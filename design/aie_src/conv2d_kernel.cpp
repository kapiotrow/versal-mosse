/*
 * conv2d_kernel.cpp
 * AIE-ML kernel: 3×3 INT8 convolution + separable Hanning window.
 *
 * One invocation processes one full PATCH_ROWS×PATCH_COLS grayscale patch:
 *   - Reads  PATCH_ROWS × PATCH_COLS  int8  samples from patch_in stream.
 *   - Writes PATCH_ROWS × PATCH_COLS  cint16 samples to feature_out stream.
 *
 * Algorithm
 * ---------
 *   Maintains a 3-row circular line buffer (with 1-sample zero-padding on
 *   each side) to form 3×3 sliding windows.
 *
 *   For each output pixel (r, c):
 *     acc  = bias_acc + Σ_{kr,kc} w[kr][kc] * buf[r+kr-1][c+kc-1]
 *     out  = saturate_int16(acc >> out_shift)
 *     wnd  = out * HANNING_128[r] / 32768 * HANNING_128[c] / 32768
 *     emit cint16{wnd, 0}
 *
 *   Border rows/cols are zero-padded (padding=1 matches the original model's
 *   same-size convolution).  The Hanning window zeroes rows 0 and PATCH_ROWS-1
 *   and cols 0 and PATCH_COLS-1, so the MAC is still performed but the output
 *   is guaranteed zero for those samples.
 *
 * ---------------------------------------------------------------------------
 * COST — per frame at 128x128, ch16, 1 GHz. Source: the aiecompiler.log
 * schedules of the build that linked; tabulated in
 * docs/thesis/results/aie_compute.csv. Claims P-04, R-01.
 *
 *   compute       4.60 ms gray (CONV_IN_CH=1) / 9.19 ms RGB (CONV_IN_CH=3).
 *                 The bank is processed serially, one channel per invocation,
 *                 so this is 16 x the per-channel schedule.
 *   vectorization 37 -> 8.75 cyc/px. aie::mac with aie::downshift, BIT-IDENTICAL
 *                 to the scalar path (CONV_VECTORIZE=0) and checked, not assumed:
 *                 make x86sim_check KUT=conv2d SCENARIO=s6 / s6rgb.
 *   pipelining    gray: MAC+post folds, 163 cyc / 16 px, critical cycle 24.
 *                 RGB: 219 cyc / 16 px and NO folding — the compiler reports
 *                 "219 (exceeds -k 64) -> no folding" against a critical cycle
 *                 of 200. 219 is therefore a give-up number, not a floor.
 *   what dominates the STREAM READ, not the arithmetic: 44% of the kernel at
 *                 CONV_IN_CH=1 and 61% at 3, because the patch is re-streamed
 *                 once per output channel. RGB makes the already-dominant term
 *                 more dominant. Optimise the read before the taps.
 *   tile memory   line buffer 3 x (PATCH_COLS+2) = 390 B at CONV_IN_CH=1;
 *                 3 planes x 3 rows with the same padding at CONV_IN_CH=3.
 *                 Weight scalars ~20 B; Hanning table 128 x 2 B = 256 B (const,
 *                 in hanning_128.h). All of it far under the 64 KB tile.
 *   stack         1024 B default is NOT enough at CONV_IN_CH=3: the 27-tap chain
 *                 needs 1344, so CONV2D_STACK=2048 is applied at RGB only. Get
 *                 this wrong and the LINK stage refuses to emit libadf.a while
 *                 the per-kernel compile succeeds either way — which is why it
 *                 went unnoticed.
 *   interface     PatchIn, a 32-bit PLIO carrying 4 packed int8 per word. Not
 *                 128-bit: that delivered one beat per readincr and starved this
 *                 kernel. `weights` is an input_buffer, so ADF re-acquires it
 *                 EVERY firing — the driver must supply PATCH_ELEMS/CONV_OUT_CHUNK
 *                 buffers per channel and must start the patch flowing first,
 *                 or the graph deadlocks.
 *   caveat        These are compiler-SCHEDULED cycles: real cycles on an in-order
 *                 VLIW core absent memory stalls. Trustworthy for sizing, not a
 *                 profile. Two traps when re-reading them: aiecompiler reuses a
 *                 cached object when the preprocessed source is unchanged, so
 *                 `rm -rf $(BUILD_DIR)/Work $(BUILD_DIR)/libadf.a` before
 *                 comparing arms; and the "140 cyc/16px" figure in the Makefile
 *                 is the main_ WRAPPER block, which does not move when the
 *                 arithmetic triples.
 *   NOT frame time. conv2d's +4.59 ms from RGB does not appear in the frame at
 *                 all: the frame is 84% CPU-bound and the AIE had the slack.
 *                 See docs/thesis/results/frame_budget_rgb_delta.csv.
 *
 * ---------------------------------------------------------------------------
 * COST — the GENERIC KxK / stride-S branch, at the pre-registered Layer-1 point
 * (7x7 stride 2, CONV_IN_CH=3, 147 taps, 128x128 crop -> 64x64 map, ch32).
 * Source: the aiecompiler.log of `make graph TARGET=hw PATCH_ROWS=64
 * PATCH_COLS=64 N_CHANNELS=32 CONV_IN_CH=3 CONV_KSIZE=7 CONV_STRIDE=2
 * CONV_RELU=1`, reworked 2026-09-02, 0 errors. Claim N-16.
 *
 *   compute       MEASURED ON HARDWARE, then optimised. The first build of
 *                 this branch cost 1.32 ms per channel = 42.4 ms/frame at ch32,
 *                 read off `roi_crop`'s ap_done poll: with ROI_CROP_PIPELINE=1
 *                 that poll measures how far the AIE lags the host, and the
 *                 per-call FLOOR rose from ~0 (sigma4, 16 ch) to 1.19 ms, which
 *                 is conv2d back-pressure and not crop work. 19.3M MACs in
 *                 42.4 ms is 0.36 MAC/cycle on a core with 128 int8 lanes.
 *                 runs/l1relu_calib/, evidence/arm_l1relu.md sec.7.4.
 *                 The doc's ~20-22 ms model was 2.4x optimistic; the frame body
 *                 was 61.48 ms against sigma4's 27.11 on the same instrument.
 *   loop schedule per-iteration cycles from aiecompiler, BEFORE -> AFTER the
 *                 2026-09-02 rework, at this same point:
 *                   PLIO read loop, per 4 pixels          143 -> 84 cycles
 *                   MAC loop, per (ic, kr) = 7 taps    7 x 11 -> 28 cycles
 *                 Weighted by iteration counts (128 read rows x CROP_COLS/4;
 *                 NC x K x PATCH_ROWS x PATCH_COLS/VEC), that is 1.00M -> 419k
 *                 cycles per channel: a 2.39x SCHEDULED improvement. The model
 *                 under-predicted hardware by 1.85x before, so the expected
 *                 landing is ~18 ms/frame of conv2d and a ~37 ms frame -- A
 *                 PREDICTION, written down before the run, not a reading.
 *   what changed  three things, none of them arithmetic, all bit-exact:
 *                 (1) the row bases moved OUT of the column loop -- they depend
 *                     on out_r, never on c, and inside they made every tap a 4-D
 *                     dynamic index ("minimum length due to resources: 10");
 *                 (2) the kc loop is UNROLLED so its phase and offset
 *                     constant-fold instead of being a div and a mod per tap;
 *                 (3) the PLIO read loop stores straight from the word through
 *                     ONE hoisted base pointer. A px[4][NC] staging array and an
 *                     srow[NC][S] pointer array were each tried and each made it
 *                     WORSE (248 and 174 cycles) -- an array indexed by unrolled
 *                     constants still went to memory. Both are recorded because
 *                     the negative result is the useful part.
 *                 The read loop is now 82% of what is left and is the next
 *                 target: 84 cycles for 4 pixels is 21 cycles/pixel for 3 byte
 *                 stores. Each (ic, phase) receives 4/S CONTIGUOUS bytes per
 *                 group, so 12 byte stores could be 6 halfword ones -- priced,
 *                 not taken, because it needs HALO aligned and a re-verification.
 *   vectorization vectorized over CONV_VEC_GEN = 32 output columns (its OWN
 *                 knob -- the 3x3 branches keep 16 and are byte-for-byte as
 *                 shipped), UNROLLED over kc, rolled over ic and kr. 32 measured
 *                 2.3x better than 16 here. BIT-EXACT against the model and
 *                 checked, not assumed: `make x86sim_check KUT=conv2d
 *                 SCENARIO=s6l1 CONV_KSIZE=7 CONV_STRIDE=2 ...` reports
 *                 4096/4096 identical, and so does CONV_VECTORIZE=0, and so do
 *                 s6/s6rgb/s7/cmul_stress after the change.
 *   stride        the line buffer is split by COLUMN PHASE on the way in, which
 *                 is what keeps every tap a unit-stride vector load at S=2. See
 *                 the branch comment.
 *   tile memory   sub[3][7][2][70] = 2940 B plus a 70 B zero row. Far under the
 *                 64 KB tile, and it is `static` -- TILE DATA MEMORY, NOT STACK.
 *   stack         CONV2D_STACK=2048 SUFFICED, unchanged from the 27-tap arm.
 *                 arm_l1relu.md called ~7.3 KB "the sharp risk" and
 *                 predicted a loop restructure would be needed; the restructure
 *                 was needed, but for the STRIDE, and it made the stack question
 *                 disappear rather than answering it -- the taps never leave the
 *                 weight input_buffer. The mapper emitted libadf.a with 0 errors
 *                 and placed conv2d at AIE_ML_CORE_X15Y0.
 *   SIGNAL LEVEL  THE THING TO WATCH ON THIS ARM, and it is measured, not
 *                 feared. export_weights.py sizes out_shift against
 *                 acc_max_theory = n_in*K^2*127^2, a bound LINEAR in the tap
 *                 count, while a real decorrelated tap sum grows like its square
 *                 root. On the s6l1 Stage-A patch the observed max|acc| over all
 *                 32 channels is 125354 against a bound of 2370963 -- 18.9x
 *                 loose -- so out_shift lands at 7 where 2 would fit, and the
 *                 feature map uses 9.9 of 15 bits (weakest channel 5.4).
 *                 The shipping 3x3 RGB bank does NOT have this problem: its
 *                 bias_acc is large and real, observed max|acc| 571420 against a
 *                 bound of 435483, and it uses 14.1 of 15 bits. The Layer-1 PCA
 *                 bank has b_fold == 0 by construction, so nothing anchors its
 *                 shift to a measured quantity.
 *                 CONFIRMED ON HARDWARE and worse than estimated: the first
 *                 200-frame run measured F_ch at 0.13% of int16 against the
 *                 res64 arm's 15.4%, and mean_prev seeded to 0 on every channel.
 *                 FIXED by `--acc-bound l1` (export_weights.py), which sizes
 *                 out_shift against 127*sum|w_int8| per channel -- also an exact
 *                 worst case, attained at x = 127*sign(w), so it cannot rail --
 *                 taking out_shift from 7 flat to 3..5 and the feature map from
 *                 1.9% to 10.3% of int16 on a real Stage-A patch, next to the
 *                 shipping bank's 12.3%. It is NOT the default: flipping it
 *                 would move the shipping bank's shifts and every arm in
 *                 claims.md with them.
 *   caveat        The loop schedules are COMPILER SCHEDULES. The 42.4 ms, the
 *                 61.48 ms frame and the F_ch level are HARDWARE, from the
 *                 2026-09-02 calibration run. The post-rework figures are
 *                 predictions and have NOT run on the board.
 */

#include "conv2d_kernel.h"
#include <aie_api/aie.hpp>
#include <cstring>

// MAC-loop implementation.
//   1 (default) = vectorized aie::mac, BIT-IDENTICAL to the scalar path
//   0           = the original scalar loop, kept for bisection and cycle
//                 comparison
// Verified with: make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0
#ifndef CONV_VECTORIZE
#  define CONV_VECTORIZE 1
#endif

// Output pixels per vector iteration. PATCH_COLS must be a whole multiple.
#ifndef CONV_VEC
#  define CONV_VEC 16
#endif

// ...and the same for the GENERIC KxK branch, which is a SEPARATE knob on
// purpose. 32 measured 2.3x better than 16 there (aiecompiler schedule at the
// 7x7/2 point: 28 cycles per 7 unrolled taps against 32, over half as many
// column iterations), but the two 3x3 branches are byte-for-byte as shipped and
// a shared knob would move them too. Their 16 is left exactly where it was.
#ifndef CONV_VEC_GEN
#  define CONV_VEC_GEN 32
#endif

// Half-wave rectifier after the output shift.
//   1 (default) = as shipped
//   0           = no ReLU, saturate only
//
// 0 IS THE BETTER SETTING and it is not the default yet on purpose. Measured
// offline 2026-08-14 (scripts/phase1_sweep.py), held-out evaluation at
// 128x128/ch16: removing ReLU takes peak/max-sidelobe from 12.8 to 16.3 and, more
// importantly, the planned bias_acc fix goes from 3.9 (a 3x REGRESSION) to 16.3
// once ReLU is off. A DCF is linear in its features; a half-wave rectifier throws
// away half the signal and the linear filter cannot undo it. export_weights.py
// already makes this argument when it drops Hardswish.
//
// Left off by default because it changes numerics, so it needs its own before/
// after run rather than riding along with a rewrite that is bit-identical by
// design. See the ReLU entry in CLAUDE.md Known Issues.
#ifndef CONV_RELU
#  define CONV_RELU 1
#endif

// Include the Hanning table sized for this build's patch (square: PATCH_ROWS==PATCH_COLS).
// HTAB aliases the size-specific symbol. Generate a table with: make weights PATCH_COLS=<n>
#if   PATCH_COLS == 128
#  include "hanning_128.h"
// @thesis subsec:przetwarzanieWstepne | A-06 | The Hanning table: PERIODIC sin^2(pi*i/N), not symmetric.
//   Its 2-D DFT has exactly 9 non-zero bins, which is what makes the Stage B2 correction exact.
#  define HTAB HANNING_128
#elif PATCH_COLS == 64
#  include "hanning_64.h"
#  define HTAB HANNING_64
#elif PATCH_COLS == 32
#  include "hanning_32.h"
#  define HTAB HANNING_32
#else
#  error "No Hanning table for this PATCH_COLS. Add a case here and run: make weights PATCH_COLS=<n>"
#endif

// PatchIn PLIO is 32-bit (plio_32_bits), carrying 4 densely-packed int8 pixels
// per beat. The stream is consumed as int32 words and unpacked below, 1:1 with
// PLIO beats: a scalar readincr on an int8 stream would pop a full 32-bit word
// per call and starve the kernel 4:1. Requires PATCH_COLS % 4 == 0.
//
// aiesim stimulus must match this: gen_aiesim_vectors.py writes ONE packed int32
// per line of patch_in.txt (the ISS parses that file in units of the port's
// stream element type, and rejects 4-values-per-line with
// "Invalid number of data samples on line 1, got 4 expected 1").
// Build mode. Note there is no stream->window adapter in this path any more:
// mosse_graph.h wires conv2d's output_buffer directly into fft2d.fft_row_in.
// 0 = real 3x3 convolution
// 1 = echo the input stream (isolates dataflow from conv/weights logic)
// 2 = BISECT: synthesize output WITHOUT reading the stream at all. Distinguishes
//     "conv2d is blocked on readincr" (PLIO not delivering) from "the blockage is
//     downstream" (conv2d->fft_row_in window, row-FFT, or the GMIO drain).
// Override from the Makefile with CONV2D_MODE=<n>.
#ifndef CONV2D_ECHO_TEST
#  define CONV2D_ECHO_TEST 1
#endif

void conv2d_kernel(
    input_stream<int32>   *patch_in,
    output_buffer<cint16> &feature_out,
    input_buffer<int8_t>  &weights)
{
#if CONV2D_ECHO_TEST == 2
    // BISECT: fill the output window with a deterministic ramp and never touch
    // patch_in. If gmio_fft_row_out now drains, conv2d was stuck on readincr and
    // the fault is the PLIO stream. If it still hangs, the PLIO is not the
    // blocker and the fault is downstream (window link / row-FFT / GMIO drain).
    (void)weights;
    {
        cint16_t *out = feature_out.data();
        for (int i = 0; i < CONV_OUT_CHUNK; i++)
        chess_prepare_for_pipelining
        chess_loop_range(CONV_OUT_CHUNK, CONV_OUT_CHUNK)
        {
            out[i].real = (int16_t)i;
            out[i].imag = 0;
        }
    }
    return;
#elif CONV2D_ECHO_TEST == 1
    // One invocation = one row-FFT window: read CONV_OUT_CHUNK/4 int32 words,
    // unpack to CONV_OUT_CHUNK cint16 in the output window. Fires
    // (PATCH_ROWS*PATCH_COLS)/CONV_OUT_CHUNK times to drain the full patch.
    cint16_t *out = feature_out.data();

    for (int i = 0; i < CONV_OUT_CHUNK / 4; i++)
    chess_prepare_for_pipelining
    chess_loop_range(CONV_OUT_CHUNK / 4, CONV_OUT_CHUNK / 4)
    {
        int32_t w = readincr(patch_in);
        out[4 * i + 0].real = (int8_t)( w        & 0xFF);  out[4 * i + 0].imag = 0;
        out[4 * i + 1].real = (int8_t)((w >>  8) & 0xFF);  out[4 * i + 1].imag = 0;
        out[4 * i + 2].real = (int8_t)((w >> 16) & 0xFF);  out[4 * i + 2].imag = 0;
        out[4 * i + 3].real = (int8_t)((w >> 24) & 0xFF);  out[4 * i + 3].imag = 0;
    }
    return;

// ====================================================================
// GENERIC KxK / STRIDE-S PATH — taken whenever CONV_KSIZE != 3 or
// CONV_STRIDE != 1. The two 3x3 stride-1 branches below are left byte for
// byte as they shipped, so selecting this path is the only thing that can
// change a shipped arm's numerics.
//
// WHY A SEPARATE BRANCH AND NOT A GENERALISATION OF THE 3x3 ONE. The shipped
// branches hoist every tap into a named scalar (9 of them gray, 27 RGB) and
// unroll the MAC by hand. At 147 taps that is what arm_l1relu.md
// flags as the sharp risk: "~7.3 KB of stack ... likely restructuring the
// inner loop rather than raising a number". This branch keeps the taps IN THE
// WEIGHT BUFFER and loops over them, so CONV2D_STACK does not have to grow
// with the tap count at all -- the cost moves to a rolled loop the compiler
// can still pipeline over the CONV_VEC output columns.
//
// THE STRIDE IS WHY THE LINE BUFFER IS DE-INTERLEAVED.
// At stride 2 output column c reads input column 2c + kc - P, so consecutive
// output columns are 2 input columns apart and a contiguous vector load no
// longer lines up with the outputs. Splitting each input row into its S column
// PHASES as it is read puts every tap back on a unit stride:
//
//     input col 2c + d, d = kc - P
//       = phase (d mod S), half-column c + floor(d / S)
//
// so tap kc is a single unaligned load at a CONSTANT offset from c. One pass
// over the row does the split, which the read loop was already doing byte by
// byte. At S = 1 the phase collapses to 0 and this is the ordinary sliding
// window.
//
// GEOMETRY. The input map is the ROI CROP -- PATCH_ROWS*S x PATCH_COLS*S --
// and the output map is PATCH_ROWS x PATCH_COLS, which is what the Hann table,
// the FFTs and the whole host tail are sized for. roi_crop takes its patch size
// as a RUNTIME AXI-Lite argument, so a 128x128 crop feeding a 64x64 feature map
// costs no PL rebuild; see CROP_ROWS/CROP_COLS in the Makefile.
//
// Padding is 'same' with an odd kernel: P = K/2 zero columns and rows on each
// side, which reproduces the 3x3 branches' padding=1 exactly at K=3.
// ====================================================================
// @thesis subsec:wyborSieci | N-16 | The Layer-1 datapath: a KxK stride-S INT8 convolution
//   whose taps stay in the weight buffer and whose line buffer is split by column phase, so
//   the stride does not break vectorization.
#elif (CONV_KSIZE != 3) || (CONV_STRIDE != 1)

    constexpr int K  = CONV_KSIZE;
    constexpr int P  = K / 2;               // 'same' padding, both sides
    constexpr int S  = CONV_STRIDE;
    constexpr int NC = CONV_IN_CH;

    // The ROI crop conv2d consumes. NOT the feature map.
    constexpr int CROP_ROWS = PATCH_ROWS * S;
    constexpr int CROP_COLS = PATCH_COLS * S;

    // Half-row halo. off = floor(d/S) for d in [-P, P], so |off| <= P, and a
    // VEC load starting at c + off must stay inside the buffer. P on each
    // side covers both.
    constexpr int HALO = P;
    constexpr int HW   = PATCH_COLS + 2 * HALO;   // phase-row width

    static_assert(K % 2 == 1, "'same' padding needs an odd kernel");
    static_assert(S == 1 || S == 2, "phase split is written for S = 1 or 2");
    static_assert(CROP_COLS % 4 == 0, "the PatchIn read loop consumes 4 pixels at a time");
    static_assert(4 % S == 0, "the read loop splits (c+b)/S as c/S + b/S, which needs S | 4");
    constexpr int VEC = CONV_VEC_GEN;   // this branch's own vector width
    static_assert(PATCH_COLS % VEC == 0, "output row must divide into vectors");

    const int8_t *wb = weights.data();
    const int out_shift = (int)(uint8_t)wb[CONV_W_OFF_SHIFT];

    int32_t bias;
    bias  = (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 0];
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 1] << 8;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 2] << 16;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 3] << 24;

    int32_t mean_prev;
    mean_prev  = (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 0];
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 1] << 8;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 2] << 16;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 3] << 24;

    constexpr int ROWS_PER_INV = CONV_OUT_CHUNK / PATCH_COLS;
    static_assert(CONV_OUT_CHUNK % PATCH_COLS == 0,
                  "output window must be a whole number of feature-map rows");
    static_assert(PATCH_ROWS % ROWS_PER_INV == 0,
                  "feature map must divide evenly into output windows");

    // sub[plane][slot][phase][HALO + half-col]. Input row y lives in slot y % K:
    // the K rows an output row needs are K CONSECUTIVE input rows, so they
    // occupy K distinct slots, and reading row y evicts row y-K, which is one
    // below the window. STATIC, i.e. tile data memory, not stack:
    // 3 x 7 x 2 x 70 = 2940 B at the 7x7/2 RGB point.
    static int8_t sub[NC][K][S][HW];
    static int8_t zsub[HW];                 // the implicit zero rows and columns
    static int    rows_read = 0;            // input rows consumed this patch
    static int    rows_out  = 0;            // feature rows produced this patch

    if (rows_out == 0) {                    // first firing of a new patch
        memset(sub,  0, sizeof(sub));
        memset(zsub, 0, sizeof(zsub));
        rows_read = 0;
    }

    cint16_t *out = feature_out.data();
    int o = 0;

    for (int kk = 0; kk < ROWS_PER_INV; ++kk) {

        const int out_r = rows_out;

        // Read forward until input row out_r*S + P is buffered. Reads per firing
        // are uneven (the first pulls S*0+P+1 rows, later ones S) but total
        // exactly CROP_ROWS over the patch, which is what roi_crop emits.
        while (rows_read <= out_r * S + P && rows_read < CROP_ROWS) {
            const int slot = rows_read % K;

            // Destination base, hoisted: `slot` does not depend on c. ONE
            // pointer, not an array of NC*S -- an array of pointers indexed by
            // unrolled constants still went to memory, and 12 pointer loads per
            // 4 pixels is what the read loop was paying. `sub` is
            // [NC][K][S][HW], so the (ic, ph) offset from this base is the
            // compile-time constant SB_IC*ic + HW*ph.
            int8_t *const sb = &sub[0][slot][0][HALO];
            constexpr int SB_IC = K * S * HW;

            for (int c = 0; c < CROP_COLS; c += 4)
            chess_prepare_for_pipelining
            chess_loop_range(CROP_COLS / 4, CROP_COLS / 4)
            {
                // NC words carry exactly 4 pixels: 1 word gray, 3 words RGB
                // (R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3), which is what
                // roi_crop puts on the wire at ROI_IN_CH=3.
                //
                // BOTH INNER LOOPS ARE FULLY UNROLLED, and that is the point:
                // rolled, this body's `lin/NC`, `lin%NC`, `x%S` and `x/S` are
                // four integer div/mod per byte on a machine that has none, and
                // the nest is a NON-LEAF loop, which aiecompiler then refuses to
                // software-pipeline ("Skipping pipelining of non-leaf loop").
                // It scheduled at 143 cycles per 4 pixels -- 64% of conv2d's
                // whole cost at the 7x7/2 point, more than the 147 MACs it
                // feeds. Unrolled, every index is a compile-time constant and
                // the body is straight-line stores, which is exactly the shape
                // the 3x3 RGB branch's hand-written unpacker already has.
                //
                // `half_c` uses c/S + b/S == (c+b)/S and b%S == (c+b)%S, both exact
                // because S divides 4 and c steps by 4 -- asserted below.
                const int half_c = c / S;   // c/S + b/S == (c+b)/S, exact since S | 4
                for (int j = 0; j < NC; ++j)
                chess_unroll_loop(NC)
                {
                    const int32_t w = readincr(patch_in);
                    for (int b = 0; b < 4; ++b)
                    chess_unroll_loop(4)
                    {
                        const int lin = j * 4 + b;   // sample index within the 4 pixels
                        const int pix = lin / NC;    // which of the 4 pixels
                        const int ic  = lin % NC;    // which plane
                        // Stored STRAIGHT FROM THE WORD. A px[4][NC] staging
                        // array here cost 24 extra memory ops and a read-after-
                        // write chain the scheduler could not break: it put this
                        // loop at 248 cycles against a 192-cycle resource bound.
                        sb[SB_IC * ic + HW * (pix % S) + half_c + pix / S] =
                            (int8_t)((w >> (8 * b)) & 0xFF);
                    }
                }
            }
            // Left/right zero pad. The halo columns are written once per row
            // because a row buffer is reused every K rows and must not carry a
            // previous row's edge pixels into this one's padding.
            for (int ph = 0; ph < S; ++ph)
                for (int ic = 0; ic < NC; ++ic)
                    for (int h = 0; h < HALO; ++h) {
                        sub[ic][slot][ph][h] = 0;
                        sub[ic][slot][ph][HALO + PATCH_COLS + h] = 0;
                    }
            ++rows_read;
        }

        const int16_t h_r = HTAB[out_r];

        // ROW BASES, HOISTED OUT OF THE COLUMN LOOP. Everything that selects a
        // tap's source row -- the input row y, whether it is inside the crop,
        // its slot in the K-row ring, and the phase split -- depends on out_r
        // and NOT on c. Left inside the c loop (as this branch first shipped)
        // it made the address a 4-D dynamic index recomputed per tap, which is
        // what put the kc loop at "minimum length due to resources: 10" in the
        // aiecompiler schedule and cost the arm 42.4 ms/frame of conv2d on
        // hardware -- runs/l1relu_calib/run_l1relu_calib.log, read off
        // roi_crop's ap_done poll (docs/thesis/evidence/arm_l1relu.md
        // sec.7.4).
        //
        // rowb[ic][kr][ph] points at half-column 0 of the row that tap
        // (ic, kr, phase ph) reads, so the tap address is a base plus the
        // COMPILE-TIME offset `off`. The zero row has no phase dimension, so
        // both phases point at zsub -- which is correct because zsub is zero
        // everywhere and the two phases of a zero row are the same row.
        //
        // BIT-EXACTNESS: the mac order (ic, kr, kc) and every operand value are
        // unchanged; only address arithmetic moved. Verified by
        // `make x86sim_check KUT=conv2d SCENARIO=s6l1` after the change.
        const int8_t *rowb[NC][K][S];
        for (int ic = 0; ic < NC; ++ic)
            for (int kr = 0; kr < K; ++kr) {
                const int  y    = out_r * S + kr - P;
                const bool in_r = (y >= 0) && (y < CROP_ROWS);
                const int  slot = in_r ? (y % K) : 0;
                for (int ph = 0; ph < S; ++ph)
                    rowb[ic][kr][ph] = in_r ? &sub[ic][slot][ph][HALO]
                                            : &zsub[HALO];
            }

#if CONV_VECTORIZE

        // Vectorized over VEC output columns, rolled over the K*K*NC taps.
        // BIT-IDENTICAL to the scalar loop below: every shift is aie::downshift
        // (arithmetic/floor, matching signed `>>`), never srs, for the reasons
        // spelled out in the 3x3 branch.
        for (int c = 0; c < PATCH_COLS; c += VEC)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS / VEC, PATCH_COLS / VEC)
        {
            aie::accum<acc32, VEC> a;
            a.from_vector(aie::broadcast<int32_t, VEC>(bias), 0);

            for (int ic = 0; ic < NC; ++ic) {
                const int8_t *wp = wb + CONV_W_OFF_PLANE(ic);
                for (int kr = 0; kr < K; ++kr) {
                    const int8_t *wrow = wp + kr * K;
                    // UNROLLED so `ph` and `off` constant-fold: they are pure
                    // functions of kc, and K and S are constexpr. Rolled, they
                    // are a modulo and a division per tap on the critical path.
                    for (int kc = 0; kc < K; ++kc)
                    chess_unroll_loop(K)
                    {
                        const int d   = kc - P;
                        // Floor division / positive modulo: d is negative for
                        // half the taps and C's / and % truncate toward zero.
                        const int ph  = ((d % S) + S) % S;
                        const int off = (d - ph) / S;
                        a = aie::mac(a,
                                     aie::load_unaligned_v<VEC>(
                                         rowb[ic][kr][ph] + c + off),
                                     wrow[kc]);
                    }
                }
            }

            aie::vector<int32_t, VEC> sh =
                aie::downshift(a.to_vector<int32_t>(0), out_shift);
#if CONV_RELU
            aie::vector<int32_t, VEC> r = aie::min(aie::max(sh, 0), 32767);
#else
            aie::vector<int32_t, VEC> r = aie::min(aie::max(sh, -32768), 32767);
#endif
            aie::vector<int32_t, VEC> cen =
                aie::sub(r, aie::broadcast<int32_t, VEC>(mean_prev));
            cen = aie::min(aie::max(cen, -32768), 32767);

            aie::vector<int32_t, VEC> w1v =
                aie::downshift(aie::mul(cen, (int32_t)h_r).template to_vector<int32_t>(0), 15);
            aie::vector<int32_t, VEC> hc =
                aie::unpack(aie::load_unaligned_v<VEC>((const int16_t *)HTAB + c));
            aie::vector<int32_t, VEC> w2v =
                aie::downshift(aie::mul(w1v, hc).template to_vector<int32_t>(0), 15);
            w2v = aie::min(aie::max(w2v, -32768), 32767);

            aie::accum<acc32, VEC> t;
            t.from_vector(w2v, 0);
            const aie::vector<int16_t, VEC> re16 = t.template to_vector<int16_t>(0);
            const aie::vector<int16_t, VEC> zero =
                aie::zeros<int16_t, VEC>();
            const auto zp = aie::interleave_zip(re16, zero, 1);

            int16_t *dst = (int16_t *)(out + o);
            aie::store_unaligned_v(dst,            zp.first);
            aie::store_unaligned_v(dst + VEC, zp.second);
            o += VEC;
        }

#else   // scalar reference — the bisection path, and BIT-IDENTICAL to the above

        for (int c = 0; c < PATCH_COLS; ++c)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS, PATCH_COLS)
        {
            int32_t acc = bias;
            for (int ic = 0; ic < NC; ++ic) {
                const int8_t *wp = wb + CONV_W_OFF_PLANE(ic);
                for (int kr = 0; kr < K; ++kr) {
                    // Same hoisted bases as the vectorized path, so the two
                    // stay one expression of one addressing scheme. An
                    // out-of-crop row reads zsub, which is zero, instead of
                    // being `continue`d -- arithmetically identical, and it
                    // keeps this loop's shape the same as the one above.
                    const int8_t *wrow = wp + kr * K;
                    for (int kc = 0; kc < K; ++kc) {
                        const int d   = kc - P;
                        const int ph  = ((d % S) + S) % S;
                        const int off = (d - ph) / S;
                        acc += (int32_t)wrow[kc]
                             * (int32_t)rowb[ic][kr][ph][c + off];
                    }
                }
            }

            int32_t shifted = acc >> out_shift;
#if CONV_RELU
            if      (shifted >  32767) shifted =  32767;
            else if (shifted <      0) shifted =  0;
#else
            if      (shifted >  32767) shifted =  32767;
            else if (shifted < -32768) shifted = -32768;
#endif
            int32_t centred = shifted - mean_prev;
            if      (centred >  32767) centred =  32767;
            else if (centred < -32768) centred = -32768;

            const int16_t h_c = HTAB[c];
            int32_t wnd = (centred * (int32_t)h_r) >> 15;
            wnd         = (wnd * (int32_t)h_c) >> 15;
            if      (wnd >  32767) wnd =  32767;
            else if (wnd < -32768) wnd = -32768;

            out[o].real = (int16_t)wnd;
            out[o].imag = 0;
            ++o;
        }

#endif  // CONV_VECTORIZE

        ++rows_out;
        if (rows_out >= PATCH_ROWS) rows_out = 0;   // patch complete — rearm
    }
    return;

// ====================================================================
// RGB PATH — CONV_IN_CH == 3. Present to MEASURE, not yet to ship: nothing
// else in the design (roi_crop, export_weights.py, the host) produces or
// consumes 3-plane data, so this compiles and schedules but is not wired up.
// Its purpose is to replace the estimated "conv2d 4.1 -> ~9.0 ms/frame" in
// CLAUDE.md with the compiler's own scheduled cycle count, via
//     make graph TARGET=hw_emu CONV_IN_CH=3
// and reading "Total number of cycles" for node 14_0-main_ in aiecompiler.log.
// TARGET=hw_emu is a SANDBOX: AIE_FLAGS is --target=hw regardless, so this
// compiles the identical graph without overwriting build/hw's libadf.a or the
// aiecompiler.log that documents what actually ran on the board.
//
// Two design choices this encodes, both from the RGB section of CLAUDE.md:
//   * INTERLEAVED ON THE WIRE, PLANAR IN THE LINE BUFFER. roi_crop must send
//     interleaved (planar would need whole planes resident); the read loop
//     de-interleaves into three 3-row plane buffers as it unpacks, which it was
//     already doing byte by byte. 3 int32 words carry exactly 4 RGB pixels, so
//     the scatter pattern repeats every 4 columns instead of rotating.
//   * PLANAR WEIGHT ORDER: [0:9] R, [9:18] G, [18:27] B. The remaining fields
//     follow the tap block exactly as they do at CONV_IN_CH=1, and their
//     offsets come from conv_weight_layout.h rather than from literals here:
//     out_shift 27, bias_acc 28, dequant_scale 32, mean_prev 36. The grayscale
//     layout has 27 taps overrunning ALL FOUR of those fields, which is why the
//     offsets stopped being hand-written in four files.
// ====================================================================
// @thesis subsec:wyborSieci | R-01 | The RGB datapath: 27 vectorized taps over three
//   de-interleaved planes. This is the branch the shipping arm takes.
#elif CONV_IN_CH == 3

    const int8_t *wb = weights.data();
    const int8_t *wR = wb + CONV_W_OFF_PLANE(0);
    const int8_t *wG = wb + CONV_W_OFF_PLANE(1);
    const int8_t *wB = wb + CONV_W_OFF_PLANE(2);
    const int8_t wR00 = wR[0], wR01 = wR[1], wR02 = wR[2];
    const int8_t wR10 = wR[3], wR11 = wR[4], wR12 = wR[5];
    const int8_t wR20 = wR[6], wR21 = wR[7], wR22 = wR[8];
    const int8_t wG00 = wG[0], wG01 = wG[1], wG02 = wG[2];
    const int8_t wG10 = wG[3], wG11 = wG[4], wG12 = wG[5];
    const int8_t wG20 = wG[6], wG21 = wG[7], wG22 = wG[8];
    const int8_t wB00 = wB[0], wB01 = wB[1], wB02 = wB[2];
    const int8_t wB10 = wB[3], wB11 = wB[4], wB12 = wB[5];
    const int8_t wB20 = wB[6], wB21 = wB[7], wB22 = wB[8];
    const int out_shift = (int)(uint8_t)wb[CONV_W_OFF_SHIFT];

    int32_t bias;
    bias  = (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 0];
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 1] << 8;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 2] << 16;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 3] << 24;

// @thesis subsec:kwantyzacjaImpl | B-08 | The per-channel quantization parameters as the kernel
//   consumes them: out_shift, bias_acc, and mean_prev for Stage B1.
    int32_t mean_prev;
    mean_prev  = (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 0];
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 1] << 8;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 2] << 16;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 3] << 24;

    constexpr int ROWS_PER_INV = CONV_OUT_CHUNK / PATCH_COLS;
    static_assert(PATCH_COLS % 4 == 0, "RGB read loop consumes 4 pixels per 3 words");

    // 3 planes x 3 rows, each with the 1-sample zero pad on both sides.
    static int8_t buf[3][3][PATCH_COLS + 2];
    static int8_t zrow[PATCH_COLS + 2];
    static int    rows_read = 0;
    static int    rows_out  = 0;

    if (rows_out == 0) {
        memset(buf,  0, sizeof(buf));
        memset(zrow, 0, sizeof(zrow));
        rows_read = 0;
    }

    cint16_t *out = feature_out.data();
    int o = 0;

    for (int k = 0; k < ROWS_PER_INV; ++k) {

        const int out_r = rows_out;

        while (rows_read <= out_r + 1 && rows_read < PATCH_ROWS) {
            int8_t *dR = buf[0][rows_read % 3];
            int8_t *dG = buf[1][rows_read % 3];
            int8_t *dB = buf[2][rows_read % 3];
            dR[0] = dG[0] = dB[0] = 0;                  // left zero pad
            for (int c = 0; c < PATCH_COLS; c += 4)
            chess_prepare_for_pipelining
            chess_loop_range(PATCH_COLS / 4, PATCH_COLS / 4)
            {
                // 12 bytes = 4 RGB pixels: R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3
                const int32_t w0 = readincr(patch_in);
                const int32_t w1 = readincr(patch_in);
                const int32_t w2 = readincr(patch_in);
                dR[c + 1] = (int8_t)( w0        & 0xFF);
                dG[c + 1] = (int8_t)((w0 >>  8) & 0xFF);
                dB[c + 1] = (int8_t)((w0 >> 16) & 0xFF);
                dR[c + 2] = (int8_t)((w0 >> 24) & 0xFF);
                dG[c + 2] = (int8_t)( w1        & 0xFF);
                dB[c + 2] = (int8_t)((w1 >>  8) & 0xFF);
                dR[c + 3] = (int8_t)((w1 >> 16) & 0xFF);
                dG[c + 3] = (int8_t)((w1 >> 24) & 0xFF);
                dB[c + 3] = (int8_t)( w2        & 0xFF);
                dR[c + 4] = (int8_t)((w2 >>  8) & 0xFF);
                dG[c + 4] = (int8_t)((w2 >> 16) & 0xFF);
                dB[c + 4] = (int8_t)((w2 >> 24) & 0xFF);
            }
            dR[PATCH_COLS + 1] = dG[PATCH_COLS + 1] = dB[PATCH_COLS + 1] = 0;
            ++rows_read;
        }

        const int8_t *rR_top = (out_r >= 1) ? buf[0][(out_r - 1) % 3] : zrow;
        const int8_t *rG_top = (out_r >= 1) ? buf[1][(out_r - 1) % 3] : zrow;
        const int8_t *rB_top = (out_r >= 1) ? buf[2][(out_r - 1) % 3] : zrow;
        const int8_t *rR_mid = buf[0][out_r % 3];
        const int8_t *rG_mid = buf[1][out_r % 3];
        const int8_t *rB_mid = buf[2][out_r % 3];
        const int8_t *rR_bot = (out_r + 1 < PATCH_ROWS) ? buf[0][(out_r + 1) % 3] : zrow;
        const int8_t *rG_bot = (out_r + 1 < PATCH_ROWS) ? buf[1][(out_r + 1) % 3] : zrow;
        const int8_t *rB_bot = (out_r + 1 < PATCH_ROWS) ? buf[2][(out_r + 1) % 3] : zrow;

        const int16_t h_r = HTAB[out_r];

        for (int c = 0; c < PATCH_COLS; c += CONV_VEC)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS / CONV_VEC, PATCH_COLS / CONV_VEC)
        {
            aie::accum<acc32, CONV_VEC> a;
            a.from_vector(aie::broadcast<int32_t, CONV_VEC>(bias), 0);

            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_top + c), wR00);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_top + c + 1), wR01);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_top + c + 2), wR02);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_mid + c), wR10);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_mid + c + 1), wR11);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_mid + c + 2), wR12);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_bot + c), wR20);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_bot + c + 1), wR21);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rR_bot + c + 2), wR22);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_top + c), wG00);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_top + c + 1), wG01);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_top + c + 2), wG02);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_mid + c), wG10);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_mid + c + 1), wG11);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_mid + c + 2), wG12);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_bot + c), wG20);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_bot + c + 1), wG21);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rG_bot + c + 2), wG22);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_top + c), wB00);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_top + c + 1), wB01);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_top + c + 2), wB02);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_mid + c), wB10);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_mid + c + 1), wB11);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_mid + c + 2), wB12);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_bot + c), wB20);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_bot + c + 1), wB21);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(rB_bot + c + 2), wB22);

            // Post-chain IDENTICAL to the grayscale path — it is fixed cost and
            // does not scale with the input channel count, which is exactly what
            // the cycle measurement is meant to show.
            aie::vector<int32_t, CONV_VEC> sh =
                aie::downshift(a.to_vector<int32_t>(0), out_shift);
// @thesis subsec:wyborSieci | N-16 | ReLU, compiled out by default: a DCF is linear in feature
//   space, so half-wave rectification costs ~3x the peak/sidelobe ratio.
#if CONV_RELU
            aie::vector<int32_t, CONV_VEC> r = aie::min(aie::max(sh, 0), 32767);
#else
            aie::vector<int32_t, CONV_VEC> r = aie::min(aie::max(sh, -32768), 32767);
#endif
            aie::vector<int32_t, CONV_VEC> cen =
                aie::sub(r, aie::broadcast<int32_t, CONV_VEC>(mean_prev));
            cen = aie::min(aie::max(cen, -32768), 32767);

            aie::vector<int32_t, CONV_VEC> w1v =
                aie::downshift(aie::mul(cen, (int32_t)h_r).template to_vector<int32_t>(0), 15);
            aie::vector<int32_t, CONV_VEC> hc =
                aie::unpack(aie::load_unaligned_v<CONV_VEC>((const int16_t *)HTAB + c));
            aie::vector<int32_t, CONV_VEC> w2v =
                aie::downshift(aie::mul(w1v, hc).template to_vector<int32_t>(0), 15);
            w2v = aie::min(aie::max(w2v, -32768), 32767);

            aie::accum<acc32, CONV_VEC> t;
            t.from_vector(w2v, 0);
            const aie::vector<int16_t, CONV_VEC> re16 = t.template to_vector<int16_t>(0);
            const aie::vector<int16_t, CONV_VEC> zero =
                aie::zeros<int16_t, CONV_VEC>();
            const auto zp = aie::interleave_zip(re16, zero, 1);

            int16_t *dst = (int16_t *)(out + o);
            aie::store_unaligned_v(dst,            zp.first);
            aie::store_unaligned_v(dst + CONV_VEC, zp.second);
            o += CONV_VEC;
        }

        ++rows_out;
        if (rows_out == PATCH_ROWS) rows_out = 0;
    }
    return;

#else
    // ----------------------------------------------------------------
    // Load per-channel kernel parameters from the 64-byte weight buffer.
    // Layout (see conv2d_kernel.h):
    //   [0:9]   int8  w[KSIZE][KSIZE]   9 bytes, row-major
    //   [9]     int8  out_shift
    //   [10:14] int32 bias_acc (LE)
    //   [14:18] float32 dequant_scale (LE)  — host-only, unused here
    //   [18:22] int32 mean_prev (LE)        — Stage B1
    // ----------------------------------------------------------------
    const int8_t *wb = weights.data();

    const int8_t w00 = wb[0], w01 = wb[1], w02 = wb[2];
    const int8_t w10 = wb[3], w11 = wb[4], w12 = wb[5];
    const int8_t w20 = wb[6], w21 = wb[7], w22 = wb[8];
    // Offsets below come from conv_weight_layout.h. At CONV_IN_CH=1 they are
    // the historical 9/10/14/18 — verified by the shipped .bin round-tripping
    // byte-for-byte through the shared packer.

    const int out_shift = (int)(uint8_t)wb[CONV_W_OFF_SHIFT];

    int32_t bias;
    // Byte-by-byte copy avoids alignment UB when wb is not 4-byte aligned.
    bias  = (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 0];
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 1] << 8;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 2] << 16;
    bias |= (int32_t)(uint8_t)wb[CONV_W_OFF_BIAS + 3] << 24;

    // Stage B1: previous frame's post-ReLU feature mean for this channel.
    // Subtracted after the ReLU and before the window. Zero on the first frame,
    // which simply degrades to the old behaviour for that one frame.
    int32_t mean_prev;
    mean_prev  = (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 0];
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 1] << 8;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 2] << 16;
    mean_prev |= (int32_t)(uint8_t)wb[CONV_W_OFF_MEAN + 3] << 24;

    // ----------------------------------------------------------------
    // STATEFUL ACROSS INVOCATIONS.
    //
    // The graph fires this kernel once per row-FFT input window
    // (CONV_OUT_CHUNK samples = ROWS_PER_INV rows), NOT once per patch. The
    // previous implementation assumed one invocation = one whole patch: it read
    // all PATCH_ROWS rows and emitted PATCH_ELEMS samples via writeincr() on a
    // stream. That stopped compiling when feature_out became an output_buffer
    // (writeincr has no buffer overload), and it had never been updated for the
    // chunked firing model — emitting PATCH_ELEMS samples into a CONV_OUT_CHUNK
    // buffer would have overrun it regardless. So MODE=0 has been dead code since
    // the stream->window adapter was removed from mosse_graph.h.
    //
    // The 3-row sliding window and the row counters therefore have to persist
    // between firings, in tile-local static storage.
    // ----------------------------------------------------------------
    constexpr int ROWS_PER_INV = CONV_OUT_CHUNK / PATCH_COLS;
    static_assert(CONV_OUT_CHUNK % PATCH_COLS == 0,
                  "output window must be a whole number of rows");
    static_assert(PATCH_ROWS % ROWS_PER_INV == 0,
                  "patch must divide evenly into output windows");

    // buf[r % 3] holds input row r. zrow supplies the implicit zero rows above
    // row 0 and below row PATCH_ROWS-1 (padding=1, matching the model's 'same'
    // convolution). The [0] and [PATCH_COLS+1] slots are the left/right zero pad.
    static int8_t buf[3][PATCH_COLS + 2];
    static int8_t zrow[PATCH_COLS + 2];
    static int    rows_read = 0;   // input rows consumed from patch_in this patch
    static int    rows_out  = 0;   // output rows produced this patch

    if (rows_out == 0) {           // first firing of a new patch
        memset(buf,  0, sizeof(buf));
        memset(zrow, 0, sizeof(zrow));
        rows_read = 0;
    }

    cint16_t *out = feature_out.data();
    int o = 0;

    for (int k = 0; k < ROWS_PER_INV; ++k) {

        const int out_r = rows_out;

        // Read forward until the row BELOW out_r is in the buffer. Rows arrive in
        // order, so the three slots then hold out_r-1, out_r, out_r+1.
        // Read counts per firing are uneven (3 rows for the first, 1 for the last)
        // but total exactly PATCH_ROWS over the patch.
        while (rows_read <= out_r + 1 && rows_read < PATCH_ROWS) {
            int8_t *dst = buf[rows_read % 3];
            dst[0] = 0;                       // left zero pad
            for (int c = 0; c < PATCH_COLS; c += 4)
            chess_prepare_for_pipelining
            chess_loop_range(PATCH_COLS / 4, PATCH_COLS / 4)
            {
                int32_t w = readincr(patch_in);
                dst[c + 1] = (int8_t)( w        & 0xFF);
                dst[c + 2] = (int8_t)((w >>  8) & 0xFF);
                dst[c + 3] = (int8_t)((w >> 16) & 0xFF);
                dst[c + 4] = (int8_t)((w >> 24) & 0xFF);
            }
            dst[PATCH_COLS + 1] = 0;          // right zero pad
            ++rows_read;
        }

        const int8_t *row_top = (out_r >= 1)              ? buf[(out_r - 1) % 3] : zrow;
        const int8_t *row_mid =                             buf[ out_r      % 3];
        const int8_t *row_bot = (out_r + 1 < PATCH_ROWS)  ? buf[(out_r + 1) % 3] : zrow;

        const int16_t h_r = HTAB[out_r];

#if CONV_VECTORIZE

        // ------------------------------------------------------------------
        // Vectorized column loop — BIT-IDENTICAL to the scalar loop below.
        //
        // The scalar version schedules at 37 cycles per OUTPUT PIXEL (measured,
        // aiecompiler.log), which is ~11.5 ms/frame at 128x128 x 16 channels and
        // makes conv2d the largest single compute cost in the design — nine
        // scalar int32xint8 MACs on a core whose int8 vector datapath does 256
        // MAC/cycle.
        //
        // BIT-EXACTNESS: every shift here is aie::downshift, which is an
        // ARITHMETIC (floor) shift, matching C++ `>>` on a signed value. It is
        // deliberately NOT srs: srs rounds to nearest, and while that is
        // arguably better numerics it would change every pixel by up to 1 LSB
        // and invalidate Stage B2 (whose ~1e-3 residual is documented as
        // depending on these truncations), the shift budget, and every s6/s7
        // expected value. `aie::logical_downshift` would be wrong for a
        // different reason — it zero-fills instead of sign-extending.
        //
        // The clamps are min/max chains rather than selects for the same reason
        // the scalar sat16 is branchless: control flow inside this loop costs
        // more than the arithmetic.
        for (int c = 0; c < PATCH_COLS; c += CONV_VEC)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS / CONV_VEC, PATCH_COLS / CONV_VEC)
        {
            // Seed the accumulator with the bias, exactly as `int32_t acc = bias`.
            aie::accum<acc32, CONV_VEC> a;
            a.from_vector(aie::broadcast<int32_t, CONV_VEC>(bias), 0);

            // 3x3 MAC. The taps sit at column offsets c, c+1, c+2 in the padded
            // row buffer, so they are unaligned by construction — the sliding
            // window is what makes this a convolution. load_unaligned_v handles
            // that directly; aligned loads plus shuffles would be faster but the
            // first version should be obviously correct.
            //
            // CONV_IN_CH is fixed at 1 (grayscale). The static_assert below is a
            // guard, not a limitation being asserted away: RGB needs more than a
            // loop bound here (interleaved 3x row stride, a joint Stage-A
            // normalization, and the weight-buffer relayout) — see the RGB
            // section in CLAUDE.md.
            static_assert(CONV_IN_CH == 1, "vectorized path is grayscale-only; "
                                           "see the RGB section in CLAUDE.md");

            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_top + c),     w00);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_top + c + 1), w01);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_top + c + 2), w02);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_mid + c),     w10);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_mid + c + 1), w11);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_mid + c + 2), w12);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_bot + c),     w20);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_bot + c + 1), w21);
            a = aie::mac(a, aie::load_unaligned_v<CONV_VEC>(row_bot + c + 2), w22);

            aie::vector<int32_t, CONV_VEC> sh =
                aie::downshift(a.to_vector<int32_t>(0), out_shift);

            // Nonlinearity. CONV_RELU=1 reproduces the shipped kernel exactly:
            //   >32767 -> 32767 ; <=0 -> 0 ; else pass
            // which is min(max(x,0),32767) for integers.
            // CONV_RELU=0 removes the half-wave rectifier and only saturates.
            // See the ReLU entry in CLAUDE.md Known Issues for why that is the
            // better setting; it is OFF by default because it changes numerics.
#if CONV_RELU
            aie::vector<int32_t, CONV_VEC> r = aie::min(aie::max(sh, 0), 32767);
#else
            aie::vector<int32_t, CONV_VEC> r = aie::min(aie::max(sh, -32768), 32767);
#endif

            // Stage B1, then the separable Hann with both >>15 floor shifts.
            aie::vector<int32_t, CONV_VEC> cen =
                aie::sub(r, aie::broadcast<int32_t, CONV_VEC>(mean_prev));
            cen = aie::min(aie::max(cen, -32768), 32767);

            aie::vector<int32_t, CONV_VEC> w1 =
                aie::downshift(aie::mul(cen, (int32_t)h_r).template to_vector<int32_t>(0), 15);
            // unpack, NOT cast_to: this must SIGN-EXTEND int16 -> int32 lane for
            // lane. cast_to reinterprets the bits and would silently halve the
            // lane count.
            aie::vector<int32_t, CONV_VEC> hc =
                aie::unpack(aie::load_unaligned_v<CONV_VEC>((const int16_t *)HTAB + c));
            aie::vector<int32_t, CONV_VEC> w2 =
                aie::downshift(aie::mul(w1, hc).template to_vector<int32_t>(0), 15);
            w2 = aie::min(aie::max(w2, -32768), 32767);

            // Emit cint16 {wnd, 0}. interleave_zip with a zero vector builds the
            // {re,im} pairs in one shot; storing 2*CONV_VEC int16 scalars instead
            // would put the store on the critical path.
            aie::accum<acc32, CONV_VEC> t;
            t.from_vector(w2, 0);
            const aie::vector<int16_t, CONV_VEC> re16 = t.template to_vector<int16_t>(0);
            const aie::vector<int16_t, CONV_VEC> zero =
                aie::zeros<int16_t, CONV_VEC>();
            const auto zp = aie::interleave_zip(re16, zero, 1);

            int16_t *dst = (int16_t *)(out + o);
            aie::store_unaligned_v(dst,             zp.first);
            aie::store_unaligned_v(dst + CONV_VEC,  zp.second);
            o += CONV_VEC;
        }

#else   // scalar reference implementation

        for (int c = 0; c < PATCH_COLS; c++)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS, PATCH_COLS)
        {
            const int c1 = c + 1;   // column index in padded buffer

            // 3×3 MAC: 9 int8 × int8 → accumulated into int32
            int32_t acc = bias;
            acc += (int32_t)w00 * row_top[c1 - 1];
            acc += (int32_t)w01 * row_top[c1];
            acc += (int32_t)w02 * row_top[c1 + 1];
            acc += (int32_t)w10 * row_mid[c1 - 1];
            acc += (int32_t)w11 * row_mid[c1];
            acc += (int32_t)w12 * row_mid[c1 + 1];
            acc += (int32_t)w20 * row_bot[c1 - 1];
            acc += (int32_t)w21 * row_bot[c1];
            acc += (int32_t)w22 * row_bot[c1 + 1];

            // Scale int32 acc → int16
            int32_t shifted = acc >> out_shift;
            int16_t out16;
            if      (shifted >  32767) out16 =  32767;
            else if (shifted <=     0) out16 =  0;        // ReLU
            else                       out16 = (int16_t)shifted;

            // Stage B1: remove the previous frame's mean BEFORE the window.
            // Order matters — w*(f-µ) is what we want; windowing first and
            // subtracting after would leave µ smeared across the spectrum by W.
            // Output is now signed, so the negative window clamp below is live.
            int32_t centred = (int32_t)out16 - mean_prev;
            if      (centred >  32767) centred =  32767;
            else if (centred < -32768) centred = -32768;

            // Separable Hanning window (Q1.15 × Q1.15 → int16)
            int16_t h_c   = HTAB[c];
            int32_t wnd   = (centred * h_r) >> 15;
            wnd           = (wnd * h_c) >> 15;
            int16_t wnd16;
            if      (wnd >  32767) wnd16 =  32767;
            else if (wnd < -32768) wnd16 = -32768;
            else                   wnd16 = (int16_t)wnd;

            out[o].real = wnd16;
            out[o].imag = 0;
            ++o;
        }

#endif  // CONV_VECTORIZE

        ++rows_out;
        if (rows_out >= PATCH_ROWS) rows_out = 0;   // patch complete — rearm
    }
#endif  // CONV2D_ECHO_TEST
}
