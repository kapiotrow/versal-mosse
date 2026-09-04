/*
 * roi_crop.cpp
 * Extracts a ROI from a DDR frame, applies the MOSSE preprocessing chain, and
 * streams int8 samples to AIE via the PatchIn PLIO.
 *
 * See roi_crop.h for the stage list and for the meaning of ROI_NORM_Q.
 *
 * Frame layout: linear uint8 array, row-major, frame_rows × frame_cols.
 *
 * Why two passes: the mean and the L2 norm are global reductions over the
 * patch, so no output sample can be emitted until every input sample has been
 * seen. Pass 1 resamples into a BRAM scratch buffer and accumulates Σx and Σx²;
 * pass 2 re-reads the buffer and emits normalized int8.
 *
 * ---------------------------------------------------------------------------
 * COST — MEASURED, not scheduled: an hw_emu VCD probe at 64×64, recompute=1,
 * 312.5 MHz, 2026-08-17. hw_emu WALL time is meaningless, but hw_emu SIMULATED
 * PL CYCLES are RTL-accurate and transfer directly. Claims A-05, P-03.
 * These supersede the II=1 claim this comment used to make, which was the
 * pragma's request rather than what the loops achieve:
 *
 *   PASS1   44,600 cyc   142.7 µs   10.9 cyc / output px   (asked II=1, got ~11)
 *   NORM     4,100 cyc    13.1 µs    1.0 cyc / element     (asked II=1, got it)
 *   PASS2   27,900 cyc    89.2 µs   27.2 cyc / AXIS beat   (asked II=1, got ~27)
 *   total   76,725 cyc   245.5 µs   ap_start -> ap_done
 *
 * The pattern is worth remembering: NORM_LOOP is the only one of the three that
 * touches nothing but BRAM, and it is the only one that hits II=1. PASS1 misses
 * by ~11x on m_axi read latency and PASS2 by ~27x on AXIS backpressure from the
 * AIE. A PIPELINE II=1 pragma bounds the datapath, not the interfaces.
 *
 * Scaled to 128×128 that is ~1 ms for recompute=1 and ~0.36 ms for recompute=0.
 * The host measured 277 ms and 492 ms respectively for the same calls, so
 * ~99.7% of roi_crop's apparent cost is NOT this kernel — it is XRT's blocking
 * crop_run.wait(). ap_done asserts in the same cycle as TLAST; there is zero
 * completion latency in the PL. Do not optimise this file for frame rate until
 * that is fixed. See "Frame time" in CLAUDE.md.
 *
 *   interface     m_axi to the DDR frame buffer; a 32-bit AXIS to the PatchIn
 *                 PLIO; all geometry over AXI-Lite at RUN TIME, so a ROI or box
 *                 change needs no rebuild of anything but the host ELF. Driven
 *                 as a user-managed CU (ROI_CROP_USER_MANAGED=1) because a KDS
 *                 launch costs ~503 ms on this platform — claim P-03.
 *   memory        one BRAM scratch patch (16 KB at ROI_IN_CH=1, 48 KB at 3).
 *                 sum_x2 peaks at 2.111e14 against ap_uint<48>'s 2.815e14, so a
 *                 FOURTH plane would not fit; the reference model asserts the width.
 *   PL resources  ROUTED, shipping build: this kernel is LUT 5983, FF 7738,
 *                 DSP 53, 13 BRAM tiles (= 26 BRAM18) — nearly all of the
 *                 design's PL cost. Whole routed design incl. the base platform:
 *                 LUT 10527 of 520704, FF 13252 of 1041408, DSP 56 of 1312,
 *                 BRAM18 26 of 1200. docs/thesis/results/resources.csv
 *                 (build=rgb_l1relu). The figures this line carried until
 *                 2026-09-04 (LUT 7694 / FF 7539 / DSP 44 / BRAM18 10, "the
 *                 whole design") were THIS FILE's HLS csynth ESTIMATE on an
 *                 hw_emu single-channel gray build — not a routed number and not
 *                 the whole design. See embedded_comparison.md sec.6.
 *   bilinear      Still NEVER EXERCISED ON HARDWARE: every build to date has
 *                 roi_h == patch_rows, which collapses the interpolator to a copy.
 *                 Real ground-truth boxes would activate it for the first time.
 *                 Bit-exactness (make test_roi_crop, 25 cases, zero tolerance)
 *                 says nothing about timing — that suite passed throughout the
 *                 period when the launch path cost 98% of the frame.
 *
 * HLS notes:
 * - The only divisions are the two step computations and the mean, all outside
 *   any pipeline, so no divider is instantiated in the datapath.
 * - The reciprocal of sigma is computed once per patch in float and converted
 *   to Q16.16, so pass 2 uses integer multiplies only.
 *
 * @thesis sec:architekturaSystemu | A-01,A-05 | The PL half of the split: DDR frame -> bilinear
 *   resample -> Stage A preprocessing -> int8 -> AXIS to the AIE array.
 */

#include "roi_crop.h"
#include <cmath>

// Precondition checks. Compiled out for synthesis, so they cost nothing in
// hardware and fire in the native harness (make test_roi_crop), which is the only
// place a bad geometry can be caught before it reaches a ~50 min hw_emu frame.
//
// Each guards a real hole. None of these was checked before 2026-08-16, and the
// first is the worst: PASS2_COL steps by 4 and reads patch_buf[... + i] for
// i in 0..3 unconditionally, while total_beats = n_elems >> 2 undercounts, so a
// patch_cols that is not a multiple of 4 BOTH reads past the row and never
// asserts word.last — and a PLIO that never sees last stalls the whole graph.
#ifndef __SYNTHESIS__
#  include <cassert>
#  define ROI_ASSERT(cond, msg) assert((cond) && (msg))
#else
#  define ROI_ASSERT(cond, msg) ((void)0)
#endif

// log(1+v) mapped to [0, 65535], v = 0..255.
//
// Bolme §3.1: "the pixel values are transformed using a log function which
// helps with low contrast lighting situations." Only the monotone shape
// matters here — the affine scale is removed by the zero-mean/unit-norm steps
// that follow, so the table is normalized to fill uint16 for precision.
static const ap_uint<16> LOG_LUT[256] = {
#define L(v) (ap_uint<16>)(v)
    L(    0), L( 8192), L(12984), L(16384), L(19021), L(21176), L(22998), L(24576),
    L(25968), L(27213), L(28339), L(29368), L(30314), L(31189), L(32005), L(32768),
    L(33484), L(34160), L(34798), L(35405), L(35981), L(36531), L(37056), L(37559),
    L(38042), L(38505), L(38951), L(39381), L(39796), L(40197), L(40584), L(40959),
    L(41323), L(41676), L(42018), L(42351), L(42675), L(42990), L(43297), L(43597),
    L(43888), L(44173), L(44451), L(44723), L(44989), L(45248), L(45502), L(45751),
    L(45995), L(46234), L(46468), L(46697), L(46922), L(47143), L(47360), L(47573),
    L(47782), L(47988), L(48190), L(48389), L(48584), L(48776), L(48965), L(49151),
    L(49334), L(49515), L(49693), L(49868), L(50040), L(50210), L(50378), L(50543),
    L(50706), L(50867), L(51026), L(51182), L(51337), L(51489), L(51640), L(51788),
    L(51935), L(52080), L(52224), L(52365), L(52505), L(52643), L(52780), L(52915),
    L(53048), L(53180), L(53311), L(53440), L(53568), L(53694), L(53819), L(53943),
    L(54066), L(54187), L(54307), L(54426), L(54543), L(54660), L(54775), L(54889),
    L(55002), L(55114), L(55225), L(55335), L(55444), L(55552), L(55659), L(55765),
    L(55870), L(55974), L(56077), L(56180), L(56281), L(56382), L(56481), L(56580),
    L(56678), L(56776), L(56872), L(56968), L(57063), L(57157), L(57250), L(57343),
    L(57435), L(57526), L(57617), L(57707), L(57796), L(57885), L(57972), L(58060),
    L(58146), L(58232), L(58317), L(58402), L(58486), L(58570), L(58653), L(58735),
    L(58817), L(58898), L(58979), L(59059), L(59139), L(59218), L(59296), L(59374),
    L(59452), L(59529), L(59605), L(59681), L(59757), L(59832), L(59906), L(59980),
    L(60054), L(60127), L(60200), L(60272), L(60344), L(60415), L(60486), L(60557),
    L(60627), L(60697), L(60766), L(60835), L(60904), L(60972), L(61039), L(61107),
    L(61174), L(61240), L(61306), L(61372), L(61438), L(61503), L(61568), L(61632),
    L(61696), L(61760), L(61823), L(61886), L(61949), L(62011), L(62073), L(62135),
    L(62196), L(62258), L(62318), L(62379), L(62439), L(62499), L(62558), L(62618),
    L(62676), L(62735), L(62793), L(62852), L(62909), L(62967), L(63024), L(63081),
    L(63138), L(63194), L(63250), L(63306), L(63362), L(63417), L(63472), L(63527),
    L(63582), L(63636), L(63690), L(63744), L(63798), L(63851), L(63904), L(63957),
    L(64010), L(64062), L(64114), L(64166), L(64218), L(64269), L(64321), L(64372),
    L(64422), L(64473), L(64523), L(64574), L(64624), L(64673), L(64723), L(64772),
    L(64821), L(64870), L(64919), L(64968), L(65016), L(65064), L(65112), L(65160),
    L(65207), L(65255), L(65302), L(65349), L(65396), L(65442), L(65489), L(65535)
#undef L
};

void roi_crop(
    const ap_uint<8>                  *frame_buf,
    hls::stream<ap_axiu<32,0,0,0>>   &patch_out,
    int  frame_rows,
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  roi_h,
    int  roi_w,
    int  patch_rows,
    int  patch_cols,
    int  recompute)
{
#pragma HLS INTERFACE m_axi     port=frame_buf  bundle=gmem0  depth=ROI_FRAME_DEPTH
#pragma HLS INTERFACE axis      port=patch_out
#pragma HLS INTERFACE s_axilite port=frame_rows bundle=control
#pragma HLS INTERFACE s_axilite port=frame_cols bundle=control
#pragma HLS INTERFACE s_axilite port=roi_row    bundle=control
#pragma HLS INTERFACE s_axilite port=roi_col    bundle=control
#pragma HLS INTERFACE s_axilite port=roi_h      bundle=control
#pragma HLS INTERFACE s_axilite port=roi_w      bundle=control
#pragma HLS INTERFACE s_axilite port=patch_rows bundle=control
#pragma HLS INTERFACE s_axilite port=patch_cols bundle=control
#pragma HLS INTERFACE s_axilite port=recompute  bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Holds the resampled patch (raw 8-bit) during Stage A, then the FINAL
    // quantized int8 samples, rewritten in place by the normalize pass. Keeping
    // the quantized result here is what lets recompute=0 skip straight to the
    // stream-out pass. static → persists across the frame's channel calls.
    static ap_uint<8> patch_buf[ROI_MAX_PATCH_BYTES];
#pragma HLS BIND_STORAGE variable=patch_buf type=ram_2p impl=bram

    ROI_ASSERT(patch_rows > 0 && patch_cols > 0,
               "patch dimensions must be positive (the step divisions below trap)");
    ROI_ASSERT((patch_cols & 3) == 0,
               "patch_cols must be a multiple of 4: PASS2 packs 4 bytes/beat and "
               "total_beats = n_bytes>>2, so otherwise word.last never asserts");
    ROI_ASSERT(((patch_cols * ROI_IN_CH) & 3) == 0,
               "a patch ROW must be a whole number of AXIS beats, or PASS2 would "
               "straddle rows; 3*patch_cols is fine whenever patch_cols is a "
               "multiple of 4, since gcd(3,4) = 1");
    ROI_ASSERT((long)patch_rows * (long)patch_cols * (long)ROI_IN_CH
                   <= (long)ROI_MAX_PATCH_BYTES,
               "patch exceeds ROI_MAX_PATCH_BYTES — patch_buf would overrun");
    ROI_ASSERT(roi_h > 0 && roi_w > 0,
               "roi extent must be positive: 0 collapses every row onto one "
               "source row, negative runs the sampler backwards");
    ROI_ASSERT(frame_rows > 0 && frame_cols > 0, "frame dimensions must be positive");

    const int n_elems    = patch_rows * patch_cols;   // PIXELS
    // SAMPLES: one per plane per pixel. Every reduction below is over n_samples,
    // not n_elems — that is what makes the normalization joint across planes.
    const int n_samples  = n_elems * ROI_IN_CH;
    const int row_bytes  = patch_cols * ROI_IN_CH;
    const int total_beats = n_samples >> 2;

    // Q8 source-coordinate step. Computed once, outside every pipeline, so no
    // divider lands in the datapath.
    //
    // Exact at every supported geometry: patch dims are powers of two <= 128, so
    // 256/patch_rows is an integer and step = roi_h * (256/patch_rows) divides
    // evenly. Truncation only becomes reachable if ROI_MAX_PATCH_ROWS is raised
    // past 256, which roi_crop_ref.py's step_is_exact() asserts against.
    const int step_y = (roi_h << ROI_FRAC_BITS) / patch_rows;
    const int step_x = (roi_w << ROI_FRAC_BITS) / patch_cols;

    // roi_row * ROI_FRAC_ONE, NOT roi_row << ROI_FRAC_BITS: the host computes
    // roi_row = pos_row - roi_h/2 and passes it as (uint32_t), so it is routinely
    // negative, and left-shifting a negative signed value is undefined behaviour
    // before C++20 — which the native harness compiles as (-std=c++17). The
    // multiply is well defined and generates the same hardware.
    const int base_y = roi_row * ROI_FRAC_ONE;
    const int base_x = roi_col * ROI_FRAC_ONE;

    const int max_y = frame_rows - 1;
    const int max_x = frame_cols - 1;

    ap_uint<48> sum_x  = 0;
    ap_uint<48> sum_x2 = 0;

    // Stage A (passes 1 and 1b) depends only on the frame and the ROI, not on
    // the channel, so it runs once per frame. See the note in roi_crop.h.
    if (recompute) {

    // ---------------------------------------------------------------------
    // Pass 1: bilinear resample into patch_buf, accumulate Σlog(x), Σlog(x)²
    //
    // MEASURED 10.9 cycles per output pixel, not the II=4 this comment used to
    // estimate and not the II=1 the pragma asks for. Bilinear needs 4 scattered
    // reads per output pixel through a single m_axi port; HLS serialises them
    // and each carries DDR read latency that no amount of pipelining hides.
    //
    // Left as-is deliberately. It runs once per FRAME (recompute=1 on channel 0
    // only), so 128×128 costs ~571 µs — against a frame that is currently 8.26 s
    // and would still be ~500 ms with the crop_run.wait() fix in. Forcing II=1
    // would need source-row line buffers or a wider/split AXI port; revisit only
    // if the frame ever gets close to 33 ms, and re-measure first.
    // ---------------------------------------------------------------------
// @thesis subsec:przetwarzanieWstepne | A-05 | The bilinear resample and the two reduction passes;
//   measured cycle counts for each pass are in the file header.
PASS1_ROW:
    for (int r = 0; r < patch_rows; ++r) {
        const int    sy  = base_y + r * step_y;
        const int    fy  = sy & (ROI_FRAC_ONE - 1);
        int          y0  = sy >> ROI_FRAC_BITS;
        int          y1  = y0 + 1;
        // Clamp to the frame: a target near an edge replicates the border
        // instead of reading outside frame_buf.
        y0 = (y0 < 0) ? 0 : ((y0 > max_y) ? max_y : y0);
        y1 = (y1 < 0) ? 0 : ((y1 > max_y) ? max_y : y1);

        // PIXEL index of the two source rows, not a byte offset — the
        // * ROI_IN_CH that converts to interleaved bytes is applied at the tap,
        // so this stays common to all three planes. At 1 plane it vanishes.
        const int row0_base = y0 * frame_cols;
        const int row1_base = y1 * frame_cols;

    PASS1_COL:
        for (int c = 0; c < patch_cols; ++c) {
#pragma HLS PIPELINE II=1
            const int sx = base_x + c * step_x;
            const int fx = sx & (ROI_FRAC_ONE - 1);
            int       x0 = sx >> ROI_FRAC_BITS;
            int       x1 = x0 + 1;
            x0 = (x0 < 0) ? 0 : ((x0 > max_x) ? max_x : x0);
            x1 = (x1 < 0) ? 0 : ((x1 > max_x) ? max_x : x1);

            // The four tap ADDRESSES are shared by every plane — same geometry,
            // same fx/fy weights — so only the +p byte offset changes. Unrolled,
            // so the three planes are 3 parallel datapaths against one m_axi
            // port, and the 3 bytes of one source pixel are contiguous.
        PASS1_PLANE:
            for (int p = 0; p < ROI_IN_CH; ++p) {
#pragma HLS UNROLL
                const ap_uint<8> p00 = frame_buf[(row0_base + x0) * ROI_IN_CH + p];
                const ap_uint<8> p01 = frame_buf[(row0_base + x1) * ROI_IN_CH + p];
                const ap_uint<8> p10 = frame_buf[(row1_base + x0) * ROI_IN_CH + p];
                const ap_uint<8> p11 = frame_buf[(row1_base + x1) * ROI_IN_CH + p];

                // Q8 bilinear. top/bot ≤ 255·256 (17 bits); val ≤ 255·256·256
                // (25 bits) before the >>16.
                const ap_uint<18> top = p00 * (ROI_FRAC_ONE - fx) + p01 * fx;
                const ap_uint<18> bot = p10 * (ROI_FRAC_ONE - fx) + p11 * fx;
                const ap_uint<27> val = top * (ROI_FRAC_ONE - fy) + bot * fy;
                const ap_uint<8>  pix = (ap_uint<8>)(val >> (2 * ROI_FRAC_BITS));

                // INTERLEAVED in the scratch buffer, because that is what the
                // AXIS wire carries and what conv2d unpacks. PASS2 then stays a
                // linear byte read.
                patch_buf[(r * patch_cols + c) * ROI_IN_CH + p] = pix;

                // ONE pair of accumulators for all planes — the joint statistic.
                const ap_uint<16> lv = LOG_LUT[pix];
                sum_x  += lv;
                sum_x2 += (ap_uint<32>)lv * (ap_uint<32>)lv;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Statistics: mean and 1/σ, once per patch
    // ---------------------------------------------------------------------
    // sum_x  <= 3 * 16384 * 65535       = 3.22e9   (32 bits)
    // sum_x2 <= 3 * 16384 * 65535^2      = 2.11e14  vs ap_uint<48> = 2.81e14.
    // RGB fits with 33% headroom; a fourth plane would NOT, which is why the
    // reference asserts the width rather than assuming it.
    const ap_uint<32> mean = (ap_uint<32>)(sum_x / n_samples);

    // var = E[x²] - E[x]².  Both terms fit comfortably; the subtraction is
    // exact in integer arithmetic because sum_x2 and mean are exact.
    const ap_uint<48> mean_sq = (ap_uint<48>)mean * (ap_uint<48>)mean;
    const ap_uint<48> ex2     = sum_x2 / n_samples;
    const ap_uint<48> var     = (ex2 > mean_sq) ? (ap_uint<48>)(ex2 - mean_sq) : (ap_uint<48>)0;

// @thesis subsec:przetwarzanieWstepne | A-05 | Stage A's normalization: one mean and one inverse sigma in
//   Q16.16, joint over all planes at ROI_IN_CH=3 -- per-plane statistics would delete the
//   chromatic contrast RGB exists for.
    // inv_q = ROI_NORM_Q / σ in Q16.16.
    int inv_q;
    if (var == 0) {
        inv_q = 0;                      // flat patch → emit all zeros
    } else {
        // sqrtf (not hls::rsqrtf): rsqrtf has no C-simulation implementation, so
        // using it would make this kernel un-testable outside the HLS tool.
        const float invsig = 1.0f / std::sqrt((float)var);
        const float scaled = invsig * (float)ROI_NORM_Q * 65536.0f;
        inv_q = (scaled >= (float)ROI_INV_Q_MAX) ? ROI_INV_Q_MAX : (int)scaled;
    }

    // ---------------------------------------------------------------------
    // Pass 1b: normalize in place. patch_buf holds raw resampled pixels on
    // entry and the final quantized int8 samples on exit, so the stream-out
    // pass below becomes a pure buffer read.
    //   q = clip(round((log(x) - mean) · inv_q >> 16), ±127)
    // ---------------------------------------------------------------------
    // Every sample of every plane gets the SAME mean and inv_q — see the joint
    // normalization note in roi_crop.h. The loop is over bytes, so it does not
    // need to know the interleaving.
NORM_LOOP:
    for (int i = 0; i < n_samples; ++i) {
#pragma HLS PIPELINE II=1
        const ap_uint<8> pix = patch_buf[i];
        const ap_int<18> dev = (ap_int<18>)LOG_LUT[pix] - (ap_int<18>)mean;
        // dev ≤ ±2^16, inv_q ≤ 2^20 → product ≤ 2^36, so 48 bits is safe.
        const ap_int<48> prod = (ap_int<48>)dev * (ap_int<48>)inv_q;
        // Round-to-nearest before the >>16.
        const ap_int<48> rnd  = (prod + (1 << 15)) >> 16;

        ap_int<8> q;
        if      (rnd >  127) q =  127;
        else if (rnd < -127) q = -127;
        else                 q = (ap_int<8>)rnd;

        patch_buf[i] = (ap_uint<8>)q;   // reinterpret: same 8 bits, signed
    }

    }   // end if (recompute)

    // ---------------------------------------------------------------------
    // Pass 2: stream the quantized patch out, 4 int8 per 32-bit AXIS beat.
    // Runs on every call, including recompute=0.
    //
    // MEASURED 27.2 cycles per beat, not the II=1 the pragma asks for. This loop
    // reads BRAM and writes AXIS, so the only thing it can be waiting on is
    // TREADY: the AIE accepts a beat about every 87 ns, while conv2d's compiler
    // schedule (~8.75 cyc/px at 1 GHz, 4 px per beat) implies ~35 ns. So the AIE
    // consumes ~2.5x slower than its schedule and roi_crop is backpressured for
    // roughly 26 of every 27 cycles here.
    //
    // That is real and it is the thing that will matter once the host-side
    // crop_run.wait() cost is removed — but it is 89 µs against 505 ms today, so
    // it is not the bottleneck and must not be treated as one. If it does become
    // one, the lever is conv2d's stream-read loop (CLAUDE.md records it as 44%
    // of that kernel and never vectorised), not this loop.
    // ---------------------------------------------------------------------
    int beat = 0;
PASS2_ROW:
    for (int r = 0; r < patch_rows; ++r) {
    PASS2_COL:
        for (int c = 0; c < row_bytes; c += 4) {
#pragma HLS PIPELINE II=1
            ap_axiu<32,0,0,0> word;
            word.keep = (ap_uint<4>)-1;
            word.strb = (ap_uint<4>)-1;
            word.last = (beat == total_beats - 1) ? 1 : 0;

        PASS2_PACK:
            for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
                word.data.range(8*i + 7, 8*i) = patch_buf[r * row_bytes + c + i];
            }

            patch_out.write(word);
            beat++;
        }
    }
}
