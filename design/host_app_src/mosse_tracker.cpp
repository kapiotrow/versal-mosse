/*
 * mosse_tracker.cpp
 * Host application: MOSSE correlation filter tracker on CNN features.
 *
 * Runs on the A72 processor using XRT APIs to orchestrate:
 *   - camera_capture  (PL): fill DDR frame buffer (stub: zero-fill)
 *   - roi_crop        (PL): extract patch → AIE PatchIn PLIO
 *   - MOSSE_graph     (AIE): conv2d + FFT2D + cmul_accum + IFFT2D
 *
 * APU manages all inter-AIE-stage data via GMIO (DDR round-trips):
 *   For ch = 0..N_CHANNELS-1:
 *     1. Start roi_crop → patch → AIE → conv2d → fft_rows → gmio_fft_row_out
 *     2. APU: transpose_inplace() on row-FFT output in DDR
 *     3. APU: write transposed data via gmio_fft_col_in → fft_cols → cmul_accum
 *     4. APU: read partial accumulation from gmio_accum_out
 *   After all channels:
 *     5. APU: write accum to gmio_ifft_row_in → IFFT rows
 *     6. APU: transpose_inplace() on IFFT row output
 *     7. APU: write transposed data via gmio_ifft_col_in → IFFT cols → gmio_response
 *     8. APU: peak_detect_sw() → displacement → update position
 *     9. APU: filter_init() on frame 0, filter_update() thereafter (mosse_filter.h)
 *
 * The filter maths lives in mosse_filter.{h,cpp}, which includes no XRT header so
 * `make test_host` can check it natively against a NumPy golden in seconds.
 *
 * F_ch reaches the host through gmio_fft_col_out, a broadcast tap on the column
 * FFT added for exactly this purpose — before it, the host could see the
 * half-transformed row FFT and the accumulated Σ H*⊙F but never F_ch itself.
 *
 * TODO: add OpenCV or V4L2 video capture loop.
 * TODO: affine perturbations for initialisation (Bolme §3.4); this is the N=1 case.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <thread>          // std::this_thread::yield() in rc_poll_until_done()

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_aie.h"
#include "experimental/xrt_aie.h"
#include "experimental/xrt_ip.h"   // user-managed roi_crop — see CropIp below

#include "mosse_filter.h"

// -----------------------------------------------------------------------
// Build-time constants (set via Makefile -D flags)
// -----------------------------------------------------------------------
#ifndef PATCH_ROWS
#  define PATCH_ROWS  128
#endif
#ifndef PATCH_COLS
#  define PATCH_COLS  128
#endif
#ifndef N_CHANNELS
#  define N_CHANNELS  16
#endif
#ifndef FFT_COL_WS
#  define FFT_COL_WS  2   // must match fft_graph.h FFT_COL_WS
#endif
#ifndef FFT_ROW_WS
#  define FFT_ROW_WS  2   // must match fft_graph.h FFT_ROW_WS
#endif
#ifndef ITER_CNT
#  define ITER_CNT    1
#endif
// Offset of the synthetic test impulse from the tracked position, in frame
// pixels. A correct pipeline must report exactly this displacement. Non-zero,
// asymmetric and opposite-signed on purpose: (0,0) is indistinguishable from a
// zero response, equal magnitudes would not catch a row/col transpose, and
// same-signed values would not catch a sign flip.
// Must stay within ±PATCH_ROWS/2 and ±PATCH_COLS/2 or the impulse falls outside
// the cropped patch entirely.
#ifndef IMPULSE_DR
#  define IMPULSE_DR  10
#endif
#ifndef IMPULSE_DC
#  define IMPULSE_DC  (-7)
#endif
static_assert(IMPULSE_DR >  -PATCH_ROWS/2 && IMPULSE_DR < PATCH_ROWS/2,
              "IMPULSE_DR puts the test impulse outside the cropped patch");
static_assert(IMPULSE_DC >  -PATCH_COLS/2 && IMPULSE_DC < PATCH_COLS/2,
              "IMPULSE_DC puts the test impulse outside the cropped patch");
// Occlusion injection, for testing the PSR gate. Bitmask over frame index: bit f
// set => frame f gets a checkerboard instead of the target, i.e. the object is
// behind something. A MASK rather than a single index because the interesting
// test is CONSECUTIVE occlusion (does it still reacquire after two gated
// frames?), which a mask expresses without a second parameter.
//
// Bit 0 is ignored: frame 0 trains the filter, so occluding it produces a filter
// trained on a checkerboard — not a test of the gate, just a broken run.
//
// The occlude-then-reacquire test is:  ITER_CNT=3 OCCLUDE_MASK=0x2
#ifndef OCCLUDE_MASK
#  define OCCLUDE_MASK  0
#endif
#ifndef OCCLUDE_SQUARE
#  define OCCLUDE_SQUARE  8
#endif
// Target box size in FRAME pixels. The tracker's state was pos_row/pos_col only
// until 2026-08-16; see the TargetBox note in mosse_filter.h for what that cost.
//
// 64 is chosen so that at TARGET_PADDING=2 the ROI is 128x128 — EXACTLY the
// geometry that shipped before the box existed. The resample therefore stays 1:1
// and roi_crop's bilinear interpolator stays dormant, which makes adopting the
// box a single-variable change rather than two at once.
//
// TARGET_SIZE also drives the SHAPE inject_target_frame draws, so the declared
// box and the drawn object agree by construction. Before this they did not: the
// shipped target was ~11x11 inside a 128x128 ROI, i.e. an effective padding of
// ~11.6, so any sigma anchored to a declared 64 px box would have been anchored
// to a fiction.
#ifndef TARGET_H
#  define TARGET_H  64
#endif
#ifndef TARGET_W
#  define TARGET_W  64
#endif
// Background for the synthetic test frame.
//   1 (default) = band-limited texture   0 = the flat BACKGROUND=40 fill
//
// This is not cosmetic. Padding exists so the filter can learn
// target-vs-background (Bolme §3.1, Danelljan §3.1); against a flat fill there is
// no background to learn, so more padding is strictly less target and any padding
// comparison is decided before it runs. The flat fill remains reachable for
// reproducing pre-2026-08-16 runs.
#ifndef FRAME_TEXTURE
#  define FRAME_TEXTURE  1
#endif
// Per-frame sensor noise, PEAK amplitude in LSB. 0 disables it.
//
// WHY THIS EXISTS. fill_background() already had a sensor-noise term
// (`3.0 * (next() - 0.5)`), but it was drawn ONCE and cached with the rest of the
// field, so the "noise" was byte-identical on every frame. Combined with
// dirty-rect restore, that made the background outside the target a perfectly
// repeating signal — and a correlation filter fed a perfectly repeating
// background correlates with it at exactly zero shift.
//
// Measured on the 2026-08-17 board run (128x128, ch16, 4-3-3, TRAJECTORY=1):
// the response carried TWO peaks every frame — the true motion peak at ~(9,-3),
// and a static-background peak at (0,0)/(-1,0)/(-1,-1) sitting at 69-86% of it.
// The static one won 21 of 48 frames, and because MOSSE measures only RELATIVE
// displacement, each win cost a permanent ~9.4 px offset. Centre error went
// 1.35 -> 9.56 -> 87 -> 292 px in steps, with PSR reading 24-35 the whole time:
// the response really was sharply peaked, just in the wrong place. Neither the
// PSR gate nor `err=0 px` can see this failure mode; only IoU and the
// resp00_over_peak column in the CSV can.
//
// Real video never does this — sensor noise and camera motion guarantee the
// background does not repeat to the LSB. So this is the harness being
// unrepresentative, not the tracker being wrong, and the minimal honest fix is
// to re-draw the noise term that was already there once per frame.
//
// 2 LSB peak sits in the 1-3 LSB range typical of real 8-bit sensor noise.
// NOTE this is the one test-sequence default that does NOT reproduce the
// previous behaviour: FRAME_NOISE=0 restores it, for reproducing runs before
// 2026-08-17 or for deliberately re-testing the pathological case.
#ifndef FRAME_NOISE
#  define FRAME_NOISE  2
#endif
#ifndef FRAME_ROWS
#  define FRAME_ROWS  1080
#endif
#ifndef FRAME_COLS
#  define FRAME_COLS  1920
#endif
// conv2d INT8 weights, produced by `make weights` and packaged onto the SD card
// next to the ELF (run_script.sh runs from that directory).
#ifndef WEIGHTS_FILE
#  define WEIGHTS_FILE  "layer0_weights.bin"
#endif

// -----------------------------------------------------------------------
// Buffer sizes
// -----------------------------------------------------------------------
constexpr size_t PATCH_ELEMS       = PATCH_ROWS * PATCH_COLS;
constexpr size_t FFT_BYTES         = PATCH_ELEMS * 4;           // cint16 = 4 B/sample
constexpr size_t FILTER_BYTES      = PATCH_ELEMS * 4;           // per channel
constexpr size_t ACCUM_BYTES       = PATCH_ELEMS * 4;
constexpr size_t CMUL_IN_BYTES     = PATCH_ELEMS * 4 * 2;       // [filter|accum] interleaved by chunk
constexpr int    CMUL_CHUNK_INT16  = PATCH_COLS * FFT_COL_WS * 2; // int16_t per half-chunk
constexpr int    CMUL_N_CHUNKS     = PATCH_ROWS / FFT_COL_WS;
constexpr size_t RESP_BYTES        = PATCH_ELEMS * 4;
constexpr size_t FRAME_BYTES       = (size_t)FRAME_ROWS * FRAME_COLS;  // single-channel grayscale uint8
// conv2d weights: 3×3×3 INT8 = 27 bytes, padded to 64-byte GMIO alignment
constexpr size_t WEIGHT_CH_BYTES   = 64;
// conv2d emits one row-FFT window (PATCH_ROWS*FFT_ROW_WS samples) per invocation,
// so it fires this many times per patch. Its `weights` input_buffer is consumed
// once per invocation, so the host must send the weight buffer once per firing.
constexpr int    CONV_OUT_CHUNK    = PATCH_ROWS * FFT_ROW_WS;
constexpr int    CONV_INVOCATIONS  = (int)PATCH_ELEMS / CONV_OUT_CHUNK;

// -----------------------------------------------------------------------
// AIE→DDR GMIO drain granularity
// -----------------------------------------------------------------------
// aie2gm_nb() moves ONE producing-kernel invocation per call — NOT the byte
// count handed to it. A single async() for a whole buffer therefore drains only
// the first window; the producer then blocks on a full output window and the
// backpressure propagates all the way to roi_crop, which never asserts ap_done.
// Measured in hw_emu 2026-08-01: with one async() for all of gmio_fft_row_out,
// the design stalled after 6 of 64 weight buffers and the AIE DMA status register
// froze for 243k consecutive polls.
//
// Every AIE→DDR GMIO must be drained one async/wait pair per invocation, with a
// chunk equal to the producer's output window (cint16 = 4 B/sample):
//   gmio_fft_row_out  <- FFTrows  window FFT_ROW_TP_WINDOW_VSIZE = PATCH_ROWS*FFT_ROW_WS
//   gmio_accum_out    <- cmul     dimensions                     = PATCH_COLS*FFT_COL_WS
//   gmio_ifft_row_out <- IFFTrows window FFT_ROW_TP_WINDOW_VSIZE
//   gmio_response     <- IFFTcols window FFT_COL_TP_WINDOW_VSIZE
constexpr size_t FFT_SAMPLE_BYTES  = 4;                                        // cint16
constexpr size_t ROW_CHUNK_BYTES   = (size_t)PATCH_ROWS * FFT_ROW_WS * FFT_SAMPLE_BYTES;
constexpr size_t COL_CHUNK_BYTES   = (size_t)PATCH_COLS * FFT_COL_WS * FFT_SAMPLE_BYTES;
constexpr int    ROW_CHUNKS        = (int)(FFT_BYTES   / ROW_CHUNK_BYTES);
constexpr int    COL_CHUNKS        = (int)(ACCUM_BYTES / COL_CHUNK_BYTES);

static_assert(FFT_BYTES   % ROW_CHUNK_BYTES == 0,
              "row-FFT buffer is not a whole number of kernel invocations");
static_assert(ACCUM_BYTES % COL_CHUNK_BYTES == 0,
              "col/accum buffer is not a whole number of kernel invocations");
static_assert(RESP_BYTES  % COL_CHUNK_BYTES == 0,
              "response buffer is not a whole number of kernel invocations");
// conv2d fires once per row-FFT window, which is what lets the weights feed and
// the row-FFT drain interleave 1:1 in a single loop below.
static_assert(ROW_CHUNKS == CONV_INVOCATIONS,
              "row-FFT drain count must match conv2d firing count");

// -----------------------------------------------------------------------
// GMIO transaction cost instrumentation
// -----------------------------------------------------------------------
// The open question this answers: the per-frame async/wait count is ~6500 at 16
// channels, and at a plausible 2-10 µs of driver cost each that is 13-65 ms —
// which would blow the whole 33 ms budget and outweigh every compute-placement
// decision in the design. It has never been measured. This measures it, at the
// cost of two clock reads per transaction.
//
// Deliberately NOT a wrapper around the GMIO objects: the async/wait ORDER in the
// loops below is load-bearing (three documented deadlocks came from getting it
// wrong), so the macros leave every call site textually where it was and only
// bracket it with a timer. `DMA_TX` also counts the transaction and is used on
// `async` only; `DMA_T` adds time without counting and is used on `wait`. So
// `calls` is the transaction count and `us` is total host time in async+wait.
enum {
    DMA_WEIGHTS, DMA_FFT_ROW_OUT, DMA_FFT_COL_IN, DMA_FFT_COL_OUT, DMA_CMUL_IN,
    DMA_ACCUM_OUT, DMA_IFFT_ROW_IN, DMA_IFFT_ROW_OUT, DMA_IFFT_COL_IN,
    DMA_RESPONSE, DMA_N
};

struct DmaStat {
    const char   *name;
    unsigned long calls;
    double        us;
};

static DmaStat g_dma[DMA_N] = {
    {"gmio_weights",      0, 0.0}, {"gmio_fft_row_out",  0, 0.0},
    {"gmio_fft_col_in",   0, 0.0}, {"gmio_fft_col_out",  0, 0.0},
    {"gmio_cmul_in",      0, 0.0}, {"gmio_accum_out",    0, 0.0},
    {"gmio_ifft_row_in",  0, 0.0}, {"gmio_ifft_row_out", 0, 0.0},
    {"gmio_ifft_col_in",  0, 0.0}, {"gmio_response",     0, 0.0},
};
static DmaStat g_dma_total[DMA_N];

#define DMA_T(slot, stmt) do {                                                  \
        const auto _t0 = std::chrono::steady_clock::now();                       \
        stmt;                                                                    \
        g_dma[slot].us += std::chrono::duration<double, std::micro>(              \
            std::chrono::steady_clock::now() - _t0).count();                      \
    } while (0)

#define DMA_TX(slot, stmt) do {                                                 \
        DMA_T(slot, stmt);                                                        \
        ++g_dma[slot].calls;                                                      \
    } while (0)

static void dma_reset_frame(void)
{
    for (int i = 0; i < DMA_N; ++i) { g_dma[i].calls = 0; g_dma[i].us = 0.0; }
}

// -----------------------------------------------------------------------
// roi_crop launch-phase instrumentation
// -----------------------------------------------------------------------
// WHY THIS EXISTS. Timestamped console capture on 2026-08-17 put 478.7 ms x 16
// channels = 7.66 s in the interval between the "weights sent + row-FFT drained"
// and "roi_crop done" prints — 94% of an 8.18 s frame, against a design budget
// of 0.7 ms/frame for this kernel. That interval contained exactly one
// statement, crop_run.wait(), and nothing timed it.
//
// It is NOT the kernel's datapath, proven three ways:
//   - ch0 (recompute=1, 36864 loop iterations) took 277 ms; ch1-15 (recompute=0,
//     4096 iterations) took 492 ms. 9x the work, 1.8x FASTER.
//   - the drain loop below completes in 2-6 ms for every channel, and it cannot
//     complete until conv2d has consumed all 16384 patch pixels — so roi_crop
//     has written all 4096 AXIS beats within ~5 ms and has nothing left to do.
//   - Pass 2 alone is 4096 beats at II=1, 312.5 MHz = 13 us. The measured floor
//     is 277 ms, 21000x off. No HLS pathology spans four orders of magnitude.
//
// So the cost is in the launch/completion path. These four slots split it into
// construction, argument setting, submission and completion, which is the one
// thing the log could not distinguish. Same shape as DmaStat above, and the same
// reason for macros over a wrapper: the call sites stay textually where they are.
// RC_POLL vs RC_WAIT is THE discriminating measurement, and it is deliberately
// both in one run because hardware access is the scarce resource:
//
//   poll ~5 ms, wait ~500 ms  -> the CU finished long ago and XRT's BLOCKING
//                                path is the cost. The spin is then also the fix.
//   poll ~500 ms              -> the CU genuinely is not asserting ap_done, and
//                                the problem is in the PL, not the host.
//
// The spin is bounded so a misread state enum cannot hang an unattended run: on
// timeout it gives up, flags it, and falls through to wait() which is correct
// regardless. `yield` keeps a tight spin from starving the QEMU/simulator pair
// under hw_emu; on hardware it returns in ~100 ns when nothing else is runnable,
// which is three orders below anything being measured here.
// RC_WAIT2 is a SECOND wait() on the same, already-completed run. It exists to
// separate "wait() blocks until the command completes" from "wait() does work
// beyond waiting". Once poll() has returned a terminal state the command IS
// complete, so a correct wait() must return in microseconds. If RC_WAIT2 is also
// hundreds of ms, then the cost is per-CALL bookkeeping in the completion path
// and not a wait on anything at all — a different bug, with a different fix, and
// one that no amount of poll-vs-wait comparison can distinguish.
enum { RC_CTOR, RC_ARGS, RC_START, RC_POLL, RC_WAIT, RC_WAIT2, RC_N };

// Which roi_crop launch path is compiled in. Defined here rather than next to
// CropIp because the slot LABELS below depend on it, and a report that names
// the wrong mechanism is worse than no report — see the CropIp block for the
// measurements that made 1 the default.
#ifndef ROI_CROP_USER_MANAGED
#define ROI_CROP_USER_MANAGED 1
#endif

#if ROI_CROP_USER_MANAGED
// User-managed: RC_POLL is a spin on the CU's own ap_done bit, and the two
// wait slots are no-ops retained so the table shape stays comparable.
static const char *g_rc_name[RC_N] = {
    "crop_ip ctor", "crop_ip write args", "crop_ip ap_start",
    "crop_ip poll ap_done", "crop (no wait)", "crop (no wait #2)"
};
#else
static const char *g_rc_name[RC_N] = {
    "crop_run ctor", "crop_run set_arg", "crop_run start",
    "crop_run poll(state)", "crop_run wait", "crop_run wait #2"
};
#endif

// Spin bound, seconds. Generous: hw_emu host time is ~1000x hardware, and a
// spurious trip there would only mislabel a measurement that is meaningless in
// emulation anyway.
static constexpr double RC_POLL_MAX_S = 60.0;
static unsigned long    g_rc_poll_iters = 0;   // per frame
static unsigned long    g_rc_poll_timeouts = 0;

// True while the command has not reached a terminal state. Enumerated rather
// than using `!= COMPLETED`: SUBMITTED is 7, i.e. NUMERICALLY ABOVE COMPLETED(4),
// so any ordering comparison would exit the spin early and report a completion
// that has not happened.
static inline bool rc_pending(ert_cmd_state s)
{
    return s == ERT_CMD_STATE_NEW       || s == ERT_CMD_STATE_QUEUED
        || s == ERT_CMD_STATE_RUNNING   || s == ERT_CMD_STATE_SUBMITTED;
}
static double        g_rc_us[RC_N];
static unsigned long g_rc_n[RC_N];
static double        g_rc_us_total[RC_N];
static unsigned long g_rc_n_total[RC_N];

// -------- per-call detail, not just the frame mean --------
//
// WHY. A per-frame mean destroys the single most diagnostic feature of this
// measurement: whether the cost is QUANTIZED or DISPERSED. 16 calls all landing
// at 500.0 +- 0.1 ms is a periodic sleep/poll interval in the completion path —
// a host software constant, which nothing in the PL can produce. A spread of
// 200-800 ms is a cost that tracks data or contention. The two demand opposite
// next steps, and the mean is identical for both.
//
// It is also the only way to read the anomaly already on record: ch0 does 9x the
// loop iterations of ch1-15 and took 277 ms against their 492 ms. Faster with
// more work has no datapath explanation, but it is exactly what beating against
// a fixed tick looks like. Per-call values make the tick visible directly, as
// clustering at multiples of a base interval.
// -----------------------------------------------------------------------
// APU-side per-frame cost — the ~90 ms nobody has ever measured
// -----------------------------------------------------------------------
// WHY. After console gating the frame is 177 ms (runs/run_0820_1513.log) of
// which GMIO is a measured 87 ms and the roi_crop launch path a measured
// 0.085 ms. The remaining ~90 ms — half the frame — has only ever been
// ATTRIBUTED, never measured: "transposes + packing memcpy + filter update".
// CLAUDE.md records the cost of exactly that habit: the dumps were inferred to
// cost ~9.4 s and measured 2 s.
//
// THE RULE THIS FOLLOWS: measure the TOTAL and print the RESIDUAL. A profiler
// that does not account for the whole frame lets you conclude confidently and
// wrongly. AP_TOTAL is wall time across the whole frame body; the report prints
// total - (GMIO + roi_crop + every AP_ slot) as an explicit "unattributed" line.
// If that line is large, the breakdown below is not the answer and says so.
enum {
    AP_SCENE,        // inject_target_frame + scene_restore + BG_PAN
    AP_FRAME_PUSH,   // memcpy g_frame_host -> frame_bo (2 MB)
    AP_FRAME_SYNC,   // frame_bo.sync host->device (2 MB)
    AP_TRANSPOSE,    // transpose_inplace, 17x 64 KB per frame
    AP_WINMEAN,      // measure_window_mean + the Parseval energy loop
    AP_FCOL_SYNC,    // fcol_bo.sync(FROM_DEVICE), 64 KB
    AP_BO_STAGE,     // bulk memcpy between a BO mapping and the heap staging
                     // buffers. Given its OWN slot rather than folded into the
                     // callers: this is the cost the copy pattern ADDS, and a
                     // fix whose overhead is invisible cannot be evaluated.
    AP_UNPACK,       // unpack_spectrum: cint16 col-FFT -> cfloat row-major
    AP_CMUL_PACK,    // [H|accum] packing memcpy, ~2 MB/frame
    AP_B2,           // apply_dc_correction
    AP_PSR,          // compute_psr x2
    AP_FILTER,       // filter_init / filter_update
    AP_PUBLISH,      // filter_quantize_q15 + pack_filter + sync
    AP_SCALE_EXTRACT,// scale_extract, x2/frame — reads frame_bo DIRECTLY
    AP_SCALE_MODEL,  // scale_detect + scale_update, pure heap
    AP_DIAG_SCAN,    // report_cint16's max/rails scan. It runs at EVERY verbosity
                     // because rails detection is the point, so it is a real
                     // per-frame cost that was sitting in UNATTRIBUTED — and one
                     // of its four calls still scans a BO mapping (accum_bo).
    AP_N
};
static const char *g_ap_name[AP_N] = {
    "scene gen", "frame push (2MB)", "frame_bo.sync", "transpose",
    "window mean+energy",
    "fcol_bo.sync", "BO<->heap stage", "unpack F_ch", "cmul packing",
    "B2 correction", "PSR scan",
    "filter update", "publish filter", "scale extract", "scale detect+update",
    "diag scan (rails)"
};
// The enum and the name table are two lists that must stay the same length, in
// the same order — exactly the class of coupling CLAUDE.md flags as "duplicated
// in four files with no compile-time check". Here there IS one:
static_assert(sizeof(g_ap_name) / sizeof(g_ap_name[0]) == AP_N,
              "g_ap_name is out of sync with the AP_* enum");

static double        g_ap_us[AP_N];        // this frame
static unsigned long g_ap_n[AP_N];         // calls this frame
static double        g_ap_tot_us[AP_N];    // whole run
static unsigned long g_ap_tot_n[AP_N];
static double        g_ap_frame_us;        // whole frame body, wall
static double        g_ap_run_us;          // whole run, summed frames
static std::chrono::steady_clock::time_point g_ap_frame_t0;

#define AP_T(slot, stmt) do {                                                   \
        const auto _a0 = std::chrono::steady_clock::now();                       \
        stmt;                                                                    \
        g_ap_us[slot] += std::chrono::duration<double, std::micro>(              \
            std::chrono::steady_clock::now() - _a0).count();                     \
        ++g_ap_n[slot];                                                          \
    } while (0)

static void ap_reset_frame(void)
{
    for (int i = 0; i < AP_N; ++i) { g_ap_us[i] = 0.0; g_ap_n[i] = 0; }
    g_ap_frame_t0 = std::chrono::steady_clock::now();
}

// `dma_us` and `rc_us` are passed in rather than read from the DMA/RC globals so
// this function cannot disagree with what those reports printed.
// Split out of the printer for the reason dma_accumulate_frame() was: the report
// runs on two frames, the accumulation must run on all of them.
static void ap_accumulate_frame(void)
{
    g_ap_frame_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - g_ap_frame_t0).count();
    g_ap_run_us += g_ap_frame_us;
    for (int i = 0; i < AP_N; ++i) {
        g_ap_tot_us[i] += g_ap_us[i];
        g_ap_tot_n[i]  += g_ap_n[i];
    }
}

static void ap_report_frame(int frame, double dma_us, double rc_us)
{
    ap_accumulate_frame();
    double apu = 0.0;
    for (int i = 0; i < AP_N; ++i) apu += g_ap_us[i];
    printf("[apu] frame %d cost breakdown, frame body = %.2f ms:\n",
           frame, g_ap_frame_us / 1000.0);
    printf("  %-20s %8s %10s %9s %7s\n", "stage", "calls", "ms", "us/call", "share");
    for (int i = 0; i < AP_N; ++i) {
        if (!g_ap_n[i]) continue;
        printf("  %-20s %8lu %10.3f %9.1f %6.1f%%\n",
               g_ap_name[i], g_ap_n[i], g_ap_us[i] / 1000.0,
               g_ap_us[i] / (double)g_ap_n[i],
               100.0 * g_ap_us[i] / g_ap_frame_us);
    }
    printf("  %-20s %8s %10.3f %9s %6.1f%%\n", "-- APU subtotal", "",
           apu / 1000.0, "", 100.0 * apu / g_ap_frame_us);
    printf("  %-20s %8s %10.3f %9s %6.1f%%\n", "-- GMIO (DMA_T)", "",
           dma_us / 1000.0, "", 100.0 * dma_us / g_ap_frame_us);
    printf("  %-20s %8s %10.3f %9s %6.1f%%\n", "-- roi_crop launch", "",
           rc_us / 1000.0, "", 100.0 * rc_us / g_ap_frame_us);
    const double resid = g_ap_frame_us - apu - dma_us - rc_us;
    printf("  %-20s %8s %10.3f %9s %6.1f%%   <-- console, dumps, printf, "
           "and anything not instrumented\n", "== UNATTRIBUTED", "",
           resid / 1000.0, "", 100.0 * resid / g_ap_frame_us);
    if (resid > 0.25 * g_ap_frame_us)
        printf("  NOTE: the unattributed share is over 25%%. The breakdown above "
               "is NOT the frame; find the missing cost before acting on it.\n");
    fflush(stdout);
}

// -----------------------------------------------------------------------
// CONSOLE VERBOSITY — and the console is not a cosmetic concern here.
//
// MEASURED 2026-08-20, 198 frames of runs/run_0820_1244.log at DUMP_BUFFERS=0:
// regressing frame period against bytes printed gives slope 92.5 us/byte
// (115200 8N1 is 86.8) and an INTERCEPT OF ZERO. The frame time was the UART,
// exactly; the 87 ms of GMIO, the 17 transposes, the ~2 MB/frame of packing
// memcpy and the filter update were all already hidden behind the tty drain.
// 79% of the ~10 KB/frame was instrumentation for problems that are now closed.
//
//   0  one compact line per frame (~45 B, ~4 ms) plus warnings and the
//      end-of-run summaries. USE THIS FOR ANY LONG RUN. track.csv carries the
//      per-frame data and loses nothing. This is the first level at which
//      something other than the UART sets the frame time.
//   1  human-readable per-frame block; the roi_crop and DMA tables on the FIRST
//      and LAST frame only. The default.
//   2  everything, including the 96 per-channel progress lines. This is what
//      every run before 2026-08-20 did.
//
// LEVEL 0 STILL PRINTS ONE LINE PER FRAME, DELIBERATELY. Gating the console to
// literally nothing would delete the instrument that produced the measurement
// above — `picocom | ts` needs a per-frame marker to time. ~4 ms against an
// ~87 ms floor is 4%, which is the right price for keeping the frame period
// measurable at all.
#ifndef VERBOSITY
#define VERBOSITY 1
#endif

// -----------------------------------------------------------------------
// Row-FFT drain pipeline depth — THE GMIO PROBE
// -----------------------------------------------------------------------
// gmio_fft_row_out costs 286 us/tx against 17.7-19.6 for its sibling AIE->DDR
// ports, which drain the same 4096 B chunks in a structurally identical loop.
// The one difference is that this loop interleaves the weights feed and waits
// once PER FIRING, so all 256 firings a frame pay a full host<->AIE round trip.
// GMIO is now 67% of a 127.7 ms frame, so this is the whole remaining question.
//
// THIS PROBE CAN KILL THE HYPOTHESIS, NOT ONLY CONFIRM IT. If the 286 us is
// per-barrier latency, raising the depth divides it. If it is throughput — the
// AIE genuinely producing at that rate — the depth changes nothing at all. Those
// are opposite predictions and one run settles it.
//
// RESULT 2026-08-20 (runs/run_0820_1629.log): **ONLY DEPTH 1 IS POSSIBLE.** The
// sweep ran 40 clean frames at depth 1 and then aborted the instant it tried
// depth 2:
//
//     terminate called after throwing an instance of 'xrt_core::error'
//       what():  Asynchronous operation is already initiated.
//                Multiple 'async' calls are not supported: Invalid argument
//
// XRT's GMIO permits exactly ONE outstanding async per port. The per-firing
// barrier is imposed by the API, not by this loop, so there is no pipeline to
// deepen and DO NOT TRY AGAIN — this block is kept only to record that.
// It cost 11 seconds of board time to retire, which is the whole argument for
// probes that can fail loudly.
//
// WHAT IT DOES NOT KILL: the sibling output ports drain the same 4096 B chunks
// under the SAME one-async-at-a-time rule at 17.7-19.6 us/tx. So the constraint
// is not what makes gmio_fft_row_out cost 286. The difference is that its wait()
// is interleaved with the weights feed, so each firing pays a full
// host->AIE->host round trip that only starts once its weights buffer lands.
// Removing gmio_weights from the loop (the RTP change) should therefore leave a
// plain drain like its siblings. That remains the fix; this was not it.
//
//   FFT_DRAIN_DEPTH = n  fixed depth n — ONLY 1 WORKS, see above
//   FFT_DRAIN_DEPTH = 0  sweep 1,2,4,8,16 in 40-frame blocks (aborts at 2)
//
// The sweep ASCENDS on purpose. adf acquires conv2d's `weights` input_buffer per
// firing and only ~2 fit in flight, so a deep queue is the one thing here that
// could deadlock — and a deadlock on the board needs a power cycle, not a
// timeout. Ascending means every depth that works is already measured and
// printed before the one that might not, so a hang still leaves a usable log and
// names the depth that caused it.
#ifndef FFT_DRAIN_DEPTH
#define FFT_DRAIN_DEPTH 1
#endif

static inline int drain_depth_for_frame(int frame)
{
#if FFT_DRAIN_DEPTH > 0
    (void)frame;
    return FFT_DRAIN_DEPTH;
#else
    static const int d[5] = { 1, 2, 4, 8, 16 };
    int b = frame / 40;
    if (b > 4) b = 4;
    return d[b];
#endif
}

// Compile-time constant, so at VERBOSITY 0 the format strings themselves are
// dead-code-eliminated rather than merely skipped at runtime.
#define VP1(...) do { if (VERBOSITY >= 1) { printf(__VA_ARGS__); } } while (0)
#define VP2(...) do { if (VERBOSITY >= 2) { printf(__VA_ARGS__); fflush(stdout); } } while (0)

// The instrumentation tables (~3.2 KB/frame of roi_crop timeline + DMA split)
// are printed on the first and last frame only. Two frames of it in a 200-frame
// run is ~6 KB total against 640 KB before — and the LAST frame matters as much
// as the first, since it is the converged state.
static inline bool trace_frame(int frame)
{
    return frame == 0 || frame == ITER_CNT - 1;
}

#define RC_MAX_CALLS 64
#ifndef RC_TRACE_FRAMES
#define RC_TRACE_FRAMES 3      // frames for which every call is printed
#endif
static double g_rc_call_us[RC_N][RC_MAX_CALLS];   // this frame, per call
static double g_rc_min_us[RC_N];                  // over the whole run
static double g_rc_max_us[RC_N];

// -------- completion timeline, anchored to the drain (item 5) --------
//
// The strongest existing evidence that the CU is innocent — the row-FFT drain
// completes in 2-6 ms, and it CANNOT complete until conv2d has consumed all
// 16384 patch pixels, so roi_crop has emitted its last AXIS beat by then, and
// the hw_emu VCD shows ap_done asserting in the same cycle as TLAST — is
// currently assembled by hand from two unrelated aggregates (a DMA total and an
// RC total) that share no clock. These five marks put all of it on ONE clock,
// zeroed at the crop_run.start() call, so the interval that matters can be read
// off directly rather than inferred:
//
//   drain -> poll   is time spent AFTER the CU had finished. If this is the
//                   500 ms, the cost is in the host/scheduler completion path,
//                   and that holds whether the 500 ms shows up in poll() or in
//                   wait() — both observe the same ERT command state.
//   start -> drain   is time the CU (or its AIE backpressure) was genuinely
//                   busy. If THIS is the 500 ms, the investigation moves into
//                   the PL and the poll/wait split is beside the point.
enum { TL_START, TL_DRAIN, TL_POLL, TL_WAIT, TL_WAIT2, TL_N };
static const char *g_tl_name[TL_N] = { "start()", "drain", "poll", "wait", "wait#2" };
static double g_tl_ms[RC_MAX_CALLS][TL_N];
static int    g_tl_rows;
static std::chrono::steady_clock::time_point g_tl_t0;

static inline void rc_tl_begin(void)
{
    if (g_tl_rows < RC_MAX_CALLS)
        for (int i = 0; i < TL_N; ++i) g_tl_ms[g_tl_rows][i] = -1.0;
    g_tl_t0 = std::chrono::steady_clock::now();
}
static inline void rc_tl_mark(int which)
{
    if (g_tl_rows >= RC_MAX_CALLS) return;
    g_tl_ms[g_tl_rows][which] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - g_tl_t0).count();
}
static inline void rc_tl_end(void) { if (g_tl_rows < RC_MAX_CALLS) ++g_tl_rows; }

// min/max are seeded on the first call of the run via the n_total==0 test rather
// than from a sentinel, so there is no "0.0 means unset" ambiguity to misread.
#define RC_T(slot, stmt) do {                                                   \
        const auto _t0 = std::chrono::steady_clock::now();                       \
        stmt;                                                                    \
        const double _dt = std::chrono::duration<double, std::micro>(             \
            std::chrono::steady_clock::now() - _t0).count();                      \
        if (g_rc_n[slot] < RC_MAX_CALLS)                                          \
            g_rc_call_us[slot][g_rc_n[slot]] = _dt;                               \
        if (!g_rc_n_total[slot] || _dt < g_rc_min_us[slot])                       \
            g_rc_min_us[slot] = _dt;                                              \
        if (!g_rc_n_total[slot] || _dt > g_rc_max_us[slot])                       \
            g_rc_max_us[slot] = _dt;                                              \
        g_rc_us[slot] += _dt;  ++g_rc_n[slot];                                    \
        g_rc_us_total[slot] += _dt;  ++g_rc_n_total[slot];                        \
    } while (0)

static void rc_reset_frame(void)
{
    for (int i = 0; i < RC_N; ++i) { g_rc_us[i] = 0.0; g_rc_n[i] = 0; }
    g_rc_poll_iters = 0;
    g_tl_rows = 0;
    // g_rc_call_us needs no clearing: it is written before it is read, indexed by
    // g_rc_n[slot], which was just zeroed. g_rc_min_us/g_rc_max_us deliberately
    // survive the reset — they are run-scoped, not frame-scoped.
}

// Spin until the command reaches a terminal state, then report how long that
// took and how many polls it needed. Returns the final state so the caller can
// see an ERROR/ABORT rather than silently treating it as success.
static ert_cmd_state rc_poll_until_done(xrt::run &r)
{
    const auto     t0 = std::chrono::steady_clock::now();
    ert_cmd_state  st = r.state();
    while (rc_pending(st)) {
        ++g_rc_poll_iters;
        if (std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count() > RC_POLL_MAX_S) {
            ++g_rc_poll_timeouts;
            break;
        }
        std::this_thread::yield();
        st = r.state();
    }
    return st;
}

static void rc_report_frame(int frame)
{
    double tot = 0.0;
    printf("[roi_crop] frame %d launch-phase cost:\n", frame);
    for (int i = 0; i < RC_N; ++i) {
        if (!g_rc_n[i]) continue;
        printf("  %-20s %4lu x  %9.3f ms  %8.2f ms each"
               "   [run min %8.2f  max %8.2f]\n",
               g_rc_name[i], g_rc_n[i], g_rc_us[i] / 1000.0,
               g_rc_us[i] / g_rc_n[i] / 1000.0,
               g_rc_min_us[i] / 1000.0, g_rc_max_us[i] / 1000.0);
        tot += g_rc_us[i];
    }
    printf("  %-20s        %9.3f ms  = %.1f%% of a 33 ms frame\n",
           "TOTAL", tot / 1000.0, 100.0 * tot / 33000.0);
    // The verdict, stated inline so it does not have to be re-derived from the
    // table every time.
    //
    // CORRECTED. This used to print "CU itself is slow to signal done (PL side)"
    // whenever poll dominated wait. That conclusion does not follow, and printing
    // it would have sent the next investigation into the PL on no evidence:
    // xrt::run::state() reads the ERT command-packet state, which is updated by
    // the same scheduler path that unblocks wait(). poll and wait are therefore
    // NOT independent observers of the CU — a slow poll is equally consistent with
    // "the CU finished at 5 ms and the observation of that fact is slow". Only the
    // drain-anchored timeline below can separate those two, so the PL-side verdict
    // is issued there and this one is confined to what the split can actually
    // establish: WHICH host call carries the cost.
    if (g_rc_n[RC_POLL] && g_rc_n[RC_WAIT]) {
        const double p  = g_rc_us[RC_POLL]  / g_rc_n[RC_POLL]  / 1000.0;
        const double w  = g_rc_us[RC_WAIT]  / g_rc_n[RC_WAIT]  / 1000.0;
        const double w2 = g_rc_n[RC_WAIT2]
                        ? g_rc_us[RC_WAIT2] / g_rc_n[RC_WAIT2] / 1000.0 : 0.0;
        printf("  poll %.2f ms | wait %.2f ms | wait#2 %.2f ms per call, "
               "%lu poll iters%s\n",
               p, w, w2, g_rc_poll_iters,
               g_rc_poll_timeouts ? "  [SPIN TIMED OUT — poll figure truncated]" : "");
#if ROI_CROP_USER_MANAGED
        // The verdict on the fix itself. The KDS path costs 503.4 ms per call on
        // this board and the CU is done at 4.8 ms, so anything still in the
        // hundreds of ms means reading ap_done directly did NOT help — which
        // would be a genuinely new fact, and would move the investigation into
        // the PL/AIE rather than the driver. 50 ms sits two orders below the
        // broken figure and one above the expected one, so neither outcome can
        // land near the boundary.
        printf("  -> %s\n",
               (p < 50.0)
                   ? "USER-MANAGED PATH IS WORKING: ap_done observed in ~ms, not "
                     "~500 ms. The control-CU probe stays on KDS and should still "
                     "pay ~512 ms in this same run — that contrast is the proof"
                   : "USER-MANAGED PATH DID NOT HELP: the CU's own ap_done is "
                     "still slow to appear, so the cost is NOT the driver's "
                     "completion path. Re-read the timeline before changing "
                     "anything else");
#else
        printf("  -> %s\n",
               (p > 10.0 * w)
                   ? "the blocking cost is absorbed by the POLL SPIN; see the "
                     "timeline for whether the CU or its observation was slow"
                   : "the blocking cost is in XRT's wait(); the spin is the fix "
                     "IF the timeline shows the CU already done");
#endif
        // Item 8. A wait() on a command that poll() has already seen reach a
        // terminal state cannot legitimately block on anything.
        if (g_rc_n[RC_WAIT2] && w2 > 1.0)
            printf("  -> wait#2 costs %.2f ms on an ALREADY-COMPLETED command: "
                   "wait() is doing per-call work, not waiting. Neither a spin "
                   "nor a PL fix addresses this.\n", w2);
    }
    fflush(stdout);
}

// -----------------------------------------------------------------------
// Per-call dump (item 6). Printed for the first RC_TRACE_FRAMES frames only:
// 5 slots x 16 channels x 3 frames is ~240 short lines, which is 2 s of console
// at 115200 and worth it exactly once per run.
//
// What to read from it: are the values CLUSTERED at multiples of a base interval
// (a tick in the completion path — a host constant), or DISPERSED (a cost that
// tracks data)? The `gcd-ish` hint is deliberately not computed here; eyeballing
// 16 numbers is more reliable than a heuristic that can be fooled by one outlier.
// -----------------------------------------------------------------------
static void rc_report_calls(int frame)
{
    // The caller gates this to the first and last frame; RC_TRACE_FRAMES is kept
    // as an additional cap so `RC_TRACE_FRAMES=0` silences the per-call detail
    // without silencing the timeline next to it.
    if (frame >= RC_TRACE_FRAMES && frame != ITER_CNT - 1) return;
    printf("[roi_crop] frame %d per-call detail, ms (look for tick quantization):\n",
           frame);
    for (int i = 0; i < RC_N; ++i) {
        if (g_rc_n[i] < 2) continue;          // ctor is once-per-run, skip it
        const unsigned long n = g_rc_n[i] < RC_MAX_CALLS ? g_rc_n[i] : RC_MAX_CALLS;
        printf("  %-20s", g_rc_name[i]);
        for (unsigned long c = 0; c < n; ++c) {
            printf(" %8.2f", g_rc_call_us[i][c] / 1000.0);
            if ((c % 8) == 7 && c + 1 < n) printf("\n  %-20s", "");
        }
        printf("\n");
    }
    fflush(stdout);
}

// -----------------------------------------------------------------------
// Completion timeline (item 5). One row per channel, all five marks on one
// clock zeroed at the crop_run.start() call.
// -----------------------------------------------------------------------
static void rc_report_timeline(int frame)
{
    if (!g_tl_rows) return;
    printf("[roi_crop] frame %d completion timeline, ms since start() entry:\n",
           frame);
    printf("  ch %9s %9s %9s %9s %9s | %11s\n",
           g_tl_name[TL_START], g_tl_name[TL_DRAIN], g_tl_name[TL_POLL],
           g_tl_name[TL_WAIT], g_tl_name[TL_WAIT2], "drain->poll");
    double drain_sum = 0.0, gap_sum = 0.0, drain_max = 0.0, gap_max = 0.0;
    for (int c = 0; c < g_tl_rows; ++c) {
        const double drain = g_tl_ms[c][TL_DRAIN];
        const double gap   = g_tl_ms[c][TL_POLL] - drain;
        printf("  %2d %9.3f %9.3f %9.3f %9.3f %9.3f | %11.3f\n", c,
               g_tl_ms[c][TL_START], drain, g_tl_ms[c][TL_POLL],
               g_tl_ms[c][TL_WAIT], g_tl_ms[c][TL_WAIT2], gap);
        drain_sum += drain;  gap_sum += gap;
        if (drain > drain_max) drain_max = drain;
        if (gap   > gap_max)   gap_max   = gap;
    }
    const double drain_mean = drain_sum / g_tl_rows;
    const double gap_mean   = gap_sum   / g_tl_rows;
    printf("  mean: start->drain %.3f ms (max %.3f), drain->done %.3f ms "
           "(max %.3f)\n", drain_mean, drain_max, gap_mean, gap_max);

    // THE VERDICT THIS PATCH EXISTS FOR.
    //
    // The drain loop cannot complete until conv2d has consumed all PATCH_ELEMS
    // pixels, so by the `drain` mark roi_crop has emitted every AXIS beat — and
    // the hw_emu VCD shows ap_done asserting in the same cycle as TLAST. So the
    // `drain` mark is an upper bound on when the CU finished that is INDEPENDENT
    // of the ERT command state, which is precisely what poll() and wait() are not.
    if (gap_mean > 10.0 * drain_mean && gap_mean > 20.0)
        printf("  -> CU WAS ALREADY DONE. %.1f%% of the per-channel cost lands "
               "AFTER the last AXIS beat was consumed. The cost is in the host "
               "completion path (scheduler/driver), NOT in the PL. Compare the "
               "control-CU probe: if camera_capture pays the same, it is "
               "scheduler-wide.\n",
               100.0 * gap_mean / (drain_mean + gap_mean));
    else if (drain_mean > 20.0)
        printf("  -> THE CU (or its AIE backpressure) IS THE COST: %.3f ms "
               "elapses before conv2d has consumed the patch, against ~1 ms of "
               "measured PL datapath. This is a PL/AIE-side investigation and "
               "the poll-vs-wait split above is beside the point.\n", drain_mean);
    else
        printf("  -> both intervals are small; roi_crop is no longer the frame. "
               "Re-read the frame total before optimising anything here.\n");
    fflush(stdout);
}

// -----------------------------------------------------------------------
// roi_crop as a USER-MANAGED CU — the fix for the 503 ms completion cost
// -----------------------------------------------------------------------
// WHY THIS EXISTS. The instrumentation above localised roi_crop's ~505 ms per
// channel to the host completion path, and four measurements then closed it:
//
//   1. poll(state) costs 503.4 ms and the wait() after it costs 2 us, so wait()
//      was only ever blocking on a command state that had not flipped.
//   2. drain -> poll is 503.4 ms against a 4.8 ms start -> drain, i.e. 99% of the
//      cost lands after the CU consumed its last AXIS beat.
//   3. camera_capture — no AXIS port, ~6 us of work — pays the same 512 ms, and
//      1080 rows costs the same as 1 row. Fixed cost, not a datapath.
//   4. /proc/interrupts shows ZERO on both zocl IRQs across every run, while the
//      CU's own registers read GIER=1, IER=0x3, ISR=0x3. The CU raises its
//      interrupt on every completion and re-latches within one run after the ISR
//      is cleared by hand; nothing in the kernel ever receives it.
//
// So the CU is healthy, the interrupt is armed and asserted, and the delivery
// path between the CU and the GIC is dead — a platform defect (see the boot-time
// `zocl-drm: error -ENXIO: IRQ index 32 not found`, present on every probe).
// KDS therefore waits on an interrupt that cannot arrive and falls back to a
// ~500 ms timer. Nothing reachable from userspace fixes that: poll_threshold
// (1000000), a hand-cleared ISR, and Runtime.ert_polling all left the number at
// 503.4 ms.
//
// The fix is to stop asking KDS when the CU finished and read the CU's own
// status register instead — the same ap_done bit, four orders of magnitude
// sooner. xrt::ip does exactly that, and its 2025.2 implementation imposes no
// control-protocol restriction: xrt_ip.cpp's ctor requires only that the IP is
// in IP_LAYOUT with a base address and an address range.
//
// TWO CONSTRAINTS, both load-bearing:
//
//   - xrt::ip takes an EXCLUSIVE CU context by default, so the xrt::kernel for
//     roi_crop must not be constructed at the same time. It isn't — see the
//     #if at the kernel handles. (Runtime.rw_shared=true relaxes this if some
//     future consumer needs the CU too.)
//   - frame_buf is an m_axi pointer, so the CU needs a DEVICE address, which
//     set_arg(0, bo) used to supply implicitly. bo.address() is that value.
//
// Register map is not guessed — it is read back from the running board:
//   cat /sys/bus/platform/devices/CU.3.auto/cu_info
// which reports base 0xa4010000, Protocol CTRL_CHAIN, and every offset below.
namespace roi_crop_reg {
    constexpr uint32_t CTRL          = 0x00;
    constexpr uint32_t FRAME_BUF_LO  = 0x10;   // 64-bit m_axi pointer
    constexpr uint32_t FRAME_BUF_HI  = 0x14;
    constexpr uint32_t FRAME_ROWS_R  = 0x1c;   // _R suffix: FRAME_ROWS/FRAME_COLS
    constexpr uint32_t FRAME_COLS_R  = 0x24;   // are already object-like macros
    constexpr uint32_t ROI_ROW       = 0x2c;
    constexpr uint32_t ROI_COL       = 0x34;
    constexpr uint32_t ROI_H         = 0x3c;
    constexpr uint32_t ROI_W         = 0x44;
    constexpr uint32_t PATCH_ROWS_R  = 0x4c;
    constexpr uint32_t PATCH_COLS_R  = 0x54;
    constexpr uint32_t RECOMPUTE     = 0x5c;

    // HLS ap_ctrl block, 0x00.
    constexpr uint32_t AP_START      = 1u << 0;
    constexpr uint32_t AP_DONE       = 1u << 1;
    constexpr uint32_t AP_IDLE       = 1u << 2;
    constexpr uint32_t AP_READY      = 1u << 3;
    constexpr uint32_t AP_CONTINUE   = 1u << 4;
}

#if ROI_CROP_USER_MANAGED
class CropIp {
public:
    CropIp(xrt::device &dev, const xrt::uuid &xclbin_id)
        : m_ip(dev, xclbin_id, "roi_crop:{roi_crop_0}") {}

    // Frame-invariant arguments, written once — the direct analogue of the
    // hoisted set_arg() calls on the KDS path.
    void set_static_args(const xrt::bo &frame_bo,
                         uint32_t frame_rows, uint32_t frame_cols,
                         uint32_t patch_rows, uint32_t patch_cols)
    {
        const uint64_t addr = frame_bo.address();
        m_ip.write_register(roi_crop_reg::FRAME_BUF_LO, (uint32_t)(addr & 0xffffffffu));
        m_ip.write_register(roi_crop_reg::FRAME_BUF_HI, (uint32_t)(addr >> 32));
        m_ip.write_register(roi_crop_reg::FRAME_ROWS_R, frame_rows);
        m_ip.write_register(roi_crop_reg::FRAME_COLS_R, frame_cols);
        m_ip.write_register(roi_crop_reg::PATCH_ROWS_R, patch_rows);
        m_ip.write_register(roi_crop_reg::PATCH_COLS_R, patch_cols);
    }

    // Per-frame geometry + the per-channel recompute flag.
    void set_frame_args(uint32_t roi_row, uint32_t roi_col,
                        uint32_t roi_h,   uint32_t roi_w, uint32_t recompute)
    {
        m_ip.write_register(roi_crop_reg::ROI_ROW,   roi_row);
        m_ip.write_register(roi_crop_reg::ROI_COL,   roi_col);
        m_ip.write_register(roi_crop_reg::ROI_H,     roi_h);
        m_ip.write_register(roi_crop_reg::ROI_W,     roi_w);
        m_ip.write_register(roi_crop_reg::RECOMPUTE, recompute);
    }

    void start() { m_ip.write_register(roi_crop_reg::CTRL, roi_crop_reg::AP_START); }

    // Spin on the CU's own ap_done. Returns an ert_cmd_state so the call site's
    // error check is identical to the KDS path's.
    //
    // ap_done is read-clear under ap_ctrl_hs and held until ap_continue under
    // ap_ctrl_chain; this CU is CTRL_CHAIN, so the loop exits on the read that
    // observes the bit and ap_continue then releases it. Writing AP_CONTINUE to
    // an ap_ctrl_hs CU is ignored, so the sequence is correct under both.
    ert_cmd_state poll_done()
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const uint32_t ctrl = m_ip.read_register(roi_crop_reg::CTRL);
            ++g_rc_poll_iters;
            if (ctrl & roi_crop_reg::AP_DONE) {
                m_ip.write_register(roi_crop_reg::CTRL, roi_crop_reg::AP_CONTINUE);
                return ERT_CMD_STATE_COMPLETED;
            }
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count() > RC_POLL_MAX_S) {
                ++g_rc_poll_timeouts;
                return ERT_CMD_STATE_TIMEOUT;
            }
            std::this_thread::yield();
        }
    }

private:
    xrt::ip m_ip;
};
#endif  // ROI_CROP_USER_MANAGED

// -----------------------------------------------------------------------
// Control CU probe (item 7)
// -----------------------------------------------------------------------
// NOTE: with ROI_CROP_USER_MANAGED=1 this probe becomes a WITHIN-RUN CONTROL
// rather than a diagnostic. camera_capture stays on the KDS path, so it should
// keep paying ~512 ms in the same run where roi_crop no longer does. That
// contrast is the cleanest possible evidence that the fix is the fix, and it
// costs 4 s at startup. Set CONTROL_CU_RUNS=0 once it stops being interesting.
// WHY. Every launch-path number in this design is measured on roi_crop, so
// nothing yet distinguishes "roi_crop's completion is slow" from "ANY CU
// completion costs ~500 ms on this stack". Those have nothing in common: the
// first is a datapath or PLIO question, the second is XRT/zocl configuration and
// roi_crop is a bystander.
//
// camera_capture is the ideal control. It is already in the xclbin, it has no
// AXIS port (so no AIE backpressure and no graph dependency), and its runtime is
// known from the source: one II=1 loop over frame_rows*frame_cols bytes, i.e.
// ~6.6 ms full-size at 312.5 MHz and ~6 us for a single row. Two sizes are probed
// on purpose — completion cost that is identical at 1920 and 2073600 bytes is a
// fixed host cost by construction, and no PL explanation survives that.
//
// Safe to call here: it zero-fills frame_bo, which is exactly what the
// (currently commented-out) intended call did, and every frame overwrites the
// whole buffer from the host map before syncing it to the device.
//
// Must run BEFORE the frame loop and AFTER the graph is up, and it deliberately
// does NOT touch crop_run — a probe that perturbed the thing being measured
// would be worthless.
#ifndef CONTROL_CU_RUNS
#define CONTROL_CU_RUNS 0
#endif
static void rc_control_cu_probe(xrt::kernel &cam, xrt::bo &frame_bo,
                                int frame_rows, int frame_cols)
{
#if CONTROL_CU_RUNS
    printf("\n[control-cu] camera_capture launch-path probe, %d runs "
           "(alternating 1 row / %d rows)\n", CONTROL_CU_RUNS, frame_rows);
    printf("  run  rows      start(ms)   poll(ms)   wait(ms)  wait#2(ms)  state\n");
    double poll_small = 0.0, poll_full = 0.0, wait_small = 0.0, wait_full = 0.0;
    int    n_small = 0, n_full = 0;

    xrt::run r(cam);
    r.set_arg(0, frame_bo);
    for (int i = 0; i < CONTROL_CU_RUNS; ++i) {
        const bool full = (i & 1) != 0;
        const int  rows = full ? frame_rows : 1;
        r.set_arg(1, (uint32_t)rows);
        r.set_arg(2, (uint32_t)frame_cols);

        const auto t0 = std::chrono::steady_clock::now();
        r.start();
        const auto t1 = std::chrono::steady_clock::now();
        const ert_cmd_state st = rc_poll_until_done(r);
        const auto t2 = std::chrono::steady_clock::now();
        r.wait();
        const auto t3 = std::chrono::steady_clock::now();
        r.wait();
        const auto t4 = std::chrono::steady_clock::now();

        const double ms_start = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms_poll  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double ms_wait  = std::chrono::duration<double, std::milli>(t3 - t2).count();
        const double ms_wait2 = std::chrono::duration<double, std::milli>(t4 - t3).count();
        printf("  %3d %5d   %9.3f  %9.3f  %9.3f   %9.3f  %d\n",
               i, rows, ms_start, ms_poll, ms_wait, ms_wait2, (int)st);
        if (full) { poll_full  += ms_poll; wait_full  += ms_wait; ++n_full; }
        else      { poll_small += ms_poll; wait_small += ms_wait; ++n_small; }
    }
    if (n_small && n_full) {
        const double s = (poll_small + wait_small) / n_small;
        const double f = (poll_full  + wait_full)  / n_full;
        // Expected PL datapath: rows*cols cycles at II=1. The point of printing it
        // is that the SLOPE, not the level, is what a PL explanation has to match.
        printf("  completion cost: 1 row %.3f ms (PL expects ~%.3f ms), "
               "%d rows %.3f ms (PL expects ~%.3f ms)\n",
               s, 1.0 * frame_cols / (PL_FREQ_MHZ * 1000.0),
               frame_rows, f,
               1.0 * frame_rows * frame_cols / (PL_FREQ_MHZ * 1000.0));
        if (s > 50.0)
            printf("  -> A CU WITH NO AXIS PORT AND ~%.0f us OF WORK ALSO PAYS "
                   "%.1f ms. The cost is scheduler-wide and roi_crop is a "
                   "bystander; do not optimise roi_crop.\n",
                   1000.0 * frame_cols / (PL_FREQ_MHZ * 1000.0), s);
        else if (f < 50.0)
            printf("  -> the control CU completes at its datapath cost, so the "
                   "completion path is healthy in general and whatever roi_crop "
                   "pays is specific to roi_crop.\n");
    }
    fflush(stdout);
#else
    (void)cam; (void)frame_bo; (void)frame_rows; (void)frame_cols;
    printf("[control-cu] probe disabled (CONTROL_CU_RUNS=0). It is a HARDWARE "
           "measurement: under hw_emu camera_capture zero-fills at II=1 in cosim "
           "and would cost hours for a number that means nothing there.\n");
#endif
}

// Per-frame breakdown. The number that decides whether the transposes have to
// move off the APU is the total, against 33 ms.
// Roll this frame's counters into the run totals. SPLIT OUT OF THE PRINTER on
// purpose: the per-frame table is now printed on two frames only, and folding
// accumulation into a function that is no longer called every frame would have
// silently made the CUMULATIVE report at exit a two-frame report. That is the
// exact failure mode the DUMP_BUFFERS residual taught — a number that still
// looks plausible after it stops meaning what it says.
static void dma_accumulate_frame(void)
{
    for (int i = 0; i < DMA_N; ++i) {
        if (!g_dma[i].calls && g_dma[i].us == 0.0) continue;
        g_dma_total[i].name   = g_dma[i].name;
        g_dma_total[i].calls += g_dma[i].calls;
        g_dma_total[i].us    += g_dma[i].us;
    }
}

static void dma_report_frame(int frame)
{
    double        tot_us = 0.0;
    unsigned long tot_n  = 0;
    printf("[dma] frame %d per-port cost:\n", frame);
    for (int i = 0; i < DMA_N; ++i) {
        if (!g_dma[i].calls && g_dma[i].us == 0.0) continue;
        const double per = g_dma[i].calls ? g_dma[i].us / g_dma[i].calls : 0.0;
        printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx\n",
               g_dma[i].name, g_dma[i].calls, g_dma[i].us / 1000.0, per);
        tot_us += g_dma[i].us;
        tot_n  += g_dma[i].calls;
    }
    dma_accumulate_frame();
    printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx  = %.1f%% of a 33 ms frame\n",
           "TOTAL", tot_n, tot_us / 1000.0,
           tot_n ? tot_us / tot_n : 0.0, 100.0 * tot_us / 33000.0);
    // GUARDED. This line used to print unconditionally, including on TARGET=hw,
    // and it caused a set of genuine hardware numbers to be discounted — see
    // CLAUDE.md, "Measurement / methodology".
#ifdef HW_EMU_BUILD
    printf("  NOTE: hw_emu wall time is not real hardware time. Treat the "
           "tx COUNT and the\n        per-port split as the real findings; the "
           "us/tx figure needs a `TARGET=hw` run.\n");
#endif
    fflush(stdout);
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// In-place 2-D matrix transpose via a temporary scratch buffer.
// elem_bytes must be 4 (cint16).
// HEAP -> HEAP. Was transpose_inplace() operating directly on the BO mapping,
// which made every one of the rows*cols element reads an uncached load: measured
// 547 us for 64 KB, i.e. 33 ns/element. The scatter is unavoidable; doing it on
// an uncached buffer is not. Callers now bulk-copy in and out around this.
static void transpose_to(const void *src_v, void *dst_v,
                         int rows, int cols, size_t elem_bytes)
{
    const uint8_t *src = static_cast<const uint8_t *>(src_v);
    uint8_t       *dst = static_cast<uint8_t *>(dst_v);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            memcpy(dst + ((size_t)c * rows + r) * elem_bytes,
                   src + ((size_t)r * cols + c) * elem_bytes, elem_bytes);
}

// Peak detection used to live here as peak_detect_sw(). It is gone: it performed
// a bit-identical |real| scan to mosse::compute_psr(..., use_abs=true) — same
// strict-`>` first-wins tie-break, same wrap arithmetic — so keeping both meant
// two independent scans that could silently decide different things, one driving
// the position and the other driving the gate. The tracker now takes its
// displacement from the same PsrResult the gate judges.
//
// The rationale that lived on it is preserved on compute_psr in mosse_filter.h:
// the scan is over |real| because Stage B1 makes the response bipolar (aiesim s6
// peaks at {-417,0}), and `resp[2*i]` rather than `resp[i]` because the map is
// interleaved cint16. A zero response, which used to be indistinguishable from a
// correct centred answer, is now GateReason::ZeroResponse.

// PsrResult and compute_psr() moved to mosse_filter.{h,cpp} on 2026-08-15.
// They are pure (no XRT, no ADF), so living there puts them under `make
// test_host` — seconds instead of a ~26 min hw_emu frame. That matters more for
// the gate than for anything else in this file: the gate CANNOT fire on the
// current synthetic test data (PSR ~172 against a threshold of 7), so the native
// tests are the only place its failure paths are exercised at all.
//
// Call them FULLY QUALIFIED (mosse::compute_psr). A leftover file-static copy
// here would win by unqualified lookup, link cleanly, run correctly, and mean
// the unit tests were validating a different function than the hardware runs.

// Prints the PSR at both peak definitions and classifies against Bolme's ranges.
//
// Takes the results rather than the response map: the caller computes them once
// and feeds the SAME objects to this and to the gate, so what is printed and what
// is acted on cannot drift apart. (This used to rescan the map twice on its own,
// and peak_detect_sw scanned it a third time.)
//
// Presentation only — the policy is mosse::psr_gate() in mosse_filter.cpp, and
// the authoritative verdict is printed by the caller AFTER this.
static void report_psr(const mosse::PsrResult &a,   // argmax|re| — what we act on
                       const mosse::PsrResult &s)   // argmax re  — paper-literal
{
    // The two table rows are TRACE ONLY: track.csv already carries both PSRs and
    // the peak, one row per frame. The two ANOMALY branches below are printed at
    // every verbosity — a disagreeing or negative peak must never be silenced to
    // save console.
    VP2("  [psr] at argmax|re| (%d,%d): peak %ld  sidelobe mu %.1f sd %.1f "
        "max %.1f  PSR %.2f (Bolme)  ratio %.2fx (aiesim metric)  n=%ld\n",
        a.dr, a.dc, a.peak, a.mean, a.sdev, a.side_max, a.psr, a.ratio, a.n_side);
    VP2("  [psr] at argmax re  (%d,%d): peak %ld  sidelobe mu %.1f sd %.1f "
        "max %.1f  PSR %.2f (Bolme)  ratio %.2fx\n",
        s.dr, s.dc, s.peak, s.mean, s.sdev, s.side_max, s.psr, s.ratio);

    if (a.dr != s.dr || a.dc != s.dc)
        printf("  [psr] DISAGREE: the |real| scan and the signed max pick "
               "DIFFERENT peaks. The tracker acted on (%d,%d) with value %ld; "
               "the paper-literal peak is (%d,%d) with %ld.\n",
               a.dr, a.dc, a.peak, s.dr, s.dc, s.peak);
    else
        VP2("  [psr] peak definitions agree.\n");

    if (a.peak < 0)
        printf("  [psr] WARNING: the acted-on peak is NEGATIVE (%ld). The target "
               "output g is a positive Gaussian, so a negative peak is "
               "anti-correlation, not a detection.\n", a.peak);

    // Classified on Bolme's PSR only. His 20-60 / ~7 numbers do NOT apply to the
    // `ratio` column — that is a different statistic (see the note on PsrResult).
    //
    // The >60 band is separate on purpose: Bolme's 20-60 comes from real video with
    // background clutter, and this harness injects a synthetic target onto an
    // otherwise empty frame. A near-empty sidelobe inflates PSR well past his range
    // (measured 125.0 at 128x128/ch16), which is a property of the TEST INPUT, not
    // evidence of a better tracker. So PSR here is a good RELATIVE metric for
    // comparing builds and a bad absolute one; it only becomes comparable to the
    // paper once real video is feeding it.
    const double p = a.psr;
    const char *verdict =
        (a.sdev <= 0.0) ? "VOID — flat sidelobe, the response carries nothing"
        : (p > 60.0)    ? "ABOVE Bolme's 20-60 range — expected on a synthetic "
                          "target with no clutter; use only as a relative metric"
        : (p >= 20.0)   ? "OK — inside Bolme's normal 20-60 range"
        : (p >= 7.0)    ? "WEAK — below Bolme's normal range, above his ~7 failure mark"
                        : "FAIL — at or below Bolme's ~7.0 occlusion/failure indicator";
    VP2("  [psr] verdict: %s\n", verdict);
    VP2("  [psr]          (Bolme PSR only — his thresholds do NOT apply to the "
        "ratio column)\n");
    (void)verdict;
}

// -----------------------------------------------------------------------
// Update gating — Bolme §3.5 (the control path; the policy is mosse::psr_gate)
// -----------------------------------------------------------------------
// "when PSR drops to around 7.0 it is an indication that the object is occluded
//  or tracking has failed" — at which point the tracker stops the online update
//  and reacquires when the appearance returns.
//
// Before this existed, filter_update ran unconditionally at eta=0.125 every
// frame, so an occluded frame trained the filter on background and the target was
// lost irrecoverably — the failure mode Bolme's Fig. 5 shows for the *Naive*
// filter and that MOSSE exists to avoid.
//
// Reacquisition needs no code of its own: a gated frame HOLDS the position, so
// the ROI stays on the last known good location and the filter keeps being
// applied there. When the target reappears PSR recovers and tracking resumes. The
// alternative — moving to the peak anyway — walks the ROI off the target on noise
// and it can never re-enter the search window.
static int    g_gate_run   = 0;    // consecutive gated frames, current run
static int    g_gate_worst = 0;    // longest such run
static int    g_gate_eval  = 0;    // frames the gate actually judged
static int    g_gate_hold  = 0;    // of those, how many were gated out
static int    g_gate_reason_n[8] = {0};
static double g_psr_min = 0.0, g_psr_max = 0.0, g_psr_sum = 0.0;

// Consecutive gated frames after which the run is called lost. REPORT ONLY —
// nothing changes behaviour at this count. A "lost" state that stopped holding,
// reset the position or widened the search would break the reacquisition
// property above, and Bolme defines no such state. Report it now so a real policy
// can be calibrated against data later, alongside the scale search.
#define PSR_LOST_FRAMES 5

static void gate_track(int frame, const mosse::GateDecision &g)
{
    if (g_gate_eval == 0) { g_psr_min = g_psr_max = g.psr; }
    else { if (g.psr < g_psr_min) g_psr_min = g.psr;
           if (g.psr > g_psr_max) g_psr_max = g.psr; }
    ++g_gate_eval;
    g_psr_sum += g.psr;
    ++g_gate_reason_n[(int)g.reason];

    // A HOLD is an event; an ACCEPT is the steady state. The verdict line drops
    // to VERBOSITY 1 and the prose explanation to 2, but a HOLD prints at every
    // level (below) because "the tracker stopped updating" is never noise.
    VP1("  [gate] frame %d: %s  reason=%s  psr=%.2f  threshold=%.2f\n",
        frame, g.accept ? "ACCEPT" : "HOLD",
        mosse::gate_reason_tag(g.reason), g.psr, (double)g.threshold);
    VP2("  [gate]         %s\n", mosse::gate_reason_why(g.reason));
    if (!g.accept && VERBOSITY < 1)
        printf("  [gate] frame %d: HOLD  reason=%s  psr=%.2f\n",
               frame, mosse::gate_reason_tag(g.reason), g.psr);

    if (g.accept) {
        // The line that proves the hold-position policy works: the target came
        // back and was re-detected without any reacquisition search.
        if (g_gate_run > 0)
            printf("  [gate]         REACQUIRED after %d gated frame(s)\n", g_gate_run);
        g_gate_run = 0;
    } else {
        ++g_gate_hold;
        ++g_gate_run;
        if (g_gate_run > g_gate_worst) g_gate_worst = g_gate_run;
        printf("  [gate]         position HELD; filter A/B frozen; H not "
               "republished. Consecutive gated frames: %d\n", g_gate_run);
        if (g_gate_run == PSR_LOST_FRAMES)
            printf("  [gate]         LOST: %d consecutive gated frames — the target "
                   "has not been re-detected (report only, nothing changes)\n",
                   g_gate_run);
    }
    fflush(stdout);
}

static void gate_report_run(int frames)
{
    printf("\n[gate] SUMMARY over %d frame(s): %d evaluated, %d accepted, %d gated\n",
           frames, g_gate_eval, g_gate_eval - g_gate_hold, g_gate_hold);
    if (g_gate_eval == 0) {
        // ITER_CNT=1 lands here: frame 0 is consumed by initialisation, so the
        // gate never judged anything. The warning at startup says the same.
        printf("[gate]   no frames were evaluated — frame 0 initialises the filter, "
               "so a meaningful run needs ITER_CNT >= 2\n");
        fflush(stdout);
        return;
    }
    printf("[gate]   reasons:");
    for (int i = 0; i < 8; ++i)
        if (g_gate_reason_n[i])
            printf("  %s x%d", mosse::gate_reason_tag((mosse::GateReason)i),
                   g_gate_reason_n[i]);
    printf("\n[gate]   longest gated run: %d    threshold %.2f\n",
           g_gate_worst, (double)mosse::DEFAULT_PSR_MIN);
    printf("[gate]   PSR  min %.2f / mean %.2f / max %.2f  (evaluated frames only)\n",
           g_psr_min, g_psr_sum / g_gate_eval, g_psr_max);
    fflush(stdout);
}

// -----------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------
// A hw_emu frame costs ~45 min, so a run that only reports a displacement
// wastes most of what it computed. These print the numbers needed to tell
// "H is wrong" from "the response is wrong" from "F_ch arrives scrambled"
// WITHOUT another run, and cost nothing measurable.
//
// Two channels on purpose: the binary dumps are richer but land on the SD card
// mount, which may be read-only under QEMU, whereas stdout is captured in
// run_emu.log unconditionally. Anything decisive is printed, not just dumped.
#ifndef DUMP_BUFFERS
#  define DUMP_BUFFERS 1
#endif

// Best-effort binary dump. Tries the cwd (SD card) first, then /tmp on the
// target. Never fatal: losing a dump must not abort a 90-minute run.
static void dump_buffer(const char *tag, int frame, const void *p, size_t bytes)
{
#if DUMP_BUFFERS
    char path[128];
    snprintf(path, sizeof(path), "%s_f%d.bin", tag, frame);
    FILE *f = fopen(path, "wb");
    if (!f) {
        snprintf(path, sizeof(path), "/tmp/%s_f%d.bin", tag, frame);
        f = fopen(path, "wb");
    }
    if (!f) {
        printf("  [dump] %s: no writable location — stdout diagnostics only\n", tag);
        return;
    }
    const size_t wrote = fwrite(p, 1, bytes, f);
    fclose(f);
    printf("  [dump] %s -> %s (%zu of %zu B)\n", tag, path, wrote, bytes);
#else
    (void)tag; (void)frame; (void)p; (void)bytes;
#endif
}

// -----------------------------------------------------------------------
// Per-frame CSV — the run's actual product
// -----------------------------------------------------------------------
// One row per frame, ~40 B, flushed every row. The three things it exists for:
//
//   1. The binary dumps are 1216 KB/frame and set the frame rate of a board run
//      (see DUMP_BUFFERS in the Makefile). This carries what the tracking curves
//      need at 1/30000th the volume, so a long run can afford it.
//   2. stdout is a 115200-baud UART and the only surviving record of the
//      2026-08-17 run was a hand-copied fragment. A flushed CSV survives a power
//      cut up to the frame in progress.
//   3. The end-of-run [track] SUMMARY reports means. Means hide the thing that
//      actually happens here — a single missed frame costs a permanent ~9.4 px
//      offset, and that is a step in a per-frame series, invisible in an average.
//
// resp00_over_peak is the background-lock diagnostic: |resp(0,0)| / |peak|. A
// DCF fed a byte-identical static background correlates with it at zero shift,
// which produces a second peak competing with the true motion peak. On the
// 2026-08-17 run it sat at 0.69-0.86 and won 21 of 48 frames. Under ~0.3 is
// healthy. The number was computable from data already on stdout and nothing
// consumed it, which is why the failure took two runs to see.
#ifndef CSV_LOG
#  define CSV_LOG 1
#endif

static FILE *g_csv = nullptr;

static void csv_open(void)
{
#if CSV_LOG
    // Same best-effort placement as dump_buffer(): cwd (the SD card) first, then
    // /tmp. Never fatal — losing the CSV must not abort a 500-frame run.
    const char *path = "track.csv";
    g_csv = fopen(path, "w");
    if (!g_csv) { path = "/tmp/track.csv"; g_csv = fopen(path, "w"); }
    if (!g_csv) {
        printf("[csv] no writable location — stdout diagnostics only\n");
        return;
    }
    fprintf(g_csv,
            "frame,occluded,evaluated,accept,reason,psr_bolme,psr_ratio,peak,"
            "dr_bin,dc_bin,resp00_over_peak,est_row,est_col,est_h,est_w,"
            "truth_row,truth_col,truth_h,truth_w,iou,centre_err,published,"
            // The scale filter's own verdict. Added 2026-08-20: the previous run
            // logged only est_h, so the level the detector actually PROPOSED had
            // to be reverse-engineered from log(est_h/64)/log(a) — which is how
            // "the detector only ever proposes +-1" was found, slowly. Log it.
            "scale_idx,scale_conf,scale_reason\n");
    fflush(g_csv);
    printf("[csv] per-frame log -> %s\n", path);
#endif
}

static void csv_row(int frame, bool occluded, bool evaluated,
                    const mosse::GateDecision &gate,
                    const mosse::PsrResult &p, double resp00_over_peak,
                    const mosse::TargetBox &est, const mosse::TargetBox &truth,
                    double iou, double cerr, bool published,
                    bool scale_evaluated, int scale_idx,
                    const mosse::ScaleDecision &sd)
{
#if CSV_LOG
    if (!g_csv) return;
    fprintf(g_csv,
            "%d,%d,%d,%d,%s,%.4f,%.4f,%ld,%d,%d,%.4f,"
            "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.2f,%d,"
            "%d,%.4f,%s\n",
            frame, occluded ? 1 : 0, evaluated ? 1 : 0, gate.accept ? 1 : 0,
            mosse::gate_reason_tag(gate.reason),
            p.psr, p.ratio, p.peak, p.dr, p.dc, resp00_over_peak,
            est.row, est.col, est.h, est.w,
            truth.row, truth.col, truth.h, truth.w,
            iou, cerr, published ? 1 : 0,
            // NOT_RUN is distinct from ACCEPT with idx 0: frame 0, an occluded
            // frame and a PSR-gated frame never reach the scale filter at all,
            // and reporting those as "level 0, accepted" would read as a healthy
            // settled scale — the exact confusion this column exists to remove.
            scale_evaluated ? scale_idx : 0,
            scale_evaluated ? sd.conf : 0.0,
            scale_evaluated ? mosse::scale_veto_tag(sd.reason) : "NOT_RUN");
    // Per ROW, not per run: the whole point is surviving a power cut. A 500-frame
    // run writes 500 flushes of ~40 B, which is nothing against 1216 KB/frame of
    // binaries or 8.3 KB/frame of console.
    fflush(g_csv);
#else
    (void)frame; (void)occluded; (void)evaluated; (void)gate; (void)p;
    (void)resp00_over_peak; (void)est; (void)truth; (void)iou; (void)cerr;
    (void)published; (void)scale_evaluated; (void)scale_idx; (void)sd;
#endif
}

static void csv_close(void)
{
#if CSV_LOG
    if (g_csv) { fclose(g_csv); g_csv = nullptr; }
#endif
}

// Peak magnitude and saturation count for a cint16 buffer. `rails > 0` means the
// stage clipped, which is the failure mode the shift budget exists to prevent.
// Indices are reported in the buffer's own layout — the caller says which.
static void report_cint16(const char *tag, const int16_t *b, int rows, int cols,
                          const char *layout)
{
    const auto _ds0 = std::chrono::steady_clock::now();
    double max_m = -1.0;
    int    max_i = 0, rails = 0;
    for (int i = 0; i < rows * cols; ++i) {
        const double re = b[2 * i], im = b[2 * i + 1];
        const double m  = re * re + im * im;
        if (m > max_m) { max_m = m; max_i = i; }
        if (re >= 32767.0 || re <= -32768.0 || im >= 32767.0 || im <= -32768.0) ++rails;
    }
    g_ap_us[AP_DIAG_SCAN] += std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - _ds0).count();
    ++g_ap_n[AP_DIAG_SCAN];

    // Printed at VERBOSITY >= 1, but ALWAYS when something railed. `rails` is
    // the shift-budget instrument and it is the one number here that track.csv
    // does not carry, so a quiet run must still shout when a bin saturates —
    // silencing an anomaly to save console is how a budget hunt goes wrong.
    if (VERBOSITY >= 1 || rails > 0)
        printf("  [diag] %-9s max|.|=%7.0f at %s idx %d (%d,%d)  rails=%d%s\n",
               tag, sqrt(max_m), layout, max_i, max_i / cols, max_i % cols, rails,
               rails > 0 ? "   <-- RAILED" : "");
}

// The decisive report: where the response actually peaks, and — the number that
// settles it — what the response is AT the injected offset.
//
// If (IMPULSE_DR, IMPULSE_DC) is a strong local peak that merely lost a contest
// against something else, the correlation is working and an artifact is winning.
// If it is at the noise floor, the displacement information never arrived, which
// is a different bug entirely. The old pass/fail line could not tell these apart.
static void report_response(const int16_t *resp, int rows, int cols)
{
    // Trace only. The routine question this answers — "is the origin peak
    // taking over?" — is `resp00_over_peak` in track.csv, one number per frame
    // instead of ~500 B of profile.
    if (VERBOSITY < 2) return;
    // Top 5 by |real| — peak_detect_sw's own metric, so these are exactly the
    // candidates it chose between. Selection sort over 5 ranks: 5 linear passes
    // with an explicit exclusion list, which avoids sorting the whole map.
    constexpr int NTOP = 5;
    int taken[NTOP];
    int ntaken = 0;
    for (int rank = 0; rank < NTOP; ++rank) {
        long best = -1;
        int  best_i = -1;
        for (int i = 0; i < rows * cols; ++i) {
            bool skip = false;
            for (int t = 0; t < ntaken; ++t)
                if (taken[t] == i) { skip = true; break; }
            if (skip) continue;
            const long re  = resp[2 * i];
            const long mag = (re < 0) ? -re : re;
            if (mag > best) { best = mag; best_i = i; }
        }
        if (best_i < 0) break;
        taken[ntaken++] = best_i;
        int r = best_i / cols, c = best_i % cols;
        if (r > rows / 2) r -= rows;
        if (c > cols / 2) c -= cols;
        printf("  [diag] resp rank %d: (%3d,%3d) |re|=%ld\n", rank, r, c, best);
    }

    // Value at the injected offset, and at a few reference bins.
    const int er = ((IMPULSE_DR % rows) + rows) % rows;
    const int ec = ((IMPULSE_DC % cols) + cols) % cols;
    auto at = [&](int r, int c) {
        const long re = resp[2 * (r * cols + c)];
        return (long)((re < 0) ? -re : re);
    };
    printf("  [diag] resp at injected (%d,%d) = %ld   |   (0,0) = %ld, "
           "(%d,0) = %ld, (0,%d) = %ld\n",
           IMPULSE_DR, IMPULSE_DC, at(er, ec), at(0, 0),
           IMPULSE_DR, at(er, 0), IMPULSE_DC, at(0, ec));

    // Profiles through the injected row and column: this is what distinguishes
    // "the column axis carries no information" (flat profile) from "the peak
    // moved" (sharp profile in the wrong place). Printed coarsely to keep the
    // log readable — every 4th bin.
    printf("  [diag] row %d profile (every 4th col): ", IMPULSE_DR);
    for (int c = 0; c < cols; c += 4) printf("%ld ", at(er, c));
    printf("\n  [diag] col %d profile (every 4th row): ", IMPULSE_DC);
    for (int r = 0; r < rows; r += 4) printf("%ld ", at(r, ec));
    printf("\n");
}

// Load the INT8 conv2d weights exported by `make weights` into a host buffer.
// Layout per channel (64 B, see design/aie_src/weights/layer0.h):
//   [0:9] int8 3×3 kernel, [9] out_shift, [10:14] int32 bias_acc (LE)
// The file ships 16 channels; a build with fewer uses the leading prefix.
// Returns false (and leaves the buffer zeroed) if the file cannot be read, so a
// missing weights file degrades to "output is zero" instead of garbage.
static bool load_conv_weights(const char *path, uint8_t *dst, size_t bytes)
{
    memset(dst, 0, bytes);

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "WARNING: cannot open %s — conv2d weights left zeroed\n", path);
        return false;
    }
    f.read(reinterpret_cast<char *>(dst), (std::streamsize)bytes);
    size_t got = (size_t)f.gcount();
    if (got != bytes) {
        fprintf(stderr, "WARNING: %s short read (%zu of %zu bytes)\n", path, got, bytes);
        return false;
    }
    printf("loaded %zu bytes of conv2d weights from %s\n", bytes, path);
    return true;
}

// -----------------------------------------------------------------------
// Stage B — feature-map normalization (see conv2d_kernel.h for the rationale)
// -----------------------------------------------------------------------

// Per-channel window-weighted feature mean, fed back to conv2d as mean_prev in
// the next frame's weight buffer (bytes [18:22]). Zero on the first frame.
static int32_t g_mean_prev[N_CHANNELS] = {0};
// Per-channel spectral energy, for the B3 filter scaling.
static double  g_energy[N_CHANNELS]    = {0.0};

// Q1.15 periodic Hann, regenerated by `make weights` into hanning_<N>.h.
// The host needs the same table the kernel uses: B2's correction is built from
// this window's DFT.
//
// Selected by geometry, mirroring conv2d_kernel.cpp — the two MUST resolve to the
// same table. This used to be a hardcoded hanning_128.h, which the header's own
// PATCH_ROWS guard turned into a compile error on any non-128 build.
#if   PATCH_COLS == 128
#  include "hanning_128.h"
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

// Measure the window-weighted feature mean from the row-FFT output.
//
// The row-FFT DC bin of row r is Σ_c w_r·w_c·g[r,c], where g is whatever conv2d
// emitted (the post-ReLU map with mean_prev already removed). Summing over rows
// gives Σ(w⊗w)·g exactly, so this recovers the window-weighted mean with 128
// adds per channel over data the APU is about to transpose anyway — no extra
// traffic, no extra pass.
//
// Called BEFORE transpose_inplace(), while the layout is still row-major:
// element [r][k] lives at index r*PATCH_COLS + k, so the row DC bins are at
// stride PATCH_COLS.
static int32_t measure_window_mean(const int16_t *row_fft, int32_t mean_prev)
{
    // Σw over one axis, in Q1.15 units; the 2D weight sum is its square.
    int64_t sum_w = 0;
    for (int i = 0; i < PATCH_COLS; ++i) sum_w += (int64_t)HTAB[i];

    int64_t dc_sum = 0;
    for (int r = 0; r < PATCH_ROWS; ++r)
        dc_sum += (int64_t)row_fft[2 * (r * PATCH_COLS)];   // real part of bin 0

    // conv2d applies two >>15 stages, so the emitted sample is
    // g·w_r·w_c / 2^30; scale back to recover Σ(w⊗w)·g / (Σw)².
    const double sw = (double)sum_w / 32768.0;              // Σw in window units
    if (sw <= 0.0) return mean_prev;
    const double residual = (double)dc_sum / (sw * sw);

    // conv2d subtracted mean_prev before windowing, so what we just measured is
    // the mean of (f - mean_prev). Add it back to get the absolute mean. This is
    // a feedback loop: even if the scale factor above is slightly off, mean_prev
    // converges to the true window-weighted mean over a few frames.
    return mean_prev + (int32_t)llround(residual);
}

// Stage B2 — remove the residual pre-window mean from the accumulated spectrum.
//
// conv2d removed mean_prev, but the true mean is mean_now, so the spectrum still
// carries (mean_now - mean_prev)·W where W = DFT(w⊗w). For the PERIODIC Hann, W
// has exactly 9 non-zero bins at (r,c) ∈ {0,±1}², so only those 9 need touching,
// and because correlation is linear they can be handled once on the ACCUMULATED
// spectrum rather than per channel.
//
// TWO MODES. The default changed on 2026-08-11 from SUBTRACT to NULL:
//
// B2_NULL_BINS=0 — SUBTRACT (the original design)
//   Σ_ch (X_ch - µ_ch·W) ⊙ H_ch*  =  Σ_ch X_ch ⊙ H_ch*  -  Σ_ch µ_ch · W ⊙ H_ch*
//   Not bit-exact: conv2d's window multiply has two >>15 truncations, which are
//   nonlinear, so linearity holds only up to quantization. Measured relative
//   error after correction ~1e-3, versus 2.5e-2 .. 9.9 without.
//
// B2_NULL_BINS=1 — NULL (default)
//   Zero the 9 bins outright.
//
// HISTORY — read this before changing the default back.
//
// The subtracting mode was BROKEN until 2026-08-11: it omitted the >>H_SHIFT that
// matches cmul_accum's output scale (see the fix inside the loop below), so the
// correction landed 1024x oversized and SATURATED the very bins it was meant to
// clean. Measured in hw_emu at 64x64/ch1: the DC bin came out of cmul at 4000,
// the subtraction added ~28767, and the bin railed at 32767 every frame. That
// rail produced a response pedestal which at 3/0/6 left the true peak only 1.43x
// above it, and at 4/2/2 BEAT it outright — reported displacement (6,-1) with the
// top five peaks within 4% of each other, a broad hump rather than a peak.
//
// Note what this means for the diagnosis: the accumulator was never railed by the
// pipeline. cmul's output matches (F ⊙ conj(H)) >> H_SHIFT to within 1 LSB across
// the whole map. The corruption was injected HERE, on the host, after the FFT,
// which is why no FFT_SHIFT setting could fix it.
//
// WHY NULL IS STILL THE DEFAULT NOW THAT SUBTRACT IS FIXED: the correction, once
// correctly scaled, is only ~28 against a DC bin of 4000 — so it does not remove
// the pedestal, because most of that DC is NOT the residual window mean. It is
// the genuine DC that conv2d's ReLU creates by making the feature map
// non-negative, which Stage B1 only partly removes (CLAUDE.md records 5.4 bits
// of DC/AC remaining; measured 5.1 here). Nulling removes it; subtracting cannot.
//
// Measured on the device with nulling active, 64x64/ch1/4-2-2:
//   displacement (10,-7) exact, peak 12618, PSR 22.7, peak/pedestal 8.2x
// against PSR 2.1 for the same build without it. (Bolme §3.5 puts the failure
// indicator at ~7; aiesim s7 measures 19.6.)
//
// COST: nulling also discards genuine target energy — at sigma=2 the Gaussian G
// is broad in frequency, so bins (0,±1) carry real signal, and the DC bin held
// 4000. It is a workaround for the ReLU-driven DC, not a fix for it. The real fix
// is upstream: stop the feature map's mean from reaching the FFT at all. Note
// gmio_fft_col_out still reports rails=5 per frame, which is that same upstream
// DC and is NOT addressed by anything in this function.
#ifndef B2_NULL_BINS
#  define B2_NULL_BINS 1
#endif
static void apply_dc_correction(int16_t *accum, const int16_t *filter_all,
                                const double *residual_mean)
{
    // The 9 bins where DFT(w⊗w) is non-zero for a PERIODIC Hann, in the
    // TRANSPOSED layout the col-FFT produced: element [c * PATCH_ROWS + r] is
    // 2D spectrum bin (r, c). Indexing it row-major (r * PATCH_COLS + c) happens
    // to give the same answer on a square patch — the bin set is transpose-
    // symmetric — but it is wrong the moment PATCH_ROWS != PATCH_COLS.
    const int ridx[3] = { 0, 1, PATCH_ROWS - 1 };   // row-frequency index
    const int kidx[3] = { 0, 1, PATCH_COLS - 1 };   // column-frequency index

#if B2_NULL_BINS
    (void)filter_all;
    (void)residual_mean;

    // Report the magnitude being discarded. If these come back near 32767 the
    // bins were railed, which is the condition this mode exists for — and if
    // they are ever SMALL, the railing has been fixed upstream and the
    // subtracting mode becomes viable again. This one number tells you which.
    long max_removed = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const int  bin = kidx[b] * PATCH_ROWS + ridx[a];
            const long re  = accum[2 * bin];
            const long im  = accum[2 * bin + 1];
            const long m   = (long)llround(sqrt((double)(re * re + im * im)));
            if (m > max_removed) max_removed = m;
            accum[2 * bin]     = 0;
            accum[2 * bin + 1] = 0;
        }
    }
    // VP1, but ALWAYS when it rails — same rule as report_cint16()'s rails>0.
    // This line and the Q1.15 one below were the two that escaped the first pass
    // of console gating: 138 of the 183 B/frame the board actually printed at
    // VERBOSITY=0, i.e. 12 ms of a 180 ms frame.
    if (VERBOSITY >= 1 || max_removed >= 32000)
        printf("  [B2] nulled 9 low-frequency bins, max|removed|=%ld%s\n",
               max_removed,
               max_removed >= 32000 ? "  (RAILED — as expected)"
                                    : "  (not railed — reconsider B2_NULL_BINS=0)");
#else
    // W[k] for a periodic Hann of length L: W[0] = L/2, W[±1] = -L/4, else 0.
    // The two axes have different lengths when the patch is not square.
    const double wrow[3] = { PATCH_ROWS * 0.5, -PATCH_ROWS * 0.25, -PATCH_ROWS * 0.25 };
    const double wcol[3] = { PATCH_COLS * 0.5, -PATCH_COLS * 0.25, -PATCH_COLS * 0.25 };

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const int    bin = kidx[b] * PATCH_ROWS + ridx[a];
            const double Wrc = wrow[a] * wcol[b] / (double)(PATCH_ROWS * PATCH_COLS);

            double corr_re = 0.0, corr_im = 0.0;
            for (int ch = 0; ch < N_CHANNELS; ++ch) {
                // µ_ch · W ⊙ H_ch*  (H stored un-conjugated; cmul conjugates)
                const int16_t hr = filter_all[2 * (ch * PATCH_ELEMS + bin)];
                const int16_t hi = filter_all[2 * (ch * PATCH_ELEMS + bin) + 1];
                const double  m  = residual_mean[ch] * Wrc;
                corr_re += m * hr;
                corr_im -= m * hi;      // conj
            }

            // MATCH cmul_accum's OUTPUT SCALE. The kernel emits
            // (F ⊙ conj(H)) >> H_SHIFT, so a correction built from the raw Q1.15
            // filter is 2^H_SHIFT too large and must be shifted the same way.
            //
            // This was missing until 2026-08-11 and it was not a rounding-level
            // mistake: at H_SHIFT=10 the correction landed 1024x oversized and
            // SATURATED the bins it was supposed to clean. Measured in hw_emu at
            // 64x64/ch1 — the DC bin came out of cmul at 4000, this subtraction
            // added ~28767, and the bin railed at 32767 every frame. That rail
            // was the source of the response pedestal that beat the true peak at
            // 4/2/2, and no FFT_SHIFT could fix it because the corruption was
            // injected on the HOST, after the FFT. Correctly scaled the same
            // correction is ~28 against a bin of 4000.
            //
            // The error scales with N_CHANNELS (the sum above runs over
            // channels), so it is worse at the ch16 design point than at ch1.
            corr_re /= (double)(1 << CMUL_H_SHIFT);
            corr_im /= (double)(1 << CMUL_H_SHIFT);

            int32_t re = (int32_t)accum[2 * bin]     - (int32_t)llround(corr_re);
            int32_t im = (int32_t)accum[2 * bin + 1] - (int32_t)llround(corr_im);
            accum[2 * bin]     = (int16_t)(re >  32767 ?  32767 : (re < -32768 ? -32768 : re));
            accum[2 * bin + 1] = (int16_t)(im >  32767 ?  32767 : (im < -32768 ? -32768 : im));
        }
    }
#endif
}

// Persistent filter state (A_ch, B) across frames. See mosse_filter.h.
static mosse::FilterState g_filter;
// Target spectrum G, CENTRED at (0,0). Used for DETECTION scale-setting and for
// filter_init() on frame 0, where the crop really is centred on the target.
// Heap staging for BO contents. MEASURED 2026-08-20 (runs/run_0820_1539.log):
// the xrt::bo host mappings are WRITE-COMBINING — a bulk read out of one runs at
// 696 MB/s against 7359 MB/s heap-to-heap (10.6x), while writes INTO one manage
// 3470 MB/s (2.1x). So any loop that touches a BO element-by-element pays the
// uncached read on every access. Copy the whole buffer out once, compute on the
// copy, copy back only if the device needs the result. Sized once at startup;
// never reallocated, because a per-call std::vector would put a malloc inside
// the very loop being optimised.
// The scene, HOST-SIDE. scale_extract() takes 33 bilinear crops per call, two
// calls a frame, i.e. ~64k scattered reads — and it used to take them straight
// out of frame_bo, a write-combining mapping where every load is a DRAM round
// trip. Measured 6732 us/call against 261 us for scale_detect+update, so 96% of
// the scale filter's cost was the mapping and none of it the DSST maths.
//
// So the scene now lives on the heap and is PUSHED to the device once per frame.
// That direction is the cheap one: the probe measured writes into a BO at
// 3470 MB/s against reads out at 696, so 2 MB costs ~600 us to push and would
// cost ~2.9 ms to pull. inject_target_frame() also gets faster for free, since it
// was writing into the same uncached mapping.
//
// INVARIANT: g_frame_host is the authority and frame_bo is a copy of it. Anything
// that writes frame_bo directly (rc_control_cu_probe's zero-fill) must run BEFORE
// the first push, or it silently reverts the scene — the same ordering trap the
// background seeding already documents.
static std::vector<uint8_t> g_frame_host;
static std::vector<uint8_t> g_stage_a;   // row_bo in
static std::vector<uint8_t> g_stage_b;   // transpose destination -> row_bo out
static std::vector<uint8_t> g_stage_c;   // fcol_bo / resp_bo — a THIRD buffer on
                                         // purpose. Reusing g_stage_b for the
                                         // F_ch copy is safe only because the
                                         // transpose has already been flushed to
                                         // row_bo by then; that is an ordering
                                         // constraint nothing would enforce, and
                                         // 64 KB is cheaper than the bug.
static std::vector<mosse::cfloat> g_target(PATCH_ELEMS);
// Target spectrum G re-centred at THIS frame's measured displacement, rebuilt
// per accepted frame for filter_update(). See the note at the update site: the
// training target must sit where the object actually is in the patch being
// trained on, and that is not (0,0) on any frame after the first.
static std::vector<mosse::cfloat> g_target_shift(PATCH_ELEMS);
// Per-channel 2-D spectra drained from gmio_fft_col_out this frame.
static std::vector<mosse::cfloat> g_F_all((size_t)N_CHANNELS * PATCH_ELEMS);

// Convert one channel's cint16 tap output into the float spectrum the filter
// maths consumes. The tap delivers the col-FFT layout — element
// [k*PATCH_ROWS + m] is spectrum bin (m, k) — which is the TRANSPOSE of row-major.
// Un-transposing here means everything downstream (G, A, H) is row-major, and the
// filter written back to filter_bo has to be re-transposed on the way out. Doing
// it in one place beats carrying two layouts through the maths.
static void unpack_spectrum(const int16_t *src, mosse::cfloat *dst)
{
    for (int k = 0; k < PATCH_COLS; ++k)
        for (int m = 0; m < PATCH_ROWS; ++m) {
            const size_t s = (size_t)k * PATCH_ROWS + m;
            dst[(size_t)m * PATCH_COLS + k] =
                mosse::cfloat((float)src[2 * s], (float)src[2 * s + 1]);
        }
}

// Write the quantized filter into filter_bo, converting row-major back to the
// col-FFT layout cmul_accum consumes. Inverse of unpack_spectrum().
static void pack_filter(const int16_t *rowmajor, int16_t *dst_bo)
{
    for (int ch = 0; ch < N_CHANNELS; ++ch) {
        const int16_t *s = rowmajor + (size_t)ch * PATCH_ELEMS * 2;
        int16_t       *d = dst_bo   + (size_t)ch * PATCH_ELEMS * 2;
        for (int m = 0; m < PATCH_ROWS; ++m)
            for (int k = 0; k < PATCH_COLS; ++k) {
                const size_t si = (size_t)m * PATCH_COLS + k;
                const size_t di = (size_t)k * PATCH_ROWS + m;
                d[2 * di]     = s[2 * si];
                d[2 * di + 1] = s[2 * si + 1];
            }
    }
}

// Build H from the current filter state and push it to the device.
// Reports the Q1.15 scale and the peak magnitude: a spiky filter that leaves the
// response far below the cint16 rails shows up here and nowhere else.
static void publish_filter(xrt::bo &filter_bo, std::vector<int16_t> &scratch)
{
    float scale = 0.0f, max_abs = 0.0f;
    mosse::filter_quantize_q15(g_filter, g_energy, mosse::DEFAULT_EPS_REL,
                               scratch.data(), &scale, &max_abs);
    pack_filter(scratch.data(), filter_bo.map<int16_t *>());
    filter_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // VP1. NOT carried by track.csv, so at VERBOSITY=0 this number is only
    // visible via the report_cint16("H(q15)") line next to it, which prints
    // unconditionally whenever a bin rails — the failure this would warn about.
    VP1("  filter: Q1.15 scale %.4g, max|H| %.4g\n", (double)scale, (double)max_abs);
}

// -----------------------------------------------------------------------
// Test data injection (for hw_emu validation)
// -----------------------------------------------------------------------

// Generate a synthetic test frame: impulse at (impulse_row, impulse_col).
// Useful for functional validation of the pipeline.
static void inject_impulse_frame(uint8_t *frame_buf, int rows, int cols,
                                 int impulse_row, int impulse_col, uint8_t value)
{
    // Zero-fill the entire frame
    for (int i = 0; i < rows * cols; ++i)
        frame_buf[i] = 0;

    // Place impulse at specified location
    if (impulse_row >= 0 && impulse_row < rows &&
        impulse_col >= 0 && impulse_col < cols) {
        frame_buf[impulse_row * cols + impulse_col] = value;
    }
}

// Generate a synthetic test frame: an ASYMMETRIC structured target centred at
// (tr, tc) on a flat background.
//
// This replaces the single-pixel impulse as the tracking test pattern, for two
// independent reasons:
//
// 1. A one-pixel impulse is a degenerate MOSSE training image. After Stage A's
//    zero-mean/unit-L2 normalisation an isolated bright pixel on a zero field
//    quantizes to exactly one non-zero int8 sample (the background sits ~4000x
//    below it and rounds away), so the filter is trained on a single sample.
//
// 2. More importantly, it is SYMMETRIC, and that hides bugs. A centred impulse
//    gives a spectrum invariant under both transposition and conjugation, so a
//    transposed pack_filter()/unpack_spectrum() or a wrong conjugation produces
//    byte-identical results to the correct code. This was verified against a
//    float model: every one of those mistakes still localises perfectly on an
//    impulse. It is the same degeneracy mosse_filter.h warns about and the
//    reason aiesim s7 puts its target off-centre — frame 0 cannot be moved
//    off-centre here (G is centred), so the asymmetry must live in the target.
//
// The shape is asymmetric in three independent ways:
//   - extents differ between axes (11 rows x 5 cols), so a transpose is visible;
//   - the spur is on one side only, so point-symmetry is broken and a
//     conjugation error shows up;
//   - the two features have different amplitudes.
//
// The background is flat: Stage A removes the mean anyway, and flat background
// keeps the patch spectrum analysable. Every pixel is a function of (r-tr, c-tc)
// alone, so a later frame is EXACTLY the earlier frame translated — which makes
// the expected displacement unambiguous.
// Band-limited background, written into the whole frame.
//
// Band-limited rather than white noise on purpose: the padding experiments
// compare different resample ratios, and roi_crop's bilinear sampler has NO
// prefilter, so a broadband background would alias differently at every ratio and
// the comparison would be measuring aliasing of the test signal rather than
// padding. A smooth field resamples predictably.
//
// Structurally the same construction as scripts/synth_frame.py (six low-order
// sinusoids plus a small noise floor), but NOT bit-identical to it — the two use
// different RNGs. The sweep and this harness therefore agree on ordering and
// mechanism, not on absolute values, which is the same relationship every other
// model in this project has with the hardware.
static void fill_background(uint8_t *frame_buf, int rows, int cols)
{
#if FRAME_TEXTURE
    // Fixed LCG so the frame is reproducible across runs and machines. rand() is
    // libc-dependent, which the PSR fixtures already record as a reason to avoid it.
    uint32_t s = 20260816u;
    auto next = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (double)(s >> 8) / (double)(1u << 24);   // [0,1)
    };
    // WHOLE CYCLES PER FRAME, so the field is exactly periodic in both axes and
    // BG_PAN's wraparound is seamless. With continuous frequencies the row wrap is
    // a hard edge — measured 8.10 LSB mean discontinuity against 0.98 LSB between
    // interior rows, i.e. 8x the natural texture gradient, running the full width
    // of the frame. A 128-px ROI straddles it ~12% of frames at a fast pan, and an
    // artificial edge is exactly the kind of feature a DCF locks onto. Rounding
    // takes that to 1.02 LSB, indistinguishable from the interior.
    //
    // The LCG is consumed in the same order and the same number of times, so this
    // only quantizes the six frequencies (they were already in [1,6] cycles);
    // amplitudes, phases and the dither are untouched.
    struct { double fy, fx, ph, amp; } comp[6];
    for (auto &k : comp) {
        k.fy  = std::round(1.0 + 5.0 * next()) / (double)rows;
        k.fx  = std::round(1.0 + 5.0 * next()) / (double)cols;
        k.ph  = 2.0 * M_PI * next();
        k.amp = 0.4 + 0.6 * next();
    }
    double amp_sum = 0.0;
    for (const auto &k : comp) amp_sum += k.amp;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double f = 0.0;
            for (const auto &k : comp)
                f += k.amp * std::sin(2.0 * M_PI * (k.fy * r + k.fx * c) + k.ph);
            f /= amp_sum;                                  // roughly [-1, 1]
            const double v = 110.0 + 90.0 * 0.35 * f + 3.0 * (next() - 0.5);
            frame_buf[(size_t)r * cols + c] =
                (uint8_t)(v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v));
        }
    }
#else
    memset(frame_buf, 40, (size_t)rows * cols);   // the pre-2026-08-16 flat fill
#endif
}

// -----------------------------------------------------------------------
// Scene generation: static background + dirty-rect restore
// -----------------------------------------------------------------------
// fill_background() runs six sinusoids per pixel over 1080x1920 — 12.4 M sin()
// calls. Measured at 193 ms/frame on an x86 host, so ~0.6-1.2 s on the A72.
//
// That was free in hw_emu, where a frame costs ~11.5 h at ch16. On real hardware
// the pipeline is ~13-22 ms, so regenerating the background every frame would be
// 30-90x the entire design's frame time and the FPS measurement — one of the two
// things a board run is FOR — would be measuring this function.
//
// The background is static by construction (fixed LCG seed, no frame dependence),
// so it is generated ONCE and only the region that actually changed is restored.
// g_dirty tracks that region; the occluder dirties the whole frame, which is why
// the rect has to be able to grow to full size rather than being assumed
// target-sized.
static std::vector<uint8_t> g_background;

struct DirtyRect { int r0 = 0, c0 = 0, r1 = -1, c1 = -1; };   // inclusive; empty if r1 < r0
static DirtyRect g_dirty;

static void scene_init(int rows, int cols)
{
    g_background.resize((size_t)rows * cols);
    fill_background(g_background.data(), rows, cols);
    g_dirty = DirtyRect{};                       // frame starts equal to background
}

// -----------------------------------------------------------------------
// Background pan — camera motion, and the actual fix for background lock
// -----------------------------------------------------------------------
// WHY NOISE IS NOT THE FIX. `FRAME_NOISE` was introduced against a background
// that repeated byte-for-byte, and it worked — in the scene of 2026-08-17, where
// the frame buffer was never seeded and the ROI was mostly zeros, so the noise
// really was the dominant varying content. Once the buffer WAS seeded the same
// lever stopped working (resp00_over_peak back to 0.86; see the trap entry), and
// the reason is structural: **independent additive noise does not decorrelate a
// static pattern.** The background still correlates with itself at exactly zero
// shift; the extra variance inflates the MOSSE numerator and the shared
// denominator alike. Byte-identity was never the mechanism — high correlation
// was — so raising the amplitude cannot help, it only buries the target too.
//
// What real video actually provides is that the background is not the SAME
// background from frame to frame under the tracking window, because the camera
// moves. Panning reproduces exactly that, and it is the honest test: correlating
// this frame's ROI against a filter trained on a differently-offset patch of the
// same texture field gives no zero-shift peak, which is the whole point.
//
// SCOPED TO THE ROI, same argument as scene_add_noise(). The pipeline only reads
// the ROI window, so only that has to carry the new offset; the rest of the frame
// keeps whatever pan it last had and nobody looks. The existing dirty-rect
// machinery undoes it, so there is no new bookkeeping and no accumulation.
//
// SAMPLED, NOT REGENERATED. fill_background() is ~0.6-1.2 s on the A72. Reading
// the cached field at a wrapped offset costs two memcpys per row.
//
// The pan is CONSTANT-VELOCITY and independent of the target trajectory. Both
// matter: a pan that tracked the target would hold the background still in ROI
// coordinates and rebuild the bug, and a periodic pan could resonate with
// TRAJ_PERIOD. 31 and 47 are coprime to 1080 and 1920, so the offsets visit every
// row and column before repeating, and fill_background() uses whole cycles per
// frame so the wraparound is seamless.
//
// **THE MAGNITUDE HAD TO BE SWEPT** (scripts/bg_pan_sweep.py; the table is in the
// Makefile next to BG_PAN_R). The texture's shortest wavelength is 180 rows, so
// the obvious first guess of 3-5 px/frame moved the zero-shift correlation from
// 0.60 to 0.61 — nothing. It takes ~31/47 to reach 0.09. Re-sweep if the texture
// changes; the right pan is a property of its spectrum, not a constant.
//
// This should NOT move the shift budget: a panned sample of the same texture
// field has the same amplitude statistics as an unpanned one. Watch |F| on the
// first run anyway — that assumption is cheap to check and expensive to assume.
#ifndef BG_PAN
#  define BG_PAN 1
#endif
#ifndef BG_PAN_R
#  define BG_PAN_R 3
#endif
#ifndef BG_PAN_C
#  define BG_PAN_C 5
#endif

static int g_pan_r = 0, g_pan_c = 0;

static inline int scene_wrap(int v, int n) { v %= n; return v < 0 ? v + n : v; }

// Frame 0 must be pan (0,0): the startup seed memcpys the unpanned field into the
// whole frame buffer, and the training frame has to agree with it.
static void scene_set_pan(int frame)
{
#if BG_PAN
    g_pan_r = BG_PAN_R * frame;
    g_pan_c = BG_PAN_C * frame;
#else
    (void)frame;
#endif
}

// With g_pan_r = g_pan_c = 0 this is bit-identical to the original single-memcpy
// restore: sc == c0 and first == w, so the second memcpy is never reached.
static void scene_restore(uint8_t *frame_buf, int rows, int cols)
{
    if (g_dirty.r1 < g_dirty.r0) return;
    const int r0 = std::max(0, g_dirty.r0), r1 = std::min(rows - 1, g_dirty.r1);
    const int c0 = std::max(0, g_dirty.c0), c1 = std::min(cols - 1, g_dirty.c1);
    const int w  = c1 - c0 + 1;
    for (int r = r0; r <= r1; ++r) {
        const uint8_t *src = g_background.data()
                           + (size_t)scene_wrap(r + g_pan_r, rows) * cols;
        uint8_t       *dst = frame_buf + (size_t)r * cols + c0;
        const int sc    = scene_wrap(c0 + g_pan_c, cols);
        const int first = std::min(w, cols - sc);
        memcpy(dst, src + sc, (size_t)first);
        if (first < w) memcpy(dst + first, src, (size_t)(w - first));
    }
    g_dirty = DirtyRect{};
}

static void scene_mark_dirty(int r0, int c0, int r1, int c1)
{
    if (g_dirty.r1 < g_dirty.r0) { g_dirty = DirtyRect{r0, c0, r1, c1}; return; }
    g_dirty.r0 = std::min(g_dirty.r0, r0);  g_dirty.c0 = std::min(g_dirty.c0, c0);
    g_dirty.r1 = std::max(g_dirty.r1, r1);  g_dirty.c1 = std::max(g_dirty.c1, c1);
}

// Per-frame sensor noise over a rectangle — see FRAME_NOISE at the top.
//
// SCOPED TO THE ROI, NOT THE FRAME, and that is the whole reason this is cheap.
// The pipeline only ever sees the ROI: roi_crop reads exactly that window, and
// scale_extract reads box.h x box.w concentric inside it. Noising 1080x1920
// would be 2.07 M pixels of RNG plus a full-frame restore every frame; the ROI
// is ~130x130, about 122x less work — tens of microseconds against an 88 ms
// pipeline. Regenerating the six-sinusoid field is what must NOT happen here:
// fill_background() is 193 ms on x86 and ~0.6-1.2 s on the A72, which is why it
// is cached in the first place.
//
// The rect is marked dirty, so the EXISTING restore machinery undoes it at the
// start of the next frame exactly as it undoes the drawn target. No new
// bookkeeping, and the noise cannot accumulate frame over frame.
//
// One continuously-advancing LCG, seeded once: deterministic run to run (so a
// result is reproducible) but different every frame (which is the entire point).
// Reseeding per frame with a fixed seed would rebuild the bug.
static uint32_t g_noise_s = 0x9E3779B9u;

static void scene_add_noise(uint8_t *frame_buf, int rows, int cols,
                            int r0, int c0, int r1, int c1)
{
#if FRAME_NOISE > 0
    r0 = std::max(0, r0);  c0 = std::max(0, c0);
    r1 = std::min(rows - 1, r1);  c1 = std::min(cols - 1, c1);
    if (r1 < r0 || c1 < c0) return;

    constexpr int      A    = FRAME_NOISE;          // peak amplitude, LSB
    constexpr uint32_t SPAN = 2u * (uint32_t)A + 1u; // uniform over [-A, +A]

    for (int r = r0; r <= r1; ++r) {
        uint8_t *p = frame_buf + (size_t)r * cols;
        for (int c = c0; c <= c1; ++c) {
            g_noise_s = g_noise_s * 1664525u + 1013904223u;
            // High bits: the low bits of an LCG have short periods, and a short
            // period in the noise is a repeating background again.
            const int n = (int)((g_noise_s >> 16) % SPAN) - A;
            const int v = (int)p[c] + n;
            p[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    scene_mark_dirty(r0, c0, r1, c1);
#else
    (void)frame_buf; (void)rows; (void)cols;
    (void)r0; (void)c0; (void)r1; (void)c1;
#endif
}

// -----------------------------------------------------------------------
// Scripted target trajectory and size envelope
// -----------------------------------------------------------------------
// TRAJECTORY=0 reproduces the legacy behaviour exactly: the target is injected at
// pos + (IMPULSE_DR, IMPULSE_DC), i.e. RELATIVE TO THE TRACKER'S OWN ESTIMATE.
// That makes the expected displacement a constant and `err=0 px` meaningful, but
// it also means the tracker is marking its own homework — it cannot drift,
// because the ground truth follows wherever it goes.
//
// TRAJECTORY=1 puts the target on an ABSOLUTE closed ellipse around the initial
// centre. Ground truth no longer depends on the estimate, so drift becomes
// visible and measurable — and because the path is closed the run length is
// unbounded. The legacy scheme walks off a 1080-row frame at about frame 48
// (constant velocity (10,-7) from row 540), which is what capped hw_emu runs.
//
// Amplitudes and period are chosen so the peak per-frame step is ~9 px/axis,
// comparable to the (10,-7) the pipeline is already proven against.
#ifndef TRAJECTORY
#  define TRAJECTORY 0
#endif
#ifndef TRAJ_AMP_R
#  define TRAJ_AMP_R 180.0
#endif
#ifndef TRAJ_AMP_C
#  define TRAJ_AMP_C 180.0
#endif
#ifndef TRAJ_PERIOD
#  define TRAJ_PERIOD 120.0
#endif

// Size envelope. SCALE_TRAJ=0 holds the drawn target at TARGET_H/W, which is what
// shipped — and which means the scale filter has NOTHING TO TRACK: it can be
// shown not to crash and nothing more.
//
// The rate limit that matters: SCALE_STEP=1.02 bounds one frame's correction to
// 1.02^((S-1)/2) = 1.37x, but SCALE_ETA=0.025 means the model adapts slowly, so
// the envelope must change far more slowly than the filter's single-frame range.
// amp 0.3 over a 200-frame period peaks at 0.94%/frame, comfortably inside it.
#ifndef SCALE_TRAJ
#  define SCALE_TRAJ 0
#endif
#ifndef SCALE_TRAJ_AMP
#  define SCALE_TRAJ_AMP 0.30
#endif
#ifndef SCALE_TRAJ_PERIOD
#  define SCALE_TRAJ_PERIOD 200.0
#endif

// Periodic occlusion. OCCLUDE_PERIOD=0 keeps the legacy OCCLUDE_MASK, which is a
// 32-bit mask indexed by frame number — so it can only express frames 0..31, and
// `mask >> frame` is undefined once frame >= 32. For a run of hundreds of frames
// a period is both expressible and more useful.
//
// OCCLUDE_START is a WARM-UP: no occlusion before this frame, and the period runs
// from there. Without it the first occlusion lands on frame 1, i.e. the filter is
// occluded immediately after being initialised from a single patch — which tests
// the gate against a filter that has not converged and conflates two different
// failures if it goes wrong.
//
// Choosing it: the translation filter smooths at MOSSE_ETA (0.125 by default), so
// its time constant is 1/eta = 8 frames and it is essentially converged after
// ~3-4 of those. 30 frames is comfortably past that. THE SCALE FILTER IS MUCH
// SLOWER — SCALE_ETA is 0.025, so 40 frames per time constant and ~120 to
// converge. If a run is meant to test occlusion against a settled SCALE estimate
// rather than a settled position, this wants to be ~120, not 30.
#ifndef OCCLUDE_PERIOD
#  define OCCLUDE_PERIOD 0
#endif
#ifndef OCCLUDE_LEN
#  define OCCLUDE_LEN 1
#endif
#ifndef OCCLUDE_START
#  define OCCLUDE_START 30
#endif

static void target_pose(int frame, double row0, double col0,
                        double *row, double *col, double *scale)
{
#if TRAJECTORY
    const double th = 2.0 * M_PI * (double)frame / (double)(TRAJ_PERIOD);
    *row = row0 + (double)(TRAJ_AMP_R) * std::sin(th);
    *col = col0 + (double)(TRAJ_AMP_C) * (std::cos(th) - 1.0);   // starts at (row0,col0)
#else
    *row = row0; *col = col0;                                    // caller applies the offset
#endif
#if SCALE_TRAJ
    *scale = 1.0 + (double)(SCALE_TRAJ_AMP)
                 * std::sin(2.0 * M_PI * (double)frame / (double)(SCALE_TRAJ_PERIOD));
#else
    (void)frame; *scale = 1.0;
#endif
}

static bool frame_is_occluded(int frame, bool init_frame)
{
    if (init_frame) return false;               // never occlude the training frame
#if OCCLUDE_PERIOD
    if (frame < (OCCLUDE_START)) return false;  // warm-up: let the filter converge
    return ((frame - (OCCLUDE_START)) % (OCCLUDE_PERIOD)) < (OCCLUDE_LEN);
#else
    // Guard the shift: `mask >> frame` is UB for frame >= 32.
    return frame < 32 && (((OCCLUDE_MASK) >> frame) & 1) != 0;
#endif
}

// Draw the target at (tr, tc) with the given box size.
//
// The shape is the shipped bar-plus-spur, now SCALED by the target box instead of
// being fixed at 11x11. Its three asymmetries are load-bearing and every extent
// below is therefore a fraction of the box rather than an absolute pixel count:
// different extents in r and c catch a transpose, the one-sided spur catches a
// reflection, and neither is symmetric about the centre so a sign flip shows up.
// At target_h = target_w = 11 this reproduces the original exactly.
static void inject_target_frame(uint8_t *frame_buf, int rows, int cols,
                                int tr, int tc, int target_h, int target_w)
{
    constexpr uint8_t BAR_VALUE  = 220;
    constexpr uint8_t SPUR_VALUE = 150;
    constexpr double  NOMINAL    = 11.0;   // the box the shape was drawn in

    // Restore only what the previous frame dirtied, instead of regenerating the
    // whole background — see the note on scene_init().
    scene_restore(frame_buf, rows, cols);

    const double sh = (double)target_h / NOMINAL;
    const double sw = (double)target_w / NOMINAL;
    const int    r0 = (int)std::floor(tr - 5.0 * sh);
    const int    r1 = (int)std::ceil (tr + 5.0 * sh);
    // The shape spans dc in [-2*sw, 8*sw], so the column extent is asymmetric.
    scene_mark_dirty(r0, (int)std::floor(tc - 2.0 * sw) - 1,
                     r1, (int)std::ceil (tc + 8.0 * sw) + 1);

    for (int r = std::max(0, r0); r <= std::min(rows - 1, r1); ++r) {
        const double dr = (double)r - tr;
        if (dr < -5.0 * sh || dr > 5.0 * sh) continue;
        for (int c = 0; c < cols; ++c) {
            const double dc = (double)c - tc;
            uint8_t v = 0;
            if (dc >= -2.0 * sw && dc <= 2.0 * sw)
                v = BAR_VALUE;
            else if (dr >= 2.0 * sh && dc >= 3.0 * sw && dc <= 8.0 * sw)
                v = SPUR_VALUE;
            if (v) frame_buf[(size_t)r * cols + c] = v;
        }
    }
}

// Generate a synthetic test frame: gradient pattern (for edge/feature testing).
static void inject_gradient_frame(uint8_t *frame_buf, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Ramp from 0 (top-left) to 255 (bottom-right)
            int val = ((r * 256 / rows) + (c * 256 / cols)) / 2;
            frame_buf[r * cols + c] = (uint8_t)(val & 0xFF);
        }
    }
}

// Generate a synthetic test frame: checkerboard pattern.
static void inject_checkerboard_frame(uint8_t *frame_buf, int rows, int cols, int square_size)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int sq_r = r / square_size;
            int sq_c = c / square_size;
            frame_buf[r * cols + c] = ((sq_r + sq_c) & 1) ? 255 : 0;
        }
    }
    // The occluder overwrites everything, so the NEXT frame has to restore the
    // whole buffer, not just a target-sized rect. Getting this wrong would leave
    // checkerboard fragments in the background for the rest of the run.
    scene_mark_dirty(0, 0, rows - 1, cols - 1);
}

// -----------------------------------------------------------------------
// Main tracking loop
// -----------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------
    // Device and xclbin setup
    // ------------------------------------------------------------------
    xrt::device device(0);
    xrt::uuid   uuid = device.load_xclbin(argv[1]);

    // ------------------------------------------------------------------
    // AIE graph
    // ------------------------------------------------------------------
    xrt::graph gr(device, uuid, "mosse_graph");
    gr.run();  // run forever, driven by GMIO transactions in the loop

    // ------------------------------------------------------------------
    // GMIO handles (names must match MOSSE_graph constructor strings exactly)
    // ------------------------------------------------------------------
    xrt::aie::buffer gm_weights     (device, uuid, "gmio_weights");
    xrt::aie::buffer gm_fft_row_out (device, uuid, "gmio_fft_row_out");
    xrt::aie::buffer gm_fft_col_in  (device, uuid, "gmio_fft_col_in");
    xrt::aie::buffer gm_fft_col_out (device, uuid, "gmio_fft_col_out");
    xrt::aie::buffer gm_cmul_in     (device, uuid, "gmio_cmul_in");
    xrt::aie::buffer gm_accum_out   (device, uuid, "gmio_accum_out");
    xrt::aie::buffer gm_ifft_row_in (device, uuid, "gmio_ifft_row_in");
    xrt::aie::buffer gm_ifft_row_out(device, uuid, "gmio_ifft_row_out");
    xrt::aie::buffer gm_ifft_col_in (device, uuid, "gmio_ifft_col_in");
    xrt::aie::buffer gm_response    (device, uuid, "gmio_response");

    // ------------------------------------------------------------------
    // XRT BOs (host-accessible DDR buffers)
    // ------------------------------------------------------------------
    // Frame buffer for camera_capture output
    auto frame_bo   = xrt::bo(device, FRAME_BYTES,
                               xrt::bo::flags::normal, 0);
    // Shared row-FFT ↔ IFFT row scratch (cint16, 64 KB)
    auto row_bo     = xrt::bo(device, FFT_BYTES,
                               xrt::bo::flags::normal, 0);
    // Partial accumulator (cint16, 64 KB)
    auto accum_bo   = xrt::bo(device, ACCUM_BYTES,
                               xrt::bo::flags::normal, 0);
    // Combined cmul input: [filter_chunk | accum_chunk] interleaved per kernel invocation
    auto cmul_bo    = xrt::bo(device, CMUL_IN_BYTES,
                               xrt::bo::flags::normal, 0);
    // Filter H_ch* for all channels (cint16, 64 KB × N_CHANNELS)
    auto filter_bo  = xrt::bo(device, FILTER_BYTES * N_CHANNELS,
                               xrt::bo::flags::normal, 0);
    // Weights for all channels (64 B × N_CHANNELS)
    auto weights_bo = xrt::bo(device, WEIGHT_CH_BYTES * N_CHANNELS,
                               xrt::bo::flags::normal, 0);
    // Correlation response map (cint16, 64 KB)
    auto resp_bo    = xrt::bo(device, RESP_BYTES,
                               xrt::bo::flags::normal, 0);
    // Per-channel 2-D spectrum F_ch, drained from the gmio_fft_col_out tap
    auto fcol_bo    = xrt::bo(device, FFT_BYTES,
                               xrt::bo::flags::normal, 0);

    // ------------------------------------------------------------------
    // PL kernel handles
    // ------------------------------------------------------------------
    auto cam  = xrt::kernel(device, uuid, "camera_capture:{camera_capture_0}");
#if !ROI_CROP_USER_MANAGED
    auto crop = xrt::kernel(device, uuid, "roi_crop:{roi_crop_0}");
#endif
    // roi_crop is driven directly when ROI_CROP_USER_MANAGED=1. The xrt::kernel
    // above must NOT also exist in that mode: xrt::ip opens an EXCLUSIVE CU
    // context, and the two would contend for the same CU.

    // ------------------------------------------------------------------
    // One-time init
    // ------------------------------------------------------------------
    // conv2d weights: read the exported INT8 kernels into weights_bo and push
    // them to the device. Without this the AIE reads whatever was in DDR.
    // (No-op while conv2d is built with CONV2D_ECHO_TEST=1, which ignores them.)
    load_conv_weights(WEIGHTS_FILE, weights_bo.map<uint8_t *>(),
                      WEIGHT_CH_BYTES * N_CHANNELS);

    // SEED mean_prev BEFORE FRAME 0. Without this Stage B1 is INERT on the one
    // frame the filter is trained from, and at 16 channels that is fatal.
    //
    // layer0_weights.bin has bytes [18:22] = 0, so frame 0 ran with mean_prev=0
    // and conv2d emitted its full DC pedestal (bias_acc >> out_shift ~ 24689 for
    // ch0). filter_init then learned from a DC-dominated spectrum, and frame 1 —
    // where B1 IS active — applies a filter matched to features that no longer
    // exist. Modelled at the real design point (scripts/phase1_sweep.py on the
    // actual inject_target_frame patch), ch16, budget 5-2-2:
    //     mean_prev=0        response RAILED, peak/sidelobe ratio 1.00, peak
    //                        displaced to (8,120) — a flat saturated map
    //     seeded (this fix)  ratio 36.81, peak (10,121), response profile
    //                        d=2 0.5994 / d=6 -0.0087 against an ideal sigma=2
    //                        Gaussian's 0.6065 / 0.0111
    // The same model reproduces the measured hw_emu ch1 numbers to within a few
    // percent once mean_prev=0 is modelled (response 20263 vs 19668 measured,
    // ratio 3.89 vs 3.59), which is what identified this.
    //
    // bias_acc >> out_shift is the right seed because Stage A delivers a
    // zero-mean patch, so the post-conv mean is the bias term alone. Measured
    // against the converged value on ch0: 24689 predicted vs 24686 actual.
    {
        uint8_t *wb = weights_bo.map<uint8_t *>();
        for (int ch = 0; ch < N_CHANNELS; ++ch) {
            uint8_t *w = wb + ch * WEIGHT_CH_BYTES;
            int32_t bias;
            memcpy(&bias, w + 10, sizeof(int32_t));
            const int shift = (int)w[9];
            const int32_t seed = bias >> shift;
            memcpy(w + 18, &seed, sizeof(int32_t));
            g_mean_prev[ch] = seed;
        }
        printf("mean_prev seeded from bias_acc>>out_shift: ch0=%d ch%d=%d "
               "(Stage B1 is active on frame 0)\n",
               g_mean_prev[0], N_CHANNELS - 1, g_mean_prev[N_CHANNELS - 1]);
    }
    weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Frame 0 runs with a zeroed filter on purpose: it is the INITIALISATION pass,
    // whose only job is to produce F_ch so filter_init() has something to learn
    // from. Its response is discarded — with H = 0 it is identically zero, and
    // peak_detect_sw would report a meaningless (0,0).
    memset(filter_bo.map<void *>(), 0, FILTER_BYTES * N_CHANNELS);
    filter_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ------------------------------------------------------------------
    // Target box and the ROI derived from it
    // ------------------------------------------------------------------
    // The box is the tracker's state now, not just a position. roi = box *
    // padding, so the filter sees background context — which is the whole reason
    // both papers use a window larger than the object.
    mosse::TargetBox box;
    box.row = FRAME_ROWS / 2.0;
    box.col = FRAME_COLS / 2.0;
    box.h   = (double)TARGET_H;
    box.w   = (double)TARGET_W;

    // Recomputed per frame inside the loop, because the scale filter moves
    // box.h/box.w. This copy is for the startup report only.
    const mosse::RoiGeometry roi0 =
        mosse::roi_for(box, mosse::DEFAULT_PADDING, PATCH_ROWS, PATCH_COLS);

    float sigma_r = 0.0f, sigma_c = 0.0f;
    mosse::sigma_for(box, roi0, &sigma_r, &sigma_c);
    // sigma does NOT need recomputing per frame even under SIGMA_FROM_TARGET:
    // the target always occupies patch/padding patch pixels regardless of its
    // size in frame pixels, so the rule is scale-invariant and G is constant.

    // ------------------------------------------------------------------
    // DSST 1-D scale filter (docs/1609.06141v1.pdf §5.1)
    // ------------------------------------------------------------------
    mosse::ScaleFilter scale;
    mosse::scale_filter_config(scale, mosse::DEFAULT_SCALE_N,
                               mosse::DEFAULT_SCALE_STEP,
                               box.h, box.w, (float)SCALE_SIGMA_FACTOR);
    std::vector<mosse::cfloat> scale_sample(
        (size_t)(scale.enabled() ? scale.sample_elems() : 1));
    // The initial box, which the absolute drift bounds are measured against. The
    // bounds themselves now live with the rest of the scale gate — see
    // DEFAULT_SCALE_MIN_REL / MAX_REL and scale_gate() in mosse_filter.h. They
    // moved out of here because they were one veto of three and the other two
    // have to be applied in the same place and reported in the same vocabulary;
    // they also tightened from 0.25/4.0, which was so loose it never fired.
    const double box_h0 = box.h, box_w0 = box.w;

    printf("box: %.0fx%.0f px at (%.0f,%.0f)  padding %.2f  ->  roi %dx%d @(%d,%d)\n",
           box.h, box.w, box.row, box.col, (double)mosse::DEFAULT_PADDING,
           roi0.roi_h, roi0.roi_w, roi0.roi_row, roi0.roi_col);
    printf("     %.4f frame px per patch bin (the localisation quantum)  "
           "sigma %.2f/%.2f  target %.1f patch px\n",
           mosse::patch_dr_to_frame(1, roi0), (double)sigma_r, (double)sigma_c,
           mosse::target_h_in_patch(box, roi0));
    if (scale.enabled()) {
        printf("scale: DSST 1-D filter, S=%d step=%.3f eta=%.3f sigma=%.2f  "
               "template %dx%d (d=%d)  range %.0f%%..%.0f%%\n",
               scale.n_scales, (double)scale.step, (double)mosse::DEFAULT_SCALE_ETA,
               (double)scale.sigma, scale.tmpl_h, scale.tmpl_w, scale.dims(),
               100.0 * std::pow((double)scale.step, -(scale.n_scales - 1) / 2.0),
               100.0 * std::pow((double)scale.step, (scale.n_scales - 1) / 2.0));
        printf("scale gate: conf >= %.2f, argmax must be interior (|idx| < %d), "
               "box within %.2fx..%.2fx of %.0fx%.0f — a veto HOLDS the size and "
               "SKIPS the model update\n",
               (double)mosse::DEFAULT_SCALE_CONF_MIN, (mosse::DEFAULT_SCALE_N - 1) / 2,
               mosse::DEFAULT_SCALE_MIN_REL, mosse::DEFAULT_SCALE_MAX_REL,
               box_h0, box_w0);
    } else {
        printf("scale: DISABLED (SCALE_N=1) — box size is held fixed\n");
    }
    if (roi0.roi_h != PATCH_ROWS || roi0.roi_w != PATCH_COLS)
        printf("     NOTE resample is NOT 1:1 — roi_crop's bilinear interpolator "
               "is live. It is covered by `make test_roi_crop` (17 cases) and by "
               "nothing before 2026-08-16.\n");

    // Target spectrum: closed form, no FFT (see mosse_filter.h). CENTRED, and
    // that is correct for exactly two things: setting the response scale, and
    // filter_init() on frame 0, whose crop IS centred on the target.
    //
    // It is NOT the training target for any later frame. The old comment here
    // read "a target displaced by (dr,dc) must produce a peak at (dr,dc), so G
    // itself carries no offset" — true of DETECTION, false of TRAINING, and that
    // conflation was the tracker's primary failure mode until 2026-08-20. See
    // the filter_update() call site.
    //
    // Per-axis sigma, because DSST §6.1 anchors it to "the target size in the
    // translation dimensionS" and a non-square box has a different extent in
    // each.
    mosse::gaussian_target_spectrum(g_target.data(), PATCH_ROWS, PATCH_COLS,
                                    sigma_r, sigma_c, 0, 0);

    // Scratch for the row-major quantized filter, before pack_filter() converts
    // it to the col-FFT layout.
    g_stage_a.resize(FFT_BYTES);
    g_stage_b.resize(FFT_BYTES);
    g_stage_c.resize(FFT_BYTES > RESP_BYTES ? FFT_BYTES : RESP_BYTES);
    std::vector<int16_t> filter_scratch((size_t)N_CHANNELS * PATCH_ELEMS * 2);

    if (ITER_CNT < 2)
        printf("WARNING: ITER_CNT=%d. Frame 0 is consumed by filter initialisation,\n"
               "         so a single-frame run cannot test localisation. Build with\n"
               "         ITER_CNT=2 or more for a meaningful result.\n", ITER_CNT);
    printf("filter: sigma=%.1f eta=%.3f H_SHIFT=%d — frame 0 initialises, "
           "frame 1+ tracks\n",
           (double)mosse::DEFAULT_SIGMA, (double)mosse::DEFAULT_ETA, CMUL_H_SHIFT);
    // MAKE THE LOG SELF-DESCRIBING. runs/.last_cfg recorded ITER_CNT=500 for a
    // run that executed exactly 200 frames, and neither the frame count nor the
    // console level appeared anywhere in the log itself — so which of the two
    // was stale could not be settled from the artifact. A log that cannot state
    // its own configuration is one reflash away from the stale-card trap.
    printf("run: ITER_CNT=%d  VERBOSITY=%d  DUMP_BUFFERS=%d  shift %d-%d-%d  "
           "N_CHANNELS=%d  %dx%d\n",
           ITER_CNT, VERBOSITY, DUMP_BUFFERS,
           FFT_SHIFT_CFG, IFFT_ROW_SHIFT_CFG, IFFT_COL_SHIFT_CFG,
           N_CHANNELS, PATCH_ROWS, PATCH_COLS);
    fflush(stdout);

    // Tracked position, kept as ints because roi_crop takes integer coordinates.
    // They mirror box.row/box.col, which stay the authoritative state.
    int pos_row = (int)llround(box.row);
    int pos_col = (int)llround(box.col);

    // Where the target was actually injected, for the IoU report.
    mosse::TargetBox truth = box;

    // Anchor for the scripted trajectory. Fixed for the run, so the path is a
    // closed curve about the INITIAL centre and not about wherever the tracker
    // has wandered to.
    const double traj_row0 = box.row, traj_col0 = box.col;

    // Background is generated ONCE — see scene_init(). Regenerating it per frame
    // costs ~0.6-1.2 s on the A72, which would be 30-90x the whole pipeline.
    scene_init(FRAME_ROWS, FRAME_COLS);

    // Per-run tracking statistics. The point of a long hardware run is the CURVE,
    // not a single frame, and these are what make it a result rather than a smoke
    // test: mean IoU is the OTB-style overlap metric both papers report, and the
    // failure count is what a tracker is actually judged on.
    int    trk_eval = 0, trk_ok = 0, trk_lost = 0;
    double trk_iou_sum = 0.0, trk_iou_min = 1.0;
    double trk_cerr_sum = 0.0, trk_cerr_max = 0.0;
    // Scale-gate tally, mirroring the PSR gate's. Reported per REASON, not just
    // as a hold count: a run that holds on AT_SEARCH_RAIL and one that holds on
    // LOW_CONF need different next steps, and at these frame times the log has to
    // say which without a rerun.
    int    scale_n_eval = 0, scale_n_accept = 0, scale_n_hold = 0;
    int    scale_reason_n[8] = {0};
    double scale_conf_min_seen = 1e300, scale_conf_max_seen = -1e300;
#if TRAJECTORY
    printf("trajectory: ELLIPSE amp (%.0f,%.0f) px, period %.0f frames, "
           "peak step ~%.1f px/frame — closed path, run length unbounded\n",
           (double)TRAJ_AMP_R, (double)TRAJ_AMP_C, (double)TRAJ_PERIOD,
           2.0 * M_PI * (double)TRAJ_AMP_R / (double)TRAJ_PERIOD);
#else
    printf("trajectory: LEGACY relative offset (%d,%d) per frame — the target "
           "follows the ESTIMATE, so drift is not measurable, and the ROI leaves "
           "a %dx%d frame at about frame %d\n",
           IMPULSE_DR, IMPULSE_DC, FRAME_ROWS, FRAME_COLS,
           IMPULSE_DR ? (int)((FRAME_ROWS / 2 - PATCH_ROWS) / (IMPULSE_DR ? IMPULSE_DR : 1))
                      : 0);
#endif
#if SCALE_TRAJ
    printf("size envelope: %.2fx..%.2fx over %.0f frames (peak %.2f%%/frame, "
           "filter range %.2f%%/frame)\n",
           1.0 - (double)SCALE_TRAJ_AMP, 1.0 + (double)SCALE_TRAJ_AMP,
           (double)SCALE_TRAJ_PERIOD,
           100.0 * 2.0 * M_PI * (double)SCALE_TRAJ_AMP / (double)SCALE_TRAJ_PERIOD,
           100.0 * ((double)mosse::DEFAULT_SCALE_STEP - 1.0));
#else
    printf("size envelope: FIXED at %dx%d — the scale filter has nothing to "
           "track; set SCALE_TRAJ=1 to exercise it\n", TARGET_H, TARGET_W);
#endif
#if OCCLUDE_PERIOD
    printf("occlusion: %d frame(s) every %d, starting at frame %d "
           "(warm-up = %.1f translation time constants at eta=%.3f; "
           "%.1f scale time constants at eta=%.3f)\n",
           (int)(OCCLUDE_LEN), (int)(OCCLUDE_PERIOD), (int)(OCCLUDE_START),
           (double)(OCCLUDE_START) * (double)mosse::DEFAULT_ETA,
           (double)mosse::DEFAULT_ETA,
           (double)(OCCLUDE_START) * (double)mosse::DEFAULT_SCALE_ETA,
           (double)mosse::DEFAULT_SCALE_ETA);
#elif OCCLUDE_MASK
    printf("occlusion: legacy OCCLUDE_MASK=0x%x (frames 0-31 only)\n",
           (unsigned)(OCCLUDE_MASK));
#else
    printf("occlusion: none\n");
#endif

#if BG_PAN
    printf("background: PANNING %+d,%+d px/frame (camera motion) — wraps after "
           "%d/%d frames, so it does not repeat within the run\n",
           BG_PAN_R, BG_PAN_C, FRAME_ROWS / (BG_PAN_R ? BG_PAN_R : 1),
           FRAME_COLS / (BG_PAN_C ? BG_PAN_C : 1));
#else
    printf("background: STATIC — the ROI sees the same texture every frame, so a "
           "zero-shift peak will grow and win. Watch resp00_over_peak in "
           "track.csv; set BG_PAN=1 unless you are deliberately reproducing it\n");
#endif

    // ------------------------------------------------------------------
    // Per-frame tracking loop
    // ------------------------------------------------------------------
    csv_open();

    // HOISTED OUT OF THE PER-CHANNEL LOOP (was constructed at every channel of
    // every frame — 16 per frame, 8000 over a 500-frame run).
    //
    // Each xrt::run construction allocates a command BO and registers it with the
    // KDS scheduler, then tears it down again; none of that depends on the
    // channel. Reuse is the documented XRT pattern: set_arg / start / wait,
    // repeat. See the RC_* instrumentation above for why this became suspect.
    //
    // The four geometry arguments are frame-invariant, so they are set ONCE here.
    // Only roi_row/roi_col/roi_h/roi_w (per frame) and recompute (per channel)
    // are re-set in the loop. Arguments are set by explicit index for the reason
    // documented at the call site — the AXIS port at id=1 consumes a positional
    // slot, so positional assignment silently shifts every scalar after it.
    // Timed even though it now happens once: if a single construction costs
    // hundreds of ms, that alone explained the old per-channel figure. Printed at
    // startup rather than through rc_report_frame(), which resets every frame.
    const auto _rc_t0 = std::chrono::steady_clock::now();
#if ROI_CROP_USER_MANAGED
    CropIp crop_ip(device, uuid);
    crop_ip.set_static_args(frame_bo, (uint32_t)FRAME_ROWS, (uint32_t)FRAME_COLS,
                            (uint32_t)PATCH_ROWS, (uint32_t)PATCH_COLS);
    printf("[roi_crop] USER-MANAGED (xrt::ip) launch path, CU driven directly; "
           "KDS completion bypassed. Constructed in %.3f ms\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - _rc_t0).count());
#else
    xrt::run crop_run(crop);
    crop_run.set_arg(0, frame_bo);
    crop_run.set_arg(2, (uint32_t)FRAME_ROWS);
    crop_run.set_arg(3, (uint32_t)FRAME_COLS);
    crop_run.set_arg(8, (uint32_t)PATCH_ROWS);
    crop_run.set_arg(9, (uint32_t)PATCH_COLS);
    printf("[roi_crop] KDS (xrt::run) launch path. crop_run constructed once "
           "(hoisted): %.3f ms\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - _rc_t0).count());
#endif
    fflush(stdout);

    // The launch path is expressed as three lambdas so the per-channel call site
    // below stays textually identical between the two modes — same RC_T slots,
    // same timeline marks, same report. Only the bodies differ, which is what
    // makes the two directly comparable in one log.
    //
    // crop_release() is the KDS wait()/wait#2 pair. It is deliberately kept as a
    // no-op rather than deleted in user-managed mode: the report then still
    // prints both rows at ~0.00 ms, which documents in the log itself that
    // nothing is hiding there.
    auto crop_set_args = [&](uint32_t roi_row, uint32_t roi_col,
                             uint32_t roi_h, uint32_t roi_w, int ch) {
#if ROI_CROP_USER_MANAGED
        crop_ip.set_frame_args(roi_row, roi_col, roi_h, roi_w,
                               (uint32_t)((ch == 0) ? 1 : 0));
#else
        crop_run.set_arg(4,  roi_row);
        crop_run.set_arg(5,  roi_col);
        crop_run.set_arg(6,  roi_h);                 // roi_h, FRAME px
        crop_run.set_arg(7,  roi_w);                 // roi_w, FRAME px
        crop_run.set_arg(10, (uint32_t)((ch == 0) ? 1 : 0));
#endif
    };
    auto crop_start = [&]() {
#if ROI_CROP_USER_MANAGED
        crop_ip.start();
#else
        crop_run.start();
#endif
    };
    auto crop_poll_done = [&]() -> ert_cmd_state {
#if ROI_CROP_USER_MANAGED
        return crop_ip.poll_done();
#else
        return rc_poll_until_done(crop_run);
#endif
    };
    auto crop_release = [&]() {
#if !ROI_CROP_USER_MANAGED
        crop_run.wait();
#endif
    };

    // Control CU. Runs before the first frame so its zero-fill of frame_bo cannot
    // race the per-frame injection, and after the graph is up so the device state
    // matches what roi_crop will see. See rc_control_cu_probe().
    rc_control_cu_probe(cam, frame_bo, FRAME_ROWS, FRAME_COLS);

    // ------------------------------------------------------------------
    // BO-MAPPING ACCESS PROBE — is the ~40 ms in the element loops the MEMORY
    // or the CODE?
    // ------------------------------------------------------------------
    // Measured 2026-08-20: transpose 33 ns/element, window mean+energy 55, unpack
    // 63 — about 71 MB/s on a 64 KB buffer, i.e. 50-100x below what an A72 does on
    // cached DRAM. All three run directly on xrt::bo mappings. Either those
    // mappings are uncached/write-combining (fix: copy once, compute on the heap
    // copy) or the loops are simply bad code (fix: -O3 -mcpu, drop fp64).
    //
    // THE TWO HYPOTHESES PREDICT OPPOSITE RESULTS HERE, which is the point — this
    // probe can kill either one rather than only confirm the one being tested:
    //   uncached mapping  -> BO memcpy and BO sum are BOTH ~50-100x slower than heap
    //   bad codegen       -> BO memcpy ~= heap memcpy, and only the SUM is slow
    {
        const size_t N = FFT_BYTES;                 // 64 KB, the working size
        std::vector<uint8_t> heap_a(N), heap_b(N);
        int16_t *bo_p = row_bo.map<int16_t *>();
        memset(heap_a.data(), 1, N);
        // Warm up: first touch faults pages, and a cold TLB would be charged to
        // whichever case ran first.
        memcpy(heap_b.data(), heap_a.data(), N);
        memcpy((void *)bo_p, heap_a.data(), N);

        constexpr int REP = 64;
        auto bench = [&](const char *tag, auto &&fn) {
            fn();                                    // warm
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < REP; ++i) fn();
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count() / REP;
            printf("  %-28s %8.1f us   %8.1f MB/s\n", tag, us, N / us);
            return us;
        };
        printf("[mem] BO-mapping access probe, %zu B x %d reps:\n", N, REP);
        volatile long sink = 0;
        const double h_cpy = bench("memcpy heap -> heap",
            [&]{ memcpy(heap_b.data(), heap_a.data(), N); });
        const double r_cpy = bench("memcpy BO   -> heap  (read)",
            [&]{ memcpy(heap_b.data(), (const void *)bo_p, N); });
        bench("memcpy heap -> BO    (write)",
            [&]{ memcpy((void *)bo_p, heap_a.data(), N); });
        const double h_sum = bench("sum int16, heap", [&]{
            const int16_t *q = (const int16_t *)heap_a.data(); long a2 = 0;
            for (size_t i = 0; i < N / 2; ++i) a2 += (long)q[i] * q[i];
            sink = a2; });
        const double b_sum = bench("sum int16, BO mapping", [&]{
            long a2 = 0;
            for (size_t i = 0; i < N / 2; ++i) a2 += (long)bo_p[i] * bo_p[i];
            sink = a2; });
        const double b_sumf = bench("sum fp64, BO mapping", [&]{
            double a2 = 0.0;
            for (size_t i = 0; i < N / 2; ++i) a2 += (double)bo_p[i] * bo_p[i];
            sink = (long)a2; });
        (void)sink;
        printf("  -> BO read / heap read  = %.1fx      BO sum / heap sum = %.1fx\n",
               h_cpy > 0 ? r_cpy / h_cpy : 0.0, h_sum > 0 ? b_sum / h_sum : 0.0);
        printf("  -> fp64 sum / int sum on the SAME BO = %.2fx\n",
               b_sum > 0 ? b_sumf / b_sum : 0.0);
        printf("  -> VERDICT: %s\n",
               (r_cpy > 4.0 * h_cpy)
                 ? "UNCACHED MAPPING — the BO itself is slow to touch; copy once, "
                   "compute on the heap copy"
                 : "MAPPING IS FINE — the cost is in the loops; try -O3 -mcpu=cortex-a72 "
                   "and drop fp64");
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Seed the frame buffer with the generated background — ONCE.
    // ------------------------------------------------------------------
    // WITHOUT THIS, THE BACKGROUND WAS NEVER IN THE FRAME AT ALL. scene_init()
    // fills g_background, and g_background is read in exactly one place:
    // scene_restore(), which copies only the PREVIOUS frame's dirty rect. g_dirty
    // starts empty, nothing ever copied the whole thing in, and the camera_capture
    // zero-fill that used to initialise the buffer is commented out above. So the
    // frame the pipeline read was the BO as allocated, plus a target, plus
    // whatever narrow rect a previous frame happened to dirty.
    //
    // MEASURED, by replaying these exact scene functions offline (2026-08-18),
    // fraction of the 128x128 ROI that had never been written:
    //
    //   frame 0                          88.53%   <- the frame the filter trains on
    //   frames 2+, TRAJECTORY=0           6.92%   (a 6-row + 3-col leading band)
    //   frames 2+, TRAJECTORY=1        3.1-4.7%
    //   frames 2+, without FRAME_NOISE   55.68%   <- the HEAD-era code path
    //
    // The last row is the configuration that produced run_0_17-08-2026.txt, whose
    // frames 307-313 report ratio 1.08x, peak 7989 against max sidelobe 7379,
    // identical to +-3 every frame, box collapsed to 27x27, 302 px error. A patch
    // that is more than half a flat saturated field produces exactly that.
    //
    // WHAT IT COST, through roi_crop_ref's bit-exact Stage A: the band saturates
    // the int8 rail (clipped count == band count, exactly), which inflates the
    // patch sigma ~4.3x, which divides the REAL scene content by 4.3 before
    // quantization. Signal std reaching conv2d 7.0-7.6 instead of 32.2-32.4, i.e.
    // 77% lost on a normal frame and 88.7% on frame 0.
    //
    // WHAT IT DID NOT COST, because this was the tempting wrong answer: the band
    // sits at patch rows 123-127, where conv2d's Hann window is 177 against 32767
    // at the centre. Those rows carry 7.3e-6 of the windowed patch energy. The
    // band cannot correlate with anything and has nothing to do with the (0,0)
    // background lock that FRAME_NOISE fixed. The damage is entirely upstream, in
    // Stage A's normalisation, and is already baked into the int8 patch by the
    // time the window would have suppressed it.
    //
    // ONE memcpy, ~2 MB, ~1-2 ms once. No per-frame cost: the dirty-rect machinery
    // was always written on the assumption that this had happened.
    //
    // ORDER IS LOAD-BEARING, TWICE. It must come AFTER rc_control_cu_probe(),
    // which zero-fills frame_bo by design, and after scene_init() (which fills
    // g_background). Putting it next to scene_init(), where it naturally belongs,
    // silently hands the probe an opportunity to erase it.
    //
    // NOTE THE COUPLING: this raises the patch amplitude 4.3x and the response
    // ~9.2x, which is why the shift budget moved from 4-3-3 to 4-5-5 in the same
    // change. See the FFT_SHIFT block in the Makefile. Seeding without that is a
    // railed response.
    {
        g_frame_host.assign(g_background.begin(), g_background.end());
        memcpy(frame_bo.map<uint8_t *>(), g_frame_host.data(), FRAME_BYTES);
        frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        printf("[scene] frame buffer seeded with the generated background "
               "(%zu B). Before this the pipeline read an unwritten buffer "
               "outside the dirty rect — see the note here.\n", FRAME_BYTES);
        fflush(stdout);
    }

    for (int frame = 0; frame < ITER_CNT; ++frame) {

        dma_reset_frame();
        rc_reset_frame();
        ap_reset_frame();      // zeroes the slots AND starts the frame-body clock

        // Recomputed every frame: the scale filter moves box.h/box.w, so the
        // ROI is no longer a constant of the run.
        const mosse::RoiGeometry roi =
            mosse::roi_for(box, mosse::DEFAULT_PADDING, PATCH_ROWS, PATCH_COLS);

        // 1. Camera capture → DDR frame buffer (zeros the buffer)
        // SKIPPED for hw_emu: camera_capture zeros the full 1080×1920 frame at
        // II=1 (~2M PL cosim cycles), which dominates emulation runtime. The host
        // fully initializes the frame below via inject_impulse_frame() +
        // sync(TO_DEVICE), so this call is redundant. Re-enable when
        // camera_capture becomes a real MIPI/V4L2 capture source.
        // {
        //     auto run = cam(frame_bo, FRAME_ROWS, FRAME_COLS);
        //     run.wait();
        // }

        // 1b. Inject test data into frame buffer (for hw_emu validation)
        //
        // The impulse is placed OFF-CENTRE, at pos + (IMPULSE_DR, IMPULSE_DC).
        // A centred impulse is useless as a test: the expected displacement is
        // then (0,0), which is also exactly what peak_detect_sw returns for an
        // all-zero response — so it passes without the data being looked at.
        // Off-centre, a correct pipeline MUST report (IMPULSE_DR, IMPULSE_DC).
        //
        // The defaults are deliberately asymmetric — different magnitudes and
        // opposite signs — so a row/col swap (a transpose bug) and a sign flip
        // are both caught, not just "something non-zero came out".
        // TODO: Replace with real video capture loop (OpenCV, V4L2).
        //
        // Frame 0 is the exception: the impulse goes at pos EXACTLY. That frame
        // trains the filter, and G is centred, so the target it learns must sit at
        // the patch centre. Injecting it off-centre on frame 0 would teach the
        // filter that the target IS the offset, and frame 1 would then correctly
        // report (0,0) — a passing-looking result from a filter trained on the
        // wrong thing.
        //
        // `occluded` outlives the block: the diagnostics further down report
        // against the injected offset, which does not exist on an occluded frame.
        bool occluded = false;
        {
            uint8_t *frame_ptr = g_frame_host.data();
            // Keyed off the filter state, not off `frame == 0`. The two agree in
            // the current flow, but "is the filter trained yet" is the condition
            // that actually decides where the target must be, and the init branch
            // at the bottom of the loop uses the same one.
            const bool init_frame = !g_filter.initialized;

            // Where the target actually is this frame, and how big.
            //
            // TRAJECTORY=0 (default) keeps the legacy RELATIVE scheme: the target
            // is placed at the tracker's own estimate plus a fixed offset, so the
            // expected displacement is the constant (IMPULSE_DR, IMPULSE_DC).
            // TRAJECTORY=1 uses an ABSOLUTE scripted path, which is what makes
            // drift measurable and the run length unbounded.
            double traj_r = 0.0, traj_c = 0.0, traj_s = 1.0;
            target_pose(frame, traj_row0, traj_col0, &traj_r, &traj_c, &traj_s);
#if TRAJECTORY
            int test_row = (int)llround(traj_r);
            int test_col = (int)llround(traj_c);
            (void)init_frame;
#else
            int test_row = pos_row + (init_frame ? 0 : IMPULSE_DR);
            int test_col = pos_col + (init_frame ? 0 : IMPULSE_DC);
#endif
            const int test_h = std::max(4, (int)llround((double)TARGET_H * traj_s));
            const int test_w = std::max(4, (int)llround((double)TARGET_W * traj_s));

            // Occlusion test for the PSR gate. Never on the training frame — see
            // frame_is_occluded() for why the legacy mask is bounded at frame 32.
            occluded = frame_is_occluded(frame, init_frame);

            // Advance the camera pan, then mark the window the pipeline is about
            // to read as dirty. BOTH are needed and the order is load-bearing.
            //
            // scene_restore() — called at the top of inject_target_frame(), i.e.
            // immediately after this — repaints the dirty rect from the background
            // at the CURRENT pan. Without the extra mark it would only repaint
            // what the PREVIOUS frame dirtied, and the ROI moves, so the part of
            // the new ROI outside the old rect would still carry the old offset:
            // a seam straight through the patch the filter is about to see. That
            // is worse than no pan at all — an artificial edge is a feature the
            // tracker would happily lock onto.
            //
            // scene_mark_dirty() unions, so this composes with the target and
            // noise rects rather than replacing them. Centred on the TRACKER's
            // position, not the target's, for the same reason scene_add_noise()
            // is: when the two diverge it is the tracker's window that gets read.
            // +4 px of margin covers roi_crop's bilinear taps at the border.
            scene_set_pan(frame);
            {
                const int pr = pos_row - roi.roi_h / 2, pc = pos_col - roi.roi_w / 2;
                scene_mark_dirty(pr - 4, pc - 4,
                                 pr + roi.roi_h + 3, pc + roi.roi_w + 3);
            }

            if (occluded) {
                // A checkerboard, not a flat fill: full-scale structure with real
                // spectral content in every band, so Stage A's zero-mean/unit-L2
                // does not degenerate, and no copy of the target anywhere. That
                // makes it the honest "the object is behind something" case, and
                // it exercises Bolme's ACTUAL threshold. A flat occluder would
                // produce a near-zero response and trip ZeroResponse or
                // FlatSidelobe instead, which tests the structural guards rather
                // than the PSR one.
                inject_checkerboard_frame(frame_ptr, FRAME_ROWS, FRAME_COLS,
                                          OCCLUDE_SQUARE);
            } else {
                // Asymmetric structured target, not a single-pixel impulse: an
                // impulse is symmetric, and a symmetric training patch makes a
                // transposed pack_filter() and a wrong conjugation both invisible.
                // See inject_target_frame().
                AP_T(AP_SCENE, inject_target_frame(frame_ptr, FRAME_ROWS, FRAME_COLS,
                                    test_row, test_col, test_h, test_w));
                // Ground truth for the IoU report. Taken from what was actually
                // DRAWN, not from the tracker's box — that is the whole point of
                // TRAJECTORY=1, where the two are allowed to diverge.
                truth.row = test_row;
                truth.col = test_col;
                truth.h   = (double)test_h;
                truth.w   = (double)test_w;
            }
            // Per-frame sensor noise over the ROI — see FRAME_NOISE at the top.
            // AFTER the target/occluder is drawn, so the target is noisy too (a
            // noiseless target on a noisy background would be its own giveaway),
            // and BEFORE the sync, so the device sees it.
            //
            // The rect is the ROI the pipeline will actually read, centred on the
            // TRACKER's position rather than the target's — when the two diverge
            // it is the tracker's window that must be noisy. +4 px of margin
            // covers roi_crop's bilinear taps at the border.
            //
            // Applied on occluded frames too: inject_checkerboard_frame() is just
            // as perfectly repeating as the background is.
            {
                const int nr = pos_row - roi.roi_h / 2, nc = pos_col - roi.roi_w / 2;
                scene_add_noise(frame_ptr, FRAME_ROWS, FRAME_COLS,
                                nr - 4, nc - 4,
                                nr + roi.roi_h + 3, nc + roi.roi_w + 3);
            }
            // Push the host scene to the device, then flush. The memcpy is the
            // price of keeping the authority on the heap; the probe puts it at
            // ~600 us for 2 MB, against the ~13 ms of scattered uncached reads it
            // removes from scale_extract.
            AP_T(AP_FRAME_PUSH,
                 memcpy(frame_bo.map<uint8_t *>(), g_frame_host.data(),
                        FRAME_BYTES));
            AP_T(AP_FRAME_SYNC,
                 frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE));  // Flush host → device
            if (occluded)
                printf("Frame %d: [OCCLUDED] checkerboard (square %d) — no target "
                       "in frame. The gate MUST hold position and freeze the "
                       "filter.\n", frame, OCCLUDE_SQUARE);
            else if (init_frame)
                VP1("Frame %d: [INIT] target CENTRED at (%d,%d)\n",
                       frame, test_row, test_col);
            else
                VP1("Frame %d: target at pos+(%d,%d) = (%d,%d)\n",
                       frame, IMPULSE_DR, IMPULSE_DC, test_row, test_col);
            fflush(stdout);
        }

        // 2. Per-channel: conv2d + FFT2D + cmul_accum
        //
        // The ROI is now derived from the BOX, not from the patch size. It is
        // roi.roi_h x roi.roi_w frame pixels centred on the tracked position, and
        // roi_crop resamples that to the fixed PATCH_ROWS x PATCH_COLS. Both
        // offsets may be negative for a target near an edge; roi_crop clamps by
        // border replication.
        int roi_row = pos_row - roi.roi_h / 2;
        int roi_col = pos_col - roi.roi_w / 2;

        // Stage B2: per-channel mean that conv2d did NOT remove this frame,
        // filled in as each channel's row-FFT comes back and applied once to the
        // accumulated spectrum after the loop.
        double residual_mean[N_CHANNELS] = {0.0};

        for (int ch = 0; ch < N_CHANNELS; ++ch) {

            VP2("[ch %d] START\n", ch);

            // ORDERING IS LOAD-BEARING — do not hoist the weights loop above this.
            //
            // conv2d consumes one `weights` buffer per firing, and a firing cannot
            // complete until it has read its share of the patch from the PatchIn
            // stream. Only ~2 weight buffers fit in flight (ping-pong), so sending
            // all CONV_INVOCATIONS of them with a blocking wait *before* starting
            // roi_crop deadlocks on the 2nd one: the host waits for the AIE to
            // free a buffer, the AIE waits for patch data that the host has not
            // sent yet. That deadlock previously looked like "the PatchIn PLIO
            // never delivers".
            //
            // So: start the patch flowing, then feed weights and drain the
            // row-FFT output together — conv2d drains the weights as it consumes
            // the patch, and fft_rows drains into DDR as conv2d feeds it.
            //
            // roi_crop → PatchIn PLIO → conv2d → fft_rows → gmio_fft_row_out
            //
            // recompute = (ch == 0): Stage A (resample + log + zero-mean +
            // unit-L2 + int8 quantize) depends only on the frame and the ROI,
            // not on the channel, so it runs once per frame and the remaining
            // channels re-stream the cached patch. See roi_crop.h.
            //
            // roi_h/roi_w are the ROI's size in FRAME pixels; roi_crop resamples
            // that to PATCH_ROWS × PATCH_COLS. They now come from the target box
            // (roi = box × TARGET_PADDING). Scale estimation, when it lands, only
            // has to update box.h/box.w and recompute `roi` per frame — the
            // plumbing is already here.
            // Arguments MUST be set by explicit index, NOT positionally via
            // `crop(...)`. roi_crop's AXIS port sits in the MIDDLE of the
            // argument list, and xrt::kernel::operator() assigns positionally
            // from index 0 — including over the stream port, which is not a
            // settable register. kernel.xml:
            //     id=0  frame_buf   (m_axi)
            //     id=1  patch_out   (AXIS  <-- consumes a positional slot)
            //     id=2  frame_rows      id=3  frame_cols
            //     id=4  roi_row         id=5  roi_col
            //     id=6  roi_h           id=7  roi_w
            //     id=8  patch_rows      id=9  patch_cols
            //     id=10 recompute
            // The old positional call shifted everything after frame_buf down by
            // one: patch_cols received (ch==0)?1:0 and recompute was never set at
            // all, so roi_crop emitted ~nothing and always skipped Stage A. That
            // is what produced "PL->AIE PLIO delivers nothing" and 0.00 MBps on
            // S00_AXIS. Confirmed on the plio_smoke testcase, which had the same
            // bug in miniature: TVALID never asserted while TREADY stayed high.
            // crop_run is HOISTED above the frame loop and reused — see there.
            // frame_buf / frame_rows / frame_cols / patch_rows / patch_cols are
            // already set and never change; only the ROI geometry (per frame) and
            // recompute (per channel) are re-set here.
            RC_T(RC_ARGS, crop_set_args((uint32_t)roi_row, (uint32_t)roi_col,
                                        (uint32_t)roi.roi_h, (uint32_t)roi.roi_w,
                                        ch));
            // Timeline zero. Everything the drain-anchored verdict rests on is
            // measured from here — see rc_tl_begin().
            rc_tl_begin();
            RC_T(RC_START, crop_start());
            rc_tl_mark(TL_START);

            // Feed one weight buffer per conv2d firing AND drain one row-FFT
            // window per firing, in the same loop.
            //
            // These MUST interleave. Draining after the weights loop deadlocks
            // (fft_rows fills its one armed output window, blocks conv2d, and the
            // weights queue freezes — the observed 6-of-64 stall). Draining before
            // it deadlocks the other way, waiting on data conv2d has not been fed.
            // The two counts are equal by construction (static_assert above), so
            // one loop serves both.
            // Depth `dd` firings are issued before either port is waited on.
            // dd == 1 is byte-for-byte the historical loop. gmio::wait() drains
            // ALL outstanding transfers on its port, so the barrier count is
            // CONV_INVOCATIONS/dd — which is exactly the quantity under test.
            const int dd = drain_depth_for_frame(frame);
            for (int k0 = 0; k0 < CONV_INVOCATIONS; k0 += dd) {
                const int kn = (k0 + dd < CONV_INVOCATIONS) ? k0 + dd
                                                            : CONV_INVOCATIONS;
                for (int k = k0; k < kn; ++k) {
                    // Arm this firing's drain before feeding the weights that
                    // trigger it, so the output window is never the thing that
                    // blocks.
                    DMA_TX(DMA_FFT_ROW_OUT,
                        gm_fft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                             ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES));
                    DMA_TX(DMA_WEIGHTS,
                        gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                         WEIGHT_CH_BYTES, ch * WEIGHT_CH_BYTES));
                }
                DMA_T(DMA_WEIGHTS,     gm_weights.wait());
                DMA_T(DMA_FFT_ROW_OUT, gm_fft_row_out.wait());
            }
            // Marked BEFORE the printf: this line is ~50 chars, i.e. ~4 ms of
            // console at 115200 baud, and it sits between the drain and the poll.
            // 4 ms against a 500 ms gap changes no conclusion, but it would be
            // charged to the completion path rather than to the console, and this
            // instrumentation exists because of exactly that kind of leak.
            rc_tl_mark(TL_DRAIN);
            VP2("[ch %d] weights sent + row-FFT drained (%d x %zu B)\n",
                ch, CONV_INVOCATIONS, ROW_CHUNK_BYTES);

            // Poll to completion FIRST, then wait(). Both are timed; the gap
            // between them is the whole question — see the RC_POLL note above.
            // wait() is still called: it is what actually releases the run object
            // for reuse, and it must be correct regardless of what poll reports.
            ert_cmd_state _st = ERT_CMD_STATE_COMPLETED;
            RC_T(RC_POLL, _st = crop_poll_done());
            rc_tl_mark(TL_POLL);
            RC_T(RC_WAIT, crop_release());
            rc_tl_mark(TL_WAIT);
            // Second wait on the same completed command — see RC_WAIT2. Free when
            // the completion path is healthy, and the whole answer when it is not.
            // A no-op in user-managed mode; see crop_release().
            RC_T(RC_WAIT2, crop_release());
            rc_tl_mark(TL_WAIT2);
            rc_tl_end();
            if (_st != ERT_CMD_STATE_COMPLETED)
                printf("[ch %d] roi_crop WARNING: terminal state %d, not COMPLETED\n",
                       ch, (int)_st);
            VP2("[ch %d] roi_crop done\n", ch);
            VP2("[ch %d] fft_row_out received\n", ch);

            // Stage B: measure this channel's window-weighted feature mean and
            // spectral energy BEFORE transposing, while the row-major layout
            // still puts each row's DC bin at stride PATCH_COLS.
            // ONE bulk read of row_bo, then mean, energy AND the transpose all
            // run on the heap copy — three passes that used to pay the uncached
            // load individually.
            AP_T(AP_BO_STAGE,
                 memcpy(g_stage_a.data(), row_bo.map<void *>(), FFT_BYTES));
            {
                const int16_t *rf = (const int16_t *)g_stage_a.data();
                const auto _wm0 = std::chrono::steady_clock::now();
                const int32_t  mean_now = measure_window_mean(rf, g_mean_prev[ch]);
                // B2 needs what conv2d failed to remove this frame.
                residual_mean[ch] = (double)(mean_now - g_mean_prev[ch]);
                g_mean_prev[ch]   = mean_now;

                // B3: Parseval energy, for the filter scaling in the update step.
                //
                // int64, NOT double, and this is BIT-EXACT rather than an
                // approximation worth arguing about: every term is a product of
                // two int16, so |term| <= 32767^2 = 1.07e9, and the full sum over
                // 2*PATCH_ELEMS terms is at most 3.5e13 — below 2^53 = 9.0e15, so
                // the old double accumulator was already representing exact
                // integers at every step. Same value, and the probe measured fp64
                // at 4.02x the int cost on the SAME buffer.
                int64_t e = 0;
                for (int i = 0; i < PATCH_ELEMS; ++i)
                    e += (int64_t)rf[2*i] * rf[2*i] + (int64_t)rf[2*i+1] * rf[2*i+1];
                g_energy[ch] = (double)e / (double)PATCH_ELEMS;

                // Feed mean_now back as the next frame's mean_prev (bytes 18:22).
                uint8_t *wb = weights_bo.map<uint8_t *>() + ch * WEIGHT_CH_BYTES;
                memcpy(wb + 18, &mean_now, sizeof(int32_t));
                g_ap_us[AP_WINMEAN] += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - _wm0).count();
                ++g_ap_n[AP_WINMEAN];
            }

            // APU: transpose row-FFT output in-place
            AP_T(AP_TRANSPOSE,
                 transpose_to(g_stage_a.data(), g_stage_b.data(),
                              PATCH_ROWS, PATCH_COLS, 4));
            AP_T(AP_BO_STAGE,
                 memcpy(row_bo.map<void *>(), g_stage_b.data(), FFT_BYTES));
            VP2("[ch %d] transpose done\n", ch);

            // Pack [filter_chunk_c | accum_chunk_c] into cmul_bo for all chunks.
            // For ch=0 the accum half is zeroed; for ch>0 it carries the running sum.
            {
                const auto _cp0 = std::chrono::steady_clock::now();
                int16_t *flt = filter_bo.map<int16_t*>() + ch * (int)(PATCH_ELEMS * 2);
                int16_t *acc = (ch == 0) ? nullptr : accum_bo.map<int16_t*>();
                int16_t *dst = cmul_bo.map<int16_t*>();
                for (int c = 0; c < CMUL_N_CHUNKS; ++c) {
                    memcpy(dst + c * 2 * CMUL_CHUNK_INT16,
                           flt + c * CMUL_CHUNK_INT16,
                           CMUL_CHUNK_INT16 * sizeof(int16_t));
                    if (acc)
                        memcpy(dst + c * 2 * CMUL_CHUNK_INT16 + CMUL_CHUNK_INT16,
                               acc + c * CMUL_CHUNK_INT16,
                               CMUL_CHUNK_INT16 * sizeof(int16_t));
                    else
                        memset(dst + c * 2 * CMUL_CHUNK_INT16 + CMUL_CHUNK_INT16, 0,
                               CMUL_CHUNK_INT16 * sizeof(int16_t));
                }
                g_ap_us[AP_CMUL_PACK] += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - _cp0).count();
                ++g_ap_n[AP_CMUL_PACK];
            }

            // Feed transposed data to col-FFT + combined [filter|accum] to cmul_accum
            DMA_TX(DMA_FFT_COL_IN,
                gm_fft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0));
            DMA_TX(DMA_CMUL_IN,
                gm_cmul_in.async(cmul_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, CMUL_IN_BYTES, 0));

            // Drain the accumulator WHILE the inputs are still in flight.
            //
            // This must not be sequenced after the two input waits: cmul stalls on
            // a full output window, which stalls the column FFT that feeds it,
            // which stalls the very input DMAs we would be waiting on. Draining
            // first breaks that cycle.
            // The F_ch tap drains in the SAME loop: gmio_fft_col_out and
            // gmio_accum_out are driven 1:1 by the same col-FFT invocation, so
            // this adds one async/wait per iteration rather than a second loop
            // (~1024 more per frame, about +16% on the host's DMA transactions).
            // It must be armed here for the same reason accum_out is — an
            // un-drained output window stalls the col FFT, which stalls the input
            // DMAs we would otherwise be waiting on.
            for (int k = 0; k < COL_CHUNKS; ++k) {
                DMA_TX(DMA_ACCUM_OUT,
                    gm_accum_out.async(accum_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                       COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
                DMA_TX(DMA_FFT_COL_OUT,
                    gm_fft_col_out.async(fcol_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                         COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
                DMA_T(DMA_FFT_COL_OUT, gm_fft_col_out.wait());
                DMA_T(DMA_ACCUM_OUT,   gm_accum_out.wait());
            }
            VP2("[ch %d] accum_out + F_ch received (%d x %zu B)\n",
                ch, COL_CHUNKS, COL_CHUNK_BYTES);

            // Stash this channel's spectrum for the filter update after the loop.
            // Converted out of the col-FFT layout here so all the filter maths
            // works in one consistent row-major order.
            // SPLIT, 2026-08-20. These were one slot and its 16.59 ms could not
            // be apportioned between the transfer and the conversion — the same
            // mistake as DMA_T timing async and wait together.
            AP_T(AP_FCOL_SYNC, fcol_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE));
            AP_T(AP_BO_STAGE,
                 memcpy(g_stage_c.data(), fcol_bo.map<void *>(), FFT_BYTES));
            AP_T(AP_UNPACK,
                 unpack_spectrum((const int16_t *)g_stage_c.data(),
                                 g_F_all.data() + (size_t)ch * PATCH_ELEMS));

            // F_ch is the filter's only input. If it is wrong, everything
            // downstream is wrong for a reason that has nothing to do with the
            // filter maths — so check it before trusting any later number.
            // Reported in the col-FFT layout it arrives in: idx = v*ROWS + u.
            if (ch == 0) {
                report_cint16("F_ch", (const int16_t *)g_stage_c.data(),
                              PATCH_ROWS, PATCH_COLS, "colFFT");
                dump_buffer("F_ch", frame, g_stage_c.data(), FFT_BYTES);
            }

            DMA_T(DMA_FFT_COL_IN, gm_fft_col_in.wait());
            VP2("[ch %d] fft_col_in sent\n", ch);
            DMA_T(DMA_CMUL_IN, gm_cmul_in.wait());
            VP2("[ch %d] cmul_in sent\n", ch);
        }

        // Stage B2: cancel the residual pre-window mean on the accumulated
        // spectrum. 9 bins × N_CHANNELS complex MACs — 144 operations for the
        // whole frame. Must run before the IFFT consumes accum_bo.
        AP_T(AP_B2, apply_dc_correction(accum_bo.map<int16_t *>(),
                                        filter_bo.map<int16_t *>(),
                                        residual_mean));

        // Push the updated mean_prev values (written into bytes [18:22] of each
        // channel's weight buffer above) so the NEXT frame's conv2d sees them.
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // The accumulated spectrum, after B2, as the IFFT will see it. Same
        // col-FFT layout as F_ch. `rails > 0` here is the saturation the shift
        // budget exists to prevent.
        report_cint16("accum", accum_bo.map<int16_t *>(),
                      PATCH_ROWS, PATCH_COLS, "colFFT");
        dump_buffer("accum", frame, accum_bo.map<void *>(), ACCUM_BYTES);

        // 3. IFFT: APU feeds accumulated spectrum to IFFT row input
        VP2("[ifft] START\n");
        DMA_TX(DMA_IFFT_ROW_IN,
            gm_ifft_row_in.async(accum_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, ACCUM_BYTES, 0));
        // Drain per invocation, and before waiting on the input — see the
        // accum_out loop above for why the input wait cannot come first.
        for (int k = 0; k < ROW_CHUNKS; ++k) {
            DMA_TX(DMA_IFFT_ROW_OUT,
                gm_ifft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                      ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES));
            DMA_T(DMA_IFFT_ROW_OUT, gm_ifft_row_out.wait());
        }
        DMA_T(DMA_IFFT_ROW_IN, gm_ifft_row_in.wait());
        VP2("[ifft] rows done (%d x %zu B)\n", ROW_CHUNKS, ROW_CHUNK_BYTES);

        // APU: transpose IFFT row output in-place
        AP_T(AP_BO_STAGE,
             memcpy(g_stage_a.data(), row_bo.map<void *>(), FFT_BYTES));
        AP_T(AP_TRANSPOSE,
             transpose_to(g_stage_a.data(), g_stage_b.data(),
                          PATCH_ROWS, PATCH_COLS, 4));
        AP_T(AP_BO_STAGE,
             memcpy(row_bo.map<void *>(), g_stage_b.data(), FFT_BYTES));
        VP2("[ifft] transpose done\n");

        DMA_TX(DMA_IFFT_COL_IN,
            gm_ifft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0));
        for (int k = 0; k < COL_CHUNKS; ++k) {
            DMA_TX(DMA_RESPONSE,
                gm_response.async(resp_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                  COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
            DMA_T(DMA_RESPONSE, gm_response.wait());
        }
        DMA_T(DMA_IFFT_COL_IN, gm_ifft_col_in.wait());
        VP2("[ifft] cols done → response received (%d x %zu B)\n",
            COL_CHUNKS, COL_CHUNK_BYTES);

        // Response diagnostics run on EVERY frame, including frame 0. Frame 0's
        // response is not a tracking result (the filter is zero, so it should be
        // ~0), but "is it actually zero" is worth knowing: a non-zero frame-0
        // response would mean the filter buffer is not what the host thinks.
        // One copy of the response out of the BO; every reader below uses it —
        // report_cint16, both PSR scans, the resp00 ratio and the dump.
        AP_T(AP_BO_STAGE,
             memcpy(g_stage_c.data(), resp_bo.map<void *>(), RESP_BYTES));
        report_cint16("response", (const int16_t *)g_stage_c.data(),
                      PATCH_ROWS, PATCH_COLS, "rowmajor");
        dump_buffer("resp", frame, g_stage_c.data(), RESP_BYTES);
        // The peak is located ONCE, here, and the same PsrResult drives the
        // reporting, the position update and the gate. It used to be scanned three
        // times (twice inside report_psr, once in peak_detect_sw) — three
        // independent scans that could in principle disagree about which peak the
        // tracker was acting on.
        //
        // `evaluate` is keyed off the filter state, not off `frame == 0`, matching
        // the injection block and the init branch below. Frame 0's response is
        // identically zero by construction (H is zeroed until filter_init runs at
        // the END of this frame), so it would gate out under ANY threshold — it
        // must never reach the gate at all or the tracker never bootstraps.
        const bool          evaluate = g_filter.initialized;
        mosse::PsrResult    psr_abs{}, psr_sgn{};
        mosse::GateDecision gate{};          // accept=false until proven otherwise
        mosse::ScaleDecision scale_gate_dec{};   // ditto for the size axis
        bool scale_evaluated = false;
        int  scale_idx = 0;                      // the level the detector proposed

        if (evaluate) {
            const int16_t *resp = (const int16_t *)g_stage_c.data();
            AP_T(AP_PSR, {
                psr_abs = mosse::compute_psr(resp, PATCH_ROWS, PATCH_COLS, true);
                psr_sgn = mosse::compute_psr(resp, PATCH_ROWS, PATCH_COLS, false);
            });
            gate    = mosse::psr_gate(psr_abs, mosse::DEFAULT_PSR_MIN);

            // Suppressed on an occluded frame: report_response prints values "at
            // the injected offset", and there is no injected target to be at.
            if (!occluded)
                report_response(resp, PATCH_ROWS, PATCH_COLS);
            else
                printf("  [diag] response profiles suppressed — no target was "
                       "injected this frame\n");

            report_psr(psr_abs, psr_sgn);
            gate_track(frame, gate);
        }

        // 4. Peak detection and position update.
        if (!evaluate) {
            VP1("Frame %d: [INIT] response not evaluated — filter is being "
                "trained from this frame\n", frame);
        } else {
            const int  dr   = psr_abs.dr, dc = psr_abs.dc;
            const long peak = psr_abs.peak;

            // PATCH BINS -> FRAME PIXELS. One bin is roi_h/patch_rows frame
            // pixels, so these are the same number ONLY while the ROI is 1:1 with
            // the patch — which it was for every build before the box existed.
            // Omitting the conversion gives a tracker that localises confidently
            // and drifts by the resample ratio every frame, which `err=0 px`
            // cannot see. Checked by run_box_tests() in test_mosse_filter.cpp.
            const double dr_frame = mosse::patch_dr_to_frame(dr, roi);
            const double dc_frame = mosse::patch_dc_to_frame(dc, roi);
            // And the inverse for the expectation: the impulse is injected in
            // FRAME pixels, so the bin a correct pipeline must report is
            // IMPULSE_DR scaled into patch coordinates.
            // Expected displacement, derived from WHERE THE TARGET WAS DRAWN
            // rather than from a constant. pos_row/pos_col are still the values
            // the ROI was formed around — the position update below has not run
            // yet — so this is the true offset the pipeline had to find.
            //
            // At TRAJECTORY=0 truth.row is pos + IMPULSE_DR by construction, so
            // this reduces exactly to the old constant. At TRAJECTORY=1 it
            // tracks the scripted path, which is the whole point: the expectation
            // no longer follows the estimate around.
            const double inj_dr = truth.row - (double)pos_row;   // FRAME px, pre-update
            const double inj_dc = truth.col - (double)pos_col;
            const int exp_dr = mosse::frame_dr_to_patch(inj_dr, roi);
            const int exp_dc = mosse::frame_dc_to_patch(inj_dc, roi);

            // THE POSITION IS ONLY UPDATED ON AN ACCEPTED FRAME. A gated frame's
            // peak is noise; following it walks the ROI off the target and the
            // target can then never re-enter the search window. Holding is what
            // makes Bolme's "reacquire when the appearance returns" work with no
            // reacquisition code — and it is also why the recovery frame's
            // expected displacement is still exactly (IMPULSE_DR, IMPULSE_DC),
            // so the check below stays meaningful across an occlusion.
            if (gate.accept) {
                box.row += dr_frame;
                box.col += dc_frame;
                pos_row = (int)llround(box.row);
                pos_col = (int)llround(box.col);
            }

            const bool ok = (peak != 0 && dr == exp_dr && dc == exp_dc);
            VP1("Frame %d: displacement (%d,%d) bins = (%.2f,%.2f) frame px %s "
                "pos (%d,%d)  peak=%ld  [%s]\n",
                   frame, dr, dc, dr_frame, dc_frame,
                   gate.accept ? "→" : "HELD, pos stays",
                   pos_row, pos_col, peak,
                   peak == 0   ? "VOID: zero response — result carries no information"
                   : occluded  ? "occluded frame — no target to match, gate decides"
                   : ok        ? "OK: matches injected offset"
                               : "MISMATCH vs injected offset");
            if (peak == 0)
                printf("       (filter is non-zero now, so this points at the shift "
                       "budget or the filter scale — check the Q1.15 report above)\n");
            // The hint is suppressed when the frame was gated: there is no
            // expected displacement for a frame the tracker deliberately did not
            // act on, so printing one would read as a failure when it is a pass.
            // VP1, NOT unconditional. The `ok` criterion derives exp_dr from
            // IMPULSE_DR, so under TRAJECTORY=1 it fires on HEALTHY frames — see
            // CLAUDE.md, "[MISMATCH vs injected offset] is meaningless under
            // TRAJECTORY=1". An anomaly print that cries wolf every frame is not
            // an anomaly print, and at VERBOSITY 0 it would have been most of the
            // console on exactly the long trajectory run this gating exists for.
            // IoU and centre error in track.csv are the valid scores.
            else if (!ok && !occluded && gate.accept)
                VP1("       expected displacement (%d,%d) bins "
                    "(= injected (%d,%d) frame px / %.4f px per bin)\n",
                       exp_dr, exp_dc, (int)llround(inj_dr), (int)llround(inj_dc),
                       mosse::patch_dr_to_frame(1, roi));

            // SCALE ESTIMATION — DSST §5.1, Algorithm 1.
            //
            // Order is load-bearing and the paper states why: "Typically, the
            // target scale difference between two frames is small compared to the
            // difference in the location. We therefore first apply the translation
            // filter... The scale filter is then applied at the new target
            // location estimate." So this runs AFTER the position update above.
            //
            // Gated on gate.accept for the same reason filter_update is: an
            // occluded frame's scale peak is noise, and resizing the box from it
            // would walk the ROI off the target just as surely as moving it would.
            if (scale.enabled() && gate.accept && scale.initialized) {
                const uint8_t *fp = g_frame_host.data();
                mosse::ScaleResult sr{};
                // Split: scale_extract reads frame_bo DIRECTLY (33 crops), so it
                // pays the uncached read; scale_detect is pure heap. 13.86 ms was
                // one number and could not be apportioned.
                AP_T(AP_SCALE_EXTRACT,
                     mosse::scale_extract(scale, fp, FRAME_ROWS, FRAME_COLS,
                                          box.row, box.col, box.h, box.w,
                                          scale_sample.data()));
                AP_T(AP_SCALE_MODEL,
                     sr = mosse::scale_detect(scale, scale_sample.data(),
                                              mosse::DEFAULT_EPS_REL));
                // Gate before resizing — see scale_gate(). A veto holds the box
                // AND, via scale_ok below, skips the model update, exactly as
                // psr_gate holds the position and skips filter_update.
                scale_gate_dec =
                    mosse::scale_gate(sr, mosse::DEFAULT_SCALE_N,
                                      box.h, box.w, box_h0, box_w0,
                                      mosse::DEFAULT_SCALE_CONF_MIN,
                                      mosse::DEFAULT_SCALE_MIN_REL,
                                      mosse::DEFAULT_SCALE_MAX_REL);
                scale_evaluated = true;
                scale_idx       = sr.idx;
                if (scale_gate_dec.accept) { box.h = scale_gate_dec.new_h;
                                             box.w = scale_gate_dec.new_w; }
                VP1("  [scale] level %+d  factor %.4f  peak %.3g  conf %.2f"
                    "  box %.1fx%.1f  %s\n",
                       sr.idx, sr.factor, sr.peak, sr.psr, box.h, box.w,
                       mosse::scale_veto_tag(scale_gate_dec.reason));
                if (!scale_gate_dec.accept)
                    printf("  [scale]   HOLD — %s  (proposed %.1fx%.1f, "
                           "threshold %.2f)\n",
                           mosse::scale_veto_why(scale_gate_dec.reason),
                           scale_gate_dec.new_h, scale_gate_dec.new_w,
                           (double)scale_gate_dec.threshold);
                ++scale_n_eval;
                if (scale_gate_dec.conf < scale_conf_min_seen)
                    scale_conf_min_seen = scale_gate_dec.conf;
                if (scale_gate_dec.conf > scale_conf_max_seen)
                    scale_conf_max_seen = scale_gate_dec.conf;
                if (scale_gate_dec.accept) ++scale_n_accept;
                else { ++scale_n_hold;
                       ++scale_reason_n[(int)scale_gate_dec.reason]; }
            } else if (scale.enabled() && !gate.accept) {
                printf("  [scale] FROZEN — gated frame, size held at %.1fx%.1f\n",
                       box.h, box.w);
            }

            // IoU against the injected box. This is what makes the tracker
            // SCOREABLE: both papers report overlap precision, and until the box
            // existed there was no way to compute it at all. Suppressed on an
            // occluded frame, where there is no ground-truth box to score.
            if (!occluded) {
                const mosse::TargetBox est = box;
                const double iou  = mosse::box_iou(est, truth);
                const double cerr = std::hypot(est.row - truth.row,
                                               est.col - truth.col);
                VP1("       IoU %.4f  centre err %.2f px  vs injected box "
                       "(%.0fx%.0f at (%.1f,%.1f); estimate %.0fx%.0f at "
                       "(%.1f,%.1f))%s\n",
                       iou, cerr, truth.h, truth.w, truth.row, truth.col,
                       est.h, est.w, est.row, est.col,
                       iou >= 0.5 ? "  [OK: above the PASCAL 0.5 criterion]"
                                  : "  [BELOW 0.5 — PASCAL would score this a miss]");
                // Accumulate the curve. PASCAL's 0.5 overlap is the criterion both
                // papers report against, so "lost" means IoU < 0.5 — not err != 0,
                // which is meaningless once the target moves sub-pixel.
                ++trk_eval;
                trk_iou_sum  += iou;
                trk_cerr_sum += cerr;
                if (iou < trk_iou_min)  trk_iou_min  = iou;
                if (cerr > trk_cerr_max) trk_cerr_max = cerr;
                if (iou >= 0.5) ++trk_ok; else ++trk_lost;
            }
        }

        // 5. Filter init / update (PS-side). Bolme eq. 10-12; see mosse_filter.h.
        //
        // Runs AFTER peak detection so the filter used for this frame's response
        // is the one learned from previous frames — updating first would leak the
        // current frame into its own detection and make tracking look better than
        // it is.
        //
        // THREE-WAY, AND THE ORDER OF THE TESTS IS LOAD-BEARING: the init branch
        // is selected before `gate` is ever consulted, so the gate can never block
        // the bootstrap.
        // Scale model update, DSST Algorithm 1 step 9. Trained on frame 0 (so the
        // filter exists before it is ever applied) and on every ACCEPTED frame
        // thereafter — the same freeze rule as the translation filter, so one
        // gated frame cannot teach the scale filter what an occluder looks like.
        //
        // AND on the SCALE gate as well, which is the half that stops the runaway.
        // Holding the box while still training on the rejected frame is not a
        // gate: the model learns the bad estimate anyway and proposes it again
        // next frame, harder. Measured 2026-08-20: one accepted level -12 at
        // conf 1.22 was enough to take the box 64.0 -> 50.5 and then thrash for
        // the rest of the run. `scale_evaluated` distinguishes "the gate ran and
        // said no" from "the gate never ran" (frame 0, an occluded frame, or a
        // PSR-gated one) — only the first should block training, because the
        // others are already covered by the conditions above.
        if (scale.enabled() && (!g_filter.initialized || gate.accept)
                            && !(scale_evaluated && !scale_gate_dec.accept)) {
            const uint8_t *fp = g_frame_host.data();
            AP_T(AP_SCALE_EXTRACT,
                 mosse::scale_extract(scale, fp, FRAME_ROWS, FRAME_COLS,
                                      box.row, box.col, box.h, box.w,
                                      scale_sample.data()));
            AP_T(AP_SCALE_MODEL,
                 mosse::scale_update(scale, scale_sample.data(),
                                     mosse::DEFAULT_SCALE_ETA));
        }

        bool published = false;
        if (!g_filter.initialized) {
            AP_T(AP_FILTER,
                 mosse::filter_init(g_filter, g_F_all.data(), g_target.data(),
                                    N_CHANNELS, PATCH_ROWS, PATCH_COLS));
            VP1("  filter: INITIALISED from frame %d (single patch, no affine "
                   "perturbations)\n", frame);
            AP_T(AP_PUBLISH, publish_filter(filter_bo, filter_scratch));
            published = true;
        } else if (gate.accept) {
            // THE TRAINING TARGET IS RE-CENTRED ON THE MEASURED DISPLACEMENT.
            //
            // g_F_all is the patch cropped at the PRE-update position, so the
            // object in it sits at (dr,dc) — the bin the response just peaked at
            // — not at the origin. Training that patch against a G centred at
            // (0,0) teaches "object at (dr,dc) => peak at (0,0)".
            //
            // Why that compounds instead of averaging out. Let Q_t be the patch
            // cropped exactly ON the object, so the patch we actually hold is
            // P_t = Q_t shifted by +d_t. Correlation is shift equivariant, so
            // resp(P_t) = resp(Q_t) shifted by +d_t. We need an on-target patch
            // to peak at 0, i.e. resp(P_t) to peak at d_t — hence G must be
            // centred at +d_t, the SAME sign as the detected peak. Training at 0
            // instead teaches resp(Q_t) -> peak at -d_t; the next frame's patch
            // is Q_{t+1} shifted by d_{t+1}, so that contribution lands at
            // d_{t+1} - d_t, which under near-constant motion is (0,0). The
            // origin peak therefore grows at the LEARNING RATE, which is why
            // resp00_over_peak tracked 1-(1-eta)^k on hardware and was unmoved
            // by BG_PAN cutting background correlation 6.6x. It was never
            // background lock.
            //
            // By the shift theorem this IS "re-crop at the updated position",
            // exactly, and for free — the alternative is 16 more roi_crop
            // launches per frame. The only difference is that the Hann window
            // stays centred where the crop was taken rather than on the object.
            //
            // Regression test, no hardware: scripts/mosse_loop_sim.py. Both arms,
            // 30 frames — centred reaches resp00/peak 0.76 and loses lock, shifted
            // holds 0.21 flat at 0.00 px error.
            mosse::gaussian_target_spectrum(g_target_shift.data(),
                                            PATCH_ROWS, PATCH_COLS,
                                            sigma_r, sigma_c,
                                            psr_abs.dr, psr_abs.dc);
            AP_T(AP_FILTER,
                 mosse::filter_update(g_filter, g_F_all.data(),
                                      g_target_shift.data(), mosse::DEFAULT_ETA));
            AP_T(AP_PUBLISH, publish_filter(filter_bo, filter_scratch));
            published = true;
        } else {
            // publish_filter is skipped as well as filter_update, and that is a
            // CORRECTNESS requirement, not an optimization: filter_quantize_q15
            // reads g_energy, which the per-channel loop above has already
            // overwritten with THIS frame's per-channel energies. Publishing would
            // therefore re-scale the old A/B by the occluder's norms. Skipping
            // leaves filter_bo holding the H from the last accepted frame, which
            // is what "frozen" has to mean.
            printf("  filter: FROZEN — update and publish both skipped (%s). A/B "
                   "unchanged; the device still holds the H published on the last "
                   "accepted frame.\n", mosse::gate_reason_tag(gate.reason));
        }

        // The quantized filter, row-major, as filter_quantize_q15() produced it —
        // i.e. BEFORE pack_filter() converts it to the col-FFT layout. Dumping
        // this side of the conversion is deliberate: comparing it against a
        // NumPy H built from the dumped F_ch isolates the filter maths from the
        // layout conversion, which are the two remaining untested steps.
        //
        // Guarded on `published`: on a frozen frame filter_scratch still holds the
        // PREVIOUS frame's bytes, and printing those as if they were current is
        // exactly the kind of plausible-looking output that costs a day.
        if (published) {
            report_cint16("H(q15)", filter_scratch.data(),
                          PATCH_ROWS, PATCH_COLS, "rowmajor");
            dump_buffer("H_q15", frame, filter_scratch.data(),
                        (size_t)N_CHANNELS * PATCH_ELEMS * 2 * sizeof(int16_t));
        } else {
            printf("  H(q15): unchanged (filter frozen this frame) — no dump "
                   "written, so the previous frame's H_q15 dump is still the "
                   "current device contents\n");
        }

        // Per-frame CSV row. Emitted here, at the very end of the frame body, so
        // `box` carries BOTH the position update and the scale update — the same
        // state the IoU line above printed — and `published` is known.
        //
        // Recomputed rather than captured from the reporting block above: those
        // locals live inside `if (evaluate)`, and widening their scope to reach
        // here would put the tracker's state in mutable frame-scope variables
        // purely for logging. box/truth are unchanged between the two points, so
        // the CSV and the printed IoU line agree by construction.
        {
            const double iou  = occluded ? 0.0 : mosse::box_iou(box, truth);
            const double cerr = occluded ? 0.0
                                         : std::hypot(box.row - truth.row,
                                                      box.col - truth.col);
            // |resp(0,0)| / |peak| — see the csv_row() comment. Frame 0's
            // response is identically zero (H is zeroed until filter_init runs
            // at the end of this frame), so it is only meaningful once the
            // filter exists; 0 is reported otherwise rather than a 0/0.
            double resp00_over_peak = 0.0;
            if (evaluate && psr_abs.peak != 0) {
                const int16_t *r = (const int16_t *)g_stage_c.data();
                // cint16: bin (0,0) is the first sample, real part at index 0.
                // The peak scan is on |real|, so compare like with like.
                resp00_over_peak = std::fabs((double)r[0])
                                 / std::fabs((double)psr_abs.peak);
            }
            csv_row(frame, occluded, evaluate, gate, psr_abs, resp00_over_peak,
                    box, truth, iou, cerr, published,
                    scale_evaluated, scale_idx, scale_gate_dec);

            // VERBOSITY 0: the ONE line per frame that keeps the frame period
            // measurable. ~45 B, ~4 ms at 115200, 4% of the ~87 ms GMIO floor.
            // Everything on it is already in track.csv — the point is not the
            // data, it is having a timestampable per-frame marker for
            // `picocom | ts`, which is how the frame time was measured in the
            // first place. Printing nothing at all would have deleted that.
            if (VERBOSITY < 1)
                printf("f%d d%d,%d psr%.0f iou%.2f r00 %.2f\n",
                       frame, psr_abs.dr, psr_abs.dc, gate.psr, iou,
                       resp00_over_peak);
            // Printed at EVERY verbosity while the sweep is armed: if a deep
            // queue deadlocks, this is the record of which depths already worked.
            if (FFT_DRAIN_DEPTH == 0)
                printf("  [drain] frame %d depth %d: gmio_fft_row_out %.2f ms "
                       "(%lu tx), weights %.2f ms\n",
                       frame, drain_depth_for_frame(frame),
                       g_dma[DMA_FFT_ROW_OUT].us / 1000.0,
                       g_dma[DMA_FFT_ROW_OUT].calls,
                       g_dma[DMA_WEIGHTS].us / 1000.0);
        }
        fflush(stdout);

        // ~3.2 KB/frame of tables — 32% of the console, and at 115200 that was
        // 277 ms of a 880 ms frame. First and last frame only; the DMA counters
        // still accumulate every frame into the CUMULATIVE report at exit, so
        // nothing is lost but the repetition. See trace_frame().
        // ap_report_frame() stops the frame-body clock, so it must come LAST and
        // must run on every frame — the run total would otherwise be a two-frame
        // total, the same trap dma_accumulate_frame() exists to avoid.
        double _dma_us = 0.0, _rc_us = 0.0;
        for (int i = 0; i < DMA_N; ++i) _dma_us += g_dma[i].us;
        for (int i = 0; i < RC_N;  ++i) _rc_us  += g_rc_us[i];
        if (trace_frame(frame)) {
            rc_report_frame(frame);
            rc_report_timeline(frame);
            rc_report_calls(frame);
            dma_report_frame(frame);
            ap_report_frame(frame, _dma_us, _rc_us);
        } else {
            dma_accumulate_frame();
            ap_accumulate_frame();
        }
    }

    csv_close();

    // Tracking result across the run — the thing a long hardware run exists to
    // produce, and which no hw_emu run could ever afford to compute (2-3 frames
    // is not a curve). Mean overlap precision at the PASCAL 0.5 threshold is the
    // metric both papers report; until the target box existed it could not be
    // computed at all.
    if (trk_eval > 0) {
        printf("\n[track] SUMMARY over %d evaluated frame(s):\n", trk_eval);
        printf("  overlap precision @0.5 : %5.1f%%  (%d of %d; %d lost)\n",
               100.0 * trk_ok / trk_eval, trk_ok, trk_eval, trk_lost);
        printf("  mean IoU               : %.4f   (worst %.4f)\n",
               trk_iou_sum / trk_eval, trk_iou_min);
        printf("  mean centre error      : %.2f px (worst %.2f px)\n",
               trk_cerr_sum / trk_eval, trk_cerr_max);
        printf("  final box              : %.0fx%.0f at (%.1f,%.1f)  "
               "truth %.0fx%.0f at (%.1f,%.1f)\n",
               box.h, box.w, box.row, box.col,
               truth.h, truth.w, truth.row, truth.col);
    }

    // Scale-gate outcome across the run, reported per reason for the same reason
    // the PSR gate's is: "held 14 frames" is unactionable, "held 14 frames, all
    // AT_SEARCH_RAIL" says the search range is the problem, and "all LOW_CONF"
    // says the scale response never became peaked. The conf range is printed
    // because the threshold has to be re-derived whenever the geometry or the
    // feature scale moves — it is not a universal constant.
    if (scale_n_eval > 0) {
        printf("\n[scale] SUMMARY over %d evaluated frame(s): %d accepted, "
               "%d held  (threshold %.2f)\n",
               scale_n_eval, scale_n_accept, scale_n_hold,
               (double)mosse::DEFAULT_SCALE_CONF_MIN);
        printf("  conf range %.2f .. %.2f\n",
               scale_conf_min_seen, scale_conf_max_seen);
        if (scale_n_hold) {
            printf("  holds: ");
            for (int i = 0; i < 8; ++i)
                if (scale_reason_n[i])
                    printf("%s x%d  ", mosse::scale_veto_tag((mosse::ScaleVeto)i),
                           scale_reason_n[i]);
            printf("\n");
        }
        // The two failure modes that a hold COUNT alone cannot distinguish.
        if (scale_n_accept == 0)
            printf("  -> NOTHING was ever accepted. The size estimate is frozen at "
                   "its initial value, so a growing or shrinking target is not "
                   "being tracked at all. Either the threshold is too high for "
                   "this geometry or the scale filter is not working.\n");
        else if (scale_n_hold > scale_n_accept)
            printf("  -> more holds than accepts. The gate is doing its job only "
                   "if the accepted frames track the true size; check the final "
                   "box against truth above before raising the threshold.\n");
    }

    // Gate outcome across the run. Printed here for the same reason the DMA
    // summary below is: gr.end(0) never returns, so anything after it is lost.
    gate_report_run(ITER_CNT);

    // Cumulative GMIO cost across all frames. Printed before teardown because
    // gr.end(0) never returns (see the "host does not exit" note in CLAUDE.md) —
    // anything printed after it is lost.
    {
        double        tot_us = 0.0;
        unsigned long tot_n  = 0;
        printf("\n[dma] CUMULATIVE over %d frame(s):\n", ITER_CNT);
        for (int i = 0; i < DMA_N; ++i) {
            if (!g_dma_total[i].calls) continue;
            printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx\n",
                   g_dma_total[i].name, g_dma_total[i].calls,
                   g_dma_total[i].us / 1000.0,
                   g_dma_total[i].us / g_dma_total[i].calls);
            tot_us += g_dma_total[i].us;
            tot_n  += g_dma_total[i].calls;
        }
        printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx\n", "TOTAL", tot_n,
               tot_us / 1000.0, tot_n ? tot_us / tot_n : 0.0);
        printf("  per frame: %.0f tx, %.3f ms  (N_CHANNELS=%d, %dx%d)\n",
               (double)tot_n / ITER_CNT, tot_us / 1000.0 / ITER_CNT,
               N_CHANNELS, PATCH_ROWS, PATCH_COLS);
        fflush(stdout);

        // APU CUMULATIVE. This is the number to read, not the two per-frame
        // tables: frame 0 does filter_init rather than filter_update and never
        // runs the scale detector, so it is the least representative frame in
        // the run. Averaged over ITER_CNT, against the measured mean frame body.
        {
            const double mf = g_ap_run_us / ITER_CNT;
            double apu = 0.0;
            for (int i = 0; i < AP_N; ++i) apu += g_ap_tot_us[i];
            double rc = 0.0;
            for (int i = 0; i < RC_N; ++i) rc += g_rc_us_total[i];
            printf("\n[apu] CUMULATIVE over %d frame(s), mean frame body %.2f ms:\n",
                   ITER_CNT, mf / 1000.0);
            printf("  %-20s %9s %10s %9s %7s\n",
                   "stage", "calls/fr", "ms/frame", "us/call", "share");
            for (int i = 0; i < AP_N; ++i) {
                if (!g_ap_tot_n[i]) continue;
                printf("  %-20s %9.1f %10.3f %9.1f %6.1f%%\n",
                       g_ap_name[i], (double)g_ap_tot_n[i] / ITER_CNT,
                       g_ap_tot_us[i] / 1000.0 / ITER_CNT,
                       g_ap_tot_us[i] / (double)g_ap_tot_n[i],
                       100.0 * g_ap_tot_us[i] / g_ap_run_us);
            }
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- APU subtotal", "",
                   apu / 1000.0 / ITER_CNT, "", 100.0 * apu / g_ap_run_us);
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- GMIO (DMA_T)", "",
                   tot_us / 1000.0 / ITER_CNT, "", 100.0 * tot_us / g_ap_run_us);
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- roi_crop launch", "",
                   rc / 1000.0 / ITER_CNT, "", 100.0 * rc / g_ap_run_us);
            const double resid = g_ap_run_us - apu - tot_us - rc;
            printf("  %-20s %9s %10.3f %9s %6.1f%%   <-- console, dumps, printf, "
                   "uninstrumented\n", "== UNATTRIBUTED", "",
                   resid / 1000.0 / ITER_CNT, "", 100.0 * resid / g_ap_run_us);
            if (resid > 0.25 * g_ap_run_us)
                printf("  NOTE: unattributed is over 25%% of the frame. This "
                       "breakdown is NOT the frame — find the missing cost "
                       "before acting on any line above it.\n");
            fflush(stdout);
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    gr.end(0);  // block until graph completes
    return EXIT_SUCCESS;
}
