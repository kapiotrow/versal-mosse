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

// The conv2d weight-buffer layout, shared verbatim with the AIE kernel. The
// host WRITES into that buffer every frame (mean_prev), so a layout it merely
// remembers is a layout it can silently corrupt: at CONV_IN_CH=3 the grayscale
// mean_prev offset (18) lands inside the B plane's taps. No <adf.h> in this
// header, by design.
#include "conv_weight_layout.h"

// Luma scene -> the interleaved buffer the device reads. Its own translation
// unit, with no XRT header, so `make test_scene` compiles and tests it natively
// in seconds — the same reason mosse_filter.{h,cpp} is separate. Both of its
// failure modes (wrong interleave, missed touch) are silent on hardware.
#include <map>

#include "scene_colour.h"

// WHERE FRAMES COME FROM. The Makefile passes -DFRAME_SOURCE_VOT on BOTH arms
// (FRAME_SOURCE=synth emits 0, =vot emits 1) rather than only on one -- a bare
// #ifndef default in the host is not a safety net, it is what makes a
// build/flag mismatch silent, which is precisely how SCENE_VERIFY=1 once built
// the instrument DISABLED. The default below exists only so a stray compile of
// this file outside the Makefile still means "synthetic scene".
#ifndef FRAME_SOURCE_VOT
#  define FRAME_SOURCE_VOT 0
#endif
#if FRAME_SOURCE_VOT
// Manifest, blob, run order and trajectory. No XRT header in it, so
// `make test_vot_source` runs its 19 mutants natively in seconds.
#  include "vot_source.h"
// RGB + VOT: WIRED 2026-08-26. It used to be a deliberate `#error`, because
// scale_extract() reads an intensity template out of scene_luma() while a VOT
// frame arrives pixel-interleaved -- building the pair without the sidecar
// would have given a colour datapath with the scale filter reading the R plane
// as luma, i.e. a tracker that works slightly WORSE rather than one that fails.
//
// What makes it safe now is not that the sidecar is loaded but that its ABSENCE
// is fatal: load_vot_sequence() refuses a manifest with no `luma_blob` at
// CONV_IN_CH=3 rather than falling back to the interleaved plane. The failure
// mode this guard existed to prevent is plausible-but-wrong, so the replacement
// has to be loud too.
#endif

// -----------------------------------------------------------------------
// Build-time constants (set via Makefile -D flags)
// -----------------------------------------------------------------------
#ifndef PATCH_ROWS
#  define PATCH_ROWS  128
#endif
#ifndef PATCH_COLS
#  define PATCH_COLS  128
#endif

// conv2d's kernel size and stride, and the ROI CROP they imply.
//
// PATCH_ROWS/PATCH_COLS are the FEATURE MAP -- the FFT point sizes, the
// accumulator, the response, the filter and every bin<->pixel conversion below
// are sized on them. The crop roi_crop must produce to feed that map is
// CONV_STRIDE times larger on each axis, and it is the only geometry the host
// programs into roi_crop's AXI-Lite registers.
//
// The Makefile passes all four from ONE place (see the CONV_KSIZE block there).
// These #ifndef defaults exist for the native test builds; a mismatch with the
// graph is exactly the silent failure CLAUDE.md's shared-constant rule is about,
// which is why the weight file carries both as tag bytes and this file checks
// them at startup.
#ifndef CONV_KSIZE
#  define CONV_KSIZE  3
#endif
#ifndef CONV_STRIDE
#  define CONV_STRIDE 1
#endif
#ifndef CROP_ROWS
#  define CROP_ROWS   (PATCH_ROWS * CONV_STRIDE)
#endif
#ifndef CROP_COLS
#  define CROP_COLS   (PATCH_COLS * CONV_STRIDE)
#endif
static_assert(CROP_ROWS == PATCH_ROWS * CONV_STRIDE &&
              CROP_COLS == PATCH_COLS * CONV_STRIDE,
              "CROP_ROWS/COLS must be PATCH_ROWS/COLS x CONV_STRIDE -- the "
              "Makefile derives them; do not set them by hand");
static_assert(CROP_ROWS <= 128 && CROP_COLS <= 128,
              "crop exceeds roi_crop's BRAM scratch (ROI_MAX_PATCH_ROWS/COLS)");
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
// Pixels in a frame, and BYTES in the device-side frame buffer. They differ at
// CONV_IN_CH=3, where frame_bo is PIXEL-INTERLEAVED R G B — the layout
// roi_crop.h documents and roi_crop reads. 2 MB grayscale, 6 MB RGB.
//
// frame_cols is still passed to roi_crop in PIXELS: the kernel applies the
// * ROI_IN_CH itself. Do not pre-multiply it, or the ROI walks off the row.
constexpr size_t FRAME_PIXELS      = (size_t)FRAME_ROWS * FRAME_COLS;
constexpr size_t FRAME_BYTES       = FRAME_PIXELS * CONV_IN_CH;

// ---- RUNTIME frame geometry --------------------------------------------
// FRAME_ROWS/FRAME_COLS above are the MAXIMUM, i.e. what frame_bo, g_frame_host
// and the background are ALLOCATED at. What the pipeline RUNS at is these, and
// at FRAME_SOURCE=vot they come from the sequence's manifest and change per
// sequence. Everything downstream already took rows/cols as arguments --
// scene_*, scale_extract, roi_crop's AXI-Lite registers -- so this is a
// substitution, not a rewrite, and at FRAME_SOURCE=synth they hold the macro
// values and every existing result stands unchanged.
//
// The stride is g_frame_cols, so a smaller sequence occupies a dense PREFIX of
// the buffer rather than a sub-rectangle of a 1920-wide one. g_frame_host and
// frame_bo agree on that because the only writer of frame_bo is a bulk memcpy
// of g_frame_bytes out of g_frame_host.
static int    g_frame_rows  = FRAME_ROWS;
static int    g_frame_cols  = FRAME_COLS;
static size_t g_frame_bytes = FRAME_BYTES;

static inline void set_frame_geometry(int rows, int cols)
{
    g_frame_rows  = rows;
    g_frame_cols  = cols;
    g_frame_bytes = (size_t)rows * cols * CONV_IN_CH;
}

// Frames this run executes. ITER_CNT at FRAME_SOURCE=synth; the job's length at
// FRAME_SOURCE=vot, where the dataset decides and ITER_CNT is IGNORED (the run
// covers [anchor .. end] or [anchor .. 0] and nothing else -- a run truncated to
// ITER_CNT would emit a short trajectory, which the toolkit reads without
// complaint and scores as a tracker that stopped early).
static int g_run_frames = ITER_CNT;
// Frames executed across ALL runs in this process. The [dma] and [apu]
// cumulative reports are process-wide, so they must divide by this and not by
// the last run's length -- which at multi-start would be a per-frame average
// computed over the wrong denominator, i.e. a plausible wrong number.
static int g_frames_total = 0;

// THE RUN-STATE DIGEST, and why the trajectory alone is not enough.
//
// The first determinism run (RESET_MUTANT=1, runs/run_0825_1443.log) had the
// mean_prev re-seed DELIBERATELY skipped and still produced a BYTE-IDENTICAL
// trajectory. The leak was real and visible in the diagnostics from frame 1 —
// accum 1963 vs 1961, response 1556 vs 1553, B2 removing 667 vs 671 — but a
// trajectory is quantised: the box comes from an integer peak bin, so a 0.1%
// difference in the response never reaches it. "Nearly identical tracking" is
// precisely the outcome CLAUDE.md records as making a bit-identical criterion
// useless, and this test had walked into it.
//
// So the digest hashes what the pipeline actually PRODUCED, before any argmax:
// the full response buffer, plus the tracker's own continuous state. Two runs
// that agree here agree bit-for-bit on the datapath, not merely on their
// conclusions.
//
// @thesis subsec:weryfikacja | M-04,R-09 | The run-state digest. A byte-identical trajectory is
//   NOT a bit-identical run -- a box comes from an integer argmax, so a 0.1% response
//   difference never reaches it.
// FNV-1a, because it needs to be reproducible across builds and is not
// security-relevant. Its cost is measured in its own AP slot.
static uint64_t g_det_hash = 1469598103934665603ULL;

static inline void det_hash_bytes(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = g_det_hash;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    g_det_hash = h;
}

// Available on BOTH arms deliberately. This project's acceptance criterion for
// every optimisation has been "tracking comes back bit-identical", checked by
// eye across two logs; the digest makes that one number. It is printed at the
// end of every run, so any two runs anywhere can be compared without a diff.

// ---- COASTING THROUGH A HOLD ------------------------------------------
// On a gate veto the host holds position. That is right for occlusion and it
// assumes the target STAYS PUT while the filter is frozen — an assumption
// stb2022 violates on most of its sequences. Measured from groundtruth alone
// (scripts/vot_hold_budget.py): the HOLD BUDGET, i.e. frames before the target
// leaves the frozen box*padding window and recovery becomes impossible for ANY
// tracker, has a median of 6 frames, is <= 4 on 30 of 62 sequences, and is 0 on
// four of them. car1's budget is 4 and its longest hold on hardware was 53.
//
// So during a hold the window COASTS at the last measured velocity instead of
// freezing, with the velocity DECAYING each held frame. The decay is what makes
// this safe rather than a second way to lose the target: total drift over a hold
// run is bounded by v/(1-decay) = 2v at the default, so a hold can never walk
// the window further than two frames' worth of motion from where it froze. Pure
// constant velocity (decay 1.0) is NOT the right answer and was measured to be
// worse — on a near-stationary target the "velocity" is mostly detection noise,
// and it took `nature` from 83 frames of budget to 34 and `girl` from 39 to 21.
//
// Swept offline over all 62 sequences, mean over sequences:
//
//   policy              P(survive 1 held frame)   P(survive 3)   median budget
//   freeze (was)                  90.3%              69.9%             6
//   coast decay 0.0               94.8%              74.4%             7
//   coast decay 0.5  <-- ship     94.9%              76.2%             8
//   coast decay 1.0               94.9%              74.6%             6
//
// At 0.5: 40 of 62 sequences improve, 15 unchanged, 7 marginally worse (the
// worst being soccer1 12 -> 9), and the mean per-sequence escape rate — frames
// where even a ONE-frame hold is already fatal — halves, 9.7% -> 5.1%.
// ball2 goes 55.3% -> 13.5%.
//
// DEFAULT OFF, AFTER BEING ON FOR PART OF 2026-08-25. THE TWO METRICS DISAGREE.
// It shipped OFF first, because the budget metric measures TIME TO FIRST ESCAPE
// rather than time to unrecoverable, and at car1's three hold onsets it gave
// freeze 10/0/0 against coast 1/1/1 — i.e. it predicted no rescue. Hardware over
// 8 sequences and 54 runs said otherwise on mean IoU
// (docs/thesis/evidence/metric_ar_vs_ioum_ab.md): 0.2709 -> 0.3005 frame-weighted, nothing worse
// than -0.0042, car1 job 0 tracking all the way out. Then the SAME 54
// trajectory pairs were scored by the toolkit (docs/thesis/evidence/metric_ar_vs_iou.md) and the
// ordering reversed: accuracy 0.638 -> 0.616, robustness 0.309 -> 0.288, EAO
// 0.208 -> 0.194. AR is the metric of record, so the default follows it.
//
// The disagreement is mechanical, not statistical. vot fails a run on 10
// CONSECUTIVE frames at overlap <= 0.1; on a direction change the coast carries
// the box the old way while the target reverses, so car1 anchor 741 drops out
// for 13 frames coasting against 7 freezing — over the grace instead of under
// it. That run then REACQUIRES and tracks ~470 more frames at overlap 0.82,
// every one of which the rule discards. Failure counts barely move (48 of 54
// runs vs 49); only their timing does, and a mean cannot see timing.
//
// So the coast wins on MANY SHORT holds and loses on ONE LONG hold that crosses
// a turn. The untested middle is a cap on consecutive coasted frames, k from
// the per-sequence budget above (median 6, car1 4). Closed loop: it needs a
// board run, not more analysis of the trajectories already on disk.
//
// THE MODEL WAS OPEN LOOP AND THE TRACKER IS CLOSED LOOP. It treated the
// observed 29-frame gated run as a fixed input, when in fact coasting keeps the
// window on the target so frames that would have been GATED are ACCEPTED — and
// every accept resets the velocity and restarts the coast. The 29-frame hold was
// a consequence of freezing, not a property of the scene.
//
// What the coast CANNOT do, also measured: rescue a tracker that never acquires.
// ball3 gates on 69% of frames and coasted ZERO times, because it has no
// accept->hold transitions at all — every hold run starts before any frame has
// been accepted, so there is no measured velocity and coast_step() correctly
// refuses. Holding a lot is not the same as having something to coast on.
#ifndef HOLD_COAST
#  define HOLD_COAST 0
#endif
#ifndef COAST_DECAY
#  define COAST_DECAY 0.5
#endif

#if FRAME_SOURCE_VOT
// -----------------------------------------------------------------------
// @thesis subsec:zrodlaObrazu | R-08,R-09 | The VOT run: one sequence, one job, the
//   anchor-based run order, and the reset between jobs that the determinism test exercises.
// THE VOT RUN — one sequence, one job (Phase 2). Everything here happens
// OUTSIDE the frame loop: a manifest parse and one sequential blob read per
// sequence, and one trajectory write at the end. Nothing in this block runs
// per frame except vot_frame(), which is a memcpy.
// -----------------------------------------------------------------------
#ifndef VOT_DATA_DIR
#  define VOT_DATA_DIR    "/mnt/vot"
#endif
#ifndef VOT_RESULTS_DIR
#  define VOT_RESULTS_DIR "/mnt/vot-results"
#endif
#ifndef VOT_SEQUENCE
#  define VOT_SEQUENCE    "car1"
#endif
#ifndef VOT_JOB
#  define VOT_JOB         0
#endif
// @thesis subsec:weryfikacja | M-04,R-09 | RESET_MUTANT: the determinism test's ability to FAIL
//   is demonstrated, not assumed. It passed with the mutant active until the digest existed.
// DELIBERATELY BREAK ONE PIECE OF run_reset(), to prove the determinism test can
// FAIL. A reset test that has never been shown to fail is worth nothing on a
// path with no prior coverage — the same rule every RGB suite was built to.
// A flag rather than a code edit so the negative control is repeatable, is in
// the log, and cannot be left commented out by accident.
//   0 none        1 skip mean_prev   2 skip filter_bo zeroing
//   3 skip g_filter   4 skip coast    5 skip the scale reconfigure
#ifndef RESET_MUTANT
#  define RESET_MUTANT 0
#endif

struct VotRun {
    std::string   data_dir    = VOT_DATA_DIR;
    std::string   results_dir = VOT_RESULTS_DIR;
    std::string   sequence    = VOT_SEQUENCE;
    int           job_index   = VOT_JOB;
    int           max_frames  = 0;      // >0 = bring-up truncation, see below
    std::vector<int> job_list;          // runs to execute, in order
    std::string   job_spec;             // --vot-jobs: "all" or "0,3,0"; empty = --vot-job
    vot::Manifest manifest;
    vot::Blob     blob;
#if CONV_IN_CH == 3
    // The single-plane sidecar that feeds scale_extract. Only allocated on the
    // colour arm; at CONV_IN_CH=1 g_frame_host IS the luma and there is nothing
    // to carry.
    vot::Blob     luma;
#endif
    // THE STREAMING ARM. Both readers are constructed either way and exactly one
    // is armed per sequence -- `streamed` says which, and every per-frame site
    // branches on it. Two readers rather than one polymorphic one because the
    // resident path must stay EXACTLY what it is: 57 of 62 RGB sequences already
    // ran on it, and their results are only comparable to the remaining five if
    // nothing about their frame delivery moved.
    bool          streamed = false;
    vot::StreamBlob stream;
#if CONV_IN_CH == 3
    vot::StreamBlob stream_luma;
#endif
    // auto = decide from the blob size against VOT_RESIDENT_MAX_MB;
    // always/never force it. `always` is what makes the mode-equivalence test
    // runnable: a sequence that fits in heap, run both ways, must produce
    // identical run-state digests, because streaming changes no arithmetic.
    std::string   stream_mode = "auto";
    vot::Job      job;
    std::vector<int> order;             // dataset frame index, in RUN order
    vot::Trajectory  traj;
};
static VotRun g_vot;

// argv[1] is the xclbin; everything after it is ours. UNRECOGNISED ARGUMENTS
// ARE FATAL, not ignored: a typo in --vot-seq that silently falls back to the
// compiled-in default produces a complete run attributed to the wrong sequence,
// and the console would not say so.
static bool vot_parse_args(int argc, char **argv)
{
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool needs_value = (a == "--vot-data" || a == "--vot-results" ||
                                  a == "--vot-seq" || a == "--vot-job" ||
                                  a == "--vot-jobs" || a == "--vot-max-frames" ||
                                  a == "--vot-stream");
        if (needs_value && i + 1 >= argc) {
            fprintf(stderr, "%s needs a value\n", a.c_str());
            return false;
        }
        if      (a == "--vot-data")    g_vot.data_dir    = argv[++i];
        else if (a == "--vot-results") g_vot.results_dir = argv[++i];
        else if (a == "--vot-seq")     g_vot.sequence    = argv[++i];
        else if (a == "--vot-job")     g_vot.job_index   = atoi(argv[++i]);
        else if (a == "--vot-jobs")    g_vot.job_spec    = argv[++i];
        else if (a == "--vot-max-frames") g_vot.max_frames = atoi(argv[++i]);
        else if (a == "--vot-stream")  g_vot.stream_mode  = argv[++i];
        else {
            fprintf(stderr,
                    "unrecognised argument '%s'\n"
                    "usage: %s <xclbin> [--vot-data DIR] [--vot-results DIR]\n"
                    "                   [--vot-seq NAME] [--vot-job N]\n"
                    "                   [--vot-jobs all|N,M,...] [--vot-max-frames N]\n"
                    "                   [--vot-stream auto|always|never]\n"
                    "  --vot-jobs runs several anchors in ONE process. Naming the\n"
                    "  same job twice is the determinism test: its two trajectories\n"
                    "  must come back byte-identical.\n",
                    a.c_str(), argv[0]);
            return false;
        }
    }
    // Checked here rather than at the first use, for the same reason an
    // unrecognised ARGUMENT is fatal above: an unrecognised VALUE that fell back
    // to "auto" would run the whole sequence in the other mode and say nothing,
    // and the one thing --vot-stream exists for is comparing the two modes.
    if (g_vot.stream_mode != "auto" && g_vot.stream_mode != "always" &&
        g_vot.stream_mode != "never") {
        fprintf(stderr, "--vot-stream must be auto|always|never, got '%s'\n",
                g_vot.stream_mode.c_str());
        return false;
    }
    return true;
}

// Manifest + blob into heap, geometry and run order set. Returns false with a
// message rather than aborting: this is the staging slot, and Phase 5 runs it 62
// times in one process.
static bool vot_stage(void)
{
    const std::string mpath = g_vot.data_dir + "/" + g_vot.sequence + ".json";
    std::string err;
    if (!vot::manifest_load(mpath, g_vot.manifest, err)) {
        fprintf(stderr, "[vot] %s\n", err.c_str());
        return false;
    }
    const vot::Manifest &m = g_vot.manifest;

    // The channel count is a BUILD property (CONV_IN_CH picks roi_crop's
    // datapath and the conv weight layout), so a blob converted for the other
    // arm cannot be fed to this ELF. Caught here, where it is one line, rather
    // than as sixteen plausible feature channels.
    if (m.channels != CONV_IN_CH) {
        fprintf(stderr, "[vot] %s has %d channel(s), this build is CONV_IN_CH=%d\n",
                m.sequence.c_str(), m.channels, CONV_IN_CH);
        return false;
    }
    if (m.rows > FRAME_ROWS || m.cols > FRAME_COLS) {
        fprintf(stderr, "[vot] %s is %dx%d, frame_bo is allocated at %dx%d\n",
                m.sequence.c_str(), m.rows, m.cols, FRAME_ROWS, FRAME_COLS);
        return false;
    }
    if (g_vot.job_index < 0 || g_vot.job_index >= (int)m.jobs.size()) {
        fprintf(stderr, "[vot] job %d out of range, %s has %zu\n",
                g_vot.job_index, m.sequence.c_str(), m.jobs.size());
        return false;
    }
    g_vot.job = m.jobs[g_vot.job_index];

    // 41 frames of stb2022 (0.21%) carry an EMPTY groundtruth box, and Phase 1
    // checked that no anchor lands on one -- 0 of 419. Asserted anyway because
    // it is one comparison and the alternative is a filter initialised on a
    // degenerate box, which tracks confidently and means nothing.
    if (g_vot.job.init_box.empty()) {
        fprintf(stderr, "[vot] %s job %d has an empty init box\n",
                m.sequence.c_str(), g_vot.job_index);
        return false;
    }

    // ----------------------------------------------------------------------
    // RESIDENT OR STREAMED, decided here and announced.
    //
    // The board maps 2 GB of the part's 12 and 512 MB of that is CMA, so usable
    // heap is ~0.9-1.2 GB. flamingo1 (3631 MB), zebrafish1 (2373), nature
    // (1482), frisbee (1471) and girl (1318) do not fit at CONV_IN_CH=3, and on
    // 2026-08-26 all five died on std::bad_alloc or the OOM killer AFTER the
    // manifest had been read and the sweep had reported a start.
    //
    // PRINTED, ALWAYS, ON BOTH ARMS. A run whose frame delivery mechanism is not
    // in its own log is a run whose frame time cannot be compared to any other,
    // and the streamed arm's frame time is NOT comparable to the resident one on
    // the two 1080p sequences (7.9 MB/frame against 117 MB/s of NFS is 67 ms,
    // against ~28 ms of compute).
    const size_t luma_fb   = (CONV_IN_CH == 3) ? vot::luma_frame_bytes(m) : 0;
    const size_t resident_bytes =
        (m.frame_bytes + luma_fb) * (size_t)m.frames;
    const double resident_mb = resident_bytes / 1048576.0;
    if (g_vot.stream_mode == "always")      g_vot.streamed = true;
    else if (g_vot.stream_mode == "never")  g_vot.streamed = false;
    else g_vot.streamed = (resident_mb > (double)VOT_RESIDENT_MAX_MB);
    printf("[vot] %s: %.0f MB of frames -> %s (mode %s, threshold %d MB)\n",
           m.sequence.c_str(), resident_mb,
           g_vot.streamed ? "STREAMED through a prefetched ring" : "resident in heap",
           g_vot.stream_mode.c_str(), VOT_RESIDENT_MAX_MB);

    const std::string bpath = g_vot.data_dir + "/" + m.blob;
    if (g_vot.streamed) {
        if (!g_vot.stream.open_blob(bpath, m, VOT_STREAM_RING, err)) {
            fprintf(stderr, "[vot] %s\n", err.c_str());
            return false;
        }
        printf("[vot] stream ring %d frames, %.1f MB\n",
               g_vot.stream.ring(), g_vot.stream.ring_bytes() / 1048576.0);
    } else if (!g_vot.blob.load(bpath, m, err)) {
        fprintf(stderr, "[vot] %s\n", err.c_str());
        return false;
    }
#if CONV_IN_CH == 3
    // REFUSED, NOT DERIVED. Computing luma here from the interleaved plane
    // would work and would be wrong in a way nothing downstream can see: it
    // puts a rows*cols BT.601 pass in the frame loop AND it would silently
    // accept a blob converted without the sidecar, which is the case this
    // check exists for. vot_prepare.py emits it at --channels 3; a manifest
    // without it was converted for the grayscale arm.
    if (m.luma_blob.empty()) {
        fprintf(stderr, "[vot] %s has no luma_blob; this build is CONV_IN_CH=3 "
                "and scale_extract needs the plane. Re-convert with "
                "vot_prepare.py --channels 3\n", m.sequence.c_str());
        return false;
    }
    {
        const std::string lpath = g_vot.data_dir + "/" + m.luma_blob;
        // The sidecar follows the blob's mode. Streaming one and staging the
        // other would defeat the point on exactly the sequences that need it --
        // the sidecar is a third of the bytes, and 494 MB of nature's luma is
        // most of the heap the streamed blob just freed.
        if (g_vot.streamed) {
            if (!g_vot.stream_luma.open_luma(lpath, m, VOT_STREAM_RING, err)) {
                fprintf(stderr, "[vot] luma sidecar: %s\n", err.c_str());
                return false;
            }
            printf("[vot] luma sidecar %s STREAMED, ring %.1f MB\n",
                   m.luma_blob.c_str(), g_vot.stream_luma.ring_bytes() / 1048576.0);
        } else {
            if (!g_vot.luma.load_luma(lpath, m, err)) {
                fprintf(stderr, "[vot] luma sidecar: %s\n", err.c_str());
                return false;
            }
            printf("[vot] luma sidecar %s staged, %.1f MB in %.2f s\n",
                   m.luma_blob.c_str(), g_vot.luma.bytes() / 1048576.0,
                   g_vot.luma.load_seconds());
        }
    }
#endif

    set_frame_geometry(m.rows, m.cols);
    g_vot.order = vot::job_order(g_vot.job, m.frames);
    if ((int)g_vot.order.size() != g_vot.job.length) {
        fprintf(stderr, "[vot] run order is %zu frames, job length is %d\n",
                g_vot.order.size(), g_vot.job.length);
        return false;
    }
    g_run_frames = (int)g_vot.order.size();

    // Bring-up truncation. It shortens the run and therefore the trajectory, so
    // the trajectory is NOT written -- a short result file is read back without
    // complaint and scored as a tracker that stopped early.
    if (g_vot.max_frames > 0 && g_vot.max_frames < g_run_frames) {
        printf("[vot] TRUNCATED to %d of %d frames for bring-up. NO TRAJECTORY "
               "WILL BE WRITTEN — a short result scores as a tracker that "
               "stopped early.\n", g_vot.max_frames, g_run_frames);
        g_run_frames = g_vot.max_frames;
    }
    // The run list. "all" is every job in the manifest, in manifest order; a
    // comma list runs exactly those, in the order given, and MAY repeat an index
    // — that is the determinism test, not a mistake, so it is not de-duplicated.
    g_vot.job_list.clear();
    if (g_vot.job_spec.empty()) {
        g_vot.job_list.push_back(g_vot.job_index);
    } else if (g_vot.job_spec == "all") {
        for (size_t k = 0; k < m.jobs.size(); ++k) g_vot.job_list.push_back((int)k);
    } else {
        const char *p = g_vot.job_spec.c_str();
        while (*p) {
            char *end = nullptr;
            const long v = strtol(p, &end, 10);
            if (end == p) { fprintf(stderr, "[vot] bad --vot-jobs '%s'\n",
                                    g_vot.job_spec.c_str()); return false; }
            g_vot.job_list.push_back((int)v);
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
    }
    for (int j : g_vot.job_list)
        if (j < 0 || j >= (int)m.jobs.size()) {
            fprintf(stderr, "[vot] job %d out of range, %s has %zu\n",
                    j, m.sequence.c_str(), m.jobs.size());
            return false;
        }

    g_vot.traj.begin(g_vot.job);

    printf("[vot] %s  %dx%d x%d  %d frames  md5 %s\n",
           m.sequence.c_str(), m.rows, m.cols, m.channels, m.frames,
           m.blob_md5.c_str());
    printf("[vot] job %d/%zu: anchor %d %s, %d frames  init box %.1fx%.1f at "
           "(%.1f,%.1f)  anchors=%s\n",
           g_vot.job_index, m.jobs.size(), g_vot.job.anchor,
           g_vot.job.forward ? "forward" : "BACKWARD", g_vot.job.length,
           g_vot.job.init_box.h, g_vot.job.init_box.w,
           g_vot.job.init_box.row, g_vot.job.init_box.col,
           m.anchors_source.c_str());
    // Reported as its own slot, the way the AP_* slots are. phase0a.md budgets
    // ~4 s for the largest sequence at 117.2 MB/s and expects staging to be
    // 2-4% of the run -- measure it, do not assume it amortises.
    if (RESET_MUTANT)
        printf("[vot] *** RESET_MUTANT=%d IS ACTIVE — run_reset() is DELIBERATELY "
               "BROKEN (1 mean_prev, 2 filter_bo, 3 g_filter, 4 coast, 5 scale). "
               "The determinism test MUST fail. Results from this build are not "
               "tracking results. ***\n", RESET_MUTANT);
    printf("[vot] %zu run(s) queued: %s\n", g_vot.job_list.size(),
           g_vot.job_spec.empty() ? "one" : g_vot.job_spec.c_str());
    // On the streamed path there IS no staging read, and printing the resident
    // reader's zeroed counters here read as "staged 0.0 MB in 0.00 s = 0.0 MB/s"
    // -- a line that looks exactly like a blob that failed to load. Caught on
    // the first streamed hardware run (2026-08-27); the run was correct and the
    // log said otherwise, which is the wrong way round.
    if (g_vot.streamed)
        printf("[vot] no staging read: frames are fetched per frame from the "
               "mount and the wait is in the AP_VOT_STAGE slot\n");
    else
        printf("[vot] staged %.1f MB in %.2f s = %.1f MB/s\n",
               g_vot.blob.bytes() / 1048576.0, g_vot.blob.load_seconds(),
               g_vot.blob.bytes() / 1048576.0 / (g_vot.blob.load_seconds() > 0.0
                                                 ? g_vot.blob.load_seconds() : 1.0));
    // Advisory, both of them, and both are things Phase 1 measured in this
    // dataset rather than hypothesised: 11 of 62 sequences push the padded ROI
    // past the frame edge on 9.6% of frames (roi_crop border-clamps), and 22 of
    // 419 anchors init from a box under 16 px, the smallest being tennis at 2.0.
    if (m.roi_exceeds_frame > 0)
        printf("[vot] NOTE: roi = box x %.1f exceeds the frame on %d of %d "
               "frames — roi_crop border-clamps them\n",
               (double)mosse::DEFAULT_PADDING, m.roi_exceeds_frame, m.frames);
    {
        const double side = std::min(g_vot.job.init_box.h, g_vot.job.init_box.w);
        if (side < 16.0)
            printf("[vot] NOTE: init box side %.1f px — roi_crop's bilinear "
                   "interpolator runs at %.1fx UPSAMPLE\n",
                   side, PATCH_ROWS / (side * (double)mosse::DEFAULT_PADDING));
    }
    fflush(stdout);
    return true;
}

// The per-frame seam: dataset frame for run index k. One memcpy.
//
// CAN NOW RETURN nullptr, and the caller checks. On the resident path it never
// did -- the index comes from g_vot.order, which is built from the manifest --
// so the check is new with the streamed path, where an I/O error mid-run is a
// real outcome. Aborting the sequence with a message beats memcpy'ing from
// nullptr, and beats the alternative this project keeps meeting: continuing with
// a plausible frame.
static inline const uint8_t *vot_frame(int k)
{
    if (g_vot.streamed) return g_vot.stream.at((size_t)k);
    return g_vot.blob.frame(g_vot.order[(size_t)k]);
}

// Cumulative seconds this sequence has spent BLOCKED waiting for a streamed
// frame, both readers together. Zero on the resident path by construction, so
// the slot it feeds is a no-op there rather than a small lie.
static inline double vot_stage_wait_seconds(void)
{
    if (!g_vot.streamed) return 0.0;
    double w = g_vot.stream.wait_seconds();
#if CONV_IN_CH == 3
    w += g_vot.stream_luma.wait_seconds();
#endif
    return w;
}

// The message for a frame that did not arrive. Both readers can fail, and
// naming which one failed is the difference between a diagnosable log and
// "[vot] frame 4102 unavailable".
static void vot_report_frame_failure(int k)
{
    fprintf(stderr, "[vot] %s run index %d: frame unavailable\n",
            g_vot.sequence.c_str(), k);
    if (!g_vot.stream.error().empty())
        fprintf(stderr, "[vot]   blob stream: %s\n", g_vot.stream.error().c_str());
#if CONV_IN_CH == 3
    if (!g_vot.stream_luma.error().empty())
        fprintf(stderr, "[vot]   luma stream: %s\n",
                g_vot.stream_luma.error().c_str());
#endif
}

#if CONV_IN_CH == 3
// Same run-order indexing as vot_frame(), deliberately sharing g_vot.order
// rather than recomputing it: the sidecar is the SAME sequence in the SAME
// frame order, and two copies of that indexing is how a backward run ends up
// reading forward luma against interleaved colour -- a mismatch that degrades
// the scale filter and nothing else, so nothing would report it.
static inline const uint8_t *vot_luma_frame(int k)
{
    if (g_vot.streamed) return g_vot.stream_luma.at((size_t)k);
    return g_vot.luma.frame(g_vot.order[(size_t)k]);
}
#endif

// Frames whose groundtruth box is empty. 41 of 19,903 in stb2022, across 12
// sequences. The toolkit's failure rule ignores them; the board's IoU line
// cannot, so they are COUNTED and reported rather than quietly scored as 0.
static int g_vot_empty_gt = 0;

// Trajectories already produced in this process, keyed by job index. A repeated
// job must come back byte-identical; see the DETERMINISM block after the run
// loop. Kept in memory because both runs write the same filename.
static std::map<int, std::string> g_det_seen;
static int g_det_failures = 0;
static bool g_det_repeat = false;   // this run repeats an earlier job index

#endif  // FRAME_SOURCE_VOT
// conv2d weights: CONV_WEIGHT_BYTES_RAW INT8 taps (9 gray / 27 RGB) plus the
// scalar fields, padded to 64-byte GMIO alignment. See conv_weight_layout.h.
constexpr size_t WEIGHT_CH_BYTES   = CONV_WEIGHT_BYTES_PAD;
// conv2d emits one row-FFT window (PATCH_ROWS*FFT_ROW_WS samples) per invocation,
// so it fires this many times per patch. Its `weights` input_buffer is consumed
// once per invocation, so the host must send the weight buffer once per firing.
constexpr int    CONV_OUT_CHUNK    = PATCH_ROWS * FFT_ROW_WS;
constexpr int    CONV_INVOCATIONS  = (int)PATCH_ELEMS / CONV_OUT_CHUNK;

// -----------------------------------------------------------------------
// @thesis sec:przeplywDanych | A-01,P-06 | GMIO drain granularity: aie2gm_nb moves ONE kernel
//   invocation per call, so every output port needs an ordered async/wait pair per invocation.
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
// @thesis sec:przeplywDanych | P-06 | The DMA instrumentation that priced a transaction: 80
//   us/tx is per-transaction OVERHEAD, not bandwidth, and the largest transfer reaches 5.76
//   GB/s.
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
    DMA_ACCUM_IN,
    DMA_ACCUM_OUT, DMA_IFFT_ROW_IN, DMA_IFFT_ROW_OUT, DMA_IFFT_COL_IN,
    DMA_RESPONSE, DMA_N
};

struct DmaStat {
    const char   *name;
    unsigned long calls;
    double        us;        // async + wait, i.e. total host time on this port
    double        wait_us;   // the BLOCKING half alone — see the macros below
};

static DmaStat g_dma[DMA_N] = {
    {"gmio_weights",      0, 0.0, 0.0}, {"gmio_fft_row_out",  0, 0.0, 0.0},
    {"gmio_fft_col_in",   0, 0.0, 0.0}, {"gmio_fft_col_out",  0, 0.0, 0.0},
    {"gmio_cmul_in",      0, 0.0, 0.0}, {"gmio_accum_in",     0, 0.0, 0.0},
    {"gmio_accum_out",    0, 0.0, 0.0},
    {"gmio_ifft_row_in",  0, 0.0, 0.0}, {"gmio_ifft_row_out", 0, 0.0, 0.0},
    {"gmio_ifft_col_in",  0, 0.0, 0.0}, {"gmio_response",     0, 0.0, 0.0},
};
// The enum and this table are two ordered lists that must stay in step — the
// same coupling the AP_* slots already guard. Adding DMA_ACCUM_IN in the middle
// of the enum without adding it here would have silently RENAMED every port
// after it in the report, which is the kind of plausible-looking output this
// project has lost days to.
static_assert(sizeof(g_dma) / sizeof(g_dma[0]) == DMA_N,
              "g_dma name table is out of sync with the DMA_* enum");
static DmaStat g_dma_total[DMA_N];

// SPLIT 2026-08-21. These two used to share one accumulator, so every port's
// figure was `async` + `wait` fused — and this file's own methodology note says
// "DMA_T times async AND wait together — split them before drawing conclusions
// from any single port". That split was done for the roi_crop path in the 503 ms
// hunt and never for the GMIO ports.
//
// WHY IT MATTERS NOW: the wait half is the host BLOCKED, i.e. CPU that a second
// A72 core could be using. The async half is real host work that a second core
// cannot recover. Sizing the threading work needs the two apart, and the
// CMUL_ACCUM_MEMTILE result (16x fewer transactions, identical cost) says most
// of it should be wait — this measures rather than infers that.
//
// `us` keeps its old meaning (async + wait) so every figure recorded before
// today stays directly comparable; `wait_us` is the new half.
#define DMA_T(slot, stmt) do {                                                  \
        const auto _t0 = std::chrono::steady_clock::now();                       \
        stmt;                                                                    \
        const double _d = std::chrono::duration<double, std::micro>(              \
            std::chrono::steady_clock::now() - _t0).count();                      \
        g_dma[slot].us      += _d;                                               \
        g_dma[slot].wait_us += _d;                                               \
    } while (0)

#define DMA_TX(slot, stmt) do {                                                 \
        const auto _t0 = std::chrono::steady_clock::now();                       \
        stmt;                                                                    \
        g_dma[slot].us += std::chrono::duration<double, std::micro>(              \
            std::chrono::steady_clock::now() - _t0).count();                      \
        ++g_dma[slot].calls;                                                      \
    } while (0)

static void dma_reset_frame(void)
{
    for (int i = 0; i < DMA_N; ++i) { g_dma[i].calls = 0; g_dma[i].us = 0.0;
                                      g_dma[i].wait_us = 0.0; }
}

// -----------------------------------------------------------------------
// roi_crop launch-phase instrumentation
// -----------------------------------------------------------------------
// WHY THIS EXISTS. Timestamped console capture on 2026-08-17 put 478.7 ms x 16
// channels = 7.66 s in the interval between the "weights sent + row-FFT drained"
// and "roi_crop done" prints — 94% of an 8.18 s frame, against a design budget
// of 0.7 ms/frame for this kernel (a BUDGET, not a measurement; claim P-03). That interval contained exactly one
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
// @thesis sec:metodykaBadan | P-02,M-05 | The APU per-frame cost breakdown. Measure the total
//   and print the RESIDUAL: a profiler that does not account for the whole frame lets you
//   conclude confidently and wrongly.
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
    AP_COLOURISE,    // luma scene -> interleaved RGB, touched rect only.
                     // Zero at CONV_IN_CH=1 (the call compiles away). Given its
                     // OWN slot for the same reason AP_BO_STAGE has one: it is
                     // the cost RGB ADDS to the host, and a cost folded into
                     // AP_FRAME_PUSH would also have doubled that slot's call
                     // count and halved its per-call figure.
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
    AP_FILTER,       // filter_init, or filter_update_quantize (update AND the
                     // Q1.15 conversion) since the two were fused 2026-08-21
    AP_PUBLISH,      // pack_filter + sync. NO LONGER the quantiser: that moved
                     // into AP_FILTER with the fusion, so this slot dropping to
                     // ~1 ms is the fusion landing, not the publish getting fast
    AP_SCALE_EXTRACT,// scale_extract. ONE call/frame since 2026-08-21 (the
                     // update reuses the detection sample); 2 on frame 0
    AP_SCALE_MODEL,  // scale_detect + scale_update/_shifted, pure heap
    AP_DIAG_SCAN,    // report_cint16's max/rails scan. It runs at EVERY verbosity
                     // because rails detection is the point, so it is a real
                     // per-frame cost that was sitting in UNATTRIBUTED. THREE
                     // calls/frame since 2026-08-21, not four: F_ch's scan now
                     // rides along with unpack_spectrum and its cost is in
                     // AP_UNPACK. The accum scan reads a heap copy, not the BO.
    AP_DET_HASH,     // the run-state digest — see g_det_hash. Its own slot
                     // because it is an INSTRUMENT added to the timed path, and
                     // an instrument whose cost is invisible cannot be judged.
#if FRAME_SOURCE_VOT
    // GUARDED, so the synth arm's ELF is byte-identical to a build without any
    // of this. It is not enough that the slot would read zero there: an extra
    // enumerator shifts nothing but does add a row to every frame report, and
    // this project's own rule is that the two arms differ STRUCTURALLY rather
    // than by a value that happens to be zero. `cmp` on the synth ELF against
    // HEAD is the check, and it is the same instrument that caught
    // PROGRESS_EVERY changing codegen through an unused static counter.
    AP_VOT_STAGE,    // BLOCKED waiting for a streamed frame, both readers. Zero
                     // on the resident path. Its own slot
                     // because the whole argument for a prefetched ring over a
                     // synchronous refill is that the wait is usually zero — and
                     // a claim like that is worth nothing unless the run prints
                     // the number. On the two 1080p sequences it will NOT be
                     // zero (7.9 MB/frame against 117 MB/s is 67 ms), and this
                     // is the slot that says so instead of inflating
                     // UNATTRIBUTED, which is exactly what mmap would have done.
#endif
    AP_N
};
static const char *g_ap_name[AP_N] = {
    "scene gen", "frame push (2MB)", "colourise RGB", "frame_bo.sync", "transpose",
    "window mean+energy",
    "fcol_bo.sync", "BO<->heap stage", "unpack F_ch", "cmul packing",
    "B2 correction", "PSR scan",
    "filter upd+quant", "publish (pack)", "scale extract", "scale detect+update",
    "diag scan (rails)", "determinism hash",
#if FRAME_SOURCE_VOT
    "vot frame stage (wait)",
#endif
};
// The enum and the name table are two lists that must stay the same length, in
// the same order — exactly the class of coupling CLAUDE.md flags as "duplicated
// in four files with no compile-time check". Here there IS one:
static_assert(sizeof(g_ap_name) / sizeof(g_ap_name[0]) == AP_N,
              "g_ap_name is out of sync with the AP_* enum");

// Wall time the helper thread's work OVERLAPPED the main thread's, per frame and
// cumulative. WITHOUT THIS THE TABLE LIES: with TAIL_PARALLEL the AP_FILTER slot
// is filled on core 1 while AP_SCALE_* is filled on core 0, so the subtotal
// counts wall time the frame never spent and UNATTRIBUTED goes NEGATIVE
// (-1.503 ms in runs/run_0821_1712.log). A residual that cannot physically exist
// is exactly the kind of plausible-looking output this project has lost days to.
static double        g_ap_overlap_us     = 0.0;   // this frame
static double        g_ap_overlap_tot_us = 0.0;   // run

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
    g_ap_overlap_us = 0.0;
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
    g_ap_overlap_tot_us += g_ap_overlap_us;
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
// @thesis sec:metodykaBadan | P-09 | Console verbosity is part of the measured system: at
//   115200 the console was 15% of the frame, and 58% on one sequence.
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
// @thesis subsec:kosztTransferow | P-05 | The GMIO drain-depth probe: XRT allows ONE
//   outstanding async per port, so there is no pipeline to deepen.
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

// HOW OFTEN THE LEVEL-0 PROGRESS LINE IS PRINTED.
//
// The comment above prices that line at "~4 ms against an ~87 ms floor, 4%".
// Both halves of that were true in 2026-08-20 terms and the floor has since
// moved: the frame is 26.29 ms (docs/thesis/results/perf.csv, run_0821_1725, claims
// P-01 and P-09), so the same line is now 15% of it, and on a
// gate-heavy sequence like `animal` the console is 58% of the frame
// (correlation(gated%, unattributed) = 0.963 over the 8-sequence sweep).
//
// The fix is NOT to silence level 0 -- its per-frame marker is what
// `picocom | ts` times, and deleting it would delete the instrument. It is to
// print the marker every Nth frame on runs that are measuring something else.
//
// DEFAULT 1 = every frame = byte-identical to the pre-2026-08-25 console. A VOT
// sweep sets it to 10 or 25. Frame 0 and the last frame ALWAYS print, whatever
// N: the first is the run's start marker and the last is its end, and a run
// whose final line is missing looks exactly like a run that hung.
#ifndef PROGRESS_EVERY
#  define PROGRESS_EVERY 1
#endif
#if PROGRESS_EVERY < 1
#  error "PROGRESS_EVERY must be >= 1 (1 = every frame)"
#endif

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
    return frame == 0 || frame == g_run_frames - 1;
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
    if (frame >= RC_TRACE_FRAMES && frame != g_run_frames - 1) return;
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
// @thesis subsec:petlaSterowania | P-03 | roi_crop driven as a user-managed CU through xrt::ip:
//   the host writes AXI-Lite and polls the CU's own ap_done. Worth 20.6x on frame rate.
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
// @thesis subsec:petlaSterowania | P-03 | The control-CU probe: a kernel with no AXIS port paying
//   the same 512 ms is what turned 'roi_crop is slow' into 'any KDS completion is slow'.
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
        g_dma_total[i].calls   += g_dma[i].calls;
        g_dma_total[i].us      += g_dma[i].us;
        g_dma_total[i].wait_us += g_dma[i].wait_us;
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
        printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx   wait %8.3f ms (%3.0f%%)\n",
               g_dma[i].name, g_dma[i].calls, g_dma[i].us / 1000.0, per,
               g_dma[i].wait_us / 1000.0,
               g_dma[i].us > 0.0 ? 100.0 * g_dma[i].wait_us / g_dma[i].us : 0.0);
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
#if !MEMTILE_TRANSPOSE   // both call sites are the DDR transpose path
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
#endif  // !MEMTILE_TRANSPOSE

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
// @thesis sec:metodykaBadan | R-06 | The PSR gate control path, Bolme 3.5. Four veto reasons
//   reported separately, because 88% of vetoes are NEGATIVE_PEAK, which the threshold cannot
//   disable.
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

// Multi-start: every one of these is PER RUN, and a run that inherits the
// previous one's counters reports a plausible summary for a run that did not
// happen. Reset from run_reset() together with the tracking and scale counters.
static void gate_reset_run(void)
{
    g_gate_run = g_gate_worst = g_gate_eval = g_gate_hold = 0;
    for (int i = 0; i < 8; ++i) g_gate_reason_n[i] = 0;
    g_psr_min = g_psr_max = g_psr_sum = 0.0;
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
// @thesis sec:metodykaBadan | M-07 | track.csv, the run's actual product. Any per-frame reader
//   must key on (job, frame): a bare frame index collides across anchors and once
//   under-reported rails 66x.
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

// HOW OFTEN track.csv IS FLUSHED, IN ROWS.
//
// csv_row() used to fflush() every row, justified as "nothing against 1216 KB
// of binaries per frame". At DUMP_BUFFERS=0 there are no binaries, so what is
// left is a filesystem sync in the timed path, once per frame.
//
// DEFAULT 1 = flush every row = the previous behaviour exactly, and that
// default is deliberate: the justification for per-row flushing was surviving a
// power cut, and a power cut is not hypothetical here -- one took out arm B's
// car1 run on 2026-08-25. Raising it is a decision to trade the tail of a run
// for frame time, and a sweep makes that trade knowingly.
//
// The whole dataset's rows are ~4.6 MB, so buffering an entire run costs
// nothing in memory. csv_close() flushes, and so does fclose() underneath it.
#ifndef CSV_FLUSH_EVERY
#  define CSV_FLUSH_EVERY 1
#endif
#if CSV_FLUSH_EVERY < 1
#  error "CSV_FLUSH_EVERY must be >= 1 (1 = every row)"
#endif

static FILE *g_csv = nullptr;
// Which run each CSV row belongs to. -1 at FRAME_SOURCE=synth, where there is
// exactly one run and no anchor.
static int g_csv_job = 0, g_csv_anchor = -1;

struct FrameDiag {
    double fch = 0.0, accum = 0.0, resp = 0.0, h = 0.0;
    int    rails = 0;
    // PER-BUFFER rails, added 2026-08-26. `rails` above is the SUM over all four
    // scans, so a railed row has never said WHICH buffer saturated -- and the
    // lever differs by buffer: H_SHIFT is upstream of both the accumulator and
    // the response, while IFFT_ROW_SHIFT/IFFT_COL_SHIFT reach only the response.
    // Sizing the budget from an unattributed total is choosing a knob blind.
    //
    // The one attributed sample that existed (car1 anchor 0, VERBOSITY=1,
    // runs/run_0825_1314.log) is 8 response-railed frames against 4 accum, i.e.
    // it does NOT support the accumulator being the problem. Two columns retire
    // the guess.
    int    rails_fch = 0, rails_accum = 0, rails_resp = 0, rails_h = 0;
};
static FrameDiag g_fdiag;

// FILTER_MASK_STAT — the mechanism check for the spatial mask, and the only way
// a mask run can answer its own falsifier.
//
// docs/thesis/evidence/arm_mask.md 4: "the fraction of the filter's energy inside
// the target box should rise from the measured 51.6% (car1) / 54.9% (tiger). If
// EAO moves while that does not, the gain is not the mask and the result is
// unattributable." Those two numbers existed only as a COMMENT in
// rgb_vs_gray_loop.py — nothing computed them, on the host or offline — so the
// falsifier had no instrument until this.
//
// Independent of FILTER_MASK on purpose: the baseline arm has to be measurable
// with the SAME instrument, or the comparison is between a number and a memory.
// Default 0, so the shipping ELF is untouched.
#ifndef FILTER_MASK_STAT
#  define FILTER_MASK_STAT 0
#endif
// The sampling schedule for the statistic above. Defaults chosen against
// vot_mask_stat.py's PROFILE_FRAMES (1, 5, 20, 40): 1 and 5 fall inside WARM,
// 20 and 40 are multiples of EVERY. Move either and that reader's per-frame
// columns thin out silently, which is why both are named here and there.
#ifndef FILTER_MASK_STAT_WARM
#  define FILTER_MASK_STAT_WARM 5
#endif
#ifndef FILTER_MASK_STAT_EVERY
#  define FILTER_MASK_STAT_EVERY 20
#endif
#if FILTER_MASK_STAT
// Fraction of SUM|h|^2 inside a centred box the size of the target, or -1 on a
// frame where the filter was NOT re-formed (frame 0's filter_init path and every
// held frame). -1 rather than the previous frame's value: a stale reading that
// looks live is exactly the failure this project keeps paying for, and a reader
// that averages a mask statistic over held frames is measuring the hold rate.
static double g_mask_ebox = -1.0;
#endif

static void csv_open(void)
{
#if CSV_LOG
    // THE NAME CARRIES THE SEQUENCE at FRAME_SOURCE=vot, because one evidence
    // sweep is one ELF invocation PER SEQUENCE and this file opens in "w" mode:
    // a fixed name means eight sequences leave one file, and the loss is silent
    // — the survivor looks like a perfectly good CSV. The arm (HOLD_COAST and
    // the rest) is NOT in the name: that separation comes from --vot-results,
    // which already has to differ per arm because the trajectories collide too.
    //
    // It stays on the SD card and NOT on the results mount: csv_row() still
    // fflush()es every row, and putting that on NFS would move a filesystem sync
    // into the timed path (Phase 4's open item).
    char namebuf[128];
    const char *path = "track.csv";
#if FRAME_SOURCE_VOT
    {
        // Defensive sanitisation: the sequence name reaches here from a manifest
        // on a mount, so it is INPUT. Anything outside [A-Za-z0-9._-] becomes
        // '_', which keeps a hostile or merely odd name from composing a path.
        char safe[64];
        size_t k = 0;
        for (const char *c = g_vot.manifest.sequence.c_str();
             *c && k < sizeof(safe) - 1; ++c) {
            const bool ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                            (*c >= '0' && *c <= '9') || *c == '.' || *c == '_' ||
                            *c == '-';
            safe[k++] = ok ? *c : '_';
        }
        safe[k] = '\0';
        if (k > 0) {
            snprintf(namebuf, sizeof namebuf, "track_%s.csv", safe);
            path = namebuf;
        }
    }
#else
    (void)namebuf;
#endif
    // Same best-effort placement as dump_buffer(): cwd (the SD card) first, then
    // /tmp. Never fatal — losing the CSV must not abort a 500-frame run.
    char tmpbuf[160];
    g_csv = fopen(path, "w");
    if (!g_csv) {
        snprintf(tmpbuf, sizeof tmpbuf, "/tmp/%s", path);
        path = tmpbuf;
        g_csv = fopen(path, "w");
    }
    if (!g_csv) {
        printf("[csv] no writable location — stdout diagnostics only\n");
        return;
    }
    // job,anchor are TRAILING columns, added 2026-08-25 for multi-start. One
    // track.csv now carries several runs, and rows from different anchors are
    // otherwise indistinguishable -- a mean IoU computed over the whole file
    // would silently average unrelated runs. Trailing so every existing reader
    // that selects by header name is unaffected.
    fprintf(g_csv,
            "frame,occluded,evaluated,accept,reason,psr_bolme,psr_ratio,peak,"
            "dr_bin,dc_bin,resp00_over_peak,est_row,est_col,est_h,est_w,"
            "truth_row,truth_col,truth_h,truth_w,iou,centre_err,published,"
            // The scale filter's own verdict. Added 2026-08-20: the previous run
            // logged only est_h, so the level the detector actually PROPOSED had
            // to be reverse-engineered from log(est_h/64)/log(a) — which is how
            // "the detector only ever proposes +-1" was found, slowly. Log it.
            "scale_idx,scale_conf,scale_reason,"
            // Added 2026-08-24. rails was THE number track.csv did not carry,
            // so a budget verdict needed the console and therefore VERBOSITY=1.
            // accum_max/h_max are the other two the console alone had; fch0_max
            // is ch0 ONLY (see diag_record) — not a bank maximum. `response` is
            // deliberately absent: `peak` above already is it.
            "rails,accum_max,fch0_max,h_max,job,anchor,"
            // Added 2026-08-26, for the real-video shift-budget re-derivation.
            //
            // resp_max: the response scan's max COMPLEX MAGNITUDE. `peak` is a
            // near-perfect proxy for it (identical on 739 of 741 car1 frames and
            // on 199 of 199 synthetic ones, the two exceptions being amplitude
            // ~25 of 32767) but it is not the same quantity: `peak` is the
            // SIGNED REAL PART at the argmax of |real|, so it cannot see a bin
            // that saturates in the IMAGINARY part alone. Budget sizing reads
            // the buffer the rail detector reads.
            //
            // rails_*: which buffer railed. See FrameDiag.
            "resp_max,rails_fch,rails_accum,rails_resp,rails_h"
#if FILTER_MASK_STAT
            // Added 2026-08-29. TRAILING, like job/anchor, so every reader that
            // selects by header name is unaffected. -1 means "not re-formed this
            // frame" and must not be averaged in as a zero.
            ",mask_ebox"
#endif
            "\n");
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
            "%d,%.4f,%s,%d,%.0f,%.0f,%.0f,%d,%d,"
            "%.0f,%d,%d,%d,%d"
#if FILTER_MASK_STAT
            ",%.4f"
#endif
            "\n",
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
            scale_evaluated ? mosse::scale_veto_tag(sd.reason) : "NOT_RUN",
            g_fdiag.rails, g_fdiag.accum, g_fdiag.fch, g_fdiag.h,
            g_csv_job, g_csv_anchor,
            g_fdiag.resp, g_fdiag.rails_fch, g_fdiag.rails_accum,
            g_fdiag.rails_resp, g_fdiag.rails_h
#if FILTER_MASK_STAT
            , g_mask_ebox
#endif
            );
    // Per ROW by default: the point is surviving a power cut, and one really did
    // happen mid-sweep. CSV_FLUSH_EVERY > 1 trades that tail for frame time on a
    // run that is measuring frame time.
    //
    // A RAILED row flushes whatever N is. A rail invalidates the shift budget
    // and is the one row worth interrupting a sweep over, and rails are rare on
    // a healthy run -- 8 frames in 742 on the worst one recorded -- so the
    // exception costs nothing.
    //
    // A GATE VETO deliberately does NOT force a flush, though it is tempting.
    // Vetoes are a routine tracking outcome, not a defect, and they are
    // COMMONEST on exactly the runs this knob exists for: `animal` gates 76% of
    // its frames, so flushing on vetoes would flush 76% of rows and save nothing
    // on the sequence where the console/IO cost is worst (58% of the frame).
#if CSV_FLUSH_EVERY > 1
    static long csv_rows = 0;
    ++csv_rows;
    if ((csv_rows % CSV_FLUSH_EVERY) == 0 || g_fdiag.rails > 0)
        fflush(g_csv);
#else
    fflush(g_csv);      // the default arm is the ORIGINAL line, textually
#endif
#else
    (void)frame; (void)occluded; (void)evaluated; (void)gate; (void)p;
    (void)resp00_over_peak; (void)est; (void)truth; (void)iou; (void)cerr;
    (void)published; (void)scale_evaluated; (void)scale_idx; (void)sd;
#endif
}

static void csv_close(void)
{
#if CSV_LOG
    // fclose() flushes -- no explicit fflush needed, and adding one would make
    // the CSV_FLUSH_EVERY=1 build differ from the pre-knob binary for no reason.
    if (g_csv) { fclose(g_csv); g_csv = nullptr; }
#endif
}

// Peak magnitude and saturation count for a cint16 buffer. `rails > 0` means the
// stage clipped, which is the failure mode the shift budget exists to prevent.
//
// SPLIT into a scan and a printer 2026-08-21, so the scan can be folded into a
// pass that is already reading the same bytes (see unpack_spectrum) instead of
// being a second pass over 64 KB.
struct Cint16Scan {
    int64_t max_m2 = -1;      // squared magnitude of the peak bin
    int     max_i  = 0;
    int     rails  = 0;
};

// INTEGER, not fp64. The operands are int16 so re*re + im*im is at most
// 2 * 32768^2 = 2.1e9 — exact in int64, and exact in double too, which is why
// this selects bitwise the same peak bin as the fp64 loop it replaces. The
// change is the arithmetic, not the answer: the same substitution took
// `window mean + energy` from 901 us/call to 26 (34x) on the A72, because an
// int64 MAC loop vectorises where an fp64 one on this core does not.
static inline void scan_cint16_into(Cint16Scan &s, int re, int im, int i)
{
    const int64_t m = (int64_t)re * re + (int64_t)im * im;
    // TIE-BREAK ON THE LOWER INDEX, not on visit order. The scan used to be a
    // strict `>` fed in increasing-index order, so ties resolved to the lowest
    // index by accident of the loop. unpack_spectrum now visits in BLOCKED
    // order, which would silently pick a different bin whenever two bins share a
    // magnitude — an all-zero buffer is the obvious case, and it would report a
    // different `max|.| at idx N` line from one build to the next with nothing
    // else changed. Making the rule explicit keeps the reported index identical
    // and makes the scan independent of traversal order for good.
    if (m > s.max_m2 || (m == s.max_m2 && i < s.max_i)) { s.max_m2 = m; s.max_i = i; }
    // re and im are int16, so `>= 32767` IS `== 32767`. Written as the
    // inequality to keep it readable next to the widened arithmetic.
    if (re >= 32767 || re <= -32768 || im >= 32767 || im <= -32768) ++s.rails;
}

static Cint16Scan scan_cint16(const int16_t *b, int n)
{
    Cint16Scan s;
    for (int i = 0; i < n; ++i) scan_cint16_into(s, b[2 * i], b[2 * i + 1], i);
    return s;
}

// Printed at VERBOSITY >= 1, but ALWAYS when something railed. `rails` is the
// shift-budget instrument and it is the one number here that track.csv does not
// carry, so a quiet run must still shout when a bin saturates — silencing an
// anomaly to save console is how a budget hunt goes wrong.
// Per-frame diagnostic maxima, kept for track.csv.
//
// WHY THIS EXISTS: at VERBOSITY=0 the [diag] lines print only when something
// rails, so a healthy run yields NO amplitude data at all — which is why every
// calibration run so far had to pay VERBOSITY=1's console cost (62 ms/frame
// against 26) and therefore could not also be an FPS measurement. The scan
// itself ALREADY RUNS at every verbosity (report_cint16 gates the print, not
// scan_cint16), so these four numbers were being computed and thrown away.
// Recording them makes a VERBOSITY=0 run fully diagnosable.
//
// @thesis subsec:arytmetyka | B-04,B-07 | The saturation instrument: rails and accum_max are
//   scanned every frame regardless of console verbosity and land in track.csv.
// Single-threaded by construction: with TAIL_PARALLEL all four scans and
// csv_row() run on the main thread AFTER g_filter_thr.join(), so no lock.
// `fch` is CHANNEL 0 ONLY — report_cint16_scan("F_ch", ...) is called under
// `if (ch == 0)`. The column is named fch0_max so the CSV cannot be misread as
// a bank maximum; at CONV_IN_CH=3 ch0 is a colour-opponent channel and its
// amplitude is the cheap on-board test that colour reaches conv2d.
static void diag_record(const char *tag, const Cint16Scan &s)
{
    const double m = (s.max_m2 > 0) ? sqrt((double)s.max_m2) : 0.0;
    if      (!strcmp(tag, "F_ch"))   { g_fdiag.fch   = m; g_fdiag.rails_fch   += s.rails; }
    else if (!strcmp(tag, "accum"))  { g_fdiag.accum = m; g_fdiag.rails_accum += s.rails; }
    else if (!strcmp(tag, "response")){g_fdiag.resp  = m; g_fdiag.rails_resp  += s.rails; }
    else if (!strcmp(tag, "H(q15)")) { g_fdiag.h     = m; g_fdiag.rails_h     += s.rails; }
    // The TOTAL still accumulates unconditionally, including for any tag not
    // named above: a buffer that starts railing under a tag nobody added a
    // column for must still make `rails` non-zero. The per-buffer columns are
    // an attribution of the total, never a replacement for it -- if they stop
    // summing to it, that discrepancy is itself the finding.
    g_fdiag.rails += s.rails;
}

static void report_cint16_scan(const char *tag, const Cint16Scan &s, int cols,
                               const char *layout)
{
    diag_record(tag, s);
    if (VERBOSITY >= 1 || s.rails > 0)
        printf("  [diag] %-9s max|.|=%7.0f at %s idx %d (%d,%d)  rails=%d%s\n",
               tag, sqrt((double)s.max_m2), layout, s.max_i,
               s.max_i / cols, s.max_i % cols, s.rails,
               s.rails > 0 ? "   <-- RAILED" : "");
}

// Scan and print, for the buffers that no other pass touches. Indices are
// reported in the buffer's own layout — the caller says which.
static void report_cint16(const char *tag, const int16_t *b, int rows, int cols,
                          const char *layout)
{
    const auto _ds0 = std::chrono::steady_clock::now();
    const Cint16Scan s = scan_cint16(b, rows * cols);
    g_ap_us[AP_DIAG_SCAN] += std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - _ds0).count();
    ++g_ap_n[AP_DIAG_SCAN];
    report_cint16_scan(tag, s, cols, layout);
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
// Layout per channel (64 B): conv_weight_layout.h, derived from CONV_IN_CH.
// Byte 63 of each buffer carries the layout tag the exporter wrote; a build/file
// mismatch is checked at load time below.
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
// @thesis subsec:operacjeCzestotliwosc | A-05 | Stage B2: the 9-bin frequency-domain mean
//   correction on the APU, exact only because the window is a periodic Hann.
// Stage B — feature-map normalization (see conv2d_kernel.h for the rationale)
// -----------------------------------------------------------------------

// Per-channel window-weighted feature mean, fed back to conv2d as mean_prev in
// the next frame's weight buffer (at CONV_W_OFF_MEAN). Zero on the first frame.
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
#if !MEMTILE_TRANSPOSE
// Reads the ROW-FFT output, which does not reach the host on the memtile path —
// measure_window_mean_fch() replaces it. Compiled out rather than left dead so
// the build stays warning-clean and the dependency is visible in the source.
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
#endif  // !MEMTILE_TRANSPOSE

// -----------------------------------------------------------------------
// @thesis subsec:operacjeCzestotliwosc | A-05,P-05 | Stage B1's mean and Stage B3's per-channel
//   energy recovered from F_ch -- the non-obvious cost of deleting gmio_fft_row_out.
// MEMTILE_TRANSPOSE: recovering Stage B1's mean and Stage B3's energy from F_ch
// -----------------------------------------------------------------------
// THE NON-OBVIOUS COST OF DELETING gmio_fft_row_out. measure_window_mean() and
// the Parseval energy loop both read row_bo — the ROW-FFT output — and with the
// transpose inside the graph that buffer stops existing on the host. Two things
// depend on them and both are load-bearing:
//
//   mean_now  -> weights[CONV_W_OFF_MEAN] -> conv2d Stage B1 on the NEXT frame.
//                Without it B1 is inert and the ch16 response rails flat
//                (CLAUDE.md, "mean_prev seeding": F_ch 32768 railed -> 53).
//   g_energy  -> Stage B3 in filter_quantize_q15.
//
// Both are recoverable from F_ch, which the host still receives on
// gmio_fft_col_out, and at no extra DMA — it is already staged in g_stage_c.
//
//   MEAN. measure_window_mean sums the real part of each row's bin 0, i.e.
//   SUM_r X_row[r][0]. The column FFT evaluated at frequency 0 IS that sum, so
//   it lands in F_ch bin (0,0) — element 0 of the col-FFT layout — attenuated by
//   the column pass's FFT_SHIFT and by DSPLib's ADDITIVE cint16 loss of ~21 per
//   pass (CLAUDE.md: "the loss is additive, not a gain factor"). Undo both.
//
//   ENERGY. Parseval: the column DFT multiplies total energy by PATCH_ROWS and
//   the shift divides it by 2^(2*FFT_SHIFT). Those are the SAME constants for
//   every channel, and filter_quantize_q15 renormalises H globally to full
//   scale, so a common factor on every g_energy[ch] cancels exactly. The factor
//   is applied anyway, purely so the logged magnitudes stay comparable with
//   every run recorded before this change.
//
// THIS IS NOT BIT-EXACT WITH THE DDR PATH, AND THAT MATTERS FOR HOW THE RUN IS
// SCORED. The shift, the rounding and the additive loss are undone
// approximately, so mean_now differs by a few LSB, feeds back into conv2d's B1,
// and moves the whole datapath. **The memtile build CANNOT be accepted on the
// bit-identical criterion that every host change since 2026-08-20 used.** Score
// it on IoU/PSR/centre error being statistically unchanged instead, and expect
// the frame-0 F_ch fingerprint to move. mean_now is a feedback loop by
// construction ("even if the scale factor above is slightly off, mean_prev
// converges over a few frames"), which is what makes the approximation safe.
#if MEMTILE_TRANSPOSE
// DSPLib's per-pass additive DC loss, cint16. See CLAUDE.md: row_dc =
// PATCH_COLS*c - 21, accum0 = PATCH_ROWS*row_dc - 21.
static constexpr int DSPLIB_DC_LOSS = 21;

static int32_t measure_window_mean_fch(const int16_t *fch, int32_t mean_prev)
{
    int64_t sum_w = 0;
    for (int i = 0; i < PATCH_COLS; ++i) sum_w += (int64_t)HTAB[i];

    // Col-FFT layout: element [k*PATCH_ROWS + m] is bin (m,k), so bin (0,0) is
    // element 0 and its real part is int16 index 0.
    const int64_t dc_sum = ((int64_t)fch[0] + DSPLIB_DC_LOSS) << FFT_SHIFT_CFG;

    const double sw = (double)sum_w / 32768.0;
    if (sw <= 0.0) return mean_prev;
    const double residual = (double)dc_sum / (sw * sw);
    return mean_prev + (int32_t)llround(residual);
}

// Parseval, with the column pass's constants undone so the value stays on the
// same scale as the row-FFT figure this replaces.
static double measure_energy_fch(const int16_t *fch)
{
    int64_t e = 0;
    for (size_t i = 0; i < PATCH_ELEMS; ++i)
        e += (int64_t)fch[2*i] * fch[2*i] + (int64_t)fch[2*i+1] * fch[2*i+1];
    const double col_gain = (double)PATCH_ROWS
                          / (double)((int64_t)1 << (2 * FFT_SHIFT_CFG));
    return (double)e / (double)PATCH_ELEMS / col_gain;
}

#if !B2_NULL_BINS
#  error "MEMTILE_TRANSPOSE with B2_NULL_BINS=0 is not supported: the SUBTRACT \
mode needs residual_mean at the accuracy of the row-FFT DC bins, and the F_ch \
derivation above is only good to a few LSB. Use B2_NULL_BINS=1 (the default), \
which ignores residual_mean entirely."
#endif
#endif  // MEMTILE_TRANSPOSE

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
// Software-pipeline roi_crop: launch channel ch+1's crop immediately after ch's
// ap_done, so the CU's PL execution overlaps the host's APU work instead of
// being spun on. Only meaningful with MEMTILE_TRANSPOSE — on the DDR path the
// row-FFT drain already covers the CU, which is exactly why RC_POLL read
// 0.067 ms/frame there and 5.196 once the drain was deleted
// (docs/thesis/results/apu_stages.csv, claim P-05).
#ifndef ROI_CROP_PIPELINE
#  define ROI_CROP_PIPELINE 1
#endif
#define RC_PIPELINE (ROI_CROP_PIPELINE && MEMTILE_TRANSPOSE)

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

// THE SCENE IS GENERATED IN LUMA AND COLOURISED ON THE WAY OUT.
//
// Every scene function — fill_background, the pan/dirty-rect restore,
// inject_target_frame, scene_add_noise, the occluder — is single-plane, and so
// is scale_extract, which crops 33 windows out of the frame every frame. Making
// all of that plane-aware would be a large invasive rewrite of code that is
// correct and measured. Instead the luma scene stays exactly as it is and one
// colourise pass expands it into the interleaved buffer the device reads.
//
// At CONV_IN_CH=1 there is NO second buffer and NO copy: g_frame_host IS the
// luma scene, byte for byte as before. The shipping path does not pay for RGB.
#if CONV_IN_CH == 3
static std::vector<uint8_t> g_scene_luma;
static inline uint8_t *scene_luma() { return g_scene_luma.data(); }
#else
static inline uint8_t *scene_luma() { return g_frame_host.data(); }
#endif

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
//
// `scan`, when non-null, collects the max/rails diagnostic FOR FREE: this loop
// already reads every one of the 16384 cint16 values that report_cint16() would
// re-read in a second pass. The four standalone scans cost 1.19 ms/frame on the
// A72 (runs/run_0820_1807.log), which is 1.9% of the frame spent reading data
// twice. Indices are the SOURCE's, i.e. col-FFT layout — the same convention
// report_cint16("F_ch", ...) used, so the printed line does not change meaning.
static void unpack_spectrum(const int16_t *src, mosse::cfloat *dst,
                            Cint16Scan *scan = nullptr)
{
    // BLOCKED, because this is a transpose and a transpose written as two nested
    // loops is a cache-miss generator.
    //
    // The source walk is contiguous (s = k*PATCH_ROWS + m, m innermost) but the
    // destination stride is PATCH_COLS * sizeof(cfloat) = 1024 B, so the naive
    // form issues 16384 scattered 8-byte stores, each landing on its own cache
    // line, against a 128 KB destination that does not fit in L1 or stay in L2.
    // It measured 181 us/call = 11 ns/element, ~2.9 ms/frame over 16 channels
    // (docs/thesis/results/apu_stages.csv, claim P-05).
    //
    // With BLK = 16 the working set per tile is 16 destination rows x 128 B plus
    // 16 source columns x 64 B — about 3 KB, comfortably L1-resident — so each
    // cache line fetched is fully used before eviction instead of once.
    //
    // BLK divides both dimensions at every supported patch size (64, 128, 256),
    // asserted rather than assumed so a future geometry cannot silently drop the
    // tail of the spectrum.
    constexpr int BLK = 16;
    static_assert(PATCH_ROWS % BLK == 0 && PATCH_COLS % BLK == 0,
                  "unpack_spectrum's block size must divide the patch dimensions");

    for (int k0 = 0; k0 < PATCH_COLS; k0 += BLK)
        for (int m0 = 0; m0 < PATCH_ROWS; m0 += BLK)
            for (int k = k0; k < k0 + BLK; ++k) {
                const int16_t     *sp = src + 2 * ((size_t)k * PATCH_ROWS + m0);
                mosse::cfloat     *dp = dst + (size_t)m0 * PATCH_COLS + k;
                for (int m = 0; m < BLK; ++m) {
                    const int re = sp[2 * m], im = sp[2 * m + 1];
                    dp[(size_t)m * PATCH_COLS] = mosse::cfloat((float)re, (float)im);
                    if (scan)
                        scan_cint16_into(*scan,
                                         re, im,
                                         (int)((size_t)k * PATCH_ROWS + m0 + m));
                }
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

static std::vector<mosse::cfloat> g_h_scratch;


// Q1.15 scale/peak produced by filter_update_quantize. GLOBAL rather than local
// because with TAIL_PARALLEL the producer is a different thread from the
// consumer (publish_packed), joined in between.
static float g_q15_scale = 0.0f, g_q15_max = 0.0f;
// AP_FILTER's accumulator at the moment the helper was launched, so the join can
// recover the helper's own elapsed time by difference.
static double g_ap_filter_at_launch = 0.0;
// @thesis subsec:petlaSterowania | P-05 | TAIL_PARALLEL: a whole function moves to core 1, not a
//   loop split. Everything the helper reads must already be written when it starts.
#if TAIL_PARALLEL
static std::thread g_filter_thr;
#endif

// The translation-filter update, as a callable so it can run on either thread.
// Pure heap: no XRT, no shared state with the scale filter. AP_FILTER is written
// from whichever thread runs it — a distinct slot from the scale path's
// AP_SCALE_*, so the two never touch the same counter.
static std::vector<int16_t> *g_filter_scratch_p = nullptr;
static void filter_update_work(void)
{
    AP_T(AP_FILTER,
         mosse::filter_update_quantize(g_filter, g_F_all.data(),
// @thesis subsec:aktualizacjaFiltra | A-03 | THE TRAINING TARGET: G is centred at the MEASURED
//   displacement, not at (0,0). Centring it teaches the filter that an off-target patch peaks
//   at zero shift.
                                       g_target_shift.data(),
                                       mosse::DEFAULT_ETA,
                                       g_energy, mosse::DEFAULT_EPS_REL,
                                       g_h_scratch, g_filter_scratch_p->data(),
                                       &g_q15_scale, &g_q15_max));
}

// Build H from the current filter state and push it to the device.
// Reports the Q1.15 scale and the peak magnitude: a spiky filter that leaves the
// response far below the cint16 rails shows up here and nowhere else.
// H already quantised into `scratch`: convert to the col-FFT layout and push.
// Split out of publish_filter() so the fused update+quantise path can share it —
// the layout conversion and the sync are the same either way, and duplicating
// them is how the two paths would drift.
static void publish_packed(xrt::bo &filter_bo, const std::vector<int16_t> &scratch,
                           float scale, float max_abs)
{
    pack_filter(scratch.data(), filter_bo.map<int16_t *>());
    filter_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // VP1. NOT carried by track.csv, so at VERBOSITY=0 this number is only
    // visible via the report_cint16("H(q15)") line next to it, which prints
    // unconditionally whenever a bin rails — the failure this would warn about.
    VP1("  filter: Q1.15 scale %.4g, max|H| %.4g\n", (double)scale, (double)max_abs);
}

// Quantise from the current filter state and push. Frame 0's path: filter_init()
// has just run, so there is no update to fuse the quantiser into.
static void publish_filter(xrt::bo &filter_bo, std::vector<int16_t> &scratch)
{
    float scale = 0.0f, max_abs = 0.0f;
    mosse::filter_quantize_q15(g_filter, g_energy, mosse::DEFAULT_EPS_REL,
                               scratch.data(), &scale, &max_abs);
    publish_packed(filter_bo, scratch, scale, max_abs);
}

// H before the Q1.15 scaling, handed to filter_update_quantize() so the 2 MB
// allocation happens once rather than per frame. Private to the fused path.


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
// @thesis subsec:zrodlaObrazu | N-12 | The synthetic scene: static background with dirty-rect
//   restore. It repeats to the LSB outside the target, which is a test-bench property, not a
//   tracker one.
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

// ONE rect type, shared with scene_colour. It used to be declared here with the
// same four fields, which is how two vocabularies for the same thing start.
using DirtyRect = scene::Rect;
static DirtyRect g_dirty;

// EVERY LUMA WRITE MUST BE COLOURISED, so a second rect accumulates the union of
// everything written this frame. g_dirty cannot serve: scene_restore CLEARS it,
// and the restored region is itself a write that must be re-colourised. So this
// one is fed from both the restore and every mark_dirty, and is cleared only by
// the colourise pass just before the push.
//
// The invariant to preserve when adding a scene function: if it writes luma, its
// rect must reach scene_touch(). Miss one and the device sees last frame's
// colour there — which would look like a plausible tracking result, not a bug.
static DirtyRect g_touched;

static inline void scene_touch(int r0, int c0, int r1, int c1)
{
    scene::rect_union(g_touched, r0, c0, r1, c1);
}

static void scene_init(int rows, int cols)
{
    g_background.resize((size_t)rows * cols);
    fill_background(g_background.data(), rows, cols);
    g_dirty = DirtyRect{};                       // frame starts equal to background
    g_touched = DirtyRect{};
}

// -----------------------------------------------------------------------
// @thesis subsec:zrodlaObrazu | N-12 | Background pan: the instrument that refuted
//   background lock as the explanation. It measurably decorrelates the background and changed
//   the tracker not at all.
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
    // The restored region is a luma write like any other.
    scene_touch(r0, c0, r1, c1);
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

// -----------------------------------------------------------------------
// Colourise — luma scene -> the interleaved RGB buffer the device reads
// -----------------------------------------------------------------------
// ONLY THE TOUCHED RECT. The frame is 2.07 M pixels and colourising all of it
// every frame would cost more than the push it feeds; the scene machinery
// already tracks what changed, and outside that rect the RGB buffer is still
// correct from the startup pass. This is the same argument that makes the
// dirty-rect restore cheap, applied one stage later.
//
// FRAME_RGB_MODE picks what "colour" means for the SYNTHETIC bench. Neither
// choice is a claim about real video — the VOT frame source supplies its own
// colour and ignores this entirely.
//
//   0  REPLICATE. All three planes carry luma. This is the hardware analogue of
//      the offline `rgb-lum` control arm: 27 taps, RGB plumbing, no colour. If
//      an RGB run beats gray here, the win is bookkeeping and not chroma.
//   1  TINT (default). A fixed per-plane gain, so the scene has real chromatic
//      contrast for the JOINT normalisation to preserve. The gains are applied
//      in Q8 and the target rect gets a different, warmer set than the
//      background, which is what makes the target distinguishable in chroma and
//      not only in luma.
//
// The gains are deliberately mild (0.75x .. 1.25x): a saturating tint would clip
// at 255 and hand Stage A a flat plane, which is the one outcome that would make
// the RGB path look broken for a reason that is not the RGB path.
#ifndef FRAME_RGB_MODE
#  define FRAME_RGB_MODE 1
#endif
static_assert(FRAME_RGB_MODE == scene::MODE_REPLICATE ||
              FRAME_RGB_MODE == scene::MODE_TINT,
              "FRAME_RGB_MODE must be 0 (replicate) or 1 (tint)");

#if CONV_IN_CH == 3
// Target rect for the tint, set once per frame before the colourise pass.
static DirtyRect g_tint_target;

// Thin wrappers over scene_colour, which owns the colour rule, the clipping and
// the verifier. Nothing here reimplements any of it: the tracker's job is to
// supply its globals and the frame geometry.
static inline void scene_colourise(int rows, int cols)
{
    scene::colourise(g_frame_host.data(), g_scene_luma.data(),
                     rows, cols, CONV_IN_CH, FRAME_RGB_MODE,
                     g_tint_target, g_touched);
}

// SCENE_VERIFY: re-expand the WHOLE frame and compare.
//
// The incremental pass is correct only if EVERY luma write reached
// scene_touch(). That invariant is invisible: miss one and the device reads last
// frame's colour there, the tracker still produces a peak, and the run looks
// like a slightly worse result rather than a bug. This turns it into an abort
// with coordinates.
//
// O(frame) per frame, so it is opt-in. Run it for a few frames after any change
// to a scene function, then turn it off — a debugging instrument, not a per-run
// cost. `make test_scene` covers the same invariant natively, including the
// missed-touch case, so this is the on-board backstop rather than the only net.
#ifndef SCENE_VERIFY
#  define SCENE_VERIFY 0
#endif
#if SCENE_VERIFY
static std::vector<uint8_t> g_colour_ref;
static void scene_verify(int rows, int cols, int frame)
{
    size_t first = 0;
    const size_t bad = scene::verify(g_frame_host.data(), g_scene_luma.data(),
                                     rows, cols, CONV_IN_CH, FRAME_RGB_MODE,
                                     g_tint_target, g_colour_ref, &first);
    if (!bad) return;
    const size_t px = first / CONV_IN_CH;
    printf("FATAL [scene] frame %d: incremental colourise disagrees with a full "
           "pass on %zu of %zu bytes, first at (r=%zu, c=%zu, plane=%zu).\n"
           "  A luma write did not reach scene_touch(). See the invariant note "
           "on g_touched and in scene_colour.h.\n",
           frame, bad, FRAME_BYTES, px / (size_t)cols, px % (size_t)cols,
           first % CONV_IN_CH);
    fflush(stdout);
    abort();
}
#else
static inline void scene_verify(int, int, int) {}
#endif
#else
static inline void scene_colourise(int, int) { /* gray: g_frame_host IS the scene */ }
static inline void scene_verify(int, int, int) {}
#endif

static void scene_mark_dirty(int r0, int c0, int r1, int c1)
{
    scene_touch(r0, c0, r1, c1);
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
// @thesis subsec:zrodlaObrazu | N-12 | The scripted trajectory and size envelope: absolute ground
//   truth, so drift becomes measurable where the legacy scheme made err=0 px self-fulfilling.
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
#if CONV_IN_CH == 3
    // The same rect is what the tint treats as "target", so the drawn object
    // carries a different hue from the background it sits on. Set here rather
    // than at the call site so the two can never disagree about where the
    // target is — they are the same two lines of geometry.
    g_tint_target = DirtyRect{r0, (int)std::floor(tc - 2.0 * sw) - 1,
                              r1, (int)std::ceil (tc + 8.0 * sw) + 1};
#endif

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
// @thesis subsec:petlaSterowania | A-01 | The per-frame orchestration loop: the APU drives
//   every GMIO port, per channel, and owns the whole control path.
// Main tracking loop
// -----------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin>\n", argv[0]);
        return EXIT_FAILURE;
    }

#if !FRAME_SOURCE_VOT
    // FRAME_SOURCE=synth MUST reproduce today's behaviour exactly, and the
    // runtime-geometry substitution above is only safe if these hold. Checked
    // rather than argued: it is four comparisons once per run, and the argument
    // is exactly the kind that reads as obviously true right up until someone
    // adds a knob that sets one of them. Not a static_assert — g_frame_rows and
    // friends are mutable globals precisely so the vot arm can move them.
    if (g_frame_rows != FRAME_ROWS || g_frame_cols != FRAME_COLS ||
        g_frame_bytes != FRAME_BYTES || g_run_frames != ITER_CNT) {
        fprintf(stderr, "FRAME_SOURCE=synth but the runtime geometry moved: "
                "%dx%d (%zu B), %d frames\n",
                g_frame_rows, g_frame_cols, g_frame_bytes, g_run_frames);
        return EXIT_FAILURE;
    }
#endif

#if FRAME_SOURCE_VOT
    // BEFORE the device is opened. Staging is a manifest parse and one blob read
    // (up to 1.27 GB for flamingo1), all of it host-side and none of it needing
    // hardware — so a wrong sequence name, an out-of-range job or a truncated
    // blob costs a message instead of an xclbin load and a graph start.
    if (!vot_parse_args(argc, argv)) return EXIT_FAILURE;
    if (!vot_stage())                return EXIT_FAILURE;
#endif

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
#if CMUL_SPLIT_ACCUM
    xrt::aie::buffer gm_accum_in    (device, uuid, "gmio_accum_in");
#endif
#if !MEMTILE_TRANSPOSE
    xrt::aie::buffer gm_fft_row_out (device, uuid, "gmio_fft_row_out");
    xrt::aie::buffer gm_fft_col_in  (device, uuid, "gmio_fft_col_in");
#endif
    xrt::aie::buffer gm_fft_col_out (device, uuid, "gmio_fft_col_out");
    xrt::aie::buffer gm_cmul_in     (device, uuid, "gmio_cmul_in");
    xrt::aie::buffer gm_accum_out   (device, uuid, "gmio_accum_out");
    xrt::aie::buffer gm_ifft_row_in (device, uuid, "gmio_ifft_row_in");
#if !MEMTILE_TRANSPOSE
    xrt::aie::buffer gm_ifft_row_out(device, uuid, "gmio_ifft_row_out");
    xrt::aie::buffer gm_ifft_col_in (device, uuid, "gmio_ifft_col_in");
#endif
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
// @thesis sec:przeplywDanych | A-04 | The DDR accumulator BO. Never compute on a BO mapping:
//   reads run at 696 MB/s against 7359 on the heap.
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
    // layer0_weights.bin ships mean_prev = 0, so frame 0 ran with mean_prev=0
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

            // THE LAYOUT TAG IS CHECKED HERE, ONCE, BEFORE ANY BYTE IS READ.
            // A CONV_IN_CH=3 file read by a CONV_IN_CH=1 build takes out_shift
            // out of the G plane and bias_acc out of G/B taps: no crash, no
            // warning, sixteen plausible channels and a meaningless tracker.
            // Byte 63 is 0 in files exported before the tag existed, which were
            // all grayscale.
            const int tag = (int)w[CONV_W_OFF_TAG];
            if (tag != CONV_IN_CH && !(tag == 0 && CONV_IN_CH == 1)) {
                printf("FATAL: %s ch%d has layout tag %d, this build is "
                       "CONV_IN_CH=%d. Re-run: make weights CONV_IN_CH=%d\n",
                       WEIGHTS_FILE, ch, tag, CONV_IN_CH, CONV_IN_CH);
                return 1;
            }

            // The SECOND tag, for the same reason. The tap count is a product of
            // CONV_IN_CH and CONV_KSIZE, so one tag cannot separate a 3x3 RGB
            // bank (27 taps) from a 7x7 gray one (49) — and once a second kernel
            // size exists the pair is the only thing that pins the layout.
            // Byte PAD-2 is 0 in files exported before this tag, which were all
            // 3x3.
            const int tag_k = (int)w[CONV_W_OFF_TAG_K];
            if (tag_k != CONV_KSIZE && !(tag_k == 0 && CONV_KSIZE == 3)) {
                printf("FATAL: %s ch%d has kernel-size tag %d, this build is "
                       "CONV_KSIZE=%d. Re-run: make weights CONV_KSIZE=%d\n",
                       WEIGHTS_FILE, ch, tag_k, CONV_KSIZE, CONV_KSIZE);
                return 1;
            }

            int32_t bias;
            memcpy(&bias, w + CONV_W_OFF_BIAS, sizeof(int32_t));
            const int shift = (int)w[CONV_W_OFF_SHIFT];
            const int32_t seed = bias >> shift;
            memcpy(w + CONV_W_OFF_MEAN, &seed, sizeof(int32_t));
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
#if FRAME_SOURCE_VOT
    // The job's init box, i.e. the dataset's groundtruth at the anchor frame.
    // TARGET_H/TARGET_W are inert here: the box is the dataset's, and with it the
    // ROI, sigma and the DSST template size, all of which are DERIVED from the
    // box and so follow it without further plumbing.
    box.row = g_vot.job.init_box.row;
    box.col = g_vot.job.init_box.col;
    box.h   = g_vot.job.init_box.h;
    box.w   = g_vot.job.init_box.w;
#else
    box.row = g_frame_rows / 2.0;
    box.col = g_frame_cols / 2.0;
    box.h   = (double)TARGET_H;
    box.w   = (double)TARGET_W;
#endif

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
    double box_h0 = box.h, box_w0 = box.w;   // reset per run — see run_reset()

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
    // Sized here, not on first use: filter_update_quantize() would otherwise do a
    // 2 MB allocation inside frame 1's timed body and put it in AP_FILTER.
    g_h_scratch.resize((size_t)N_CHANNELS * PATCH_ELEMS);
    std::vector<int16_t> filter_scratch((size_t)N_CHANNELS * PATCH_ELEMS * 2);
    // filter_scratch is a local; filter_update_work() may run on the second core
    // and needs it by address. Bound once, immediately after it exists.
    g_filter_scratch_p = &filter_scratch;

    if (g_run_frames < 2)
        printf("WARNING: this run is %d frame(s). Frame 0 is consumed by filter\n"
               "         initialisation, so a single-frame run cannot test\n"
               "         localisation. Build with ITER_CNT=2 or more (synth), or\n"
               "         pick an anchor that is not the last frame (vot).\n",
               g_run_frames);
    printf("filter: sigma=%.1f eta=%.3f H_SHIFT=%d — frame 0 initialises, "
           "frame 1+ tracks\n",
           (double)mosse::DEFAULT_SIGMA, (double)mosse::DEFAULT_ETA, CMUL_H_SHIFT);
    // MAKE THE LOG SELF-DESCRIBING. runs/.last_cfg recorded ITER_CNT=500 for a
    // run that executed exactly 200 frames, and neither the frame count nor the
    // console level appeared anywhere in the log itself — so which of the two
    // was stale could not be settled from the artifact. A log that cannot state
    // its own configuration is one reflash away from the stale-card trap.
    // The frame SOURCE and the frame COUNT are both on this line, and the count
    // is the one the loop will actually run (g_run_frames), not the ITER_CNT the
    // build was configured with -- at FRAME_SOURCE=vot those differ by design,
    // and a log that reports the configured number rather than the executed one
    // is the exact artifact runs/.last_cfg was.
    printf("run: source=%s  frames=%d (ITER_CNT=%d)  VERBOSITY=%d  DUMP_BUFFERS=%d  "
           "shift %d-%d-%d  N_CHANNELS=%d  %dx%d  frame %dx%d\n",
           FRAME_SOURCE_VOT ? "vot" : "synth",
           g_run_frames, ITER_CNT, VERBOSITY, DUMP_BUFFERS,
           FFT_SHIFT_CFG, IFFT_ROW_SHIFT_CFG, IFFT_COL_SHIFT_CFG,
           N_CHANNELS, PATCH_ROWS, PATCH_COLS, g_frame_rows, g_frame_cols);
    fflush(stdout);

    // Tracked position, kept as ints because roi_crop takes integer coordinates.
    // They mirror box.row/box.col, which stay the authoritative state.
    int pos_row = (int)llround(box.row);
    int pos_col = (int)llround(box.col);

    // Coast state — see HOLD_COAST at the top. `coast_scale` is the decay factor
    // for the CURRENT hold run and is reset to 1.0 by every accepted frame, so a
    // long hold fades to a freeze and the next accept restores full velocity.
    // Zero-initialised, so the first hold before any accepted frame (which can
    // only be frame 0's, and frame 0 initialises rather than detects) coasts by
    // nothing at all rather than by an undefined velocity.
    mosse::CoastState coast;
    (void)coast;                                 // unused at HOLD_COAST=0

    // Where the target was actually injected, for the IoU report.
    mosse::TargetBox truth = box;

    // Anchor for the scripted trajectory. Fixed for the run, so the path is a
    // closed curve about the INITIAL centre and not about wherever the tracker
    // has wandered to.
    const double traj_row0 = box.row, traj_col0 = box.col;

    // Background is generated ONCE — see scene_init(). Regenerating it per frame
    // costs ~0.6-1.2 s on the A72, which would be 30-90x the whole pipeline.
#if !FRAME_SOURCE_VOT
    scene_init(g_frame_rows, g_frame_cols);
#endif

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
    // CROP_ROWS/COLS, not PATCH_ROWS/COLS. Identical while CONV_STRIDE == 1,
    // which is every arm shipped to date; at stride 2 roi_crop must resample the
    // ROI to twice the feature map on each axis or conv2d starves.
    crop_ip.set_static_args(frame_bo, (uint32_t)g_frame_rows, (uint32_t)g_frame_cols,
                            (uint32_t)CROP_ROWS, (uint32_t)CROP_COLS);
    printf("[roi_crop] USER-MANAGED (xrt::ip) launch path, CU driven directly; "
           "KDS completion bypassed. Constructed in %.3f ms\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - _rc_t0).count());
#else
    xrt::run crop_run(crop);
    crop_run.set_arg(0, frame_bo);
    crop_run.set_arg(2, (uint32_t)g_frame_rows);
    crop_run.set_arg(3, (uint32_t)g_frame_cols);
    crop_run.set_arg(8, (uint32_t)CROP_ROWS);   // the CROP, not the feature map
    crop_run.set_arg(9, (uint32_t)CROP_COLS);
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
    // FRAME_COLS * CONV_IN_CH, not FRAME_COLS: camera_capture zero-fills
    // rows*cols BYTES, and at 3 planes the buffer is three times as wide. The
    // seed below overwrites all of it either way, so this only matters for the
    // probe's own claim to be filling the buffer it says it fills.
    rc_control_cu_probe(cam, frame_bo, FRAME_ROWS, FRAME_COLS * CONV_IN_CH);

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
#if FRAME_SOURCE_VOT
    // NOTHING TO SEED. The seeding above exists because the synthetic scene only
    // ever writes its dirty rect, so the rest of the frame would be whatever the
    // probe left; a VOT frame is memcpy'd WHOLE every frame, so the first push
    // covers the buffer completely. The buffers are still sized here, at the
    // SEQUENCE's geometry rather than the maximum, because that is what the push
    // copies and what scene_luma() indexes with a g_frame_cols stride.
    {
        g_frame_host.assign(g_frame_bytes, 0);
#if CONV_IN_CH == 3
        g_scene_luma.assign((size_t)g_frame_rows * g_frame_cols, 0);
#endif
        printf("[vot] frame buffers sized at %dx%d x%d = %zu B (frame_bo is "
               "allocated at %dx%d)\n",
               g_frame_rows, g_frame_cols, CONV_IN_CH, g_frame_bytes,
               FRAME_ROWS, FRAME_COLS);
        fflush(stdout);
    }
#else
    {
#if CONV_IN_CH == 3
        // Luma scene first, then colourise the WHOLE frame once. The per-frame
        // pass only covers the touched rect, so everything outside it must be
        // correct from here — this is the pass that makes that true.
        g_scene_luma.assign(g_background.begin(), g_background.end());
        g_frame_host.assign(FRAME_BYTES, 0);
        g_tint_target = DirtyRect{};              // no target drawn yet
        scene_touch(0, 0, g_frame_rows - 1, g_frame_cols - 1);
        scene_colourise(g_frame_rows, g_frame_cols);
#else
        g_frame_host.assign(g_background.begin(), g_background.end());
#endif
        memcpy(frame_bo.map<uint8_t *>(), g_frame_host.data(), FRAME_BYTES);
        frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        printf("[scene] frame buffer seeded with the generated background "
               "(%zu B, %d plane%s). Before this the pipeline read an unwritten "
               "buffer outside the dirty rect — see the note here.\n",
               FRAME_BYTES, CONV_IN_CH, CONV_IN_CH == 1 ? "" : "s");
        fflush(stdout);
    }
#endif  // FRAME_SOURCE_VOT

    // How many runs this process executes. One at FRAME_SOURCE=synth, one per
    // entry of the job list at FRAME_SOURCE=vot (which may name the same job
    // twice — that is the determinism test).
#if FRAME_SOURCE_VOT
    const size_t n_jobs = g_vot.job_list.size();
#else
    const size_t n_jobs = 1;
#endif

#if FRAME_SOURCE_VOT
    // EVERY piece of state that outlives a run. The list is the deliverable of
    // Phase 3; each entry is here because leaving it out is silent:
    //
    //   g_filter        run B would start from A's trained filter
    //   filter_bo       ...and the DEVICE would still hold A's H on frame 0
    //   scale           the template size is DERIVED from the init box, so
    //                   reusing it mis-sizes the DSST template for the new box
    //   g_target/sigma  G is built from the box; a new box needs a new G
    //   mean_prev       missing this makes Stage B1 inert on the one frame the
    //                   filter trains from, and the ch16 response rails flat
    //   g_energy        filter_quantize_q15 reads it; stale values mis-scale H
    //   g_target_shift  the exact variable the first TAIL_PARALLEL attempt read
    //                   stale, taking mean IoU 0.9188 -> 0.4794
    //                   (runs/run_0821_1706.log, claim P-05)
    //   coast           NEW 2026-08-25: run B would coast on A's velocity
    //   counters        a run that inherits them reports a plausible summary
    //                   for a run that did not happen
    auto run_reset = [&](size_t ji) -> bool {
        g_vot.job_index = g_vot.job_list[ji];
        if (g_vot.job_index < 0 ||
            g_vot.job_index >= (int)g_vot.manifest.jobs.size()) {
            fprintf(stderr, "[vot] job %d out of range\n", g_vot.job_index);
            return false;
        }
        g_vot.job = g_vot.manifest.jobs[g_vot.job_index];
        if (g_vot.job.init_box.empty()) {
            fprintf(stderr, "[vot] job %d has an empty init box\n", g_vot.job_index);
            return false;
        }
        g_vot.order = vot::job_order(g_vot.job, g_vot.manifest.frames);
        g_run_frames = (int)g_vot.order.size();
        if (g_vot.max_frames > 0 && g_vot.max_frames < g_run_frames)
            g_run_frames = g_vot.max_frames;
        // THE STREAM'S PER-JOB RE-ARM, and it belongs here for the same reason
        // every other line of run_reset() does: this list is what the prefetcher
        // walks, a job that inherited the previous job's list would stream real
        // frames in the wrong order, and nothing downstream can tell the
        // difference — it is the backward-run-in-sequence-order failure wearing
        // a new hat. g_run_frames, not order.size(), so --vot-max-frames does
        // not prefetch frames the run will never ask for.
        if (g_vot.streamed) {
            g_vot.stream.begin_run(g_vot.order, (size_t)g_run_frames);
#if CONV_IN_CH == 3
            g_vot.stream_luma.begin_run(g_vot.order, (size_t)g_run_frames);
#endif
        }
        g_vot.traj.begin(g_vot.job);
        g_vot_empty_gt = 0;

        // --- the tracker's own state ---
        box.row = g_vot.job.init_box.row;  box.col = g_vot.job.init_box.col;
        box.h   = g_vot.job.init_box.h;    box.w   = g_vot.job.init_box.w;
        box_h0  = box.h;                   box_w0  = box.w;
        pos_row = (int)llround(box.row);   pos_col = (int)llround(box.col);
        truth   = box;
        if (RESET_MUTANT != 4) coast = mosse::CoastState{};

        // G is derived from the box, so both sigma and the target spectrum move
        // with it. Rebuilt exactly as at startup rather than scaled, because the
        // startup path is the one that is known to be right.
        const mosse::RoiGeometry r0 =
            mosse::roi_for(box, mosse::DEFAULT_PADDING, PATCH_ROWS, PATCH_COLS);
        mosse::sigma_for(box, r0, &sigma_r, &sigma_c);
        mosse::gaussian_target_spectrum(g_target.data(), PATCH_ROWS, PATCH_COLS,
                                        sigma_r, sigma_c, 0, 0);

        if (RESET_MUTANT != 5) {
            mosse::scale_filter_config(scale, mosse::DEFAULT_SCALE_N,
                                       mosse::DEFAULT_SCALE_STEP,
                                       box.h, box.w, (float)SCALE_SIGMA_FACTOR);
            scale_sample.assign((size_t)(scale.enabled() ? scale.sample_elems() : 1),
                                mosse::cfloat{});
        }

        if (RESET_MUTANT != 3)
            g_filter = mosse::FilterState{};    // exactly the startup condition
        std::fill(g_target_shift.begin(), g_target_shift.end(), mosse::cfloat{});
        for (int ch = 0; ch < N_CHANNELS; ++ch) g_energy[ch] = 0.0;

        // THE DEVICE holds state too. Without this, frame 0 of run B correlates
        // against run A's trained H — and frame 0 is the frame run B trains
        // FROM, so the damage is in the initialisation, where it is hardest to
        // see. Same memset as at startup.
        if (RESET_MUTANT != 2) {
            memset(filter_bo.map<void *>(), 0, FILTER_BYTES * N_CHANNELS);
            filter_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Re-seed mean_prev from the weight buffer's own bias/shift fields, then
        // push. Idempotent: bias_acc and out_shift are never written during a
        // run, so this recomputes the same seed the startup path produced.
        if (RESET_MUTANT != 1) {
            uint8_t *wmap = weights_bo.map<uint8_t *>();
            for (int ch = 0; ch < N_CHANNELS; ++ch) {
                uint8_t *w = wmap + (size_t)ch * WEIGHT_CH_BYTES;
                int32_t bias;
                memcpy(&bias, w + CONV_W_OFF_BIAS, sizeof(int32_t));
                const int32_t seed = bias >> (int)w[CONV_W_OFF_SHIFT];
                memcpy(w + CONV_W_OFF_MEAN, &seed, sizeof(int32_t));
                g_mean_prev[ch] = seed;
            }
            weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // --- per-run reporting counters ---
        g_det_hash   = 1469598103934665603ULL;   // FNV-1a offset basis
        g_csv_job    = g_vot.job_index;
        g_csv_anchor = g_vot.job.anchor;
        gate_reset_run();
        trk_eval = trk_ok = trk_lost = 0;
        trk_iou_sum = 0.0; trk_iou_min = 1.0;
        trk_cerr_sum = 0.0; trk_cerr_max = 0.0;
        scale_n_eval = scale_n_accept = scale_n_hold = 0;
        for (int i = 0; i < 8; ++i) scale_reason_n[i] = 0;
        scale_conf_min_seen = 1e300; scale_conf_max_seen = -1e300;

        printf("\n========== RUN %zu/%zu: %s job %d, anchor %d %s, %d frames, "
               "init box %.0fx%.0f at (%.0f,%.0f) ==========\n",
               ji + 1, n_jobs, g_vot.manifest.sequence.c_str(), g_vot.job_index,
               g_vot.job.anchor, g_vot.job.forward ? "forward" : "BACKWARD",
               g_run_frames, box.h, box.w, box.row, box.col);
        fflush(stdout);
        return true;
    };
#endif  // FRAME_SOURCE_VOT

    // ================= THE RUN LOOP (multi-start) =====================
    // One iteration per JOB. At FRAME_SOURCE=synth there is exactly one, and
    // run_reset() below is a no-op on the first pass, so this arm executes the
    // same sequence of operations it always has.
    //
    // EVERY piece of state that survives a run is reset here, and the list is
    // not obvious — each entry has caused a documented failure somewhere in this
    // project. See docs/thesis/evidence/phase3.md. The determinism test (job A, job B, job A
    // in one process, A's two trajectories byte-identical) is the instrument
    // that can actually see a miss; nothing else here can.
    for (size_t job_i = 0; job_i < n_jobs; ++job_i) {
#if FRAME_SOURCE_VOT
    if (!run_reset(job_i)) return EXIT_FAILURE;
#endif

    for (int frame = 0; frame < g_run_frames; ++frame) {

        dma_reset_frame();
        rc_reset_frame();
        ap_reset_frame();      // zeroes the slots AND starts the frame-body clock
        g_fdiag = FrameDiag{};  // rails must be per-frame, not cumulative

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
#if FRAME_SOURCE_VOT
        // ---- THE FRAME SEAM, VOT ARM ----------------------------------
        // The whole synthetic scene block below is replaced by two memcpys.
        // Everything the generator existed to do — background, trajectory,
        // target injection, occluder, noise, pan — is inert here: the frames are
        // whatever the dataset holds, and the groundtruth comes from the
        // manifest instead of from what the host drew.
        {
            const bool init_frame = !g_filter.initialized;
            const int  src = g_vot.order[(size_t)frame];   // dataset frame index

            // TWO copies, blob -> g_frame_host -> frame_bo, and the first one is
            // deliberate. Repointing scene_luma() at the blob would save it, and
            // would put the colourise path, the dirty-rect invariant and
            // scale_extract's luma pointer on a buffer nothing else in this file
            // expects to be read-only. Heap-to-heap runs at 7359 MB/s: 0.04 ms
            // for a 640x480 frame, against a refactor with no measurable gain.
            // THE FETCH IS SPLIT FROM THE COPY. On the resident path the two
            // are the same instruction and the fetch costs nothing; on the
            // streamed path the fetch can BLOCK and the copy cannot, so folding
            // them would put NFS latency in a slot named "scene gen" and leave
            // the only per-frame cost this feature adds invisible. AP_SCENE
            // therefore keeps meaning exactly what it meant before on both.
            const uint8_t *vot_px = nullptr;
            AP_T(AP_VOT_STAGE, vot_px = vot_frame(frame));
            if (!vot_px) { vot_report_frame_failure(frame); return EXIT_FAILURE; }
            AP_T(AP_SCENE, memcpy(g_frame_host.data(), vot_px, g_frame_bytes));
#if CONV_IN_CH == 3
            // The interleaved frame went to g_frame_host above; the scale
            // filter reads THIS. No colourise pass on the VOT arm -- the blob is
            // already interleaved, so scene_colourise() would overwrite a
            // correct colour frame with a tint of the luma plane. (It is only
            // called in the synth branch; this comment is here because the
            // asymmetry is the kind that gets "fixed" by symmetry later.)
            const uint8_t *vot_lu = nullptr;
            AP_T(AP_VOT_STAGE, vot_lu = vot_luma_frame(frame));
            if (!vot_lu) { vot_report_frame_failure(frame); return EXIT_FAILURE; }
            AP_T(AP_SCENE, memcpy(g_scene_luma.data(), vot_lu,
                                  (size_t)g_frame_rows * g_frame_cols));
#endif

            // Groundtruth for the IoU line and track.csv. REPORTING ONLY — it
            // never steers the tracker, which is the entire point of the board
            // knowing nothing about failure detection.
            const vot::Box &g = g_vot.manifest.groundtruth[(size_t)src];
            truth.row = g.row; truth.col = g.col;
            truth.h   = g.h;   truth.w   = g.w;
            if (g.empty()) ++g_vot_empty_gt;

            AP_T(AP_FRAME_PUSH,
                 memcpy(frame_bo.map<uint8_t *>(), g_frame_host.data(),
                        g_frame_bytes));
            AP_T(AP_FRAME_SYNC,
                 frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE));

            if (init_frame)
                VP1("Frame %d: [INIT] %s frame %d (anchor), box %.0fx%.0f at "
                    "(%.0f,%.0f)\n", frame, g_vot.manifest.sequence.c_str(), src,
                    box.h, box.w, box.row, box.col);
            else
                VP1("Frame %d: [vot] %s frame %d%s\n", frame,
                    g_vot.manifest.sequence.c_str(), src,
                    g.empty() ? "  (groundtruth EMPTY)" : "");
            fflush(stdout);
        }
#else
        {
            // The scene functions are all single-plane; scene_colourise()
            // below expands what they wrote into g_frame_host.
            uint8_t *frame_ptr = scene_luma();
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
                inject_checkerboard_frame(frame_ptr, g_frame_rows, g_frame_cols,
                                          OCCLUDE_SQUARE);
            } else {
                // Asymmetric structured target, not a single-pixel impulse: an
                // impulse is symmetric, and a symmetric training patch makes a
                // transposed pack_filter() and a wrong conjugation both invisible.
                // See inject_target_frame().
                AP_T(AP_SCENE, inject_target_frame(frame_ptr, g_frame_rows, g_frame_cols,
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
                scene_add_noise(frame_ptr, g_frame_rows, g_frame_cols,
                                nr - 4, nc - 4,
                                nr + roi.roi_h + 3, nc + roi.roi_w + 3);
            }
            // Push the host scene to the device, then flush. The memcpy is the
            // price of keeping the authority on the heap; the probe puts it at
            // ~600 us for 2 MB, against the ~13 ms of scattered uncached reads it
            // removes from scale_extract.
            // LAST luma write to FIRST device byte. Everything that draws into
            // the scene has run by here — restore, target, occluder, noise — so
            // the touched rect is complete and this is the only place the two
            // representations are guaranteed to agree.
            AP_T(AP_COLOURISE, scene_colourise(g_frame_rows, g_frame_cols));
            scene_verify(g_frame_rows, g_frame_cols, frame);   // no-op unless SCENE_VERIFY
            AP_T(AP_FRAME_PUSH,
                 memcpy(frame_bo.map<uint8_t *>(), g_frame_host.data(),
                        g_frame_bytes));
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
#endif  // FRAME_SOURCE_VOT — the frame seam

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

        // ---- roi_crop launch, hoisted so it can be issued a channel ahead ----
        //
        // Sets this channel's geometry, starts the CU, and feeds the weights the
        // resulting patch needs. The weights feed lives HERE and not at the call
        // site because the ordering between the two is load-bearing: conv2d
        // consumes one weights buffer per firing and a firing cannot complete
        // until it has read its share of the PatchIn stream, so sending
        // CONV_INVOCATIONS of them with blocking waits BEFORE starting the crop
        // deadlocks on the second — the host waits for the AIE to free a buffer,
        // the AIE waits for patch data nobody has sent. Keeping start and feed in
        // one function makes that order impossible to get wrong from outside.
#if RC_PIPELINE
        auto crop_launch = [&](int c) {
            RC_T(RC_ARGS, crop_set_args((uint32_t)roi_row, (uint32_t)roi_col,
                                        (uint32_t)roi.roi_h, (uint32_t)roi.roi_w,
                                        c));
            rc_tl_begin();
            RC_T(RC_START, crop_start());
            rc_tl_mark(TL_START);
            for (int k = 0; k < CONV_INVOCATIONS; ++k) {
                DMA_TX(DMA_WEIGHTS,
                    gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                     WEIGHT_CH_BYTES, c * WEIGHT_CH_BYTES));
                DMA_T(DMA_WEIGHTS, gm_weights.wait());
            }
        };

        // PRIME THE PIPELINE. Channel 0's crop is launched before the loop; from
        // then on each iteration launches ch+1 immediately after retiring ch, so
        // roi_crop's PL execution runs concurrently with the host's APU work for
        // the channel that just finished.
        crop_launch(0);
#endif

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
#if !RC_PIPELINE
            RC_T(RC_ARGS, crop_set_args((uint32_t)roi_row, (uint32_t)roi_col,
                                        (uint32_t)roi.roi_h, (uint32_t)roi.roi_w,
                                        ch));
            rc_tl_begin();
            RC_T(RC_START, crop_start());
            rc_tl_mark(TL_START);
#if MEMTILE_TRANSPOSE
            for (int k = 0; k < CONV_INVOCATIONS; ++k) {
                DMA_TX(DMA_WEIGHTS,
                    gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                     WEIGHT_CH_BYTES, ch * WEIGHT_CH_BYTES));
                DMA_T(DMA_WEIGHTS, gm_weights.wait());
            }
#else
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
#endif
#endif  // !RC_PIPELINE — pipelined builds launched this channel last iteration
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

#if RC_PIPELINE
            // LAUNCH THE NEXT CHANNEL HERE — as early as the CU allows.
            //
            // roi_crop is a single user-managed CU, so ch+1 cannot start until
            // ch's ap_done, which the poll above just confirmed. Everything below
            // in this iteration — the cmul feed, the col-FFT and accumulator
            // drains, and ~0.4 ms/channel of APU work — now runs while ch+1's
            // crop executes in the PL.
            //
            // WHY THIS IS WORTH 5 ms. roi_crop's own execution was never free; it
            // was hidden behind the row-FFT drain, and CLAUDE.md said so in
            // advance: "the CU finishes inside the drain loop. If the drain ever
            // shrinks the spin will start spinning for real." The memtile deleted
            // that drain outright and the spin duly started spinning — RC_POLL
            // went 0.067 -> 5.196 ms/frame, 325 us/channel (apu_stages.csv, claim
            // P-05). This puts the CU back
            // under cover, without the DDR round trip that used to provide it.
            //
            // Ordering safety: conv2d cannot run ahead of the memtile's
            // ping-pong, so the graph self-limits to ONE channel of lookahead —
            // fft_rows(ch+1) fills the second buffer while fft_cols(ch) drains
            // the first, and fft_rows(ch+2) blocks until the first is free. The
            // host cannot outrun that even if it wanted to.
            if (ch + 1 < N_CHANNELS) crop_launch(ch + 1);
#endif

            // Stage B: measure this channel's window-weighted feature mean and
            // spectral energy BEFORE transposing, while the row-major layout
            // still puts each row's DC bin at stride PATCH_COLS.
            // ONE bulk read of row_bo, then mean, energy AND the transpose all
            // run on the heap copy — three passes that used to pay the uncached
            // load individually.
#if !MEMTILE_TRANSPOSE
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

                // Feed mean_now back as the next frame's mean_prev.
                uint8_t *wb = weights_bo.map<uint8_t *>() + ch * WEIGHT_CH_BYTES;
                memcpy(wb + CONV_W_OFF_MEAN, &mean_now, sizeof(int32_t));
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
#endif  // !MEMTILE_TRANSPOSE — no row-FFT output on the host, nothing to
        // transpose, and Stage B1's mean / Stage B3's energy move to the F_ch
        // block further down.

#if CMUL_SPLIT_ACCUM
            // NO PACKING. accum_prev has its own port, so H comes straight from
            // filter_bo and the running sum straight from accum_bo — the 2 MB of
            // BO->BO memcpy that `cmul packing` measured (2.871 ms/frame — see
            // docs/thesis/results/apu_stages.csv, claim P-05 — which
            // is exactly 2 MB at the probe's 696 MB/s uncached BO read rate) is
            // simply not performed.
            //
            // IN-PLACE ON accum_bo IS SAFE, and it is worth saying why rather
            // than discovering it. cmul reads chunk c of accum_bo and later
            // writes chunk c of accum_bo; chunks are distinct addresses and the
            // read of chunk c always precedes the write of chunk c, so whichever
            // order the two DMAs interleave, no chunk is read after it has been
            // updated this channel. A separate double buffer would reintroduce
            // the copy this change exists to remove.
            //
            // ch0 must see a ZERO accumulator — the packed path memset the accum
            // half for ch0 and this replaces that.
            if (ch == 0) {
                const auto _cp0 = std::chrono::steady_clock::now();
                memset(accum_bo.map<void *>(), 0, ACCUM_BYTES);
                g_ap_us[AP_CMUL_PACK] += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - _cp0).count();
                ++g_ap_n[AP_CMUL_PACK];
            }
#else
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
#endif  // CMUL_SPLIT_ACCUM

            // Feed transposed data to col-FFT + combined [filter|accum] to cmul_accum
#if !MEMTILE_TRANSPOSE
            DMA_TX(DMA_FFT_COL_IN,
                gm_fft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0));
#endif
#if CMUL_SPLIT_ACCUM
            DMA_TX(DMA_CMUL_IN,
                gm_cmul_in.async(filter_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                 FFT_BYTES, (size_t)ch * FFT_BYTES));
            DMA_TX(DMA_ACCUM_IN,
                gm_accum_in.async(accum_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                  ACCUM_BYTES, 0));
#else
            DMA_TX(DMA_CMUL_IN,
                gm_cmul_in.async(cmul_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, CMUL_IN_BYTES, 0));
#endif

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
#if CMUL_ACCUM_MEMTILE
            // THE ACCUMULATOR IS NO LONGER DRAINED PER CHUNK. memTileAcc buffers
            // the whole channel's accumulator on-chip, so one transfer of
            // ACCUM_BYTES replaces COL_CHUNKS of COL_CHUNK_BYTES: 256 tx/frame
            // becomes 16. gmio_accum_out's cost is per-transaction, not per-byte
            // — 14.4 us for 64 B against 22.8 us for 128 KB on the DMA probe — so
            // this is the whole ~4 ms.
            //
            // THE DEADLOCK THE OLD LOOP AVOIDED IS GONE WITH IT, and this is the
            // load-bearing half of the argument. The comment above says draining
            // must not be sequenced after the input waits, because cmul stalls on
            // a full OUTPUT WINDOW, which stalls the col FFT, which stalls the
            // input DMAs. cmul's output window now drains into the memory tile at
            // AIE speed rather than waiting on a host-driven DDR transfer, so it
            // cannot be the thing that blocks: the memtile holds a full channel
            // (64 KB) and ping-pongs, so channel k's drain to DDR overlaps
            // channel k+1's accumulation.
            //
            // The F_ch tap still drains per chunk — it is fed directly by the col
            // FFT, not through the tile, and its window granularity is unchanged.
            // Decoupling the two is the entire point: FFT_COL_WS used to move
            // them together, and at WS=32 the accumulator WON (4.42 -> 1.25 ms)
            // while the tap lost catastrophically (4.57 -> 17.07).
            DMA_TX(DMA_ACCUM_OUT,
                gm_accum_out.async(accum_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                   ACCUM_BYTES, 0));
            for (int k = 0; k < COL_CHUNKS; ++k) {
                DMA_TX(DMA_FFT_COL_OUT,
                    gm_fft_col_out.async(fcol_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                         COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
                DMA_T(DMA_FFT_COL_OUT, gm_fft_col_out.wait());
            }
            DMA_T(DMA_ACCUM_OUT, gm_accum_out.wait());
            VP2("[ch %d] accum_out (1 x %zu B) + F_ch (%d x %zu B) received\n",
                ch, ACCUM_BYTES, COL_CHUNKS, COL_CHUNK_BYTES);
#else
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
#endif

            // Stash this channel's spectrum for the filter update after the loop.
            // Converted out of the col-FFT layout here so all the filter maths
            // works in one consistent row-major order.
            // SPLIT, 2026-08-20. These were one slot and its 16.59 ms could not
            // be apportioned between the transfer and the conversion — the same
            // mistake as DMA_T timing async and wait together.
            AP_T(AP_FCOL_SYNC, fcol_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE));
            AP_T(AP_BO_STAGE,
                 memcpy(g_stage_c.data(), fcol_bo.map<void *>(), FFT_BYTES));
            //
            // F_ch is the filter's only input. If it is wrong, everything
            // downstream is wrong for a reason that has nothing to do with the
            // filter maths — so it is checked before any later number is
            // trusted. The check now rides ALONG WITH the unpack rather than
            // rescanning the same 64 KB afterwards; its cost therefore lands in
            // AP_UNPACK, and AP_DIAG_SCAN reports 3 calls/frame, not 4.
            Cint16Scan fch_scan;
            AP_T(AP_UNPACK,
                 unpack_spectrum((const int16_t *)g_stage_c.data(),
                                 g_F_all.data() + (size_t)ch * PATCH_ELEMS,
                                 ch == 0 ? &fch_scan : nullptr));

#if MEMTILE_TRANSPOSE
            // Stage B1's mean and Stage B3's energy, derived from F_ch because
            // the row-FFT output no longer reaches the host. See the derivation
            // above measure_window_mean_fch(). Same slot as before so the
            // CUMULATIVE table stays comparable — it is the same work on a
            // different buffer, and it reads g_stage_c, which the unpack just
            // brought onto the heap, so it costs no DMA.
            {
                const int16_t *fc = (const int16_t *)g_stage_c.data();
                const auto _wm0 = std::chrono::steady_clock::now();
                const int32_t mean_now = measure_window_mean_fch(fc, g_mean_prev[ch]);
                residual_mean[ch] = (double)(mean_now - g_mean_prev[ch]);
                g_mean_prev[ch]   = mean_now;
                g_energy[ch]      = measure_energy_fch(fc);

                // Feed mean_now back as the NEXT frame's mean_prev.
                // weights_bo is synced once after the channel loop, unchanged.
                uint8_t *wb = weights_bo.map<uint8_t *>() + ch * WEIGHT_CH_BYTES;
                memcpy(wb + CONV_W_OFF_MEAN, &mean_now, sizeof(int32_t));

                g_ap_us[AP_WINMEAN] += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - _wm0).count();
                ++g_ap_n[AP_WINMEAN];
            }
#endif
            if (ch == 0) {
                report_cint16_scan("F_ch", fch_scan, PATCH_COLS, "colFFT");
                dump_buffer("F_ch", frame, g_stage_c.data(), FFT_BYTES);
            }

#if !MEMTILE_TRANSPOSE
            DMA_T(DMA_FFT_COL_IN, gm_fft_col_in.wait());
#endif
            VP2("[ch %d] fft_col_in sent\n", ch);
            DMA_T(DMA_CMUL_IN, gm_cmul_in.wait());
#if CMUL_SPLIT_ACCUM
            DMA_T(DMA_ACCUM_IN, gm_accum_in.wait());
#endif
            VP2("[ch %d] cmul_in sent\n", ch);
        }

        // Stage B2: cancel the residual pre-window mean on the accumulated
        // spectrum. 9 bins × N_CHANNELS complex MACs — 144 operations for the
        // whole frame. Must run before the IFFT consumes accum_bo.
        AP_T(AP_B2, apply_dc_correction(accum_bo.map<int16_t *>(),
                                        filter_bo.map<int16_t *>(),
                                        residual_mean));

        // Push the updated mean_prev values (written at CONV_W_OFF_MEAN of each
        // channel's weight buffer above) so the NEXT frame's conv2d sees them.
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // The accumulated spectrum, after B2, as the IFFT will see it. Same
        // col-FFT layout as F_ch. `rails > 0` here is the saturation the shift
        // budget exists to prevent.
        // Staged to the heap FIRST. This was the one diagnostic still scanning a
        // BO mapping element by element, and those mappings are write-combining:
        // the startup probe measures reads out of them at 294 MB/s against
        // 1707 on the heap, so a 64 KB scan in place costs ~223 us where a bulk
        // memcpy costs 94 and the scan of the copy ~25. Same fix, same reason as
        // the copy pattern applied everywhere else — see CLAUDE.md, "never
        // compute on a BO mapping". g_stage_a is free here; the IFFT transpose
        // below overwrites it anyway.
        AP_T(AP_BO_STAGE,
             memcpy(g_stage_a.data(), accum_bo.map<void *>(), ACCUM_BYTES));
        report_cint16("accum", (const int16_t *)g_stage_a.data(),
                      PATCH_ROWS, PATCH_COLS, "colFFT");
        dump_buffer("accum", frame, accum_bo.map<void *>(), ACCUM_BYTES);

        // 3. IFFT: APU feeds accumulated spectrum to IFFT row input
        VP2("[ifft] START\n");
        DMA_TX(DMA_IFFT_ROW_IN,
            gm_ifft_row_in.async(accum_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, ACCUM_BYTES, 0));
        // Drain per invocation, and before waiting on the input — see the
        // accum_out loop above for why the input wait cannot come first.
#if !MEMTILE_TRANSPOSE
        for (int k = 0; k < ROW_CHUNKS; ++k) {
            DMA_TX(DMA_IFFT_ROW_OUT,
                gm_ifft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                      ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES));
            DMA_T(DMA_IFFT_ROW_OUT, gm_ifft_row_out.wait());
        }
#endif
        DMA_T(DMA_IFFT_ROW_IN, gm_ifft_row_in.wait());
        VP2("[ifft] rows done (%d x %zu B)\n", ROW_CHUNKS, ROW_CHUNK_BYTES);

#if !MEMTILE_TRANSPOSE
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
#endif
        for (int k = 0; k < COL_CHUNKS; ++k) {
            DMA_TX(DMA_RESPONSE,
                gm_response.async(resp_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                  COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
            DMA_T(DMA_RESPONSE, gm_response.wait());
        }
#if !MEMTILE_TRANSPOSE
        DMA_T(DMA_IFFT_COL_IN, gm_ifft_col_in.wait());
#endif
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
        // The response is the pipeline's output and the last thing that exists
        // before an argmax throws its precision away, so it is what the digest
        // is taken over. Hashed from the HEAP copy, never the BO mapping.
        AP_T(AP_DET_HASH, det_hash_bytes(g_stage_c.data(), RESP_BYTES));
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
#if HOLD_COAST
                // The velocity is the displacement THIS accepted frame applied,
                // i.e. one frame's motion, because the position advances every
                // frame on both paths. See coast_observe() for why a coasted
                // frame must never feed this.
                mosse::coast_observe(coast, dr_frame, dc_frame);
#endif
            }
#if HOLD_COAST
            else {
                double cr = 0.0, cc = 0.0;
                if (mosse::coast_step(coast, (double)(COAST_DECAY), &cr, &cc)) {
                    // COAST. Moves the position only; the box size, the filter
                    // and the scale model stay frozen exactly as before.
                    box.row += cr;
                    box.col += cc;
                    pos_row  = (int)llround(box.row);
                    pos_col  = (int)llround(box.col);
                    // Printed at EVERY verbosity, like the other anomalies: a
                    // coast moves the search window on NO evidence, and a run of
                    // them is the shape of a loss in progress.
                    printf("Frame %d: [coast] hold + (%.2f,%.2f) px of the last "
                           "measured velocity (%.2f,%.2f) -> pos (%d,%d)\n",
                           frame, cr, cc, coast.vr, coast.vc, pos_row, pos_col);
                }
            }
#endif

            const bool ok = (peak != 0 && dr == exp_dr && dc == exp_dc);
            VP1("Frame %d: displacement (%d,%d) bins = (%.2f,%.2f) frame px %s "
                "pos (%d,%d)  peak=%ld  [%s]\n",
                   frame, dr, dc, dr_frame, dc_frame,
                   gate.accept ? "→"
                               : (HOLD_COAST ? "HELD, pos COASTS" : "HELD, pos stays"),
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

            // ---- TRANSLATION FILTER UPDATE, ON THE SECOND CORE ----------
            //
            // Launched HERE, before the scale filter, because from this point
            // the two are independent: both need the PSR result and the updated
            // position, neither needs the other. State is disjoint —
            // filter_update_quantize touches g_filter / g_F_all / g_energy /
            // g_target_shift / g_h_scratch / filter_scratch, the scale path
            // touches box / scale / scale_sample / the scale counters. The one
            // frame-scope value they might have shared, sigma_r/sigma_c, is
            // computed once before the frame loop and never written here.
            //
            // WHY THIS SHAPE AND NOT THE OBVIOUS ONE. The first cut moved the
            // SCALE work onto the helper instead. That breaks a real ordering
            // constraint: the IoU/centre-error accumulation a few lines below
            // reads `box` AFTER the scale update, so with the scale work in
            // flight it would have scored a stale box and quietly reported the
            // wrong tracking numbers. Launching the FILTER side instead needs no
            // code motion at all, which is worth more here than the extra ms.
            //
            // PUBLISH DELIBERATELY STAYS ON THE MAIN THREAD. pack_filter +
            // filter_bo.sync are the only XRT calls in the tail, and keeping XRT
            // single-threaded is a discipline this design should not trade for
            // 1.9 ms. So the overlap is scale (2.62) against the update (4.72),
            // and the tail goes 9.25 -> ~6.63 ms.
            //
            // Frame 0 never gets here (evaluate == g_filter.initialized), so the
            // bootstrap runs filter_init serially exactly as before.
#if TAIL_PARALLEL
            if (gate.accept) {
                // BUILD THE TRAINING TARGET BEFORE THE LAUNCH, NOT AFTER.
                //
                // filter_update_quantize CONSUMES g_target_shift, and in the
                // serial code this call sits down in the `else if (gate.accept)`
                // branch — i.e. after the scale filter and after the join. The
                // first cut left it there and started the helper anyway, so the
                // helper trained on the PREVIOUS frame's target. Measured
                // (runs/run_0821_1706.log): mean IoU 0.9188 -> 0.4794, centre
                // error 1.37 -> 13.01 px, 22 scale HOLDs where there had been 0,
                // diverging from frame 2.
                //
                // The state review that preceded that run asked "does the scale
                // path TOUCH anything the filter path touches?" and answered no,
                // correctly. It failed to ask the other question: "is everything
                // the helper READS already written when it starts?" For a thread
                // launch both must hold, and only the second one catches this.
                mosse::gaussian_target_spectrum(g_target_shift.data(),
                                                PATCH_ROWS, PATCH_COLS,
                                                sigma_r, sigma_c,
                                                psr_abs.dr, psr_abs.dc);
                g_ap_filter_at_launch = g_ap_us[AP_FILTER];
                g_filter_thr = std::thread(filter_update_work);
            }
#endif

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
                // LUMA, not the interleaved buffer: the DSST scale filter is an
                // intensity template and keeping it on luma means RGB costs it
                // no recalibration at all.
                const uint8_t *fp = scene_luma();
                mosse::ScaleResult sr{};
                // Split: scale_extract reads frame_bo DIRECTLY (33 crops), so it
                // pays the uncached read; scale_detect is pure heap. 13.86 ms was
                // one number and could not be apportioned.
                AP_T(AP_SCALE_EXTRACT,
                     mosse::scale_extract(scale, fp, g_frame_rows, g_frame_cols,
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
                                      mosse::DEFAULT_SCALE_MAX_REL,
                                      mosse::DEFAULT_SCALE_MAX_STEP);
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
            if (scale_evaluated && scale_gate_dec.accept) {
                // THE SECOND EXTRACTION IS REDUNDANT AND IT WAS 4.73 ms.
                //
                // Both calls crop at the SAME centre (box.row, box.col); only the
                // size differs, and it differs by exactly the accepted factor
                // a^idx. Level n of an extraction at box*a^idx is therefore
                // box*a^(idx+n) — level idx+n of the extraction already sitting
                // in scale_sample. Training on that sample against a target
                // shifted by idx levels is the same update, by the shift theorem,
                // and the shift is exact rather than approximate.
                //
                // This is the manoeuvre filter_update() already makes with
                // (psr_abs.dr, psr_abs.dc), one axis down, and with the same sign
                // convention: G is centred at +idx, the level the detector
                // reported. See scale_update_shifted() in mosse_filter.h for the
                // derivation and what the |idx| edge levels cost — on hardware
                // the detector proposed only -1/0/+1 across 199 frames, and at
                // idx = 0 (174 of them) the two paths are bitwise identical.
                AP_T(AP_SCALE_MODEL,
                     mosse::scale_update_shifted(scale, scale_sample.data(),
                                                 scale_idx,
                                                 mosse::DEFAULT_SCALE_ETA));
            } else {
                // The gate never ran — frame 0's bootstrap, or a frame where the
                // scale filter is not yet trained. There is no detection sample
                // to reuse and no idx to shift by, so extract as before.
                // LUMA, not the interleaved buffer: the DSST scale filter is an
                // intensity template and keeping it on luma means RGB costs it
                // no recalibration at all.
                const uint8_t *fp = scene_luma();
                AP_T(AP_SCALE_EXTRACT,
                     mosse::scale_extract(scale, fp, g_frame_rows, g_frame_cols,
                                          box.row, box.col, box.h, box.w,
                                          scale_sample.data()));
                AP_T(AP_SCALE_MODEL,
                     mosse::scale_update(scale, scale_sample.data(),
                                         mosse::DEFAULT_SCALE_ETA));
            }
        }

        bool published = false;
#if FILTER_MASK_STAT
        // Cleared EVERY frame, before anything can set it. filter_init()'s path
        // does not fill g_h_scratch and a held frame does not re-form H, so
        // without this the column would repeat the last computed value and read
        // as a filter that never changes.
        g_mask_ebox = -1.0;
#endif
#if TAIL_PARALLEL
        // JOIN POINT for the translation-filter update started before the scale
        // block. See the launch site for the dependency argument.
        //
        // The overlap is measured here rather than estimated. Let H be the
        // helper's own elapsed time (the AP_FILTER delta across the region) and
        // W the time this thread then spends blocked in join(). The region's
        // WALL cost is (main's own work + W), while the slots credit
        // (main's own work + H) — so the double-count is exactly H - W, which is
        // min(helper, main) however the two happen to line up.
        if (g_filter_thr.joinable()) {
            const auto   _j0 = std::chrono::steady_clock::now();
            g_filter_thr.join();
            const double _jw = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - _j0).count();
            // H IS READ AFTER THE JOIN, NOT BEFORE. The helper writes
            // g_ap_us[AP_FILTER] when it finishes, so sampling it before join()
            // reads a value that has not been written yet — it came back 0 every
            // frame, the `ovl > 0` guard suppressed both new lines, and
            // runs/run_0821_1720.log still printed the impossible -1.492 ms
            // residual. join() is the synchronisation point that makes the
            // helper's write visible; reading before it was also a data race.
            //
            // The same read-before-write ordering mistake as the g_target_shift
            // bug two builds ago, made in the code written to explain that bug.
            const double _hf = g_ap_us[AP_FILTER] - g_ap_filter_at_launch;
            g_ap_overlap_us += (_hf > _jw) ? (_hf - _jw) : 0.0;
        }
#endif
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
#if !TAIL_PARALLEL
            mosse::gaussian_target_spectrum(g_target_shift.data(),
                                            PATCH_ROWS, PATCH_COLS,
                                            sigma_r, sigma_c,
                                            psr_abs.dr, psr_abs.dc);
#endif  // TAIL_PARALLEL builds fill g_target_shift BEFORE launching the helper
            //
            // FUSED with the quantiser. Unfused, A (2 MB at ch16) is streamed
            // four times a frame — written by the update, then read twice by
            // filter_quantize_q15, which recomputes the same 262144 divides on
            // both passes. filter_update_quantize() forms H in the pass that
            // writes A and keeps it, so the write-out is a scalar multiply.
            // Measured cost before the fusion (docs/thesis/results/apu_stages.csv,
            // claim P-01): 10.18 + 5.60 = 15.78 ms of a
            // 62.7 ms frame. BIT-IDENTICAL output, asserted by memcmp in
            // run_fusion_tests() at BOTH -O2 and -ffp-contract=fast — the first
            // cut was not, and only the contraction build could see it.
#if TAIL_PARALLEL
            // Already computed on core 1 while this thread ran the scale filter;
            // g_q15_scale/g_q15_max were filled there. Only the publish is left,
            // and it stays HERE because it is the one part that touches XRT.
#else
            filter_update_work();
#endif
            AP_T(AP_PUBLISH,
                 publish_packed(filter_bo, filter_scratch, g_q15_scale, g_q15_max));
            published = true;
#if FILTER_MASK_STAT
            // g_h_scratch holds H for this frame's publish — masked already, if
            // FILTER_MASK=1, because filter_update_quantize() projects BEFORE
            // the max scan. Read after the join, so the helper's writes are
            // visible; this is on the main thread and the helper is finished.
            //
            // The box is measured in PATCH BINS, from the same roi geometry the
            // crop used, so it tracks the DSST scale filter frame by frame
            // rather than assuming the initial size. A hardcoded box*2/128 is
            // the bug vot_detector_gain.py still has.
            //
            // SUBSAMPLED, and that is not a nicety. Sum|h|^2 is a SPATIAL
            // quantity read off a FREQUENCY-domain H, so the call carries an
            // inverse FFT per channel: 9.4 ms on an x86 -O3 host at ch16 /
            // 128x128, i.e. more than a whole 26 ms frame once it is on the
            // A72. Every frame would make the instrument cost more than the
            // tracker it is instrumenting, and frame time is a reported number.
            //
            // The schedule is the first FILTER_MASK_STAT_WARM frames of each
            // run (at-init and the early profile, which is where the mask is
            // theorised to act) plus every FILTER_MASK_STAT_EVERY-th frame
            // after. Unsampled frames log -1, the same value a held frame logs,
            // and every reader already excludes it.
            //
            // vot_mask_stat.py's PROFILE_FRAMES must stay inside WARM or land on
            // a multiple of EVERY, or its per-frame columns silently thin out.
            if (frame <= FILTER_MASK_STAT_WARM
                || (FILTER_MASK_STAT_EVERY > 0
                    && frame % FILTER_MASK_STAT_EVERY == 0)) {
                g_mask_ebox = mosse::filter_box_energy_fraction(
                    g_h_scratch.data(), N_CHANNELS, PATCH_ROWS, PATCH_COLS,
                    (int)llround(mosse::target_h_in_patch(box, roi)),
                    (int)llround(mosse::target_w_in_patch(box, roi)));
                VP1("  filter: energy in target box %.4f\n", g_mask_ebox);
            }
#endif
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

            // The HOST side of the state, in full precision. box carries the
            // position and the scale filter's output; psr carries the gate's
            // input. Together with the response hash above this covers the
            // datapath and the decisions taken from it.
            {
                const double st[6] = { box.row, box.col, box.h, box.w,
                                       gate.psr, (double)psr_abs.peak };
                AP_T(AP_DET_HASH, det_hash_bytes(st, sizeof st));
            }

#if FRAME_SOURCE_VOT
            // The result, accumulated in RAM. Nothing touches the NFS mount
            // inside the frame loop — the whole trajectory is one write at the
            // end of the run (phase0a.md).
            //
            // Index 0 is the anchor, and it is written as Special(INITIALIZATION)
            // rather than as the init box: EVERY VOT trajectory begins with a
            // special code, and a file that begins with a rectangle is read back
            // without complaint and scored one frame out of step (phase0b.md).
            {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - g_ap_frame_t0).count();
                if (frame == 0) {
                    g_vot.traj.push_init(ms);
                } else {
                    vot::Box b;
                    b.row = box.row; b.col = box.col; b.h = box.h; b.w = box.w;
                    g_vot.traj.push(b, ms);
                }
            }
#endif

            // VERBOSITY 0: the ONE line per frame that keeps the frame period
            // measurable. ~45 B, ~4 ms at 115200, 4% of the ~87 ms GMIO floor.
            // Everything on it is already in track.csv — the point is not the
            // data, it is having a timestampable per-frame marker for
            // `picocom | ts`, which is how the frame time was measured in the
            // first place. Printing nothing at all would have deleted that.
            // PROGRESS_EVERY thins this line without silencing it; frame 0 and
            // the last frame always print. At the default 1 the condition folds
            // away and the output is byte-identical to before the knob existed.
#if PROGRESS_EVERY > 1
            const bool progress_due = (frame % PROGRESS_EVERY) == 0 ||
                                      (frame == g_run_frames - 1);
#else
            const bool progress_due = true;   // default arm: every frame
#endif
            if (VERBOSITY < 1 && progress_due)
                printf("f%d d%d,%d psr%.0f iou%.2f r00 %.2f\n",
                       frame, psr_abs.dr, psr_abs.dc, gate.psr, iou,
                       resp00_over_peak);
            // Printed at EVERY verbosity while the sweep is armed: if a deep
            // queue deadlocks, this is the record of which depths already worked.
#if !MEMTILE_TRANSPOSE
            // The sweep instruments gmio_fft_row_out, which no longer exists on
            // the memtile path; it would print a row of zeros every frame.
            if (FFT_DRAIN_DEPTH == 0)
                printf("  [drain] frame %d depth %d: gmio_fft_row_out %.2f ms "
                       "(%lu tx), weights %.2f ms\n",
                       frame, drain_depth_for_frame(frame),
                       g_dma[DMA_FFT_ROW_OUT].us / 1000.0,
                       g_dma[DMA_FFT_ROW_OUT].calls,
                       g_dma[DMA_WEIGHTS].us / 1000.0);
#endif
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

#if FRAME_SOURCE_VOT
    // ONE write per run, after the loop. Reported as its own step, and a failure
    // here is fatal to the run's meaning even though every frame tracked fine —
    // so it is announced rather than returned quietly.
    {
        // THE DETERMINISM TEST. If this job index has already run in this
        // process, its trajectory must come back byte-identical — that is the
        // only cheap instrument that can see state leaking across run_reset(),
        // and it needs no groundtruth to be meaningful. Compared in memory
        // because both runs write the same filename.
        const std::string traj_text = g_vot.traj.as_text();
        char hashline[64];
        snprintf(hashline, sizeof hashline, "\nstate %016llx",
                 (unsigned long long)g_det_hash);
        // The key is trajectory AND digest. Reported separately below, because
        // "same boxes, different datapath" is a distinct and more informative
        // verdict than either half alone.
        const std::string key = traj_text + hashline;
        printf("\n[vot] run state digest %016llx  (response + box + psr, every "
               "frame)\n", (unsigned long long)g_det_hash);
        auto prev = g_det_seen.find(g_vot.job_index);
        g_det_repeat = (prev != g_det_seen.end());
        if (prev == g_det_seen.end()) {
            g_det_seen.emplace(g_vot.job_index, key);
        } else if (prev->second == key) {
            printf("[vot] DETERMINISM: job %d re-run, trajectory AND state digest "
                   "IDENTICAL (%zu B) — no state leaked across run_reset()\n",
                   g_vot.job_index, traj_text.size());
        } else if (prev->second.compare(0, traj_text.size(), traj_text) == 0) {
            // The case RESET_MUTANT=1 produced on 2026-08-25 and the reason this
            // digest exists: identical boxes, different arithmetic. A leak too
            // small to move an argmax is still a leak, and on a longer sequence
            // or a different scene it would move one.
            printf("[vot] DETERMINISM FAILED: job %d re-run has the SAME "
                   "trajectory but a DIFFERENT state digest. State leaked across "
                   "run_reset() without moving a peak — see the [diag] lines.\n",
                   g_vot.job_index);
            g_det_failures++;
        } else {
            size_t at = 0;
            while (at < key.size() && at < prev->second.size() &&
                   key[at] == prev->second[at]) ++at;
            printf("\n[vot] DETERMINISM FAILED: job %d re-run differs at byte %zu "
                   "(%zu B vs %zu B). STATE LEAKED ACROSS run_reset().\n",
                   g_vot.job_index, at, prev->second.size(), key.size());
            g_det_failures++;
        }
    }
    {
        if (g_vot.max_frames > 0 && g_run_frames < g_vot.job.length) {
            printf("\n[vot] trajectory NOT written: this was a %d-frame bring-up "
                   "of a %d-frame job.\n", g_run_frames, g_vot.job.length);
        } else if (g_det_repeat) {
            // A repeat exists to be COMPARED, not to overwrite the first run's
            // result. On a determinism failure the file on disk is then the
            // first run's, which is the one the comparison reports against.
            printf("\n[vot] job %d already written this process; the re-run was "
                   "compared, not written\n", g_vot.job_index);
        } else {
            std::string err;
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = g_vot.traj.write(g_vot.results_dir,
                                             g_vot.manifest.sequence, err);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (ok)
                printf("\n[vot] wrote %s/%s_%08d.txt  (%zu regions, %.1f ms)\n",
                       g_vot.results_dir.c_str(), g_vot.manifest.sequence.c_str(),
                       g_vot.job.anchor, g_vot.traj.size(), ms);
            else
                fprintf(stderr, "\n[vot] TRAJECTORY NOT WRITTEN: %s\n", err.c_str());
        }
        // TWO INDEPENDENT INSTRUMENTS ON THE SAME WAIT, which is this project's
        // rule and not decoration: AP_VOT_STAGE times the at() CALL from the
        // frame loop, this is the reader's own accounting from inside. They
        // measure the same seconds by two routes, so a divergence means the
        // wait is not where the slot says it is -- and a prefetcher that is
        // secretly synchronous would show up here as the two agreeing at a
        // number far above zero, rather than as a plausible frame time.
        if (g_vot.streamed)
            printf("[vot] streamed frame wait, cumulative for this job: %.3f s "
                   "over %d frames (%.3f ms/frame)\n",
                   vot_stage_wait_seconds(), g_run_frames,
                   g_run_frames > 0 ? 1000.0 * vot_stage_wait_seconds() / g_run_frames
                                    : 0.0);
        if (g_vot_empty_gt > 0)
            printf("[vot] %d frame(s) of this run have an EMPTY groundtruth box. "
                   "The IoU figures below score them as 0; the toolkit's failure "
                   "rule ignores them, so the PC-side AR is the number of "
                   "record.\n", g_vot_empty_gt);
        fflush(stdout);
    }
#endif

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
    gate_report_run(g_run_frames);

#if !FRAME_SOURCE_VOT
    // Same instrument on the synthetic arm: one number that stands in for "does
    // this build track bit-identically to the last one". Compare it across two
    // logs instead of diffing per-frame lines by eye.
    printf("\n[run] state digest %016llx  (response + box + psr, every frame)\n",
           (unsigned long long)g_det_hash);
#endif

    g_frames_total += g_run_frames;
    }   // ===== end of the run loop =====================================

    csv_close();

#if FRAME_SOURCE_VOT
    if (g_det_seen.size() < g_vot.job_list.size()) {
        printf("\n[vot] DETERMINISM: %d failure(s) across %zu run(s)\n",
               g_det_failures, g_vot.job_list.size());
    }
    printf("\n[vot] %zu run(s), %d frames total\n",
           g_vot.job_list.size(), g_frames_total);
#endif

    // Cumulative GMIO cost across all frames. Printed before teardown because
    // gr.end(0) never returns (see the "host does not exit" note in CLAUDE.md) —
    // anything printed after it is lost.
    {
        double        tot_us = 0.0;
        unsigned long tot_n  = 0;
        double tot_wait = 0.0;
        printf("\n[dma] CUMULATIVE over %d frame(s):\n", g_frames_total);
        for (int i = 0; i < DMA_N; ++i) {
            if (!g_dma_total[i].calls) continue;
            printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx   wait %9.3f ms (%3.0f%%)\n",
                   g_dma_total[i].name, g_dma_total[i].calls,
                   g_dma_total[i].us / 1000.0,
                   g_dma_total[i].us / g_dma_total[i].calls,
                   g_dma_total[i].wait_us / 1000.0,
                   g_dma_total[i].us > 0.0
                       ? 100.0 * g_dma_total[i].wait_us / g_dma_total[i].us : 0.0);
            tot_us  += g_dma_total[i].us;
            tot_wait += g_dma_total[i].wait_us;
            tot_n   += g_dma_total[i].calls;
        }
        printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx   wait %9.3f ms (%3.0f%%)\n",
               "TOTAL", tot_n, tot_us / 1000.0, tot_n ? tot_us / tot_n : 0.0,
               tot_wait / 1000.0, tot_us > 0.0 ? 100.0 * tot_wait / tot_us : 0.0);
        printf("  per frame: %.0f tx, %.3f ms  (N_CHANNELS=%d, %dx%d)\n",
               (double)tot_n / g_frames_total, tot_us / 1000.0 / g_frames_total,
               N_CHANNELS, PATCH_ROWS, PATCH_COLS);
        // THE NUMBER THE THREADING DECISION RESTS ON.
        //
        // `wait` is the host blocked in gmio::wait() with nothing to do — CPU a
        // second A72 core could be spending on the per-channel unpack/mean work.
        // `async` is real host time (descriptor setup, the driver path) that a
        // second core cannot recover. Sizing the helper thread against the wrong
        // one of these is exactly the mistake the CMUL_ACCUM_MEMTILE attempt
        // made in the other direction.
        printf("  HOST BLOCKED in wait(): %.3f ms/frame of %.3f ms GMIO "
               "(%.0f%%); host-side async work %.3f ms/frame\n",
               tot_wait / 1000.0 / g_frames_total, tot_us / 1000.0 / g_frames_total,
               tot_us > 0.0 ? 100.0 * tot_wait / tot_us : 0.0,
               (tot_us - tot_wait) / 1000.0 / g_frames_total);
        fflush(stdout);

        // APU CUMULATIVE. This is the number to read, not the two per-frame
        // tables: frame 0 does filter_init rather than filter_update and never
        // runs the scale detector, so it is the least representative frame in
        // the run. Averaged over ITER_CNT, against the measured mean frame body.
        {
            const double mf = g_ap_run_us / g_frames_total;
            double apu = 0.0;
            for (int i = 0; i < AP_N; ++i) apu += g_ap_tot_us[i];
            double rc = 0.0;
            for (int i = 0; i < RC_N; ++i) rc += g_rc_us_total[i];
            printf("\n[apu] CUMULATIVE over %d frame(s), mean frame body %.2f ms:\n",
                   g_frames_total, mf / 1000.0);
            printf("  %-20s %9s %10s %9s %7s\n",
                   "stage", "calls/fr", "ms/frame", "us/call", "share");
            for (int i = 0; i < AP_N; ++i) {
                if (!g_ap_tot_n[i]) continue;
                printf("  %-20s %9.1f %10.3f %9.1f %6.1f%%\n",
                       g_ap_name[i], (double)g_ap_tot_n[i] / g_frames_total,
                       g_ap_tot_us[i] / 1000.0 / g_frames_total,
                       g_ap_tot_us[i] / (double)g_ap_tot_n[i],
                       100.0 * g_ap_tot_us[i] / g_ap_run_us);
            }
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- APU subtotal", "",
                   apu / 1000.0 / g_frames_total, "", 100.0 * apu / g_ap_run_us);
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- GMIO (DMA_T)", "",
                   tot_us / 1000.0 / g_frames_total, "", 100.0 * tot_us / g_ap_run_us);
            printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- roi_crop launch", "",
                   rc / 1000.0 / g_frames_total, "", 100.0 * rc / g_ap_run_us);
            // The slots sum to more wall time than the frame spent, by exactly
            // the overlap measured at the join. Report BOTH: the subtotal keeps
            // its old meaning so every figure recorded before threading stays
            // comparable, and the wall figure is the one the residual is
            // computed from.
            const double ovl = g_ap_overlap_tot_us;
            if (ovl > 0.0) {
                printf("  %-20s %9s %10.3f %9s %6.1f%%   <-- ran CONCURRENTLY on "
                       "core 1; the subtotal above counts it, the frame does not\n",
                       "-- of which OVERLAP", "", -ovl / 1000.0 / g_frames_total, "",
                       -100.0 * ovl / g_ap_run_us);
                printf("  %-20s %9s %10.3f %9s %6.1f%%\n", "-- APU wall", "",
                       (apu - ovl) / 1000.0 / g_frames_total, "",
                       100.0 * (apu - ovl) / g_ap_run_us);
            }
            const double resid = g_ap_run_us - (apu - ovl) - tot_us - rc;
            printf("  %-20s %9s %10.3f %9s %6.1f%%   <-- console, dumps, printf, "
                   "uninstrumented\n", "== UNATTRIBUTED", "",
                   resid / 1000.0 / g_frames_total, "", 100.0 * resid / g_ap_run_us);
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
