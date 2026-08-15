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

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_aie.h"
#include "experimental/xrt_aie.h"

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

// Per-frame breakdown. The number that decides whether the transposes have to
// move off the APU is the total, against 33 ms.
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
        g_dma_total[i].name   = g_dma[i].name;
        g_dma_total[i].calls += g_dma[i].calls;
        g_dma_total[i].us    += g_dma[i].us;
    }
    printf("  %-18s %6lu tx  %9.3f ms  %7.2f us/tx  = %.1f%% of a 33 ms frame\n",
           "TOTAL", tot_n, tot_us / 1000.0,
           tot_n ? tot_us / tot_n : 0.0, 100.0 * tot_us / 33000.0);
    printf("  NOTE: hw_emu wall time is not real hardware time. Treat the "
           "tx COUNT and the\n        per-port split as the real findings; the "
           "us/tx figure needs a `TARGET=hw` run.\n");
    fflush(stdout);
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// In-place 2-D matrix transpose via a temporary scratch buffer.
// elem_bytes must be 4 (cint16).
static void transpose_inplace(void *buf, int rows, int cols, size_t elem_bytes)
{
    // Allocate scratch (stack for 64 KB is too large; use heap).
    size_t total = (size_t)rows * cols * elem_bytes;
    std::vector<uint8_t> tmp(total);
    const uint8_t *src = static_cast<const uint8_t *>(buf);
    uint8_t       *dst = tmp.data();

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const uint8_t *s = src + (r * cols + c) * elem_bytes;
            uint8_t       *d = dst + (c * rows + r) * elem_bytes;
            memcpy(d, s, elem_bytes);
        }
    memcpy(buf, tmp.data(), total);
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
    printf("  [psr] at argmax|re| (%d,%d): peak %ld  sidelobe mu %.1f sd %.1f "
           "max %.1f  PSR %.2f (Bolme)  ratio %.2fx (aiesim metric)  n=%ld\n",
           a.dr, a.dc, a.peak, a.mean, a.sdev, a.side_max, a.psr, a.ratio, a.n_side);
    printf("  [psr] at argmax re  (%d,%d): peak %ld  sidelobe mu %.1f sd %.1f "
           "max %.1f  PSR %.2f (Bolme)  ratio %.2fx\n",
           s.dr, s.dc, s.peak, s.mean, s.sdev, s.side_max, s.psr, s.ratio);

    if (a.dr != s.dr || a.dc != s.dc)
        printf("  [psr] DISAGREE: the |real| scan and the signed max pick "
               "DIFFERENT peaks. The tracker acted on (%d,%d) with value %ld; "
               "the paper-literal peak is (%d,%d) with %ld.\n",
               a.dr, a.dc, a.peak, s.dr, s.dc, s.peak);
    else
        printf("  [psr] peak definitions agree.\n");

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
    printf("  [psr] verdict: %s\n", verdict);
    printf("  [psr]          (Bolme PSR only — his thresholds do NOT apply to the "
           "ratio column)\n");
    fflush(stdout);
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

    printf("  [gate] frame %d: %s  reason=%s  psr=%.2f  threshold=%.2f\n",
           frame, g.accept ? "ACCEPT" : "HOLD",
           mosse::gate_reason_tag(g.reason), g.psr, (double)g.threshold);
    printf("  [gate]         %s\n", mosse::gate_reason_why(g.reason));

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

// Peak magnitude and saturation count for a cint16 buffer. `rails > 0` means the
// stage clipped, which is the failure mode the shift budget exists to prevent.
// Indices are reported in the buffer's own layout — the caller says which.
static void report_cint16(const char *tag, const int16_t *b, int rows, int cols,
                          const char *layout)
{
    double max_m = -1.0;
    int    max_i = 0, rails = 0;
    for (int i = 0; i < rows * cols; ++i) {
        const double re = b[2 * i], im = b[2 * i + 1];
        const double m  = re * re + im * im;
        if (m > max_m) { max_m = m; max_i = i; }
        if (re >= 32767.0 || re <= -32768.0 || im >= 32767.0 || im <= -32768.0) ++rails;
    }
    printf("  [diag] %-9s max|.|=%7.0f at %s idx %d (%d,%d)  rails=%d\n",
           tag, sqrt(max_m), layout, max_i, max_i / cols, max_i % cols, rails);
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
// Target spectrum G — constant for the whole run, so it is built once at startup
// from the closed form rather than transformed per frame.
static std::vector<mosse::cfloat> g_target(PATCH_ELEMS);
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
    printf("  filter: Q1.15 scale %.4g, max|H| %.4g\n", (double)scale, (double)max_abs);
    fflush(stdout);
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
static void inject_target_frame(uint8_t *frame_buf, int rows, int cols,
                                int tr, int tc)
{
    constexpr uint8_t BACKGROUND = 40;
    constexpr uint8_t BAR_VALUE  = 220;
    constexpr uint8_t SPUR_VALUE = 150;

    memset(frame_buf, BACKGROUND, (size_t)rows * cols);

    for (int r = 0; r < rows; ++r) {
        const int dr = r - tr;
        if (dr < -5 || dr > 5) continue;          // bar and spur both fit in |dr| <= 5
        for (int c = 0; c < cols; ++c) {
            const int dc = c - tc;
            uint8_t v = 0;
            if (dc >= -2 && dc <= 2)                       v = BAR_VALUE;   // 11 x 5
            else if (dr >= 2 && dc >= 3 && dc <= 8)        v = SPUR_VALUE;  // one side only
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
    auto crop = xrt::kernel(device, uuid, "roi_crop:{roi_crop_0}");

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

    // Target spectrum: closed form, no FFT (see mosse_filter.h). CENTRED — a
    // target displaced by (dr,dc) must produce a peak at (dr,dc), so G itself
    // carries no offset. Constant for the whole run.
    mosse::gaussian_target_spectrum(g_target.data(), PATCH_ROWS, PATCH_COLS,
                                    mosse::DEFAULT_SIGMA, 0, 0);

    // Scratch for the row-major quantized filter, before pack_filter() converts
    // it to the col-FFT layout.
    std::vector<int16_t> filter_scratch((size_t)N_CHANNELS * PATCH_ELEMS * 2);

    if (ITER_CNT < 2)
        printf("WARNING: ITER_CNT=%d. Frame 0 is consumed by filter initialisation,\n"
               "         so a single-frame run cannot test localisation. Build with\n"
               "         ITER_CNT=2 or more for a meaningful result.\n", ITER_CNT);
    printf("filter: sigma=%.1f eta=%.3f H_SHIFT=%d — frame 0 initialises, "
           "frame 1+ tracks\n",
           (double)mosse::DEFAULT_SIGMA, (double)mosse::DEFAULT_ETA, CMUL_H_SHIFT);
    fflush(stdout);

    // Tracked position (centre of search patch in frame coordinates)
    int pos_row = FRAME_ROWS / 2;
    int pos_col = FRAME_COLS / 2;

    // ------------------------------------------------------------------
    // Per-frame tracking loop
    // ------------------------------------------------------------------
    for (int frame = 0; frame < ITER_CNT; ++frame) {

        dma_reset_frame();

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
            uint8_t *frame_ptr = frame_bo.map<uint8_t *>();
            // Keyed off the filter state, not off `frame == 0`. The two agree in
            // the current flow, but "is the filter trained yet" is the condition
            // that actually decides where the target must be, and the init branch
            // at the bottom of the loop uses the same one.
            const bool init_frame = !g_filter.initialized;
            int test_row = pos_row + (init_frame ? 0 : IMPULSE_DR);
            int test_col = pos_col + (init_frame ? 0 : IMPULSE_DC);

            // Occlusion test for the PSR gate. Guarded on !init_frame so bit 0 is
            // a no-op — see the OCCLUDE_MASK comment at the top of this file.
            occluded = !init_frame && (((OCCLUDE_MASK) >> frame) & 1);

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
                inject_target_frame(frame_ptr, FRAME_ROWS, FRAME_COLS,
                                    test_row, test_col);
            }
            frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // Flush host → device
            if (occluded)
                printf("Frame %d: [OCCLUDED] checkerboard (square %d) — no target "
                       "in frame. The gate MUST hold position and freeze the "
                       "filter.\n", frame, OCCLUDE_SQUARE);
            else if (init_frame)
                printf("Frame %d: [INIT] target CENTRED at (%d,%d)\n",
                       frame, test_row, test_col);
            else
                printf("Frame %d: target at pos+(%d,%d) = (%d,%d)\n",
                       frame, IMPULSE_DR, IMPULSE_DC, test_row, test_col);
            fflush(stdout);
        }

        // 2. Per-channel: conv2d + FFT2D + cmul_accum
        int roi_row = pos_row - PATCH_ROWS / 2;
        int roi_col = pos_col - PATCH_COLS / 2;

        // Stage B2: per-channel mean that conv2d did NOT remove this frame,
        // filled in as each channel's row-FFT comes back and applied once to the
        // accumulated spectrum after the loop.
        double residual_mean[N_CHANNELS] = {0.0};

        for (int ch = 0; ch < N_CHANNELS; ++ch) {

            printf("[ch %d] START\n", ch); fflush(stdout);

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
            // that to PATCH_ROWS × PATCH_COLS. They are fixed to the patch size
            // here because scale estimation is not implemented yet — once it is,
            // this is where the estimated target size goes.
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
            xrt::run crop_run(crop);
            crop_run.set_arg(0,  frame_bo);
            crop_run.set_arg(2,  (uint32_t)FRAME_ROWS);
            crop_run.set_arg(3,  (uint32_t)FRAME_COLS);
            crop_run.set_arg(4,  (uint32_t)roi_row);
            crop_run.set_arg(5,  (uint32_t)roi_col);
            crop_run.set_arg(6,  (uint32_t)PATCH_ROWS);   // roi_h
            crop_run.set_arg(7,  (uint32_t)PATCH_COLS);   // roi_w
            crop_run.set_arg(8,  (uint32_t)PATCH_ROWS);
            crop_run.set_arg(9,  (uint32_t)PATCH_COLS);
            crop_run.set_arg(10, (uint32_t)((ch == 0) ? 1 : 0));
            crop_run.start();

            // Feed one weight buffer per conv2d firing AND drain one row-FFT
            // window per firing, in the same loop.
            //
            // These MUST interleave. Draining after the weights loop deadlocks
            // (fft_rows fills its one armed output window, blocks conv2d, and the
            // weights queue freezes — the observed 6-of-64 stall). Draining before
            // it deadlocks the other way, waiting on data conv2d has not been fed.
            // The two counts are equal by construction (static_assert above), so
            // one loop serves both.
            for (int k = 0; k < CONV_INVOCATIONS; ++k) {
                // Arm this firing's drain before feeding the weights that trigger
                // it, so the output window is never the thing that blocks.
                DMA_TX(DMA_FFT_ROW_OUT,
                    gm_fft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                         ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES));
                DMA_TX(DMA_WEIGHTS,
                    gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                     WEIGHT_CH_BYTES, ch * WEIGHT_CH_BYTES));
                DMA_T(DMA_WEIGHTS,     gm_weights.wait());
                DMA_T(DMA_FFT_ROW_OUT, gm_fft_row_out.wait());
            }
            printf("[ch %d] weights sent + row-FFT drained (%d x %zu B)\n",
                   ch, CONV_INVOCATIONS, ROW_CHUNK_BYTES); fflush(stdout);

            crop_run.wait();
            printf("[ch %d] roi_crop done\n", ch); fflush(stdout);
            printf("[ch %d] fft_row_out received\n", ch); fflush(stdout);

            // Stage B: measure this channel's window-weighted feature mean and
            // spectral energy BEFORE transposing, while the row-major layout
            // still puts each row's DC bin at stride PATCH_COLS.
            {
                const int16_t *rf = row_bo.map<int16_t *>();
                const int32_t  mean_now = measure_window_mean(rf, g_mean_prev[ch]);
                // B2 needs what conv2d failed to remove this frame.
                residual_mean[ch] = (double)(mean_now - g_mean_prev[ch]);
                g_mean_prev[ch]   = mean_now;

                // B3: Parseval energy, for the filter scaling in the update step.
                double e = 0.0;
                for (int i = 0; i < PATCH_ELEMS; ++i)
                    e += (double)rf[2*i] * rf[2*i] + (double)rf[2*i+1] * rf[2*i+1];
                g_energy[ch] = e / (double)PATCH_ELEMS;

                // Feed mean_now back as the next frame's mean_prev (bytes 18:22).
                uint8_t *wb = weights_bo.map<uint8_t *>() + ch * WEIGHT_CH_BYTES;
                memcpy(wb + 18, &mean_now, sizeof(int32_t));
            }

            // APU: transpose row-FFT output in-place
            transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);
            printf("[ch %d] transpose done\n", ch); fflush(stdout);

            // Pack [filter_chunk_c | accum_chunk_c] into cmul_bo for all chunks.
            // For ch=0 the accum half is zeroed; for ch>0 it carries the running sum.
            {
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
            printf("[ch %d] accum_out + F_ch received (%d x %zu B)\n",
                   ch, COL_CHUNKS, COL_CHUNK_BYTES); fflush(stdout);

            // Stash this channel's spectrum for the filter update after the loop.
            // Converted out of the col-FFT layout here so all the filter maths
            // works in one consistent row-major order.
            fcol_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            unpack_spectrum(fcol_bo.map<int16_t *>(),
                            g_F_all.data() + (size_t)ch * PATCH_ELEMS);

            // F_ch is the filter's only input. If it is wrong, everything
            // downstream is wrong for a reason that has nothing to do with the
            // filter maths — so check it before trusting any later number.
            // Reported in the col-FFT layout it arrives in: idx = v*ROWS + u.
            if (ch == 0) {
                report_cint16("F_ch", fcol_bo.map<int16_t *>(),
                              PATCH_ROWS, PATCH_COLS, "colFFT");
                dump_buffer("F_ch", frame, fcol_bo.map<void *>(), FFT_BYTES);
            }

            DMA_T(DMA_FFT_COL_IN, gm_fft_col_in.wait());
            printf("[ch %d] fft_col_in sent\n", ch); fflush(stdout);
            DMA_T(DMA_CMUL_IN, gm_cmul_in.wait());
            printf("[ch %d] cmul_in sent\n", ch); fflush(stdout);
        }

        // Stage B2: cancel the residual pre-window mean on the accumulated
        // spectrum. 9 bins × N_CHANNELS complex MACs — 144 operations for the
        // whole frame. Must run before the IFFT consumes accum_bo.
        apply_dc_correction(accum_bo.map<int16_t *>(),
                            filter_bo.map<int16_t *>(),
                            residual_mean);

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
        printf("[ifft] START\n"); fflush(stdout);
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
        printf("[ifft] rows done (%d x %zu B)\n", ROW_CHUNKS, ROW_CHUNK_BYTES);
        fflush(stdout);

        // APU: transpose IFFT row output in-place
        transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);
        printf("[ifft] transpose done\n"); fflush(stdout);

        DMA_TX(DMA_IFFT_COL_IN,
            gm_ifft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0));
        for (int k = 0; k < COL_CHUNKS; ++k) {
            DMA_TX(DMA_RESPONSE,
                gm_response.async(resp_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                  COL_CHUNK_BYTES, k * COL_CHUNK_BYTES));
            DMA_T(DMA_RESPONSE, gm_response.wait());
        }
        DMA_T(DMA_IFFT_COL_IN, gm_ifft_col_in.wait());
        printf("[ifft] cols done → response received (%d x %zu B)\n",
               COL_CHUNKS, COL_CHUNK_BYTES); fflush(stdout);

        // Response diagnostics run on EVERY frame, including frame 0. Frame 0's
        // response is not a tracking result (the filter is zero, so it should be
        // ~0), but "is it actually zero" is worth knowing: a non-zero frame-0
        // response would mean the filter buffer is not what the host thinks.
        report_cint16("response", resp_bo.map<int16_t *>(),
                      PATCH_ROWS, PATCH_COLS, "rowmajor");
        dump_buffer("resp", frame, resp_bo.map<void *>(), RESP_BYTES);
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

        if (evaluate) {
            const int16_t *resp = resp_bo.map<int16_t *>();
            psr_abs = mosse::compute_psr(resp, PATCH_ROWS, PATCH_COLS, true);
            psr_sgn = mosse::compute_psr(resp, PATCH_ROWS, PATCH_COLS, false);
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
            printf("Frame %d: [INIT] response not evaluated — filter is being "
                   "trained from this frame\n", frame);
        } else {
            const int  dr   = psr_abs.dr, dc = psr_abs.dc;
            const long peak = psr_abs.peak;

            // THE POSITION IS ONLY UPDATED ON AN ACCEPTED FRAME. A gated frame's
            // peak is noise; following it walks the ROI off the target and the
            // target can then never re-enter the search window. Holding is what
            // makes Bolme's "reacquire when the appearance returns" work with no
            // reacquisition code — and it is also why the recovery frame's
            // expected displacement is still exactly (IMPULSE_DR, IMPULSE_DC),
            // so the check below stays meaningful across an occlusion.
            if (gate.accept) { pos_row += dr; pos_col += dc; }

            const bool ok = (peak != 0 && dr == IMPULSE_DR && dc == IMPULSE_DC);
            printf("Frame %d: displacement (%d,%d) %s pos (%d,%d)  peak=%ld  [%s]\n",
                   frame, dr, dc, gate.accept ? "→" : "HELD, pos stays",
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
            else if (!ok && !occluded && gate.accept)
                printf("       expected displacement (%d,%d)\n", IMPULSE_DR, IMPULSE_DC);
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
        bool published = false;
        if (!g_filter.initialized) {
            mosse::filter_init(g_filter, g_F_all.data(), g_target.data(),
                               N_CHANNELS, PATCH_ROWS, PATCH_COLS);
            printf("  filter: INITIALISED from frame %d (single patch, no affine "
                   "perturbations)\n", frame);
            publish_filter(filter_bo, filter_scratch);
            published = true;
        } else if (gate.accept) {
            mosse::filter_update(g_filter, g_F_all.data(), g_target.data(),
                                 mosse::DEFAULT_ETA);
            publish_filter(filter_bo, filter_scratch);
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

        dma_report_frame(frame);
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
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    gr.end(0);  // block until graph completes
    return EXIT_SUCCESS;
}
