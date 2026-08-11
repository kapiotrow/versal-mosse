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

// Software peak detection on the real part of the IFFT response map.
// Returns displacement relative to patch centre (range: [-rows/2, rows/2)).
//
// `resp` points at cint16 samples: {real, imag} interleaved, so sample i's real
// part is resp[2*i]. Indexing it as resp[i] would scan the real AND imaginary
// halves of only the first half of the map and report bogus peaks.
//
// Scans |real|, not real. Before Stage B1 the feature map was non-negative
// (ReLU), so a correlation peak was always positive and a signed maximum was
// fine. Mean subtraction makes the map bipolar: the strongest correlation is now
// as likely to be a trough as a crest — the aiesim s6 scenario peaks at
// {-417,0}. A signed max would walk straight past it and return whatever the
// largest positive sidelobe happened to be.
// Returns the peak |real| magnitude, which the caller MUST report.
//
// Without it a zero response is indistinguishable from a correct answer: with
// H_ch* zeroed the response is identically zero, and then only i=0 satisfies
// `mag > max_val` (0 > -1), so max_idx stays 0 and the wrap below maps it to
// displacement (0,0) — the right answer for a centred target, produced without
// looking at the data at all. peak == 0 means the result is meaningless.
static int peak_detect_sw(const int16_t *resp, int rows, int cols,
                          int *dr, int *dc)
{
    int max_val = -1, max_idx = 0;
    for (int i = 0; i < rows * cols; ++i) {
        int re  = (int)resp[2 * i];
        int mag = (re < 0) ? -re : re;
        if (mag > max_val) { max_val = mag; max_idx = i; }
    }
    int r = max_idx / cols;
    int c = max_idx % cols;
    // Wrap: centre of response map is dc-shifted to (0,0)
    if (r > rows / 2) r -= rows;
    if (c > cols / 2) c -= cols;
    *dr = r;
    *dc = c;
    return max_val;
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

// Stage B2 — cancel the residual mean in the frequency domain.
//
// conv2d removed mean_prev, but the true mean is mean_now, so the spectrum still
// carries (mean_now - mean_prev)·W where W = DFT(w⊗w). For the PERIODIC Hann, W
// has exactly 9 non-zero bins at (r,c) ∈ {0,±1}², so the correction is 9 complex
// subtractions per channel — and because correlation is linear, they can be
// applied once to the ACCUMULATED spectrum rather than per channel:
//
//   Σ_ch (X_ch - µ_ch·W) ⊙ H_ch*  =  Σ_ch X_ch ⊙ H_ch*  -  Σ_ch µ_ch · W ⊙ H_ch*
//
// NOT bit-exact: the two >>15 truncations in conv2d's window multiply are
// nonlinear, so linearity holds only up to quantization. Measured relative error
// after correction is ~1e-3, versus 2.5e-2 .. 9.9 without it.
static void apply_dc_correction(int16_t *accum, const int16_t *filter_all,
                                const double *residual_mean)
{
    // W[k] for a periodic Hann of length L: W[0] = L/2, W[±1] = -L/4, else 0.
    // The two axes have different lengths when the patch is not square.
    const double wrow[3] = { PATCH_ROWS * 0.5, -PATCH_ROWS * 0.25, -PATCH_ROWS * 0.25 };
    const double wcol[3] = { PATCH_COLS * 0.5, -PATCH_COLS * 0.25, -PATCH_COLS * 0.25 };
    const int    ridx[3] = { 0, 1, PATCH_ROWS - 1 };   // row-frequency index
    const int    kidx[3] = { 0, 1, PATCH_COLS - 1 };   // column index

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            // accum is in the TRANSPOSED layout the col-FFT produced:
            // element [c * PATCH_ROWS + r] is 2D spectrum bin (r, c).
            // Indexing it row-major (r * PATCH_COLS + c) happens to give the
            // same answer on a square patch — the bin set is transpose-symmetric
            // and the separable coefficient is symmetric in (a,b) — but it is
            // wrong the moment PATCH_ROWS != PATCH_COLS.
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

            int32_t re = (int32_t)accum[2 * bin]     - (int32_t)llround(corr_re);
            int32_t im = (int32_t)accum[2 * bin + 1] - (int32_t)llround(corr_im);
            accum[2 * bin]     = (int16_t)(re >  32767 ?  32767 : (re < -32768 ? -32768 : re));
            accum[2 * bin + 1] = (int16_t)(im >  32767 ?  32767 : (im < -32768 ? -32768 : im));
        }
    }
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
        {
            uint8_t *frame_ptr = frame_bo.map<uint8_t *>();
            // Keyed off the filter state, not off `frame == 0`. The two agree in
            // the current flow, but "is the filter trained yet" is the condition
            // that actually decides where the target must be, and the init branch
            // at the bottom of the loop uses the same one.
            const bool init_frame = !g_filter.initialized;
            int test_row = pos_row + (init_frame ? 0 : IMPULSE_DR);
            int test_col = pos_col + (init_frame ? 0 : IMPULSE_DC);
            // Asymmetric structured target, not a single-pixel impulse: an
            // impulse is symmetric, and a symmetric training patch makes a
            // transposed pack_filter() and a wrong conjugation both invisible.
            // See inject_target_frame().
            inject_target_frame(frame_ptr, FRAME_ROWS, FRAME_COLS, test_row, test_col);
            frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // Flush host → device
            if (init_frame)
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
                gm_fft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                     ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES);
                gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE,
                                 WEIGHT_CH_BYTES, ch * WEIGHT_CH_BYTES);
                gm_weights.wait();
                gm_fft_row_out.wait();
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
            gm_fft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0);
            gm_cmul_in.async(cmul_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, CMUL_IN_BYTES, 0);

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
                gm_accum_out.async(accum_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                   COL_CHUNK_BYTES, k * COL_CHUNK_BYTES);
                gm_fft_col_out.async(fcol_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                     COL_CHUNK_BYTES, k * COL_CHUNK_BYTES);
                gm_fft_col_out.wait();
                gm_accum_out.wait();
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

            gm_fft_col_in.wait();
            printf("[ch %d] fft_col_in sent\n", ch); fflush(stdout);
            gm_cmul_in.wait();
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
        gm_ifft_row_in.async(accum_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, ACCUM_BYTES, 0);
        // Drain per invocation, and before waiting on the input — see the
        // accum_out loop above for why the input wait cannot come first.
        for (int k = 0; k < ROW_CHUNKS; ++k) {
            gm_ifft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                                  ROW_CHUNK_BYTES, k * ROW_CHUNK_BYTES);
            gm_ifft_row_out.wait();
        }
        gm_ifft_row_in.wait();
        printf("[ifft] rows done (%d x %zu B)\n", ROW_CHUNKS, ROW_CHUNK_BYTES);
        fflush(stdout);

        // APU: transpose IFFT row output in-place
        transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);
        printf("[ifft] transpose done\n"); fflush(stdout);

        gm_ifft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0);
        for (int k = 0; k < COL_CHUNKS; ++k) {
            gm_response.async(resp_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO,
                              COL_CHUNK_BYTES, k * COL_CHUNK_BYTES);
            gm_response.wait();
        }
        gm_ifft_col_in.wait();
        printf("[ifft] cols done → response received (%d x %zu B)\n",
               COL_CHUNKS, COL_CHUNK_BYTES); fflush(stdout);

        // Response diagnostics run on EVERY frame, including frame 0. Frame 0's
        // response is not a tracking result (the filter is zero, so it should be
        // ~0), but "is it actually zero" is worth knowing: a non-zero frame-0
        // response would mean the filter buffer is not what the host thinks.
        report_cint16("response", resp_bo.map<int16_t *>(),
                      PATCH_ROWS, PATCH_COLS, "rowmajor");
        dump_buffer("resp", frame, resp_bo.map<void *>(), RESP_BYTES);
        if (g_filter.initialized)
            report_response(resp_bo.map<int16_t *>(), PATCH_ROWS, PATCH_COLS);

        // 4. Peak detection — read real parts (stride-2 for cint16)
        //
        // Skipped entirely on frame 0: the filter is zero there by construction,
        // so the response is identically zero and any displacement it reports is
        // an artifact of peak_detect_sw's scan order, not a measurement.
        if (!g_filter.initialized) {
            printf("Frame %d: [INIT] response not evaluated — filter is being "
                   "trained from this frame\n", frame);
        } else {
            int dr = 0, dc = 0;
            int peak = peak_detect_sw(resp_bo.map<int16_t *>(),
                                      PATCH_ROWS, PATCH_COLS, &dr, &dc);
            pos_row += dr;
            pos_col += dc;

            // peak==0 means the response map is identically zero, so the (0,0)
            // displacement carries no information — say so rather than printing a
            // plausible-looking position. Now that the filter is real this should
            // no longer happen; if it does, the filter or the shift budget is
            // wrong, not the test.
            const bool ok = (peak > 0 && dr == IMPULSE_DR && dc == IMPULSE_DC);
            printf("Frame %d: displacement (%d,%d) → pos (%d,%d)  peak|re|=%d  [%s]\n",
                   frame, dr, dc, pos_row, pos_col, peak,
                   peak == 0   ? "VOID: zero response — result carries no information"
                   : ok        ? "OK: matches injected offset"
                               : "MISMATCH vs injected offset");
            if (peak == 0)
                printf("       (filter is non-zero now, so this points at the shift "
                       "budget or the filter scale — check the Q1.15 report above)\n");
            else if (!ok)
                printf("       expected displacement (%d,%d)\n", IMPULSE_DR, IMPULSE_DC);
        }

        // 5. Filter init / update (PS-side). Bolme eq. 10-12; see mosse_filter.h.
        //
        // Runs AFTER peak detection so the filter used for this frame's response
        // is the one learned from previous frames — updating first would leak the
        // current frame into its own detection and make tracking look better than
        // it is.
        if (!g_filter.initialized) {
            mosse::filter_init(g_filter, g_F_all.data(), g_target.data(),
                               N_CHANNELS, PATCH_ROWS, PATCH_COLS);
            printf("  filter: INITIALISED from frame %d (single patch, no affine "
                   "perturbations)\n", frame);
        } else {
            mosse::filter_update(g_filter, g_F_all.data(), g_target.data(),
                                 mosse::DEFAULT_ETA);
        }
        publish_filter(filter_bo, filter_scratch);

        // The quantized filter, row-major, as filter_quantize_q15() produced it —
        // i.e. BEFORE pack_filter() converts it to the col-FFT layout. Dumping
        // this side of the conversion is deliberate: comparing it against a
        // NumPy H built from the dumped F_ch isolates the filter maths from the
        // layout conversion, which are the two remaining untested steps.
        report_cint16("H(q15)", filter_scratch.data(),
                      PATCH_ROWS, PATCH_COLS, "rowmajor");
        dump_buffer("H_q15", frame, filter_scratch.data(),
                    (size_t)N_CHANNELS * PATCH_ELEMS * 2 * sizeof(int16_t));
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    gr.end(0);  // block until graph completes
    return EXIT_SUCCESS;
}
