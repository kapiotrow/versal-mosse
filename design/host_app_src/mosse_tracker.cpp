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
 *     9. APU: filter_update_kissfft() (stub)
 *
 * TODO: add OpenCV or V4L2 video capture loop.
 * TODO: implement first-frame filter initialization (compute H from Gaussian target).
 * TODO: implement PS-side filter update (KissFFT for A_ch, B, H_ch*).
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

// TODO: include KissFFT for PS-side filter update
// #include "kiss_fft.h"

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
#ifndef ITER_CNT
#  define ITER_CNT    1
#endif
#ifndef FRAME_ROWS
#  define FRAME_ROWS  1080
#endif
#ifndef FRAME_COLS
#  define FRAME_COLS  1920
#endif

// -----------------------------------------------------------------------
// Buffer sizes
// -----------------------------------------------------------------------
constexpr size_t PATCH_ELEMS       = PATCH_ROWS * PATCH_COLS;
constexpr size_t FFT_BYTES         = PATCH_ELEMS * 4;           // cint16 = 4 B/sample
constexpr size_t FILTER_BYTES      = PATCH_ELEMS * 4;           // per channel
constexpr size_t ACCUM_BYTES       = PATCH_ELEMS * 4;
constexpr size_t RESP_BYTES        = PATCH_ELEMS * 4;
constexpr size_t FRAME_BYTES       = (size_t)FRAME_ROWS * FRAME_COLS * 3; // RGB uint8
// conv2d weights: 3×3×3 INT8 = 27 bytes, padded to 64-byte GMIO alignment
constexpr size_t WEIGHT_CH_BYTES   = 64;

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
static void peak_detect_sw(const int16_t *resp_re, int rows, int cols,
                            int *dr, int *dc)
{
    int max_val = -32768, max_idx = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (resp_re[i] > max_val) { max_val = resp_re[i]; max_idx = i; }
    }
    int r = max_idx / cols;
    int c = max_idx % cols;
    // Wrap: centre of response map is dc-shifted to (0,0)
    if (r > rows / 2) r -= rows;
    if (c > cols / 2) c -= cols;
    *dr = r;
    *dc = c;
}

// Filter update stub — TODO: implement with KissFFT on A72.
static void filter_update_kissfft(/* ... */) { /* TODO */ }

// Compute ideal Gaussian response G (spatial domain); kept as a utility.
// TODO: implement using KissFFT to obtain G_star = conj(FFT(G)).
static void compute_gaussian_response(/* ... */) { /* TODO */ }

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
    gr.run(-1);  // streaming — driven by GMIO transactions in the loop

    // ------------------------------------------------------------------
    // GMIO handles (names must match MOSSE_graph constructor strings exactly)
    // ------------------------------------------------------------------
    xrt::aie::gmio gm_weights     (device, uuid, "gmio_weights");
    xrt::aie::gmio gm_fft_row_out (device, uuid, "gmio_fft_row_out");
    xrt::aie::gmio gm_fft_col_in  (device, uuid, "gmio_fft_col_in");
    xrt::aie::gmio gm_filter      (device, uuid, "gmio_filter");
    xrt::aie::gmio gm_accum_in    (device, uuid, "gmio_accum_in");
    xrt::aie::gmio gm_accum_out   (device, uuid, "gmio_accum_out");
    xrt::aie::gmio gm_ifft_row_in (device, uuid, "gmio_ifft_row_in");
    xrt::aie::gmio gm_ifft_row_out(device, uuid, "gmio_ifft_row_out");
    xrt::aie::gmio gm_ifft_col_in (device, uuid, "gmio_ifft_col_in");
    xrt::aie::gmio gm_response    (device, uuid, "gmio_response");

    // ------------------------------------------------------------------
    // XRT BOs (host-accessible DDR buffers)
    // ------------------------------------------------------------------
    // Frame buffer for camera_capture output
    auto frame_bo   = xrt::bo(device, FRAME_BYTES,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());
    // Shared row-FFT ↔ IFFT row scratch (cint16, 64 KB)
    auto row_bo     = xrt::bo(device, FFT_BYTES,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());
    // Partial accumulator (cint16, 64 KB)
    auto accum_bo   = xrt::bo(device, ACCUM_BYTES,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());
    // Filter H_ch* for all channels (cint16, 64 KB × N_CHANNELS)
    auto filter_bo  = xrt::bo(device, FILTER_BYTES * N_CHANNELS,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());
    // Weights for all channels (64 B × N_CHANNELS)
    auto weights_bo = xrt::bo(device, WEIGHT_CH_BYTES * N_CHANNELS,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());
    // Correlation response map (cint16, 64 KB)
    auto resp_bo    = xrt::bo(device, RESP_BYTES,
                               xrt::bo::flags::normal, device.get_info<xrt::info::device::bdf>());

    // ------------------------------------------------------------------
    // PL kernel handles
    // ------------------------------------------------------------------
    auto cam  = xrt::kernel(device, uuid, "camera_capture:{camera_capture_0}");
    auto crop = xrt::kernel(device, uuid, "roi_crop:{roi_crop_0}");

    // ------------------------------------------------------------------
    // One-time init
    // ------------------------------------------------------------------
    // TODO: load first frame, select initial ROI, compute initial H_ch*
    //       and write to filter_bo.
    printf("TODO: initialize filters from first frame\n");

    // Tracked position (centre of search patch in frame coordinates)
    int pos_row = FRAME_ROWS / 2;
    int pos_col = FRAME_COLS / 2;

    // ------------------------------------------------------------------
    // Per-frame tracking loop
    // ------------------------------------------------------------------
    for (int frame = 0; frame < ITER_CNT; ++frame) {

        // 1. Camera capture → DDR frame buffer
        {
            auto run = cam(frame_bo, FRAME_ROWS, FRAME_COLS);
            run.wait();
        }

        // 2. Per-channel: conv2d + FFT2D + cmul_accum
        int roi_row = pos_row - PATCH_ROWS / 2;
        int roi_col = pos_col - PATCH_COLS / 2;

        for (int ch = 0; ch < N_CHANNELS; ++ch) {

            // Load weights for channel ch
            int8_t *wp = weights_bo.map<int8_t *>() + ch * WEIGHT_CH_BYTES;
            gm_weights.gm2aie_nb(wp, WEIGHT_CH_BYTES);
            gm_weights.wait();

            // roi_crop → PatchIn PLIO → conv2d → fft_rows → gmio_fft_row_out
            auto crop_run = crop(frame_bo, FRAME_COLS,
                                 roi_row, roi_col, PATCH_ROWS, PATCH_COLS);
            gm_fft_row_out.aie2gm_nb(row_bo.map<void *>(), FFT_BYTES);
            crop_run.wait();
            gm_fft_row_out.wait();

            // APU: transpose row-FFT output in-place
            transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);

            // Feed transposed data to col-FFT + filter to cmul_accum
            gm_fft_col_in.gm2aie_nb(row_bo.map<void *>(), FFT_BYTES);
            int16_t *fp = filter_bo.map<int16_t *>()
                          + ch * (int)(PATCH_ELEMS * 2);  // cint16: 2 × int16
            gm_filter.gm2aie_nb(fp, FILTER_BYTES);

            if (ch > 0) {
                gm_accum_in.gm2aie_nb(accum_bo.map<void *>(), ACCUM_BYTES);
                gm_accum_in.wait();
            }

            gm_fft_col_in.wait();
            gm_filter.wait();

            // Read updated partial accumulator
            gm_accum_out.aie2gm_nb(accum_bo.map<void *>(), ACCUM_BYTES);
            gm_accum_out.wait();
        }

        // 3. IFFT: APU feeds accumulated spectrum to IFFT row input
        gm_ifft_row_in.gm2aie_nb(accum_bo.map<void *>(), ACCUM_BYTES);
        gm_ifft_row_out.aie2gm_nb(row_bo.map<void *>(), FFT_BYTES);
        gm_ifft_row_in.wait();
        gm_ifft_row_out.wait();

        // APU: transpose IFFT row output in-place
        transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);

        gm_ifft_col_in.gm2aie_nb(row_bo.map<void *>(), FFT_BYTES);
        gm_response.aie2gm_nb(resp_bo.map<void *>(), RESP_BYTES);
        gm_ifft_col_in.wait();
        gm_response.wait();

        // 4. Peak detection — read real parts (stride-2 for cint16)
        int dr = 0, dc = 0;
        peak_detect_sw(resp_bo.map<int16_t *>(), PATCH_ROWS, PATCH_COLS, &dr, &dc);
        pos_row += dr;
        pos_col += dc;
        printf("Frame %d: displacement (%d,%d) → pos (%d,%d)\n",
               frame, dr, dc, pos_row, pos_col);

        // 5. Filter update (PS-side, stub)
        filter_update_kissfft();
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    gr.end(1000);  // 1 s timeout
    return EXIT_SUCCESS;
}
