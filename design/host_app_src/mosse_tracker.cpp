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
#ifndef FFT_COL_WS
#  define FFT_COL_WS  2   // must match fft_graph.h FFT_COL_WS
#endif
#ifndef FFT_ROW_WS
#  define FFT_ROW_WS  2   // must match fft_graph.h FFT_ROW_WS
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
static void peak_detect_sw(const int16_t *resp, int rows, int cols,
                            int *dr, int *dc)
{
    int max_val = -32768, max_idx = 0;
    for (int i = 0; i < rows * cols; ++i) {
        int16_t re = resp[2 * i];
        if (re > max_val) { max_val = re; max_idx = i; }
    }
    int r = max_idx / cols;
    int c = max_idx % cols;
    // Wrap: centre of response map is dc-shifted to (0,0)
    if (r > rows / 2) r -= rows;
    if (c > cols / 2) c -= cols;
    *dr = r;
    *dc = c;
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

// Filter update stub — TODO: implement with KissFFT on A72.
static void filter_update_kissfft(/* ... */) { /* TODO */ }

// Compute ideal Gaussian response G (spatial domain); kept as a utility.
// TODO: implement using KissFFT to obtain G_star = conj(FFT(G)).
static void compute_gaussian_response(/* ... */) { /* TODO */ }

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

    // H_ch* is not computed yet (see filter_update_kissfft). Zero it explicitly
    // so cmul_accum consumes a defined value; the response is then zero rather
    // than garbage, which is a meaningful "pipeline ran end to end" result.
    // TODO: load first frame, select initial ROI, compute initial H_ch*
    //       and write to filter_bo.
    memset(filter_bo.map<void *>(), 0, FILTER_BYTES * N_CHANNELS);
    filter_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("TODO: initialize filters from first frame (H_ch* zeroed for now)\n");

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
        // For functional testing, use an impulse at the tracked position.
        // TODO: Replace with real video capture loop (OpenCV, V4L2).
        {
            uint8_t *frame_ptr = frame_bo.map<uint8_t *>();
            int test_row = pos_row;
            int test_col = pos_col;
            inject_impulse_frame(frame_ptr, FRAME_ROWS, FRAME_COLS, test_row, test_col, 200);
            frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // Flush host → device
        }

        // 2. Per-channel: conv2d + FFT2D + cmul_accum
        int roi_row = pos_row - PATCH_ROWS / 2;
        int roi_col = pos_col - PATCH_COLS / 2;

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
            // So: arm the output DMA, start the patch flowing, and only then feed
            // weights — conv2d drains them as it consumes the patch.
            gm_fft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO, FFT_BYTES, 0);

            // roi_crop → PatchIn PLIO → conv2d → fft_rows → gmio_fft_row_out
            auto crop_run = crop(frame_bo, FRAME_COLS,
                                 roi_row, roi_col, PATCH_ROWS, PATCH_COLS);

            // Load weights for channel ch — once per conv2d invocation, since the
            // kernel's `weights` input_buffer is consumed on every firing.
            for (int k = 0; k < CONV_INVOCATIONS; ++k) {
                gm_weights.async(weights_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, WEIGHT_CH_BYTES, ch * WEIGHT_CH_BYTES);
                gm_weights.wait();
            }
            printf("[ch %d] weights sent (%d buffers)\n", ch, CONV_INVOCATIONS); fflush(stdout);

            crop_run.wait();
            printf("[ch %d] roi_crop done\n", ch); fflush(stdout);
            gm_fft_row_out.wait();
            printf("[ch %d] fft_row_out received\n", ch); fflush(stdout);

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

            gm_fft_col_in.wait();
            printf("[ch %d] fft_col_in sent\n", ch); fflush(stdout);
            gm_cmul_in.wait();  // Wait for filter/accum data to reach AIE before accum reads
            printf("[ch %d] cmul_in sent\n", ch); fflush(stdout);

            // Read updated partial accumulator
            gm_accum_out.async(accum_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO, ACCUM_BYTES, 0);
            gm_accum_out.wait();
            printf("[ch %d] accum_out received\n", ch); fflush(stdout);
        }

        // 3. IFFT: APU feeds accumulated spectrum to IFFT row input
        printf("[ifft] START\n"); fflush(stdout);
        gm_ifft_row_in.async(accum_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, ACCUM_BYTES, 0);
        gm_ifft_row_out.async(row_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO, FFT_BYTES, 0);
        gm_ifft_row_in.wait();
        gm_ifft_row_out.wait();
        printf("[ifft] rows done\n"); fflush(stdout);

        // APU: transpose IFFT row output in-place
        transpose_inplace(row_bo.map<void *>(), PATCH_ROWS, PATCH_COLS, 4);
        printf("[ifft] transpose done\n"); fflush(stdout);

        gm_ifft_col_in.async(row_bo, XCL_BO_SYNC_BO_GMIO_TO_AIE, FFT_BYTES, 0);
        gm_response.async(resp_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO, RESP_BYTES, 0);
        gm_ifft_col_in.wait();
        gm_response.wait();
        printf("[ifft] cols done → response received\n"); fflush(stdout);

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
    gr.end(0);  // block until graph completes
    return EXIT_SUCCESS;
}
