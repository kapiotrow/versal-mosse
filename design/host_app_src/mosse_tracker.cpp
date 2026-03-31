/*
 * mosse_tracker.cpp
 * Host application: MOSSE correlation filter tracker on CNN features.
 *
 * Runs on the A72 processor using XRT APIs to orchestrate:
 *   - conv_layer   (PL)
 *   - fmt_adapter  (PL)
 *   - cmul_accum   (PL)
 *   - peak_detect  (PL)
 *   - MOSSE_graph  (AIE: N_CHANNELS FFT2D + 1 IFFT2D)
 *
 * Algorithm per frame:
 *   1. Decode frame, extract grayscale/RGB patch around tracked position.
 *   2. Run conv_layer: patch → feature maps (N_CHANNELS channels).
 *   3. For ch = 0..N_CHANNELS-1 (serial):
 *        Run fmt_adapter: feature_map[ch] + Hanning → AIE FFT row input.
 *        Run cmul_accum (ch): transpose + AIE FFT col + H_ch* multiply + accumulate.
 *   4. On last channel, cmul_accum flushes accum → AIE IFFT → peak_detect.
 *   5. Read (peak_row, peak_col) from peak_detect → update tracked position.
 *   6. Filter update (on PS): update A_ch[], B[], recompute H_ch* → write to filter_buf.
 *
 * TODO: add OpenCV or V4L2 video capture.
 * TODO: add initial filter training (first frame: compute H from scratch).
 * TODO: add tracking confidence check (PSR from peak_detect).
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <stdexcept>

#include "adf/adf_api/XRTConfig.h"
#include "xrt/xrt_aie.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

// TODO: include lightweight FFT library for PS-side filter update
// (e.g. kiss_fft.h) for computing A_ch and B accumulators.
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

#define PATCH_SIZE      (PATCH_ROWS * PATCH_COLS)
#define FEAT_MAP_SIZE   (PATCH_SIZE * N_CHANNELS)
#define FILTER_SIZE     (PATCH_SIZE * N_CHANNELS)  // one H_ch* per channel

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
static std::vector<char> load_xclbin(xrtDeviceHandle dhdl, const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open xclbin: " + path);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    if (xrtDeviceLoadXclbin(dhdl, reinterpret_cast<const axlf*>(buf.data())))
        throw std::runtime_error("xclbin load failed");
    return buf;
}

// Compute 2D Hanning window in Q1.15 fixed-point, stored row-major.
static void compute_hanning_q15(int16_t *win, int rows, int cols)
{
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double h = 0.5 * (1.0 - cos(2.0*M_PI*r / (rows-1)))
                     * 0.5 * (1.0 - cos(2.0*M_PI*c / (cols-1)));
            win[r*cols + c] = (int16_t)(h * 32767.0);
        }
}

// Compute ideal Gaussian response G (desired correlation peak) in spatial domain,
// then take its FFT to get G_star = conj(FFT(G)) needed for filter update.
// TODO: implement using kiss_fft or equivalent on A72.
static void compute_gaussian_response(/* ... */) { /* TODO */ }

// -----------------------------------------------------------------------
// XRT kernel wrappers
// -----------------------------------------------------------------------
class ConvLayerKernel {
public:
    xrtKernelHandle khdl;
    xrtRunHandle    rhdl;

    void init(xrtDeviceHandle dhdl, const axlf *top) {
        khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "conv_layer:{conv_layer_0}");
        rhdl = xrtRunOpen(khdl);
    }

    void run(xrtBufferHandle input_bo, xrtBufferHandle output_bo) {
        xrtRunSetArg(rhdl, 0, input_bo);
        xrtRunSetArg(rhdl, 1, output_bo);
        xrtRunSetArg(rhdl, 2, PATCH_ROWS);
        xrtRunSetArg(rhdl, 3, PATCH_COLS);
        xrtRunStart(rhdl);
    }

    void wait() { xrtRunWait(rhdl); }

    void close() { xrtRunClose(rhdl); xrtKernelClose(khdl); }
};

class FmtAdapterKernel {
public:
    xrtKernelHandle khdl;
    xrtRunHandle    rhdl;

    void init(xrtDeviceHandle dhdl, const axlf *top) {
        khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "fmt_adapter:{fmt_adapter_0}");
        rhdl = xrtRunOpen(khdl);
    }

    // feature_bo: full N_CHANNELS feature map buffer; offset selects channel ch
    void run(xrtBufferHandle feature_bo, xrtBufferHandle hanning_bo, int channel) {
        // TODO: use sub-buffer or offset argument to select channel slice
        xrtRunSetArg(rhdl, 0, feature_bo);   // TODO: channel offset
        xrtRunSetArg(rhdl, 1, hanning_bo);
        xrtRunSetArg(rhdl, 2, PATCH_ROWS);
        xrtRunSetArg(rhdl, 3, PATCH_COLS);
        xrtRunStart(rhdl);
    }

    void wait() { xrtRunWait(rhdl); }
    void close() { xrtRunClose(rhdl); xrtKernelClose(khdl); }
};

class CmulAccumKernel {
public:
    xrtKernelHandle khdl;
    xrtRunHandle    rhdl;

    void init(xrtDeviceHandle dhdl, const axlf *top) {
        khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "cmul_accum:{cmul_accum_0}");
        rhdl = xrtRunOpen(khdl);
    }

    void run(xrtBufferHandle transpose_bo, xrtBufferHandle filter_bo,
             xrtBufferHandle accum_bo, int channel) {
        xrtRunSetArg(rhdl, 6, transpose_bo);
        xrtRunSetArg(rhdl, 7, filter_bo);
        xrtRunSetArg(rhdl, 8, accum_bo);
        xrtRunSetArg(rhdl, 9,  channel);
        xrtRunSetArg(rhdl, 10, PATCH_ROWS);
        xrtRunSetArg(rhdl, 11, PATCH_COLS);
        xrtRunSetArg(rhdl, 12, (channel == N_CHANNELS - 1) ? 1 : 0);
        xrtRunStart(rhdl);
    }

    void wait() { xrtRunWait(rhdl); }
    void close() { xrtRunClose(rhdl); xrtKernelClose(khdl); }
};

class PeakDetectKernel {
public:
    xrtKernelHandle khdl;
    xrtRunHandle    rhdl;

    void init(xrtDeviceHandle dhdl, const axlf *top) {
        khdl = xrtPLKernelOpenExclusive(dhdl, top->m_header.uuid, "peak_detect:{peak_detect_0}");
        rhdl = xrtRunOpen(khdl);
    }

    void run() {
        xrtRunSetArg(rhdl, 2, PATCH_ROWS);
        xrtRunSetArg(rhdl, 3, PATCH_COLS);
        xrtRunStart(rhdl);
    }

    void wait() { xrtRunWait(rhdl); }

    void read_result(int *disp_row, int *disp_col) {
        uint32_t r, c;
        xrtKernelReadRegister(khdl, 0x10, &r);   // peak_row register
        xrtKernelReadRegister(khdl, 0x18, &c);   // peak_col register
        *disp_row = (int)(int32_t)r;
        *disp_col = (int)(int32_t)c;
    }

    void close() { xrtRunClose(rhdl); xrtKernelClose(khdl); }
};

// -----------------------------------------------------------------------
// Main tracking loop
// -----------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin> [video_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------
    // Device and xclbin setup
    // ------------------------------------------------------------------
    auto dhdl  = xrtDeviceOpen(0);
    auto xclbin = load_xclbin(dhdl, argv[1]);
    auto top   = reinterpret_cast<const axlf*>(xclbin.data());

    // ------------------------------------------------------------------
    // Allocate XRT buffers
    // ------------------------------------------------------------------
    // Input patch: uint8 RGB [PATCH_ROWS * PATCH_COLS * 3]
    auto patch_bo    = xrtBOAlloc(dhdl, PATCH_SIZE * 3,              0, 0);
    // Feature maps: int8 [PATCH_ROWS * PATCH_COLS * N_CHANNELS]
    auto feat_bo     = xrtBOAlloc(dhdl, FEAT_MAP_SIZE,               0, 0);
    // Hanning window: int16 [PATCH_ROWS * PATCH_COLS]
    auto hanning_bo  = xrtBOAlloc(dhdl, PATCH_SIZE * sizeof(int16_t), 0, 0);
    // Filter H_ch*: cint16 [N_CHANNELS * PATCH_ROWS * PATCH_COLS]
    auto filter_bo   = xrtBOAlloc(dhdl, FILTER_SIZE * 4,             0, 0);
    // Scratch transpose buffer and accumulator (cint16)
    auto transpose_bo = xrtBOAlloc(dhdl, PATCH_SIZE * 4,             0, 0);
    auto accum_bo     = xrtBOAlloc(dhdl, PATCH_SIZE * 4,             0, 0);

    // ------------------------------------------------------------------
    // One-time init: Hanning window + initial filter
    // ------------------------------------------------------------------
    int16_t *hanning_ptr = (int16_t*)xrtBOMap(hanning_bo);
    compute_hanning_q15(hanning_ptr, PATCH_ROWS, PATCH_COLS);
    xrtBOSync(hanning_bo, XCL_BO_SYNC_BO_TO_DEVICE, PATCH_SIZE * sizeof(int16_t), 0);

    // TODO: load first frame, select initial ROI, compute initial H_ch*
    // and write to filter_bo, then sync to device.
    printf("TODO: initialize filter from first frame\n");

    // ------------------------------------------------------------------
    // Open AIE graph
    // ------------------------------------------------------------------
    auto aie_graph = xrtGraphOpen(dhdl, top->m_header.uuid, "mosse_graph");
    if (!aie_graph) throw std::runtime_error("Cannot open mosse_graph");
    xrtGraphReset(aie_graph);
    xrtGraphRun(aie_graph, -1);   // run indefinitely, driven by PLIO

    // ------------------------------------------------------------------
    // Kernel handles
    // ------------------------------------------------------------------
    ConvLayerKernel  conv;     conv.init(dhdl, top);
    FmtAdapterKernel fmt;      fmt.init(dhdl, top);
    CmulAccumKernel  cmul;     cmul.init(dhdl, top);
    PeakDetectKernel peak;     peak.init(dhdl, top);

    // Tracked position (centre of search patch in frame coordinates)
    int pos_row = PATCH_ROWS / 2;
    int pos_col = PATCH_COLS / 2;

    // ------------------------------------------------------------------
    // Per-frame tracking loop
    // TODO: replace with actual video decode loop
    // ------------------------------------------------------------------
    for (int frame = 0; frame < ITER_CNT; ++frame) {
        printf("Frame %d: tracked position (%d, %d)\n", frame, pos_row, pos_col);

        // 1. Extract patch around pos_row, pos_col from current frame
        //    and write to patch_bo (uint8 RGB).
        // TODO: implement patch extraction with border handling.
        xrtBOSync(patch_bo, XCL_BO_SYNC_BO_TO_DEVICE, PATCH_SIZE * 3, 0);

        // 2. Convolutional feature extraction
        conv.run(patch_bo, feat_bo);
        conv.wait();

        // 3. Per-channel: format + FFT + cmul + accumulate
        for (int ch = 0; ch < N_CHANNELS; ++ch) {
            fmt.run(feat_bo, hanning_bo, ch);
            // cmul_accum starts in parallel: it reads fft_row_in from AIE
            cmul.run(transpose_bo, filter_bo, accum_bo, ch);
            // On last channel, cmul_accum also drives IFFT and IFFTcol
            if (ch == N_CHANNELS - 1)
                peak.run();
            fmt.wait();
            cmul.wait();
        }
        peak.wait();

        // 4. Read displacement and update position
        int disp_row, disp_col;
        peak.read_result(&disp_row, &disp_col);
        pos_row += disp_row;
        pos_col += disp_col;
        printf("  displacement (%d, %d) → new pos (%d, %d)\n",
               disp_row, disp_col, pos_row, pos_col);

        // 5. Filter update (on PS)
        // TODO: re-extract patch at new position, compute FFT(patch) via KissFFT,
        //       update A_ch += eta * G* ⊙ FFT(patch_ch)*
        //                  B += eta * |FFT(patch_ch)|²
        //       then H_ch* = A_ch / (B + lambda), write to filter_bo.
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    conv.close(); fmt.close(); cmul.close(); peak.close();
    xrtGraphClose(aie_graph);
    xrtBOFree(patch_bo); xrtBOFree(feat_bo); xrtBOFree(hanning_bo);
    xrtBOFree(filter_bo); xrtBOFree(transpose_bo); xrtBOFree(accum_bo);
    xrtDeviceClose(dhdl);

    return EXIT_SUCCESS;
}
