/*
 * mosse_graph.cpp
 * Instantiation and aiesim entry point for the MOSSE AIE graph.
 *
 * In the new architecture there is only one FFT2D instance and one IFFT2D
 * instance (serial per-channel processing). The old per-instance counters
 * (fftRows_grInsts etc.) are removed.
 *
 * For aiesim: the graph runs with run(-1) (streaming) while the host
 * manually drives GMIO ports. For a ch=0 smoke test, we push zeroed data
 * through the full pipeline and verify no deadlock occurs.
 *
 * Production use: APU calls gr.run(-1) and drives the GMIO ports in the
 * per-frame, per-channel loop in mosse_tracker.cpp.
 */

#include "mosse_graph.h"

MOSSE_graph mosse_graph;

#ifdef __AIESIM__

#include <cstring>

int main(int argc, char **argv)
{
    mosse_graph.init();
    mosse_graph.run(-1);  // streaming — driven by GMIO transactions below

    // ----------------------------------------------------------------
    // Zeroed test buffers (all transfers are GMIO-aligned: multiple of 64 B)
    // ----------------------------------------------------------------
    constexpr int PATCH_ELEMS  = PATCH_ROWS * PATCH_COLS;
    constexpr int PATCH_BYTES  = PATCH_ELEMS * 4;   // cint16 = 4 bytes per sample

    static int8_t   weights_buf[64]          = {};   // 27 weights padded to 64 B
    static int16_t  fft_scratch[PATCH_ELEMS * 2] = {};  // cint16 ping buffer (re-used)
    static int16_t  filter_buf [PATCH_ELEMS * 2] = {};
    static int16_t  accum_buf  [PATCH_ELEMS * 2] = {};
    static int16_t  resp_buf   [PATCH_ELEMS * 2] = {};

    // ----------------------------------------------------------------
    // Channel 0 smoke test (ch=0: no gmio_accum_in read-back)
    // ----------------------------------------------------------------

    // Load weights for channel 0 (zeroed stub)
    mosse_graph.gmio_weights.gm2aie_nb(weights_buf, 64);
    mosse_graph.gmio_weights.wait();

    // PatchIn is driven from patch_in.txt (aiesim_data/patch_in.txt)
    // conv2d → fft_rows → gmio_fft_row_out
    mosse_graph.gmio_fft_row_out.aie2gm_nb(fft_scratch, PATCH_BYTES);
    mosse_graph.gmio_fft_row_out.wait();

    // APU would transpose here; for aiesim skip (send same data, no transpose)

    // Send (un-transposed) data to col-FFT and filter
    mosse_graph.gmio_fft_col_in.gm2aie_nb(fft_scratch, PATCH_BYTES);
    mosse_graph.gmio_filter.gm2aie_nb(filter_buf, PATCH_BYTES);
    mosse_graph.gmio_fft_col_in.wait();
    mosse_graph.gmio_filter.wait();

    // cmul → gmio_accum_out
    mosse_graph.gmio_accum_out.aie2gm_nb(accum_buf, PATCH_BYTES);
    mosse_graph.gmio_accum_out.wait();

    // ----------------------------------------------------------------
    // IFFT pass (APU feeds accum to IFFT row input)
    // ----------------------------------------------------------------
    mosse_graph.gmio_ifft_row_in.gm2aie_nb(accum_buf, PATCH_BYTES);
    mosse_graph.gmio_ifft_row_out.aie2gm_nb(fft_scratch, PATCH_BYTES);
    mosse_graph.gmio_ifft_row_in.wait();
    mosse_graph.gmio_ifft_row_out.wait();

    // (skip IFFT transpose for smoke test)
    mosse_graph.gmio_ifft_col_in.gm2aie_nb(fft_scratch, PATCH_BYTES);
    mosse_graph.gmio_response.aie2gm_nb(resp_buf, PATCH_BYTES);
    mosse_graph.gmio_ifft_col_in.wait();
    mosse_graph.gmio_response.wait();

    mosse_graph.end();
    return 0;
}

#endif  // __AIESIM__
