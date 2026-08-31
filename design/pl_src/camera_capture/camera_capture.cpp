/*
 * camera_capture.cpp
 * Minimal PL kernel: zeros the frame buffer.
 *
 * For hw_emu testing: The host app pre-loads frame_buf with test data
 * after calling this kernel, or can inject data directly via XRT BO.
 *
 * For hardware: TODO: add hls::stream<ap_axiu<32,0,0,0>> vid_in port
 *              to consume AXIS pixels from MIPI CSI-2 RX or V4L2 source.
 *
 * Frame layout: linear array of uint8 pixels (no RGB interleave for now).
 * Total size: frame_rows * frame_cols (not 3× for RGB channels).
 *
 * ---------------------------------------------------------------------------
 * COST — ~6 us of work: a zero-fill of the frame buffer, no AXIS port, no
 * datapath to speak of. Claim P-03.
 *
 *   why it matters anyway  That is exactly what makes it the KNOWN-GOOD
 *                 COMPARATOR in the launch-cost probe. A kernel with ~6 us of
 *                 work paying the same ~503 ms per KDS launch is what turned
 *                 "roi_crop is slow" into "any CU completion is slow" — and,
 *                 after the fix, the same probe still paying 503 ms in the same
 *                 run is what proves the fix is the fix. CONTROL_CU_RUNS keeps
 *                 it in the shipping build for that reason, not for its output.
 *   ordering      It zero-fills frame_bo BY DESIGN, so anything seeding that
 *                 buffer (the background memcpy) must run AFTER it.
 *
 * @thesis sec:architekturaSystemu | A-01 | The second PL kernel, a frame-buffer stub standing
 *   in for MIPI RX; it also serves as the known-good comparator in the KDS launch-cost probe
 *   (P-03).
 */

#include "camera_capture.h"

void camera_capture(
    ap_uint<8>  *frame_buf,
    int          frame_rows,
    int          frame_cols)
{
#pragma HLS INTERFACE m_axi     port=frame_buf  bundle=gmem0  depth=2073600
#pragma HLS INTERFACE s_axilite port=frame_rows bundle=control
#pragma HLS INTERFACE s_axilite port=frame_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Minimal stub: zero-fill the frame buffer
    // For hw_emu: Host app can pre-populate frame_buf with test data
    //             before calling this kernel (via xrt::bo map()).
    // For hw:     TODO: Replace with MIPI CSI-2 RX or V4L2 stream.
    int total = frame_rows * frame_cols;
    for (int i = 0; i < total; ++i) {
#pragma HLS PIPELINE II=1
        frame_buf[i] = 0;
    }
}
