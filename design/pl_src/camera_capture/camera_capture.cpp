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
