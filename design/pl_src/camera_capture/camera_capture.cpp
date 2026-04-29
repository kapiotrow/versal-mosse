/*
 * camera_capture.cpp
 * PL stub: zero-fills the frame buffer.
 * TODO: add hls::stream<ap_axiu<32,0,0,0>> vid_in port and consume
 *       AXIS pixels from MIPI CSI-2 RX Subsystem output.
 */

#include "camera_capture.h"

void camera_capture(
    ap_uint<8>  *frame_buf,
    int          frame_rows,
    int          frame_cols)
{
#pragma HLS INTERFACE m_axi     port=frame_buf  bundle=gmem0  depth=6220800
#pragma HLS INTERFACE s_axilite port=frame_rows bundle=control
#pragma HLS INTERFACE s_axilite port=frame_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    int total = frame_rows * frame_cols * 3;
    for (int i = 0; i < total; ++i) {
#pragma HLS PIPELINE II=1
        frame_buf[i] = 0;
    }
}
