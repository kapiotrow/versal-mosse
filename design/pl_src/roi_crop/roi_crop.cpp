/*
 * roi_crop.cpp
 * Extracts a ROI patch from a DDR frame buffer and streams it to AIE via PatchIn PLIO.
 *
 * Reads frame pixels in row-major order from frame_buf starting at (roi_row, roi_col),
 * packs 16 consecutive uint8 pixels per 128-bit AXIS beat, and streams to PLIO.
 *
 * Frame layout: [row 0: pixel 0, pixel 1, ...] [row 1: ...] etc. (linear uint8 array)
 * ROI extraction: start at frame_buf[roi_row * frame_cols + roi_col],
 *                 read patch_rows × patch_cols pixels in row-major order
 *
 * Packing (128-bit = 16 bytes):
 *   beat[0].data[7:0]     = pixel[0]
 *   beat[0].data[15:8]    = pixel[1]
 *   ...
 *   beat[0].data[127:120] = pixel[15]
 *
 * HLS Optimization:
 * - Nested row/column loops (no division by runtime patch_cols in pipeline)
 * - UNROLL inner pixel packing loop for II=1 pipelining
 * - Total beats counter computed once outside pipeline
 */

#include "roi_crop.h"

void roi_crop(
    const ap_uint<8>                  *frame_buf,
    hls::stream<ap_axiu<32,0,0,0>>   &patch_out,
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  patch_rows,
    int  patch_cols)
{
#pragma HLS INTERFACE m_axi     port=frame_buf  bundle=gmem0  depth=2073600
#pragma HLS INTERFACE axis      port=patch_out
#pragma HLS INTERFACE s_axilite port=frame_cols bundle=control
#pragma HLS INTERFACE s_axilite port=roi_row    bundle=control
#pragma HLS INTERFACE s_axilite port=roi_col    bundle=control
#pragma HLS INTERFACE s_axilite port=patch_rows bundle=control
#pragma HLS INTERFACE s_axilite port=patch_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Pre-compute total beats outside the pipeline
    // >> 2 is bit-shift (divide by 4); no hardware divider. 4 pixels per 32-bit beat.
    int total_beats = (patch_rows * patch_cols) >> 2;
    int beat = 0;

    // Nested loops: row-major iteration without division inside pipeline
    for (int r = 0; r < patch_rows; ++r) {
        for (int c = 0; c < patch_cols; c += 4) {
#pragma HLS PIPELINE II=1
            ap_axiu<32,0,0,0> word;
            word.keep = (ap_uint<4>)-1;
            word.strb = (ap_uint<4>)-1;
            word.last = (beat == total_beats - 1) ? 1 : 0;

            // Unroll inner pixel packing loop: 4 pixels read in parallel
            for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
                // Frame address: (roi_row + r) * frame_cols + (roi_col + c + i)
                int frame_idx = (roi_row + r) * frame_cols + (roi_col + c + i);
                ap_uint<8> pix = frame_buf[frame_idx];
                word.data.range(8*i + 7, 8*i) = pix;
            }

            patch_out.write(word);
            beat++;
        }
    }
}
