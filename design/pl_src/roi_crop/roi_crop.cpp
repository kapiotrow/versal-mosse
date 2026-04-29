/*
 * roi_crop.cpp
 * PL stub: reads the ROI patch from DDR and streams it to AIE via PatchIn PLIO.
 *
 * The stub writes zeroed 128-bit beats; the real implementation should
 * read frame_buf[(roi_row+r)*frame_cols*3 + (roi_col+c)*3 + ch] and pack
 * pixels correctly.
 *
 * Packing: 16 uint8 values per 128-bit word, sequential row-major order.
 * The PLIO is plio_128_bits so each beat transfers 16 bytes.
 * Total beats = patch_rows * patch_cols / 16 (assuming patch size is
 * a multiple of 16; for 128×128 = 16384 pixels = 1024 beats).
 */

#include "roi_crop.h"

void roi_crop(
    const ap_uint<8>                   *frame_buf,
    hls::stream<ap_axiu<128,0,0,0>>   &patch_out,
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  patch_rows,
    int  patch_cols)
{
#pragma HLS INTERFACE m_axi     port=frame_buf  bundle=gmem0  depth=6220800
#pragma HLS INTERFACE axis      port=patch_out
#pragma HLS INTERFACE s_axilite port=frame_cols bundle=control
#pragma HLS INTERFACE s_axilite port=roi_row    bundle=control
#pragma HLS INTERFACE s_axilite port=roi_col    bundle=control
#pragma HLS INTERFACE s_axilite port=patch_rows bundle=control
#pragma HLS INTERFACE s_axilite port=patch_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Total pixels in the patch; must be a multiple of 16 for 128-bit packing.
    // Default: 128×128 = 16384 pixels → 1024 beats.
    int total_pixels = patch_rows * patch_cols;
    int total_beats  = total_pixels / 16;

    for (int beat = 0; beat < total_beats; ++beat) {
#pragma HLS PIPELINE II=1
        ap_axiu<128,0,0,0> word;
        word.data = 0;  // TODO: read 16 pixels from frame_buf at correct offsets
        word.keep = (ap_uint<16>)-1;
        word.strb = (ap_uint<16>)-1;
        word.last = (beat == total_beats - 1) ? 1 : 0;
        patch_out.write(word);
    }
}
