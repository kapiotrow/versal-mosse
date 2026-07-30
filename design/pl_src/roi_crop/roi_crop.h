/*
 * roi_crop.h
 * PL stub kernel: extracts a patch from a DDR frame and streams it to AIE.
 *
 * Reads a PATCH_ROWS × PATCH_COLS region of the full frame starting at
 * (roi_row, roi_col) and writes it as a 32-bit AXIS stream to the AIE
 * PatchIn PLIO port (32-bit so it is 1:1 with conv2d's input_stream<int32>).
 *
 * Output packing (32-bit word = 4 uint8 pixels, no RGB interleave in stub):
 *   word.data[7:0]   = pixel 0
 *   word.data[15:8]  = pixel 1
 *   word.data[23:16] = pixel 2
 *   word.data[31:24] = pixel 3
 *
 * The final beat has word.last = 1.
 * Called once per channel per frame (APU loops N_CHANNELS times).
 */

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

void roi_crop(
    const ap_uint<8>                  *frame_buf,   // DDR input frame
    hls::stream<ap_axiu<32,0,0,0>>   &patch_out,   // AXIS → AIE PatchIn PLIO
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  patch_rows,
    int  patch_cols
);
