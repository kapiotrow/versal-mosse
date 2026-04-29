/*
 * roi_crop.h
 * PL stub kernel: extracts a patch from a DDR frame and streams it to AIE.
 *
 * Reads a PATCH_ROWS × PATCH_COLS region of the full frame starting at
 * (roi_row, roi_col) and writes it as a 128-bit AXIS stream to the AIE
 * PatchIn PLIO port.
 *
 * Output packing (128-bit word = 16 uint8 pixels, no RGB interleave in stub):
 *   word.data[7:0]   = pixel 0
 *   word.data[15:8]  = pixel 1
 *   ...
 *   word.data[127:120] = pixel 15
 *
 * The final beat has word.last = 1.
 * Called once per channel per frame (APU loops N_CHANNELS times).
 */

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

void roi_crop(
    const ap_uint<8>                   *frame_buf,   // DDR input frame
    hls::stream<ap_axiu<128,0,0,0>>   &patch_out,   // AXIS → AIE PatchIn PLIO
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  patch_rows,
    int  patch_cols
);
