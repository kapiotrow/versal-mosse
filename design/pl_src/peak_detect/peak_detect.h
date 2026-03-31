/*
 * peak_detect.h
 * HLS kernel: find peak location in the IFFT correlation response map.
 *
 * Reads the PATCH_ROWS × PATCH_COLS spatial response map streamed from
 * the AIE IFFT col stage, finds the (row, col) of the maximum real value,
 * and writes the result to an AXI-Lite register readable by the host.
 *
 * The peak location gives the displacement of the object from the center
 * of the current search patch (after wrapping from [0,N) to [-N/2, N/2)).
 *
 * Output: (peak_row, peak_col) as int16 signed displacements from center.
 */

#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

void peak_detect(
    hls::stream<ap_axiu<128, 0, 0, 0>> &resp_in,  // ← AIE IFFTColOut0
    int                                  patch_rows,
    int                                  patch_cols,
    int                                 *peak_row,  // output displacement (signed)
    int                                 *peak_col   // output displacement (signed)
);
