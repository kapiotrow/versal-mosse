/*
 * fmt_adapter.h
 * HLS kernel: feature map formatter + Hanning window.
 *
 * For each feature channel in turn, this kernel:
 *   1. Reads one INT8 feature map [PATCH_ROWS][PATCH_COLS] from DDR.
 *   2. Applies a precomputed 2D Hanning window (stored in BRAM).
 *   3. Packs samples as cint16 (real = windowed INT8 value, imag = 0).
 *   4. Streams 128-bit words (4 cint16 samples/word) to the AIE row-FFT PLIO.
 *
 * After the AIE row-FFT completes, the PL transpose kernel (part of cmul_accum)
 * reads the row-FFT output and re-streams it column-by-column back into this
 * kernel's col_out port for the AIE col-FFT stage.
 *
 * Called N_CHANNELS times per frame (once per channel, serially).
 */

#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

void fmt_adapter(
    const ap_int<8>                    *feature_map,   // DDR: one channel [PATCH_ROWS*PATCH_COLS]
    const ap_int<16>                   *hanning_win,   // DDR or BRAM: [PATCH_ROWS*PATCH_COLS]
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_row_out,  // → AIE FFTRowIn{ch}
    int                                 patch_rows,
    int                                 patch_cols
);
