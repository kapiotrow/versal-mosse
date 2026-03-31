/*
 * fmt_adapter.cpp
 * Feature map formatter + Hanning window application — HLS skeleton.
 *
 * Packing (cint16, 128-bit word):
 *   word = { sample3.imag, sample3.real, sample2.imag, sample2.real,
 *            sample1.imag, sample1.real, sample0.imag, sample0.real }
 *   Each sample: real = (int16)(hanning[i] * feature[i]) >> HANNING_FRAC_BITS
 *                imag = 0
 *
 * The Hanning window is precomputed as fixed-point INT16 values
 * (e.g., Q1.15 format) on the host and transferred to DDR once at init.
 *
 * TODO: move hanning_win to BRAM (once, at startup) using
 *       #pragma HLS bind_storage variable=... type=RAM_2P impl=BRAM
 */

#include "fmt_adapter.h"

#define HANNING_FRAC_BITS 15   // Q1.15 fixed-point Hanning window

void fmt_adapter(
    const ap_int<8>                    *feature_map,
    const ap_int<16>                   *hanning_win,
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_row_out,
    int                                 patch_rows,
    int                                 patch_cols)
{
    #pragma HLS INTERFACE m_axi port=feature_map  bundle=gmem0 depth=16384  // 128*128
    #pragma HLS INTERFACE m_axi port=hanning_win  bundle=gmem1 depth=16384
    #pragma HLS INTERFACE axis  port=fft_row_out
    #pragma HLS INTERFACE s_axilite port=patch_rows  bundle=control
    #pragma HLS INTERFACE s_axilite port=patch_cols  bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control

    const int total_samples = patch_rows * patch_cols;

    FMT_LOOP: for (int i = 0; i < total_samples; i += 4) {
        #pragma HLS PIPELINE II=1
        #pragma HLS loop_tripcount min=4096 max=4096  // 128*128/4

        ap_axiu<128, 0, 0, 0> word;

        for (int k = 0; k < 4; ++k) {
            ap_int<16> real_val = 0;
            if (i + k < total_samples) {
                // Fixed-point multiply: (int16 hanning) * (int8 feature) >> FRAC_BITS
                ap_int<32> prod = (ap_int<32>)hanning_win[i+k] * (ap_int<32>)feature_map[i+k];
                real_val = (ap_int<16>)(prod >> HANNING_FRAC_BITS);
            }
            // Pack: cint16 = {imag(16b), real(16b)}
            word.data.range(k*32 + 15,  k*32)      = real_val;
            word.data.range(k*32 + 31,  k*32 + 16) = (ap_int<16>)0;  // imag = 0
        }

        word.keep = -1;
        word.last = (i + 4 >= total_samples) ? 1 : 0;
        fft_row_out.write(word);
    }
}
