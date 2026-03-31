/*
 * peak_detect.cpp
 * Argmax on correlation response map — HLS skeleton.
 *
 * Streams through the response map once, tracking the maximum real-part
 * value and its flat index, then converts to (row, col) and wraps to
 * signed displacement from patch center.
 *
 * cint16 response: real part in bits [15:0], imaginary in [31:16].
 * Only the real part is used (imaginary should be ~0 for a valid correlation).
 *
 * TODO: add PSR (peak-to-sidelobe ratio) computation for tracking confidence.
 */

#include "peak_detect.h"

void peak_detect(
    hls::stream<ap_axiu<128, 0, 0, 0>> &resp_in,
    int                                  patch_rows,
    int                                  patch_cols,
    int                                 *peak_row,
    int                                 *peak_col)
{
    #pragma HLS INTERFACE axis      port=resp_in
    #pragma HLS INTERFACE s_axilite port=patch_rows  bundle=control
    #pragma HLS INTERFACE s_axilite port=patch_cols  bundle=control
    #pragma HLS INTERFACE s_axilite port=peak_row    bundle=control
    #pragma HLS INTERFACE s_axilite port=peak_col    bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control

    const int mat_sz = patch_rows * patch_cols;
    const int words  = mat_sz / 4;

    ap_int<16> max_val  = ap_int<16>(0x8000);   // most negative INT16
    int        max_idx  = 0;

    SCAN: for (int w = 0; w < words; ++w) {
        #pragma HLS PIPELINE II=1
        #pragma HLS loop_tripcount min=4096 max=4096

        ap_axiu<128, 0, 0, 0> beat = resp_in.read();

        for (int k = 0; k < 4; ++k) {
            ap_int<16> real_val = beat.data.range(k*32 + 15, k*32);
            if (real_val > max_val) {
                max_val = real_val;
                max_idx = w * 4 + k;
            }
        }
    }

    // Convert flat index → (row, col)
    int r = max_idx / patch_cols;
    int c = max_idx % patch_cols;

    // Wrap to signed displacement from center: [0, N) → [-N/2, N/2)
    if (r > patch_rows / 2) r -= patch_rows;
    if (c > patch_cols / 2) c -= patch_cols;

    *peak_row = r;
    *peak_col = c;
}
