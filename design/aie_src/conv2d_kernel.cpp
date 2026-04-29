/*
 * conv2d_kernel.cpp
 * AIE-ML kernel stub: pass-through (no windowing, no convolution yet).
 *
 * Reads patch_rows * patch_cols int8 samples from patch_in, converts each
 * to cint16 {real=sample, imag=0}, and writes to feature_out.
 *
 * TODO (full implementation):
 *   1. Load 3×3 weights from the `weights` buffer for the current channel.
 *   2. Maintain a 3-line circular line buffer (3 × PATCH_COLS × CONV_IN_CH bytes).
 *   3. For each output pixel: accumulate weighted sum over 3×3×3 neighborhood,
 *      apply bias, clip to int8 range (ReLU-like activation).
 *   4. Apply separable Hanning window: multiply output by
 *      w_row[r] * w_col[c] / 32768 (Q1.15 fixed-point, computed on-the-fly).
 *   5. Pack as cint16 {real = windowed_feature, imag = 0}.
 */

#include "conv2d_kernel.h"

void conv2d_kernel(
    input_stream<int8>     *patch_in,
    output_stream<cint16>  *feature_out,
    input_buffer<int8_t>   &weights,
    int32_t                 patch_rows,
    int32_t                 patch_cols,
    int32_t                 channel)
{
    // Suppress unused-variable warnings for stub
    (void)weights;
    (void)channel;

    int total = patch_rows * patch_cols;
    for (int i = 0; i < total; ++i) {
        int8_t s = readincr(patch_in);
        cint16_t out;
        out.real = (int16_t)s;
        out.imag = 0;
        writeincr(feature_out, out);
    }
}
