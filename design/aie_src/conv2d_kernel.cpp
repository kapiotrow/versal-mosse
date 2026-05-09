/*
 * conv2d_kernel.cpp
 * AIE-ML kernel stub: pass-through cast (no windowing, no convolution yet).
 *
 * Reads PATCH_ROWS * PATCH_COLS int8 samples from patch_in, converts each
 * to cint16 {real = sample, imag = 0}, and writes to feature_out.
 *
 * TODO (full implementation):
 *   1. Load 3×3 weights from the `weights` buffer for the current channel.
 *   2. Maintain a 3-line circular line buffer (3 × PATCH_COLS × CONV_IN_CH bytes).
 *   3. For each output pixel: accumulate weighted sum over 3×3×3 neighbourhood,
 *      apply bias, clip to int8 range.
 *   4. Apply separable Hanning window: multiply by w_row[r] * w_col[c] / 32768
 *      (Q1.15 on-the-fly, see hardware note in header).
 *   5. Add RTP port wiring in mosse_graph.h for channel index once step 1 is done.
 */

#include "conv2d_kernel.h"

void conv2d_kernel(
    input_stream<int8>    *patch_in,
    output_stream<cint16> *feature_out,
    input_buffer<int8_t>  &weights)
{
    (void)weights;

    for (int i = 0; i < PATCH_ROWS * PATCH_COLS; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(PATCH_ROWS * PATCH_COLS, PATCH_ROWS * PATCH_COLS)
    {
        int8_t s = readincr(patch_in);
        cint16_t out;
        out.real = (int16_t)s;
        out.imag = 0;
        writeincr(feature_out, out);
    }
}
