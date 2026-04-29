/*
 * conv2d_kernel.h
 * AIE-ML kernel: 3×3 INT8 convolution with Hanning window (stub).
 *
 * Processes one feature channel per invocation. Called N_CHANNELS times
 * per frame with different weights loaded via gmio_weights each time.
 *
 * Input:  int8 pixel stream from PatchIn PLIO (patch_rows * patch_cols samples)
 * Output: cint16 stream to fft2d row-FFT input (real = windowed feature, imag = 0)
 *
 * Weights: loaded into tile local memory via gmio_weights GMIO before each call.
 *          Layout: [IN_CHANNELS=3][KSIZE=3][KSIZE=3] int8, padded to 64 bytes
 *          for GMIO alignment.
 *
 * Hanning window: TODO — apply separable 1D Hanning to each output sample.
 *                 Compute on-the-fly from row/col index to avoid 32 KB table.
 *
 * Hardware note (R1): a 128×128 int16 Hanning table = 32 KB, which combined
 * with other tile buffers risks exceeding 64 KB tile data memory on AIE-ML.
 * Use on-the-fly computation; verify with aiecompiler --report-all.
 */

#pragma once

#include <adf.h>

// Number of input channels (RGB)
#ifndef CONV_IN_CH
#  define CONV_IN_CH  3
#endif
// Kernel size
#ifndef CONV_KSIZE
#  define CONV_KSIZE  3
#endif
// Weight array size for one output channel, padded to 64-byte GMIO alignment
#define CONV_WEIGHT_BYTES_RAW  (CONV_IN_CH * CONV_KSIZE * CONV_KSIZE)   // 27
#define CONV_WEIGHT_BYTES_PAD  64                                         // padded

void conv2d_kernel(
    input_stream<int8>     *patch_in,     // from PatchIn PLIO
    output_stream<cint16>  *feature_out,  // to fft2d.fft_row_in (via window buffer)
    input_buffer<int8_t>   &weights,      // loaded via gmio_weights; size = CONV_WEIGHT_BYTES_PAD
    int32_t                 patch_rows,
    int32_t                 patch_cols,
    int32_t                 channel
);
