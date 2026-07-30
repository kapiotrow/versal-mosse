/*
 * conv2d_kernel.h
 * AIE-ML kernel: 3×3 INT8 convolution + separable Hanning window.
 *
 * Processes one feature channel per invocation. Called N_CHANNELS times
 * per frame with different weights loaded via gmio_weights each time.
 *
 * Input:  int8 grayscale pixel stream from PatchIn PLIO
 *         (PATCH_ROWS * PATCH_COLS samples, row-major)
 * Output: cint16 stream to fft2d row-FFT input
 *         (real = Hanning-windowed feature value, imag = 0)
 *
 * Weights: 64-byte buffer loaded via gmio_weights before each invocation.
 *   [0 : 9]   int8  w[KSIZE=3][KSIZE=3]  — 3×3 grayscale conv weights (row-major)
 *   [9]       int8  out_shift             — right-shift: int32 acc → int16
 *   [10:14]   int32 bias_acc (LE)         — bias in accumulator domain
 *   [14:18]   float32 dequant_scale (LE)  — for host validation only, not used by kernel
 *   [18:64]   zero padding
 *
 * Weights are derived from the first conv layer of MobileNetV3-Small (pretrained
 * ImageNet) collapsed to grayscale via luminance coefficients.
 * Generate with:  make weights  (runs scripts/export_weights.py)
 *
 * Input quantization contract:
 *   x_int8 in [-127, 127]  represents  x_gray_float = x_int8 / 127
 *   where x_gray_float is the ImageNet-normalized grayscale float:
 *     x_gray = 0.2989*R + 0.5870*G + 0.1140*B   (then ImageNet mean/std normalized)
 *   roi_crop / host should apply this quantization before sending to PLIO.
 *
 * Hanning window: applied to the int16 convolution output using a precomputed
 *   Q1.15 table (hanning_128.h).  The window zeros the patch borders, reducing
 *   spectral leakage in the downstream FFT.
 */

#pragma once

#include <adf.h>
using namespace adf;

// Single grayscale input channel
#ifndef CONV_IN_CH
#  define CONV_IN_CH  1
#endif
// Kernel size
#ifndef CONV_KSIZE
#  define CONV_KSIZE  3
#endif
// Weight array size for one output channel, padded to 64-byte GMIO alignment
#define CONV_WEIGHT_BYTES_RAW  (CONV_IN_CH * CONV_KSIZE * CONV_KSIZE)   // 9
#define CONV_WEIGHT_BYTES_PAD  64                                         // padded

#ifndef FFT_ROW_WS
#  define FFT_ROW_WS  2
#endif

// Output chunk = exactly one row-FFT input window.
// MUST equal FFT_ROW_TP_WINDOW_VSIZE in fft_graph.h (= PATCH_ROWS * FFT_ROW_WS),
// so conv2d.out connects to fft_row_in as a direct window→window link with NO
// stream→window adapter (that adapter is the suspected cause of the hw_emu hang).
// conv2d therefore fires (PATCH_ROWS*PATCH_COLS)/CONV_OUT_CHUNK times per patch.
#define CONV_OUT_CHUNK  (PATCH_ROWS * FFT_ROW_WS)

void conv2d_kernel(
    input_stream<int32>   *patch_in,     // from PatchIn PLIO (int32 words = 4 packed int8 pixels)
    output_buffer<cint16> &feature_out,  // window → fft2d.fft_row_in (CONV_OUT_CHUNK samples)
    input_buffer<int8_t>  &weights       // loaded via gmio_weights; CONV_WEIGHT_BYTES_PAD bytes
);
