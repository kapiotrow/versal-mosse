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
 *   THE LAYOUT IS IN conv_weight_layout.h, derived from CONV_IN_CH — it is not
 *   repeated here, because it used to be repeated in four places and RGB's 27
 *   taps overrun every grayscale field. At CONV_IN_CH=1 it resolves to the
 *   historical offsets: taps [0:9), out_shift 9, bias_acc 10, dequant_scale 14,
 *   mean_prev 18. Byte 63 is a layout tag.
 *
 *   mean_prev is Stage B1's previous-frame per-channel feature mean, written by
 *   the HOST every frame — see below.
 *
 * Weights are derived from the first conv layer of MobileNetV3-Small (pretrained
 * ImageNet), collapsed to grayscale via luminance coefficients at CONV_IN_CH=1
 * or kept as three planes at CONV_IN_CH=3.
 * Generate with:  make weights  (runs scripts/export_weights.py)
 *
 * Input quantization contract:
 *   x_int8 in [-127, 127], SIGNED, produced by roi_crop's Stage A preprocessing
 *   (bilinear resample → log → zero mean → unit L2 norm × ROI_NORM_Q → clip).
 *   See design/pl_src/roi_crop/roi_crop.h. Before Stage A existed, roi_crop emitted
 *   *unsigned* 0..255 into this signed contract and every pixel ≥128 wrapped
 *   negative; do not regress that.
 *
 * Stage B1 — mean removal (Bolme §3.1 "normalized to have a mean value of 0.0"):
 *   ReLU makes the feature map strictly non-negative, so windowing it directly
 *   pushes a large DC pedestal (≈ µ·N·mean(w) = µ·4096 at 128×128) into the FFT,
 *   starving the informative AC bins. We subtract the PREVIOUS frame's
 *   per-channel mean before the window; tracking is temporally smooth so
 *   µ_t ≈ µ_{t-1}, and the host cancels the residual (µ_t - µ_{t-1})·W with a
 *   9-bin frequency-domain correction (the periodic Hann's 2D DFT has exactly 9
 *   non-zero bins). One subtract per element, no buffering, no extra pass, no
 *   new port.
 *
 *   mean_prev is the WINDOW-WEIGHTED mean Σ(w⊗w)·f / (Σw)², not the plain mean.
 *   Bolme's text says plain mean, but the plain mean does not zero a
 *   window-weighted DC sum. Measured on channel 0 with real exported weights and
 *   a realistic patch, DC/AC ratio:
 *       no subtraction  393600  (18.59 bits)
 *       plain mean        2179  (11.09 bits)
 *       window-weighted      43  ( 5.43 bits)
 *   Both are "subtract a constant, then window"; the window-weighted one is the
 *   constant that actually zeros the DC bin, and it is worth 5.7 extra bits of
 *   cint16 headroom. It also costs the host nothing: summing the PATCH_ROWS
 *   row-FFT DC bins (which the APU already touches when transposing) yields
 *   Σ(w⊗w)·f exactly.
 *
 *   The residual correction is NOT bit-exact: the two >>15 truncations in the
 *   window multiply are nonlinear, so linearity holds only up to quantization.
 *   Measured relative error after correction is ~1e-3 (vs 2.5e-2 .. 9.9 without),
 *   which is 3-4 orders of magnitude better but not zero.
 *
 * Hanning window: applied to the int16 convolution output using a precomputed
 *   Q1.15 table (hanning_128.h).  The window zeros the patch borders, reducing
 *   spectral leakage in the downstream FFT.
 *   The table is the PERIODIC Hann; the host's mean correction depends on it.
 */

#pragma once

#include <adf.h>
using namespace adf;

// CONV_IN_CH (1 = luminance, 3 = RGB), CONV_KSIZE, CONV_WEIGHT_BYTES_RAW/_PAD
// and every field offset. One definition, shared with the host, roi_crop and
// (mirrored) scripts/conv_weight_layout.py.
#include "conv_weight_layout.h"

// AIE stack for conv2d, BYTES. Only applied at CONV_IN_CH=3 (see mosse_graph.h):
// the 27-tap MAC chain needs 1344 where the default is 1024. Powers of two keep
// the tile allocator's arithmetic simple; 2048 leaves ~700 bytes of headroom for
// a future tap or post-chain change without another round trip through the
// mapper.
#ifndef CONV2D_STACK
#  define CONV2D_STACK 2048
#endif

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
