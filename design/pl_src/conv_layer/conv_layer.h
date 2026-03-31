/*
 * conv_layer.h
 * HLS kernel: quantized first convolutional layer.
 *
 * Applies N_FILTERS 3x3 convolutions (with same padding) to an input
 * RGB patch, producing N_FILTERS feature maps of the same spatial size.
 *
 * Input:  uint8 pixels,   shape [PATCH_ROWS][PATCH_COLS][3]    (AXI master read from DDR)
 * Output: int8  features, shape [PATCH_ROWS][PATCH_COLS][N_FILTERS] (AXI master write to DDR)
 *
 * Weights are stored as int8 constants compiled into the kernel (baked from
 * Brevitas/FINN export or hand-set for initial testing).
 *
 * TODO: replace weight arrays with FINN-generated values or link FINN IP core.
 * TODO: evaluate replacing this kernel with a FINN-generated RTL module if
 *       Versal PL support matures in FINN.
 */

#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Kernel dimensions
#define CONV_KSIZE   3
#define CONV_PAD     (CONV_KSIZE / 2)   // same padding
#define IN_CHANNELS  3                   // RGB input

void conv_layer(
    const ap_uint<8>  *input_patch,     // DDR: [PATCH_ROWS * PATCH_COLS * IN_CHANNELS]
    ap_int<8>         *output_features, // DDR: [PATCH_ROWS * PATCH_COLS * N_FILTERS]
    int                patch_rows,
    int                patch_cols
);
