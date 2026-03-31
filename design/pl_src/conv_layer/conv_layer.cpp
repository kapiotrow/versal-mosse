/*
 * conv_layer.cpp
 * Quantized 3x3 convolutional layer — HLS implementation skeleton.
 *
 * TODO: implement sliding window buffer + MAC array.
 * TODO: fill weight_data[N_FILTERS][IN_CHANNELS][KSIZE][KSIZE] with trained values.
 * TODO: implement post-activation (ReLU or clipped ReLU for INT8 output range).
 *
 * Suggested HLS directives:
 *   #pragma HLS ARRAY_PARTITION variable=weight_data complete dim=1
 *   #pragma HLS PIPELINE II=1  (inner spatial loop)
 *   #pragma HLS DATAFLOW        (between line-buffer fill and MAC stages)
 */

#include "conv_layer.h"

// Placeholder zero weights — replace with trained INT8 values.
static const ap_int<8> weight_data[N_FILTERS][IN_CHANNELS][CONV_KSIZE][CONV_KSIZE] = {};
static const ap_int<16> bias_data[N_FILTERS] = {};

void conv_layer(
    const ap_uint<8>  *input_patch,
    ap_int<8>         *output_features,
    int                patch_rows,
    int                patch_cols)
{
    #pragma HLS INTERFACE m_axi port=input_patch     bundle=gmem0 depth=49152  // 128*128*3
    #pragma HLS INTERFACE m_axi port=output_features bundle=gmem1 depth=262144 // 128*128*16 (N_FILTERS=16)
    #pragma HLS INTERFACE s_axilite port=patch_rows   bundle=control
    #pragma HLS INTERFACE s_axilite port=patch_cols   bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // TODO: implement line buffer (3 lines x PATCH_COLS x IN_CHANNELS) for
    //       sliding-window 3x3 convolution.
    //
    // For each output pixel (r, c, f):
    //   acc = bias_data[f]
    //   for kr in [0, KSIZE): for kc in [0, KSIZE): for ic in [0, IN_CHANNELS):
    //     acc += weight_data[f][ic][kr][kc] * line_buf[kr][(c + kc - PAD)][ic]
    //   output_features[(r*patch_cols + c)*N_FILTERS + f] = relu8(acc)
    (void)input_patch;
    (void)output_features;
    (void)patch_rows;
    (void)patch_cols;
}
