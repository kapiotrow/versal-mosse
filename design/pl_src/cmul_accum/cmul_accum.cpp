/*
 * cmul_accum.cpp
 * Complex multiply + accumulate + transpose — HLS skeleton.
 *
 * This kernel is called N_CHANNELS times per detection frame (once per channel).
 * On the last channel (is_last_channel=1) it also drains the accumulator into
 * the IFFT stage and handles the IFFT row→col transpose.
 *
 * Transpose strategy: store entire row-FFT output matrix in DDR transpose_buf
 * in row-major order, then re-read column-by-column to feed AIE col-FFT.
 * For 128×128 cint16: 128*128*4 = 64 KB — fits in a single URAM36 block.
 * TODO: bind transpose_buf to URAM for lower latency.
 *
 * TODO: add #pragma HLS DATAFLOW between transpose fill, col-FFT feed,
 *       cmul-accumulate, and (on last channel) accum flush stages.
 */

#include "cmul_accum.h"

void cmul_accum(
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_row_in,
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_col_out,
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_col_in,
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_row_out,
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_row_in,
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_col_out,
    ap_int<32>  *transpose_buf,
    ap_int<32>  *filter_buf,
    ap_int<32>  *accum_buf,
    int          channel,
    int          patch_rows,
    int          patch_cols,
    int          is_last_channel)
{
    #pragma HLS INTERFACE axis        port=fft_row_in
    #pragma HLS INTERFACE axis        port=fft_col_out
    #pragma HLS INTERFACE axis        port=fft_col_in
    #pragma HLS INTERFACE axis        port=ifft_row_out
    #pragma HLS INTERFACE axis        port=ifft_row_in
    #pragma HLS INTERFACE axis        port=ifft_col_out
    #pragma HLS INTERFACE m_axi       port=transpose_buf  bundle=gmem0 depth=16384
    #pragma HLS INTERFACE m_axi       port=filter_buf     bundle=gmem1 depth=262144
    #pragma HLS INTERFACE m_axi       port=accum_buf      bundle=gmem2 depth=16384
    #pragma HLS INTERFACE s_axilite   port=channel          bundle=control
    #pragma HLS INTERFACE s_axilite   port=patch_rows       bundle=control
    #pragma HLS INTERFACE s_axilite   port=patch_cols       bundle=control
    #pragma HLS INTERFACE s_axilite   port=is_last_channel  bundle=control
    #pragma HLS INTERFACE s_axilite   port=return           bundle=control

    const int mat_sz = patch_rows * patch_cols;
    // Samples per 128-bit word (cint16: 32 bits each → 4 per word)
    const int words  = mat_sz / 4;

    // -------------------------------------------------------------------
    // Stage 1: drain AIE row-FFT output → transpose_buf (row-major)
    // -------------------------------------------------------------------
    S1_FILL: for (int w = 0; w < words; ++w) {
        #pragma HLS PIPELINE II=1
        ap_axiu<128, 0, 0, 0> beat = fft_row_in.read();
        for (int k = 0; k < 4; ++k)
            transpose_buf[w*4 + k] = beat.data.range(k*32+31, k*32);
    }

    // -------------------------------------------------------------------
    // Stage 2: re-stream transpose_buf column-by-column → AIE col-FFT
    // -------------------------------------------------------------------
    S2_COL: for (int c = 0; c < patch_cols; c += 4) {
        #pragma HLS PIPELINE II=patch_rows
        for (int r = 0; r < patch_rows; r += 4) {
            ap_axiu<128, 0, 0, 0> beat;
            beat.keep = -1;
            beat.last = (r + 4 >= patch_rows && c + 4 >= patch_cols) ? 1 : 0;
            for (int k = 0; k < 4; ++k)
                beat.data.range(k*32+31, k*32) = transpose_buf[(r+k)*patch_cols + c];
            // TODO: verify column-major re-streaming order matches AIE col-FFT expectation
            fft_col_out.write(beat);
        }
    }

    // -------------------------------------------------------------------
    // Stage 3: read col-FFT output (F_ch), multiply by H_ch*, accumulate
    // -------------------------------------------------------------------
    const int filter_offset = channel * mat_sz;

    S3_CMUL: for (int w = 0; w < words; ++w) {
        #pragma HLS PIPELINE II=1
        ap_axiu<128, 0, 0, 0> f_beat = fft_col_in.read();

        for (int k = 0; k < 4; ++k) {
            ap_int<32> f_sample  = f_beat.data.range(k*32+31, k*32);
            ap_int<32> h_sample  = filter_buf[filter_offset + w*4 + k];
            ap_int<32> product   = cmul_cint16(f_sample, h_sample);
            // Accumulate (saturating add TODO)
            accum_buf[w*4 + k] += product;
        }
    }

    // -------------------------------------------------------------------
    // Stage 4 (last channel only): stream accum_buf → AIE IFFT row stage
    //          + transpose IFFT row output for IFFT col stage
    // -------------------------------------------------------------------
    if (is_last_channel) {
        S4_ACCUM_TO_IFFT: for (int w = 0; w < words; ++w) {
            #pragma HLS PIPELINE II=1
            ap_axiu<128, 0, 0, 0> beat;
            beat.keep = -1;
            beat.last = (w == words - 1) ? 1 : 0;
            for (int k = 0; k < 4; ++k)
                beat.data.range(k*32+31, k*32) = accum_buf[w*4 + k];
            // Zero accumulator for next frame
            accum_buf[w*4 + 0] = 0; accum_buf[w*4 + 1] = 0;
            accum_buf[w*4 + 2] = 0; accum_buf[w*4 + 3] = 0;
            ifft_row_out.write(beat);
        }

        // Re-use transpose_buf for IFFT row → col transpose
        S4_IFFT_TRANSPOSE_FILL: for (int w = 0; w < words; ++w) {
            #pragma HLS PIPELINE II=1
            ap_axiu<128, 0, 0, 0> beat = ifft_row_in.read();
            for (int k = 0; k < 4; ++k)
                transpose_buf[w*4 + k] = beat.data.range(k*32+31, k*32);
        }

        S4_IFFT_COL_FEED: for (int c = 0; c < patch_cols; c += 4) {
            #pragma HLS PIPELINE II=patch_rows
            for (int r = 0; r < patch_rows; r += 4) {
                ap_axiu<128, 0, 0, 0> beat;
                beat.keep = -1;
                beat.last = (r + 4 >= patch_rows && c + 4 >= patch_cols) ? 1 : 0;
                for (int k = 0; k < 4; ++k)
                    beat.data.range(k*32+31, k*32) = transpose_buf[(r+k)*patch_cols + c];
                ifft_col_out.write(beat);
            }
        }
    }
}
