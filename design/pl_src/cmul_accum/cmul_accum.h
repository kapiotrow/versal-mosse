/*
 * cmul_accum.h
 * HLS kernel: complex multiply + accumulate across feature channels.
 *
 * Responsibilities per frame:
 *
 *  [Detection pass]
 *   For each channel ch = 0 .. N_CHANNELS-1:
 *     1. Read row-FFT output from AIE [FFTRowOut{ch}] → write to DDR transpose buffer.
 *     2. Re-stream DDR transpose buffer column-by-column → AIE [FFTColIn{ch}].
 *     3. Read col-FFT output from AIE [FFTColOut{ch}] = F_ch(u,v).
 *     4. Read H_ch*(u,v) from DDR filter buffer.
 *     5. Compute F_ch ⊙ H_ch* and accumulate into resp_accum[u][v].
 *   After all channels:
 *     6. Stream resp_accum → AIE [IFFTRowIn0].
 *     7. Transpose IFFT row output → AIE [IFFTColIn0].
 *
 *  [Training pass — called by host after detection]
 *   Streams FFT outputs back to DDR so the host can update A_ch and B.
 *   (Alternatively, implement the numerator/denominator update here in PL.)
 *
 * TODO: decide whether filter update (A_ch, B accumulation) stays on PS
 *       or moves here. PS is simpler for a first implementation.
 */

#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Complex multiply: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
// All operands are cint16 packed as {imag[31:16], real[15:0]} in 32-bit.
inline ap_int<32> cmul_cint16(ap_int<32> f, ap_int<32> h_conj)
{
    #pragma HLS INLINE
    ap_int<16> ar = f.range(15, 0),      ai = f.range(31, 16);
    ap_int<16> cr = h_conj.range(15, 0), ci = h_conj.range(31, 16);
    // h_conj = c - jd (conjugate), so ci here is already negated by caller
    ap_int<32> result;
    result.range(15,  0) = (ap_int<16>)((ar*cr - ai*ci) >> 15);  // TODO: adjust shift
    result.range(31, 16) = (ap_int<16>)((ar*ci + ai*cr) >> 15);
    return result;
}

void cmul_accum(
    // Row-FFT output from AIE (per channel)
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_row_in,    // ← AIE FFTRowOut{ch}
    // Column-FFT feed to AIE (per channel, after transpose)
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_col_out,   // → AIE FFTColIn{ch}
    // Column-FFT output from AIE (F_ch)
    hls::stream<ap_axiu<128, 0, 0, 0>> &fft_col_in,    // ← AIE FFTColOut{ch}
    // Accumulated response feed to IFFT
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_row_out,  // → AIE IFFTRowIn0
    // IFFT row output (for transpose → IFFTColIn0)
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_row_in,   // ← AIE IFFTRowOut0
    hls::stream<ap_axiu<128, 0, 0, 0>> &ifft_col_out,  // → AIE IFFTColIn0
    // DDR buffers
    ap_int<32> *transpose_buf,   // scratch: [PATCH_ROWS * PATCH_COLS] cint16 words
    ap_int<32> *filter_buf,      // H_ch*: [N_CHANNELS * PATCH_ROWS * PATCH_COLS] cint16
    ap_int<32> *accum_buf,       // Σ H_ch* ⊙ F_ch: [PATCH_ROWS * PATCH_COLS] cint16
    int         channel,         // current channel index (0 .. N_CHANNELS-1)
    int         patch_rows,
    int         patch_cols,
    int         is_last_channel  // 1 when channel == N_CHANNELS-1: flush accum → IFFT
);
