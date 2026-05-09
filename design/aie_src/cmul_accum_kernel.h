/*
 * cmul_accum_kernel.h
 * AIE-ML kernel: element-wise complex multiply H_ch* ⊙ F_ch and accumulate.
 *
 * Called once per channel (N_CHANNELS times per frame):
 *   - fft_col_in: frequency-domain feature F_ch from fft2d col-FFT output
 *   - filter:     conjugate filter H_ch* for channel ch (from gmio_filter)
 *   - accum_prev: running accumulator from previous channels (from gmio_accum_in)
 *   - accum_out:  updated accumulator written to DDR via gmio_accum_out
 *
 * After all N_CHANNELS channels the accumulator holds Σ_c H_c* ⊙ F_c,
 * which the APU feeds to the IFFT via gmio_ifft_row_in.
 *
 * Scalar parameters (channel, patch_rows, patch_cols) are absent for the same
 * reason as conv2d_kernel.h: ADF requires RTP port wiring for scalars.
 * The stub uses PATCH_ROWS / PATCH_COLS compile-time defines. The 'channel'
 * conditional (initialise vs. accumulate) is handled by always reading accum_prev
 * from gmio_accum_in (APU sends zeros on ch=0).
 *
 * Hardware note: accumulator is cint16 in the stub (64 KB for 128×128).
 * Production should use cint32 (128 KB) stored in DDR via GMIO, or
 * migrate to an AIE-ML Memory Tile (512 KB on-chip). The GMIO round-trip
 * adds ~0.6 ms/frame at 3.3 GB/s for N_CHANNELS=16 — acceptable at 30 fps.
 */

#pragma once

#include <adf.h>
using namespace adf;

void cmul_accum_kernel(
    input_stream<cint16>   *fft_col_in,   // ← fft2d.fft_col_out
    output_stream<cint16>  *accum_out,    // → gmio_accum_out (DDR)
    input_buffer<cint16_t> &filter,       // H_ch*; PATCH_ROWS*PATCH_COLS cint16
    input_buffer<cint16_t> &accum_prev    // previous Σ; same size (APU sends zeros for ch=0)
);
