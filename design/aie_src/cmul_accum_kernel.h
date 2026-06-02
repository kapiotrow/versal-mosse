/*
 * cmul_accum_kernel.h
 * AIE-ML kernel: element-wise complex multiply H_ch* ⊙ F_ch and accumulate.
 *
 * Called once per channel (N_CHANNELS times per frame):
 *   - fft_col_in: frequency-domain feature F_ch from fft2d col-FFT output
 *   - cmul_in:    packed [H_ch* | prev_Σ] from gmio_cmul_in
 *                 First PATCH_COLS*FFT_COL_WS elements = filter H_ch*
 *                 Next  PATCH_COLS*FFT_COL_WS elements = accum_prev
 *   - accum_out:  updated accumulator written to DDR via gmio_accum_out
 *
 * After all N_CHANNELS channels the accumulator holds Σ_c H_c* ⊙ F_c,
 * which the APU feeds to the IFFT via gmio_ifft_row_in.
 *
 * Single combined GMIO port (gmio_cmul_in) note:
 *   Vitis 2025.2 cycle-approximate aiesim deadlocks when a kernel reads from
 *   two distinct GMIO-backed input_buffer objects in the same invocation,
 *   regardless of loop structure or chess pragmas.  Packing filter and
 *   accum_prev into one buffer (one lock pair) avoids this ISS limitation.
 *   On real AIE-ML hardware both approaches are equivalent.
 *
 *   The PS interleaves the two halves per chunk before calling gm2aie_nb:
 *     for c in 0..N_CHUNKS-1:
 *       combined[ c*2*CHUNK .. c*2*CHUNK+CHUNK-1 ] = filter_chunk_c
 *       combined[ c*2*CHUNK+CHUNK .. c*2*CHUNK+2*CHUNK-1 ] = accum_chunk_c
 *   where CHUNK = PATCH_COLS * FFT_COL_WS.
 *
 * Port order note: fft_col_in MUST be in[0] (first input_buffer arg).
 * In Vitis 2025.2 cycle-approximate aiesim, INPUT GMIO locks on tile 23_0
 * (cmul_in=in[1]) can only be delivered after the tile-to-tile lock for
 * fft_col_in (from col-FFT tile 22_0) fires first in the iteration.
 */

#pragma once

#include <adf.h>
using namespace adf;

void cmul_accum_kernel(
    input_buffer<cint16_t>  &fft_col_in,  // in[0]: F_ch ← fft2d.fft_col_out (tile-to-tile, must be first)
    output_buffer<cint16_t> &accum_out,   // out[0]: updated Σ → gmio_accum_out
    input_buffer<cint16_t>  &cmul_in      // in[1]: [H_ch* | prev_Σ] packed ← gmio_cmul_in
);
