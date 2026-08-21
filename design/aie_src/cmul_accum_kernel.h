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
 * H is Q1.15 (host normalizes max|H| across all channels to 32767) and the
 * product is shifted right by CMUL_H_SHIFT with round-to-nearest. See the
 * H_SHIFT block in the Makefile for why, and why the existing FFT/IFFT shift
 * budget survives the change unmodified.
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

// Filter-product shift, from the Makefile (H_SHIFT). 15 = H is Q1.15.
// 0 reproduces the historical no-shift behaviour, kept reachable for bisection.
// Defined in the header, not the .cpp, so the aiesim harness can report the value
// its binary was actually built with.
#ifndef CMUL_H_SHIFT
#  define CMUL_H_SHIFT 15
#endif
static_assert(CMUL_H_SHIFT >= 0 && CMUL_H_SHIFT <= 30, "CMUL_H_SHIFT out of range");

// CMUL_SPLIT_ACCUM: give accum_prev its OWN input port instead of packing it
// behind H in cmul_in.
//
// WHY. The packing is a host memcpy of 128 KB per channel, 2 MB/frame, and both
// sources are xrt::bo mappings — which are write-combining, so the READ side runs
// at 696 MB/s (startup probe). 2 MB at 696 MB/s is 3.01 ms; the `cmul packing`
// slot measures 2.871. It is not a copy that happens to be slow, it IS the
// uncached read, and splitting the port deletes it outright rather than
// optimising it.
//
// The single-port design was never about hardware — the header note below says
// so: it works around a Vitis 2025.2 cycle-approximate aiesim deadlock, and "on
// real AIE-ML hardware both approaches are equivalent". `make aiesim` therefore
// needs CMUL_SPLIT_ACCUM=0; hardware does not.
//
// This is also the prerequisite for holding the accumulator in a memory tile:
// once accum_prev arrives on its own port, its source can be changed from a GMIO
// to a shared_buffer without touching the kernel again.
#ifndef CMUL_SPLIT_ACCUM
#  define CMUL_SPLIT_ACCUM 0
#endif

void cmul_accum_kernel(
    input_buffer<cint16_t>  &fft_col_in,  // in[0]: F_ch ← fft2d.fft_col_out (tile-to-tile, must be first)
    output_buffer<cint16_t> &accum_out,   // out[0]: updated Σ → gmio_accum_out
#if CMUL_SPLIT_ACCUM
    input_buffer<cint16_t>  &cmul_in,     // in[1]: H_ch* only ← gmio_cmul_in
    input_buffer<cint16_t>  &accum_in     // in[2]: prev_Σ    ← gmio_accum_in
#else
    input_buffer<cint16_t>  &cmul_in      // in[1]: [H_ch* | prev_Σ] packed ← gmio_cmul_in
#endif
);
