/*
 * ifft_graph.h
 * AIE IFFT2D graph for the MOSSE correlation response.
 *
 * Single instance; fed by the accumulated correlation spectrum
 * Σ_c H_c* ⊙ F_c written to DDR by cmul_accum_kernel, then passed to
 * the IFFT row input via gmio_ifft_row_in by the APU.
 *
 * Same row/col-transpose-via-DDR pattern as fft_graph.h:
 *   ifft_row_out → gmio_ifft_row_out → DDR   (APU transposes)
 *   DDR → gmio_ifft_col_in → ifft_col_in
 *   ifft_col_out → gmio_response → DDR       (APU reads for peak detection)
 *
 * Normalization shift (empirically calibrated):
 *   Row IFFT: shift = 0   (no normalization on row pass)
 *   Col IFFT: shift = 12  (DSPLib col raw acc ~5120 for all-{43} row input; 5120>>12=1)
 *
 * Note: DSPLib cint16 IFFT with TP_SHIFT=0 does NOT scale by N (unlike mathematical IFFT).
 * It applies fixed-point twiddle normalization per stage, giving a non-obvious scaling constant.
 * The col shift was determined empirically to make round-trip FFT→IFFT of a unit impulse give 1.
 *
 * fft_graph.h must be included before this file (provides point-size macros).
 */

#pragma once

#include "adf.h"
#include "fft_ifft_dit_1ch_graph.hpp"
#include "fft_graph.h"

using namespace adf;
namespace dsplib = xf::dsp::aie;

// ---------------------------------------------------------------
// IFFT normalization shifts (split across row and col passes)
// For 128-point: log2(128) = 7; total = 14.
// Apply entire shift on col pass so row output stays at full precision.
// ---------------------------------------------------------------
// Both overridable from the Makefile (IFFT_ROW_SHIFT / IFFT_COL_SHIFT) so the
// normalization can be swept without editing source.
//
// WARNING — the col shift is NOT a free parameter. It was calibrated so that a
// BROADBAND (impulse) spectrum round-trips to 1, which assumes the IFFT gets ~64x
// summation gain from energy spread across all bins. A narrowband / DC-concentrated
// spectrum gets no such gain, and shift=12 then destroys it outright: scenario s2
// (constant patch, DC bin only) produced an identically ZERO response.
// Real image patches are DC-dominated, so this matters beyond the tests.
//
// If you change this, regenerate the aiesim vectors with the SAME value —
// gen_aiesim_vectors.py scales every expected peak by 2^(12-shift). The Makefile
// feeds both from one variable; do not set them independently.
#ifndef FFT_2D_TP_IFFT_ROW_SHIFT
#  define FFT_2D_TP_IFFT_ROW_SHIFT 0    // no shift on row IFFT
#endif
#ifndef FFT_2D_TP_IFFT_COL_SHIFT
#  define FFT_2D_TP_IFFT_COL_SHIFT 12   // empirically calibrated for BROADBAND input only
#endif

#define FFT_2D_TP_IFFT_NIFFT       0    // 0 = inverse FFT

// Window sizes reuse FFT sizes (same point size, same window stride)
#define IFFT_ROW_WINDOW_BUFF_SIZE  FFT_ROW_WINDOW_BUFF_SIZE
#define IFFT_COL_WINDOW_BUFF_SIZE  FFT_COL_WINDOW_BUFF_SIZE

// ---------------------------------------------------------------
// IFFTrows_graph
// PATCH_COLS-point IFFT, row shift = 0.
// ---------------------------------------------------------------
class IFFTrows_graph : public graph
{
public:
    port<input>  row_in;
    port<output> row_out;

    dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
        FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
        FFT_ROW_TP_POINT_SIZE,
        FFT_2D_TP_IFFT_NIFFT,
        FFT_2D_TP_IFFT_ROW_SHIFT,
        FFT_ROW_CASCADE_LEN,
        FFT_2D_TP_DYN_PT_SIZE,
        FFT_ROW_TP_WINDOW_VSIZE> IFFTrow_gr;

    IFFTrows_graph()
    {
        runtime<ratio>(*IFFTrow_gr.getKernels()) = 0.8;

        adf::connect<window<IFFT_ROW_WINDOW_BUFF_SIZE>>(row_in,             IFFTrow_gr.in[0]);
        adf::connect<window<IFFT_ROW_WINDOW_BUFF_SIZE>>(IFFTrow_gr.out[0],  row_out);
    }
};

// ---------------------------------------------------------------
// IFFTcols_graph
// PATCH_ROWS-point IFFT, col shift = 14 (full normalization).
// ---------------------------------------------------------------
class IFFTcols_graph : public graph
{
public:
    port<input>  col_in;
    port<output> col_out;

    dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
        FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
        FFT_COL_TP_POINT_SIZE,
        FFT_2D_TP_IFFT_NIFFT,
        FFT_2D_TP_IFFT_COL_SHIFT,
        FFT_COL_CASCADE_LEN,
        FFT_2D_TP_DYN_PT_SIZE,
        FFT_COL_TP_WINDOW_VSIZE> IFFTcol_gr;

    IFFTcols_graph()
    {
        runtime<ratio>(*IFFTcol_gr.getKernels()) = 0.8;

        adf::connect<window<IFFT_COL_WINDOW_BUFF_SIZE>>(col_in,             IFFTcol_gr.in[0]);
        adf::connect<window<IFFT_COL_WINDOW_BUFF_SIZE>>(IFFTcol_gr.out[0],  col_out);
    }
};

// ---------------------------------------------------------------
// IFFT2D_graph — single instance
// Row→col path broken: APU manages DDR transpose via GMIO.
// ---------------------------------------------------------------
class IFFT2D_graph : public graph
{
public:
#if MEMTILE_TRANSPOSE
    port<input>  ifft_row_in;   // ← gmio_ifft_row_in  (accumulated spectrum from APU)
    port<output> ifft_col_out;  // → gmio_response       (final correlation response)
#else
    port<input>  ifft_row_in;   // ← gmio_ifft_row_in  (accumulated spectrum from APU)
    port<output> ifft_row_out;  // → gmio_ifft_row_out  (APU reads and transposes)
    port<input>  ifft_col_in;   // ← gmio_ifft_col_in   (APU-transposed data)
    port<output> ifft_col_out;  // → gmio_response       (final correlation response)
#endif

    IFFTrows_graph ifft_rows;
    IFFTcols_graph ifft_cols;

#if MEMTILE_TRANSPOSE
    // The inverse transpose. Identical in structure to memTileFwd — the recorded
    // claim that only the forward pass was a plain 2D-FFT transpose was wrong in
    // both directions: cmul sits DOWNSTREAM of the column FFT, so both row->col
    // paths in this design are textbook transposes. This one runs once per
    // FRAME rather than once per channel, so it is the cheaper half.
    adf::shared_buffer<FFT_2D_TT_DATA> memTileInv;
#endif

    IFFT2D_graph()
    {
#if MEMTILE_TRANSPOSE
        memTileInv = adf::shared_buffer<FFT_2D_TT_DATA>::create(
                         {PATCH_COLS, PATCH_ROWS}, 1, 1);
        adf::num_buffers(memTileInv) = 2;

        adf::write_access(memTileInv.in[0]) = adf::tiling({
            .buffer_dimension = {PATCH_COLS, PATCH_ROWS},
            .tiling_dimension = {PATCH_COLS, PATCH_ROWS},
            .offset           = {0, 0}});

        adf::read_access(memTileInv.out[0]) = adf::tiling({
            .buffer_dimension = {PATCH_COLS, PATCH_ROWS},
            .tiling_dimension = {1, 1},
            .offset           = {0, 0},
            .tile_traversal   = {{.dimension = 1, .stride = 1, .wrap = PATCH_ROWS},
                                 {.dimension = 0, .stride = 1, .wrap = PATCH_COLS}}});

        adf::connect<>(ifft_row_in,        ifft_rows.row_in);
        adf::connect<>(ifft_rows.row_out,  memTileInv.in[0]);
        adf::connect<>(memTileInv.out[0],  ifft_cols.col_in);
        adf::connect<>(ifft_cols.col_out,  ifft_col_out);
#else
        adf::connect<>(ifft_row_in,          ifft_rows.row_in);
        adf::connect<>(ifft_rows.row_out,    ifft_row_out);

        adf::connect<>(ifft_col_in,          ifft_cols.col_in);
        adf::connect<>(ifft_cols.col_out,    ifft_col_out);

        // NOTE: ifft_row_out and ifft_col_in are NOT connected here.
        // APU reads via gmio_ifft_row_out, transposes, writes via gmio_ifft_col_in.
#endif
    }
};
