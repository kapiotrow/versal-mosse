/*
 * fft_graph.h
 * AIE FFT2D graph for the MOSSE tracker feature pipeline.
 *
 * Wraps DSPLib fft_ifft_dit_1ch_graph for the forward FFT pass.
 * A single FFT2D_graph instance is reused for all N_CHANNELS channels
 * (serial processing driven by the APU via GMIO).
 *
 * The row→col connection is intentionally broken at the graph boundary:
 *   fft_row_out → GMIO → DDR   (APU reads, transposes in software)
 *   DDR → GMIO → fft_col_in    (APU writes transposed data)
 * This avoids the need for a PL transpose kernel.
 *
 * FFT2D_graph exposes four ports:
 *   fft_row_in   ← conv2d_kernel output (window connection from MOSSE_graph)
 *   fft_row_out  → gmio_fft_row_out    (DDR, APU transposes)
 *   fft_col_in   ← gmio_fft_col_in     (DDR, APU-transposed data)
 *   fft_col_out  → cmul_accum_kernel   (internal connection in MOSSE_graph)
 *
 * Row FFT point size = PATCH_COLS
 * Col FFT point size = PATCH_ROWS
 * Both equal for square patches (default 128×128).
 */

#pragma once

#include "adf.h"
#include "fft_ifft_dit_1ch_graph.hpp"

using namespace adf;
namespace dsplib = xf::dsp::aie;

// ---------------------------------------------------------------
// FFT point sizes derived from patch dimensions
// ---------------------------------------------------------------
#define FFT_ROW_TP_POINT_SIZE  PATCH_COLS   // 1D FFT across columns (one row at a time)
#define FFT_COL_TP_POINT_SIZE  PATCH_ROWS   // 1D FFT across rows    (one col at a time)

#define FFT_2D_TP_FFT_NIFFT    1            // 1 = forward FFT
#define FFT_2D_TP_SHIFT        0            // no shift on forward pass
#define FFT_2D_TP_DYN_PT_SIZE  0            // fixed point size

// Cascade lengths (increase for cfloat or large point sizes)
#ifndef FFT_ROW_CASCADE_LEN
#  define FFT_ROW_CASCADE_LEN  1
#endif
#ifndef FFT_COL_CASCADE_LEN
#  define FFT_COL_CASCADE_LEN  1
#endif

// Window sizes: process FFT_ROW_WS rows / FFT_COL_WS cols per kernel invocation
#ifndef FFT_ROW_WS
#  define FFT_ROW_WS  2
#endif
#ifndef FFT_COL_WS
#  define FFT_COL_WS  2
#endif

#define FFT_ROW_TP_WINDOW_VSIZE  (PATCH_ROWS * FFT_ROW_WS)
#define FFT_COL_TP_WINDOW_VSIZE  (PATCH_COLS * FFT_COL_WS)

// ---------------------------------------------------------------
// Data / twiddle types
// ---------------------------------------------------------------
#if FFT_2D_DT == 0
#  define FFT_2D_TT_DATA    cint16
#  define FFT_2D_TT_TWIDDLE cint16
#  define FFT_SAMPLE_BYTES  4
#else
#  define FFT_2D_TT_DATA    cfloat
#  define FFT_2D_TT_TWIDDLE cfloat
#  define FFT_SAMPLE_BYTES  8
#endif

#define FFT_ROW_WINDOW_BUFF_SIZE  (FFT_ROW_TP_WINDOW_VSIZE * FFT_SAMPLE_BYTES)
#define FFT_COL_WINDOW_BUFF_SIZE  (FFT_COL_TP_WINDOW_VSIZE * FFT_SAMPLE_BYTES)

// ---------------------------------------------------------------
// FFTrows_graph
// PATCH_COLS-point FFT, one row at a time.
// Exposes port<input> / port<output> for wiring in FFT2D_graph.
// ---------------------------------------------------------------
class FFTrows_graph : public graph
{
public:
    port<input>  row_in;
    port<output> row_out;

    dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
        FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
        FFT_ROW_TP_POINT_SIZE,
        FFT_2D_TP_FFT_NIFFT,
        FFT_2D_TP_SHIFT,
        FFT_ROW_CASCADE_LEN,
        FFT_2D_TP_DYN_PT_SIZE,
        FFT_ROW_TP_WINDOW_VSIZE> FFTrow_gr;

    FFTrows_graph()
    {
        runtime<ratio>(*FFTrow_gr.getKernels()) = 0.8;

        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(row_in,            FFTrow_gr.in[0]);
        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(FFTrow_gr.out[0],  row_out);
    }
};

// ---------------------------------------------------------------
// FFTcols_graph
// PATCH_ROWS-point FFT, one column at a time (fed transposed data from DDR).
// Exposes port<input> / port<output> for wiring in FFT2D_graph.
// ---------------------------------------------------------------
class FFTcols_graph : public graph
{
public:
    port<input>  col_in;
    port<output> col_out;

    dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
        FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
        FFT_COL_TP_POINT_SIZE,
        FFT_2D_TP_FFT_NIFFT,
        FFT_2D_TP_SHIFT,
        FFT_COL_CASCADE_LEN,
        FFT_2D_TP_DYN_PT_SIZE,
        FFT_COL_TP_WINDOW_VSIZE> FFTcol_gr;

    FFTcols_graph()
    {
        runtime<ratio>(*FFTcol_gr.getKernels()) = 0.8;

        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(col_in,            FFTcol_gr.in[0]);
        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(FFTcol_gr.out[0],  col_out);
    }
};

// ---------------------------------------------------------------
// FFT2D_graph
// Combines FFTrows_graph + FFTcols_graph.
// The row→col path is broken: row_out and col_in are separate external
// ports wired to GMIO in MOSSE_graph (APU manages the DDR transpose).
// ---------------------------------------------------------------
class FFT2D_graph : public graph
{
public:
    // Four external ports — all wired from the parent MOSSE_graph
    port<input>  fft_row_in;   // ← conv2d_kernel output (window)
    port<output> fft_row_out;  // → gmio_fft_row_out (DDR, APU reads and transposes)
    port<input>  fft_col_in;   // ← gmio_fft_col_in  (APU-transposed data)
    port<output> fft_col_out;  // → cmul_accum_kernel input (internal)

    FFTrows_graph fft_rows;
    FFTcols_graph fft_cols;

    FFT2D_graph()
    {
        adf::connect<>(fft_row_in,        fft_rows.row_in);
        adf::connect<>(fft_rows.row_out,  fft_row_out);

        adf::connect<>(fft_col_in,        fft_cols.col_in);
        adf::connect<>(fft_cols.col_out,  fft_col_out);

        // NOTE: fft_row_out and fft_col_in are NOT connected to each other here.
        // The APU reads fft_row_out via gmio_fft_row_out, transposes the matrix
        // in DDR, then writes to fft_col_in via gmio_fft_col_in.
    }
};
