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
 *
 * ---------------------------------------------------------------------------
 * COST — the forward and inverse chains together are ~2.2 ms/frame (band
 * 1.3-3.5) at 128x128, ch16. docs/thesis/results/aie_compute.csv. Claim A-01.
 *
 *   caveat        THE TRIP COUNTS ARE INFERRED, NOT LOGGED. Every other row of
 *                 that table is read off a compiler schedule; this one is not,
 *                 and the band is the honest uncertainty. Do not quote 2.2 as
 *                 though it were measured.
 *   windowing     FFT_ROW_WS rows and FFT_COL_WS cols per invocation — the DMA
 *                 transaction-count knob, which took total traffic from 4258 to
 *                 1090 tx/frame. FFT_ROW_WS=64 is exhausted. FFT_COL_WS=32 is a
 *                 9.57 ms LOSS and must not be raised (claim P-08).
 *                 A windowing result from one port does NOT generalise to
 *                 another: the identical change was 7x better on one GMIO and 4x
 *                 worse on its sibling.
 *   memory tiles  1 of 76 used, for the transpose between the row and column
 *                 passes (MEMTILE_TRANSPOSE=1, which also deleted four GMIO
 *                 ports). A ONE-SIDED flag is a board deadlock, not a compile
 *                 error. docs/thesis/results/resources.csv.
 *   numerics      DSPLib's cint16 FFT loss is ADDITIVE, not a gain factor: each
 *                 pass subtracts ~21 from a summed DC bin, independent of
 *                 amplitude. So row_dc = PATCH_COLS*c - 21 and accum0 =
 *                 PATCH_ROWS*row_dc - 21; any "expected = N" calculation is
 *                 wrong. The fixed-point transform is also NOT Hermitian for
 *                 real input (claim P-10) — 95.8% of bins differ from their
 *                 conjugate partner, so the host filter cannot be halved.
 *
 * @thesis subsec:fftAie | A-01 | Forward 2-D FFT: PATCH_COLS-point row FFT, memory-tile
 *   transpose, PATCH_ROWS-point column FFT, built on DSPLib fft_ifft_dit_1ch.
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
#define FFT_2D_TP_DYN_PT_SIZE  0            // fixed point size

// Forward-FFT output shift. Overridable from the Makefile (FFT_SHIFT) so the
// normalization can be swept without editing source — same pattern as
// FFT_ROW_CASCADE_LEN below. Default 0: no shift on the forward pass, so the
// spectrum keeps full precision and all normalization happens on the IFFT side.
#ifndef FFT_2D_TP_SHIFT
#  define FFT_2D_TP_SHIFT      0
#endif

// Cascade lengths (increase for cfloat or large point sizes)
#ifndef FFT_ROW_CASCADE_LEN
#  define FFT_ROW_CASCADE_LEN  1
#endif
#ifndef FFT_COL_CASCADE_LEN
#  define FFT_COL_CASCADE_LEN  1
#endif

// @thesis sec:wydajnoscZasoby | P-08 | FFT_COL_WS=32 was built and measured: a
//   9.57 ms LOSS. The window that wins on one GMIO port loses on its sibling, so a windowing
//   result does not generalise without a hardware run.
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
// MEMTILE_TRANSPOSE: do the row->col transpose in an AIE-ML memory tile instead
// of a DDR round trip through the APU. Off by default — the DDR path is what
// every recorded run used. See CLAUDE.md, "Memory-tile transpose".
#ifndef MEMTILE_TRANSPOSE
#  define MEMTILE_TRANSPOSE 0
#endif

class FFT2D_graph : public graph
{
public:
#if MEMTILE_TRANSPOSE
    // TWO external ports. The row->col path closes INSIDE the graph, so
    // fft_row_out and fft_col_in (and their two GMIOs) do not exist.
    port<input>  fft_row_in;   // ← conv2d_kernel output (window)
    port<output> fft_col_out;  // → cmul_accum_kernel input (internal)
#else
    // Four external ports — all wired from the parent MOSSE_graph
    port<input>  fft_row_in;   // ← conv2d_kernel output (window)
    port<output> fft_row_out;  // → gmio_fft_row_out (DDR, APU reads and transposes)
    port<input>  fft_col_in;   // ← gmio_fft_col_in  (APU-transposed data)
    port<output> fft_col_out;  // → cmul_accum_kernel input (internal)
#endif

    FFTrows_graph fft_rows;
    FFTcols_graph fft_cols;

#if MEMTILE_TRANSPOSE
    // The transpose itself. Ported from DSPLib's own 2D FFT — see
    // Vitis_Libraries/dsp/L2/include/aie/fft_ifft_2d_graph.hpp:242-260, which
    // puts exactly this between its front and back FFT. Write the plane
    // contiguously, read it with the traversal dimensions swapped; walking
    // dimension 1 in the INNER loop is what makes the read a transpose.
    //
    // buffer_dimension[0] is the contiguous dimension (adf/types.h:393). The row
    // FFT emits each row's PATCH_COLS bins contiguously, so dimension 0 is COLS
    // and dimension 1 is ROWS — the same {D1, D2} = {front point size, back
    // point size} convention DSPLib uses.
    //
    // 128x128 cint16 = 64 KB, 128 KB with the ping-pong, against a 512 KB
    // AIE-ML memory tile. num_buffers = 2 is load-bearing and not an
    // optimisation: a transpose is a global dependency (the column pass cannot
    // start until the whole plane is written), so without ping-pong this
    // serialises the channels harder than the DDR path it replaces.
    adf::shared_buffer<FFT_2D_TT_DATA> memTileFwd;
#endif

    FFT2D_graph()
    {
#if MEMTILE_TRANSPOSE
        memTileFwd = adf::shared_buffer<FFT_2D_TT_DATA>::create(
                         {PATCH_COLS, PATCH_ROWS}, 1, 1);
        adf::num_buffers(memTileFwd) = 2;

        adf::write_access(memTileFwd.in[0]) = adf::tiling({
            .buffer_dimension = {PATCH_COLS, PATCH_ROWS},
            .tiling_dimension = {PATCH_COLS, PATCH_ROWS},
            .offset           = {0, 0}});

        adf::read_access(memTileFwd.out[0]) = adf::tiling({
            .buffer_dimension = {PATCH_COLS, PATCH_ROWS},
            .tiling_dimension = {1, 1},
            .offset           = {0, 0},
            .tile_traversal   = {{.dimension = 1, .stride = 1, .wrap = PATCH_ROWS},
                                 {.dimension = 0, .stride = 1, .wrap = PATCH_COLS}}});

        adf::connect<>(fft_row_in,        fft_rows.row_in);
        adf::connect<>(fft_rows.row_out,  memTileFwd.in[0]);
        adf::connect<>(memTileFwd.out[0], fft_cols.col_in);
        adf::connect<>(fft_cols.col_out,  fft_col_out);
#else
        adf::connect<>(fft_row_in,        fft_rows.row_in);
        adf::connect<>(fft_rows.row_out,  fft_row_out);

        adf::connect<>(fft_col_in,        fft_cols.col_in);
        adf::connect<>(fft_cols.col_out,  fft_col_out);

        // NOTE: fft_row_out and fft_col_in are NOT connected to each other here.
        // The APU reads fft_row_out via gmio_fft_row_out, transposes the matrix
        // in DDR, then writes to fft_col_in via gmio_fft_col_in.
#endif
    }
};
