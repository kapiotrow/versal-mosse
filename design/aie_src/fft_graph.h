/*
 * fft_graph.h
 * AIE FFT2D graph for the MOSSE tracker feature pipeline.
 *
 * Wraps DSPLib fft_ifft_dit_1ch_graph for the forward FFT pass.
 * Instantiated N_CHANNELS times (one per feature channel).
 * Processed serially: fmt_adapter feeds one channel at a time.
 *
 * Row FFT point size = PATCH_COLS
 * Col FFT point size = PATCH_ROWS
 * Both are equal for square patches (default 128x128).
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
#define FFT_COL_TP_POINT_SIZE  PATCH_ROWS   // 1D FFT across rows (one col at a time)

#define FFT_2D_TP_FFT_NIFFT    1            // 1 = forward FFT
#define FFT_2D_TP_SHIFT        0            // no shift for forward FFT
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
// External instance counters (defined in mosse_graph.cpp)
// ---------------------------------------------------------------
extern uint8_t fftRows_grInsts, fftCols_grInsts;

// ---------------------------------------------------------------
// FFTrows_graph
// One instance per feature channel; processes PATCH_COLS-point FFT
// across each row of the patch.
// PLIO name pattern: "FFTRowIn{ch}", "FFTRowOut{ch}"
// ---------------------------------------------------------------
class FFTrows_graph : public graph
{
public:
    input_plio  row_in;
    output_plio row_out;

    FFTrows_graph()
    {
        dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
            FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
            FFT_ROW_TP_POINT_SIZE,
            FFT_2D_TP_FFT_NIFFT,
            FFT_2D_TP_SHIFT,
            FFT_ROW_CASCADE_LEN,
            FFT_2D_TP_DYN_PT_SIZE,
            FFT_ROW_TP_WINDOW_VSIZE> FFTrow_gr;

        runtime<ratio>(*FFTrow_gr.getKernels()) = 0.8;

        std::string in_name  = "FFTRowIn"  + std::to_string(fftRows_grInsts);
        std::string out_name = "FFTRowOut" + std::to_string(fftRows_grInsts);
        std::string in_file  = "fft_row_in_"  + std::to_string(fftRows_grInsts) + ".txt";
        std::string out_file = "data/fft_row_out_" + std::to_string(fftRows_grInsts) + ".txt";

        row_in  = input_plio::create(in_name.c_str(),  plio_128_bits, in_file.c_str());
        row_out = output_plio::create(out_name.c_str(), plio_128_bits, out_file.c_str());

        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(row_in.out[0],    FFTrow_gr.in[0]);
        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(FFTrow_gr.out[0], row_out.in[0]);

        ++fftRows_grInsts;
    }
};

// ---------------------------------------------------------------
// FFTcols_graph
// One instance per feature channel; processes PATCH_ROWS-point FFT
// across each column of the patch (fed transpose data from PL).
// PLIO name pattern: "FFTColIn{ch}", "FFTColOut{ch}"
// ---------------------------------------------------------------
class FFTcols_graph : public graph
{
public:
    input_plio  col_in;
    output_plio col_out;

    FFTcols_graph()
    {
        dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
            FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
            FFT_COL_TP_POINT_SIZE,
            FFT_2D_TP_FFT_NIFFT,
            FFT_2D_TP_SHIFT,
            FFT_COL_CASCADE_LEN,
            FFT_2D_TP_DYN_PT_SIZE,
            FFT_COL_TP_WINDOW_VSIZE> FFTcol_gr;

        runtime<ratio>(*FFTcol_gr.getKernels()) = 0.8;

        std::string in_name  = "FFTColIn"  + std::to_string(fftCols_grInsts);
        std::string out_name = "FFTColOut" + std::to_string(fftCols_grInsts);
        std::string in_file  = "fft_col_in_"  + std::to_string(fftCols_grInsts) + ".txt";
        std::string out_file = "data/fft_col_out_" + std::to_string(fftCols_grInsts) + ".txt";

        col_in  = input_plio::create(in_name.c_str(),  plio_128_bits, in_file.c_str());
        col_out = output_plio::create(out_name.c_str(), plio_128_bits, out_file.c_str());

        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(col_in.out[0],    FFTcol_gr.in[0]);
        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(FFTcol_gr.out[0], col_out.in[0]);

        ++fftCols_grInsts;
    }
};

// ---------------------------------------------------------------
// FFT2D_graph
// One FFTrows_graph + one FFTcols_graph for a single feature channel.
// MOSSE_graph instantiates this N_CHANNELS times (fft2d[N_CHANNELS]).
// Serial processing: the PL fmt_adapter feeds one channel at a time;
// the PL transpose buffer sits between row and col outputs.
// ---------------------------------------------------------------
class FFT2D_graph : public graph
{
public:
    FFTrows_graph fft_rows;
    FFTcols_graph fft_cols;
};
