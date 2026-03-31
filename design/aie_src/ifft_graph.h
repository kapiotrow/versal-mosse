/*
 * ifft_graph.h
 * AIE IFFT2D graph for the MOSSE tracker response computation.
 *
 * Single instance — applied after the PL cmul_accum kernel has
 * accumulated the element-wise product  Σ_c H_c* ⊙ FFT(f_c)
 * across all N_CHANNELS feature channels.
 *
 * Point sizes are identical to the FFT graph (same patch dimensions).
 * The IFFT normalization shift: log2(PATCH_COLS) for row IFFT and
 * log2(PATCH_ROWS) for col IFFT.  Adjust FFT_2D_TP_IFFT_SHIFT below
 * to match the DSPLib convention (shift is applied per stage).
 */

#pragma once

#include "adf.h"
#include "fft_ifft_dit_1ch_graph.hpp"

using namespace adf;
namespace dsplib = xf::dsp::aie;

// Reuse point-size and type macros from fft_graph.h — include order matters.
// fft_graph.h must be included before ifft_graph.h.

#define FFT_2D_TP_IFFT_NIFFT   0    // 0 = inverse FFT
// TODO: set shift to log2(PATCH_COLS) + log2(PATCH_ROWS) split appropriately
// For 128-point: log2(128)=7; typical split: row shift=0, col shift=7 (or both=0 and
// normalise in cmul_accum). Start with 0 and verify numerically.
#define FFT_2D_TP_IFFT_SHIFT   0

extern uint8_t ifftRows_grInsts, ifftCols_grInsts;

// ---------------------------------------------------------------
// IFFTrows_graph
// Applies PATCH_COLS-point IFFT across each row of the accumulated
// frequency-domain response map.
// PLIO: "IFFTRowIn0", "IFFTRowOut0"
// ---------------------------------------------------------------
class IFFTrows_graph : public graph
{
public:
    input_plio  row_in;
    output_plio row_out;

    IFFTrows_graph()
    {
        dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
            FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
            FFT_ROW_TP_POINT_SIZE,
            FFT_2D_TP_IFFT_NIFFT,
            FFT_2D_TP_IFFT_SHIFT,
            FFT_ROW_CASCADE_LEN,
            FFT_2D_TP_DYN_PT_SIZE,
            FFT_ROW_TP_WINDOW_VSIZE> IFFTrow_gr;

        runtime<ratio>(*IFFTrow_gr.getKernels()) = 0.8;

        std::string in_name  = "IFFTRowIn"  + std::to_string(ifftRows_grInsts);
        std::string out_name = "IFFTRowOut" + std::to_string(ifftRows_grInsts);
        std::string in_file  = "ifft_row_in_"  + std::to_string(ifftRows_grInsts) + ".txt";
        std::string out_file = "data/ifft_row_out_" + std::to_string(ifftRows_grInsts) + ".txt";

        row_in  = input_plio::create(in_name.c_str(),  plio_128_bits, in_file.c_str());
        row_out = output_plio::create(out_name.c_str(), plio_128_bits, out_file.c_str());

        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(row_in.out[0],     IFFTrow_gr.in[0]);
        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(IFFTrow_gr.out[0], row_out.in[0]);

        ++ifftRows_grInsts;
    }
};

// ---------------------------------------------------------------
// IFFTcols_graph
// Applies PATCH_ROWS-point IFFT across each column of the row-IFFT
// output (fed transposed data from PL).
// PLIO: "IFFTColIn0", "IFFTColOut0"
// ---------------------------------------------------------------
class IFFTcols_graph : public graph
{
public:
    input_plio  col_in;
    output_plio col_out;

    IFFTcols_graph()
    {
        dsplib::fft::dit_1ch::fft_ifft_dit_1ch_graph<
            FFT_2D_TT_DATA, FFT_2D_TT_TWIDDLE,
            FFT_COL_TP_POINT_SIZE,
            FFT_2D_TP_IFFT_NIFFT,
            FFT_2D_TP_IFFT_SHIFT,
            FFT_COL_CASCADE_LEN,
            FFT_2D_TP_DYN_PT_SIZE,
            FFT_COL_TP_WINDOW_VSIZE> IFFTcol_gr;

        runtime<ratio>(*IFFTcol_gr.getKernels()) = 0.8;

        std::string in_name  = "IFFTColIn"  + std::to_string(ifftCols_grInsts);
        std::string out_name = "IFFTColOut" + std::to_string(ifftCols_grInsts);
        std::string in_file  = "ifft_col_in_"  + std::to_string(ifftCols_grInsts) + ".txt";
        std::string out_file = "data/ifft_col_out_" + std::to_string(ifftCols_grInsts) + ".txt";

        col_in  = input_plio::create(in_name.c_str(),  plio_128_bits, in_file.c_str());
        col_out = output_plio::create(out_name.c_str(), plio_128_bits, out_file.c_str());

        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(col_in.out[0],     IFFTcol_gr.in[0]);
        adf::connect<window<FFT_COL_WINDOW_BUFF_SIZE>>(IFFTcol_gr.out[0], col_out.in[0]);

        ++ifftCols_grInsts;
    }
};

// ---------------------------------------------------------------
// IFFT2D_graph — single instance
// ---------------------------------------------------------------
class IFFT2D_graph : public graph
{
public:
    IFFTrows_graph ifft_rows;
    IFFTcols_graph ifft_cols;
};
