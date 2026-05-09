/*
 * fft_only_graph.h
 * Minimal graph for isolated FFT row-kernel smoke test.
 *
 * Topology:
 *   gmio_fft_in → FFTrows_graph → gmio_fft_out
 *
 * No PLIO, no custom kernels.  Used to verify the DSPLib FFT works in
 * aiesim before adding the PLIO + conv2d/cmul complexity back in.
 */

#pragma once

#include "adf.h"
#include "fft_graph.h"

using namespace adf;

class FFTOnly_graph : public graph
{
public:
    input_gmio  gmio_fft_in;
    output_gmio gmio_fft_out;

    FFTrows_graph fft_rows;

    FFTOnly_graph()
    {
        gmio_fft_in  = input_gmio::create("gmio_fft_in",  64, 1000);
        gmio_fft_out = output_gmio::create("gmio_fft_out", 64, 1000);

        adf::connect<>(gmio_fft_in.out[0], fft_rows.row_in);
        adf::connect<>(fft_rows.row_out,   gmio_fft_out.in[0]);
    }
};
