/*
 * kernel_only_graph.h
 * Minimal single-kernel graphs for BIT-EXACTNESS testing under x86sim.
 *
 * Why this exists
 * ---------------
 * Before this harness there was no cheap way to check that a change to conv2d or
 * cmul_accum preserves the arithmetic. The options were aiesim (hours, and it
 * asserts tolerances on an END-TO-END response, not the kernel's own output) or
 * hw_emu (~24 h at the ch16 design point). Neither can answer "did this rewrite
 * change any sample?", which is the only question that matters when vectorizing.
 *
 * So: isolate ONE kernel, feed it scenario data, dump its raw output, and diff
 * that against the Python model in gen_aiesim_vectors.py. Under --target=x86sim
 * the whole loop is seconds.
 *
 * It also validates the Python model itself. gen_aiesim_vectors.py:205 claims
 * simulate_conv2d() "replicates the integer arithmetic in conv2d_kernel.cpp
 * exactly" — a docstring that has never been checked against the kernel, and the
 * offline shift-budget work all rests on it. Running this harness against the
 * UNMODIFIED scalar kernel is therefore a real test, not a warm-up.
 *
 * Topology (selected by KERNEL_UNDER_TEST)
 * ----------------------------------------
 *   KUT_CONV2D:  PatchIn (PLIO) -> conv2d -> gmio_kernel_out
 *                gmio_weights   -> conv2d.in[1]
 *
 *   KUT_CMUL:    gmio_fft_col_in -> cmul.in[0]  (a GMIO here; a tile-to-tile
 *                                                window in the real graph)
 *                gmio_cmul_in    -> cmul.in[1]
 *                cmul.out[0]     -> gmio_kernel_out
 *
 * No DSPLib, no FFT, no second kernel: whatever comes out of gmio_kernel_out was
 * produced by the kernel under test and nothing else.
 *
 * Port order for cmul is preserved from mosse_graph.h: fft_col_in MUST be in[0].
 * cmul_accum_kernel.h documents why, and keeping the harness's wiring identical
 * to production is the point — a harness that wires it differently is testing a
 * different kernel.
 */

#pragma once

#include "adf.h"
#include "conv2d_kernel.h"
#include "cmul_accum_kernel.h"

using namespace adf;

#define KUT_CONV2D 0
#define KUT_CMUL   1

#ifndef KERNEL_UNDER_TEST
#  define KERNEL_UNDER_TEST KUT_CONV2D
#endif

#if KERNEL_UNDER_TEST == KUT_CONV2D

class KernelOnly_graph : public graph
{
public:
    input_plio  patch_in;
    input_gmio  gmio_weights;
    output_gmio gmio_kernel_out;

    kernel conv2d;

    KernelOnly_graph()
    {
        // Same PLIO width and stimulus filename as mosse_graph.h — a 32-bit PLIO
        // carrying 4 packed int8 per beat. Changing it here would test a
        // different unpack path than production runs.
        patch_in        = input_plio::create("PatchIn", plio_32_bits, "patch_in.txt");
        gmio_weights    = input_gmio::create("gmio_weights",    64, 1000);
        gmio_kernel_out = output_gmio::create("gmio_kernel_out", 64, 1000);

        conv2d = kernel::create(conv2d_kernel);
        source(conv2d) = "conv2d_kernel.cpp";
        runtime<ratio>(conv2d) = 0.9;

        adf::connect<stream>(patch_in.out[0], conv2d.in[0]);

        adf::connect<>(gmio_weights.out[0], conv2d.in[1]);
        adf::dimensions(conv2d.in[1]) = {CONV_WEIGHT_BYTES_PAD};

        adf::connect<>(conv2d.out[0], gmio_kernel_out.in[0]);
        adf::dimensions(conv2d.out[0]) = {CONV_OUT_CHUNK};
    }
};

#else  // KERNEL_UNDER_TEST == KUT_CMUL

class KernelOnly_graph : public graph
{
public:
    input_gmio  gmio_fft_col_in;
    input_gmio  gmio_cmul_in;
    output_gmio gmio_kernel_out;

    kernel cmul;

    KernelOnly_graph()
    {
        gmio_fft_col_in = input_gmio::create("gmio_fft_col_in", 64, 1000);
        gmio_cmul_in    = input_gmio::create("gmio_cmul_in",    64, 1000);
        gmio_kernel_out = output_gmio::create("gmio_kernel_out", 64, 1000);

        cmul = kernel::create(cmul_accum_kernel);
        source(cmul) = "cmul_accum_kernel.cpp";
        runtime<ratio>(cmul) = 0.9;

        // in[0] first — see the port-order note in cmul_accum_kernel.h.
        adf::connect<>(gmio_fft_col_in.out[0], cmul.in[0]);
        adf::dimensions(cmul.in[0]) = {PATCH_COLS * FFT_COL_WS};

        adf::connect<>(gmio_cmul_in.out[0], cmul.in[1]);
        adf::dimensions(cmul.in[1]) = {2 * PATCH_COLS * FFT_COL_WS};

        adf::connect<>(cmul.out[0], gmio_kernel_out.in[0]);
        adf::dimensions(cmul.out[0]) = {PATCH_COLS * FFT_COL_WS};
    }
};

#endif // KERNEL_UNDER_TEST
