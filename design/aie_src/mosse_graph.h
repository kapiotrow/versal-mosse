/*
 * mosse_graph.h
 * Top-level AIE graph for the MOSSE correlation filter tracker.
 *
 * New architecture (all compute on AIE, GMIO for DDR transfers):
 *
 *   PatchIn (PLIO) ← roi_crop PL kernel
 *     ↓
 *   conv2d_kernel  — 3×3 INT8 conv + Hanning window (stub: pass-through)
 *     ↓ (window)
 *   fft2d.fft_rows — PATCH_COLS-point row FFT (DSPLib)
 *     ↓ gmio_fft_row_out → DDR
 *     APU: transpose_inplace()
 *     DDR → gmio_fft_col_in ↓
 *   fft2d.fft_cols — PATCH_ROWS-point col FFT (DSPLib)
 *     ↓ (internal)
 *   cmul_accum_kernel — H_ch* ⊙ F_ch + accumulate (stub: pass-through)
 *     ← gmio_filter      (H_ch* from APU, per channel)
 *     ← gmio_accum_in    (previous partial sum; skipped on ch=0)
 *     ↓ gmio_accum_out → DDR
 *
 *   After all N_CHANNELS channels, APU reads accum_out and writes to
 *   gmio_ifft_row_in:
 *     DDR → gmio_ifft_row_in ↓
 *   ifft2d.ifft_rows — PATCH_COLS-point row IFFT (DSPLib)
 *     ↓ gmio_ifft_row_out → DDR
 *     APU: transpose_inplace()
 *     DDR → gmio_ifft_col_in ↓
 *   ifft2d.ifft_cols — PATCH_ROWS-point col IFFT (DSPLib)
 *     ↓ gmio_response → DDR
 *   APU: peak_detect_sw() + filter_update_kissfft()
 *
 * GMIO summary (10 ports: 6 input + 4 output):
 *   IN:  gmio_weights, gmio_fft_col_in, gmio_filter, gmio_accum_in,
 *        gmio_ifft_row_in, gmio_ifft_col_in
 *   OUT: gmio_fft_row_out, gmio_accum_out, gmio_ifft_row_out, gmio_response
 *
 * PLIO summary (1 port):
 *   IN: PatchIn
 *
 * Hardware (VEK280 AIE-ML): 32 input + 32 output GMIO available.
 * Using 10 total — well within budget.
 */

#pragma once

#include "adf.h"
#include "fft_graph.h"
#include "ifft_graph.h"
#include "conv2d_kernel.h"
#include "cmul_accum_kernel.h"

using namespace adf;

class MOSSE_graph : public graph
{
public:
    // -------------------------------------------------------
    // PLIO (1 input only)
    // -------------------------------------------------------
    input_plio  patch_in;   // "PatchIn" ← roi_crop PL kernel

    // -------------------------------------------------------
    // GMIO (6 input + 4 output = 10 total)
    // -------------------------------------------------------
    // conv2d weights: loaded once per channel before conv starts
    input_gmio  gmio_weights;

    // Forward FFT transpose scratch (shared serially across channels)
    output_gmio gmio_fft_row_out;   // row-FFT output → DDR (APU transposes)
    input_gmio  gmio_fft_col_in;    // DDR (transposed) → col-FFT input

    // cmul_accum filter and accumulator (shared serially)
    input_gmio  gmio_filter;        // H_ch* per channel ← DDR (APU writes)
    input_gmio  gmio_accum_in;      // previous partial sum ← DDR (skipped on ch=0)
    output_gmio gmio_accum_out;     // updated partial sum → DDR

    // IFFT input: APU writes accumulated spectrum after all channels
    input_gmio  gmio_ifft_row_in;   // accumulated Σ H_c*⊙F_c ← DDR

    // IFFT transpose scratch
    output_gmio gmio_ifft_row_out;  // row-IFFT output → DDR (APU transposes)
    input_gmio  gmio_ifft_col_in;   // DDR (transposed) → col-IFFT input

    // Final correlation response
    output_gmio gmio_response;      // col-IFFT output → DDR (APU reads for peak)

    // -------------------------------------------------------
    // Sub-graphs (single instances, reused serially)
    // -------------------------------------------------------
    FFT2D_graph  fft2d;
    IFFT2D_graph ifft2d;

    // -------------------------------------------------------
    // Custom kernels
    // -------------------------------------------------------
    kernel conv2d;
    kernel cmul;

    MOSSE_graph()
    {
        // --- PLIO ---
        patch_in = input_plio::create("PatchIn", plio_128_bits, "patch_in.txt");

        // --- GMIO (burst_length = 64 bytes — minimum AXI4 burst, GMIO-aligned) ---
        gmio_weights     = input_gmio::create("gmio_weights",      64);
        gmio_fft_row_out = output_gmio::create("gmio_fft_row_out", 64);
        gmio_fft_col_in  = input_gmio::create("gmio_fft_col_in",   64);
        gmio_filter      = input_gmio::create("gmio_filter",       64);
        gmio_accum_in    = input_gmio::create("gmio_accum_in",     64);
        gmio_accum_out   = output_gmio::create("gmio_accum_out",   64);
        gmio_ifft_row_in = input_gmio::create("gmio_ifft_row_in",  64);
        gmio_ifft_row_out= output_gmio::create("gmio_ifft_row_out",64);
        gmio_ifft_col_in = input_gmio::create("gmio_ifft_col_in",  64);
        gmio_response    = output_gmio::create("gmio_response",    64);

        // --- Custom kernel instantiation ---
        conv2d = kernel::create(conv2d_kernel);
        cmul   = kernel::create(cmul_accum_kernel);

        source(conv2d) = "conv2d_kernel.cpp";
        source(cmul)   = "cmul_accum_kernel.cpp";

        runtime<ratio>(conv2d) = 0.9;
        runtime<ratio>(cmul)   = 0.9;

        // --- Wiring ---

        // PatchIn → conv2d: stream (PLIO produces a stream of int8 samples)
        adf::connect<stream>(patch_in.out[0], conv2d.in[0]);

        // gmio_weights → conv2d weights buffer (in[1])
        // R4: input_buffer<int8_t> in kernel signature → window/buffer connect type
        adf::connect<>(gmio_weights.out[0], conv2d.in[1]);

        // conv2d → fft2d row-FFT input
        // DSPLib expects input_buffer (window); use explicit window connect.
        // conv2d output is a stream; ADF will insert a stream-to-window FIFO automatically.
        adf::connect<window<FFT_ROW_WINDOW_BUFF_SIZE>>(conv2d.out[0], fft2d.fft_row_in);

        // fft2d row-FFT output → GMIO (DDR, APU reads and transposes)
        adf::connect<>(fft2d.fft_row_out, gmio_fft_row_out.in[0]);

        // GMIO (APU-transposed) → fft2d col-FFT input
        adf::connect<>(gmio_fft_col_in.out[0], fft2d.fft_col_in);

        // fft2d col-FFT output → cmul kernel (stream)
        adf::connect<stream>(fft2d.fft_col_out, cmul.in[0]);

        // gmio_filter → cmul filter buffer (in[1])
        adf::connect<>(gmio_filter.out[0], cmul.in[1]);

        // gmio_accum_in → cmul previous accumulator (in[2]; skipped ch=0 at runtime)
        adf::connect<>(gmio_accum_in.out[0], cmul.in[2]);

        // cmul output → gmio_accum_out (DDR)
        adf::connect<>(cmul.out[0], gmio_accum_out.in[0]);

        // IFFT: APU reads gmio_accum_out, writes accumulated spectrum to gmio_ifft_row_in
        adf::connect<>(gmio_ifft_row_in.out[0],  ifft2d.ifft_row_in);

        // ifft2d row-IFFT output → GMIO (APU reads and transposes)
        adf::connect<>(ifft2d.ifft_row_out, gmio_ifft_row_out.in[0]);

        // GMIO (APU-transposed) → ifft2d col-IFFT input
        adf::connect<>(gmio_ifft_col_in.out[0], ifft2d.ifft_col_in);

        // ifft2d col-IFFT output → gmio_response (APU reads for peak detection)
        adf::connect<>(ifft2d.ifft_col_out, gmio_response.in[0]);
    }
};
