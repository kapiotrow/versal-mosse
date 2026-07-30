/*
 * mosse_graph.h
 * Top-level AIE graph for the MOSSE correlation filter tracker.
 *
 * New architecture (all compute on AIE, GMIO for DDR transfers):
 *
 *   PatchIn (PLIO) ← roi_crop PL kernel
 *     ↓
 *   conv2d_kernel  — 3×3 INT8 conv + separable Hanning window (int8 → cint16)
 *     ↓ (window)
 *   fft2d.fft_rows — PATCH_COLS-point row FFT (DSPLib)
 *     ↓ gmio_fft_row_out → DDR
 *     APU: transpose_inplace()
 *     DDR → gmio_fft_col_in ↓
 *   fft2d.fft_cols — PATCH_ROWS-point col FFT (DSPLib)
 *     ↓ (internal)
 *   cmul_accum_kernel — H_ch* ⊙ F_ch + accumulate
 *     ← gmio_cmul_in     ([H_ch* | prev_Σ] packed, per channel)
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
 * GMIO summary (9 ports: 5 input + 4 output):
 *   IN:  gmio_weights, gmio_fft_col_in, gmio_cmul_in,
 *        gmio_ifft_row_in, gmio_ifft_col_in
 *   OUT: gmio_fft_row_out, gmio_accum_out, gmio_ifft_row_out, gmio_response
 *
 * gmio_cmul_in carries [H_ch* | prev_Σ] packed per chunk — see cmul_accum_kernel.h.
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

    // cmul_accum combined input and output (shared serially)
    input_gmio  gmio_cmul_in;       // [H_ch* | prev_Σ] packed per chunk ← DDR (APU writes)
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
        // 32-bit width so conv2d's input_stream<int32> reads are 1:1 with PLIO
        // beats (a 128-bit PLIO delivered one 128-bit beat per readincr, starving
        // conv2d 4:1). roi_crop emits 32-bit AXIS beats (4 packed int8 each).
        patch_in = input_plio::create("PatchIn", plio_32_bits, "patch_in.txt");

        // --- GMIO (burst_length = 64 bytes, bandwidth = 1000 MB/s estimate) ---
        gmio_weights     = input_gmio::create("gmio_weights",      64, 1000);
        gmio_fft_row_out = output_gmio::create("gmio_fft_row_out", 64, 1000);
        gmio_fft_col_in  = input_gmio::create("gmio_fft_col_in",   64, 1000);
        gmio_cmul_in     = input_gmio::create("gmio_cmul_in",       64, 1000);
        gmio_accum_out   = output_gmio::create("gmio_accum_out",   64, 1000);
        gmio_ifft_row_in = input_gmio::create("gmio_ifft_row_in",  64, 1000);
        gmio_ifft_row_out= output_gmio::create("gmio_ifft_row_out",64, 1000);
        gmio_ifft_col_in = input_gmio::create("gmio_ifft_col_in",  64, 1000);
        gmio_response    = output_gmio::create("gmio_response",    64, 1000);

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
        adf::connect<>(gmio_weights.out[0], conv2d.in[1]);
        adf::dimensions(conv2d.in[1]) = {CONV_WEIGHT_BYTES_PAD};  // 64 int8_t elements

        // conv2d (window) → fft2d row-FFT input (window)
        // conv2d now emits an output_buffer of exactly CONV_OUT_CHUNK samples =
        // one row-FFT input window, so this is a direct window→window link with NO
        // stream→window adapter (that adapter is the suspected hw_emu hang).
        adf::connect<>(conv2d.out[0], fft2d.fft_row_in);
        adf::dimensions(conv2d.out[0]) = {CONV_OUT_CHUNK};

        // fft2d row-FFT output → GMIO (DDR, APU reads and transposes)
        adf::connect<>(fft2d.fft_row_out, gmio_fft_row_out.in[0]);

        // GMIO (APU-transposed) → fft2d col-FFT input
        adf::connect<>(gmio_fft_col_in.out[0], fft2d.fft_col_in);

        // fft2d col-FFT output → cmul.in[0] (fft_col_in, acquired 1st — must be first)
        // fft_col_out is a tile-to-tile window port; its lock is set by the col-FFT tile
        // (22_0), not by a GMIO DMA.  In Vitis 2025.2 cycle-approximate aiesim, INPUT
        // GMIO locks on this tile (filter/accum) only deliver correctly after the
        // tile-to-tile lock fires first in the iteration — see cmul_accum_kernel.h note.
        adf::connect<>(fft2d.fft_col_out, cmul.in[0]);
        adf::dimensions(cmul.in[0]) = {PATCH_COLS * FFT_COL_WS};

        // gmio_cmul_in → cmul.in[1] ([filter | accum_prev] packed; single GMIO lock)
        adf::connect<>(gmio_cmul_in.out[0], cmul.in[1]);
        adf::dimensions(cmul.in[1]) = {2 * PATCH_COLS * FFT_COL_WS};

        // cmul output → gmio_accum_out (DDR)
        adf::connect<>(cmul.out[0], gmio_accum_out.in[0]);
        adf::dimensions(cmul.out[0]) = {PATCH_COLS * FFT_COL_WS};

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
