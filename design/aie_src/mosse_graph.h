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
 *     ↓ gmio_fft_col_out → DDR   (broadcast tap: F_ch for the PS filter update)
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
 *   APU: peak_detect_sw() + filter_init/filter_update (mosse_filter.h)
 *
 * GMIO summary (10 ports: 5 input + 5 output):
 *   IN:  gmio_weights, gmio_fft_col_in, gmio_cmul_in,
 *        gmio_ifft_row_in, gmio_ifft_col_in
 *   OUT: gmio_fft_row_out, gmio_fft_col_out, gmio_accum_out,
 *        gmio_ifft_row_out, gmio_response
 *
 * gmio_fft_col_out is a broadcast tap on the col-FFT output carrying the full
 * 2-D spectrum F_ch. The PS-side filter update needs it; nothing else consumes it.
 *
 * gmio_cmul_in carries [H_ch* | prev_Σ] packed per chunk — see cmul_accum_kernel.h.
 *
 * PLIO summary (1 port):
 *   IN: PatchIn
 *
 * Hardware (VEK280 AIE-ML): 32 input + 32 output GMIO available.
 * Using 10 total — well within budget.
 *
 * ---------------------------------------------------------------------------
 * COST — the whole design, per frame at 128x128, ch16. AIE core clock 1 GHz, PL
 * 312.5 MHz. docs/thesis/results/{aie_compute,resources,frame_budget}.csv.
 * Claims A-01, A-02, P-01, P-02.
 *
 *   AIE compute   ~6.4 ms/frame total, from ~21.6 before the kernels were
 *                 vectorized. conv2d is 4.1 of it, cmul_accum 0.13, the FFT and
 *                 IFFT chains ~2.2 (inferred).
 *   frame time    26.29 ms gray / 28.58 ms RGB on hardware. AIE COMPUTE IS NOT
 *                 FRAME TIME: the frame is 84% CPU-bound, only 41% of GMIO
 *                 blocks, and RGB's +4.59 ms of conv2d does not appear in the
 *                 frame at all. Size host work first.
 *   utilisation   ROUTED shipping build: 6 of 304 AIE-ML cores (2%), 2 of 76
 *                 memory tiles, BRAM18 26 of 1200, DSP 56 of 1312, LUT 10527 of
 *                 520704, FF 13252 of 1041408 (base platform included; the two
 *                 PL kernels alone are LUT 7499 / FF 9523).
 *                 docs/thesis/results/resources.csv, build=rgb_l1relu.
 *                 CORRECTED 2026-09-04 — this line previously carried 1 memory
 *                 tile / BRAM18 10 / DSP 44 / LUT 7694 / FF 7539, which was
 *                 roi_crop's HLS csynth ESTIMATE on an hw_emu single-channel
 *                 gray build, not a utilisation of this design.
 *                 Check any "we cannot afford it on AIE" claim against that: the
 *                 binding constraints here have always been TILE MEMORY (64 KB)
 *                 and host DMA orchestration, never core count. Note that the
 *                 mapper's Utilization column reports DECLARED budgets —
 *                 runtime<ratio> is not occupancy.
 *   DMA           1090 tx/frame after windowing (from 4258). 80 us/tx is
 *                 per-transaction OVERHEAD, not bandwidth: 64 B costs 14.4 us
 *                 and 128 KB costs 22.8 us, i.e. 2048x the payload for 1.6x the
 *                 time, and the largest transfer reaches 5.76 GB/s. DMA is not
 *                 the bottleneck. Claim P-06.
 *   host cost     6.6 ms/frame of XRT descriptor work: the two 256-tx ports cost
 *                 11.0 us of host CPU per async(). The only lever is fewer,
 *                 larger transactions.
 *
 * @thesis sec:architekturaSystemu | A-01,A-02 | The top-level AIE graph: 1 PLIO in, 7 GMIO
 *   ports, conv2d + FFT2D + cmul_accum + IFFT2D, one instance each, reused serially across the
 *   16 feature channels.
 */

#pragma once

#include "adf.h"
#include "fft_graph.h"
#include "ifft_graph.h"
#include "conv2d_kernel.h"
#include "cmul_accum_kernel.h"

using namespace adf;

// Regroup cmul's accumulator output through a memory tile so the host can drain
// a whole channel in ONE transfer instead of one per chunk. Off by default; the
// DDR-per-chunk path is what every run before 2026-08-21 used.
#ifndef CMUL_ACCUM_MEMTILE
#  define CMUL_ACCUM_MEMTILE 0
#endif

class MOSSE_graph : public graph
{
public:
    // -------------------------------------------------------
    // PLIO (1 input only)
    // -------------------------------------------------------
    input_plio  patch_in;   // "PatchIn" ← roi_crop PL kernel

    // -------------------------------------------------------
    // GMIO (5 input + 5 output = 10 total)
    // -------------------------------------------------------
    // conv2d weights: loaded once per channel before conv starts
    input_gmio  gmio_weights;

    // Forward FFT transpose scratch (shared serially across channels)
    output_gmio gmio_fft_row_out;   // row-FFT output → DDR (APU transposes)
    input_gmio  gmio_fft_col_in;    // DDR (transposed) → col-FFT input

    // Per-channel 2-D spectrum tap.
    //
    // The filter update needs F_ch, and until this port existed nothing exposed
    // it: fft_col_out went only to cmul, so the host could see the half-transformed
    // row FFT and the accumulated Σ H*⊙F but never F_ch itself. This is a pure
    // broadcast — cmul's connection is unchanged — and it drains 1:1 with
    // gmio_accum_out, so the host folds it into the existing chunk loop rather
    // than adding a new synchronisation structure.
    output_gmio gmio_fft_col_out;   // F_ch (full 2-D spectrum) → DDR (APU reads)

    // cmul_accum combined input and output (shared serially)
    input_gmio  gmio_cmul_in;       // [H_ch* | prev_Σ] packed per chunk ← DDR (APU writes)
                                    //   H_ch* ALONE when CMUL_SPLIT_ACCUM
    input_gmio  gmio_accum_in;      // prev_Σ, its own port when CMUL_SPLIT_ACCUM
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

#if CMUL_ACCUM_MEMTILE
    // Regrouping buffer for the accumulator output — see the connect below.
    // PATCH_ROWS*PATCH_COLS cint16 = 64 KB, 128 KB with the ping-pong, against a 512 KB
    // AIE-ML memory tile. num_buffers = 2 so channel k's drain to DDR can overlap
    // channel k+1's accumulation.
    adf::shared_buffer<FFT_2D_TT_DATA> memTileAcc;
#endif

    MOSSE_graph()
    {
        // --- PLIO ---
        // 32-bit width so conv2d's input_stream<int32> reads are 1:1 with PLIO
        // beats (a 128-bit PLIO delivered one 128-bit beat per readincr, starving
        // conv2d 4:1). roi_crop emits 32-bit AXIS beats (4 packed int8 each).
// @thesis sec:przeplywDanych | A-01 | PatchIn is 32-bit, not 128-bit: a 128-bit PLIO delivered
//   one beat per readincr and starved conv2d.
        patch_in = input_plio::create("PatchIn", plio_32_bits, "patch_in.txt");

        // --- GMIO (burst_length = 64 bytes, bandwidth = 1000 MB/s estimate) ---
        gmio_weights     = input_gmio::create("gmio_weights",      64, 1000);
// @thesis subsec:fftAie | A-01,P-05 | MEMTILE_TRANSPOSE: the transposes move into AIE-ML memory
//   tiles and four GMIO ports disappear. A one-sided flag is a board deadlock, not a compile
//   error.
#if !MEMTILE_TRANSPOSE
        gmio_fft_row_out = output_gmio::create("gmio_fft_row_out", 64, 1000);
        gmio_fft_col_in  = input_gmio::create("gmio_fft_col_in",   64, 1000);
#endif
        gmio_fft_col_out = output_gmio::create("gmio_fft_col_out", 64, 1000);
        gmio_cmul_in     = input_gmio::create("gmio_cmul_in",       64, 1000);
#if CMUL_SPLIT_ACCUM
        gmio_accum_in    = input_gmio::create("gmio_accum_in",      64, 1000);
#endif
        gmio_accum_out   = output_gmio::create("gmio_accum_out",   64, 1000);
        gmio_ifft_row_in = input_gmio::create("gmio_ifft_row_in",  64, 1000);
#if !MEMTILE_TRANSPOSE
        gmio_ifft_row_out= output_gmio::create("gmio_ifft_row_out",64, 1000);
        gmio_ifft_col_in = input_gmio::create("gmio_ifft_col_in",  64, 1000);
#endif
        gmio_response    = output_gmio::create("gmio_response",    64, 1000);

#if CMUL_ACCUM_MEMTILE
        memTileAcc = adf::shared_buffer<FFT_2D_TT_DATA>::create({PATCH_ROWS * PATCH_COLS}, 1, 1);
        adf::num_buffers(memTileAcc) = 2;
        adf::write_access(memTileAcc.in[0]) = adf::tiling({
            .buffer_dimension = {PATCH_ROWS * PATCH_COLS},
            .tiling_dimension = {PATCH_ROWS * PATCH_COLS},
            .offset           = {0}});
        adf::read_access(memTileAcc.out[0]) = adf::tiling({
            .buffer_dimension = {PATCH_ROWS * PATCH_COLS},
            .tiling_dimension = {PATCH_ROWS * PATCH_COLS},
            .offset           = {0}});
#endif

        // --- Custom kernel instantiation ---
        conv2d = kernel::create(conv2d_kernel);
        cmul   = kernel::create(cmul_accum_kernel);

        source(conv2d) = "conv2d_kernel.cpp";
        source(cmul)   = "cmul_accum_kernel.cpp";

        runtime<ratio>(conv2d) = 0.9;
        runtime<ratio>(cmul)   = 0.9;

#if CONV_IN_CH == 3
        // RGB conv2d NEEDS MORE THAN THE 1024-BYTE DEFAULT STACK.
        //
        // Measured, not guessed: without this the mapper stops with
        //   "Stack size requirement of total (1344 + 0) bytes for 15_0 exceeds
        //    the allotted stack size of 1024 bytes"
        // and NO libadf.a is produced. The per-kernel compile succeeds either
        // way, which is why the RGB cycle schedules in CLAUDE.md were readable
        // from a build that never linked.
        //
        // The cause is the 27-tap MAC chain: three planes' worth of live
        // vectors and the fixed post chain (downshift / clip / B1 / two Hann
        // multiplies, each an int32 x CONV_VEC vector) spill where nine taps did
        // not. Confirmed pre-existing by counterfactual — the same 1344 bytes
        // with the branch's original literal weight offsets, so it is not an
        // artifact of the conv_weight_layout refactor.
        //
        // Set ONLY for CONV_IN_CH=3. The grayscale build keeps the default, so
        // its mapping and its 38 FPS measurement are not perturbed by a change
        // made for the other arm.
        stack_size(conv2d) = CONV2D_STACK;
#endif

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

#if !MEMTILE_TRANSPOSE
        // fft2d row-FFT output → GMIO (DDR, APU reads and transposes)
        adf::connect<>(fft2d.fft_row_out, gmio_fft_row_out.in[0]);

        // GMIO (APU-transposed) → fft2d col-FFT input
        adf::connect<>(gmio_fft_col_in.out[0], fft2d.fft_col_in);
#endif

        // fft2d col-FFT output → cmul.in[0] (fft_col_in, acquired 1st — must be first)
        // fft_col_out is a tile-to-tile window port; its lock is set by the col-FFT tile
        // (22_0), not by a GMIO DMA.  In Vitis 2025.2 cycle-approximate aiesim, INPUT
        // GMIO locks on this tile (filter/accum) only deliver correctly after the
        // tile-to-tile lock fires first in the iteration — see cmul_accum_kernel.h note.
        adf::connect<>(fft2d.fft_col_out, cmul.in[0]);
        adf::dimensions(cmul.in[0]) = {PATCH_COLS * FFT_COL_WS};

        // Broadcast the same col-FFT output to DDR so the APU can run the filter
        // update on F_ch. Declared AFTER the cmul connection so cmul's fft_col_in
        // keeps in[0] — the port order that cmul_accum_kernel.h documents as
        // load-bearing.
        //
        // MEASURED 2026-08-05: this fan-out DOES change the mapping. cmul.pi0 goes
        // from a direct tile-to-tile buffer
        //     fft_cols...po0 -> cmul.pi0        @ tile (22,0)
        // to a DMA path
        //     mosse_graph._dma[0].po0 -> cmul.pi0  @ tile (24,0)
        // with fft_cols relocated to (29,0). Same at 64x64 and 128x128.
        //
        // cmul_accum_kernel.h warns that the aiesim ISS only delivers the cmul_in
        // GMIO lock after the tile-to-tile lock fires, so this looked like it would
        // break the workaround. It does NOT: the 64x64 s6 run reached step 6
        // (row-IFFT) with the tap in place, i.e. the col FFT, the tap drain and
        // cmul all completed. Both apparent "hangs" during this work were the
        // aiesim wall clock, not a deadlock — see the SIM_WALL_TIMEOUT note in
        // the Makefile before diagnosing one.
        adf::connect<>(fft2d.fft_col_out, gmio_fft_col_out.in[0]);

#if CMUL_SPLIT_ACCUM
        // gmio_cmul_in → cmul.in[1] (H only), gmio_accum_in → cmul.in[2] (prev_Σ).
        // Splitting them deletes the host's 2 MB/frame packing memcpy, which the
        // startup probe shows IS an uncached BO read: 2.871 ms measured against
        // (docs/thesis/results/apu_stages.csv, claim P-05)
        // 3.01 predicted at 696 MB/s. Same bytes on the wire, one fewer touch of
        // them. See cmul_accum_kernel.h for why the packed form existed at all.
        adf::connect<>(gmio_cmul_in.out[0], cmul.in[1]);
        adf::dimensions(cmul.in[1]) = {PATCH_COLS * FFT_COL_WS};
        adf::connect<>(gmio_accum_in.out[0], cmul.in[2]);
        adf::dimensions(cmul.in[2]) = {PATCH_COLS * FFT_COL_WS};
#else
        // gmio_cmul_in → cmul.in[1] ([filter | accum_prev] packed; single GMIO lock)
        adf::connect<>(gmio_cmul_in.out[0], cmul.in[1]);
        adf::dimensions(cmul.in[1]) = {2 * PATCH_COLS * FFT_COL_WS};
#endif

#if CMUL_ACCUM_MEMTILE
        // cmul output → memory tile → gmio_accum_out (DDR)
        //
        // WHY. gmio_accum_out is 256 tx/frame (16 chunks x 16 channels) at
        // ~16.8 us each = 4.31 ms, the largest single GMIO item. Its cost is
        // per-TRANSACTION, not per-byte: the DMA probe measured 14.4 us for 64
        // bytes and 22.8 us for 128 KB. Buffering a whole channel's accumulator
        // on-tile lets the host drain it in ONE transfer per channel instead of
        // 16 — 256 tx -> 16 tx, ~4.31 -> ~0.3 ms.
        //
        // THIS IS THE FEED-FORWARD PATTERN, NOT THE READ-MODIFY-WRITE ONE. The
        // accumulator's *state* still round-trips through DDR on gmio_accum_in;
        // what moves on-tile is only the OUTPUT REGROUPING. That is why this is a
        // plain producer->consumer shared_buffer and needs no cycle and no
        // 16-deep delay line — see the accumulator entry in CLAUDE.md for why the
        // full on-tile accumulator does.
        //
        // Whole-buffer tiling on both sides, exactly as memTileFwd does: the
        // producer writes it in CMUL_N-sized pieces (16 firings) and the consumer
        // takes the plane in one go. memTileFwd already proved a chunked producer
        // write against a whole-buffer access pattern.
        //
        // It also DECOUPLES the accumulator from gmio_fft_col_out, which is the
        // whole point. Both were drained at the col-FFT's window granularity, so
        // FFT_COL_WS moved them together — and at WS=32 accum_out WON
        // (4.42 -> 1.25) while fft_col_out lost catastrophically
        // (4.57 -> 17.07), which is what made that knob a net loss.
        adf::connect<>(cmul.out[0], memTileAcc.in[0]);
        adf::dimensions(cmul.out[0]) = {PATCH_COLS * FFT_COL_WS};
        adf::connect<>(memTileAcc.out[0], gmio_accum_out.in[0]);
#else
        // cmul output → gmio_accum_out (DDR)
        adf::connect<>(cmul.out[0], gmio_accum_out.in[0]);
        adf::dimensions(cmul.out[0]) = {PATCH_COLS * FFT_COL_WS};
#endif

        // IFFT: APU reads gmio_accum_out, writes accumulated spectrum to gmio_ifft_row_in
        adf::connect<>(gmio_ifft_row_in.out[0],  ifft2d.ifft_row_in);

#if !MEMTILE_TRANSPOSE
        // ifft2d row-IFFT output → GMIO (APU reads and transposes)
        adf::connect<>(ifft2d.ifft_row_out, gmio_ifft_row_out.in[0]);

        // GMIO (APU-transposed) → ifft2d col-IFFT input
        adf::connect<>(gmio_ifft_col_in.out[0], ifft2d.ifft_col_in);
#endif

        // ifft2d col-IFFT output → gmio_response (APU reads for peak detection)
        adf::connect<>(ifft2d.ifft_col_out, gmio_response.in[0]);
    }
};
