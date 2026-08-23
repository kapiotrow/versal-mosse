/*
 * roi_crop.h
 * PL kernel: extracts a target ROI from a DDR frame, applies the MOSSE
 * preprocessing chain, and streams the result to AIE.
 *
 * Pipeline (Bolme et al. §3.1, Danelljan et al. §3.3):
 *   1. bilinear resample  — arbitrary roi_h × roi_w → fixed patch_rows × patch_cols
 *                           (source coordinates clamped to the frame, so a target
 *                            near an edge replicates the border instead of reading
 *                            out of bounds). At ROI_IN_CH=3 all three planes are
 *                            resampled with the SAME geometry and weights.
 *   2. log transform      — LOG_LUT[v] ≈ log(1+v), "helps with low contrast
 *                           lighting situations". Per sample, per plane.
 *   3. zero mean          — subtract the patch mean
 *   4. unit L2 norm       — divide by the patch standard deviation, then rescale
 *                           by ROI_NORM_Q so the int8 grid is actually used
 *                           (see the note on ROI_NORM_Q below)
 *   5. int8 quantize      — clip to [-127, 127], SIGNED
 *
 * Steps 3-4 are JOINT ACROSS THE PLANES at ROI_IN_CH=3 — one mean and one
 * 1/sigma over all 3·patch_rows·patch_cols samples, applied to all three. See
 * the ROI_IN_CH note below; getting this wrong is silent and self-defeating.
 *
 * Steps 3-4 are global reductions: nothing can be emitted until every pixel has
 * been seen. The kernel therefore buffers the resampled patch in BRAM and runs
 * two passes over it.
 *
 * Output packing (32-bit word = 4 signed int8 samples, 1:1 with conv2d's
 * input_stream<int32>):
 *   word.data[7:0]   = sample 0
 *   word.data[15:8]  = sample 1
 *   word.data[23:16] = sample 2
 *   word.data[31:24] = sample 3
 *
 * At ROI_IN_CH=3 the SAMPLES are pixel-interleaved, so a row is 3·patch_cols
 * bytes and three words carry exactly four RGB pixels:
 *   R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3
 * which is byte-for-byte what conv2d_kernel.cpp's RGB read loop unpacks. A row
 * stays word-aligned because patch_cols is a multiple of 4 and 3·4 = 12.
 *
 * The final beat has word.last = 1.
 * Called once per channel per frame (APU loops N_CHANNELS times).
 */

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// Input planes. 1 = grayscale (what ships), 3 = interleaved RGB.
//
// DRIVEN FROM THE SAME MAKEFILE VARIABLE AS CONV_IN_CH. The two MUST agree:
// this kernel decides what the AXIS wire carries and conv2d decides how to
// unpack it, and a disagreement is not a compile error at either end — it is a
// graph that runs and produces a plausible, wrong feature map. The Makefile
// passes -DROI_IN_CH=$(CONV_IN_CH) so there is one knob, per CLAUDE.md's rule
// about constants both engines derive from.
//
// WHY NORMALIZATION MUST BE JOINT ACROSS PLANES. Normalizing each plane on its
// own mean and sigma equalizes the three and deletes exactly the chromatic
// contrast RGB is for: a red patch and a grey patch of the same luminance
// structure would come out identical. So Stage A takes ONE mean and ONE inv_q
// over all 3N samples. This matches scripts/rgb_vs_gray_holdout.py's
// stage_a_rgb(), which is the model the offline 51 -> 42 failure result was
// measured with, so the board reproduces that arm rather than a variant of it.
//
// FRAME LAYOUT AT ROI_IN_CH=3 IS PIXEL-INTERLEAVED: frame_buf holds
// frame_rows x frame_cols x 3 bytes as R0 G0 B0 R1 G1 B1 ... per row.
// Interleaved, not planar, for two reasons: it is what a camera delivers, and
// the three taps for one source pixel are then CONTIGUOUS, so the 12 scattered
// reads per output pixel are 4 bursts of 3 bytes rather than 12 unrelated
// addresses. At ROI_IN_CH=1 the indexing collapses to the historical
// frame_buf[y * frame_cols + x] exactly.
#ifndef ROI_IN_CH
#  define ROI_IN_CH 1
#endif

// Largest patch the BRAM scratch buffer supports. PATCH_ROWS/PATCH_COLS come
// from the Makefile and must not exceed this.
#define ROI_MAX_PATCH_ROWS 128
#define ROI_MAX_PATCH_COLS 128
#define ROI_MAX_PATCH_ELEMS (ROI_MAX_PATCH_ROWS * ROI_MAX_PATCH_COLS)
// Scratch buffer bytes: one byte per sample per plane. 16 KB gray, 48 KB RGB.
#define ROI_MAX_PATCH_BYTES (ROI_MAX_PATCH_ELEMS * ROI_IN_CH)

// m_axi depth for frame_buf, in BYTES. Cosim/interface sizing only — it must
// cover the largest frame the host will pass. 1920*1080 per plane.
#define ROI_FRAME_DEPTH (1920 * 1080 * ROI_IN_CH)

// Fractional bits used for bilinear interpolation weights and for the source
// coordinate step. Q8 is ample for 8-bit pixels.
#define ROI_FRAC_BITS 8
#define ROI_FRAC_ONE  (1 << ROI_FRAC_BITS)

// Post-normalization scale.
//
// Bolme normalizes the patch to "a mean value of 0.0 and a norm of 1.0". Taken
// literally into fixed point that is useless: a unit-L2 patch of N=16384 pixels
// has elements of magnitude ~1/128, which quantizes to 0 in int8. The norm
// constraint only fixes a global scale, and correlation is linear in the patch,
// so any constant rescale is absorbed downstream by the filter. We therefore
// emit the z-score (x-µ)/σ scaled by ROI_NORM_Q.
//
// ROI_NORM_Q = 32 puts ±1σ at ±32 and clips at ±127 ≈ ±3.97σ — 5 bits of
// resolution per σ, with mild clipping only in the far tails.
#define ROI_NORM_Q 32

// Upper bound on the Q16.16 reciprocal-sigma factor, to keep the pass-2 product
// bounded and to stop a near-flat patch (σ→0) amplifying sensor noise to
// full scale.
#define ROI_INV_Q_MAX (1 << 20)

void roi_crop(
    const ap_uint<8>                  *frame_buf,   // DDR input frame
    hls::stream<ap_axiu<32,0,0,0>>   &patch_out,   // AXIS → AIE PatchIn PLIO
    int  frame_rows,                                // frame height (for clamping)
    int  frame_cols,                                // frame width
    int  roi_row,                                   // ROI top edge, in frame pixels
    int  roi_col,                                   // ROI left edge, in frame pixels
    int  roi_h,                                     // ROI height, in frame pixels
    int  roi_w,                                     // ROI width, in frame pixels
    int  patch_rows,                                // output patch height
    int  patch_cols,                                // output patch width
    int  recompute                                  // see below
);

/*
 * recompute — set to 1 on the first channel of a frame, 0 on the rest.
 *
 * The serial-channel architecture re-streams the SAME patch to conv2d once per
 * channel (only the conv weights differ), so this kernel is invoked N_CHANNELS
 * times per frame. Stage A depends only on the frame and the ROI, not on the
 * channel, so recomputing it every time would burn 16× the work for an identical
 * result.
 *
 * recompute=1 runs the resample + statistics + normalize passes and leaves the
 * quantized int8 patch in the on-chip buffer. recompute=0 skips straight to
 * streaming that buffer out.
 *
 * COST, corrected 2026-08-17 from a hw_emu VCD probe (the previous figures here
 * were estimates and both were low). Scaling the 64×64 measurement to 128×128:
 *
 *   Stage A (PASS1 + NORM)  ~195k cycles   once per frame     (was estimated 81k)
 *   stream-out (PASS2)      ~112k cycles   EVERY channel      (was estimated  4k)
 *
 * The stream-out estimate was low by ~28× because PASS2 does not achieve II=1:
 * it is backpressured by the AIE at ~27 cycles/beat. So the saving is
 *
 *   without cache  16 × (195k + 112k) ≈ 4.90M cycles
 *   with cache      195k + 16 × 112k  ≈ 1.98M cycles     -> ~2.5×, not ~9×.
 *
 * The cache is still clearly worth having, but it is no longer the dominant
 * term: 90% of the remaining cost is the 16 stream-out passes, and their cost is
 * set by how fast conv2d consumes, not by anything in this file.
 *
 * None of this is currently the bottleneck — see the MEASURED COST block in
 * roi_crop.cpp: XRT's crop_run.wait() dwarfs the whole kernel by ~300×.
 *
 * The buffer persists across calls (it is static), so recompute=0 before any
 * recompute=1 call streams stale or zero data. The host must order them.
 */
