/*
 * roi_crop.h
 * PL kernel: extracts a target ROI from a DDR frame, applies the MOSSE
 * preprocessing chain, and streams the result to AIE.
 *
 * Pipeline (Bolme et al. §3.1, Danelljan et al. §3.3):
 *   1. bilinear resample  — arbitrary roi_h × roi_w → fixed patch_rows × patch_cols
 *                           (source coordinates clamped to the frame, so a target
 *                            near an edge replicates the border instead of reading
 *                            out of bounds)
 *   2. log transform      — LOG_LUT[v] ≈ log(1+v), "helps with low contrast
 *                           lighting situations"
 *   3. zero mean          — subtract the patch mean
 *   4. unit L2 norm       — divide by the patch standard deviation, then rescale
 *                           by ROI_NORM_Q so the int8 grid is actually used
 *                           (see the note on ROI_NORM_Q below)
 *   5. int8 quantize      — clip to [-127, 127], SIGNED
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
 * The final beat has word.last = 1.
 * Called once per channel per frame (APU loops N_CHANNELS times).
 */

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// Largest patch the BRAM scratch buffer supports. PATCH_ROWS/PATCH_COLS come
// from the Makefile and must not exceed this.
#define ROI_MAX_PATCH_ROWS 128
#define ROI_MAX_PATCH_COLS 128
#define ROI_MAX_PATCH_ELEMS (ROI_MAX_PATCH_ROWS * ROI_MAX_PATCH_COLS)

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
 * streaming that buffer out. Measured on the 128×128 build: the Stage A passes
 * cost ~81k cycles, the stream-out pass ~4k, so per frame this turns
 * 16 × 85k ≈ 1.36M cycles into 81k + 16 × 4k ≈ 146k — roughly a 9× saving.
 *
 * The buffer persists across calls (it is static), so recompute=0 before any
 * recompute=1 call streams stale or zero data. The host must order them.
 */
