/*
 * hanning_64.h
 * Precomputed separable Hanning window in Q1.15 for 64-point patches.
 *
 * HANNING_64[i] = round(sin(π*i/63)^2 * 32767)   for i = 0..63
 *
 * Usage in conv2d_kernel: out_windowed = (out * HANNING_64[r] / 32768)
 *                                        * HANNING_64[c] / 32768
 *
 * Regenerate with:  make weights        (runs scripts/export_weights.py)
 * Guard: kernel will fail to compile if PATCH_ROWS or PATCH_COLS != 64.
 */

#pragma once
#include <stdint.h>

/* Guard: only enforce when building with AIE pre-processor defines. */
#if defined(PATCH_ROWS) && defined(PATCH_COLS)
#  if PATCH_ROWS != 64 || PATCH_COLS != 64
#    error "hanning_64.h is generated for PATCH_ROWS=64, PATCH_COLS=64. Re-run make weights."
#  endif
#endif

static const int16_t HANNING_64[64] = {
         0,     81,    325,    728,   1286,   1995,   2847,   3833,
      4944,   6169,   7495,   8909,  10398,  11946,  13539,  15159,
     16792,  18421,  20029,  21601,  23122,  24575,  25947,  27224,
     28393,  29443,  30363,  31145,  31779,  32260,  32584,  32747,
     32747,  32584,  32260,  31779,  31145,  30363,  29443,  28393,
     27224,  25947,  24575,  23122,  21601,  20029,  18421,  16792,
     15159,  13539,  11946,  10398,   8909,   7495,   6169,   4944,
      3833,   2847,   1995,   1286,    728,    325,     81,      0,
};
