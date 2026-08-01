/*
 * hanning_64.h
 * Precomputed separable Hanning window in Q1.15 for 64-point patches.
 *
 * HANNING_64[i] = round(sin(π*i/64)^2 * 32767)   for i = 0..63
 *
 * PERIODIC window (denominator n). Its 2D DFT has exactly 9 non-zero bins,
 * which is what lets the host cancel the pre-window mean in the frequency
 * domain. Do not switch to the symmetric (n-1) form — see _gen_hanning_h
 * in scripts/export_weights.py.
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
         0,     79,    315,    705,   1247,   1935,   2761,   3719,
      4799,   5990,   7281,   8660,  10114,  11628,  13187,  14778,
     16383,  17989,  19580,  21139,  22653,  24107,  25486,  26777,
     27968,  29048,  30006,  30832,  31520,  32062,  32452,  32688,
     32767,  32688,  32452,  32062,  31520,  30832,  30006,  29048,
     27968,  26777,  25486,  24107,  22653,  21139,  19580,  17989,
     16384,  14778,  13187,  11628,  10114,   8660,   7281,   5990,
      4799,   3719,   2761,   1935,   1247,    705,    315,     79,
};
