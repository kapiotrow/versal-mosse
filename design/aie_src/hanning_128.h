/*
 * hanning_128.h
 * Precomputed separable Hanning window in Q1.15 for 128-point patches.
 *
 * HANNING_128[i] = round(sin(π*i/128)^2 * 32767)   for i = 0..127
 *
 * PERIODIC window (denominator n). Its 2D DFT has exactly 9 non-zero bins,
 * which is what lets the host cancel the pre-window mean in the frequency
 * domain. Do not switch to the symmetric (n-1) form — see _gen_hanning_h
 * in scripts/export_weights.py.
 *
 * Usage in conv2d_kernel: out_windowed = (out * HANNING_128[r] / 32768)
 *                                        * HANNING_128[c] / 32768
 *
 * Regenerate with:  make weights        (runs scripts/export_weights.py)
 * Guard: kernel will fail to compile if PATCH_ROWS or PATCH_COLS != 128.
 */

#pragma once
#include <stdint.h>

/* Guard: only enforce when building with AIE pre-processor defines. */
#if defined(PATCH_ROWS) && defined(PATCH_COLS)
#  if PATCH_ROWS != 128 || PATCH_COLS != 128
#    error "hanning_128.h is generated for PATCH_ROWS=128, PATCH_COLS=128. Re-run make weights."
#  endif
#endif

static const int16_t HANNING_128[128] = {
         0,     20,     79,    177,    315,    491,    705,    958,
      1247,   1573,   1935,   2331,   2761,   3224,   3719,   4244,
      4799,   5381,   5990,   6624,   7281,   7961,   8660,   9379,
     10114,  10864,  11628,  12403,  13187,  13980,  14778,  15580,
     16383,  17187,  17989,  18787,  19580,  20364,  21139,  21903,
     22653,  23388,  24107,  24806,  25486,  26143,  26777,  27386,
     27968,  28523,  29048,  29543,  30006,  30436,  30832,  31194,
     31520,  31809,  32062,  32276,  32452,  32590,  32688,  32747,
     32767,  32747,  32688,  32590,  32452,  32276,  32062,  31809,
     31520,  31194,  30832,  30436,  30006,  29543,  29048,  28523,
     27968,  27386,  26777,  26143,  25486,  24806,  24107,  23388,
     22653,  21903,  21139,  20364,  19580,  18787,  17989,  17187,
     16384,  15580,  14778,  13980,  13187,  12403,  11628,  10864,
     10114,   9379,   8660,   7961,   7281,   6624,   5990,   5381,
      4799,   4244,   3719,   3224,   2761,   2331,   1935,   1573,
      1247,    958,    705,    491,    315,    177,     79,     20,
};
