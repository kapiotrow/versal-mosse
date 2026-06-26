/*
 * hanning_128.h
 * Precomputed separable Hanning window in Q1.15 for 128-point patches.
 *
 * HANNING_128[i] = round(sin(π*i/127)^2 * 32767)   for i = 0..127
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
         0,     20,     80,    180,    320,    499,    717,    973,
      1267,   1597,   1965,   2367,   2803,   3273,   3775,   4308,
      4870,   5461,   6078,   6721,   7387,   8075,   8784,   9511,
     10254,  11013,  11785,  12569,  13361,  14161,  14967,  15776,
     16586,  17396,  18203,  19006,  19803,  20591,  21369,  22135,
     22886,  23622,  24340,  25039,  25716,  26371,  27001,  27605,
     28181,  28729,  29247,  29733,  30186,  30606,  30990,  31340,
     31652,  31927,  32164,  32363,  32522,  32642,  32722,  32762,
     32762,  32722,  32642,  32522,  32363,  32164,  31927,  31652,
     31340,  30990,  30606,  30186,  29733,  29247,  28729,  28181,
     27605,  27001,  26371,  25716,  25039,  24340,  23622,  22886,
     22135,  21369,  20591,  19803,  19006,  18203,  17396,  16586,
     15776,  14967,  14161,  13361,  12569,  11785,  11013,  10254,
      9511,   8784,   8075,   7387,   6721,   6078,   5461,   4870,
      4308,   3775,   3273,   2803,   2367,   1965,   1597,   1267,
       973,    717,    499,    320,    180,     80,     20,      0,
};
