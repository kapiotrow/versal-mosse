/*
 * design/aie_src/conv_weight_layout.h
 *
 * THE conv2d weight-buffer layout. Defined ONCE, here, and derived from
 * CONV_IN_CH so the grayscale (9-tap) and RGB (27-tap) layouts cannot drift
 * apart by hand-editing.
 *
 * Why this file exists
 * --------------------
 * The layout used to be four hardcoded copies of the same byte offsets:
 * conv2d_kernel.h's comment, conv2d_kernel.cpp's two branches,
 * export_weights.py's pack_channel(), the host's mean_prev feedback, plus
 * check_collapse.py / gen_aiesim_vectors.py / phase1_sweep.py as readers.
 * CLAUDE.md's "Preprocessing constants are coupled across engines with no
 * compile-time check" entry is about exactly this set.
 *
 * RGB is what forced the issue. 27 taps occupy [0:27], which OVERRUNS the
 * grayscale out_shift[9], bias_acc[10:14], dequant_scale[14:18] and
 * mean_prev[18:22]. Every one of those overlaps is silent: the host would
 * happily write a mean_prev over three B-plane taps and the kernel would
 * happily convolve with it.
 *
 * The layout, as a function of the tap count
 * ------------------------------------------
 *   [0 : RAW)          int8   conv taps. GRAY: w[kr][kc] row-major.
 *                             RGB:  PLANAR -- [0:9] R, [9:18] G, [18:27] B,
 *                             each row-major. Planar, not interleaved: the
 *                             kernel hoists 27 scalars once per invocation and
 *                             the plane grouping is what makes that readable.
 *   [RAW]              int8   out_shift   (int32 accumulator -> int16)
 *   [RAW+1 : RAW+5)    int32  bias_acc     LE
 *   [RAW+5 : RAW+9)    fp32   dequant_scale LE -- host validation only, the
 *                             kernel never reads it
 *   [RAW+9 : RAW+13)   int32  mean_prev    LE -- Stage B1. Written by the HOST
 *                             every frame, not by export_weights.py.
 *   [PAD-2]            int8   layout tag = CONV_KSIZE
 *   [PAD-1]            int8   layout tag = CONV_IN_CH
 *
 * Gray resolves to 9 / 9 / 10 / 14 / 18, which is byte-for-byte the shipped
 * layout -- this file changes no grayscale offset. RGB 3x3 resolves to
 * 27 / 27 / 28 / 32 / 36 / end 40, comfortably inside the 64-byte pad.
 *
 * THE BUFFER IS NO LONGER FIXED AT 64 BYTES, because CONV_KSIZE=7 does not fit
 * -----------------------------------------------------------------------
 * A 7x7 RGB bank is 147 taps, and 147 + 13 field bytes + 2 tag bytes = 162 --
 * two and a half times the old buffer. CONV_WEIGHT_BYTES_PAD is therefore
 * DERIVED: the smallest multiple of 64 that holds taps, fields and tags. It
 * resolves to the historical 64 for every 3x3 bank (gray 24, RGB 42) and to 192
 * for 7x7 RGB, so no existing arm's byte layout moves. 64 is kept as the
 * granularity because it is the GMIO alignment the transfer wants.
 *
 * The two tag bytes
 * -----------------
 * The .bin has no header, so a reader that assumes the wrong layout gets
 * plausible garbage rather than an error -- check_collapse.py reading an RGB
 * file as gray would report 16 healthy channels built from R-plane taps and a
 * bias sliced out of the G plane. The LAST byte carries CONV_IN_CH and the one
 * before it CONV_KSIZE, so every reader can assert instead of guess. Two tags,
 * not one, because the tap count is a product of both and 27 gray-7x7 taps
 * would otherwise be indistinguishable from 27 RGB-3x3 taps -- a collision that
 * exists the moment a second kernel size does.
 * Files exported before this convention have 0 in both; readers treat (0,0) as
 * (1, 3), legacy grayscale 3x3, which is what they all were.
 *
 * No <adf.h> here on purpose: the AIE kernel, the PS host and roi_crop must all
 * be able to include this.
 *
 * @thesis subsec:kwantyzacjaImpl | B-08 | The single-sourced weight-buffer layout, derived from
 *   CONV_IN_CH, plus the layout tag byte that makes a gray/RGB mismatch loud instead of
 *   plausible.
 */

#pragma once

#ifndef CONV_IN_CH
#  define CONV_IN_CH  1
#endif
#ifndef CONV_KSIZE
#  define CONV_KSIZE  3
#endif

#define CONV_WEIGHT_BYTES_RAW  (CONV_IN_CH * CONV_KSIZE * CONV_KSIZE)  /* 9, 27 or 147 */

/* taps + the 13 bytes of fields + the 2 tag bytes, rounded up to the 64-byte
 * GMIO granularity. 64 for every 3x3 bank, 192 for 7x7 RGB. */
#define CONV_WEIGHT_BYTES_MIN  (CONV_WEIGHT_BYTES_RAW + 13 + 2)
#define CONV_WEIGHT_BYTES_PAD  ((((CONV_WEIGHT_BYTES_MIN) + 63) / 64) * 64)

#define CONV_W_OFF_TAPS     0
#define CONV_W_OFF_SHIFT    (CONV_WEIGHT_BYTES_RAW)
#define CONV_W_OFF_BIAS     (CONV_W_OFF_SHIFT + 1)
#define CONV_W_OFF_DEQUANT  (CONV_W_OFF_BIAS + 4)
#define CONV_W_OFF_MEAN     (CONV_W_OFF_DEQUANT + 4)
#define CONV_W_OFF_END      (CONV_W_OFF_MEAN + 4)
#define CONV_W_OFF_TAG      (CONV_WEIGHT_BYTES_PAD - 1)   /* = CONV_IN_CH */
#define CONV_W_OFF_TAG_K    (CONV_WEIGHT_BYTES_PAD - 2)   /* = CONV_KSIZE */

/* Per-plane tap offsets, for the RGB kernel branch's scalar hoist. */
#define CONV_W_OFF_PLANE(ic)  ((ic) * CONV_KSIZE * CONV_KSIZE)

#ifdef __cplusplus
static_assert(CONV_W_OFF_END <= CONV_W_OFF_TAG_K,
              "conv2d weight fields overrun the buffer -- CONV_WEIGHT_BYTES_PAD is wrong");
static_assert(CONV_IN_CH == 1 || CONV_IN_CH == 3,
              "CONV_IN_CH must be 1 (luminance) or 3 (RGB)");
static_assert(CONV_KSIZE == 3 || CONV_KSIZE == 5 || CONV_KSIZE == 7,
              "CONV_KSIZE must be 3, 5 or 7 (odd, so the 'same' padding is symmetric)");
#endif
