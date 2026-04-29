/*
 * cmul_accum_kernel.cpp
 * AIE-ML kernel stub: pass-through accumulator (no multiply yet).
 *
 * Reads patch_rows * patch_cols cint16 samples from fft_col_in and
 * writes them straight to accum_out (ignoring filter and accum_prev).
 *
 * TODO (full implementation):
 *   1. On channel == 0: for each element i:
 *        accum[i] = cmul_conj(fft_col_in[i], filter[i])
 *      where cmul_conj(a, b*) = {a.r*b.r + a.i*b.i, a.i*b.r - a.r*b.i}
 *      Use int32 intermediate to avoid overflow; output cint16 with >>15 shift.
 *   2. On channel > 0: for each element i:
 *        accum[i] = sat16(accum_prev[i] + cmul_conj(fft_col_in[i], filter[i]))
 *   3. Write updated accumulator to accum_out.
 *   4. Migrate accumulator to cint32 (128 KB DDR) or AIE-ML Memory Tile
 *      once correctness is validated at cint16 precision.
 */

#include "cmul_accum_kernel.h"

void cmul_accum_kernel(
    input_stream<cint16>   *fft_col_in,
    output_stream<cint16>  *accum_out,
    input_buffer<cint16_t> &filter,
    input_buffer<cint16_t> &accum_prev,
    int32_t                 patch_rows,
    int32_t                 patch_cols,
    int32_t                 channel)
{
    // Suppress unused-variable warnings for stub
    (void)filter;
    (void)accum_prev;
    (void)channel;

    int total = patch_rows * patch_cols;
    for (int i = 0; i < total; ++i) {
        cint16_t f = readincr(fft_col_in);
        writeincr(accum_out, f);  // stub: pass through, no multiply
    }
}
