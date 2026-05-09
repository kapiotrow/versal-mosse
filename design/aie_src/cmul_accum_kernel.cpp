/*
 * cmul_accum_kernel.cpp
 * AIE-ML kernel stub: pass-through (no complex multiply, no accumulation yet).
 *
 * Reads PATCH_ROWS * PATCH_COLS cint16 samples from fft_col_in and writes them
 * straight to accum_out.  The filter and accum_prev buffers are consumed (to
 * avoid stalling the GMIO DMA) but not used in computation.
 *
 * TODO (full implementation):
 *   1. For each element i:
 *        A_ch[i] = cmul_conj(fft_col_in[i], filter[i])
 *      where cmul_conj(a, b*) = {a.re*b.re + a.im*b.im, a.im*b.re - a.re*b.im}
 *      Use int32 intermediate to avoid overflow; output cint16 with >>15 shift.
 *   2. acc[i] = sat16(accum_prev[i] + A_ch[i])   (APU sends zeros for ch=0)
 *   3. Write acc[i] to accum_out.
 *   4. Add RTP port for channel index once step 2 is in place; remove the
 *      "APU sends zeros for ch=0" contract.
 */

#include "cmul_accum_kernel.h"

void cmul_accum_kernel(
    input_stream<cint16>   *fft_col_in,
    output_stream<cint16>  *accum_out,
    input_buffer<cint16_t> &filter,
    input_buffer<cint16_t> &accum_prev)
{
    (void)filter;
    (void)accum_prev;

    // chess_prepare_for_pipelining: use modulo scheduling instead of
    // whole-graph scheduling.  Avoids OOM in noodle when compiled in
    // parallel with the large DSPLib FFT tiles.
    for (int i = 0; i < PATCH_ROWS * PATCH_COLS; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(PATCH_ROWS * PATCH_COLS, PATCH_ROWS * PATCH_COLS)
    {
        cint16_t f = readincr(fft_col_in);
        writeincr(accum_out, f);  // stub: pass through, no multiply/accumulate
    }
}
