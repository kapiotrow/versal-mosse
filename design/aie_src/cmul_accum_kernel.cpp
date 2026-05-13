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
    input_buffer<cint16_t>  &fft_col_in,
    output_buffer<cint16_t> &accum_out,
    input_buffer<cint16_t>  &filter,
    input_buffer<cint16_t>  &accum_prev)
{
    (void)filter;
    (void)accum_prev;

    // .data() returns a raw pointer — same pattern used in DSPLib kernels
    // (e.g. bitonic_sort.cpp).  aie::begin() / operator[] require aie_api/aie.hpp
    // which is not in the chess-clang include path.
    cint16_t* __restrict in_ptr  = (cint16_t*)fft_col_in.data();
    cint16_t* __restrict out_ptr = (cint16_t*)accum_out.data();

    // chess_prepare_for_pipelining: use modulo scheduling instead of
    // whole-graph scheduling.  Avoids OOM in noodle when compiled in
    // parallel with the large DSPLib FFT tiles.
    // Process one FFT window chunk per invocation; the kernel is called
    // PATCH_ROWS/FFT_COL_WS = 64 times per frame by ADF's ping-pong scheduler.
    for (int i = 0; i < PATCH_COLS * FFT_COL_WS; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(PATCH_COLS * FFT_COL_WS, PATCH_COLS * FFT_COL_WS)
    {
        out_ptr[i] = in_ptr[i];  // stub: pass through, no multiply/accumulate
    }
}
