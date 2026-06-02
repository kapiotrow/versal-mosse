/*
 * cmul_accum_kernel.cpp
 * AIE-ML kernel: element-wise F_ch * conj(H_ch) + accumulate.
 *
 * For each element i in the current FFT window chunk (N = PATCH_COLS * FFT_COL_WS):
 *   A[i]   = fft_col_in[i] * conj(filter[i])
 *          = { re_in*re_flt + im_in*im_flt,
 *              im_in*re_flt - re_in*im_flt }   (int32 intermediates)
 *   out[i] = (cint16){ accum_prev[i].re + A[i].re,
 *                      accum_prev[i].im + A[i].im }  (truncating to int16)
 *
 * filter stores H (not pre-conjugated); conjugation is applied here via the
 * sign flip on the imaginary product.
 *
 * No >>15 shift: PS pre-scales H so products stay within int16 range.
 * APU sends a zero buffer for accum_prev on ch=0 to initialise the accumulator.
 *
 * Memory tile access strategy:
 *   cmul_in (flt + accum halves) lives in memory tile 13_0 (GMIO-backed).
 *   Scalar reads from memory tile cost ~100K cycles each in cycle-approximate
 *   aiesim.  To avoid the timeout, we first vector-copy both halves into
 *   tile-local static arrays using 128-bit (v8cint16) vector loads, then run
 *   the scalar arithmetic loop over the local copies.  Vector loads from
 *   memory tile use the vector bus and are not subject to the scalar penalty.
 *
 * cmul_in layout per invocation (2 * N elements total):
 *   [0 .. N-1]   : H_ch* (filter)
 *   [N .. 2N-1]  : prev accumulator
 * Both halves come from the same GMIO-backed buffer (one lock pair).
 */

#include "cmul_accum_kernel.h"

static constexpr int CMUL_N = PATCH_COLS * FFT_COL_WS;  // 256

// Tile-local scratch for filter and accumulator halves.
// Placed in tile LDM (not stack) to avoid overflow; 2×1024 B is negligible.
static cint16_t flt_local[CMUL_N];
static cint16_t acc_local[CMUL_N];

void cmul_accum_kernel(
    input_buffer<cint16_t>  &fft_col_in,
    output_buffer<cint16_t> &accum_out,
    input_buffer<cint16_t>  &cmul_in)
{
    cint16_t* flt_src = cmul_in.data();
    cint16_t* acc_src = flt_src + CMUL_N;

    // Vector copy: 128-bit (v8cint16) loads from memory tile 13_0.
    // Vector loads use the vector bus — not subject to the scalar read penalty.
    v8cint16* __restrict s_flt = (v8cint16*)flt_src;
    v8cint16* __restrict s_acc = (v8cint16*)acc_src;
    v8cint16* __restrict d_flt = (v8cint16*)flt_local;
    v8cint16* __restrict d_acc = (v8cint16*)acc_local;

    for (int i = 0; i < CMUL_N / 8; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(CMUL_N / 8, CMUL_N / 8)
    {
        d_flt[i] = s_flt[i];
        d_acc[i] = s_acc[i];
    }

    // Scalar arithmetic on tile-local arrays — fast (~1 cycle per read).
    // No chess_prepare_for_pipelining: that pragma + int32 arithmetic causes
    // the noodle assembler to OOM-kill during compilation on this kernel.
    cint16_t* __restrict in_ptr  = fft_col_in.data();
    cint16_t* __restrict out_ptr = accum_out.data();

    for (int i = 0; i < CMUL_N; ++i) {
        int32_t re = (int32_t)in_ptr[i].real * (int32_t)flt_local[i].real
                   + (int32_t)in_ptr[i].imag * (int32_t)flt_local[i].imag;
        int32_t im = (int32_t)in_ptr[i].imag * (int32_t)flt_local[i].real
                   - (int32_t)in_ptr[i].real * (int32_t)flt_local[i].imag;
        out_ptr[i].real = (int16_t)(acc_local[i].real + re);
        out_ptr[i].imag = (int16_t)(acc_local[i].imag + im);
    }
}
