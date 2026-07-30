/*
 * cmul_accum_kernel.cpp
 * AIE-ML kernel: element-wise F_ch * conj(H_ch) + accumulate.
 *
 * For each element i in the current FFT window chunk (N = PATCH_COLS * FFT_COL_WS):
 *   A[i]   = fft_col_in[i] * conj(filter[i])
 *          = { re_in*re_flt + im_in*im_flt,
 *              im_in*re_flt - re_in*im_flt }   (int32 intermediates)
 *   out[i] = (cint16){ accum_prev[i].re + A[i].re,
 *                      accum_prev[i].im + A[i].im }  (SATURATING to int16)
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

// Saturating narrow to int16.
//
// The accumulate step used to be a bare `(int16_t)(acc + re)` cast. `acc` is
// int16 and `re` is int32, so the sum is computed in int32 and the cast WRAPS on
// overflow: one channel too many flips the accumulated spectrum's sign instead of
// clamping. That is silent and catastrophic for peak detection — the argmax moves
// to a wrapped element. (The {-32768,0} values seen in the MODE=2 ramp runs are
// consistent with exactly this.)
//
// Saturation is the right failure mode here: MOSSE only needs the argmax, so a
// clamped peak still points at the correct location, while a wrapped one does not.
// NOTE this makes overflow benign, not absent — with N_CHANNELS=16 the DDR
// accumulator is still cint16, so 16 x per-channel magnitude must fit 32767 or
// the sum clips. Sizing that headroom (or widening the accumulator to int32) is
// a separate open task.
// BRANCHLESS on purpose. The first version used early returns:
//     if (v >  32767) return  32767;
//     if (v < -32768) return -32768;
// which emits control flow inside the inner loop and measurably slowed the
// kernel (aiesim s0 went from completing comfortably to not finishing within the
// 1200 s wall clock). This kernel is already documented as schedule-fragile —
// see the note below about chess_prepare_for_pipelining and the assembler OOM.
//
// The min/max form compiles to two conditional selects with no branches, so the
// hardware do-loop stays tight. Compare the reported cycles for the arithmetic
// loop in aiecompiler.log against the branching version's 26.
static inline int16_t sat16(int32_t v)
{
    v = (v >  32767) ?  32767 : v;
    v = (v < -32768) ? -32768 : v;
    return (int16_t)v;
}

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
        out_ptr[i].real = sat16((int32_t)acc_local[i].real + re);
        out_ptr[i].imag = sat16((int32_t)acc_local[i].imag + im);
    }
}
