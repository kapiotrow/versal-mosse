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
 * H is Q1.15: the host normalizes max|H| to 32767 across all channels, and the
 * product is shifted right by CMUL_H_SHIFT (default 15, set from the Makefile)
 * with round-to-nearest. Earlier revisions did NO shift, on the strength of a
 * comment claiming the PS pre-scaled H — nothing implemented that, and every
 * aiesim scenario passed a literal H = 1, so the shift budget in the Makefile
 * was calibrated at a filter gain of one. With Q1.15 the product is |H|/2^15 <= 1
 * times the old value, so that budget still holds as an upper bound.
 *
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
 *
 * ---------------------------------------------------------------------------
 * COST — per frame at 128x128, ch16, 1 GHz. docs/thesis/results/aie_compute.csv.
 * Claims A-04, P-05.
 *
 *   compute       0.13 ms. Four multiplies and two adds per bin, so this kernel
 *                 is ~2% of the AIE total and is NOT where to look for time.
 *   vectorization 30 -> 2 cyc/element AND it now pipelines; the scalar loop did
 *                 not, which is what made it 7.9 ms and the second-largest
 *                 compute cost in the design. BIT-IDENTICAL, checked by
 *                 make x86sim_check KUT=cmul SCENARIO=s7 / cmul_stress.
 *   tile memory   flt_local + acc_local, 2 x CMUL_N cint16 = 2 x PATCH_COLS x
 *                 FFT_COL_WS x 4 B — 8 KB at the shipping FFT_COL_WS=8. Both are
 *                 alignas(32): x86sim does NOT enforce alignment, so omitting it
 *                 passes every bit-exactness check and then misbehaves on hardware.
 *   interface     Two input ports at CMUL_SPLIT_ACCUM=1 (H on gmio_cmul_in,
 *                 the running sum on gmio_accum_in) instead of one packed buffer.
 *                 That deletes 2 MB/frame of host BO->BO memcpy outright rather
 *                 than optimising it; see docs/thesis/results/apu_stages.csv.
 *   accumulator   In DDR, 128x128 cint16 = 64 KB. On-tile does not work as usually
 *                 stated: it is a read-modify-write by ONE kernel across
 *                 invocations, so as a shared_buffer it is a graph CYCLE whose
 *                 required delay is CMUL_N_CHUNKS (16) invocations, not 1.
 *   caveat        This kernel is SCHEDULE-FRAGILE. A branching sat16 slowed it
 *                 enough to blow the aiesim wall clock, and chess_prepare_for_
 *                 pipelining has put the assembler into OOM here. Re-measure the
 *                 arithmetic loop's reported cycles after any edit.
 *                 `make aiesim` needs CMUL_SPLIT_ACCUM=0 (a 2025.2 aiesim
 *                 deadlock); hardware is fine either way.
 *
 * @thesis subsec:operacjeCzestotliwosc | A-01,A-04 | Complex multiply against the conjugated
 *   filter plus cross-channel accumulation; the accumulator lives in DDR, and the file header
 *   explains why an on-tile version is a graph cycle.
 */

#include "cmul_accum_kernel.h"
#include <aie_api/aie.hpp>

// Arithmetic implementation.
//   1 (default) = vectorized aie::mac / srs
//   0           = the original scalar loop, kept reachable for bisection
// The two are BIT-IDENTICAL by construction (see the derivation above the
// vector loop) and that is checked, not asserted:
//   make x86sim_check KUT=cmul SCENARIO=s7
//   make x86sim_check KUT=cmul SCENARIO=cmul_stress
#ifndef CMUL_VECTORIZE
#  define CMUL_VECTORIZE 1
#endif

static constexpr int CMUL_N = PATCH_COLS * FFT_COL_WS;  // 256

// Round-to-nearest bias. A bare arithmetic >> truncates toward -infinity, which
// biases every negative bin of the spectrum by up to one LSB in the same
// direction — a systematic DC offset in the response, not just noise. One add
// per product removes it, and it is branchless so the do-loop stays tight.
// @thesis subsec:arytmetyka | B-02,B-03 | H_SHIFT applied here: the one knob upstream of BOTH
//   the accumulator and the response, and the only non-host-only parameter in the build table.
static constexpr int32_t CMUL_RND = (CMUL_H_SHIFT > 0) ? (1 << (CMUL_H_SHIFT - 1)) : 0;

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
// @thesis subsec:arytmetyka | B-04 | Saturating int32->int16: the rail is per COMPONENT, which
//   is why a magnitude of 46340 = 32767*sqrt(2) is not overshoot and why `rails` is the only
//   saturation instrument.
static inline int16_t sat16(int32_t v)
{
    v = (v >  32767) ?  32767 : v;
    v = (v < -32768) ? -32768 : v;
    return (int16_t)v;
}

// Tile-local scratch for filter and accumulator halves.
// Placed in tile LDM (not stack) to avoid overflow; 2×1024 B is negligible.
//
// alignas(32) is REQUIRED, not decorative: both the v8cint16 copy below and the
// aie::load_v in the vectorized arithmetic are ALIGNED vector accesses, and a
// cint16_t array carries only 4-byte natural alignment. x86sim does not enforce
// this, so a missing alignment here passes every bit-exactness check and then
// misbehaves on hardware — the worst possible failure mode.
alignas(32) static cint16_t flt_local[CMUL_N];
alignas(32) static cint16_t acc_local[CMUL_N];

void cmul_accum_kernel(
    input_buffer<cint16_t>  &fft_col_in,
    output_buffer<cint16_t> &accum_out,
#if CMUL_SPLIT_ACCUM
    input_buffer<cint16_t>  &cmul_in,
    input_buffer<cint16_t>  &accum_in)
{
    cint16_t* flt_src = cmul_in.data();
    cint16_t* acc_src = accum_in.data();
#else
    input_buffer<cint16_t>  &cmul_in)
{
    cint16_t* flt_src = cmul_in.data();
    cint16_t* acc_src = flt_src + CMUL_N;
#endif

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

    cint16_t* __restrict in_ptr  = fft_col_in.data();
    cint16_t* __restrict out_ptr = accum_out.data();

#if CMUL_VECTORIZE

    // ------------------------------------------------------------------
    // Vectorized arithmetic — BIT-IDENTICAL to the scalar loop below.
    //
    // The scalar loop cost 30 cycles per complex element and was NOT pipelined
    // (see the note below about the assembler OOM), which at 64 invocations x 16
    // channels made this kernel ~7.9 ms/frame (docs/thesis/results/aie_compute.csv,
    // row `cmul_accum (scalar)`) — the second-largest compute cost
    // in the design after conv2d, for four multiplies and two adds.
    //
    // Why this is exactly equal to the scalar code, not merely close:
    //
    //   from_vector(acc, S) puts acc*2^S in the accumulator, EXACTLY — acc is
    //   int16 and cacc64 has room to spare. Then
    //
    //     srs_S( acc*2^S + prod )
    //       = sat16( floor( (acc*2^S + prod + 2^(S-1)) / 2^S ) )
    //       = sat16( acc + floor( (prod + 2^(S-1)) / 2^S ) )
    //       = sat16( acc + ((prod + CMUL_RND) >> CMUL_H_SHIFT) )
    //
    //   because acc*2^S is an exact multiple of 2^S and so passes through the
    //   floor untouched. That identity is the whole trick, and it is why the
    //   accumulator must be folded in BEFORE the shift.
    //
    //   Saturating once at the end also matters. Converting the product to
    //   cint16 first and then adding would clamp twice, and double clamping is
    //   WRONG whenever the product overflows int16 but the sum comes back in
    //   range (prod=+40000, acc=-20000: true 20000, double-clamped 12767).
    //
    // The two mode settings are load-bearing:
    //   positive_inf = "round to nearest, ties toward +inf" = (x + 2^(S-1)) >> S
    //     with an arithmetic shift. conv_even (a common default) would differ on
    //     exact ties and break bit-exactness.
    //   saturate reproduces sat16(). The alternative wraps, and a wrap here
    //     flips the accumulated spectrum's sign and sends the argmax to a
    //     garbage index — this kernel has shipped that bug once already.
    aie::set_rounding(aie::rounding_mode::positive_inf);
    aie::set_saturation(aie::saturation_mode::saturate);

    constexpr int VEC = 8;
    static_assert(CMUL_N % VEC == 0, "CMUL_N must be a whole number of vectors");

    for (int i = 0; i < CMUL_N / VEC; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(CMUL_N / VEC, CMUL_N / VEC)
    {
        const aie::vector<cint16, VEC> vF = aie::load_v<VEC>((cint16*)in_ptr + i * VEC);
        const aie::vector<cint16, VEC> vH = aie::load_v<VEC>((cint16*)flt_local + i * VEC);
        const aie::vector<cint16, VEC> vA = aie::load_v<VEC>((cint16*)acc_local + i * VEC);

        // Seed the accumulator with acc << S, then MAC the product onto it.
        aie::accum<cacc64, VEC> a;
        a.from_vector(vA, CMUL_H_SHIFT);

        // F * conj(H): the filter is stored un-conjugated and the conjugation
        // happens here, exactly as the scalar code's sign flip on the imaginary
        // product did. See the conjugation note in mosse_filter.h — getting this
        // backwards is invisible whenever the target is centred.
        a = aie::mac(a, vF, aie::op_conj(vH));

        aie::store_v((cint16*)out_ptr + i * VEC, a.to_vector<cint16>(CMUL_H_SHIFT));
    }

#else

    // Scalar arithmetic on tile-local arrays — fast (~1 cycle per read).
    // No chess_prepare_for_pipelining: that pragma + int32 arithmetic causes
    // the noodle assembler to OOM-kill during compilation on this kernel.
    // (That OOM is why this loop ran unpipelined at 30 cycles/element. The
    // vectorized path above does not hit it — different code path entirely.)
    for (int i = 0; i < CMUL_N; ++i) {
        int32_t re = (int32_t)in_ptr[i].real * (int32_t)flt_local[i].real
                   + (int32_t)in_ptr[i].imag * (int32_t)flt_local[i].imag;
        int32_t im = (int32_t)in_ptr[i].imag * (int32_t)flt_local[i].real
                   - (int32_t)in_ptr[i].real * (int32_t)flt_local[i].imag;
        out_ptr[i].real = sat16((int32_t)acc_local[i].real
                                + ((re + CMUL_RND) >> CMUL_H_SHIFT));
        out_ptr[i].imag = sat16((int32_t)acc_local[i].imag
                                + ((im + CMUL_RND) >> CMUL_H_SHIFT));
    }

#endif  // CMUL_VECTORIZE
}
