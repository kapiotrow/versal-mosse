#!/usr/bin/env python3
"""
gen_aiesim_vectors.py
Generate per-scenario aiesim test data for the MOSSE pipeline.

Usage:
    python3 gen_aiesim_vectors.py [output_dir]

output_dir defaults to design/aie_src/aiesim_data.

Per-scenario layout under output_dir/<scenario>/:
    patch_in.txt       — PLIO input (int8, plio_128_bits format)
    cmul_filter.bin    — H* as flat int16 LE pairs (re,im)  [PATCH_ELEMS × 2 int16]
    cmul_accum.bin     — accum_prev as flat int16 LE pairs   [PATCH_ELEMS × 2 int16]
    expected.txt       — pass/fail bounds for mosse_graph.cpp

Legacy top-level files (kept for backward compat):
    patch_in.txt       — impulse at (0,0)
    patch_in_const.txt — constant all-ones

Scenarios
---------
s0  Baseline: impulse@(0,0), H*={1,0}, acc={1,0}.  Tests basic cmul+accum.
s1  Off-centre: impulse@(17,42), H*={1,0}, acc={0,0}.  Tests spatial localisation.
s2  Constant patch: H*={1,0}, acc={0,0}.  Tests DC/large-value cmul path.
s3  Imaginary filter: impulse@(0,0), H*={0,1}, acc={0,0}.  Tests sign of im cross-term.
s4  Gaussian filter: impulse@(0,0), H*=2D-Gaussian(σ=4), acc={0,0}.  Tests non-uniform flt.
s6  Stage-A patch through the REAL conv2d path (CONV2D_MODE=0), H=unity.
s7  s6's patch through a REAL MOSSE filter (per-bin complex H, Q1.15), target
    off-centre. The only scenario that exercises H_SHIFT with a non-identity filter,
    and the only one that asserts a peak-to-sidelobe ratio.

Filters are written in Q1.15: cmul_accum computes (F*H + rnd) >> H_SHIFT, so
"unity gain" is H_UNITY (32767), not the integer 1. See the H_SHIFT block in the
Makefile.
"""

import sys
import os
import struct
import numpy as np

# Geometry must match the build under test. The Makefile passes these through
# from PATCH_ROWS / PATCH_COLS / PLIO_BEAT_SAMPLES so `make aiesim` and the AIE
# graph never disagree — a 128×128 vector set fed to a 64×64 graph silently
# starves or overruns the PLIO instead of failing loudly.
PATCH_ROWS = int(os.environ.get('GEN_PATCH_ROWS', 128))
PATCH_COLS = int(os.environ.get('GEN_PATCH_COLS', 128))
N = PATCH_ROWS * PATCH_COLS
# MUST match ifft_graph.h FFT_2D_TP_IFFT_COL_SHIFT. The Makefile feeds both from
# the single IFFT_COL_SHIFT variable — never set them independently, because every
# expected response peak scales by 2^(REF-shift) and a mismatch silently invalidates
# all of them.
IFFT_SHIFT_COL = int(os.environ.get('GEN_IFFT_COL_SHIFT', 12))
IFFT_SHIFT_ROW = int(os.environ.get('GEN_IFFT_ROW_SHIFT', 0))
FFT_SHIFT      = int(os.environ.get('GEN_FFT_SHIFT', 0))
# MUST match CMUL_H_SHIFT in cmul_accum_kernel.cpp — the Makefile feeds both from
# the single H_SHIFT variable. The kernel computes (F*H + rnd) >> H_SHIFT, so a
# filter that is meant to be UNITY GAIN is not the integer 1 any more; it is
# H_UNITY below. Every scenario written before this shift existed passed a literal
# 1, which now multiplies the spectrum by 1/32768 and produces an all-zero
# response — hence H_UNITY rather than a hand-edited constant per scenario.
H_SHIFT = int(os.environ.get('GEN_H_SHIFT', 15))

# Q1.15 representation of 1.0, clamped to int16. At H_SHIFT=15 this is 32767, so
# the effective gain is 32767/32768 — one part in 32768 below unity, which is far
# under the ~2% round-trip loss the peak bounds already tolerate. At H_SHIFT=0 it
# degenerates to the literal 1 the old scenarios used, so setting H_SHIFT=0
# reproduces the pre-shift behaviour exactly for bisection.
H_UNITY = min(32767, max(1, 1 << H_SHIFT))
assert H_SHIFT <= 15, (
    f"H_SHIFT={H_SHIFT} cannot represent unity gain in int16 "
    f"(needs {1 << H_SHIFT} > 32767); scenarios would all under-scale"
)

# Additive DC-bin loss per cint16 FFT pass, measured at 64-point / TP_SHIFT=0.
# See the long note in main()'s S2 section for why this is additive, not a gain
# factor. Module scope because S2_CONST is derived from it.
FFT_DC_TRUNC = 21

# Response magnitude depends on the TOTAL normalization budget, not on the col shift
# alone. The forward FFT applies FFT_SHIFT on BOTH the row and col pass, so:
#
#     total = 2*FFT_SHIFT + IFFT_SHIFT_ROW + IFFT_SHIFT_COL
#
# The scenarios were calibrated at total = 12 (FFT_SHIFT=0, row=0, col=12). Any
# budget summing to 12 leaves the response scale unchanged — which is exactly how
# normalization gets moved onto the forward pass (real conv2d output overflows the
# cint16 FFT at FFT_SHIFT=0) without recalibrating every expected value.
#
# Scaling on the col shift alone would be silently wrong: moving the budget to
# FFT_SHIFT=6 / col=0 would inflate every expected peak by 4096x.
#
# accum0 values are NOT response-domain — but they ARE affected by FFT_SHIFT, since
# they are read after the forward FFT. See the note at the accum0 definitions.
IFFT_REF_TOTAL   = 12
IFFT_SHIFT_TOTAL = 2 * FFT_SHIFT + IFFT_SHIFT_ROW + IFFT_SHIFT_COL
# Shift applied by the INVERSE transform alone (both passes). Distinct from
# IFFT_SHIFT_TOTAL, which also counts the two forward passes. Use this for values
# read in the response domain but derived from an accumulator-domain quantity —
# the forward shift is already baked into the accumulator, so counting it twice
# would double-scale.
IFFT_SHIFT_IFFT  = IFFT_SHIFT_ROW + IFFT_SHIFT_COL


def scale_peak(v: int) -> int:
    """Scale a total-budget-12 response magnitude to the configured shift budget."""
    d = IFFT_REF_TOTAL - IFFT_SHIFT_TOTAL
    return (int(v) << d) if d >= 0 else (int(v) >> -d)


def scale_accum(v: int) -> int:
    """Scale a pre-IFFT (accumulator-domain) value for the forward FFT shift.

    accum0 is read after both forward passes, so it scales by 2^(2*FFT_SHIFT).
    """
    return int(v) >> (2 * FFT_SHIFT)


def peak_lo(v: int) -> int:
    """Loss-tolerant lower bound, scaled to the configured shift.

    A full round trip rounds at every cint16 stage (row-FFT -> col-FFT -> cmul ->
    row-IFFT -> col-IFFT); measured loss is ~2%, so a bound at the exact ideal
    value fails. Allow 10%, floor of 1.
    """
    return max(1, scale_peak(max(1, (9 * int(v)) // 10)))


def peak_hi(v: int) -> int:
    """Upper bound, scaled and clamped to what cint16 can actually represent."""
    return min(32767, max(1, scale_peak(int(v))))


def peak_sym(v: int) -> int:
    """Symmetric (+/-) bound magnitude, scaled and clamped to int16."""
    return min(32767, max(1, scale_peak(abs(int(v)))))


def q15_gain(f: int, h: int) -> int:
    """One cmul product exactly as cmul_accum_kernel.cpp computes it.

    (f*h + rnd) >> H_SHIFT, with round-to-nearest. Mirrors CMUL_RND in the kernel;
    keeping the two in one place is the point — an expected accumulator value that
    models the multiply differently from the kernel is worse than no check at all.
    """
    rnd = (1 << (H_SHIFT - 1)) if H_SHIFT > 0 else 0
    return (int(f) * int(h) + rnd) >> H_SHIFT

# int8 samples per PLIO beat: plio_128_bits → 16, plio_32_bits → 4.
PLIO_BEAT_SAMPLES   = int(os.environ.get('GEN_PLIO_BEAT_SAMPLES', 16))
# Pack 4 int8 into one int32 per line (plio_32_bits + input_stream<int32>).
PLIO_PACK_INT32     = int(os.environ.get('GEN_PLIO_PACK_INT32', 1)) != 0
PLIO_PADDING_FRAMES = 4    # zero-pad frames to prevent PLIO starvation in cycle-approx ISS


# ---------------------------------------------------------------------------
# Hanning window (matches hanning_128.h — DO NOT edit independently)
# ---------------------------------------------------------------------------
import math as _math
# PERIODIC window (denominator PATCH_ROWS), matching _gen_hanning_h in
# scripts/export_weights.py. Its 2D DFT has exactly 9 non-zero bins, which the
# host relies on to cancel the pre-window feature mean in the frequency domain.
# The symmetric (n-1) form leaks across all bins and breaks that identity.
HANNING = np.array(
    [round(_math.sin(_math.pi * i / PATCH_ROWS) ** 2 * 32767) for i in range(PATCH_ROWS)],
    dtype=np.int32
)


# ---------------------------------------------------------------------------
# Conv2d kernel simulation (matches conv2d_kernel.cpp exactly)
# ---------------------------------------------------------------------------

# Whether conv2d applies the half-wave rectifier. MUST match the CONV_RELU the
# kernel was built with — the Makefile passes it as GEN_CONV_RELU from the same
# variable that feeds AIE_FLAGS. A scenario generated with the wrong setting
# produces expected values for a datapath the kernel does not run, which fails
# the scenario for a reason that has nothing to do with the pipeline.
CONV_RELU = int(os.environ.get('GEN_CONV_RELU', 1))


def conv2d_relu_map(patch_int8: np.ndarray, weights_64b: bytes,
                    relu: bool = None) -> np.ndarray:
    """The post-nonlinearity, pre-window feature map — conv2d_kernel.cpp up to
    the Stage B1 subtraction.

    Split out from simulate_conv2d because Stage B1 needs this map's mean: the
    host feeds the previous frame's value back as mean_prev.

    `relu` defaults to the module-level CONV_RELU; pass it explicitly to model a
    setting other than the one this build uses.
    """
    if relu is None:
        relu = bool(CONV_RELU)
    w     = np.frombuffer(weights_64b[0:9], dtype=np.int8).reshape(3, 3).astype(np.int64)
    shift = int(weights_64b[9])
    bias  = struct.unpack_from('<i', weights_64b, 10)[0]

    x  = patch_int8.reshape(PATCH_ROWS, PATCH_COLS).astype(np.int64)
    xp = np.pad(x, 1, mode='constant')     # zero-padding = conv padding=1

    acc = np.full((PATCH_ROWS, PATCH_COLS), bias, dtype=np.int64)
    for kr in range(3):
        for kc in range(3):
            acc += w[kr, kc] * xp[kr:kr + PATCH_ROWS, kc:kc + PATCH_COLS]

    shifted = acc >> shift
    # ReLU + saturate, matching conv2d_kernel.cpp's `#if CONV_RELU` branch
    # exactly. NOTE this used to be a bare clip(-32768, 32767): the model was
    # missing the ReLU entirely and so did not describe the kernel it claimed to
    # replicate. With relu=False it becomes that clip again — but deliberately,
    # and only when the kernel was built the same way.
    if relu:
        out = np.where(shifted > 32767, 32767, np.where(shifted <= 0, 0, shifted))
    else:
        out = np.clip(shifted, -32768, 32767)
    return out.astype(np.int64)


def simulate_conv2d(patch_int8: np.ndarray, weights_64b: bytes,
                    mean_prev: int = 0, relu: bool = None) -> np.ndarray:
    """Apply one channel of conv2d_kernel: 3×3 INT8 MAC + ReLU + B1 + Hanning.

    Replicates the integer arithmetic in conv2d_kernel.cpp exactly:
      acc     = bias_acc + Σ_{kr,kc} w[kr][kc] * x_pad[r+kr, c+kc]
      out16   = 0 if (acc >> out_shift) <= 0 else saturate_int16(acc >> out_shift)
      centred = saturate_int16(out16 - mean_prev)          # Stage B1
      wnd     = (((centred * h_r) >> 15) * h_c) >> 15

    Returns float64 array shape (PATCH_ROWS, PATCH_COLS) representing the
    real part of the cint16 output (imag = 0 as per the kernel).
    """
    out = conv2d_relu_map(patch_int8, weights_64b, relu)

    # Stage B1: remove the previous frame's mean BEFORE the window.
    centred = np.clip(out - int(mean_prev), -32768, 32767)

    # Separable Hanning window (Q1.15 integer arithmetic).
    # NumPy >> on negative int64 is arithmetic, matching signed C++ >>.
    h_r = HANNING[:, None]   # [128, 1]
    h_c = HANNING[None, :]   # [1, 128]
    wnd = (centred * h_r) >> 15
    wnd = (wnd * h_c) >> 15
    wnd = np.clip(wnd, -32768, 32767)

    return wnd.astype(np.float64)


# ---------------------------------------------------------------------------
# cmul_accum kernel simulation (matches cmul_accum_kernel.cpp exactly)
# ---------------------------------------------------------------------------

def simulate_cmul(in_re, in_im, flt_re, flt_im, acc_re, acc_im):
    """One cmul_accum invocation: F (*) conj(H) >> H_SHIFT, accumulated.

    Replicates cmul_accum_kernel.cpp:116-127 exactly:
        re = in.re*flt.re + in.im*flt.im
        im = in.im*flt.re - in.re*flt.im
        out.re = sat16(acc.re + ((re + CMUL_RND) >> CMUL_H_SHIFT))
        out.im = sat16(acc.im + ((im + CMUL_RND) >> CMUL_H_SHIFT))

    Three details that are easy to get wrong and silent when wrong:
      * The filter is stored UN-conjugated; the sign flip on the imaginary
        product is where the conjugation happens. See the conjugation note in
        mosse_filter.h — getting this backwards is invisible on a centred target.
      * The shift is round-to-nearest (CMUL_RND), not truncating. A bare >>
        biases every negative bin one way and shows up as a DC offset in the
        response, not as noise.
      * The accumulate SATURATES. It used to wrap, which flipped the spectrum's
        sign on overflow and sent the argmax to a garbage index.

    numpy's >> on int64 is arithmetic, matching signed C++ >>.

    Inputs and outputs are int64 arrays of any matching shape.
    """
    rnd = (1 << (H_SHIFT - 1)) if H_SHIFT > 0 else 0

    in_re  = np.asarray(in_re,  dtype=np.int64)
    in_im  = np.asarray(in_im,  dtype=np.int64)
    flt_re = np.asarray(flt_re, dtype=np.int64)
    flt_im = np.asarray(flt_im, dtype=np.int64)

    re = in_re * flt_re + in_im * flt_im
    im = in_im * flt_re - in_re * flt_im

    out_re = np.asarray(acc_re, dtype=np.int64) + ((re + rnd) >> H_SHIFT)
    out_im = np.asarray(acc_im, dtype=np.int64) + ((im + rnd) >> H_SHIFT)

    return (np.clip(out_re, -32768, 32767),
            np.clip(out_im, -32768, 32767))


# ---------------------------------------------------------------------------
# PLIO text file
# ---------------------------------------------------------------------------

def write_plio_txt(path: str, samples: np.ndarray) -> None:
    """Write the PatchIn PLIO stimulus file.

    The ISS parses this file in units of the *stream element type of the kernel
    port*, not in pixels, and it enforces exactly one beat per line:

        got 4 expected 1  →  4 values on a line where the port reads one int32

    mosse_graph.h creates PatchIn as plio_32_bits and conv2d reads
    input_stream<int32>, so a beat is a single int32 packing 4 int8 pixels
    little-endian — matching conv2d_kernel.cpp's unpack:
        pixel0 = w & 0xFF, pixel1 = (w>>8) & 0xFF, ...
    Set GEN_PLIO_PACK_INT32=0 for the older plio_128_bits + int8-stream build,
    which took raw int8 values, PLIO_BEAT_SAMPLES per line.
    """
    assert samples.dtype == np.int8 and len(samples) == N
    padding = np.zeros(N * PLIO_PADDING_FRAMES, dtype=np.int8)
    all_samples = np.concatenate([samples, padding])

    with open(path, 'w') as f:
        if not PLIO_PACK_INT32:
            for i in range(0, len(all_samples), PLIO_BEAT_SAMPLES):
                f.write(' '.join(str(int(s)) for s in all_samples[i:i + PLIO_BEAT_SAMPLES]) + '\n')
            return

        assert len(all_samples) % 4 == 0, "packed int32 stimulus needs a multiple of 4 pixels"
        # Two's-complement bytes → unsigned 32-bit word → signed, since the ISS
        # reads these lines as signed int32 decimals.
        b = all_samples.astype(np.uint8).astype(np.uint32)
        words = b[0::4] | (b[1::4] << 8) | (b[2::4] << 16) | (b[3::4] << 24)
        words = words.astype(np.uint32).astype(np.int32)   # wrap to signed
        for w in words:
            f.write(f'{int(w)}\n')


# ---------------------------------------------------------------------------
# Binary GMIO data
# ---------------------------------------------------------------------------

def write_cint16_bin(path: str, re_flat: np.ndarray, im_flat: np.ndarray) -> None:
    """Write PATCH_ELEMS cint16 values as flat int16 LE binary (re,im interleaved)."""
    assert len(re_flat) == N and len(im_flat) == N
    buf = np.empty(N * 2, dtype='<i2')
    buf[0::2] = np.clip(np.round(re_flat), -32768, 32767).astype(np.int16)
    buf[1::2] = np.clip(np.round(im_flat), -32768, 32767).astype(np.int16)
    buf.tofile(path)


# ---------------------------------------------------------------------------
# cmul saturation stress scenario (kernel-only bit-exactness harness)
# ---------------------------------------------------------------------------

def write_cmul_stress_scenario(out_dir: str) -> None:
    """Write a scenario that actually drives cmul_accum's saturating add.

    NO OTHER SCENARIO DOES. Measured across s0-s7, only s0 has a non-zero
    accum_prev at all (max 1024) and none comes near the +/-32767 rail, so the
    sat16() clamp in cmul_accum_kernel.cpp is dead code under every existing
    test. That clamp is not incidental: it replaced a cast that WRAPPED, and the
    wrap flipped the accumulated spectrum's sign on overflow and sent the argmax
    to a garbage index (see the cmul_accum entry in CLAUDE.md). A rewrite that
    reintroduced wrapping would pass every scenario in the repo.

    This directory holds only the three .bin files the kernel-only harness reads
    — no expected.txt, because it is not an end-to-end scenario and mosse_graph's
    aiesim harness should not be pointed at it.

    Cases, cycled per element so all of them appear in every 1024-sample chunk:
      0  high rail   : positive product onto a near-max accumulator
      1  low rail    : negative product onto a near-min accumulator
      2  no clamp    : near-rail accumulator that must NOT clamp
      3  max operands: |F| and |H| both at full scale (largest int32 intermediate)
      4  min accum   : accum_prev at -32768 exactly, zero product
      5  mid-range   : ordinary values, so the common path is still covered

    F is allowed to be -32768 (it comes from the FFT and can be), but H is not:
    filter_quantize_q15() clamps the filter to -32767 precisely so that
    in.re*flt.re + in.im*flt.im cannot reach 2^31. Generating H = -32768 here
    would test a value production never emits.
    """
    os.makedirs(out_dir, exist_ok=True)

    f_re = np.zeros(N, dtype=np.int64); f_im = np.zeros(N, dtype=np.int64)
    h_re = np.zeros(N, dtype=np.int64); h_im = np.zeros(N, dtype=np.int64)
    a_re = np.zeros(N, dtype=np.int64); a_im = np.zeros(N, dtype=np.int64)

    # A product big enough to clear the rail after >>H_SHIFT regardless of shift.
    BIG = 32767

    for i in range(N):
        case = i % 6
        if case == 0:
            f_re[i], f_im[i] = BIG, 0
            h_re[i], h_im[i] = BIG, 0
            a_re[i], a_im[i] = 32000, 32000
        elif case == 1:
            f_re[i], f_im[i] = BIG, 0
            h_re[i], h_im[i] = -32767, 0
            a_re[i], a_im[i] = -32000, -32000
        elif case == 2:
            f_re[i], f_im[i] = 100, -100
            h_re[i], h_im[i] = 1, 1
            a_re[i], a_im[i] = 32700, -32700
        elif case == 3:
            f_re[i], f_im[i] = -32768, -32768
            h_re[i], h_im[i] = -32767, 32767
            a_re[i], a_im[i] = 0, 0
        elif case == 4:
            f_re[i], f_im[i] = 0, 0
            h_re[i], h_im[i] = 12345, -6789
            a_re[i], a_im[i] = -32768, 32767
        else:
            f_re[i], f_im[i] = (i % 4001) - 2000, (i % 2999) - 1500
            h_re[i], h_im[i] = (i % 5003) - 2500, (i % 3001) - 1500
            a_re[i], a_im[i] = (i % 601) - 300, (i % 809) - 400

    write_cint16_bin(os.path.join(out_dir, "fft_col_in.bin"), f_re, f_im)
    write_cint16_bin(os.path.join(out_dir, "cmul_filter.bin"), h_re, h_im)
    write_cint16_bin(os.path.join(out_dir, "cmul_accum.bin"), a_re, a_im)

    # Report how much of the output actually rails, so a future change that
    # silently stops exercising saturation is visible here rather than never.
    o_re, o_im = simulate_cmul(f_re, f_im, h_re, h_im, a_re, a_im)
    rails = int(np.sum((np.abs(o_re) == 32767) | (o_re == -32768)) +
                np.sum((np.abs(o_im) == 32767) | (o_im == -32768)))
    print(f"  cmul_stress: {rails} of {2*N} int16 outputs at a rail "
          f"({100.0*rails/(2*N):.1f}%) — saturation IS exercised"
          if rails else
          "  cmul_stress: WARNING - nothing rails, the stress case is not stressing")


# ---------------------------------------------------------------------------
# Expected-output text file (parsed by mosse_graph.cpp at runtime)
# ---------------------------------------------------------------------------

def write_expected_txt(path: str, *,
                       peak_idx: int,
                       peak_re_lo: int, peak_re_hi: int,
                       peak_im_lo: int, peak_im_hi: int,
                       max_noise: int,
                       peak_tol: int = 0,
                       skip_snr: bool = False,
                       snr_ratio_pct: int = 0,
                       fcol_corr_pct: int = 0,
                       check_accum0: bool = False,
                       accum0_re: int = 0,
                       accum0_im: int = 0,
                       description: str = "") -> None:
    with open(path, 'w') as f:
        f.write(f"peak_idx     {peak_idx}\n")
        f.write(f"peak_re_lo   {peak_re_lo}\n")
        f.write(f"peak_re_hi   {peak_re_hi}\n")
        f.write(f"peak_im_lo   {peak_im_lo}\n")
        f.write(f"peak_im_hi   {peak_im_hi}\n")
        f.write(f"max_noise    {max_noise}\n")
        # Allowed peak displacement in PIXELS (Chebyshev radius). 0 = exact argmax.
        # Smooth responses have near-flat peaks, so an exact-argmax assertion tests
        # rounding luck rather than correctness; +/-1 px is the real tracking criterion.
        f.write(f"peak_tol     {peak_tol}\n")
        f.write(f"skip_snr     {1 if skip_snr else 0}\n")
        # RELATIVE peak-to-sidelobe assertion, in percent: the measured peak must
        # be at least this fraction of the largest non-peak element. Scale-invariant,
        # so unlike max_noise it survives a change to the shift budget. 0 disables.
        f.write(f"snr_ratio_pct {snr_ratio_pct}\n")
        # Minimum normalized correlation (percent) between the drained
        # gmio_fft_col_out tap and fft_col_out.bin. 0 disables. See the note in
        # generate_scenario() for why this is a correlation and not an equality.
        f.write(f"fcol_corr_pct {fcol_corr_pct}\n")
        f.write(f"check_accum0 {1 if check_accum0 else 0}\n")
        f.write(f"accum0_re    {accum0_re}\n")
        f.write(f"accum0_im    {accum0_im}\n")


# ---------------------------------------------------------------------------
# Scenario generation helper
# ---------------------------------------------------------------------------

def compute_fft_col_in(patch_int8: np.ndarray,
                       weights_64b: bytes = None) -> tuple:
    """
    Compute the pre-transposed row-FFT output that should be fed to gmio_fft_col_in.

    In the MOSSE pipeline:
        patch (int8, row-major) → conv2d (cint16 feature) → row FFT → transpose → col FFT

    If weights_64b is provided, the conv2d_kernel transform (3×3 MAC + Hanning window)
    is applied to the patch before computing the FFT — this makes fft_col_in.bin represent
    the actual expected output of the conv2d kernel rather than the raw patch signal.

    If weights_64b is None, the raw int8 patch is used directly (no conv2d simulation).
    This is only used for the legacy top-level patch_in.txt files.

    Returns (fft_col_re, fft_col_im), each a flat ndarray of length N in the same memory
    order as gmio_fft_col_in expects:
        fft_col_in[col_k * PATCH_ROWS + row_m]  =  F[row_m, col_k]
    After the transpose the col-FFT input is organised so that the col FFT (along axis=0)
    yields the full 2-D spectrum.
    """
    if weights_64b is not None:
        x = simulate_conv2d(patch_int8, weights_64b).reshape(PATCH_ROWS, PATCH_COLS)
    else:
        x = patch_int8.astype(np.float64).reshape(PATCH_ROWS, PATCH_COLS)
    # Row FFT (axis=1): PATCH_COLS-pt DFT of each row
    F_rows = np.fft.fft(x, axis=1)  # shape (PATCH_ROWS, PATCH_COLS)
    # Transpose so that each "row" fed to col-FFT is one column of F_rows
    F_T = F_rows.T               # shape (PATCH_COLS, PATCH_ROWS)
    # Flatten in row-major: element [k, r] → flat index k*PATCH_ROWS + r
    F_flat = F_T.flatten()       # length N = PATCH_ROWS * PATCH_COLS
    return F_flat.real, F_flat.imag


def generate_scenario(out_dir: str, name: str, patch_int8: np.ndarray,
                      H_re: np.ndarray, H_im: np.ndarray,
                      acc_re: np.ndarray, acc_im: np.ndarray,
                      weights_64b: bytes = None,
                      use_conv2d: bool = False,
                      mean_prev: int = 0,
                      **kwargs) -> None:
    """Write all files for one scenario into out_dir/name/.

    fft_col_in.bin is computed from the raw patch by default (no conv2d
    simulation). This keeps the expected test values analytically predictable
    regardless of which conv2d weights are in use.

    use_conv2d=True instead routes the patch through simulate_conv2d (3×3 MAC +
    ReLU + Stage B1 mean subtraction + Hanning window), so fft_col_in.bin
    represents what the real kernel emits. That makes `make aiesim` agree with
    `make aiesim_plio` for the scenario instead of testing a different signal.
    Used by s6.

    weights_ch0.bin is written when weights_64b is provided so that mosse_graph.cpp
    can load real weights for the GMIO transfer, exercising the weight-load path even
    though the conv2d output is discarded via the bypass mechanism.
    """
    sdir = os.path.join(out_dir, name)
    os.makedirs(sdir, exist_ok=True)
    write_plio_txt(os.path.join(sdir, 'patch_in.txt'), patch_int8)
    write_cint16_bin(os.path.join(sdir, 'cmul_filter.bin'), H_re, H_im)
    write_cint16_bin(os.path.join(sdir, 'cmul_accum.bin'), acc_re, acc_im)
    write_expected_txt(os.path.join(sdir, 'expected.txt'), **kwargs)
    # Pre-computed col-FFT input.
    if use_conv2d:
        if weights_64b is None:
            raise ValueError("use_conv2d=True requires weights_64b")
        x = simulate_conv2d(patch_int8, weights_64b, mean_prev).reshape(PATCH_ROWS, PATCH_COLS)
        F_T = np.fft.fft(x, axis=1).T.flatten()
        fci_re, fci_im = F_T.real, F_T.imag
    else:
        # Raw patch (bypasses PLIO→conv2d→row_FFT).
        fci_re, fci_im = compute_fft_col_in(patch_int8)
    write_cint16_bin(os.path.join(sdir, 'fft_col_in.bin'), fci_re, fci_im)

    # Golden col-FFT OUTPUT — the full 2-D spectrum F_ch, in the same layout the
    # new gmio_fft_col_out tap delivers:
    #     fft_col_out[k*PATCH_ROWS + m] = F2d[m, k]
    # The harness compares its drained tap against this by normalized correlation
    # rather than element-wise: cint16 rounding and DSPLib's additive DC loss make
    # an exact match impossible, but a shuffled, aliased or empty tap drops the
    # correlation immediately. This is what makes the tap's ORDERING testable.
    fci = (np.asarray(fci_re, dtype=np.float64)
           + 1j * np.asarray(fci_im, dtype=np.float64)).reshape(PATCH_COLS, PATCH_ROWS)
    # fci is already transposed (col-major); the column FFT runs along axis=1.
    fco = np.fft.fft(fci, axis=1).flatten() / float(1 << (2 * FFT_SHIFT))
    write_cint16_bin(os.path.join(sdir, 'fft_col_out.bin'), fco.real, fco.imag)
    # Single-channel weight buffer: mosse_graph.cpp loads this instead of zeroing.
    # Stage B1's mean_prev lives in bytes [18:22] — see conv2d_kernel.h.
    if weights_64b is not None:
        wb = bytearray(weights_64b)
        struct.pack_into('<i', wb, 18, int(mean_prev))
        with open(os.path.join(sdir, 'weights_ch0.bin'), 'wb') as f:
            f.write(bytes(wb))

        # ALSO write every other channel, for the kernel-only bit-exactness
        # harness (make x86sim_check KUT=conv2d KUT_CH=<n>).
        #
        # Channel 0 alone is NOT sufficient coverage, and that is measured, not
        # theoretical: on the s6 patch, ReLU never fires for ch0 — nor for 12 of
        # the 16 channels, because bias_acc is oversized (see the ReLU entry in
        # CLAUDE.md). A harness pinned to ch0 therefore cannot tell CONV_RELU=1
        # from CONV_RELU=0 at all; it passes either way. ch11 is the channel to
        # reach for: it is the only one where ReLU CLAMPS SOME BUT NOT ALL pixels
        # (12434 of 16384), so it exercises both sides of the branch. ch3, ch7
        # and ch15 clamp every pixel, which tests the dead-channel path.
        allw = _load_all_weights(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if allw:
            w2d = np.outer(HANNING, HANNING).astype(np.float64)
            for oc, w64 in enumerate(allw):
                relu = conv2d_relu_map(patch_int8, w64)
                mp = int(round(float((w2d * relu).sum()) / float(w2d.sum())))
                wbn = bytearray(w64)
                struct.pack_into('<i', wbn, 18, mp)
                with open(os.path.join(sdir, f'weights_ch{oc}.bin'), 'wb') as f:
                    f.write(bytes(wbn))
    desc = kwargs.get('description', '')
    print(f"  Written: {sdir}/  [{desc}]")


# ---------------------------------------------------------------------------
# Legacy helpers (kept for simulate_roundtrip compatibility check)
# ---------------------------------------------------------------------------

def simulate_roundtrip(patch_int8: np.ndarray) -> np.ndarray:
    x = patch_int8.astype(np.float64).reshape(PATCH_ROWS, PATCH_COLS)
    F_row  = np.fft.fft(x, axis=1)
    F_rowT = F_row.T
    F_2d   = np.fft.fft(F_rowT, axis=1)
    Y_row  = np.fft.ifft(F_2d, axis=1) * PATCH_COLS
    Y_rowT = Y_row.T
    Y_2d   = np.fft.ifft(Y_rowT, axis=1) * PATCH_ROWS
    Y_2d  /= (2 ** IFFT_SHIFT_COL)
    return Y_2d.real


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_all_weights(repo_root: str):
    """Every 64-byte channel buffer from layer0_weights.bin, or [] if absent.

    Used to give the kernel-only harness a choice of channel — see the note in
    generate_scenario about ch0 being unable to exercise ReLU.
    """
    path = os.path.join(repo_root, "design", "aie_src", "weights", "layer0_weights.bin")
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        data = f.read()
    return [data[i * 64:(i + 1) * 64] for i in range(len(data) // 64)]


def _load_ch0_weights(repo_root: str) -> bytes:
    """Load 64-byte channel-0 weight buffer from layer0_weights.bin.

    Returns bytes of length 64 on success, or 64 zero bytes if the file is not
    available (weights not yet exported — run 'make weights' first).
    """
    path = os.path.join(repo_root, "design", "aie_src", "weights", "layer0_weights.bin")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — using zero weights (run 'make weights' first)")
        return bytes(64)
    with open(path, 'rb') as f:
        data = f.read(64)
    if len(data) < 64:
        print(f"  WARNING: {path} too short — using zero weights")
        return bytes(64)
    print(f"  Loaded channel-0 weights from {path}")
    return data



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    default_out = os.path.join(repo_root, "design", "aie_src", "aiesim_data")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default_out
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load channel-0 weights for conv2d simulation
    # ------------------------------------------------------------------
    weights_ch0 = _load_ch0_weights(repo_root)

    # ------------------------------------------------------------------
    # Legacy top-level files (backward compat with old make aiesim)
    # ------------------------------------------------------------------
    # Impulse amplitude for ALL impulse scenarios.
    #
    # At the original amplitude of 1 these tests sit ON the fixed-point quantization
    # floor and cannot pass: a unit impulse has |F(k)| = 1 in every bin of its row,
    # which rounds to 0 or +/-1 in cint16 (measured: 20/4096 bins non-zero, max 3),
    # and the IFFT column shift of 12 then divides by 4096 and erases the rest.
    # int8 allows 127 and the host injects pixel values around 200 (uint8), so
    # amplitude 1 was never representative either. At AMP=100 the s1 impulse
    # round-trips to its exact index returning 98, with a noise floor of 2.
    AMP = int(os.environ.get('GEN_IMPULSE_AMP', 100))
    assert 1 <= AMP <= 127, "impulse amplitude must fit in int8"

    impulse = np.zeros(N, dtype=np.int8); impulse[0] = AMP
    write_plio_txt(os.path.join(out_dir, "patch_in.txt"), impulse)

    # s2 constant-patch level. The DC bin is ~N * S2_CONST, so this cannot be
    # scaled like the impulse scenarios — it must shrink as the patch grows.
    #
    # DERIVED, not hardcoded. It used to be a literal 7, calibrated at 64x64,
    # which made `make gen_vectors` fail outright at the Makefile's DEFAULT
    # 128x128 geometry: accum0 came out at 111979, well past cint16, and the
    # assertion below aborted the whole generator. (Pre-existing, and unrelated
    # to preprocessing — it just went unnoticed because every build on disk was
    # 64x64.) Inverting the accum0 formula against a target that leaves margin
    # below 32767 reproduces the calibrated 7 at 64x64 and yields 1 at 128x128.
    S2_DC_TARGET = 28000
    S2_CONST = max(1, int((S2_DC_TARGET / PATCH_ROWS + FFT_DC_TRUNC) / PATCH_COLS))
    const_img = np.full(N, S2_CONST, dtype=np.int8)
    write_plio_txt(os.path.join(out_dir, "patch_in_const.txt"), const_img)

    print(f"\nGenerating aiesim scenarios in: {out_dir}/")

    # Harness-only scenario for the kernel bit-exactness check. Not an aiesim
    # scenario — it has no expected.txt and no patch_in.txt. See the function
    # docstring for why it exists.
    write_cmul_stress_scenario(os.path.join(out_dir, "cmul_stress"))

    # uniform arrays used in multiple scenarios.
    # ones_re is UNITY GAIN in the kernel's Q1.15 filter format, not the integer 1
    # — see H_UNITY. Every expected value below was calibrated against a gain of
    # exactly 1 and stays valid, because H_UNITY/2^H_SHIFT = 32767/32768.
    ones_re  = np.full(N, float(H_UNITY), dtype=np.float64)
    zeros_re = np.zeros(N, dtype=np.float64)

    # ------------------------------------------------------------------
    # S0 — Baseline: impulse@(0,0), H*={1,0}, acc_prev={1,0}
    # ------------------------------------------------------------------
    # fft_col_in.bin = FFT of the raw impulse → flat spectrum of magnitude AMP.
    # cmul is a plain integer multiply (verified by s1: H*=1 gives accum0 = AMP),
    # so accum_out[0] = col_FFT[0]*H[0] + acc_prev[0] = AMP*1 + 1 = {AMP+1, 0}.
    # acc_prev stays at 1 (it is a filter/accumulator value, not pixel data, so it
    # does not scale with the impulse amplitude).
    generate_scenario(
        out_dir, "s0",
        impulse,
        H_re=ones_re,   H_im=zeros_re,
        acc_re=ones_re, acc_im=zeros_re,
        weights_64b=weights_ch0,
        peak_idx=0,
        # Peak is the round-tripped impulse (~AMP) plus the acc_prev contribution.
        # Upper bound kept generous — the acc term's exact IFFT scaling is not
        # analytically pinned down; tighten after a calibration run if desired.
        peak_re_lo=peak_lo(AMP), peak_re_hi=peak_hi(6 * AMP + 8),
        peak_im_lo=-peak_sym(3 * AMP), peak_im_hi=peak_sym(3 * AMP),
        max_noise=peak_sym(AMP // 2), skip_snr=False,
        check_accum0=True, accum0_re=scale_accum(AMP + 1), accum0_im=0,
        description="impulse@(0,0), H*={1,0}, acc={1,0} — baseline accumulation test",
    )

    # ------------------------------------------------------------------
    # S1 — Off-centre impulse: impulse@(17,42), H*={1,0}, acc_prev={0,0}
    # ------------------------------------------------------------------
    # F[0,0] = 1 (sum of raw impulse = 1), accum_out[0] = {1,0}.
    # IFFT round-trip → peak at flat index 17*128+42 = 2218.
    # Expected values scale linearly with amplitude (cmul is an integer multiply),
    # so the original 1:6 / +/-4 calibration ratios are preserved below.
    impulse_off = np.zeros(N, dtype=np.int8)
    impulse_off[17 * PATCH_COLS + 42] = AMP
    generate_scenario(
        out_dir, "s1",
        impulse_off,
        H_re=ones_re,    H_im=zeros_re,
        acc_re=zeros_re, acc_im=zeros_re,
        weights_64b=weights_ch0,
        peak_idx=17 * PATCH_COLS + 42,
        # The lower bound must allow for fixed-point loss. A full round trip is
        # row-FFT -> col-FFT -> cmul -> row-IFFT -> col-IFFT, every stage rounding
        # in cint16; measured end-to-end for AMP=100 the peak comes back as 98.
        # A bound of exactly 1*AMP assumes lossless arithmetic and fails by 2%.
        # 0.9*AMP keeps the test meaningful while tolerating the real error.
        peak_re_lo=peak_lo(AMP), peak_re_hi=peak_hi(6 * AMP),
        peak_im_lo=-peak_sym(4 * AMP), peak_im_hi=peak_sym(4 * AMP),
        # 6*AMP would be 6x the peak itself — a threshold that passes even when the
        # peak is buried in noise, i.e. no test at all. Measured noise floor is 2
        # against a peak of 98, so AMP/2 asserts a real >=2:1 SNR with ~25x margin.
        max_noise=peak_sym(AMP // 2), skip_snr=False,
        check_accum0=True, accum0_re=scale_accum(AMP), accum0_im=0,   # F[0,0]=sum=AMP
        description="impulse@(17,42), H*={1,0}, acc={0,0} — spatial localisation",
    )

    # ------------------------------------------------------------------
    # S2 — Constant patch: H*={1,0}, acc_prev={0,0}
    # ------------------------------------------------------------------
    # Raw constant patch → all energy in F[0,0], every other bin 0.
    #
    # Two corrections here, both from measurement rather than theory:
    #
    # 1. accum0_re was hardcoded to 16384 = N for a 128x128 patch ONLY. At 64x64
    #    the geometry gives 4096, so this scenario was silently wrong by 4x for
    #    every non-128 build.
    #
    # 2. The ideal-DFT value N is ALSO wrong. ifft_graph.h documents that DSPLib's
    #    cint16 FFT "does NOT scale by N ... it applies fixed-point twiddle
    #    normalization per stage, giving a non-obvious scaling constant". Measured
    #    at 64x64 with a constant patch: row DC = 43 (not 64) and accum0 = 2731
    #    (not 4096) — a factor of 2/3, and the same "43" already cited in
    #    ifft_graph.h's own calibration note.
    #
    #    The loss is ADDITIVE, not a gain factor. Two measured points at 64x64:
    #        const=1: row DC 43 vs ideal 64   (loss 21);  accum0 2731 vs 2752 (loss 21)
    #        const=7: row DC 427 vs ideal 448 (loss 21);  accum0 27307 vs 27328 (loss 21)
    #    i.e. each FFT pass subtracts a constant ~21 from a summed DC bin. (A 2/3
    #    "gain" fits the const=1 point only by coincidence — 21 is a third of 64 —
    #    and predicts 19114 for const=7, which measurement refutes: it is 27307.)
    #
    #    The loss tracks how much SUMMATION the input causes, which is why the s1
    #    impulse loses only ~3 (97 of 100): its DC bin has a single non-zero term,
    #    so it accumulates almost no per-butterfly truncation.
    #
    #    21 is measured for 64-point cint16 at TP_SHIFT=0 and will likely differ at
    #    other point sizes — re-measure if PATCH_ROWS/COLS change.
    #    (FFT_DC_TRUNC is defined at module scope; S2_CONST is derived from it.)
    s2_row_dc = PATCH_COLS * S2_CONST - FFT_DC_TRUNC     # row pass (PATCH_COLS-pt)
    s2_accum0 = PATCH_ROWS * s2_row_dc  - FFT_DC_TRUNC   # col pass (PATCH_ROWS-pt)
    assert s2_accum0 < 32768, \
        f"s2 DC bin {s2_accum0} would saturate cint16 — lower S2_CONST"
    generate_scenario(
        out_dir, "s2",
        const_img,
        H_re=ones_re,    H_im=zeros_re,
        acc_re=zeros_re, acc_im=zeros_re,
        weights_64b=weights_ch0,
        peak_idx=0,
        # Response is uniform at accum0 >> IFFT_SHIFT_COL. This is small because a
        # DC-only spectrum gets NO summation gain from the IFFT, while the col
        # shift of 12 assumes it does — see the narrowband note in ifft_graph.h.
        # At S2_CONST=1 the response was exactly 0 (2731 >> 12 == 0); S2_CONST=7
        # lifts it to ~4 so this scenario asserts something again.
        # Uniform response of accum0 >> (IFFT_SHIFT_ROW + IFFT_SHIFT_COL). Derived
        # from the measured accum0 rather than hardcoded, so it tracks the shift
        # being swept: at col shift 12 this is ~6, at col shift 6 it would be ~426.
        #
        # BOTH IFFT passes shift, so both belong here. This used to use
        # IFFT_SHIFT_COL alone, which is correct only while IFFT_SHIFT_ROW == 0 —
        # its default, which is why it never bit. It would silently inflate s2's
        # expected bounds by 2^IFFT_SHIFT_ROW the moment the row shift is used to
        # carry part of the budget.
        peak_re_lo=max(1, (scale_accum(s2_accum0) >> IFFT_SHIFT_IFFT) * 9 // 10),
        peak_re_hi=min(32767, (scale_accum(s2_accum0) >> IFFT_SHIFT_IFFT) * 2 + 8),
        peak_im_lo=-peak_sym(4), peak_im_hi=peak_sym(4),
        max_noise=0, skip_snr=True,   # uniform response: all elements ≈ equal
        check_accum0=True, accum0_re=scale_accum(s2_accum0), accum0_im=0,
        description="constant patch, H*={1,0}, acc={0,0} — DC/large-value path",
    )

    # ------------------------------------------------------------------
    # S3 — Imaginary filter: impulse@(0,0), H*={0,1}, acc_prev={0,0}
    # ------------------------------------------------------------------
    # F = {AMP,0} everywhere (raw impulse FFT).
    # accum_out[i] = {AMP*0 + 0*1, 0*0 - AMP*1} = {0,-AMP} — sign test for the
    # cmul conjugation. The magnitude is incidental here; the SIGN is the assertion.
    generate_scenario(
        out_dir, "s3",
        impulse,
        H_re=zeros_re,   H_im=ones_re,
        acc_re=zeros_re, acc_im=zeros_re,
        weights_64b=weights_ch0,
        peak_idx=0,
        # Real part must stay near zero *relative to* the signal, not within a
        # fixed +/-4 that scaling would make trivially satisfiable.
        peak_re_lo=-peak_sym(AMP // 4), peak_re_hi=peak_sym(AMP // 4),
        # Sign test: must remain strictly negative, magnitude ~AMP.
        peak_im_lo=-peak_sym(8 * AMP), peak_im_hi=-peak_lo(AMP),
        max_noise=peak_sym(AMP // 2), skip_snr=False,
        check_accum0=True, accum0_re=0, accum0_im=-scale_accum(AMP),
        description="impulse@(0,0), H*={0,1}, acc={0,0} — imaginary cross-term sign",
    )

    # ------------------------------------------------------------------
    # S4 — Gaussian filter: impulse@(0,0), H*=Gaussian(σ=4), acc_prev={0,0}
    # ------------------------------------------------------------------
    # F = {1,0} everywhere (raw impulse FFT).
    # accum_out[0] = {H_MAX, 0}; IFFT → narrow Gaussian blob at (0,0).
    k1 = np.arange(PATCH_COLS, dtype=np.float64)
    k2 = np.arange(PATCH_ROWS, dtype=np.float64)
    k1_c = np.where(k1 > PATCH_COLS // 2, k1 - PATCH_COLS, k1)
    k2_c = np.where(k2 > PATCH_ROWS // 2, k2 - PATCH_ROWS, k2)
    K1, K2 = np.meshgrid(k1_c, k2_c, indexing='ij')   # shape (PATCH_COLS, PATCH_ROWS)
    sigma = 4.0
    # Peak of the filter is full-scale Q1.15, i.e. a gain of 1.0 at the DC bin.
    # cmul now computes (F*H + rnd) >> H_SHIFT, so accum_out[0] = AMP, and the
    # headroom question this used to hand-tune (H_MAX = 20000 // AMP, chosen so the
    # raw integer product stayed under 32767) no longer exists: Q1.15 caps the gain
    # at 1 by construction, so the product can never exceed |F|.
    H_MAX = H_UNITY
    s4_accum0 = q15_gain(AMP, H_MAX)
    H_gauss = np.exp(-(K1**2 + K2**2) / (2.0 * sigma**2)) * H_MAX
    H_gauss_flat = H_gauss.flatten()   # row-major of (PATCH_COLS, PATCH_ROWS)
    generate_scenario(
        out_dir, "s4",
        impulse,
        H_re=H_gauss_flat, H_im=zeros_re,
        acc_re=zeros_re,   acc_im=zeros_re,
        weights_64b=weights_ch0,
        peak_idx=0,
        # Deliberately broad: the response peak depends on DSPLib's internal IFFT
        # scaling of a Gaussian spectrum, which is not analytically pinned down here.
        # accum0 below is the precise assertion for this scenario; these bounds only
        # catch gross breakage. Tighten from a calibration run if s4 is to be a real
        # numerical test.
        peak_re_lo=1, peak_re_hi=32767,
        peak_im_lo=-peak_sym(2 * s4_accum0), peak_im_hi=peak_sym(2 * s4_accum0),
        max_noise=0, skip_snr=True,        # Gaussian has non-zero off-peak values
        check_accum0=True, accum0_re=scale_accum(s4_accum0), accum0_im=0,
        description=f"impulse@(0,0), H*=Gaussian(σ={sigma:.0f},Q1.15 peak gain 1.0), "
                    f"acc={{0,0}} — per-element flt_local",
    )

    # ------------------------------------------------------------------
    # S6 — Full preprocessing path: Stage A output → conv2d → ReLU → B1 → Hanning
    # ------------------------------------------------------------------
    # Every other scenario feeds a synthetic patch straight to the FFT and skips
    # conv2d, so none of them exercise the preprocessing chain at all. s6 is the
    # one that does:
    #   - the patch is what roi_crop's Stage A actually emits (log → zero mean →
    #     unit L2 × ROI_NORM_Q → clip to int8), not an impulse or a constant
    #   - fft_col_in.bin is generated through simulate_conv2d, so `make aiesim`
    #     and `make aiesim_plio` are testing the same signal
    #   - mean_prev is non-zero, so Stage B1's subtraction is live
    #
    # H* = {1,0} makes the response a round trip, so the peak must land on the
    # brightest point of the windowed feature map. That location comes from the
    # golden model rather than a closed form — the 3×3 MobileNet kernel plus the
    # Hanning taper has no analytic argmax.
    ROI_NORM_Q = 32          # must match roi_crop.h
    s6_rng = np.random.default_rng(20260731)
    # Blob geometry is a FRACTION of the patch, not absolute pixels. Hardcoding
    # (44,76) put the target at column 76 — off the edge of a 64×64 patch, so the
    # scenario silently degraded to "gradient plus the tail of a blob" at any
    # geometry other than 128×128. Same trap that had S2_CONST calibrated at one
    # patch size and broken at the other.
    s6_r = int(round(0.35 * PATCH_ROWS))
    s6_c = int(round(0.60 * PATCH_COLS))
    s6_sigma = PATCH_COLS / 9.0
    rr6 = np.arange(PATCH_ROWS, dtype=np.float64).reshape(-1, 1)
    cc6 = np.arange(PATCH_COLS, dtype=np.float64).reshape(1, -1)
    # Target blob + illumination gradient + sensor noise, i.e. something with a
    # non-trivial mean and contrast — the case a bare impulse never covers.
    s6_img = (180.0 * np.exp(-(((rr6 - s6_r) ** 2 + (cc6 - s6_c) ** 2) / (2.0 * s6_sigma ** 2)))
              + 0.12 * cc6 + 0.06 * rr6 + 30.0
              + 5.0 * s6_rng.standard_normal((PATCH_ROWS, PATCH_COLS)))
    # Stage A, in float — the fixed-point version is verified separately by the
    # roi_crop C simulation.
    s6_log = np.log1p(np.clip(s6_img, 0.0, 255.0))
    s6_z   = (s6_log - s6_log.mean()) / s6_log.std()
    s6_patch = np.clip(np.round(s6_z * ROI_NORM_Q), -127, 127).astype(np.int8).flatten()

    # mean_prev = window-weighted mean of the post-ReLU map (see conv2d_kernel.h).
    # Using the exact current-frame value here; on hardware it lags by one frame
    # and the host's 9-bin correction absorbs the difference.
    s6_relu = conv2d_relu_map(s6_patch, weights_ch0)
    s6_w2d  = np.outer(HANNING, HANNING).astype(np.float64)
    s6_mean_prev = int(round(float((s6_w2d * s6_relu).sum()) / float(s6_w2d.sum())))

    s6_feat = simulate_conv2d(s6_patch, weights_ch0, s6_mean_prev)
    s6_peak_idx = int(np.argmax(np.abs(s6_feat)))
    assert int(np.abs(s6_patch).max()) <= 127, "s6 patch violates the int8 contract"

    # The response is SIGNED once Stage B1 is active, and the peak here is in fact
    # negative. Every other scenario can assume a positive peak because ReLU left
    # the feature map non-negative; subtracting the mean makes it bipolar, so the
    # largest-magnitude point is as likely to be a trough as a crest.
    # Bounds follow the sign the golden model predicts, which keeps the "response
    # was not crushed to zero" floor AND additionally asserts the sign is right.
    s6_peak_negative = bool(s6_feat.flatten()[s6_peak_idx] < 0)
    s6_re_lo, s6_re_hi = (-32767, -1) if s6_peak_negative else (1, 32767)

    generate_scenario(
        out_dir, "s6",
        s6_patch,
        H_re=ones_re,    H_im=zeros_re,
        acc_re=zeros_re, acc_im=zeros_re,
        weights_64b=weights_ch0,
        use_conv2d=True,
        mean_prev=s6_mean_prev,
        peak_idx=s6_peak_idx,
        # Location and sign are the assertions. Magnitude depends on the shift
        # settings and the conv weights, neither of which this scenario calibrates.
        peak_re_lo=s6_re_lo, peak_re_hi=s6_re_hi,
        peak_im_lo=-32768, peak_im_hi=32767,
        max_noise=0, skip_snr=True,
        peak_tol=1,
        # Check the gmio_fft_col_out tap on the two scenarios whose signal the
        # golden model reproduces exactly. Left off for s0-s4: they run in echo
        # mode, where the col-FFT input comes from file and the tap check would be
        # asserting something the scenario was not built to test.
        fcol_corr_pct=95,
        check_accum0=False,
        description=f"Stage A patch → conv2d+ReLU+B1(mean_prev={s6_mean_prev})+Hanning "
                    f"— FULL preprocessing path, peak@{s6_peak_idx} "
                    f"({'negative' if s6_peak_negative else 'positive'})",
    )

    # ------------------------------------------------------------------
    # S7 — s6's patch driven through a REAL MOSSE filter
    # ------------------------------------------------------------------
    # Everything before this passes H = unity (or an ad-hoc Gaussian), so no
    # scenario has ever exercised cmul with a filter that varies per bin in both
    # real and imaginary parts — which is the only kind a real tracker produces,
    # and the only kind that tests H_SHIFT for anything beyond "did not crash".
    #
    # H is built exactly as mosse_filter.cpp builds it on the host:
    #     H = G * conj(F) / (F*conj(F) + eps)     then normalized to Q1.15
    # so this scenario also pins down the host's quantization convention. If the
    # two ever disagree about conjugation or scale, s7 fails and the hw_emu run
    # does not have to be the thing that discovers it.
    #
    # The target G is a Gaussian placed OFF-CENTRE at (IMPULSE_DR, IMPULSE_DC) =
    # (10,-7), matching the host's synthetic test offset. Centring it would put the
    # expected peak at flat index 0 — indistinguishable from an all-zero response,
    # which is precisely the degenerate test CLAUDE.md warns about. The two offsets
    # differ in magnitude and sign so a row/col transpose and a sign flip both fail.
    S7_DR, S7_DC = 10, -7
    s7_sigma = 2.0

    # Spatial Gaussian target, wrapped (circular), centred at (S7_DR, S7_DC).
    rr7 = np.arange(PATCH_ROWS, dtype=np.float64).reshape(-1, 1)
    cc7 = np.arange(PATCH_COLS, dtype=np.float64).reshape(1, -1)
    dr7 = np.minimum((rr7 - S7_DR) % PATCH_ROWS, (S7_DR - rr7) % PATCH_ROWS)
    dc7 = np.minimum((cc7 - S7_DC) % PATCH_COLS, (S7_DC - cc7) % PATCH_COLS)
    g7  = np.exp(-(dr7**2 + dc7**2) / (2.0 * s7_sigma**2))
    G7  = np.fft.fft2(g7)                       # shape (PATCH_ROWS, PATCH_COLS)

    # F: the 2-D spectrum of the same Stage-A → conv2d → B1 → Hanning feature map
    # s6 uses, so the PLIO path feeds s7 a signal the golden model knows exactly.
    s7_feat = simulate_conv2d(s6_patch, weights_ch0, s6_mean_prev).reshape(PATCH_ROWS, PATCH_COLS)
    F7 = np.fft.fft2(s7_feat)

    # MOSSE closed form for a single training image (Bolme eq. 6 with N=1), which
    # is exactly what filter_init() computes on frame 0.
    #
    # CONJUGATION. Bolme writes the filter as H* = G ⊙ F* / (F ⊙ F*), because the
    # correlation he forms is F ⊙ H*. cmul_accum applies the conjugation itself, so
    # what gets STORED is H, not H*:
    #     F ⊙ conj(H_stored) = G   =>   H_stored = conj(G) ⊙ F / (|F|² + ε)
    # Storing Bolme's expression verbatim gives F ⊙ conj(H) = conj(G)·F/conj(F),
    # whose phase is garbage — the response then peaks at an arbitrary bin. The
    # assertion below caught exactly that. The distinction is invisible for a
    # CENTRED real target (conj(G) = G), which is another reason this scenario puts
    # the target off-centre.
    s7_eps = 1e-3 * float(np.mean(np.abs(F7) ** 2))
    H7 = np.conj(G7) * F7 / (np.abs(F7) ** 2 + s7_eps)

    # Normalize the largest |H| bin to FULL int16 scale — 32767, NOT H_UNITY.
    # Same rule as filter_quantize_q15() in mosse_filter.cpp, and the distinction
    # matters: H_UNITY is the value that represents a GAIN OF ONE (2^H_SHIFT), which
    # is what the unity-filter scenarios s0-s6 need. A real filter instead wants
    # every one of the 15 bits, and lets H_SHIFT decide where the product lands.
    # Using H_UNITY here would drop H to H_SHIFT bits of resolution.
    s7_scale = 32767.0 / float(np.max(np.abs(H7)))
    H7q = H7 * s7_scale

    # Layout: the filter is consumed in the col-FFT output order, i.e.
    # element [k*PATCH_ROWS + m] is spectrum bin (m, k) — the transpose of the
    # natural row-major order. Getting this wrong is a silent transpose bug, and it
    # only shows up on a non-square patch or an asymmetric target like this one.
    H7_flat = H7q.T.flatten()

    # Golden response: correlation is IFFT(F * conj(H)) — cmul conjugates the
    # stored filter, so the stored value is H itself, un-conjugated.
    s7_resp = np.real(np.fft.ifft2(F7 * np.conj(H7q)))
    s7_peak_idx = int(np.argmax(np.abs(s7_resp)))
    s7_exp_idx  = (S7_DR % PATCH_ROWS) * PATCH_COLS + (S7_DC % PATCH_COLS)
    assert s7_peak_idx == s7_exp_idx, (
        f"s7 golden model peaks at {s7_peak_idx}, not the target offset "
        f"{s7_exp_idx} — the filter construction is wrong, not the hardware"
    )

    # Peak-to-sidelobe ratio, Bolme §3.5: the sidelobe region EXCLUDES an 11×11
    # window around the peak. Without that exclusion the "largest non-peak element"
    # of a smooth σ=2 Gaussian is simply its neighbour at exp(-1/8) = 0.88 of the
    # peak, giving a ratio of 1.13 — an assertion that passes on any blurry blob
    # and tests nothing. Excluding the mainlobe is what makes PSR discriminative.
    s7_absr = np.abs(s7_resp)
    s7_peak_mag = float(s7_absr.flat[s7_peak_idx])
    s7_mask = np.ones_like(s7_absr, dtype=bool)
    _pr, _pc = s7_peak_idx // PATCH_COLS, s7_peak_idx % PATCH_COLS
    for _dr in range(-5, 6):
        for _dc in range(-5, 6):
            s7_mask[(_pr + _dr) % PATCH_ROWS, (_pc + _dc) % PATCH_COLS] = False
    s7_sidelobe = float(np.max(s7_absr[s7_mask]))
    s7_positive = bool(s7_resp.flat[s7_peak_idx] > 0)

    generate_scenario(
        out_dir, "s7",
        s6_patch,
        H_re=H7_flat.real, H_im=H7_flat.imag,
        acc_re=zeros_re,   acc_im=zeros_re,
        weights_64b=weights_ch0,
        use_conv2d=True,
        mean_prev=s6_mean_prev,
        peak_idx=s7_exp_idx,
        # Sign comes from the golden model, as in s6 — Stage B1 makes the response
        # bipolar, so a positive peak cannot be assumed.
        peak_re_lo=(1 if s7_positive else -32767),
        peak_re_hi=(32767 if s7_positive else -1),
        peak_im_lo=-32768, peak_im_hi=32767,
        # The absolute noise bound is useless here — the response magnitude depends
        # on the shift budget, which this scenario deliberately does not calibrate.
        # Assert the peak-to-sidelobe RATIO instead: scale-invariant, and the thing
        # that actually distinguishes a matched filter from a filter that localises
        # by luck.
        #
        # THRESHOLD IS MEASURED, NOT A FRACTION OF THE GOLDEN IDEAL. The first
        # version used 0.7 x golden on the assumption that fixed point would retain
        # 70% of the float ratio. That assumption was never checked and is wrong:
        # measured at 64x64 / FFT_SHIFT=3-0-6 / H_SHIFT=10 / ch1, the golden 38x
        # comes back as 19.6x, i.e. ~51%.
        #
        # The gap is SPECTRAL QUANTIZATION, not a defect. The accumulator peaks at
        # 466 of 32767, so the spectrum carries ~9 bits and its rounding noise
        # spreads over the whole response, lifting the sidelobe to 40 where the true
        # value is ~20. More gain would fix it (H_SHIFT=8 quadruples the signal while
        # the noise stays absolute) but rails the accumulator at 16 channels — and
        # ch1 is the WORST case here, since channels add coherently and their
        # quantization noise does not.
        #
        # 15x keeps this a real assertion: it still fails a ~25% regression, and it
        # is far above Bolme's own failure indicator (§3.5: PSR ~7 means occluded or
        # lost, 20-60 is healthy tracking). Re-measure if the shift budget moves.
        max_noise=0, skip_snr=True,
        snr_ratio_pct=1500,
        peak_tol=1,
        fcol_corr_pct=95,
        check_accum0=False,
        description=f"s6 patch + REAL MOSSE filter H=conj(G)·F/(|F|²+ε) in Q1.15 "
                    f"(H_SHIFT={H_SHIFT}), target off-centre at ({S7_DR},{S7_DC}) "
                    f"→ idx {s7_exp_idx}, PSR {s7_peak_mag / max(s7_sidelobe, 1e-12):.1f}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("Normalization reference (matches ifft_graph.h):")
    print(f"  IFFT col shift = {IFFT_SHIFT_COL}  (empirically calibrated)")
    print(f"  fft_col_in.bin computed from raw patch (expected values remain analytically calibrated)")
    print(f"  weights_ch0.bin written to each scenario dir — mosse_graph.cpp loads real weights")
    print()
    print("Run scenarios with:")
    print("  make aiesim SCENARIO=s0  N_CHANNELS=1 ITER_CNT=1")
    print("  make aiesim SCENARIO=s1  N_CHANNELS=1 ITER_CNT=1")
    print("  ... etc.")
    print()
    print(f"  impulse amplitude AMP = {AMP}  (override with GEN_IMPULSE_AMP)")
    print(f"  s4 Gaussian H_MAX = {H_MAX}  (scaled so AMP*H_MAX stays inside cint16)")
    print()
    print("NOTE: S0/S3/S4 use impulse@(0,0). With CONV2D_MODE=0 the Hanning window")
    print("  zeroes the first row/col, so their response is near-zero — they test")
    print("  cmul/IFFT numerics, not conv2d. S1 and S2 are the informative ones for")
    print("  conv2d + Hanning correctness.")
    print()
    print("NOTE: only S1 is fully calibrated against a measured run (peak 98 @ idx 1130,")
    print("  noise 2). S0/S3/S4 peak bounds are derived-but-unverified and may need one")
    print("  calibration pass; their accum0 values ARE analytically exact.")


if __name__ == "__main__":
    main()
