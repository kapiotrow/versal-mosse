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

def conv2d_relu_map(patch_int8: np.ndarray, weights_64b: bytes) -> np.ndarray:
    """The post-ReLU, pre-window feature map — conv2d_kernel.cpp up to line 226.

    Split out from simulate_conv2d because Stage B1 needs this map's mean: the
    host feeds the previous frame's value back as mean_prev.
    """
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
    # ReLU + saturate, matching conv2d_kernel.cpp:224-227 exactly. NOTE this
    # used to be a bare clip(-32768, 32767): the model was missing the ReLU
    # entirely and so did not describe the kernel it claimed to replicate.
    out = np.where(shifted > 32767, 32767, np.where(shifted <= 0, 0, shifted))
    return out.astype(np.int64)


def simulate_conv2d(patch_int8: np.ndarray, weights_64b: bytes,
                    mean_prev: int = 0) -> np.ndarray:
    """Apply one channel of conv2d_kernel: 3×3 INT8 MAC + ReLU + B1 + Hanning.

    Replicates the integer arithmetic in conv2d_kernel.cpp exactly:
      acc     = bias_acc + Σ_{kr,kc} w[kr][kc] * x_pad[r+kr, c+kc]
      out16   = 0 if (acc >> out_shift) <= 0 else saturate_int16(acc >> out_shift)
      centred = saturate_int16(out16 - mean_prev)          # Stage B1
      wnd     = (((centred * h_r) >> 15) * h_c) >> 15

    Returns float64 array shape (PATCH_ROWS, PATCH_COLS) representing the
    real part of the cint16 output (imag = 0 as per the kernel).
    """
    out = conv2d_relu_map(patch_int8, weights_64b)

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
# Expected-output text file (parsed by mosse_graph.cpp at runtime)
# ---------------------------------------------------------------------------

def write_expected_txt(path: str, *,
                       peak_idx: int,
                       peak_re_lo: int, peak_re_hi: int,
                       peak_im_lo: int, peak_im_hi: int,
                       max_noise: int,
                       peak_tol: int = 0,
                       skip_snr: bool = False,
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
    # Single-channel weight buffer: mosse_graph.cpp loads this instead of zeroing.
    # Stage B1's mean_prev lives in bytes [18:22] — see conv2d_kernel.h.
    if weights_64b is not None:
        wb = bytearray(weights_64b)
        struct.pack_into('<i', wb, 18, int(mean_prev))
        with open(os.path.join(sdir, 'weights_ch0.bin'), 'wb') as f:
            f.write(bytes(wb))
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

    # uniform arrays used in multiple scenarios
    ones_re  = np.ones(N,  dtype=np.float64)
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
    # H_MAX must be chosen against AMP, not fixed. cmul is an integer multiply, so
    # accum_out[0] = F[0,0] * H[0,0] = AMP * H_MAX. The original 4096 was safe only
    # because AMP was 1; at AMP=100 it would be 409600 and saturate cint16 (32767),
    # turning this scenario into a saturation test instead of a filter-shape test.
    # Cap the product at ~20000 to leave headroom.
    H_MAX = max(1, 20000 // AMP)
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
        peak_im_lo=-peak_sym(2 * H_MAX), peak_im_hi=peak_sym(2 * H_MAX),
        max_noise=0, skip_snr=True,        # Gaussian has non-zero off-peak values
        check_accum0=True, accum0_re=scale_accum(AMP * H_MAX), accum0_im=0,
        description=f"impulse@(0,0), H*=Gaussian(σ={sigma:.0f},max={H_MAX}), acc={{0,0}} — per-element flt_local",
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
        check_accum0=False,
        description=f"Stage A patch → conv2d+ReLU+B1(mean_prev={s6_mean_prev})+Hanning "
                    f"— FULL preprocessing path, peak@{s6_peak_idx} "
                    f"({'negative' if s6_peak_negative else 'positive'})",
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
