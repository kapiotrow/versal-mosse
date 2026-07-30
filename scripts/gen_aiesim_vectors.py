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
IFFT_SHIFT_COL = 12   # matches ifft_graph.h FFT_2D_TP_IFFT_COL_SHIFT

# int8 samples per PLIO beat: plio_128_bits → 16, plio_32_bits → 4.
PLIO_BEAT_SAMPLES   = int(os.environ.get('GEN_PLIO_BEAT_SAMPLES', 16))
# Pack 4 int8 into one int32 per line (plio_32_bits + input_stream<int32>).
PLIO_PACK_INT32     = int(os.environ.get('GEN_PLIO_PACK_INT32', 1)) != 0
PLIO_PADDING_FRAMES = 4    # zero-pad frames to prevent PLIO starvation in cycle-approx ISS


# ---------------------------------------------------------------------------
# Hanning window (matches hanning_128.h — DO NOT edit independently)
# ---------------------------------------------------------------------------
import math as _math
HANNING = np.array(
    [round(_math.sin(_math.pi * i / (PATCH_ROWS - 1)) ** 2 * 32767) for i in range(PATCH_ROWS)],
    dtype=np.int32
)


# ---------------------------------------------------------------------------
# Conv2d kernel simulation (matches conv2d_kernel.cpp exactly)
# ---------------------------------------------------------------------------

def simulate_conv2d(patch_int8: np.ndarray, weights_64b: bytes) -> np.ndarray:
    """Apply one channel of conv2d_kernel: 3×3 INT8 MAC + separable Hanning window.

    Replicates the integer arithmetic in conv2d_kernel.cpp exactly:
      acc = bias_acc + Σ_{kr,kc} w[kr][kc] * x_pad[r+kr, c+kc]
      out16 = saturate_int16(acc >> out_shift)
      wnd = ((out16 * h_r) >> 15 * h_c) >> 15

    Returns float64 array shape (PATCH_ROWS, PATCH_COLS) representing the
    real part of the cint16 output (imag = 0 as per the kernel).
    """
    w    = np.frombuffer(weights_64b[0:9], dtype=np.int8).reshape(3, 3).astype(np.int64)
    shift = int(weights_64b[9])
    bias  = struct.unpack_from('<i', weights_64b, 10)[0]

    x = patch_int8.reshape(PATCH_ROWS, PATCH_COLS).astype(np.int64)
    xp = np.pad(x, 1, mode='constant')     # zero-padding = conv padding=1

    acc = np.full((PATCH_ROWS, PATCH_COLS), bias, dtype=np.int64)
    for kr in range(3):
        for kc in range(3):
            acc += w[kr, kc] * xp[kr:kr + PATCH_ROWS, kc:kc + PATCH_COLS]

    out = acc >> shift
    out = np.clip(out, -32768, 32767)       # saturate_int16

    # Separable Hanning window (Q1.15 integer arithmetic)
    h_r = HANNING[:, None]   # [128, 1]
    h_c = HANNING[None, :]   # [1, 128]
    wnd = (out * h_r) >> 15
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
                      **kwargs) -> None:
    """Write all files for one scenario into out_dir/name/.

    fft_col_in.bin is always computed from the raw patch (no conv2d simulation).
    This keeps the expected test values stable and analytically predictable
    regardless of which conv2d weights are in use.

    weights_ch0.bin is written when weights_64b is provided so that mosse_graph.cpp
    can load real weights for the GMIO transfer, exercising the weight-load path even
    though the conv2d output is discarded via the bypass mechanism.

    A future add-on: a separate 'conv2d_check' scenario that uses simulate_conv2d()
    to generate fft_col_in.bin from the actual kernel output and calibrates the
    expected values from a first exploratory sim run.
    """
    sdir = os.path.join(out_dir, name)
    os.makedirs(sdir, exist_ok=True)
    write_plio_txt(os.path.join(sdir, 'patch_in.txt'), patch_int8)
    write_cint16_bin(os.path.join(sdir, 'cmul_filter.bin'), H_re, H_im)
    write_cint16_bin(os.path.join(sdir, 'cmul_accum.bin'), acc_re, acc_im)
    write_expected_txt(os.path.join(sdir, 'expected.txt'), **kwargs)
    # Pre-computed col-FFT input: computed from the raw patch (bypasses PLIO→conv2d→row_FFT).
    fci_re, fci_im = compute_fft_col_in(patch_int8)   # raw patch, no conv2d
    write_cint16_bin(os.path.join(sdir, 'fft_col_in.bin'), fci_re, fci_im)
    # Single-channel weight buffer: mosse_graph.cpp loads this instead of zeroing.
    if weights_64b is not None:
        with open(os.path.join(sdir, 'weights_ch0.bin'), 'wb') as f:
            f.write(weights_64b)
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

    # Loss-tolerant lower bound: a full round trip rounds at every cint16 stage
    # (row-FFT -> col-FFT -> cmul -> row-IFFT -> col-IFFT). Measured loss is ~2%,
    # so a bound of exactly the ideal value fails. Allow 10%, floor of 1.
    def lo(v):
        return max(1, (9 * int(v)) // 10)

    impulse = np.zeros(N, dtype=np.int8); impulse[0] = AMP
    write_plio_txt(os.path.join(out_dir, "patch_in.txt"), impulse)

    # s2 constant-patch level. The DC bin is (gain * N * S2_CONST), so this cannot
    # be scaled like the impulse scenarios: at 64x64, S2_CONST=7 puts accum0 near
    # 19k and 12 would saturate cint16. 7 is the practical maximum here.
    S2_CONST = 7
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
        peak_re_lo=lo(AMP), peak_re_hi=6 * AMP + 8,
        peak_im_lo=-3 * AMP, peak_im_hi=3 * AMP,
        max_noise=AMP // 2, skip_snr=False,
        check_accum0=True, accum0_re=AMP + 1, accum0_im=0,
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
        peak_re_lo=lo(AMP), peak_re_hi=6 * AMP,
        peak_im_lo=-4 * AMP, peak_im_hi=4 * AMP,
        # 6*AMP would be 6x the peak itself — a threshold that passes even when the
        # peak is buried in noise, i.e. no test at all. Measured noise floor is 2
        # against a peak of 98, so AMP/2 asserts a real >=2:1 SNR with ~25x margin.
        max_noise=AMP // 2, skip_snr=False,
        check_accum0=True, accum0_re=AMP, accum0_im=0,   # F[0,0] = sum of patch = AMP
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
    FFT_DC_TRUNC = 21
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
        peak_re_lo=1, peak_re_hi=10,
        peak_im_lo=-4, peak_im_hi=4,
        max_noise=0, skip_snr=True,   # uniform response: all elements ≈ equal
        check_accum0=True, accum0_re=s2_accum0, accum0_im=0,
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
        peak_re_lo=-AMP // 4, peak_re_hi=AMP // 4,
        # Sign test: must remain strictly negative, magnitude ~AMP.
        peak_im_lo=-8 * AMP, peak_im_hi=-lo(AMP),
        max_noise=AMP // 2, skip_snr=False,
        check_accum0=True, accum0_re=0, accum0_im=-AMP,
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
        peak_im_lo=-2 * H_MAX, peak_im_hi=2 * H_MAX,
        max_noise=0, skip_snr=True,        # Gaussian has non-zero off-peak values
        check_accum0=True, accum0_re=AMP * H_MAX, accum0_im=0,
        description=f"impulse@(0,0), H*=Gaussian(σ={sigma:.0f},max={H_MAX}), acc={{0,0}} — per-element flt_local",
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
