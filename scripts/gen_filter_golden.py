#!/usr/bin/env python3
"""
gen_filter_golden.py
NumPy reference for design/host_app_src/mosse_filter.{h,cpp}.

Usage:
    python3 gen_filter_golden.py <output_dir>

The filter init/update is the one part of the host application that can be tested
without hardware, and it is the part where a sign or conjugation error is silent:
a wrong filter still produces a plausible-looking response map with a peak
somewhere. `make test_host` runs the C++ against this reference in seconds, which
is the only practical alternative to a ~90 min hw_emu frame.

Deliberately SMALL (32x32, 4 channels): the geometry is irrelevant to the maths
and a small case keeps the test instant. The C++ is geometry-agnostic.

Writes into <output_dir>/:
    params.txt   — geometry and constants, `key value` per line
    F_all.bin    — input spectra, float32 (re,im) interleaved [ch][row][col]
    energy.bin   — per-channel Stage B3 energies, float64
    G.bin        — target spectrum, float32 (re,im)
    A_init.bin   — expected A after filter_init,   float32 (re,im)
    B_init.bin   — expected B after filter_init,   float32
    A_upd.bin    — expected A after filter_update, float32 (re,im)
    B_upd.bin    — expected B after filter_update, float32
    H_q15.bin    — expected quantized filter, int16 (re,im)
"""

import os
import sys
import numpy as np

# Must match CMUL_H_SHIFT. The Makefile passes H_SHIFT to both this script's
# consumer and the C++; keeping the default here in sync with mosse_filter.h is
# enough because test_host compiles with the same -D.
H_SHIFT = int(os.environ.get('GEN_H_SHIFT', 15))

ROWS, COLS, CHANNELS = 32, 32, 4
N = ROWS * COLS
SIGMA = 2.0
# Non-zero and asymmetric on purpose: a centred target has conj(G) == G, which
# makes the conjugation convention untestable. See mosse_filter.h.
DR, DC = 3, -5
ETA = 0.125
EPS_REL = 1e-3


def signed_freq(k, n):
    return np.where(k > n // 2, k - n, k)


def gaussian_target_spectrum(rows, cols, sigma, dr, dc):
    """Closed-form spectrum of a circularly-wrapped Gaussian — mirrors the C++.

    Note this is NOT computed via np.fft.fft2 of a spatial Gaussian: the point is
    to check the C++ against the same closed form it claims to implement. The
    closed form itself is validated separately by test_gaussian_matches_fft()
    below, which DOES compare against an actual FFT.
    """
    u = signed_freq(np.arange(rows), rows).reshape(-1, 1).astype(np.float64)
    v = signed_freq(np.arange(cols), cols).reshape(1, -1).astype(np.float64)
    s2 = sigma * sigma
    mag = np.exp(-2.0 * np.pi**2 * s2 * (u**2 / rows**2 + v**2 / cols**2))
    ph = -2.0 * np.pi * (u * dr / rows + v * dc / cols)
    return mag * np.exp(1j * ph)


def test_gaussian_matches_fft():
    """The closed form must agree with an actual FFT of a wrapped Gaussian."""
    rr = np.arange(ROWS).reshape(-1, 1)
    cc = np.arange(COLS).reshape(1, -1)
    dr = np.minimum((rr - DR) % ROWS, (DR - rr) % ROWS)
    dc = np.minimum((cc - DC) % COLS, (DC - cc) % COLS)
    g = np.exp(-(dr**2 + dc**2) / (2.0 * SIGMA**2))
    G_fft = np.fft.fft2(g)
    G_cf = gaussian_target_spectrum(ROWS, COLS, SIGMA, DR, DC)
    # The closed form drops the constant gain 2*pi*sigma^2, so compare shapes.
    G_cf_scaled = G_cf * (np.abs(G_fft).max() / np.abs(G_cf).max())
    err = np.max(np.abs(G_fft - G_cf_scaled)) / np.max(np.abs(G_fft))
    assert err < 1e-3, f"closed-form Gaussian spectrum deviates from FFT by {err:.2e}"
    print(f"  closed-form G vs FFT(wrapped Gaussian): max rel err {err:.2e}  OK")


def filter_update(A, B, F_all, G, eta):
    keep = 1.0 - eta
    B_new = eta * np.sum(np.abs(F_all)**2, axis=0) + keep * B
    # conj(G) * F, NOT G * conj(F) — cmul_accum conjugates the stored filter.
    A_new = eta * np.conj(G)[None, :, :] * F_all + keep * A
    return A_new, B_new


def quantize_q15(A, B, energy, eps_rel):
    eps = eps_rel * float(np.mean(B))
    chscale = np.where(energy > 0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
    H = A * chscale[:, None, None] / (B + eps)[None, :, :]
    max_abs = float(np.max(np.abs(H)))
    # Full int16 scale, independent of H_SHIFT. H always uses all 15 bits; H_SHIFT
    # only scales the F*H product inside cmul_accum. Deriving this ceiling from
    # H_SHIFT was a real bug in filter_quantize_q15() — it cost one bit of filter
    # resolution per bit of accumulator gain.
    q15_one = 32767.0
    scale = (q15_one / max_abs) if max_abs > 0 else 0.0
    Hq = H * scale
    # -32767, not -32768 — see the clamp comment in mosse_filter.cpp: excluding
    # the most-negative int16 keeps cmul_accum's int32 product from reaching 2^31.
    re = np.clip(np.round(Hq.real), -32767, 32767).astype(np.int16)
    im = np.clip(np.round(Hq.imag), -32767, 32767).astype(np.int16)
    out = np.empty(Hq.size * 2, dtype='<i2')
    out[0::2] = re.flatten()
    out[1::2] = im.flatten()
    return out, scale, max_abs


def write_c64(path, arr):
    """Write complex as float32 (re,im) interleaved."""
    buf = np.empty(arr.size * 2, dtype='<f4')
    buf[0::2] = arr.real.flatten()
    buf[1::2] = arr.imag.flatten()
    buf.tofile(path)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'golden'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating filter golden data in: {out_dir}/")
    test_gaussian_matches_fft()

    rng = np.random.default_rng(20260805)
    # Two independent frames' worth of spectra, so the update step is exercised
    # against something that actually differs from the init input.
    F0 = (rng.standard_normal((CHANNELS, ROWS, COLS))
          + 1j * rng.standard_normal((CHANNELS, ROWS, COLS))) * 100.0
    F1 = (rng.standard_normal((CHANNELS, ROWS, COLS))
          + 1j * rng.standard_normal((CHANNELS, ROWS, COLS))) * 100.0
    # float32 round trip: the C++ works in float32, so the reference must consume
    # exactly the same input values, not the float64 originals.
    F0 = F0.astype(np.complex64).astype(np.complex128)
    F1 = F1.astype(np.complex64).astype(np.complex128)

    energy = np.array([float(np.mean(np.abs(F0[ch])**2)) for ch in range(CHANNELS)])

    G = gaussian_target_spectrum(ROWS, COLS, SIGMA, DR, DC)
    G = G.astype(np.complex64).astype(np.complex128)

    A0 = np.zeros((CHANNELS, ROWS, COLS), dtype=np.complex128)
    B0 = np.zeros((ROWS, COLS), dtype=np.float64)

    A_init, B_init = filter_update(A0, B0, F0, G, 1.0)
    A_upd,  B_upd  = filter_update(A_init, B_init, F1, G, ETA)
    H_q15, scale, max_abs = quantize_q15(A_upd, B_upd, energy, EPS_REL)

    with open(os.path.join(out_dir, 'params.txt'), 'w') as f:
        f.write(f"rows      {ROWS}\n")
        f.write(f"cols      {COLS}\n")
        f.write(f"channels  {CHANNELS}\n")
        f.write(f"h_shift   {H_SHIFT}\n")
        f.write(f"sigma     {SIGMA}\n")
        f.write(f"dr        {DR}\n")
        f.write(f"dc        {DC}\n")
        f.write(f"eta       {ETA}\n")
        f.write(f"eps_rel   {EPS_REL}\n")

    write_c64(os.path.join(out_dir, 'F0.bin'), F0)
    write_c64(os.path.join(out_dir, 'F1.bin'), F1)
    write_c64(os.path.join(out_dir, 'G.bin'), G)
    energy.astype('<f8').tofile(os.path.join(out_dir, 'energy.bin'))
    write_c64(os.path.join(out_dir, 'A_init.bin'), A_init)
    B_init.astype('<f4').tofile(os.path.join(out_dir, 'B_init.bin'))
    write_c64(os.path.join(out_dir, 'A_upd.bin'), A_upd)
    B_upd.astype('<f4').tofile(os.path.join(out_dir, 'B_upd.bin'))
    H_q15.tofile(os.path.join(out_dir, 'H_q15.bin'))

    print(f"  geometry {ROWS}x{COLS} x {CHANNELS} ch, target offset ({DR},{DC})")
    print(f"  Q1.15 scale {scale:.6g}, max|H| {max_abs:.6g}")
    print(f"  wrote 9 files")


if __name__ == '__main__':
    main()
