#!/usr/bin/env python3
"""Choose BG_PAN_R / BG_PAN_C without hardware.

WHY THIS EXISTS. Background lock is the failure where the tracker correlates with
a static background at exactly zero shift and wins there instead of at the true
displacement (see "background lock" in CLAUDE.md). BG_PAN fixes it by moving the
background under the tracking window, the way a real camera does. But the pan
MAGNITUDE is not obvious: it has to be large against the texture's WAVELENGTHS,
not against a pixel. fill_background() uses six sinusoids of 1-6 cycles per frame,
so the shortest period is 180 rows, and the natural first guess of 3-5 px/frame
changed the zero-shift correlation from 0.60 to 0.61 — i.e. did nothing at all.

The metric is the normalised zero-shift correlation between the preprocessed ROI
patch at frame k and at frame k+1, after Stage A (log / zero-mean / unit-L2), the
periodic Hann window and Stage B2's 9-bin null. That is the quantity that becomes
resp00_over_peak on hardware. Lower |corr| is better; the sign does not matter.

Re-run this after ANY change to fill_background(), FRAME_TEXTURE or the ROI size.
The right pan is a property of the texture's spectrum, not a universal number.

    python3 scripts/bg_pan_sweep.py
    python3 scripts/bg_pan_sweep.py --roi 128 --pans 0,0 31,47 63,97
"""
import argparse
import numpy as np

ROWS, COLS = 1080, 1920          # must match FRAME_ROWS / FRAME_COLS
SEED       = 20260816            # must match fill_background()'s LCG seed


def build_background(rows, cols, integer_freq=True):
    """Replicate fill_background() from mosse_tracker.cpp."""
    s = SEED

    def nxt():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return (s >> 8) / (1 << 24)

    comp = []
    for _ in range(6):
        ky, kx = 1.0 + 5.0 * nxt(), 1.0 + 5.0 * nxt()
        if integer_freq:                      # whole cycles => seamless wrap
            ky, kx = round(ky), round(kx)
        comp.append((ky / rows, kx / cols, 2 * np.pi * nxt(), 0.4 + 0.6 * nxt()))

    amp_sum = sum(c[3] for c in comp)
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    f = np.zeros((rows, cols))
    for fy, fx, ph, amp in comp:
        f += amp * np.sin(2 * np.pi * (fy * r + fx * c) + ph)
    f /= amp_sum
    # The C++ dither consumes the LCG in raster order; only its distribution
    # matters here, so a separate generator is fine.
    rng = np.random.default_rng(0)
    dither = 3.0 * (rng.random((rows, cols)) - 0.5)
    return np.clip(110.0 + 90.0 * 0.35 * f + dither, 0.0, 255.0)


def preprocess(bg, r0, c0, n, pan_r, pan_c, window):
    """Stage A + periodic Hann over one ROI window, with wraparound."""
    rr = (np.arange(r0, r0 + n) + pan_r) % bg.shape[0]
    cc = (np.arange(c0, c0 + n) + pan_c) % bg.shape[1]
    p = np.log(bg[np.ix_(rr, cc)] + 1.0)
    p = p - p.mean()
    norm = np.linalg.norm(p)
    return (p / norm if norm else p) * window


def null_low_bins(spec):
    """Stage B2: the periodic Hann's 2-D DFT has exactly 9 non-zero bins."""
    out = spec.copy()
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            out[a, b] = 0.0
    return out


def zero_shift_corr(bg, pan_r, pan_c, n, window, frames=8, step_r=9, step_c=-3):
    """Mean |corr| at zero shift between consecutive frames' ROI patches."""
    vals = []
    for k in range(frames):
        r0, c0 = 476 + step_r * k, 896 + step_c * k
        a = null_low_bins(np.fft.fft2(
            preprocess(bg, r0, c0, n, pan_r * k, pan_c * k, window)))
        b = null_low_bins(np.fft.fft2(
            preprocess(bg, r0 + step_r, c0 + step_c, n,
                       pan_r * (k + 1), pan_c * (k + 1), window)))
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        vals.append(np.real(np.sum(a * np.conj(b))) / den)
    return float(np.mean(vals))


def wrap_seam(bg):
    """Mean |row 0 - row N-1| against the interior row-to-row gradient, LSB."""
    seam = float(np.mean(np.abs(bg[0] - bg[-1])))
    interior = float(np.mean(np.abs(bg[bg.shape[0] // 2] - bg[bg.shape[0] // 2 - 1])))
    return seam, interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, default=128, help="ROI size, frame px")
    ap.add_argument("--pans", nargs="*", default=None,
                    help='pan pairs, e.g. "31,47"')
    ap.add_argument("--non-integer-freq", action="store_true",
                    help="use the pre-2026-08-20 continuous frequencies")
    args = ap.parse_args()

    pans = ([tuple(int(x) for x in p.split(",")) for p in args.pans] if args.pans
            else [(0, 0), (3, 5), (7, 11), (15, 23), (23, 37),
                  (31, 47), (47, 71), (63, 97)])

    bg = build_background(ROWS, COLS, integer_freq=not args.non_integer_freq)
    hann = np.sin(np.pi * np.arange(args.roi) / args.roi) ** 2
    window = np.outer(hann, hann)

    seam, interior = wrap_seam(bg)
    print(f"row-wrap seam {seam:.2f} LSB vs interior row-to-row {interior:.2f} LSB"
          f"  ({'SEAMLESS' if seam < 2.0 * interior else 'VISIBLE EDGE — the ROI '
             'will straddle it and the tracker can lock onto it'})")
    print(f"\nROI {args.roi}x{args.roi}, zero-shift correlation after Stage A + Hann + B2")
    print("  pan r,c/frame    corr@0shift")
    for pr, pc in pans:
        print(f"    {pr:4d},{pc:4d}      {zero_shift_corr(bg, pr, pc, args.roi, window):+.4f}")
    print("\nLower |corr| is better; the sign is irrelevant. Pick the minimum, then")
    print("confirm on hardware with resp00_over_peak in track.csv (healthy < 0.3).")


if __name__ == "__main__":
    main()
