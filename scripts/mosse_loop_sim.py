#!/usr/bin/env python3
"""Closed-loop MOSSE simulation — the hardware-free regression test for the
training-target defect.

WHAT THIS TESTS, AND WHY IT IS NOT gen_filter_golden.py: that script checks ONE
filter update against a NumPy golden, which the C++ already passes. The defect
here is invisible to a single update — it is a feedback loop. The filter is
trained against a target Gaussian centred at (0,0) while the patch it is trained
on has the object at the measured displacement (dr,dc), so every accepted frame
teaches "object at (dr,dc) => peak at (0,0)". Under roughly constant motion the
error is coherent frame to frame and compounds at the learning rate until the
zero-shift peak wins and the tracker jumps. Only a CLOSED LOOP over many frames
can see that.

The mechanism, stated exactly, because the sign matters and is easy to get
backwards:

  Let Q_t be the patch cropped exactly ON the target and P_t = crop at the
  PREVIOUS position, so P_t is Q_t shifted by +d_t. Correlation is shift
  equivariant, so resp(P_t) = resp(Q_t) shifted by +d_t. We want an on-target
  patch to peak at 0, i.e. resp(Q_t) peaked at 0, hence resp(P_t) must peak at
  d_t -- so the training target must be centred at +d_t, the SAME sign as the
  detected peak.

  Training at 0 instead teaches resp(Q_t) -> peak at -d_t. On the next frame the
  patch is Q_{t+1} shifted by d_{t+1}, so the learned contribution peaks at
  d_{t+1} - d_t, which for near-constant motion is (0,0). That is precisely the
  growing origin peak, and it explains why resp00_over_peak follows
  1-(1-eta)^k -- it is governed by the learning rate, not the scene.

Arms:
  centred  the shipped behaviour: G built once at (0,0)
  shifted  the fix: G rebuilt per frame at the measured (dr,dc)

Usage:
  python3 scripts/mosse_loop_sim.py                 # both arms, 40 frames
  python3 scripts/mosse_loop_sim.py --frames 100
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gen_filter_golden import gaussian_target_spectrum, filter_update  # noqa: E402

# Geometry mirrors the hardware build: 128x128 patch, sigma 2, eta 0.125.
ROWS = COLS = 128
SIGMA = 2.0
ETA = 0.125
EPS_REL = 1e-3
N_CH = 4          # a small feature bank; the defect is per-channel identical
FRAME_R, FRAME_C = 512, 512


def make_background(rng):
    """Band-limited texture, whole cycles per frame -- mirrors fill_background()."""
    rr = np.arange(FRAME_R).reshape(-1, 1) / FRAME_R
    cc = np.arange(FRAME_C).reshape(1, -1) / FRAME_C
    bg = np.zeros((FRAME_R, FRAME_C))
    for fr, fc in [(1, 2), (2, 1), (3, 5), (5, 3), (4, 6), (6, 4)]:
        bg += np.cos(2 * np.pi * (fr * rr + fc * cc) + rng.uniform(0, 2 * np.pi))
    return 128.0 + 20.0 * bg / 6.0


def stamp_target(frame, row, col, size=64):
    """An asymmetric, structured target -- a centred blob hides sign errors."""
    r0, c0 = int(round(row - size / 2)), int(round(col - size / 2))
    rr = np.arange(size).reshape(-1, 1)
    cc = np.arange(size).reshape(1, -1)
    t = 60.0 * np.exp(-((rr - size * 0.35)**2 + (cc - size * 0.6)**2) / (2 * 12.0**2))
    t += 40.0 * ((rr > size * 0.6) & (cc < size * 0.4))
    frame[r0:r0 + size, c0:c0 + size] += t
    return frame


def hann2d(rows, cols):
    wr = np.sin(np.pi * np.arange(rows) / rows)**2      # PERIODIC, as on hardware
    wc = np.sin(np.pi * np.arange(cols) / cols)**2
    return np.outer(wr, wc)


W = hann2d(ROWS, COLS)

# Four cheap "conv layer 1" surrogates. The defect does not depend on what the
# features are, only on there being several of them accumulating coherently.
KERNELS = [
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float),
]


def features(patch):
    """3x3 conv bank -> zero mean -> Hann window -> FFT. Mirrors conv2d + B1."""
    out = np.empty((N_CH, ROWS, COLS), dtype=np.complex128)
    P = np.fft.fft2(patch)
    for k, ker in enumerate(KERNELS):
        kk = np.zeros((ROWS, COLS))
        kk[:3, :3] = ker
        kk = np.roll(np.roll(kk, -1, 0), -1, 1)
        f = np.real(np.fft.ifft2(P * np.fft.fft2(kk)))
        f = (f - f.mean()) * W
        out[k] = np.fft.fft2(f)
    return out


def crop(frame, row, col):
    r0, c0 = int(round(row - ROWS / 2)), int(round(col - COLS / 2))
    rr = np.clip(np.arange(r0, r0 + ROWS), 0, FRAME_R - 1)
    cc = np.clip(np.arange(c0, c0 + COLS), 0, FRAME_C - 1)
    return frame[np.ix_(rr, cc)].astype(np.float64)


def signed_bin(k, n):
    return k - n if k > n // 2 else k


def run(arm, n_frames, vel, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    bg = make_background(rng)

    truth_r, truth_c = FRAME_R / 2.0, FRAME_C / 2.0
    est_r, est_c = truth_r, truth_c

    A = np.zeros((N_CH, ROWS, COLS), dtype=np.complex128)
    B = np.zeros((ROWS, COLS), dtype=np.complex128)

    G0 = gaussian_target_spectrum(ROWS, COLS, SIGMA, 0, 0)
    rows = []

    for t in range(n_frames):
        frame = stamp_target(bg.copy(), truth_r, truth_c)
        F = features(crop(frame, est_r, est_c))

        if t == 0:
            # Bootstrap: the crop IS centred on the target, so a centred G is
            # correct here in BOTH arms. eta=1 against a zeroed state == init.
            A, B = filter_update(A, B, F, G0, 1.0)
            rows.append((t, 0, 0, 0.0, 0.0))
        else:
            eps = EPS_REL * float(np.mean(np.abs(B)))
            H = A / (B + eps)
            R = np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))
            idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            dr, dc = signed_bin(idx[0], ROWS), signed_bin(idx[1], COLS)
            peak = abs(R[idx])
            resp00 = abs(R[0, 0]) / peak if peak else 0.0

            est_r += dr
            est_c += dc
            cerr = np.hypot(est_r - truth_r, est_c - truth_c)
            rows.append((t, dr, dc, resp00, cerr))

            # THE DEFECT AND THE FIX, in one line.
            G = G0 if arm == 'centred' else \
                gaussian_target_spectrum(ROWS, COLS, SIGMA, dr, dc)
            A, B = filter_update(A, B, F, G, ETA)

        truth_r += vel[0]
        truth_c += vel[1]

    if verbose:
        print(f"  {'frame':>5} {'dr':>4} {'dc':>4} {'resp00/peak':>12} {'centre err':>11}")
        for t, dr, dc, r00, cerr in rows:
            flag = '  <-- LOST' if cerr > 8 else ('  <-- resp00 high' if r00 > 0.3 else '')
            print(f"  {t:5d} {dr:4d} {dc:4d} {r00:12.3f} {cerr:11.2f}{flag}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--vel', type=float, nargs=2, default=[3.0, -2.0],
                    help='target velocity in frame px/frame')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    summary = {}
    for arm in ('centred', 'shifted'):
        print(f"\n=== arm: {arm}"
              f"{'  (shipped -- G always at (0,0))' if arm == 'centred' else '  (fix -- G at the measured (dr,dc))'}")
        rows = run(arm, args.frames, tuple(args.vel), verbose=not args.quiet)
        post = rows[1:]
        summary[arm] = (
            max(r[3] for r in post),
            post[-1][4],
            sum(1 for r in post if r[4] <= 8) / len(post),
        )

    print("\n=== summary")
    print(f"  {'arm':>8} {'max resp00/peak':>16} {'final centre err':>18} {'frames on target':>18}")
    for arm, (r00, cerr, frac) in summary.items():
        print(f"  {arm:>8} {r00:16.3f} {cerr:18.2f} {100 * frac:17.1f}%")

    ok = (summary['shifted'][0] < 0.3 and summary['shifted'][1] < 4.0
          and summary['centred'][0] > summary['shifted'][0])
    print("\n  VERDICT:", "PASS — the shifted target holds lock and the centred one does not"
          if ok else "FAIL — check the arms, this script is the regression test")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
