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
    """An asymmetric, structured target -- a centred blob hides sign errors.

    The target is rendered at its exact SUB-PIXEL position, evaluated on the
    frame's own grid rather than stamped into an integer box. That matters for
    the sub-bin experiment and only for it: with integer stamping a target moving
    0.35 px/frame does not move at all for three frames and then jumps one pixel,
    which is a different phenomenon from the one under test and would let a
    quantised detector look correct. The 1 px/frame arms are unaffected -- at
    integer offsets this evaluates to the same picture the old code stamped.
    """
    rr = np.arange(FRAME_R).reshape(-1, 1) - (row - size / 2.0)
    cc = np.arange(FRAME_C).reshape(1, -1) - (col - size / 2.0)
    inside = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    t = 60.0 * np.exp(-((rr - size * 0.35)**2 + (cc - size * 0.6)**2) / (2 * 12.0**2))
    t = t + 40.0 * ((rr > size * 0.6) & (cc < size * 0.4))
    frame += np.where(inside, t, 0.0)
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


def crop(frame, row, col, ratio=1.0):
    """ROI of ROWS*ratio frame pixels, bilinearly resampled to ROWS x COLS.

    `ratio` is roi_h / patch_rows -- the resample factor roi_crop applies on
    hardware, and therefore the size of ONE PATCH BIN in frame pixels. At 1.0
    this is the plain integer crop the earlier arms used, evaluated identically.

    This is the whole reason sub-bin motion exists: on `nature` the box is
    103x178, the ROI is twice that, and roi_crop squeezes it into 128x128, so one
    bin is 1.61 x 2.78 frame px while the target moves 2.06 px/frame.
    """
    if ratio == 1.0:
        r0, c0 = int(round(row - ROWS / 2)), int(round(col - COLS / 2))
        rr = np.clip(np.arange(r0, r0 + ROWS), 0, FRAME_R - 1)
        cc = np.clip(np.arange(c0, c0 + COLS), 0, FRAME_C - 1)
        return frame[np.ix_(rr, cc)].astype(np.float64)

    # Sample centres of the ROI, mapped onto the frame grid.
    rr = (np.arange(ROWS) + 0.5) * ratio + (row - ROWS * ratio / 2.0) - 0.5
    cc = (np.arange(COLS) + 0.5) * ratio + (col - COLS * ratio / 2.0) - 0.5
    r0 = np.clip(np.floor(rr).astype(int), 0, FRAME_R - 2)
    c0 = np.clip(np.floor(cc).astype(int), 0, FRAME_C - 2)
    fr = (rr - r0).reshape(-1, 1)
    fc = (cc - c0).reshape(1, -1)
    a = frame[np.ix_(r0,     c0)];      b = frame[np.ix_(r0,     c0 + 1)]
    c = frame[np.ix_(r0 + 1, c0)];      d = frame[np.ix_(r0 + 1, c0 + 1)]
    return ((a * (1 - fc) + b * fc) * (1 - fr) +
            (c * (1 - fc) + d * fc) * fr).astype(np.float64)


def parabolic(y_m, y_0, y_p):
    """3-point parabola vertex offset in [-0.5, 0.5], 0 when not a maximum.

    THE SAME EXPRESSION AS THE C++ (mosse_filter.cpp, refine_peak_axis). Two
    implementations of one rule is how this project has been bitten before, so
    the C++ unit test pins the identical numeric cases -- see
    run_subbin_tests() in test_mosse_filter.cpp.
    """
    den = y_m - 2.0 * y_0 + y_p
    if den >= 0.0:            # flat or a minimum: the fit says nothing
        return 0.0
    d = 0.5 * (y_m - y_p) / den
    return float(np.clip(d, -0.5, 0.5))


def signed_bin(k, n):
    return k - n if k > n // 2 else k


def run(arm, n_frames, vel, seed=0, verbose=True, ratio=1.0, subbin=False):
    """One closed-loop arm.

    arm     'centred' | 'shifted'   -- where the TRAINING target is centred
    ratio   frame px per patch bin  -- 1.0 is the historical 1:1 crop
    subbin  refine the integer argmax with a 3-point parabola before using it

    `subbin` deliberately affects BOTH uses of the measurement, because they are
    the same number: the position update AND the centre of the training G. A
    refinement applied only to the position would leave the filter still being
    taught that a drifted appearance is centred, which is the actual mechanism.
    """
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
        F = features(crop(frame, est_r, est_c, ratio))

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

            fdr, fdc = float(dr), float(dc)
            if subbin:
                # Neighbours WRAP: the response map is circular, exactly as the
                # PSR exclusion window is. Fit on sign(peak)*R so a legitimately
                # negative peak is still a maximum of the fitted parabola.
                sg = 1.0 if R[idx] >= 0 else -1.0
                r_, c_ = idx
                fdr += parabolic(sg * R[(r_ - 1) % ROWS, c_], sg * R[r_, c_],
                                 sg * R[(r_ + 1) % ROWS, c_])
                fdc += parabolic(sg * R[r_, (c_ - 1) % COLS], sg * R[r_, c_],
                                 sg * R[r_, (c_ + 1) % COLS])

            # Patch bins -> frame pixels. At ratio 1.0 this is the old `+= dr`.
            est_r += fdr * ratio
            est_c += fdc * ratio
            cerr = np.hypot(est_r - truth_r, est_c - truth_c)
            rows.append((t, dr, dc, resp00, cerr))

            # THE DEFECT AND THE FIX, in one line. The training target is centred
            # at the SAME measurement the position moved by -- refined or not.
            G = G0 if arm == 'centred' else \
                gaussian_target_spectrum(ROWS, COLS, SIGMA, fdr, fdc)
            A, B = filter_update(A, B, F, G, ETA)

        truth_r += vel[0]
        truth_c += vel[1]

    if verbose:
        print(f"  {'frame':>5} {'dr':>4} {'dc':>4} {'resp00/peak':>12} {'centre err':>11}")
        for t, dr, dc, r00, cerr in rows:
            flag = '  <-- LOST' if cerr > 8 else ('  <-- resp00 high' if r00 > 0.3 else '')
            print(f"  {t:5d} {dr:4d} {dc:4d} {r00:12.3f} {cerr:11.2f}{flag}")
    return rows


def subbin_experiment(frames, quiet):
    """Does sub-bin motion compound into unbounded lag? MEASURED: no.

    The hypothesis (docs/thesis/evidence/subbin_lag.md) was that a target moving less than
    one patch bin per frame is reported as (0,0), the filter is then trained
    against a G centred at that (0,0), and the lag compounds forever. The first
    half is true and the second is not, and the reason is structural: the
    detector measures the offset that EXISTS RIGHT NOW, not the increment. Lag
    accumulates only until it exceeds half a bin, at which point the next
    measurement is a whole bin and takes it back. The loop is self-correcting and
    the error is bounded by half a bin, whatever the speed or the resample ratio.

    This bench is known to be CAPABLE of showing compounding drift: the 'centred'
    arm above is exactly that, and it ends tens of pixels off. So a flat error
    here is a result, not an insensitive instrument.
    """
    print("\n=== sub-bin motion: does quantised measurement compound?")
    print("  The claim under test is GROWTH, not magnitude: a compounding lag makes")
    print("  the second half of a run much worse than the first. A bounded error does")
    print("  not, however often the detector reports (0,0).\n")
    print(f"  {'ratio':>6}{'vel px/f':>10}{'bins/f':>8}{'(0,0)%':>8}"
          f"{'argmax 1st/2nd half':>22}{'parabolic':>11}")
    worst_growth = 0.0
    for ratio in (1.0, 2.0, 3.0):
        for v in (0.3, 0.7, 1.5):
            res = {}
            for sub in (False, True):
                rows = run('shifted', frames, (v * 0.8, v * 0.6), verbose=False,
                           ratio=ratio, subbin=sub)
                post = rows[1:]
                half = len(post) // 2
                early = max(r[4] for r in post[:half])
                late = max(r[4] for r in post[half:])
                res[sub] = (early, late)
                if not sub:
                    z = 100 * sum(1 for r in post if r[1] == 0 and r[2] == 0) / len(post)
            e, l = res[False]
            growth = l / e if e > 0 else (0.0 if l == 0 else 99.0)
            worst_growth = max(worst_growth, growth)
            print(f"  {ratio:6.1f}{v:10.2f}{v / ratio:8.2f}{z:8.0f}"
                  f"{e:11.2f} /{l:8.2f}{res[True][1]:11.2f}")

    ok = worst_growth < 1.5
    print(f"\n  worst late/early error ratio: {worst_growth:.2f}")
    print("  VERDICT:", "sub-bin quantisation does NOT compound -- the loop measures the\n"
          "           offset that exists now, so lag is corrected the moment it exceeds\n"
          "           half a bin. Error is bounded, not accumulating."
          if ok else
          "the error GREW -- the compounding hypothesis survives this bench")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--vel', type=float, nargs=2, default=[3.0, -2.0],
                    help='target velocity in frame px/frame')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--subbin', action='store_true',
                    help='run the sub-bin experiment instead of the training-target one')
    args = ap.parse_args()

    if args.subbin:
        return subbin_experiment(max(args.frames, 200), args.quiet)

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
