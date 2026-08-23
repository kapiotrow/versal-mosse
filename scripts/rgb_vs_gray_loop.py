#!/usr/bin/env python3
"""
scripts/rgb_vs_gray_loop.py

Closed-loop MOSSE on real video, gray vs RGB vs a colour-free control.

rgb_vs_gray_holdout.py measures RESPONSE QUALITY with the filter frozen. It
found RGB worth 1.63x on Bolme PSR and nothing measurable on localisation, and
it could not tell whether that was RGB's fault or the protocol's: a filter
trained on one frame and evaluated ten frames later is mostly just failing.

This closes the loop. The filter updates every frame at MOSSE_ETA, the position
is the tracker's own, and errors compound — which is the only regime in which
"does RGB track better" is a meaningful question. Same three arms, same exact
integer Stage A / int8 / conv datapath, same float FFT (see the holdout's
docstring for why the FFT is float).

Faithful to the hardware in the ways that matter
------------------------------------------------
  * THE FILTER TRAINS AGAINST G CENTRED AT THE MEASURED DISPLACEMENT, not at
    (0,0). This is the defect fixed 2026-08-20 and it is invisible to any
    single-update test; a closed loop is the only thing that sees it. See the
    training-target entry in CLAUDE.md.
  * Stage B1 subtracts the PREVIOUS frame's per-channel mean.
  * The PSR gate holds position AND skips the filter update, both.
  * Position updates by peak * roi/patch — the resample ratio, not 1:1.

Deliberately NOT modelled: the DSST scale filter. The box size is held at its
initial value (equivalent to SCALE_N=1). That caps IoU on any sequence whose
target changes size — car1's width goes 122 -> 83 px — but it caps all three
arms identically, and adding a second estimator would confound the comparison.
--oracle-scale takes the box size from ground truth instead, which isolates
localisation from scale; run both and read the pair.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py
  ... --oracle-scale --frames 400
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('GEN_PATCH_ROWS', '128')
os.environ.setdefault('GEN_PATCH_COLS', '128')

import gen_filter_golden as FG                      # noqa: E402
from rgb_vs_gray_holdout import (                   # noqa: E402
    LUM, N_OUT, PADDING, SIGMA, EPS_REL, R, C,
    load_gt, load_frame_rgb, to_luma,
    stage_a_gray, stage_a_rgb, folded_weights, quantize,
    conv_features, metrics, wrap,
)

ETA = 0.125            # MOSSE_ETA
PSR_GATE_MIN = 7.0     # Bolme 3.5


def box_iou(a, b):
    """a, b = (row, col, h, w) centre-form. Axis-aligned overlap."""
    ar0, ac0, ar1, ac1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    br0, bc0, br1, bc1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ih = max(0.0, min(ar1, br1) - max(ar0, br0))
    iw = max(0.0, min(ac1, bc1) - max(ac0, bc0))
    inter = ih * iw
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def make_patch(arm, planes, lum, roi_row, roi_col, roi_h, roi_w):
    if arm == 'gray':
        return stage_a_gray(lum, roi_row, roi_col, roi_h, roi_w)[None]
    if arm == 'rgb':
        return stage_a_rgb(planes, roi_row, roi_col, roi_h, roi_w)
    return stage_a_rgb(np.stack([lum] * 3), roi_row, roi_col, roi_h, roi_w)


def run_arm(arm, wq, bias, shift, gt, n_frames, oracle_scale, verbose):
    """One full pass over the sequence. Returns a per-frame record."""
    row, col, bh, bw = gt[0]
    A = B = None
    mean_prev = None
    rec = {'iou': [], 'cerr': [], 'psr': [], 'holds': 0, 'lost_at': None}

    for f in range(1, n_frames + 1):
        if oracle_scale:
            bh, bw = gt[f - 1][2], gt[f - 1][3]
        roi_h, roi_w = int(round(bh * PADDING)), int(round(bw * PADDING))
        roi_row = int(round(row - roi_h / 2.0))
        roi_col = int(round(col - roi_w / 2.0))

        planes = load_frame_rgb(f)
        lum = to_luma(planes)
        patch = make_patch(arm, planes, lum, roi_row, roi_col, roi_h, roi_w)
        feats, own_mean = conv_features(patch, wq, bias, shift, mean_prev=mean_prev)
        mean_prev = own_mean

        F = np.fft.fft2(feats.astype(np.float64), axes=(1, 2))

        if A is None:
            # filter_init. Frame 1's crop really is centred on the target, so G
            # is centred here and only here.
            Gt = FG.gaussian_target_spectrum(R, C, SIGMA, 0, 0)
            A = np.conj(Gt)[None] * F
            B = np.sum(np.abs(F)**2, axis=0)
            rec['iou'].append(box_iou((row, col, bh, bw), gt[f - 1]))
            rec['cerr'].append(float(np.hypot(row - gt[f-1][0], col - gt[f-1][1])))
            rec['psr'].append(float('nan'))
            continue

        # detect
        energy = np.sum(np.abs(F)**2, axis=(1, 2))
        chscale = np.where(energy > 0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
        H = A * chscale[:, None, None] / (B + EPS_REL * B.mean())[None]
        resp = np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))
        idx, peak, bolme, _ratio = metrics(resp)
        dr, dc = wrap(idx[0], R), wrap(idx[1], C)

        gate_ok = (peak > 0) and (bolme >= PSR_GATE_MIN)
        if gate_ok:
            # Patch bins -> frame pixels by the resample ratio.
            row += dr * roi_h / R
            col += dc * roi_w / C
            # THE TRAINING TARGET IS SHIFTED BY THE MEASURED DISPLACEMENT.
            # g_F_all was cropped at the PRE-update position, where the object
            # sits at (dr,dc); training against a centred G teaches "target at
            # (dr,dc) peaks at 0" and compounds at ETA until zero-shift wins.
            Gt = FG.gaussian_target_spectrum(R, C, SIGMA, dr, dc)
            A, B = FG.filter_update(A, B, F, Gt, ETA)
        else:
            rec['holds'] += 1        # hold position AND skip the update, both

        iou = box_iou((row, col, bh, bw), gt[f - 1])
        rec['iou'].append(iou)
        rec['cerr'].append(float(np.hypot(row - gt[f-1][0], col - gt[f-1][1])))
        rec['psr'].append(bolme)
        if verbose and f % 50 == 0:
            print(f"    {arm:<8} f{f:<4} IoU {iou:.3f}  PSR {bolme:6.2f}  "
                  f"cerr {rec['cerr'][-1]:6.1f}px", flush=True)

    # First frame after which IoU never recovers above 0.5 — "permanently lost".
    iou = np.array(rec['iou'])
    below = iou < 0.5
    if below.any():
        k = len(iou)
        while k > 0 and below[k - 1]:
            k -= 1
        if k < len(iou):
            rec['lost_at'] = k + 1
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=0, help='0 = whole sequence')
    ap.add_argument('--arms', nargs='+', default=['gray', 'rgb', 'rgb-lum'])
    ap.add_argument('--oracle-scale', action='store_true',
                    help='take box size from ground truth (isolates localisation)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    gt = load_gt()
    n = len(gt) if args.frames == 0 else min(args.frames, len(gt))
    w_rgb, b_fold = folded_weights()
    w_gray = (w_rgb * LUM[None, :, None, None]).sum(axis=1, keepdims=True)
    W = {'gray': quantize(w_gray, b_fold),
         'rgb': quantize(w_rgb, b_fold),
         'rgb-lum': quantize(w_rgb, b_fold)}

    print(f"car1, {n} frames, closed loop: eta {ETA}, padding {PADDING}, "
          f"sigma {SIGMA}, PSR gate {PSR_GATE_MIN}")
    print(f"scale: {'ORACLE (from ground truth)' if args.oracle_scale else 'HELD FIXED (SCALE_N=1 equivalent)'}")
    print()

    out = {}
    for a in args.arms:
        print(f"  running {a} ...", flush=True)
        out[a] = run_arm(a, *W[a], gt, n, args.oracle_scale, args.verbose)

    print()
    print(f"{'arm':<9} {'mean IoU':>9} {'worst':>7} {'>=0.5':>7} "
          f"{'cerr mean':>10} {'cerr max':>9} {'PSR mean':>9} {'holds':>6} {'lost at':>8}")
    print("-" * 82)
    for a in args.arms:
        r = out[a]
        iou = np.array(r['iou']); ce = np.array(r['cerr'])
        psr = np.array(r['psr'])[1:]
        print(f"{a:<9} {iou.mean():9.4f} {iou.min():7.4f} "
              f"{100*np.mean(iou >= 0.5):6.1f}% "
              f"{ce.mean():10.2f} {ce.max():9.2f} {np.nanmean(psr):9.2f} "
              f"{r['holds']:6d} "
              f"{(str(r['lost_at']) if r['lost_at'] else 'never'):>8}")

    if all(a in out for a in ('gray', 'rgb', 'rgb-lum')):
        print()
        print("=" * 82)
        g, r, l = (np.array(out[a]['iou']) for a in ('gray', 'rgb', 'rgb-lum'))
        print(f"mean IoU   gray {g.mean():.4f}   rgb {r.mean():.4f} "
              f"({r.mean()-g.mean():+.4f})   rgb-lum {l.mean():.4f} "
              f"({l.mean()-g.mean():+.4f})")
        print(f"per-frame  rgb better than gray on {100*np.mean(r>g+1e-9):.1f}% of frames, "
              f"worse on {100*np.mean(r<g-1e-9):.1f}%")
        print(f"           rgb better than rgb-lum on {100*np.mean(r>l+1e-9):.1f}%, "
              f"worse on {100*np.mean(r<l-1e-9):.1f}%")
        print()
        print("The control carries RGB's taps, bias and quantization grid and NO")
        print("colour. An rgb win that rgb-lum matches is not a colour result.")


if __name__ == '__main__':
    main()
