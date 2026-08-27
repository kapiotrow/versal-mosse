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
    load_gt, load_frame_rgb, to_luma, set_sequence,
    stage_a_gray, stage_a_rgb, folded_weights, quantize, conv_features_float,
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
    if arm in ('gray', 'gray-float'):
        return stage_a_gray(lum, roi_row, roi_col, roi_h, roi_w)[None]
    if arm in ('rgb', 'rgb-float'):
        return stage_a_rgb(planes, roi_row, roi_col, roi_h, roi_w)
    return stage_a_rgb(np.stack([lum] * 3), roi_row, roi_col, roi_h, roi_w)


def run_arm(arm, wq, bias, shift, gt, n_frames, oracle_scale, verbose,
            detect_iters=1, detect_gain=1.0, float_conv=None):
    # float_conv = (w_float, b_fold) runs the UNQUANTIZED conv instead of the
    # int8 one, everything else identical. See conv_features_float's docstring:
    # this arm exists to answer whether quantization causes the tracker's poor
    # robustness, and it is the second half of that question -- the first half
    # (the cint16/Q1.15 correlation pipeline) is already answered by this whole
    # model being float64 downstream of the features.
    """One full pass over the sequence. Returns a per-frame record."""
    row, col, bh, bw = gt[0]
    A = B = None
    mean_prev = None
    # `step` records the measurement itself, per frame, in BINS and in frame px,
    # because the failure under investigation is a detector that reports (0,0)
    # while the target moves several bins -- invisible in IoU or PSR, which both
    # look healthy while it happens.
    rec = {'iou': [], 'cerr': [], 'psr': [], 'holds': 0, 'lost_at': None,
           'step': [], 'resp00': []}

    for f in range(1, n_frames + 1):
        if oracle_scale:
            bh, bw = gt[f - 1][2], gt[f - 1][3]
        roi_h, roi_w = int(round(bh * PADDING)), int(round(bw * PADDING))
        roi_row = int(round(row - roi_h / 2.0))
        roi_col = int(round(col - roi_w / 2.0))

        planes = load_frame_rgb(f)
        lum = to_luma(planes)

        def crop_features(rr, rc, mp):
            patch = make_patch(arm, planes, lum, rr, rc, roi_h, roi_w)
            if float_conv is not None:
                ft, om = conv_features_float(patch, float_conv[0], float_conv[1],
                                             mean_prev=mp)
            else:
                ft, om = conv_features(patch, wq, bias, shift, mean_prev=mp)
            return np.fft.fft2(ft.astype(np.float64), axes=(1, 2)), om

        F, own_mean = crop_features(roi_row, roi_col, mean_prev)
        mean_prev = own_mean

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

        # ITERATED LOCALISATION. A windowed correlation systematically reports
        # LESS than the true displacement: the patch is Hann-weighted, so a
        # target that has moved far from the ROI centre is attenuated and the
        # peak is pulled back toward zero. On `tiger` the report is roughly half
        # of what was needed, every frame, which accumulates into a standing lag
        # rather than a loss. Re-cropping at the updated position and detecting
        # again attacks exactly that: each pass shrinks the residual, because the
        # second crop has the target much closer to the centre.
        #
        # detect_iters=1 is the shipped behaviour, evaluated identically.
        it_dr, it_dc = dr, dc
        for _ in range(detect_iters - 1):
            if not ((peak > 0) and (bolme >= PSR_GATE_MIN)) or (it_dr == 0 and it_dc == 0):
                break
            row += it_dr * roi_h / R
            col += it_dc * roi_w / C
            rr = int(round(row - roi_h / 2.0))
            rc = int(round(col - roi_w / 2.0))
            F, _ = crop_features(rr, rc, mean_prev)
            en2 = np.sum(np.abs(F)**2, axis=(1, 2))
            ch2 = np.where(en2 > 0, 1.0 / np.sqrt(np.maximum(en2, 1e-300)), 0.0)
            resp = np.real(np.fft.ifft2(np.sum(F * np.conj(A * ch2[:, None, None]
                                               / (B + EPS_REL * B.mean())[None]), axis=0)))
            idx, peak, bolme, _ratio = metrics(resp)
            it_dr, it_dc = wrap(idx[0], R), wrap(idx[1], C)
        dr, dc = it_dr, it_dc

        # resp00/peak: how much of the peak sits at ZERO SHIFT. The discriminator
        # for origin lock -- a value near 1 means the response is peaked at the
        # origin no matter where the target went.
        rec['resp00'].append(float(abs(resp[0, 0]) / abs(peak)) if peak else 0.0)
        # The displacement the detector REPORTED, and the one it NEEDED to report
        # to land on the target -- (groundtruth now) minus (where the tracker was
        # when it cropped). Comparing the report against the TARGET'S motion
        # instead is a different and much weaker question: it is only the same
        # number while the tracker is exactly on target, and it made a healthy
        # car1 look like a 20%-correct detector.
        need_r = (gt[f-1][0] - row) / (roi_h / R)
        need_c = (gt[f-1][1] - col) / (roi_w / C)
        rec['step'].append((dr, dc, roi_h / R, roi_w / C, need_r, need_c))

        gate_ok = (peak > 0) and (bolme >= PSR_GATE_MIN)
        if gate_ok:
            # Patch bins -> frame pixels by the resample ratio.
            row += dr * detect_gain * roi_h / R
            col += dc * detect_gain * roi_w / C
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
    ap.add_argument('--sequence', default=None,
                    help='sequence name; stb2022 under $VOT_ROOT is preferred, '
                         'then test-sequences/. Default: the built-in car1.')
    args = ap.parse_args()

    navail = set_sequence(args.sequence)
    gt = load_gt()
    n = min(navail, len(gt)) if args.frames == 0 else min(args.frames, navail, len(gt))
    w_rgb, b_fold = folded_weights()
    w_gray = (w_rgb * LUM[None, :, None, None]).sum(axis=1, keepdims=True)
    W = {'gray': quantize(w_gray, b_fold),
         'rgb': quantize(w_rgb, b_fold),
         'rgb-lum': quantize(w_rgb, b_fold),
         'gray-float': quantize(w_gray, b_fold),   # unused, keeps the call shape
         'rgb-float': quantize(w_rgb, b_fold)}
    # The unquantized counterparts. Same folded BN weights, same bias, no int8
    # grid, no out_shift, no int16 clips, no integer Hann.
    FLOATW = {'gray-float': (w_gray, b_fold), 'rgb-float': (w_rgb, b_fold)}

    print(f"{args.sequence or 'car1'}, {n} frames, closed loop: eta {ETA}, padding {PADDING}, "
          f"sigma {SIGMA}, PSR gate {PSR_GATE_MIN}")
    print(f"scale: {'ORACLE (from ground truth)' if args.oracle_scale else 'HELD FIXED (SCALE_N=1 equivalent)'}")
    print()

    out = {}
    for a in args.arms:
        print(f"  running {a} ...", flush=True)
        out[a] = run_arm(a, *W[a], gt, n, args.oracle_scale, args.verbose,
                         float_conv=FLOATW.get(a))

    print()
    print(f"{'arm':<9} {'frozen, truth>=1bin':>20} {'frozen, <1bin':>14} "
          f"{'moved':>7} {'resp00/peak':>12}")
    print("-" * 82)
    for a in args.arms:
        st = out[a]['step']
        big = small = moved = 0
        # need_r/need_c COME FROM THE RECORD, they are not recomputed here.
        # This used to unpack 4 fields from a 6-field record (a ValueError since
        # need_* were added) and recompute the truth motion as a groundtruth
        # DIFFERENCE -- which is the weaker question run_arm's own comment warns
        # about: the target's motion only equals the required displacement while
        # the tracker is exactly on target, and reading it that way "made a
        # healthy car1 look like a 20%-correct detector".
        for k, (dr, dc, _br, _bc, tdr, tdc) in enumerate(st, start=2):
            if k >= len(gt):
                break
            if dr == 0 and dc == 0:
                if max(abs(tdr), abs(tdc)) >= 1.0:
                    big += 1
                else:
                    small += 1
            else:
                moved += 1
        tot = max(1, big + small + moved)
        print(f"{a:<9} {100*big/tot:19.1f}% {100*small/tot:13.1f}% "
              f"{100*moved/tot:6.1f}% {np.mean(out[a]['resp00']):12.3f}")

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
