#!/usr/bin/env python3
"""
scripts/rgb_vs_gray_vot.py

Multi-sequence VOT supervised evaluation: gray vs RGB vs a colour-free control.

Why this exists
---------------
rgb_vs_gray_loop.py ran one sequence and produced a +0.31 mean IoU for RGB that
decomposed into ONE avoided failure at car1 frame 462 — with gray AHEAD on the
461 frames before it. A single event cannot decide anything. The VOT supervised
protocol exists precisely to turn "it survived / it didn't" into a countable
statistic, so that is what this runs.

The protocol (VOT2015-style)
----------------------------
Initialise at frame 1 from ground truth. Track. When the overlap with ground
truth reaches ZERO, record a FAILURE, skip SKIP frames, and re-initialise from
ground truth. Report two numbers that deliberately do not combine:

  Accuracy   A = mean overlap while tracking, EXCLUDING the BURNIN frames after
                 each (re)initialisation — those are trivially perfect and
                 would reward a tracker for failing often.
  Robustness R = failures, reported per 100 frames so sequences of different
                 length are comparable.

A tracker cannot game both: never moving scores R=0 and a terrible A; chasing
every distractor scores a good A on the frames it survives and a ruinous R.

Arms, and the control
---------------------
  gray     BT.601 luminance collapse, 9-tap int8 kernels — what ships
  rgb      27-tap int8 kernels, jointly-normalised 3-plane patch
  rgb-lum  CONTROL: the 27-tap kernels fed three IDENTICAL luminance planes.
           Same taps, same bias, same quantization grid, NO colour. An rgb win
           that rgb-lum matches is bookkeeping, not colour.

Scale is not modelled (box size held from the last initialisation, i.e.
SCALE_N=1). --oracle-scale takes size from ground truth instead and isolates
localisation. Run both; they bracket what the DSST scale filter would give.

Everything else is the closed loop from rgb_vs_gray_loop.py: exact integer
Stage A / int8 / conv, float FFT, G centred at the MEASURED displacement,
Stage B1 on the previous frame's means, and a PSR gate that holds position and
skips the update together.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_vot.py
  ... --oracle-scale --sequences car1 tiger --match-shift
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('GEN_PATCH_ROWS', '128')
os.environ.setdefault('GEN_PATCH_COLS', '128')

import gen_filter_golden as FG                      # noqa: E402
from rgb_vs_gray_holdout import (                   # noqa: E402
    LUM, PADDING, SIGMA, EPS_REL, R, C,
    to_luma, stage_a_gray, stage_a_rgb,
    folded_weights, quantize, conv_features, metrics, wrap,
    sat_reset, sat_frac,
)
from rgb_vs_gray_loop import box_iou, ETA, PSR_GATE_MIN   # noqa: E402
from vot_prepare import reduce_box                  # noqa: E402

ROOT = Path("test-sequences")
SKIP = 5        # frames skipped after a failure before re-initialisation
BURNIN = 10     # frames after each (re)init excluded from the accuracy average


# ---------------------------------------------------------------------------
# sequence discovery — the annotation directories are named inconsistently
# ("car1-annotations" but "fernando - annotations"), so match loosely.
# ---------------------------------------------------------------------------

def discover():
    seqs = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or 'annotation' in d.name.lower():
            continue
        frames = sorted(d.glob("*.jpg"))
        if not frames:
            continue
        stem = d.name
        ann = None
        for cand in ROOT.iterdir():
            if not cand.is_dir() or 'annotation' not in cand.name.lower():
                continue
            if re.sub(r'[^a-z0-9]', '', cand.name.lower()).startswith(
                    re.sub(r'[^a-z0-9]', '', stem.lower())):
                ann = cand
                break
        gtf = ann / "groundtruth.txt" if ann else None
        if gtf and gtf.exists():
            seqs[stem] = (frames, gtf)
        else:
            print(f"  [skip] {stem}: no groundtruth.txt found")
    return seqs


def load_gt(path):
    """One groundtruth file -> axis-aligned (row, col, h, w) per frame.

    THE REDUCTION IS SINGLE-SOURCED FROM vot_prepare.reduce_box, DELIBERATELY.

    This function used to carry its own polygon-only copy of the rule. VOT uses
    two groundtruth formats -- 4-value `x,y,w,h` rectangles (ALL of stb2022) and
    2n-value rotated polygons (the VOT2015-era sequences in test-sequences/) --
    and the polygon rule applied to a rectangle silently gives x=[x,w], y=[y,h]:
    `fernando` frame 291 (440,229,198,230) reduces to a 1.0 x 242.0 sliver
    instead of 230 x 198. Complete, plausible, and wrong in every box.

    On test-sequences/ the polygon-only rule is correct, so this file was right
    for every run it has ever done -- and would have produced garbage the first
    time it was pointed at stb2022. Phase 1 found it in vot_prepare; the copy
    here was the reason Phase 0c's "independent" cross-check agreed, since it
    was two implementations of ONE wrong rule.

    So there is now one implementation. `vot_prepare.verify --check-gt` compares
    the manifests against the toolkit's own parse_region(), a genuinely
    independent parser, which means this function inherits that check instead of
    needing its own. Importing the converter costs nothing: its module level is
    numpy and constants, and the dependency runs offline-harness -> converter,
    never the other way (vot_prepare imports nothing from the offline stack, on
    purpose).
    """
    return [reduce_box([float(t) for t in line.split(',')])
            for line in path.read_text().split()]


def load_frame(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"),
                      dtype=np.uint8).transpose(2, 0, 1)


def make_patch(arm, planes, lum, rr, rc, rh, rw):
    if arm == 'gray':
        return stage_a_gray(lum, rr, rc, rh, rw)[None]
    if arm == 'rgb':
        return stage_a_rgb(planes, rr, rc, rh, rw)
    return stage_a_rgb(np.stack([lum] * 3), rr, rc, rh, rw)


# ---------------------------------------------------------------------------

class Arm:
    """Per-arm tracker state for one sequence."""

    def __init__(self, name, weights):
        self.name = name
        self.wq, self.bias, self.shift = weights
        self.reset_all()

    def reset_all(self):
        self.A = self.B = None
        self.mean_prev = None
        self.box = None
        self.failures = 0
        self.overlaps = []       # post-burn-in only
        self.psr = []
        self.holds = 0
        self.skip_until = 0      # frames < this are not processed
        self.since_init = 0
        self.sat = []            # per-frame int16 saturation fraction

    def init_at(self, gt_box):
        self.A = self.B = None
        self.mean_prev = None
        self.box = list(gt_box)
        self.since_init = 0


def run_sequence(frames, gt, arms, oracle_scale, progress=None):
    n = len(frames)
    for a in arms:
        a.reset_all()
        a.init_at(gt[0])

    for f in range(1, n + 1):
        if all(f < a.skip_until for a in arms):
            continue
        planes = load_frame(frames[f - 1])
        lum = to_luma(planes)

        for a in arms:
            if f < a.skip_until:
                continue
            if f == a.skip_until:            # re-initialise from ground truth
                a.init_at(gt[f - 1])
                a.skip_until = 0

            if oracle_scale:
                a.box[2], a.box[3] = gt[f - 1][2], gt[f - 1][3]
            rh = max(4, int(round(a.box[2] * PADDING)))
            rw = max(4, int(round(a.box[3] * PADDING)))
            rr = int(round(a.box[0] - rh / 2.0))
            rc = int(round(a.box[1] - rw / 2.0))

            patch = make_patch(a.name, planes, lum, rr, rc, rh, rw)
            sat_reset()
            feats, own_mean = conv_features(patch, a.wq, a.bias, a.shift,
                                            mean_prev=a.mean_prev)
            a.sat.append(sat_frac())
            a.mean_prev = own_mean
            F = np.fft.fft2(feats.astype(np.float64), axes=(1, 2))

            if a.A is None:
                Gt = FG.gaussian_target_spectrum(R, C, SIGMA, 0, 0)
                a.A = np.conj(Gt)[None] * F
                a.B = np.sum(np.abs(F)**2, axis=0)
            else:
                energy = np.sum(np.abs(F)**2, axis=(1, 2))
                chs = np.where(energy > 0,
                               1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
                H = a.A * chs[:, None, None] / (a.B + EPS_REL * a.B.mean())[None]
                resp = np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))
                idx, peak, bolme, _ = metrics(resp)
                dr, dc = wrap(idx[0], R), wrap(idx[1], C)
                a.psr.append(bolme)
                if peak > 0 and bolme >= PSR_GATE_MIN:
                    a.box[0] += dr * rh / R
                    a.box[1] += dc * rw / C
                    Gt = FG.gaussian_target_spectrum(R, C, SIGMA, dr, dc)
                    a.A, a.B = FG.filter_update(a.A, a.B, F, Gt, ETA)
                else:
                    a.holds += 1

            ov = box_iou(tuple(a.box), gt[f - 1])
            a.since_init += 1

            if ov <= 0.0:
                # VOT failure: count it, skip SKIP frames, reinit from truth.
                a.failures += 1
                a.skip_until = min(f + SKIP, n + 1)
                a.A = None
            elif a.since_init > BURNIN:
                a.overlaps.append(ov)

        if progress and f % 250 == 0:
            print(f"      f{f}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequences', nargs='+', default=None)
    ap.add_argument('--arms', nargs='+', default=['gray', 'rgb', 'rgb-lum'])
    ap.add_argument('--oracle-scale', action='store_true')
    ap.add_argument('--match-shift', action='store_true',
                    help="force the RGB arms' out_shift to the gray arm's, testing "
                         "whether RGB's steady-state deficit is the extra shift "
                         "the tripled ACC_MAX_THEORY forces")
    ap.add_argument('--progress', action='store_true')
    args = ap.parse_args()

    w_rgb, b_fold = folded_weights()
    w_gray = (w_rgb * LUM[None, :, None, None]).sum(axis=1, keepdims=True)
    Wg, Wr = quantize(w_gray, b_fold), quantize(w_rgb, b_fold)
    if args.match_shift:
        # Same shift as gray. Overflow is possible in principle; the clip in
        # conv_features saturates rather than wrapping, and the count is
        # reported so a silent saturation cannot masquerade as a result.
        Wr = (Wr[0], Wr[1], Wg[2].copy())
    W = {'gray': Wg, 'rgb': Wr, 'rgb-lum': Wr}

    seqs = discover()
    if args.sequences:
        seqs = {k: v for k, v in seqs.items() if k in args.sequences}
    if not seqs:
        sys.exit("no sequences found under test-sequences/")

    print(f"VOT supervised: skip {SKIP}, burn-in {BURNIN}, "
          f"gate {PSR_GATE_MIN}, eta {ETA}, padding {PADDING}")
    print(f"scale: {'ORACLE (ground truth)' if args.oracle_scale else 'HELD FIXED (SCALE_N=1)'}"
          f"   shifts: {'MATCHED to gray' if args.match_shift else 'natural'}")
    print(f"sequences: {', '.join(f'{k}({len(v[0])})' for k, v in seqs.items())}")
    print()

    arms = [Arm(a, W[a]) for a in args.arms]
    tot = {a.name: {'ov': [], 'fail': 0, 'frames': 0, 'psr': [], 'holds': 0,
                    'sat': []}
           for a in arms}

    hdr = f"{'sequence':<12} {'n':>5}  " + "  ".join(
        f"{a.name:>9} {'A':>6} {'R':>5}" for a in arms)
    print(f"{'sequence':<12} {'n':>5}  " +
          "  ".join(f"{a.name:^22}" for a in arms))
    print(f"{'':<12} {'':>5}  " +
          "  ".join(f"{'A':>7}{'fails':>8}{'R/100f':>8}" for a in arms))
    print("-" * len(hdr) * 1)

    for name, (frames, gtf) in seqs.items():
        gt = load_gt(gtf)
        n = min(len(frames), len(gt))
        run_sequence(frames[:n], gt[:n], arms, args.oracle_scale, args.progress)
        row = f"{name:<12} {n:>5}  "
        for a in arms:
            A = float(np.mean(a.overlaps)) if a.overlaps else 0.0
            row += f"{A:>7.4f}{a.failures:>8d}{100.0*a.failures/n:>8.2f}  "
            t = tot[a.name]
            t['ov'] += a.overlaps
            t['fail'] += a.failures
            t['frames'] += n
            t['psr'] += a.psr
            t['holds'] += a.holds
            t['sat'] += a.sat
        print(row, flush=True)

    print("-" * len(hdr))
    row = f"{'OVERALL':<12} {sum(t['frames'] for t in tot.values())//len(tot):>5}  "
    for a in arms:
        t = tot[a.name]
        row += (f"{np.mean(t['ov']):>7.4f}{t['fail']:>8d}"
                f"{100.0*t['fail']/t['frames']:>8.2f}  ")
    print(row)
    print()
    for a in arms:
        t = tot[a.name]
        sat = np.array(t['sat'])
        print(f"  {a.name:<9} mean PSR {np.mean(t['psr']):7.2f}   gate holds "
              f"{t['holds']:>5}   accuracy frames {len(t['ov']):>5}   "
              f"int16 saturation mean {100*sat.mean():.4f}%  max {100*sat.max():.3f}%")

    if all(k in tot for k in ('gray', 'rgb', 'rgb-lum')):
        print()
        print("=" * 72)
        g, r, l = (tot[k] for k in ('gray', 'rgb', 'rgb-lum'))
        ga, ra, la = (np.mean(x['ov']) for x in (g, r, l))
        print(f"Accuracy    gray {ga:.4f}   rgb {ra:.4f} ({ra-ga:+.4f})   "
              f"rgb-lum {la:.4f} ({la-ga:+.4f})")
        print(f"Failures    gray {g['fail']:>4}   rgb {r['fail']:>4} "
              f"({r['fail']-g['fail']:+d})   rgb-lum {l['fail']:>4} "
              f"({l['fail']-g['fail']:+d})")
        print()
        print("The control has RGB's taps, bias and quantization grid and NO colour.")
        print("Read the rgb-vs-rgb-lum gap, not the rgb-vs-gray one.")


if __name__ == '__main__':
    main()
