#!/usr/bin/env python3
"""
scripts/vot_motion_check.py -- is a sequence's annotated motion actually IMAGE
motion? No tracker, no filter, no board.

WHY THIS EXISTS
---------------
`nature` scored 0.1535 mean IoU and was diagnosed twice from tracker-side
evidence: first as sub-bin lag, then as origin lock. Both diagnoses assumed the
premise that the target moves and the tracker fails to follow. This script tests
that premise, and on `nature` it is false.

THE TEST, and why it has no dominance failure mode
--------------------------------------------------
Take frame f-1's groundtruth box content as a reference. Compare it, by
normalised cross-correlation, against frame f at two places:

    (a) where the ANNOTATION says the target moved to
    (b) where it was, unmoved

If the annotation's motion is real image translation, (a) must win. When (b)
wins, staying still explains the pixels better than following the box -- the
content is deforming, rotating or changing in place while the box centre moves
as a side effect of a min-max reduction over a changing shape.

Phase correlation was tried first and is the WRONG instrument here: it returns
the DOMINANT motion in a window, so it reads zero whenever static background
fills the box, and it read 0.00 px on `car1` -- a car crossing the frame at 20
px/frame. This comparison asks about the target's own pixels at two specific
hypotheses and cannot be fooled that way.

Usage
-----
  ./.venv/bin/python scripts/vot_motion_check.py nature tiger car1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgb_vs_gray_holdout as HO                     # noqa: E402
from rgb_vs_gray_holdout import (                    # noqa: E402
    load_gt, load_frame_rgb, to_luma)


def crop(lum, r, c, h, w):
    r0, c0 = int(round(r - h / 2)), int(round(c - w / 2))
    rr = np.clip(np.arange(r0, r0 + int(h)), 0, lum.shape[0] - 1)
    cc = np.clip(np.arange(c0, c0 + int(w)), 0, lum.shape[1] - 1)
    return lum[np.ix_(rr, cc)].astype(np.float64)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = a.std() * b.std()
    return float((a * b).mean() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequences', nargs='+')
    ap.add_argument('--frames', type=int, default=120)
    args = ap.parse_args()

    print(f"{'sequence':<10}{'|gt step| px':>14}{'NCC moved':>11}{'NCC still':>11}"
          f"{'still wins':>12}{'appearance f1 vs f50':>22}")
    for name in args.sequences:
        HO.set_sequence(name)
        gt = load_gt()
        n = min(args.frames, len(gt))
        A, Z, steps, wins = [], [], [], 0
        for f in range(2, n):
            lum0, lum1 = to_luma(load_frame_rgb(f - 1)), to_luma(load_frame_rgb(f))
            h, w = gt[f-2][2], gt[f-2][3]
            ref   = crop(lum0, gt[f-2][0], gt[f-2][1], h, w)
            moved = crop(lum1, gt[f-1][0], gt[f-1][1], h, w)
            still = crop(lum1, gt[f-2][0], gt[f-2][1], h, w)
            if ref.size == 0 or moved.size == 0 or still.size == 0:
                continue          # box off-frame: no pixels to compare
            a, z = ncc(ref, moved), ncc(ref, still)
            A.append(a); Z.append(z); wins += (z > a)
            steps.append(float(np.hypot(gt[f-1][0]-gt[f-2][0], gt[f-1][1]-gt[f-2][1])))

        # How fast the target stops looking like itself, for context: a filter at
        # eta=0.125 has a ~8-frame memory, so a target that decorrelates in 2 is
        # a different problem from one that merely moves.
        a0 = crop(to_luma(load_frame_rgb(1)), gt[0][0], gt[0][1], gt[0][2], gt[0][3])
        k = min(50, len(gt))
        bk = crop(to_luma(load_frame_rgb(k)), gt[k-1][0], gt[k-1][1], gt[0][2], gt[0][3])
        print(f"{name:<10}{np.mean(steps):14.2f}{np.mean(A):11.3f}{np.mean(Z):11.3f}"
              f"{100*wins/len(A):11.0f}%{ncc(a0, bk):22.3f}")

    print("\n  'still wins' high  -> the annotation moves but the PIXELS do not:")
    print("                        a translation tracker is being asked the wrong question.")
    print("  'still wins' low   -> the motion is real; a frozen detector is a real defect.")


if __name__ == '__main__':
    main()
