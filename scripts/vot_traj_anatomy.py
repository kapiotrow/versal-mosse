#!/usr/bin/env python3
"""
scripts/vot_traj_anatomy.py -- what a board trajectory did, frame by frame,
against the groundtruth, in units of the tracker's own PATCH BIN.

WHY THIS EXISTS
---------------
`nature`'s failure was diagnosed from console lines as sub-bin lag: the target
moves less than one patch bin per frame, the integer argmax reports (0,0), the
filter is trained on that, and the error compounds. Half of that is observable
from the console and half is inference, and the inference was wrong.

The discriminator is one number this script prints: **how often the tracker
reports NO MOTION on a frame whose true motion is bigger than a bin.** Sub-bin
lag cannot do that -- by construction it only bites when the motion is under one
bin. A detector pinned to the origin does it constantly.

Everything is expressed in bins because that is the tracker's actual resolution:
one bin = roi/patch = box * TARGET_PADDING / 128 frame pixels, per axis, and the
two axes differ whenever the box is not square.

Usage
-----
  ./.venv/bin/python scripts/vot_traj_anatomy.py nature \
      --traj /srv/vot/results/coast0/nature_00000000.txt [--frames 25]
"""
import argparse
import os
from collections import Counter

import numpy as np


def load_boxes(path, centre=True):
    out = []
    for line in open(path):
        s = line.strip()
        if not s or s == '1':          # Special(INITIALIZATION)
            out.append(None); continue
        v = [float(x) for x in s.split(',')]
        out.append((v[1] + v[3] / 2, v[0] + v[2] / 2, v[3], v[2]) if centre else tuple(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequence')
    ap.add_argument('--traj', required=True)
    ap.add_argument('--frames', type=int, default=25, help='rows of the per-frame table')
    ap.add_argument('--padding', type=float, default=2.0)
    ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--seqdir', default=None)
    args = ap.parse_args()

    root = os.environ.get('VOT_ROOT', os.path.expanduser('~/vot'))
    seqdir = args.seqdir or f"{root}/workspace/sequences/{args.sequence}"
    gt = load_boxes(f"{seqdir}/groundtruth.txt")
    tr = load_boxes(args.traj)
    tr[0] = gt[0]                       # the init line carries no box
    n = min(len(gt), len(tr))
    if n < 2:
        raise SystemExit("trajectory too short")

    # One bin, from the tracker's OWN box each frame -- the scale filter moves it.
    binpx = lambda k, ax: args.padding * tr[k][2 + ax] / args.patch

    print(f"{args.sequence}: {n} frames, init box "
          f"{gt[0][2]:.0f} x {gt[0][3]:.0f}  ->  bin = "
          f"{binpx(0,0):.2f} x {binpx(0,1):.2f} frame px "
          f"(padding {args.padding}, patch {args.patch})")

    print(f"\n{'f':>4}{'truth d(r,c)':>18}{'track d(r,c)':>18}"
          f"{'track in BINS':>16}{'lag px':>8}")
    for k in range(1, min(args.frames, n)):
        td = (gt[k][0] - gt[k-1][0], gt[k][1] - gt[k-1][1])
        kd = (tr[k][0] - tr[k-1][0], tr[k][1] - tr[k-1][1])
        lag = np.hypot(tr[k][0] - gt[k][0], tr[k][1] - gt[k][1])
        print(f"{k:4d}  ({td[0]:+7.2f},{td[1]:+7.2f})  ({kd[0]:+7.2f},{kd[1]:+7.2f})"
              f"  ({kd[0]/binpx(k,0):+5.1f},{kd[1]/binpx(k,1):+5.1f}){lag:8.1f}")

    # --- the discriminator --------------------------------------------------
    frozen_big = frozen_small = moved = 0
    for k in range(1, n):
        rb, cb = binpx(k, 0), binpx(k, 1)
        tdr, tdc = (gt[k][0]-gt[k-1][0]) / rb, (gt[k][1]-gt[k-1][1]) / cb
        kdr, kdc = (tr[k][0]-tr[k-1][0]) / rb, (tr[k][1]-tr[k-1][1]) / cb
        still = abs(kdr) < 0.5 and abs(kdc) < 0.5
        big = max(abs(tdr), abs(tdc)) >= 1.0
        if still and big:   frozen_big += 1
        elif still:         frozen_small += 1
        else:               moved += 1

    print(f"\n  reported no motion while the target moved >= 1 bin : "
          f"{frozen_big:5d} / {n-1} = {100*frozen_big/(n-1):.1f}%   <- NOT sub-bin")
    print(f"  reported no motion while the target moved  < 1 bin : "
          f"{frozen_small:5d} / {n-1} = {100*frozen_small/(n-1):.1f}%   <- sub-bin")
    print(f"  reported motion                                    : "
          f"{moved:5d} / {n-1} = {100*moved/(n-1):.1f}%")

    # --- mean signed displacement, per axis ---------------------------------
    # Mean SPEED and mean DISPLACEMENT are different statistics and conflating
    # them is how "the tracker captures 83% of the motion" was arrived at: a
    # tracker that moves smoothly through a jittering groundtruth has a lower
    # mean speed while tracking the mean displacement exactly.
    for ax, name in ((0, 'row'), (1, 'col')):
        td = [gt[k][ax] - gt[k-1][ax] for k in range(1, n)]
        kd = [tr[k][ax] - tr[k-1][ax] for k in range(1, n)]
        print(f"\n  {name}: mean displacement  truth {np.mean(td):+.3f}  "
              f"track {np.mean(kd):+.3f} px/frame")
        print(f"       mean SPEED         truth {np.mean(np.abs(td)):.3f}  "
              f"track {np.mean(np.abs(kd)):.3f} px/frame")
        h = Counter(int(round(kd[k-1] / binpx(k, ax))) for k in range(1, n))
        print(f"       reported step histogram (bins): {dict(sorted(h.items()))}")


if __name__ == '__main__':
    main()
