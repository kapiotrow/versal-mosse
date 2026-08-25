#!/usr/bin/env python3
"""
scripts/vot_hold_budget.py -- how many frames can the tracker HOLD before the
target has left the window it is holding?

WHY THIS EXISTS
---------------
On a gated frame the host holds position and freezes the filter (Bolme sec 3.5).
That is correct for occlusion and it assumes the target does not go anywhere
while the filter is frozen. `car1` on hardware (runs/run_0825_1314.log) held for
53 consecutive frames and never recovered, so the assumption is worth measuring
rather than arguing about -- and it can be measured from GROUNDTRUTH ALONE, with
no tracker, no board and no simulation. Nothing here depends on how well the
tracker works; it is an upper bound that a PERFECT tracker also obeys.

THE BOUND
---------
While the position is held at frame i, `roi_crop` keeps reading the same window:
roi = box_i * TARGET_PADDING, centred on the frozen centre. A correlation filter
can only report a peak for something inside that window, so once the target's
centre leaves it, no amount of later evidence brings the tracker back -- the
response is computed over pixels the target is not in. At the shipping padding
of 2.0 the window half-extent is exactly box/2, i.e. the target may drift half
its own size before it is unrecoverable.

So the HOLD BUDGET at frame i is the largest k such that the target centre at
frame i+k is still inside the window frozen at frame i. It is measured in frames
and it is what a hold policy actually spends.

This says nothing about whether the tracker SHOULD have gated -- only what a
hold costs once it does.

Usage:
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_hold_budget.py
  ... --data $VOT_ROOT/data --sequences car1,bag,agility --padding 2.0
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path


def hold_budget(gt, i, padding, direction=+1, cap=200,
                coast=False, decay=1.0, coast_max=0):
    """Frames the target stays inside the window held from frame i.

    policy `freeze` (coast=False): the window stays where it was, which is what
    the tracker does today.

    policy `coast`: the window moves at the velocity the tracker would have
    KNOWN -- gt[i] - gt[i-1], i.e. the displacement it measured on the last
    accepted frame, not a velocity read from the future. That distinction is the
    whole validity of this comparison: a model that steers the window with the
    motion it is trying to predict measures nothing.

    `decay` shrinks the velocity on each successive held frame, so the policy
    interpolates between "assume constant velocity" (1.0) and "assume stopped"
    (0.0). With decay < 1 the window's total drift is bounded by v/(1-decay),
    which is what keeps a coast from running away when the target has in fact
    stopped -- the occlusion case the freeze policy exists for.
    `coast_max` (0 = unlimited) stops coasting after that many held frames.
    """
    r0, c0, h0, w0 = gt[i]
    if not (h0 > 0 and w0 > 0):
        return None
    half_r = h0 * padding / 2.0 - h0 / 2.0     # = h0/2 at padding 2.0
    half_c = w0 * padding / 2.0 - w0 / 2.0

    vr = vc = 0.0
    if coast:
        prev = i - direction
        if 0 <= prev < len(gt) and gt[prev][2] > 0:
            vr = (r0 - gt[prev][0]) * direction * direction   # per-frame, run order
            vc = (c0 - gt[prev][1]) * direction * direction
        # A backward run reads frames in reverse, so the velocity the tracker
        # measures is also reversed; direction*direction == 1 keeps the sign in
        # RUN order, which is the order the window moves in.

    wr, wc = r0, c0
    k = 0
    j = i
    step = 0
    while k < cap:
        j += direction
        step += 1
        if j < 0 or j >= len(gt):
            return None                        # ran out of sequence, not a budget
        if coast and (coast_max <= 0 or step <= coast_max):
            wr += vr
            wc += vc
            vr *= decay
            vc *= decay
        r, c, h, w = gt[j]
        if not (h > 0 and w > 0):
            k += 1
            continue                           # empty gt: no evidence either way
        if abs(r - wr) > half_r or abs(c - wc) > half_c:
            return k
        k += 1
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--sequences', default=None, help="comma list; default all")
    ap.add_argument('--padding', type=float, default=2.0)
    ap.add_argument('--cap', type=int, default=200)
    ap.add_argument('--policy', default='freeze', choices=('freeze', 'coast', 'both'))
    ap.add_argument('--coast-decay', type=float, default=1.0)
    ap.add_argument('--coast-max', type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data or (os.environ.get('VOT_ROOT') or '') ) / 'data' \
        if not args.data else Path(args.data)
    if not data.is_dir():
        raise SystemExit(f"no such directory: {data} (set VOT_ROOT or pass --data)")

    names = (args.sequences.split(',') if args.sequences
             else sorted(p.stem for p in data.glob('*.json')))

    print(f"hold budget at padding {args.padding} -- frames before the target "
          f"leaves the frozen window\n")
    print(f"{'sequence':<14} {'frames':>6} {'motion med':>10} {'p95':>6} "
          f"{'budget med':>10} {'p10':>5} {'>=20':>6} {'1-frame escapes':>16}")
    rows = []
    for name in names:
        f = data / f"{name}.json"
        if not f.exists():
            print(f"{name:<14} (no manifest)")
            continue
        m = json.loads(f.read_text())
        gt = m['groundtruth']
        mot = [math.hypot(gt[i][0] - gt[i-1][0], gt[i][1] - gt[i-1][1])
               for i in range(1, len(gt))
               if gt[i][2] > 0 and gt[i-1][2] > 0]
        def budgets(coast):
            return [b for b in (hold_budget(gt, i, args.padding, +1, args.cap,
                                            coast, args.coast_decay, args.coast_max)
                                for i in range(len(gt) - 1)) if b is not None]
        buds = budgets(args.policy == 'coast')
        buds_alt = budgets(True) if args.policy == 'both' else None
        if not buds or not mot:
            print(f"{name:<14} (degenerate)")
            continue
        mot.sort(); buds.sort()
        med_m = statistics.median(mot)
        p95_m = mot[int(0.95 * (len(mot) - 1))]
        med_b = statistics.median(buds)
        p10_b = buds[int(0.10 * (len(buds) - 1))]
        long_b = 100.0 * sum(1 for b in buds if b >= 20) / len(buds)
        esc = 100.0 * sum(1 for b in buds if b == 0) / len(buds)
        rows.append((name, len(gt), med_m, p95_m, med_b, p10_b, long_b, esc))
        line = (f"{name:<14} {len(gt):>6} {med_m:>10.1f} {p95_m:>6.1f} "
                f"{med_b:>10.0f} {p10_b:>5.0f} {long_b:>5.1f}% {esc:>15.1f}%")
        if buds_alt:
            buds_alt.sort()
            med_c = statistics.median(buds_alt)
            esc_c = 100.0 * sum(1 for b in buds_alt if b == 0) / len(buds_alt)
            line += f"   | coast {med_c:>4.0f} {esc_c:>6.1f}%"
            rows[-1] = rows[-1] + (med_c, esc_c)
        print(line)

    if len(rows) > 1:
        print()
        allb = [r[4] for r in rows]
        print(f"across {len(rows)} sequences: median hold budget "
              f"{statistics.median(allb):.0f} frames, "
              f"min {min(allb):.0f} ({rows[allb.index(min(allb))][0]}), "
              f"max {max(allb):.0f} ({rows[allb.index(max(allb))][0]})")
        print(f"sequences whose MEDIAN budget is under 10 frames: "
              f"{sum(1 for b in allb if b < 10)} of {len(rows)}")


if __name__ == '__main__':
    main()
