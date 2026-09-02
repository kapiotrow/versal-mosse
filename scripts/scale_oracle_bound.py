#!/usr/bin/env python3
"""What is a PERFECT scale filter worth? An oracle bound from board trajectories.

WHY THIS EXISTS. 2026-09-02 produced a long, well-attributed chain about the
scale filter -- frozen on ~90% of frames, detector gain -0.003 against the
position detector's 0.93, root-caused to a self-confirming loop fed by a
scale-NORMALISING feature. Every step of it was measured. NONE of it asked the
prior question: even repaired perfectly, how much R is there to win?

The answer is +0.0023, and it retired the whole direction. This script is that
measurement, kept so the retirement is reproducible rather than remembered.

METHOD. Take the board's own trajectories, keep the tracker's CENTRE, replace
the box SIZE with ground truth, recompute IoU, re-apply VOT's failure rule
(threshold 0.1, grace 10, burn-in 10). Everything except size is left exactly as
the hardware produced it.

  scripts/scale_oracle_bound.py <run_dir> [run_dir ...]

CAVEAT, PRICED: this is OPEN-LOOP. A genuinely better-sized box would also change
the ROI content and therefore later frames, so it is not a strict ceiling. For
that to matter the second-order feedback would have to exceed the first-order
effect by an order of magnitude, which nothing measured supports.
"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np

def iou(r0, c0, h0, w0, r1, c1, h1, w1):
    t0, l0, b0, e0 = r0 - h0/2, c0 - w0/2, r0 + h0/2, c0 + w0/2
    t1, l1, b1, e1 = r1 - h1/2, c1 - w1/2, r1 + h1/2, c1 + w1/2
    ih = max(0.0, min(b0, b1) - max(t0, t1))
    iw = max(0.0, min(e0, e1) - max(l0, l1))
    inter = ih * iw
    u = h0*w0 + h1*w1 - inter
    return inter / u if u > 0 else 0.0

def robustness(series):
    """VOT's rule: progress is the index of the first failure, R = sum/sum."""
    prog = tot = 0
    for s in series:
        bad, cut = 0, len(s)
        for i, v in enumerate(s):
            if i < 10:
                continue
            bad = bad + 1 if v <= 0.1 else 0
            if bad >= 10:
                cut = i - 9
                break
        prog += cut
        tot += len(s)
    return (prog / tot if tot else float('nan')), prog

def main(dirs):
    for d in dirs:
        real, orc = [], []
        for f in sorted(glob.glob(os.path.join(d, 'track_*.csv'))):
            by = defaultdict(list)
            for r in csv.DictReader(open(f)):
                by[r.get('job', '0')].append(r)
            for _, rows in by.items():
                s1, s2 = [], []
                for r in rows:
                    try:
                        er, ec = float(r['est_row']), float(r['est_col'])
                        tr, tc = float(r['truth_row']), float(r['truth_col'])
                        th, tw = float(r['truth_h']), float(r['truth_w'])
                        v = float(r['iou'])
                    except (ValueError, KeyError):
                        continue
                    s1.append(v)
                    # tracker's CENTRE, ground truth's SIZE
                    s2.append(iou(er, ec, th, tw, tr, tc, th, tw))
                if s1:
                    real.append(s1); orc.append(s2)
        Rr, pr = robustness(real)
        Ro, po = robustness(orc)
        fr = [v for s in real for v in s]
        fo = [v for s in orc for v in s]
        print(f"\n{d}: {len(real)} runs")
        print(f"  R  as tracked   {Rr:.4f}  ({pr} frames survived)")
        print(f"  R  ORACLE SIZE  {Ro:.4f}  ({po} frames survived)   dR = {Ro-Rr:+.4f}")
        print(f"  mean IoU        {np.mean(fr):.4f} -> {np.mean(fo):.4f}"
              f"   d = {np.mean(fo)-np.mean(fr):+.4f}")
        print("  -> a perfect scale filter converts a large BOX-QUALITY gain into"
              " almost no SURVIVAL gain.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
