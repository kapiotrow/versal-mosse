#!/usr/bin/env python3
"""Paired per-sequence statistics for a merged grid, against one control cell.

WHY THIS EXISTS. `vot_ar_offline.py` prints POOLED A/R and a per-sequence dR
listing, and nothing else -- the trim / sign / bootstrap columns quoted all over
docs/thesis/evidence/ are not in it. Pooled R is exactly the statistic this
project has repeatedly been misled by: the spatial mask was +0.0330 pooled and
-0.0101 after dropping three sequences, and `vot_ar_offline`'s own resolution is
MEASURED at ~0.02 in R. So a pooled maximum is a reason to look, never a result.

  scripts/grid_stats.py <merged.json> <control_arm> [arm ...]

Columns, all computed on the 62 PAIRED per-sequence R values (paired, so the
sequence-to-sequence variance that dominates the pooled means cancels):

  dR mean   mean of per-sequence (arm - control). NOT the pooled dR: pooled
            weights by sequence length, this weights sequences equally.
  trim3/5   dR mean after DROPPING the 3 (5) sequences most FAVOURABLE to the
            arm. One-sided on purpose -- the question is whether a handful of
            sequences carry the result, which is how the mask arm failed.
  b/w/t     sequences better / worse / tied on R.
  sign p    two-sided binomial on better-vs-worse, ties excluded.
  P(dR<=0)  fraction of 10k bootstrap resamples (over sequences, with
            replacement) whose mean dR is <= 0. `dec2`, the one arm that
            transferred to hardware, scored 0.000 here.

Reuses vot_ar_offline.score so the R per sequence is identical to that tool's.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vot_ar_offline import score

def seq_R(d, seq, arm):
    return score([d[f"{seq}|{arm}"]['iou']])[1]

def binom_two_sided(k, n):
    if n == 0:
        return float('nan')
    from math import comb
    obs = comb(n, k)
    tot = 2 ** n
    p = sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= obs) / tot
    return min(1.0, p)

def main(path, ctrl, arms):
    d = json.load(open(path))
    seqs = sorted({k.rsplit('|', 1)[0] for k in d})
    present = {k.rsplit('|', 1)[1] for k in d}
    if ctrl not in present:
        sys.exit(f"control {ctrl!r} not in file; have {sorted(present)}")
    arms = arms or sorted(a for a in present if a != ctrl)
    rc = np.array([seq_R(d, s, ctrl) for s in seqs])

    print(f"{len(seqs)} sequences, paired against control {ctrl}")
    print(f"{'arm':<12} {'poolR':>7} {'dRmean':>8} {'trim3':>8} {'trim5':>8} "
          f"{'b/w/t':>10} {'signp':>7} {'P(dR<=0)':>9}")
    print("-" * 78)
    rng = np.random.default_rng(20260902)
    for a in arms:
        ra = np.array([seq_R(d, s, a) for s in seqs])
        dr = ra - rc
        order = np.argsort(dr)                      # ascending; drop the top
        t3 = dr[order[:-3]].mean()
        t5 = dr[order[:-5]].mean()
        b = int((dr > 1e-9).sum()); w = int((dr < -1e-9).sum())
        t = len(seqs) - b - w
        idx = rng.integers(0, len(dr), size=(10000, len(dr)))
        boot = dr[idx].mean(axis=1)
        # pooled R over all frames, the vot_ar_offline convention
        pooled = score([d[f"{s}|{a}"]['iou'] for s in seqs])[1]
        print(f"{a:<12} {pooled:7.4f} {dr.mean():+8.4f} {t3:+8.4f} {t5:+8.4f} "
              f"{b:3d}/{w:2d}/{t:2d} {binom_two_sided(min(b,w), b+w):7.3f} "
              f"{(boot <= 0).mean():9.3f}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
