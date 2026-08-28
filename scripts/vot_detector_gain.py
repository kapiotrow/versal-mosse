#!/usr/bin/env python3
"""
scripts/vot_detector_gain.py -- IS THE DETECTOR THE PROBLEM? From track_*.csv alone.

WHY THIS EXISTS
---------------
`vot_loss_anatomy.py` says the tracker loses by ACCEPTING every frame while it
walks off the target, and calls that drift. Drift is a claim about what the
FILTER LEARNS. But the same log is equally consistent with a detector that
simply under-reports how far the target moved, which is a claim about
LOCALISATION and needs the opposite fix. Nothing in the harness separated them.

The instrument is a displacement GAIN. `track_*.csv` carries both the annotated
box (truth_row/col) and the detector's own reported peak offset (dr_bin/dc_bin),
so on every accepted, on-target frame we have a matched pair:

    true component motion  t  (frame px, from the annotation)
    reported shift         r  (dr_bin * box*2/128, i.e. bins -> frame px)

and alpha = sum(t*r)/sum(t*t) is the fraction of the true motion the detector
recovers. alpha = 1 is a perfect detector; alpha < 1 leaves a residual lag that
the next frame must absorb.

TWO THINGS THIS GETS RIGHT THAT AN EYEBALL DOES NOT
  * On-target frames only (IoU > 0.3) and accepted frames only. After a loss the
    box motion is uncorrelated with anything, and mixing those in makes every
    arm look equally broken -- the same "aftermath, not cause" confound that
    made 88% of this design's gate vetoes look like a gate problem.
  * Bucketed by |t| / sqrt(target area). A detector can be perfect on small
    motion and useless on large, and the pooled number hides it. Per-sequence R
    correlates -0.48 with the fraction of frames moving > 0.25 target, so the
    large-motion bucket is the one that matters.

AND ONE CONFOUND IT MUST CONTROL FOR, WHICH IS THE WHOLE POINT OF --movers.
On sequences like `nature`, `girl` and `wiper` the annotated box centre moves
because the box is a min-max over a DEFORMING shape, while the object's pixels
stay put (`runs/vot/frozen_detector.md` measured this directly). There, alpha
near 0 is the correct behaviour of a correlation filter, not a defect. Pooling
those together with genuinely translating targets produces a low alpha and an
entirely wrong conclusion. Pass --movers to split them.

Keys rows on (job, frame): a bare frame index is not a key in a multi-start CSV.

Usage
-----
  python3 scripts/vot_detector_gain.py runs/vot/<arm> [--movers car1,drone_across,...]
  python3 scripts/vot_detector_gain.py runs/vot/<arm> --per-sequence
"""
import csv, glob, os, sys, math, collections

ON_TARGET = 0.3
BUCKET    = 0.05
NBUCKET   = 6          # last bucket is "> NBUCKET*BUCKET"


def pairs(path):
    """Yield (seq, t, r, rel) per accepted on-target frame COMPONENT."""
    for f in sorted(glob.glob(os.path.join(path, 'track_*.csv'))):
        seq = os.path.basename(f)[len('track_'):-len('.csv')]
        runs = collections.defaultdict(list)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                runs[row['job']].append(row)
        for rs in runs.values():
            rs = [r for r in rs if r['evaluated'] == '1']
            for prev, cur in zip(rs, rs[1:]):
                if cur['accept'] != '1' or float(cur['iou']) <= ON_TARGET:
                    continue
                h, w = float(cur['est_h']), float(cur['est_w'])
                size = math.sqrt(float(cur['truth_h']) * float(cur['truth_w']))
                if h <= 0 or w <= 0 or size <= 0:
                    continue
                # bins -> frame px: the ROI is box * TARGET_PADDING(2) resampled to 128
                for t, r in ((float(cur['truth_row']) - float(prev['truth_row']),
                              float(cur['dr_bin']) * h * 2.0 / 128.0),
                             (float(cur['truth_col']) - float(prev['truth_col']),
                              float(cur['dc_bin']) * w * 2.0 / 128.0)):
                    yield seq, t, r, abs(t) / size


class Acc:
    __slots__ = ('sxy', 'sxx', 'n')
    def __init__(self): self.sxy = self.sxx = 0.0; self.n = 0
    def add(self, t, r): self.sxy += t * r; self.sxx += t * t; self.n += 1
    @property
    def alpha(self): return self.sxy / self.sxx if self.sxx > 0 else float('nan')


def report(title, buckets):
    tot = Acc()
    for b in buckets.values():
        tot.sxy += b.sxy; tot.sxx += b.sxx; tot.n += b.n
    print(f"-- {title}: n={tot.n} components, alpha={tot.alpha:.3f}")
    for k in sorted(buckets):
        lab = (f">{k*BUCKET:.2f}" if k == NBUCKET
               else f"{k*BUCKET:.2f}-{(k+1)*BUCKET:.2f}")
        b = buckets[k]
        print(f"     |motion|/size {lab:>10s}  n={b.n:6d}  alpha={b.alpha:6.3f}")


def main(argv):
    path = argv[0]
    movers = set()
    per_seq = False
    if '--movers' in argv:
        movers = set(argv[argv.index('--movers') + 1].split(','))
    per_seq = '--per-sequence' in argv

    groups = collections.defaultdict(lambda: collections.defaultdict(Acc))
    by_seq = collections.defaultdict(lambda: collections.defaultdict(Acc))
    for seq, t, r, rel in pairs(path):
        k = min(int(rel / BUCKET), NBUCKET)
        g = ('mover' if seq in movers else 'other') if movers else 'all'
        groups[g][k].add(t, r)
        by_seq[seq][k].add(t, r)

    print(f"== {os.path.basename(path)}  (accepted, on-target frames only)")
    for g in ('all', 'mover', 'other'):
        if g in groups:
            report(g, groups[g])

    if per_seq:
        print(f"\n{'sequence':18s} {'n':>7s} {'alpha':>7s} {'alpha>0.15':>11s} {'%>0.15':>7s}")
        rows = []
        for seq, bs in by_seq.items():
            tot, fast = Acc(), Acc()
            for k, b in bs.items():
                tot.sxy += b.sxy; tot.sxx += b.sxx; tot.n += b.n
                if k >= 3:
                    fast.sxy += b.sxy; fast.sxx += b.sxx; fast.n += b.n
            if tot.n >= 400:
                rows.append((fast.n / tot.n, seq, tot, fast))
        for frac, seq, tot, fast in sorted(rows, reverse=True):
            print(f"{seq:18s} {tot.n:7d} {tot.alpha:7.3f} {fast.alpha:11.3f} {frac*100:6.1f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
