#!/usr/bin/env python3
"""
scripts/vot_init_anatomy.py -- is a run's loss an INITIALISATION failure or drift?

`robustness_gap.md` attributed losses to model drift: ACCEPT 82% at PSR ~19, the
box walking off at 1.88 px/frame. That is a statement about the 5 frames before
the loss, and it is correct for runs that HAVE a healthy stretch to drift away
from. It says nothing about runs that never acquired -- and under the anchored
multi-start protocol there are 419 inits per arm, so "never acquired" is a
failure mode with 419 chances to happen.

This splits the two apart on the one axis that separates them: WHEN the run
fails, measured with the toolkit's own rule, and what IoU/PSR looked like on the
frames immediately after init. A drifting run is healthy at frame 1 by
definition; a run that is already at IoU 0.57 and PSR 7 one frame after
filter_init() never had a filter.

KEYED ON (job, frame), NEVER ON frame ALONE. A FRAME_SOURCE=vot CSV holds every
anchor of a sequence in one file, so a bare frame index collapses the runs
together -- the bug that made calib_report.py under-report rails by 66x.

Usage
-----
  python3 scripts/vot_init_anatomy.py runs/vot/0827_1642-eta05_g5p0
  python3 scripts/vot_init_anatomy.py <dir> --early 10 --per-sequence
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict, Counter

# The toolkit's multistart failure rule, same constants as vot_ar_offline.py.
THRESHOLD = 0.1
GRACE = 10


def load_runs(d):
    """{(sequence, job): [row, ...]} in frame order."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, 'track_*.csv'))):
        seq = os.path.basename(path)[len('track_'):-len('.csv')]
        with open(path) as fh:
            for r in csv.DictReader(fh):
                runs[(seq, r['job'])].append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r['frame']))
    return runs


def first_loss(rows):
    """Index of the first frame of the losing streak, or len(rows) if never."""
    run = 0
    for i, r in enumerate(rows):
        run = run + 1 if float(r['iou']) <= THRESHOLD else 0
        if run >= GRACE:
            return i - GRACE + 1
    return len(rows)


def profile(group, label, frames):
    """Median IoU and PSR on the first few frames after init, over a group."""
    for j in frames:
        ious = [float(v[j]['iou']) for v in group if len(v) > j]
        psr = [float(v[j]['psr_bolme']) for v in group if len(v) > j]
        if not ious:
            continue
        print(f"  {label:<14} f{j:<3} median IoU {st.median(ious):6.3f}   "
              f"median PSR {st.median(psr):7.2f}   n={len(ious)}")


def drift_warning(runs, losses, lead=5, base_skip=5, ctrl=(20, 15)):
    """Is PSR low RELATIVE TO THIS RUN'S OWN median before a drift loss?

    `robustness_gap.md` measured pre-loss PSR at 18.83 against a 7.00 threshold
    and concluded, correctly, that an absolute gate cannot see the loss coming.
    A per-run relative reading is a different question: the same 18.83 means
    something else on a run whose healthy level is 40 than on one whose healthy
    level is 20.

    The control window is the point. A ratio below 1.0 before a loss proves
    nothing on its own -- PSR could be below its own median a third of the time
    everywhere. The -20..-15 window is still healthy by construction (the run
    has not lost yet and is 15 frames clear of the burn-in on the loss), so the
    two columns are the same statistic on the same runs at two times.
    """
    pre, con = [], []
    for k, idx in losses.items():
        v = runs[k]
        if idx < 25:                    # no room for a healthy baseline
            continue
        psr = [float(r['psr_bolme']) for r in v]
        base = st.median(psr[base_skip:idx - 10])
        if base <= 0:
            continue
        pre.append(st.median(psr[idx - lead:idx]) / base)
        con.append(st.median(psr[idx - ctrl[0]:idx - ctrl[1]]) / base)
    if not pre:
        print("\nno runs long enough for the relative-PSR reading")
        return

    def q(x, f):
        return sorted(x)[int(f * len(x))]

    print()
    print(f"relative PSR before a loss ({len(pre)} runs with a healthy baseline)")
    print(f"  {'window':<22} {'median':>7} {'p25':>7} {'p75':>7}   frac < 0.6x")
    for lab, x in (("pre-loss (-%d..0)" % lead, pre),
                   ("control (-%d..-%d)" % ctrl, con)):
        frac = sum(1 for y in x if y < 0.6) / len(x)
        print(f"  {lab:<22} {st.median(x):7.3f} {q(x,0.25):7.3f} "
              f"{q(x,0.75):7.3f} {100*frac:12.0f}%")
    print()
    print("A lift, not a detector: gating on it would veto healthy frames at the")
    print("control rate to catch losses at the pre-loss rate. Use it to SCALE eta.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--early', type=int, default=10,
                    help='a loss at or before this frame index is an INIT '
                         'failure, not drift (default 10 -- one grace window)')
    ap.add_argument('--per-sequence', action='store_true')
    ap.add_argument('--drift-warning', action='store_true',
                    help='the complementary half: is there a RELATIVE PSR '
                         'warning before a drift loss, where the absolute '
                         'threshold has none?')
    args = ap.parse_args()

    runs = load_runs(args.run_dir)
    if not runs:
        raise SystemExit(f"no track_*.csv in {args.run_dir}")

    losses = {}
    for k, v in runs.items():
        idx = first_loss(v)
        if idx < len(v):
            losses[k] = idx

    nframes = sum(len(v) for v in runs.values())
    print(f"{args.run_dir}: {len(runs)} runs, {nframes} frames, "
          f"{len(losses)} lose (rule: IoU <= {THRESHOLD}, grace {GRACE})")
    print()

    buckets = [(0, args.early), (args.early + 1, 30), (31, 100),
               (101, 300), (301, 10**9)]
    hist = Counter()
    for idx in losses.values():
        for lo, hi in buckets:
            if lo <= idx <= hi:
                hist[(lo, hi)] += 1
                break
    print("frames from init to first loss")
    for lo, hi in buckets:
        name = f"<={hi}" if lo == 0 else (f">{lo-1}" if hi > 10**8 else f"{lo}-{hi}")
        print(f"  {name:>8}  {hist[(lo, hi)]:5d}")
    early_n = hist[(0, args.early)]
    if losses:
        print(f"\n  {early_n} of {len(losses)} losses ({100*early_n/len(losses):.0f}%) "
              f"happen within {args.early} frames of init")
    print()

    early = [runs[k] for k, i in losses.items() if i <= args.early]
    other = [v for k, v in runs.items()
             if k not in losses or losses[k] > args.early]
    print("the frames immediately after filter_init()")
    profile(early, 'early-loss', [1, 2, 3, 5])
    profile(other, 'all others', [1, 2, 3, 5])
    print()
    print("An init failure is already visible at f1. A drifting run is not -- that")
    print("is the whole discriminator, and it is why these need opposite fixes.")

    if args.drift_warning:
        drift_warning(runs, losses)

    if args.per_sequence and early:
        print()
        print("early-loss runs by sequence")
        c = Counter(k[0] for k, i in losses.items() if i <= args.early)
        for seq, n in c.most_common():
            print(f"  {seq:<16} {n}")


if __name__ == '__main__':
    main()
