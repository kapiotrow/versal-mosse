#!/usr/bin/env python3
"""
scripts/vot_ar_offline.py -- VOT's A/R rule applied to OFFLINE closed-loop runs.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
`vot analysis` is the metric of record and this is NOT it. The toolkit scores the
MULTI-START protocol: each sequence is entered at its own anchors, forward and
backward, and a run that fails is simply over -- 419 runs per arm on the board.
`rgb_vs_gray_loop.py` runs ONE pass from frame 0 per sequence. So the numbers
here are computed with the toolkit's RULE on a different set of RUNS, and they
are not comparable to `~/vot/analysis/*` or to the published VOT-STb2022 table.

Why it is still worth computing. `runs/vot/evidence_ar.md` showed mean IoU and
AR ordering two arms OPPOSITELY on identical trajectories: the failure rule
discards everything after 10 consecutive frames at overlap <= 0.1, so a short
excursion outweighs hundreds of good frames, and mean IoU cannot see the TIMING
of a loss at all. Any arm that changes the hold rate -- which the coarse-feature
arm does, 2.3x -- has exactly that failure mode. Applying the rule here converts
"mean overlap" into "how long did it survive, and how good was it while alive",
which is the shape of the question even when the runs are not the toolkit's.

The rule, from the toolkit's multistart experiment:
    threshold 0.1, grace 10 consecutive frames, burn-in 10 frames after init.
    A run's PROGRESS is the index of the first failure, or its full length.
    Robustness = sum(progress) / sum(length).
    Accuracy   = mean overlap over the tracked frames.
The init frame counts in the accuracy denominator with overlap 0 (Phase 0b
caveat), so a perfect tracker cannot score 1.0 -- reproduced here so the two
conventions do not silently differ.

VALIDATED, AND ITS RESOLUTION IS MEASURED -- NOT ASSUMED
-------------------------------------------------------
Run against the board's own multi-start CSVs, whose toolkit A/R is known:

    arm                 A(mine)  A(tool)   R(mine)  R(tool)
    rgb_eta125           0.5352   0.5043    0.3333   0.3065
    rgb_eta05            0.5022   0.5100    0.3492   0.3283
    eta05_g5p0           0.5010   0.5100    0.3471   0.3417

Absolute values are biased (R runs ~0.02 high: the toolkit computes bounded
overlaps from the trajectory itself, this reads the board's own `iou` column).
So only DELTAS between arms scored identically mean anything. On the two pairs
whose answer is known:

    eta125 -> eta05    toolkit +0.0218    here +0.0159    AGREE
    eta05  -> g5p0     toolkit +0.0134    here -0.0021    DISAGREE

**RESOLUTION IS ABOUT 0.02 IN R. It reproduced a +0.022 ordering and got a
+0.013 one backwards.** Do not read a dR smaller than ~0.02 from this tool, in
either direction, and do not use it to accept an arm -- only to decide whether
an arm is worth a board run. The failed pair is stated here rather than dropped
because a tool validated on its successes only is not validated.

Usage
-----
  python3 scripts/vot_ar_offline.py <runs.json> [armA armB ...]
where <runs.json> maps "<sequence>|<arm>" -> {"iou": [...per frame...]}.
"""
import json, sys, statistics as st

THRESHOLD = 0.1
GRACE     = 10
BURNIN    = 10


def progress(iou):
    """Index of the first failure under the toolkit's rule, else len(iou)."""
    n = len(iou)
    run = 0
    for i in range(n):
        if i < BURNIN:
            continue
        run = run + 1 if iou[i] <= THRESHOLD else 0
        if run >= GRACE:
            return i - GRACE + 1
    return n


def score(runs):
    """runs: list of per-frame overlap lists. Returns (A, R, tracked, total)."""
    tracked = total = 0
    acc = []
    for iou in runs:
        p = progress(iou)
        tracked += p
        total += len(iou)
        # The init frame enters accuracy with overlap 0, matching the toolkit.
        acc.extend([0.0] + list(iou[:p]))
    return (st.fmean(acc) if acc else 0.0), (tracked / total if total else 0.0), tracked, total


def main(path, arms):
    d = json.load(open(path))
    seqs = sorted({k.split('|')[0] for k in d})
    if not arms:
        arms = sorted({k.split('|')[1] for k in d})

    print(f"{len(seqs)} sequences, single-start, toolkit rule "
          f"(threshold {THRESHOLD}, grace {GRACE}, burn-in {BURNIN})")
    print("NOT the toolkit's AR: these are single-start runs, not the anchored")
    print("multi-start protocol. Do not compare to ~/vot/analysis or to Table 12.")
    print()
    print(f"{'arm':<12} {'accuracy':>9} {'robustness':>11} {'tracked':>8} {'total':>7}")
    print("-" * 52)
    pooled = {}
    for a in arms:
        A, R, tr, to = score([d[f"{s}|{a}"]['iou'] for s in seqs])
        pooled[a] = (A, R)
        print(f"{a:<12} {A:9.4f} {R:11.4f} {tr:8d} {to:7d}")

    if len(arms) == 2:
        x, y = arms
        print()
        print(f"{'sequence':<16} {'A '+x:>9} {'A '+y:>9} {'R '+x:>9} {'R '+y:>9}   dR")
        print("-" * 66)
        wins = losses = 0
        for s in seqs:
            Ax, Rx, *_ = score([d[f"{s}|{x}"]['iou']])
            Ay, Ry, *_ = score([d[f"{s}|{y}"]['iou']])
            if Ry > Rx + 1e-9: wins += 1
            elif Ry < Rx - 1e-9: losses += 1
            if abs(Ry - Rx) > 0.05:
                print(f"{s:<16} {Ax:9.3f} {Ay:9.3f} {Rx:9.3f} {Ry:9.3f} {Ry-Rx:+7.3f}")
        print("-" * 66)
        print(f"(only |dR| > 0.05 listed)  R: {y} better on {wins}, "
              f"worse on {losses}, tied on {len(seqs)-wins-losses}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
