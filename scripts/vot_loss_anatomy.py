#!/usr/bin/env python3
"""
scripts/vot_loss_anatomy.py -- HOW a sweep loses the target, from track_*.csv alone.

WHY THIS EXISTS
---------------
AR says how much a run lost and when. It cannot say what the tracker was DOING
at the moment it lost, and that is the only thing that distinguishes the two
failure stories this project has confused before: a gate that freezes the
tracker into a loss, versus a filter that drifts off the target while accepting
every frame. Those need opposite fixes.

The instrument is the 5 frames before each run's FIRST drop to IoU <= 0.1:

  * the gate verdict distribution there -- ACCEPT-dominated means drift,
    veto-dominated means the gate;
  * the PSR in that window, against PSR_GATE_MIN;
  * how far the box moved per frame -- a drift walks, a mis-detection jumps.

It also separates gate vetoes that happen WHILE STILL ON TARGET from those that
happen after the run is already lost, because the second kind is the aftermath
of a loss and reading it as a cause is exactly the mistake this was written to
stop (88% of this design's vetoes are the aftermath kind).

Keys rows on (job, frame). A bare frame index is NOT a key in a multi-start CSV
-- that bug once under-reported railed frames 66x.

Usage
-----
  python3 scripts/vot_loss_anatomy.py runs/vot/<armA> [runs/vot/<armB> ...]
"""
import csv, glob, os, sys, math, statistics, collections

LOSS_IOU = 0.1
WINDOW   = 5

def analyse(d):
    files = sorted(glob.glob(os.path.join(d, 'track_*.csv')))
    if not files:
        sys.exit(f"{d}: no track_*.csv")
    vetoes = collections.Counter()
    pre    = collections.Counter()          # gate verdict before first loss
    ctx    = collections.Counter()          # veto context: before/after the loss
    psrs, jumps, lossat, holdrate = [], [], [], []
    nframes = nruns = nsurvive = 0
    for f in files:
        runs = collections.defaultdict(list)
        for r in csv.DictReader(open(f)):
            runs[r['job']].append(r)
        for rs in runs.values():
            rs = [r for r in rs if r['evaluated'] == '1']
            if not rs:
                continue
            nruns += 1
            nframes += len(rs)
            held = sum(1 for r in rs if r['accept'] == '0')
            holdrate.append(held / len(rs))
            first = next((i for i, r in enumerate(rs)
                          if float(r['iou']) <= LOSS_IOU), None)
            for i, r in enumerate(rs):
                if r['accept'] == '0':
                    vetoes[r['reason']] += 1
                    ctx['after_loss' if (first is not None and i > first)
                        else 'on_target' if float(r['iou']) > 0.3
                        else 'ambiguous'] += 1
            if first is None:
                nsurvive += 1
                continue
            lossat.append(first / len(rs))
            w = rs[max(0, first - WINDOW):first]
            for r in w:
                pre[r['reason']] += 1
            if w:
                psrs.append(statistics.median(float(r['psr_bolme']) for r in w))
                for a, b in zip(w, w[1:]):
                    jumps.append(abs(float(b['est_row']) - float(a['est_row'])) +
                                 abs(float(b['est_col']) - float(a['est_col'])))
    npre = sum(pre.values()) or 1
    nvet = sum(vetoes.values()) or 1
    return dict(
        dir=d, seqs=len(files), runs=nruns, frames=nframes,
        survive=nsurvive / max(nruns, 1),
        holdrate=statistics.mean(holdrate) if holdrate else 0.0,
        veto_top=vetoes.most_common(3),
        veto_after_loss=ctx['after_loss'] / nvet,
        veto_on_target=ctx['on_target'] / nvet,
        pre_accept=pre['ACCEPT'] / npre,
        pre_top=pre.most_common(3),
        pre_psr=statistics.median(psrs) if psrs else float('nan'),
        jump=statistics.median(jumps) if jumps else float('nan'),
        lossat=statistics.median(lossat) if lossat else float('nan'))

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for a in (analyse(d) for d in sys.argv[1:]):
        print(f"\n=== {a['dir']}  ({a['seqs']} sequences, {a['runs']} runs, "
              f"{a['frames']} frames)")
        print(f"  runs that never reach IoU<={LOSS_IOU}   {a['survive']:.2%}")
        print(f"  median first-loss position in run       {a['lossat']:.3f} of the run")
        print(f"  mean hold rate                          {a['holdrate']:.2%}")
        print(f"  vetoes                                  {a['veto_top']}")
        print(f"    of them, after the run was lost       {a['veto_after_loss']:.1%}")
        print(f"    of them, while still on target        {a['veto_on_target']:.1%}")
        print(f"  --- the {WINDOW} frames before first loss")
        print(f"    verdicts                              {a['pre_top']}")
        print(f"    ACCEPT share                          {a['pre_accept']:.1%}")
        print(f"    median PSR                            {a['pre_psr']:.2f}")
        print(f"    median box motion                     {a['jump']:.2f} px/frame")

main()
