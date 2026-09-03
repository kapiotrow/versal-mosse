#!/usr/bin/env python3
"""Is the long-horizon SCALE error a random walk, and what feeds it?

WHY. arm_l1relu.md sec.10.3 measured that 71% of the EAO window
(301-755) sees no improvement from better features, and that on surviving runs
`est_h/truth_h` stays centred on 1.0 while its IQR MORE THAN DOUBLES between
frames 100 and 500. That is a description, not a cause, and it does NOT match
the failure `scale_loop_sim.cpp` was built for -- a FROZEN estimate (f130, 70
identical frames). A freeze and a random walk are opposite things, so the knob
to sweep cannot be chosen from the sim without checking which one the board
actually does.

Reads only existing track_*.csv. No hardware, no board, seconds.

  scripts/scale_drift_anatomy.py <run_dir> [run_dir ...]

Columns reported, each pointing at a different knob:
  var(e) vs t   e = log(est_h/truth_h). LINEAR growth => diffusion (a random
                walk); saturation => mean-reverting. This is the discriminator.
  lag-1 rho     autocorrelation of the STEP de = e[t]-e[t-1]. ~0 => white
                driving noise; strongly negative => over-correction ringing;
                strongly positive => persistent bias (the freeze/lag mode).
  frozen        fraction of frames with est_h EXACTLY unchanged -- the f130
                signature, counted rather than assumed absent.
  scale_idx     the DSST argmax actually applied. Its spread is what feeds the
                walk; SCALE_ETA and SCALE_CONF_MIN act here.
  MAX_STEP      fraction of frames the rate limiter clipped. If diffusion is
                fed by clipped REAL motion the knob is SCALE_MAX_STEP, not eta.
"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np

BURN = 10          # toolkit burn-in; the same convention as vot_ar_offline
def runs_of(d):
    for f in sorted(glob.glob(os.path.join(d, 'track_*.csv'))):
        by = defaultdict(list)
        for r in csv.DictReader(open(f)):
            by[r.get('job', '0')].append(r)
        for job, rows in by.items():
            yield os.path.basename(f)[6:-4], job, rows

def first_loss(rows):
    bad = 0
    for i, r in enumerate(rows):
        if i < BURN: continue
        try: iou = float(r['iou'])
        except (ValueError, KeyError): iou = 0.0
        bad = bad + 1 if iou <= 0.1 else 0
        if bad >= 10: return i - 9
    return len(rows)

def analyse(d):
    e_by_t = defaultdict(list)      # frame index -> log ratio, ALIVE runs only
    steps, idxs = [], []
    frozen = tot = maxstep = lowconf = 0
    for seq, job, rows in runs_of(d):
        cut = first_loss(rows)
        prev_e = prev_h = None
        for i, r in enumerate(rows[:cut]):
            try:
                eh, th = float(r['est_h']), float(r['truth_h'])
            except (ValueError, KeyError):
                continue
            if th <= 0 or eh <= 0: continue
            e = np.log(eh / th)
            e_by_t[i].append(e)
            if prev_e is not None:
                steps.append(e - prev_e)
                frozen += (eh == prev_h); tot += 1
            prev_e, prev_h = e, eh
            try: idxs.append(int(r['scale_idx']))
            except (ValueError, KeyError): pass
            rs = r.get('scale_reason', '')
            maxstep += (rs == 'MAX_STEP'); lowconf += (rs == 'LOW_CONF')
    return e_by_t, np.array(steps), np.array(idxs), frozen, tot, maxstep, lowconf

for d in sys.argv[1:]:
    e_by_t, steps, idxs, frozen, tot, maxstep, lowconf = analyse(d)
    print(f"\n=== {d}")
    print(f"  frames analysed (alive only): {tot}   scale_idx samples: {len(idxs)}")
    print(f"  {'t':>5} {'n':>5} {'var(e)':>9} {'IQR':>8}")
    for t in (25, 50, 100, 200, 300, 400, 500):
        v = np.array(e_by_t.get(t, []))
        if len(v) >= 15:
            print(f"  {t:5d} {len(v):5d} {v.var():9.5f} "
                  f"{np.percentile(v,75)-np.percentile(v,25):8.4f}")
    if len(steps) > 2:
        rho = np.corrcoef(steps[:-1], steps[1:])[0, 1]
        print(f"  step de: sd {steps.std():.5f}  lag-1 rho {rho:+.3f}")
    if tot:
        print(f"  frozen frames (est_h exactly unchanged): {100*frozen/tot:.1f}%")
        print(f"  scale_reason MAX_STEP {100*maxstep/tot:.2f}%   LOW_CONF {100*lowconf/tot:.2f}%")
    if len(idxs):
        nz = 100 * (idxs != 0).mean()
        print(f"  scale_idx: {nz:.1f}% non-zero  sd {idxs.std():.3f}  "
              f"|idx|>=2 {100*(np.abs(idxs)>=2).mean():.2f}%  max|idx| {np.abs(idxs).max()}")
