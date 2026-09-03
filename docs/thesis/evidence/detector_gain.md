# The detector is EXONERATED on targets that actually translate

**Status:** closed · **Updated:** 2026-08-28 · **Scope:** the position detector is EXONERATED: it recovers 93% of annotated motion

**2026-08-28.** From the existing `track_*.csv` of the 62-sequence arms — no board time,
no dataset. Instrument: `scripts/vot_detector_gain.py`.

## The question this settles

`robustness_gap.md` measured *how* runs lose (ACCEPT 82% at PSR ~19, box walking 1.88
px/frame) and read it as **model drift** — the filter learning the background. The same
log is equally consistent with a **detector that under-reports displacement**: a tracker
that recovers only part of each frame's motion also walks off smoothly, also accepts every
frame, and also has a healthy-looking PSR. Those need opposite fixes and nothing separated
them.

`track_*.csv` already carries both halves of the matched pair — the annotation
(`truth_row/col`) and the detector's own peak offset (`dr_bin/dc_bin`) — so the
displacement gain

```
alpha = sum(t*r) / sum(t*t)      t = true component motion, r = reported shift, both frame px
```

is computable per bucket, over accepted on-target frames (IoU > 0.3) only.

## The result, and the split IS the result

`0827_1642-eta05_g5p0`, 102,146 matched components:

```
                        MOVERS (17 seq)          EVERYTHING ELSE
|motion|/size   n        alpha              n        alpha
0.00-0.05    15187       0.895            63000       0.405
0.05-0.10     6867       0.950             5507       0.513
0.10-0.15     3808       0.966             1225       0.481
0.15-0.20     2179       0.971              386       0.593
0.20-0.25     1204       0.978              234       0.679
0.25-0.30      914       0.973              103       0.738
     >0.30     1309       0.879              223       0.080
pooled       31468       0.930            70678       0.218
```

**On genuinely translating targets the detector recovers 93% of the annotated motion, and
95-98% in every bucket from 0.05 to 0.30 target-sizes per frame.** There is no
large-displacement deficit, no search-range limit and no lag to fix. Per sequence:
`drone_across` 0.99, `handball2` 0.97, `conduction1` 0.95, `car1` 0.93 — and those are the
sequences with the most fast frames.

Pooled over all 62 the figure is 0.686, which reads as a broken detector. **It is not: it
is `nature`, `girl`, `wiper`, `graduate`, `basketball` and friends, where the annotated box
centre moves because the box is a min-max over a deforming shape while the object's pixels
stay put.** `frozen_detector.md` measured that directly (on 80% of `nature`'s frames NOT
moving correlates better, NCC 0.940 vs 0.816). alpha near 0 there is the CORRECT behaviour
of a correlation filter, not a defect. **Reading the pooled 0.686 as a detection problem
would have been the fourth wrong diagnosis of `nature`.**

The >0.30 bucket looks like a cliff (0.879 movers, 0.080 other) for the same reason: at
that speed the non-movers' "motion" is almost entirely annotation jitter on a deforming
box.

## What it rules out

- **Iterated re-detection / multi-pass re-cropping.** It converges alpha toward 1 and alpha
  is already 0.93-0.98 where it matters. `tiger.md` had already found it helps `car1`
  (cerr 5.6 -> 4.7 px) and hurts `tiger`; this says why, and says the `car1`-class gain is
  worth ~5% of a displacement the tracker mostly already has.
- **Sub-bin / parabolic peak refinement.** Same argument, and `subbin_lag.md` had it.
- **More search range (higher `TARGET_PADDING`) for fast motion.** The movers do not run out
  of range: alpha holds to 0.973 at 0.25-0.30 target/frame.

## The gate, re-checked on-target with the same rows

The pre-loss window makes holds look expensive (7.4% of frames carrying 13% of the true
motion, 7.4 px/frame on held frames vs 3.5 on accepted). **That is the aftermath confound
again** — those windows are already partly lost. Restricted to genuinely on-target frames
(IoU > 0.3) the hold rate is **1.8%**, flat across every speed bucket (1.7% at
motion < 0.05 target, 1.7% at > 0.30), and 914 of the 925 on-target vetoes are
`NEGATIVE_PEAK`. **The gate is not vetoing the fast frames and there is nothing here for a
coast or a threshold to recover.** Consistent with `robustness_gap.md`'s 95.8% figure,
reached from the other side.

## What is left

Localisation is exonerated; the gate is exonerated (twice); quantization is exonerated
(settled questions); `MOSSE_ETA` has been harvested (+7.1% R). **The deficit is in what the
features can represent under deformation, and in what the filter is allowed to learn.**
69% of on-target frames are on targets the appearance model cannot follow by translation at
all. That is the pooling / receptive-field item and the spatial-reliability item, in that
order — see the proposal at the end of `robustness_gap.md`.

## Reproduce

```bash
python3 scripts/vot_detector_gain.py runs/vot/0827_1642-eta05_g5p0 \
    --movers car1,drone_across,handball1,handball2,conduction1,diver,helicopter,\
soccer1,birds1,bolt1,motocross1,marathon,rowing,surfing,agility,polo,tennis
python3 scripts/vot_detector_gain.py runs/vot/0826_1550-rgb15 --per-sequence
```

The mover list is hand-picked and that is a weakness: it is a judgement about which
sequences translate, not a measurement. `scripts/vot_motion_check.py` answers exactly that
question from pixels and should be used to derive the list before this number is quoted in
the write-up.
