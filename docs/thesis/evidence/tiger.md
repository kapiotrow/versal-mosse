# Chasing `tiger`: eta, sigma, eps_rel — and why none of them is the answer

**Status:** closed · **Updated:** 2026-08-25 · **Scope:** eta, sigma and eps_rel all chased on `tiger`, and why none of them is the answer

**2026-08-25, entirely offline.** `tiger` was identified in
[frozen_detector.md](frozen_detector.md) as the one sequence with a real defect:
the motion is real (still-wins 8%), the target never leaves the search window
(0% of steps exceed ROI/2), and the detector still reports zero on 62% of frames
on hardware and 66% in the model.

Every number here comes from `rgb_vs_gray_loop.py --sequence tiger`, 13 s a run,
with `car1` carried as a known-good control in every table.

## The three knobs

```
knob            value    froze&needed%   meanIoU    cerr   lost at
MOSSE_ETA       0.025            48.2     0.1210   131.5        33
                0.05             40.7     0.2535    62.7       107
                0.125 (ship)     65.8     0.2119    72.4       137
                0.25             49.7     0.1863   139.0        33
                0.5              54.3     0.1773    92.8        33
MOSSE_SIGMA     1.0               0.5     0.1618   109.5        81
                2.0 (ship)       65.8     0.2119    72.4       137
                4.0               0.0     0.1758    85.3        34
                8.0               0.0     0.0720   144.4       100
EPS_REL         1e-3 (ship)      65.8     0.2119    72.4       137
                1e-2             49.2     0.2130   112.9        70
                1e-1              0.0     0.2166   111.5        70
```

**`SIGMA=1` and `EPS_REL=0.1` unfreeze the detector completely — 65.8% → 0.5% /
0.0% — and tracking does not improve.** IoU stays ~0.21 and centre error gets
*worse*. That is the most useful line in the table: **the freeze rate is a
symptom and not the objective.** A detector that moves every frame to the wrong
place scores no better than one that refuses to move, and it is easier to
mistake for health.

`ETA=0.05` is the only setting that improves anything (+0.034 IoU), and the
sweep is not monotone — 0.025 is much worse than 0.05 — so read it as a shallow
optimum, not a trend.

## Three hypotheses, all killed by measurement

**1. Attenuation, i.e. the tracker systematically under-reports.** The early
trace supports it beautifully: frame 4 needs +10.3 col bins and reports +3, frame
5 needs +15.0 and reports +7, frame 10 needs +21.1 and reports +8. The deficit
compounds into a standing lag rather than a loss.
**Refuted by ITERATED LOCALISATION.** Re-cropping at the updated position and
detecting again attacks exactly that, and on `car1` it works as predicted
(cerr 5.6 → 4.7 px, agreement 45.7% → 54.3%). On `tiger` it makes things worse:
lost at 137 → 70, cerr 72 → 112. If the report were merely a shrunk version of
the truth, a second look from closer in would fix it. It does not.

**2. The online update is poisoning the filter.** Refuted directly: with the
crop placed at the GROUNDTRUTH position every frame — no feedback loop, no
accumulated lag, so the correct answer is 0 — the detector reports a median
**17-bin** displacement, and it reports the same 17 bins with the online update
switched off entirely (`eta = 0`, frame-1 filter only). `car1`'s equivalent
number is 2 bins. The filter is wrong before learning gets a chance to make it
wrong.

**3. The filter is inventing an offset.** Also no. The oracle probe's reports
cluster: mean (−10.2, −15.7) bins, i.e. a CONSISTENT direction. A plain NCC
template search — frame-1 luma template, no MOSSE filter, no conv features, no
Hann window — puts the best match at **(−6.7, −8.9) px** from the annotation
centre, within 8 px on only 27% of frames. `car1`: (+2.8, −0.4) px, 100%.

**So `tiger`'s appearance genuinely does not stay centred in its annotation
box**, by ~11 px ≈ 8 bins, and the MOSSE filter amplifies that real offset by
about 1.7×. It is the same disease as `nature` in a milder form: the box centre
is a min–max reduction over a deforming object, and it drifts against the
appearance any translation tracker can lock onto. `nature` is 80% still-wins and
untrackable; `tiger` is 8% still-wins with an 11 px standing bias, which is
trackable but permanently penalised.

## Two candidates worth a hardware A/B, neither a fix

**`MOSSE_ETA = 0.05`**, full sequences, 8 sequences, gray:

```
seq       frames   eta=.125   eta=.05     delta
car1         742     0.7131    0.7163   +0.0032
tiger        365     0.1696    0.2037   +0.0341
nature       999     0.1121    0.1104   -0.0017
crabs1       160     0.0188    0.0188   +0.0000
book         175     0.0187    0.0212   +0.0025
soccer2      129     0.0141    0.0525   +0.0384
animal       100     0.0227    0.0246   +0.0019
ball3        171     0.0366    0.0366   +0.0000
          frame-weighted 0.2533 -> 0.2599   unweighted 0.1382 -> 0.1480
```

Six better, two tied, one worse by 0.0017 — and the one worse is `nature`, the
sequence that cannot be tracked anyway. Uniform in sign, which `HOLD_COAST` was
not.

**A detection GAIN** (multiply the reported displacement by g before applying
it) is the crudest possible attenuation compensation, and on `tiger` it is the
largest single effect found: IoU 0.2119 → **0.3230** at g=1.5, cerr 72 → 46,
lost at 137 → 187. It costs `car1` a little and both collapse at g=2.0. Over the
full 8 sequences:

```
seq        gain=1.0   gain=1.25   gain=1.5
car1         0.7131      0.7135     0.6990
tiger        0.1696      0.1528     0.3047
nature       0.1121      0.1132     0.1124
crabs1       0.0188      0.0197     0.0207
book         0.0187      0.0211     0.0203
soccer2      0.0141      0.0157     0.0078
animal       0.0227      0.0219     0.0234
ball3        0.0366      0.0366     0.0366
        frame-weighted 0.2533 / 0.2519 / 0.2670
```

**Read that the way the coast A/B taught us to read a concentrated win.** The
+0.0137 frame-weighted is almost entirely `tiger`; `car1` and `soccer2` both get
worse, and 1.25 is worse than 1.0 overall while 1.5 is better — a non-monotone
response is what a knob looks like when it is compensating for something it does
not model. It is a hack aimed at a mechanism the oracle probe says is not the
real one, so treat it as a measurement of what the standing lag costs, not as a
proposal.

## What this run added to the bench

`run_arm()` gained `detect_iters` and `detect_gain`, both defaulting to the
shipped behaviour and verified to reproduce the baseline digit-for-digit at
`detect_iters=1, detect_gain=1.0`.

**A caution on `froze&needed%` with `detect_iters > 1`:** the "needed"
displacement is recorded after the iteration loop, so it measures the residual
of the last pass, not the original frame's requirement. The two are the same
number only at one iteration. Compare `cerr` and IoU across iteration counts,
never the freeze rate.
