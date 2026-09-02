# Scale filter (DSST) — findings

Moved out of CLAUDE.md 2026-08-31; content unchanged.

### Scale filter — root-caused offline, confirmed on hardware (2026-08-20)

`design/host_app_src/test/scale_loop_sim.cpp` (`make scale_sim`) drives the REAL
`scale_extract`/`scale_detect`/`scale_gate`/`scale_update` in a closed loop with position held.
Native g++, seconds per arm. It **reproduces the board** (the `moving` arm parks for 42 frames
from f131; hardware froze at f130) and refuses to report a verdict when the premise arm fails to
reproduce.

**`SCALE_STEP=1.04` confirmed on hardware** (`run_0820_1513`): mean/worst IoU 0.807/0.579 →
**0.917/0.833**, max box error 31.4% → **9.6%**, mean/worst centre error 2.47/11.07 px →
**1.30/3.52**. **Centre error fell 3.2× from a size-only change** — independent confirmation
that position error was downstream of the scale error.

**On hardware the detector proposed only −1, 0 or +1 over 199 frames** (174/13/12) — but **that
sentence describes ONE SYNTHETIC SCENE and failed twice on 2026-08-25.** On `car1` the detector
proposed ±2 or more seven times, up to **+9**, every one on a frame whose IoU was 0.000; and in
`scale_loop_sim` it uses ±2 *legitimately* on a smooth envelope — capping at 1 parks the
`moving` arm for 123 of 200 frames and ends it 28.0% wrong against 1.0% uncapped. Reading ±1 as
a property of the DETECTOR rather than of that scene is exactly what `SCALE_MAX_STEP=1` would
have been; the sim, not the hardware observation, is the bench that decides that parameter.

**The scale filter is ALREADY slaved to the position gate, and that is not sufficient.** The
whole block runs under `if (scale.enabled() && gate.accept && scale.initialized)`, verified on
hardware (`run_0825_1314`: 577 ACCEPTs, 577 scale evaluations, zero on a held frame). `car1`
frame 490 still got through because the POSITION gate accepted it at PSR 7.87 while the tracker
was 227 px off target. **Do not "add" that slaving; it is there.** `SCALE_MAX_STEP` exists
because the precondition is only ever as good as the PSR gate.

**`SCALE_CONF_MIN` blocks legitimate large corrections.** On the sim's `step` arm the detector
proposes the correct `idx=-14` and the gate vetoes it as `LOW_CONF` for four frames; the box
then walks at 2%/frame where bypassing the gate corrects it in ONE frame. **`conf` cannot
distinguish "wrong proposal" from "big correct correction"** — both match the model poorly, for
the same reason. Exonerated for smooth envelopes; it will bite on any abrupt scale change.

**Where it stops.** `SCALE_ETA` does not help (8.6/10.3/9.2% at 0.025/0.05/0.1); `SCALE_N=65`
gives 7.0% against 8.6% for **3.9× the cost**. ~8-10% box error is this filter's practical floor
— and it is what the worst IoU frames on every 2026-08-24 run are. The next gain needs a
different estimator, not a tuning change.

**Calibration honesty: the sim predicted a=1.04 well and a=1.02 badly** (12.6% → 8.6% predicted;
31.4% → 9.6% measured), and it recovers where hardware never did — **unexplained**; position
error and background pan were both ruled out. **Trust the ORDERING; do not quote the sim's
absolute magnitudes as the board's.** The truth rate matters and the test bench chose it:
`SCALE_TRAJ_AMP=0.30` over 200 frames shrinks the target at ~0.94%/frame, under half of one
scale level, so on most frames the correct proposal rounds to 0.


---

### THE FREEZE IS THE NORMAL OPERATING MODE — measured on 838 real runs, 2026-09-02

`scripts/scale_drift_anatomy.py`, reading the existing `track_*.csv` of `0901_1601-sigma4` and
`0902_1413-l1relu` (419 trajectories each, ~82k alive frames per arm). No hardware, seconds.

**Everything above was measured on ONE synthetic scene.** This generalises it to real video and
to two feature banks, and it changes what the freeze means.

| | `sigma4` | `l1relu` (SHIPPING) |
|---|---|---|
| **`est_h` EXACTLY unchanged** | **91.4%** of frames | **89.4%** |
| `scale_idx` non-zero | 15.6% | 19.7% |
| vetoed: `MAX_STEP` / `LOW_CONF` | 3.59% / 2.14% | 5.03% / 2.26% |
| step `de` autocorrelation, lag-1 | −0.089 | −0.084 |
| `scale_idx` sd, \|idx\|>=2, max\|idx\| | 1.953, 4.82%, **16** | 2.146, 6.95%, **16** |

**The f130 park is not an anomaly — the scale estimate is frozen on ~90% of all frames.** The
driving noise is white (no ringing, no persistent lag), and **about a third of every non-zero
scale decision is thrown away by a veto**, `MAX_STEP` being the larger half. `max|idx|` reaches
16, the very edge of the 33-point filter, so the argmax is hitting its own boundary.

**A "diffusing scale" reading was tested and REFUTED.** `proposed_build_l1relu.md` sec.10.3
observed the `est_h/truth_h` IQR more than doubling between frames 100 and 500 and read it as a
random walk. It is not one:

```
var(log est_h/truth_h) at t=500 / at t=25   MEASURED 1.95x
a random walk over the same 20x in t        PREDICTS  20x
saturating fit R^2 0.824   vs   linear R^2 0.682
```

**The variance saturates while the IQR grows, which is SURVIVORSHIP** — runs whose scale goes
badly wrong fail and leave the population, trimming the tails while the body spreads. The spread
across runs is many runs each PARKED at its own offset, not any run diffusing. Freeze and spread
are the same phenomenon at two scales, and the IQR was read without controlling for selection.

**`SCALE_ETA` is confirmed inert over a 12x range**, extending the 0.025/0.05/0.1 sweep above:

| eta | 0.025 | 0.05 | 0.075 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| max\|rel err\| | 10.4% | 8.3% | 8.3% | 8.3% | 8.3% | 10.4% | 8.3% | 41.5% |
| terminal freeze | 41 fr | 43 | 43 | 43 | 43 | 42 | 42 | 40 (27 held) |

**A 43-frame park at eta 0.3 is the same park as at 0.025.** That is mechanically forced by the
board data: `scale_idx` is ALREADY 0 on 84% of frames, so no learning rate can move a filter that
is not asking to move. **The freeze is a DETECTION failure, not a model-update one** — which is
the same conclusion this file reached from the sim ("the next gain needs a different estimator,
not a tuning change"), now with the mechanism named.

**What is still untried**, in order of how much of the veto they own:
- **`SCALE_MAX_STEP=3`.** `=1` was measured and rejected; `=3` never was, and the rejection was
  taken at 128x128/sigma 2. It owns 3.6-5.0% of frames.
- **`SCALE_N`/`SCALE_STEP`.** `max|idx|=16` is the filter boundary; either the +-1.87x range is
  too narrow or the argmax runs to the edge on noise, and `scale_reason` separates those.
- **`SCALE_CONF_MIN`**, the other ~2.2%, already known here to veto legitimate large corrections.

**Second-order, and it may explain sec.10.3's long-horizon result:** the SHIPPING arm's scale
filter is measurably BUSIER than `sigma4`'s — 19.7% vs 15.6% non-zero, |idx|>=2 at 6.95% vs
4.82%, `MAX_STEP` up 40% — and its survivors carry worse long-horizon IoU (0.388 vs 0.446 at
frame 500) despite more of them surviving. Noisier scale on a longer-surviving tracker is a
candidate cause; it is not established, and the arms differ in geometry as well as bank.

---

### THE DETECTOR IS LOCKED TO "NO CHANGE" — root-caused 2026-09-02, and it retires three fixes

Four measurements, in the order they were taken. **Each one killed the hypothesis the previous
one suggested**, which is why they are all recorded rather than only the last.

#### 1. Scale error is CAUSAL, not cosmetic

Board CSVs, 419 trajectories x 2 arms. In the 5 frames BEFORE a first loss against surviving
runs:

| | pre-loss | survivors |
|---|---|---|
| box >25% mis-sized | **60.4% / 60.6%** | 20.2% / 30.6% |
| median \|log(est_h/truth_h)\| | 0.32 | 0.11 |

A **2.8x** gap, median ratio 1.00 both times (mis-sized in both directions, no bias). Runs that
die are badly mis-scaled while the POSITION gate still accepts them. So scale carries R capacity,
not only A/EAO capacity.

#### 2. The deadband hypothesis — REFUTED

`SCALE_STEP=1.04` quantises the axis to 3.92%, so a half-level deadband is 1.96%. Ground truth
(annotation only, no tracker) over 19,787 frame pairs: **median per-frame size change 2.41%**,
p75 5.97%, p90 13.06%. **Only 44.3% of frames fall inside the deadband** — more than half have a
true change the filter has the resolution to see, and it still reports 0. Quantisation explains
at most half the freeze.

#### 3. DETECTOR GAIN — the scale analogue of `vot_detector_gain.py`'s alpha

Regress the proposed `scale_idx` on the correction the frame warranted,
`log(truth_h/est_h)/log(step)`, on ACCEPT frames:

| detector | alpha | r^2 |
|---|---|---|
| POSITION (`vot_detector_gain.py`) | **0.93** | — |
| scale, `sigma4` (3x3, ch16, 128x128) | **−0.003** | 0.0029 |
| scale, `l1relu` (7x7/2, ch32, 64x64) | **−0.003** | 0.0029 |

Identical to three decimals across two banks and two geometries, on ~140k accepted frames. The
proposal is uncorrelated with the fix **in every error bucket**, including the 43,553 frames
where the box is already >16% wrong (proposes 0 on 75.8% of them).

**Annotation noise does NOT explain it.** VOT ground truth is boxes fitted to per-frame masks and
is heavy-tailed (p99 57.89%), so a tracker arguably SHOULD ignore some of it. Re-running the
regression against LOW-PASS ground truth, +-1 to +-20 frames, moves alpha by **nothing**:
−0.003 at every width, r^2 0.0026-0.0031. The detector is not correctly ignoring jitter.

#### 4. The sim says the ESTIMATOR IS FINE — so it is not an algorithm defect

`scale_loop_sim` now reports the same alpha (`--period` raises the true rate by shortening the
bounded sinusoid; a constant geometric `--rate` ramp is degenerate — 200 frames at 2.4%/frame is
115x and the target leaves the frame, measured). Premise arm still reproduces the 41-frame board
freeze, so the instrument is valid.

| period | peak %/frame | freeze | alpha |
|---|---|---|---|
| 200 (**the hardware config**) | 0.94% | 45 fr | **+0.174** |
| 78 (VOT median) | 2.41% | 14 fr | +0.545 |
| 32 (VOT p75) | 5.89% | 5 fr | +0.678 |
| 12 | 15.7% | 200 fr | **+0.930** |

**Given a genuine smooth size change the detector recovers it, reaching the position detector's
0.93.** The freeze collapses 45 -> 5 frames over the same range. The DSST implementation is not
broken and the freeze is a CONSEQUENCE of slow motion, not a defect.

#### 5. ROOT CAUSE: self-confirmation, not blindness

| | |
|---|---|
| P(idx == 0) measured | **88.4%** |
| P(idx == 0) if the argmax were NOISE | **3.0%** |
| P(\|idx\| <= 1) | 99.3% |

**A blind detector would land on 0 three percent of the time; this lands there 88%.** The scale
response is strongly and consistently peaked at the CURRENT box size, and it stays peaked there
however wrong that size is:

| box wrong by | n | P(idx==0) | mean conf |
|---|---|---|---|
| <1 level | 9257 | 88.9% | 2.85 |
| 2-4 levels | 14495 | 89.6% | 2.84 |
| **>8 levels (>36% off)** | **19715** | **87.1%** | **2.92** |

**`scale_conf` is flat to +-0.04 across the entire range** — the filter reports the same
confidence at 36% wrong as at correct. That generalises this file's existing note that conf
cannot separate a wrong proposal from a big correct one: it separates NOTHING.

**The mechanism is the one this file's sim header hypothesised and never connected to a
statistic**: the model is retrained at the current scale every frame, so the response peaks at
the current scale, so the argmax says "no change", so the model is retrained at the same wrong
scale. The loop is closed and stable. It breaks only when the target moves fast enough to
outrun it, which is exactly the alpha-vs-rate curve in §4.

#### What this RETIRES

Three fixes proposed earlier the same day, all now refuted by the above:

- **Sub-level interpolation of the scale peak** — would interpolate a peak that is in the wrong
  place, more precisely. The argmax is not quantisation-limited; it is mis-located.
- **`SCALE_MAX_STEP=3` / `SCALE_N` / `SCALE_STEP`** — widen a search whose argmax is already
  interior on 99.3% of frames and at 0 on 88.4%.
- **`SCALE_CONF_MIN`** — gates on a statistic measured FLAT against the error it should track.

**The lever is the UPDATE, not the detection or the search.** The model must not be retrained at
a size it has just declined to change. `SCALE_ETA` is inert (§ above) precisely because it scales
the RATE of an update whose TARGET is the problem. Next step is to read `scale_update_shifted`
against Danelljan §3.2 before proposing anything; the sim can now score a candidate in seconds
via alpha, and `--period` gives it a rate axis to be scored on.

---

### THE DIRECTION IS CLOSED — a PERFECT scale filter is worth +0.0023 R (2026-09-02)

`scripts/scale_oracle_bound.py`. Board trajectories, tracker's CENTRE kept, box SIZE replaced
with ground truth, IoU recomputed, VOT's failure rule re-applied. 419 runs per arm.

| arm | R as tracked | R ORACLE SIZE | dR | mean IoU |
|---|---|---|---|---|
| `l1relu` (SHIPPING) | 0.4536 | 0.4559 | **+0.0023** | 0.2625 -> 0.3174 (**+0.0548**) |
| `sigma4` | 0.4549 | 0.4460 | **−0.0089** | 0.2576 -> 0.3113 (+0.0537) |

**A perfect scale filter is worth +0.0023 on one arm and −0.0089 on the other.** For scale:
`l1relu` bought +0.0184 R, the spatial mask +0.0192, RGB +0.0322. An ORACLE is an order of
magnitude below any of them and inside this bench's own noise. On `sigma4` it is NEGATIVE —
oracle sizing shrinks boxes that were accidentally overlapping, tipping frames under the 0.1
threshold.

**The mechanism is visible in the same numbers.** Oracle scale lifts mean IoU by **+0.054** on
both arms — a large, real gain in box quality — and converts essentially none of it into
survival. **Boxes get better; runs do not last longer.** VOT fails a run at IoU <= 0.1 for 10
consecutive frames, and a run at IoU <= 0.1 has lost the target's POSITION. Resizing a box that
is not on the target rescues nothing.

**This also RE-READS the causal evidence above.** §1's ">25% mis-sized on 60% of pre-loss frames
against 20-31% on survivors" is real and reproducible, and it was over-read as causation. The
temporal profile shows both errors ALREADY LARGE 40 frames before the loss (size 30%, centre 0.23
target-units) with **position accelerating faster into it** (x2.22 against size's x1.54). Scale
error is a MARKER of a run in trouble, not its cause.

#### What this retires

**Everything in this file that proposes a scale FIX.** The freeze, the −0.003 gain, the
self-confirming loop and the scale-normalising feature are all correctly measured and remain the
explanation of how the filter behaves — they are simply not worth repairing. Specifically dead:
sub-level interpolation, `SCALE_MAX_STEP`/`SCALE_N`/`SCALE_STEP`, `SCALE_CONF_MIN`, `SCALE_ETA`,
and the HOG feature (`SCALE_FEATURE=1`, built and left in the tree at the default OFF — it
measured WORSE on the sim and 1.20x slower, and the sim was an invalid instrument for it anyway).

#### THE METHOD LESSON, which is the expensive part

**The prior question was never asked.** A day of correct measurement went into attributing a
defect nobody had shown was worth fixing — and the bound that retired it took ten minutes and
data already on disk. **Before attributing a defect, bound what removing it is worth.** An oracle
over the existing trajectories is usually available and always cheaper than the attribution.

The same instrument would have saved the sim detour too: the sim's raw-feature alpha is 0.174-0.83
where the board's is −0.003, so it never reproduced the defect and could not have validated a
repair. **An instrument that cannot show the failure cannot score its fix** — the rule this
project already wrote down for `rgb_vs_gray_loop.py` having no scale filter at all, applied one
level up and missed.
