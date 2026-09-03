# Confidence-modulated learning rate — REFUTED on both statistics, and the MUTANT does not lose

**Status:** closed · **Updated:** 2026-09-03 · **Scope:** LMCF-style confidence-modulated eta (PSR and APCE) and the two-filter ensemble premise behind it: both refuted, and the confidence class is closed

**2026-09-03.** `runs/vot/0903_offline-ceta/`, `scripts/rgb_vs_gray_loop.py` arm suffix
`-ceta<N>`, 62 sequences / 19,903 frames, shipping bank and operating point
(`rgb-l1relu`, `--eta 0.05 --psr-min 5.0`), scored with `vot_ar_offline.py` and
`scripts/grid_stats.py`. Offline only — no board time. Claim id: `N-22`; corrects `O-03`.

`robustness_proposals.md` §4 proposed this and `baselines.md` ranked the two-filter ensemble
around it. The literature form is LMCF's high-confidence update (Wang et al. 2017 §3.2), which
uses **APCE alongside the peak** rather than PSR alone — the reason APCE is here at all is this
project's own rule that two INDEPENDENT instruments beat one instrument twice, and PSR is
already what `PSR_GATE_MIN` tests.

The law, one-sided on purpose (`hi = 1.0`, eta is only ever REDUCED) because the upward half is
already refuted on hardware — `MOSSE_ETA=0.1` lost EAO 0.1960 -> 0.1817 (`arm_l1relu.md` §13)
and 0.025 is "much worse":

```
eta_eff = eta * clamp(conf / median(conf over this run's PAST frames), lo, 1.0)
```

## The prediction, written down first

**Accept only on `dR >= +0.02` surviving drop-top-3, AND the mutant losing.** `-cetaneg`
inverts the law (raises eta when confidence is LOW); if it does not lose, the statistic is inert
and any gain is "perturbing eta", not confidence. That is the role `-chrelneg` played in
`channel_reliability.md`, and the reason that null was informative rather than merely
disappointing.

Expectation recorded before the run: **a small effect or none.** PSR's within-run separation is
`P[pre-loss < control] = 0.618` — a weak discriminator (see §"What decided the warm-up").

## The result

```
arm                     poolR   dRmean    trim3    trim5    b/w/t    P(dR<=0)
rgb-l1relu (control)   0.4364        —        —        —        —          —
psr  ceta6  (floor .6) 0.4179  -0.0177  -0.0199  -0.0206   2/ 7/53     0.952
psr  ceta4  (floor .4) 0.4126  -0.0215  -0.0248  -0.0257   5/ 8/49     0.962
psr  cetaneg (MUTANT)  0.4307  -0.0130  -0.0139  -0.0144   2/ 5/55     0.973
apce ceta6             0.4236  -0.0108  -0.0127  -0.0131   2/ 7/53     0.878
apce cetaneg (MUTANT)  0.4198  -0.0241  -0.0353  -0.0365   4/ 6/52     0.913
psr  ceta6 warmup 20   0.4177  -0.0170  -0.0192  -0.0199   2/ 7/53     0.953
```

**Every arm loses on every column.** The best is −0.0108 against a bar of +0.02.

**More modulation is monotonically worse**: floor 0.4 (−0.0215) loses to floor 0.6 (−0.0177).
That is a wrong direction, not a tuning miss.

## Did the mechanism hold? NO — and this is the decisive part

Each arm scored directly against **its own mutant**:

```
psr  ceta6 vs psr  cetaneg   pooled -0.0047   trim3 -0.0169   P(dR<=0) 0.571   5/ 7/50
apce ceta6 vs apce cetaneg   pooled +0.0133   trim3 -0.0153   P(dR<=0) 0.265   7/ 8/47
```

**Inverting the law is not worse than the correct law.** For PSR the mutant is if anything
better. For APCE the pooled **+0.0133 looks like the mechanism holding and does NOT survive a
trim** (−0.0153), with a sign test of exactly 1.000 (7 better, 8 worse, 47 tied).

**Both statistics are INERT for this purpose.** Unlike `N-20`, where the anti-reliability mutant
lost 0.0396 and established that the statistic carried real information, **nothing survives this
null.** There is no "mechanism real, lever short" reading available here and it must not be
written as one.

**THE APCE POOLED NUMBER WAS NEARLY REPORTED AS A MECHANISM CONFIRMATION.** `vot_ar_offline.py`
prints pooled R and stopping there would have produced exactly the claim the trim refutes. This
is the failure mode `grid_stats.py` exists for and claims.md rule 2 warns about, reached once
more by a new route.

## What decided the warm-up, and why it was not the limit

Measured on the SHIPPING arm's own logs before any bench arm was written
(`runs/vot/0902_1413-l1relu`, 419 runs / 180,125 evaluated frames), because the operating point
had moved since `robustness_proposals.md` §4 (claim `M-14`):

* **The running median is biased high early** — `median(psr[:k])` reads **1.86x** the run's
  settled level at k=1, 1.05x by k=12, 0.98x by k=20. Seeding it from frame 1 would cut eta
  across the board for a spurious reason.
* **The relative statistic does not SEPARATE early**: `P[doomed < healthy]` is 0.608 at frame 1,
  0.529 at frame 3 and **0.461 by frame 12** — worse than useless. ABSOLUTE psr separates the
  same runs at **0.82-0.85**, because dividing by a doomed run's own depressed median removes
  exactly the evidence.
* So the early population is not reachable by this law — and not by eta at all: those are INIT
  failures (61 runs broken one frame after `filter_init`, f1 IoU 0.571), where lowering eta
  PRESERVES the bad init. Closed separately as `N-02`.
* The warm-up therefore EXEMPTS them, priced not hidden: **N=12 leaves ~17% of all losses
  untouched, N=20 leaves 24.3%.**

**And the warm-up turned out not to be the binding constraint**: N=20 (−0.0170) and N=12
(−0.0177) are the same arm. The diagnosis was correct and the law is what fails.

**The within-run premise itself DID transfer** — re-measured here rather than reused, and it is
slightly stronger than on the arm §4 measured (0.892 vs 1.004 there):

```
window                median      p25      p75   frac < 0.6x
pre-loss (-5..0)       0.808    0.559    1.062        29.6%
control (-20..-15)     0.978    0.770    1.135        11.7%
P[pre-loss < control] = 0.618      paired 147/230 (63.9%)
```

**So a real signal existed and acting on it still lost.** That is the substantive finding: the
dip is detectable and is not a usable control input.

## Controls

* **Degenerate control: `-ceta10` (floor 1.0) reproduces the baseline BIT-FOR-BIT**, maxdiff
  exactly 0 on every frame. The plumbing perturbs nothing.
* **The arms really modulate**: `etascale` (persisted per frame, the mechanism check —
  `mask_ebox`'s role) shows 51.1% of frames at exactly 1.0 (conf >= median) and the rest
  reduced, floor respected at 0.600. An arm that moved AR with a flat `etascale` would not have
  moved it by modulating eta.
* **The control arm reproduces its recorded value digit-for-digit** (0.5077 / 0.4364 / 8686,
  `layer1_features.md`), so the arms are readable.
* Causality: `conf_hist` is appended AFTER the frame's own scale is computed, so a frame can
  never normalise itself; vetoed frames never enter the history, because the median must
  describe frames the filter actually learned from.

## ROOT CAUSE — PSR is NON-MONOTONE in correctness, so a monotone law is misspecified

**2026-09-03, after the null.** The arms above say the law fails; this says why, and it is a
result about the STATISTIC rather than about the law. Measured on the shipping arm's own
trajectories (`runs/vot/0902_1413-l1relu`, 180,125 evaluated frames), no new runs.

### The literature's premise, and the regime it holds in

LMCF's high-confidence update (Wang et al., CVPR 2017, arXiv 1703.05020) targets **occlusion and
motion blur**: the target is ABSENT, the response degrades, APCE falls, and skipping the update
stops the model learning the occluder. In that regime confidence IS monotone in correctness.
The limitation is already stated in the CF literature and treated there as a footnote — APCE
"will treat the occlusion as the real target", and a tracker "can produce larger response values
but concentrate on the wrong target". **On this tracker that footnote is the dominant failure.**

### The measurement — fraction of frames ALREADY LOST (IoU <= 0.1), by PSR band

Computed PER SEQUENCE and then medianed, so pooling cannot carry it:

```
  PSR band       n_seq   median frac lost
   0 - 10          59          70.2%
  10 - 20          60          50.0%
  20 - 30          50          48.4%   <- minimum
  30 - 50          37          77.8%
  50 +             20          96.9%   <- the MOST CONFIDENT band
```

**The highest-confidence band is 96.9% lost.** Not three sequences: 17 of the 20 sequences with
>= 20 frames at PSR > 50 exceed 50% lost. Pooled, `P[PSR(lost) < PSR(on target)] = 0.625` — the
two tails cancel, which is why the pooled statistic looks merely weak instead of misshapen.

### The mechanism — those frames are FROZEN on background

Median frame-to-frame box motion:

```
  population                  n     |d est| px    |d truth| px
  on target (IoU>0.5)      44866         2.56            3.04
  lost, PSR<=20            45973         2.81            3.16
  lost, PSR>50              9034         0.00            2.24
```

**Zero pixels of motion while the target moves 2.24 px/frame.** The filter has welded itself to a
static background patch and is correlating that patch against a filter trained on it, which is a
SHARPER correlation than a deforming target against a temporally averaged filter. Hence the
highest PSR in the dataset. This generalises CLAUDE.md's recorded anecdote — "a tracker 179 px
off target, confidently locked to background, reported PSR 33" — from one case to 9,034 frames,
and it identifies TWO distinct lost populations, not one: wandering (low PSR, still moving) and
welded (high PSR, frozen).

### Why this explains the arm AND the mutant

A confidence-modulated eta assumes `confidence up => correctness up`. The relation has an
interior optimum instead, so:

* the law holds `eta_eff` at its MAXIMUM exactly in the PSR > 50 band that is 96.9% lost — it
  learns hardest while welded to background;
* **inverting a U-shaped relation is about as wrong as not inverting it**, which is why
  `-cetaneg` failed to lose and why that non-result was otherwise hard to read;
* the real within-run dip (P = 0.618) is an AVERAGE OVER TWO OPPOSITE MEANINGS — a fall from
  60 to 30 is a tracker un-sticking from background, a fall from 30 to 15 is a tracker losing the
  target. No single monotone law serves both.

### Three secondary causes, in descending weight

1. **Protocol.** The high-confidence update is validated on OTB-100 one-pass with NO RESET,
   scored over the whole sequence, where delaying corruption pays through eventual recovery.
   VOT terminates 10 frames after failure and re-anchors, so recovery is worth nothing — already
   retired here as `N-13`.
2. **LMCF's confidence check is not standalone**: it is coupled to multi-peak detection, where
   the secondary peaks are candidate target locations and re-detection picks among them. Ported
   without that partner, the confidence half keeps the information and loses the mechanism that
   acts on it.
3. **Learning-rate regime.** Classical CF trackers run eta ~ 0.0125-0.025 (BACF 0.0125, DSST
   0.025); this ships at 0.05, so there is less accumulated corruption to protect against. The
   weakest of the three and it cuts both ways.

### What this makes worth trying, and what it does not

The U-shape suggests a **two-sided** law — `eta_eff = eta * f(|log(conf/median)|)` — which is not
what the literature does and follows directly from the measurement. **DO NOT BUILD IT.** Its
upper tail is the welded population, which the rows below show is post-loss; its lower tail is
the 0-10 band, which `R-06` already established is the aftermath too. A two-sided law is
therefore two aftermath detectors bolted together, and even at the optimum the band is 48.4%
lost. **Confidence-derived per-frame statistics are CLOSED as a class for this tracker**: pre-loss
the failure looks confident and MOVING (2.81 px/frame), not frozen and not low-PSR.

**THE WELDED POPULATION IS NOT AN OPPORTUNITY, AND THIS PARAGRAPH FIRST SAID IT WAS.** It is
trivially detectable (`|d est| == 0` at high PSR, 5% of all frames) and detectable ONLY AFTER THE
RUN IS ALREADY DEAD. Measured, 221 losing runs with >= 25 frames of history, nature and tiger
excluded:

```
  welded-frame rate in the 20 frames BEFORE first loss     0.70%
  welded-frame rate AFTER first loss                       4.59%   (6.6x)
  losing runs with ANY welded frame pre-loss             6 / 221   (2.7%)
```

That is the SAME SHAPE as `PSR_GATE_MIN`'s 95.8%-after-loss problem (`R-06`), and under VOT's
reset rule post-loss information is worth nothing (`N-13`). **The direction is closed.** The
first version of this section proposed it as the follow-up and cited `M-13` — bound the prize
first — without doing so; the prior check costs ten minutes on stored CSVs and closes it.

The descriptive finding is unaffected and survives the trim: excluding `nature` (`N-09`, its
pixels do not move) and `tiger` (`N-10`, deformation), the welded population is still 4,842
frames at a 0.00 px median across 17 of 60 sequences.

Same root cause explains `R-03`/`R-06`: raising `PSR_GATE_MIN` would cut the 0-10 band, which is
mostly runs already dead, and would never touch the welded population — whose signature is zero
MOTION, not low confidence.

## THE ENSEMBLE PREMISE — long-term/short-term peak DISAGREEMENT does not predict a loss either

**2026-09-03, the M-13 prior question for `O-03`, asked before building anything.**
`runs/vot/0903_offline-ltprobe/`, `scripts/rgb_vs_gray_loop.py --lt-probe ETA`, 62 sequences.

The class result above (`N-23`) is that confidence statistics read the CURRENT response map, and
pre-loss this tracker looks confident and MOVING. Disagreement between two models with different
TEMPORAL SUPPORT is the one candidate signal that is not inside a single response map, and it is
the premise of the two-filter ensemble that `baselines.md` ranked on TCLCFcpp's R 0.598.

**The probe is a PURE OBSERVER.** `A_lt`/`B_lt` ride the live trajectory -- same crops, same
shifted training target, same regularizer -- and differ only in `lt_eta`. Nothing feeds back, so
inertness is the control and it is exact: **a probe run is BIT-IDENTICAL to the same arm without
it, maxdiff 0.** Any divergence signal therefore cannot be the probe disturbing the tracker.
Distance is CIRCULAR in bins; a response map wraps, and a naive `|a-b|` would score a 1-bin
disagreement across the wrap as 63.

### The result — per frame, 426 doomed frames against 6,606 healthy

A frame is "doomed" if the run's first loss is within the next 10 frames, "healthy" if the loss
is more than 40 frames away or never.

```
                        median LT divergence          P[healthy < doomed]
                        doomed    healthy              (0.5 = no signal)
frozen  lt_eta=0         3.80      3.61 bins                 0.461
slow    lt_eta=0.005     1.41      1.00 bins                 0.555
```

**The frozen filter is BELOW chance and the slow one is barely above it — and both are weaker
than PSR's 0.618, which is itself closed as too weak to act on.** Note the filters do NOT agree:
the frozen one sits 3.61 bins away even on healthy frames. The disagreement is large and
UNINFORMATIVE, which is a different thing from absent.

### Why this closes the direction and not merely one form of it

An ensemble could in principle work by SELECTING the better filter per frame rather than by
detecting trouble. Selection needs a signal, and the only two available are confidence (closed,
`N-22`/`N-23`) and disagreement (this, 0.555). **With no basis for selection, running two filters
buys nothing but cost** — and the expensive form is a second AIE bank, not a host-only change.

### THE FIRST VERSION OF THIS TEST WAS UNDERPOWERED AND READ THE OTHER WAY

Written down because it nearly became the result. The first analysis compared two 5-frame windows
per run, matching the PSR pre-loss test:

```
                        pre-loss   control   P[ctl < pre]   paired
frozen                     7.21      4.00       0.565       11/27
slow                       5.83      2.83       0.631       16/27
```

**n = 27**, and slow's 0.631 looks like a signal above PSR's 0.618. It is not: the PSR number came
from the BOARD's 419 anchored trajectories (n = 230 losing runs), while this bench is
SINGLE-START, so only 27 of 62 sequences lose late enough to have both windows. Per-run window
pairing then discards almost every frame. The per-frame form uses the same runs, has 7,032
samples instead of 27 pairs, and **inverts the reading**.

## What not to re-derive

* **Neither direction of eta modulation beats a constant eta.** The tempting mechanism story —
  "a confidence dip is a real appearance change, so it needs MORE adaptation, not less" — is
  refuted by the mutant, which adapts more and also loses to the control. The defensible
  statement is narrower: **PSR and APCE do not identify frames at which a different eta would
  help.** Consistent with everything on record: eta 0.1 rejected on hardware, 0.025 much worse,
  and eta arms trading whole sequences (`polo` +0.332 against `rowing` −0.332).
* **47-55 of 62 sequences are EXACTLY TIED in every arm.** The law's footprint is small, because
  the position update consumes an integer argmax bin: eta moves continuously while the
  trajectory only moves when the peak bin flips. A `-ceta` arm changing few frames is expected
  and is not evidence the flag is dead — check `etascale`, not the IoU diff.
* **APCE cannot be pre-screened from board logs.** `track.csv` carries `psr_bolme` and no
  response map, so its separation could only ever be measured in this bench. It has now been
  measured, and it is inert.
* **Match the test to how much data the runs actually give.** A per-run window comparison on a
  single-start bench yields tens of samples; the per-frame form yields thousands from the SAME
  runs. That difference inverted this result (`M-16`).
* The PSR definitions agree between bench and board — both `(peak − sidelobe_mean)/sidelobe_std`
  at `argmax|resp|` with an 11x11 exclusion (`rgb_vs_gray_loop.py` `excl = 5`;
  `mosse_filter.h:170 PSR_EXCL_HALF = 5`). Checked rather than assumed, because CLAUDE.md
  records that two different statistics are both called PSR. So this null is about the law, not
  about a definitional mismatch.

## Reproduce

```bash
scripts/offline_sweep_par.sh runs/vot/0903_offline-ceta/ceta_psr_w12.json 14 \
    rgb-l1relu rgb-l1relu-ceta6 rgb-l1relu-ceta4 rgb-l1relu-cetaneg \
    -- --eta 0.05 --psr-min 5.0 --ceta-stat psr --ceta-warmup 12
scripts/offline_sweep_par.sh runs/vot/0903_offline-ceta/ceta_apce_w12.json 14 \
    rgb-l1relu-ceta6 rgb-l1relu-cetaneg \
    -- --eta 0.05 --psr-min 5.0 --ceta-stat apce --ceta-warmup 12
scripts/offline_sweep_par.sh runs/vot/0903_offline-ceta/ceta_psr_w20.json 14 \
    rgb-l1relu-ceta6 -- --eta 0.05 --psr-min 5.0 --ceta-stat psr --ceta-warmup 20
# --ceta-stat and --ceta-warmup are GLOBAL flags, so each cell is its own file and the
# arms are renamed before scoring (merge_grid.py does not apply -- it requires bare `rgb`).
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/grid_stats.py \
    runs/vot/0903_offline-ceta/ceta_all.json rgb-l1relu ...
```

14 minutes wall on 14 workers, all three sweeps.
