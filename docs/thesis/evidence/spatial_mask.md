# The spatial mask: EAO +0.0110 and the accuracy "loss" is a selection effect — but the gain is carried by 3 sequences of 62 and is NOT distinguishable from a null

**Status:** closed · **Updated:** 2026-08-31 · **Scope:** the spatial mask on hardware: EAO +0.0110, not separable from a null, mechanism refuted

**2026-09-03 — THIS IS A RESULT ABOUT THE OLD BANK.** Re-screened offline on the SHIPPING
Layer-1 arm (7x7/2, 32ch, 64x64) the mask's sign INVERTS: dR +0.0601 there against −0.0127
paired here, trim-3 −0.0358, P(dR<=0)=0.706. The `e_box` mechanism separates on both banks.
Do not carry any number below onto the shipping configuration — `evidence/mask_bank_transfer.md`,
claim `N-21`.

**2026-08-31.** `runs/vot/0831_1528-mask` against `runs/vot/0831_1340-base_stat`, 62 sequences /
419 trajectories per arm, workspace `~/vot/analysis/0831_mask`. The two arms differ in ONE
`-D`: `app.flagstamp` diffs to a single line, `-DFILTER_MASK=0` against `=1`, and the
disassembly shows `filter_mask_project` called from 4 sites in the masked ELF and 0 in the
baseline. Same `a.xclbin` (`52235f49221e`) on the card for both — host-only, an scp, no reflash.
Claim id: `O-01` in `docs/thesis/claims.md`, result row `R-10`. Pre-registration:
`evidence/arm_mask.md`.

## The prediction, written down first

From `arm_mask.md` sec.4, before any board time:

- **Accept only on `vot analysis`, and only if EAO rises. `dEAO >= +0.005` to ship.** Below that
  it is inside the noise of a 419-run comparison. EAO is the arbiter for an A/R trade — that is
  the pad30 lesson, where R rose +0.0077 and EAO FELL.
- **`dR > 0` with `dA < -0.02` is the failure-rule artifact, not a win**, demonstrated by the
  `gsign` mutant at dR +0.0525 / dA -0.0975. Require A not to fall, or price the fall on the
  COMMON survived prefix.
- **Mechanism:** `mask_ebox` should sit near 0.91 at init against a baseline starting near
  0.52-0.55 and climbing. **If EAO moves while `mask_ebox` does not separate the arms, the gain
  is not the mask and the result is unattributable.**
- Offline, in board form: dR +0.0601, dA -0.0284 pooled / -0.0103 on the common prefix.

## The result

| arm | A | R | EAO | tracked frames |
|---|---|---|---|---|
| `eta05_g5p0` (stored shipping) | 0.5100 | 0.3417 | 0.1629 | 61831 |
| `base_stat` (control, `FILTER_MASK=0`) | 0.5100 | 0.3417 | 0.1629 | 61831 |
| **`mask` (`FILTER_MASK=1`)** | 0.4913 | **0.3608** | **0.1740** | **72075** |
| **delta (mask - baseline)** | -0.0187 | **+0.0192** | **+0.0110** | **+16.6%** |

**dEAO = +0.0110, twice the +0.005 bar: the mask ships.** R rises 0.3417 -> 0.3608 and 16.6%
more frames survive. A falls 0.0187 pooled — inside the -0.02 artifact threshold, so the
`gsign` test does not fire, but see below: scored properly the accuracy does not fall at all.

## Did the mechanism hold?

Two different questions, and they have different answers. **Does the projection do what it
claims? HELD, decisively.** **Does that explain the tracking gain? NO — see the next section.**

`mask_ebox` (`FILTER_MASK_STAT=1`, carried by BOTH arms — the flag is independent of
`FILTER_MASK` for exactly this reason):

| | at init | f1 | f5 | f20 | f40 |
|---|---|---|---|---|---|
| baseline | 0.6049 | 0.6016 | 0.7024 | 0.8174 | 0.8626 |
| mask | 0.9500 | 0.9495 | 0.9622 | 0.9760 | 0.9838 |
| **delta** | **+0.3451** | +0.3479 | +0.2598 | +0.1586 | +0.1212 |

The distributions do not overlap — the mask arm's p25 (0.932) clears the baseline's p75
(0.665) — and on the 8859 frames valid in BOTH arms the gap is +0.2122 against a pooled
+0.2128, so it is not a selection effect either. **The EAO gain is attributable to the mask.**

**The accuracy falsifier: the drop is a SELECTION EFFECT, and scoring it properly REVERSES the
sign.** A is the mean overlap over TRACKED frames, so an arm surviving 16.6% longer is averaged
over a harder set. On the frames both arms survived (identical 52452-frame set):

```
                              baseline      mask     delta
A, own survived frames          0.5113    0.4957   -0.0156   (62477 vs 71910 frames)
A, COMMON survived prefix       0.5319    0.5499   +0.0179   (52452 frames, identical set)
```

**On the same frames the masked filter is MORE accurate, not less.** The pooled drop is entirely
the extra hard frames it reached. This is better than predicted: the pre-registration expected a
small genuine cost (-0.0103 on the prefix) and there is none.

## Why R improves is NOT explained — and two tempting explanations are refuted

`mask_ebox` proves the projection does what it claims (it concentrates filter energy). It says
nothing about WHY tracking improves, and the obvious downstream stories do not survive.

**Refuted: "the mask cuts `NEGATIVE_PEAK` vetoes."** It looks compelling in aggregate — 15.42%
of evaluated frames -> 10.22%, and `PSR_GATE_MIN` cannot disable that veto, so it seemed like
the lever the gate never had. Split by whether the tracker was ON TARGET that frame, it inverts:

| arm | on-target frames | NEG% | lost frames | NEG% |
|---|---|---|---|---|
| baseline | 78846 | **3.07** | 101279 | 25.03 |
| mask | 86875 | **3.87** | 93250 | 16.13 |

**On target the mask produces MORE anti-correlated peaks, not fewer** (+26% relative), and in
the 5 frames before the first loss — where a causal story would have to live — it is also worse
(13.05% -> 15.06%). Direct standardisation says the aggregate fall is 90% a RATE change
(-4.66 pp) rather than composition (-0.55 pp), but that rate change sits almost entirely inside
frames the run had ALREADY lost. Same conclusion as `robustness_gap.md`: `NEGATIVE_PEAK` is the
aftermath of a loss, not its cause. **Do not build on it.**

**Refuted: "the mask prevents losses."** Losing runs barely move — 373 -> 369 of 419, a net
rescue of 4 (15 runs saved, 11 newly lost). Nor are losses consistently delayed: of the 358 runs
that lose in both arms the mask loses LATER on 129, EARLIER on 109, and at the same frame on
120. **The median time-to-first-loss does not move at all (54 -> 54 frames).**

**What is actually true is a TAIL effect.** Mean time-to-first-loss rises 119.4 -> 141.4 frames
(+18%) while the median is unchanged, so the +16.6% of tracked frames and the EAO gain are
carried by a minority of runs that survive substantially longer, not by a broad improvement.
That is the same shape as `eta05.md` — per-sequence a coin flip, the pooled figure
frame-weighted — and it is why EAO, which is dominated by how long runs survive, moves while the
median run is untouched. **The arm ships on EAO; the explanation for it is open.**

## HOW MUCH OF THIS IS REAL — the gain is carried by 3 sequences of 62

The pre-registered falsifier was `dEAO >= +0.005` and the arm cleared it at +0.0110. **The
falsifier did not include a stability test, and this project's own standard elsewhere does**
("survives a symmetric trim" is what distinguished the offline mask arm from eleven others the
same day). Applied here, on the per-sequence robustness from `vot analysis`:

```
per-sequence dR   mean +0.0087   median +0.0000   better 29 / worse 26 / tied 7
  symmetric trim (drop 3 each end)      +0.0059
  drop the top-3 GAINERS only           -0.0030     <- the sign FLIPS
  bootstrap over sequences (20k)        95% CI [-0.0135, +0.0318],  P(dR <= 0) = 0.22
```

**The top 3 sequences of 62 carry 133% of the total gain** — `surfing` (+0.271),
`flamingo1` (+0.243), `iceskater2` (+0.199) — so the other 59 net NEGATIVE. Largest losers are
comparable in size (`conduction1` -0.182, `handball1` -0.171, `shaking` -0.154). The pooled
dR of +0.0192 is frame-weighted and is 2.2x the per-sequence mean, the same gap `eta05.md`
records.

**EAO cannot be bootstrapped the same way** — the toolkit reports one EAO per tracker, not per
sequence — so the arbiter itself has no stability estimate here. Given R's CI includes zero,
**the honest reading is that this arm is not distinguishable from a null on 62 sequences**, and
the +0.0110 EAO should be quoted with that caveat rather than as a clean win.

## Why the CSR-DCF mechanism cannot transfer to this design

Three structural differences, in decreasing order of how much they explain:

1. **THIS IS A PROJECTION, NOT A CONSTRAINT — and that is the big one.** CSR-DCF solves a
   CONSTRAINED optimisation: the filter adapts so that the best filter *supported inside the
   mask* is found. Here `A` and `B` are untouched, `H = A/(B+eps)` is re-formed unmasked every
   frame and only then projected, so `h_masked = m (.) h_opt` is a TRUNCATION of the
   unconstrained optimum, not the constrained one. The energy the mask removes carried signal
   that a constrained solve would have redistributed INSIDE the mask; here it is simply
   discarded, and next frame's `A`/`B` have not learned anything from the projection. The mask
   therefore cannot compound — it is re-applied from scratch to an unmasked model forever.
2. **CSR-DCF's mask is ESTIMATED PER FRAME from colour segmentation; ours is FIXED.** Their own
   ablation prices a uniform box mask at **-21% EAO relative to their estimated one**, so the
   literature already says a fixed mask captures only part of the value. Ours is fixed AND
   smooth (a Hann), i.e. even less shape-adapted than the box they priced.
3. **THERE WAS LESS TO FIX THAN THE LITERATURE ASSUMES, BECAUSE THE FILTER SELF-CONCENTRATES.**
   Unmasked `mask_ebox` climbs 0.6049 -> 0.8626 over a run's first 40 frames with no help at
   all, so the mask's marginal contribution SHRINKS from +0.345 at init to +0.121 by f40. The
   design's `TARGET_PADDING=2.0` (settled three ways) already keeps the context small.

**The sharpest single piece of evidence that the concentration mechanism does not operate:** the
mask's `mask_ebox` advantage is LARGEST at init (+0.345), and init failures do not move —
**61 -> 60 runs lost within 10 frames of init**. If concentration were the mechanism, that is
the bucket that should have fallen most. The whole distribution barely shifts (−1 / −5 / −6 /
+8 across the <=10 / 11-50 / 51-200 / >200 buckets).

**And nothing about a run predicts whether the mask helps it.** Correlations of the mask's gain
in frames-to-loss against baseline properties: median PSR +0.003, hold rate +0.007, target area
-0.017, box aspect +0.015, median IoU +0.026, median |displacement| -0.106, median
`resp00/peak` +0.128 — and against baseline `mask_ebox` itself, **+0.028 with the WRONG SIGN**
(CSR-DCF predicts negative: most help where most background). A filter-shaping change with no
predictor is what you get when filter shaping is not the binding constraint.

**So the answer to "is something else wrong with the design" is the one already in CLAUDE.md,
now with a fourth independent line of support: the deficit is UPSTREAM OF THE FILTER.**
Quantization is exonerated (removing it makes tracking worse), localisation is exonerated
(alpha 0.93 on targets that translate), padding is rejected on hardware, and now filter support
is a null too. What is left is the feature bank — 16 channels of 3x3 conv1, no pooling, against
HOG's 31 pooled dims. **No amount of shaping a filter fixes the features it is built from.**

## Controls

- **The baseline is the shipping arm, proven rather than assumed: all 419 `base_stat`
  trajectories are BYTE-IDENTICAL to the stored `eta05_g5p0` arm**, and its A/R/EAO reproduce
  the published 0.5100 / 0.3417 / 0.1629 to four decimals. That proves three things at once —
  `FILTER_MASK=0` is inert on hardware, the `FILTER_MASK_STAT` instrument perturbs tracking not
  at all, and the clean-HEAD rebuild reproduces the DIRTY-tree ELF the shipping arm was run
  with (the open provenance question in `arm_mask.md` sec.3, now closed).
- **The instrument itself is cross-checked and mutation-tested.**
  `scripts/check_ebox_crosscheck.py` runs the board's `filter_box_energy_fraction()` and the
  offline `box_energy_fraction()` on the SAME H over 4 geometries (including a rectangular
  64x32, which catches a row/column confusion). They agree to 6 decimals, and 5 injected
  mutants — no inverse transform, box at the origin, column pass skipped, forward instead of
  inverse, box extent off by one — are each caught on every case.
- **The board's `mask_ebox` reproduces the offline prediction on an unmasked arm.** car1 reads
  0.5306 / 0.5879 / 0.6981 at init / f5 / f20 against the offline 0.5140 / 0.5832 / 0.6920.
  Two independent implementations on different hardware, agreeing to ~0.01-0.02.
- **The analysis is run over three arms at once**, so the stored shipping arm is a control
  INSIDE the same `vot analysis` invocation rather than a number copied from a previous one.

## What not to re-derive

- **THE INSTRUMENT SHIPPED BROKEN THE FIRST TIME, AND IT READ 0.0000 ON EVERY FRAME OF A WHOLE
  SWEEP.** `filter_box_energy_fraction()` squared `H` directly, but `Sum|h|^2` is a SPATIAL
  quantity and `h_scratch` holds the FREQUENCY-domain H. A centred box in the frequency domain
  holds the high frequencies — under 5e-5 for this filter. The offline half transformed first
  and its docstring said so; the two were described as "written against the same definition"
  and were not. **Five unit tests passed throughout**, because they fed the function SPATIAL
  arrays — self-consistent with a function that does no transform, and agreeing with the caller
  about nothing. Only a cross-implementation check can see this class of defect. The tests now
  build H by forward-transforming the pattern each case is about.
- **The statistic costs an inverse FFT per channel: 9.4 ms/call on x86 -O3, 30.1 ms on the
  A72** — more than a whole 26 ms frame. It is therefore SAMPLED (`FILTER_MASK_STAT_WARM=5`,
  `FILTER_MASK_STAT_EVERY=20`), and unsampled frames log -1 exactly as a held frame does. **A
  reader that averages the -1 rows as zeros measures the SCHEDULE, not the filter.** At the
  shipped schedule 94.6% of rows are -1 and that number is NOT a hold rate.
- **The offline bench over-predicted the gain by 3x: dR +0.0601 offline against +0.0192 on
  hardware.** The direction transferred, the magnitude did not — consistent with
  `vot_ar_offline.py`'s measured ~0.02 resolution in R and with the pad30 lesson that
  single-start offline does not map onto 419 anchored runs. Do not quote the offline delta.
- **`mask_ebox` came in ABOVE the offline prediction: 0.9500 at init against 0.913.** Not
  explained. The board applies the exact periodic Hann from `hanning_128.h` while the offline
  bench builds its mask by resampling; that is the obvious suspect and it was not chased,
  because the arms separate either way.
- **7 runs appear only in the mask arm's `mask_ebox` table.** These are baseline runs whose gate
  vetoed every SAMPLED frame, so `publish` never ran and H was never re-formed; under the mask
  they publish. `vot_mask_stat.py` names such runs rather than shrinking its run count, because
  419 reported as 412 hides exactly this.
- **THE GATE IS NOW STALE, and this arm inherited it.** Masking moves the PSR scale (offline:
  car1 mean PSR 48.2 -> 36.7) and `PSR_GATE_MIN`'s worth is conditional on that scale. This arm
  ran at the inherited 5.0, one variable at a time. The re-tune is a second host-only sweep.
- **Do not kill a board sweep mid-sequence.** Doing so leaves the free-running AIE graph and the
  XRT device context inconsistent, and the NEXT process stalls: both threads in
  `clock_nanosleep`, 33% of one core, CU reading `ap_done=1 ap_idle=1`, zero frames produced for
  14 minutes on a sequence that takes 14 SECONDS. Killing the leftover does not clear it — the
  sweep's own leftover-kill ran and the fresh process stalled anyway. A reboot does.
  Two things hid the stall for those 14 minutes and neither is a fault: the board's stdout is
  block-buffered at 4 KB, and `CSV_FLUSH_EVERY=200` holds the rows. **Watch the trajectory
  count, which is the only signal that separates a healthy long sequence from a hung one.**

## The mask WIDTH knob exists, and k=1 is already past its optimum

**2026-08-31.** CLAUDE.md and `arm_mask.md` both state the window is forced so
"there is no width knob". True of a SINGLE Hann, false of the family: `sin^2` raised to `k` is a
cosine polynomial of degree `k`, so **`m^k` has exactly `2k+1` non-zero DFT bins per axis** and
is still board-implementable, and multiplying by `m^k` is EXACTLY applying `filter_mask_project`
`k` times — verified against an exact FFT to **4.94e-16**. So `k` costs `8k` complex adds per
bin, still no multiplies, and NO NEW BOARD CODE. Mask energy in the centred 64x64 box:
k=1 0.6695, k=2 0.8544, k=3 0.9347.

Swept offline at k=2 over all 62 at shipping eta/gate (`--mask-power 2`), baseline reproducing
the board-form control exactly:

```
arm                    A        R  tracked      dR        dA
rgb               0.5394   0.2910     5792                       <- control
k=1 (shipped)     0.5109   0.3512               +0.0601   -0.0284
k=2 (m^2)         0.4930   0.3485     6937      +0.0575   -0.0464

per sequence      mean dR   median    trim3  drop-top3
k=1               +0.0330  +0.0000  +0.0231    +0.0101
k=2               +0.0275  +0.0000  +0.0192    +0.0022
k=2 minus k=1     -0.0056  +0.0000     better on 13 / worse on 20
```

**k=2 is worse than k=1 on every statistic and costs more accuracy.** The knob is real and k=1
is already at or past the optimum, which is what the refuted concentration mechanism predicts:
if the gain is not concentration, more concentration cannot help. **The `m^k` family is closed
at k=1.**
