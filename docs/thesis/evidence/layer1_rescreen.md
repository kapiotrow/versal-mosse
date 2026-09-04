# Refuted R-boosters re-screened on the Layer-1 bank — M-14 run in reverse

**Status:** current · **Updated:** 2026-09-04 · **Scope:** which pre-2026-09-02 refutations survive the Layer-1 operating point, and the denominator-conditioning measurement that decides two of them

**2026-09-04.** No board time. `scripts/rgb_vs_gray_loop.py` on the shipping
`rgb-l1relu` bank against the `rgb` (mobilenet 3x3) bank it was refuted on.
Claims: `R-17`, `R-18`.

## Why this pass exists

`M-14` says a prior POSITIVE screen expires when the operating point moves. **The
converse holds identically and had never been applied systematically.** Six things
moved on 2026-09-02 — bank (mobilenet 3x3 -> resnet18 7x7/2 PCA), channels
(16 -> 32), map (128 -> 64), `CONV_RELU` (0 -> 1), `ACC_BOUND`, budget
(4-4-4 -> 3-3-3) — and this repo already had **two proofs that a refutation is
bank-scoped**: ReLU (refuted on mobilenet 3x3, ships on Layer-1) and the spatial
mask (+0.0601 on the old bank, **-0.0127** here — the SIGN inverted, `N-21`).

Already re-screened on Layer-1 and NOT revisited here: aggregation (a loss,
-0.0242), the spatial mask (`N-21`), Stage B3 channel reliability, the
confidence-modulated eta and two-filter ensemble (`N-22`/`N-24`/`O-03`),
`MOSSE_ETA=0.1` (hardware).

## The prediction, written down first

Written before the measurement, and **it was wrong in sign** — recorded because
the wrong prediction is what makes the result worth trusting.

Bolme 3.3 presents regularization and perturbations as ALTERNATIVE cures for one
defect, low-energy denominator bins, and his Figure 3 — the whole empirical case
for perturbations — is captioned *"Results shown without regularization"*. The
2026-08-28 refutation argued this design already bought that stability via
`eps_rel=1e-3` and a shared denominator over **16 channels**, evidenced by
"bins below `1e-6*mean(B)` are 0.00%".

**Predicted:** a 7x7 stride-2 RECTIFIED map is heavily low-pass and DC-heavy, so
`B`'s high-frequency bins would be relatively SMALLER, conditioning would be
WORSE, and Bolme's regime would reopen. **Falsifier:** if the low-energy-bin
fraction does not rise, perturbations are refuted a fortiori and no arm runs.

## The result

Median over `car1`/`tiger`/`nature`, 120 frames each, `B` normalised by its own mean:

| bank | min B/mean | p1 | frac < 1e-6·mean | frac < 1e-3·mean |
|---|---|---|---|---|
| mobilenet 3x3, 16ch — **the bank it was refuted on** | 2.96e-05 | 7.87e-05 | 0.000% | **15.80%** |
| resnet18 7x7/2 rectified, 32ch — **SHIPPING** | **5.13e-03** | 6.98e-03 | 0.000% | **0.000%** |

**The denominator floor is 173x HIGHER on the shipping bank and the fraction of
bins below `1e-3*mean(B)` falls from 15.8% to exactly zero.** Bolme's defect is
not merely cured here, it is ABSENT.

The predicted mechanism was backwards: 32 rectified channels each contribute
NON-NEGATIVE energy at every bin, so summing `|F|^2` over twice as many channels
FILLS the spectrum in rather than thinning it. The rectifier and the channel
count push the same way and both moved favourably.

## Did the mechanism hold?

- **Perturbations (Bolme 3.4) — REFUTED A FORTIORI. FIRED against my prediction.**
  The refutation is STRONGER at the new operating point, and better evidenced: the
  old entry cleared a `1e-6` threshold, the new bank clears one **1000x tighter**.
  No arm was run and none is needed.
- **COROLLARY, and it is worth more than the arm was: the 16% of losing runs that
  fail within 10 frames of init are NOT a conditioning problem.** They survive with
  `B` essentially perfectly conditioned (median IoU 0.571 / PSR 7.35 one frame after
  `filter_init()` against 0.915 / 36.73). So no regularizer and no perturbation
  scheme addresses them, and that whole family of cures is closed. The question
  re-points at the init box and target distinctiveness.
- **`eps_rel` — EXPIRED, and this measurement is the reason.** `eps = 1e-3*mean(B)`
  while the SMALLEST bin is `5.13e-3*mean(B)`: the regularizer is now five times
  below the worst bin it exists to protect, so it moves that bin by ~20% and a
  typical bin by 0.1%. The sweep that fixed `eps_rel=1e-3` (ratio 12.90 / 13.96 /
  **16.15** / 10.62 / 4.01 at 1e-5..1e-1) ran on a bank where **15.8% of bins sat
  below that value** — a different regime entirely.
  **RE-SCREENED, and the mechanism prediction held while the hoped-for gain did not.** 12
  sequences, 250 frames, paired against `1e-3`:

  | eps_rel | mean dIoU | median | trim-3 | b/w |
  |---|---|---|---|---|
  | 1e-5 | +0.0003 | 0.0000 | **0.0000** | 1/0 |
  | 1e-4 | +0.0003 | 0.0000 | **0.0000** | 1/0 |
  | 3e-3 | -0.0000 | 0.0000 | -0.0000 | 2/1 |
  | 1e-2 | +0.0014 | 0.0000 | -0.0001 | 4/2 |
  | 3e-2 | **+0.0112** | 0.0000 | **-0.0009** | 4/4 |
  | 1e-1 | +0.0060 | +0.0024 | **-0.0090** | 8/4 |

  **Inert across four decades** — that is the conditioning measurement confirmed on TRACKING and
  not merely on a spectrum statistic, which is the stronger form of the claim. Above the inert
  band nothing survives a trim. `eps_rel=1e-3` STANDS, but the original entry's peak (16.15 at
  1e-3 against 4.01 at 1e-1) does not exist on this bank: the curve is flat, so the optimality
  claim is vacuous rather than wrong. **`eps` is inert because `B` is well conditioned, not
  because it is unnecessary** — it remains the safety net for a bank that is not.

## Padding — the refutation was CONFOUNDED, and correcting it strengthens the verdict

The target spans `map / padding` BINS, so `sigma/target = sigma * padding / map`.
`R-11` puts the optimum at 1/16, and the shipping arm is padding 2.0 / sigma 2.0
= 1/16. **The 2026-08-28 hardware `pad30` arm ran padding 3.0 at sigma 2.0 =
1/10.7 — a mainlobe 1.5x too wide — and nobody re-tuned sigma.** That arm moved
two magnitudes at once, the exact defect that re-attributed the 64x64 arm (`R-11`)
and gutted the resolution term (`R-14`).

Re-screened with sigma matched (8 sequences, 250 frames, mean IoU, paired):

| arm | mean dIoU | median | b/w | d frames-to-loss |
|---|---|---|---|---|
| pad 2.0 / sigma 2.00 — SHIPPING (1/16) | — | — | — | — |
| pad 3.0 / sigma 2.00 — **what hardware ran** (1/10.7) | -0.1182 | -0.1650 | 2/6 | -25.0 |
| pad 3.0 / sigma 1.33 — **matched** (1/16) | **-0.1946** | -0.2148 | 1/7 | -66.8 |
| pad 2.5 / sigma 1.60 — matched (1/16) | -0.0567 | -0.0774 | 3/5 | -20.2 |

**`TARGET_PADDING=2.0` stands, now for a better reason**: removing the confound
makes padding 3.0 WORSE, not better. The unmatched arm reproduced the known
hardware direction, so the screen is known to be able to detect what it looked for.

## `R-18` — R-11's invariant CHALLENGED AND UPHELD

**CORRECTED 2026-09-04, SAME DAY.** The first reading of this grid concluded the invariant was
under-determined and that `sigma/map` won. **That was taken before the sigma range was extended,
with the padding-3.0 column still rising at the grid edge — an argmax read off an unbracketed
curve.** The corrected result is below. The original error is recorded because it is the same
mistake in a new costume: a conclusion drawn from a measurement that had not yet bounded the
thing it was measuring.

The two candidate invariants are

    sigma/target = sigma * padding / map        sigma/map = sigma / map

and `R-11` was established from 64x64/sigma2 against 128x128/sigma4 **with padding held at 2.0 in
both**, where the two ratios move together. Every width arm since held padding constant, so the
padding sweep is the first arm that separates them at all.

Mean IoU, 12 sequences, 250 frames, all three columns now BRACKETED:

| padding | `sigma/target` predicts | `sigma/map` predicts | **observed argmax** |
|---|---|---|---|
| 1.5 | **2.667** | 2.000 | **2.667** |
| 2.0 | 2.000 | 2.000 | **2.000** (degenerate — both agree) |
| 3.0 | 1.333 | 2.000 | **3.500** |

**`sigma/target` matches two of three; `sigma/map` matches only the degenerate column. `R-11`
STANDS.** The lone exception is padding 3.0 — the padding `settled.md` independently records as
**tripping the aliasing detector** (bilinear has no prefilter; 3.0 also clips 3.57% of samples) —
and it deviates in exactly the direction that defect predicts: noisier, aliased features favour a
WIDER training target. So the exception has a cause unrelated to the width law, and **padding is
not a clean lever for testing width above 2.0.**

**Independently corroborated on Danilowicz's tracker** (`evidence/deepdcf_reproduction.md`): their
`sigma` also has no geometry term, and width-controlling their 224-vs-128 comparison made the
geometry effect **LARGER** (+0.0060 -> +0.0149 EAO), not smaller. Two trackers, two datasets, same
conclusion — the width confound was masking part of a real geometry effect rather than creating it.

**CONSEQUENCE FOR THE ROADMAP:** `R-14` downgraded the 128x128 Layer-1 arm because its only
remaining support was Danilowicz's +0.024 EAO "with no per-sequence data to trim". That support is
now stronger, not weaker — it survives a width control on their own tracker.

**The control that makes the grid readable:** padding 2.0 declines hard at high sigma
(0.4324 at sigma 2 -> 0.2178 at sigma 8), reproducing the prior INDEPENDENT 22-cell sweep that
found sigma 8 the worst cell. That is the only thing in this grid capable of catching a wiring
error in the `padding`/`sigma` knobs added the same day. Sigma 8 returns 0.2178 at ALL THREE
paddings — the degenerate limit, where the mainlobe is too wide to discriminate and padding stops
mattering.

## Controls

- **The banks differ ONLY in the bank.** Both arms run the same geometry, the same
  Stage A, the same 64x64 map and the same loop; `rgb` selects the mobilenet 3x3
  weights and `rgb-l1relu` the resnet18 7x7/2 PCA ones through `l1_banks`.
- **The padding screen carries a positive control**: the unmatched arm must
  reproduce the hardware direction (a loss), and does.
- **The prediction was registered and FAILED.** A conditioning measurement that
  confirmed my guess would be much weaker evidence than one that reversed it.

## What not to re-derive

- **A refutation is scoped to its bank until re-screened.** Three instances now:
  ReLU (inverted), the spatial mask (inverted), and this pass (perturbations
  strengthened). The direction is not predictable — screen, do not argue.
- **Do not read "low-pass features" as "worse-conditioned denominator".** It is
  the CHANNEL COUNT and the rectifier's non-negativity that set `B`'s floor, and
  they dominate the spectral shape.
- **`vot_ar_offline.py`'s resolution is ~0.02 in R and the 2026-08-28 perturbation
  falsifier was +0.02** — i.e. the original screen's bar sat at its instrument's
  own resolution. Any future perturbation claim needs the multistart path.
- Not covered here: the `hq` filter-quantization null and the `R-06` sample-memory
  result from the same day. Different topics, recorded with their own claims.
