# The robustness gap — where it comes from

**Status:** closed · **Updated:** 2026-08-27 · **Scope:** where the robustness gap comes from: the tracker walks off target confidently

**2026-08-27.** Published CF baselines vs our full-62 run (`~/vot/analysis/full62`), plus an
attribution of our own failures from the existing `track_*.csv` (180,544 frames, 434 runs, no
board time). **The headline reverses the obvious suspect.**

## 1. Where we sit — VOT-STb2022 Table 12, classical end

| tracker | EAO | A | R |
|---|---|---|---|
| TCLCFcpp — CF ensemble, *explicitly embedded/CPU-only* | 0.267 | 0.550 | 0.598 |
| ASMS | 0.255 | 0.526 | 0.599 |
| CSRDCF — HOG/CN + spatial & channel reliability | 0.251 | 0.519 | 0.580 |
| KCF — kernelized DCF + HOG | 0.239 | 0.542 | 0.532 |
| LGT — last of 41 | 0.195 | 0.461 | 0.486 |
| **this, RGB `H_SHIFT=15`** | **0.147** | **0.504** | **0.307** |

A = 0.504 is inside the classical band (0.015 under CSRDCF, 0.038 under KCF, above ANT/LGT).
R = 0.307 is **below all 41**, 0.58x of KCF. Being embedded is no excuse — TCLCFcpp is in
exactly this niche at R = 0.598.

*(The old "0.541, parity with KCF" line was the 57-sequence figure. Only full-62 is quotable.)*

## 2. What the literature prices our structural gaps at

- **Features.** KCF paper, same filter, features only: DCF on raw pixels **0.451** -> DCF on
  HOG **0.728** precision, and it states DCF-on-raw-pixels *is* a MOSSE filter. Danelljan
  ICCVW15: intensity 37.0 / HOG 50.0 / **VGG-M conv1 52.1** mean OP. **That conv1 is 96-dim,
  7x7, PCA'd to 40** — ours is 16 channels of 3x3 int8 with participation ratio 4.94 (gray) /
  7.43 (RGB). "conv1 beats HOG" is a claim about a 40-dimensional conv1.
- **Spatial reliability.** CSR-DCF ablation on VOT2016: uniform box mask **-21% EAO**; uniform
  channel reliability **-12%**; **removing the mask entirely — "reduces the tracker to a
  standard DCF with a large receptive field" — >50%.** That sentence describes this design:
  target is 27% of the ROI at `TARGET_PADDING=2`, nothing masks the rest, and Stage B3
  normalises channels by *energy*, not discriminative power.
- **Features vs regularization, priced in failures** (ICCVW15, VOT2015 failure rate, lower
  better): DeepSRDCF 1.05, SRDCF 1.24, **DeepDCF (conv1, plain DCF) 1.75**, KCF 2.51. Accuracy
  barely separates them; failures separate them 2.4x — and **spatial regularization buys more
  than conv1 features do.** We have the features and not the regularization.

## 3. What our own 62 sequences say

Per-sequence R against groundtruth statistics (lengths in target-size units):

```
frac. frames with displacement > 0.25 target   -0.480     VOT attribute tags:
90th-pct displacement / target size            -0.460       camera_motion +0.138
hold rate (gate vetoes / frames)               -0.486       illum_change  +0.108
median |dlog area| / |dlog aspect|      -0.335 / -0.316     motion_change -0.083
per-sequence accuracy A                        +0.399       size_change   +0.110
```

**VOT's own attributes explain nothing.** Target speed relative to target size, deformation,
and hold rate all land near -0.45. A stays 0.3-0.7 on sequences where R is 0.02: **the boxes
are fine right up until the target is gone.**

### The gate is the AFTERMATH of a loss, not its cause

33,867 vetoes over 180,544 frames (18.8%). **NEGATIVE_PEAK is 29,955 of them — 88% — and
`PSR_GATE_MIN` does not touch it** (it is a structural veto; the knob only disables LOW_PSR,
12%). On veto frames the response has collapsed: median |peak| **4** against **616** on
accepted frames, PSR **-5.2**. And:

```
95.8% of NEGATIVE_PEAK frames occur AFTER the run has already dropped to IoU <= 0.1
 1.4% occur while still on target (IoU > 0.3)
```

So the freeze is what the log looks like once the target is gone. **Relaxing the gate would
have won almost nothing, and that was the change I was about to propose.**

### How it actually loses: confident, gradual drift

The 5 frames before each run's first loss, over 394 runs that lose:

```
gate verdict   ACCEPT 82.0%   NEGATIVE_PEAK 10.3%   LOW_PSR 7.6%
median PSR in that window                18.83   (threshold is 7.00)
median box motion pre-loss          1.88 px/frame  (p90 16.5)
median max |scale_idx| pre-loss              1
```

**It does not freeze into a loss and it does not jump. It walks off the target smoothly,
accepting every frame at PSR ~19.** That is model drift: at `MOSSE_ETA=0.125` the filter has
an 8-frame time constant and re-trains on 73% background every accepted frame, with nothing
that can tell it the content it is locking onto is wrong. It is the "PSR is a weak pass
criterion" trap, now measured at the loss boundary across every run instead of on one frame.

**Everything the baselines have that we lack is anti-drift machinery**: CSR-DCF's spatial
mask (don't learn the background), its channel reliability (weight the discriminative
channels), TCLCF's temporal confidence learning and two-filter ensemble. None of it is about
peak-finding; all of it is about what the filter is allowed to learn.

## 4. Next step

**Arm `eta05`: `MOSSE_ETA=0.05`, RGB, all 62, host-only.** One sweep, no rebuild, no reflash.

Why this one first:
- It attacks the mechanism section 3 just identified — drift rate — and nothing else moves.
- It is the only anti-drift change already measured offline: 8 sequences, frame-weighted mean
  IoU 0.2533 -> 0.2599, **six better, two tied, one worse by 0.0017** (and that one is
  `nature`, which does not move). **Uniform in sign, which `HOLD_COAST` was not** — and
  `HOLD_COAST` is exactly the case where mean IoU and AR disagreed, so this arm must be
  scored on **AR**, with mean IoU reported only as a secondary.
- The sweep is not monotone (0.025 is much worse than 0.05), so it is a shallow optimum: run
  0.05 alone, not a ladder.

**Predictions, written down first.** Halving eta doubles the filter's time constant to ~16
frames, so: the pre-loss ACCEPT share stays ~80% (this change does not touch the gate); first
losses move **later** in the run; and because AR scores loss *timing*, R should move more than
A. If R moves and the pre-loss ACCEPT share also moves, something other than drift changed and
the result is unattributable. Caveat already known: the offline model holds box size fixed, so
eta has never been tested against a live DSST scale filter.

**Then, in order:** (2) a fixed spatial mask on the filter update — the CSR-DCF ablation prices
this highest of anything on the list, and a static box-shaped mask is host-only; (3) replace
channel *energy* normalisation in Stage B3 with a discriminative weight (their -12% row);
(4) a feature bank with pooling / a larger receptive field — offline in `rgb_vs_gray_loop.py`
first, and it is a weights export, not a re-synthesis.

**Not on the list, and now excluded rather than argued:** the fixed-point pipeline. Rails are
uncorrelated with IoU (-0.025), the `H_SHIFT=15` arm rails zero times over 101,564 frames, and
removing quantization makes tracking worse. The 150x amplitude gap between dead and live
frames is not something 1-2 bits of shift can explain. **The frame rate costs no robustness.**

## Sources

- Kristan et al., *The Tenth VOT2022 Challenge Results*, ECCVW 2022, Table 12 —
  https://data.vicos.si/publications/kristan2022the.pdf
- Henriques et al., *High-Speed Tracking with Kernelized Correlation Filters*, TPAMI 2015 —
  https://arxiv.org/abs/1404.7584
- Danelljan et al., *Convolutional Features for CF Based Visual Tracking*, ICCVW 2015 —
  https://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/ConvDCF_ICCV15_VOTworkshop.pdf
- Lukezic et al., *DCF with Channel and Spatial Reliability*, CVPR 2017 —
  https://arxiv.org/abs/1611.08461
