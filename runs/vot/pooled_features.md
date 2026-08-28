# Pooled features — AGGREGATION IS REFUTED ON BOTH BANKS.
# The RESOLUTION half is bank-dependent and was NOT.

**2026-08-28.** Offline, `scripts/rgb_vs_gray_loop.py`, 8 stb2022 sequences, 2841 frames,
14 arms across both feature banks. No hardware.

**Two separate answers, and running only the gray bank got the second one wrong.**

1. **AGGREGATION — the actual hypothesis — is a null on BOTH banks.** A 2x2 box average at
   stride 1, which changes the receptive field and nothing else, is -0.0010 on gray and
   -0.0012 on RGB. Rectifying first does not rescue it on either.
2. **HALVING THE FEATURE-MAP RESOLUTION is bank-dependent, and the sign FLIPS**: gray
   -0.0143, RGB **+0.0062** (and +0.0198 at `MOSSE_ETA=0.05`). The first sweep was gray-only
   and read as "pooled features lose"; that was a statement about the gray bank, quoted as a
   statement about pooling. See "The RGB arms" below — the RGB aggregate is one sequence
   deep and is NOT yet a result.

## Why it was worth running

`runs/vot/detector_gain.md` exonerated localisation (alpha = 0.93 on targets that
translate) and left the feature bank as the ranking suspect: 16 channels of 3x3 conv at
stride 1 is a 3-pixel receptive field on a ~73 px target, participation ratio 4.94 (gray) /
7.43 (RGB). The literature prices this highest of anything cheap — KCF raw pixels -> HOG
0.451 -> 0.728 precision, Danelljan intensity 37.0 / HOG 50.0 / conv1 52.1 mean OP — and
HOG's deformation tolerance comes from aggregating over cells. On hardware a 2x2 average at
the end of conv2d would leave conv2d's cost unchanged and make FFT, cmul, IFFT and the
whole APU filter tail **4x cheaper**, so a win would have been nearly free.

## The arms, and what each one isolates

```
                                                              frame-wtd  unweighted
gray            shipping 128x128, no aggregation  (CONTROL)      0.2533      0.1382
gray-blur2      2x2 box average, STRIDE 1, map stays 128x128     0.2523      0.1380
gray-blur3      3x3 box average, stride 1                        0.2449      0.1315
gray-relublur2  ReLU then 2x2 stride-1 average                   0.2438      0.1314
gray-relu       ReLU only, no aggregation                        0.2449      0.1314
gray-pool2      2x2 AVERAGE pool -> 64x64                        0.2420      0.1329
gray-relupool2  ReLU then 2x2 average pool -> 64x64              0.2384      0.1315
gray-dec2       2x2 SUBSAMPLE -> 64x64  (no aggregation)         0.2390      0.1295
gray-pool4      4x4 average pool -> 32x32                        0.2309      0.1255
gray-relupool4  ReLU then 4x4 average pool -> 32x32              0.1795      0.1002
```

Decomposed against the baseline:

```
resolution alone            gray  -> dec2    -0.0143
pooling at that resolution  dec2  -> pool2   +0.0030
aggregation, resolution held gray -> blur2   -0.0010     <- THE HYPOTHESIS, ISOLATED
rectification alone         gray  -> relu    -0.0084
```

**Aggregation is worth zero and the resolution it costs is worth -0.014.** Pooling helps
slightly *relative to* the subsampling it implies (+0.0030, and it is visible in PSR: pool2
30.4 -> 13.4 against dec2's 22.3, so averaging a SIGNED edge map over a cell cancels the
lobes — rectifying first recovers most of it, relupool2 PSR 29.4). But the whole effect is
an order of magnitude smaller than the resolution it has to pay for, and at constant
resolution it is a null.

`blur2` was the arm that mattered and it did not exist in the first sweep. Without it the
result reads "pooling loses", which is true and unattributable — `pool` and `dec` both move
the receptive field AND the bin size, sigma in frame pixels, and the sub-bin quantisation.

## THE POSITIVE CONTROL, which is what makes the null mean anything

A bench that shows nothing is indistinguishable from a bench that cannot show anything.
`MOSSE_ETA = 0.05` is the one change already known to move this exact bench up, so it was
re-run through the same code path:

```
                    gray      gray-blur2   gray-pool2
eta = 0.125       0.2533        0.2523       0.2420
eta = 0.05        0.2599        0.2584       0.2318
```

**gray at eta 0.05 returns 0.2599 frame-weighted and 0.1480 unweighted — digit-for-digit
the numbers in CLAUDE.md, with `nature` the lone loser at -0.0017, also exact.** The bench
reproduces a known +0.0066 and reports -0.0010 for aggregation. And the verdict is not
eta-dependent: blur2 stays neutral and pool2 stays clearly worse at the slower learning
rate.

The baseline arm also reproduces the recorded per-sequence table (`car1` 0.7131, `tiger`
0.1696, `nature` 0.1121) unchanged, so the pooling plumbing did not disturb the shipped path.

## The RGB arms — where the gray-only conclusion broke

The shipping arm is RGB (participation ratio 7.43 against gray's 4.94) and it wins AR on
hardware. Gray was chosen first because the documented positive control is a gray number,
which is an argument about instrumentation and not about transferability. It does not
transfer:

```
eta = 0.125          rgb   rgb-blur2   rgb-pool2   rgb-dec2
frame-wtd         0.2544      0.2532      0.2599     0.2606
eta = 0.05
frame-wtd         0.2577      0.2601      0.2712     0.2775
unweighted        0.1427      0.1457      0.1618     0.1654
mean PSR            48.1        46.9        19.0       18.7
holds                115         132        1172       1165
```

`rgb` reproduces its recorded 0.2544 exactly, so the baseline is intact on this bank too.

**Read `pool2` against `dec2`, not against `rgb`: pooling is <= subsampling on both rows.**
So even here the win is not aggregation — it is the coarser grid, and the averaging on top
of it contributes nothing or slightly less than nothing. The aggregation verdict is
unchanged by the RGB data; only the resolution verdict moved.

**And the resolution win is ONE SEQUENCE.** At eta 0.05, `rgb-dec2` beats `rgb` on 4 of 8
and loses on 4. `tiger` alone goes 0.1784 -> 0.3855 (+0.2071 over 365 frames = 75.6
IoU-frames) against a total aggregate delta of 56.3 IoU-frames — **without `tiger` the
change is negative**, and `nature` is -0.0204. That is exactly the trap this project has
written down twice: `nature` is 35% of these frames, `tiger` another 13%, and both are
documented pathologies that must not be tuned against.

The `tiger` gain is at least NOT the known freeze artifact — it is the reverse. `rgb-dec2`
takes the frozen-detector rate from 68.6% to **1.9%** while IoU rises 0.1784 -> 0.3855 and
centre error falls 77.4 -> 39.9 px. `tiger.md` had already unfrozen that detector two other
ways (`SIGMA=1`, `EPS_REL=0.1`), and both times tracking got WORSE, so this is a different
mechanism and it is **unexplained**. A plausible reading is that at 2x the bin size the ~11
px (~8 bin) drift of the annotated box against the appearance becomes ~4 bins and stops
being chased — but that is post hoc, and PSR collapsing 45.0 -> 11.5 with holds going 1 ->
68 says the response is much weaker, not better conditioned.

### The 62-sequence run — direction real, magnitude NOT

`rgb` vs `rgb-dec2`, eta 0.05, all 62 sequences, 19,903 frames:

```
                        rgb    rgb-dec2
frame-weighted IoU   0.1841      0.2169   +0.0328
unweighted IoU       0.1620      0.1709   +0.0089
median per-seq delta                      +0.0023
per sequence            --   39 better / 23 worse   (sign test p = 0.028)
survived fraction     0.281       0.312   (mean; median 0.114 -> 0.154)
holds per frame       0.135       0.306   2.3x
mean PSR               27.9        16.7   halved
```

**It is NOT the pathologies:** dropping `tiger` and `nature` leaves +0.0322 frame-weighted.
**It IS three sequences:** dropping the top-3 gainers (`girl` +0.574, `bolt1` +0.552,
`rowing` +0.261) turns it into **-0.0239 frame-weighted / -0.0142 unweighted**. The median
sequence gains +0.0023, i.e. nothing.

So the honest statement is: **more sequences improve than not (39/23, p = 0.028), and the
size of the aggregate is carried entirely by three of 62.** A direction without a magnitude.

**And mean IoU is the wrong metric for this arm specifically.** Holds go up 2.3x and PSR
halves — a weaker, more frequently vetoed detector. `evidence_ar.md` showed mean IoU and AR
ordering `HOLD_COAST` OPPOSITELY on identical trajectories, and the mechanism there was
exactly loss TIMING, which mean IoU cannot see. The survived-fraction row above is the
nearest free proxy and it agrees (0.281 -> 0.312), but it is not AR: no anchors, no
re-initialisation, no 10-consecutive-frame rule.

### Scored with VOT's RULE — and the verdict changes

`scripts/vot_ar_offline.py` applies the toolkit's failure rule (threshold 0.1, grace 10,
burn-in 10) to these trajectories. **This is not the toolkit's AR** — these are single-start
runs, not the anchored multi-start protocol — but it is the metric's SHAPE, and mean IoU
demonstrably cannot see loss timing.

```
arm         accuracy  robustness   tracked / 19903
rgb           0.5519      0.2850      5673
rgb-dec2      0.5433      0.3417      6800      +19.9% frames survived
                -0.0086     +0.0567
```

**dR = +0.0567 against a mean-IoU median of +0.0023.** That gap IS the point: the coarser
arm survives longer, which mean IoU averages away and the failure rule does not. A falls
slightly, which is the shape a robustness change is supposed to have and the same shape the
`eta05` arm had on hardware.

**The instrument's resolution is 0.02 and it is measured, not assumed.** Validated against
the board's own multi-start CSVs: it reproduced `eta125 -> eta05` (+0.0159 against the
toolkit's +0.0218) and got `eta05 -> g5p0` BACKWARDS (-0.0021 against +0.0134). +0.0567
clears that by 3x; the +0.0328 mean-IoU figure would not have.

**Per sequence it is still a coin flip: R better on 19, worse on 19, tied on 24**, with
enormous swings both ways (`bolt1` +0.969, `girl` +0.927, `rowing` +0.634 against `shaking`
-0.962, `conduction1` -0.788, `basketball` -0.557). That is very nearly the profile `eta05`
had on hardware (28 better / 24 worse / 10 tied, pooled +0.0218) — and `eta05` shipped on it.

**THE OBVIOUS CONFOUND IS RULED OUT.** `rgb-dec2` holds 2.3x more, and a held box on a
static target trivially avoids the failure rule — the freeze-is-protective artifact
`tiger.md` warns about. If that were the mechanism, dR would rise with the hold-rate
increase. It does the opposite: **corr(dR, d hold-rate) = -0.198**, and the R-GAIN sequences
add less holding (median +0.083) than the R-LOSS ones (+0.184). The survival gain is not
bought by freezing.

**Status: worth a hardware arm.** It is NOT host-only — a 64x64 feature map changes the
FFT/IFFT geometry, so it is a graph rebuild, re-package and re-flash — but it also makes
FFT, cmul, IFFT and the whole APU filter tail **4x cheaper**, so the same run measures a
robustness change and a large frame-rate change at once. Score it on `vot analysis`, not on
this proxy and not on mean IoU.

**Write the falsifier down first.** dR here is +0.0567 with A -0.0086 and +19.9% frames
tracked. If the board returns a dR under +0.02, this proxy is not usable for feature-geometry
arms and the result is a subset artifact. If A falls by more than ~0.02, the coarser grid is
costing box quality rather than buying survival, and the trade is not obviously worth it.

## Read the per-sequence table, not the aggregate

```
sequence      gray   blur2   pool2   relupool2
car1        0.7131  0.7129  0.7119     0.7107
tiger       0.1696  0.1715  0.1303     0.1119
nature      0.1121  0.1086  0.0929     0.0885
crabs1      0.0188  0.0188  0.0241     0.0244
book        0.0187  0.0187  0.0198     0.0198
soccer2     0.0141  0.0141  0.0250     0.0373
animal      0.0227  0.0227  0.0231     0.0231
ball3       0.0366  0.0366  0.0366     0.0366
```

`pool2` is "better" on 4 of 8 — but every one of those is a sequence sitting at IoU 0.02-0.04,
i.e. already lost, where the difference is between two ways of being wrong. Its real losses
are `tiger` and `nature`, which are 48% of the frames and are the two sequences CLAUDE.md
says never to tune against. **Neither direction of that table is evidence.** The frame-weighted
row is the summary, and the isolated `blur2` arm is the verdict.

## What this does and does not refute

**Refuted:** that AGGREGATING this feature bank buys deformation tolerance. Box-averaging
its responses, with or without rectification, at stride 1 or with decimation, at 2x2 or
4x4, does not improve tracking on this bench.

**Not refuted, and now the ranking candidate:** that the BANK ITSELF is the constraint.
HOG's tolerance is not pooling alone — it is orientation binning of gradient MAGNITUDES
into 31 dimensions, of which pooling is the second half. What was tested here is the first
bank aggregated, not a different bank. Participation ratio 4.94 (gray) / 7.43 (RGB) against
HOG's 31, and Danelljan's conv1 result is for a 96-channel 7x7 layer PCA'd to 40. **The next
experiment is a REPLACEMENT bank, not an aggregation of this one** — and it is still a
weights export plus this same offline loop, no re-synthesis.

Also untouched: the spatial-reliability item (`robustness_gap.md` #2), which constrains what
the filter learns rather than what the features see, and is a different mechanism entirely.

## Caveats on the bench itself

- Box size is held fixed (no DSST). Applies identically to every arm.
- Mean IoU, not AR — and `evidence_ar.md` showed those can order two arms oppositely. A
  +0.0066-scale effect would need AR to be believed; a null this flat would not survive
  becoming a win under a different metric, but that is an argument, not a measurement.
- `nature` is 35% of the frames and `tiger` another 13%.

## Reproduce

```bash
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
    --sequence car1 --arms gray gray-blur2 gray-pool2 gray-dec2
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
    --sequence tiger --arms gray gray-blur2 --eta 0.05      # the positive control
```

Arm suffixes: `-pool<N>` average pool, `-dec<N>` subsample (the pooling control),
`-blur<N>` stride-1 box average (resolution held), `-relu`, `-relupool<N>`, `-relublur<N>`.
`--sigma` holds the target sigma in frame pixels instead of bins; `--eta` sets MOSSE_ETA.

---

# The axis is px/bin — and padding 2.0 is right AFTER ALL

**2026-08-28, later.** Ten configurations, RGB, eta 0.05, gate 5.0, all 62 sequences, scored
with `vot_ar_offline.py`. `px/bin` = frame pixels per response bin = `padding * pool / 128`,
written below in units of the target box.

```
config           px/bin        A        R   tracked   holds/fr
pad 1.25 @128   box/102   0.4870   0.2047      4074      0.118
pad  1.5 @128    box/85   0.5390   0.2251      4481      0.082
pad 1.75 @128    box/73   0.5354   0.2461      4898      0.053
pad  1.0 @ 64    box/64   0.4758   0.2605      5185      0.208
pad  2.0 @128    box/64   0.5394   0.2910      5792      0.052   <- SHIPPING geometry
pad  2.5 @128    box/51   0.5081   0.3400      6768      0.066
pad  3.0 @128    box/43   0.4957   0.3789      7541      0.059
pad  2.0 @ 64    box/32   0.5005   0.3981      7923      0.094   <- BEST
pad  4.0 @128    box/32   0.4870   0.3677      7318      0.044
pad  3.0 @ 64    box/21   0.5390   0.3698      7360      0.105
```

## Two findings, and the second corrects the first

**1. R is governed mostly by px/bin, and it has an optimum near box/32.** Monotone from
box/102 up to box/32, then it turns over at box/21. That single number explains why `blur2`
was a null (it aggregates without changing px/bin), why `pool2`/`dec2` helped, and why the
padding sweep looked monotone.

**2. px/bin is NOT a sufficient statistic, and padding 2.0 wins at BOTH matched pairs.**
The test was two pairs constructed to have identical px/bin by different routes:

```
box/64:  pad 2.0 @128  R=0.2910   vs   pad 1.0 @ 64  R=0.2605   dR -0.0305
box/32:  pad 4.0 @128  R=0.3677   vs   pad 2.0 @ 64  R=0.3981   dR +0.0304
```

Both gaps are ~0.03, above the proxy's 0.02 resolution, and they point in OPPOSITE
directions — so the two routes are not interchangeable. **In both pairs the arm at padding
2.0 wins.** Padding and sampling scale are separable, and 2.0 is the right padding once
px/bin is controlled for.

**So the earlier "more padding is monotonically better" was a confound, and so was the
CSR-DCF reasoning that motivated the sweep.** Raising padding at a fixed 128x128 map raises
px/bin, and px/bin was doing the work. The shipping `TARGET_PADDING=2.0` is vindicated — by
an experiment run to overturn it, and for a reason unrelated to the static-scene holdout that
originally set it. The "1.5 verdict is REOPENED" note in CLAUDE.md can be closed: 1.5 is
worse (R 0.2251 vs 0.2910), and so is every value below 2.0.

**And the spatial-mask premise is weakened.** The sweep was proposed as the cheap stand-in for
CSR-DCF's mask: less padding, less background learned, better robustness. Every value below
2.0 is worse. Background contamination is not this tracker's binding constraint, and
`robustness_gap.md` ranking the mask second was reading a ">50% EAO" ablation — which prices
REMOVING a mask from a tracker built around one — as if it priced ADDING a crude one here.

## Where that leaves the config

Two candidates, both at `MOSSE_ETA=0.05`, `PSR_GATE_MIN=5.0`, RGB, `H_SHIFT=15`:

| | padding | map | offline A | offline R | cost |
|---|---|---|---|---|---|
| shipping-equivalent | 2.0 | 128x128 | 0.5394 | 0.2910 | — |
| **(a) host-only** | **3.0** | 128x128 | 0.4957 | 0.3789 | host ELF only |
| **(b) reflash** | **2.0** | **64x64** | 0.5005 | 0.3981 | graph rebuild + reflash |

**(a) captures 87% of (b)'s robustness gain with no reflash**, and the gap between them
(0.019) is inside the proxy's resolution. ROI geometry is runtime AXI-Lite, so (a) is a host
ELF change. (b) additionally makes FFT/cmul/IFFT and the APU filter tail 4x cheaper.

**Both cost accuracy — about -0.04 A** — and that is past the falsifier written for the
`dec2` arm. Part is selection (30% more surviving frames are scored, and they are harder
frames — the same argument that applies to RGB), but that cannot be claimed after the fact.
EAO is what prices an A/R trade and this proxy does not compute it.

**Recommended order: run (a) first.** It is one host build, it tests the px/bin mechanism on
the metric of record, and it costs no silicon. If the board reproduces a dR of roughly +0.09,
the mechanism is confirmed and (b) becomes worth its reflash — with the frame-rate win as a
bonus rather than the justification. If (a) returns under +0.02, this proxy does not transfer
to geometry arms and nothing here should be shipped.

**Caveat on padding 3.0 specifically:** CLAUDE.md's settled entry records that 3.0 clips 3.57%
of samples and trips the aliasing detector (bilinear has no prefilter). That was measured on
the synthetic holdout, and it is a real risk on hardware that the offline model reproduces
only partially. `pad 2.5 @128` (R 0.3400) is the conservative version of the same move.

---

# HARDWARE VERDICT: config (a) REJECTED. The falsifier fired.

**2026-08-28, `runs/vot/0828_1451-pad30`.** 62 sequences, 419 trajectories, 180,125 frames,
host-only rebuild (`TARGET_PADDING` 2.0 -> 3.0, one flag, `aie.flagstamp` byte-identical,
board `a.xclbin` matched at `52235f49221e`). Workspace `~/vot/analysis/pad30ab`.

```
arm                        A         R       EAO    frames
g5p0   (padding 2.0)  0.5100    0.3417    0.1629     61831
pad30  (padding 3.0)  0.5030    0.3494    0.1570     59784
                     -0.0070   +0.0077   -0.0059     -2047
```

**Predicted dR +0.088. Measured +0.0077** — an order of magnitude short, and below the
+0.02 floor written down before the run. **EAO, which is the metric of record and the arbiter
for an A/R trade, went DOWN.** Frames tracked went down by 2047. Per-sequence R is better on
34 / worse on 26 / tied on 2, i.e. the usual coin flip with no pooled gain behind it.

By the stated criterion: **the offline proxy does not transfer to geometry arms, and neither
this nor the 64x64 reflash should proceed on its strength.** The reflash is off the table
until something else motivates it.

## Why the proxy failed, as far as can be said

The likeliest cause is a known and stated gap rather than a new one: **the offline loop has no
DSST scale filter** (box size held fixed, `SCALE_N=1` equivalent). Padding sets the ROI that
`scale_extract` draws its template from, so every scale-axis consequence of tripling the ROI
was invisible offline by construction. Second candidate: offline runs are SINGLE-START from
frame 0, where a larger search region has many frames to earn back a drift; the board runs 419
short anchored runs, where it has far less room to.

Both were written down as caveats before the run. Neither was quantified, and that is the
lesson: **a caveat that is not priced is not a caveat, it is a hope.**

## Two things the run got right that the prediction did not

- **Padding 3.0 costs NO frame time.** Predicted `roi_crop launch` would rise with a 3x ROI;
  measured 1.477 -> 1.473 ms, and the frame is unchanged. `roi_crop` resamples the ROI to a
  FIXED 128x128 patch, so its cost is set by OUTPUT pixels and the four bilinear taps each,
  not by input extent. A bigger ROI is free. Worth remembering before pricing any future
  geometry change.
- **The gate got LOOSER, not tighter**: hold rate 15.51% -> 10.94%, `NEGATIVE_PEAK` 27772 ->
  19586. More context makes the response better-formed, which is the mechanism the offline
  model did capture — it simply is not worth much once the scale filter is live.

## A tool bug this exposed

`scripts/vot_detector_gain.py` hardcodes the bin->pixel conversion as `box * 2 / 128`, i.e. it
assumes `TARGET_PADDING=2.0`. Pointed at the pad30 CSVs it reports alpha 0.442 against g5p0's
0.686, which looks like a collapse and is an artifact: corrected by the real 3.0 the value is
~0.66, essentially unchanged. **Any per-bin instrument in this repo carries the padding
assumption until shown otherwise** — the same class of defect as the multi-start CSV keyed on
frame index. Not fixed here; recorded so the next reader does not trust that column.
