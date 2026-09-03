# Pooled features — AGGREGATION IS REFUTED ON BOTH BANKS.
# The RESOLUTION half is bank-dependent and was NOT.

**Status:** closed · **Updated:** 2026-09-02 · **Scope:** aggregation refuted on both banks; the resolution half, and the linearity explanation WITHDRAWN

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

`docs/thesis/evidence/detector_gain.md` exonerated localisation (alpha = 0.93 on targets that
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
halves — a weaker, more frequently vetoed detector. `metric_ar_vs_iou.md` showed mean IoU and AR
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
- Mean IoU, not AR — and `metric_ar_vs_iou.md` showed those can order two arms oppositely. A
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

# MAX POOLING IS THE SAME NULL AS AVERAGE — and px/bin is confirmed as the only axis

**2026-08-31.** Prompted by Danilowicz & Kryjak 2022 (`docs/papers/danilowicz2022_embedded_dcf.pdf`),
whose deepDCF stem is VGG11 conv1 **including ReLU and 2x2 MAXPOOL** — i.e. the aggregation the
DCF literature actually ships, and one this file had never tested. Every pooling arm above is
an AVERAGE, and averaging a SIGNED edge map cancels lobes (`pool2` PSR 30.4 -> 13.4). That is a
property of the average, not of pooling, so the null above was arguably measured on the wrong
operator. `-mpool<N>` (max) and `-relumpool<N>` (ReLU then max) were added to test it.

62 sequences, RGB, **shipping eta 0.05 / gate 5.0**, `vot_ar_offline.py`. Baseline re-run in the
same invocation and it reproduces the recorded board-form control EXACTLY (A 0.5394 / R 0.2910 /
5792 tracked), so the arms are readable.

```
arm                  A        R   tracked        dR        dA
rgb             0.5394   0.2910      5792                        <- control
rgb-dec2        0.5005   0.3981      7923   +0.1071   -0.0389    subsample, NO aggregation
rgb-mpool2      0.4976   0.3971      7904   +0.1061   -0.0418    MAX
rgb-pool2       0.4977   0.3970      7901   +0.1060   -0.0417    AVERAGE
rgb-relumpool2  0.5393   0.3294      6557   +0.0384   -0.0001    ReLU + MAX
```

**MAX, AVERAGE AND NO AGGREGATION AT ALL AGREE TO 0.001.** The hypothesis was that max on a
rectified map would behave differently from an average on a signed one; the arithmetic is right
(2x2 of `[[-4,1],[2,3]]` gives avg 0.5, max 3.0, dec -4.0) and the tracking consequence is nil.
**The aggregation operator is irrelevant; px/bin is doing all of the work**, which is what the
section above concluded and what this was written to falsify. Danilowicz's maxpool is not why
their pipeline works.

`relumpool2` is the one arm that differs, and not in a useful way: accuracy is pinned at the
baseline (0.5393 vs 0.5394) for a third of the robustness. It tracks 6557 frames against the
other pooled arms' ~7900 — a more conservative tracker scored on an easier prefix, i.e. the
selection effect, not better boxes.

## THE RESOLUTION ARM SURVIVES EVERY STABILITY TEST THE SPATIAL MASK FAILED

The same trim/bootstrap applied to the hardware mask arm (`spatial_mask.md`), per sequence:

```
arm               mean dR   median    trim3  drop-top3            95% CI   P(dR<=0)
rgb-dec2          +0.0826  +0.0000  +0.0648    +0.0503  [+0.0312, +0.1403]    0.000
rgb-mpool2        +0.0842  +0.0000  +0.0662    +0.0521  [+0.0354, +0.1406]    0.000
rgb-pool2         +0.0876  +0.0000  +0.0704    +0.0557  [+0.0343, +0.1461]    0.000
rgb-relumpool2    +0.0623  +0.0000  +0.0499    +0.0291  [+0.0131, +0.1169]    0.005
(shipped spatial mask, k=1, same instrument)
                  +0.0330  +0.0000  +0.0231    +0.0101                        --
```

**The bootstrap CI excludes zero, and it survives dropping the top-3 gainers at +0.050 — five
times the mask's +0.010 and still above this bench's own 0.02 resolution.** It is the first arm
measured here that is not carried by a handful of sequences. The median is still 0.0000, so it
is a tail effect like everything else on this bench; the difference is the size of the tail.

**Read the magnitude with the standing discount.** This instrument over-predicted the spatial
mask 3x (+0.0601 offline against +0.0192 on hardware) and `TARGET_PADDING=3.0` by ~11x. What is
different here is that the pad30 failure had a NAMED cause — the offline loop has no DSST scale
filter and padding sets the ROI `scale_extract` draws from — and **this arm holds padding at
2.0, so that specific blindness does not apply.** It remains single-start against 419 anchored
runs.

**It is a RESOLUTION change, not a pooling change.** Plain `dec2` scores as well as either
pooled variant, so nothing needs to aggregate: the arm is `PATCH_ROWS=PATCH_COLS=64` at
`TARGET_PADDING=2.0`, i.e. px/bin = box/32. That is also exactly the geometry Danilowicz ships
(128x128 ROI -> 64x64 filter) and their 8-channel row scores EAO 0.183 against DSST/KCF HOG at
0.17 — though on VOT2015 with an inverted R, so no value there is comparable to ours.

---

# AGGREGATION ON THE RECTIFIED LAYER-1 BANK — REFUTED, and the LINEARITY EXPLANATION with it

**2026-09-02.** `runs/vot/0902_offline-pool/pool62.json`, 62 sequences / 19,903 frames, shipping
eta 0.05 / gate 5.0, `vot_ar_offline.py` + `scripts/grid_stats.py`. Offline only.

## Why it was re-opened

`settled.md` and `feature_bank.md` refute aggregation with an ARGUMENT as well as a measurement:

> *"A box average of a linear map is another linear map with the same span. It CANNOT do
> anything. That file read it as a fact about aggregation; it is a fact about LINEARITY."*

**`CONV_RELU=1` shipped on 2026-09-02**, so the map this tracker computes is no longer linear and
that argument no longer covers it. The literature all points the same way and all of it is
post-nonlinearity: HOG's deformation tolerance IS the cell; Danelljan takes layer-1 activations
after the ReLU; Danilowicz & Kryjak's embedded stem is conv + ReLU + 2x2 MAXPOOL. This project's
own screen had a consistent hit — `rgb-danrelu` (vgg conv1 3x3 + maxpool + ReLU) beat its linear
twin by **+0.0398**, the pooled arm that beat its control.

## The design: a 2x2, because an ARGUMENT is under test, not an arm

`l1linblur` is the NEGATIVE CONTROL. The linearity argument predicts it reproduces the `blur2`
null (−0.0010/−0.0012); the hypothesis predicts `l1relublur` GAINS. Bank, geometry, channels and
`sigma/target` are identical across all four — only the rectifier and the aggregation move.

```
arm              A        R    tracked
rgb-l1relu    0.5077   0.4364     8686    <- SHIPPING, the control
rgb-l1relublur 0.5154  0.4209     8377
rgb-l1lin     0.5031   0.3910     7783
rgb-l1linblur 0.5465   0.3255     6478
```

Paired per-sequence, against each arm's OWN no-aggregation twin:

| comparison | dR mean | trim-3 | trim-5 | b/w/t | sign p | P(dR<=0) |
|---|---|---|---|---|---|---|
| `l1relublur` vs `l1relu` (rectified) | **−0.0242** | −0.0260 | −0.0269 | 3/7/52 | 0.344 | **0.995** |
| `l1linblur` vs `l1lin` (linear) | **−0.0180** | −0.0198 | −0.0205 | 2/6/54 | 0.289 | 0.971 |

## Verdict: BOTH PREDICTIONS FAILED

**Aggregation LOSES on the rectified map** — −0.0242, stable under both trims, worse on 7
sequences and better on 3, bootstrap 0.995 AGAINST. That is not a null; it is a measured loss
larger than several effects this project has accepted.

**And the negative control fires, which is what makes the result readable.** The linear arm loses
−0.0180, where the linearity argument predicts a NULL. **So that argument was not the operative
mechanism in 2026-08-28 either.** It is arithmetically correct and trackingwise irrelevant — the
same verdict this file already reached for the signed-lobe-cancellation hypothesis, now applied
to its replacement. Blur hurts both maps by a similar amount; the rectifier changes almost
nothing about how aggregation behaves.

**The likely operative mechanism is RESOLUTION, not linearity.** The map is already 64x64 from a
128x128 crop; a further 2x2 average takes the effective resolution below what localisation needs.
That is consistent with the hardware result that at matched `sigma/target` the FINER map wins
(+0.0222 R, `arm_res64.md` sec.25) and with `dec2`/`pool2`/`mpool2` agreeing to 0.001
— aggregation and decimation are the same knob seen twice.

**Accuracy rises in both pairs (0.5077→0.5154, 0.5031→0.5465) and it is the documented SELECTION
effect** — the blur arms track 309 and 1305 fewer frames, so they are scored on an easier prefix.

## What this closes, and what it does not

- **CLOSED: aggregation over THIS map, on any bank, rectified or not.** Three operators (box
  average, max, decimate) x two banks x rectified/linear, all null-to-negative.
- **NOT closed: Danilowicz's stem as a whole.** Their maxpool follows a STRIDE-1 3x3 conv and
  pools down TO the working resolution; ours would pool BELOW it, because the stride-2 conv has
  already decimated. `rgb-danrelu`'s +0.0398 was measured with that 3x3/stride-1 geometry, not on
  top of this bank. The two are not the same operation and this result does not speak to theirs.
