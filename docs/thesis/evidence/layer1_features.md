# Layer-1 conv features — THE NONLINEARITY MECHANISM IS CONFIRMED, THE ARMS ARE NOT

**2026-09-01.** `runs/vot/0901_offline-layer1b/l1b62.json`, 62 sequences / 19,903 frames,
shipping eta 0.05 / gate 5.0, `vot_ar_offline.py`. Offline only. Claim `N-16` in
`docs/thesis/claims.md`.

## Why this screen existed

`feature_bank.md` proves this tracker is Danelljan 2015's **Layer 0**: with `CONV_RELU=0` and 16
channels over a 27-dim tap space the conv is a linear lift the online filter absorbs, and a
one-hot bank with no network in it ties the pretrained one. His Figure 2 puts Layer 0 at ~45% OP
against Layer 1's ~61%, and his sec.3.3 takes activations **after the ReLU**.

Stage 0 (`feature_bank.md`) had refuted pointwise ReLU — but on the SHIPPING 3x3/16 bank, which
has no orientation selectivity to rectify. **The open question was whether the rectifier works
once the bank is Layer-1 shaped**, so every arm here carries its own LINEAR TWIN.

**The control is `rgb-dec2`, not `rgb`.** Stride 2 and 2x2 pooling both halve the map, which
doubles `sigma/target` onto the 1/16 optimum — the free win that turned out to be all of the
res64 result (`proposed_build_res64.md` sec.25). Scoring against the 128x128 baseline would hand
these arms that win a second time.

## The prediction, written down first

**Each `relu` arm must beat ITS OWN LINEAR TWIN.** If it does not, the nonlinearity story is dead
at every kernel size and the direction closes. The headline arm additionally needs
`dR >= +0.02` surviving drop-top-3 against the control.

## The result

```
arm             A        R    tracked   vs dec2     what it is
rgb-dec2   0.5005   0.3981      7923    control     3x3 stride 1, 16ch, 64x64 map
rgb-danlin 0.5018   0.3880      7722    -0.0101     vgg16 conv1 3x3 -> 16ch + 2x2 MAXPOOL
rgb-danrelu 0.4945  0.4278      8514    +0.0297     ...+ ReLU  = Danilowicz & Kryjak's stem
rgb-l1lin16 0.5324  0.3394      6755    -0.0587     resnet18 conv1 7x7/2 -> 16ch (PCA)
rgb-l1relu16 0.5106 0.3993      7948    +0.0012     ...+ ReLU
```

**THE FALSIFIER PASSES, TWICE:**

```
danrelu  - danlin   = +0.0398      (3x3 + maxpool)
l1relu16 - l1lin16  = +0.0599      (7x7 stride 2)
```

**This is the first time in this project that a nonlinearity has beaten its matched linear
control**, and it does so on two different banks at two different kernel sizes, each 2-3x the
bench's 0.02 resolution. The mechanism named in `feature_bank.md` is confirmed: **rectification
needs something to encode, which a 3x3 signed edge map does not provide and an AGGREGATED or
LARGE-KERNEL map does.** Accuracy does not pay for it — on the common survived prefix `danrelu`
is **+0.0059** and `l1relu16` **+0.0017** against the control.

## Did the arms hold? NO — none survives a symmetric trim against the control

```
comparison                  dR mean   trim-3   better/worse/tied
danrelu  vs dec2            +0.0119  -0.0203      16/13/33
l1relu16 vs dec2            +0.0088  -0.0156      18/13/31
danrelu  vs danlin          +0.0190  -0.0155      15/ 8/39
l1relu16 vs l1lin16         +0.0472  +0.0144      18/12/32
```

Only the WITHIN-PAIR 7x7 comparison survives drop-top-3. Against the control both ReLU arms
collapse — **worse stability than the spatial mask** (+0.0330 / +0.0101), which this project
already judged not separable from a null. For scale, `dec2`, the one arm that transferred to
hardware, was +0.1071 pooled AND +0.050 trimmed.

**And the pattern is the CReLU pattern again: the nonlinearity buys back what the alternative
bank costs, and little more.** The 7x7 resnet-PCA bank costs -0.0587 and ReLU returns +0.0599,
netting zero. The vgg 3x3 bank costs -0.0101 and ReLU returns +0.0398, netting +0.0297 pooled
and not trim-stable.

## What Danilowicz & Kryjak 2022 contributes, and what it does not

Their Table 1 (VOT2015, 128x128 ROI / 64x64 filter — exactly this geometry) shows **8 channels
tying 32** (EAO 0.183 vs 0.184), with the collapse only at 4 and 16 anomalously low at 0.174.
That is why this screen ran at 16 channels: **`N_CHANNELS` unchanged means FFT, cmul, IFFT and
the whole APU tail are untouched and only `conv2d` moves**, i.e. the expensive half of a Layer-1
build never has to happen. Their stem is VGG11 conv1 including ReLU AND 2x2 MAXPOOL — a **3x3**
kernel, not Danelljan's 7x7 — which is why `danlin`/`danrelu` exist. vgg11 is not in the local
torch cache; vgg16_bn conv1 is the stand-in and is labelled as one.

What it does not settle: their EAO 0.183-0.184 is VOT2015 against this project's VOT-STb2022, so
the numbers are not comparable, only the ORDERING within their own table is.

## The gap this screen has, and it is the important one

**Every arm here sits at a 64x64 map**, because stride 2 or a maxpool forces it. Hardware
measured the opposite preference the same day: at matched `sigma/target`, **128x128 beats 64x64
by +0.0222 R and +0.0082 EAO** (`proposed_build_res64.md` sec.25). So this screen ran the
nonlinearity at the geometry that loses.

**The untested configuration the evidence now points at is 7x7 STRIDE 1, 16 channels, ReLU, at a
128x128 map with `MOSSE_SIGMA=4`** — the rectifier combined with the geometry that actually won.
It costs 4x the conv of the stride-2 version and leaves the downstream untouched. On hardware
that is `conv2d` only: ~4x the AIE conv against a frame that is 84% CPU-bound.

## Cost, if any of this is ever built

`CONV_RELU` and the kernel size both reach `AIE_FLAGS`, so this is a graph rebuild, re-package,
re-flash and a shift-budget calibration — a rectified map has a large DC that Stage B1/B2 must
absorb, and `conv_weight_layout.h` grows from 64 B/channel to 147 B for a 7x7. `CONV2D_STACK`
needs raising again (27 taps already forced 2048).

---

# THE 32-CHANNEL BATCH — the falsifier passes ROBUSTLY, the arm still ties the control

**2026-09-01, later.** `runs/vot/0901_offline-layer1/l1a62.json`, same 62 sequences, same
protocol, arms at 32 channels plus the analytic diagnostic.

```
arm               A        R    tracked   vs dec2    what it is
rgb          0.5394   0.2910      5792              128x128 reference (WRONG sigma/target, see above)
rgb-dec2     0.5005   0.3981      7923    control   3x3 stride 1, 16ch, 64x64 map
rgb-l1lin    0.5031   0.3910      7783    -0.0071   resnet18 7x7/2 -> 32ch (PCA)
rgb-l1relu   0.5077   0.4364      8686    +0.0383   ...+ ReLU   <- BEST ARM IN EITHER BATCH
rgb-gablin   0.4890   0.3518      7002    -0.0463   analytic Gabor 7x7/2, 32ch  [DIAGNOSTIC]
rgb-gabrelu  0.5275   0.3292      6552    -0.0689   ...+ ReLU                   [DIAGNOSTIC]
rgb-gabrelublur 0.5229 0.3232     6433    -0.0749   ...+ 2x2 stride-1 average   [DIAGNOSTIC]
```

## 1. The falsifier passes, and at 32 channels it passes the TRIM as well

```
comparison                    dR mean   trim-3   trim-5   b/w/tied
l1relu(32) vs l1lin(32)       +0.0575  +0.0200  +0.0064    15/ 8/39
l1relu(16) vs l1lin(16)       +0.0472  +0.0144  -0.0027    18/12/32
danrelu    vs danlin          +0.0190  -0.0155  -0.0273    15/ 8/39
```

**`l1relu(32)` vs its own linear twin is the ONLY comparison in this entire screen that survives
a drop-top-FIVE.** Three learned-bank pairs, three positive pooled results, and the strongest one
is trim-stable. The rectifier is doing real work on a learned Layer-1 bank; that is settled.

## 2. AND THE ANALYTIC BANK GOES THE OTHER WAY — rectification is not about "oriented"

```
gabrelu vs gablin             +0.0246  -0.0116  -0.0265    20/12/30      pooled -0.0226
```

**The Gabor bank LOSES when rectified**, pooled R 0.3518 -> 0.3292, where both learned banks gain.
So the rectifier's value does NOT come from the filters merely being oriented — an analytic
quadrature bank is as oriented as it gets. It comes from the bank being LEARNED. That retires the
"HOG by convolution" reading of this direction, and it retires it on the one arm that was outside
this project's conv-feature requirement anyway, so nothing of value is lost.

## 3. Channels: pooled says 32 > 16, the trim says they are not separable

```
l1relu 32ch vs 16ch           +0.0339  -0.0029   16/ 8/38      pooled R 0.4364 vs 0.3993
l1lin  32ch vs 16ch           +0.0236  -0.0024   16/ 8/38      pooled R 0.3910 vs 0.3394
```

**This does NOT reproduce Danilowicz & Kryjak's "8 channels ties 32"** on pooled R, and the trim
does not separate them either way. Their axis is not ours: they TRAIN/quantise at N channels,
while these arms keep N of 64 PCA directions of a donor, so "16 channels" here means "half the
donor's principal directions discarded". Both readings are consistent with the trim result, and
neither is worth a board arm on this evidence.

## 4. But against the CONTROL the best arm is exactly a null

```
l1relu(32) vs dec2            +0.0427  -0.0001  -0.0104    19/ 6/37
```

Pooled +0.0383, per-sequence mean +0.0427, **trim-3 -0.0001**. Accuracy is fine — common survived
prefix **+0.0072** — so this is not an A/R trade, it is simply not separable from the control once
three sequences are removed. For scale, `dec2` itself was +0.1071 pooled AND +0.050 trimmed
against ITS control, and it transferred to hardware at 43%.

## Verdict

**Mechanism: CONFIRMED.** ReLU on a learned Layer-1 bank beats its matched linear control, on
three pairs, surviving drop-top-3 twice and drop-top-5 once. `feature_bank.md`'s scope note is
now measured rather than argued: pointwise rectification is refuted on the SHIPPING 3x3/16 bank
and works on a Layer-1 one.

**Deliverable: NOT YET.** No arm beats the geometry-matched control after a symmetric trim. On
this project's own standard — the one that rejected the spatial mask and accepted `dec2` — that
is a null, and it does not justify a rebuild, a reflash and a shift-budget calibration.

**Frame time is NOT the obstacle, which is worth recording separately.** At 7x7 stride 2 with 32
channels the conv is ~19.9 ms of AIE against a ~15.1 ms host tail (model in sec.25 of
`proposed_build_res64.md`, validated on two measured points), so the frame lands ~20-22 ms
against today's 24.55 ms — **no frame-rate cost, possibly a small gain**. It is the
STRIDE-1/128x128 variant that costs 2x (conv ~39.8 ms against a 24.5 ms host).

## What would change the verdict

Every arm here ran at a 64x64 map, which hardware measured as the WORSE geometry (-0.0222 R at
matched sigma/target). The untested cell is **a learned Layer-1 bank with ReLU at a 128x128 map**
-- 7x7 stride 1 (2x frame time) or 5x5 stride 1 (frame-rate neutral, conv ~21.4 ms). If the
rectifier's +0.02 trim-stable gain over its linear twin SURVIVES at the better geometry, the arm
would be starting from `sigma4`'s 0.1931 rather than `dec2`'s baseline, and that is the only
version of this direction with a route to a board result.

---

# 5x5 STRIDE 1 AT THE 128x128 MAP — the better geometry did NOT strengthen the case

**2026-09-01, last.** `runs/vot/0901_offline-l5/l562.json`, 62 sequences, **`--sigma 4`**, so
`sigma/target` stays at the 1/16 optimum on a 128x128 map. **Control: `rgb` at sigma 4
reproduces `rgb-s4` from the sigma sweep EXACTLY (0.4965 / 0.3718 / 7399 tracked)** — this is the
offline twin of the best hardware arm, not `dec2`.

```
arm             A        R    tracked   vs control
rgb        0.4965   0.3718      7399    control   128x128 map, sigma 4
rgb-l5lin  0.5047   0.3545      7055    -0.0173   5x5 stride 1, 32ch, linear
rgb-l5relu 0.5087   0.4316      8590    +0.0598   ...+ ReLU  <- largest pooled gain of any L1 arm
rgb-l5lin16 0.5419  0.2830      5633    -0.0888
rgb-l5relu16 0.5413 0.3431      6829    -0.0287

comparison                    mean   trim-3   trim-5   +/-/tie   sign p   P(dR<=0)
l5relu(32) vs control       +0.0441  -0.0010  -0.0106  18/ 9/35   0.122     0.060
l5relu vs its linear twin   +0.0655  +0.0242  +0.0060  19/12/31   0.281     0.024
l5lin(32) vs control        -0.0213  -0.0524  -0.0578  14/10/38   0.541     0.768
l5relu(16) vs control       +0.0243  -0.0198  -0.0337  19/11/32   0.201     0.229
```

**The nonlinearity is confirmed a FOURTH time** — against its own linear twin it survives
drop-top-FIVE (+0.0060) at P(dR<=0)=0.024. Common-prefix accuracy +0.0076, so no A/R trade.

**But the pooled gain grew while the per-sequence statistics got WEAKER** (P(dR<=0) 0.060 here
against 0.041 at 64x64; sign p 0.122 against 0.015). The same borderline picture at both
geometries: consistent direction, magnitude carried by a few sequences, bootstrap sitting on the
0.05 line. `dec2` was accepted at **P(dR<=0)=0.000**.

## THE COST FINDING THAT DECIDES THE CONFIGURATION

Frame model of sec.25 (`proposed_build_res64.md`), validated on two measured points, plus
conv2d = `S + t*taps` with S = 2.31 ms, t = 0.255 ms/tap from `rgb.md`'s 4.60 (9 taps) / 9.19
(27 taps):

```
config                          host frame   AIE conv   frame     R vs its control
sigma4 today: 128^2, 16ch, 3x3    24.55 ms     9.2 ms   24.55 ms  (the control)
l5relu(32):   128^2, 32ch, 5x5    43.3  ms    42.8 ms   ~45   ms  +0.0598  1.8x SLOWER
l5relu16:     128^2, 16ch, 5x5    24.55 ms    21.4 ms   ~25   ms  -0.0287  a null
l1relu(32):    64^2, 32ch, 7x7/2  15.1  ms    19.9 ms   ~20-22 ms +0.0383  FASTER than today
```

**At 128x128 it is the 32 CHANNELS, not the kernel, that double the downstream.** The working
configuration is not affordable and the affordable one is a null. **The one cell with both
properties is `l1relu(32)` at the 64x64 map** — see `proposed_build_l1relu.md`.

## Verdict on the whole Layer-1 direction

**Mechanism: CONFIRMED, four times, on two geometries, two kernel sizes, two learned banks.**
ReLU on a learned Layer-1 bank beats its matched linear control; on the analytic Gabor bank it
LOSES, so the property that matters is that the bank is LEARNED, not that it is oriented.

**Tracking: BORDERLINE and now tested at both ends.** No arm reaches the acceptance level
`dec2`/`sigma4` cleared against their own controls. Expected hardware transfer at the geometry
arms' 43-84%: +0.016 to +0.032 R, i.e. EAO roughly +0.008 to +0.016 — above the +0.005 bar if it
transfers, and it may land at zero.

**The argument that does NOT depend on any of this:** `feature_bank.md` proves the shipping
configuration makes the CNN redundant — a one-hot bank with no network in it ties the pretrained
one. A build whose conv layer is provably a change of basis is hard to describe as a CNN-feature
tracker, and this project's requirement is conv features. **That is a thesis-integrity reason to
adopt Layer 1 and it is independent of the tracking delta — but it must be reported as that, not
dressed as a robustness win.**
