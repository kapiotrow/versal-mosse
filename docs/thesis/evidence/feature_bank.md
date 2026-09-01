# The feature bank — how much is the PRETRAINING actually worth?

**2026-08-29.** Prompted by a design question: would swapping the network the conv1 layer comes
from help, or taking a wider network and PCA-compressing it to 16 channels? Both are cheap on
this hardware (`make weights` + a 1 KB file copy, no re-synthesis), so the question is whether
they are worth doing at all.

**Answer: probably not, and the reason is measurable rather than arguable — at layer 1 the
specific weights barely matter.** Two independent measurements below, plus a third that retires
the statistic the case for "compress to fewer, better features" would have rested on.

## 1. PARTICIPATION RATIO IS NOT A QUALITY METRIC. IT IS MAXIMISED BY NOISE

CLAUDE.md cites the bank's participation ratio (4.94 gray / 7.43 RGB) against HOG's 31 dims, and
"PCA to get fewer, better-conditioned features" is an attempt to raise it. The control:

```
                                              rank      PR
random ORTHONORMAL 16x27                        16   16.00   <- theoretical maximum
random Gaussian 16x27 (no training at all)      16   10.69
shipping RGB bank                               16    7.43
random Gaussian 16x9 (grayscale ambient)         9    5.65
shipping GRAY bank                               9    4.94
```

**Random untrained weights beat the trained bank, and an orthonormal basis hits the maximum by
construction.** Any objective of the form "raise the participation ratio" is therefore satisfied
perfectly by noise, and §2/§3 show noise does not track better.

On real activations rather than weights (20 `car1` ROI patches through the real int8 conv):

```
                                        act PR   act rank   weight PR
shipping (mobilenet_v3_small conv1)       1.43        3.6        7.43
random Gaussian, matched row norms        1.99        5.0       10.69
```

Same ordering, and a second finding: **the activation-space effective width is 1.43, not 7.43.**
That, not the weight-space number, is what should be compared with HOG's 31 — the headline
overstates the bank's usable width by about 5x.

## 2. HELD-OUT PSR: RANDOM MATCHES PRETRAINED

Held-out Bolme PSR is the instrument that sold RGB (gray 12.97 -> RGB 21.18, 1.63x). Same
instrument, same RGB path, everything identical except the 16x3x3x3 weights, which are replaced
by Gaussian taps of MATCHED PER-CHANNEL ROW NORMS so the int8 grid, `out_shift` and `bias_acc`
see a comparable scale:

```
                          dt=1     dt=5    dt=10        (PSR mean)
car1    pretrained       36.09    13.52    12.63
        RANDOM           37.25    14.45    13.40
tiger   pretrained       31.22     7.44     5.26
        RANDOM           30.99     6.72     4.97
nature  pretrained       83.07    26.43    16.20
        RANDOM           75.51    24.01    15.07
bolt1   pretrained       19.29     8.10    16.19
        RANDOM           18.89     7.73    16.25
```

Four sequences, no systematic gap, and localisation error is a tie. This is consistent with the
known result that first-layer filters are close to a generic oriented-edge/colour-blob basis —
which also explains why every 3x3x3 stem in torchvision looks alike:

```
stem (3x3x3, folded BN)          n_out   rank      PR
mobilenet_v3_small (SHIPPING)       16     16    7.43
mobilenet_v2                        32     26    7.60
efficientnet_b0                     32     26    6.87
regnet_y_400mf                      32     25    9.50
shufflenet_v2_x1_0                  24     22    7.61
vgg16_bn                            64     24    5.17
```

The WIDEST stem is the worst conditioned. There is no obviously better donor.

## 3. CLOSED-LOOP AR — the check this project actually decides on

`rgb_vs_gray_loop.py` arm suffix `-rand<seed>`, 62 sequences, 19,903 frames, shipping
eta/gate, scored with `vot_ar_offline.py`. TWO seeds, because one random draw is one sample.
Trajectories: `runs/vot/0829_offline-bank/rand62.json`.

**Control first:** the pretrained arm reproduces the stored board-form baseline on all 62
sequences with maxdiff 0, so the only thing that moved is the weights.

```
arm                   A        R  tracked   meanIoU        dA        dR
rgb (pretrained)   0.5394   0.2910     5792    0.1792
rgb-rand0          0.5651   0.2801     5574    0.1899   +0.0258   -0.0110
rgb-rand1          0.5572   0.2763     5499    0.1781   +0.0178   -0.0147

SEED SPREAD (rand0 vs rand1):                                -0.0079   -0.0038
```

**Pretraining is worth about 0.011-0.015 in R — BELOW this instrument's measured ~0.02
resolution.** The sign is consistent across both seeds and the seed-to-seed spread (0.0038) is
3-4x smaller than the gap, so the ordering is probably real; the MAGNITUDE is not resolvable.

Two things sharpen it:

* **The pooled dR is carried by a handful of sequences and does not survive a symmetric trim** —
  it FLIPS POSITIVE, +0.0090 (rand0) and +0.0069 (rand1). `flamingo1` alone is -0.538/-0.534.
  Same pattern this file's neighbours keep finding.
* **The random arms' higher accuracy is the selection effect, not better boxes.** They track
  3.8-5.1% fewer frames, so they are scored on an easier prefix. On the frames BOTH arms
  survived, random is still marginally ahead (+0.0075, +0.0060) — i.e. random is not producing
  worse boxes, it just survives slightly less long.

## What this does and does not establish

**Establishes:** within the family of reasonable 16x3x3x3 banks, *which* filters you pick barely
matters. So swapping the donor network is very unlikely to reach the +0.02 in R this project
requires of an arm, and "wider network + PCA to 16" is worse than that — its motivating
statistic is maximised by noise. Note the PCA MECHANISM is sound and free (a linear map of conv
outputs composes with the conv, so a 64->16 projection folds into the weights and stays a 3x3
conv); it is the OBJECTIVE that does not survive contact with the control.

**Does not establish:** that no feature change can help. What is untouched here is the
GEOMETRY — 16 channels, 3x3, stride 1, no pooling. Danelljan's "conv1 beats HOG" is a 96-dim
7x7 conv1 PCA'd to 40, and HOG's deformation tolerance is the CELL. Neither is reachable by a
weights export; both are `N_CHANNELS` / kernel-size / stride changes, i.e. an AIE rebuild.
Aggregation over THIS map is separately refuted (`pooled_features.md`: `blur2` -0.0010 gray /
-0.0012 RGB).

**AND A METHODOLOGICAL RESULT WORTH MORE THAN THE ARM.** `vot_ar_offline` cannot separate a
pretrained bank from a random one at 62 sequences. **A bench that cannot tell trained from noise
cannot rank two trained banks either**, so no future feature-bank comparison should be decided
on this instrument unless the predicted effect is far larger than 0.02 in R. That is a real
limit on the offline loop, discovered by running a control rather than by reasoning about it.

## Reproduce

```bash
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
    --arms rgb rgb-rand0 rgb-rand1 --sequence <seq> --eta 0.05 --psr-min 5.0 --json out.json
python3 scripts/vot_ar_offline.py out.json rgb rgb-rand0
```

`-rand<N>` is a SEED. The bank is Gaussian with per-channel row norms matched to the pretrained
one — without that match the arm also moves `out_shift` and `bias_acc`, and a loss would be
unattributable between "the weights are random" and "the fixed-point scale moved".

---

# WHY random matches pretrained: THE BANK IS A LINEAR LIFT AND ONLY ITS SPAN MATTERS

**2026-09-01.** §2/§3 above measured a null and read it as "at layer 1 the specific weights
barely matter", which is the known literature result. There is a sharper reason, it is specific
to THIS datapath, and it makes predictions that were then tested.

## The derivation

`CONV_RELU=0` ships. `conv2d_kernel.cpp` computes `out = saturate_int16(acc >> out_shift)` and
then applies the separable Hann window — **there is no nonlinearity in the feature extractor at
all**, only saturation and a shift. So with `u_k` (k = 1..27) the raw lifted patch planes (3x3
spatial shifts x 3 colour planes), `W` the 16x27 bank, `w` the window (IDENTICAL for every
channel), and `h_c` the learned per-channel filter:

```
y = SUM_c h_c * ( w . SUM_k W_ck u_k )  =  SUM_k ( SUM_c W_ck h_c ) * ( w . u_k )
```

The detector the tracker can express is `g_k = SUM_c W_ck h_c` — **an element of the ROW SPACE of
W and nothing else.** The online MOSSE filter does the discriminative learning and absorbs any
change of basis inside that span. Pretraining picks one 16-dim subspace of a 27-dim space; a
random draw picks another; both capture most of it. The null in §3 is not a surprise, it is
forced.

**Basis-invariance is not exact, and the exceptions name themselves:** the DSST shared
denominator couples channels, `eps_rel` is applied in that basis, Stage B3 normalises
PER-CHANNEL ENERGY, and conv outputs are quantized per channel. All four are CONDITIONING terms.
A residual of the size §3 measured (0.011-0.015 R, below resolution) is what that predicts.

## It retro-explains three recorded results

- **`blur2` is an exact null** (-0.0010 gray / -0.0012 RGB, `pooled_features.md`). A box average
  of a linear map is another linear map with the same span. It CANNOT do anything. That file
  read it as a fact about aggregation; it is a fact about linearity.
- **The gray rank cap.** Gray's 16 channels live in 9 dimensions — the ENTIRE gray patch space.
  The gray bank is already an identity lift up to conditioning, which is why the cap is
  "structural" and why RGB (16 of 27) had room to help.
- **`-rand` losing slightly** rather than tying exactly: a random Gaussian basis is worse
  CONDITIONED than the trained one, and conditioning is the only channel through which the bank
  can act.

## The two controls this predicts — RUN, 62 sequences, `runs/vot/0901_offline-bank/`

`rgb_vs_gray_loop.py` arms `-eye` (one-hot taps: 16 of the 27 raw planes, NO network at all) and
`-orth<N>` (random ORTHONORMAL rows via QR: the best-conditioned lift of the same dimension).
Both match the pretrained per-channel row norms, as `-rand<N>` does. Shipping eta 0.05 / gate 5.0.
**Control: the `rgb` arm reproduces the stored baseline 0.5394 / 0.2910 / 5792 exactly.**

```
arm             A        R    tracked      dR       dA    trim-3   trim-5   better/worse/tied
rgb        0.5394   0.2910       5792                                       (control)
rgb-eye    0.5386   0.2963       5898  +0.0053  -0.0008  +0.0105  -0.0007      19 /  9 / 34
rgb-orth0  0.5642   0.3017       6004  +0.0107  +0.0248  +0.0105  -0.0051      14 /  5 / 43
rgb-orth1  0.5605   0.2974       5919  +0.0064  +0.0211  -0.0027  -0.0124      16 /  8 / 38
```

**1. THE IDENTITY LIFT TIES THE NETWORK.** `-eye` has no network in it — 16 one-hot taps — and
scores R +0.0053 pooled with a common-survived-prefix accuracy of **0.5833 against 0.5832, a tie
to four decimals over 4724 frames**. 34 of 62 sequences are EXACTLY tied. The prediction holds:
**at this geometry, with ReLU off, the conv layer contributes a choice of basis and nothing
else.** Calling these "CNN features" overstates what the datapath does. (The one-hot bank also
quantizes exactly — one tap per channel at +-127 — so it carries no quantization penalty; that
favours the arm, which is the right way round for a control trying to show the network is
unnecessary.)

**2. ORTHONORMAL IS THE BEST OF THE THREE, AND ONLY ON ACCURACY.** Both seeds beat pretrained on
A by +0.021/+0.025, and **it survives the selection-effect check**: unlike the `-rand` arms of §3
(higher A while tracking FEWER frames), these track MORE frames and still gain, +0.0070/+0.0056
on the common survived prefix. Ordering across all five banks is
`random Gaussian < pretrained < orthonormal`, exactly the conditioning story. **On R it is a
null** — trim-5 negative on both seeds, seed 1 already negative at trim-3.

**3. NOTHING HERE IS SEPARABLE FROM A NULL ON R**, `-eye` included: every arm's pooled dR
collapses under a symmetric trim and 34-43 of 62 sequences never move. Same pattern as §3, and
the same standing limit — this bench could not tell trained from noise, so it can only FAIL a
bank arm, never confirm one.

## What this settles, and what it opens

**Settles: the weights axis is closed, and now for a structural reason rather than an empirical
one.** Not "a better donor is unlikely to help" but "with ReLU off, ANY 16x27 bank of comparable
conditioning is the same tracker". More channels only walks toward 27 = the full space, i.e.
toward MOSSE on 3x3-lifted RGB pixels — a bounded ceiling, and the literature's raw-pixels ->
HOG jump (0.451 -> 0.728 precision) says it is a low one.

**Opens, in order:**
1. **ReLU deserves re-examination IN THIS FRAME.** It was switched off because it costs ~25% of
   peak/sidelobe — a CONDITIONING statistic — but this derivation says the nonlinearity is the
   only thing that would make the bank more than a basis. That is a different question from the
   one asked when it was disabled, and it is host-... no: `CONV_RELU` reaches `AIE_FLAGS`, so it
   is a rebuild. Screen it offline first (`--arms rgb rgb-relu`, already implemented).
2. **Stage B3 channel reliability** is one of the four terms that BREAKS basis-invariance, so it
   is one of the few places the representation can still matter. Independent argument, same
   recommendation as `baselines.md` item 3.
3. **Geometry** (kernel size, stride, receptive field) stays the untouched axis — claim O-04.
   Aggregation over this map is refuted twice over, now including "because it is linear".

**An orthonormal bank is free to adopt** (`make weights` + a 1 KB copy) if the accuracy edge is
ever worth having, but it is inside the bench's resolution on R and must not be shipped on this
evidence alone.

---

# THE NONLINEARITY AXIS — ReLU RE-EXAMINED ON AR, AND CLOSED. CReLU TOO

**2026-09-01, `runs/vot/0901_offline-relu/relu62.json`.** The section above makes ReLU the GATE
on the whole weights axis: with it off the bank is a linear lift and every bank of comparable
conditioning is the same tracker. So the standing rejection had to be re-checked, and it needed
re-checking on its own merits — `settled.md` rejected ReLU on **held-out peak/max-sidelobe,
ONE patch (s6), by its own caveat**, and this project has three separate records of PSR-family
statistics ranking nothing. **`rgb-relu` alone had never been AR-scored on 62 sequences**; the
only 62-sequence ReLU data was `relumpool2`, which confounds it with max pooling.

Five arms, 62 sequences, 19,903 frames, shipping eta 0.05 / gate 5.0, `vot_ar_offline.py`.
**Control: `rgb` reproduces the stored baseline 0.5394 / 0.2910 / 5792 exactly.**

```
arm             A        R    tracked        dR (pooled)   per-seq dR   trim-3
rgb        0.5394   0.2910       5792                                          (control)
rgb-relu   0.5556   0.2578       5131          -0.0332       +0.0063   -0.0129
rgb-abs    0.5501   0.2678       5330          -0.0232       +0.0102   -0.0187
rgb-crelu  0.5564   0.2930       5831          +0.0020       +0.0227   -0.0080
rgb-half8  0.5587   0.2685       5343          -0.0225       -0.0106   -0.0226
```

`-abs` is the FULL-WAVE rectifier (the nonlinearity HOG actually uses); `-crelu` is 8 filters
AND THEIR NEGATIONS with ReLU on, so the rectifier emits both halves and the map spans
{linear, |.|}; `-half8` is its control — the same 8 directions DUPLICATED, no rectifier — which
prices the span 16->8 loss on its own.

## The decomposition, and it is clean

```
span 16 -> 8, no nonlinearity      half8 vs linear      -0.0225
add the sign pairing + ReLU        crelu vs half8       +0.0245   (trim-3 +0.0007)
                                   ------------------------------
net                                crelu vs linear      +0.0020
```

**The nonlinearity buys back exactly the span it costs, and nothing more.** That is the whole
result. `crelu` is the only rectified arm that ties the linear baseline, and it is the one that
KEEPS THE LINEAR PART — consistent with the derivation above rather than an exception to it.

The ordering across all five arms is coherent and matches the mechanism: **relu < abs ~ half8 <
linear ~ crelu.** Half-wave (throws away half the signal) is worse than full-wave (keeps the
magnitude, loses the sign), which is level with simply having fewer directions.

**The accuracy column is the selection effect and the standard check says so.** Every rectified
arm shows higher pooled A while tracking FEWER frames (5131-5831 against 5792). On the common
survived prefix the differences vanish: `-relu` +0.0017, `-abs` -0.0006, `-crelu` -0.0021,
`-half8` -0.0039.

## Verdict against the pre-registered falsifier

The bar was **dR >= +0.02 with trim-3 surviving AND `crelu - half8` positive.** The contrast is
positive (+0.0245) but `crelu` versus the actual baseline is **+0.0020**, an order of magnitude
under the bar, and every arm's per-sequence mean collapses or goes negative under a symmetric
trim with 37-46 of 62 sequences exactly tied. **REJECTED. `CONV_RELU=0` stays, and no board time
is spent.**

## What this settles

**The old verdict survives, but the evidence under it is now the right kind.** "ReLU off"
was a one-patch peak/sidelobe result that diverged from Danelljan §3.3; it is now a
62-sequence AR result with the mechanism decomposed. The `settled.md` reasoning — "a DCF is
linear in feature space; a half-wave rectifier throws away half the signal and the filter cannot
undo it" — is confirmed, AND the obvious repair (sign-paired channels, which does not throw
anything away) is measured and is a null.

**SCOPE — what is closed is POINTWISE rectification, not "nonlinearity".** The literature's
nonlinearity is not this. Deep trackers put it INSIDE a stack trained end to end (Danelljan's
conv1 is his shallowest and weakest layer, used ALONGSIDE deeper ones, never alone), and HOG
decomposes as |grad| -> ORIENTATION BINNING (a soft argmax ACROSS channels) -> BLOCK
NORMALISATION (divisive, over a local spatial neighbourhood). **Only the first of those three is
in the arms above** (`-abs`). Cross-channel competition and local contrast normalisation are a
different class and are UNTESTED. What this project already has is global (Stage A's z-score),
per-channel mean removal (B1) and the window; what it lacks is the LOCAL and the CROSS-CHANNEL
DISCRIMINATIVE forms.

**Two axes reachable without a rebuild are closed:**

- the WEIGHTS (any 16x27 bank of comparable conditioning is the same tracker — the one-hot
  control ties it),
- POINTWISE rectification (relu, abs, and crelu are a loss, a loss and a tie).

**`basketball` is an unexplained outlier and is recorded as one:** `crelu` survives 295 frames
longer on it and scores dA +0.343, the single largest per-sequence movement in the set. The
pooled A +0.0170 is that sequence and `rowing`; the MEDIAN sequence moves by 0.0000 and the
common-prefix dA is -0.0021. **It is NOT the selection effect** — corr(dA, d tracked) = +0.111,
i.e. this arm is not being scored on an easier prefix, which is how the other rectified arms and
the `-rand` banks earned their apparent accuracy. Anything revisiting this should start there.

What remains is **GEOMETRY — kernel size, stride, receptive field (claim O-04)** — which is an
AIE rebuild and has no offline evidence yet, and **CONDITIONING**, i.e. the four terms that break
basis-invariance: the DSST shared denominator, `eps_rel`, per-channel quantization, and **Stage
B3's per-channel energy normalisation**. B3 is the only one of the four that is host-only and
untested, which is the third independent argument for putting channel reliability next.
