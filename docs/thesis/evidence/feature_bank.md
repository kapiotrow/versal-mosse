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
