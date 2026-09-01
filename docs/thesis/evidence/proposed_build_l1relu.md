# NEXT BUILD (PROPOSED, NOT BUILT): Layer-1 features — 7x7 stride 2, 32ch, ReLU, 64x64 map

**Pre-registered 2026-09-01.** Nothing built. Claim `N-16` / `O-04` in `docs/thesis/claims.md`;
the screens behind it are `evidence/layer1_features.md` and `evidence/feature_bank.md`.

```
PATCH_ROWS=128 PATCH_COLS=128          # roi_crop output unchanged
conv:  7x7, STRIDE 2, 32 channels      # -> 64x64 feature map
CONV_RELU=1                            # the point of the build
MOSSE_SIGMA=2.0                        # 2 bins over a 32-bin target = sigma/target 1/16
N_CHANNELS=32  CONV_IN_CH=3  TARGET_PADDING=2.0  MOSSE_ETA=0.05  PSR_GATE_MIN=5.0
shift budget: TO BE CALIBRATED (see "What it costs")
weights: resnet18 conv1 (7x7/2/64, BatchNorm folded) reduced 64 -> 32 by weight PCA
```

## Why this configuration and not another

Four Layer-1 cells were screened offline on 62 sequences. This is the only one that is both
**better than its control** and **not slower than today**:

```
config                        frame     R vs its control   P(dR<=0)
l5relu(32) 128^2 5x5/1        ~45 ms    +0.0598            0.060      1.8x slower
l5relu16   128^2 5x5/1 16ch   ~25 ms    -0.0287            0.229      a null
l1relu(32)  64^2 7x7/2        ~20-22 ms +0.0383            0.041      <- THIS ONE
danrelu     64^2 3x3+maxpool  ~15 ms    +0.0119            0.314      a null
```

At 128x128 it is the 32 CHANNELS that double the downstream, not the kernel; at 64x64 the map is
a quarter the size, so 32 channels cost half of today's per-channel work and the arm comes out
FASTER than the current best while carrying the features.

## What it is expected to be worth — and the honest range

Offline **+0.0383 pooled R** over the geometry-matched control, sign test 19 better / 6 worse
(p=0.015), bootstrap **P(dR<=0)=0.041**, common-prefix accuracy **+0.0072**. At the geometry
arms' measured transfer of 43-84%: **+0.016 to +0.032 R, EAO roughly +0.008 to +0.016.**

**This is BORDERLINE and must not be sold as a robustness win.** `dec2` and `sigma4`, the two
arms that transferred, were accepted at P(dR<=0)=0.000; this one sits on the 0.05 line and does
NOT survive drop-top-3 against its control (-0.0001). It may land at zero.

**The argument that does not depend on the tracking delta**, and is the real reason to build it:
`feature_bank.md` proves that with `CONV_RELU=0` and 16 channels over a 27-dim tap space the conv
layer is a linear lift the online filter absorbs — **a one-hot bank with no network in it ties
the pretrained one to four decimals.** The project's requirement is conv features, and the
shipping build makes the CNN provably redundant. `CONV_RELU=1` on a Layer-1 bank is what makes
the network do work; the nonlinearity beats its own linear twin on four independent screens
(+0.0398, +0.0472, +0.0575, +0.0655 pooled; drop-top-5 positive twice).

## The falsifier, written before the build

- **ACCEPT** on `dEAO >= +0.005` against `sigma4`'s 0.1931, on 62 sequences / 419 trajectories.
- **REJECT** if EAO falls. R alone does not carry it — that is how `pad30` died.
- **The mechanism check is separate from the arm**: `CONV_RELU=0` on the SAME bank and geometry
  is the linear twin, and it is one host-only ELF away. If the ReLU arm wins, run the twin; if
  the twin wins too, the gain is the bank, not the nonlinearity, and the thesis argument is void.

## What it costs

**NOT host-only.** `CONV_RELU`, the kernel size and `N_CHANNELS` all reach `AIE_FLAGS`: graph
rebuild, re-package, re-flash, and a shift-budget calibration.

- **`CONV2D_STACK` is the sharp risk.** 27 taps already forced 2048 (1344 B); 147 taps at the
  same unrolling wants ~7.3 KB. That likely means restructuring the inner loop rather than
  raising a number — **and a restructured loop invalidates the `S + t*taps` cycle model this
  build's frame-time estimate rests on.** Re-measure conv2d before quoting a frame time.
- `conv_weight_layout.h` grows 64 -> 147 B/channel, and the weight file 16x64 B -> 32x147 B. The
  host's runtime tag check makes a layout mismatch loud instead of plausible.
- **The shift budget must be re-calibrated and it is NOT the res64 budget.** A rectified map has
  a large DC for Stage B1/B2 to absorb, 32 channels sum 2x as many terms into the accumulator,
  and res64 already runs at 44-47% of ceiling on real video (sec.20.1 of
  `proposed_build_res64.md`). Expect `H_SHIFT` to move UP, and size it on the TAIL over 200
  frames, not the typical frame.
- Downstream (FFT, cmul, IFFT, APU tail) doubles with the channel count but quarters with the
  map: net ~0.5x of today per the sec.25 frame model.

## Order of work

1. `make weights` variant emitting the 32-channel 7x7 bank (new export path; `l1_banks.py`
   already builds it offline and caches to `build/l1_resnet18_pca32.npz`).
2. `x86sim_check KUT=conv2d` at 7x7/32ch before any hardware — bit-exactness first.
3. `scripts/calib_build.sh` with the new geometry, then a 200-frame `rails=0` calibration.
4. 62-sequence sweep against `sigma4`, then the `CONV_RELU=0` twin.
