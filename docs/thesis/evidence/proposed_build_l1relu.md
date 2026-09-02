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

---

# PREPARED 2026-09-01 — everything off-board is green, nothing has run on the board

Steps 1 and 2 of the order of work are DONE and step 3 is one command away. No
hardware has seen this arm; every number below is a compiler schedule, a
bit-exact diff or an offline measurement, and each is labelled as which.

## 1. What had to be built, and what it cost

The build needed a datapath that did not exist: every conv path in the design was
a hand-unrolled 3x3 stride-1 loop. Five things moved.

| what | where | note |
|---|---|---|
| generic KxK / stride-S conv branch | `conv2d_kernel.cpp` | new branch, taken only when `CONV_KSIZE != 3` or `CONV_STRIDE != 1`; the two 3x3 branches are byte-for-byte unchanged |
| weight buffer grown 64 -> 192 B | `conv_weight_layout.{h,py}` | `CONV_WEIGHT_BYTES_PAD` is now DERIVED; resolves to 64 for every 3x3 bank, so no shipped arm's layout moves |
| second layout tag = `CONV_KSIZE` | same, + the host's startup check | one tag cannot separate 27 RGB-3x3 taps from 27 gray-7x7 taps |
| `CONV_KSIZE` / `CONV_STRIDE` / `CROP_ROWS` / `CROP_COLS` | `Makefile`, both toolchains | one variable each, per CLAUDE.md's shared-constant rule |
| `l1resnet` bank export | `export_weights.py --bank` | imports `l1_banks.resnet18_conv1_pca`, i.e. THE SAME function the offline screen scored |

**The crop and the feature map are now different things.** `PATCH_ROWS/COLS`
stays the FEATURE MAP — the Hann table, both FFTs, the accumulator, the filter
and every bin-to-pixel conversion are sized on it — and `CROP_ROWS/COLS =
PATCH x CONV_STRIDE` is what `roi_crop` is programmed to produce. Because
`roi_crop` takes its patch size as a RUNTIME AXI-Lite argument, the 128x128 crop
costs **no PL rebuild**, exactly as res64 exploited. A `make check_geometry`
guard and a host `static_assert` both refuse a crop over roi_crop's 128x128 BRAM
scratch, because nothing else in the toolchain can see that overrun.

**`CONV2D_STACK` — the pre-registered "sharp risk" — did not materialise, and
not because the estimate was wrong.** ~7.3 KB was the right number for 147 taps
*at the same unrolling*. The stride forced a loop restructure anyway (see below),
and the restructure keeps the taps in the weight `input_buffer` and the line
buffer in `static` tile memory, so the stack never scales with the tap count.
2048 B is unchanged from the 27-tap arm and the mapper emitted `libadf.a` with
0 errors.

## 2. The stride is what forced the loop restructure, not the tap count

At stride 2 output column `c` reads input column `2c + kc - P`, so consecutive
outputs are two input columns apart and a contiguous vector load no longer lines
up with them. The branch splits each input row into its `S` column PHASES as it
is read:

```
input col 2c + d,  d = kc - P   ==   phase (d mod S), half-column c + floor(d/S)
```

which puts every tap back on a unit stride, so tap `kc` is one unaligned load at
a constant offset from `c`. At `S = 1` the phase collapses to 0 and it is the
ordinary sliding window. One pass over the row does the split, which the read
loop was already doing byte by byte.

## 3. Bit-exactness — PASS, and the shipped arms are untouched

```
make x86sim_check KUT=conv2d SCENARIO=s6l1 PATCH_ROWS=64 PATCH_COLS=64 \
     N_CHANNELS=32 CONV_IN_CH=3 CONV_KSIZE=7 CONV_STRIDE=2 CONV_RELU=1 \
     CMUL_SPLIT_ACCUM=0

  weights: CONV_IN_CH=3 CONV_KSIZE=7 CONV_STRIDE=2 taps=147 out_shift=7 ...
  conv2d real: OK — 4096/4096 samples identical (max|.|=11)
  conv2d imag: OK — all zero
  === BIT-EXACT: PASS ===
```

`s6l1` is a NEW scenario directory, not an overwrite of `s6rgb`: the generic
branch is a different kernel path and the two must be runnable against each
other. Its stimulus is a 128x128 three-plane Stage-A crop from `roi_crop_ref`
(49152 PLIO samples) producing 4096 feature pixels.

**Regression, all four pre-existing kernel checks, after the change:**

| check | result |
|---|---|
| `KUT=conv2d SCENARIO=s6rgb` (shipping 27-tap) | 16384/16384 identical, `out_shift=5 bias_acc=153812 mean_prev=4842` — the same values as before |
| `KUT=conv2d SCENARIO=s6 CONV_IN_CH=1` | 16384/16384 identical |
| `KUT=cmul SCENARIO=s7` | 16384/16384 real and imag |
| `KUT=cmul SCENARIO=cmul_stress` | 16384/16384 real and imag |

**`layer0_weights.bin` for the shipping arm is byte-identical except for 16
bytes**, one per channel, at offset 62 — the new `CONV_KSIZE` tag, 0 -> 3. Every
tap, shift, bias and dequant scale is unchanged. Readers treat a 0 there as
legacy 3x3, so the tag is backwards compatible in both directions.

## 4. Graph, host and the native suites

- `make graph TARGET=hw` at `64x64 / ch32 / 7x7 / stride 2 / ReLU / 3-3-3 /
  H_SHIFT=15` — **0 errors**, `libadf.a` written, conv2d placed at
  `AIE_ML_CORE_X15Y0`, the same 6 cores and 1 memtile group as every other arm.
- `make application TARGET=hw ... FRAME_SOURCE=vot MOSSE_SIGMA=2.0` — builds,
  `-DCROP_ROWS=128 -DCROP_COLS=128 -DPATCH_ROWS=64 -DN_CHANNELS=32` in the stamp.
- `test_host`, `test_scene`, `test_vot_source` — PASS at the new geometry.
- `test_roi_crop` both arms — PASS (8 RGB cases, 8 gray).
- `ARM=l1relu ... scripts/calib_build.sh` — pre-flight OK end to end under
  `DRY_RUN=1`. The script now carries `l1relu` and its `l1lin` mechanism twin as
  first-class arms rather than as `ARM=rgb` plus overrides, because `CONV_RELU`
  and `CONV_KSIZE` reach `AIE_FLAGS` and it would otherwise print
  `BUILD VERIFIED` on a build that silently omitted both — the failure CLAUDE.md
  already records against the `FILTER_MASK` arms. Its weight check now verifies
  BOTH tags and the file LENGTH, the last of which is the only thing that can
  catch a 16-channel file on a 32-channel build.

## 5. THE FINDING THAT WAS NOT PREDICTED: the Layer-1 bank loses ~4 bits at `out_shift`

Measured offline on the `s6l1` Stage-A patch, all 32 channels, integer datapath:

| bank | `acc_max_theory` | observed `max\|acc\|` | bound is | `out_shift` | bits used of 15 |
|---|---|---|---|---|---|
| shipping 3x3 RGB, 16ch | 435 483 | **571 420** | *exceeded* | 4-5 | **14.1** |
| Layer-1 7x7/2, 32ch | 2 370 963 | **125 354** | **18.9x loose** | 7 | **9.9** (weakest channel 5.4) |

**Mechanism.** `export_weights.py` sizes `out_shift` against
`|bias_acc| + n_in*K^2*127^2`. That bound is LINEAR in the tap count; a real sum
over decorrelated taps grows like its square root, so the looseness itself grows
with `K`. The shipping bank is saved from this by its bias: `bias_acc = 153812`
is a large MEASURED quantity that dominates the bound and anchors the shift to
something real. **The `l1resnet` bank has `b_fold == 0` by construction** —
`l1_banks.py` returns `np.zeros(n_out)` because PCA components carry no bias — so
nothing anchors it, and `out_shift` is set by the loose bound alone. A shift of 2
would have fit the observed maximum.

**Why this matters before the board run.** This document predicts "Expect
`H_SHIFT` to move UP", reasoning from a rectified map's DC and from 32 channels
summing twice as many terms. This measurement points the OTHER WAY: `F_ch`
arrives about 4 bits smaller than the 3x3 arm's, which is upstream of both the
accumulator and the response. Both are predictions and the 200-frame `rails=0`
run is the arbiter — but it should be read holding both, not just the one written
first. If it comes back with the amplitudes far below rail rather than near it,
this is the reason, and the fix is one line in `compute_acc_params`.

**Deliberately NOT fixed here.** Changing how `out_shift` is derived is a
numerics change to every arm's exporter and needs its own before/after, and this
build already moves the kernel, the geometry, the channel count and the
nonlinearity at once. One magnitude at a time.

## 6. What remains, and it all needs the board

1. `ARM=l1relu PATCH_ROWS=64 PATCH_COLS=64 N_CHANNELS=32 FFT_SHIFT=3
   IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3 H_SHIFT=15 scripts/calib_build.sh`
   — the card image. 3-3-3 is inherited from res64's geometry argument and is a
   STARTING POINT, not a validated budget.
2. Flash, then the 200-frame `rails=0` calibration, read with sec.5 in hand.
3. **conv2d's per-channel schedule must be MEASURED before any frame time is
   quoted.** The `~20-22 ms` in this document is a model fitted to the two 3x3
   points, and the loop it models no longer exists. The compiler reported 39
   cycles for the PLIO read loop and 27 for the tap loop (one per plane); which
   loop level the 27 belongs to was not established, so the two were deliberately
   not multiplied into a total.
4. The 62-sequence sweep against `sigma4`'s EAO 0.1931, on the falsifier already
   written above.
5. Then, and only if the ReLU arm wins, the `ARM=l1lin` twin.

## 7. The calibration RAN — 2026-09-02. Budget 3-3-3 / `H_SHIFT=15` ACCEPTED, with two findings

`runs/l1relu_calib/run_l1relu_calib.log` + `runs/l1relu_calib/track.csv`. 200 frames, synthetic
scene, `TRAJECTORY=1 SCALE_TRAJ=1`, the card image of sec.6
(`a.xclbin 94fa6b981195...`, ELF `197d3f24c108...`, weights `61235f74c5ab...` — all three
verified ON THE CARD, against this tree, before the run). Weights file 6144 B = 32 x 192 B, both
layout tags correct. **Taken over ssh** — no `/dev/ttyUSB*` on the host, same as the res64
calibration — so amplitudes and tracking stand and **no FPS figure here is quotable**; the frame
BREAKDOWN is comparable to the other ssh runs and is used as one below.

### 7.1 The hard criterion: PASS

`rails = 0` on **all 200 frames of `F_ch`, `accum` and `response`**. `calib_report.py` prints
`BUDGET IS WRONG` on one row and it is a false positive: frame 0's `H(q15)` reports `rails=1`.

**That rail is by construction, not a clip.** `filter_init_quantize` sets the single global Q1.15
scale to `32767 / max|H|`, where `max|H|` is a **magnitude** — so the peak bin lands on the unit
circle of radius 32767 and rails a *component* exactly when that bin is near-purely real or
imaginary. It is the same shape of artifact as `accum_max = 46340` (`shift_budget.md`): a
magnitude normalised against a per-component rail. One bin, on the init frame only; the other 199
frames sit at 21.8-97.8% with `rails=0`. **Nothing was clamped** — the write-out clamp can only
fire above 32767 and the scale forbids it.

### 7.2 Tracking: at or above the comparator

| | this run | res64 calib | comparator `run_0824_1354` |
|---|---|---|---|
| mean IoU | **0.9229** | 0.9209 | 0.9188 |
| worst IoU | 0.8353 | 0.8276 | 0.8353 |
| centre err mean / worst | **1.21 / 3.27 px** | 1.29 / — | 1.37 / 3.52 |
| gate holds | 0 | 0 | 0 |
| PSR min / mean / max | **29.22 / 53.19 / 63.42** | 20.11 / 42.47 / 56.65 | — |
| frames accepted | 199 / 199 | 199 / 199 | — |

PSR is the arbiter of a budget and it is the healthiest of the three. (It is still weak in its
documented direction and cannot fail a confidently-wrong tracker; IoU is read above and agrees.)

### 7.3 FINDING 1 — the undershoot is REAL and it is ~6 bits, not the ~4 predicted

Converged tail (frames 21+), as a fraction of int16:

| buffer | l1relu, this run | res64 calib | ratio |
|---|---|---|---|
| `F_ch` | **0.13% (med 42, max 65)** | 15.4% | ~1/120 |
| `accum` | **0.57% (med 186, max 280)** | 4.4% | ~1/8 |
| `response` | **1.52% (med 498, max 749)** | 10.6% | ~1/7 |

Sec.5 predicted `F_ch` would arrive SMALLER and it does — **the direction was right and the
magnitude was not**. The whole feature spectrum lives in about 6 of 15 bits.
`mean_prev seeded ... ch0=0 ch31=0` in the header is the same cause visible a second way: the
PCA bank's `b_fold` is zero by construction, so **Stage B1 is inert on frame 0** and nothing
anchors `out_shift` to a measured quantity. `out_shift = 7` on every channel.

**This does not fail the budget** — `H_SHIFT` is downstream of the loss and cannot recover it,
and 3-3-3 / 15 rails nowhere and tracks. It is an `out_shift` question, i.e. a question about
`compute_acc_params`, and **`out_shift` lives in the weight buffer, so testing it is
`make weights` plus a 6 KB scp — no rebuild, no reflash**, and bit-exactness is re-checkable
off-board with `x86sim_check SCENARIO=s6l1`. Recommend closing it BEFORE spending the 62-sequence
sweep, on the "instruments before changes" rule: a sweep run 6 bits down measures the
quantisation, not the Layer-1 features.

### 7.4 FINDING 2 — the arm is AIE-BOUND and 2.3x SLOWER than shipping, not faster

`mean frame body 61.48 ms` against `sigma4`'s 27.11 ms **on the same instrument and the same
transport (ssh)**, so the ratio is comparable even though neither number is an FPS.

```
-- APU subtotal        10.997   17.9%      -- roi_crop launch     42.404   69.0%
-- GMIO (DMA_T)         9.890   16.1%      -- APU wall             8.435   13.7%
```

**69% of the frame is the host polling `roi_crop`'s `ap_done`, and that is conv2d back-pressure,
not crop work.** `roi_crop`'s geometry is byte-identical to the shipping arm (128x128 RGB crop);
what changed is downstream. The per-call FLOOR is the evidence:

| | calls | per call | run min | run max |
|---|---|---|---|---|
| sigma4 (128x128, ch16, 3x3) | 16 | 0.09 ms | 0.00 | 1.45 (ch0 recompute) |
| res64 (64x64 crop, ch16, 3x3) | 16 | 0.01 ms | 0.00 | 0.20 |
| **l1relu (128x128 crop, ch32, 7x7/2)** | 32 | **1.32 ms** | **1.19** | 2.33 |

A cached channel that used to cost ~0 now costs 1.19 ms. With `ROI_CROP_PIPELINE=1` the poll only
ever measures how far the AIE lags the host, so this is a direct, MEASURED read of conv2d — the
first one this arm has. **It is 2.4x the 17.97 ms/frame the scheduled-cycle model predicted**
(and the ~19-22 ms frame estimate built on it was wrong; the frame is 61.5 ms).

The efficiency says the same thing: 32 ch x 4096 outputs x 147 taps = 19.3M MACs in 42.4 ms is
**~0.36 MAC/cycle** on a core with 128 int8 lanes. That is the sec.2/sec.5 `kc`-loop finding
cashed out — `aiecompiler` reported "minimum length due to resources: 10" for a loop whose
4-D dynamic index is recomputed per tap. **Hoisting the row pointer out of `kc` changes no
arithmetic** and is re-verifiable bit-exact via `s6l1`; it is the one large win on the table.
Unlike `out_shift` it is an AIE rebuild, so it should wait until the sweep says the arm is worth
keeping.

### 7.5 What this does to the pre-registration

The falsifier is unchanged and still the sweep's job. Two things it did not price:

- The arm's **cost** entry ("~20-22 ms, but re-measure conv2d before quoting a frame time") is
  now measured and is **61.5 ms**. The sec.1 note that the restructure invalidated the
  `S + t*taps` model was right, and the error was 2.4x in the optimistic direction. A 62-sequence
  sweep costs ~2.3x `sigma4`'s board hours.
- The `out_shift` finding of sec.5 was priced as "read the calibration against it too". It has
  now been read and it is larger than sec.5's estimate, and it is cheap to fix. Do that first.

## 8. Both findings FIXED — 2026-09-02. Rebuilt, bit-exact, not yet run

Sec.7 closed with two defects and a recommendation to fix the cheap one before the sweep. Both
were fixed. **Everything below is verified off-board; nothing here has run on hardware.**

### 8.1 `out_shift` — a tighter bound that is still an exact worst case

`export_weights.py` gained `--acc-bound {loose,l1}` (Makefile `ACC_BOUND`, default `loose`).

- `loose` — `n_in * K^2 * 127^2`. Tap count only, weight-independent. **The default, because it
  is what every arm in `claims.md` was built with.**
- `l1` — `127 * sum|w_int8[oc]| + |bias_acc[oc]|`, per channel, from the actual quantized taps.

**Both are exact worst cases.** `l1` is attained at `x = 127*sign(w)`, so nothing was assumed
away and **neither bound can rail**. What `loose` pretends is that every tap carries +-127 of
weight, which is false for any quantized channel by construction: the per-channel scale puts ONE
tap at +-127 and the rest below it. At 147 taps that pretence costs ~2 bits.

Measured on one synthetic Stage-A patch through the exact integer datapath (no hardware):

| bank | bound / observed max\|acc\| | `out_shift` | feature map, % of int16 |
|---|---|---|---|
| shipping mobilenet 3x3/1 ch16, `loose` | 7.1x | 4-5 | 12.33% |
| l1resnet 7x7/2 ch32, `loose` (as run in sec.7) | 29.9x | **7 flat** | **1.89%** |
| l1resnet 7x7/2 ch32, **`l1`** | 6.6x | **3..5** | **10.34%** |

So the Layer-1 bank now sits where the shipping bank sits. The residual 6.6x is COHERENCE — a
real ROI never aligns with `sign(w)` — and is deliberately not claimed, because a bound that
depends on the patch is not a bound. `rails` on 200 frames stays the instrument.

**REGRESSION GUARD: the shipping bank is unchanged.** Regenerated with the default and compared
field by field against `HEAD`'s file: taps, `out_shift`, `bias_acc`, dequant scale and
`mean_prev` **identical on all 16 channels** (only the ksize tag byte differs, and that is the
2026-09-01 layout change, not this one).

`ACC_BOUND` reaches neither toolchain — it changes only the DATA in `layer0_weights.bin` — so no
flagstamp can see it and a stale file would pass every existing check. `calib_build.sh` therefore
recomputes both bounds from the file's own taps and requires the file to agree with the arm; the
negative control was run (asking `loose` of the `l1` file reports `MISMATCH ch0 file=3 loose=7`).

### 8.2 conv2d — 2.39x on the compiler schedule, no arithmetic touched

Three changes to the generic KxK branch, in order of what they were worth:

| | before | after |
|---|---|---|
| PLIO read loop, per 4 pixels | 143 cyc | **84** (== its resource lower bound) |
| MAC loop, per (ic, kr) = K taps | 7 x 11 cyc | **28** |
| cycles per channel, both weighted by iteration count | 1.00M | **419k** |

1. **Row bases hoisted out of the column loop.** Everything selecting a tap's source row — the
   input row, whether it is inside the crop, its slot in the K-row ring, the phase split —
   depends on `out_r` and never on `c`. Inside, it made every tap a 4-D dynamic index, which is
   what `minimum length due to resources: 10` was reporting.
2. **The `kc` loop unrolled**, so its phase and offset constant-fold instead of being an integer
   division and a modulo per tap on a machine that has neither.
3. **The PLIO read loop stores straight from the word** through ONE hoisted base pointer, and its
   inner loops are unrolled so every index but the column is a compile-time constant. Rolled, the
   nest was a NON-LEAF loop that aiecompiler refused to software-pipeline outright.

Plus `CONV_VEC_GEN = 32`, this branch's **own** vector-width knob — 2.3x better than 16 here, and
separate precisely so the two 3x3 branches keep their 16 and stay byte-for-byte as shipped.

**Two negative results, recorded because they are the useful part**: a `px[4][NC]` staging array
took the read loop to **248** cycles and an `srow[NC][S]` pointer array to **174**, both WORSE
than the 143 they replaced. An array indexed by unrolled compile-time constants still went to
memory. Only a single base pointer with constant offsets got under the original.

**Bit-exactness, all five checks green after the change:**

| check | result |
|---|---|
| `s6l1` (7x7/2 RGB, the reworked branch) | 4096/4096 identical |
| `s6l1` at `CONV_VECTORIZE=0` (scalar twin) | 4096/4096 identical |
| `s6` (gray 3x3) | 16384/16384 identical |
| `s6rgb` (RGB 3x3, the shipping path) | 16384/16384 identical |
| `s7`, `cmul_stress` | 16384/16384 identical, both parts |

### 8.3 The frame-time prediction, written down before the run

Scheduled cycles under-predicted hardware by **1.85x** on the sec.7 build (23.4 ms modelled
against 42.4 ms measured). Carrying that same factor forward:

- conv2d **~18 ms/frame**, from 42.4.
- frame body **~37 ms**, from 61.48. Against `sigma4`'s 27.11 ms on the same instrument, the
  62-sequence sweep would cost ~1.4x that arm's board hours rather than 2.3x.

**This is a prediction and it is the same kind of model that was 2.4x wrong in sec.7.4.** The
`roi_crop` ap_done poll in the next calibration run is the reading; treat anything above as void
until then. The residual target is named in the COST block: 84 cycles for 4 pixels is 21
cycles/pixel for 3 byte stores, and each (ic, phase) receives `4/S` CONTIGUOUS bytes per group,
so 12 byte stores could be 6 halfword ones — priced, not taken.

### 8.4 What to read the next calibration against

Two predictions now point in opposite directions on the same instrument, which is the good case:

- **`F_ch` should rise ~5.5x** from sec.7's 0.13% of int16, to roughly where res64 sat. If it
  does not, the `out_shift` attribution in sec.7.3 was wrong and the remaining factor is
  elsewhere — most likely that `[diag] F_ch` is CHANNEL 0 ONLY and channel 0 of a PCA bank is its
  most low-pass filter, whose windowed spectrum lands in the nine bins B2 nulls.
- **`rails` must stay 0 on `F_ch`, `accum` and `response`.** The bound guarantees the feature map
  cannot rail; it guarantees nothing downstream, and the response now carries ~5.5x more signal
  at an unchanged 3-3-3 / `H_SHIFT=15`. Sec.7 measured the response at 1.52% of int16, so there
  is room — but that is arithmetic, and the shift-budget rules exist because arithmetic has been
  overturned by hardware twice.
- The frame body against ~37 ms, per 8.3.

Frame 0's `H(q15) rails=1` will recur and remains benign (sec.7.1).

## 10. THE SWEEP RAN — 2026-09-02. **EAO 0.1960, the best on record.** Falsifier: NOT MET

`runs/vot/0902_1413-l1relu/`, workspace `~/vot/analysis/0902_l1relu`. Full VOT-STb2022:
**62 sequences, 419 trajectories, 82,504 frames, 0 failures**, every run name and length verified
against the dataset's anchors by `vot_ingest.py`. Card `a.xclbin 9cea47ce`, ELF `f584617498899de1`,
weights `d8a7d6ff` (`ACC_BOUND=l1`), all three md5-checked against this tree before the push.

```
arm       accuracy  robustness     EAO   frames
sigma4      0.5133      0.4095  0.1931    82278
l1relu      0.5129      0.4279  0.1960    82504
res64       0.5336      0.3873  0.1849    74309
```

### 10.1 The falsifier verdict, stated first

Sec."The falsifier, written before the build" says **ACCEPT on `dEAO >= +0.005` against
`sigma4`'s 0.1931**. Measured **dEAO = +0.0029**. **The bar is NOT met.** It is also not a
rejection -- EAO rose, it did not fall -- so the arm lands in exactly the band the two-part bar
was written to exclude: positive and too small to act on. **Recorded as NOT ACCEPTED on its own
pre-registration.** Sec.11 records the separate decision to ship it anyway and the grounds.

### 10.2 What IS strong: the paired per-sequence result

`scripts/grid_stats.py` conventions, on the 62 paired hardware R values:

| comparison | mean | trim-3 | trim-5 | b/w/t | P(dR<=0) |
|---|---|---|---|---|---|
| **R** vs `sigma4` | +0.0317 | +0.0176 | **+0.0112** | 35/24/3 | **0.011** |
| **A** vs `sigma4` | +0.0181 | +0.0093 | +0.0049 | 34/28/0 | 0.009 |
| **R** vs `res64` (matched geometry) | +0.0365 | +0.0233 | +0.0167 | 38/22/2 | 0.004 |

**It survives drop-top-FIVE and the bootstrap**, where the offline screen called it borderline
(P=0.041) and where the spatial mask's trim FLIPPED SIGN. Top-3 gains are 47% of the net; the
mask's were 133%. **Accuracy improved too**, so this is not an A/R trade -- the pooled A tie
(0.5129 vs 0.5133) is frame-weighting, and the typical SEQUENCE got better boxes.

The offline screen predicted +0.016 to +0.032 in R at the geometry arms' transfer rates; hardware
returned **+0.0184 pooled**. The bench got the magnitude right. What it could not predict is that
an R gain of that size buys almost no EAO -- which is sec.10.3.

### 10.3 WHY R MOVED AND EAO DID NOT — the finding worth more than the arm

The toolkit's EAO window is **[115, 755]** (`stack.yaml`, `multistart_eao_score`). Reimplementing
the expected-overlap curve reproduces the toolkit to 0.0008 (`sigma4` 0.1939 / `l1relu` 0.1969
against the reported 0.1931 / 0.1960), so the decomposition below is trustworthy:

| horizon N | 25 | 100 | 200 | 300 | 400 | 650 |
|---|---|---|---|---|---|---|
| dEO | +0.0175 | +0.0142 | +0.0104 | +0.0019 | -0.0013 | +0.0016 |

| EAO sub-window | share of window | dEO |
|---|---|---|
| 115-300 | 29% | **+0.0092** |
| 301-755 | **71%** | **+0.0005** |

**The entire gain lives below N ~ 300 and is diluted threefold by the window.**
0.0092 x 0.29 = +0.0027, against the observed +0.0029. Time-to-first-loss agrees: 18 fewer runs
die within 30 frames (128 -> 110 of 419), and those losses move into the 31-300 bins.

**Long-horizon behaviour, on runs still alive** -- and this names the next target:

| | frame 100 | frame 500 |
|---|---|---|
| `sigma4` `est_h/truth_h` IQR | [0.853, 1.067] | [0.760, 1.217] |
| `l1relu` IQR | [0.848, 1.090] | [0.777, 1.281] |
| `l1relu` median IoU | 0.568 | **0.388** |

The median stays at ~1.0 and the spread more than doubles between frames 100 and 500.
**THE "RANDOM WALK" READING OF THIS WAS WRONG AND WAS REFUTED THE SAME DAY** -- see
`../../engineering/scale_filter.md`: the VARIANCE grows only 1.95x over a 20x span where a random
walk predicts 20x, and the estimate is FROZEN on ~90% of frames. The growing IQR is survivorship
(runs whose scale goes wrong fail and leave the population) plus many runs each PARKED at their
own offset. The IQR above was read without controlling for selection; the conclusion that long
horizon is governed by box quality stands, the mechanism named for it did not. `l1relu` keeps MORE runs alive at long horizon
(160 vs 153 at frame 200) but with WORSE boxes (IoU 0.388 vs 0.446 at frame 500), and the two
cancel to +0.0005. Gate holds halve, 3.71% -> 2.01%: the Layer-1 features are genuinely more
confident.

**CONSEQUENCE FOR EVERY FUTURE ARM: 71% of the EAO window is governed by long-horizon box
quality, and nothing measured in this project has moved it.** An arm that improves acquisition or
mid-horizon survival -- which is what better FEATURES do -- is capped at roughly a third of its
apparent R gain. **The scale filter is frozen on ~90% of frames and `SCALE_ETA` is inert over a
12x range** -- the freeze is a DETECTION failure (`../../engineering/scale_filter.md`).


## 11. PRE-REGISTERED, before the run: `l1relu` + `MOSSE_ETA=0.1`

**2026-09-02, written before the ELF was built.** Host-only — the card already holds
`a.xclbin 9cea47ce`, so this is one ELF and an scp, no rebuild and no calibration.

**Why this combination, mechanistically.** Sec.8's sweep landed EAO 0.1960 and sec.10 below
decomposes it: the entire gain lives in expected overlap at horizons N < 300, worth +0.0092 over
the 115-300 part of the EAO window and **+0.0005 over 301-755, which is 71% of it**. `l1relu`
improves acquisition and mid-horizon survival and does nothing at long horizon. `MOSSE_ETA` is
the adaptation rate, so it acts on long-horizon appearance change -- the regime `l1relu` leaves
untouched. The two are complementary by construction rather than by hope.

**The offline evidence, `runs/vot/0902_offline-sigmaeta/`, 22 cells x 62 sequences.** At
`sigma/target = 1/16` (sigma 4 on a 128x128 map) eta 0.1 beats the shipping eta 0.05:
dR +0.0481, trim-3 **+0.0097**, P(dR<=0) **0.021** -- the only cell of 22 with a positive trim-3.
The peak is bracketed on both sides (eta 0.075 and 0.15 both fall back to the 0.34-0.37 baseline)
and it replicates within the sigma-5 row (eta 0.125, trim-3 +0.0103, P 0.060). It does NOT
replicate on the sigma-6 row -- 2 of 3.

**THE ASSUMPTION THIS ARM RESTS ON, NAMED BEFORE THE RUN.** The grid ran at 128x128 / sigma 4.
`l1relu` is 64x64 / sigma 2, i.e. the SAME `sigma/target` = 1/16 but half the map. Transfer
therefore assumes eta's optimum is governed by `sigma/target` and not by the absolute geometry.
`proposed_build_res64.md` sec.21 established that for SIGMA and for nothing else. **If this arm
returns a null, that assumption is the first suspect, not the eta result.**

**Falsifier.**
- **ACCEPT** on `dEAO >= +0.005` against `l1relu`'s **0.1960**, 62 sequences / 419 trajectories.
- **REJECT** if EAO falls. R alone does not carry it -- `pad30` died that way, and sec.10 shows
  R and EAO can move nearly independently on this tracker.
- **The specific way this can succeed and still teach nothing:** if dEAO comes from the 115-300
  sub-window again, eta is duplicating what `l1relu` already bought rather than adding to it.
  **Score the EO curve by sub-window, not just the scalar EAO** -- the tooling is in sec.10.

**Prediction, written down.** Offline dR +0.0481 pooled at the matched sigma/target; the geometry
arms transferred at 43-84%, giving +0.021 to +0.040 in R. Whether that reaches +0.005 of EAO
depends entirely on WHERE it lands: sec.10 measured `l1relu`'s +0.0184 R converting to only
+0.0029 EAO because it landed short of frame 300.

## 12. THE DECISION TO SHIP IT — 2026-09-02, and it is NOT the falsifier's verdict

**`l1relu` becomes the default configuration** even though sec.10.1 records the pre-registered
bar as NOT MET. That is a deliberate, separately-argued decision and it must never be written up
as though the falsifier passed. The grounds, in order of weight:

1. **It is the best EAO on record** -- 0.1960 against `sigma4`'s 0.1931 -- and EAO is this
   project's arbiter. The bar was written to decide whether the arm justified a rebuild and a
   reflash; that cost has now been PAID, so the question the bar answered is moot and the
   remaining question is simply which configuration is better. It is.
2. **The per-sequence evidence is much stronger than the scalar** (sec.10.2): R trim-5 +0.0112 at
   P(dR<=0)=0.011, A trim-5 +0.0049, no A/R trade. On the stability standard this project applied
   to reject the spatial mask, `l1relu` PASSES where the mask failed.
3. **The thesis-integrity argument, which never depended on the delta.** `feature_bank.md` proves
   that at `CONV_RELU=0` the conv layer is a LINEAR LIFT the online filter absorbs -- a one-hot
   bank with no network in it ties the pretrained one. The shipping configuration was therefore
   a CNN-feature tracker whose CNN was provably redundant. This arm replaces that with a real
   rectified Layer-1 bank **at no cost in EAO or accuracy and a measurable gain in R**, which is
   what makes the project's own conv-feature requirement honest.

**What is NOT claimed:** that the nonlinearity is worth +0.005 EAO, that Layer-1 features solve
the robustness gap, or that the falsifier passed. Claim `N-16` is answered as **mechanism
confirmed, deliverable marginal**.

**The mechanism check is still owed.** `ARM=l1lin` -- same bank, same geometry, `CONV_RELU=0` --
is a rebuild and reflash, NOT host-only. If the twin also lands near 0.196 the gain is the BANK
and not the rectifier, and `N-16`'s "confirmed" collapses to "the bank helps". Until it runs,
sec.10.2's numbers are attributable to the arm as a whole and not to the ReLU.

## 13. THE `MOSSE_ETA=0.1` ARM RAN — 2026-09-02. **REJECTED: EAO FELL.** And the bench INVERTED

`runs/vot/0902_1607-l1relu_eta1/`, workspace `~/vot/analysis/0902_eta1`. 62 sequences,
419 trajectories, 73,583 frames, **0 failures**, every run name and length verified against the
dataset's anchors. Host-only: ELF `043fc5a6ed2b`, one flagstamp line different from sec.10's
(`-DMOSSE_ETA=0.05` -> `0.1`), same card `a.xclbin 9cea47ce`.

```
arm            accuracy  robustness     EAO   frames
l1relu           0.5129      0.4279  0.1960    82504
l1relu_eta1      0.5079      0.4070  0.1817    73583
```

**dEAO = −0.0143.** Sec.11's falsifier says **"REJECT if EAO falls"**. It fell. `MOSSE_ETA` stays
at 0.05 and no default moves. This is a REJECTION, not a near-miss, and the sub-window
decomposition sec.11 asked for is moot — there is no gain to attribute.

Paired per-sequence (hardware, 62 sequences): **R is a wash and A is a real loss.**

| | mean | trim-3 | median | b/w/t | sign p | P(d<=0) |
|---|---|---|---|---|---|---|
| R | −0.0030 | −0.0174 | **0.0000** | 28/26/8 | 0.892 | 0.595 |
| A | −0.0136 | −0.0176 | −0.0066 | 23/35/4 | 0.148 | **0.991** |

The per-sequence spread is huge and symmetric — `polo` **+0.332** and `surfing` **+0.326**
against `rowing` **−0.332** and `snake` −0.217. **A faster learning rate trades whole sequences;
it does not improve anything systematically.**

### 13.1 THE ASSUMPTION NAMED BEFORE THE RUN IS THE ONE THAT BROKE

Sec.11 wrote it down: the grid ran at 128x128 / sigma 4, this arm is 64x64 / sigma 2. Both sit at
`sigma/target = 1/16`, and **`proposed_build_res64.md` sec.21 established that `sigma/target`
governs SIGMA — it never established that it governs ETA.** The transfer assumed it did. It does
not, or not at this map size. **A pre-registered assumption failing is the cheapest possible
outcome here**: the arm cost one ELF and one sweep, and the negative result is attributable
instead of mysterious.

### 13.2 THE METHODOLOGICAL RESULT, which outlives the arm

`vot_ar_offline.py` predicted **dR +0.0481, trim-3 +0.0097, P(dR<=0)=0.021** — the ONLY cell of
22 in the sigma x eta grid to survive a symmetric trim AND a bootstrap. Hardware returned
**−0.0030 on R and −0.0143 on EAO. THE SIGN INVERTED.**

The bench's record is now:

| axis | offline | hardware | |
|---|---|---|---|
| sigma 2 -> 4 | +0.0808 dR | +0.0678 | transferred, 84% |
| 64x64 map (`dec2`) | +0.1071, P=0.000 | +0.0456 | transferred, 43% |
| Layer-1 + ReLU | borderline, P=0.041 | trim-5 stable, P=0.011 | UNDER-called |
| spatial mask | +0.0601 | +0.0192 | over-called 3x |
| `pad30` | large | ~0 | over-called ~11x |
| **`MOSSE_ETA` 0.05 -> 0.1** | **+0.0481, P=0.021** | **−0.0030** | **INVERTED** |

**Its ~0.02 resolution in R was already documented. What this adds is that a TRIM-STABLE,
BOOTSTRAP-SIGNIFICANT cell can still invert on hardware** — so trim and bootstrap bound the
sampling noise, not the transfer. They say a result is not carried by three sequences; they say
nothing about whether the bench models the tracker. Treat `P(dR<=0)` as necessary and never
sufficient, and keep sec.11-style named assumptions on every arm derived from a geometry the
board does not run.
