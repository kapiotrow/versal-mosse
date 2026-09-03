# Against the nearest published embedded equivalent: the cost axis is comparable and this design wins it 20-54x; the tracking axis is not comparable at all

**Status:** current · **Updated:** 2026-09-03 · **Scope:** Danilowicz & Kryjak 2022 vs this work — what compares, what does not, the two orderings that transfer, and the measured EAO window term

**2026-09-03.** No new run. Danilowicz & Kryjak 2022 (`docs/papers/danilowicz2022_embedded_dcf.pdf`)
against this project's existing `results/resources.csv`, `results/perf.csv`, `results/power.csv`
and `results/arms.csv`. Every number lives in `results/embedded_baselines.csv`.
Claim ids: **`P-13`** (the cost axis) and **`M-17`** (the incomparability of the tracking axis).

## Why this paper and not another

Their deepDCF is the **architecturally nearest neighbour in the literature**, and the match is
unusually tight: conv1 features into a multichannel MOSSE filter (their eq. 12-15 are this
project's filter), quantised weights, **128x128 ROI -> 64x64 filter — the same geometry this
project ships**, a 2D FFT built as row transform -> BRAM transpose -> column transform (this
project's `fft_graph.h` with a memory tile instead of BRAM), and an embedded SoC-FPGA target
with the PS cropping the patch and the fabric doing the filter. Their Section 3 reviews 25
implementations; none of the others reports tracking quality on a common benchmark at all.

The differences that matter for reading the numbers: their features are VGG11 conv1 — **3x3,
including a 2x2 maxpool**, 4-bit, 8 channels — against this project's resnet18 conv1 **7x7
stride 2, int8, 32 channels PCA'd**; they run 3 scales by multi-resolution search against this
project's 33-scale DSST filter; and their filter datapath is 32-bit fixed point in fabric
against this project's int16/cint16 on AIE-ML.

## 1. THE TRACKING AXIS IS NOT COMPARABLE — three independent breaks

Their hardware arm reads **A 0.491, R 2.082, EAO 0.183**; this project's shipping arm reads
**A 0.5129, R 0.4279, EAO 0.1960**. **The EAO proximity is a coincidence and must never be
written up as a result.** Three things break independently, and any one of them is sufficient:

1. **`R` is a different quantity with the opposite sign convention.** VOT2015 `R` is the average
   number of failures per sequence, lower better. VOT-STb2022 `R` is the fraction of frames
   tracked before failure, higher better. There is no monotone map between them: 2.082 and
   0.4279 are not two readings of one axis.
2. **The EAO windows differ — [108, 371] against [115, 755].** The VOT2015 window is in the
   toolkit as `vot/stack/vot2015/rgb.yaml`; this project's is pinned by `vot_ingest.py`.
   This is the break that is easy to wave away and is not small: `arm_l1relu.md` sec.10.3
   measured that the first 29% of THIS project's window carries +0.0092 dEO while the
   remaining 71% carries +0.0005. A window half as long weights a different part of the curve.
   **`M-17` measures the term directly** rather than arguing about it — see section 5, and
   it is worth **+0.0827**, more than this project's entire measured improvement history.
3. **The ground truth differs in kind.** VOT2015 is rotated polygons; VOT-STb2022 is
   axis-aligned boxes fitted to segmentation masks. An axis-aligned tracker is penalised on `A`
   by the former and not by the latter — and this project outputs axis-aligned boxes only.

**What does transfer is the ORDERING inside their own table**, and this project has already
used both orderings:

- **Channel count saturates.** 8ch (0.183) ties 32ch (0.184); 64ch at the larger geometry
  (0.203) does not beat 32ch (0.207); only 4ch collapses (0.145). Their 16ch row at 0.174 is
  anomalously low in their own table. `layer1_features.md` screened at 16 channels on exactly
  this reading.
- **The larger geometry wins, by +0.024 EAO at 32 channels.** *Independently corroborated on
  this project's own hardware*: at matched `sigma/target` = 1/16 the finer map wins by
  **+0.0222 R / +0.0082 EAO** (`arm_res64.md` sec.25.1). **Two implementations, two datasets,
  two protocols, same sign.** The shipping arm sits at the geometry both sources call the worse
  one, which is the standing argument for the untested cell named in `layer1_features.md`
  (Layer-1 features at a 128x128 map, `MOSSE_SIGMA=4`).

## 2. THE COST AXIS IS COMPARABLE, AND IT IS THE HONEST ONE

Both sides are post-implementation utilisation of a named device for a named build.
**Quote the absolute counts.** The percentages are in the CSV only so each number traces back
to its source table — the devices are not the same size, so percent flatters this work:

| resource | deepDCF (ZCU104) | this work (VEK280) | ratio |
|---|---|---|---|
| LUT | 156,663 (68.0%) | 7,694 (1.5%) | **20.4x** |
| FF | 334,373 (72.6%) | 7,539 (0.7%) | **44.4x** |
| BRAM (36Kb equiv.) | 270.5 (86.7%) | 5 (0.8%) | **54.1x** |
| DSP | 480 (27.8%) | 44 (3.4%) | **10.9x** |
| AIE-ML cores | — | 6 of 304 (2.0%) | no counterpart |

**And the explanation is the architecture, not cleverness — say so in the write-up.** The
transforms did not get cheaper; they moved off the fabric onto 6 of 304 AIE-ML cores. The claim
that survives scrutiny is *"the same algorithm class costs 20-54x less programmable-logic fabric
when the FFT/conv/cmul chain runs on an AIE array"*, which is a result about the platform choice
and is exactly what `cha:przeglad` sets up. The claim that does not survive is any suggestion
that this is a better-engineered filter.

**One honesty check that cuts the other way:** the VEK280 is 2.26x the ZCU104 in LUT and FF and
1.92x in BRAM, but it has **fewer** DSPs (1312 against 1728). So the percentage column is not
uniformly generous, and the DSP row is the one place where absolute and relative tell nearly the
same story.

Their design is also **at the edge of its device** — 86.7% BRAM and 72.6% FF for **8 channels at
one scale**. This project runs 32 channels and 33 scales at 0.8% BRAM. That headroom is the
practical form of the result and is worth one sentence.

## 3. THROUGHPUT DOES NOT COMPARE AS FPS — and this is the trap

Their **467.3 fps is the PL module's throughput for ONE scale at 375 MHz**, with the PS cropping
the patch and feeding it over DMA (their Fig. 2). It is not an end-to-end frame rate. Their
**150 fps for 3 scales is a projection and they state it as one** ("tracking speeds reaching
150 fps can be expected"). This project's **38.04 FPS is measured wall clock for a whole frame**
including the APU tail, on a frame that is **84% CPU-bound** (`P-02`) and that carries a
33-scale DSST filter they do not have.

Put those in one column and a reader will divide them and conclude this design is 4x slower. The
defensible statements are narrower and there are two of them:

- **Neither number bounds the other**, because they measure different spans of the pipeline.
- **A like-for-like row is buildable and cheap**: this project at `SCALE_N=1` removes the DSST
  filter they do not have, and `results/frame_budget.csv` already isolates the AIE/PL share from
  the APU tail. That is the row to build if the thesis wants a speed comparison at all.
  **It has not been built, so do not quote one.**

## 4. ENERGY HAS NO COUNTERPART

They report none, and **none of the 25 implementations their Section 3 reviews reports energy
either** — the closest any of them comes is a remark that a lower clock could be chosen. This
project's **12.2 mJ/frame** (`P-12`, INA226 rails, `results/power.csv`) is therefore an
uncontested figure, not a winning one. Write it as filling a gap in the literature; there is
nothing to beat.


## 5. THE WINDOW TERM, MEASURED — +0.0827, or 1.39x the whole arm ladder

`scripts/eao_window.py`, 2026-09-03, no board time. Every arm's existing trajectories re-scored
under both windows by **the toolkit's own `multistart_eao_score`**, one analysis run per
workspace. `results/eao_window.csv`.

| arm | A | R | EAO [115, 755] | EAO [108, 371] | dEAO | ratio |
|---|---|---|---|---|---|---|
| `gray_h14` | 0.4890 | 0.2743 | 0.1367 | 0.2056 | +0.0689 | 1.504 |
| `rgb_h15` | 0.5043 | 0.3065 | 0.1474 | 0.2192 | +0.0718 | 1.487 |
| `rgb_eta05` | 0.5100 | 0.3283 | 0.1600 | 0.2307 | +0.0707 | 1.442 |
| `rgb_eta05_gate5` | 0.5100 | 0.3417 | 0.1629 | 0.2367 | +0.0737 | 1.453 |
| `rgb_res64` | 0.5336 | 0.3873 | 0.1849 | 0.2652 | +0.0802 | 1.434 |
| `rgb_l1lin` | 0.5294 | 0.3790 | 0.1851 | 0.2629 | +0.0777 | 1.420 |
| `rgb_sigma4` | 0.5133 | 0.4095 | 0.1931 | 0.2719 | +0.0788 | 1.408 |
| **`rgb_l1relu` — shipping** | 0.5129 | 0.4279 | **0.1960** | **0.2786** | **+0.0827** | 1.422 |

**The window is worth +0.0827 on the shipping arm. The entire arm ladder — eight arms, five
weeks, gray to Layer-1 features — spans 0.0593.** The choice of integration window is therefore
**1.39x everything this project has measured**, and it is a property of the challenge, not of the
tracker. That is the result: *any* cross-era EAO comparison that does not control for the window
is dominated by an artefact.

Applied to the paper this note is about: reading this project's 0.1960 against Danilowicz's 0.183
and concluding "comparable" is wrong before the dataset, the failure rule or the ground truth are
even considered. Correcting **only** the window moves this project's figure to 0.2786 — past
their hardware arm (0.183), past their best software arm (0.207), past the VOT2015 committee's
DSST and KCF (0.17 each). **That does not mean this tracker beats them, and 0.2786 MUST NOT be
put in a table beside 0.183.** Breaks 1 and 3 are untouched, and the curve is still generated by
multistart's failure rule rather than VOT2015's reset protocol. The number's only job is to show
that the window alone can manufacture a rank change of that size.

**The ratio falls monotonically as the arms improve, 1.504 -> 1.422**, and the mechanism is the
one `arm_l1relu.md` sec.10.3 already named: a better arm survives longer, so the long tail that
only the wider window includes is worth relatively more to it. The window does not merely shift
the scores, it **compresses the differences between good arms** — which is why 71% of the
[115, 755] window returned +0.0005 for `l1relu`.

**The ladder does not reorder** — one inversion in eight arms, `rgb_res64` (0.1849) against
`rgb_l1lin` (0.1851), a pair separated by 0.0002 at the published window and therefore tied.
**So the window changes the SCALE of every EAO in this project and not the ORDERING of its
decisions.** No arm accepted at [115, 755] would have been rejected at [108, 371]. That is worth
stating plainly, because it bounds the damage: the window artefact invalidates cross-paper
comparison, not this project's own internal history.

### Controls

- **The published column is the control and it is exact.** All eight `eao_115_755` values
  reproduce their `results/arms.csv` rows to four decimals — 0.1367, 0.1474, 0.1600, 0.1629,
  0.1849, 0.1851, 0.1931, 0.1960. The re-analysis is scoring the same runs the thesis quotes.
- **The scores are re-derived from the returned EAO curve to 0.00e+00** on every arm and both
  windows, so the sub-window decomposition integrates the same array the scores did.
- The two-arm decomposition reproduces sec.10.3's independent reimplementation: `l1relu -
  l1lin` gives +0.0175 over [115, 300] against +0.0081 over [301, 755], the same
  concentrate-early / dilute-late shape measured there for `l1relu - sigma4`.

### What not to re-derive

- **The toolkit's `EAOScore` has an off-by-one and it is load-bearing.** It is written as
  `mean(curve[low:high+1])`, but the curve it integrates has *length* `high`, so index `high`
  does not exist and the slice silently ends at `high-1`. **This is the published behaviour** —
  it is inside `arms.csv` exactly as it is inside the new column — so it cancels in every
  comparison and `eao_window.py` reproduces it rather than fixing it. Implementing the docstring
  instead leaves a 3.1e-04 residual. That residual is three orders below the effect and would
  have been invisible without the curve-vs-score control; **it is the whole reason that control
  exists**, and it fired on the first run.
- **Do not compute the curve once at `high=755` and slice it for the narrow window without
  reproducing the truncation.** The curves themselves are element-identical at both `high`
  settings (verified: max difference 0.0), so the curve is *not* the problem — the slice is.
- Do not read the falling `ratio` column as a measurement artefact to be normalised away. It is
  the finding of sec.10.3 restated on a different axis.

## Controls

- **The two orderings in section 1 are the control on the incomparability argument.** If nothing
  transferred, the paper would be useless here rather than partly useful; the fact that its
  channel-count and geometry orderings both reproduce on this project's hardware is what shows
  the two systems are measuring the same underlying tracker behaviour despite incomparable
  metrics. The geometry ordering is the stronger of the two because this project measured it
  independently and in a *matched pair*, not as a single arm.
- **The `M-17` re-analysis is the falsifier for the window half of section 1.** If re-scoring
  this project's own trajectories at [108, 371] moves EAO by less than the ~0.008 resolution the
  window argument needs, then break 2 is real but negligible and the note should say so.
- No control exists for breaks 1 and 3 short of running VOT2015 end to end; they are definitional
  rather than empirical and are argued from the two stack files and the two ground-truth formats.

## What not to re-derive

- **Do not "fix" the R comparison by inverting one of them.** Failures-per-sequence and
  tracked-fraction are not reciprocals; a sequence with one failure at frame 5 and one with one
  failure at frame 500 score identically on the former and very differently on the latter.
- **Do not compare BRAM counts without normalising the block size.** The Zynq part counts 36Kb
  blocks, the Versal part 18Kb blocks. `resources.csv` is in BRAM18 and their Table 2 is in
  BRAM36; the CSV carries a `bram36_equiv` row for both so the ratio is not off by 2x.
- **Do not read their 5.40% "CNN only" LUT column as the cost of their features.** It is the
  FINN-generated convolution module alone; the 68% total is dominated by the eight parallel
  channel-filter modules of their Fig. 3, which is the part this project moved to AIE.
- The percentage columns were the first thing written here and they overstated the result at
  every row. Absolute counts, then the device-size caveat, is the order that survives review.
