# Robustness — the ranked candidate list, and why this order

**Status:** closed · **Updated:** 2026-08-29 · **Scope:** the ranked robustness candidate list and why this order; several entries have since been decided

**2026-08-28.** Written after `robustness_gap.md` (how runs lose), `detector_gain.md`
(localisation exonerated) and `pooled_features.md` (aggregation refuted, geometry rejected
on hardware). Two new measurements here, both from the existing `track_*.csv` of
`runs/vot/0827_1642-eta05_g5p0` — 419 runs, 180,544 frames, no board time.

**The list in CLAUDE.md ("Next, in order — ROBUSTNESS") is reordered by this document.** Its
item 1 (channel reliability) stays live and gains a method; a candidate that was on nobody's
list moves to the top.

## 0. The scoring rule constrains the whole list

VOT terminates a run 10 frames after failure and re-enters the sequence at the next anchor.
So **recovery after a loss is worth exactly nothing to R or to EAO** — only not-losing, or
recovering inside the 10-frame grace, scores. That retires re-detection, search-window
expansion and full-frame rescue outright, however good they look on mean IoU. It is also
why the early-loss runs in §1 are worth more than their frame count suggests: a run that
dies at frame 2 contributes ~0 to both metrics no matter how long the sequence is.

## 1. NEW: 16% of losses are an INITIALISATION failure, not drift

Time-to-first-loss under the toolkit's rule (threshold 0.1, grace 10), over the 373 losing
runs of 419:

```
frames to first loss    <=10   11-30   31-100   101-300   >300
runs                      61      71      102        99      40
```

**61 runs — 16% of all losses — are already broken one frame after `filter_init()`:**

```
                f1 IoU   f1 PSR    f2 IoU   f3 IoU   f5 IoU     n
early-loss       0.571     7.35     0.173    0.016    0.000    61
all other runs   0.915    36.73     0.866    0.833    0.787   358
```

That is not the drift mechanism `robustness_gap.md` measured — drift takes tens of frames
and walks at 1.88 px/frame with PSR ~19. These runs never acquire. PSR 7.35 at frame 1 is
above the 5.0 gate, so they are accepted and then train on whatever they locked onto.

**And the code already names the cause.** `mosse_tracker.cpp:31`:

```
TODO: affine perturbations for initialisation (Bolme §3.4); this is the N=1 case.
```

`filter_init()` is `filter_update()` at eta = 1 against a zeroed state — the exact closed
form for **one** training image. Bolme's §3.4 regularises it with eight random affine
perturbations of the first frame, and the anchored protocol makes that 419 inits per arm,
i.e. the init is exercised ~419 times more often here than in a single-start benchmark.

**Cost is one-off and tiny**: N-1 extra `roi_crop -> conv2d -> FFT` passes on frame 0 only,
~25 ms each at ch16, so ~200 ms per anchor at N = 8 and **zero effect on steady-state FPS**.

**The honest caveat, priced before the run.** `roi_crop` resamples an **axis-aligned** ROI,
so the hardware can deliver *translation and scale* jitter and **not rotation**. That is a
weaker perturbation set than Bolme's, and if the win turns out to come from rotation
specifically it is not implementable without a new PL datapath. Any offline arm must
therefore be restricted to the warps `roi_crop` can actually produce — an arm that jitters
rotation offline would be a result about hardware that does not exist. It does have a free
side effect: it is the first thing to exercise the bilinear interpolator, which per CLAUDE.md
has never run on hardware.

**Falsifier, written first.** `vot_ar_offline.py` has a MEASURED resolution of ~0.02 in R.
A dR under +0.02 on the 62-sequence offline set is not a result. Secondary and more
specific: the EARLY-FRAME IoU should rise, and if R moves while the early frames do not, the
mechanism is wrong even though the number is good — that is the `eta05` outcome (gain real,
explanation refuted) and it must not be repeated by accident.

### Offline arm — IMPLEMENTED 2026-08-28, `rgb_vs_gray_loop.py` arm suffix `-warp<N>`

`rgb-warp8` runs Bolme's N-sample closed form at init — `A = SUM conj(G_i) F_i`,
`B = SUM |F_i|^2` over 8 views of frame 1 — then normalises by N so the first
`filter_update`'s eta still weights the new frame against a state of the same magnitude. The
warp set is deterministic from a fixed seed, so two arms see the same warps and a rerun
reproduces exactly. Each warp moves the CROP: centre by a fraction of the ROI, extent by a
log-uniform scale factor, both defaulting to 5% (`--warp-shift` / `--warp-scale`). **There is
no rotation knob, deliberately** — see `warp_set()`.

The G-centring sign is the one thing that can be silently wrong: a crop centred at
`+(dr,dc)` puts the target at `-(dr,dc)` inside the patch, so `G` is centred there. Same rule
as the measured-displacement training target, and getting it backwards would train the filter
to peak away from the object in a way that looks like ordinary noise.

**Two controls, both pass before any result is read:**

| control | asserts | outcome |
|---|---|---|
| `gray` on full `car1` | the plumbing did not disturb the shipped path | **0.7131**, digit-for-digit the value recorded in `pooled_features.md` |
| `gray-warp8 --warp-shift 0 --warp-scale 0` | 8 degenerate warps == the 1-sample init, i.e. the accumulate-and-normalise arithmetic is right | output IDENTICAL to `gray` on every column |

The degenerate control is the one that matters: an accumulator that forgot to normalise, or
summed B over the wrong axis, passes every "does it run" check and fails this one.

**And two MUTANTS, so the arm is known to be able to fail** (`--warp-mutant`). A warped init
that helps must hurt when its geometry is wrong, or what the arm measures is "more training
samples" and not "warps": `gsign` centres `G` at `+`the crop offset instead of `-`, the one
error that is invisible by inspection and reads as ordinary noise; `noshift` drops the
correction entirely. Both print a banner and invalidate the run as a result, in the style of
`RESET_MUTANT`.

```bash
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
    --arms rgb rgb-warp8 --sequence <seq> --eta 0.05 --psr-min 5.0 --json out.json
python3 scripts/vot_ar_offline.py out.json rgb rgb-warp8
```

### RESULT — the offline arm does NOT clear its falsifier

62 sequences, 19,903 frames, RGB bank at the shipping `MOSSE_ETA=0.05` / `PSR_GATE_MIN=5.0`,
`vot_ar_offline.py`. (The baseline reproduces the recorded 19,903-frame set exactly, so this
is a null from a working bench and not a broken one.)

```
arm            accuracy   robustness   tracked / 19903
rgb              0.5394       0.2910      5792
rgb-warp8        0.5416       0.2992      5954
                 +0.0022      +0.0081
per sequence R:  better 8, worse 11, tied 43
```

**+0.0081 against a written-down floor of +0.02, and the sign test leans the wrong way.**

**IT IS ONE SEQUENCE.** `shaking` goes R 0.299 -> 1.000 (+0.701) and carries the whole
aggregate:

```
drop nothing                        n=62   dA +0.0022   dR +0.0081
drop shaking                        n=61   dA +0.0053   dR -0.0048
drop shaking, rabbit, kangaroo      n=59   dA +0.0063   dR -0.0063
```

That is the trap this project has now written down three times (`tiger` in
`pooled_features.md`, `nature` everywhere): an aggregate carried by one member of 62.

**And the MECHANISM falsifier — the specific one — is flat.** If warped init worked, the
frames right after init would improve. They do not:

```
                f1      f2      f3      f5     seqs lost within 10 frames of init
rgb           0.896   0.825   0.801   0.687              10
rgb-warp8     0.898   0.830   0.800   0.698               9
```

**Verdict: item 1 does not earn board time in this form**, and it drops below item 2. The
remaining perturbation axes were then tested rather than left as an open hope — see below,
where aspect and rotation both fail too.

**The board measurement in §1 still stands** — 16% of losses are init failures. What is now
known is that *this* regulariser does not fix them.

### Every perturbation axis this hardware can reach — ALL FOUR TESTED, 2026-08-28

The first arm moved translation and isotropic scale together. Rather than leave "maybe a
different warp" as an open hope, the remaining two axes were run the same way, 62 sequences
each. **Cost per axis on the board is the reason they are separate arms:**

| axis | board cost | how |
|---|---|---|
| translation | **free** | `base_x` / `base_y`, runtime AXI-Lite |
| isotropic scale | **free** | `roi_h` / `roi_w` together |
| aspect | **free** | `roi_h` / `roi_w` INDEPENDENTLY — they are separate registers |
| rotation | **host-side, ~2 ms/warp at init** | host pre-rotates the ROI region into `frame_bo`, then `roi_crop` crops axis-aligned as usual. Inside `roi_crop` it is a kernel rebuild + re-package + reflash: `sy` becomes column-dependent, which is two incremental adders on a loop that is m_axi-latency-bound at 10.9 cyc/px, so the silicon is nearly free and the PROCESS is not |

Magnitudes chosen so each axis perturbs comparably: rotation 10 deg displaces a patch-edge
bin by about the same distance as the 5% shift jitter (64 bins x 0.175 rad ~ 11 bins).

```
arm                    A        R   tracked   meanIoU        dA        dR
rgb (baseline)    0.5394   0.2910      5792    0.1792
rgb-warp8         0.5416   0.2992      5954    0.1821   +0.0022   +0.0081
warp8-aspect      0.5521   0.2589      5152    0.1659   +0.0127   -0.0322
warp8-rot10       0.5448   0.3056      6082    0.1875   +0.0055   +0.0146
warp8-gsign       0.4419   0.3435      6837    0.1683   -0.0975   +0.0525
```

* **Aspect LOSES**, and it loses on robustness and mean IoU while gaining accuracy — a
  tighter-boxing, shorter-surviving tracker. Stretching the ROI teaches the filter a target
  shape the detection crop never presents, and the DSST scale filter is a separate estimator
  that this cannot help.
* **Rotation is the best of the four and still does not clear the falsifier.** dR +0.0146
  against a +0.02 floor. It is the ONLY arm that moves A, R and mean IoU together — the
  mutant's signature (R up, A and IoU down) is absent — so if any variant is ever revived it
  is this one.
* **And it is three sequences again:**

```
drop nothing                      n=62   dA +0.0055   dR +0.0146   dMeanIoU +0.0083
drop rowing                       n=61   dA +0.0023   dR +0.0109   dMeanIoU +0.0054
drop rowing, singer2, birds2      n=59   dA +0.0016   dR -0.0020   dMeanIoU -0.0010
```

**VERDICT ON ITEM 1: the init regulariser is a null on every perturbation axis this design
can produce**, not just on the one that was convenient. That is a much stronger negative than
the first sweep alone, and it is why the other three axes were worth the compute: the cheap
follow-ups ("bigger jitter", "a different warp") are now closed rather than left open.

**Two things that were expected to short-circuit this and did NOT** — both worth not
re-deriving:

* **Translation warps are NOT algebraically degenerate.** In exact arithmetic they should be:
  a crop shifted by d gives `F.exp(-i th)` trained against `G.exp(-i th)`, and
  `conj(G_d) F_d = conj(G) F` exactly, so the sample contributes nothing. The fixed Hann
  window and border inflow break it hard — measured `rel|dA|` of 0.19 / 0.30 / 0.51 at
  2 / 5 / 10% shift. **So "the jitter was too timid" is refuted**: each warp already produced
  a substantially different training sample and averaging them still did not help.
* **Photometric warps are nearly worthless, and Stage A is why.** log -> zero-mean -> unit-L2
  annihilates gain (log-gain becomes a constant, removed by the mean) and contrast/gamma
  (scales the log, removed by the L2). Measured residue after int8 rounding and clipping:
  `max|diff|` 7-9 LSB against a patch sigma of ~32. That residue is quantization noise, not
  illumination variation, so brightness/contrast jitter buys no illumination invariance here.

### WHAT BOLME ACTUALLY SAYS — the null has a candidate mechanism, and it is testable

Re-reading `docs/papers/bolme2010_mosse.pdf` after the four arms
failed. **Section 3.3 presents regularization and initialisation perturbations as ALTERNATIVE
cures for the same defect**, and the defect is not "too few training samples" in the general
sense — it is specifically low-energy denominator bins:

> "ASEF filters are unstable when trained on small numbers of images because the element-wise
> division [...] becomes unstable when a frequency in the training image contains very little
> energy [...] **Averaging large numbers of exact filters compensates for this problem** [...]
> **Alternatively, regularization can be used to correct for low-energy frequencies** [...]
> Because the denominator for MOSSE is the sum of the energies over more images, it will
> rarely produce small numbers and is therefore more stable."

And the decisive detail: **Figure 3 — the perturbation-count curve, the entire empirical case
for perturbations — is captioned "Results shown without regularization."** Figure 4 then shows
that with a proper epsilon "all three filters are producing good peaks", using a FIXED eight
images.

**This design already has both of Bolme's other cures, and one of them is not obvious.**
`eps_rel = 1e-3` on `mean(B)` is the first. The second is that `B` is a SHARED denominator
summed over **16 channels** — Bolme's own stability argument ("the sum of the energies over
more images"), applied across the feature bank instead of across time. Measured fraction of
denominator bins below `eps` at init:

```
sequence   1 ch, 1 image   16 ch (shipping)   16 ch + 8 warps
car1              48.78%             19.94%            13.76%
tiger             64.73%             27.20%            19.55%
nature            48.07%              4.43%             1.08%
ants1             67.77%             69.90%            69.26%
```

**The 16-channel sum alone removes more ill-conditioning than the 8 warps then add on top of
it** (car1 48.8 -> 19.9 by channels, 19.9 -> 13.8 by warps). So the warps are a third cure
applied to a disease already treated twice — which is a mechanism for the null, not merely a
restatement of it.

**PREDICTION, WRITTEN BEFORE THE RUN.** If this reading is right, the warp benefit is a
function of the regularizer, and lowering `eps_rel` into Bolme's regime should make it appear:

* at `eps_rel = 1e-3` (shipping), dR(warp) ~ +0.008 — measured, above;
* at `eps_rel = 1e-6` and `1e-8`, the no-warp baseline should FALL, and dR(warp) should be
  clearly positive and much larger — that is Figure 3;
* the arm is REVIVED only if (low eps + warp8) beats (1e-3 + no warp) outright; recovering
  ground the low eps just gave away is a mechanism confirmation, not a win.

**And the falsifier for my own explanation:** if dR(warp) stays ~0 at every eps, then
low-energy bins are not what the warps were ever fixing here, and this whole reading is wrong
even though the null stands.

#### RESULT — the prediction held in direction and the arm is still not revived

62 sequences, six cells, everything else identical:

```
eps_rel   arm                A        R   meanIoU     dR(warp)
1e-3      rgb           0.5394   0.2910    0.1792               <- shipping regularizer
1e-3      rgb-warp8     0.5416   0.2992    0.1821       +0.0081

1e-6      rgb           0.5422   0.2863    0.1749
1e-6      rgb-warp8     0.5435   0.3006    0.1833       +0.0143

1e-8      rgb           0.5418   0.2862    0.1748
1e-8      rgb-warp8     0.5434   0.3006    0.1833       +0.0143

REVIVAL TEST -- does any (low eps + warp8) beat (1e-3, no warp)?
  eps 1e-3   warp8:  dA +0.0022  dR +0.0081   no
  eps 1e-6   warp8:  dA +0.0041  dR +0.0095   no
  eps 1e-8   warp8:  dA +0.0041  dR +0.0095   no
```

**The warp benefit IS a function of the regularizer** — dR(warp) +0.0081 -> +0.0143, a 1.8x
increase as eps falls. That is Bolme's Figure 3 mechanism reproduced on this tracker, and it
confirms the reading above rather than merely being consistent with it.

**It saturates below 1e-6, and the reason is the whole point.** `B` never gets that small:

```
fraction of the 16-channel denominator below k*mean(B), at init
sequence      k=1e-3    k=1e-6    k=1e-8
car1          19.94%   0.0000%   0.0000%
tiger         27.20%   0.0000%   0.0000%
nature         4.43%   0.0000%   0.0000%
ants1         69.90%   0.1892%   0.0000%
```

Below 1e-6 the bin count is essentially zero, so eps stops binding and 1e-6 and 1e-8 agree to
four decimals. **Bolme's eps ~ 0 regime is not reachable on this design by lowering eps** —
the 16-channel shared denominator keeps `B` away from zero structurally. The same fact
explains why removing the regularizer costs almost nothing here (R 0.2910 -> 0.2862), where on
a single-channel MOSSE it certainly would.

### THE CONCLUSION FROM ALL OF IT

> **Perturbations do exactly what Bolme says, and this design has already bought that
> stability a different way.** The 16-channel shared denominator is a structural substitute
> for his eight affine perturbations, so the ceiling on what init perturbations can add here
> is about +0.014 in R, and it CANNOT be raised by weakening the regularizer — because the
> ill-conditioning they cure is not present to be exposed.

Read that as the reason a well-cited technique does not transfer, not as a defect. It is a
property of the FEATURE BANK -- 16 channels summed into one denominator -- and not of the
fixed-point implementation, which is exonerated separately.

**Item 1 is CLOSED.** Four perturbation axes (translation, isotropic scale, aspect, rotation),
three regularizer settings, one mutant, all 62 sequences: nothing clears +0.02 in R over the
shipping arm. What survives for the write-up is the mechanism, not a change.

**Practical corollary worth keeping:** a future feature bank with FEWER channels, or per-channel
denominators instead of the shared one, re-opens this — it would remove the structural cure and
put Bolme's perturbations back in play. That is a real coupling between the feature-bank item
(§6) and this one.

#### A NUMBER COLLISION TO GUARD AGAINST

**R = 0.3435 (the `gsign` mutant, offline) is NOT R = 0.3417 (the SHIPPING ARM on hardware).**
They agree to three decimals by coincidence and mean opposite things: 0.3417 is `vot analysis`
on the anchored multi-start protocol, 419 trajectories, the metric of record (CLAUDE.md's
shipping row, A 0.5100 / EAO 0.1629); 0.3435 is a deliberately broken filter scored by
`vot_ar_offline` on SINGLE-START runs, where the same arm has A 0.4419 and mean IoU 0.1683.
Every offline number on this page is the toolkit's RULE on different RUNS, is biased ~0.02
high in R, and is comparable only between arms scored identically. **The offline baseline at
R 0.2910 is not a regression from 0.3417** — it is a different measurement of a different
thing, and nothing on this page moved the shipping arm.

### THE MUTANT IS THE MOST USEFUL RESULT ON THIS PAGE, AND IT IS ABOUT THE INSTRUMENT

`gsign` — the init trained with `G` centred at `+`the crop offset instead of `-`, i.e.
deliberately taught to peak away from the object — run over the same 62 sequences:

```
arm                        A        R    tracked   frame-wtd IoU   frames IoU>0.5
rgb (baseline)        0.5394   0.2910      5792          0.1792           17.3%
rgb-warp8             0.5416   0.2992      5954          0.1821           17.9%
rgb-warp8 gsign       0.4419   0.3435      6837          0.1683           15.4%
                     -0.0975  +0.0525
```

**The deliberately broken arm scores dR = +0.0525 — 6.5x the correct arm, and three times
the instrument's stated resolution — while tracking measurably WORSE by every direct
measure**: accuracy -0.0975, frame-weighted mean IoU 0.1792 -> 0.1683, and the fraction of
frames above IoU 0.5 down from 17.3% to 15.4%.

Two things follow, and the second is worth more than this whole section.

1. **The bench is not blind.** A wrong warp geometry moves it hard, so the correct arm's
   +0.0081 is a genuine null and not an instrument that cannot see warps. That is exactly
   what the mutant was written to establish, and it establishes it in the opposite direction
   from the one expected — which is the only reason it is informative.
2. **`vot_ar_offline`'s R CAN BE RAISED BY DEGRADING THE FILTER.** A mis-centred init makes a
   broad, weak, under-confident filter that reports smaller displacements and stays near where
   it started; on sequences whose target does not travel far, that survives the
   10-consecutive-frame rule while overlapping the target badly the entire time. R rewards
   *not being 10 frames dead*, not tracking. `tiger.md` warned that a freeze can be protective
   and `pooled_features.md` ruled that confound out for `dec2` by correlating dR against the
   hold-rate change; this is the same artifact reached by a different lever, and it is now
   DEMONSTRATED rather than argued.

**So: never accept an arm on R alone from this instrument.** Require A not to fall, or price
the fall. An arm with dR > 0 and dA < -0.02 should be assumed to be this artifact until shown
otherwise. `pooled_features.md`'s `dec2` row (dR +0.0567, dA -0.0086) survives that test; this
mutant (dR +0.0525, dA -0.0975) does not, and the two are otherwise the same shape — which is
precisely why the rule needs stating.

## 2. Spatial reliability — and it IS host-only, via the Stage B2 trick

The literature prices this highest of anything on the list (CSR-DCF ablation: uniform box
mask -21% EAO, removing the mask entirely >50%, and their sentence "reduces the tracker to a
standard DCF with a large receptive field" describes this design exactly — the target is 27%
of the ROI at `TARGET_PADDING=2` and nothing masks the rest). CLAUDE.md carries it as
needing spatial-domain access to the filter, i.e. a host FFT this design deliberately does
not have.

**It does not.** A separable raised-cosine mask has a *compact* DFT — the periodic Hann's has
exactly 9 non-zero bins, which is the identity Stage B2 is already built on — so the
projection `h <- m (.) h` is a **9-tap sparse 2-D convolution applied to the published `H`**,
not an FFT pair. ~16 ch x 16384 bins x 9 complex MAC = 2.4 MMAC/frame, the same order as the
existing filter tail. Because it lands in `H`, the AIE sees an already-masked filter and
**detection and training stay consistent with no graph change**.

This is *not* `TARGET_PADDING` by another name, and that distinction is the reason to run it
after pad30 failed: padding moves the mask, the px/bin, the search range and the DSST
extraction region at once, which is exactly why `docs/thesis/evidence/pooled_features.md` could not
attribute its own hardware result. The mask holds every one of those fixed.

Known coupling, from this project's own history: shrinking the filter's support moves the
PSR scale, and `PSR_GATE_MIN`'s worth is conditional on the PSR scale. Expect to re-tune the
gate on the same sweep, as `MOSSE_ETA=0.05` forced.

Not CSR-DCF: theirs is an ADMM with several iterations per frame and a mask estimated per
frame from colour segmentation. This is a one-shot projection onto a fixed, box-shaped
support. If the projection wins, the estimated mask is the follow-up, not the entry point.

### §2 TESTED OFFLINE 2026-08-28 — the fixed box mask is a NULL on AR

`rgb_vs_gray_loop.py` arm suffix `-mask<N>` (N = plateau width as a percent of the patch; 50 is
exactly the target box at `TARGET_PADDING=2`), separable raised cosine, applied as `h ← m⊙h` on
`H` at detection time with `A`/`B` untouched. **Exact FFTs offline on purpose** — the 9-bin
sparse-spectrum form is a BOARD implementation detail, and testing the approximation before the
idea would confound the two.

**Measured first, not assumed: the filter's energy is centred at the PATCH CENTRE** (peak of
`Σ|h|²` at (64,64); a corner-wrapped 64x64 box holds only 8-12%), even though `resp[0,0]` is the
zero-displacement bin. Masking at the response origin would delete the filter and read as
"masking hurts". That same measurement quantifies CSR-DCF's complaint here directly: **a centred
64x64 box holds only 51.6% (`car1`) / 54.9% (`tiger`) of the filter's energy — half the filter is
spent on background.**

Controls: the no-mask path reproduces the stored baseline bit-for-bit, and a full-width mask with
zero taper is the EXACT identity on every column — which is what the projection must do if the
transform pair is right.

```
arm                  A        R   tracked   meanIoU        dA        dR
rgb             0.5394   0.2910      5792    0.1792
rgb-mask70      0.5251   0.3089      6149    0.1820   -0.0143   +0.0179
rgb-mask50      0.5413   0.3033      6037    0.1939   +0.0020   +0.0123
rgb-mask35      0.5369   0.2914      5800    0.1781   -0.0024   +0.0004
```

**None clears the +0.02 bar, and all of it is a handful of sequences:**

```
arm            R better/worse/tied   drop top-3 gainers   median dR   trim 3 BOTH ends
rgb-mask70            5 / 5 / 52              -0.0015      +0.0000            +0.0000
rgb-mask50          10 / 10 / 42              -0.0130      +0.0000            +0.0019
rgb-mask35          18 / 14 / 30              -0.0107      +0.0000            +0.0013
```

The symmetric trim is reported because drop-top-3 removes only gainers and is deliberately harsh;
it does not rescue anything. **Median per-sequence dR is 0.0000 on all three arms.**

The one real signal is `mask50`'s **mean IoU +0.0147**, the largest of any arm run this day, with
A and R both also positive — the "not a failure-rule artifact" signature. But mean IoU is the
metric that has ordered arms oppositely before, and R is the metric of record.

Free findings: over-masking has a sharp cliff (`mask25` on `car1` goes IoU 0.7225 -> 0.4089 and
loses at frame 375), and **PSR falls monotonically with mask width** (car1 48.2 / 42.8 / 38.3 /
31.0 at none / 70 / 50 / 35), which is the `PSR_GATE_MIN` coupling predicted in §2 — any arm that
ships needs the gate re-tuned on the same sweep.

**WHAT THIS DOES AND DOES NOT REFUTE.** It refutes the CHEAP stand-in: a fixed, box-shaped,
patch-centred mask that never adapts. CSR-DCF's own ablation prices a **uniform** box mask at
-21% EAO *relative to their estimated one*, so the uniform mask IS their weak arm — the >50%
figure is for removing a mask from a tracker built around one. **This is now the SECOND cheap
spatial-reliability stand-in to fail here** (every `TARGET_PADDING` below 2.0 was the first), and
two independent failures of the cheap version is the strongest available evidence that the value
in CSR-DCF is the ESTIMATION, not the masking. A per-frame mask estimated from colour
segmentation is a much larger piece of work than anything on this list and is not host-only in
spirit even if it is in fact.

**BUT THE BOARD-IMPLEMENTABLE MASK IS A DIFFERENT SHAPE, AND IT WINS.** The masks above are
Tukey (flat plateau + cosine roll-off) and a plateau is a rect, so their spectra are NOT sparse —
7 bins per axis for 99% of the energy, and truncating to 9 bins in 2D gives max error 0.30-0.50
on a [0,1] mask. **Only a HANN-SHAPED mask (plateau 0, taper 1.0) is exactly 9 bins**, which is
the Stage B2 identity and the only variant implementable host-side with no approximation. Swept
separately over the same 62:

**THE INVOCATION IS PART OF THE RESULT.** `--mask-taper` DEFAULTS TO 0.25, and `mask0` at the
default taper is a narrow raised cosine that reaches zero at 12.5% of the patch — 99 non-zero
bins per axis, only 47% of the energy in the 3 that survive truncation. The arm below needs
`--mask-taper 1.0` EXPLICITLY, which is the full-patch Hann and the only exactly-9-bin case.
Verified 2026-08-29 by re-running `surfing` both ways against the stored trajectories: taper 1.0
reproduces `mask62_hann9bin.json` with maxdiff 0.0, taper 0.25 does not (mean IoU 0.2546 against
0.1087). Re-running from a command line without the flag silently scores a different,
NOT-board-implementable arm.

```bash
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
    --arms rgb rgb-mask0 --sequence <seq> --eta 0.05 --psr-min 5.0 --mask-taper 1.0
```

```
arm                  A        R   tracked/19903   meanIoU        dA        dR
rgb             0.5394   0.2910            5792    0.1792
rgb-mask50      0.5413   0.3033            6037    0.1939   +0.0020   +0.0123
rgb-mask0       0.5048   0.3628            7221    0.2040   -0.0346   +0.0718
```

**dR +0.0718 is 3.6× the instrument's resolution and survives the symmetric trim (+0.0480)** —
the only arm all day that does either. The `dA −0.0346` trips this file's own artifact rule and
three checks clear it: mean IoU RISES where the mutant's fell; the hold rate moves only +0.64%
of frames; and on the frames BOTH arms survived (identical 5563-frame set) the gap is
**−0.0085**, i.e. three quarters of the accuracy loss is the harder frames a longer-surviving
arm is scored on. See `docs/thesis/evidence/arm_mask.md`.

**AND THE WINDOW THE BOARD WILL ACTUALLY RUN IS NOT QUITE THIS ONE.** `spatial_mask()` centres
the axis at `(n-1)/2`; the exact periodic Hann — `hanning_128.h`, the only one whose DFT is
REAL — is centred at `n/2`. Half a sample, `max|Δm| = 0.0123`, and on `tiger` it is worth mean
IoU 0.1715 (lost f107) against 0.2813 (lost f360). Re-swept over all 62 in the board form:
**dR +0.0601, trim +0.0409, common-prefix dA −0.0103, hold rate +0.71%** — the arm survives the
swap, at about 16% less gain. `docs/thesis/evidence/arm_mask.md` §2 carries the table; the
trajectories are `mask62_boardform.json`.

**So §2 does NOT drop below §3 — it becomes the proposed hardware build.** The refutation above
stands and is narrower than it first looked: it refutes the TIGHT box mask, not spatial
reliability. The mask machinery stays in the bench behind `-mask<N>`.

## 3. Channel reliability in Stage B3 — the method it was missing

CLAUDE.md's item 1, held up because per-channel *response maps* do not exist on the host:
`cmul_accum` sums channels in the AIE before the IFFT. **They are not needed.** Both halves
of a per-channel reliability are available in the frequency domain, per Parseval:

```
peak at the target centre   r_ch(0) = (1/N) * SUM_i  H_i * conj(F_i)
total response energy       ||r_ch||^2 = (1/N) * SUM_i |H_i F_i|^2
reliability                 w_ch = r_ch(0)^2 / ||r_ch||^2
```

Two scalar accumulators inside the existing per-channel loop of
`filter_update_quantize()`, over two arrays that loop is already streaming — **essentially
free**, and it folds straight into `chscale`, which today is `1/sqrt(energy)`. CSR-DCF price
their version at -12% EAO.

Bit-exactness note: this changes `chscale`, so it ends the run of bit-identical tracking by
construction. Use the FNV-1a run-state digest to separate "changed as intended" from
"changed elsewhere too".

## 4. Confidence-modulated learning rate — a weak signal used weakly

Measured here, and the number is reported rather than sold. Per losing run, PSR in the 5
frames before the loss against that run's own healthy median:

```
                              median   p25    p75    frac < 0.6x own median
pre-loss window (-5..0)        0.892  0.578  1.314          27%
control window (-20..-15)      1.004  0.812  1.233          11%      n = 256 runs
```

A 2.5x lift. **Too weak to gate on** — that would veto 11% of healthy frames to catch 27% of
losses, and `robustness_gap.md` already showed the absolute gate is the aftermath of a loss,
not its cause. Adequate to *scale* eta by: `eta_eff = eta * clamp(psr / psr_running_median)`.
~10 host lines, no new state beyond a running median. It attacks the measured mechanism
directly — a filter that accepts every frame at PSR ~19 while walking off the target has
nothing telling it the content it is locking onto is wrong.

## 5. Two-filter ensemble (TCLCFcpp's contribution) — take the cheap half first

The full form costs a second AIE bank: 16 more cmul invocations plus an IFFT, ~5-8 ms on a
26 ms frame. The cheap half needs no AIE at all: keep a long-term filter at eta ~ 0 on the
host and use it **only as a validator** — one dot product per channel at the already-selected
peak, per §3's identity — feeding §4's eta modulation. Most of the anti-drift value at ~0 ms.
Run the expensive half only if the validator's signal turns out to be strong.

## 6. A replacement feature bank — ranked BELOW where CLAUDE.md puts it, and NARROWED 2026-08-29

**Before reading this section: `docs/thesis/evidence/feature_bank.md` closes the cheap half of it.** Swapping
which network conv1 comes from is worth ~0.011-0.015 in R — below this bench's resolution, not
surviving a symmetric trim, and a RANDOM bank of matched row norms ties the pretrained one on
held-out PSR. "PCA a wider net to 16 better features" is worse: its objective, participation
ratio, is maximised by noise (random Gaussian 10.69 vs the shipping bank's 7.43). What survives
is the GEOMETRY argument below — channels, kernel, stride — which is an AIE rebuild.
**And the bench cannot separate pretrained from random, so it cannot rank two banks either.**


The receptive field is hard-wired at 3x3 stride 1 in the AIE kernel. Aggregation over that
map is refuted on both banks (`pooled_features.md`), so a weights export cannot buy HOG's
deformation tolerance: HOG's tolerance is the cell, and the cell is the thing that was
measured to be worth zero here. The only remaining receptive-field lever is the coarser grid
(the 64x64 `dec2` arm), which is a graph rebuild, re-package and re-flash.

`dec2`'s offline case is stronger than pad30's was — dR = +0.0567 on `vot_ar_offline`, 3x
that instrument's measured resolution, where pad30's case was largely a mean-IoU/px-bin
argument. But it is the same proxy on the same sequences, its aggregate mean-IoU gain is
carried by three of 62, and CLAUDE.md turned the reflash off for that reason. **Do not spend
the reflash until items 1-4, which are host-only, are exhausted.**

## What is NOT on this list, and why

- **Relaxing or re-tuning the gate.** 88% of vetoes are `NEGATIVE_PEAK`, which
  `PSR_GATE_MIN` cannot disable, and 95.8% land after the run is already at IoU <= 0.1.
- **Localisation** — sub-bin refinement, iterated re-detection, more search range. alpha =
  0.93 on targets that translate, 0.95-0.98 in every speed bucket to 0.30 target/frame.
- **The fixed-point pipeline.** Removing quantization makes tracking WORSE; rails correlate
  with IoU at -0.025; the shipping arm rails zero times over 101,564 frames.
- **Re-detection after a loss** — §0.
- **The freeze rate as an objective.** Unfrozen three ways now (`SIGMA=1`, `EPS_REL=0.1`,
  float features) and tracking got worse every time.

## Reproduce the two new measurements

Both read only the existing CSVs of one arm and take seconds. `scripts/vot_init_anatomy.py`
keys on `(job, frame)` — a bare frame index is not a key on a `FRAME_SOURCE=vot` CSV.

```bash
python3 scripts/vot_init_anatomy.py runs/vot/0827_1642-eta05_g5p0 --per-sequence
python3 scripts/vot_init_anatomy.py runs/vot/0827_1642-eta05_g5p0 --drift-warning
```

The control window is what makes the §4 number readable: the same statistic on the same runs
at a time when they are still healthy. Without it, "PSR is 0.89x its own median before a
loss" is unfalsifiable.
