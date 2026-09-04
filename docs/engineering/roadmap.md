# Roadmap

**Status:** current · **Updated:** 2026-09-04 · **Scope:** what to try next, ranked, with the evidence behind each rank

## 2026-09-04 (later) — `SCALE_N=1` RAN, and the answer is the ARITHMETIC

**`R-16`, `evidence/fixed_point_cost.md`.** The section below is superseded on its top item and
on two of its factual premises; it is kept because its reasoning is what the run tested.

- **The answer is (a).** Disabling the DSST filter is a NULL — dR mean −0.0099, **trim-5 −0.0210**,
  P(dR<=0)=0.858, 19 sequences better against 29 worse. Pooled says +0.0055 and **inverts in
  sign**; do not quote it. With scale handling matched, float64 against fixed-point is dR mean
  +0.0312, **trim-5 +0.0102**, 35/12, P=**0.001** — stronger than the confounded `R-13` contrast.
- **SCOPE, and it is narrower than "quantization".** The twin runs the SAME int8 conv datapath,
  so this prices the fixed-point CORRELATION pipeline (cint16 FFT/IFFT, Q1.15, `H_SHIFT`) and
  NOT the int8 feature path. `settled.md` row 1 is contradicted; rows 2 and 3 stand.
- **It was NOT "one host-only flag, an scp rather than a card swap"** — that premise below was
  stale. The 0902 bitstream was no longer on disk and `CONV_RELU` reaches `AIE_FLAGS`, so it
  cost a full AIE rebuild, a re-flash and a regenerated xclbin (`v++` is not bit-reproducible).
- **`R-13`'s trim figure below (+0.0213 paired, "trim-stable") is CORRECTED**: its published
  drop-top-5 of +0.0222 was a TWO-SIDED trim and one-sided it is +0.0018. `R-13` was not
  trim-stable; `R-16` is.
- **The scale direction is now closed for a SECOND, independent reason.** `scale_oracle_bound.py`
  priced a perfect filter at +0.0023 R; this prices the harm of the broken one at ~0.

## 2026-09-04 — the offline path is validated, and `SCALE_N=1` is now the top item (SUPERSEDED ABOVE)

**New top of the list: `SCALE_N=1` on the board.** The float twin (`R-13`) beat the shipping arm
by +0.0213 R paired, which fired the pre-registered prediction from `settled.md` — but the twin
also has no scale filter, so **arithmetic and scale handling moved together** and the headline is
unattributable as it stands. `SCALE_N=1` gives *fixed-point + no scale filter*, the matched
comparison to the twin's *float + no scale filter*. It is one host-only flag, an scp rather than
a card swap, and it is the only thing that separates:

- **(a) the fixed-point pipeline costs robustness** — which overturns `settled.md`'s quantization
  entry, an entry that `M-14` says was due for re-screening anyway (it was scored on the 3x3
  mobilenet bank at 128x128 by single-start mean IoU); or
- **(b) the board's broken scale filter costs robustness** — a broken estimator being worse than
  no estimator, which nothing in this project has ever priced. `scale_oracle_bound.py` priced a
  PERFECT filter (+0.0023 R); it never priced the HARM of the broken one.

**The scale bracket narrowed (b) without closing it.** Fixed-scale and oracle-scale twins sit
within ~0 R of each other (dR trim-5 −0.0032), so for (b) to carry +0.0213 the broken filter has
to be substantially worse than BOTH ends of that bracket. Possible — a self-confirming drifting
estimate can sit outside the span — but it is now the more demanding branch.

**What the validation licenses.** `R-12`: CSRDCF reproduces its published row through this exact
path to EAO −0.0078, and the `oracle` control returns R = 1.0000 exactly. Host-only candidates
can now be scored by the toolkit on the real protocol instead of by `vot_ar_offline.py`'s
single-start proxy (~0.02 R resolution, two recorded sign inversions). **That matters most for
the one remaining robustness candidate, the training-sample memory (`R-06`), which is host-only.**

**Still open and unchanged by any of this:** in float, CSRDCF beats the twin by +0.0144 R
trimmed (P=0.018), so the robustness gap is not an artefact of the embedded implementation.

## 2026-09-03 — the mechanism check RAN: the gain is the RECTIFIER, not the bank

`runs/vot/0903_1227-l1lin/`, 62 sequences / 419 trajectories, 0 failures. The pre-registered linear
twin — same bank (identical weights md5), same geometry, `app.flagstamp` BYTE-IDENTICAL and
`aie.flagstamp` differing on exactly `CONV_RELU=1 -> 0` — lands at **EAO 0.1851 against 0.1960**.
Sec.12's falsifier said a twin near 0.196 would collapse `N-16` to "the bank helps"; it did not.
Paired R **+0.0447 mean, +0.0331 trim-3, +0.0274 trim-5, sign p 0.018, P(dR<=0)=0.000**, with A
moving the same way — no A/R trade, and the strongest per-sequence result on record here.
**`N-16` is confirmed ON HARDWARE**, so the conv layer is no longer provably redundant and the
thesis's conv-feature requirement is answered by measurement. The 200-frame calibration passed
first (`rails=0` on all 800 readings) and the budget was deliberately NOT re-tuned, which is what
keeps the delta attributable. **The bench transferred at 85%, its best rate.**
`../thesis/evidence/arm_l1relu.md` sec.14.

---

## 2026-09-03 — confidence-modulated eta is REFUTED, and the ensemble's cheap half goes with it

`runs/vot/0903_offline-ceta/`, offline, 62 sequences, 14 minutes. LMCF's high-confidence update as a
one-sided law `eta_eff = eta * clamp(conf/median(own past), lo, 1.0)`, on BOTH PSR and APCE.
**Every arm loses** (best −0.0108 against a +0.02 bar; floor 0.4 worse than floor 0.6, so the
direction is wrong) **and the MUTANT does not lose** — inverting the law is not worse than the
correct law (PSR −0.0047 pooled; APCE's +0.0133 pooled dies at trim-3 −0.0153, sign test 1.000).
**The statistic is inert and no mechanism survives**, unlike `N-20`. The within-run dip is REAL and
re-measured stronger than on the old arm (P[pre-loss < control] = 0.618, 29.6% vs 11.7% below 0.6x)
— so a detectable signal existed and acting on it still lost.
**Item 4 below (the two-filter ensemble) loses its cheap half**: the roadmap proposed a long-term
filter as a VALIDATOR feeding exactly this modulation. What is left is the EXPENSIVE form, two real
filters selected per frame — a second AIE bank, not host-only. `../thesis/evidence/confidence_eta.md`,
claims `N-22` / `O-03`.
**ROOT-CAUSED the same day (`N-23`), and the root cause outlives the arm:** PSR is NON-MONOTONE in
correctness — frames already lost are 70.2% of the PSR 0-10 band, 48.4% at 20-30, and **96.9%
above 50**, because the high-confidence lost frames are WELDED to static background (median box
motion **0.00 px** against the truth's 2.24 px/frame). A monotone control law is therefore
misspecified by construction, which is why the arm AND its mutant both failed.
**AND THE FOLLOW-UP THIS SUGGESTED IS ALREADY CLOSED.** The welded population looked detectable by
zero MOTION at high PSR with no confidence statistic; the prior question — is it a LEADING
indicator? — answers no: 0.70% of pre-loss frames against 4.59% after, and only 6 of 221 losing
runs show any welded frame before their loss. Same shape as the gate (`R-06`), and post-loss
information is worth nothing (`N-13`). **Confidence-derived per-frame statistics are CLOSED as a
class here** — pre-loss this tracker looks confident and MOVING, not frozen.

**AND THE TWO-FILTER ENSEMBLE WENT WITH IT (`N-24`, `O-03` CLOSED).** The M-13 prior question was
asked before building: a PURE OBSERVER long-term filter (bit-identical control, maxdiff 0) riding the
live trajectory shows that long/short PEAK DISAGREEMENT does not predict an imminent loss —
`P[healthy < doomed]` = **0.461** frozen, **0.555** slow, against PSR's 0.618 which is itself closed
as too weak. The filters do not agree (3.61 bins apart when healthy); the disagreement is simply
UNINFORMATIVE. A per-frame SELECTION rule therefore has no signal to use, so the expensive form —
a second AIE bank — is closed too, not merely deprioritised.

**WHAT IS LEFT FOR ROBUSTNESS IS ONE ITEM: THE TRAINING-SAMPLE MEMORY.** SRDCF/CSRDCF keep weighted
sample SETS; this keeps one running average, which is exactly the "walks off target confidently"
mechanism (`R-06`). It is the only remaining candidate that was never a confidence mechanism, which
is why none of 2026-09-03's results touch it.
**The warm-up was diagnosed from the shipping run's own logs and was NOT the limit** (N=20 and N=12
are the same arm): the running median is biased 1.86x high at k=1, and the relative statistic does
not separate doomed from healthy runs early (0.608 at f1, 0.461 by f12) where ABSOLUTE psr does
(0.82-0.85). The early population is init failures, which eta cannot fix (`N-02`).

---

## 2026-09-03 — the spatial mask is REFUTED on the shipping arm, and so is the chrel re-open

`runs/vot/0903_offline-l1mask/`, offline, 62 sequences, 8 minutes. **The mask was the proposed
hardware arm for the day and the pre-screen inverted it**: dR **+0.0601** on the old 3x3/16ch
bank against **−0.0127** paired on the shipping Layer-1 one (trim-3 −0.0358, P(dR<=0)=0.706,
35 of 62 tied); the k=2 width knob is worse on both trims. The `e_box` mechanism HELD on both
banks (0.6795 -> 0.9547, non-overlapping quartiles), so the projection works and the tracking
gets worse — **why it inverts is open.** `chrel05`, riding the same sweep, spends the one
re-open `channel_reliability.md` licensed: −0.0040 paired, 39 of 62 tied.
**Item 2 below (the spatial mask) and item 3 (channel reliability) are both CLOSED against
this arm.** `../thesis/evidence/mask_bank_transfer.md`, claims `N-21` / `M-14` / `N-20`.
**What is left for robustness is item 4, the two-filter temporal ensemble and the
confidence-modulated learning rate — the only untested mechanism that acts on the 71% of the
EAO window nothing has moved.**

---

## 2026-09-02 — the Layer-1 arm SHIPS, and the EAO window is now the constraint

`rgb_l1relu` (7x7 stride 2, resnet18-PCA bank, ReLU on, 32ch, 64x64 map) is the best arm on
record at **EAO 0.1960** and is the default. It did **not** meet its pre-registered `dEAO >=
+0.005` bar (+0.0029); the grounds for shipping it anyway are in
`../thesis/evidence/arm_l1relu.md` sec.12 and must never be written up as a pass.

| screened 2026-09-02 | verdict | note |
|---|---|---|
| Layer-1 7x7/2 + ReLU on hardware | **SHIPS**, best EAO; falsifier NOT met | `arm_l1relu.md` sec.10-12 |
| `out_shift` bound (`ACC_BOUND=l1`) | FIXED — `F_ch` 0.13% -> ~1% of int16 | sec.7-8 |
| conv2d generic-branch rework | 2.39x scheduled; frame 61.5 -> 24.1 ms | sec.8, COST block |
| `MOSSE_SIGMA` interior (22-cell grid) | **CLOSED** — 3, 5, 6, 8 all worse than 4 | `runs/vot/0902_offline-sigmaeta/` |
| `MOSSE_ETA` interior at sigma/target 1/16 | eta 0.1 the only trim-stable cell of 22 | board A/B in flight |
| `ARM=l1lin` linear twin | **OWED** — rebuild+reflash, decides N-16's attribution | — |
| `MOSSE_ETA=0.1` on the shipping arm | **REJECTED on hardware, EAO 0.1960 -> 0.1817**; the offline grid INVERTED | `arm_l1relu.md` sec.13 |
| aggregation on the RECTIFIED bank (2x2, blur x relu/lin) | **REFUTED, a LOSS −0.0242**; the linearity explanation WITHDRAWN | `pooled_features.md` |
| scale error before a loss | **CAUSAL**: >25% mis-sized on 60% of pre-loss frames vs 20-31% on survivors | `scale_filter.md` |
| scale detector, root cause | **LOCKED, not blind**: P(idx==0) 88.4% vs 3.0% for noise, conf FLAT, gain α −0.003 vs the position detector's 0.93; the sim's SAME estimator reaches 0.93 at speed, so the loop is SELF-CONFIRMING | `scale_filter.md` |
| **the whole SCALE direction** | **CLOSED. An ORACLE scale filter is worth +0.0023 R (and −0.0089 on `sigma4`)**; it lifts mean IoU +0.054 and converts none of it into survival. The filter is broken, understood, and NOT WORTH FIXING | `scale_filter.md`, `scripts/scale_oracle_bound.py` |

**THE RESULT THAT REFRAMES THE WHOLE LIST.** The toolkit's EAO window is [115, 755]. `l1relu`'s
+0.0184 pooled R became +0.0029 EAO because **the gain lives entirely below N ~ 300**: +0.0092
over the 29% of the window that is 115-300, and **+0.0005 over the 71% that is 301-755**. Better
FEATURES improve acquisition and mid-horizon survival and are then diluted threefold. What owns
the other 71% is long-horizon BOX QUALITY, and 2026-09-02 measured what that is: **the scale
estimate is FROZEN on ~90% of all frames** across 838 real runs, `scale_idx` is already 0 on 84%,
and a third of every non-zero decision is vetoed. A "diffusing scale" reading of the growing IQR
was tested and REFUTED (var 1.95x over a 20x span; a random walk predicts 20x) -- the spread is
survivorship plus many runs parked at their own offsets. **`SCALE_ETA` is inert over 0.025-0.3
and is CLOSED**; the freeze is a DETECTION failure. Untried: `SCALE_MAX_STEP=3`,
`SCALE_N`/`SCALE_STEP` (`max|idx|` hits the filter boundary), `SCALE_CONF_MIN`. Screen with
`make scale_sim`: `rgb_vs_gray_loop.py` has no DSST filter, the blind spot that sank `pad30`.
`scale_filter.md`.

---

## 2026-09-01 — the day's screens, and what each closed

`MOSSE_SIGMA=4.0` at 128x128 was the best arm on record that day (A 0.5133 / R 0.4095 / EAO 0.1931,
*superseded 2026-09-02 by `rgb_l1relu`, EAO 0.1960*;
host-only). Everything else screened that day was a null or borderline. Full detail in the
evidence notes; this is the index.

| screened | verdict | note |
|---|---|---|
| 64x64 feature map | CONFIRMED on hw, then RE-ATTRIBUTED | `arm_res64.md` sec.17-20, 25 |
| sigma/target sweep (1/64..1/8) | **1/16 is the optimum; sigma4 SHIPS** | `arm_res64.md` sec.21, 25 |
| `PSR_GATE_MIN=3.5` rescale | REJECTED — EAO null, and the "conditional on PSR scale" claim REFUTED | sec.22-23 |
| one-hot + orthonormal banks | the conv layer is a LINEAR LIFT; one-hot ties the network | `feature_bank.md` |
| ReLU / abs / CReLU on the shipping bank | refuted (−0.0332 / −0.0232 / +0.0020) | `feature_bank.md` |
| Stage B3 channel reliability | REJECTED — mechanism holds, gain does not | `channel_reliability.md` |
| Layer-1 banks x {16,32}ch x {7x7/2, 3x3+pool, 5x5/1} | mechanism CONFIRMED 4x, arms BORDERLINE | `layer1_features.md` |

*(That build ran on 2026-09-02 and now ships — see the section above. Its offline
P(dR<=0)=0.041 understated it: on hardware the paired R survives drop-top-5 at P=0.011.)*

**Three things that changed how to read the older entries below:**

1. **`MOSSE_SIGMA` is in BINS**, so any change to the map size silently changes `sigma/target`.
   The 64x64 arm's robustness gain was the sigma it carried, not its resolution. Every
   geometry arm scored before 2026-09-01 has this confound.
2. **px/bin is NOT the axis** `pooled_features.md` proposed — at matched sigma/target the two
   resolutions nearly agree, and on HARDWARE the finer map wins POOLED (a null when paired,
   `R-14`), which makes "nearly agree" the safer reading of both.
3. **The offline proxy's ACCURACY column is not usable** on response-shape arms: it predicted
   A −0.039 and −0.043 for the two arms that then gained +0.021 and +0.030 on hardware.

---

# Roadmap — what to try next, in order

Split out of CLAUDE.md 2026-08-31 and **maintained here since** — this file, not
CLAUDE.md, is where this topic is kept current; CLAUDE.md carries only the one-line
version and a link.

### Next, in order — ROBUSTNESS

Localisation, the gate, quantization, saturation, pooling, feature resolution, padding and
**init perturbations** are ALL exonerated or rejected (see [`settled.md`](settled.md)). Ranked list and the
supporting measurements: `docs/thesis/evidence/robustness_proposals.md`. What is left is host-only:

1. **THE NEXT BUILD: the 64x64 feature map (`PATCH_ROWS=PATCH_COLS=64`) — PROPOSED 2026-08-31,
   `docs/thesis/evidence/arm_res64.md`, claim N-03b.** The first arm whose offline
   signal survives a symmetric trim AND a bootstrap: dR **+0.1071**, drop-top-3 **+0.050**,
   P(dR<=0) = 0.000, against the shipped spatial mask's +0.0330 / +0.0101 on the same
   instrument. A RESOLUTION change, not a pooling one (`dec2` alone scores it). **Buys frame
   time too** — 0.25x the host work, ~15 ms/frame predicted against 28.58. Most of the build is
   free: `roi_crop` takes `patch_rows` as a RUNTIME AXI-Lite arg and its `.xo` is stamp-verified
   reusable, `hanning_64.h` already exists, `BUILD_DIR` is already keyed on geometry. **The real
   price is the shift budget** — the FFT gain falls ~4x at 64 points but Stage A renormalises
   over 4x fewer samples, so the net is NOT derivable; budget one 200-frame calibration run.
   Falsifier has TWO parts this time (dEAO >= +0.005 AND survives drop-top-3), because the mask
   cleared a one-part bar and was still not separable from a null.
   **PREPARED 2026-08-31 (sec.8-14 of that file), nothing built:** the graph COMPILES and PLACES
   at 64x64 (`make graph`, 0 errors, tile buffers 623.8 -> 254.5 KB, every window buffer halved),
   the PL `.xo` files are seeded into `build/hw/64x64/ch16` so HLS really is skipped, and
   `calib_build.sh` takes the geometry — it did NOT before, and would have verified the 128x128
   stamps while building 64x64. **The budget candidate is `FFT_SHIFT=3 IFFT_ROW_SHIFT=3
   IFFT_COL_SHIFT=3` with `H_SHIFT=15` unchanged** — one bit off each of the four shifts is
   exactly the halved point size and holds every intermediate magnitude, `F_ch` included, at the
   validated 128x128 values. sec.3's "2 bits, partially offset" is wrong twice over (Stage A is a
   FIXED-scale z-score, not a unit-L2 norm; the inverse pass loses gain as well): the response
   correction is 4 bits. Still a model, still decided by the one 200-frame `rails=0` run.
   **THE CALIBRATION CARD IMAGE IS BUILT AND EVERYTHING OFF-BOARD IS GREEN** (sec.14-16):
   `test_host`/`test_roi_crop` (both arms)/`test_scene`/`test_vot_source` pass at 64x64, all four
   `x86sim_check` kernels are BIT-EXACT at the new point size (conv2d s6 and s6rgb, cmul s7 and
   cmul_stress), `BUILD VERIFIED` on all 21 stamps, timing met (WNS +0.322 ns), PL utilisation
   byte-identical to the 128x128 build, image re-provisioned. **What remains needs the board:
   flash, the 200-frame calibration, then the sweep.**

2. **A spatial mask on the filter — SWEPT 2026-08-31. EAO ROSE, AND THE ARM IS STILL NOT
   DISTINGUISHABLE FROM A NULL.** `docs/thesis/evidence/spatial_mask.md`, claim R-10, `results/arms.csv` row
   `rgb_mask`. **EAO 0.1629 → 0.1740 (+0.0110), against a +0.005 bar written before the run.**
   **But the gain is carried by 3 sequences of 62 — the top 3 are 133% of it, dropping them FLIPS the sign to −0.0030, the per-sequence median dR is 0.0000 and the bootstrap CI is [−0.0135, +0.0318] with P(dR≤0)=0.22.** EAO cannot be bootstrapped (one value per tracker), so the arbiter has no stability estimate. **Quote it with that caveat, never as a clean win.** R 0.3417 → 0.3608, 16.6% more frames tracked, and the pooled A drop (−0.0187) is a SELECTION
   effect: on the common survived prefix A goes **0.5319 → 0.5499, i.e. +0.0179**. Mechanism
   confirmed by `mask_ebox`, which separates the arms at init 0.6049 → 0.9500 with
   non-overlapping quartiles. Control: the paired baseline's 419 trajectories are BYTE-IDENTICAL
   to the stored shipping arm, so `FILTER_MASK=0` is inert on hardware and the instrument
   perturbs nothing. **Offline over-predicted the gain 3× (dR +0.0601 against +0.0192) — quote
   the hardware delta.** **THE PREDICTED `PSR_GATE_MIN` RE-TUNE IS NOT SUPPORTED BY THE HARDWARE DATA — do
   not spend a 100-minute sweep on it.** Masking DID move the PSR scale, by half what offline
   predicted (median 30.87 → 27.18, −12%, against a predicted −24%), but **the gate's BITE did
   not move**: `LOW_PSR` is 0.10% of evaluated frames on the baseline and 0.09% under the mask,
   and the fraction below 7 FELL (1.56% → 1.32%). The threshold has almost no lever left —
   moving it 5.0 → 3.0 flips 161 frames (0.09% of evaluated), 5.0 → 7.0 flips 1966 (1.09%).
   99.1% of vetoes are `NEGATIVE_PEAK`, which `PSR_GATE_MIN` cannot disable. (Those flip counts
   are OPEN-LOOP — a flipped decision changes every later frame — so read them as how much room
   the knob has, not as a predicted outcome.)
   **`calib_build.sh` CANNOT build either arm** — it accepts only `ARM=gray|rgb`, never passes
   `FILTER_MASK`, and would print `BUILD VERIFIED` on a build that silently omitted it. Use
   `make application` with the flags spelled out; see `evidence/spatial_mask.md`.
   *(The pre-sweep entry follows; its numbers are OFFLINE predictions, not results.)*
   **The window swept on 08-28 is NOT the one the board runs** — the bench centred its axis at
   `(n-1)/2`, the hardware's periodic Hann at `n/2`; half a sample, worth mean IoU 0.1715 vs
   0.2813 on `tiger`. Re-swept over all 62 in the board form: **dR +0.0601** (was +0.0718),
   trim +0.0409, mean IoU 0.1792 → 0.2016, common-prefix dA −0.0103. Still 3.0× the
   instrument's resolution, and every check that cleared the old arm clears this one.
   `rgb_vs_gray_loop.py --mask-center board|bench` now carries both, defaulting to board.
   **`--mask-taper 1.0` is REQUIRED and is not the default** — at the 0.25 default `mask0` has
   99 non-zero bins per axis and is not board-implementable.
   *(Original 08-28 entry follows.)*
   **A spatial mask on the filter — MEASURED OFFLINE 2026-08-28.** `docs/thesis/evidence/arm_mask.md`. CSR-DCF's highest-priced item, applied
   as a one-shot projection `h ← m⊙h` on the published `H` (`A`/`B` untouched), so the AIE sees
   an already-masked filter and detection stays consistent. Host-only — an scp, not a card swap.
   62 sequences, shipping eta/gate, `vot_ar_offline`:

   ```
   arm                  A        R   tracked/19903   meanIoU        dA        dR
   rgb             0.5394   0.2910            5792    0.1792
   rgb-mask50      0.5413   0.3033            6037    0.1939   +0.0020   +0.0123   (Tukey)
   rgb-mask0       0.5048   0.3628            7221    0.2040   -0.0346   +0.0718   (Hann)
   ```

   **ONLY A HANN-SHAPED MASK IS EXACTLY 9 BINS, and this was written down wrong here first.**
   The 9-bin claim comes from the Stage B2 identity and holds for a periodic Hann (3 bins per
   axis, zero error). A **Tukey** mask — flat plateau plus roll-off — has a rect in it, and a
   rect transforms to a sinc: 7 bins per axis for 99% of the energy, and truncating to 9 bins in
   2D gives max error 0.30-0.50 on a mask whose range is [0,1]. So the tight-box arms that were
   swept first are NOT board-implementable without a 49-tap convolution and its own truncation
   study, and **the exactly-sparse shape is a much wider, smoother mask that behaves completely
   differently.** Checking the claim is what found the result.
   **dR +0.0718 is 3.6× the instrument's measured resolution and survives a symmetric trim
   (+0.0480)** — the only arm of that day's twelve to do either. `dA −0.0346` trips the
   failure-rule artifact test in [`measurement.md`](measurement.md) and three checks clear it: mean IoU RISES where the `gsign`
   mutant's fell; the hold rate moves only +0.64% of frames; and **on the frames BOTH arms
   survived the gap is −0.0085** (see the selection-effect rule under Measurement methodology).
   **The board is still required and not as a formality**: EAO is not computable offline and EAO
   is the arbiter for an A/R trade (pad30 lost EAO while gaining R), and this is SINGLE-START
   where the toolkit is 419 anchored runs — a mask concentrates its effect in a filter's early
   life, so the protocol may cut either way. Median per-sequence dR is 0.0000: 35 of 62 are
   untouched and the gain lives in 18.
   **Two things measured before the arm, both load-bearing.** The filter's energy is centred at
   the **PATCH CENTRE** (peak of `Σ|h|²` at (64,64); a corner-wrapped 64×64 box holds 8-12%),
   even though `resp[0,0]` is the zero-displacement bin — masking at the response origin deletes
   the filter and reads as "masking hurts". And a centred 64×64 box, exactly the target box at
   padding 2, holds only **51.6% (`car1`) / 54.9% (`tiger`)** of the filter's energy, which is
   CSR-DCF's complaint quantified on this design.
   **NOT `TARGET_PADDING` by another name**: padding moves the mask, px/bin, the search range and
   the DSST extraction region at once, which is why pad30 was unattributable. **Expect to re-tune
   `PSR_GATE_MIN` on a second sweep** — masking moves the PSR scale (car1 mean PSR 48.2 → 36.7),
   and the gate's worth is conditional on that scale. One variable at a time: mask first at the
   inherited gate, then the gate.
3. **Channel reliability in Stage B3, and it needs no per-channel response map.** B3 normalises
   by ENERGY, not discriminative power; the blocker was that `cmul_accum` sums channels before
   the IFFT. Both halves are available in the frequency domain by Parseval:
   `r_ch(0) = (1/N)·Σᵢ Hᵢ·conj(Fᵢ)` and `‖r_ch‖² = (1/N)·Σᵢ|Hᵢ Fᵢ|²`, i.e. two scalar
   accumulators inside the existing per-channel loop of `filter_update_quantize` over arrays it
   already streams. Reliability `r(0)²/‖r‖²` folds into `chscale`. CSR-DCF price theirs at −12%.
4. ~~**A two-filter temporal ensemble**~~ — **CLOSED 2026-09-03, both halves (`N-22`, `N-24`,
   `O-03`).** The cheap half was a long-term filter used as a confidence VALIDATOR feeding a
   modulated eta: the modulation is refuted on PSR and APCE with its mutant failing to lose. The
   premise underneath — that a second memory sees drift a single filter does not — is refuted
   directly by a pure-observer probe: peak disagreement predicts an imminent loss at AUC
   0.461 (frozen) / 0.555 (slow) against PSR's already-too-weak 0.618. Nothing is left for a
   per-frame selection rule to use. `../thesis/evidence/confidence_eta.md`.
5. **A REPLACEMENT feature bank — NARROWED SHARPLY 2026-08-29, see `docs/thesis/evidence/feature_bank.md`.**
   **Swapping the donor network is close to a null: at layer 1 the pretraining is worth
   ~0.011-0.015 in R, below the offline bench's own resolution, and it does not survive a
   symmetric trim.** A RANDOM bank of matched row norms ties the pretrained one on held-out PSR
   across 4 sequences and scores A 0.5651 / R 0.2801 against 0.5394 / 0.2910 over all 62 (two
   seeds; seed spread 0.0038). **And "PCA a wider net down to 16 better features" rests on a
   statistic that is MAXIMISED BY NOISE** — random Gaussian 16x27 scores participation ratio
   10.69 against the shipping bank's 7.43, a random orthonormal basis the maximum 16.00, and on
   real activations 1.99 against 1.43. (The PCA *mechanism* is sound and free — a linear map of
   conv outputs folds into the weights and stays a 3x3 conv — it is the objective that fails.)
   Also: no torchvision 3x3x3 stem is better conditioned; the widest, vgg16_bn at 64 channels,
   is the WORST (PR 5.17). What is NOT refuted is the GEOMETRY — 16ch / 3x3 / stride 1 — which
   is an AIE rebuild, not a weights export. **Corollary, and it costs more than the arm: the
   offline AR bench cannot separate a pretrained bank from noise, so it cannot rank two
   pretrained banks either.**

**Recovery AFTER a loss is worth NOTHING to R or EAO** — VOT terminates a run 10 frames after
failure and re-enters at the next anchor. Only not-losing, or recovering inside the grace,
scores. That retires re-detection and search-window expansion outright, however good they look
on mean IoU.

**Score any of these on `vot analysis`, never on mean IoU** — the two have ordered arms
oppositely on identical trajectories (`HOLD_COAST`), and the offline AR proxy
(`vot_ar_offline.py`) has a MEASURED resolution of only ~0.02 in R and did not transfer to a
geometry arm.


### Next, in order — PERFORMANCE

**RE-RANKED 2026-09-04 by the shipping arm's first per-stage measurement**
([`../thesis/evidence/frame_time_shipping.md`](../thesis/evidence/frame_time_shipping.md)).
The list below was built on the 128x128 / ch16 / 3x3 arm and its premise no longer holds:

0. **`roi_crop`'s `ap_done` POLL — 7.495 ms/frame, 28.7% of the frame, the single largest item.**
   0.23 ms per channel x 32 channels, 4386 poll iterations, and it is a **busy-wait**: the host
   spins on the PL's completion flag. It grew 7.3x from the ch16 arm's 1.013 ms because the crop
   is 128x128 (4x the output pixels) across twice the channels. Overlap it with the host's
   per-channel work, exactly as `roi_crop` itself was pipelined on 08-21 (5.196 -> 1.020 ms), or
   make the wait sleep. **This subsumes item 1 below and is a bigger, simpler target.**
   Note `scale extract` (2.211 ms) is **NOT** the head of the tail on this arm — `arm_res64.md`
   sec.19.5 concluded that from the ch16 arm and it does not transfer.

The APU was a **flat tail** on the ch16 arm — biggest single item 5.2 ms — so the remaining wins
were structural. On the shipping arm the APU wall is only 7.0 ms of 24.9 and the tail is flatter
still; the frame is now `roi_crop` + GMIO.

1. **Software-pipeline the CHANNEL loop.** The `fft_col_out` + `accum_out` pair (~8.7 ms) is the
   col-FFT + cmul production time for 16 channels, proven immune to transaction count. Overlap it
   with the host's ~0.4 ms/channel of APU work, exactly as `roi_crop` was pipelined
   (5.196 → 1.020 ms). The graph already permits one channel of lookahead; the host serialises.
2. **More of the second core.** Parallel-for over channels inside `filter_update_quantize` —
   expect well under 2×, it is memory-bound and the two cores share a controller. **See the abandoned parallel-for attempt in
   [`performance.md`](performance.md) before trying.** Then re-enable `CMUL_ACCUM_MEMTILE`, which only pays
   once a helper exists to absorb the wait (alone it loses 0.36 ms; paired, ~2.7 ms of freed CPU).
3. **NEON-vectorise the int16→float conversion in `unpack_spectrum`** (`SSHLL` + `SCVTF`) — maybe
   1 ms, and the only remaining idea for that slot.
4. **6.6 ms/frame of XRT descriptor cost** — the two 256-tx ports cost 11.0 µs of host CPU per
   `async()`. The only lever is fewer, larger transactions, i.e. item 2's memtile.
5. **At `CONV_IN_CH=3` only: `frame push` 1.385 ms.** The 6 MB `frame_bo` is now the single
   biggest RGB-specific cost — bigger than everything conv2d added to the frame.

**Retired, do not reopen:** `FFT_COL_WS` 8→32 (a 9.57 ms loss), `CMUL_ACCUM_MEMTILE` alone (0.36
ms), Hermitian symmetry in the host filter (premise refuted in fixed point), the accumulator as a
`shared_buffer`, parallel-for inside `filter_update_quantize` as attempted. Each has an entry in [`performance.md`](performance.md) or [`settled.md`](settled.md).

