# Where this tracker sits — published VOT-STb2022 baselines

Moved out of CLAUDE.md 2026-08-31; content unchanged.


**Directly comparable, which is unusual enough to state: same dataset, same anchor-based
multi-start protocol, and STb ground truth is axis-aligned boxes fitted to segmentation masks —
exactly what this harness scores.** Source: Kristan et al., *The Tenth VOT2022 Challenge
Results*, ECCVW 2022, Table 12 (41 trackers, EAO 0.602 down to 0.195).

| tracker | class | EAO | A | R |
|---|---|---|---|---|
| MixFormerL / DAMT | transformer (winners) | 0.602 | 0.831 / 0.776 | 0.859 / 0.887 |
| DiMP | deep DCF | 0.430 | 0.689 | 0.760 |
| ATOM | deep DCF | 0.386 | 0.668 | 0.716 |
| TCLCFcpp | CF ensemble, *explicitly embedded/CPU* | 0.267 | 0.550 | 0.598 |
| ASMS | mean-shift | 0.255 | 0.526 | 0.599 |
| CSRDCF | HOG/CN + spatial & channel reliability | 0.251 | 0.519 | 0.580 |
| KCF | kernelized DCF + HOG | 0.239 | 0.542 | 0.532 |
| LGT | part-based, last of 41 | 0.195 | 0.461 | 0.486 |
| **this tracker, SHIPPING (eta 0.05, gate 5.0)** | fixed-point MOSSE/DSST | **0.163** | **0.510** | **0.342** |

**The split is the finding and it is sharp: ACCURACY IS INSIDE THE CLASSICAL-DCF BAND,
ROBUSTNESS IS NOT IN THE TABLE AT ALL.** A = 0.510 sits 0.009 under CSRDCF and 0.032 under KCF,
above ANT (0.492) and LGT (0.461) — on target, its boxes are competitive. R = 0.342 is below
every one of the 41, and EAO follows robustness because EAO is dominated by how long runs
survive before the 10-consecutive-frame failure rule fires. Being embedded is no excuse:
TCLCFcpp is in exactly this niche at R = 0.598.

*(Only a full-62 row is quotable. A 57-sequence subset degrades A and R gracefully and its EAO
does NOT — the five missing sequences change the subsequence-length distribution, which is why
adding them once RAISED EAO while lowering A and R.)*

**THE LOSS MECHANISM IS ATTRIBUTED, from the CSVs, no board time** (`docs/thesis/evidence/robustness_gap.md`).
The gate is the AFTERMATH of a loss, not its cause: 88% of vetoes are `NEGATIVE_PEAK` — which
`PSR_GATE_MIN` cannot disable — and **95.8% land after the run is already at IoU ≤ 0.1**. In the
5 frames BEFORE each first loss (394 losing runs) the verdict is **ACCEPT 82.0% at median PSR
18.83**, box moving 1.88 px/frame. **It does not freeze into a loss and it does not jump — it
walks off the target confidently.** Confirmed from the other side by `detector_gain.md`: on
targets that genuinely translate the detector recovers 93% of the annotated motion (alpha 0.93,
0.95–0.98 per speed bucket), and on-target hold rate is only 1.8%, flat across speed. **Do not
spend a sweep relaxing the gate, and do not look for the fault in localisation.**

**THE MASK'S `NEGATIVE_PEAK` REDUCTION IS NOT A MECHANISM — checked 2026-08-31 and refuted.**
Aggregate `NEGATIVE_PEAK` falls 15.42% → 10.22% of evaluated frames under `FILTER_MASK=1`, which
looks like the lever the gate never had. Split by whether the tracker was ON TARGET it inverts:
on-target 3.07% → **3.87%** (the mask makes anti-correlation MORE common where it matters), lost
frames 25.03% → 16.13%, and in the 5 frames before the first loss it is also worse
(13.05% → 15.06%). The fall is 90% a rate change rather than composition, but that rate change
lives inside frames the run had already lost. Consistent with `robustness_gap.md` — the veto is
the aftermath of a loss. **And the mask does not prevent losses either**: 373 → 369 losing runs
of 419, median time-to-first-loss unchanged at 54 frames, later on 129 runs and earlier on 109.
What moves is the TAIL — mean time-to-loss 119.4 → 141.4 (+18%). The arm ships on EAO and its
explanation is open. `docs/thesis/evidence/spatial_mask.md`.

**What the baselines have that this does not**, in likely order of importance:
1. **Pooled features.** HOG is gradient-orientation histograms over cells — 31 dims, tolerant of
   deformation. This is 16 channels of 3x3 conv1, no pooling, participation ratio 4.94 gray /
   7.43 RGB — **and read that number with `docs/thesis/evidence/feature_bank.md` in hand: PR is maximised by
   NOISE (random Gaussian scores 10.69), so it ranks nothing, and the activation-space width is
   1.43, not 7.43.** The dimension comparison with HOG stands; the PR comparison does not. **NOTE: aggregating THIS bank was tested and is a null** (see the feature-geometry
   entry in [`settled.md`](settled.md)); the open question is a different bank, not a filter over this one.
2. **A spatial reliability map** (CSR-DCF's contribution). **Its cheap stand-in was tested and
   FAILED**: every `TARGET_PADDING` below 2.0 is worse offline, and 3.0 lost EAO on hardware. So
   background contamination is not the binding constraint, and CSR-DCF's ">50% EAO" ablation
   prices REMOVING a mask from a tracker built around one — not adding a crude one here.
3. **Channel reliability at detection.** Stage B3 normalises by ENERGY, not discriminative
   power. **Untested, host-only, and now the cheapest live candidate** (their ablation: -12%).
4. **A temporal ensemble** (TCLCF). eta 0.125 and eta 0.05 are already a short/long pair that win
   on different sequences; running both and selecting by PSR is host-only. Untested.
5. **KCF is kernelized** — but its own raw-vs-HOG gap is 0.451 -> 0.728 while DCF-vs-KCF on HOG
   is ~0.728 -> 0.732. **The gain is the features, not the kernel. Not worth the cmul path.**
6. **Neither gates.** They follow the peak every frame; this vetoes and freezes, and a hold
   longer than the recovery budget (median 6 frames) is an unrecoverable loss by construction.

Not a differentiator, but worth stating for the write-up: no tracker in that table estimates
aspect ratio either, so the axis-aligned-box penalty on deforming targets is shared.

