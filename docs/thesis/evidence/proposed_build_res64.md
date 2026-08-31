# Proposed hardware build — the 64x64 feature map (`PATCH_ROWS=PATCH_COLS=64`)

**2026-08-31. Status: PROPOSED, not built.** Written so the decision to spend board time is made
against a falsifier and a cost, not against a hope. Claim `N-03b` in `docs/thesis/claims.md`.
Evidence: `evidence/pooled_features.md` (both the px/bin section and the 2026-08-31 max-pool
section). This is the FIRST arm in this project whose offline signal survives a symmetric trim
and a bootstrap.

## 1. Why this is worth board time

62 sequences, RGB, shipping eta 0.05 / gate 5.0, `vot_ar_offline.py`, baseline re-run in the
same invocation and reproducing the recorded board-form control EXACTLY:

```
arm                  A        R   tracked        dR        dA
rgb             0.5394   0.2910      5792                        <- control
rgb-dec2        0.5005   0.3981      7923   +0.1071   -0.0389

per sequence      mean dR   median    trim3  drop-top3            95% CI   P(dR<=0)
rgb-dec2          +0.0826  +0.0000  +0.0648    +0.0503  [+0.0312, +0.1403]    0.000
(shipped mask)    +0.0330  +0.0000  +0.0231    +0.0101                        --
```

**It survives dropping the top-3 gainers at +0.050 — five times the shipped spatial mask's
+0.010 — and its bootstrap CI excludes zero.** Every other arm measured on this bench, including
the one that shipped, is carried by a handful of sequences.

**IT IS A RESOLUTION CHANGE, NOT A POOLING ONE.** Plain `dec2` (subsample, no aggregation)
scores the same as `pool2` (average) and `mpool2` (max) to within 0.001, so nothing needs to
aggregate and conv2d's arithmetic does not change. It is also the geometry Danilowicz & Kryjak
2022 ship (128x128 ROI -> 64x64 filter); no value in their table is comparable to ours (VOT2015,
and their R is an inverted failure count), so that is corroboration of the GEOMETRY only.

**And it is the rare arm that buys robustness AND frame rate**: host work scales with
`channels x map_elems`, so a quarter-size map is 0.25x the filter update, GMIO and filter state
(2 MB -> 0.5 MB). Predicted ~15 ms/frame against today's 28.58.

## 2. The build — what does NOT rebuild is most of it

```
PATCH_ROWS=64 PATCH_COLS=64        # the only intended change
TARGET_PADDING=2.0                 # HELD -- see sec.5, this is what keeps the arm readable
CONV_IN_CH=3  MOSSE_ETA=0.05  PSR_GATE_MIN=5.0  SCALE_STEP=1.04  SCALE_MAX_STEP=2
HOLD_COAST=0  FILTER_MASK=0        # mask OFF: one variable at a time
FRAME_SOURCE=vot VERBOSITY=0 PROGRESS_EVERY=25 DUMP_BUFFERS=0 CSV_FLUSH_EVERY=200
FFT_SHIFT / IFFT_ROW_SHIFT / IFFT_COL_SHIFT -- NOT 4-4-4, see sec.3
```

Free, and each verified rather than assumed:

| item | why it is free |
|---|---|
| `roi_crop` | `patch_rows`/`patch_cols` are runtime **AXI-Lite** args; `ROI_MAX_PATCH_ROWS 128` is a MAXIMUM, so 64 is in range. Its stamp is `--hls.clock ... -D ROI_IN_CH=3` — no `PATCH_ROWS` dependency, so the existing `.xo` is stamp-verified reusable |
| `camera_capture` | no patch dependency |
| `hanning_64.h` | ALREADY EXISTS in `design/aie_src/` |
| FFT windowing | `PATCH % FFT_ROW_WS == 0` and the DSPLib whole-window rule both hold at 64 with `FFT_ROW_WS=64`, `FFT_COL_WS=8` unchanged |
| `BUILD_DIR` | already keyed on geometry (`build/hw/64x64/ch16`), so it CANNOT clobber the 128 build |
| PLIO alignment | 64*64*3 = 12288 divides by 4 |

Not free: `libadf.a` (64-point FFTs, conv2d/cmul window sizes), `v++ --link`, `--package`, a
reflash — and **re-provisioning after packaging**, because `v++ --package` takes
`rootfs_compat.ext4` and a re-package inherits whatever that copy holds. A board that boots
unreachable reads as a cable fault.

**Answer the placement question with `make graph` (5 min), not with a 25-minute package.**

## 3. THE REAL PRICE IS THE SHIFT BUDGET, AND IT IS NOT DERIVABLE HERE

The 2-D FFT gain falls ~4x at 64 points — `row_dc = PATCH_COLS*c - 21`,
`accum0 = PATCH_ROWS*row_dc - 21` — which is 2 bits off the total-16 budget and points at 4-3-3
or 3-4-4. **But Stage A renormalises to unit L2 over 4x FEWER samples, which partially offsets
it, and the net is not derivable from first principles.** Three self-consistent offline models
have been overturned by their premises in this project; do not let a fourth set a calibrated
constant.

**Budget ONE 200-frame `calib_build.sh` run with `calib_report.py`, criterion `rails = 0`.** The
five-run campaign of 2026-08-24 was for an unknown budget; this is a bounded 2-bit correction
with a known direction. `H_SHIFT` may move with it — it is the only knob upstream of BOTH the
accumulator and the response.

## 4. Falsifier, written BEFORE the run

**Prediction.** The instrument over-predicted the spatial mask 3.1x (offline dR +0.0601 against
hardware +0.0192). Applying that same discount to +0.1071 predicts **hardware dR ~ +0.035**
(range +0.010 to +0.055), and by the mask's dR-to-EAO ratio, **dEAO ~ +0.020**.

**Accept only on `vot analysis`, and this time the bar has TWO parts** — the spatial mask
cleared a one-part bar and then turned out not to be separable from a null:

1. **`dEAO >= +0.005`**, and
2. **the per-sequence dR must survive dropping the top-3 gainers.** The mask arm failed this
   (+0.0192 pooled, but drop-top-3 flipped the sign to -0.0030, and the bootstrap CI included
   zero at P(dR<=0)=0.22). **A pooled gain carried by 3 of 62 sequences is not a result.**
   Compute it as in `spatial_mask.md`; EAO itself cannot be bootstrapped (the toolkit reports
   one value per tracker), so R carries the stability evidence.

**Accuracy.** Expect `dA` about -0.04 POOLED and near zero or positive on the COMMON SURVIVED
PREFIX — an arm tracking ~35% more frames is scored on a harder set. Score the prefix before
calling any accuracy drop real; do not accept `dR > 0 with dA < -0.02` pooled as a win without
it (the `gsign` artifact rule).

**Frame time is a SECOND claim and needs its own check.** ~15 ms/frame is predicted from the
`channels x map_elems` host model. **Quote FPS only from a serial-console run**, never from the
ssh sweep, and take the breakdown from the `AP_*` slots.

## 5. Known couplings, and one deliberate non-change

- **`TARGET_PADDING` STAYS AT 2.0, AND THAT IS WHAT MAKES THIS ARM READABLE.** The pad30 proxy
  failed for a NAMED reason: the offline loop has no DSST scale filter, and padding sets the ROI
  `scale_extract` draws from, so the whole scale axis was invisible by construction. Holding
  padding fixed removes exactly that blindness. Changing both at once re-creates it.
- **The PSR exclusion stays 11x11 and that is correct**, not an oversight: the exclusion must be
  a fixed number of SIGMAS, `MOSSE_SIGMA` stays 2.0 BINS, so at 64x64 it is still the same
  mainlobe. (The offline bench holds sigma in bins across arms for the same reason.)
- **PSR SCALE WILL MOVE ANYWAY** — a 64x64 map has a quarter of the sidelobe population — and
  `PSR_GATE_MIN`'s worth is conditional on the PSR scale. Do NOT pre-emptively re-tune it: the
  mask arm's predicted gate re-tune turned out to be unsupported (`LOW_PSR` was 0.10% of frames
  and did not move). Measure the gate's bite from `track.csv` first; a re-tune is only warranted
  if the `LOW_PSR` share actually moves.
- **`FILTER_MASK=0`.** The mask ships but is not separable from a null; stacking it here would
  make an unreadable arm out of a readable one.
- **STATED CONFOUND: the two arms are at different xclbins.** After reflashing to 64x64, the
  128x128 baseline cannot be re-run without another reflash, so the comparison is against the
  STORED shipping arm. That is legitimate — `base_stat` reproduced it byte-for-byte on
  2026-08-31 — but it is the same class of confound as the gray/RGB `H_SHIFT` difference and
  must be stated wherever the result is quoted.

## 6. Cost

- `make graph` 5 min (placement + tile memory).
- Package + flash + re-provision: ~1 h. HLS synthesis is SKIPPED (the `.xo` files are reusable).
- One 200-frame calibration run for the shift budget, plus `calib_report.py`.
- ONE 62-sequence sweep (~100 min) — the baseline is the stored shipping arm, so only the new
  geometry has to run.
- Risk: MEDIUM. Unlike every host-only arm this year it needs a reflash, and the shift budget is
  genuinely open. The upside is that it is the only candidate that improves robustness and frame
  time together.

## 7. If it is a null

Then feature GEOMETRY joins weights, pooling, padding, quantization, localisation and filter
support on the refuted list, and the conclusion for the write-up is sharp: **within this
architecture the tracker is at its ceiling, and the remaining gap to the classical-DCF band is
not reachable by tuning the front end.** That is a defensible thesis result and it retires O-04
rather than leaving it as an untested "we could have tried a wider map".
