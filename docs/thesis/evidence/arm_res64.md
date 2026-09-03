# Proposed hardware build — the 64x64 feature map (`PATCH_ROWS=PATCH_COLS=64`)

**Status:** closed · **Updated:** 2026-09-01 · **Scope:** the 64x64 feature map: pre-registered, confirmed on hardware, then RE-ATTRIBUTED to mainlobe width

**WHERE THIS ENDED UP.** Built and swept 2026-09-01: the arm was CONFIRMED (A 0.5100 ->
0.5336, R 0.3417 -> 0.3873, EAO 0.1629 -> **0.1849**, sec.18) and then **RE-ATTRIBUTED the
same day (sec.25): the gain was the MAINLOBE WIDTH the arm carried by accident, not the
resolution**, which is a small loss at matched width. Superseded as the best arm by
`arm_l1relu.md` on 2026-09-02. Everything up to sec.16 is the proposal as written.

**2026-08-31, as proposed. Status at the time: PROPOSED, not built.** Written so the decision to spend board time is made
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

**SUPERSEDED IN PART BY sec.11 — read that before using any number in this section.**
(And sec.11's own "52% / 49%" comparator is WRONG — it is `run_0824_1354`, the old railing
budget. See sec.17.) Both
premises below are wrong: Stage A emits a z-score at a FIXED scale, not a unit-L2 norm, so there
is no offset; and the inverse pass loses gain too, so the correction is 4 bits on the response,
not 2. The candidate budget is `3-3-3` with `H_SHIFT=15` unchanged. The paragraph is kept as
written because the run has not happened and the warning in it still stands.

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

---

# BUILD PREP — 2026-08-31

Everything below is preparation, not a result. No hardware has run at this geometry.

## 8. The graph is verified: it compiles, places, and the tile memory falls

`make graph TARGET=hw PATCH_ROWS=64 PATCH_COLS=64 CONV_IN_CH=3 ...` — **0 errors, `libadf.a`
produced**, ~5 min, into `build/hw/64x64/ch16/` (the 128x128 build untouched, as `BUILD_DIR`
promised). Run at BOTH candidate budgets, `4-3-3 / H_SHIFT=13` and `3-3-3 / H_SHIFT=15`; the
directory is left at the second, so its `aie.flagstamp` records the budget sec.11 recommends and
not a retired one.

The placement question sec.2 flagged is answered, and the geometry demonstrably reached the
kernels rather than being silently ignored — every window buffer halved:

| buffer | 128x128 | 64x64 |
|---|---|---|
| conv2d / row-FFT / row-IFFT window (`po0`, ping-pong ×2) | 32768 B | 16384 B |
| memory-tile transpose (×4) | 65536 B | 16384 B |
| all buffers in `Map_Report.csv` | 623 844 B | 254 532 B |

**Do not read the `fft_128_tmp*` buffer names as a point size** — they are DSPLib's internal
scratch labels and are 1024 B in BOTH builds. The window sizes are the evidence that
`PATCH_ROWS`/`PATCH_COLS` propagated.

## 9. The PL kernels are seeded, so HLS synthesis really is skipped

`BUILD_DIR` is keyed on geometry, which is what stops the arms clobbering each other — but it
also means `make` would re-synthesize `roi_crop` into the empty `build/hw/64x64/ch16/`, an hour
for a bit-identical `.xo`. Verified that neither kernel's flags carry geometry
(`CROP_VPP_FLAGS = --hls.clock 312500000:roi_crop -D ROI_IN_CH=3`, `CAM_VPP_FLAGS` likewise),
then copied `roi_crop.hw.xo`, `camera_capture.hw.xo` and `crop.flagstamp` across from
`build/hw/128x128/ch16/`. The flagstamp is copied with its ORIGINAL mtime (`cp -p`): the
`%.flagstamp: FORCE` recipe rewrites the file only when the content changes, and GNU make
re-stats the target afterwards, so an unchanged stamp does not retrigger the `.xo`. Confirmed
on a reduced Makefile rather than assumed — `make -n` is NOT a valid check here, it reports a
rebuild for the 128x128 build too.

## 10. `scripts/calib_build.sh` can now build this arm — it could not before

It hardcoded `BUILD_DIR=build/hw/128x128/ch16` and never passed `PATCH_ROWS`/`PATCH_COLS`, so
`PATCH_ROWS=64 scripts/calib_build.sh` would have built `build/hw/64x64/ch16` and then verified
the flagstamps of the 128x128 build sitting next to it — **printing `BUILD VERIFIED` for a
binary it never looked at.** Same class as the stale `libadf.a` and as `FILTER_MASK`, which it
still cannot build. Now:

- `PATCH_ROWS`/`PATCH_COLS`/`N_CHANNELS` are read from the Makefile via `print-%` (never
  copied), passed in `VARS`, and `BUILD_DIR` is derived from them;
- both flagstamps are checked for the geometry;
- pre-flight rejects a non-square or non-power-of-2 patch, a missing `hanning_<N>.h`, and a
  patch larger than `ROI_MAX_PATCH_ROWS`;
- at any geometry other than 128x128 it prints that the stored comparator does NOT apply and
  that this is a new budget.

```
PATCH_ROWS=64 PATCH_COLS=64 ARM=rgb FFT_SHIFT=3 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3 \
  scripts/calib_build.sh          # DRY_RUN=1 first
```

`hanning_64.h` was checked numerically against `round(sin^2(pi i/64)*32767)`, all 64 entries —
it is the periodic form and it is current, so `make weights` does not need re-running for the
window. The weights file itself is patch-independent (RGB tag already correct).

## 11. sec.3's premise is wrong in two ways, and the correction has a DIRECTION

Sec.3 says the FFT gain falls ~4x, "2 bits off the total-16 budget", *partially offset* because
"Stage A renormalises to unit L2 over 4x FEWER samples". Both halves need correcting:

1. **Stage A does not normalise to unit L2.** `roi_crop.h` says so explicitly and
   `roi_crop.cpp` implements it: it emits the z-score `(x-mu)*ROI_NORM_Q/sigma`, with
   `ROI_NORM_Q = 32` fixed — a unit-L2 patch "quantizes to 0 in int8" is the reason the code
   gives. The per-sample scale is therefore **independent of the patch size**, and there is
   **no offsetting gain at all**.
2. **The inverse pass loses gain too.** Sec.3 counts only the forward transform. All four
   passes sum over N instead of 2N, so at fixed shifts:

```
|F|      ~ N_r*N_c / 2^(2*FFT_SHIFT)                          -> 2 bits smaller at 64
accum    ~ |F| * 32767 / 2^H_SHIFT                            -> 2 bits smaller  (H is
                                                                 renormalised to Q1.15 full
                                                                 scale, so it carries no
                                                                 scale of its own)
response ~ accum * N_r*N_c / 2^(IFFT_ROW+IFFT_COL)            -> 4 bits smaller
```

**So the correction is 4 bits on the response and 2 on the accumulator, not 2 and 0.** Held at
4-4-4/15, this build would not rail — it would run the response 4 bits QUIETER than the
validated arm, which is the safe direction but throws away precision the PSR is read from.

**Candidate for the one calibration run: `FFT_SHIFT=3 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3`,
`H_SHIFT=15` unchanged.** Subtracting one from each of the four shifts is exactly the halved
point size, and it makes **every intermediate magnitude identical to the 128x128 arm**:

```
                          128x128 @ 4-4-4         64x64 @ 3-3-3
row-FFT output gain       128 / 2^4 = 8           64 / 2^3 = 8
col-FFT output gain       128 / 2^4 = 8           64 / 2^3 = 8
accum   (2*FFT + H)       2*4 + 15 = 23           2*3 + 15 = 21, against |F| 0 bits smaller
response(2*FFT+IFFTs+H)   31                      27, against N^2 4 bits smaller
```

That includes `F_ch`, which is what the host's filter update reads — a reason to prefer this
over the arithmetically equivalent `4-3-3 / H_SHIFT=13`, where `F` itself would be 2 bits
quieter. `4-3-3 / H_SHIFT=13` is the fallback if `FFT_SHIFT=3` turns out to rail `F` inside the
AIE (it should not: at 64 points, `FFT_SHIFT=3` restores `F` to the magnitude that measured
`rails=0` at 128).

**This is a fourth offline model and sec.3's warning applies to it in full.** It is written down
BEFORE the run, as a prediction with a direction, not as a calibrated constant: the 200-frame
`calib_build.sh` run with `calib_report.py` decides, criterion `rails = 0` plus amplitudes near
the comparator's (accum 52%, response 49% of int16). If it rails, the fix is `H_SHIFT` up, as
it has been every time.

## 12. A confound sec.5 does not list: the offline arm and the board arm are not the same operation

`rgb_vs_gray_loop.py`'s `dec<N>` **subsamples the conv OUTPUT** — the ROI is still extracted at
128x128 and convolved at that resolution, then every 2nd pixel is taken (`pool_features`,
`mode='dec'`). The board's `PATCH_ROWS=64` instead resamples the ROI to 64x64 **in `roi_crop`,
before conv2d**. Two consequences, neither measured:

- the 3x3 kernel then spans **twice the frame area** per tap, i.e. a different feature, not the
  same feature sampled coarsely;
- the board's bilinear 2:1 decimation partially prefilters where `dec2` aliases outright.

`px/bin` — the statistic this whole file identifies as the mechanism — is box/32 either way, so
the arm is still the right test of the mechanism. But the hardware is not running the offline
arm, and if it lands short of the discounted prediction this is the first place to look.

## 13. One trap found while preparing: do NOT run `make gen_vectors` at this geometry

`gen_vectors` writes `design/aie_src/aiesim_data/s*/` with no geometry key, and those directories
are **not tracked in git** (only `patch_in.txt`/`patch_in_const.txt` are). `make gen_vectors
PATCH_ROWS=64 PATCH_COLS=64` therefore overwrites the shipping 128x128 scenario vectors in place,
silently. Regenerating at 128 restores them, but nothing warns. Not needed for this build.

## 14. Off-board verification at 64x64 — all green

Run 2026-08-31, before committing to the package. None of it needs a board.

**Native suites**, `PATCH_ROWS=64 PATCH_COLS=64` (`mosse_filter` is dimension-agnostic, so this
mostly re-greens the tree; `roi_crop` is where geometry actually bites):

```
make test_host        PASS (0 failures, both builds -- -O2 and -O3 -ffp-contract=fast)
make test_roi_crop    PASS  ROI_IN_CH=1: 17 cases   ROI_IN_CH=3: 8 cases   (both arms run)
make test_scene       PASS (incl. the deliberately missed scene_touch())
make test_vot_source  PASS (62 real manifests parsed, StreamBlob against a direct read)
```

**Kernel bit-exactness at the new point size** — `x86sim_check`, the check that exists so a
kernel bug is caught in minutes instead of in an hours-long package. Every scenario re-generated
at 64x64 with the candidate budget (`3-3-3`, `H_SHIFT=15`):

```
conv2d  s6rgb  CONV_IN_CH=3  BIT-EXACT PASS   4096/4096 samples, max|.|=226
conv2d  s6     CONV_IN_CH=1  BIT-EXACT PASS   4096/4096 samples, max|.|=1669
cmul    s7     H_SHIFT=15    BIT-EXACT PASS   re+im 4096/4096, max|F|=21939 max|H|=28264
cmul    cmul_stress          BIT-EXACT PASS   re+im 4096/4096, drives sat16 to both rails
```

**The vectors were restored afterwards.** `s7`/`cmul_stress` are in the GRAYSCALE set
(`gen_aiesim_vectors.py` returns after `write_s6rgb` at `CONV_IN_CH=3`), so exercising them needs
a `CONV_IN_CH=1` pass, which also rewrites `layer0_weights.bin`. The 128x128 tree was copied out
first and copied back, and `make weights CONV_IN_CH=3` re-exported the shipping weights — layout
tag re-checked as 3, and `hanning_64.h` came back byte-identical to the committed file.

## 15. The card image is BUILT (calibration image, not the sweep image)

`ARM=rgb PATCH_ROWS=64 PATCH_COLS=64 FFT_SHIFT=3 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3 H_SHIFT=15
scripts/calib_build.sh` — the synthetic-scene, 200-frame, `TRAJECTORY=1 SCALE_TRAJ=1`,
`VERBOSITY=1` image that sec.16's step 2 needs. **The seeding paid off exactly as intended: the
build went straight to `v++ --link` — no HLS synthesis, no `aiecompiler` run**, ~10 min of
link + implementation + package instead of the estimated hour.

```
build/hw/64x64/ch16/package/sd_card.img   3.17 GB   ready to flash
build/hw/64x64/ch16/calib_cfg.txt         the provenance record
BUILD VERIFIED — all 21 stamp checks, geometry included, on both flagstamps
```

- **Timing is met**: WNS +0.322 ns, 0 failing endpoints of 32199 (hold and pulse-width clean).
- **PL utilisation is IDENTICAL to the 128x128 build** — 10527 LUTs (2.02%), 13216 registers,
  13 BRAM tiles — which is the strongest possible confirmation of sec.2's "roi_crop is free":
  the geometry is runtime AXI-Lite, so the fabric does not change at all.
- **Re-provisioned after packaging** (`make board_provision ROOTFS_IMG=.../package/sd_card.img`)
  — the step sec.2 warns about, verified: key mode 600 in a 700 dir, 192.168.10.2/24 on end0,
  filesystem clean.
- `calib_cfg.txt` recorded the 128x128 comparator string in good faith. **`calib_build.sh` now
  refuses to**: at any other geometry the comparator field reads `NONE`, because a stored number
  a later reader compares against in good faith is the `runs/.last_cfg` failure mode. The file
  for this build was corrected by hand and says so.

**The sweep ELF was deliberately NOT built yet.** It is host-only and needs no board, but
`make application` writes the same `BUILD_DIR`, so building it now would overwrite the
`app.flagstamp` that documents the image just packaged — leaving a provenance record that does
not describe the card. It also has to wait for the calibration: if `rails` sends `H_SHIFT`
moving, a sweep ELF built today is stale. Build it after the calibration run passes.

The VOT sweep image is a SEPARATE, host-only rebuild on top of the same xclbin
(`FRAME_SOURCE=vot VERBOSITY=0 PROGRESS_EVERY=25 DUMP_BUFFERS=0 CSV_FLUSH_EVERY=200`) — an scp,
not a second reflash.

## 16. The hardware run — everything it needs is ready

The artifacts, and the identities the guards will check:

```
image        build/hw/64x64/ch16/package/sd_card.img     3.17 GB, already provisioned
a.xclbin     f155eb3ef9e8...   (the 128x128 arm is 52235f49221e... -- vot_sweep.sh
                                compares these and REFUSES a mismatch)
weights      layer0_weights.bin md5 2aa7031f1f53..., identical to the repo's file and
             recorded in calib_cfg.txt
run_script   EMU_MODE empty -> XCL_EMULATION_MODE unset. Correct for real hardware
ELF          mosse_tracker.elf, ITER_CNT=200 VERBOSITY=1 TRAJECTORY=1 SCALE_TRAJ=1,
             synthetic scene -- this is the CALIBRATION image, not the sweep image
```

**1. Flash and boot.** The image already carries the static address and the authorized key
(re-provisioned after packaging), so the board should come up as an ssh target with nothing typed
at the console: `ssh root@192.168.10.2 uname -a`. **Take the calibration run on the SERIAL
CONSOLE, not over ssh** — the frame time is part of what this run measures, and launching over
ssh changes it.

**2. The 200-frame calibration.** `./run_script.sh` on the card. What it tests is the budget of
sec.11, and the criterion is written before the run:

- **`rails = 0`** over all 200 frames (`rails` is in the per-frame block at `VERBOSITY=1`, NOT in
  `track.csv` — which is why the image is built at verbosity 1).
- accum and response amplitudes in the region the 128x128 arm measured (52% / 49% of int16). The **(WRONG COMPARATOR — see sec.17.)**
  ledger predicts they land there; **that prediction is the thing under test.**
- **Read the TAIL, not frame 1** — the response grows as the filter converges, and a budget
  validated early is not validated.

Then `python3 scripts/calib_report.py <console log> <track.csv>` for the verdict. It is
geometry-clean (percentages of int16, no bin/pixel conversion), so it needs no flags.

**If it rails:** raise `H_SHIFT`, as every previous fix has been. **If the amplitudes come in far
LOW** (say under 15%), the ledger over-corrected and the budget wants a bit back — take it on
`IFFT_*`, which reach only the response, before touching `FFT_SHIFT`, which moves `F` too.

**3. Build the sweep ELF** (host-only, PC side, minutes) once the budget is accepted:

```
make application TARGET=hw PATCH_ROWS=64 PATCH_COLS=64 CONV_IN_CH=3 \
     FFT_SHIFT=3 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3 H_SHIFT=15 \
     FRAME_SOURCE=vot VERBOSITY=0 PROGRESS_EVERY=25 DUMP_BUFFERS=0 CSV_FLUSH_EVERY=200
```

It overwrites `app.flagstamp`, which is why it was NOT built alongside the card image — do it
only after the calibration passes, and the stamp then describes the sweep rather than the
calibration.

**4. The sweep.** `vot_sweep.sh` defaults `--elf` to the **128x128** path, so pass it:

```
scripts/vot_sweep.sh --arm res64 --seqs @runs/vot/seqs62.txt \
     --elf build/hw/64x64/ch16/mosse_tracker.elf --ingest --dry-run   # then without --dry-run
```

Forgetting `--elf` is not silently wrong — the `a.xclbin` md5 guard fails the run — but it costs
a restart. The VOT blobs on the board are frames and carry no geometry, so nothing needs
re-preparing. **NEVER KILL A BOARD RUN MID-SEQUENCE**; watch the trajectory count.

**5. Score it** on `vot analysis` with the TWO-part falsifier of sec.4 (`dEAO >= +0.005` AND the
per-sequence dR surviving drop-top-3), the common-survived-prefix accuracy of sec.4, and the
stored-baseline confound of sec.5 stated wherever the number is quoted. FPS only from the serial
console.

**Two instruments carry a geometry assumption — do not read them raw at 64x64:**

- `vot_traj_anatomy.py` takes `--patch` and defaults to 128. **Pass `--patch 64`.**
- `vot_detector_gain.py` hardcodes the bin->pixel conversion as `box * 2 / 128` (padding AND
  patch size baked in — the documented pad30 defect, now wrong on a second axis). At 64x64 its
  alpha is off by 2x. Left unfixed deliberately, so the numbers already recorded from it stay
  comparable; correct by hand before quoting it.

*(An `aiesim` end-to-end run at 64 points was started as a further off-board check of the sec.11
ledger and ABANDONED: ~7 minutes per channel in the cycle-approximate simulator, and the s7
scenario's expectations are calibrated for a single channel anyway. The hardware run answers the
same question in minutes. `build/hw_emu/64x64/` holds its leftovers and can be deleted.)*

## 17. The calibration RAN — 2026-09-01. VERDICT: budget 3-3-3 / `H_SHIFT=15` ACCEPTED

`runs/run_res64_calib_0901_1234.log` + `runs/res64_calib/track.csv`, 200 frames, synthetic scene,
`TRAJECTORY=1 SCALE_TRAJ=1`, the card image of sec.15 (`a.xclbin` `f155eb3ef9e8...`, weights
`2aa7031f1f53...`, ELF `36d62d3cb8be...` — all three verified ON THE CARD before the run).
**Taken over ssh, not the serial console** — no `/dev/ttyUSB*` was present on the host — so every
amplitude and tracking number below is transport-independent and stands, and the FRAME TIME is
not quotable.

**`rails = 0` on all 200 frames, all four buffers** (`F_ch`, `accum`, `response`, `H(q15)`;
`track.csv`'s four `rails_*` columns agree). That is the one hard criterion, and it is met.

**Tracking is at the comparator, not below it** — mean IoU 0.9209 against 0.9188, centre error
1.29 px against 1.37, worst IoU 0.8276 against 0.8353, 0 gate holds against 0. PSR min 20.11 /
mean 42.47 / max 56.65, 199 of 199 evaluated frames accepted. **PSR is the arbiter of a budget
(`shift_budget.md`), and it is healthy.**

Converged-tail amplitudes (frames 21+, per the read-the-tail rule):

| buffer | 64x64, this run | 128x128 RGB `H_SHIFT=15`, same instrument |
|---|---|---|
| `F_ch` | 15.4% of int16 | — |
| `accum` | 4.4% | — |
| `response` | 10.6% | ~22% (`shift_budget.md`) |

**`calib_report.py` prints UNDERSHOOT against a 49-64% band. That flag is ADVISORY and the band
is retired** — `shift_budget.md` and `shift_budget_realvideo.md` both say so in as many words: the
band came from a distribution with a 1.30x spread, the corrected build spreads 2.07x at the
converged end, and re-centring the typical frame puts the tail on the rail. The 49%/52% figures
this proposal quoted in sec.11 as "the region the 128x128 arm measured" are from
`run_0824_1354` — **the OLD budget, the one that went on to rail on 266 real-video frames.** They
are the wrong comparator and were used as one here; the shipping arm sits at ~22% response on the
same instrument, and at accum 15.6% / response 10.1% of ceiling over 101,564 real-video frames.

So the honest reading of the table is **one bit, not two**, and in a direction the shipping arm
deliberately chose for itself: 128x128 RGB stays at `H_SHIFT=15` when 13 is the tight value,
i.e. two bits over-shifted, because `rails = 0` is the only hard criterion and the benchmark is
banked on it. **This arm at 15 is simply one bit less over-shifted than that one, with the same
rails-clean margin and equal tracking.** Real video will not make it worse: at 128x128 the
response tail on real video (10.1%) came in BELOW the synthetic calibration (22%).

**Do not rebuild.** `H_SHIFT` is the one knob that is not host-only — it costs a graph rebuild,
re-package and re-flash — and it would be spent to chase a retired band. Sec.11's ledger is
CONFIRMED where it can be checked (`F_ch` at 15.4% is the forward chain landing where the
64/2^3 = 128/2^4 = 8 identity says it should) and its response prediction was scored against the
wrong reference, not falsified.

**Frame time, NOT QUOTABLE, recorded only for direction:** mean frame body 13.16 ms over ssh
against the 128x128 RGB arm's 28.58 ms on the serial console. The transports differ (the UART
alone is 3.79 ms), so the two are not comparable — but the gap far exceeds the transport
difference and is consistent with the ~15 ms/frame the proposal predicted. GMIO fell to
4.898 ms/frame over 313 tx. **Re-measure on the console before any FPS claim.**

**The host does not exit.** `gr.end(0)` never returns (the graph is `run()`-forever), so the elf
sits in `hrtimer_nanosleep` at 0% CPU after printing everything — the documented end state, not a
hang. The board wants a reboot before the next run rather than a kill.

**Next: sec.16 step 3** — build the sweep ELF (host-only) and run the 62-sequence sweep with
`--elf build/hw/64x64/ch16/mosse_tracker.elf`.

## 18. THE SWEEP RAN — 2026-09-01. THE ARM IS CONFIRMED

`runs/vot/0901_1252-res64`, workspace `~/vot/analysis/0901_res64`. 62 sequences, 419
trajectories, 0 skipped, 0 failed. ELF `81fcea3e7565`, xclbin `f155eb3ef9e8` verified against
the card, weights `2aa7031f1f53`, data `/srv/vot/data-rgb`.

| arm | A | R | EAO |
|---|---|---|---|
| `eta05_g5p0` — SHIPPING, 128x128 | 0.5100 | 0.3417 | 0.1629 |
| `base_stat` — the same config, re-run 2026-08-31 | 0.5100 | 0.3417 | 0.1629 |
| **`res64` — 64x64, shift 3-3-3, `H_SHIFT=15`** | **0.5336** | **0.3873** | **0.1849** |
| delta | +0.0236 | +0.0456 | **+0.0220** |

**`base_stat` reproduces `eta05_g5p0` to four decimals on all three figures.** That is the
comparator's own determinism check and it was free — the same numbers from a separate sweep a
week apart, so the delta above is not a re-run artifact.

**The pre-registered falsifier of sec.4 PASSES on both parts:**

- `dEAO = +0.0220`, against the threshold `>= +0.005`.
- per-sequence `dR` mean `+0.0323`, and **`+0.0187` after dropping the top 3** (`+0.0104` after
  dropping the top 5). 37 sequences better, 22 worse, 3 unchanged — not three sequences carrying
  it, which is exactly how `FILTER_MASK` failed to separate from a null.

Largest gains `singer2 +0.332  conduction1 +0.309  rowing +0.261  soldier +0.257  girl +0.250`;
largest losses `polo -0.208  rabbit -0.178  motocross1 -0.158  soccer1 -0.134`.

**The offline proxy predicted `dR +0.1071` and hardware returned `+0.0456`: right sign, 43% of
the magnitude.** That is the third data point on the same instrument — `pad30` did not transfer
at all, `eta05` did — and it is worth recording as the proxy's calibration, not as a
disappointment. **`vot_ar_offline.py` decides whether an arm deserves board time and nothing
more**, which is precisely how it was used here.

**Accuracy also rose (+0.0236), and that number needs the sec.4 caveat when quoted**: VOT
averages A over TRACKED frames, and this arm tracks 74,309 frames against 61,831, so the two A
values are not measured over the same footage. **Score A on the common survived prefix before
claiming an accuracy win.** The direction is at least not a trade — R and EAO moved with it, so
nothing here is the A/R swap that sank `pad30`.

**Both halves of the claim landed.** N-03b promised robustness; sec.15 promised ~15 ms/frame.
The frame body measured 10.39 ms on `agility` over ssh against the 128x128 arm's 28.58 ms on the
console — still not a quotable FPS (transports differ), but the arm is faster AND more robust,
which is the rare case where nothing had to be traded.

**Not yet done, and each is cheap:**
- FPS on the SERIAL CONSOLE. Every frame-time figure for this arm is an ssh number.
- The common-survived-prefix A, before the +0.0236 is quoted anywhere.
- `vot_traj_anatomy.py --patch 64` (it defaults to 128) if the per-sequence losses are worth
  anatomising; `polo`, `rabbit`, `motocross1` and `soccer1` are where to look, and the
  localisation quantum doubling to 1.78 frame px/bin is the first mechanism to test.
- `H_SHIFT` re-tune is NOT indicated — sec.17 settled that, and this result was obtained at 15.

## 19. WHY it won — read off the logs, 2026-09-01

Instruments: the two arms' per-frame CSVs (`0901_1252-res64` vs `0827_1642-eta05_g5p0`, 180,544
rows each, 419 matched runs), `vot_loss_anatomy.py` on both, and the two `agility` APU blocks
(same sequence, same 252 frames, both over ssh — comparable).

### 19.1 What actually changed

```
                                res64      g5p0(128)
runs that NEVER reach IoU<=0.1  13.13%      8.35%     <-- +57% relative
median first-loss position       0.170      0.176     <-- unchanged
tracked frames (IoU>0.1)         50.5%      43.9%
mean IoU, all frames             0.2488     0.2100
hold rate (frame-weighted)       12.48%     15.51%
vetoes NEGATIVE_PEAK             19437      27772     <-- -30%
vetoes LOW_PSR                    3046        174     <-- x17
PSR accepted, median              20.7       30.9
PSR accepted, mean                22.9       38.8
resp00/peak, median               0.72       0.40     <-- the response is BROADER
median |dr|+|dc| reported          3.0 bins   5.25 bins
median centre error, tracked     14.29 px   14.41 px  <-- UNCHANGED
frames with rails>0                 158         56    (ALL in H(q15); fch/accum/resp = 0)
```

**The arm does not lose later — it loses less often.** The median first-loss position is
identical (0.170 vs 0.176 of the run); what moved is the share of runs that never fail at all,
8.35% -> 13.13%. That is the shape R and EAO reward and mean IoU cannot see, and it is why this
arm transferred where `pad30` did not.

### 19.2 THE ACCURACY GAIN IS REAL, NOT SELECTION

The obvious objection to A 0.5100 -> 0.5336 is that VOT averages over TRACKED frames and this
arm tracks 12,478 more of them. Scored on the **common survived prefix** — per run, every frame
before EITHER arm's first loss, 47,902 frames — the gain survives intact:

```
mean IoU   res64 0.5860   g5p0 0.5650   +0.0210     (raw, unmatched: +0.0236)
per frame  better 51.6%   worse 38.3%
```

**Halving the localisation grid IMPROVED accuracy**, and the median centre error on tracked
frames did not move (14.29 vs 14.41 px) despite the quantum doubling from 0.89 to 1.78 frame px
per bin. This is the strongest confirmation yet of `detector_gain.md`'s verdict that
localisation is not the constraint — half the localisation resolution is free, and sub-bin
interpolation is evidently absorbing the difference.

**It also refutes the falsifier this arm was pre-registered against.** Sec.4 wrote "if A falls by
more than ~0.02 the coarser grid is not worth it"; the offline proxy predicted A **-0.0389**.
A rose. See 19.4.

### 19.3 Three candidate mechanisms, and what the logs say about each

**(a) The target Gaussian doubled in width relative to the target.** `MOSSE_SIGMA` is 2.0 BINS
and the target always spans `patch/padding` bins — 64 at 128x128, **32 at 64x64**. So sigma went
from box/32 to **box/16 of the target extent**, and box/16 is exactly DSST's `target/16` rule,
which this repo carries as `SIGMA_FROM_TARGET=1` and has never enabled. **The 64x64 arm
accidentally implements DSST's sigma rule; the shipping 128x128 arm sits at half of it.**
Supporting evidence: `resp00/peak` 0.40 -> 0.72 and PSR down 40% are exactly what a wider
mainlobe produces, and `settled.md` records that **sigma was never decided on real video** —
"PSR cannot select sigma; sigma needs real video". The px/bin optimum the offline sweep found
(box/32, `pooled_features.md`) corresponds to sigma = 2 x box/32 = box/16 in frame pixels, i.e.
the same number from the other direction.

**(b) Fewer degrees of freedom, better-conditioned denominator.** 4096 bins per channel instead
of 16384, with the same 16 channels feeding one shared denominator. Bolme §3.3's pathology is
low-energy denominator bins; a 4x coarser grid puts 4x the image energy in each bin. The direct
symptom is in the veto table: **NEGATIVE_PEAK, which is literally "the acted-on peak is
anti-correlated", fell 30%** with no other change that touches it. `settled.md` already credits
"the 16-channel denominator" with curing the init-perturbation pathology; this is the same cure
applied harder.

**(c) The board's resample is not the bench's subsample.** `roi_crop` bilinearly resamples the
ROI to the patch; the offline `dec2` arm subsamples. At 64x64 that difference becomes a 2:1
decimation with a prefilter versus one that aliases outright (sec.12). **This is the most likely
reason hardware beat the proxy's accuracy prediction by ~0.06 A**, and it is not a small
detail — it means the hardware arm is strictly a better-conditioned version of the arm the bench
scored.

**These are separable and the separation is HOST-ONLY.** `MOSSE_SIGMA` reaches neither
`AIE_FLAGS` nor the bitstream:

- `res64` + `MOSSE_SIGMA=1.0` restores the 128x128 arm's mainlobe in frame pixels while keeping
  the 4096-bin grid and the prefilter. Gain survives => (a) is NOT the mechanism.
- `128x128` + `MOSSE_SIGMA=4.0` (or `SIGMA_FROM_TARGET=1`) applies (a) alone, with no reflash.
  Gain reproduces => most of this arm's robustness was available without silicon, and res64 at a
  re-tuned sigma should go further.

**An offline 2x3 (arms `rgb`,`rgb-dec2` x sigma 1,2,4; 62 sequences, eta 0.05, gate 5.0) is
running as of 2026-09-01 13:1x** in `runs/vot/0901_offline-sigma/`, to rank the two before either
costs a board sweep. Score it with `vot_ar_offline.py` and remember its ~0.02 resolution.

### 19.4 What the hardware says about the OFFLINE PROXY — third data point

```
arm        offline dR   hardware dR   offline dA   hardware dA
eta05         +0.0159      +0.0218          --          --
pad30         +0.088       +0.0077          --          --
res64         +0.0567      +0.0456       -0.0086     +0.0210
```

**[SUPERSEDED BY sec.21.2 — the right comparator is +0.1071 at gate 5.0, so the transfer is
43%, as sec.18 said. The +0.0567 row below is the gate-7.0 run.]** `res64` predicted +0.0567 here
and delivered +0.0456 and its worst on A (predicted a 0.009-0.039 LOSS, delivered a 0.021 GAIN).
**The proxy models the failure rule well and the box quality badly**, which is consistent with
19.3(c): it is scoring an aliased arm the board does not run. The sec.18 line "43% of the
magnitude" was computed against the 8-sequence `+0.1071` figure and should be read against the
62-sequence `+0.0567` instead — 80%.

**And it re-reads `pad30`.** Both arms lower px/bin; only one worked. The difference is that
`pad30` changed the ROI CONTENT (3x the background) and left the grid at 16384 bins, while
`res64` held the ROI fixed and cut the grid. **px/bin is therefore NOT the sufficient statistic
`pooled_features.md` proposed** — the hardware pair box/43-via-padding (+0.0077) against
box/32-via-resolution (+0.0456) says the route matters more than the number, which is the same
conclusion its own two matched offline pairs reached and the opposite of the headline it drew.

### 19.5 The compute dividend, measured — `agility`, 252 frames, both arms over ssh

```
stage                 g5p0(128)    res64    ratio
frame body              24.740     10.430    2.37x
  GMIO (DMA_T)          11.165      4.908    2.27x
  filter upd+quant       3.811      1.060    3.60x
  unpack F_ch            1.778      0.422    4.21x
  BO<->heap stage        1.695      0.456    3.72x
  publish (pack)         1.443      0.405    3.56x
  roi_crop launch        1.473      0.373    3.95x
  PSR scan               0.509      0.130    3.92x
  scale extract          1.659      1.735    1.00x   <-- UNCHANGED
  scene gen+push+sync    0.929      0.923    1.00x   <-- UNCHANGED
```

Everything indexed by patch bins fell by ~4x; the two blocks that are not — the DSST scale
filter (its template is `SCALE_TMPL_AREA`, not the patch) and the frame-source path (1080p,
not the patch) — did not move at all. **`roi_crop` falling 3.95x confirms the `pad30` finding
that its cost is set by OUTPUT pixels**, from the opposite direction.

**The performance roadmap re-ranks.** `scale extract` is now the single largest item in the
frame at 16.6%, and the frame-source path is 8.9%. "The APU is a flat tail" is no longer true —
it has a head, and it is the scale filter.

## 20. What this re-opens, re-ranks and closes

### 20.1 THE SHIFT BUDGET, measured on real video — and §17's rejected rebuild would have RAILED

Uncensored maxima over 180,544 real-video frames per arm:

```
                 accum_max                  resp_max              fch0_max   rails frames
res64      20227  (43.6% of 46340)   15448  (47.1% of int16)        23675      158 (all H)
g5p0(128)   7707  (16.6%)             5416  (16.5%)                 32767       56 (53 H, 3 fch)
```

**The 64x64 arm runs 2.6-2.9x HOTTER on real video than the shipping arm**, at 44-47% of ceiling
with `rails_fch = rails_accum = rails_resp = 0` on every one of those frames. The 4x coarser grid
concentrates ~4x the energy into each bin; the synthetic calibration of sec.17 does not show this
because the synthetic scene is a weak stimulus.

**So sec.17's arithmetic was right and its conclusion was lucky.** `H_SHIFT=13` — the "fix" the
UNDERSHOOT flag invited — multiplies both by 4: 175% and 188% of ceiling. It would have railed
hard, on the arm the whole result rests on. `H_SHIFT=14` lands at 87%/94%, i.e. on the rail's
edge, against the standing rule to size against the tail. **`H_SHIFT=15` at 3-3-3 is not
over-shifted at 64x64 — it is correct, with about one bit of margin and no more.** Do not spend a
reflash lowering it, and re-read this table before touching the budget at any further geometry.

### 20.2 RE-OPENED — `PSR_GATE_MIN`, and it is currently MIS-SCALED. Host-only. Do this first

```
                    res64    g5p0
PSR accepted median  20.7    30.9      x0.67
PSR accepted mean    22.9    38.8      x0.59
LOW_PSR vetoes       3046     174      x17.5
NEGATIVE_PEAK       19437   27772      -30%
```

The gate is a FIXED 5.0 against a PSR scale that moved by 0.6x, so the same threshold is now
~1.7x tighter in real terms and it is firing 17x more often. CLAUDE.md already records the gate
being worth **+0.006 at 128x128 against +0.056 at 64x64** offline — an order of magnitude more
at exactly this geometry — and the standing rule is that anything moving PSR re-opens it, as
`MOSSE_ETA=0.05` did. **`PSR_GATE_MIN` ~3.0-3.5 is the indicated arm** (5.0 x 0.6-0.67), and the
correlation supports relaxing rather than tightening: **corr(per-sequence dR, d hold-rate) =
-0.307** — the sequences that hold MORE gain LESS. One host ELF, one sweep, no silicon.

### 20.3 RE-OPENED — `MOSSE_SIGMA`, which may be MOST of this result and needs no silicon at all

See 19.3(a). Two host-only arms bracket it, and one of them could hand the 128x128 arm most of
this gain without a reflash. `settled.md` closed sigma with "PSR cannot select sigma; sigma needs
real video" — this is that real video, and it says the shipping arm sits at half of DSST's rule.
The offline 2x3 in `runs/vot/0901_offline-sigma/` ranks them before either costs board time.

### 20.4 RE-OPENED, LOWER — `MOSSE_ETA`

`eta=0.05` was chosen at 128x128. Each bin now aggregates 4x the energy and the update runs on a
grid with a quarter of the degrees of freedom, so the effective time constant of the learned
filter is not the one that was tuned. Host-only, but it is a second-order re-tune behind the gate
and sigma, and `eta` is documented as NOT monotone (0.025 much worse) — sweep, do not extrapolate.

### 20.5 RE-RANKED — the spatial mask

`FILTER_MASK=1` at 128x128 gave EAO +0.0110 and could not be separated from a null (3 of 62
sequences carried it). At 64x64 it is geometrically the SAME mask — the target always spans
`patch/padding` bins, so the mask's box is the same fraction of the map — and it costs 4x less.
But its premise weakens twice over: `pooled_features.md` already showed background contamination
is not the binding constraint (every padding below 2.0 is worse), and this arm's -30%
`NEGATIVE_PEAK` says the conditioning it was meant to fix has been partly fixed by the geometry
instead. **Keep it ranked behind the gate and sigma, and re-run the drop-top-3 separability test
before believing any repeat.**

### 20.6 CLOSED HARDER — aggregation, padding, localisation, quantization

- **Aggregation (N-03) stays refuted, now from the hardware side.** The arm that shipped carries
  NO aggregation — a plain resolution change — and delivered what max, average and subsample
  agreed on offline to 0.001. The operator was never the variable.
- **Padding (N-04) stays rejected, and now has a mechanism.** `pad30` and `res64` both lower
  px/bin; only the one that held the ROI content fixed worked. See 19.4.
- **Localisation is exonerated a second way.** The quantum doubled to 1.78 frame px/bin and the
  median centre error on tracked frames moved by 0.12 px. Sub-bin interpolation is doing its job;
  `detector_gain.md`'s alpha 0.93 stands. (`vot_detector_gain.py` still needs its hardcoded
  `box*2/128` corrected by hand at this geometry.)
- **Quantization stays exonerated.** A 4x coarser grid at the SAME fixed-point budget improved
  both A and R; if quantization were binding, concentrating 4x the energy per bin against an
  unchanged 16-bit rail would have shown up as a loss. It showed up as 44-47% ceiling occupancy
  and rails=0.

### 20.7 OPENED BY THE DIVIDEND — but read the participation ratio first

The frame is 2.37x faster and every patch-indexed stage fell ~4x (19.5), so `N_CHANNELS=32` at
64x64 costs roughly what `N_CHANNELS=16` at 128x128 did. **This is affordable, and it is NOT ruled out by the participation ratio** — `feature_bank.md`
retires PR as a quality metric (random Gaussian weights score 10.69 against the trained bank's
7.43, so the statistic is maximised by noise and ranks nothing). What it IS ruled out by is that
nothing has shown the bank's WIDTH to be binding: the activation-space effective width is 1.43,
and `N_CHANNELS` is an `AIE_FLAGS` knob, i.e. a rebuild. Test it only if a channel-reliability
result (Stage B3, roadmap item 3) says the weak channels are worth keeping.

The clearer use of the dividend is PERFORMANCE, and it re-ranks too: `scale extract` is now the
largest single item in the frame at 16.6% and did not move at all, and the frame-source path
(scene gen + push + sync, 0.92 ms) is now 8.9%. **The old "the APU is a flat tail, biggest item
5.2 ms" no longer holds — the tail has a head, and it is the DSST scale filter.**

## 21. SIGMA vs RESOLUTION, separated — 2026-09-01. **75% of this arm is a HOST-ONLY knob**

`runs/vot/0901_offline-sigma/`, `combined.json`. Six arms: `rgb` and `rgb-dec2` x
`MOSSE_SIGMA` in {1, 2, 4} BINS, 62 sequences, 19,903 frames, eta 0.05, gate 5.0, scored with
`vot_ar_offline.py`. 114 of the 186 runs were executed 24-way parallel; the rest were already
banked.

**POSITIVE CONTROL: the two sigma-2 arms reproduce the recorded board-form pair DIGIT FOR
DIGIT** — `rgb` 0.5394 / 0.2910 / 5792 tracked and `rgb-dec2` 0.5005 / 0.3981 / 7923, exactly the
2026-08-31 table. The new arms are readable.

```
robustness            sigma=1   sigma=2   sigma=4
rgb        (128x128)   0.2025    0.2910    0.3718
rgb-dec2   ( 64x64 )   0.2808    0.3981    0.3506
```

`MOSSE_SIGMA` is in BINS and the target spans `patch/padding` bins — 64 at 128x128, 32 at 64x64 —
so the meaningful variable is **sigma as a fraction of the target extent**, and the table above
re-indexes onto ONE axis:

```
sigma/target    arm                R        A
   1/64      rgb-s1            0.2025   0.5273
   1/32      rgb-s2            0.2910   0.5394   <- SHIPPING 128x128
   1/32      rgb-dec2-s1       0.2808   0.5438   } matched pair
   1/16      rgb-s4            0.3718   0.4965   } matched pair
   1/16      rgb-dec2-s2       0.3981   0.5005   <- THE SHIPPED 64x64 ARM
   1/8       rgb-dec2-s4       0.3506   0.5430
```

**R is governed by sigma/target, not by px/bin, and the optimum is 1/16 — exactly DSST's
`target/16` rule.** The curve rises to 1/16 on both maps and turns over past it (0.3981 ->
0.3506). At MATCHED sigma/target the two resolutions nearly agree (-0.0102 at 1/32, +0.0263 at
1/16); at matched resolution, doubling sigma is worth +0.0808 (rgb) and +0.1173 (dec2).

**Decomposition of the shipped arm's offline +0.1071:**

```
mainlobe width alone   rgb-s2 -> rgb-s4        +0.0808    75%   HOST-ONLY, no reflash
resolution alone       rgb-s4 -> rgb-dec2-s2   +0.0263    25%   needs the graph rebuild
```

**This overturns `pooled_features.md`'s headline.** That sweep held `MOSSE_SIGMA` at 2 bins
across every arm, so px/bin and sigma/target were PERFECTLY CONFOUNDED (sigma in frame pixels =
2 x px/bin) and "R is governed mostly by px/bin, optimum near box/32" was the same finding read
off the wrong axis — box/32 px/bin at sigma 2 bins IS sigma = box/16. The bench's own `--sigma`
help had named the confound ("the other half of the geometry confound the dec arm controls
for"); it had never been run.

**And it explains the `-0.0143` gray result and the `tiger` anomaly** as the same axis: every
arm that lowered px/bin also widened the mainlobe, and `tiger.md`'s two unfreezing experiments
went the OTHER way (`SIGMA=1`, i.e. 1/64 of target — the worst cell in the table at 0.2025) and
made tracking worse, which is exactly what this curve predicts.

### 21.1 What to run next, and in what order

1. **`MOSSE_SIGMA=4.0` at 128x128 — HOST-ONLY, one ELF, one sweep.** Offline dR +0.0808. If it
   transfers it hands most of this result to any 128x128 build with no silicon, and it is the
   cleanest possible test of the mechanism on the metric of record. `SIGMA_FROM_TARGET=1`
   computes `target/16` and lands on the same 4.0 at this geometry — prefer the explicit
   `MOSSE_SIGMA=4.0` so the flagstamp says what was run.
2. **`MOSSE_SIGMA` re-tune ON res64.** The shipped arm is already AT the optimum (1/16), and the
   curve turns over on both sides of it, so the expected gain is small and the useful arms are
   the interior ones (2.5-3.0 bins, i.e. 1/13-1/11). **Do not raise it to 4.0 on res64** —
   `rgb-dec2-s4` is the measured turnover and is 0.0475 WORSE.
3. **The gate re-tune of 20.2 stays first among the host-only arms**, and this sweep adds a
   reason: the SAME resolution comparison scores +0.0567 at gate 7.0 (`pooled_features.md`,
   earlier section) and +0.1071 at gate 5.0. **Gate and geometry interact strongly**, so a
   sigma arm run at a mis-scaled gate is measuring two things.

### 21.2 The caveats that must travel with this

- **This is the proxy, not the toolkit.** Single-start, no anchors, measured resolution ~0.02 in
  R. `+0.0808` clears that 4x, but `pad30` cleared it 4x too and returned +0.0077 on hardware.
- **The proxy's ACCURACY column is not trustworthy on this axis and is now measured to be so.**
  It predicted A -0.0389 for the resolution arm; hardware delivered +0.0210 on the common
  survived prefix. Offline A here anti-correlates with R across the whole table, which is the
  selection effect (more surviving frames are harder frames), so read the A column as a warning
  flag and nothing more.
- **Correction to sec.19.4.** The right offline comparator for the shipped arm is **+0.1071**
  (eta 0.05, gate 5.0 — the configuration this sweep reproduces exactly), so hardware's +0.0456
  is **43% of it**, as sec.18 originally said. The `+0.0567` figure quoted in 19.4 is from the
  earlier gate-7.0 run and is not the shipping comparator.

## 22. `PSR_GATE_MIN=3.5` on res64 — PREDICTIONS, written before the run (2026-09-01)

Host-only. `app.flagstamp` differs from the res64 sweep in ONE flag (`-DPSR_GATE_MIN=5.0` ->
`3.5`); `aie.flagstamp` untouched, board `a.xclbin` unchanged at `f155eb3ef9e8`. ELF
`732b523a7352`.

**Why 3.5, by two independent estimators** (CLAUDE.md: two instruments beat one instrument
twice), both computed off the res64 sweep's own 180,544 rows:

```
match the 128x128 arm's LOW_PSR VETO RATE (174/152353 = 0.114%)   ->  3.51
match the PSR MEDIAN scale (20.40 / 30.87 = 0.661) x 5.0          ->  3.30
```

They agree to 0.2. 3.5 is the rate-matched value and the conservative end.

**Predictions.** Static estimates from the res64 CSVs, i.e. assuming the trajectories do not
move; they will, so these are order-of-magnitude, not targets:

```
LOW_PSR vetoes        3046  ->  ~170        (the gate stops being a routine veto)
hold rate             12.48%  ->  ~10.9%
NEGATIVE_PEAK         19437  ->  ~unchanged (the gate does not touch this path)
R                     0.3873  ->  +0.005 to +0.030
A                     0.5336  ->  flat to slightly down (fewer holds = more updates)
```

The R range brackets the two priors: the hardware 128x128 arm's 7.0 -> 5.0 was worth +0.0134 R,
and the offline estimate of the gate's worth AT 64x64 is +0.056.

**FALSIFIER, and it is the same one as sec.4: EAO is the arbiter.** Accept only on
`dEAO >= +0.005` over 0.1849. **If EAO falls, the re-tune is REJECTED and 5.0 stays**, whatever R
does — this is exactly the A/R trade that sank `pad30`, and the gate is the knob most likely to
buy R by holding a stale box.

**Second thing this run tests, for free:** whether the gate's worth is really "conditional on the
PSR scale" as CLAUDE.md asserts. If a threshold rescaled to the new PSR distribution recovers a
gain, the assertion is confirmed on hardware. If it is a null, then the gate's 7.0 -> 5.0 win at
128x128 was about something other than the scale, and that assertion needs rewriting.

## 23. `PSR_GATE_MIN=3.5` — RAN, AND IT IS A NULL. The falsifier fired

`runs/vot/0901_1442-res64_g35`, workspace `~/vot/analysis/0901_gate35`. 62 sequences, 419
trajectories, 0 failed. ELF `732b523a7352`, one flag from the res64 build, same `a.xclbin`.

```
arm                     A        R        EAO        frames
res64      (gate 5.0)  0.5336   0.3873   0.184936    74309
res64_g35  (gate 3.5)  0.5308   0.3936   0.184867    74187
delta                 -0.0028  +0.0063  -0.000069
```

**EAO is identical to four decimals.** By sec.22's pre-registered criterion (`dEAO >= +0.005`)
this is **REJECTED**, and it is the textbook shape the criterion exists to catch: R up, A down,
EAO unmoved. Per-sequence dR mean +0.0084 collapses to **+0.0017 after drop-top-3**, 26 better /
17 worse / **19 exactly tied** — the tied count alone says most sequences did not notice.

**The mechanical predictions of sec.22 were all correct. The tracking prediction was not.**

```
                     predicted     measured
LOW_PSR vetoes        ~170            38
hold rate            ~10.9%        11.74%   (from 12.48%)
NEGATIVE_PEAK      ~unchanged       20962   (from 19437)
R                 +0.005..+0.030   +0.0063  <- bottom of the range
A                 flat/slightly down -0.0028
```

R landed at the very bottom of a range whose top came from the offline estimate of +0.056.

### 23.1 The free result: **"the gate's value is conditional on the PSR scale" is REFUTED**

That assertion is in CLAUDE.md and in claim R-03, and sec.22 set this run up to test it. The
64x64 arm moved the PSR distribution by 0.66x and left a fixed 5.0 threshold firing **17x more
often** (3046 LOW_PSR vetoes against 174). Rescaling the threshold by exactly that factor — by
two independent estimators that agreed to 0.2 — recovered **nothing**.

**Why, and the answer was already in the repo:** `vot_loss_anatomy.py` on this arm reports
**95.9% of all vetoes fire AFTER the run is already lost** and only 2.3% while still on target.
That is the standing finding that the gate is the AFTERMATH of a loss, not its cause. So the
17x mis-scaling was almost entirely post-loss bookkeeping — a statistic, not a behaviour. The
3.5 gate did free the 101 pre-loss `LOW_PSR` holds the 5.0 gate was taking (the 5 frames before
first loss now contain ZERO of them, against 101), and that is worth +0.0063 R and -0.0028 A.
**That is the whole size of the effect, and it is what a correctly-scaled gate is worth here.**

**What this does NOT overturn:** the 128x128 `7.0 -> 5.0` arm really did deliver +0.0134 R. But
its EAO gain was +0.0029, also under the +0.005 bar — the gate has never been worth much on the
metric of record, and it shipped on R. The honest restatement is: **the gate is a small, mostly
post-hoc veto whose threshold is not worth re-tuning per geometry.** Leave `PSR_GATE_MIN=5.0`.

### 23.2 What this says about the remaining candidates

- **`MOSSE_SIGMA=4.0` at 128x128 (sec.21.1) is now the single best-motivated open arm**, and it
  no longer has the gate confound sec.21.1 warned about: the gate does not interact with
  tracking at this geometry to any measurable degree, so a sigma arm can be run at the shipping
  5.0 and read cleanly.
- **The offline proxy has now missed twice on the gate axis** (+0.056 predicted, +0.0063
  delivered) after missing on padding (+0.088 -> +0.0077) and hitting on resolution (43%
  transfer). Its record is: good on the resolution/geometry axis, poor on knobs that act through
  the veto path. Weight sec.21's +0.0808 sigma prediction accordingly — sigma acts on the
  response shape, which is the axis the proxy models well, but that is an argument by analogy
  and not evidence.

## 24. `MOSSE_SIGMA=4.0` at 128x128 — PREPARED, predictions written before the run (2026-09-01)

**The point of this arm.** Sec.21 separated the 64x64 result into a mainlobe-width term
(+0.0808 offline, 75%) and a resolution term (+0.0263, 25%), and found R is governed by
sigma/target with an optimum at 1/16 — DSST's `target/16` rule. At 128x128 the target spans 64
bins, so `MOSSE_SIGMA=4.0` puts the shipping geometry AT that optimum. **If this transfers, most
of the 64x64 robustness gain is available on the 128x128 bitstream with no silicon at all**, and
the res64 arm's remaining justification becomes the 2.37x frame time plus a +0.0263 top-up.

**The build.** `build/hw/128x128/ch16/mosse_tracker.elf`, md5 `62383777f3b8`. Against the
SHIPPING arm's recorded flagstamp (`0827_1642-eta05_g5p0`) the functional difference is **one
flag, `MOSSE_SIGMA` 2.0 -> 4.0**; the four `FILTER_MASK*` flags appear only because the shipping
arm predates that feature, and `FILTER_MASK=0` is proven inert — `base_stat` carries it and
reproduces `eta05_g5p0` to four decimals on A, R and EAO. `aie.flagstamp` is **byte-identical**
to the shipping arm's, so sigma is confirmed host-only by construction as well as by the table.

**A RE-FLASH IS REQUIRED, and not because of sigma.** The card currently holds the 64x64
bitstream (`a.xclbin f155eb3ef9e8`); this arm needs the 128x128 one (`52235f49221e`). The image
is already built and is the one the shipping and mask arms ran from:

```
build/hw/128x128/ch16/package/sd_card.img      3.17 GB, a.xclbin 52235f49221e
```

Nothing needs rebuilding for it — no graph compile, no re-package. `vot_sweep.sh`'s guard
compares the board's `a.xclbin` against this tree's and will refuse until the card is swapped.

**Comparator.** `eta05_g5p0` (same geometry, same everything, sigma 2.0), already scored and
already shown reproducible by `base_stat`. No baseline re-run needed.

**Predictions.**

```
R      0.3417  ->  offline says +0.0808; hardware transfer on the geometry axis was 43%,
                   so the honest range is +0.02 .. +0.06
A      0.5100  ->  DOWN. Offline says -0.043 for this exact arm (rgb-s2 0.5394 ->
                   rgb-s4 0.4965), and unlike the resolution arm there is no prefilter
                   story to argue it back
EAO    0.1629  ->  the arbiter, and the whole question: does the R gain survive the A loss
PSR              DOWN (a wider mainlobe is a broader peak) -- expect the 64x64 arm's
                   scale, ~0.6x, and DO NOT re-tune the gate for it (sec.23)
holds            up slightly
```

**FALSIFIER: `dEAO >= +0.005` over 0.1629.** Below that it is a null or an A/R trade and sigma
stays at 2.0. **A specific way this can fail that the offline bench cannot see:** the proxy has
no DSST scale filter, and a wider target Gaussian makes the response peak broader in the same
patch the scale filter draws its template from — the same blind spot that made `pad30`'s
prediction 10x too large. That is the named risk, and it is priced: if this returns under
+0.005, the sigma axis is offline-only and the 64x64 arm keeps the full credit.

**If it PASSES**, the follow-up is `MOSSE_SIGMA` on res64 in the interior (2.5-3.0 bins, i.e.
sigma/target 1/13-1/11) — NOT 4.0, which is the measured turnover there (sec.21.1).

## 25. `MOSSE_SIGMA=4.0` RAN — NEW BEST EAO 0.1931, HOST-ONLY, AND IT REVISES SEC.18-20

`runs/vot/0901_1601-sigma4`, workspace `~/vot/analysis/0901_sigma4`. 62 sequences, 419
trajectories, 0 failed. ELF `62383777f3b8`, one flag from the shipping arm, `aie.flagstamp`
byte-identical, board `a.xclbin 52235f49221e`.

| arm | A | R | EAO | frames |
|---|---|---|---|---|
| `eta05_g5p0` — shipping, 128x128, sigma 2 | 0.5100 | 0.3417 | 0.1629 | 61831 |
| `res64` — 64x64, sigma 2 | 0.5336 | 0.3873 | 0.1849 | 74309 |
| **`sigma4` — 128x128, sigma 4** | **0.5133** | **0.4095** | **0.1931** | **82278** |

**dEAO +0.0301 over shipping, against a +0.005 falsifier — it passes by 6x**, with per-sequence
dR +0.0372 mean and **+0.0235 after drop-top-3**, 38 better / 21 worse / 3 tied. **And it is
HOST-ONLY**: same bitstream as the shipping arm, an scp rather than a card swap.

**Accuracy is UP, not down, and the pooled figure understates it.** Pooled A +0.0033 while
tracking 20,447 more frames; on the common survived prefix (49,363 frames) it is
**0.5876 against 0.5576, +0.0299**. The offline bench predicted A **-0.043** for this exact arm.
That is the second time its accuracy column has been wrong in the same direction (res64:
predicted -0.039, delivered +0.021 on the prefix). **Treat the proxy's A as uninformative on
response-shape arms; its R transferred at 84% here (+0.0808 predicted, +0.0678 delivered), its
best on any axis.**

### 25.1 THE MATCHED PAIR INVERTS OFFLINE — and it rewrites the res64 story

`sigma/target` is 4/64 at this arm and 2/32 at `res64`: **both are exactly 1/16, DSST's rule.**
So `sigma4` and `res64` are the SAME point on the width axis and differ only in resolution.
That is the matched pair sec.21 constructed offline, now measured on hardware:

```
                     offline (single-start proxy)      HARDWARE (62 seq, EAO)
sigma/target 1/16    rgb-s4      R 0.3718              sigma4   R 0.4095   EAO 0.1931
                     rgb-dec2-s2 R 0.3981              res64    R 0.3873   EAO 0.1849
                     resolution term  +0.0263 (64 wins)   resolution term  -0.0222 (128 wins)
```

**CORRECTED 2026-09-04 — THE HARDWARE RESOLUTION TERM IS POOLED-ONLY AND IS A NULL WHEN PAIRED.**
Everything in this subsection is a difference of two POOLED, frame-weighted arm scores. Computed
per sequence over the same 62 (`R-14`, `harness_validation.md`, `results/geometry_calibration.csv`):
**dR mean +0.0049, median EXACTLY 0.0000, 26 better / 25 worse / 11 tied, trim-5 −0.0040,
P(dR<=0)=0.205.** The offline multistart twin reproduces that null (+0.0119 pooled, −0.0009
paired, trim-5 −0.0092). **So "the finer map wins" is not an established effect and must not be
used to justify a geometry arm** — five other documents quoted it as one until this was checked.
The pooled figure stays in `arms.csv`: it is the official metric and it is not wrong, it is just
not separable from sampling noise across sequences. *The sentence below stands as originally
written and is what was corrected.*

**The sign flips.** On hardware, at matched mainlobe width, the FINER map wins by +0.0222 R and
+0.0082 EAO. On the common survived prefix the two produce equally good boxes (+0.0014), so the
whole difference is survival.

**Therefore sec.18-20's attribution was wrong in its emphasis.** The res64 arm's +0.0220 EAO was
the mainlobe width it carried by accident — `MOSSE_SIGMA` is in BINS, so halving the map doubled
sigma/target onto the optimum — and the resolution change it was built for is, on its own, a
small LOSS. Sec.19.3 listed three candidate mechanisms; (a) mainlobe width is confirmed and
carries the result, while (b) conditioning and (c) the prefilter cannot be worth more than the
-0.0222 the resolution term costs. Sec.21's decomposition was right about the split (75/25) and
wrong about the sign of the smaller term.

### 25.2 What this changes

- **`MOSSE_SIGMA=4.0` at 128x128 is the new shipping candidate.** Best EAO on record, host-only,
  no reflash, one flag.
- **`res64` becomes the SPEED option, not the robustness one.** It is 2.37x faster (10.4 ms
  against ~25 ms) for -0.0082 EAO. That is a real trade and worth stating as one; it is no longer
  "robustness AND frame rate".
- **The obvious combination is already exhausted.** res64 sits AT the width optimum, so there is
  no sigma left to take there — the offline curve turns over on both sides of 1/16.
- **Untested and cheap: the interior of the sigma curve at 128x128.** Only 1/32 (shipping) and
  1/16 (this arm) are measured on hardware; 1/21 and 1/13 (sigma 3 and 5) are host-only sweeps,
  and the offline turnover at 1/8 (sigma 8) says the optimum is bracketed but not located.
- **The gate held**: hold rate 15.31% against shipping's 15.51% even though median accepted PSR
  fell 30.9 -> 18.6, because `NEGATIVE_PEAK` dominates and 5.0 is far below either. Consistent
  with sec.23 — do not re-tune it.
