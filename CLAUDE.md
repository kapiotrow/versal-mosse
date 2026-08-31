# CLAUDE.md

Guidance for Claude Code working in this repo.

**Thesis scaffold (2026-08-30).** Findings are indexed in `docs/thesis/claims.md` — one row
per question answered, with a verdict, an evidence note and a run directory. Every number the
thesis quotes lives in `docs/thesis/results/*.csv`, not in prose. The evidence notes moved from
`runs/vot/*.md` to `docs/thesis/evidence/` (content unchanged; all references updated). When a
sweep finishes: append a row to `results/arms.csv`, update `claims.md`, write the note from
`evidence/TEMPLATE.md`. This file keeps the operational half — environment, build parameters,
traps, commands.

**`@thesis` tags bind code to thesis sections.** A site a thesis section describes carries

```
// @thesis <label> | <claim>[,<claim>] | <one line on what this site contributes>
```

where `<label>` is the thesis's own `\label` and `<claim>` is an id from `claims.md`.
`make code-map` (or `python3 scripts/thesis_index.py`) regenerates
`docs/thesis/code_map.md` — section → source, and the reverse. It VALIDATES both fields: an
unknown claim id, a label the thesis does not define, and a label the thesis defines twice are
each reported. Add a tag when you add a site a chapter will describe; do not tag every function.
**Documentation in this repo is English-only**, including these tags — the thesis is Polish and
`docs/thesis/glossary.md` is the bridge: table A is terminology `teoria.tex`/`przeglad.tex`
already fixed and must be reused verbatim, table B is proposed implementation vocabulary to
accept or amend there, and table D lists four pairs that must not share a Polish word.

**Every kernel and graph header carries a `COST` block.** Eight of them: `conv2d_kernel.cpp`,
`cmul_accum_kernel.cpp`, `fft_graph.h`, `ifft_graph.h`, `mosse_graph.h` (whole design),
`roi_crop.cpp`, `roi_crop.h` and `camera_capture.cpp`. Fixed fields — compute, vectorization,
pipelining, tile/BRAM memory, interface, caveat — each citing the CSV or the log it was read
off, so `sec:wydajnoscZasoby` can be written from the headers. **The caveat field is
load-bearing**: AIE figures are compiler-SCHEDULED cycles and not a profile, the FFT chain's
2.2 ms is INFERRED rather than logged, and `roi_crop`'s are the only MEASURED cycle counts in
the set (an hw_emu VCD probe — hw_emu wall time is meaningless, but its simulated PL cycles are
RTL-accurate). AIE compute is not frame time: the frame is 84% CPU-bound.

**The thesis's tables are generated.** `make thesis-tables` turns `docs/thesis/results/*.csv`
into booktabs `tabular` bodies in `docs/thesis/tables/` (gitignored -- the CSV is the source).
The chapter `\input`s one inside its own float and keeps the caption, the label and the
placement; Polish row labels live in the CSV's `*_pl` columns so neither a regeneration nor an
Overleaf edit can clobber them. Copying into the thesis repo is MANUAL by default;
`python3 scripts/csv2tex.py --overleaf [PATH]` opts in and only writes files -- it never adds,
commits or pushes. `--check` exits 1 when a table is out of date.

**A measured number in a comment must say where it came from.** `make check-docs`
(`scripts/check_doc_numbers.py`) reports two things: a decimal that duplicates a value in
`results/*.csv` while citing nothing — the drift risk, since a re-swept arm silently makes that
comment a lie — and a frame-time or tracking figure that no CSV records and nothing sources.
A claim id, a `results/`/`docs/` path, or the `.log` it was read off all count as citations.
**The number itself usually stays**: deleting "8.71 -> 1.88 ms, 4.6x" from the comment that
explains the hypot fix makes the code worse. Both classes are at zero as of 2026-08-30.

MOSSE correlation-filter tracker with CNN features on Versal VEK280. Extends the AIE 2D-FFT
tutorial (XD073) with a full object-tracking pipeline. Papers in `docs/papers/` (Bolme MOSSE,
Danelljan DSST); section numbers below refer to them.

## Environment setup

```bash
source setup_env.sh
```

Sets `PLATFORM_REPO_PATHS`, `XILINX_VITIS`, `COMMON_IMAGE_VERSAL`, `PLATFORM`
(`xilinx_vek280_base_202520_1.xpfm`), and `DSPLIB_VITIS` — the Vitis Libraries **root**, not
the `dsp` subdirectory (the Makefile appends `/dsp`).

## Build parameters

| Parameter | Default | Notes |
|---|---|---|
| `TARGET` | `hw_emu` | `hw_emu` or `hw` |
| `PATCH_ROWS` / `PATCH_COLS` | `128` | Must be powers of 2 (AIE FFT constraint) |
| `N_CHANNELS` | `16` | conv feature channels |
| `FFT_2D_DT` | `0` | 0=cint16, 1=cfloat |
| `ITER_CNT` | `1` | Frames. **Needs ≥2** — frame 0 initialises the filter |
| `PL_FREQ` | `312.5` | MHz. Platform also offers 625 / 156.25 / 100 / 78.125 |
| `H_SHIFT` | `15` | Deliberately OVER-shifted: `rails=0` over 101,564 frames, and what the flashed xclbin was built with. 13 is the tight RGB value, 12 rails, gray's arm is 14. cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget. **Reaches `AIE_FLAGS` — the ONLY non-host-only knob in this table** |
| `FFT_SHIFT` / `IFFT_ROW_SHIFT` / `IFFT_COL_SHIFT` | `4` / `4` / `4` | See "Shift budget" — validated on hardware, do not change without a ≥20-frame hw run |
| `FFT_ROW_WS` | `64` | Rows per FFT invocation — the DMA transaction-count knob. Exhausted, see history |
| `FFT_COL_WS` | `8` | Cols per FFT invocation. **32 is a 9.6 ms LOSS — do not raise** |
| `MEMTILE_TRANSPOSE` | `1` | Forward+inverse transposes in AIE-ML memory tiles. A one-sided flag is a board deadlock, not a compile error |
| `ROI_CROP_PIPELINE` | `1` | Launch channel k+1's crop before polling k |
| `CMUL_SPLIT_ACCUM` | `1` | `accum_prev` gets its own kernel port. **`make aiesim` needs `0`** (2025.2 aiesim deadlock); hardware is fine either way |
| `CMUL_ACCUM_MEMTILE` | `0` | Tried, a 0.36 ms loss. Code kept behind the flag |
| `TAIL_PARALLEL` | `1` | `filter_update_quantize` on core 1 ∥ scale filter on core 0. Needs `-pthread` in `GCC_FLAGS` |
| `ROI_CROP_USER_MANAGED` | `1` | 1 = roi_crop driven via `xrt::ip` (host writes AXI-Lite, polls the CU's own `ap_done`). **20.6× on frame rate** — KDS completion costs 503 ms/launch because the CU interrupt is never delivered. Host-only |
| `CONTROL_CU_RUNS` | `8` on `hw` | camera_capture launched N times at startup on the KDS path — a within-run control that should still pay ~512 ms while roi_crop does not |
| `CONV2D_MODE` | `0` | 0 = real 3×3 conv, 1 = echo passthrough, 2 = synthesize |
| `CONV_VECTORIZE` / `CMUL_VECTORIZE` | `1` | Vectorized kernels, bit-identical to scalar; 0 restores scalar for bisection |
| `CONV_RELU` | `0` | Off — ReLU costs ~25% of the peak/sidelobe ratio |
| `CONV_IN_CH` | `3` | conv2d input planes. 1 = BT.601 luminance, 3 = RGB. Picks the **weight-buffer layout**, so it drives `AIE_FLAGS`, `GCC_FLAGS` and `ROI_IN_CH` from this one variable. Both arms build and link; see the RGB section |
| `CONV2D_STACK` | `2048` | conv2d's AIE stack, bytes. Applied **only at `CONV_IN_CH=3`**, where the 27-tap chain needs 1344 against the 1024 default and the mapper otherwise refuses to emit a `libadf.a` |
| `BIAS_SCALE` | `roi` | `bias_acc` input scale for `make weights`. **Default changed 2026-08-23**; `127` restores the pre-correction weights. See "The bias_acc correction" |
| `FRAME_RGB_MODE` | `1` | Synthetic scene colour at `CONV_IN_CH=3`. 1 = per-plane tint; 0 = replicate luma — the COLOUR-FREE CONTROL, which reproduced grayscale bit-for-bit on hardware. Inert at `CONV_IN_CH=1`; a real frame source ignores it. Host-only |
| `FRAME_SOURCE` | `synth` | `synth` = the generated scene (unchanged, and `vot_source.cpp` is not even linked); `vot` = frames memcpy'd from a converted VOT blob, geometry and init box from its manifest. At `vot` the scene generator, `TRAJECTORY`, `OCCLUDE_MASK`, `BG_PAN` and `FRAME_NOISE` are all inert, and `ITER_CNT` is ignored — the run length is the job's. `vot` + `CONV_IN_CH=3` needs the `.luma` sidecar, which the converter emits and the host streams or stages; this is the SHIPPING combination. Host-only. See `docs/thesis/evidence/phase2.md` |
| `RESET_MUTANT` | `0` | Deliberately breaks ONE item of `run_reset()` so the multi-start determinism test's ability to FAIL is demonstrated, not assumed: `1` mean_prev, `2` filter_bo, `3` g_filter, `4` coast, `5` scale reconfigure. Non-zero prints a banner and invalidates the run's tracking output. See `docs/thesis/evidence/phase3.md`. Host-only |
| `VOT_DATA_DIR` / `VOT_RESULTS_DIR` / `VOT_SEQUENCE` / `VOT_JOB` | `/mnt/vot` / `/mnt/vot-results` / `car1` / `0` | Compiled-in defaults, each overridable on the board's command line (`--vot-data`, `--vot-results`, `--vot-seq`, `--vot-job`, `--vot-jobs all|N,M,...`, `--vot-max-frames` which then REFUSES to write the trajectory). **Repeating a job index in `--vot-jobs` is the determinism test** — its two trajectories must come back byte-identical. An unrecognised argument is fatal. Host-only |
| `VOT_RESIDENT_MAX_MB` / `VOT_STREAM_RING` | `700` / `8` | Blob+sidecar size above which a sequence STREAMS from the NFS mount through a prefetched ring, and the ring depth. Exists because usable heap is ~0.9-1.2 GB, not 12 GB: five RGB sequences died on `std::bad_alloc` while 57 completed. Ring < 2 is REFUSED, not clamped. `--vot-stream auto|always|never`; `always` is the MODE-EQUIVALENCE TEST — streaming changes no arithmetic, so digests must be IDENTICAL both ways. Host-only |
| `SCENE_VERIFY` | `0` | Re-colourise the whole frame each push and abort on a mismatch. O(frame)/frame — for a short `MODE=bringup` run only (`calib_build.sh` refuses otherwise). It was a bare `#ifndef` before 2026-08-24, so `SCENE_VERIFY=1` silently built it DISABLED. Host-only |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `FILTER_MASK` | `0` | Spatial reliability: the projection `h ← m⊙h` on the published `H`, i.e. CSR-DCF's item. The window is FORCED — only the periodic Hann has an exactly sparse spectrum — so there is no width knob, and once forced the constants collapse to `H ← D_row(D_col(H))/16` with `D(X)[i]=2X[i]−X[i−1]−X[i+1]` circular: **8 complex adds/bin, no multiplies**, verified against an exact FFT to 8e-16. Applied in BOTH publish paths (the fused one AND `filter_quantize_q15`, which is frame 0's) and BEFORE the max-\|H\| scan, since it moves the Q1.15 scale. Host-only. **SWEPT AND ACCEPTED 2026-08-31: EAO 0.1629 → 0.1740 (+0.0110), R +0.3417 → 0.3608, 16.6% more frames tracked; A −0.0187 pooled but **+0.0179 on the common survived prefix**, i.e. the accuracy loss is a selection effect. Offline over-predicted the gain 3× (dR +0.0601 vs +0.0192). `docs/thesis/evidence/spatial_mask.md`, claim R-10.** The DEFAULT IS STILL `0` pending the `PSR_GATE_MIN` re-tune the mask forces. **`cmp` cannot check its inertness** — see `evidence/proposed_build_mask.md` §3 |
| `FILTER_MASK_STAT` | `0` | Logs `mask_ebox`, the fraction of `Σ\|h\|²` inside a centred target-sized box, as a trailing `track.csv` column. THE mechanism check for the above, and independent of it so the baseline is measurable with the same instrument. **51.6%/54.9% are AT-INIT values and the fraction RISES as the filter converges** (car1 0.514→0.741 unmasked); comparing a run's mean against them would confirm the mechanism on an unmasked arm. **`-1` is NOT MEASURED, and at the shipped schedule that is 94.6% of rows — do not average as zero and do not read it as a hold rate.** Sampled, not per-frame: the statistic is an inverse FFT per channel (30.1 ms/call on the A72, more than a whole frame), so see `FILTER_MASK_STAT_WARM`/`_EVERY`. Host-only |
| `FILTER_MASK_STAT_WARM` / `FILTER_MASK_STAT_EVERY` | `5` / `20` | Sampling schedule for `mask_ebox`: the first WARM frames of each run plus every EVERY-th after. **Matched to `vot_mask_stat.py`'s `PROFILE_FRAMES` (1, 5, 20, 40)** — move either and that reader's per-frame columns silently thin out. Host-only |
| `PSR_GATE_MIN` | `5.0` | +0.0134 R against 7.0 on the full benchmark. **Its worth depends on the PSR scale**, so anything moving PSR re-opens it. Bolme §3.5. Below it the host HOLDS position and skips `filter_update` + `publish_filter`. `0` disables the threshold test only (structural vetoes remain). Host-only |
| `TARGET_H` / `TARGET_W` | `64` | Target box size, frame px. Host-only |
| `TARGET_PADDING` | `2` | Settled at 2.0 three ways; 1.5 and 3.0 both measured worse. `roi = box × padding`. At 64/2 the ROI is 128 ⇒ resample is 1:1. Host-only |
| `MOSSE_SIGMA` / `SIGMA_FROM_TARGET` | `2.0` / `0` | `SIGMA_FROM_TARGET=1` applies DSST's target/16 rule (σ=4 at padding 2) |
| `MOSSE_ETA` | `0.05` | +0.0218 R, +8.5% EAO against 0.125. Not monotone — 0.025 is much worse, so it is a shallow optimum. Translation filter learning rate. Host-only |
| `SCALE_N` / `SCALE_STEP` | `33` / `1.04` | DSST scale levels; `SCALE_N=1` disables the scale filter. **1.04 beats DSST §6.1's 1.02 on hardware** (IoU 0.807 → 0.917) |
| `SCALE_ETA` | `0.025` | Scale filter learning rate (deliberately ≠ `MOSSE_ETA`) |
| `SCALE_CONF_MIN` | `2.0` | Scale gate on `conf`. A veto HOLDS the box and SKIPS `scale_update()`. `0` disables the threshold test only. Host-only |
| `HOLD_COAST` / `COAST_DECAY` | `0` / `0.5` | `1` = a held frame moves the search window at the last measured velocity, decayed each held frame (drift bounded by 2v); `0` = the freeze. **Flipped to 1 and reverted the same day**: the same 54 trajectory pairs win on mean IoU (0.2709 → 0.3005) and LOSE on the toolkit's metric (A 0.638 → 0.616, R 0.309 → 0.288, EAO 0.208 → 0.194). AR is the metric of record. See `docs/thesis/evidence/evidence_ar.md`. Host-only |
| `SCALE_MAX_STEP` | `2` | Largest `|idx|` ONE frame may move the box — a RATE limit, where `MIN_REL`/`MAX_REL` are a drift bound. `0` disables. **`1` was measured and rejected**: `scale_sim` parks the smooth arm 123 of 200 frames and ends 28.0% wrong. Added after `car1` f490 inflated the box 1.42× while 227 px off target. Host-only |
| `SCALE_MIN_REL` / `SCALE_MAX_REL` | `0.5` / `2.0` | Absolute drift bounds vs initial box size. Must still admit `SCALE_TRAJ_AMP` (0.70×..1.30×) |
| `OCCLUDE_MASK` | `0` | Bitmask over frame index: bit *f* ⇒ frame *f* occluded. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test |
| `OCCLUDE_SQUARE` / `OCCLUDE_START` | `8` / `30` | `OCCLUDE_START` is a warm-up: 30 ≈ 4 time constants at `MOSSE_ETA=0.125`. The scale filter needs ~120 frames to settle |
| `TRAJECTORY` / `TRAJ_AMP_R,C` / `TRAJ_PERIOD` | `0` | 1 = closed elliptical path (absolute ground truth) |
| `SCALE_TRAJ` / `SCALE_TRAJ_AMP` / `SCALE_TRAJ_PERIOD` | `0` | Sinusoidal size envelope |
| `FRAME_TEXTURE` | — | Band-limited background instead of a flat fill |
| `FRAME_NOISE` | `2` | Per-frame sensor noise, PEAK amplitude in LSB, over the ROI. Does **not** fix background lock |
| `BG_PAN` / `BG_PAN_R` / `BG_PAN_C` | `1` / `31` / `47` | Camera pan over the cached background, px/frame. Decorrelates the background 6.6× (swept with `scripts/bg_pan_sweep.py`) but **did not fix tracking** — see the training-target trap. Host-only |
| `PROGRESS_EVERY` | `1` | Frames between LEVEL-0 progress lines. `1` = byte-identical to every pre-2026-08-25 run (proven by an ELF `cmp`). Thins the marker, never silences it: frame 0 and the last frame always print, because a run missing its final line looks exactly like a run that hung. **Only has effect at `VERBOSITY=0`.** Worth 0.27 ms, not the 4.0 predicted. Host-only |
| `CSV_FLUSH_EVERY` | `1` | Rows between `track.csv` flushes. Per-row flushing survived a power cut that really did take out a run. A **railed** row flushes regardless of N; a gate veto deliberately does not. Worth 0.00 ms from tmpfs. Host-only |
| `VERBOSITY` | `1` | `0` = one compact line/frame (~45 B); `1` = per-frame block, roi_crop/DMA tables on first+last frame only; `2` = everything. Anomalies print at every level. **No longer a diagnostics trade-off** — since 2026-08-24 `track.csv` carries `rails`/`accum_max`, so a `VERBOSITY=0` run is a full budget verdict AND an FPS measurement. Host-only |
| `DUMP_BUFFERS` | `1` | Per-frame binary dumps. **1216 KB/frame, ~2 s/frame**. Set `0` for any run measuring tracking or FPS |
| `CSV_LOG` | `1` | One row/frame to `track.csv` (`track_<sequence>.csv` at `FRAME_SOURCE=vot`, since one sweep is one invocation per sequence and the file opens `"w"`; the arm is separated by `--vot-results` instead) — gate verdict, both PSRs, peak, displacement, `resp00_over_peak`, both boxes, IoU, centre error, scale fields, and (2026-08-24+) `rails,accum_max,fch0_max,h_max`. ~60 B/frame |

**THE DEFAULTS ARE THE SHIPPING CONFIGURATION as of 2026-08-28.** A bare
`make application FRAME_SOURCE=vot` reproduces the benchmark arm's `app.flagstamp` exactly, and
the default `aie.flagstamp` matches the flashed `a.xclbin`. Four defaults moved that day:
`CONV_IN_CH` 1->3, `H_SHIFT` 11->15, `MOSSE_ETA` 0.125->0.05, `PSR_GATE_MIN` 7.0->5.0.
**Grayscale is still fully supported** — pass `CONV_IN_CH=1`, and note it is what the aiesim
scenarios `s6`/`s7` need.

Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

### Shift budget — SETTLED: 4-4-4, `H_SHIFT` 14 (gray) / 15 (RGB)

**FFT budget closed 2026-08-24 on five 200-frame runs; `H_SHIFT` closed 2026-08-27 on real
video (see "Shift budget on real video").** The FFT budget never moved — every fix has been
`H_SHIFT`, the only knob upstream of **both** the accumulator and the response (`IFFT_*`
reaches only the response, `FFT_SHIFT` moves it two bits at once).

The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so
weight moves freely between passes (holds to 1.3% across splits). `FFT_SHIFT` stays 4 rather
than 5 because that leaves the accumulator at ~1400 instead of ~330 for the same response.
Retired points: 4-5-5 (total 16) undershot 6-11×; 5-3-4 (17) gave 0.4%; 4-2-2 (12) peaks at 56%
on frame 1, then rails from frame 15 and sign-flips to −32768, holding forever on
`NEGATIVE_PEAK`; 4-2-1 was never validated past frame 1. `IFFT_ROW_SHIFT=0` is unsafe at ch16.

**Do not re-centre the response in the 49-64% band.** That band came from a distribution with a
1.30× spread; the corrected build spreads 2.07× at the converged end, so centring the TYPICAL
frame puts the TAIL on the rail. Size against the tail. The response sits at ~28% (gray) / 22%
(RGB) and `calib_report.py` calls that UNDERSHOOT; advisory, and PSR is the arbiter.

**Four rules this budget cost real time to learn:**
1. **The response GROWS as the filter converges** — a budget validated at `ITER_CNT=2` is not
   validated. The 08-24 rail appeared at f173 and the 98% peak at f187; use the full 200.
2. **Twice an offline model set this budget and hardware overturned it.** Both times the model
   was self-consistent and its *premise* was wrong (see frame-buffer seeding under Correctness
   traps).
3. **Never size this budget against railing before checking `mean_prev` is seeded** — two budget
   hunts chased a frame-0 DC pedestal, not a scaling problem.
4. **Do not size from early frames.** RGB's response reads ~1.03× of gray at f1-4 and 0.785×
   once converged; the weights-derived estimate (0.685–0.790×) was right.

A calibration run's criterion is `rails=0` plus BIT-IDENTICAL tracking (a uniform rescale cannot
move an argmax) plus PSR not moving — PSR is where a quantization floor would show, and `F_ch` /
`H(q15)` must be digit-for-digit unchanged since both are upstream of `H_SHIFT`.

`runs/.last_cfg` is **stale and not authoritative**; `build/hw/.../aie.flagstamp` is, and
`scripts/calib_build.sh` checks it for you.

## Architecture overview

```
PS (A72) — mosse_tracker.cpp
  Drives all GMIO ports in the per-frame, per-channel loop.
  Runs PSR gating, filter init/update (mosse_filter.cpp), and the DSST scale filter.

PL kernels (2)
  camera_capture : zero-fill DDR frame buffer (stub; TODO: MIPI RX)
  roi_crop       : DDR frame → Stage A → 32-bit AXIS (int8) → AIE PatchIn.
                   Bilinear resample of roi_h×roi_w to the fixed patch size with border
                   clamping, log, zero mean, unit L2 × ROI_NORM_Q, int8 quantize.
                   Two reduction passes + a stream-out pass; `recompute=1` on channel 0,
                   channels 1..15 re-stream the cache. All geometry is runtime AXI-Lite,
                   so ROI/box changes need no rebuild of anything but the host ELF.
                   At ROI_IN_CH=3 it resamples three interleaved planes with shared
                   geometry and ONE joint mean/sigma. ROI_IN_CH is compile-time.

AIE (single instances, serial per-channel; both custom kernels vectorized)
  conv2d_kernel     : int8 patch → 3×3 MAC → Hanning window → cint16 stream
                      (MobileNet-v3 Small layer 1, INT8; RGB collapsed to grayscale at
                      CONV_IN_CH=1, all 27 taps at 3). Stage B1 subtracts mean_prev,
                      which the host SEEDS before frame 0.
  fft2d             : PATCH_COLS-pt row FFT → memory-tile transpose → PATCH_ROWS-pt col FFT
  cmul_accum_kernel : col-FFT ⊙ H_ch* + accumulate (int32 intermediates, saturating cint16
                      accumulator in DDR)
  ifft2d            : row IFFT → memory-tile transpose → col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop → conv2d, **32-bit** (one int32 = 4 packed int8 pixels). It is 32-bit, not
128-bit: `mosse_graph.h`'s `input_plio::create("PatchIn", ...)` uses `plio_32_bits` because a 128-bit PLIO delivered one beat per
`readincr`, starving the kernel. The name must match between `mosse_graph.h` and `mosse_x1.cfg`
(`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

### GMIO ports

| Name | Dir | Purpose |
|---|---|---|
| `gmio_weights` | DDR→AIE | conv2d INT8 weights per channel |
| `gmio_fft_col_out` | AIE→DDR | broadcast tap: F_ch for the PS filter update |
| `gmio_cmul_in` | DDR→AIE | H_ch* per chunk |
| `gmio_accum_in` | DDR→AIE | prev Σ per chunk (`CMUL_SPLIT_ACCUM=1`) |
| `gmio_accum_out` | AIE→DDR | updated partial sum |
| `gmio_ifft_row_in` | DDR→AIE | accumulated spectrum → ifft_rows |
| `gmio_response` | AIE→DDR | correlation response |

(The four transpose GMIOs — `fft_row_out`/`fft_col_in`/`ifft_row_out`/`ifft_col_in` — were
deleted by `MEMTILE_TRANSPOSE=1`.)

### Per-frame data flow

```
host generates scene in g_frame_host, pushes to frame_bo
for ch in 0..N_CHANNELS-1:
  roi_crop → PatchIn → conv2d → fft_rows → memtile → fft_cols → cmul_accum → DDR
after all channels:
  DDR → ifft_rows → memtile → ifft_cols → gmio_response
  APU: compute_psr → gate → update pos (or HOLD)
  APU: filter_init() on frame 0, filter_update_quantize() thereafter ∥ DSST scale update
  APU: publish_packed()
```

## Key design decisions

- **AIE-centric**: all FFT/IFFT/conv/cmul on AIE; PL is only camera_capture + roi_crop; APU
  orchestrates via GMIO DDR round-trips.
- **Serial channel processing**: one FFT2D and one IFFT2D instance reused across all channels.
  Minimal PL/PLIO count at the cost of throughput.
- **Accumulator in DDR** (128×128 cint16 = 64 KB). The on-tile version does not work as usually
  stated: it is a read-modify-write by ONE kernel across invocations, so as a `shared_buffer` it
  is a graph CYCLE, and the required delay is `CMUL_N_CHUNKS` (16) invocations, not 1 — cmul
  walks chunks inner and channels outer. Keeping it in cmul's own tile needs all 16 chunk
  accumulators resident (64 KB, the whole tile) on top of the port buffers, and AIE-ML kernels
  cannot address a memory tile as random-access scratch.
- **Filter init/update on PS, with no FFT library.** `mosse_filter.{h,cpp}` implements Bolme
  eq. 10–12 with a *shared* denominator (Danelljan/DSST form — one reciprocal map per frame,
  better conditioned). `F_ch` arrives already transformed via `gmio_fft_col_out` and `G` has a
  closed form, so KissFFT was never needed. AIE would be the worst home (2 MB of filter state
  does not fit on-tile).
- **The filter update runs AFTER peak detection** — updating first leaks the current frame into
  its own detection.
- **`mosse_filter.{h,cpp}` includes no XRT/ADF header**, so `make test_host` compiles it with
  system g++ against a NumPy golden in seconds. The alternative for a sign error is an
  hours-long hw_emu frame.
- **Preprocessing is split across PL / AIE / APU**: intensity-domain steps in `roi_crop`
  (Stage A); feature-map mean removal in `conv2d` using the *previous* frame's mean (Stage B1 —
  a full channel is 64 KB, too big for a tile); a 9-bin frequency-domain correction on the APU
  (Stage B2); per-channel energy normalization folded into `H_ch*` (Stage B3). <2% added
  arithmetic, no new AIE tiles.
- **Periodic Hann, not symmetric.** `hanning_*.h` uses `sin²(πi/N)`. Its 2D DFT has exactly 9
  non-zero bins, which is what makes B2's correction exact. Measured DC/worst-leaked-bin at
  N=128 in Q1.15: periodic 2.2e5, symmetric 373. Do not "fix" this back.
- **Grayscale collapse uses ITU-R BT.601 luminance, deliberately NOT Danelljan's unweighted
  sum.** The paper-literal sum annihilates the four colour-opponent channels (0, 2, 9, 10) to
  2.5-5% of their norm; per-channel int8 quantization then amplifies the residue to full scale.
  The 11 achromatic channels agree between conventions to cos > 0.99. Luminance is the better
  of the two conventions and still discards real information — which is why `CONV_IN_CH=3`
  exists and is built. See the RGB section.
- **Scale estimation is DSST's 1-D filter, not multi-resolution search.** DSST Table 1 beats
  exhaustive on both accuracy and speed, and here exhaustive would push ±30% resampled patches
  through roi_crop→conv2d→FFT every frame, moving `|F|` and therefore the shift budget.
- **`g_frame_host` (2 MB heap) is the authority for the scene; `frame_bo` is a copy** the host
  pushes once per frame. BO writes run at 3470 MB/s against reads at 696, so pushing costs
  0.405 ms where pulling would cost ~2.9 ms. Anything writing `frame_bo` directly
  (`rc_control_cu_probe`'s zero-fill) must run before the first push.

## Resources and cost

VEK280 `xcve2802-vsvh1760-2MP-e-S`, 12 GB LPDDR4 **of which Linux maps 2 GB, and
512 MB of that is CMA — usable heap is ~0.9-1.2 GB, NOT 12 GB** (see
`docs/thesis/evidence/TODO_board_memory.md`; it cost **5** of 62 sequences on the RGB VOT
arm — the predicted 8 included three that turned out to fit, which is why the
"derive luma on the board" step recovers zero sequences and the streaming reader
was the only fix. `VOT_RESIDENT_MAX_MB` above is that reader).
The device tree declares all three banks; `/proc/iomem` shows only
`00000000-7fffffff : System RAM`, with the 2 GB at `0x8_00000000` reserved and
the 8 GB at `0x500_00000000` absent. The figure below is the PART's capacity and
has been misread as an available-memory budget once already; AIE core clock 1 GHz
(`directives/post_sys_link.tcl`). Measured on the 128×128 ch1 build:

| Resource | Available | Used | % |
|---|---|---|---|
| AIE-ML cores | 304 | 6 | 2% |
| AIE-ML memory tiles | 76 | 1 | 1% |
| BRAM18 | 1200 | 10 | 0.8% |
| DSP | 1312 | 44 | 3.4% |
| LUT | 520704 | 7694 | 1.5% |
| FF | 1041408 | 7539 | 0.7% |

**The design uses 2% of the AIE array.** Check any "we can't afford it on AIE" claim against
that first — the binding constraints have always been tile memory (64 KB/tile) and host DMA
orchestration, never core count. **`runtime<ratio>` is not utilization**: the mapper's
`Utilization` column shows declared budgets, not measured occupancy.

### Per-frame AIE compute (128×128, ch16, from `aiecompiler.log` schedules, post-vectorization)

| kernel | ms/frame | note |
|---|---|---|
| conv2d | 4.1 | 37 → 8.75 cyc/px; the untouched **stream-read loop is now 44%** of it |
| cmul_accum | 0.13 | 30 → 2 cyc/element, now pipelined |
| FFT + IFFT chain | ~2.2 (band 1.3–3.5) | **trip counts inferred, not logged** |
| **total** | **~6.4** | from ~21.6 before vectorization |

These are the compiler's scheduled cycles — real cycles on an in-order VLIW core absent memory
stalls, trustworthy for sizing but not a profile.

## Current status (2026-08-28)

**THE SHIPPING CONFIG, and every line of it is a hardware A/B:**

```
CONV_IN_CH=3  H_SHIFT=15  budget 4-4-4  MOSSE_ETA=0.05  PSR_GATE_MIN=5.0
TARGET_PADDING=2.0  SCALE_STEP=1.04  SCALE_MAX_STEP=2  HOLD_COAST=0
```

Full VOT-STb2022, 62 sequences, 419 trajectories per arm. Each row moves ONE knob:

| arm | A | R | EAO | workspace |
|---|---|---|---|---|
| gray `H_SHIFT=14` | 0.4890 | 0.2743 | 0.1367 | `full62` |
| RGB `H_SHIFT=15` | 0.5043 | 0.3065 | 0.1474 | `full62` |
| + `MOSSE_ETA=0.05` | 0.5100 | 0.3283 | 0.1600 | `eta05ab` |
| **+ `PSR_GATE_MIN=5.0` — SHIPPING** | **0.5100** | **0.3417** | **0.1629** | `gate` |
| + `TARGET_PADDING=3.0` — REJECTED | 0.5030 | 0.3494 | 0.1570 | `pad30ab` |
| + `FILTER_MASK=1` — EAO up, but NOT SEPARABLE FROM A NULL | 0.4913 | 0.3608 | 0.1740 | `0831_mask` |

`PSR_GATE_MIN=0` is a null (13-seq probe, `g0partial`). The padding arm gained R and LOST
EAO — **EAO is the arbiter for an A/R trade**, see the feature-geometry entry under Settled.
Every arm after the first two is HOST-ONLY: an scp, not a card swap.

**The gate's value is CONDITIONAL on the PSR scale.** 7.0 -> 5.0 is worth +0.0134 on hardware,
and offline it is +0.006 at 128x128 against +0.056 at 64x64. Anything that moves PSR re-opens
it — as `MOSSE_ETA=0.05` did (it dropped pre-loss PSR to 13.91 against a 7.00 threshold, which
is why the gate arm existed at all).

**Best hardware FPS: 26.29 ms/frame = 38.04 FPS** (`runs/run_0821_1725.log`, gray, synthetic,
UART console). RGB is 28.58 ms. `car1` over ssh is **24.43 ms = 40.9 FPS** and is NOT
comparable to either — the UART alone was 3.79 ms. Quote FPS only from a serial-console run.

Heap, not tracking, was the last blocker: five RGB sequences exceed the board's ~1 GB and died
on `std::bad_alloc`. `vot::StreamBlob` (ring + prefetch, `--vot-stream`) closed it, proven by
identical run-state digests both ways. `docs/thesis/evidence/TODO_board_memory.md`, CLOSED.

Calibration closed 2026-08-24 on five 200-frame runs; the shift-budget entry above carries the
result. The full chain runs on hardware at 128x128 ch16 on the real conv path: roi_crop ->
PatchIn -> conv2d -> B1 -> row FFT -> transpose -> col FFT -> cmul(H_SHIFT) -> B2 -> IFFT rows
-> transpose -> IFFT cols -> response -> PSR gate -> filter update -> scale update.

### Performance history

| date | change | frame ms | FPS | log |
|---|---|---|---|---|
| — | baseline (console at 115200) | 880 | 1.14 | |
| 08-20 | console gating (`VERBOSITY`) | 180.6 | 5.54 | `run_0820_1528` |
| 08-20 | BO copy pattern + int64 energy | 143.3 | 6.98 | `run_0820_1554` |
| 08-20 | scene on the host | 134.6 | 7.43 | `run_0820_1604` |
| 08-20 | `-O3 -mcpu=cortex-a72` | 132.2 | 7.56 | `run_0820_1610` |
| 08-20 | hypot fix + `-fcx-limited-range` | 127.7 | 7.83 | |
| 08-20 | `FFT_ROW_WS` 8→16→32→64 | 89.5 → 70.9 → 60.7 | 16.48 | `run_0820_1807` |
| 08-21 | scale-extract real DFT, fused filter update+quantize, folded diag scan | 45.60 | 21.93 | `run_0821_1109` |
| 08-21 | memory-tile transpose | 35.58 | 28.14 | `run_0821_1348` |
| 08-21 | software-pipelined roi_crop | 31.48 | 31.81 | `run_0821_1402` |
| 08-21 | `CMUL_SPLIT_ACCUM` | 29.61 | 33.77 | `run_0821_1452` |
| 08-21 | blocked `unpack_spectrum` | 28.64 | 34.92 | `run_0821_1635` |
| 08-21 | tail split onto core 1 (`TAIL_PARALLEL`) | **26.23** | **38.15** | `run_0821_1712` |
| 08-24 | RGB (`CONV_IN_CH=3`) — cost is host memory, not conv2d | 28.58 | 34.99 | `run_0824_1457` |

**Every one of these was accepted on a bit-identical-tracking test**, and that criterion has now
caught two bugs it was not designed for (the `g_target_shift` race, the `FFT_COL_WS` datapath
check). A tolerance-based check would have shrugged at IoU 0.48.

### Where the frame goes — gray 26.29 ms, RGB 28.58 ms

```
                  gray    RGB      (VERBOSITY=0, run_0821_1725 / run_0824_1457)
APU subtotal    15.639  17.442
of which OVERLAP -2.762  -2.746   <- ran CONCURRENTLY on core 1
APU wall        12.876  14.696    <- what the frame actually spent
GMIO            11.134  11.133    (async 6.6 / wait 4.5 — UNCHANGED by RGB)
roi_crop launch  1.013   1.471    (= channel 0's Stage A, structurally exposed)
UNATTRIBUTED    +1.261  +1.284
```

**The frame is 84% CPU-BOUND**, not wait-bound: host CPU = APU 15.4 + GMIO async 6.6 + roi_crop
1.0 + unattributed 1.2 = 24.2 ms of a 28.7 ms frame (measured pre-threading). Only 41% of GMIO
blocks. **The second core's value is splitting 24 ms of work, not filling 4.5 ms of gaps**;
perfect two-core use floors at ≈12-15 ms, 65-80 FPS. That slack is also why RGB is nearly free
(see "RGB costs what the HOST pays").

**Overlap accounting is measured, not estimated.** With H the helper's elapsed time and W the
main thread's time blocked in `join()`, the region's WALL cost is (main's own work + W) while the
slots credit (main's own work + H), so the double-count is exactly **H − W**. It validated
itself: overlap 2.762 against a predicted `min(4.80, 2.89)` = 2.9, and the residual returned to
+1.261 against +1.17 before threading — a correction tuned merely to erase a negative number
would have had no reason to land there.

### Shift budget on real video — CLOSED 2026-08-27

`docs/thesis/evidence/TODO_shift_budget.md`. `car1` railed on 266 of 8434 frames at `H_SHIFT=11`, attributed
to BOTH `accum` and `response` — so `H_SHIFT`, the only knob upstream of both, was the lever. Two
deliberately over-shifted arms returned the UNCENSORED distribution: over 101,564 RGB frames
`rails_accum = rails_resp = 0`, maxima 15.6% / 10.1% of ceiling. **`H_SHIFT=13` is the
tight-but-safe RGB budget; 12 rails.** The shipped arms stay at gray 14 / RGB 15 because
`rails = 0` is the only hard criterion and the whole benchmark is banked on them. **Stated
confound: the benchmark's two arms are at different `H_SHIFT`.**

Three things not to re-derive:
- **`accum_max = 46340 = 141.4%` IS NOT OVERSHOOT** — it is 32767·√2, the largest magnitude a
  non-saturated cint16 bin can hold (the rail is per COMPONENT, `accum_max` is a magnitude).
  **`rails` is the only saturation instrument.**
- **Readings are CENSORED at the rail**, so no budget is derivable from a clipped maximum. That
  is why the over-shifted arm had to be built.
- Rails do NOT correlate with tracking loss (`corr = −0.025`) and do not explain the
  `NEGATIVE_PEAK` vetoes. A budget defect, never a tracking fix.

**`H_SHIFT` is the one knob that is NOT host-only** — it reaches `AIE_FLAGS`, so it needs a graph
rebuild, re-package and re-flash, and the sweep's xclbin guard will refuse until the card is
updated. **Re-provision after packaging**: `v++ --package` takes `build/rootfs/rootfs_compat.ext4`,
which `make rootfs` regenerates only when the pristine rootfs changes, so a re-package inherits
whatever that copy holds — it did not hold the ssh key until 2026-08-25, which produces a board
that boots unreachable and reads as a cable fault.

### Next, in order — ROBUSTNESS

Localisation, the gate, quantization, saturation, pooling, feature resolution, padding and
**init perturbations** are ALL exonerated or rejected (entries below). Ranked list and the
supporting measurements: `docs/thesis/evidence/robustness_proposals.md`. What is left is host-only:

1. **THE NEXT BUILD: the 64x64 feature map (`PATCH_ROWS=PATCH_COLS=64`) — PROPOSED 2026-08-31,
   `docs/thesis/evidence/proposed_build_res64.md`, claim N-03b.** The first arm whose offline
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
   **A spatial mask on the filter — MEASURED OFFLINE 2026-08-28.** `docs/thesis/evidence/proposed_build_mask.md`. CSR-DCF's highest-priced item, applied
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
   failure-rule artifact test below and three checks clear it: mean IoU RISES where the `gsign`
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
4. **A two-filter temporal ensemble** (TCLCFcpp, the one embedded tracker at R 0.598). eta 0.125
   and eta 0.05 already win on different sequences; run both, select per frame by PSR. Host-only,
   +2 MB filter state, roughly doubles the ~5 ms filter tail. **Take the cheap half first**: a
   long-term filter at eta≈0 used only as a VALIDATOR at the selected peak (one dot product per
   channel, item 2's identity, no second AIE bank) feeding a confidence-modulated eta.
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
   *(Original entry:)* **A REPLACEMENT feature bank** — not a filter over this one, which was tested and is a null.
   HOG is orientation binning of gradient magnitudes into 31 dims; Danelljan's conv1 is 96ch 7×7
   PCA'd to 40; this is 16×3×3 with participation ratio 7.43. A weights export plus the offline
   loop, no re-synthesis. **It re-opens item 1 of Settled below**: fewer channels, or per-channel
   denominators instead of the shared one, removes the structural cure that makes Bolme's init
   perturbations a null here.

**Recovery AFTER a loss is worth NOTHING to R or EAO** — VOT terminates a run 10 frames after
failure and re-enters at the next anchor. Only not-losing, or recovering inside the grace,
scores. That retires re-detection and search-window expansion outright, however good they look
on mean IoU.

**Score any of these on `vot analysis`, never on mean IoU** — the two have ordered arms
oppositely on identical trajectories (`HOLD_COAST`), and the offline AR proxy
(`vot_ar_offline.py`) has a MEASURED resolution of only ~0.02 in R and did not transfer to a
geometry arm.

### Next, in order — PERFORMANCE

The APU is a **flat tail** — biggest single item 5.2 ms — so the remaining wins are structural.

1. **Software-pipeline the CHANNEL loop.** The `fft_col_out` + `accum_out` pair (~8.7 ms) is the
   col-FFT + cmul production time for 16 channels, proven immune to transaction count. Overlap it
   with the host's ~0.4 ms/channel of APU work, exactly as `roi_crop` was pipelined
   (5.196 → 1.020 ms). The graph already permits one channel of lookahead; the host serialises.
2. **More of the second core.** Parallel-for over channels inside `filter_update_quantize` —
   expect well under 2×, it is memory-bound and the two cores share a controller. **See the
   abandoned attempt below before trying.** Then re-enable `CMUL_ACCUM_MEMTILE`, which only pays
   once a helper exists to absorb the wait (alone it loses 0.36 ms; paired, ~2.7 ms of freed CPU).
3. **NEON-vectorise the int16→float conversion in `unpack_spectrum`** (`SSHLL` + `SCVTF`) — maybe
   1 ms, and the only remaining idea for that slot.
4. **6.6 ms/frame of XRT descriptor cost** — the two 256-tx ports cost 11.0 µs of host CPU per
   `async()`. The only lever is fewer, larger transactions, i.e. item 2's memtile.
5. **At `CONV_IN_CH=3` only: `frame push` 1.385 ms.** The 6 MB `frame_bo` is now the single
   biggest RGB-specific cost — bigger than everything conv2d added to the frame.

**Retired, do not reopen:** `FFT_COL_WS` 8→32 (a 9.57 ms loss), `CMUL_ACCUM_MEMTILE` alone (0.36
ms), Hermitian symmetry in the host filter (premise refuted in fixed point), the accumulator as a
`shared_buffer`, parallel-for inside `filter_update_quantize` as attempted. Each has an entry
below.

### Parallel-for inside `filter_update_quantize` — ATTEMPTED AND ABANDONED (2026-08-21)

**~0.96 ms (3.6%) was not worth what every formulation cost in bit-exactness.** The gain was
bounded before it started: the tail is `max(scale 2.89, filter 4.80) + publish 1.91 = 6.71 ms`,
and **`publish` cannot move to the other core** (it consumes `filter_scratch`/`q15_scale`), so
splitting the filter internally only helps on its last 1.91 ms.

Three formulations — split by element range, split by channel, and a `noinline` wrapper — were
each bit-exact under `-O2` and each 1 ulp off under `-ffp-contract=fast`, always the same
82/1024 elements of **A** at 1.49e-08. **Root cause: GCC's FMA contraction is sensitive to
INLINING CONTEXT, not just to the expression** — at `nw == 1` the worker body inlines, at
`nw > 1` it is emitted out-of-line, and the two contract `eta*conj(G[i])*f[i] + keep*a[i]`
differently. 1 ulp of A was disqualifying because A is carried frame to frame at eta = 0.125, so
it settles at ~8 ulps and flips occasional bins across 200 frames — tracking would come back
*nearly* identical, the one outcome that makes the bit-identical criterion useless.

**What survives for a next attempt.** (1) `make test_host`'s `-ffp-contract=fast` second build
found every one of these; `-O2` found none. (2) An element-range split does protect B's
channel-order sum. (3) The `|H|` max scan needs a lowest-global-index tie-break to be split at
all, since `std::abs()` is `hypot()`. (4) **Parallelise across FUNCTIONS, not inside them** — the
`TAIL_PARALLEL` split worked precisely because it moved a whole function and touched no
arithmetic.

### Scale filter — root-caused offline, confirmed on hardware (2026-08-20)

`design/host_app_src/test/scale_loop_sim.cpp` (`make scale_sim`) drives the REAL
`scale_extract`/`scale_detect`/`scale_gate`/`scale_update` in a closed loop with position held.
Native g++, seconds per arm. It **reproduces the board** (the `moving` arm parks for 42 frames
from f131; hardware froze at f130) and refuses to report a verdict when the premise arm fails to
reproduce.

**`SCALE_STEP=1.04` confirmed on hardware** (`run_0820_1513`): mean/worst IoU 0.807/0.579 →
**0.917/0.833**, max box error 31.4% → **9.6%**, mean/worst centre error 2.47/11.07 px →
**1.30/3.52**. **Centre error fell 3.2× from a size-only change** — independent confirmation
that position error was downstream of the scale error.

**On hardware the detector proposed only −1, 0 or +1 over 199 frames** (174/13/12) — but **that
sentence describes ONE SYNTHETIC SCENE and failed twice on 2026-08-25.** On `car1` the detector
proposed ±2 or more seven times, up to **+9**, every one on a frame whose IoU was 0.000; and in
`scale_loop_sim` it uses ±2 *legitimately* on a smooth envelope — capping at 1 parks the
`moving` arm for 123 of 200 frames and ends it 28.0% wrong against 1.0% uncapped. Reading ±1 as
a property of the DETECTOR rather than of that scene is exactly what `SCALE_MAX_STEP=1` would
have been; the sim, not the hardware observation, is the bench that decides that parameter.

**The scale filter is ALREADY slaved to the position gate, and that is not sufficient.** The
whole block runs under `if (scale.enabled() && gate.accept && scale.initialized)`, verified on
hardware (`run_0825_1314`: 577 ACCEPTs, 577 scale evaluations, zero on a held frame). `car1`
frame 490 still got through because the POSITION gate accepted it at PSR 7.87 while the tracker
was 227 px off target. **Do not "add" that slaving; it is there.** `SCALE_MAX_STEP` exists
because the precondition is only ever as good as the PSR gate.

**`SCALE_CONF_MIN` blocks legitimate large corrections.** On the sim's `step` arm the detector
proposes the correct `idx=-14` and the gate vetoes it as `LOW_CONF` for four frames; the box
then walks at 2%/frame where bypassing the gate corrects it in ONE frame. **`conf` cannot
distinguish "wrong proposal" from "big correct correction"** — both match the model poorly, for
the same reason. Exonerated for smooth envelopes; it will bite on any abrupt scale change.

**Where it stops.** `SCALE_ETA` does not help (8.6/10.3/9.2% at 0.025/0.05/0.1); `SCALE_N=65`
gives 7.0% against 8.6% for **3.9× the cost**. ~8-10% box error is this filter's practical floor
— and it is what the worst IoU frames on every 2026-08-24 run are. The next gain needs a
different estimator, not a tuning change.

**Calibration honesty: the sim predicted a=1.04 well and a=1.02 badly** (12.6% → 8.6% predicted;
31.4% → 9.6% measured), and it recovers where hardware never did — **unexplained**; position
error and background pan were both ruled out. **Trust the ORDERING; do not quote the sim's
absolute magnitudes as the board's.** The truth rate matters and the test bench chose it:
`SCALE_TRAJ_AMP=0.30` over 200 frames shrinks the target at ~0.94%/frame, under half of one
scale level, so on most frames the correct proposal rounds to 0.

### hw_emu frame times — MEASURED, and they do not scale from ch1

```
                 ch1        ch16
per channel    ~50 min    ~43 min
per FRAME      ~50 min    ~11.5 h
ITER_CNT=2      ~1.7 h      ~23 h
```

**hw_emu wall clock does not track AIE compute.** The echo-mode ch16 run, where conv2d did no MAC
work at all, still took ~14 h/frame; the emulator is simulating the PL and the DMA/NoC traffic.
Vectorizing a kernel speeds up the *design*, not the emulation of it. Always size runs from
measured wall clock at the same `N_CHANNELS`.

## Measurement methodology — the rules that were paid for

Two principles that have repeatedly earned their keep: **instruments before changes**, and
**never move two magnitudes at once**.

- **NEVER SIZE A CHANGE FROM ONE MEMBER OF AN INTERLEAVED async/wait GROUP. SUM THE GROUP.**
  The first `wait()` in an interleaved pair absorbs the other's production latency. This has
  appeared four times: `gmio_fft_row_out` absorbing the weights feed (286 µs/tx vs siblings'
  18 µs), the `fft_col_out`/`accum_out` pair under `FFT_COL_WS` (single ports looked like a win;
  pair total went 9.00 → 18.32), `gmio_cmul_in` absorbing both inputs after the port split, and
  `CMUL_ACCUM_MEMTILE`. **It is the single most repeated measurement error in this design**, and
  it caused a whole optimisation to be built on an artifact.
- **`DMA_TX`/`DMA_T` must time `async` and `wait` separately.** Fused, every port figure is a
  lie about mechanism. Splitting them revealed 6.6 ms/frame of host descriptor cost that was
  invisible.
- **Measure the total and print the residual.** A profiler that does not account for the whole
  frame lets you conclude confidently and wrongly — twice here.
- **Instrument the two candidate mechanisms in ONE run and let the log print the verdict.** The
  `poll(state)`/`wait()` split cost one hardware run and **retired the planned fix before it was
  built on**. Hardware access is the scarce resource; a measurement that can only confirm your
  hypothesis is worth less than one that can also kill it.
- **Two independent instruments beat one instrument twice.** `/proc/interrupts` reading 0 is
  consistent with "no interrupt raised" *and* "raised but never delivered". The CU's own
  `ISR=0x3` (toggle-on-write) discriminates them outright.
- **Keep a known-good comparator on the old path when you change a mechanism.** camera_capture —
  no AXIS port, ~6 µs of work — paying the same 512 ms turned "roi_crop is slow" into "any CU
  completion is slow"; after the fix, the *same* probe in the *same run* still paying 512 ms is
  what proves the fix is the fix.
- **A MULTI-START CSV COLLIDES ON FRAME INDEX, AND IT SILENTLY DISARMED THE RAILS GATE.**
  `calib_report.py`'s `parse_csv_frames` keyed by frame number, so one `track.csv` holding 15
  anchors collapsed 8434 rows to 742 and reported **4 railed frames where there were 266** — a
  66× under-report, printed next to a confident "BUDGET IS WRONG" that was right for the wrong
  reason. Fixed 2026-08-26 by keying on `(job, frame)`. Any per-frame reader of a
  `FRAME_SOURCE=vot` CSV has this bug until shown otherwise; `job` is a column for exactly this
  reason and a bare frame index is not a key.
- **AN OFFLINE R CAN BE RAISED BY DEGRADING THE FILTER — DEMONSTRATED, NOT ARGUED
  (2026-08-28).** A deliberately broken init (`--warp-mutant gsign`, G centred at `+`the crop
  offset instead of `−`) scored `vot_ar_offline` **dR +0.0525 — 6.5× the correct arm** — while
  tracking measurably worse by every direct measure: A −0.0975, mean IoU 0.1792 → 0.1683,
  frames above IoU 0.5 17.3% → 15.4%. A weak, under-confident filter reports smaller
  displacements and stays near where it started, which survives the 10-consecutive-frame rule
  while overlapping badly throughout. **Never accept an arm on R alone: require A not to fall,
  or price the fall.** dR > 0 with dA < −0.02 is this artifact until shown otherwise. (`dec2`
  at dR +0.0567 / dA −0.0086 passes; the mutant at +0.0525 / −0.0975 does not, and the two are
  otherwise the same shape.) It also did its intended job — the bench is demonstrably NOT blind
  to warp geometry, which is what makes the init null readable.
- **ACCURACY IS AVERAGED OVER TRACKED FRAMES, SO A LONGER-SURVIVING ARM IS SCORED ON HARDER
  ONES — score A on the COMMON survived prefix before calling an accuracy drop real.** The
  spatial-mask arm reads `dA = −0.0346` pooled, which trips the artifact test above; on the
  frames BOTH arms survived (identical 5563-frame set) it is **−0.0085**, so three quarters of
  the "loss" is the extra 24.7% of harder frames it reached. The same structural point is
  already made about RGB ("a tracker that survives longer is scored on harder frames"), but it
  was never turned into a control. It is one `min(progress(x), progress(y))` per sequence, and
  without it a robustness win looks like an accuracy regression.
- **`R = 0.3435` (the gsign mutant, OFFLINE) IS NOT `R = 0.3417` (the SHIPPING ARM).** They
  agree to three decimals by coincidence and mean opposite things. Every `vot_ar_offline`
  number is the toolkit's RULE on SINGLE-START runs, is biased ~0.02 high in R, and is
  comparable only between arms scored identically — so the offline baseline at R 0.2910 is
  **not** a regression from the hardware 0.3417.
- **Test an analysis tool against an OLD log before the run it was written for.**
  `calib_report.py` was written to parse a calibration run, then pointed at an existing board
  log: its frame regex was anchored at line start, but board logs are captured through
  `picocom … | ts`, so every line carries a timestamp. Anchored, it found ONE frame and reported
  "no rails" with a straight face. A parser that finds nothing looks exactly like a clean run.
- **Timestamp the console instead of adding timers.** `picocom … | ts '%H:%M:%.S' | tee log`
  needs no rebuild. That localised 7.66 s to a single interval in one run, after two wrong
  inferences from reasoning-by-elimination. Do this *before* writing any in-code profiler.
- **Benchmark a host-side change on the host.** `-fcx-limited-range` gave 1.6× on x86 and 0.48 ms
  on the board, because `filter_update` streams ~8 MB/frame — inside a desktop L3 (compute-bound)
  and outside the A72's (memory-bound). **The ORDERING did not transfer, not just the
  magnitude**, because the working set crossed a cache boundary between the machines. Same for
  the blocked `unpack_spectrum`: x86 said 4.66×, the A72 gave 1.63×.
- **Test an AIE placement question with `make graph`, not with a 25-minute package.** The
  prediction that a 64 KB ping-pong would not fit was wrong (AIE-ML cores address neighbouring
  tiles' memory) and cost 3 minutes to disprove.
- **A knob that won 7× on one port lost 4× on another with no warning.** `FFT_ROW_WS` took
  `gmio_fft_row_out` 73.22 → 9.93 ms; the identical change on `gmio_fft_col_out` went the other
  way. The "overhead- vs production-dominated" discriminator has no term for where the mapper
  puts the buffers. **Do not generalise a windowing result from one port to another without a
  hardware run.**
- **Write predictions down before the run.** The `FFT_COL_WS` sweep was 2 of 3 right and
  catastrophically wrong on the one that mattered; the memtile transpose matched its frame total
  while its attribution was completely wrong (conv2d's production reappeared in `roi_crop
  launch`, 0.067 → 5.196 ms). Checking *where* the time went, not just the total, is what turned
  that into the next item.
- **`make test_roi_crop` and the HLS report cannot see launch-path bugs.** Both are correct about
  the datapath. The 503 ms was between two `printf`s, untimed.
- **An existing constraint does not become inapplicable because the code around it changed.**
  The memtile rewrite queued all `CONV_INVOCATIONS` asyncs on `gmio_weights`; XRT allows **one
  outstanding async per port**, the identical error a depth-2 drain probe had already hit. Every
  async site in `mosse_tracker.cpp` is now audited — 11 pairs, one outstanding per port.
- **For a thread launch, disjoint state is not enough.** The right question is not "does the
  other path TOUCH anything this one touches?" but **"is everything this helper READS already
  written when it starts?"** The first `TAIL_PARALLEL` attempt passed the first question and
  silently corrupted tracking (mean IoU 0.9188 → 0.4794): `filter_update_quantize` consumes
  `g_target_shift`, which was filled ~200 lines later, so it trained on the previous frame's
  target. The instrument written to explain that bug then had the same bug — it sampled
  `g_ap_us[AP_FILTER]` before `join()`. Cost differed sharply: the first took a full
  build-flash-run to find; the second only broke a diagnostic and **failed silently rather than
  printing a plausible wrong number**. Failing loudly-or-not-at-all beats failing plausibly.

## Known issues and traps

### Measurement / build hygiene

- **`runs/.last_cfg` IS NOT AUTHORITATIVE. THE FLAGSTAMPS ARE.** For `run_0820_1418.log`,
  `.last_cfg` recorded `ITER_CNT=500` and 4-3-3; the run executed **200 frames at 4-4-4**. The
  stamps are written by the recipe that runs the compiler, so they cannot disagree with the
  binary. **Diff the stamp against `AIE_FLAGS` before an expensive run:**
  ```bash
  printf 'printvar:\n\t@echo "$(AIE_FLAGS)"\n' > /tmp/pv.mk
  make -f Makefile -f /tmp/pv.mk printvar TARGET=hw | tail -1 > /tmp/aie_now
  diff <(tr ' ' '\n' < /tmp/aie_now) <(tr ' ' '\n' < build/hw/.../aie.flagstamp)
  ```
  Quoting differs; only VALUES matter. `make -n` cannot answer this — the stamp is a `FORCE`
  prerequisite. Also grep the compiler log directly:
  ```bash
  grep -o 'CONV2D_ECHO_TEST=[0-9]' build/$TARGET/${PATCH_ROWS}x${PATCH_COLS}/ch$N_CHANNELS/aiecompiler.log
  ```
  Flag-only changes have silently reused a stale `libadf.a` and produced convincing false
  results. Order matters for `:=` in the Makefile — a stamp variable defined above its inputs
  expanded to `/aie.flagstamp` and silently disarmed a test. See [[feedback-verify-the-build-ran]].
  **`scripts/calib_build.sh` does this check for you** and refuses to declare a build good when
  the stamps disagree, so prefer it for any run worth hours. Every `.xo` now carries a stamp too
  (`crop.flagstamp`): `CONV_IN_CH` touches no source file, so a source-only prerequisite list
  would have shipped a grayscale `roi_crop.xo` inside an RGB build.
- **A stale `Map_Report.csv` in the build dir survives a failed compile** and still shows the
  previous run's healthy table. Only the compiler's `ERROR:` line is true.
- **Check `CONV2D_MODE` before every expensive run.** The default was `1` (echo) until
  2026-08-14 and nothing in the build output says so; it cost a ~28 h ch16 baseline. In echo mode
  conv2d returns at the top: no MAC, no ReLU, no B1, **no Hanning window** (so B2's 9-bin
  identity does not hold), and all 16 channels are bit-identical (so the accumulator sums 16
  coherent copies and every amplitude and PSR figure is inflated).
- **NEVER KILL A BOARD RUN MID-SEQUENCE — THE NEXT PROCESS STALLS, AND KILLING THE LEFTOVER DOES
  NOT CLEAR IT.** Interrupting a sweep leaves the free-running AIE graph and the XRT device
  context inconsistent. The next run then hangs: both threads in `clock_nanosleep`, ~33% of one
  core, `roi_crop`'s CU reading `0x20E` (`ap_done=1 ap_idle=1`, so the CU is NOT the blocker),
  and ZERO frames produced for 14 minutes on a sequence that takes 14 SECONDS. `vot_sweep.sh`'s
  own leftover-kill ran first and the fresh process stalled anyway — **a reboot is the only
  clear**. Two things hide it and neither is a fault: the board's stdout is block-buffered at
  4 KB, and `CSV_FLUSH_EVERY=200` holds the rows, so a healthy long sequence and a hung one look
  identical from the log. **Watch the TRAJECTORY COUNT.** Cost: one aborted 62-sequence arm.
- **A CROSS-IMPLEMENTATION CHECK IS THE ONLY THING THAT CATCHES A CONTRACT MISMATCH.**
  `filter_box_energy_fraction()` (board) and `box_energy_fraction()` (offline) were described as
  "written against the same definition"; the board half omitted the inverse transform, so it
  measured a spatial statistic on a frequency-domain array and read **0.0000 on every frame of a
  whole sweep**. Five unit tests were green throughout, because they fed it SPATIAL arrays —
  self-consistent with the bug and agreeing with the caller about nothing.
  `scripts/check_ebox_crosscheck.py` runs BOTH implementations on the SAME H and is
  mutation-tested (5 mutants, all caught). Same family as the `generate_scenario` trap below:
  zero-tolerance comparison proves nothing when both sides share an assumption.
- **LAUNCHING OVER SSH CHANGES THE FRAME-TIME MEASUREMENT.** `scripts/vot_sweep.sh` drives the
  board over ssh, which moves the ELF's stdout off the 115200 console. That console is itself a
  distortion — 15% of the frame at `VERBOSITY=0`, **58% on `animal`** — so ssh frame times are
  more honest AND **not comparable to any run before 2026-08-25**, `run_0821_1725` included.
  `ts` on the PC side of a TCP stream is good to about a second: it locates a stall, it is not
  the instrument `picocom … | ts` was. Take frame time from the `AP_*` slots and `track.csv`, and
  quote FPS only from a serial-console run. Incidental gain: ssh without a pty emits clean `\n`,
  where picocom's bare `\r` made `readlines()` and `grep` disagree about line numbers.
  See `docs/thesis/evidence/automation.md`.
- **`debugfs`'s `mkdir` ALLOCATES THE INODE BEFORE it fails on an existing directory**, so
  re-running an image-provisioning script leaves an unconnected inode and a filesystem `e2fsck`
  calls dirty. Test-then-create. The read-back verification passed both times — only the closing
  `e2fsck -fn` caught it, which is the argument for ending image surgery with a filesystem check
  rather than with a content check.
- **hw_emu packaging stalls on `udevadm settle` when the card reader is plugged in** — ~120 s per
  repeat, turning a 45 s package into 10 min. Not a failure. Unplug the reader.
- **hw_emu wall-clock timings are not hardware timings — but hw_emu SIMULATED PL CYCLES are.**
  The host runs on QEMU (`Runtime.hw_em_driver`), so host-side latency there is meaningless
  (one run reported 210,925,994 µs/tx). The PL is simulated at RTL, so `ap_start`→`ap_done`
  counts transfer directly. Use `make debug_sim && make probe_emu` (defaults
  `PROBE_CU=roi_crop_0 PROBE_PORT=patch_out`, 26 signals). A 64×64 `N_CHANNELS=2 ITER_CNT=1` run
  costs ~1-2 h and exercises both `recompute=1` (ch0) and `recompute=0` (ch1). The VCD is written
  incrementally, so a killed run still parses. `ap_int/ext/str_blocking_n` do not resolve in this
  build; use sub-loop `ap_done` plus the handshakes.
- **The `NOTE: hw_emu wall time is not real hardware time` line is UNCONDITIONAL** (no `TARGET`
  guard) — it is not evidence a run was emulated, and it caused genuine hardware numbers to be
  discounted. Now guarded behind `HW_EMU_BUILD`.
- **`SIM_WALL_TIMEOUT` scales with patch area** (`SIM_PATCH_SCALE`); a timeout looks exactly like
  a deadlock. Check for `Error 124` before concluding "deadlock". Check the right process:
  `ps -C aiesimulator` shows only bash wrappers at 0% CPU; the simulator is `aie2simmsm`
  (~190% CPU, ~2.5 GB RSS). A quiet log is normal — 10+ minutes between prints.
- **The offline model's ratios and orderings are sound; its absolute magnitudes are
  patch-specific.** `phase1_sweep.py` runs on the s6 patch; hw_emu injects a synthetic target
  through real `roi_crop`, which after log/z-score is far hotter — it predicted accum 162 where
  hardware gave 5264. With `mean_prev` seeded the model agrees to 3-11%.

### Metrics that cannot fail a broken tracker

- **PSR is a weak pass criterion, and in a specific direction.** A tracker 179 px off target,
  confidently locked to background, reported **PSR 33**. One run: `19 evaluated, 19 accepted, 0
  gated`, PSR min 15.95 / mean 26.60 — while IoU fell to 0.0656. The response really was sharply
  peaked, just in the wrong place. **IoU is the only metric in the harness that can fail a
  confidently-wrong tracker.** Read `track.csv`, not the console.
- **`[diag] F_ch` IS CHANNEL 0 ONLY, not a bank maximum.** It is printed under `if (ch == 0)`
  (in `mosse_tracker.cpp`, under the per-channel `if (ch == 0)`), while `accum` and `response`
  are bank-wide. So "F_ch looks
  comfortable" says nothing about the other 15 channels, and a hot channel elsewhere is
  invisible. It matters most at `CONV_IN_CH=3`: **ch0 is one of the four colour-opponent
  channels** (0/2/9/10), so its amplitude is the one number in the console that discriminates a
  real colour path from a colour-free one. Predicted input-referred gain for ch0, from the
  exported weights: gray 8.72, RGB `FRAME_RGB_MODE=1` 10.87 (1.25×), RGB `FRAME_RGB_MODE=0`
  0.80 (**0.09×, near-dead**). If ch0's `F_ch` does not collapse ~10× between those two RGB
  arms, colour is not reaching conv2d.
- **`err=0 px` is a weak pass criterion too.** It cannot see mainlobe width, drift, a DC
  pedestal, or a gated frame (where a mismatch is a *pass*).
- **A centred test impulse cannot validate localisation** — `peak_detect_sw`'s old scan returned
  index 0 on an all-zero response, i.e. the right answer produced without reading the data. The
  impulse is injected at `pos + (IMPULSE_DR, IMPULSE_DC)` = (10,−7), asymmetric and
  opposite-signed so a transpose and a sign flip are both caught.
  See [[impulse-test-pattern-is-degenerate]].
- **`[MISMATCH vs injected offset]` is meaningless under `TRAJECTORY=1`** — the criterion derives
  `exp_dr` from `IMPULSE_DR`, so it fires on healthy frames.
- **Two different statistics are both called PSR.** The aiesim `snr_ratio_pct` is
  `|peak| / max|sidelobe|`; Bolme's is `(g_max − µ_sl) / σ_sl`. Same 11×11 circular exclusion,
  different statistic — they differ by several times and neither's thresholds transfer.
  `report_psr()` prints both, labelled. Bolme's is the meaningful one for occlusion.
- **PSR must exclude the mainlobe or it asserts nothing** — a neighbour of a σ=2 peak sits at
  0.88 of it. Use a *circular* distance, since the map wraps.
- **s7's PSR threshold is geometry- and budget-dependent.** 15× was calibrated at 64×64 / 3-0-6 /
  ch1; at 128×128 / 4-3-3 / ch1 the measured ratio is 11.8. A FAIL from that threshold alone is
  not evidence of a defect until it is re-derived. **ch1 is the documented worst case** —
  channels add coherently, their quantization noise does not.
- **An ordering inferred from one failing run is a hypothesis, not a finding.** This cost the
  most time on this project. Only a run where one of the two candidates is known-good can order
  them.

### Correctness traps

- **THE FILTER MUST BE TRAINED AGAINST A G CENTRED AT THE MEASURED DISPLACEMENT, NOT AT (0,0).**
  Fixed 2026-08-20. `filter_update()` was passed `g_target` — the Gaussian centred at (0,0) —
  while `g_F_all` is the patch cropped at the **pre-update** position, where the target sits at
  `(dr,dc)`. Every accepted frame taught "a patch with the target at (dr,dc) peaks at (0,0)",
  compounding at `eta` until the zero-shift peak won. Mean IoU 0.1708, 181 of 199 frames lost.
  **The sign, derived rather than guessed.** Let `Q_t` be the patch cropped exactly ON the
  object; the patch actually held is `P_t = Q_t` shifted by `+d_t`. Correlation is shift
  equivariant, so an on-target patch must peak at 0 ⇒ **G is centred at `+d_t`, the SAME sign as
  the detected peak.** That derivation **predicts analytically** the observed law
  `resp00_over_peak ≈ 1-(1-eta)^k` — i.e. it is governed by the LEARNING RATE, not the scene.
  **Fix**: a per-frame `g_target_shift` from `gaussian_target_spectrum(..., psr_abs.dr,
  psr_abs.dc)`. By the shift theorem this IS "re-crop at the new position", exactly and for free.
  `filter_init()` on frame 0 keeps the centred `g_target` — that crop really is centred. The
  exact alternative is a real re-crop at 16 more roi_crop launches ≈ 77 ms/frame.
  **A SINGLE UPDATE CANNOT SEE THIS DEFECT** — the one-shot golden check passed throughout —
  which is why both regression tests are closed loops (`scripts/mosse_loop_sim.py` and
  `run_training_target_tests()` in `make test_host`).
  **Assert the shape, not an absolute level.** In the 32×32 test both arms start at 0.39 — that
  scene's own zero-shift autocorrelation, not a defect. What is geometry-independent is that the
  defect makes the ratio GROW at the learning rate while the fix leaves it flat.
- **Background lock was the WRONG explanation for the above** — kept because the measurements
  are real and the mechanism exists. `fill_background()` is cached and only the dirty rect is
  restored, so outside the target the synthetic frame repeats to the LSB, and a DCF fed a
  perfectly repeating background correlates with it at exactly zero shift. Measured 2026-08-17:
  the static peak was worth 69-86% of the true one and won 21 of 48 frames, each win costing a
  permanent ~9.4 px offset (centre error 1.35 → 9.56 → 87 → 292 px) **while PSR read 24-35
  throughout**. **`FRAME_NOISE` is not the fix** (independent additive noise cannot decorrelate
  a static pattern). **`BG_PAN` is the right instrument** and measurably works — but changed the
  tracker not at all, which is what refuted this explanation. **Sweep its magnitude against the
  texture's wavelengths, not in pixels** (`scripts/bg_pan_sweep.py`): corr@0shift falls +0.60 →
  **+0.09** at 31,47 px/frame; the obvious guess of 3-5 px/frame is worthless, the texture's
  shortest wavelength being 180 rows. `fill_background()` rounds frequencies to WHOLE CYCLES per
  frame so the pan wraps seamlessly (else an **8.10 LSB** row-wrap discontinuity, and an
  artificial edge is exactly what a DCF locks onto); 31 and 47 are coprime to 1080 and 1920.
  **This fixes the TEST BENCH, not the tracker** — do not tune the tracker against it.
  **Discriminating background lock from a DC pedestal** (similar look, opposite fixes): a
  pedestal lifts every bin uniformly; background lock is a *localised blob* at the origin with
  sidelobe mean ≈ 0.
- **THE FRAME BUFFER WAS NEVER SEEDED WITH THE BACKGROUND. Fixed 2026-08-18** — one 2 MB
  `memcpy` at startup, which **must** come after `rc_control_cu_probe()` (which zero-fills
  `frame_bo` by design). **The interesting part is the correction.** The measured "88.53% of the
  ROI never written" came from an *offline replay*, which assumed the unwritten region held
  garbage that saturated Stage A's int8 rail and inflated σ 4.3×. On hardware that region is
  **zeros**. Hardware before/after: `F_ch` frames 0 and 1 are **unchanged** — including frame 0,
  the one the filter trains from — and frames 2+ rise 2.55-3.10×. The fix is real; the predicted
  9.2× response gain never existed, and the shift-budget change it justified was wrong.
  **The lesson: an offline replay inherits every assumption the host makes about memory it did
  not write.** Verify the premise on hardware before letting a model move a calibrated constant.
- **`mean_prev` seeding** (`mosse_tracker.cpp`, before the first `weights_bo.sync`): seed
  `mean_prev = bias_acc >> out_shift` for every channel at startup, since Stage A delivers a
  zero-mean patch. Without it Stage B1 is inert on frame 0 — the one frame the filter trains
  from — and the ch16 response rails flat. Fixing it took `F_ch` from 32768 (railed, 11 bins) to
  53, accum 5264 → 70, peak/sidelobe 3.59 → 23.04.
- **Conjugation: the stored filter is H, not Bolme's H\*.** `cmul_accum` conjugates itself, so
  the host stores `H = conj(G) ⊙ F / (B + ε)`. Storing Bolme's expression verbatim gives a
  phase-noise response peaking at an arbitrary bin — **invisible whenever the target is
  centred**, since a centred real Gaussian has `conj(G) = G`. That is why s7's target is
  off-centre.
- **H's quantization ceiling is not `2^H_SHIFT`.** The ceiling sets H's *resolution* (always use
  all 15 bits); `H_SHIFT` sets the *product scale*. Coupling them trades one bit of filter
  precision per bit of accumulator gain, i.e. does nothing. **max|H| sits where |F| is
  smallest**, because that is where the regularized inverse peaks — at `H_SHIFT=15` the
  accumulator reached 15 of 32767 and PSR collapsed to 5.2 while still localising exactly. The
  contract is encoded in four files: `mosse_filter.cpp`, `gen_filter_golden.py`,
  `test_mosse_filter.cpp`, s7 in `gen_aiesim_vectors.py`. Normalization is by complex
  *magnitude*, so `rails=1` on one frame and `rails=0` on the next with `max|.|=32767` both times
  is not a bug (a bin at 45° puts 32767/√2 in each part).
- **The correlation response is SIGNED once Stage B1 is active** (s6 peaks at `{-417,0}`). The
  scan is `|real|`; both peak definitions are computed every frame and a disagreement reported.
- **B2's correction is not bit-exact.** The linearity argument is exact in real arithmetic, but
  conv2d's window multiply applies two `>>15` truncations. Residual ~1e-3 (vs 2.5e-2…9.9
  without). Fine for argmax. With `mean_prev` seeded B2 is currently a no-op
  (`max|removed| = 0`), which is the desired state.
- **DSPLib's cint16 FFT loss is additive, not a gain factor** — each pass subtracts ~21 from a
  summed DC bin, independent of amplitude. So `row_dc = PATCH_COLS*c − 21`,
  `accum0 = PATCH_ROWS*row_dc − 21`. Any "expected = N" calculation is wrong. An impulse loses
  ~3. (A "2/3 gain" fits one data point by coincidence — don't fit a scaling law to one point.)
- **A per-element `std::abs`/`std::norm` on a complex is a TRANSCENDENTAL CALL.** `std::abs()` on
  a complex is `hypot()`, and `filter_quantize_q15`'s max scan ran it 262144×/frame. No compiler
  flag short of `-ffast-math` touched it. Fixed exactly: scan on `re²+im²` (monotone in `abs()`,
  so it selects the same element), then take **one** square root. 8.71 → 1.88 ms, 4.6×. Grep for
  it before profiling anything else in this file.
- **`std::complex<float>` blocks vectorisation** — C99 Annex G forces the libgcc `__mulsc3`
  helper. `-fcx-limited-range` removes it (NEON fp ops 10 → 17) and only drops Inf/NaN range
  handling in complex mul/div, unlike `-ffast-math` which makes every float op in the file
  unsafe. Worth 1.6× on x86 and 0.48 ms on the board — see the benchmarking rule above.
- **Never compute on a BO mapping.** The mappings are write-combining and the asymmetry is
  extreme: memcpy heap→heap 7359 MB/s, **BO→heap read 696**, heap→BO write 3470; summing int16
  on a BO mapping is 5.8× the heap rate, and **fp64 on the same buffer is another 4.02×**. Bulk
  `memcpy` out, compute on the heap copy, `memcpy` back only if needed. This was worth 33 ms/frame
  across five sites. Two independent sub-fixes: **the energy accumulator should be int64, not
  fp64** (sum of 16384 int16 squares peaks at 1.8e13 — exact in int64, and 4× faster on the same
  memory; the old `double` was already carrying exact integers, so the switch is bit-exact), and
  `filter_update` is **pure heap already**, so the BO fix does nothing for it.
- **Access PATTERN, not traffic volume, was the lever on `filter_update`.** Flipping the B loop
  to channel-major moves **exactly the same bytes** — it changes only the stream count, from 16
  strided readers 128 KB apart to one sequential reader. On an A72 whose prefetcher tracks a
  handful of streams, that was ~7 ms/frame. A traffic-volume model would never have found it,
  and did not.
- **A SELF-CONSISTENT TEST CAN PASS ON CORRUPTED DATA. Found 2026-08-23, and it is
  the best argument in this file for single-sourcing an offset.**
  `generate_scenario` patched `mean_prev` into the scenario's `weights_ch0.bin` at
  a hardcoded byte 18 — correct at 9 taps, and **inside the B plane at 27**. At
  `CONV_IN_CH=3` it therefore overwrote taps [18:22] with an int32 and left the
  real `mean_prev` field at 0, so Stage B1 was silently disabled AND three of the
  27 taps were garbage. **The bit-exactness check passed anyway**, 16384/16384,
  because the kernel and the model both read the same corrupted file. The only
  symptom was `mean_prev=0` printed where the generator had computed 4842 — a
  number nothing was asserting on. Zero-tolerance comparison is not enough when
  both sides share an input; the INPUT has to be pinned too.
- **THE CONV WEIGHT LAYOUT IS NOW SINGLE-SOURCED. The rest of this trap still stands.**
  Fixed 2026-08-23. The layout lives in `design/aie_src/conv_weight_layout.h`, derived from
  `CONV_IN_CH`, and is mirrored formula-for-formula in `scripts/conv_weight_layout.py`. It used
  to be four hardcoded copies of `[0:9]/[9]/[10:14]/[14:18]/[18:22]`.
  **RGB is what forced it: 27 taps overrun ALL FOUR grayscale fields**, so a `CONV_IN_CH=3` file
  read by a `CONV_IN_CH=1` reader takes `out_shift` out of the G plane and `bias_acc` out of G/B
  taps — no crash, sixteen plausible channels, a meaningless tracker. **Byte 63 of every channel
  buffer carries the layout tag** (= `CONV_IN_CH`; 0 in pre-tag files, read as grayscale), and
  three independent guards were claimed on a mismatch — and **one of the three is inert**.
  Corrected 2026-08-25: the `#error` in the generated `layer0.h` CANNOT fire, because no
  translation unit in either the AIE or the host build includes that header
  (`grep -rn layer0.h design/` returns nothing but the file itself). What actually fires is
  the **runtime tag check in the host**, before any weight byte is read, and `SystemExit` in
  `check_collapse.py` / `gen_aiesim_vectors.py` / `phase1_sweep.py`. The runtime check caught
  a real one on 2026-08-25: an RGB `layer0_weights.bin` (tag 3) on the SD card under a
  `CONV_IN_CH=1` ELF — `FATAL: layer0_weights.bin ch0 has layout tag 3`. **The dead guard went
  unnoticed precisely because a live guard covers the same case**, which is the argument for
  testing each guard separately rather than testing that the case is caught.
  **The weights are a RUNTIME data file**: nothing includes `layer0.h`, and `hanning_128.h`
  does not depend on `CONV_IN_CH`, so switching arms needs `make weights CONV_IN_CH=<n>` and a
  1 KB file copy onto the card — no re-synthesis, no re-package, no re-flash.
  Gray resolves to the historical offsets; **proven, not assumed** — preprocessing
  `conv2d_kernel.cpp` at `CONV_IN_CH=1` from `HEAD` and from the working tree and folding the 18
  constant index expressions gives identical text.
- **Preprocessing constants are coupled across engines with no compile-time check.**
  `hanning_*.h` must stay periodic;
  `ROI_NORM_Q` in `roi_crop.h` sets the int8 scale `out_shift` was derived against;
  `FFT_ROW_WS`/`FFT_COL_WS` must reach **both** `AIE_FLAGS` and `GCC_FLAGS` (they didn't, and
  `mosse_tracker.cpp` silently defaults them to 2 — graph and host would have disagreed about
  every DMA chunk count and deadlocked). **General rule: any constant both the graph and the host
  derive from must be passed to both toolchains from one Makefile variable. A `#ifndef` default
  in the host is not a safety net — it is what makes the mismatch silent.**
- **A `static_assert` ties each `AP_*`/`DMA_*` enum to its name table.** Inserting `DMA_ACCUM_IN`
  mid-enum without updating the table would silently RENAME every port after it.
- **The `bias_acc` correction — APPLIED 2026-08-23, and it re-opens the shift budget.**
  `export_weights.py` derived `bias_acc` for an input scale of 127 ≙ 1.0 while `roi_crop` emits
  a z-score at `ROI_NORM_Q = 32`, so `bias_acc` was ~4× oversized — and since `out_shift` comes
  from `|bias_acc| + ACC_MAX_THEORY`, an oversized bias shifts the SIGNAL down to make room for a
  DC pedestal Stage B1 subtracts away anyway. Pure loss. `make weights` now defaults to
  `BIAS_SCALE=roi`; `BIAS_SCALE=127` reverts bit-for-bit apart from the layout tag.
  Measured on the grayscale export (`check_collapse.py` Q3, input-independent):

  | | `127` | `roi` |
  |---|---|---|
  | structurally dead channels | ch3, ch15 | **none** |
  | ReLU provably a no-op on | 11 of 16 | 7 of 16 |
  | signal resolution | 7.6–13.0 of 15 bits | **9.6–13.4** (spread 5.5 → 3.8) |

  Rank and participation ratio do NOT move (9 / 4.94) — the collapse is a property of the
  weights, not the bias, so this fix and RGB are independent wins.
  **It only pays with `CONV_RELU=0`** — see the ReLU entry under Settled; corrected+ReLU is the
  worst of the three arms measured. It also spent the accumulator margin, which is what forced
  `H_SHIFT` 10 → 11 on 2026-08-24; that is now closed on hardware for both arms.
  Still unfixed, and unrelated: the semantic mismatch of weights quantized against
  ImageNet-normalized linear luminance being fed a z-score of the log.
- **`gen_aiesim_vectors.py`'s float Stage A differs from the kernel on 40.9% of samples**, by up
  to 2 LSB (rms 0.65 on a signal of std 32).
- **Test vectors can sit below the fixed-point floor** — s1's amplitude-1 impulse quantized to
  almost nothing (20/4096 bins non-zero). Now `GEN_IMPULSE_AMP=100`.
  See [[aiesim-quantization-floor]].
- **The legacy test scheme plants the target at the tracker's own estimate plus a constant**, so
  ground truth follows the tracker and `err=0 px` is nearly self-fulfilling. `TRAJECTORY=1` makes
  drift real and measurable.

### AIE / kernel traps

- **Vectorization gotchas, all found the hard way.** conv2d: use `aie::downshift`
  (arithmetic/floor, matching C++ `>>`) never `srs` (rounds to nearest); `aie::unpack` not
  `cast_to` to widen the Hann table. cmul: `from_vector(acc, S)` seeds the accumulator exactly,
  so the accumulator must be folded in BEFORE the shift — converting the product to cint16 first
  clamps twice. `rounding_mode::positive_inf` and `saturation_mode::saturate` are load-bearing.
  **`alignas(32)` on tile-local buffers is required: x86sim does not enforce alignment**, so
  omitting it passes every bit-exactness check and then misbehaves on hardware.
- **conv2d weights are consumed per FIRING, not per patch.** `weights` is an `input_buffer` and
  ADF acquires it before every invocation, so the driver must supply
  `PATCH_ELEMS/CONV_OUT_CHUNK` buffers and must start the patch flowing *first* or it deadlocks.
  This caused every historical "PLIO hang". Proper fix, not done: make weights an RTP.
  Note the host currently sends the **same 64 bytes 16 times per channel** (the `async` offset
  has no `k` dependence). See [[hw_emu_conv2d_fft_hang]].
- **`aie2gm_nb()` transfers one kernel invocation per call, not the full N bytes.** One
  `async`/`wait` pair per invocation on every output GMIO, chunked by the producer's output
  window. **The drain loops must be ordered, not just chunked**: `gmio_accum_out` /
  `gmio_response` must be drained **before** waiting on the corresponding input GMIO. Symptoms of
  getting it wrong: stall after 6 of 64 weight buffers, DMA status frozen at `0x1a080010`,
  `roi_crop_0` at `ap_start=1, ap_done=0` forever. See [[feedback_aiesim_gmio]].
- **XRT GMIO allows ONE outstanding async per port.** There is no pipeline to deepen. A depth-2
  probe aborted in 11 seconds of board time with
  `Asynchronous operation is already initiated. Multiple 'async' calls are not supported`.

### Infrastructure

- **THE CU COMPLETION INTERRUPT IS NEVER DELIVERED ON THIS PLATFORM. Every KDS launch costs
  ~503 ms.** A platform defect, not anything in this design. The CU side is healthy and armed:
  ```
  devmem 0xa4010004  GIER = 0x1     global interrupt enable ON
  devmem 0xa4010008  IER  = 0x3     ap_done + ap_ready enabled
  devmem 0xa401000c  ISR  = 0x3     BOTH LATCHED, NEVER SERVICED
  ```
  The HLS ISR is toggle-on-write and only a handler clears it, so `ISR=0x3` standing after 100+
  launches proves **no handler has ever run** — independent of `/proc/interrupts`, which reads
  `0 0` on both `zocl_irq_intc` IRQs (51, 52). Clearing the ISR by hand puts it straight back to
  `0x3` with counts still 0. Consistent with the boot-time `zocl-drm: error -ENXIO: IRQ index 32
  not found` and with IRQ 51 registered `Edge` while 52 is `Level`.
  **Everything reachable from userspace was tried and none of it moved the number**:
  `poll_threshold=1000000`, a hand-cleared ISR, `Runtime.ert_polling` — all 503.40 ms. **Fix is
  `ROI_CROP_USER_MANAGED=1`**; the interrupt wiring itself needs the base platform's device tree.
  Board-side checks in the order that settles things fastest: `/proc/interrupts | grep zocl`, the
  three `devmem` reads, then `cat /sys/bus/platform/devices/CU.N.auto/cu_stat` — its `sleep cnt`
  equalled the launch count exactly. `cu_info` in the same directory prints the base address,
  control protocol and full argument offset map, which is where `CropIp`'s register map came from.
- **`xrt::ip` details that mattered** for `CropIp`: 2025.2 imposes **no control-protocol
  restriction** (the `ap_ctrl_chain` worry was unfounded); it takes an **exclusive** CU context,
  so the `xrt::kernel` for roi_crop must not be constructed at the same time
  (`Runtime.rw_shared=true` relaxes this); `frame_buf` is an `m_axi` pointer and needs
  `bo.address()`, which `set_arg(0, bo)` used to supply implicitly. After the fix `drain → poll`
  is 0.019-0.027 ms and the spin exits on its first read. **If the drain ever shrinks the spin
  will start spinning for real** — the 60 s bound is what keeps that safe. (It did: the memtile
  transpose deleted the drain and exposed 5.196 ms/frame, exactly as predicted here.)
- **An `xrt.ini` key set as a shell variable is not a test of that key.** `ert_polling=true` at
  the prompt sets an environment variable; XRT reads `Runtime.ert_polling` from `xrt.ini` in the
  process's working directory. One "null result" in the 503 ms hunt was this and nothing else.
  The same file's header warns that an unrecognised *key* is silently ignored — the same failure
  mode one level up.
- **`v++ --package` corrupts the 2025.2 rootfs** (ext4 feature mismatch) — every hw_emu run
  panicked at boot. `make rootfs` builds a downgraded copy. See [[vpp_package_corrupts_rootfs]].
- **XRT AXIS ports consume a positional argument slot**, so scalars shift by one. Set args by
  explicit index. This was the real cause of "hw_emu PL→AIE PLIO delivers nothing".
  See [[plio-was-never-broken-xrt-arg-index]].
- **The host does not exit after the last frame.** `gr.end(0)` blocks forever on a free-running
  graph. Cosmetic, but `run_script.sh` never reports RC and emulation must be killed by hand.
- **Probing PL↔AIE signals in hw_emu takes three non-obvious steps**, each of which silently
  defeats capture: `ai_engine_0.S00_AXIS` is a SystemC/TLM socket with no TVALID/TREADY wires
  (use the `VitisRegion/out_r_*` boundary port); v++ elaborates with `--debug off` so `log_vcd`
  aborts before Linux boots (`make smoke_debug_sim` re-elaborates); and `elaborate.sh` uses
  relative include paths that fail in the copied `package/` tree **while still printing "Built
  simulation snapshot"**. See [[hw-emu-signal-probing]].
- **The pre-computed FFT bypass in `make aiesim` skips the PatchIn→conv2d→row-FFT path.** Use
  `make aiesim_plio`. Buffer dumps: [[hw-emu-buffer-dumps]].

## Validated / done (facts worth not re-deriving)

- **`roi_crop` cycle counts, measured** (hw_emu VCD, 64×64, `recompute=1`): `ap_start` →
  `ap_done` = **245.5 µs = 76,725 cycles** at 312.5 MHz, and `ap_done` asserts at the same
  instant as `TLAST`. PASS1 44,600 cyc, NORM 4,100, PASS2 27,900. **Two source comments in
  `roi_crop.cpp` are wrong**: PASS2 achieves **27.2 cycles/beat, not II=1** (real AIE
  backpressure), and PASS1 achieves **10.9 cycles/output-px, not II=4** (m_axi latency on the
  four scattered bilinear taps). 128×128 recompute is 4× the pixels ≈ 1 ms, exactly the residual
  `ROI_CROP_PIPELINE` leaves on channel 0.
- **`roi_crop` bit-exact, 25 cases, zero tolerance** (`make test_roi_crop`, golden from
  `scripts/roi_crop_ref.py`): 17 grayscale + 8 RGB, run as two builds because `ROI_IN_CH` is
  compile-time. 16 cases execute the bilinear interpolator, which **has still never run on
  hardware** — every build to date sets `roi_h == patch_rows`, collapsing the datapath to a copy.
  Real ground-truth boxes (a VOT run, say) would activate it for the first time.
  **Bit-exactness says nothing about timing**: this suite passed throughout the period when the
  launch path cost 98% of the frame.
- **Bounding-box state, ROI padding, σ anchoring.** State is a `TargetBox`;
  `roi = box × TARGET_PADDING`. `patch_dr_to_frame` / `frame_dr_to_patch` are unit-tested (33
  assertions incl. an anisotropic box), because while `roi_h == patch_rows` the two are
  accidentally the same number and padding breaks both by the resample ratio — a tracker that
  localises confidently and *drifts*, invisible to `err=0 px`.
- **`nature` IS NOT A TRACKER DEFECT — ITS PIXELS DO NOT MOVE. Measured 2026-08-25, offline.**
  Three diagnoses (sub-bin lag, origin lock, background lock) all assumed the target moves and
  the tracker fails to follow. `scripts/vot_motion_check.py` tests the premise: **on 80% of
  `nature`'s frames NOT MOVING correlates better** (NCC 0.940 vs 0.816). The target deforms in
  place — aspect 0.58 → 1.65, appearance decorrelating to 0.072 by frame 50 — and the box centre
  moves because it is a min–max over a changing shape. Corroborated twice: the response is
  healthy and says zero translation (peak at (0,0), `resp00/peak` 1.0000, PSR 33), and at
  `padding = 1.0`, with no background in the ROI at all, it still reports (0,0) on 98% of frames
  — so not background lock either. **`nature` is 46% of the frames in the 8-sequence evidence
  set**, so read per-sequence tables, never a frame-weighted aggregate, and never tune against
  it. `docs/thesis/evidence/frozen_detector.md`.
- **`tiger`: eta / sigma / eps_rel SWEPT, and the freeze is a SYMPTOM, not the objective.**
  `SIGMA=1` and `EPS_REL=0.1` each unfreeze the detector completely (froze-while-needed 65.8% →
  0.5% / 0.0%) and tracking does not improve — IoU stays ~0.21, centre error gets worse. **Do
  not optimise the freeze rate.** Three mechanisms tested and killed: ATTENUATION (iterated
  re-cropping helps `car1`, cerr 5.6 → 4.7 px, and makes `tiger` worse); the ONLINE UPDATE (with
  the crop at groundtruth every frame the detector is still wrong by a median **17 bins**, and
  identically so at `eta = 0`, so the filter is wrong before learning touches it — `car1`: 2
  bins); and the filter INVENTING the offset (a plain NCC template search with no filter and no
  features puts the best match **(−6.7, −8.9) px** from the annotation centre). **So `tiger` is
  `nature`'s disease in a milder form**: the box centre is a min–max over a deforming object and
  drifts against the appearance, ~11 px ≈ 8 bins, which the filter amplifies ~1.7×. Trackable
  but permanently penalised. `docs/thesis/evidence/tiger.md`.
- **`MOSSE_ETA = 0.05` SHIPPED — hardware R 0.3065 → 0.3283 (+7.1%), EAO +8.5%, 13.9% more
  frames. But two of three mechanism falsifiers FIRED.** The drift model predicted the gain and
  predicted the wrong reason: pre-loss ACCEPT share was predicted to stay ~80% and fell to
  73.7%, and first losses land marginally EARLIER, not later. What improves is RECOVERY. Per
  sequence it is a coin flip (R better on 28 / worse on 24 / tied 10); the pooled figure is
  frame-weighted. **Do not build on the explanation** — and note the gain forced the
  `PSR_GATE_MIN` re-tune, because a slower filter drops median PSR ~26% into a gate calibrated
  for the faster one. `docs/thesis/evidence/eta05.md`.
- **PHASE CORRELATION IS THE WRONG INSTRUMENT FOR "did the target move".** It returns the
  DOMINANT motion in the window, so static background filling the box makes it read zero: it
  reported **0.00 px on `car1`**, a car crossing the frame at 20 px/frame. Ask about the
  target's own pixels at two named hypotheses instead (`vot_motion_check.py`).
- **`rgb_vs_gray_loop.py --sequence <name>` reproduces the board's tracking failures at 3.5 s
  per 100 frames** — `nature` 38.4% frozen against hardware's 44.3%, `tiger` 65.8% against
  62.4%. Resolves stb2022 under `$VOT_ROOT` first, then `test-sequences/`. **Its `load_gt` used
  to carry a polygon-only copy of the groundtruth rule**, correct for `test-sequences/` and
  silently wrong for every stb2022 rectangle; it now single-sources `vot_prepare.reduce_box`.
- **Padding 2.0 is CLOSED, three ways.** `car1` closed-loop mean IoU 0.857 / 0.780 / 0.174 at
  2.0 / 1.5 / 1.2; all 62 offline, every value below 2.0 is worse (R 0.2251 at 1.5 vs 0.2910);
  and on hardware 3.0 LOST EAO. The original 1.5-vs-2.0 holdout was on a static scene where
  background lock costs nothing — that objection was valid and the answer came out the same.
- **SUB-BIN INTERPOLATION IS A SMALL ACCURACY WIN, NOT A FIX — and the compounding-lag argument
  for it is wrong.** The peak detector is a pure integer argmax, and on `nature` one bin is
  1.61 × 2.78 frame px against 2.06 px/frame of true motion, so 86% of frames really do report
  (0,0). What does NOT follow is that the error compounds: **the detector measures the offset
  that exists now, not the increment**, so lag accumulates only until it crosses half a bin.
  `mosse_loop_sim.py --subbin` sweeps ratio 1-3 × 0.1-1.5 bins/frame: worst late/early error
  ratio **1.00**, error bounded at ~half a bin — and the same bench's `centred` arm DOES
  compound, so the flat result is a finding, not an insensitive instrument. Parabolic refinement
  cuts the bounded error at large resample ratios (2.68 → 1.25 px at ratio 3), worth well under
  1% of IoU on a 100 px box. **The arithmetic that sold the wrong story conflated mean SPEED
  with mean DISPLACEMENT.** `docs/thesis/evidence/subbin_lag.md`.
- **The HOLD on a gated frame is NOT unconditionally correct — measured 2026-08-25.** From
  stb2022 groundtruth alone (`scripts/vot_hold_budget.py`, no tracker, no board), the **hold
  budget** — frames before the target's centre leaves the frozen `box × padding` window, i.e.
  before recovery is impossible for ANY tracker — has a median of **6 frames**, is **≤ 4 on 30
  of 62 sequences**, and is **0 on four**. `car1`'s budget is 4 and its longest hold on hardware
  was **53**. The candidate fix is `HOLD_COAST` (`coast_observe()`/`coast_step()`), OFF by
  default because the two metrics disagree about it (see the AR entry below); over 8 sequences
  it is +0.0296 frame-weighted mean IoU. **THE OFFLINE BUDGET MODEL PREDICTED THE OPPOSITE FOR
  `car1`, BECAUSE IT WAS OPEN LOOP**: it treated the observed 29-frame gated run as fixed, when
  coasting turns gated frames into accepted ones and every accept restarts the coast. Read the
  budget as a bound on ONE uninterrupted hold, never as a prediction of outcome. The coast
  cannot rescue a tracker that never acquires — `ball3` gates on 69% of frames and coasted zero
  times. `docs/thesis/evidence/hold_policy.md`, `evidence_arm_ab.md`.
- **MEAN IoU AND THE TOOLKIT'S AR CAN ORDER TWO ARMS OPPOSITELY, ON THE SAME RUNS —
  2026-08-25.** The `HOLD_COAST` A/B was decided on frame-weighted mean IoU (+0.0296, arm B
  wins); ingesting the identical 54 trajectory pairs into a VOT workspace
  (`scripts/vot_ingest.py`) reversed it — accuracy 0.638 → 0.616, robustness 0.309 → 0.288,
  EAO 0.208 → 0.194, arm A wins. **The cause is that `vot` fails a run on 10 CONSECUTIVE frames
  at overlap ≤ 0.1 and discards everything after**, so a short excursion outweighs hundreds of
  good frames: `car1` anchor 741 coasts 13 frames the wrong way through a turn (freezing drops
  out for 7, under the grace), then reacquires and tracks ~470 more frames at overlap 0.82 that
  the rule throws away. Failure COUNTS barely moved (48 of 54 runs vs 49) — only their timing
  did. **Quote which metric produced a verdict, and prefer AR for anything that will be
  reported**; mean IoU cannot see the timing of a loss, and that is what the challenge scores.
  See `docs/thesis/evidence/evidence_ar.md`.
- **PSR gating (Bolme §3.5)** — four veto reasons reported separately: `ZERO_RESPONSE`,
  `FLAT_SIDELOBE` (sdev 0 — PSR undefined, not infinite), `NEGATIVE_PEAK`, `LOW_PSR`. Only
  `LOW_PSR` is disabled by `PSR_GATE_MIN=0`. On a gated frame the host **holds the position**
  (moving to a noise peak walks the ROI off target permanently) and skips `publish_filter` —
  required, not an optimization, because `filter_quantize_q15` reads `g_energy`, which the
  per-channel loop has already overwritten with the occluder's energies. Proven on hardware:
  PSR 3.90 → HOLD → reacquire with `err=0 px`, frozen filter byte-identical to the ungated run's.
  **`HOLD_COAST=1` does NOT change the short occlusion regression**, and the reason is structural rather than lucky: `ITER_CNT=3 OCCLUDE_MASK=0x2`
  occludes frame 1, and frame 0 takes the `if (!evaluate)` branch without ever reaching the
  accept path, so no velocity has been measured and `coast_step()` refuses. **Longer occlusion
  runs DO change** — at `OCCLUDE_START=30` the frames before the occlusion set a velocity and the
  window coasts through it. Compare any such run against the shipping default (`0`) rather than
  against the 08-25 arm-B runs, and use the run-state digest to tell "unchanged" from "nearly
  unchanged".
- **`scale_gate()`** — three vetoes reported separately. `AT_SEARCH_RAIL` is checked *before*
  `LOW_CONF` because it is the more specific finding when both fire. **The update skip is the
  load-bearing half**, not the box hold. Guarded so a degenerate filter cannot veto everything
  (`n_scales > 2`) and so "the gate never ran" is distinguished from "the gate said no". 21
  assertions in `make test_host`, driven by values hardware actually produced.
- **DSST 1-D scale filter** = the existing filter at `rows = 1`; `gaussian_target_spectrum(G,1,S,
  σ,0,0)` degenerates cleanly. Defaults from §6.1 except the step: S=33, η=0.025, σ_s=S/16,
  template capped at 512 px. A single application under-corrects (+3 where +5 is exact) because
  the response is a discrete peak smoothed by a σ=S/16 target — the property to assert is that
  **repeated** application converges monotonically, which it does to 0.5%.
- **A BYTE-IDENTICAL TRAJECTORY IS NOT A BIT-IDENTICAL RUN — found 2026-08-25 BY A NEGATIVE
  CONTROL.** The multi-start determinism test (job A, B, A in one process; A's two trajectories
  must match) passed with `RESET_MUTANT=1` active, i.e. with `run_reset()`'s `mean_prev` re-seed
  deliberately skipped. The leak was real and visible from frame 1 — `accum` 1963 vs 1961,
  `response` 1556 vs 1553, B2 removing 667 vs 671, differing on ~15% of frames — but a box comes
  from an INTEGER peak bin, so a 0.1% difference in the response never reaches the trajectory.
  Same failure mode as the parallel-for FMA experiment: "nearly identical" is the outcome that
  makes a bit-identical criterion useless. **Fix: a per-frame FNV-1a digest over the full response
  buffer plus `box`/`psr` in full precision**, printed at the end of every run on BOTH arms and
  folded into the determinism key. It also replaces "diff two logs by eye" as this project's
  bit-identical check with a single number. **The lesson is about the CONTROL, not the test**: the
  test had already "passed" on hardware once, and only running the deliberately-broken build
  showed that the pass meant nothing.
- **`strings` ON THE ELF CAN REPORT A FALSE ABSENCE — found 2026-08-25.** The check recommended
  below (and for `SCENE_VERIFY`) is sound, but it answers "did the compiler emit this?", which is
  not "did the build enable this?". The `[coast]` printf was absent from a `FRAME_SOURCE=synth
  ITER_CNT=1` ELF with the feature correctly enabled: `g_run_frames` is a `static int` that
  nothing writes on that arm, so GCC proved the frame loop runs ONCE, and any state carried
  ACROSS iterations (here the coast velocity, set only in the mutually-exclusive branch) is then
  provably dead. It reappears at `ITER_CNT=200` or at `FRAME_SOURCE=vot`, where the count is
  written at runtime. **Verify a feature flag on a build that can actually exercise it**, and
  when a `strings` check says a feature is missing, check the loop bounds before the flags. Cost
  here: an hour of bisecting a compiler that was right.
- **PHASE 4's TWO KNOBS ARE WORTH 1.1%, NOT 15% — THE TRANSPORT WAS THE WHOLE WIN
  (2026-08-25).** Three `car1` runs, 8434 frames, all 15 state digests identical: UART + every-
  frame progress + every-row flush **28.48 ms**; ssh **24.69**; ssh + `CSV_FLUSH_EVERY=200`
  **24.70**; ssh + `PROGRESS_EVERY=25` **24.43**. So UART→ssh is **3.79 ms**, `CSV_FLUSH_EVERY`
  **0.00**, `PROGRESS_EVERY` **0.27** — against a written-down prediction of ~4.0 ms for the
  thinning. Premise again: 92.5 µs/byte is the cost of a byte *at 115200*, and the run being
  predicted was launched over ssh where a byte is nearly free. `CSV_FLUSH_EVERY` is zero for a
  second reason: the sweep runs from `/tmp`, which is **tmpfs**. **`car1` at 24.43 ms = 40.9 FPS
  is the best frame time recorded and is NOT comparable to the 26.29 ms in the performance
  history** (UART, synthetic scene).
- **VERIFY A FEATURE FLAG ON A BUILD THAT CAN EXERCISE IT — the check caught its own case.**
  `PROGRESS_EVERY=25` produced an ELF byte-identical to the default, i.e. the knob was INERT —
  correctly, because the default build is `VERBOSITY=1` where the level-0 line does not exist
  and the branch is dead-code-eliminated. At `VERBOSITY=0` the ELF differs at 10 and 25 and
  matches at 1. Read the first result as "it works" and a sweep runs with an unthinned console.
  **Two corrections were needed to make the defaults byte-identical**, both worth repeating: an
  unconditional `static` counter changes codegen even when its value is never used, so the
  default arm must be the original line TEXTUALLY (`#if N > 1 / #else`); and one inserted
  `fflush()` shifted every later offset, producing 9958 differing bytes from a single redundant
  call. **The control:** building the same source twice gives a byte-identical ELF, so an ELF
  `cmp` is a valid instrument on this project.
- **Console gating (`VERBOSITY`) details that mattered.** (1) The `VP1`/`VP2` macros are
  `if (VERBOSITY >= n)` on a compile-time constant, so format strings are **dead-code-eliminated,
  not merely skipped** — verify with `strings` on the ELF. (2) `dma_accumulate_frame()` had to be
  split out of `dma_report_frame()`; the printer was also the accumulator, so gating the print
  would have silently turned the CUMULATIVE report into a two-frame report. (3) **Anomalies print
  at every level.** (4) `VERBOSITY=0` still prints one line per frame deliberately: gating to
  nothing would delete the instrument `picocom | ts` needs.
- **`track.csv` carries `rails,accum_max,fch0_max,h_max` since 2026-08-24, and that retired the
  "`VERBOSITY=1` is load-bearing" rule.** The scan always ran — `report_cint16` gates the PRINT,
  not `scan_cint16` — so the numbers were computed and discarded. **Validated by reproducing a
  known answer**: the `VERBOSITY=0` run's CSV gives `F_ch`/`accum`/`response`/`H(q15)`
  digit-for-digit identical to the `VERBOSITY=1` run's console, at 28.58 ms/frame instead of
  62.67. `calib_report.py` falls back to the CSV and says which source it used; it distinguishes
  "0 rails" from "no rails column", because those must never print the same word. `fch0_max` is
  named for ch0 because `F_ch` is scanned under `if (ch == 0)` — not a bank max.
- **DMA 4258 → 1090 tx/frame** via `FFT_ROW_WS`/`FFT_COL_WS` 2→8; 96% of the traffic was the
  four per-invocation-chunked ports. **DMA is not a bottleneck and the fabric is at spec**:
  80 µs/tx is per-transaction *overhead*, not bandwidth — 64 B costs 14.4 µs and 128 KB costs
  22.8 µs (2048× the size for 1.6× the time). The largest transfer achieves **5.76 GB/s**.
- **x86sim bit-exactness harness** (`make x86sim_check`) — conv2d s6, cmul s7 and cmul_stress all
  16384/16384 identical. This verifies `simulate_conv2d`, on which the whole offline
  `phase1_sweep.py` methodology rests.
- **Test-sequence generation**: static background generated once with dirty-rect restore;
  `TRAJECTORY=1` closed elliptical path (peak 9.42 px/frame, ROI in-frame over 2000 frames);
  `SCALE_TRAJ=1` sinusoidal size (0.99%/frame peak vs the filter's 2%/frame step); per-frame IoU
  and centre error with an OTB-style run summary. All defaults reproduce previous behaviour.
- Earlier: cmul_accum saturation fix, conv2d hang root-cause (weights starvation), rootfs fix,
  uint8→int8 contract fix, filter init/update + Q1.15 export, H_SHIFT, scenarios s6/s7, hw_emu
  PLIO smoke test, N_CHANNELS=16 in aiesim (accum 7728 = 24% of cint16 — the cint16 DDR
  accumulator is sufficient). See [[cmul-accum-wrap-bug]], [[weights-bo-never-populated]].

## Where this tracker sits — the published VOT-STb2022 baselines

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
   entry under Settled); the open question is a different bank, not a filter over this one.
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

## Settled questions — do not reopen

- **`eps_rel = 1e-3` is optimal.** The response has a closed form `R = G·B/(B+ε)`. Sweeping the
  integer pipeline: ratio 12.90 / 13.96 / **16.15** / 10.62 / 4.01 at ε = 1e-5…1e-1. Bolme
  Fig. 4's flat curve does not transfer — his ε is absolute on the denominator, ours is relative
  to `mean(B)`.
- **ReLU off beats ReLU on, by ~3×, and the `bias_acc` fix must be paired with it.** Held-out
  peak/max-sidelobe: base(ReLU) 12.82, bias-corrected(ReLU) 3.92, bias-corrected(no ReLU) 16.25.
  `base` only looks decent because its oversized `bias_acc` makes ReLU a no-op on 11 of 16
  channels. **The shipping pair is now the best of the three** (`CONV_RELU=0`,
  `BIAS_SCALE=roi`, applied 2026-08-23); the middle column is what "apply the fix alone" means
  and is why the correction sat unapplied for months. A DCF is linear in feature space; a half-wave rectifier throws away half the signal
  and the filter cannot undo it. Caveat: one patch (s6), held out by circular shift, and it
  diverges from Danelljan §3.3.
- **Padding ≥2; recommend 2.0 — REOPENED 2026-08-24, and CLOSED 2026-08-28 IN FAVOUR OF 2.0.**
  See the feature-geometry entry below: on all 62 sequences, every padding under 2.0 is WORSE
  (R 0.2251 at 1.5 against 0.2910 at 2.0), and at two matched px/bin points padding 2.0
  wins both. The original static-scene objection stands and the answer came out the same.
  Historic detail: the original holdout (target 64, 4-2-2, ch16) gave PSR 18.4 at 1.5 against
  45.7 at 2.0, and 2.0 is the only value where the resample stays 1:1. **2.5 and 3.0 trip the
  aliasing detector (bilinear has no prefilter) and 3.0 clips 3.57% of samples** — which is the
  standing reason not to go up, independent of the tracking result.
- **σ stays 2.0; PSR cannot select σ.** Bolme PSR is monotone decreasing in σ all the way to
  sub-pixel (80.3 at σ=0.75 down to 19.3 at σ=5.33), so it just rewards a sharper peak. The
  tempting Stage-B2 explanation was tested with `--no-b2` and refuted — the dependence is
  intrinsic to the metric. σ needs real video, or a holdout with scale/appearance change.
- **fDSST's PCA compression is not worth it; the real-input DFT was, and is DONE.** Measured per
  frame (d=484, S=33): real-input DFT + Hermitian symmetry is **3.11×**; full PCA 1.59×, of which
  the QR is 0.46 ms — and the QR is O(d·S²), the same order as the DFT it eliminates. The 3.11×
  transferred exactly to hardware (`scale extract` 4730 → 2138 µs/call, isolating the transform
  at 3819 → 1228 µs). After that the real levers are d and S, not PCA.
- **HALVING THE HOST FILTER ON HERMITIAN SYMMETRY DOES NOT WORK — the premise is false in fixed
  point.** `conv2d` emits cint16 with `imag = 0`, so the 2-D spectrum is Hermitian *in exact
  arithmetic* and A/B look half-redundant (~4 ms). Measured on the real chain (64×64 ch1, s6,
  `make aiesim_plio`): `max|residual| = 12 LSB at bin (0,1), mean 1.379, 3924/4096 bins
  asymmetric, max|F| = 444` — **95.8% of bins differ from their conjugate partner**. DSPLib's DIT
  butterflies do not compute a conjugate pair by symmetric operations and every stage rounds.
  **The control is what makes this conclusive**: the float golden for the same scenario,
  quantised to int16, is Hermitian to **0 LSB across all 4096 bins**. So the index convention is
  right and int16 storage alone preserves the symmetry — the asymmetry is attributable to the
  fixed-point FFT and nothing else. Without that control a 12 LSB residual could just as easily
  have been a wrong conjugate index.
  **And the error lands in the worst possible place**: **max|H| sits where |F| is smallest**, so a
  fixed ~1.4 LSB asymmetry is proportionally most damaging exactly at the bins carrying the
  largest filter coefficients. The check is now permanent in the harness (`F_ch Hermitian
  check`), so any future FFT change re-tests it for free.
  **What is NOT refuted**: the same argument for the AIE's own forward transform — DSPLib's
  `fft_ifft_2d_graph` halves its memory tile for real input via `fft_dit_2ch_real_graph`, which
  computes only the independent half rather than computing both and discarding one. That saves
  work instead of reconstructing it, so per-stage rounding never enters the argument.
- **QUANTIZATION IS NOT THE CAUSE OF THE POOR ROBUSTNESS. Removing it makes tracking
  WORSE — measured 2026-08-27, offline, 8 sequences, 2841 frames.** The suspects are bracketed
  rather than argued:

  | suspect | verdict | what settles it |
  |---|---|---|
  | cint16 / Q1.15 / `H_SHIFT` pipeline | exonerated | `rgb_vs_gray_loop.py` is **float64 downstream of the features** and reproduces the board's failures anyway |
  | saturation / rails | exonerated | `corr(rail rate, mean IoU)` = **−0.025**; the shipping arm has **zero** rails over 101,564 frames and still scores R = 0.307 |
  | the int8 FEATURE path | **exonerated — it HELPS** | `--arms gray gray-float`: frame-weighted mean IoU **0.2533 → 0.2350**, **zero of eight sequences improve** |

  The control: the int8 arm reproduces the recorded per-sequence table to four decimal places,
  so the float column is the only thing that moved. **WHY float is worse** — on `tiger` it takes
  the frozen-detector rate from 74.7% to **1.4%** and IoU from 0.1696 to 0.0982. That is the
  third independent instance of the same rule: **the int8 grid acts as a DEADBAND**, suppressing
  sub-threshold responses a float pipeline follows faithfully to the wrong place. *Do not
  optimise the freeze rate*, and do not read "more precision" as "more accuracy" on a tracker
  whose features are the binding constraint.
  **So the robustness deficit is upstream of all arithmetic**: no pooling and a participation
  ratio of 4.94/7.43 against HOG's 31 dims; no spatial reliability, so the filter trains on 73%
  background; and a hold policy whose duration exceeds the recovery budget. **Corollary for the
  write-up: the fixed-point design costs nothing in accuracy or robustness, so the frame rate is
  bought at no algorithmic price.**
- **INIT PERTURBATIONS (Bolme §3.4) — CLOSED 2026-08-28. The 16-CHANNEL DENOMINATOR IS
  ALREADY THE CURE.** `mosse_tracker.cpp` carried "TODO: affine perturbations for
  initialisation; this is the N=1 case" from the start, and the board says it matters:
  `scripts/vot_init_anatomy.py` measures **61 of 373 losing runs (16%) failing within 10 frames
  of init**, already at median IoU 0.571 / PSR 7.35 one frame after `filter_init()` against
  0.915 / 36.73 for every other run. Those runs never acquire — a different mechanism from the
  drift in `robustness_gap.md`.
  Offline (`rgb_vs_gray_loop.py` arm suffix `-warp<N>`, 62 sequences, shipping eta/gate,
  scored with `vot_ar_offline.py`), **four perturbation axes × three regularizer settings, and
  nothing clears the +0.02 falsifier written down first**:

  | axis | board cost | dR |
  |---|---|---|
  | translation + isotropic scale | free (`roi_crop` runtime AXI-Lite) | +0.0081 |
  | aspect (`roi_h`/`roi_w` INDEPENDENTLY) | free | **−0.0322** |
  | rotation 10° (host pre-rotates into `frame_bo`; `roi_crop` cannot) | ~2 ms/warp at init | +0.0146 |

  Rotation is the best and is three sequences: dropping `rowing`/`singer2`/`birds2` gives
  −0.0020. It is the only arm moving A, R and mean IoU together, so it is the one variant that
  is not the failure-rule artifact below.
  **WHY IT DOES NOT TRANSFER, and this is the part worth keeping.** Bolme §3.3 presents
  regularization and perturbations as ALTERNATIVE cures for one defect — low-energy denominator
  bins — and **his Figure 3, the whole empirical case for perturbations, is captioned "Results
  shown without regularization"**. This design has both other cures: `eps_rel=1e-3`, and a
  SHARED denominator summed over **16 channels** (Bolme's own "sum of the energies over more
  images" argument, applied across the bank). The prediction that the warp gain is a function
  of the regularizer HELD — dR +0.0081 → +0.0143 as eps falls 1e-3 → 1e-6 — and then saturates,
  because `B` never gets that small: bins below `1e-6·mean(B)` are **0.00%** on car1/tiger/nature.
  **Bolme's ε≈0 regime is unreachable here by lowering ε.** So: perturbations do exactly what
  Bolme says, and this design already bought that stability structurally; the ceiling is ~+0.014
  in R. It is a property of the FEATURE BANK, not of the fixed-point path.
  Two things not to re-derive: translation warps are NOT algebraically degenerate (they should
  be — `conj(G_δ)F_δ = conj(G)F` exactly — but the fixed Hann window and border inflow give
  `rel|ΔA|` 0.19/0.30/0.51 at 2/5/10% shift, so "the jitter was too timid" is refuted); and
  photometric warps are worthless because Stage A's log → zero-mean → unit-L2 annihilates gain
  and contrast, leaving 7–9 LSB of quantization residue against a patch σ of ~32.

- **FEATURE GEOMETRY — POOLING, RESOLUTION AND PADDING ALL TESTED, ALL REJECTED (2026-08-28).**
  Offline (8 seq, then all 62, `rgb_vs_gray_loop.py` arms `-pool<N>/-dec<N>/-blur<N>`), then
  hardware. `docs/thesis/evidence/pooled_features.md`.
  * **Aggregation is a NULL on both banks, and MAX POOLING IS THE SAME NULL (2026-08-31).**
    `blur2` (2x2 box average at STRIDE 1) is −0.0010 gray / −0.0012 RGB, and `pool2 <= dec2`
    everywhere. Prompted by Danilowicz & Kryjak 2022, whose stem is VGG11 conv1 **with ReLU and
    2x2 MAXPOOL**, `-mpool`/`-relumpool` were added: `dec2` R 0.3981, `mpool2` 0.3971, `pool2`
    0.3970 — **max, average and NO aggregation agree to 0.001.** The operator is irrelevant; the
    hypothesis that averaging a SIGNED map cancelled the lobes is arithmetically true and
    trackingwise nil. `relumpool2` pins accuracy at baseline (0.5393 vs 0.5394) for a third of
    the robustness — a conservative tracker on an easier prefix.
  * **RESOLUTION IS NOT REFUTED, AND IT IS THE STRONGEST OFFLINE ARM MEASURED HERE.** A 64x64
    map at `TARGET_PADDING=2.0` (px/bin box/32) gives **dR +0.1071**, survives dropping the
    top-3 gainers at **+0.050** and a bootstrap CI excluding zero (P(dR<=0)=0.000) — against the
    shipped spatial mask's +0.0330 / +0.0101 on the same instrument. It is a RESOLUTION change,
    not a pooling one (`dec2` alone scores it), and it is the geometry Danilowicz ships. **It
    costs 0.25x the host work (~15 ms/frame) — robustness AND frame rate.** Standing discount:
    this bench over-predicted the mask 3x and pad30 ~11x; unlike pad30 it holds padding at 2.0,
    so the missing-scale-filter blindness that broke that prediction does not apply. Claim
    N-03b, `docs/thesis/evidence/pooled_features.md`. `blur2` is the arm that
    matters and it did not exist in the first sweep; without it the result reads "pooling loses",
    which is true and unattributable.
  * What moved was **px/bin**, and only on RGB. **Gray and RGB gave OPPOSITE signs for the
    resolution half; do not generalise a feature result across banks.**
  * **HARDWARE KILLED IT.** `runs/vot/0828_1451-pad30`, 62 seq / 419 trajectories, host-only:
    padding 3.0 vs 2.0 gives A 0.5100 → 0.5030, R 0.3417 → **0.3494 (+0.0077 against a predicted
    +0.088)**, and **EAO 0.1629 → 0.1570 — DOWN**. Below the +0.02 floor written before the run.
    `TARGET_PADDING=2.0` stands and the 64x64 reflash is OFF — it rested on the same proxy.
  * **Why the proxy failed:** the offline loop has NO DSST scale filter, and padding sets the
    ROI `scale_extract` draws from, so the whole scale axis was invisible by construction; and it
    is SINGLE-START, where a big search region has hundreds of frames to earn back a drift,
    against 419 short anchored runs. Both were named as caveats and neither was priced.
    **A caveat that is not priced is a hope.**
  * Free corrections: **padding costs NO frame time** (it resamples to a FIXED 128x128 patch, so
    cost is output px × 4 taps, independent of input extent), and more context makes the gate
    LOOSER (holds 15.51% → 10.94%). **Instrument bug:** `vot_detector_gain.py` hardcodes
    `box*2/128`, so on pad30 it reports alpha 0.442 where the corrected value is ~0.66.
  Still NOT refuted: a REPLACEMENT bank. That is a weights export plus the same offline loop.
- **Channel pruning is moot** with ReLU off, and doubly so at `BIAS_SCALE=roi`, which retires
  the last two structurally dead channels (ch3, ch15) outright. The real
  redundancy is the collapse: it caps the bank at **rank 9** (participation ratio 4.94) and
  leaves ch0/ch9/ch14 collinear up to sign, and collinear channels add exactly coherently. The
  fix is RGB. `check_collapse.py` Q2 used to print "14 independent filters" here — that was a
  count of near-parallel GROUPS, not a rank, and it understated the problem for months.

## RGB features — SHIPPING, and it is a ROBUSTNESS win

`CONV_IN_CH=3` is the default. Build with `make weights CONV_IN_CH=3` then
`ARM=rgb scripts/calib_build.sh`. 4-4-4 carries both arms; no RGB-specific FFT budget.

### Why RGB — CLOSED ON HARDWARE 2026-08-27

`~/vot/analysis/full62`, 419 runs per arm, both verified run-name-and-length against the
dataset's anchors before analysis:

| arm | A | R | EAO | frames |
|---|---|---|---|---|
| gray `H_SHIFT=14` | 0.4890 | 0.2743 | 0.1367 | 48,603 |
| **RGB `H_SHIFT=15`** | **0.5043** | **0.3065** | **0.1474** | **54,813** |

R better on 37 / worse on 15 / tied on 10. Largest swings are all RGB gains and all in
ROBUSTNESS: `car1` 0.634 → **1.000**, `book` +0.311, `lamb` +0.272. **12.8% more frames
survive.** (Confound: the arms are at different `H_SHIFT`.)

**RGB WAS NEVER AN ACCURACY CHANGE, and the shape matters more than the number.** Offline mean
IoU over 8 stb2022 sequences is a TIE (+0.0011); the synthetic hardware arm moved the wrong way
(0.9188 → 0.9173). Its claim was always **failures**, −18% under the supervised protocol — a
robustness metric mean IoU structurally cannot express. **Decide a colour question on AR, never
on mean IoU.**

Supporting offline measurements (`scripts/rgb_vs_gray_*.py`):

| measurement | gray | RGB |
|---|---|---|
| feature-bank rank / participation ratio | 9 (hard cap) / 4.94 | 16 / 7.43 |
| held-out Bolme PSR (147 paired evals, car1) | 12.97 | 21.18 (**+1.63×**) |
| VOT supervised failures (16 seq, 5971 frames) | 51 | **42 (−18%)** |
| conv2d scheduled cycles/frame (ch16) | 4.60 ms | 9.19 ms (**2.00×**) |

**Why the collapse costs anything.** A 3×3 grayscale kernel lives in 9 dimensions, so 16
channels CANNOT be independent — the cap is structural, not a property of the pretrained
weights. BT.601 guts the four colour-opponent channels: 0/2/9/10 keep 0.32/0.60/0.63/**0.037**
of the per-plane norm against 1.24–1.39 for the achromatic ones, and per-channel int8 then
renormalises that residue to full scale. ch0/ch9/ch14 sit within 2–6° of one line in gray,
59–72° apart in RGB.

**The 08-24 validation, kept for its method.** Three 200-frame arms, one variable each
(`run_0824_1354`/`_1432`/`_1442`). The colour-free control — same 27 taps, bias, quantization
grid and joint normalization, fed three IDENTICAL luma planes — reproduced grayscale's
decisions **bit-for-bit on all 199 frames**, which validates the whole RGB datapath against a
known-good reference AND proves the 1.65× PSR floor is colour and not bookkeeping. ch0's
`F_ch` collapse to 0.084× of gray was PREDICTED at 0.09× from the exported weights alone —
the sharpest confirmation in this file that the weight-collapse model is right, and what makes
ch0's `F_ch` the cheap on-board colour-path test.

**THE SYNTHETIC SCENE IS A WEAK COLOUR STIMULUS.** `FRAME_RGB_MODE=1` tints one luma image per
plane, i.e. **rank-1 across the plane dimension**: no chromatic texture independent of luma, so
a colour-opponent channel sees a scaled copy of luma. ch0 lands at 0.52× instead of the
decorrelated 1.25×. **A good IoU on that scene is NOT evidence for the VOT result above.**

### The datapath, end to end

**The wire format is pixel-interleaved, the line buffer is planar.** roi_crop must send
interleaved (planar would need whole planes resident); conv2d de-interleaves into three 3-row
buffers as it unpacks, costing index arithmetic and ~1.2 KB. Without it `load_unaligned_v`
gathers `[R G B R G B…]` and the MAC loop needs shuffles. Three int32 words carry exactly four
RGB pixels: `R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3`.

**`roi_crop` Stage A** at `ROI_IN_CH=3`: all three planes share one geometry and one set of
bilinear weights — only the `+p` byte offset differs — and the frame is interleaved in DDR, so
one source pixel's three taps are contiguous. Normalization is **JOINT**: one mean and one
`inv_q` over all 3·pr·pc samples. Per-plane statistics would equalize the planes and delete
exactly the chromatic contrast RGB exists for — silent, and self-defeating. Scratch 16 → 48 KB.
`sum_x2` peaks at 2.111e14 against `ap_uint<48>`'s 2.815e14, so **a fourth plane would not fit**
and the reference asserts the width.

**The host keeps the scene in LUMA and colourises on the way out.** Every scene function and
`scale_extract`'s 33 crops are single-plane; one pass expands the touched rect into the
interleaved buffer. At `CONV_IN_CH=1` there is no second buffer and no copy. `scale_extract`
reads luma, so **the DSST scale filter needs no recalibration**. The invariant is that every
luma write reaches `scene_touch()`; miss one and the device reads last frame's colour there,
which looks like a slightly worse tracking result rather than a bug. `SCENE_VERIFY=1` catches it.

**The RGB conv2d stack.** `make graph CONV_IN_CH=3` used to fail the link-stage stack check
(1344 bytes against 1024) producing NO `libadf.a`, while the per-kernel *compile* succeeded
either way — which is why it went unnoticed. Fix: `stack_size(conv2d) = CONV2D_STACK` (2048),
applied **only** at `CONV_IN_CH=3`. No kernel arithmetic changed; the RGB build allocates
`MG(15,0) size: 0x800` while every other node stays `0x400`.

**The RGB branch is VECTORIZED, not scalar** — 27 `aie::mac` with `load_unaligned_v`. The
`static_assert(CONV_IN_CH == 1)` guards the SEPARATE grayscale block, which the RGB branch
returns before reaching. Reading its 27 hoisted `int8_t` *weight* scalars as a scalar datapath
makes the 219-cycle schedule look far worse than it is.

### Testing — and why each suite means anything

Every RGB suite is **mutation-tested**, because a passing test on a path with no prior coverage
is worth nothing until it has been shown to fail.

| suite | coverage | mutants caught |
|---|---|---|
| `make test_roi_crop` | 17 gray + 8 RGB cases, zero tolerance | per-plane mean, planar scratch store, dropped plane index: 6-7 of 8 RGB, 0 of 17 gray |
| `make test_scene` | interleave, Q8 gains, saturation, clipping, missed `scene_touch()` | 8 of 8 |
| `make x86sim_check … s6rgb` | the real 27-tap kernel vs `simulate_conv2d`, 16384/16384 | de-interleave 48.4%, dropped MAC 22.8% |

The survivors are informative: `rgb_flat` survives everything (var 0 → all zeros regardless),
and a dropped plane index survives `rgb_gray_control` because with three identical planes it
genuinely IS a no-op. **A suite built only from replicated-luma frames would have caught none of
that mutant** — hence decorrelated planes in the RGB test frames. `s6rgb` writes to its OWN
directory (overwriting `s6` would silently feed RGB vectors to a grayscale check) and its patch
comes from `roi_crop_ref.stage_a_rgb`, not s6's float shortcut. **Only ONE RGB scenario exists,
on purpose**: RGB changes conv2d and nothing downstream.

### Cost — RGB COSTS WHAT THE HOST PAYS, NOT WHAT conv2d COSTS

**conv2d 2.00×**, from the compiler's schedules, reproduced byte-for-byte by the build that
linked:

| loop | gray | RGB | ratio |
|---|---|---|---|
| stream read, per 4 px | 28–31 cyc | 84–87 cyc | **3.00×** |
| MAC + post, per 16 px | 163 cyc | 219 cyc | 1.34× |
| per frame at ch16, 1 GHz | 4.60 ms | 9.19 ms | **2.00×** |

**The stream read goes from 44% of conv2d to 61%** — RGB makes the already-dominant term more
dominant, because the patch is re-streamed once per output channel. The RGB MAC loop does NOT
software-pipeline (`219 (exceeds -k 64) → no folding`, critical cycle 200 against gray's 24),
so 219 is a give-up number a tuned variant could plausibly beat.

**Two traps in reading those schedules.** (1) `aiecompiler` reuses a cached per-kernel object
when the preprocessed source is unchanged, so a `CONV_IN_CH=1` baseline silently reports NO
conv2d schedule — `rm -rf $(BUILD_DIR)/Work $(BUILD_DIR)/libadf.a` first. (2) The "conv2d 140
cyc/16px" figure in the Makefile is the `main_` WRAPPER block, and does not move when the
arithmetic triples.

Measured `run_0824_1457` vs `run_0821_1725`: **28.58 ms vs 26.29**, so +2.29 ms — and conv2d's
+4.59 ms of AIE compute **does not appear in the frame at all** (GMIO total unchanged, 11.133 vs
11.134; blocking `wait()` unchanged, 4.55 vs 4.51). The whole +2.29 is host-side:

| stage | gray | RGB | delta |
|---|---|---|---|
| frame push (`frame_bo` 2 → 6 MB) | 0.472 | 1.385 | **+0.91** |
| roi_crop launch (3× bilinear taps) | 1.013 | 1.471 | +0.46 |
| colourise RGB (new pass) | — | 0.338 | +0.34 |
| `frame_bo.sync` | 0.116 | 0.304 | +0.19 |
| filter upd+quant | 4.806 | 5.215 | +0.41 |
| **frame** | **26.29** | **28.58** | **+2.29** |

**The offline "~30 FPS" prediction was arithmetically fine and its PREMISE was wrong** — it
costed conv2d as if AIE time were frame time. The frame is 84% CPU-bound, so the AIE had slack
and absorbed the doubling. Third time a self-consistent offline model has been overturned by its
premise here. **The lever for RGB speed is host memory traffic, not the 27 taps.**

**Retired — "RGB is handicapped by its larger `out_shift`."** 27 taps triple `ACC_MAX_THEORY`
(mean out_shift 3.69 → 4.25), but forcing gray's shifts onto RGB (`--match-shift`) makes it
**worse**, 42 → 53 failures, with 0.0000% saturation at all three clip sites. Not clipping.
**Caching the patch in conv2d's tile does not fit**: at `FFT_ROW_WS=64` the output window's
ping-pong is already the whole 64 KB tile. **No alignment obstacle** — the PLIO is 32-bit and
128·128·3 = 49152 B divides exactly. See [[verify-stated-blockers-arithmetically]].

## Weight export

`make weights` → extracts torchvision mobilenet_v3_small conv1, folds BatchNorm, then either
collapses RGB→gray by luminance (`CONV_IN_CH=1`, default) or keeps all 27 taps
(`CONV_IN_CH=3`); symmetric per-channel INT8 quantization over ALL taps of a channel, never per
plane. `BIAS_SCALE` picks the activation scale `bias_acc` is derived against (`roi` by default
since 2026-08-23). Outputs `design/aie_src/weights/layer0_weights.bin` (16 × 64 B),
`weights/layer0.h`, `design/aie_src/hanning_128.h`. Byte 63 of each channel buffer is the
layout tag; three independent guards fire on a build/file mismatch.

`scripts/check_collapse.py` is the front-end diagnostic — four checks, no hardware, seconds,
where the alternative for each is an aiesim or hw_emu run. **Re-run it after any change to
`export_weights.py`, `ROI_NORM_Q`, or the collapse.** Q1 collapse convention (needs torch);
Q2 linear diversity of the kernels; Q3 `bias_acc`/`out_shift` sanity (input-independent — trust
this one); Q4 post-ReLU maps through conv2d's exact integer datapath (patch-specific).
`--skip-torch` gives Q2-Q4 without torch.

## Build commands

```bash
# DEFAULTS ARE THE SHIPPING ARM since 2026-08-28 (CONV_IN_CH=3 H_SHIFT=15
# MOSSE_ETA=0.05 PSR_GATE_MIN=5.0). Pass CONV_IN_CH=1 for the grayscale arm --
# the gray aiesim scenarios and the 17 gray roi_crop cases need it explicitly.
make weights                       # export layer-1 INT8 weights + hanning table (RGB, 27 taps)
                                   #   BIAS_SCALE=roi by default since 2026-08-23;
                                   #   BIAS_SCALE=127 reverts to the pre-correction file
make weights CONV_IN_CH=1          # ...as the 9-tap BT.601 luminance collapse instead
make gen_vectors                   # generate aiesim test vectors
make graph                         # compile AIE graph only — use this to answer placement
                                   #   questions (5 min) instead of a 25-min package
make test_host                     # native unit tests for filter/PSR/scale/scale-gate/
                                   #   training-target/fusion/scale-reuse/real-DFT/conversions.
                                   #   Runs the suite TWICE — the second time with
                                   #   -O3 -march=native -ffp-contract=fast, because the board's
                                   #   compiler contracts mul+add into FMA by default and a
                                   #   bit-exactness claim proven only at -O2 is proven on the
                                   #   wrong machine. That build caught real bugs -O2 missed.
python3 scripts/mosse_loop_sim.py  # closed-loop MOSSE regression, centred-G vs shifted-G
python3 scripts/mosse_loop_sim.py --subbin   # ...and the sub-bin experiment: sweeps resample
                                   #   ratio x speed and shows quantisation does NOT compound
# gray vs RGB vs a colour-free control, on real VOT video. Needs the venv, and the Vitis
# env MASKS it: PYTHONHOME/PYTHONPATH point python at Vivado's build, which has no _ctypes.
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_holdout.py   # frozen filter
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py      # closed loop, 1 seq
#   ... --sequence tiger   picks the sequence ($VOT_ROOT stb2022 first, then test-sequences/).
#   Reproduces the board's frozen-detector failures at 3.5 s per 100 frames.
#   ... --arms rgb rgb-warp8  is Bolme 3.4's N-sample init (see Settled). The warp
#   set is restricted to what roi_crop can PRODUCE -- translation and scale; there is
#   deliberately no rotation knob in warp_set(), and --warp-rot models the HOST
#   pre-rotation route instead. --warp-aspect uses roi_h/roi_w independently (free on
#   the board). --warp-mutant {gsign,noshift} are the negative controls, banner-printed
#   and invalidating. --eps-rel reaches Bolme's unregularized regime.
#   ... --arms gray gray-float   is the QUANTIZATION COUNTERFACTUAL: gray-float and
#   rgb-float run the same loop with unquantized folded-BN weights, no out_shift, no
#   int16 clips and a float Hann. Everything else is identical, and since the model is
#   already float64 downstream of the features, the pair brackets the whole
#   quantization question. Answer: removing it makes tracking WORSE -- see the settled
#   entry. Stage A's int8 patch is deliberately NOT removed (it is roi_crop's output,
#   a separate choice with its own scale), so a difference stays attributable.
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_vot.py       # VOT supervised, all seqs
#   ... --oracle-scale takes box size from ground truth (isolates localisation from the
#   missing DSST scale filter); the two modes BRACKET what scale handling is worth.
#   uv sync --extra weights  installs torch/torchvision (needed for the folded weights).
make scale_sim                     # closed-loop DSST scale sim — reproduces the f130 stall
make test_roi_crop                 # native bit-exact roi_crop test. ROI_IN_CH is
                                   #   COMPILE-TIME, so one build runs one arm: the default
                                   #   runs the 8 RGB cases and skips 17, and
                                   #   CONV_IN_CH=1 runs the 17 gray and skips 8. Zero
                                   #   tolerance; a build matching no case exits 2 rather
                                   #   than passing vacuously. RUN BOTH before shipping
make test_scene                    # native test of the luma -> interleaved-RGB pass,
                                   #   including a deliberately missed scene_touch() so
                                   #   the verifier is known to fire, not assumed to
make test_vot_source               # native test of the VOT manifest parser, blob
                                   #   offsets, run-order convention, trajectory
                                   #   writer and the STREAMING reader. The 19
                                   #   manifest/blob/trajectory mutants plus 7
                                   #   more for the ring, each REJECTED.
                                   #   With $VOT_ROOT exported it also parses
                                   #   every real manifest AND streams a forward
                                   #   and a backward slice of the LARGEST real
                                   #   blob against an independent fopen/fread —
                                   #   the fixtures are 40-byte frames, where a
                                   #   partial read is impossible and at 2.64 MB
                                   #   it is routine
make test_vot_format               # the BOARD's trajectory writer read back by the
                                   #   toolkit's OWN reader, with the centre -> top-left
                                   #   conversion re-derived in Python. Needs the venv
make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0 CONV_IN_CH=1   # bit-exact, gray
make x86sim_check KUT=cmul   SCENARIO=s7                 # ...same for cmul_accum
make x86sim_check KUT=cmul   SCENARIO=cmul_stress        # ...exercising sat16's rails
#   cmul needs CMUL_SPLIT_ACCUM=0 here — kernel_only_graph leaves cmul.in[2]
#   unconnected otherwise and the x86sim graph refuses to compile.
make x86sim_check KUT=conv2d SCENARIO=s6rgb              # RGB conv2d, 27 taps, bit-exact
                                   #   needs `make weights` and `make gen_vectors` first
                                   #   (both now default to RGB)
make aiesim CMUL_SPLIT_ACCUM=0     # AIE simulator — bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but forces the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine one).
                                   #   ALSO provisions the board for ssh (static end0 + an
                                   #   authorized key). BOARD_KEY=none opts out explicitly;
                                   #   there is no silent skip when the key is missing
make board_provision ROOTFS_IMG=<img>   # same, against an existing sd_card.img: no re-package
# One command per sweep: mounts, pushes the ELF, guards, runs, collects, ingests.
scripts/vot_sweep.sh --arm coast0 --seqs car1,tiger --ingest
scripts/vot_sweep.sh --arm x --seqs car1 --dry-run    # prints every remote command, runs none
make kernels / xsa / application / sd_card
make sd_card TARGET=hw
make run_emu LAUNCH_HW_EMU_EXEC=1
make cleanall
```

## Directory layout

```
design/
├── aie_src/
│   ├── fft_graph.h / ifft_graph.h    # FFT2D / IFFT2D graphs (memory-tile transpose)
│   ├── conv_weight_layout.h          # THE weight-buffer layout, derived from CONV_IN_CH.
│   │                                 #   Included by the kernel AND the host (no <adf.h>)
│   ├── conv2d_kernel.h/.cpp          # 3×3 MAC + optional ReLU + Hanning window + Stage B1
│   ├── cmul_accum_kernel.h/.cpp      # col-FFT ⊙ H_ch* + saturating accumulate
│   ├── mosse_graph.h/.cpp            # top level: PLIO + GMIO + 2 kernels + FFT2D + IFFT2D
│   ├── kernel_only_graph.h/.cpp      # x86sim single-kernel harness
│   ├── aiesim_scenario_io.h          # shared scenario loader (keeps the two harnesses in sync)
│   ├── constraints.aiecst            # PatchIn PLIO shim placement
│   ├── hanning_128.h, weights/       # auto-generated
│   └── aiesim_data/s*/               # scenarios: s0-s4 raw patches (echo mode only);
│                                     #   s6 Stage-A preprocessed, H=unity, real conv path;
│                                     #   s7 s6 + a real per-bin complex H, off-centre target
│                                     #   (the only one exercising H_SHIFT, PSR, the F_ch tap);
│                                     #   s6rgb 3-plane Stage A -> the 27-tap conv path;
│                                     #   cmul_stress exercises sat16's rails
├── pl_src/{camera_capture,roi_crop}/
├── host_app_src/
│   ├── mosse_tracker.cpp             # GMIO-driven XRT tracking loop; CropIp (user-managed
│   │                                 #   roi_crop) + the RC_*/timeline launch instrumentation
│   ├── mosse_filter.h/.cpp           # init/update/PSR/scale/Q1.15 export — NO XRT include
│   ├── scene_colour.h/.cpp           # luma scene -> interleaved RGB, rect union, the
│   │                                 #   incremental-vs-full verifier. NO XRT include, so
│   │                                 #   `make test_scene` builds it in seconds
│   ├── vot_source.h/.cpp             # VOT manifest/blob/run-order/trajectory. NO XRT
│   │                                 #   include; linked only at FRAME_SOURCE=vot. Pure
│   │                                 #   bookkeeping, so every failure mode is a
│   │                                 #   plausible-but-invalid AR report -- hence the
│   │                                 #   mutation suite. Blob is resident (one read()
│   │                                 #   per sequence); StreamBlob is the ring +
│   │                                 #   prefetch thread for the five RGB sequences
│   │                                 #   that exceed the ~1 GB of usable heap. Access
│   │                                 #   is strictly sequential in job_order() and
│   │                                 #   OUT-OF-ORDER IS AN ERROR, not a seek
│   └── test/                         # native tests + NumPy goldens
│       ├── test_mosse_filter.cpp     # make test_host
│       ├── test_scene_colour.cpp     # make test_scene
│       ├── test_vot_source.cpp       # make test_vot_source; `<dir>` argument emits a
│       │                             #   trajectory + its INPUT for make test_vot_format
│       └── scale_loop_sim.cpp        # closed-loop DSST scale sim (make scale_sim)
├── system_configs/mosse_x1.cfg       # v++ linker
├── profiling_configs/, directives/
└── exec_scripts/run_script.sh
scripts/  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py,
          check_collapse.py, check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py,
          gen_roi_crop_golden.py, synth_frame.py, sweep_shift.sh, fix_sd_rootfs.sh
          conv_weight_layout.py  # Python mirror of conv_weight_layout.h. EVERY reader of
                              #   layer0_weights.bin goes through it; the tag byte makes a
                              #   layout mismatch loud instead of plausible
          calib_build.sh      # hardware build for a budget or bring-up run: pre-flight, then
                              #   verifies the FLAGSTAMPS against the intended config. Budget
                              #   defaults DERIVED from the Makefile (print-%), never copied
          vot_prepare.py      # VOT sequences -> board blobs + manifests, and the
                              #   mutation-tested verifier. THE groundtruth reduction lives
                              #   here -- other readers import reduce_box from it
          vot_sweep.sh        # drives a whole sweep over ssh: mount, push the ELF, guard the
                              #   build, run, collect, ingest. --dry-run prints every command
          vot_ingest.py       # board trajectories -> toolkit workspace -> `vot analysis`. One
                              #   directory per ARM. Re-derives every run name from the
                              #   sequence's anchors and checks each trajectory's LENGTH --
                              #   scan() only notices a MISSING file, and a wrong-length run
                              #   is scored without complaint
          vot_roundtrip.py / vot_check_trajectory.py  # the toolkit's result format, and the
                              #   BOARD's writer against the toolkit's reader
          board_provision.sh  # static end0 address + root's authorized_keys into the rootfs
                              #   (or an sd_card.img partition) with debugfs -- no root, no
                              #   loop device. sshd is already enabled in the stock image
          calib_report.py     # a run's console+track.csv -> a budget verdict (rails,
                              #   amplitude early vs converged, IoU)
  -- diagnosis, all offline, all reading existing CSVs or the dataset --
          vot_motion_check.py # does a sequence's ANNOTATED motion appear in the PIXELS? NCC of
                              #   the box content at "moved" vs "still". Phase correlation
                              #   cannot answer this -- it returns the window's dominant motion
          vot_detector_gain.py # IS THE DETECTOR THE PROBLEM? Regresses reported peak offset on
                              #   annotated motion, per speed bucket, over accepted ON-TARGET
                              #   frames. Answer: no, alpha 0.93 on targets that translate.
                              #   --movers splits off sequences whose box moves while the
                              #   pixels do not; WITHOUT that split the pooled 0.686 reads as a
                              #   broken detector. NOTE it hardcodes padding 2.0
          vot_loss_anatomy.py # what the tracker was DOING as it lost: gate verdict, PSR and box
                              #   motion in the 5 frames before the first loss
          vot_init_anatomy.py # INIT FAILURE or DRIFT? Time-to-first-loss under the toolkit's
                              #   rule plus the IoU/PSR profile right after filter_init(). An
                              #   init failure is visible at f1; a drifting run is not, and that
                              #   is the whole discriminator. --drift-warning adds the
                              #   relative-PSR reading WITH its control window. Keys on
                              #   (job, frame)
          vot_traj_anatomy.py # a board trajectory vs groundtruth in units of the tracker's own
                              #   BIN -- discriminates a resolution limit from a pinned detector
          vot_hold_budget.py  # frames before a frozen window loses the target, from groundtruth
          vot_mask_stat.py    # reads the `mask_ebox` column -- THE mechanism check for
                              #   FILTER_MASK. Keyed on (sequence, job, frame); excludes the -1
                              #   rows rather than averaging them as zeros; reports at-init and
                              #   per-frame-index, never a pooled mean, because the fraction
                              #   RISES as the filter converges. Paired mode scores the frames
                              #   valid in BOTH arms
          check_ebox_crosscheck.py  # the board's filter_box_energy_fraction against the
                              #   offline box_energy_fraction on the SAME H, plus 5 injected
                              #   mutants (-DEBOX_MUTANT). Exists because the two disagreed
                              #   silently and every unit test stayed green
          vot_ar_offline.py   # VOT's failure rule applied to OFFLINE single-start runs. NOT the
                              #   toolkit's AR -- different RUNS, same rule -- and its
                              #   resolution is MEASURED at ~0.02 in R, including one pair it
                              #   got BACKWARDS. Decides whether an arm deserves board time,
                              #   never whether to accept one. Did NOT transfer on pad30
  -- offline benches, no hardware, seconds to minutes --
          mosse_loop_sim.py   # closed-loop MOSSE, centred-G vs shifted-G; --subbin sweeps
                              #   resample ratio x speed
          bg_pan_sweep.py     # picks BG_PAN_R/C from the texture spectrum
          rgb_vs_gray_holdout.py / _loop.py / _vot.py   # gray vs RGB vs a colour-free control
                              #   on real VOT video. _loop.py also carries the float
                              #   quantization counterfactual, the pooling/resolution arms, and
                              #   the -warp<N> init-perturbation arms with their mutants
test-sequences/   VOT sequences + annotations (16 usable, 5971 frames). The annotation
                  directories are named inconsistently ("car1-annotations" but
                  "fernando - annotations"); the harness matches them loosely.
```
