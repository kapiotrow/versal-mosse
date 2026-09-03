# Build parameters — full table

**Status:** current · **Updated:** 2026-09-02 · **Scope:** every build knob, one paragraph each; defaults checked against the Makefile

**Every build knob, one paragraph each.** Split out of CLAUDE.md 2026-08-31 and
maintained here since. CLAUDE.md carries a DIGEST — the ~28 knobs that reach a
toolchain, get set on a routine command line, or silently invalidate a run — and links
here for the rest.

**Neither table owns a default: the Makefile does.** `make check-build-table` (part of
`make check-docs`) fails if any default here or in CLAUDE.md disagrees with
`make print-<KNOB>`, if either names a knob the Makefile does not define, or if a knob
in the digest has no paragraph here. Run it after any `?=` change.

## Build parameters

| Parameter | Default | Notes |
|---|---|---|
| `TARGET` | `hw_emu` | `hw_emu` or `hw` |
| `PATCH_ROWS` / `PATCH_COLS` | `64` | The FEATURE MAP, not the crop. Powers of 2 (AIE FFT constraint). **Moving it silently moves `sigma/target`** (see `MOSSE_SIGMA`) **and invalidates the shift budget**, which follows the point size. `128` is the pre-09-02 geometry and is still fully supported — pass it back with the 3x3 bank |
| `N_CHANNELS` | `32` | conv feature channels. Host cost scales as `N_CHANNELS x map_pixels`, and so does conv2d's read loop. `16` is the pre-09-02 value; the two are not trim-separable on tracking |
| `FFT_2D_DT` | `0` | 0=cint16, 1=cfloat |
| `ITER_CNT` | `1` | Frames. **Needs ≥2** — frame 0 initialises the filter |
| `PL_FREQ` | `312.5` | MHz. Platform also offers 625 / 156.25 / 100 / 78.125 |
| `H_SHIFT` | `15` | Deliberately OVER-shifted: `rails=0` over 101,564 frames, and what the flashed xclbin was built with. 13 is the tight RGB value, 12 rails, gray's arm is 14. cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget. **Reaches `AIE_FLAGS`** — one of the few non-host-only knobs here, with `CONV_IN_CH`, `CONV_KSIZE`/`CONV_STRIDE`, `CONV_RELU`, `CONV2D_STACK`, the FFT/window knobs and the geometry |
| `FFT_SHIFT` / `IFFT_ROW_SHIFT` / `IFFT_COL_SHIFT` | `3` / `3` / `3` | The 64x64 budget, calibrated 2026-09-02 (`rails=0` over 200 frames, all four buffers). **4-4-4 is the 128x128 budget** — the budget follows the POINT SIZE, so a geometry change needs its own calibration run before any of its numbers mean anything. See [`shift_budget.md`](shift_budget.md); do not change either without a ≥20-frame hw run |
| `FFT_ROW_WS` | `64` | Rows per FFT invocation — the DMA transaction-count knob. Exhausted, see history |
| `FFT_COL_WS` | `8` | Cols per FFT invocation. **32 is a 9.6 ms LOSS — do not raise** |
| `MEMTILE_TRANSPOSE` | `1` | Forward+inverse transposes in AIE-ML memory tiles. A one-sided flag is a board deadlock, not a compile error |
| `ROI_CROP_PIPELINE` | `1` | Launch channel k+1's crop before polling k |
| `CMUL_SPLIT_ACCUM` | `1` | `accum_prev` gets its own kernel port. **`make aiesim` needs `0`** (2025.2 aiesim deadlock); hardware is fine either way |
| `CMUL_ACCUM_MEMTILE` | `0` | Tried, a 0.36 ms loss. Code kept behind the flag |
| `TAIL_PARALLEL` | `1` | `filter_update_quantize` on core 1 ∥ scale filter on core 0. Needs `-pthread` in `GCC_FLAGS` |
| `ROI_CROP_USER_MANAGED` | `1` | 1 = roi_crop driven via `xrt::ip` (host writes AXI-Lite, polls the CU's own `ap_done`). **20.6× on frame rate** — KDS completion costs 503 ms/launch because the CU interrupt is never delivered. Host-only |
| `CONTROL_CU_RUNS` | `0` (`8` on `hw`) | camera_capture launched N times at startup on the KDS path — a within-run control that should still pay ~512 ms while roi_crop does not |
| `CONV2D_MODE` | `0` | 0 = real 3×3 conv, 1 = echo passthrough, 2 = synthesize |
| `CONV_VECTORIZE` / `CMUL_VECTORIZE` | `1` | Vectorized kernels, bit-identical to scalar; 0 restores scalar for bisection |
| `CONV_RELU` | `1` | **Bank-specific, not a global preference.** On the 3x3 mobilenet bank it is refuted (dR −0.0332, and it costs ~25% of the peak/sidelobe ratio). On a LEARNED Layer-1 bank it beats its own linear twin four times offline; on an ANALYTIC Gabor bank it LOSES, so what matters is that the bank is learned. Ships on. Reaches `AIE_FLAGS` — a rebuild. **The linear twin has not run on hardware**, so the gain belongs to the arm, not yet to the rectifier |
| `CONV_KSIZE` / `CONV_STRIDE` | `7` / `2` | Conv kernel and stride. Anything but 3/1 selects conv2d's **generic KxK branch** — taps stay in the weight `input_buffer`, the line buffer is split by COLUMN PHASE so every tap stays a unit-stride vector load at S=2, and it uses its own `CONV_VEC_GEN=32`. The two 3x3 branches are byte-for-byte as shipped at `CONV_VEC=16`. Both reach BOTH toolchains |
| `CROP_ROWS` / `CROP_COLS` | derived | `PATCH x CONV_STRIDE` — the ROI CROP, **not** the feature map. `roi_crop` takes it at RUNTIME, so a 128x128 crop into a 64x64 map needs no PL rebuild; its BRAM scratch caps at 128 and `make check_geometry` plus a host `static_assert` both refuse an overrun |
| `WEIGHT_BANK` | `l1resnet` | Donor bank for `make weights`. `l1resnet` = resnet18 conv1 7x7/2, 64 filters PCA'd to `N_CHANNELS` via `scripts/l1_banks.py` — the SAME function the offline screen scored, so the board arm and the screen share a bank rather than two spellings of one. `mobilenet` = mobilenet_v3_small conv1 3x3. A RUNTIME data file: switching banks within one geometry is `make weights` plus a file copy, no re-synthesis |
| `ACC_BOUND` | derived from `WEIGHT_BANK` | Which **exact worst-case** accumulator bound sizes `out_shift`. `loose` = `n_in*K^2*127^2`, tap count only; `l1` = `127*Σ\|w_int8\|` per channel, from the actual taps and attained at `x = 127·sign(w)`. **Both are exact, so neither can rail** — `l1` is simply ~2 bits tighter, which the 147-tap bank needs (without it `F_ch` measured 0.13% of int16 on hardware against the 3x3 bank's 12.3%). Tied to the bank so neither can be picked by accident; an explicit `ACC_BOUND=` still overrides. **Reaches NEITHER toolchain** — it changes only the weights DATA — so no flagstamp sees it, and `calib_build.sh` re-derives both bounds from the file's own taps instead |
| `CONV_IN_CH` | `3` | conv2d input planes. 1 = BT.601 luminance, 3 = RGB. Picks the **weight-buffer layout**, so it drives `AIE_FLAGS`, `GCC_FLAGS` and `ROI_IN_CH` from this one variable. Both arms build and link; see [`rgb.md`](rgb.md) |
| `CONV2D_STACK` | `2048` | conv2d's AIE stack, bytes. Applied **only at `CONV_IN_CH=3`**, where the 27-tap chain needs 1344 against the 1024 default and the mapper otherwise refuses to emit a `libadf.a` |
| `BIAS_SCALE` | `roi` | `bias_acc` input scale for `make weights`. **Default changed 2026-08-23**; `127` restores the pre-correction weights. See "The bias_acc correction" |
| `FRAME_RGB_MODE` | `1` | Synthetic scene colour at `CONV_IN_CH=3`. 1 = per-plane tint; 0 = replicate luma — the COLOUR-FREE CONTROL, which reproduced grayscale bit-for-bit on hardware. Inert at `CONV_IN_CH=1`; a real frame source ignores it. Host-only |
| `FRAME_SOURCE` | `synth` | `synth` = the generated scene (unchanged, and `vot_source.cpp` is not even linked); `vot` = frames memcpy'd from a converted VOT blob, geometry and init box from its manifest. At `vot` the scene generator, `TRAJECTORY`, `OCCLUDE_MASK`, `BG_PAN` and `FRAME_NOISE` are all inert, and `ITER_CNT` is ignored — the run length is the job's. `vot` + `CONV_IN_CH=3` needs the `.luma` sidecar, which the converter emits and the host streams or stages; this is the SHIPPING combination. Host-only. See `docs/thesis/evidence/phase2.md` |
| `RESET_MUTANT` | `0` | Deliberately breaks ONE item of `run_reset()` so the multi-start determinism test's ability to FAIL is demonstrated, not assumed: `1` mean_prev, `2` filter_bo, `3` g_filter, `4` coast, `5` scale reconfigure. Non-zero prints a banner and invalidates the run's tracking output. See `docs/thesis/evidence/phase3.md`. Host-only |
| `VOT_DATA_DIR` / `VOT_RESULTS_DIR` / `VOT_SEQUENCE` / `VOT_JOB` | `/mnt/vot` / `/mnt/vot-results` / `car1` / `0` | Compiled-in defaults, each overridable on the board's command line (`--vot-data`, `--vot-results`, `--vot-seq`, `--vot-job`, `--vot-jobs all|N,M,...`, `--vot-max-frames` which then REFUSES to write the trajectory). **Repeating a job index in `--vot-jobs` is the determinism test** — its two trajectories must come back byte-identical. An unrecognised argument is fatal. Host-only |
| `VOT_RESIDENT_MAX_MB` / `VOT_STREAM_RING` | `700` / `8` | Blob+sidecar size above which a sequence STREAMS from the NFS mount through a prefetched ring, and the ring depth. Exists because usable heap is ~0.9-1.2 GB, not 12 GB: five RGB sequences died on `std::bad_alloc` while 57 completed. Ring < 2 is REFUSED, not clamped. `--vot-stream auto|always|never`; `always` is the MODE-EQUIVALENCE TEST — streaming changes no arithmetic, so digests must be IDENTICAL both ways. Host-only |
| `SCENE_VERIFY` | `0` | Re-colourise the whole frame each push and abort on a mismatch. O(frame)/frame — for a short `MODE=bringup` run only (`calib_build.sh` refuses otherwise). It was a bare `#ifndef` before 2026-08-24, so `SCENE_VERIFY=1` silently built it DISABLED. Host-only |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `FILTER_MASK` | `0` | Spatial reliability: the projection `h ← m⊙h` on the published `H`, i.e. CSR-DCF's item. The window is FORCED — only the periodic Hann has an exactly sparse spectrum — so there is no width knob, and once forced the constants collapse to `H ← D_row(D_col(H))/16` with `D(X)[i]=2X[i]−X[i−1]−X[i+1]` circular: **8 complex adds/bin, no multiplies**, verified against an exact FFT to 8e-16. Applied in BOTH publish paths (the fused one AND `filter_quantize_q15`, which is frame 0's) and BEFORE the max-\|H\| scan, since it moves the Q1.15 scale. Host-only. **SWEPT AND ACCEPTED 2026-08-31: EAO 0.1629 → 0.1740 (+0.0110), R +0.3417 → 0.3608, 16.6% more frames tracked; A −0.0187 pooled but **+0.0179 on the common survived prefix**, i.e. the accuracy loss is a selection effect. Offline over-predicted the gain 3× (dR +0.0601 vs +0.0192). `docs/thesis/evidence/spatial_mask.md`, claim R-10.** **The default is STILL `0`, and the reason has changed**: the gain is carried by 3 of 62 sequences, drop-top-3 FLIPS it to −0.0030, and P(dR<=0)=0.22 — not separable from a null. The `PSR_GATE_MIN` re-tune it was thought to force was REFUTED on hardware 2026-09-01 (EAO null), so that is no longer a reason to hold it either. It has never been measured on top of the Layer-1 arm. **`cmp` cannot check its inertness** — see `evidence/arm_mask.md` §3 |
| `FILTER_MASK_STAT` | `0` | Logs `mask_ebox`, the fraction of `Σ\|h\|²` inside a centred target-sized box, as a trailing `track.csv` column. THE mechanism check for the above, and independent of it so the baseline is measurable with the same instrument. **51.6%/54.9% are AT-INIT values and the fraction RISES as the filter converges** (car1 0.514→0.741 unmasked); comparing a run's mean against them would confirm the mechanism on an unmasked arm. **`-1` is NOT MEASURED, and at the shipped schedule that is 94.6% of rows — do not average as zero and do not read it as a hold rate.** Sampled, not per-frame: the statistic is an inverse FFT per channel (30.1 ms/call on the A72, more than a whole frame), so see `FILTER_MASK_STAT_WARM`/`_EVERY`. Host-only |
| `FILTER_MASK_STAT_WARM` / `FILTER_MASK_STAT_EVERY` | `5` / `20` | Sampling schedule for `mask_ebox`: the first WARM frames of each run plus every EVERY-th after. **Matched to `vot_mask_stat.py`'s `PROFILE_FRAMES` (1, 5, 20, 40)** — move either and that reader's per-frame columns silently thin out. Host-only |
| `PSR_GATE_MIN` | `5.0` | +0.0134 R against 7.0 on the full benchmark. **Its worth depends on the PSR scale**, so anything moving PSR re-opens it. Bolme §3.5. Below it the host HOLDS position and skips `filter_update` + `publish_filter`. `0` disables the threshold test only (structural vetoes remain). Host-only |
| `TARGET_H` / `TARGET_W` | `64` | Target box size, frame px. Host-only |
| `TARGET_PADDING` | `2` | Settled at 2.0 three ways; 1.5 and 3.0 both measured worse. `roi = box × padding`. At 64/2 the ROI is 128 ⇒ resample is 1:1. Host-only |
| `MOSSE_SIGMA` / `SIGMA_FROM_TARGET` | `2.0` / `0` | In BINS, so what matters is `sigma/target`, and 1/16 (DSST's `target/16`) is the measured optimum: `2.0` at 64x64 and `4.0` at 128x128 are the SAME operating point. `SIGMA_FROM_TARGET=1` applies the rule instead of the literal. **The interior is CLOSED** — a 22-cell grid puts 3, 5, 6 and 8 all below it |
| `MOSSE_ETA` | `0.05` | +0.0218 R, +8.5% EAO against 0.125. Not monotone — 0.025 is much worse, so it is a shallow optimum. Translation filter learning rate. Host-only |
| `SCALE_N` / `SCALE_STEP` | `33` / `1.04` | DSST scale levels; `SCALE_N=1` disables the scale filter. **1.04 beats DSST §6.1's 1.02 on hardware** (IoU 0.807 → 0.917) |
| `SCALE_ETA` | `0.025` | Scale filter learning rate (deliberately ≠ `MOSSE_ETA`) |
| `SCALE_CONF_MIN` | `2.0` | Scale gate on `conf`. A veto HOLDS the box and SKIPS `scale_update()`. `0` disables the threshold test only. Host-only |
| `HOLD_COAST` / `COAST_DECAY` | `0` / `0.5` | `1` = a held frame moves the search window at the last measured velocity, decayed each held frame (drift bounded by 2v); `0` = the freeze. **Flipped to 1 and reverted the same day**: the same 54 trajectory pairs win on mean IoU (0.2709 → 0.3005) and LOSE on the toolkit's metric (A 0.638 → 0.616, R 0.309 → 0.288, EAO 0.208 → 0.194). AR is the metric of record. See `docs/thesis/evidence/metric_ar_vs_iou.md`. Host-only |
| `SCALE_MAX_STEP` | `2` | Largest `|idx|` ONE frame may move the box — a RATE limit, where `MIN_REL`/`MAX_REL` are a drift bound. `0` disables. **`1` was measured and rejected**: `scale_sim` parks the smooth arm 123 of 200 frames and ends 28.0% wrong. Added after `car1` f490 inflated the box 1.42× while 227 px off target. Host-only |
| `SCALE_MIN_REL` / `SCALE_MAX_REL` | `0.5` / `2.0` | Absolute drift bounds vs initial box size. Must still admit `SCALE_TRAJ_AMP` (0.70×..1.30×) |
| `OCCLUDE_MASK` | `0` | Bitmask over frame index: bit *f* ⇒ frame *f* occluded. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test |
| `OCCLUDE_SQUARE` / `OCCLUDE_START` | `8` / `30` | `OCCLUDE_START` is a warm-up: 30 ≈ 4 time constants at `MOSSE_ETA=0.125`. The scale filter needs ~120 frames to settle |
| `TRAJECTORY` / `TRAJ_AMP_R` / `TRAJ_AMP_C` / `TRAJ_PERIOD` | `0` / `180.0` / `180.0` / `120.0` | `TRAJECTORY=1` puts the target on a closed elliptical path (absolute ground truth); the other three are inert until it does |
| `SCALE_TRAJ` / `SCALE_TRAJ_AMP` / `SCALE_TRAJ_PERIOD` | `0` / `0.30` / `200.0` | `SCALE_TRAJ=1` adds a sinusoidal size envelope; the amplitude must stay inside `SCALE_MIN_REL`/`SCALE_MAX_REL` |
| `FRAME_TEXTURE` | `1` | Band-limited background instead of a flat fill |
| `FRAME_NOISE` | `2` | Per-frame sensor noise, PEAK amplitude in LSB, over the ROI. Does **not** fix background lock |
| `BG_PAN` / `BG_PAN_R` / `BG_PAN_C` | `1` / `31` / `47` | Camera pan over the cached background, px/frame. Decorrelates the background 6.6× (swept with `scripts/bg_pan_sweep.py`) but **did not fix tracking** — see the training-target trap. Host-only |
| `PROGRESS_EVERY` | `1` | Frames between LEVEL-0 progress lines. `1` = byte-identical to every pre-2026-08-25 run (proven by an ELF `cmp`). Thins the marker, never silences it: frame 0 and the last frame always print, because a run missing its final line looks exactly like a run that hung. **Only has effect at `VERBOSITY=0`.** Worth 0.27 ms, not the 4.0 predicted. Host-only |
| `CSV_FLUSH_EVERY` | `1` | Rows between `track.csv` flushes. Per-row flushing survived a power cut that really did take out a run. A **railed** row flushes regardless of N; a gate veto deliberately does not. Worth 0.00 ms from tmpfs. Host-only |
| `VERBOSITY` | `1` | `0` = one compact line/frame (~45 B); `1` = per-frame block, roi_crop/DMA tables on first+last frame only; `2` = everything. Anomalies print at every level. **No longer a diagnostics trade-off** — since 2026-08-24 `track.csv` carries `rails`/`accum_max`, so a `VERBOSITY=0` run is a full budget verdict AND an FPS measurement. Host-only |
| `DUMP_BUFFERS` | `1` | Per-frame binary dumps. **1216 KB/frame, ~2 s/frame**. Set `0` for any run measuring tracking or FPS |
| `CSV_LOG` | `1` | One row/frame to `track.csv` (`track_<sequence>.csv` at `FRAME_SOURCE=vot`, since one sweep is one invocation per sequence and the file opens `"w"`; the arm is separated by `--vot-results` instead) — gate verdict, both PSRs, peak, displacement, `resp00_over_peak`, both boxes, IoU, centre error, scale fields, and (2026-08-24+) `rails,accum_max,fch0_max,h_max`. ~60 B/frame |

**THE DEFAULTS ARE THE SHIPPING CONFIGURATION as of 2026-09-02** (`rgb_l1relu`, EAO 0.1960).
A bare `make application FRAME_SOURCE=vot` reproduces the benchmark arm's `app.flagstamp`
exactly, and the default `aie.flagstamp` matches the flashed `a.xclbin`. Eight defaults moved
that day, all of them the Layer-1 arm: geometry `128->64`, `N_CHANNELS 16->32`, budget
`4-4-4 -> 3-3-3`, `CONV_KSIZE 3->7`, `CONV_STRIDE 1->2`, `CONV_RELU 0->1`,
`WEIGHT_BANK mobilenet->l1resnet`, and `ACC_BOUND` now DERIVED from the bank
(`l1resnet`→`l1`, `mobilenet`→`loose`) so neither can be selected by accident.
Four moved on 2026-08-28 and still stand: `CONV_IN_CH` 1->3, `H_SHIFT` 11->15,
`MOSSE_ETA` 0.125->0.05, `PSR_GATE_MIN` 7.0->5.0.
**The 128x128 3x3 arms and grayscale are all still supported** — pass the geometry and the bank
back explicitly; `CONV_IN_CH=1` is what the aiesim scenarios `s6`/`s7` need.

**Two knobs that are NOT independent, and that a table cannot show.** `MOSSE_SIGMA` is in BINS,
so any geometry change silently moves `sigma/target` off its 1/16 optimum — 2.0 at 64x64 and 4.0
at 128x128 are the SAME operating point. And the FFT/IFFT budget follows the POINT SIZE, so a
geometry change needs its own 200-frame calibration before any of its numbers mean anything.

Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.
