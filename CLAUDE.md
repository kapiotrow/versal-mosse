# CLAUDE.md

Guidance for Claude Code working in this repo. MOSSE correlation-filter tracker with CNN
features on Versal VEK280, extending the AIE 2D-FFT tutorial (XD073) with a full
object-tracking pipeline. Papers in `docs/papers/` (Bolme MOSSE, Danelljan DSST); section
numbers below refer to them.

**This file is the operational half only.** Every derivation, measurement and cost behind a
rule stated here lives in `docs/engineering/` — see [`docs/engineering/README.md`](docs/engineering/README.md)
for the map. Nothing was deleted when it moved there (2026-08-31); if a line below reads as a
bare assertion, the file it links to says how it was paid for.

## Thesis scaffold

Findings are indexed in `docs/thesis/claims.md` — one row per question answered, with a
verdict, an evidence note and a run directory. **Every number the thesis quotes lives in
`docs/thesis/results/*.csv`, not in prose**; evidence notes are in `docs/thesis/evidence/`.
When a sweep finishes: append a row to `results/arms.csv`, update `claims.md`, write the note
from `evidence/TEMPLATE.md`.

- **`@thesis` tags bind code to thesis sections**: `// @thesis <label> | <claim>[,<claim>] | <one line>`,
  where `<label>` is the thesis's own `\label` and `<claim>` an id from `claims.md`.
  `make code-map` (`scripts/thesis_index.py`) regenerates `docs/thesis/code_map.md` and
  VALIDATES both fields (unknown claim id, undefined label, doubly-defined label). Tag a site a
  chapter will describe; do not tag every function.
- **Documentation in this repo is English-only**, tags included — the thesis is Polish and
  `docs/thesis/glossary.md` is the bridge (table A: terminology already fixed in
  `teoria.tex`/`przeglad.tex`, reuse verbatim; table B: proposed implementation vocabulary;
  table D: four pairs that must not share a Polish word).
- **Every kernel and graph header carries a `COST` block** — eight of them (`conv2d_kernel.cpp`,
  `cmul_accum_kernel.cpp`, `fft_graph.h`, `ifft_graph.h`, `mosse_graph.h`, `roi_crop.cpp`,
  `roi_crop.h`, `camera_capture.cpp`) — with fixed fields (compute, vectorization, pipelining,
  tile/BRAM memory, interface, caveat), each citing its CSV or log, so `sec:wydajnoscZasoby` can
  be written from the headers. **The caveat field is load-bearing**: AIE figures are
  compiler-SCHEDULED cycles, the FFT chain's 2.2 ms is INFERRED, and `roi_crop`'s are the only
  MEASURED cycle counts (hw_emu VCD — hw_emu wall time is meaningless, its simulated PL cycles
  are RTL-accurate). AIE compute is not frame time: the frame is 84% CPU-bound.
- **The thesis's tables are generated.** `make thesis-tables` turns `docs/thesis/results/*.csv`
  into booktabs bodies in `docs/thesis/tables/` (gitignored — the CSV is the source). The chapter
  `\input`s one inside its own float and keeps caption/label/placement; Polish row labels live in
  the CSV's `*_pl` columns. Copying to the thesis repo is MANUAL; `scripts/csv2tex.py --overleaf
  [PATH]` opts in and only writes files. `--check` exits 1 when a table is stale.
- **A measured number in a comment must say where it came from.** `make check-docs`
  (`scripts/check_doc_numbers.py`) reports a decimal duplicating a `results/*.csv` value while
  citing nothing (drift risk) and any frame-time/tracking figure no CSV records. A claim id, a
  `results/`/`docs/` path or the `.log` it was read off all count. **The number itself usually
  stays** — deleting "8.71 -> 1.88 ms, 4.6x" from the hypot-fix comment makes the code worse.
  Both classes at zero as of 2026-08-30.

## Environment setup

```bash
source setup_env.sh
```

Sets `PLATFORM_REPO_PATHS`, `XILINX_VITIS`, `COMMON_IMAGE_VERSAL`, `PLATFORM`
(`xilinx_vek280_base_202520_1.xpfm`), and `DSPLIB_VITIS` — the Vitis Libraries **root**, not
the `dsp` subdirectory (the Makefile appends `/dsp`).

## Build parameters

**THE DEFAULTS ARE THE SHIPPING CONFIGURATION as of 2026-08-28.** A bare
`make application FRAME_SOURCE=vot` reproduces the benchmark arm's `app.flagstamp` exactly, and
the default `aie.flagstamp` matches the flashed `a.xclbin`. Four defaults moved that day:
`CONV_IN_CH` 1->3, `H_SHIFT` 11->15, `MOSSE_ETA` 0.125->0.05, `PSR_GATE_MIN` 7.0->5.0.
**Grayscale is still fully supported** — pass `CONV_IN_CH=1` (what aiesim `s6`/`s7` need).
Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

Digest below; **the reasoning for every knob is in
[`docs/engineering/build_params.md`](docs/engineering/build_params.md)** — read that entry
before moving one. Everything is host-only (an scp, not a card swap) EXCEPT `H_SHIFT`,
`CONV_IN_CH`, `CONV2D_STACK`, the FFT/window knobs and the geometry, which reach `AIE_FLAGS`.

| Parameter | Default | One line |
|---|---|---|
| `TARGET` | `hw_emu` | `hw_emu` or `hw` |
| `PATCH_ROWS`/`PATCH_COLS` | `128` | powers of 2 (AIE FFT constraint) |
| `N_CHANNELS` | `16` | conv feature channels |
| `FFT_2D_DT` | `0` | 0=cint16, 1=cfloat |
| `ITER_CNT` | `1` | frames; **needs ≥2** — frame 0 initialises the filter |
| `PL_FREQ` | `312.5` | MHz; platform also offers 625/156.25/100/78.125 |
| `H_SHIFT` | `15` | filter-product shift, deliberately OVER-shifted (`rails=0` over 101,564 frames). 13 is the tight RGB value, 12 rails, gray's arm is 14. **The one non-host-only knob here** |
| `FFT_SHIFT`/`IFFT_ROW_SHIFT`/`IFFT_COL_SHIFT` | `4`/`4`/`4` | validated on hardware; do not change without a ≥20-frame hw run |
| `FFT_ROW_WS`/`FFT_COL_WS` | `64`/`8` | rows/cols per FFT invocation. `FFT_COL_WS=32` is a 9.6 ms LOSS |
| `MEMTILE_TRANSPOSE` | `1` | transposes in AIE-ML memory tiles; a one-sided flag is a board deadlock |
| `ROI_CROP_PIPELINE` | `1` | launch channel k+1's crop before polling k |
| `CMUL_SPLIT_ACCUM` | `1` | own port for `accum_prev`. **`make aiesim` needs `0`** |
| `CMUL_ACCUM_MEMTILE` | `0` | tried, 0.36 ms loss; code kept behind the flag |
| `TAIL_PARALLEL` | `1` | filter update on core 1 ∥ scale filter on core 0; needs `-pthread` |
| `ROI_CROP_USER_MANAGED` | `1` | roi_crop via `xrt::ip`. **20.6× on frame rate** — KDS completion costs 503 ms/launch |
| `CONTROL_CU_RUNS` | `8` on `hw` | camera_capture launches: the within-run KDS control |
| `CONV2D_MODE` | `0` | 0=real 3×3 conv, 1=echo, 2=synthesize. **Check before every expensive run** |
| `CONV_VECTORIZE`/`CMUL_VECTORIZE` | `1` | bit-identical to scalar; 0 for bisection |
| `CONV_RELU` | `0` | off. **Bank-specific**: refuted on the shipping 3x3/16 bank (dR −0.0332), but it BEATS its own linear twin on a Layer-1 bank (`evidence/layer1_features.md`) |
| `CONV_IN_CH` | `3` | 1=BT.601 luma, 3=RGB. Picks the **weight-buffer layout**, so it drives `AIE_FLAGS`/`GCC_FLAGS`/`ROI_IN_CH` |
| `CONV2D_STACK` | `2048` | conv2d AIE stack; applied only at `CONV_IN_CH=3` (27 taps need 1344 > 1024 default) |
| `BIAS_SCALE` | `roi` | `bias_acc` input scale for `make weights`; `127` restores pre-correction weights |
| `FRAME_RGB_MODE` | `1` | 1=per-plane tint, 0=replicate luma (the COLOUR-FREE CONTROL) |
| `FRAME_SOURCE` | `synth` | `vot` = frames from a converted VOT blob; ignores `ITER_CNT`, the scene generator and its knobs. `vot`+RGB needs the `.luma` sidecar — the SHIPPING combination |
| `RESET_MUTANT` | `0` | deliberately breaks one item of `run_reset()` so the determinism test's ability to FAIL is demonstrated |
| `VOT_DATA_DIR`/`VOT_RESULTS_DIR`/`VOT_SEQUENCE`/`VOT_JOB` | `/mnt/vot`/`/mnt/vot-results`/`car1`/`0` | overridable on the board's command line; **repeating a job in `--vot-jobs` is the determinism test** |
| `VOT_RESIDENT_MAX_MB`/`VOT_STREAM_RING` | `700`/`8` | stream from NFS above this size; usable heap is ~0.9-1.2 GB, not 12 GB. `--vot-stream always` is the MODE-EQUIVALENCE TEST |
| `SCENE_VERIFY` | `0` | re-colourise and abort on mismatch; O(frame)/frame, `MODE=bringup` only |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `FILTER_MASK` | `0` | spatial reliability `h ← m⊙h`; window FORCED periodic Hann ⇒ 8 complex adds/bin, no multiplies. **Swept 2026-08-31 (EAO +0.0110) but default still 0** — not separable from a null (3 of 62 sequences) |
| `FILTER_MASK_STAT` | `0` | logs `mask_ebox` to `track.csv`; **`-1` is NOT MEASURED** (94.6% of rows). Sampled — an inverse FFT per channel is 30.1 ms |
| `FILTER_MASK_STAT_WARM`/`_EVERY` | `5`/`20` | **matched to `vot_mask_stat.py`'s `PROFILE_FRAMES`** |
| `PSR_GATE_MIN` | `5.0` | Bolme §3.5. Below it the host HOLDS and skips the update. **Worth is conditional on the PSR scale** |
| `TARGET_H`/`TARGET_W` | `64` | target box, frame px |
| `TARGET_PADDING` | `2` | settled three ways; 1.5 and 3.0 both worse. At 64/2 the resample is 1:1 |
| `MOSSE_SIGMA`/`SIGMA_FROM_TARGET` | `2.0`/`0` | **BINS, so what matters is sigma/target; 1/16 = DSST's `target/16` is the measured optimum. `4.0` at 128x128 is the BEST ARM ON RECORD** (EAO 0.1931) |
| `MOSSE_ETA` | `0.05` | +0.0218 R vs 0.125; not monotone (0.025 much worse) |
| `SCALE_N`/`SCALE_STEP` | `33`/`1.04` | `SCALE_N=1` disables. **1.04 beats DSST §6.1's 1.02 on hardware** |
| `SCALE_ETA` | `0.025` | deliberately ≠ `MOSSE_ETA` |
| `SCALE_CONF_MIN` | `2.0` | veto HOLDS the box and SKIPS `scale_update()` |
| `HOLD_COAST`/`COAST_DECAY` | `0`/`0.5` | coast a held frame at last velocity. **Flipped to 1 and reverted**: wins mean IoU, loses AR |
| `SCALE_MAX_STEP` | `2` | rate limit on `|idx|`/frame. **`1` measured and rejected** |
| `SCALE_MIN_REL`/`SCALE_MAX_REL` | `0.5`/`2.0` | drift bounds; must admit `SCALE_TRAJ_AMP` |
| `OCCLUDE_MASK`/`OCCLUDE_SQUARE`/`OCCLUDE_START` | `0`/`8`/`30` | bit *f* ⇒ frame *f* occluded; `OCCLUDE_START` is a warm-up |
| `TRAJECTORY`/`TRAJ_AMP_R,C`/`TRAJ_PERIOD` | `0` | closed elliptical path (absolute ground truth) |
| `SCALE_TRAJ`/`SCALE_TRAJ_AMP`/`SCALE_TRAJ_PERIOD` | `0` | sinusoidal size envelope |
| `FRAME_TEXTURE`/`FRAME_NOISE` | — /`2` | band-limited background; per-frame noise **does not** fix background lock |
| `BG_PAN`/`BG_PAN_R`/`BG_PAN_C` | `1`/`31`/`47` | pan px/frame; decorrelates background 6.6× but **did not fix tracking** |
| `PROGRESS_EVERY` | `1` | thins the level-0 line, never silences it. **Only has effect at `VERBOSITY=0`** |
| `CSV_FLUSH_EVERY` | `1` | rows between `track.csv` flushes; a railed row flushes regardless |
| `VERBOSITY` | `1` | 0=one line/frame, 1=per-frame block, 2=everything. Anomalies print at every level |
| `DUMP_BUFFERS` | `1` | **1216 KB/frame, ~2 s/frame** — set `0` for any tracking or FPS run |
| `CSV_LOG` | `1` | one row/frame to `track.csv` (`track_<sequence>.csv` at `FRAME_SOURCE=vot`), ~60 B/frame |

### Shift budget — SETTLED: 4-4-4, `H_SHIFT` 14 (gray) / 15 (RGB)

Closed on hardware (FFT budget 2026-08-24, `H_SHIFT` 2026-08-27).
[`docs/engineering/shift_budget.md`](docs/engineering/shift_budget.md) has the retired points,
the four rules it cost, and the uncensored real-video distribution. The short version:

- The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so
  weight moves freely between passes. **Every fix has been `H_SHIFT`** — the only knob upstream
  of both the accumulator and the response.
- **The response GROWS as the filter converges**: a budget validated at `ITER_CNT=2` is not
  validated, and `IFFT_ROW_SHIFT=0` is unsafe at ch16. Size against the TAIL, not the typical
  frame. **Twice an offline model set this budget and hardware overturned it.**
- Acceptance criterion: `rails=0` **plus** bit-identical tracking **plus** PSR not moving, with
  `F_ch`/`H(q15)` digit-for-digit unchanged.
- **`accum_max = 46340` IS NOT OVERSHOOT** (= 32767·√2, a magnitude against a per-component
  rail); readings are CENSORED at the rail; rails do NOT correlate with tracking loss
  (corr −0.025). **`rails` is the only saturation instrument.**
- `runs/.last_cfg` is **stale and not authoritative**; `build/hw/.../aie.flagstamp` is, and
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

**PLIO (1 port).** `PatchIn` — roi_crop → conv2d, **32-bit** (one int32 = 4 packed int8
pixels). `mosse_graph.h`'s `input_plio::create("PatchIn", ...)` uses `plio_32_bits`, not
128-bit: a 128-bit PLIO delivered one beat per `readincr`, starving the kernel. The name must
match between `mosse_graph.h` and `mosse_x1.cfg`
(`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

**GMIO ports.** `gmio_weights` (DDR→AIE, conv2d INT8 weights per channel); `gmio_fft_col_out`
(AIE→DDR, broadcast tap: F_ch for the PS filter update); `gmio_cmul_in` (H_ch* per chunk);
`gmio_accum_in` (prev Σ per chunk, `CMUL_SPLIT_ACCUM=1`); `gmio_accum_out`; `gmio_ifft_row_in`;
`gmio_response`. The four transpose GMIOs were deleted by `MEMTILE_TRANSPOSE=1`.

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

### Key design decisions

- **AIE-centric**: all FFT/IFFT/conv/cmul on AIE; PL is only camera_capture + roi_crop; APU
  orchestrates via GMIO DDR round-trips. **Serial channel processing** — one FFT2D and one
  IFFT2D reused across all channels, minimal PL/PLIO count at the cost of throughput.
- **Accumulator in DDR** (128×128 cint16 = 64 KB). On-tile does not work as usually stated: it
  is a read-modify-write by ONE kernel across invocations, so as a `shared_buffer` it is a graph
  CYCLE with a required delay of `CMUL_N_CHUNKS` (16) invocations, not 1; keeping it in cmul's
  tile needs all 16 chunk accumulators resident (the whole 64 KB tile) on top of port buffers,
  and AIE-ML kernels cannot address a memory tile as random-access scratch.
- **Filter init/update on PS, with no FFT library.** `mosse_filter.{h,cpp}` implements Bolme
  eq. 10–12 with a *shared* denominator (DSST form — one reciprocal map per frame, better
  conditioned). `F_ch` arrives already transformed via `gmio_fft_col_out` and `G` has a closed
  form, so KissFFT was never needed. AIE would be the worst home — 2 MB of filter state does not
  fit on-tile.
- **The filter update runs AFTER peak detection** — updating first leaks the current frame into
  its own detection.
- **`mosse_filter.{h,cpp}` includes no XRT/ADF header**, so `make test_host` compiles it with
  system g++ against a NumPy golden in seconds; the alternative for a sign error is an
  hours-long hw_emu frame.
- **Preprocessing is split PL / AIE / APU**: intensity-domain steps in `roi_crop` (Stage A);
  feature-map mean removal in `conv2d` from the *previous* frame's mean (Stage B1 — a full
  channel is 64 KB, too big for a tile); a 9-bin frequency-domain correction on the APU
  (Stage B2); per-channel energy normalization folded into `H_ch*` (Stage B3). <2% added
  arithmetic, no new AIE tiles.
- **Periodic Hann, not symmetric.** `hanning_*.h` uses `sin²(πi/N)`; its 2D DFT has exactly 9
  non-zero bins, which is what makes B2's correction exact. Measured DC/worst-leaked-bin at
  N=128 in Q1.15: periodic 2.2e5, symmetric 373. Do not "fix" this back.
- **Grayscale collapse uses ITU-R BT.601 luminance, deliberately NOT Danelljan's unweighted
  sum**, which annihilates the four colour-opponent channels (0, 2, 9, 10) to 2.5-5% of their
  norm before int8 quantization amplifies the residue to full scale (the 11 achromatic channels
  agree between conventions to cos > 0.99). Luminance is the better of the two conventions and
  still discards real information — which is why `CONV_IN_CH=3` exists. See `docs/engineering/rgb.md`.
- **Scale estimation is DSST's 1-D filter, not multi-resolution search** (DSST Table 1 beats
  exhaustive on both axes, and exhaustive would push ±30% patches through the whole chain every
  frame, moving `|F|` and therefore the shift budget).
- **`g_frame_host` (2 MB heap) is the authority for the scene; `frame_bo` is a copy.** BO writes
  run at 3470 MB/s against reads at 696, so pushing costs 0.405 ms where pulling would cost
  ~2.9 ms. Anything writing `frame_bo` directly (`rc_control_cu_probe`'s zero-fill) must run
  before the first push.

**Resources: the design uses 2% of the AIE array** (6 of 304 cores, 1 of 76 memory tiles, 0.8%
BRAM18, 3.4% DSP, 1.5% LUT). Check any "we can't afford it on AIE" claim against that — the
binding constraints have always been tile memory (64 KB/tile) and host DMA orchestration, never
core count, and `runtime<ratio>` is not utilization. **Usable heap is ~0.9-1.2 GB, NOT the
part's 12 GB** (Linux maps 2 GB, 512 MB of it CMA); it cost 5 of 62 sequences on the RGB VOT arm
until the streaming reader. Full tables, per-frame AIE compute and the memory forensics:
[`docs/engineering/performance.md`](docs/engineering/performance.md),
`docs/thesis/evidence/TODO_board_memory.md`.

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
| **+ 64x64 feature map (3-3-3, `H_SHIFT=15`) — CONFIRMED 2026-09-01** | **0.5336** | **0.3873** | **0.1849** | `0901_res64` |
| ...+ `PSR_GATE_MIN=3.5` (rescaled to the new PSR) — REJECTED, EAO null | 0.5308 | 0.3936 | 0.1849 | `0901_gate35` |
| **+ `MOSSE_SIGMA=4.0` at 128x128 (HOST-ONLY) — BEST ON RECORD 2026-09-01** | **0.5133** | **0.4095** | **0.1931** | `0901_sigma4` |

`PSR_GATE_MIN=0` is a null (13-seq probe, `g0partial`). The padding arm gained R and LOST EAO —
**EAO is the arbiter for an A/R trade**. Every arm after the first two is HOST-ONLY.
**The gate's worth is NOT conditional on the PSR scale — REFUTED on hardware 2026-09-01**
(`proposed_build_res64.md` sec.23): a threshold rescaled to the 64x64 PSR returned an EAO null,
because 95.9% of vetoes fire after the run is already lost. The gate is a small, mostly post-hoc
veto and its threshold is NOT worth re-tuning per geometry. **The offline gate estimate
(+0.056 at 64x64) does not transfer (+0.0063).**

The full chain runs on hardware at 128x128 ch16 on the real conv path. Heap, not tracking, was
the last blocker; `vot::StreamBlob` closed it, proven by identical run-state digests both ways.

**Best hardware FPS: 26.29 ms/frame = 38.04 FPS** (`runs/run_0821_1725.log`, gray, synthetic,
UART console). RGB is 28.58 ms. `car1` over ssh is **24.43 ms = 40.9 FPS** and is NOT comparable
to either — the UART alone was 3.79 ms. **Quote FPS only from a serial-console run.**
The frame is **84% CPU-BOUND**; perfect two-core use floors at ≈12-15 ms, 65-80 FPS.
Optimisation history and the frame breakdown: [`docs/engineering/performance.md`](docs/engineering/performance.md).

Where this sits against the 41 published VOT-STb2022 trackers, and the attributed loss
mechanism: [`docs/engineering/baselines.md`](docs/engineering/baselines.md). The split is sharp
— **A = 0.510 is inside the classical-DCF band (0.009 under CSRDCF); R = 0.342 is below every
one of the 41.** The gate is the AFTERMATH of a loss, not its cause: 95.8% of vetoes land after
the run is already at IoU ≤ 0.1, and in the 5 frames before a first loss the verdict is ACCEPT
82.0% at median PSR 18.83. **It walks off the target confidently — do not spend a sweep
relaxing the gate, and do not look for the fault in localisation.**

## What to try next

Ranked, with the evidence for each rank, in
[`docs/engineering/roadmap.md`](docs/engineering/roadmap.md). Headlines:

- **NEXT BUILD — Layer-1 features: 7x7 stride 2, 32ch, `CONV_RELU=1`, 64x64 map, sigma 2.**
  Pre-registered in [`evidence/proposed_build_l1relu.md`](docs/thesis/evidence/proposed_build_l1relu.md).
  The only screened cell that beats its control AND is not slower than today (~20-22 ms).
  **Offline +0.0383 R is BORDERLINE — P(dR<=0)=0.041 where the two arms that transferred were
  0.000 — so it must not be sold as a robustness win.** The reason to build it is that
  `CONV_RELU=0` makes the CNN *provably redundant* (a one-hot bank ties the pretrained one), and
  this project's requirement is conv features. NOT host-only: rebuild, reflash, re-calibrate.
- **ROBUSTNESS.** (1) `MOSSE_SIGMA` interior at 128x128 — sigma 3 and 5 are host-only sweeps and
  the optimum is bracketed but not located. (2) A two-filter temporal ensemble — untested.
  (3) The spatial mask — EAO up, still not separable from a null. (4) Re-screen channel
  reliability on the sigma-4 operating point ONCE, and nothing more.
- **PERFORMANCE.** `scale extract` is now the head of the tail (16.6% of the frame, and it did
  not move when everything else fell 4x), then the 1080p frame-source path at 8.9%. Structural
  wins: software-pipeline the CHANNEL loop, more of the second core, NEON in `unpack_spectrum`,
  fewer/larger DMA transactions.
  **Retired, do not reopen**: `FFT_COL_WS` 8->32, `CMUL_ACCUM_MEMTILE` alone, Hermitian symmetry
  in the host filter, the accumulator as a `shared_buffer`, parallel-for inside
  `filter_update_quantize` as attempted.

**`MOSSE_SIGMA` is in BINS and the target spans `patch/padding` bins — what matters is
sigma/target, and 1/16 (DSST's `target/16`) is the measured optimum.** That one fact re-attributed
the 64x64 arm: it sat at 1/16 by accident, so it and `sigma4` are a MATCHED PAIR on width, and at
matched width the FINER map wins. res64's gain was the sigma it carried, not its resolution; it is
now the SPEED option (2.37x faster for -0.0082 EAO).

**Recovery AFTER a loss is worth NOTHING to R or EAO** — VOT terminates 10 frames after failure
and re-enters at the next anchor. That retires re-detection and search-window expansion outright,
however good they look on mean IoU. **Score any candidate on `vot analysis`, never on mean IoU**:
the two have ordered arms oppositely on identical trajectories, and the offline AR proxy has a
MEASURED resolution of only ~0.02 in R.

## Rules that were paid for

Full statements, with the runs that bought them, in
[`docs/engineering/measurement.md`](docs/engineering/measurement.md) and
[`docs/engineering/traps.md`](docs/engineering/traps.md). The ones worth carrying in the head:

**Measurement.** Instruments before changes; never move two magnitudes at once.
**NEVER SIZE A CHANGE FROM ONE MEMBER OF AN INTERLEAVED async/wait GROUP — SUM THE GROUP**
(the single most repeated measurement error here; it built a whole optimisation on an artifact).
Time `async` and `wait` separately. Measure the total and print the residual. Instrument both
candidate mechanisms in ONE run and let the log print the verdict. Two independent instruments
beat one instrument twice. Keep a known-good comparator on the old path. Write predictions down
before the run. Benchmark a host-side change on the host — the *ordering* has failed to transfer,
not just the magnitude. **A caveat that is not priced is a hope.**

**Scoring.** An offline R can be RAISED by degrading the filter — demonstrated with a deliberate
mutant at dR +0.0525 while tracking worse by every direct measure, so **never accept an arm on R
alone**. Accuracy is averaged over TRACKED frames, so score A on the **common survived prefix**
before calling a drop real. A multi-start CSV must be keyed on `(job, frame)` — a bare frame
index once under-reported rails 66×. Test an analysis tool against an OLD log first: a parser
that finds nothing looks exactly like a clean run.

**Build hygiene.** `runs/.last_cfg` is not authoritative — the flagstamps are, and
`scripts/calib_build.sh` checks them. Check `CONV2D_MODE`. A stale `Map_Report.csv` survives a
failed compile. **NEVER KILL A BOARD RUN MID-SEQUENCE** — the next process stalls and only a
reboot clears it; watch the trajectory count. A cross-implementation check is the only thing that
catches a contract mismatch (five green unit tests coexisted with a statistic reading 0.0000 for
a whole sweep). Launching over ssh changes the frame-time measurement. Verify a feature flag on a
build that can actually exercise it — `strings` can report a false absence.

**Metrics that cannot fail a broken tracker.** PSR is weak *in a specific direction* — a tracker
179 px off target, confidently locked to background, reported PSR 33. **IoU is the only metric in
the harness that can fail a confidently-wrong tracker; read `track.csv`, not the console.**
`err=0 px` is weak too, a centred test impulse cannot validate localisation, and `[diag] F_ch` is
CHANNEL 0 ONLY. Two different statistics are both called PSR.

**Correctness.** The filter must be trained against a G centred at the MEASURED displacement, not
at (0,0) — the fix is a per-frame `g_target_shift`, and **a single update cannot see the defect**,
which is why both regression tests are closed loops. The stored filter is H, not Bolme's H\*
(cmul conjugates itself) — invisible whenever the target is centred, which is why s7's target is
off-centre. `mean_prev` must be seeded before frame 0. **Never compute on a BO mapping** (reads
are 696 MB/s; worth 33 ms/frame across five sites). A per-element `std::abs` on a complex is a
`hypot()` call. A self-consistent test can pass on corrupted data — pin the INPUT, not just the
comparison. Any constant both the graph and the host derive from must be passed to both
toolchains from one Makefile variable; a `#ifndef` default in the host is what makes a mismatch
silent.

**AIE / infrastructure.** conv2d weights are consumed per FIRING, not per patch. `aie2gm_nb()`
transfers one invocation per call, and the drain loops must be ORDERED. XRT allows ONE
outstanding async per port. **The CU completion interrupt is never delivered on this platform —
every KDS launch costs ~503 ms**; the fix is `ROI_CROP_USER_MANAGED=1`. `v++ --package` corrupts
the 2025.2 rootfs (`make rootfs`). XRT AXIS ports consume a positional argument slot, so set args
by explicit index. hw_emu wall time is meaningless but its simulated PL cycles are RTL-accurate.

**Settled — do not reopen** ([`docs/engineering/settled.md`](docs/engineering/settled.md)):
`eps_rel = 1e-3`; ReLU off (paired with the `bias_acc` fix); padding 2.0, closed three ways;
fDSST's PCA (the real-input DFT was the win, and is done);
Hermitian halving of the host filter (premise false in fixed point); **quantization is NOT the
cause of the poor robustness — removing it makes tracking WORSE**, so the fixed-point design
costs nothing in accuracy or robustness and the frame rate is bought at no algorithmic price;
init perturbations (the 16-channel denominator is already the cure); pooling and max-pooling
(a null on both banks); channel pruning. Also validated there: `roi_crop` cycle counts and
bit-exactness, the bounding-box/padding unit tests, `nature` and `tiger` being deformation not
tracker defects, sub-bin interpolation, the hold budget, mean-IoU-vs-AR disagreement, PSR gating,
`scale_gate()`, the DSST degeneracy, the per-frame FNV-1a run digest, and the Phase 4 transport
result.

## RGB features — SHIPPING

`CONV_IN_CH=3` is the default; 4-4-4 carries both arms. Build with `make weights CONV_IN_CH=3`
then `ARM=rgb scripts/calib_build.sh`. Datapath, testing and cost:
[`docs/engineering/rgb.md`](docs/engineering/rgb.md).

- **It is a ROBUSTNESS win, never an accuracy one.** Hardware, 419 runs/arm: R 0.2743 → 0.3065,
  EAO 0.1367 → 0.1474, **12.8% more frames survive**, R better on 37 sequences / worse on 15.
  Offline mean IoU is a TIE. **Decide a colour question on AR, never on mean IoU.**
  (Confound: the arms are at different `H_SHIFT`.)
- **Why**: a 3×3 grayscale kernel lives in 9 dimensions, so 16 channels CANNOT be independent —
  the rank cap is structural. Rank/participation ratio 9/4.94 gray against 16/7.43 RGB; held-out
  PSR +1.63×.
- **Cost is what the HOST pays, not what conv2d costs.** conv2d doubles (4.60 → 9.19 ms of AIE
  compute) and **none of it appears in the frame**; the whole +2.29 ms is host-side, led by the
  6 MB `frame_bo` push. The lever for RGB speed is host memory traffic, not the 27 taps.
- **The synthetic scene is a WEAK COLOUR STIMULUS** (rank-1 across planes) — a good IoU there is
  not evidence for the VOT result. The colour-free control (`FRAME_RGB_MODE=0`) reproduced
  grayscale **bit-for-bit on all 199 frames**, which validates the datapath and proves the PSR
  gain is colour and not bookkeeping. On-board colour-path test: ch0's `F_ch` must collapse ~10×
  between the two RGB arms.

## Weight export

`make weights` → extracts torchvision mobilenet_v3_small conv1, folds BatchNorm, then either
collapses RGB→gray by luminance (`CONV_IN_CH=1`) or keeps all 27 taps (`CONV_IN_CH=3`, default);
symmetric per-channel INT8 quantization over ALL taps of a channel, never per plane.
`BIAS_SCALE` picks the activation scale `bias_acc` is derived against (`roi` since 2026-08-23).
Outputs `design/aie_src/weights/layer0_weights.bin` (16 × 64 B), `weights/layer0.h`,
`design/aie_src/hanning_128.h`. **Byte 63 of each channel buffer is the layout tag**; the live
guard is the host's runtime tag check (the `#error` in `layer0.h` is inert — nothing includes
it). The weights are a RUNTIME data file: switching arms is `make weights CONV_IN_CH=<n>` plus a
1 KB file copy onto the card — no re-synthesis, no re-flash.

`scripts/check_collapse.py` is the front-end diagnostic — four checks, no hardware, seconds,
where the alternative for each is an aiesim or hw_emu run. **Re-run it after any change to
`export_weights.py`, `ROI_NORM_Q`, or the collapse.** Q1 collapse convention (needs torch); Q2
linear diversity; Q3 `bias_acc`/`out_shift` sanity (input-independent — trust this one); Q4
post-ReLU maps through conv2d's exact integer datapath (patch-specific). `--skip-torch` gives
Q2-Q4 without torch.

## Build commands

```bash
# DEFAULTS ARE THE SHIPPING ARM since 2026-08-28 (CONV_IN_CH=3 H_SHIFT=15
# MOSSE_ETA=0.05 PSR_GATE_MIN=5.0). Pass CONV_IN_CH=1 for the grayscale arm --
# the gray aiesim scenarios and the 17 gray roi_crop cases need it explicitly.
make weights                       # export layer-1 INT8 weights + hanning table (RGB, 27 taps)
make weights CONV_IN_CH=1          # ...as the 9-tap BT.601 luminance collapse instead
make gen_vectors                   # generate aiesim test vectors
make graph                         # AIE graph only — answers placement questions in 5 min
                                   #   instead of a 25-min package
make test_host                     # native unit tests for filter/PSR/scale/scale-gate/
                                   #   training-target/fusion/scale-reuse/real-DFT/conversions.
                                   #   Runs the suite TWICE, the second time with
                                   #   -O3 -march=native -ffp-contract=fast, because the board's
                                   #   compiler contracts mul+add into FMA by default and a
                                   #   bit-exactness claim proven only at -O2 is proven on the
                                   #   wrong machine. That build caught real bugs -O2 missed
make test_roi_crop                 # bit-exact roi_crop. ROI_IN_CH is COMPILE-TIME, so one build
                                   #   runs one arm: default = the 8 RGB cases, CONV_IN_CH=1 =
                                   #   the 17 gray. A build matching no case exits 2 rather than
                                   #   passing vacuously. RUN BOTH before shipping
make test_scene                    # luma -> interleaved-RGB pass, including a deliberately
                                   #   missed scene_touch() so the verifier is known to fire
make test_vot_source               # VOT manifest/blob/trajectory/streaming reader; 19+7
                                   #   mutants, each REJECTED. With $VOT_ROOT exported it also
                                   #   parses every real manifest and streams the largest real
                                   #   blob against an independent fopen/fread
make test_vot_format               # the BOARD's trajectory writer read back by the toolkit's
                                   #   OWN reader. Needs the venv
make scale_sim                     # closed-loop DSST scale sim — reproduces the f130 stall
make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0 CONV_IN_CH=1   # bit-exact, gray
make x86sim_check KUT=cmul   SCENARIO=s7                 # ...same for cmul_accum
make x86sim_check KUT=cmul   SCENARIO=cmul_stress        # ...exercising sat16's rails
#   cmul needs CMUL_SPLIT_ACCUM=0 here — kernel_only_graph leaves cmul.in[2]
#   unconnected otherwise and the x86sim graph refuses to compile.
make x86sim_check KUT=conv2d SCENARIO=s6rgb              # RGB conv2d, 27 taps, bit-exact
                                   #   needs `make weights` and `make gen_vectors` first
make aiesim CMUL_SPLIT_ACCUM=0     # AIE simulator — bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but forces the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine
                                   #   one). ALSO provisions the board for ssh (static end0 + an
                                   #   authorized key). BOARD_KEY=none opts out explicitly;
                                   #   there is no silent skip when the key is missing
make board_provision ROOTFS_IMG=<img>   # same, against an existing sd_card.img: no re-package
make kernels / xsa / application / sd_card
make sd_card TARGET=hw
make run_emu LAUNCH_HW_EMU_EXEC=1
make cleanall

# One command per sweep: mounts, pushes the ELF, guards, runs, collects, ingests.
scripts/vot_sweep.sh --arm coast0 --seqs car1,tiger --ingest
scripts/vot_sweep.sh --arm x --seqs car1 --dry-run    # prints every remote command, runs none

# Offline benches. The Vitis env MASKS the venv: PYTHONHOME/PYTHONPATH point python at
# Vivado's build, which has no _ctypes — hence the `env -u`.
python3 scripts/mosse_loop_sim.py            # closed-loop MOSSE, centred-G vs shifted-G
python3 scripts/mosse_loop_sim.py --subbin   # ...sweeps resample ratio x speed, and shows
                                             #   quantisation does NOT compound
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_holdout.py   # frozen filter
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py      # closed loop, 1 seq
#   ... --sequence tiger   picks the sequence ($VOT_ROOT stb2022 first, then test-sequences/).
#   Reproduces the board's frozen-detector failures at 3.5 s per 100 frames.
#   ... --arms rgb rgb-warp8    Bolme 3.4's N-sample init (see settled.md). The warp set is
#       restricted to what roi_crop can PRODUCE -- translation and scale; --warp-rot models the
#       HOST pre-rotation route, --warp-aspect uses roi_h/roi_w independently (free on the
#       board), --warp-mutant {gsign,noshift} are the banner-printed negative controls, and
#       --eps-rel reaches Bolme's unregularized regime.
#   ... --arms gray gray-float  the QUANTIZATION COUNTERFACTUAL: unquantized folded-BN weights,
#       no out_shift, no int16 clips, float Hann. Since the model is already float64 downstream
#       of the features, the pair brackets the whole question. Stage A's int8 patch is
#       deliberately NOT removed, so a difference stays attributable.
#   ... --mask-center board|bench  the half-sample window difference; board is the default and
#       the one the hardware runs. --mask-taper 1.0 is REQUIRED and is not the default.
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_vot.py       # VOT supervised, all seqs
#   ... --oracle-scale takes box size from ground truth (isolates localisation from the missing
#       DSST scale filter); the two modes BRACKET what scale handling is worth.
#   uv sync --extra weights  installs torch/torchvision (needed for the folded weights).
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
│   └── aiesim_data/s*/               # s0-s4 raw patches (echo mode only); s6 Stage-A
│                                     #   preprocessed, H=unity, real conv path; s7 = s6 + a
│                                     #   real per-bin complex H, off-centre target (the only
│                                     #   one exercising H_SHIFT, PSR, the F_ch tap); s6rgb
│                                     #   3-plane Stage A -> the 27-tap path; cmul_stress
│                                     #   exercises sat16's rails
├── pl_src/{camera_capture,roi_crop}/
├── host_app_src/
│   ├── mosse_tracker.cpp             # GMIO-driven XRT tracking loop; CropIp (user-managed
│   │                                 #   roi_crop) + the RC_*/timeline launch instrumentation
│   ├── mosse_filter.h/.cpp           # init/update/PSR/scale/Q1.15 export — NO XRT include
│   ├── scene_colour.h/.cpp           # luma scene -> interleaved RGB, rect union, the
│   │                                 #   incremental-vs-full verifier. NO XRT include
│   ├── vot_source.h/.cpp             # VOT manifest/blob/run-order/trajectory. NO XRT include;
│   │                                 #   linked only at FRAME_SOURCE=vot. Pure bookkeeping, so
│   │                                 #   every failure mode is a plausible-but-invalid AR
│   │                                 #   report -- hence the mutation suite. Blob is resident;
│   │                                 #   StreamBlob is the ring + prefetch thread. Access is
│   │                                 #   strictly sequential and OUT-OF-ORDER IS AN ERROR
│   └── test/                         # native tests + NumPy goldens
│       ├── test_mosse_filter.cpp     # make test_host
│       ├── test_scene_colour.cpp     # make test_scene
│       ├── test_vot_source.cpp       # make test_vot_source; `<dir>` emits a trajectory + its
│       │                             #   INPUT for make test_vot_format
│       └── scale_loop_sim.cpp        # closed-loop DSST scale sim (make scale_sim)
├── system_configs/mosse_x1.cfg       # v++ linker
├── profiling_configs/, directives/
└── exec_scripts/run_script.sh

scripts/
  -- generation and build --
  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py, check_collapse.py,
  check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py, gen_roi_crop_golden.py,
  synth_frame.py, sweep_shift.sh, fix_sd_rootfs.sh
  conv_weight_layout.py  # Python mirror of conv_weight_layout.h. EVERY reader of
                         #   layer0_weights.bin goes through it; the tag byte makes a layout
                         #   mismatch loud instead of plausible
  calib_build.sh         # hardware build for a budget or bring-up run: pre-flight, then
                         #   verifies the FLAGSTAMPS against the intended config. Budget
                         #   defaults DERIVED from the Makefile (print-%), never copied.
                         #   CANNOT build either FILTER_MASK arm (ARM=gray|rgb only).
                         #   TAKES THE GEOMETRY since 2026-08-31 (PATCH_ROWS/PATCH_COLS/
                         #   N_CHANNELS -> VARS, BUILD_DIR and the stamp checks) -- before that
                         #   it built 64x64 and verified the 128x128 stamps next to it
  board_provision.sh     # static end0 address + root's authorized_keys into the rootfs (or an
                         #   sd_card.img partition) with debugfs -- no root, no loop device.
                         #   sshd is already enabled in the stock image
  -- the VOT pipeline --
  vot_prepare.py         # VOT sequences -> board blobs + manifests, and the mutation-tested
                         #   verifier. THE groundtruth reduction lives here -- other readers
                         #   import reduce_box from it
  vot_sweep.sh           # drives a whole sweep over ssh. --dry-run prints every command
  vot_ingest.py          # board trajectories -> toolkit workspace -> `vot analysis`. One
                         #   directory per ARM. Re-derives every run name from the sequence's
                         #   anchors and checks each trajectory's LENGTH -- scan() only notices
                         #   a MISSING file, and a wrong-length run is scored without complaint
  vot_roundtrip.py / vot_check_trajectory.py  # the toolkit's result format, and the BOARD's
                         #   writer against the toolkit's reader
  calib_report.py        # a run's console+track.csv -> a budget verdict (rails, amplitude early
                         #   vs converged, IoU). Keys on (job, frame)
  -- diagnosis, all offline, reading existing CSVs or the dataset --
  vot_motion_check.py    # does a sequence's ANNOTATED motion appear in the PIXELS? NCC of the
                         #   box content at "moved" vs "still". Phase correlation cannot answer
                         #   this -- it returns the window's dominant motion
  vot_detector_gain.py   # IS THE DETECTOR THE PROBLEM? Regresses reported peak offset on
                         #   annotated motion, per speed bucket. Answer: no, alpha 0.93.
                         #   --movers splits off sequences whose box moves while the pixels do
                         #   not. NOTE it hardcodes padding 2.0
  vot_loss_anatomy.py    # what the tracker was DOING as it lost: gate verdict, PSR and box
                         #   motion in the 5 frames before the first loss
  vot_init_anatomy.py    # INIT FAILURE or DRIFT? An init failure is visible at f1; a drifting
                         #   run is not, and that is the whole discriminator
  vot_traj_anatomy.py    # a board trajectory vs groundtruth in units of the tracker's own BIN
  vot_hold_budget.py     # frames before a frozen window loses the target, from groundtruth
  vot_mask_stat.py       # reads `mask_ebox` -- THE mechanism check for FILTER_MASK. Keyed on
                         #   (sequence, job, frame); EXCLUDES the -1 rows; reports at-init and
                         #   per-frame-index, never a pooled mean
  check_ebox_crosscheck.py  # the board's filter_box_energy_fraction against the offline
                         #   box_energy_fraction on the SAME H, plus 5 injected mutants
  vot_ar_offline.py      # VOT's failure rule applied to OFFLINE single-start runs. NOT the
                         #   toolkit's AR, and its resolution is MEASURED at ~0.02 in R,
                         #   including one pair it got BACKWARDS. Decides whether an arm
                         #   deserves board time, never whether to accept one.
                         #   Did NOT transfer on pad30
  -- offline benches, no hardware, seconds to minutes --
  mosse_loop_sim.py, bg_pan_sweep.py,
  rgb_vs_gray_holdout.py / _loop.py / _vot.py   # gray vs RGB vs a colour-free control on real
                         #   VOT video. _loop.py also carries the float quantization
                         #   counterfactual, the pooling/resolution arms, and the -warp<N>
                         #   init-perturbation arms with their mutants

docs/
├── engineering/   # the operational detail this file summarises — see its README
├── thesis/        # claims.md, evidence/, results/*.csv, tables/, glossary.md
└── papers/        # Bolme MOSSE, Danelljan DSST
test-sequences/    # VOT sequences + annotations (16 usable, 5971 frames). The annotation
                   # directories are named inconsistently ("car1-annotations" but
                   # "fernando - annotations"); the harness matches them loosely.
```
