# CLAUDE.md

Guidance for Claude Code working in this repo. MOSSE correlation-filter tracker with CNN
features on Versal VEK280, extending the AIE 2D-FFT tutorial (XD073) with a full
object-tracking pipeline. Papers in `docs/papers/` (Bolme MOSSE, Danelljan DSST); section
numbers below refer to them.

## Where things live — three tiers, one job each

**This file is the operational half only**, and it is a DIGEST: it states what is true now and
links out. If a line below reads as a bare assertion, the file it points at says how it was paid
for — nothing was deleted when the detail moved (2026-08-31), and those files, not this one,
are where each topic is now maintained.

| tier | what it is | when it changes |
|---|---|---|
| `CLAUDE.md` | the entry point: environment, commands, the shipping config, the build DIGEST, the rules worth carrying in the head | every arm |
| [`docs/engineering/`](docs/engineering/README.md) | the live operational truth, one file per topic — the unabridged build table, the shift budget, performance, traps, what is settled, what to try next | when a topic moves |
| [`docs/thesis/`](docs/thesis/README.md) | the citeable record: [`claims.md`](docs/thesis/claims.md) (a verdict per question) → `results/*.csv` (every number) → [`evidence/`](docs/thesis/evidence/README.md) (the notes) | when a sweep lands |

**Every doc carries a one-line `**Status:**` header** — `current`, `closed`, `superseded` or
`generated` — with the date it was last true and a one-line scope. Read it before trusting the
file. Inside an append-only evidence note the LATEST dated section wins, and the ones whose
verdict later changed carry a `WHERE THIS ENDED UP` block under the header.

**Four things are generated or checked, so they cannot drift** — run them, do not hand-edit
their outputs: `make code-map` (code_map.md from `@thesis` tags), `make doc-index`
(the evidence index from each note's header), `make thesis-tables` (LaTeX from the CSVs), and
`make check-docs`, which fails if a documented build default disagrees with the Makefile, a
documented path does not exist, a source file is missing from the Directory layout below, a
claim id is duplicated or out of order, or a measured number in a comment cites nothing.

## Thesis scaffold

Findings are indexed in [`docs/thesis/claims.md`](docs/thesis/claims.md) — one row per question
answered, with a verdict, an evidence note and a run directory; the notes are indexed the other
way, note → status → claims, in [`docs/thesis/evidence/README.md`](docs/thesis/evidence/README.md).
**Every number the thesis quotes lives in `docs/thesis/results/*.csv`, not in prose.**
When a sweep finishes: append a row to `results/arms.csv`, update `claims.md`, write the note
from `evidence/TEMPLATE.md` (name it for its TOPIC, never its status), then `make doc-index`.

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

**THE DEFAULTS ARE THE SHIPPING CONFIGURATION as of 2026-09-02** (`rgb_l1relu`, EAO 0.1960).
A bare `make application FRAME_SOURCE=vot` reproduces the benchmark arm's `app.flagstamp`
exactly, and the default `aie.flagstamp` matches the flashed `a.xclbin`. Eight defaults moved
that day, all of them the Layer-1 arm: geometry `128->64` and `N_CHANNELS 16->32`, budget
`4-4-4 -> 3-3-3`, `CONV_KSIZE 3->7`, `CONV_STRIDE 1->2`, `CONV_RELU 0->1`,
`WEIGHT_BANK mobilenet->l1resnet`, `ACC_BOUND` now DERIVED from the bank.
**The 128x128 3x3 arms are still fully supported** and are how every pre-09-02 row in the table
below is reproduced — pass the geometry and the bank back explicitly.
**Grayscale too** — `CONV_IN_CH=1` (what aiesim `s6`/`s7` need).
Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

**The table below is a DIGEST, not the list.** It carries only the knobs that reach a toolchain,
that get set on a routine command line, or whose wrong value silently invalidates a run;
**every knob, with the reasoning behind it, is in
[`docs/engineering/build_params.md`](docs/engineering/build_params.md)** — read that entry before
moving one. Neither table is the source of truth for a DEFAULT: the Makefile is, and
`make check-build-table` (part of `make check-docs`) fails if either document disagrees with
`make print-<KNOB>`. Everything is host-only (an scp, not a card swap) EXCEPT `H_SHIFT`,
`CONV_IN_CH`, `CONV_KSIZE`/`CONV_STRIDE`, `CONV_RELU`, `CONV2D_STACK`, the FFT/window knobs and
the geometry, which reach `AIE_FLAGS`.

| Parameter | Default | One line |
|---|---|---|
| `TARGET` | `hw_emu` | `hw_emu` or `hw` |
| `PATCH_ROWS`/`PATCH_COLS` | `64` | the FEATURE MAP, powers of 2 (AIE FFT constraint). **Moving it silently moves `sigma/target`** — see `MOSSE_SIGMA` |
| `N_CHANNELS` | `32` | conv feature channels. Host cost scales as `N_CHANNELS x map_pixels` and so does conv2d's read loop |
| `ITER_CNT` | `1` | frames; **needs ≥2** — frame 0 initialises the filter |
| `H_SHIFT` | `15` | filter-product shift, deliberately OVER-shifted (`rails=0` over 101,564 frames). 13 is the tight RGB value, 12 rails, gray's arm is 14. **The one non-host-only knob here** |
| `FFT_SHIFT`/`IFFT_ROW_SHIFT`/`IFFT_COL_SHIFT` | `3`/`3`/`3` | the 64x64 budget (calibrated 2026-09-02, `rails=0` on 200 frames). **4-4-4 is the 128x128 one** — the budget follows the POINT SIZE, so a geometry change needs a new calibration run |
| `MEMTILE_TRANSPOSE` | `1` | transposes in AIE-ML memory tiles; a one-sided flag is a board deadlock |
| `CMUL_SPLIT_ACCUM` | `1` | own port for `accum_prev`. **`make aiesim` needs `0`** |
| `ROI_CROP_USER_MANAGED` | `1` | roi_crop via `xrt::ip`. **20.6× on frame rate** — KDS completion costs 503 ms/launch |
| `CONV2D_MODE` | `0` | 0=real conv, 1=echo, 2=synthesize. **Check before every expensive run** |
| `CONV_RELU` | `1` | **Bank-specific.** Refuted on the 3x3/16 mobilenet bank (dR −0.0332); beats its own linear twin four times offline on a LEARNED Layer-1 bank, and loses on an analytic Gabor one — the property that matters is that the bank is learned. Ships on. **The linear twin has NOT run on hardware**, so N-16's gain is attributable to the arm, not yet to the rectifier. **The scalar path (`CONV_VECTORIZE=0`) of the 3x3 GRAY branch hardcodes ReLU and ignores this flag**; RGB and generic honour it |
| `CONV_IN_CH` | `3` | 1=BT.601 luma, 3=RGB. Picks the **weight-buffer layout**, so it drives `AIE_FLAGS`/`GCC_FLAGS`/`ROI_IN_CH` |
| `CONV_KSIZE`/`CONV_STRIDE` | `7`/`2` | conv kernel and stride. Anything but 3/1 selects conv2d's **generic KxK branch** (taps stay in the weight buffer, line buffer split by column phase, `CONV_VEC_GEN=32`); the two 3x3 branches are byte-for-byte as shipped and keep their own `CONV_VEC=16`. Both reach BOTH toolchains |
| `CROP_ROWS`/`CROP_COLS` | derived | `PATCH x CONV_STRIDE` — the ROI crop, **not** the feature map. `roi_crop` takes it at RUNTIME so a 128x128 crop needs no PL rebuild, but its BRAM scratch caps at 128; `make check_geometry` and a host `static_assert` both refuse an overrun |
| `ACC_BOUND` | **derived from `WEIGHT_BANK`** | which exact worst-case accumulator bound sizes `out_shift`. `l1resnet`→`l1` (`127*Σ|w_int8|` per channel, ~2 bits tighter, needed at 147 taps — without it `F_ch` measured 0.13% of int16); `mobilenet`→`loose` (`n_in*K^2*127^2`, what every pre-09-02 arm used and what reproducing them requires). Both are exact worst cases, so neither can rail. **Reaches NEITHER toolchain** — it changes only the weights DATA — so no flagstamp sees it; `calib_build.sh` re-derives both bounds from the file's own taps |
| `WEIGHT_BANK` | `l1resnet` | donor bank for `make weights`. `l1resnet` = resnet18 conv1 7x7/2 PCA'd to `N_CHANNELS`, via `scripts/l1_banks.py` — the same function the offline screen scored |
| `CONV2D_STACK` | `2048` | conv2d AIE stack; applied only at `CONV_IN_CH=3` (27 taps need 1344 > 1024 default) |
| `FRAME_SOURCE` | `synth` | `vot` = frames from a converted VOT blob; ignores `ITER_CNT`, the scene generator and its knobs. `vot`+RGB needs the `.luma` sidecar — the SHIPPING combination |
| `VOT_DATA_DIR`/`VOT_RESULTS_DIR`/`VOT_SEQUENCE`/`VOT_JOB` | `/mnt/vot`/`/mnt/vot-results`/`car1`/`0` | overridable on the board's command line; **repeating a job in `--vot-jobs` is the determinism test** |
| `FILTER_MASK` | `0` | spatial reliability `h ← m⊙h`; window FORCED periodic Hann ⇒ 8 complex adds/bin, no multiplies. **Swept 2026-08-31 (EAO +0.0110) but default still 0** — not separable from a null (3 of 62 sequences) |
| `PSR_GATE_MIN` | `5.0` | Bolme §3.5. Below it the host HOLDS and skips the update. **Worth is conditional on the PSR scale** |
| `TARGET_H`/`TARGET_W` | `64` | target box, frame px |
| `TARGET_PADDING` | `2` | settled three ways; 1.5 and 3.0 both worse. At 64/2 the resample is 1:1 |
| `MOSSE_SIGMA`/`SIGMA_FROM_TARGET` | `2.0`/`0` | **BINS, so what matters is sigma/target, and 1/16 (DSST's `target/16`) is the measured optimum** — 2.0 on a 64x64 map and 4.0 on a 128x128 one both sit there. **The interior is CLOSED**: a 22-cell grid puts sigma 3, 5, 6 and 8 all below it (`runs/vot/0902_offline-sigmaeta/`) |
| `MOSSE_ETA` | `0.05` | +0.0218 R vs 0.125; not monotone (0.025 much worse). **`0.1` MEASURED AND REJECTED on hardware 2026-09-02** — EAO 0.1960 → 0.1817, R a wash (median dR exactly 0.0000), A −0.0136. The offline grid's only trim-stable cell of 22 (dR +0.0481, P=0.021) **INVERTED**; it was screened at 128x128/sigma 4 and `sigma/target` governs SIGMA, not eta |
| `SCALE_N`/`SCALE_STEP` | `33`/`1.04` | `SCALE_N=1` disables. **1.04 beats DSST §6.1's 1.02 on hardware** |
| `DUMP_BUFFERS` | `1` | **1216 KB/frame, ~2 s/frame** — set `0` for any tracking or FPS run |
| `CSV_LOG` | `1` | one row/frame to `track.csv` (`track_<sequence>.csv` at `FRAME_SOURCE=vot`), ~60 B/frame |

**The other 45 knobs are not listed here.** Each one is either host-only, a bisection switch, or
a synthetic-scene control, and every one of them has a paragraph in
[`docs/engineering/build_params.md`](docs/engineering/build_params.md) — which is where their
defaults live, so there is only ever one copy to update. By theme, so you know what exists:

- **AIE / PL cost, all measured and all closed** — `FFT_2D_DT`, `PL_FREQ`, `FFT_ROW_WS`/`FFT_COL_WS`,
  `ROI_CROP_PIPELINE`, `CMUL_ACCUM_MEMTILE`, `TAIL_PARALLEL`, `CONTROL_CU_RUNS`.
- **Bisection switches** — `CONV_VECTORIZE`/`CMUL_VECTORIZE` (bit-identical to scalar),
  `RESET_MUTANT` and `SCENE_VERIFY` (deliberate breakage, so a test is known to be able to fail).
- **Weight export and preprocessing** — `BIAS_SCALE`, `B2_NULL_BINS`.
- **Mask instrumentation** — `FILTER_MASK_STAT`, `FILTER_MASK_STAT_WARM`/`_EVERY`. `-1` in the
  stat column is NOT MEASURED, not zero.
- **Scale filter interior, all settled or refuted** — `SCALE_ETA`, `SCALE_CONF_MIN`,
  `SCALE_MAX_STEP`, `SCALE_MIN_REL`/`SCALE_MAX_REL`, `HOLD_COAST`/`COAST_DECAY`.
- **Synthetic scene only** (inert at `FRAME_SOURCE=vot`) — `FRAME_RGB_MODE`, `FRAME_TEXTURE`/
  `FRAME_NOISE`, `BG_PAN*`, `TRAJECTORY`/`TRAJ_*`, `SCALE_TRAJ*`, `OCCLUDE_*`.
- **VOT reader and console** — `VOT_RESIDENT_MAX_MB`/`VOT_STREAM_RING`, `PROGRESS_EVERY`,
  `CSV_FLUSH_EVERY`, `VERBOSITY`.

### Shift budget — SETTLED: 3-3-3 at 64x64 (SHIPPING), 4-4-4 at 128x128; `H_SHIFT` 15

Closed on hardware (128x128 2026-08-24/27; 64x64 2026-09-02, `rails=0` on 200 frames and all
four buffers). **The budget follows the POINT SIZE** — one bit off each of the four shifts is
exactly the halved transform — so a geometry change needs its own 200-frame calibration.
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
                   Bilinear resample of roi_h×roi_w to CROP_ROWS×CROP_COLS with border
                   clamping, log, zero mean, unit L2 × ROI_NORM_Q, int8 quantize.
                   Two reduction passes + a stream-out pass; `recompute=1` on channel 0,
                   the rest re-stream the cache. All geometry is runtime AXI-Lite,
                   so ROI/box changes need no rebuild of anything but the host ELF.
                   At ROI_IN_CH=3 it resamples three interleaved planes with shared
                   geometry and ONE joint mean/sigma. ROI_IN_CH is compile-time.

AIE (single instances, serial per-channel; both custom kernels vectorized)
  conv2d_kernel     : int8 crop → KxK/stride-S MAC → ReLU → Hanning window → cint16 stream.
                      SHIPPING: resnet18 conv1 7x7 stride 2, PCA'd to 32 channels, INT8,
                      128x128 crop → 64x64 map. Three branches: 3x3 gray, 3x3 RGB (both
                      byte-for-byte as shipped), and the generic KxK/stride-S one.
                      Stage B1 subtracts mean_prev, which the host SEEDS before frame 0.
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
`docs/thesis/evidence/board_memory.md`.

## Current status (2026-09-02)

**THE SHIPPING CONFIG — `rgb_l1relu`, EAO 0.1960, the best on record:**

```
64x64 map from a 128x128 crop  N_CHANNELS=32  CONV_IN_CH=3
conv 7x7 stride 2, resnet18-conv1 PCA bank, CONV_RELU=1, ACC_BOUND=l1
budget 3-3-3  H_SHIFT=15  MOSSE_SIGMA=2.0 (= sigma/target 1/16)
MOSSE_ETA=0.05  PSR_GATE_MIN=5.0  TARGET_PADDING=2.0  SCALE_STEP=1.04
```

Full VOT-STb2022, 62 sequences, 419 trajectories per arm. Each row moves ONE knob:

| arm | A | R | EAO | workspace |
|---|---|---|---|---|
| gray `H_SHIFT=14` | 0.4890 | 0.2743 | 0.1367 | `full62` |
| RGB `H_SHIFT=15` | 0.5043 | 0.3065 | 0.1474 | `full62` |
| + `MOSSE_ETA=0.05` | 0.5100 | 0.3283 | 0.1600 | `eta05ab` |
| + `PSR_GATE_MIN=5.0` | 0.5100 | 0.3417 | 0.1629 | `gate` |
| + `TARGET_PADDING=3.0` — REJECTED | 0.5030 | 0.3494 | 0.1570 | `pad30ab` |
| + `FILTER_MASK=1` — not separable from a null | 0.4913 | 0.3608 | 0.1740 | `0831_mask` |
| + 64x64 map (3-3-3) | 0.5336 | 0.3873 | 0.1849 | `0901_res64` |
| ...+ `PSR_GATE_MIN=3.5` — REJECTED, EAO null | 0.5308 | 0.3936 | 0.1849 | `0901_gate35` |
| + `MOSSE_SIGMA=4.0` at 128x128 (HOST-ONLY) | 0.5133 | 0.4095 | 0.1931 | `0901_sigma4` |
| **+ Layer-1 features (7x7/2, ReLU, 32ch, 64x64) — SHIPPING** | **0.5129** | **0.4279** | **0.1960** | `0902_l1relu` |
| ...+ `CONV_RELU=0` (`l1lin`, the linear twin) — MECHANISM CHECK, rectifier CONFIRMED | 0.5294 | 0.3790 | 0.1851 | `0903_l1lin` |

`PSR_GATE_MIN=0` is a null (13-seq probe). The padding arm gained R and LOST EAO — **EAO is the
arbiter for an A/R trade**. **The gate's worth is NOT conditional on the PSR scale** (refuted
2026-09-01): 95.9% of vetoes fire after the run is already lost, so the threshold is not worth
re-tuning per geometry.

**The shipping arm did NOT meet its own pre-registered bar** (`dEAO >= +0.005`; measured
+0.0029) and is shipped on two other grounds, both recorded in
[`evidence/arm_l1relu.md`](docs/thesis/evidence/arm_l1relu.md) sec.10-12:
the paired per-sequence result is strong where the scalar is not (R **+0.0112 after drop-top-5,
P(dR<=0)=0.011**, accuracy up too, no A/R trade), and at `CONV_RELU=0` the conv layer is a
**LINEAR LIFT the online filter absorbs** — a one-hot bank with no network ties the pretrained
one, so the previous config was a CNN-feature tracker whose CNN was provably redundant.
**Never write this arm up as having passed the falsifier** — that is a separate question and is
unchanged. **But the linear twin RAN on 2026-09-03 and the gain IS the rectifier**: `l1lin`
(same bank, `aie.flagstamp` differing on exactly `CONV_RELU`) lands at EAO **0.1851** against
0.1960, paired R **+0.0447 / +0.0274 after drop-top-5, P(dR<=0)=0.000**, A moving the same way.
So the conv layer is doing real work and the CNN is no longer provably redundant — `N-16`,
`evidence/arm_l1relu.md` sec.14.

### THE EAO WINDOW IS [115, 755], AND IT CAPS WHAT FEATURES CAN BUY

`l1relu`'s +0.0184 pooled R became only +0.0029 EAO, and the reason generalises to every future
arm (sec.10.3, reimplemented to within 0.0008 of the toolkit):

| EAO sub-window | share | dEO |
|---|---|---|
| 115-300 | 29% | **+0.0092** |
| 301-755 | **71%** | **+0.0005** |

**Better FEATURES improve acquisition and mid-horizon survival and are then diluted threefold.**
18 fewer runs die within 30 frames; nothing changes after ~300. What governs the other 71% is
long-horizon BOX QUALITY — **but that is NOT the scale filter, and the whole scale direction is
CLOSED** (2026-09-02, [`docs/engineering/scale_filter.md`](docs/engineering/scale_filter.md)).
**A PERFECT scale filter is worth +0.0023 R on the shipping arm and −0.0089 on `sigma4`**
(`scripts/scale_oracle_bound.py`: the board's own trajectories, tracker's centre, ground-truth
size, VOT's rule re-applied). It lifts mean IoU by **+0.054** and converts almost none of that
into survival, because a run at IoU <= 0.1 has lost the target's POSITION and resizing a box that
is not on the target rescues nothing. The filter IS broken and it is well understood — frozen on
~90% of frames, detector gain **−0.003** against the position detector's 0.93, root-caused to a
self-confirming loop fed by a scale-NORMALISING feature — **it is simply not worth fixing.**
The pre-loss mis-sizing (>25% on 60% of frames) is a MARKER of a run in trouble, not its cause:
both errors are already large 40 frames out and POSITION accelerates faster into the loss.

The full chain runs on hardware on the real conv path. Heap, not tracking, was the last blocker;
`vot::StreamBlob` closed it, proven by identical run-state digests both ways.

**ENERGY PER FRAME: 12.2 mJ (0.487 W dynamic at 25.16 ms), measured 2026-09-03** — the last
thesis debt with no number, now paid. From the VEK280 System Controller's INA226 rails
(`sc_app` over `/dev/ttyUSB3`; **the APU exposes no current sensor at all**), as a DIFFERENCE
against an idle baseline. **68% is the APU rail, 23% LPDDR4, 9% NoC — and the PL+AIE-ML rail
does not move (< 33 mW)**, nor does merely having the graph resident (unresolved on every
rail). That is P-02's CPU-bound frame confirmed in the energy domain by an instrument on a
different chip. `results/power.csv`, `evidence/power.md`, claim `P-12`.
**Never quote a rail delta without its error bar**: the rails are quantized very differently
(149 distinct values on `VCC_PSFP`, three on `VCC_PMC`), and on a 3-level rail the sample sd
collapses and any difference clears "2 s.e." — which is exactly how three null rails first
reported `CONTROL FAILED` at ±0.000 W.

**Best hardware FPS: 26.29 ms/frame = 38.04 FPS** (`runs/run_0821_1725.log`, gray, synthetic,
UART console). **Quote FPS only from a serial-console run** — over ssh the same work reads
differently, and the UART alone was 3.79 ms. On the ssh instrument the shipping arm is **24.07 ms
on `agility` against `sigma4`'s 24.55** — 32 channels and 147 taps for slightly less than the old
16-channel 3x3 arm, after the 2026-09-02 conv2d rework (2.39x on the schedule; row bases hoisted
out of the column loop, `kc` unrolled, the PLIO read stored straight from the word).
Frame breakdown: [`docs/engineering/performance.md`](docs/engineering/performance.md).

Where this sits against the 41 published VOT-STb2022 trackers, and against the nearest
published EMBEDDED equivalent (Danilowicz & Kryjak 2022):
[`docs/engineering/baselines.md`](docs/engineering/baselines.md). The split is sharp — **A is
inside the classical-DCF band; R is below every one of the 41.**
**On COST the comparison is valid and this design wins it** — 20.4x fewer LUT, 44.4x fewer FF,
54.1x less BRAM, 10.9x fewer DSP than their ZCU104 deepDCF at the same geometry, because the
transforms moved onto 6 of 304 AIE-ML cores (`P-13`). **On TRACKING it is not valid**: their EAO
is on VOT2015 with a [108, 371] window, an inverted `R`, and polygon ground truth.
**The window term alone is +0.0827 on the shipping arm — 1.39x this project's entire arm
ladder** (`M-17`, `results/eao_window.csv`), so 0.1960-against-0.183 is an artefact, not a
result. **Never put a re-windowed number in a table beside theirs.** The gate is the AFTERMATH of a
loss, not its cause: 95.8% of vetoes land after the run is already at IoU <= 0.1. **It walks off
the target confidently — do not spend a sweep relaxing the gate, and do not look for the fault in
localisation.**

## What to try next

Ranked, with the evidence for each rank, in
[`docs/engineering/roadmap.md`](docs/engineering/roadmap.md). Headlines:

- **DONE, REJECTED — `MOSSE_ETA=0.1`.** EAO 0.1960 → 0.1817 (sec.13). The named assumption is
  what broke: the grid ran at 128x128/sigma 4, the arm at 64x64/sigma 2, and `sigma/target`
  governs SIGMA only. **The methodological result outlives the arm — see below.**
- **DONE, FALSIFIER MET — `ARM=l1lin`, the mechanism check.** EAO **0.1851** against 0.1960, so the
  gain is the RECTIFIER and not the bank; paired R +0.0447, trim-5 +0.0274, P(dR<=0)=0.000, no A/R
  trade. `N-16` is confirmed ON HARDWARE and the conv layer is no longer provably redundant.
- **ROBUSTNESS — after 2026-09-03 there is ONE candidate left.** 71% of the EAO window is
  long-horizon box quality and the scale direction is CLOSED by the oracle bound above, so what
  ends runs is POSITION drift. Everything host-only that attacked it is now refuted:
  - **THE ONE REMAINING ITEM: a training-sample memory.** SRDCF/CSRDCF keep weighted sample SETS;
    this keeps one running average, which is exactly the "walks off target confidently" mechanism
    (R-06). It is the only candidate that was never a confidence mechanism, which is why none of
    2026-09-03's results touch it. Untested.
  - ~~two-filter temporal ensemble~~ — **CLOSED, both halves (N-22, N-24, O-03).** The cheap half
    (a long-term filter as a confidence VALIDATOR feeding a modulated eta) loses on PSR and APCE
    with its mutant failing to lose. The premise underneath — a second memory sees drift one
    filter does not — is refuted by a pure-observer probe: peak disagreement predicts an imminent
    loss at AUC **0.461** frozen / **0.555** slow, against PSR's already-too-weak 0.618. A
    per-frame selection rule has no signal left, so the second AIE bank has nothing to select on.
  - ~~confidence-modulated eta, and CONFIDENCE-DERIVED PER-FRAME STATISTICS AS A CLASS~~ —
    **CLOSED (N-22, N-23).** PSR is U-SHAPED in correctness: the most confident band is 96.9%
    lost, because those frames are welded to static background (box motion 0.00 px while the
    target moves 2.24 px/frame). Both tails are POST-LOSS, so a two-sided law is two aftermath
    detectors bolted together. Pre-loss the tracker looks confident and MOVING.
  - ~~the spatial mask on this arm~~ — **REFUTED (N-21):** its sign INVERTS on the Layer-1 bank
    (+0.0601 on the old 3x3 one, −0.0127 paired here, P(dR<=0)=0.706); the k=2 width knob is
    worse. Stage B3 channel reliability was re-screened on the same sweep and is a null too.
  - `MOSSE_SIGMA`'s interior is CLOSED (22-cell grid; 3, 5, 6, 8 all worse).

  Notes: `docs/thesis/evidence/confidence_eta.md`, `docs/thesis/evidence/mask_bank_transfer.md`.
- **FEATURES — mostly closed, and the ceiling is now known.** The bank's weights are a linear
  lift (random ties pretrained, one-hot ties the network), pooling is a null, channel count is
  not trim-separable between 16 and 32, and Danilowicz & Kryjak measure 8ch ~ 32ch ~ 64ch. What
  is left is geometry, and the EAO-window result caps any feature arm at ~1/3 of its R gain. The
  one untested affordable cell is a **16-channel Layer-1 bank at a 128x128 map** (host cost =
  `sigma4`'s), which is the only route that puts better features at the finer geometry.
- **PERFORMANCE.** conv2d's read loop is now 82% of that kernel and its scheduled floor is
  84 cycles per 4 pixels; each (plane, phase) receives `4/S` CONTIGUOUS bytes, so 12 byte stores
  could be 6 halfword ones — priced, not taken. Host side: `scale extract` is invariant to the
  map and is the head of the tail; the rest scales as `N_CHANNELS x map_pixels`.
  **Retired, do not reopen**: `FFT_COL_WS` 8->32, `CMUL_ACCUM_MEMTILE` alone, Hermitian symmetry
  in the host filter, the accumulator as a `shared_buffer`, parallel-for inside
  `filter_update_quantize` as attempted, a `px[4][NC]` staging array or an `srow[NC][S]` pointer
  array in conv2d's read loop (both measured WORSE than the rolled original).

**`MOSSE_SIGMA` is in BINS and the target spans `patch/padding` bins — what matters is
sigma/target, and 1/16 (DSST's `target/16`) is the measured optimum.** That one fact re-attributed
the 64x64 arm: it sat at 1/16 by accident, so it and `sigma4` are a MATCHED PAIR on width, and at
matched width the FINER map wins. **Any geometry change silently moves this knob.**

**THE OFFLINE PROXY BOUNDS SAMPLING NOISE, NOT TRANSFER.** `vot_ar_offline.py` has now
transferred on the RECTIFIER (**85%, the best rate on record**), sigma (84%) and `dec2` (43%),
over-called the mask 3x and `pad30` ~11x, INVERTED the mask's SIGN on a new bank (N-21), and on
2026-09-02 **INVERTED on `MOSSE_ETA`**. **The emerging pattern, fitted to seven points and offered
as a prior rather than a rule: it transfers on arms that change the FEATURES or the response
WIDTH, and misleads on arms acting through the filter/veto path** — its only trim-stable,
bootstrap-significant cell of a 22-cell grid (dR +0.0481, P(dR<=0)=0.021) returned −0.0030 on
hardware. **A trim and a bootstrap say a result is not carried by three sequences; they say
NOTHING about whether the bench models the tracker.** Treat `P(dR<=0)` as necessary, never
sufficient, and write down the transfer assumption before any arm screened at a geometry the
board does not run.

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

**Metrics that cannot fail a broken tracker.** **PSR IS NON-MONOTONE IN CORRECTNESS** (N-23,
measured over 180,125 frames): fraction already lost is 70.2% at PSR 0-10, 48.4% at 20-30 (the
optimum) and **96.9% above 50** — the most confident band is the worst. Those frames are WELDED to
static background (median box motion **0.00 px** while the truth moves 2.24 px/frame), so the filter
self-correlates more sharply than on a real target. The old one-case version of this —  a tracker
179 px off target, confidently locked to background, reported PSR 33 — is the same effect.
**So no MONOTONE law on PSR can work** (it sank the confidence-modulated eta, N-22), and the welded
population is detectable by zero MOTION, never by low confidence. **IoU is the only metric in
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

**A PRIOR POSITIVE SCREEN EXPIRES WHEN THE OPERATING POINT MOVES** (M-14, 2026-09-03) — re-screen
an arm whenever the bank, the geometry or the response shape has changed since it was scored, not
only when something feels suspect. Eight minutes of CPU inverted the proposed mask arm and saved a
sweep, an ingest and a card round-trip.

**Settled — do not reopen** ([`docs/engineering/settled.md`](docs/engineering/settled.md)):
`eps_rel = 1e-3`; padding 2.0, closed three ways;
fDSST's PCA (the real-input DFT was the win, and is done);
Hermitian halving of the host filter (premise false in fixed point); **quantization is NOT the
cause of the poor robustness — removing it makes tracking WORSE**, so the fixed-point design
costs nothing in accuracy or robustness and the frame rate is bought at no algorithmic price;
init perturbations (the 16-channel denominator is already the cure); **aggregation over this map
— box average, max and decimate, on the linear AND the rectified bank** (a null at best, and a
LOSS of −0.0242 on the shipping bank; its old "linearity" explanation is WITHDRAWN — the linear
negative control lost too); channel pruning; **the sigma interior** (22-cell grid) and
**`SCALE_ETA`** (inert 0.025-0.3 — the scale filter is FROZEN on ~90% of frames, so the freeze is
a DETECTION failure and no learning rate reaches it), both 2026-09-02.
*ReLU left this list on 2026-09-02* — refuted on a 3x3 mobilenet bank, ships on a learned
Layer-1 one; the flag is bank-specific, not settled. Also validated there: `roi_crop` cycle counts and
bit-exactness, the bounding-box/padding unit tests, `nature` and `tiger` being deformation not
tracker defects, sub-bin interpolation, the hold budget, mean-IoU-vs-AR disagreement, PSR gating,
`scale_gate()`, the DSST degeneracy, the per-frame FNV-1a run digest, and the Phase 4 transport
result.

## RGB features — SHIPPING

`CONV_IN_CH=3` is the default and carries into the Layer-1 arm. To rebuild the 3x3 RGB arm:
`make weights WEIGHT_BANK=mobilenet CONV_KSIZE=3 CONV_STRIDE=1 N_CHANNELS=16` then
`ARM=rgb PATCH_ROWS=128 PATCH_COLS=128 N_CHANNELS=16 scripts/calib_build.sh`. Datapath, testing and cost:
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

`make weights` → loads the donor bank picked by `WEIGHT_BANK` and folds BatchNorm:
**`l1resnet` (default) = resnet18 conv1 7x7/2, 64 filters PCA'd to `N_CHANNELS`**
(`scripts/l1_banks.py`, the same function the offline screen scored); `mobilenet` =
mobilenet_v3_small conv1 3x3, either collapsed RGB→gray by luminance (`CONV_IN_CH=1`) or all 27
taps. Then symmetric per-channel INT8 quantization over ALL taps of a channel, never per plane.
`BIAS_SCALE` picks the activation scale `bias_acc` is derived against (`roi` since 2026-08-23);
`ACC_BOUND` (derived from the bank) picks which exact worst-case bound sizes `out_shift`.
Outputs `design/aie_src/weights/layer0_weights.bin` (32 × 192 B on the shipping bank, 16 × 64 B
on a 3x3 one), `weights/layer0.h`, `design/aie_src/hanning_<PATCH_COLS>.h`.
**The LAST TWO bytes of each channel buffer are the layout tags** (`CONV_IN_CH`, `CONV_KSIZE`);
the live
guard is the host's runtime tag check (the `#error` in `layer0.h` is inert — nothing includes
it). The weights are a RUNTIME data file: switching BANKS within one geometry is `make weights`
plus a small file copy onto the card — no re-synthesis, no re-flash. Changing the KERNEL or the
geometry is not: those reach `AIE_FLAGS`.

`scripts/check_collapse.py` is the front-end diagnostic — four checks, no hardware, seconds,
where the alternative for each is an aiesim or hw_emu run. **Re-run it after any change to
`export_weights.py`, `ROI_NORM_Q`, or the collapse.** Q1 collapse convention (needs torch); Q2
linear diversity; Q3 `bias_acc`/`out_shift` sanity (input-independent — trust this one); Q4
post-ReLU maps through conv2d's exact integer datapath (patch-specific). `--skip-torch` gives
Q2-Q4 without torch.

## Build commands

```bash
# DEFAULTS ARE THE SHIPPING ARM since 2026-09-02: 64x64 map / ch32 / 7x7 stride 2 /
# CONV_RELU=1 / l1resnet / 3-3-3 / H_SHIFT=15. The older 3x3 arms need their
# geometry AND bank back explicitly -- see the reproduction lines below.
make weights                       # 147-tap resnet18 conv1 PCA'd to N_CHANNELS (the shipping bank)
make weights WEIGHT_BANK=mobilenet CONV_KSIZE=3 CONV_STRIDE=1 N_CHANNELS=16 \
     PATCH_ROWS=128 PATCH_COLS=128            # ...the pre-09-02 RGB 3x3 bank
make weights WEIGHT_BANK=mobilenet CONV_KSIZE=3 CONV_STRIDE=1 N_CHANNELS=16 \
     PATCH_ROWS=128 PATCH_COLS=128 CONV_IN_CH=1   # ...the 9-tap BT.601 luminance collapse
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
make x86sim_check KUT=conv2d SCENARIO=s6rgb CONV_KSIZE=3 CONV_STRIDE=1 N_CHANNELS=16 \
     PATCH_ROWS=128 PATCH_COLS=128 CONV_RELU=0            # RGB 3x3, 27 taps, bit-exact
make x86sim_check KUT=conv2d SCENARIO=s6l1               # the SHIPPING 147-tap 7x7/2 path
                                   #   each needs its own `make weights` + `make gen_vectors`
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
│   ├── conv2d_kernel.h/.cpp          # KxK/stride-S MAC + ReLU + Hanning window + Stage B1.
│   │                                 #   3 branches: 3x3 gray, 3x3 RGB, generic (SHIPPING)
│   ├── cmul_accum_kernel.h/.cpp      # col-FFT ⊙ H_ch* + saturating accumulate
│   ├── mosse_graph.h/.cpp            # top level: PLIO + GMIO + 2 kernels + FFT2D + IFFT2D
│   ├── kernel_only_graph.h/.cpp      # x86sim single-kernel harness
│   ├── aiesim_scenario_io.h          # shared scenario loader (keeps the two harnesses in sync)
│   ├── constraints.aiecst            # PatchIn PLIO shim placement
│   ├── hanning_<N>.h, weights/       # auto-generated (64 and 128 both live)
│   └── aiesim_data/s*/               # s0-s4 raw patches (echo mode only); s6 Stage-A
│                                     #   preprocessed, H=unity, real conv path; s7 = s6 + a
│                                     #   real per-bin complex H, off-centre target (the only
│                                     #   one exercising H_SHIFT, PSR, the F_ch tap); s6rgb
│                                     #   3-plane Stage A -> the 27-tap path; s6l1 the
│                                     #   147-tap 7x7/2 SHIPPING path; cmul_stress the rails
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
├── pl_src/test/test_roi_crop.cpp     # make test_roi_crop; ROI_IN_CH is COMPILE-TIME, so one
│                                     #   build runs one arm (RGB by default, gray at
│                                     #   CONV_IN_CH=1) and a build matching no case exits 2
├── system_configs/mosse_x1.cfg       # v++ linker (plio_smoke.cfg links the smoke graph)
├── profiling_configs/, directives/
├── exec_scripts/run_script.sh        # ...and run_smoke.sh, the PLIO smoke test's board script
└── (bring-up harnesses, all still built by the Makefile and kept for bisection)
    fft_only_graph.h/.cpp + fft_only_constraints.aiecst   # the FFT chain with no conv2d
    plio_smoke_graph.h/.cpp + plio_smoke_host.cpp + smoke_passthrough.h/.cpp
                                      #   PL->AIE AXIS handshake only, scored by
                                      #   scripts/analyze_plio_vcd.py
    pl_src/stream_src/stream_src.h/.cpp   # the smoke test's PL producer

scripts/
  -- generation and build --
  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py, check_collapse.py,
  check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py, gen_roi_crop_golden.py,
  synth_frame.py, sweep_shift.sh, fix_sd_rootfs.sh
  l1_banks.py            # THE Layer-1 donor banks (7x7/2, PCA'd to N_CHANNELS). The offline
                         #   screen and `make weights` call the SAME function, so the board arm
                         #   and the screened arm cannot be two spellings of one bank
  conv_weight_layout.py  # Python mirror of conv_weight_layout.h. EVERY reader of
                         #   layer0_weights.bin goes through it; the tag byte makes a layout
                         #   mismatch loud instead of plausible
  calib_build.sh         # hardware build for a budget or bring-up run: pre-flight, then
                         #   verifies the FLAGSTAMPS against the intended config. Budget
                         #   defaults DERIVED from the Makefile (print-%), never copied.
                         #   ARM=gray|rgb|l1relu|l1lin; CANNOT build a FILTER_MASK arm.
                         #   Takes the GEOMETRY, and cross-checks the weights file's out_shift
                         #   against ACC_BOUND -- which no flagstamp can see
  board_provision.sh     # static end0 address + root's authorized_keys into the rootfs (or an
                         #   sd_card.img partition) with debugfs -- no root, no loop device.
                         #   sshd is already enabled in the stock image
  -- the VOT pipeline --
  vot_prepare.py         # VOT sequences -> board blobs + manifests, and the mutation-tested
                         #   verifier. THE groundtruth reduction lives here -- other readers
                         #   import reduce_box from it
  vot_sweep.sh           # drives a whole sweep over ssh. --dry-run prints every command
  board_run.sh           # runs the tracker on the board and RETURNS when it is done. `gr.end(0)`
                         #   blocks forever on a free-running graph, which is cosmetic by hand
                         #   and fatal to automation -- and NEVER KILL A RUN MID-SEQUENCE
  vot_ingest.py          # board trajectories -> toolkit workspace -> `vot analysis`. One
                         #   directory per ARM. Re-derives every run name from the sequence's
                         #   anchors and checks each trajectory's LENGTH -- scan() only notices
                         #   a MISSING file, and a wrong-length run is scored without complaint
  vot_roundtrip.py / vot_check_trajectory.py  # the toolkit's result format, and the BOARD's
                         #   writer against the toolkit's reader
  calib_report.py        # a run's console+track.csv -> a budget verdict (rails, amplitude early
                         #   vs converged, IoU). Keys on (job, frame)
  -- power / energy, the one thesis debt with no claim behind it --
  power_probe.sh         # samples whatever power/thermal channels a host exposes, as long-format
                         #   CSV. Backends: `sc_app` (INA226 rails, runs ON the System
                         #   Controller -- the only source of WATTS) and `sysmon` (the board's
                         #   own iio: regulated voltages + die temp, NO current). Discovers its
                         #   channels; exits 3 rather than emit an empty CSV
  power_measure.py       # the protocol and the arithmetic: static -> graph -> run -> graph_post
                         #   -> tail, and the answer is a DIFFERENCE. Refuses to print a delta
                         #   smaller than the instrument's own noise, and cross-checks frame
                         #   time two ways. Phase boundaries come from the tracker's own
                         #   `[power] PHASE` markers, never from an assumed duration -- VOT
                         #   staging reads up to 1.27 GB before frame 0.
                         #   The tracker's `--power-pause <s>` (HOST-ONLY, no flagstamp, >= 5 s)
                         #   is what creates the graph-idle windows: it holds twice, before
                         #   frame 0 and after the last frame, and the SECOND hold is the
                         #   thermal-drift CONTROL, not a duplicate
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
  scale_oracle_bound.py  # WHAT IS A PERFECT SCALE FILTER WORTH? Board trajectories, tracker's
                         #   centre, ground-truth size, VOT's rule re-applied: +0.0023 R. It is
                         #   what CLOSED the scale direction, and it is cheaper than attribution
  scale_drift_anatomy.py # is the long-horizon scale error a random walk, and what feeds it?
  analyze_plio_vcd.py    # the PL->AIE AXIS handshake from plio_probe.vcd: TVALID/TREADY, beats
                         #   completed, where it stalled
  vot_hold_budget.py     # frames before a frozen window loses the target, from groundtruth
  vot_mask_stat.py       # reads `mask_ebox` -- THE mechanism check for FILTER_MASK. Keyed on
                         #   (sequence, job, frame); EXCLUDES the -1 rows; reports at-init and
                         #   per-frame-index, never a pooled mean
  check_ebox_crosscheck.py  # the board's filter_box_energy_fraction against the offline
                         #   box_energy_fraction on the SAME H, plus 5 injected mutants
  offline_multistart.py  # runs a HOST-SIDE tracker under the BOARD's multistart protocol and
                         #   writes the board's trajectory format, so vot_ingest.py scores it
                         #   exactly like a sweep. THIS IS WHAT VALIDATED THE SCORING PATH:
                         #   CSRDCF reproduces its published row to 0.008 EAO (R-12), and the
                         #   `oracle` control returns R = 1.0000 EXACTLY -- run it first, always.
                         #   Controls: oracle / oracle-lag1 / static / opencv:<kind>-rgb.
                         #   Needs opencv-contrib-python for csrt+kcf. Parallelise per RUN;
                         #   pin cv2.setNumThreads(1) or 30 workers become 780 threads
  eao_window.py          # RE-SCORES EXISTING trajectories under a second EAO window, using the
                         #   toolkit's own analysis. The window is a property of the CHALLENGE:
                         #   [115,755] here, [108,371] on VOT2015. Worth +0.0827 on the shipping
                         #   arm -- 1.39x the whole arm ladder -- so no cross-era EAO comparison
                         #   means anything without it. NOT a VOT2015 score; see M-17
  vot_ar_offline.py      # VOT's failure rule applied to OFFLINE single-start runs. NOT the
                         #   toolkit's AR, and its resolution is MEASURED at ~0.02 in R,
                         #   including one pair it got BACKWARDS. Decides whether an arm
                         #   deserves board time, never whether to accept one.
                         #   Did NOT transfer on pad30
  -- offline benches, no hardware, seconds to minutes --
  mosse_loop_sim.py, bg_pan_sweep.py,
  rgb_vs_gray_holdout.py / rgb_vs_gray_loop.py / rgb_vs_gray_vot.py
                         #   gray vs RGB vs a colour-free control on real VOT video.
                         #   rgb_vs_gray_loop.py also carries the float quantization
                         #   counterfactual, the pooling/resolution arms, and the -warp<N>
                         #   init-perturbation arms with their mutants
  offline_sweep_par.sh   # one process per sequence over all 62, then a merge. --json is a
                         #   read-modify-write, so parallel workers MUST own private files
  merge_grid.py          # one-arm-per-file grid cells -> one JSON with distinct arm names,
                         #   because sigma and eta are global flags, not per-arm suffixes
  grid_stats.py          # paired per-sequence stats for a merged grid: trim, sign test,
                         #   bootstrap. `vot_ar_offline.py` prints POOLED R, which is the
                         #   statistic this project has most often been misled by
  -- documentation, generated or linted; all of these run in seconds --
  thesis_index.py        # @thesis tags -> docs/thesis/code_map.md (make code-map)
  csv2tex.py             # results/*.csv -> booktabs bodies (make thesis-tables)
  check_build_table.py   # documented build defaults == `make print-<KNOB>` (make check-docs)
  check_doc_links.py     # every documented path exists, and this map names every source file
  doc_index.py           # each doc's status header -> docs/thesis/evidence/README.md
                         #   (make doc-index); --check validates the headers without writing
  check_doc_numbers.py   # a measured number in a comment must say where it came from
  figs/fig_perf_history.py   # results/perf.csv -> docs/thesis/figures/perf_history.pdf. Every
                         #   figure script reads a CSV and takes no arguments

docs/
├── engineering/   # the operational detail this file summarises — see its README
├── thesis/        # claims.md, evidence/, results/*.csv, tables/, glossary.md
└── papers/        # Bolme MOSSE, Danelljan DSST
test-sequences/    # VOT sequences + annotations. 17 directories, 16 USABLE, 5971 frames:
                   # `fish4` holds 7 frames and is the one excluded. The annotation
                   # directories are named inconsistently ("car1-annotations" but
                   # "fernando - annotations"); the harness matches them loosely.
```
