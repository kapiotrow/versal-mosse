# CLAUDE.md

Guidance for Claude Code working in this repo.

MOSSE correlation-filter tracker with CNN features on Versal VEK280. Extends the AIE 2D-FFT
tutorial (XD073) with a full object-tracking pipeline. Papers in `docs/` (Bolme MOSSE,
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
| `H_SHIFT` | `11` | cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget — and that independence is what fixed the 2026-08-24 rail. Was 10 until then |
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
| `CONV_IN_CH` | `1` | conv2d input planes. 1 = BT.601 luminance, 3 = RGB. Picks the **weight-buffer layout**, so it drives `AIE_FLAGS`, `GCC_FLAGS` and `ROI_IN_CH` from this one variable. Both arms build and link; see the RGB section |
| `CONV2D_STACK` | `2048` | conv2d's AIE stack, bytes. Applied **only at `CONV_IN_CH=3`**, where the 27-tap chain needs 1344 against the 1024 default and the mapper otherwise refuses to emit a `libadf.a` |
| `BIAS_SCALE` | `roi` | `bias_acc` input scale for `make weights`. **Default changed 2026-08-23**; `127` restores the pre-correction weights. See "The bias_acc correction" |
| `FRAME_RGB_MODE` | `1` | Synthetic scene colour at `CONV_IN_CH=3`. 1 = per-plane tint; 0 = replicate luma — the COLOUR-FREE CONTROL, run on hardware 2026-08-24 (`run_0824_1442`) where it reproduced grayscale bit-for-bit. Inert at `CONV_IN_CH=1`; a real frame source ignores it. Host-only |
| `SCENE_VERIFY` | `0` | Re-colourise the whole frame each push and abort with coordinates on a mismatch. O(frame)/frame — for a short `MODE=bringup` run, never a 200-frame one (`calib_build.sh` refuses that combination). Plumbed 2026-08-24; it was a bare `#ifndef` before, so `SCENE_VERIFY=1` silently built it DISABLED. Host-only |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `PSR_GATE_MIN` | `7.0` | Bolme §3.5. Below it the host HOLDS position and skips `filter_update` + `publish_filter`. `0` disables the threshold test only (structural vetoes remain). Host-only |
| `TARGET_H` / `TARGET_W` | `64` | Target box size, frame px. Host-only |
| `TARGET_PADDING` | `2` | `roi = box × padding`. At 64/2 the ROI is 128 ⇒ resample is 1:1 |
| `MOSSE_SIGMA` / `SIGMA_FROM_TARGET` | `2.0` / `0` | `SIGMA_FROM_TARGET=1` applies DSST's target/16 rule (σ=4 at padding 2) |
| `MOSSE_ETA` | `0.125` | Translation filter learning rate |
| `SCALE_N` / `SCALE_STEP` | `33` / `1.04` | DSST scale levels; `SCALE_N=1` disables the scale filter. **1.04 beats DSST §6.1's 1.02 on hardware** (IoU 0.807 → 0.917) |
| `SCALE_ETA` | `0.025` | Scale filter learning rate (deliberately ≠ `MOSSE_ETA`) |
| `SCALE_CONF_MIN` | `2.0` | Scale gate on `conf`. A veto HOLDS the box and SKIPS `scale_update()`. `0` disables the threshold test only. Host-only |
| `SCALE_MIN_REL` / `SCALE_MAX_REL` | `0.5` / `2.0` | Absolute drift bounds vs initial box size. Must still admit `SCALE_TRAJ_AMP` (0.70×..1.30×) |
| `OCCLUDE_MASK` | `0` | Bitmask over frame index: bit *f* ⇒ frame *f* occluded. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test |
| `OCCLUDE_SQUARE` / `OCCLUDE_START` | `8` / `30` | `OCCLUDE_START` is a warm-up: 30 ≈ 4 time constants at `MOSSE_ETA=0.125`. The scale filter needs ~120 frames to settle |
| `TRAJECTORY` / `TRAJ_AMP_R,C` / `TRAJ_PERIOD` | `0` | 1 = closed elliptical path (absolute ground truth) |
| `SCALE_TRAJ` / `SCALE_TRAJ_AMP` / `SCALE_TRAJ_PERIOD` | `0` | Sinusoidal size envelope |
| `FRAME_TEXTURE` | — | Band-limited background instead of a flat fill |
| `FRAME_NOISE` | `2` | Per-frame sensor noise, PEAK amplitude in LSB, over the ROI. Does **not** fix background lock |
| `BG_PAN` / `BG_PAN_R` / `BG_PAN_C` | `1` / `31` / `47` | Camera pan over the cached background, px/frame. Decorrelates the background 6.6× (swept with `scripts/bg_pan_sweep.py`) but **did not fix tracking** — see the training-target trap. Host-only |
| `VERBOSITY` | `1` | `0` = one compact line/frame (~45 B); `1` = per-frame block, roi_crop/DMA tables on first+last frame only; `2` = everything. Anomalies print at every level. **No longer a diagnostics trade-off** — since 2026-08-24 `track.csv` carries `rails`/`accum_max`, so a `VERBOSITY=0` run is a full budget verdict AND an FPS measurement. Host-only |
| `DUMP_BUFFERS` | `1` | Per-frame binary dumps. **1216 KB/frame, ~2 s/frame**. Set `0` for any run measuring tracking or FPS |
| `CSV_LOG` | `1` | One row/frame to `track.csv` — gate verdict, both PSRs, peak, displacement, `resp00_over_peak`, both boxes, IoU, centre error, scale fields, and (2026-08-24+) `rails,accum_max,fch0_max,h_max`. ~60 B/frame |

Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

### Shift budget — SETTLED: 4-4-4 at `H_SHIFT=11`, validated for BOTH arms

**Closed 2026-08-24.** gray `run_0824_1354`: `rails=0`, `accum` max 52.1%, `response` max 49.0%,
mean IoU 0.9188. RGB `run_0824_1432`: `rails=0`, response max 38.5%. Both 200 frames,
`BIAS_SCALE=roi`, `TRAJECTORY=1 SCALE_TRAJ=1`. **The FFT budget never moved — the fix was
`H_SHIFT` 10 → 11.** The bias correction returned ~2.5× of signal and spent the margin
`H_SHIFT=10` had: at 10 the corrected build railed (`run_calib.log`, `accum` 104% on f173,
response 98% on f187, still growing at f200). `H_SHIFT` is the only knob upstream of **both** the
accumulator and the response — `IFFT_*` reaches only the response, `FFT_SHIFT` moves it two bits
at once — and both needed exactly one bit.

**What made it a validation and not just a passing run:** tracking came back BIT-IDENTICAL on all
199 frames (a uniform rescale cannot move an argmax, so that was the prediction); PSR did not
move (25.92 / 84.06 / 127.36 vs 25.92 / 84.08 / 127.41), which was written down as the falsifier
BEFORE the run because PSR is where a quantization floor would show; and `F_ch` / `H(q15)` are
digit-for-digit unchanged, as they must be — both are upstream of `H_SHIFT`.

**Do not re-centre the response in the 49-64% band.** That band came from a distribution with a
1.30× spread; the corrected build spreads 2.07× at the converged end, so centring the TYPICAL
frame puts the TAIL on the rail — exactly how f187 reached 98%. Size against the tail. The
response now sits at ~28% (gray) / 22% (RGB) typically, and `calib_report.py` will call that
UNDERSHOOT; it is advisory, and PSR is the arbiter.

The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so weight
moves freely between passes (holds to 1.3% across splits). `FFT_SHIFT` stays 4 rather than 5
because that leaves the accumulator at ~1400 instead of ~330 for the same response.

Retired points: 4-5-5 (total 16) undershot 6-11×. 5-3-4 (total 17) gave 0.4%. 4-2-2 (total 12)
peaks at 56% on frame 1, then rails from frame 15 and sign-flips to −32768, holding forever on
`NEGATIVE_PEAK`. 4-2-1 was never validated past frame 1. `IFFT_ROW_SHIFT=0` is unsafe at ch16.

**Four rules this budget cost real time to learn:**
1. **The response GROWS as the filter converges** — a budget validated at `ITER_CNT=2` is not
   validated. The 08-24 rail appeared at f173 and the 98% peak at f187, so for `H_SHIFT` even
   "≥20 frames" is not enough: use the full 200.
2. **Twice an offline model set this budget and hardware overturned it.** Both times the model
   was self-consistent and its *premise* was wrong (see the frame-buffer-seeding entry under
   Correctness traps).
3. **Never size this budget against railing before checking `mean_prev` is seeded** — two budget
   hunts chased a frame-0 DC pedestal, not a scaling problem.
4. **Do not size from early frames.** RGB's response reads ~1.03× of gray at f1-4 and 0.785× once
   converged; the weights-derived estimate (0.685–0.790×) was right and the 4-frame read was not.

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
128-bit: `mosse_graph.h:121` uses `plio_32_bits` because a 128-bit PLIO delivered one beat per
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

VEK280 `xcve2802-vsvh1760-2MP-e-S`, 12 GB LPDDR4; AIE core clock 1 GHz
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

## Current status (2026-08-24)

**Best hardware FPS: `runs/run_0821_1725.log`, 26.29 ms/frame = 38.04 FPS** —
`MEMTILE_TRANSPOSE=1 ROI_CROP_PIPELINE=1 CMUL_SPLIT_ACCUM=1 TAIL_PARALLEL=1`. That run predates
the `bias_acc` correction, so quote it for SPEED only; the tracking numbers below supersede it.

**CALIBRATION CLOSED, AND RGB RUNS ON HARDWARE.** Five 200-frame runs on 2026-08-24 settled
both. All at 128×128 ch16, `BIAS_SCALE=roi`, 4-4-4, `TRAJECTORY=1 SCALE_TRAJ=1`, `VERBOSITY=1`:

| run | arm | `H_SHIFT` | rails | mean IoU | PSR min/mean | note |
|---|---|---|---|---|---|---|
| `run_calib` | gray | 10 | **1** (f173) | 0.9188 | 25.92 / 84.08 | `accum` 104%, response 98% |
| `run_0824_1354` | gray | **11** | 0 | 0.9188 | 25.92 / 84.06 | **the comparator** |
| `run_0824_1426` | RGB | 11 | 0 | — | — | 5-frame bring-up, `SCENE_VERIFY=1` |
| `run_0824_1432` | RGB tint | 11 | 0 | 0.9173 | **42.65 / 100.44** | colour arm |
| `run_0824_1442` | RGB luma | 11 | 0 | 0.9188 | 25.69 / 84.83 | colour-free control |

Those five are `VERBOSITY=1`, so their frame times are console-bound (62.3 ms). **RGB's frame
rate is 28.58 ms = 34.99 FPS** (`runs/run_0824_1457.log`, `VERBOSITY=0`) against gray's 38.04 —
**a 8.0% cost for colour, not the 21% the offline model predicted.** See "RGB costs what the
HOST pays" below.

The full chain runs on real hardware at 128×128 ch16 on the real conv path: roi_crop → PatchIn →
conv2d → B1 → row FFT → transpose → col FFT → cmul(H_SHIFT) → B2 → IFFT rows → transpose →
IFFT cols → response → PSR gate → filter update → scale update.

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

### Next, in order

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

**`SCALE_STEP=1.04` confirmed on hardware** (`runs/run_0820_1513.log`, otherwise identical to
`run_0820_1418`): mean/worst IoU 0.807/0.579 → **0.917/0.833**, max box error 31.4% → **9.6%**,
mean/worst centre error 2.47/11.07 px → **1.30/3.52**. **Centre error fell 3.2× from a size-only
change** — independent confirmation that position error was downstream of the scale error.

**The detector proposed only −1, 0 or +1 over 199 frames** (174/13/12) — never ±2, at either step
size, and still true on both RGB arms. The decisions overlap heavily in error (idx 0 spans
−5.3%..+9.6%), so near zero it is a noisy estimator, not a dead zone with a clean threshold.

**`SCALE_CONF_MIN` blocks legitimate large corrections.** On the sim's `step` arm the detector
proposes the correct `idx=-14` after a jump and the gate vetoes it as `LOW_CONF` for four frames;
the box then walks at 2%/frame with a 37% peak, where bypassing the gate corrects it in ONE
frame. **`conf` cannot distinguish "wrong proposal" from "big correct correction"** — both match
the model poorly, for the same reason. Exonerated for smooth envelopes; it will bite on any
abrupt scale change.

**Where it stops.** `SCALE_ETA` does not help (sim at a=1.04: 8.6/10.3/9.2% at 0.025/0.05/0.1);
`SCALE_N=65` gives 7.0% against 8.6% for **3.9× the cost**. ~8-10% box error is this filter's
practical floor — and it is what the worst IoU frames on EVERY 2026-08-24 run are: f125 box
54.71 vs truth 50.00 with centre error 0.06 px. The next gain needs a different estimator, not a
tuning change.

**Calibration honesty: the sim predicted a=1.04 well and a=1.02 badly** (12.6% → 8.6% predicted;
31.4% → 9.6% measured), and it recovers when the envelope turns where hardware never did —
**unexplained**; position error and background pan were both tested and ruled out. **Trust the
ORDERING; do not quote the sim's absolute magnitudes as the board's.**

**The truth rate matters and the test bench chose it.** `SCALE_TRAJ_AMP=0.30` over
`SCALE_TRAJ_PERIOD=200` shrinks the target at ~0.94%/frame — under half of one 2% scale level —
so on most frames the correct proposal rounds to 0. A slower envelope would hide this entirely.

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
  (`mosse_tracker.cpp:3966`), while `accum` and `response` are bank-wide. So "F_ch looks
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
  Fixed 2026-08-20. The defect: `filter_update()` was passed `g_target` — the Gaussian centred at
  (0,0) — while `g_F_all` is the patch cropped at the **pre-update** position, where the target
  sits at `(dr,dc)`. Every accepted frame taught "a patch with the target at (dr,dc) peaks at
  (0,0)", compounding at `eta=0.125` until the zero-shift peak won. Mean IoU 0.1708, 181 of 199
  frames lost.
  **The sign, derived rather than guessed.** Let `Q_t` be the patch cropped exactly ON the
  object; the patch actually held is `P_t = Q_t` shifted by `+d_t`. Correlation is shift
  equivariant, so `resp(P_t) = resp(Q_t)` shifted by `+d_t`, and an on-target patch must peak at
  0 ⇒ **G is centred at `+d_t`, the SAME sign as the detected peak.** Training at 0 teaches
  `resp(Q_t) → peak at −d_t`; the next frame's contribution lands at `d_{t+1} − d_t`, which under
  near-constant motion is exactly (0,0). That derivation **predicts analytically** the observed
  law `resp00_over_peak ≈ 1-(1-eta)^k` (0.125/0.234/0.330/0.414/0.487/0.551 predicted vs
  0.08/0.29/0.42/0.38/0.65/0.69 measured) — i.e. it is governed by the LEARNING RATE, not the
  scene.
  **Fix**: `filter_update()` gets a per-frame `g_target_shift` from
  `gaussian_target_spectrum(..., psr_abs.dr, psr_abs.dc)`. By the shift theorem this IS "re-crop
  at the new position", exactly and for free. `filter_init()` on frame 0 keeps the centred
  `g_target` — that crop really is centred. The exact alternative is a real re-crop (Bolme/DSST
  do this) at 16 more roi_crop launches ≈ 77 ms/frame; the only difference is that the Hann
  window stays centred where the crop was taken.
  **A SINGLE UPDATE CANNOT SEE THIS DEFECT** — `gen_filter_golden.py`'s one-shot check passed
  throughout — which is why both regression tests are closed loops:
  `scripts/mosse_loop_sim.py` (128×128, 60 frames: centred reaches `resp00/peak` 0.764 and ends
  39.7 px off; shifted holds 0.207 flat, 0.00 px) and `run_training_target_tests()` in
  `make test_host`.
  **Assert the shape, not an absolute level.** In the 32×32 test both arms start at 0.39 — that
  scene's own zero-shift autocorrelation, not a defect. The 0.3 healthy ceiling is calibrated for
  the 128×128 ch16 geometry and does **not** transfer. What is geometry-independent is that the
  defect makes the ratio GROW at the learning rate while the fix leaves it flat.
- **Background lock was the WRONG explanation for the above** — worth keeping because the
  measurements are real and the mechanism does exist. `fill_background()` is cached and only the
  dirty rect is restored, so outside the target the frame repeats to the LSB, and a DCF fed a
  perfectly repeating background correlates with it at exactly zero shift. Measured 2026-08-17:
  the static peak was worth 69-86% of the true one and won 21 of 48 frames, each win costing a
  permanent ~9.4 px offset (centre error 1.35 → 9.56 → 87 → 292 px) **while PSR read 24-35
  throughout**.
  **`FRAME_NOISE` is not the fix**: independent additive noise cannot decorrelate a static
  pattern, and it inflates numerator and shared denominator alike. It appeared to work on
  08-17 only because the frame buffer was not yet seeded and the ROI was mostly zeros.
  **`BG_PAN` is the right instrument** and measurably works — but it changed the tracker not at
  all, which is what refuted this explanation. **Sweep the magnitude against the texture's
  wavelengths, not in pixels** (`scripts/bg_pan_sweep.py`, seconds, no hardware): corr@0shift is
  +0.60 / +0.61 / +0.64 / +0.54 / +0.31 / **+0.09** / −0.25 at 0,0 / 3,5 / 7,11 / 15,23 / 23,37 /
  **31,47** / 47,71 px per frame. The obvious guess of 3-5 px/frame is worthless — the texture's
  shortest wavelength is 180 rows. Re-run the sweep after any change to `fill_background()`,
  `FRAME_TEXTURE` or the ROI size.
  **`fill_background()` rounds its frequencies to WHOLE CYCLES per frame** so the pan wraps
  seamlessly: with continuous frequencies the row wrap is a **8.10 LSB** discontinuity against
  0.98 LSB between interior rows, and an artificial edge is exactly what a DCF locks onto.
  Rounded, the seam is 1.02 LSB. 31 and 47 are coprime to 1080 and 1920.
  **This fixes the TEST BENCH, not the tracker** — do not tune the tracker against this artifact.
  On a white-noise background a rigid pan creates a correlation peak at exactly the pan offset.
  **Discriminating background lock from a DC pedestal** (similar look, opposite fixes): a pedestal
  lifts every bin uniformly; background lock is a *localised blob* at the origin with sidelobe
  mean ≈ 0.
- **THE FRAME BUFFER WAS NEVER SEEDED WITH THE BACKGROUND. Fixed 2026-08-18** — one 2 MB `memcpy`
  at startup, which **must** come after `rc_control_cu_probe()` (which zero-fills `frame_bo` by
  design).
  **The interesting part is the correction.** The measured "88.53% of the ROI never written" came
  from an *offline replay*, which assumed the unwritten region held garbage that saturated Stage
  A's int8 rail and inflated σ 4.3×. On hardware that region is **zeros**, so no rail, no σ
  inflation. Hardware before/after: `F_ch` frames 0 and 1 are **unchanged** (1854→2010,
  1893→1894) — including frame 0, the one the filter trains from — and frames 2+ rise 2.55-3.10×.
  The fix is real and worth keeping; the predicted 9.2× response gain never existed, and the
  shift-budget change it justified (4-3-3 → 4-5-5) was wrong.
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
  three independent guards fire on a mismatch: a `#error` in the generated `layer0.h`, a runtime
  tag check in the host before any weight byte is read, and `SystemExit` in
  `check_collapse.py` / `gen_aiesim_vectors.py` / `phase1_sweep.py`. All three tested firing.
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
  backpressure — `TREADY` gates the stream at ~87 ns/beat against conv2d's scheduled ~35 ns per
  4-px beat), and PASS1 achieves **10.9 cycles/output-px, not II=4** (m_axi latency on the four
  scattered bilinear taps). 128×128 recompute is 4× the pixels ≈ 1 ms, which is exactly the
  residual `ROI_CROP_PIPELINE` leaves on channel 0.
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
- **PSR gating (Bolme §3.5)** — four veto reasons reported separately: `ZERO_RESPONSE`,
  `FLAT_SIDELOBE` (sdev 0 — PSR undefined, not infinite), `NEGATIVE_PEAK`, `LOW_PSR`. Only
  `LOW_PSR` is disabled by `PSR_GATE_MIN=0`. On a gated frame the host **holds the position**
  (moving to a noise peak walks the ROI off target permanently) and skips `publish_filter` —
  required, not an optimization, because `filter_quantize_q15` reads `g_energy`, which the
  per-channel loop has already overwritten with the occluder's energies. Proven on hardware:
  PSR 3.90 → HOLD → reacquire with `err=0 px`, frozen filter byte-identical to the ungated run's.
- **`scale_gate()`** — three vetoes reported separately for the same reason. `AT_SEARCH_RAIL` is
  checked *before* `LOW_CONF` because it is the more specific finding when both fire. **The
  update skip is the load-bearing half**, not the box hold. Guarded so a degenerate filter cannot
  veto everything (`n_scales > 2`) and so "the gate never ran" is distinguished from "the gate
  said no". 21 assertions in `make test_host`, driven by values hardware actually produced.
- **DSST 1-D scale filter** = the existing filter at `rows = 1`; `gaussian_target_spectrum(G,1,S,
  σ,0,0)` degenerates cleanly. Defaults from §6.1 except the step: S=33, η=0.025, σ_s=S/16,
  template capped at 512 px. A single application under-corrects (+3 where +5 is exact) because
  the response is a discrete peak smoothed by a σ=S/16 target — the property to assert is that
  **repeated** application converges monotonically, which it does to 0.5%.
- **Console gating (`VERBOSITY`) details that mattered.** (1) The `VP1`/`VP2` macros are
  `if (VERBOSITY >= n)` on a compile-time constant, so format strings are **dead-code-eliminated,
  not merely skipped** — verify with `strings` on the ELF, a stronger check than reading a log.
  (2) `dma_accumulate_frame()` had to be split out of `dma_report_frame()`; the printer was also
  the accumulator, so gating the print would have silently turned the CUMULATIVE report into a
  two-frame report. (3) **Anomalies print at every level** — a railed bin, a PSR or scale HOLD, a
  peak-definition disagreement, a negative peak. (4) `VERBOSITY=0` still prints one line per
  frame deliberately: gating to nothing would delete the instrument `picocom | ts` needs.
- **`track.csv` carries `rails,accum_max,fch0_max,h_max` since 2026-08-24, and that retired the
  "`VERBOSITY=1` is load-bearing" rule.** The scan always ran — `report_cint16` gates the PRINT,
  not `scan_cint16` — so the numbers were computed and discarded, and a healthy `VERBOSITY=0` run
  yielded no amplitude data at all. **Validated by reproducing a known answer**: the
  `VERBOSITY=0` run's CSV gives `F_ch`/`accum`/`response`/`H(q15)` digit-for-digit identical to
  the `VERBOSITY=1` run's console, at 28.58 ms/frame instead of 62.67. `calib_report.py` falls
  back to the CSV when a log has no `[diag]` lines and says which source it used; it distinguishes
  "0 rails" from "no rails column", because those must never print the same word.
  `fch0_max` is named for ch0 because `F_ch` is scanned under `if (ch == 0)` — not a bank max.
- **DMA 4258 → 1090 tx/frame** via `FFT_ROW_WS`/`FFT_COL_WS` 2→8; 96% of the traffic was the four
  per-invocation-chunked ports. **DMA is not a bottleneck and the fabric is at spec**: 80 µs/tx is
  per-transaction *overhead*, not bandwidth — 64 B costs 14.4 µs and 128 KB costs 22.8 µs
  (2048× the size for 1.6× the time). The largest transfer achieves **5.76 GB/s**, inside AMD's
  own measured GMIO range.
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
- **Padding ≥2; recommend 2.0 — but the 1.5 verdict is REOPENED.** Held-out at target 64, budget
  4-2-2, ch16: padding 1.5 gives PSR 18.4 / ratio 2.59 / 0.75 px; 2.0 gives 45.7 / 4.97 / 0 px;
  2.5 and 3.0 edge it out but trigger the aliasing detector (bilinear has no prefilter) and 3.0
  clips 3.57% of samples. 2.0 is the only value where the resample stays 1:1. **But that holdout
  used a STATIC scene, where background lock costs nothing** — and background lock is precisely
  what more padding buys more of (target is 27% of ROI area at 2.0, 44% at 1.5). The measurement
  was blind to the failure mode. Re-test under `TRAJECTORY=1` with `resp00_over_peak` as the
  metric before treating 1.5 as settled.
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
- **Channel pruning is moot** with ReLU off, and doubly so at `BIAS_SCALE=roi`, which retires
  the last two structurally dead channels (ch3, ch15) outright. The real
  redundancy is the collapse: it caps the bank at **rank 9** (participation ratio 4.94) and
  leaves ch0/ch9/ch14 collinear up to sign, and collinear channels add exactly coherently. The
  fix is RGB. `check_collapse.py` Q2 used to print "14 independent filters" here — that was a
  count of near-parallel GROUPS, not a rank, and it understated the problem for months.

## RGB features — VALIDATED ON HARDWARE 2026-08-24

Code was complete 2026-08-23; three hardware runs on 08-24 closed it. Build with
`make weights CONV_IN_CH=3` then `ARM=rgb scripts/calib_build.sh`. `H_SHIFT=11` / 4-4-4 carries
both arms — no RGB-specific budget was needed.

### What hardware proved, and how

Three 200-frame arms, one variable moved at a time (`run_0824_1354` / `_1432` / `_1442`):

| | gray (9-tap) | RGB tint | RGB replicated luma (CONTROL) |
|---|---|---|---|
| tracking-identical to gray | — | 121 / 199 | **199 / 199** |
| mean IoU | 0.9188 | 0.9173 | 0.9188 |
| **PSR min / mean** | 25.92 / 84.06 | **42.65 / 100.44** | 25.69 / 84.83 |
| ch0 `F_ch` median | 7398 | 3834 (0.52×) | **623 (0.084×)** |
| rails / gate holds | 0 / 0 | 0 / 0 | 0 / 0 |

**The control is what makes this conclusive, and it is stronger than the offline version.** Same
27 taps, bias, quantization grid and joint normalization, fed three IDENTICAL luma planes: it
reproduces grayscale's decisions **bit-for-bit on all 199 frames** and its PSR lands within 1%.
So (a) the whole RGB datapath — 3-plane Stage A, interleaved wire format, de-interleave, 27-tap
MAC — is validated against a known-good reference, and (b) the **1.65× PSR floor is colour, not
bookkeeping**. This reproduces the offline finding on hardware, including its shape:
**RGB IS A ROBUSTNESS CHANGE, NOT AN ACCURACY CHANGE** — mean IoU actually moves the wrong way
(0.9188 → 0.9173) and the colour-free control is tied for most accurate, exactly as offline.

**ch0's collapse was PREDICTED to 7%.** From the exported weights alone, the input-referred gain
for ch0 is gray 8.72 / RGB-decorrelated 10.87 / RGB-luma-replicated 0.80, i.e. the control should
sit at 0.09× of gray. Measured 0.084×. That is the sharpest confirmation in this file that the
weight-collapse model is right, and it makes ch0's `F_ch` the cheap on-board colour-path test
(see the ch0-only trap under "Metrics that cannot fail a broken tracker").

**Amplitudes.** RGB's converged response is **0.785× of gray on both median and max**, against a
prediction of 0.685–0.790× computed from `‖taps‖₂ / 2^out_shift` summed over the bank. Do NOT
size RGB from early frames: at f1-4 the ratio reads ~1.03× because the filter is barely trained,
and trusting that over the weights model was a wrong call made on this project. `calib_report.py`
flags RGB as UNDERSHOOT (response med 22.2%, max 38.5%) — advisory only: the band is calibrated
for gray at `BIAS_SCALE=127`, and PSR went UP, which is the metric a quantization floor shows in.

### THE SYNTHETIC SCENE IS A WEAK COLOUR STIMULUS — do not read accuracy from it

`FRAME_RGB_MODE=1` tints one luma image per plane, i.e. `plane_p ≈ a_p·luma + b_p`. That is
**rank-1 across the plane dimension**: there is no chromatic texture independent of luma, and a
colour-opponent channel sees a scaled copy of luma rather than real chroma. Hence ch0 lands at
0.52× (between the 0.09× replicated case and the 1.25× decorrelated case) instead of 1.25×.
**A good IoU on this scene is NOT evidence for the VOT result below.** The offline accuracy and
failure-rate numbers stay the claim of record until real video is fed through the board.

### Why RGB — the offline evidence (still the accuracy claim of record)

Reproducible from `scripts/rgb_vs_gray_*.py`:

| measurement | gray | RGB |
|---|---|---|
| feature-bank rank / participation ratio | 9 (hard cap) / 4.94 | 16 / 7.43 |
| held-out Bolme PSR (147 paired evals, car1) | 12.97 | 21.18 (**+1.63×**) |
| VOT supervised failures (16 seq, 5971 frames) | 51 | **42 (−18%)** |
| VOT supervised accuracy | 0.4484 | 0.4642 (+0.016) |
| conv2d scheduled cycles/frame (ch16) | 4.60 ms | 9.19 ms (**2.00×**) |

The offline control scores 55 failures against gray's 51 and PSR 28.67 against 33.86 — it
reproduces gray, as the hardware control now does. Per sequence the failure delta is 6 wins /
8 ties / 2 losses and survives dropping `tiger` (40 → 37). Read accuracy and failures together:
RGB survives 18% longer between resets, so its accuracy is measured on harder frames.

**Why the collapse costs anything.** A 3×3 grayscale kernel lives in 9 dimensions, so 16 channels
CANNOT be independent — the cap is structural, not a property of the pretrained weights. BT.601
guts the four colour-opponent channels: 0/2/9/10 keep 0.32/0.60/0.63/**0.037** of the per-plane
norm against 1.24–1.39 for the achromatic ones, and per-channel int8 then renormalises that
residue to full scale. ch0/ch9/ch14 sit within 2–6° of one line in gray, 59–72° apart in RGB.

**Retired — "RGB is handicapped by its larger `out_shift`."** 27 taps triple `ACC_MAX_THEORY`,
pushing mean out_shift 3.69 → 4.25 (confirmed on the shipping export). But forcing gray's shifts
onto RGB (`--match-shift`) makes it **worse**, 42 → 53 failures, with 0.0000% saturation at all
three clip sites — so it is not clipping. Hypothesis: the per-channel re-weighting of the shared
denominator `B = Σ|F|²`. Not verified, and hardware did not need it.

### The datapath, end to end

**The wire format is pixel-interleaved, the line buffer is planar.** roi_crop must send
interleaved (planar would need whole planes resident); conv2d de-interleaves into three 3-row
buffers as it unpacks, which costs index arithmetic and ~1.2 KB. Without it `load_unaligned_v`
gathers `[R G B R G B…]` and the MAC loop needs shuffles. Three int32 words carry exactly four
RGB pixels: `R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3`.

**`roi_crop` Stage A**, at `ROI_IN_CH=3` (driven from `CONV_IN_CH`): all three planes share one
geometry and one set of bilinear weights — only the `+p` byte offset differs — and the frame is
interleaved in DDR, so one source pixel's three taps are contiguous. Normalization is **JOINT**:
one mean and one `inv_q` over all 3·pr·pc samples. Per-plane statistics would equalize the planes
and delete exactly the chromatic contrast RGB exists for — silent, and self-defeating. This
matches `rgb_vs_gray_holdout.stage_a_rgb`, the model the 51 → 42 result was measured with.
Scratch buffer 16 → 48 KB. `sum_x2` peaks at 2.111e14 against `ap_uint<48>`'s 2.815e14, so **a
fourth plane would not fit** and the reference asserts the width.

**The host keeps the scene in LUMA and colourises on the way out.** Every scene function —
background, pan/dirty-rect restore, target injection, noise, occluder — and `scale_extract`'s 33
crops are single-plane and stay that way; one pass expands the touched rect into the interleaved
buffer. At `CONV_IN_CH=1` there is no second buffer and no copy, so the shipping path pays
nothing. `scale_extract` reads luma, so **the DSST scale filter needs no recalibration at all**.
The invariant is that every luma write reaches `scene_touch()`; miss one and the device reads
last frame's colour there, which looks like a slightly worse tracking result rather than a bug.
`SCENE_VERIFY=1` re-colourises the whole frame and aborts with coordinates.

**The RGB conv2d stack — fixed 2026-08-23.** `make graph CONV_IN_CH=3` used to fail at the
link-stage stack check: `Stack size requirement of total (1344 + 0) bytes for 15_0 exceeds the
allotted stack size of 1024 bytes`, producing NO `libadf.a`. The per-kernel *compile* succeeds
either way, which is why the cycle schedules below were readable from a build that never linked —
and is exactly why it went unnoticed. **Confirmed pre-existing by counterfactual**: the branch
with its original literal weight offsets gives the identical 1344 bytes, so it was not an
artifact of the layout refactor. Fix: `stack_size(conv2d) = CONV2D_STACK` (2048) in
`mosse_graph.h`, applied **only** at `CONV_IN_CH=3`. No kernel arithmetic changed. **Verified
from the linker log both ways**: the RGB build allocates `MG(15,0) size: 0x800` while every other
node stays `0x400`; the grayscale build allocates `0x400` everywhere. The RGB schedules are
byte-for-byte the pre-fix ones, which is the correct outcome for a memory-allocation change. The
cause is the 27-tap chain — three planes of live vectors plus the fixed post chain (downshift /
clip / B1 / two Hann multiplies, each an `int32 × CONV_VEC` vector) spilling where nine taps did
not.

**The RGB branch is VECTORIZED, not scalar** — 27 `aie::mac` with `load_unaligned_v`. The
`static_assert(CONV_IN_CH == 1, "vectorized path is grayscale-only")` guards the SEPARATE
grayscale block, which the RGB branch returns before reaching. Reading its 27 hoisted `int8_t`
*weight* scalars as a scalar datapath is an easy mistake and makes the 219-cycle schedule look
far worse than it is.

### Testing — and why each suite means anything

Every RGB suite is **mutation-tested**, because a passing test on a path with no prior coverage
is worth nothing until it has been shown to fail.

| suite | coverage | mutants caught |
|---|---|---|
| `make test_roi_crop` | 17 gray + 8 RGB cases, zero tolerance | per-plane mean, planar scratch store, dropped plane index: 6-7 of 8 RGB, 0 of 17 gray |
| `make test_scene` | interleave, Q8 gains, saturation, clipping, missed `scene_touch()` | 8 of 8 |
| `make x86sim_check … s6rgb` | the real 27-tap kernel vs `simulate_conv2d`, 16384/16384 | de-interleave 48.4%, dropped MAC 22.8% |

The survivors are informative, not gaps: `rgb_flat` survives everything (var 0 → all zeros
regardless), and a dropped plane index survives `rgb_gray_control` because with three identical
planes it genuinely IS a no-op. **A suite built only from replicated-luma frames would have
caught none of that mutant** — which is why the RGB test frames use decorrelated planes.

`s6rgb` writes to its OWN directory: overwriting `s6` would silently feed RGB vectors to a
grayscale check, and the two are indistinguishable from outside. Its patch comes from
`roi_crop_ref.stage_a_rgb` — the exact integer Stage A — not from s6's float shortcut, which
differs from the kernel on 40.9% of samples. **Only ONE RGB scenario exists, on purpose**: RGB
changes conv2d and nothing downstream (`N_CHANNELS` is conv2d's OUTPUT count), so s0-s4 would
test identical arithmetic with three times the stimulus.

The colour pass is its own translation unit (`scene_colour.{h,cpp}`, no XRT header) for the
reason `mosse_filter` is — off-board testability. `verify()` and `colourise()` share
`colourise_rect()`, because a verifier carrying its own copy of the rule proves only that two
copies agree.

### Cost — predicted offline, then measured on hardware

- **conv2d 2.00×**, from the compiler's schedules. The 08-24 RGB build that actually LINKED
  reproduces them byte-for-byte, so the figure no longer rests on a build that never linked:

  | loop | gray | RGB | ratio |
  |---|---|---|---|
  | stream read, per 4 px | 28–31 cyc | 84–87 cyc | **3.00×** |
  | MAC + post, per 16 px | 163 cyc | 219 cyc | 1.34× |
  | per frame at ch16, 1 GHz | 4.60 ms | 9.19 ms | **2.00×** |

  **The stream read goes from 44% of conv2d to 61%** — RGB makes the already-dominant term more
  dominant, because the patch is re-streamed once per output channel. **The RGB MAC loop does NOT
  software-pipeline**: `(algo 1a) -> # cycles: ..219 (exceeds -k 64) -> no folding`, critical
  cycle 200 against gray's 24, resource floor 90. So 219 is a give-up number and a tuned variant
  (smaller `CONV_VEC`, or splitting the 27 MACs) could plausibly beat it. Treat 2.00× as an upper
  bound on the MAC half and a hard floor on the read half.
- **Two traps in reading those schedules.** (1) `aiecompiler` reuses a cached per-kernel object
  when the preprocessed source is unchanged, so a `CONV_IN_CH=1` baseline silently reports NO
  conv2d schedule — `rm -rf $(BUILD_DIR)/Work $(BUILD_DIR)/libadf.a` first. (2) The "conv2d 140
  cyc/16px" figure in the Makefile is the `main_` WRAPPER block, not conv2d's body: it measures
  130/135 identically in both arms, i.e. it does not move when the arithmetic triples.
- **Host cost, MEASURED 08-24** (`run_0824_1432` vs `run_0824_1354`): `roi_crop launch`
  0.993 → 1.456 ms (predicted ≈ +1.1 ms for 3× the bilinear taps), `colourise RGB` 0.31 ms,
  **GMIO unchanged at 11.23 vs 11.14 ms and the same transaction count** — RGB touches none of
  the DMA or the CPU-bound APU tail, as predicted. `frame_bo` 2 → 6 MB.
- **Caching the patch in conv2d's tile does not fit.** At `FFT_ROW_WS=64` the output window is
  32 KB and its ping-pong is 64 KB — the whole tile. A 48 KB RGB patch cache would force
  `FFT_ROW_WS` down, and 64→32 cost 10.2 ms/frame on hardware. Net loss.
- **RGB COSTS WHAT THE HOST PAYS, NOT WHAT conv2d COSTS.** Measured `run_0824_1457` (RGB,
  `VERBOSITY=0`) against `run_0821_1725` (gray): **28.58 ms = 34.99 FPS vs 26.29 = 38.04**, so
  +2.29 ms. conv2d's +4.59 ms of AIE compute **does not appear in the frame at all** — `GMIO`
  total is unchanged (11.133 vs 11.134) and the host's blocking `wait()` is unchanged (4.55 vs
  4.51 ms). The whole +2.29 is host-side and adds up:

  | stage | gray | RGB | delta |
  |---|---|---|---|
  | frame push (`frame_bo` 2 → 6 MB) | 0.472 | 1.385 | **+0.91** |
  | roi_crop launch (3× bilinear taps) | 1.013 | 1.471 | +0.46 |
  | colourise RGB (new pass) | — | 0.338 | +0.34 |
  | `frame_bo.sync` | 0.116 | 0.304 | +0.19 |
  | filter upd+quant | 4.806 | 5.215 | +0.41 |
  | **frame** | **26.29** | **28.58** | **+2.29** |

  **The offline "~30 FPS, bracket 27-34" was arithmetically fine and its PREMISE was wrong** — it
  costed conv2d as if AIE time were frame time. It is not: the frame is 84% CPU-bound, so the AIE
  had slack and absorbed the doubling. Third time a self-consistent offline model has been
  overturned by its premise on this project. **The lever for RGB speed is host memory traffic
  (`frame push`, `colourise`), not the 27 taps.**
- **No alignment obstacle** — the old "does not divide evenly into 16-byte beats" was false twice
  over: the PLIO is 32-bit, and 128·128·3 = 49152 B divides exactly.
  See [[verify-stated-blockers-arithmetically]].

**Calibration is DONE** — see the hardware table above. The gray `BIAS_SCALE=roi` run went
first deliberately (one variable against a known-good comparator); `ARM=rgb` moves the bias
scale, the feature bank and the input scale at once, and a bad result would have been
unattributable. That ordering is why the RGB result was readable on the first try.

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
make weights                       # export layer-1 INT8 weights + hanning table
                                   #   BIAS_SCALE=roi by default since 2026-08-23;
                                   #   BIAS_SCALE=127 reverts to the pre-correction file
make weights CONV_IN_CH=3          # ...as 27-tap RGB instead of the luminance collapse
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
# gray vs RGB vs a colour-free control, on real VOT video. Needs the venv, and the Vitis
# env MASKS it: PYTHONHOME/PYTHONPATH point python at Vivado's build, which has no _ctypes.
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_holdout.py   # frozen filter
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py      # closed loop, 1 seq
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_vot.py       # VOT supervised, all seqs
#   ... --oracle-scale takes box size from ground truth (isolates localisation from the
#   missing DSST scale filter); the two modes BRACKET what scale handling is worth.
#   uv sync --extra weights  installs torch/torchvision (needed for the folded weights).
make scale_sim                     # closed-loop DSST scale sim — reproduces the f130 stall
make test_roi_crop                 # native bit-exact roi_crop test. BOTH arms: 17 gray +
                                   #   8 RGB cases, zero tolerance. A build matching no
                                   #   case exits 2 rather than passing vacuously
make test_scene                    # native test of the luma -> interleaved-RGB pass,
                                   #   including a deliberately missed scene_touch() so
                                   #   the verifier is known to fire, not assumed to
make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0   # bit-exact kernel diff (seconds)
make x86sim_check KUT=cmul   SCENARIO=s7                 # ...same for cmul_accum
make x86sim_check KUT=cmul   SCENARIO=cmul_stress        # ...exercising sat16's rails
#   cmul needs CMUL_SPLIT_ACCUM=0 here — kernel_only_graph leaves cmul.in[2]
#   unconnected otherwise and the x86sim graph refuses to compile.
make x86sim_check KUT=conv2d SCENARIO=s6rgb CONV_IN_CH=3 # RGB conv2d, 27 taps, bit-exact
                                   #   needs `make weights CONV_IN_CH=3` and
                                   #   `make gen_vectors CONV_IN_CH=3` first
make aiesim CMUL_SPLIT_ACCUM=0     # AIE simulator — bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but forces the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine one)
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
│   └── test/                         # native tests + NumPy goldens
│       ├── test_mosse_filter.cpp     # make test_host
│       ├── test_scene_colour.cpp     # make test_scene
│       └── scale_loop_sim.cpp        # closed-loop DSST scale sim (make scale_sim)
├── system_configs/mosse_x1.cfg       # v++ linker
├── profiling_configs/, directives/
└── exec_scripts/run_script.sh
scripts/  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py,
          check_collapse.py, check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py,
          conv_weight_layout.py  # Python mirror of conv_weight_layout.h. EVERY reader of
                              #   layer0_weights.bin goes through it; the tag byte makes a
                              #   layout mismatch loud instead of plausible.
          gen_roi_crop_golden.py, synth_frame.py, sweep_shift.sh, fix_sd_rootfs.sh
          calib_build.sh      # hardware build for a shift-budget or bring-up run: pre-flight,
                              #   then verifies the FLAGSTAMPS against the intended config and
                              #   records calib_cfg.txt. Budget defaults are DERIVED from the
                              #   Makefile (print-%), never copied. ARM=gray|rgb, MODE=budget|
                              #   bringup, H_SHIFT=, FRAME_RGB_MODE=, SCENE_VERIFY=, COMPARATOR=
          calib_report.py     # turns a run's console+track.csv into a budget verdict
                              #   (rails, amplitude early vs converged, IoU)
          bg_pan_sweep.py     # picks BG_PAN_R/C from the texture spectrum, no hardware
          mosse_loop_sim.py   # closed-loop MOSSE, centred-G vs shifted-G, no hardware
          rgb_vs_gray_holdout.py / _loop.py / _vot.py   # gray vs RGB vs a colour-free
                              #   control, on real VOT video. No hardware, minutes.
test-sequences/   VOT sequences + annotations (16 usable, 5971 frames). The annotation
                  directories are named inconsistently ("car1-annotations" but
                  "fernando - annotations"); the harness matches them loosely.
```
