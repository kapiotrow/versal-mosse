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
| `H_SHIFT` | `10` | cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget |
| `FFT_SHIFT` | `4` | Forward FFT shift, per pass |
| `IFFT_ROW_SHIFT` | `4` | Must stay non-zero at high channel counts |
| `IFFT_COL_SHIFT` | `4` | 4-4-4 **validated on hardware** 2026-08-20: 200 frames, rails=0 every frame, peak 49-64% of int16 range. See "Shift budget" |
| `FFT_ROW_WS` / `FFT_COL_WS` | `8` | Rows/cols per FFT invocation — **the DMA transaction-count knob** (2→8 gave 4258→1090 tx/frame) |
| `ROI_CROP_USER_MANAGED` | `1` | 1 = roi_crop driven as a user-managed CU via `xrt::ip` (host writes AXI-Lite, polls the CU's own `ap_done`); 0 = the old KDS `xrt::run` path. **1 is 20.6× on frame rate** — KDS completion costs 503 ms/launch on this board because the CU interrupt is never delivered. Host-only |
| `CONTROL_CU_RUNS` | `8` on `hw` | camera_capture launched N times at startup on the KDS path. With `ROI_CROP_USER_MANAGED=1` this is a **within-run control**, not a diagnostic: it should still pay ~512 ms while roi_crop does not |
| `CONV2D_MODE` | `0` | 0 = real 3×3 conv, 1 = echo passthrough, 2 = synthesize |
| `CONV_VECTORIZE` / `CMUL_VECTORIZE` | `1` | Vectorized kernels, bit-identical to scalar; 0 restores scalar for bisection |
| `CONV_RELU` | `0` | Half-wave rectifier after the output shift. Off — ReLU costs ~25% of the peak/sidelobe ratio |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `PSR_GATE_MIN` | `7.0` | Bolme §3.5 threshold. Below it the host HOLDS position and skips `filter_update` + `publish_filter`. `0` disables the threshold test only (structural vetoes remain). Host-only |
| `TARGET_H` / `TARGET_W` | `64` | Target box size, frame px. Host-only |
| `TARGET_PADDING` | `2` | `roi = box × padding`. At 64/2 the ROI is 128 ⇒ resample is 1:1 |
| `MOSSE_SIGMA` / `SIGMA_FACTOR` / `SIGMA_FROM_TARGET` | `2.0` / — / `0` | σ=2 default; `SIGMA_FROM_TARGET=1` applies DSST's target/16 rule (σ=4 at padding 2) |
| `MOSSE_ETA` | `0.125` | Translation filter learning rate |
| `SCALE_N` | `33` | DSST scale levels; `1` disables the scale filter entirely |
| `SCALE_ETA` | `0.025` | Scale filter learning rate (deliberately ≠ `MOSSE_ETA`) |
| `SCALE_CONF_MIN` | `2.0` | Scale-gate threshold on `conf`, the size-axis analogue of `PSR_GATE_MIN`. A veto HOLDS the box and SKIPS `scale_update()`. `0` disables the threshold test only (structural vetoes remain). Host-only |
| `SCALE_MIN_REL` / `SCALE_MAX_REL` | `0.5` / `2.0` | Absolute drift bounds on the box vs its initial size. Tightened from 0.25/4.0, which never fired. Must still admit `SCALE_TRAJ_AMP` (0.70×..1.30×) |
| `OCCLUDE_MASK` | `0` | Bitmask over frame index: bit *f* ⇒ frame *f* is occluded. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test |
| `OCCLUDE_SQUARE` / `OCCLUDE_START` / `OCCLUDE_PERIOD` / `OCCLUDE_LEN` | `8` / `30` / — / — | Occluder shape and schedule |
| `TRAJECTORY` / `TRAJ_AMP_R,C` / `TRAJ_PERIOD` | `0` | 1 = closed elliptical path (absolute ground truth) |
| `SCALE_TRAJ` / `SCALE_TRAJ_AMP` / `SCALE_TRAJ_PERIOD` | `0` | Sinusoidal size envelope |
| `FRAME_TEXTURE` | — | Band-limited background instead of a flat fill |
| `FRAME_NOISE` | `2` | Per-frame sensor noise, PEAK amplitude in LSB, applied over the ROI. Sensor noise only — it does **not** fix background lock, see `BG_PAN`. `0` restores the pre-2026-08-17 (pathological) static background |
| `BG_PAN` / `BG_PAN_R` / `BG_PAN_C` | `1` / `31` / `47` | Camera pan over the cached background, px/frame. Decorrelates the static background (6.6x, swept with `scripts/bg_pan_sweep.py` — 3-5 px/frame does nothing). Correct and worth keeping, but **it did not fix tracking** — see the training-target trap. `0` restores the static background. Host-only |
| `VERBOSITY` | `1` | **A frame-rate parameter, not a cosmetic one** — the frame time IS the console, see "Frame time". `0` = one compact line/frame (~45 B), use for long runs; `1` = per-frame block, roi_crop/DMA tables on first+last frame only; `2` = everything (pre-2026-08-20 behaviour). Anomalies (railed bin, PSR/scale HOLD, peak disagreement, negative peak) print at every level. Host-only |
| `DUMP_BUFFERS` | `1` | Per-frame binary dumps of F_ch/accum/resp/H_q15. **1216 KB/frame**, ~2 s/frame on the board. Set `0` for any run measuring tracking or FPS |
| `CSV_LOG` | `1` | One row per frame to `track.csv` — gate verdict, both PSRs, peak, displacement, `resp00_over_peak`, both boxes, IoU, centre error, and (since 2026-08-20) `scale_idx` / `scale_conf` / `scale_reason`. ~40 B/frame, flushed every row |

Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

**Shift budget. SETTLED AT 4-4-4, AND THE MAKEFILE NOW DEFAULTS TO IT.** Validated on hardware
2026-08-20 over 200 frames at ch16 (`runs/run_0820_1418.log`, `TRAJECTORY=1 SCALE_TRAJ=1`):
**`rails=0` on every frame** and the response peak at 16157-20994 = **49-64% of int16 range** at
the converged end — the band 4-5-5 undershot by 6-11×. Note `runs/.last_cfg` recorded 4-3-3 for
that run and is **stale**; `build/hw/.../aie.flagstamp` is the authority and reads 4-4-4.
The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so
weight moves freely between passes. Earlier point of reference, hardware at ch1: **5-2-2**
(F_ch 53, accum 70, response 2534, ratio 23.0, nothing railed, correct σ=2 Gaussian).
*(Historical, on why 4-5-5 was wrong:)*
Measured on hardware 2026-08-20, ch16, 20 frames, `TRAJECTORY=1`: nothing rails anywhere
(`F_ch`, `accum`, `response`, `H` all `rails=0` every frame) but the response is **1367 at
frame 1 = 4.2% of int16 range**, and 371-1489 (1.1-4.5%) across the run. The prediction was
25-46%. Six to eleven times under.

**The premise for the +4 bits was refuted by the same run.** 4-5-5 replaced 4-3-3 on an offline
claim that seeding the frame buffer raises the response ~9.2× (|feat| 4.3×, |F| 4.4×, plus
~2.3× from H's Q1.15 grid). Comparing `F_ch` (ch0) between the last unseeded run and the seeded
one — same `TRAJECTORY=1`, same geometry, same `FFT_SHIFT`, so directly comparable:

| frame | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| unseeded | 1854 | 1893 | 746 | 753 | 726 | 623 |
| seeded | 2010 | 1894 | 1901 | 1945 | 1967 | 1929 |
| ratio | 1.08× | **1.00×** | 2.55× | 2.58× | 2.71× | 3.10× |

Frames 0 and 1 are **unchanged — including frame 0, the one the filter trains from**. The
offline replay assumed the never-written region held garbage that saturated Stage A's int8 rail
and inflated σ 4.3×; on hardware that region is zeros (the BO is allocated zeroed and
`rc_control_cu_probe` zero-fills it), so no rail, no σ inflation, no 4.4×. Seeding does help
from frame 2 on — 2.55-3.10×, which is real and worth keeping — but it is not 9.2× and it does
nothing at all where the budget is anchored.

The arithmetic closes: 4-5-5 is 4 bits below 4-3-3, and 1367 × 16 = **21.9k**, inside the
14-26k measured on hardware at 4-3-3. The datapath scale never moved.

**Sizing from the measurement rather than a model.** Scaling this run's healthy-region peak
(~1500, frames 1-6 before the tracker drifts): total 17 → 9%, **total 16 → 18%**, total 15 →
37%, total 14 (4-3-3) → 73%. Going back to 4-3-3 puts frame 1 at 73% *before* convergence, and
the response grows with convergence — that is exactly what killed 4-2-2. 4-4-4 leaves room.
`FFT_SHIFT` stays 4 rather than 5 because that leaves the accumulator at ~1400 instead of ~330
for the same response, and the split is free (invariant holds to 1.3% across splits).

**This is the second time an offline model set this budget and hardware overturned it.** Both
times the model was self-consistent and its *premise* was wrong. Size this budget from a
hardware run of ≥20 frames, or don't change it.

**4-2-1 was wrong and 4-2-2 rails — both are retired.** The old 4-2-1 recommendation
("response 23.9%") came from the offline model plus a 2-frame hw_emu run, i.e. it was only ever
validated at frame 1. Measured on hardware at 4-2-2 (one bit *tighter* than 4-2-1): frame 1
peaks at 18276 = **56% of range already**, then grows as the filter converges at `eta=0.125`
and rails from frame 15 onward (57-72 bins), after which the peak sign-flips to −32768 and the
gate holds forever on `NEGATIVE_PEAK`. At 4-2-1 it would have railed on frame 1.

**The general lesson: the response GROWS as the filter converges, so a budget validated at
ITER_CNT=2 is not validated.** Size it from frames 1-20, not frame 1. 5-3-4 (total 17) remains
retired at the other extreme — response 0.4% of range. `IFFT_ROW_SHIFT=0` is unsafe at 16
channels. **Never size this budget against railing before checking `mean_prev` is seeded** —
two budget hunts chased a frame-0 DC pedestal, not a scaling problem.

## Architecture overview

```
PS (A72) — mosse_tracker.cpp
  Drives all GMIO ports in the per-frame, per-channel loop.
  Runs transpose_inplace(), PSR gating, filter init/update (mosse_filter.cpp),
  and the DSST scale filter.

PL kernels (2)
  camera_capture : zero-fill DDR frame buffer (stub; TODO: MIPI RX)
  roi_crop       : DDR frame → Stage A → 32-bit AXIS (int8) → AIE PatchIn.
                   Bilinear resample of roi_h×roi_w to the fixed patch size with border
                   clamping, log, zero mean, unit L2 × ROI_NORM_Q, int8 quantize.
                   Two reduction passes + a stream-out pass; `recompute=1` on channel 0,
                   channels 1..15 re-stream the cache. All geometry is runtime AXI-Lite,
                   so ROI/box changes need no rebuild of anything but the host ELF.

AIE (single instances, serial per-channel; both custom kernels vectorized)
  conv2d_kernel     : int8 patch → 3×3 MAC → Hanning window → cint16 stream
                      (MobileNet-v3 Small layer 1, INT8, RGB collapsed to grayscale).
                      Stage B1 subtracts mean_prev, which the host SEEDS before frame 0.
  fft2d             : PATCH_COLS-pt row FFT → GMIO → DDR; APU transposes; → PATCH_ROWS-pt col FFT
  cmul_accum_kernel : col-FFT ⊙ H_ch* + accumulate (int32 intermediates, saturating cint16
                      accumulator in DDR)
  ifft2d            : same DDR-transpose pattern; row IFFT + col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop → conv2d, **32-bit** (one int32 = 4 packed int8 pixels). It is 32-bit,
not 128-bit: `mosse_graph.h:121` uses `plio_32_bits` because a 128-bit PLIO delivered one beat
per `readincr`, starving the kernel. The name must match between `mosse_graph.h` and
`mosse_x1.cfg` (`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

### GMIO ports (10: 5 in, 5 out)

| Name | Dir | Purpose |
|---|---|---|
| `gmio_weights` | DDR→AIE | conv2d INT8 weights per channel |
| `gmio_fft_row_out` | AIE→DDR | fft_rows output; APU transposes |
| `gmio_fft_col_in` | DDR→AIE | transposed data → fft_cols |
| `gmio_fft_col_out` | AIE→DDR | broadcast tap: F_ch for the PS filter update |
| `gmio_cmul_in` | DDR→AIE | [H_ch* \| prev_Σ] packed per chunk |
| `gmio_accum_out` | AIE→DDR | updated partial sum |
| `gmio_ifft_row_in` | DDR→AIE | accumulated spectrum → ifft_rows |
| `gmio_ifft_row_out` | AIE→DDR | ifft_rows output; APU transposes |
| `gmio_ifft_col_in` | DDR→AIE | transposed data → ifft_cols |
| `gmio_response` | AIE→DDR | correlation response |

### Per-frame data flow

```
camera_capture → DDR frame
for ch in 0..N_CHANNELS-1:
  roi_crop → PatchIn → conv2d → fft_rows → DDR ; APU transpose
  DDR → fft_cols → cmul_accum → DDR
after all channels:
  DDR → ifft_rows → DDR ; APU transpose ; → ifft_cols → gmio_response
  APU: compute_psr → gate → update pos (or HOLD)
  APU: filter_init() on frame 0, filter_update() thereafter, then DSST scale update
```

## Key design decisions

- **AIE-centric**: all FFT/IFFT/conv/cmul on AIE; PL is only camera_capture + roi_crop; APU
  orchestrates via GMIO DDR round-trips.
- **Serial channel processing**: one FFT2D and one IFFT2D instance reused across all channels.
  Minimal PL/PLIO count at the cost of throughput.
- **Transpose in DDR on the APU** between row and col passes (~64 KB memcpy + reorder).
- **Accumulator in DDR** (128×128 cint16 = 64 KB); on-tile would need a Memory Tile.
- **Filter init/update on PS, with no FFT library.** `mosse_filter.{h,cpp}` implements Bolme
  eq. 10–12 with a *shared* denominator (Danelljan/DSST form — one reciprocal map per frame,
  better conditioned). `F_ch` arrives already transformed via `gmio_fft_col_out` and `G` has a
  closed form, so KissFFT was never needed. The update is ~5% of the pipeline's arithmetic;
  AIE would be the worst home (2 MB of filter state does not fit on-tile).
- **The filter update runs AFTER peak detection** — updating first leaks the current frame into
  its own detection.
- **`mosse_filter.{h,cpp}` includes no XRT/ADF header**, so `make test_host` compiles it with
  system g++ and checks it against a NumPy golden in seconds. The alternative for a sign error
  is an hours-long hw_emu frame.
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
  The 11 achromatic channels agree between conventions to cos > 0.99.
- **Scale estimation is DSST's 1-D filter, not multi-resolution search.** DSST Table 1 beats
  exhaustive on both accuracy and speed, and here exhaustive would push ±30% resampled patches
  through roi_crop→conv2d→FFT every frame, moving `|F|` and therefore the shift budget.

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
`Utilization` column shows the declared budgets, not measured occupancy.

### Per-frame AIE compute (128×128, ch16, from `aiecompiler.log` schedules, post-vectorization)

| kernel | ms/frame | note |
|---|---|---|
| conv2d | 4.1 | 37 → 8.75 cyc/px; the untouched **stream-read loop is now 44%** of it |
<!-- NB: AIE compute is ~6.4 ms of a 181 ms frame. The frame is host-bound, not
     AIE-bound — see "APU per-frame cost" and "Frame time". -->
| cmul_accum | 0.13 | 30 → 2 cyc/element, now pipelined |
| FFT + IFFT chain | ~2.2 (band 1.3–3.5) | **trip counts inferred, not logged** — do not quote as measured |
| **total** | **~6.4** | from ~21.6 before vectorization |

These are the compiler's scheduled cycles — real cycles on an in-order VLIW core absent memory
stalls, trustworthy for sizing but not a profile. Tile map: `15_0` conv2d, `24_1` cmul_accum (spanning `24_0`..`24_2` since `FFT_COL_WS=32`),
`15_1`/`29_0` forward FFT, `14_0`/`22_0` IFFT.

### Frame time — 177 ms/frame, 5.65 FPS, and it is now COMPUTE (2026-08-20)

**Console gating is done as a lever** (`runs/run_0820_1513.log`): 36 B/frame = 3.1 ms, 1.8% of
the frame. 0.88 s → **0.177 s, 5.0×**. Note the last 12.8 ms of console removed bought only
3.4 ms of frame (180.4 → 177.0) — the tty drains concurrently with compute, so removing console
below the compute floor buys nothing. **The frame is now ~177 ms of real compute: 87 ms GMIO +
~90 ms APU work** (transposes, ~2 MB/frame of cmul packing memcpy, filter update). That ~90 ms
has never been measured directly and is the largest unattributed cost in the design.

*The measurement that got us here, kept because the method is the lesson:*

**At 115200 with ~10 KB/frame, the frame time WAS the console, exactly.** Regressing frame
period against console bytes over all 198 frames of `runs/run_0820_1244.log` (`DUMP_BUFFERS=0`):

```
console 9018..10055 B/frame, frame 0.784..0.880 s
slope     = 92.5 us/byte      (115200 8N1 = 86.8 us/byte)
intercept = -51 ms            i.e. zero, or negative
```

Slope is the baud; **the intercept is zero**. The 87.4 ms of GMIO, the 17 in-place transposes,
the ~2 MB/frame of cmul packing memcpy and the filter update are all already hidden behind the
tty drain. So the compute floor is the measured GMIO total, **≥87 ms ≈ 11 FPS**, and no change
below the console moves the frame at all until the console is gated.

Where the 10 KB goes — 79% is instrumentation for problems that are now closed:

| bytes | ms | what |
|---|---|---|
| 3792 | 329 | `[ch N]` progress, 6 lines × 16 ch |
| 3196 | 277 | roi_crop timeline + per-call tables, reprinted **every** frame |
| 1450 | 126 | psr / gate / diag |
| 965 | 84 | DMA per-port table, **every** frame |
| 652 | 57 | tracking result + rest |

**The 0.400 s/frame figure recorded earlier on 2026-08-20 does not reproduce** — `run_0820_1223`
and `run_0820_1244` both give 0.88/0.78 s at `DUMP_BUFFERS=0` (`run_0820_1112`, at 8.25 s, is
the dumps run). Treat 0.88 s as the number. The `ROI_CROP_USER_MANAGED=1` win below is
unaffected: it is measured from the RC_* timers, not from the frame period.

**`gmio_fft_row_out`'s 286 µs/tx is very likely the weights lock-step, not the FFT.**
`gmio_fft_col_out` drains the same 256 × 4096 B from an AIE output in a structurally identical
async/wait loop at **18.2 µs/tx** — 15× less. The difference is the interleaved weights feed:
per firing the host does `async(row_out); async(weights); wait(weights); wait(row_out)`, so all
256 firings pay a full host↔AIE round trip. It is not roi_crop either — `start→drain` is
4.78 ms/channel and **the same** for ch0 (`recompute=1`, Stage A runs) as for ch1-15 (cached
re-stream), against 0.36 ms for roi_crop's stream pass in the VCD. Note also that the host sends
the **same 64 bytes 16 times per channel** (the `async` offset is `ch * WEIGHT_CH_BYTES`, no `k`
dependence) — 256 tx/frame of identical data, purely because ADF acquires an `input_buffer` per
firing.

| stage | ms/frame | how |
|---|---|---|
| **console (115200 baud, ~10 KB)** | **~880** | measured, 198-frame regression, intercept 0 |
| GMIO total (1090 tx @ 80.2 µs) | 87 | measured, DMA_T — the floor once the console is gated |
| — of which `gmio_fft_row_out` | 73 | 286 µs/tx; this IS the per-channel drain |
| **roi_crop launch path** | **0.085** | measured, RC_* timers |

The drain and the GMIO are the same thing, not two costs: `gmio_weights` (0.23 ms/ch) +
`gmio_fft_row_out` (4.58 ms/ch) = 4.81 ms, and the `TL_DRAIN` mark lands at 4.75-5.45 ms. No
double-counting.

**How the 503 ms was found and killed.** Four measurements, each of which the previous one made
possible — the sequence is the lesson:

1. `poll(state)` costs 503.4 ms and the `wait()` after it costs **2 µs**. So `wait()` was never
   the problem; it was blocking on a command state that had not flipped. *This retired the
   planned fix (replace `wait()` with a spin) before it was built on.*
2. `drain → poll` is 503.4 ms against a 4.8 ms `start → drain`. 99% of the cost lands after the
   CU consumed its last AXIS beat. The drain mark is an upper bound on completion that does
   **not** go through the driver, which is what makes it decisive.
3. camera_capture — no AXIS port, ~6 µs of work — pays the same 512 ms, and **1080 rows costs
   the same as 1 row** (512.9 vs 514.0). Fixed cost, scheduler-wide, no PL explanation survives.
4. `/proc/interrupts` reads **0** on both zocl IRQs across every run, while the CU's own
   registers read `GIER=1, IER=0x3, ISR=0x3`. See "the CU interrupt is never delivered" under
   Infrastructure.

**Everything reachable from userspace was tried and none of it moved the number**: `poll_threshold`
1000000 (503.40), a hand-cleared ISR (503.40), `Runtime.ert_polling` (503.40, and see the note
below on how that one was mis-tested). The fix was to stop asking KDS.

**The within-run control is the proof.** With `ROI_CROP_USER_MANAGED=1`, camera_capture stays
on KDS and still pays 509-512 ms in the *same run, same process, same driver* where roi_crop
pays microseconds. Keep `CONTROL_CU_RUNS=8` until that stops being interesting — it costs 4 s
and it is stronger evidence than any before/after across runs.

After the fix, `drain → poll` is **0.019-0.027 ms** and the spin exits on its **first** read
every time (16 poll iters / 16 calls) — the CU finishes inside the drain loop. If the drain
ever shrinks (`FFT_ROW_WS` 8→16 halves it) the spin will start spinning for real; the 60 s
bound is what keeps that safe.

**DMA is not a bottleneck and the fabric is at spec.** 80.3 µs/tx average, but that is
per-transaction *overhead*, not bandwidth: a 64-byte transfer costs 14.4 µs and a 128 KB one
costs 22.8 µs — 2048× the size for 1.6× the time, i.e. latency-bound. The largest transfer
achieves **5.76 GB/s**, inside AMD's own measured GMIO range (5.08 GB/s at 1 port, 10.25 GB/s
at 32). Actual bytes moved are 6.27 MB/frame ≈ **1.1 ms of real transfer** — 1.2% of the 88 ms.
The rest is 1090 × ~14 µs of fixed cost. `FFT_ROW_WS/COL_WS` 8→16 would halve the four 256-tx
ports; now that the frame is 0.4 s rather than 8.26 s, those ~7 ms are finally visible.

`gmio_fft_row_out` at 287 µs/tx is a 15× outlier over its sibling output ports (17.7-19.6 µs)
because its `wait()` is interleaved with the weights feed and therefore absorbs AIE pipeline
latency, not because the port is slow. `DMA_T` times `async` **and** `wait` together — split
them before drawing conclusions from any single port.

### Instrumentation lessons (these earned their keep)

- **Timestamp the console instead of adding timers.** `picocom … | ts '%H:%M:%.S' | tee log`
  needs no rebuild, and the design already prints 128 per-channel markers per frame. That is
  what localised the 7.66 s to a single interval in one run, after two wrong inferences from
  reasoning-by-elimination. Do this *before* writing any in-code profiler.
- **Measure the total and print the residual.** A profiler that does not account for the whole
  frame lets you conclude confidently and wrongly — twice here (the dumps were inferred to cost
  ~9.4 s; measured, they cost ~2 s).
- **Instrument the two candidate mechanisms in ONE run, and let the log print the verdict.**
  The `poll(state)` / `wait()` / `wait#2` split cost one hardware run and **retired the planned
  fix before it was built on** — `wait()` returned in 2 µs, so replacing it with a spin would
  have changed nothing. Hardware access is the scarce resource; a measurement that can only
  confirm your hypothesis is worth less than one that can also kill it.
- **Two independent instruments beat one instrument twice.** `/proc/interrupts` reading 0 is
  consistent with "no interrupt raised" *and* with "raised but never delivered". The CU's own
  `ISR=0x3` (toggle-on-write, so only a handler clears it) discriminates them outright: the
  interrupt was raised and no handler ever ran. Neither reading alone would have justified
  abandoning the platform-level fix.
- **A control CU is worth its 4 seconds.** camera_capture — no AXIS port, ~6 µs of work — paying
  the same 512 ms is what turned "roi_crop is slow" into "any CU completion is slow", and after
  the fix the *same* probe in the *same run* still paying 512 ms is what proves the fix is the
  fix. Keep a known-good comparator on the old path when you change a mechanism.
- **`make test_roi_crop` and the HLS report cannot see this class of bug.** Both are correct
  about the datapath. The cost was in the launch path, between two `printf`s, untimed.

## RGB features (not implemented; the old blocker was a miscalculation)

`export_weights.py` justified grayscale by claiming RGB "does not divide evenly into 16-byte
beats". Both halves are false — the PLIO is 32-bit, and 128·128·3 = 49152 B divides exactly
(one row = 384 B = 24 beats), also at 64×64 and 256×256. There is no alignment obstacle.

Actual cost and constraints:

- conv2d 4.1 → ~9.0 ms/frame (9→27 MACs/px, but load-bound so ~2.1×, not 3×).
- **Interleaved layout is forced** — conv2d keeps a 3-row sliding window; planar would need two
  whole planes resident (32 KB).
- `roi_crop`: patch cache 16→48 KB, DSP 44→~132, both trivial; but Pass 1 is measured at
  **10.9 cycles/output-px** (not the II=4 the source comment claims — m_axi latency on the four
  scattered taps), and RGB triples the taps. Treat 10.9→~33 cyc/px as a floor — HLS may re-bank
  the BRAM array. Even so this is ~0.4 ms/frame, i.e. noise until `crop_run.wait()` is fixed.
- **Stage A's normalization must become JOINT across the three planes.** Normalizing each plane
  independently equalizes them and destroys the chromatic information RGB was for. Silent and
  self-defeating if done wrong.
- **Weight-buffer layout collision**: 27 weights occupy `[0:27]`, overrunning `out_shift`[9],
  `bias_acc`[10:14], `dequant_scale`[14:18] and `mean_prev`[18:22]. The layout is duplicated in
  four files — a coordinated edit.
- Unchanged: `N_CHANNELS` (conv2d's *output* count), the FFT→cmul→IFFT loop, the shift budget,
  `H_SHIFT`, the filter state, the DMA count.

**Sequencing.** (1) RGB's joint normalization changes the effective input scale, so it forces
`bias_acc` to be re-derived and the shift budget re-swept — fold it into any calibration work
rather than doing it after. (2) The accuracy premise is unproven: Danelljan Fig. 3's 52.1-vs-37.0
is *CNN features vs raw intensity*, already banked here; the RGB-vs-gray-input delta is
unquantified, and the nearest proxy (colour names, 49.7) is *below* conv layer-1. (3) Do the
DMA-cost measurement first — it could force an architectural change.

## Weight export

`make weights` → extracts torchvision mobilenet_v3_small conv1, folds BatchNorm, collapses
RGB→gray by luminance, symmetric per-channel INT8 quantization. Outputs
`design/aie_src/weights/layer0_weights.bin` (16 × 64 B), `weights/layer0.h`,
`design/aie_src/hanning_128.h`.

`scripts/check_collapse.py` is the front-end diagnostic — four checks, no hardware, seconds,
where the alternative for each is an aiesim or hw_emu run. **Re-run it after any change to
`export_weights.py`, `ROI_NORM_Q`, or the collapse.** Q1 collapse convention (needs torch);
Q2 linear diversity of the kernels; Q3 `bias_acc`/`out_shift` sanity (input-independent — trust
this one); Q4 post-ReLU maps through conv2d's exact integer datapath (patch-specific).
`--skip-torch` gives Q2-Q4 without torch.

## Build commands

```bash
make weights                       # export layer-1 INT8 weights + hanning table
make gen_vectors                   # generate aiesim test vectors
make graph                         # compile AIE graph only
make test_host                     # native unit tests for filter/PSR/scale/scale-gate/training-target/
                                   #   fusion/scale-reuse/real-DFT/conversions (seconds).
                                   #   Builds and runs the suite TWICE — the second time with
                                   #   -O3 -march=native -ffp-contract=fast, because the board's
                                   #   compiler contracts mul+add into FMA by default and a
                                   #   bit-exactness claim proven only at -O2 is proven on the
                                   #   wrong machine. That build caught a real one.
python3 scripts/mosse_loop_sim.py  # closed-loop MOSSE regression, centred-G vs shifted-G
make scale_sim                     # closed-loop DSST scale sim — reproduces the f130 stall
make test_roi_crop                 # native bit-exact roi_crop test, 17 cases, zero tolerance
make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0   # bit-exact kernel diff (seconds)
make x86sim_check KUT=cmul   SCENARIO=s7                 # ...same for cmul_accum
make x86sim_check KUT=cmul   SCENARIO=cmul_stress        # ...exercising sat16's rails
make aiesim                        # AIE simulator — NOTE: bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but forces the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine one)
make kernels / xsa / application / sd_card
make sd_card TARGET=hw
make run_emu LAUNCH_HW_EMU_EXEC=1
make cleanall
```

**Verify the flags reached the build before any expensive run** — flag-only changes have
silently reused a stale `libadf.a` and produced convincing false results:

```bash
grep -o 'CONV2D_ECHO_TEST=[0-9]' build/$TARGET/${PATCH_ROWS}x${PATCH_COLS}/ch$N_CHANNELS/aiecompiler.log
```

## Directory layout

```
design/
├── aie_src/
│   ├── fft_graph.h / ifft_graph.h    # FFT2D / IFFT2D graphs (single instance, GMIO row→col)
│   ├── conv2d_kernel.h/.cpp          # 3×3 MAC + optional ReLU + Hanning window + Stage B1
│   ├── cmul_accum_kernel.h/.cpp      # col-FFT ⊙ H_ch* + saturating accumulate
│   ├── mosse_graph.h/.cpp            # top level: PLIO + 10 GMIO + 2 kernels + FFT2D + IFFT2D
│   ├── kernel_only_graph.h/.cpp      # x86sim single-kernel harness
│   ├── aiesim_scenario_io.h          # shared scenario loader (keeps the two harnesses in sync)
│   ├── constraints.aiecst            # PatchIn PLIO shim placement
│   ├── hanning_128.h, weights/       # auto-generated
│   └── aiesim_data/s*/               # scenarios: s0-s4 raw patches (echo mode only);
│                                     #   s6 Stage-A preprocessed, H=unity, real conv path;
│                                     #   s7 s6 + a real per-bin complex H, off-centre target
│                                     #   (the only one exercising H_SHIFT, PSR, the F_ch tap);
│                                     #   cmul_stress exercises sat16's rails
├── pl_src/{camera_capture,roi_crop}/
├── host_app_src/
│   ├── mosse_tracker.cpp             # GMIO-driven XRT tracking loop; CropIp (user-managed
│   │                                 #   roi_crop) + the RC_*/timeline launch instrumentation
│   ├── mosse_filter.h/.cpp           # init/update/PSR/scale/Q1.15 export — NO XRT include
│   └── test/                         # native tests + NumPy goldens
│       └── scale_loop_sim.cpp        # closed-loop DSST scale sim (make scale_sim)
├── system_configs/mosse_x1.cfg       # v++ linker
├── profiling_configs/, directives/
└── exec_scripts/run_script.sh
scripts/  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py,
          check_collapse.py, check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py,
          bg_pan_sweep.py     # picks BG_PAN_R/C from the texture spectrum, no hardware
          mosse_loop_sim.py   # closed-loop MOSSE, centred-G vs shifted-G, no hardware
```

## Current status (2026-08-21)

**Best hardware to date: `runs/run_0821_1635.log`, 28.64 ms/frame = 34.92 FPS** — adds the
blocked `unpack_spectrum` to the config below. `unpack F_ch` 2.902 → 1.780 ms, tracking
bit-identical, GMIO unchanged (11.093 → 11.142) as the control.
**The day: 62.71 → 28.64 ms, 15.95 → 34.92 FPS — 2.19×.**

*(previous:)* **`runs/run_0821_1452.log`, 29.61 ms/frame = 33.77 FPS** — memory-tile
transposes + software-pipelined roi_crop + split cmul ports (`MEMTILE_TRANSPOSE=1
ROI_CROP_PIPELINE=1 CMUL_SPLIT_ACCUM=1`). Tracking unchanged throughout: mean IoU 0.9188,
worst 0.8353, centre 1.37/3.52 px, PSR 25.75/83.75/127.08.
APU 16.33 ms (55.2%), GMIO 11.09 (37.5%), roi_crop 1.02 (3.4%).
**The day: 62.71 → 29.61 ms, 15.95 → 33.77 FPS — 2.12×, with tracking never once regressing.**, APU 23.37 ms (51.3%)
against GMIO 20.98 ms (46.0%), tracking mean IoU 0.9188 / worst 0.8353 / centre 1.37 px,
rails 0, no gate or scale holds. APU and GMIO are now within 2.4 ms of each other, so
**neither one alone is worth more than ~2×** — see "Result 2026-08-21" and the next-steps
list under it.

*(Historical, from 2026-08-20:)*

**The tracker runs end to end on real VEK280 hardware at 128×128 ch16 on the real conv path, at
0.88 s/frame.** Box state, patch↔frame conversions, IoU and the DSST scale filter all execute.
Frame time was fixed today (8.26 s → 0.88 s, 9.4×) and that changes what experiments are
affordable: **a 200-frame run is now ~3 min instead of 28 minutes** — and gating the console
(see "Frame time": the 0.88 s IS the UART) would take it to ~20 s — so the scale filter can be
let settle (~120 frames) and `FRAME_NOISE` can be swept rather than guessed.

Best tracking to date: 2026-08-17 at 4-3-3, IoU 0.94-0.99, centre error 0.00-1.27 px, PSR 24-35,
rails=0 for ~20 frames. Earlier: hw_emu 128×128 ch1 `ITER_CNT=2` at 5-2-2 gives `err=0 px`,
peak/max-sidelobe 23.0, Bolme PSR 172, response matching `exp(-d²/8)` to within 4% at d=2, d=6.
PSR gating fires correctly under occlusion and reacquires.

**Two open defects, in order of severity.** Both are visible in the same 20-frame run
(`runs/run_0820_1112.log`) and both are separately observable in one log, so they can be
fixed in one build without violating "never move two magnitudes at once". The primary tracking
failure — the filter training against a centred G on a pre-update crop — is **FIXED in the host
2026-08-20, native-tested, NOT YET ON HARDWARE**; see the Correctness traps entry.

1. **The shift budget default is 4 bits too generous.** 4-5-5 gives 1.1-4.5% of int16 range.
   Use 4-4-4. See "Shift budget" — and note the offline premise that produced 4-5-5 was refuted
   by the same run.
2. **The DSST scale filter STALLS — and the runaway it was gated against is a different,
   rarer failure.** Measured on hardware 2026-08-20 (`runs/run_0820_1418.log`, 200 frames) and
   root-caused offline the same day with `make scale_sim`. `est_h` tracked the size envelope
   well for 130 frames (64 → 82.79 at f45 against a truth of 83), then FROZE at 59.13 for
   frames 130-199 — never moving in either direction while truth went 48 → 45 → 63, passing
   back *through* 59.13 around f190 with no response. Peak error +31%. The gate accepted all
   199 frames (`conf` 2.15..3.31 against a 2.00 threshold), so it neither caused nor caught it.
   **Root cause: the search RANGE, not the resolution.** See "Scale filter stall" below.
   *(Historical, still valid as a description of the other failure mode:)*
   Measured 2026-08-20 with position tracking exact for five frames (see the correction below):
   level −12 at conf 1.22 on frame 6 took the box 64.0 → 50.5 in one step, then thrashed
   −12/+5/+10/−14/+4/+10/−14/**+16** (the ±16 rail). Final box 76×76 vs truth 64×64; mean IoU
   0.6657, worst 0.4844. In the earlier `TRAJECTORY=1` run: box 67.9 → 51.5 → 45.7 → 40.6 →
   37.5 → 34.0, ending 31×31 against a truth of 75×75.
   **`conf` separates the populations on three independent runs** — healthy 2.24-3.31, collapsed
   0.72-1.87 — so 2.0 sits in the gap every time.
   **FIX WRITTEN, NOT YET ON HARDWARE**: `scale_gate()` in `mosse_filter.{h,cpp}`, three vetoes
   (`AT_SEARCH_RAIL`, `LOW_CONF`, `OUT_OF_RANGE`), tunable via `SCALE_CONF_MIN` /
   `SCALE_MIN_REL` / `SCALE_MAX_REL`, 21 native assertions in `make test_host`. **A veto holds
   the box AND skips `scale_update()`** — that second half is the one that stops the runaway,
   because holding the box while still training on the rejected frame teaches the model the bad
   estimate and it proposes it again harder next frame.
   **`conf` alone would not have been enough**: frames 16-18 of the earlier run report conf
   3.33-3.37 at level +0 with the box already collapsed to 34 — high confidence in a wrong,
   *settled* scale, which only the structural vetoes catch.

**PSR remains the wrong instrument for all of this.** That run: `19 evaluated, 19 accepted,
0 gated`, PSR min 15.95 / mean 26.60 / max 52.97 — while IoU fell to 0.0656. The response really
was sharply peaked, just in the wrong place. Read `track.csv`, not the console.

**Scale collapse is a PRIMARY defect. The "it is only a symptom" reading was wrong, and it was
wrong twice in opposite directions.** First it was read as a root cause from frames 307-313 of
one long run (a collapsed box, no context) and a scale clamp was proposed. Then it was read as
a pure symptom — "static-background lock → missed frames → position drift → the scale filter
re-learns on off-target patches → collapse" — because fixing the background made it visible.
**Measured 2026-08-20, `TRAJECTORY=0`, background panning, position tracking EXACT** (IoU
1.0000, centre error 0.00 px, displacement (10,−7) every frame through frame 5): the scale
filter jumped to level −12 on frame 6, took the box 64.0 → 50.5 in one step, and *that* shrank
the ROI and broke position tracking. Nothing upstream was wrong. The correct ordering here is
scale → position, the reverse of what was recorded.

**The general lesson, and it is the one that cost the most time on this project: an ordering
inferred from one failing run is a hypothesis, not a finding.** Both readings above were
consistent with the data available when they were written. Only a run where one of the two
candidates is known-good (here, position exact for five frames) can order them.

### Scale filter stall — root-caused offline 2026-08-20 with `make scale_sim`

`design/host_app_src/test/scale_loop_sim.cpp` drives the REAL
`scale_extract`/`scale_detect`/`scale_gate`/`scale_update` in a closed loop with the position
held — which the hardware run could not do. Native g++, seconds per arm. Three arms: `moving`
(the board's ±30%/200-frame envelope), `static`, `step` (one jump, then hold).

**It reproduces the board**: `moving` parks for 42 frames starting at **f131**; hardware froze at
**f130**. That match is what makes the counterfactuals below worth anything, and the program
refuses to report a verdict when the premise arm fails to reproduce.

**Four hypotheses were tested and THREE were refuted.** Each was plausible and each is now dead:

| hypothesis | test | result |
|---|---|---|
| per-frame under-correction | `--corrections 1..5` | **refuted** — settles at +10.4%, +8.3% at 5 iters |
| self-training feedback (the centred-G defect one axis down) | `--eta 0 .. 0.1`, gate bypassed | **refuted** — `eta=0` gives the IDENTICAL +8.3%; the model is frozen and the detector still says level 0 |
| the σ = S/16 target is too smooth | `--sigma-factor 16..132` | **refuted, and backwards** — sharpening σ below ~2 levels stops the filter correcting at all (+42.9%) |
| position drift prevents recovery | `--pos-err 0..11` | **refuted** — an 11 px standing offset, the board's own worst case, changes nothing |
| background decorrelation | `--bg-pan 31 47` | **refuted** — no effect |

**What survives, and it is a resolution floor, not a feedback.** With the model frozen (`eta=0`)
and the gate bypassed, the detector settles **1 to 4 levels (2-8%) off and reports level 0 with
`conf` 3.1-3.3**. Sweeping the step size shows the residual is set by how far the search can
see, not how finely it is divided — `a` barely matters, `a^((S-1)/2)` is everything:

| a | S | range | `moving` max err | `step` max err | cost (d·S²) |
|---|---|---|---|---|---|
| **1.02** | **33** | **1.373** | **12.6%** | **37.3%** | **1.00×** (today) |
| 1.03 | 33 | 1.605 | 9.5% | 42.9% | 1.00× |
| **1.04** | **33** | **1.873** | **8.6%** | **4.4%** | **1.00×** |
| 1.02 | 49 | 1.608 | 8.3% | 42.9% | 2.20× |
| **1.02** | **65** | **1.885** | **6.1%** | **6.1%** | **3.88×** |
| 1.03 | 49 | 2.033 | 8.1% | 6.3% | 2.20× |

**`SCALE_STEP=1.04` is free** (no cost change, S unchanged) and halves the moving error while
taking the step arm 37.3% → 4.4%. `SCALE_N=65` at a=1.02 is better still (6.1% on both) but
costs 3.9× the scale filter — 1.54 → ~6 ms, i.e. 3% of a 180 ms frame, affordable now.
**Both deviate from DSST §6.1's a=1.02/S=33**; neither is yet on hardware.

**Separately, `SCALE_CONF_MIN` blocks legitimate large corrections.** On the `step` arm the
detector proposes the correct `idx=-14` on the first frame after the jump, and the gate vetoes
it as `LOW_CONF` (conf 1.48) for four frames; after that the detector only ever proposes ±1 and
the box walks at 2%/frame with a 37% peak. Bypassing the gate corrects the same step in ONE
frame to +8.3%. **`conf` cannot distinguish "wrong proposal" from "big correct correction"** —
both match the model poorly, for the same reason. On the smooth `moving` envelope the gate makes
no difference at any threshold (proposals stay small, conf stays 1.76-3.30), so it is exonerated
for the hardware run — but it will bite on any abrupt scale change.

**UNEXPLAINED, and flagged rather than hand-waved**: the sim peaks at 12.6% where hardware peaked
at **+31%**, and the sim recovers when the envelope turns while hardware never did. Position
error and background pan were both tested and are not it. Do not treat the sim's absolute
magnitudes as the board's — the ORDERING and the range dependence are what transfer.

**The truth rate matters and the test bench chose it.** `SCALE_TRAJ_AMP=0.30` over
`SCALE_TRAJ_PERIOD=200` shrinks the target at ~0.94%/frame — under half of one 2% scale level —
so on most frames the correct proposal rounds to level 0 and the box does not move at all. The
error accumulates until it crosses the detector's 1-4 level floor. A slower envelope would hide
this defect entirely.

**CONFIRMED ON HARDWARE, `runs/run_0820_1513.log`** — `SCALE_STEP=1.04`, everything else
identical to `run_0820_1418`:

| | a=1.02 | a=1.04 |
|---|---|---|
| mean IoU | 0.807 | **0.917** |
| worst IoU | 0.579 | **0.833** |
| max box error | 31.4% | **9.6%** |
| mean / worst centre error | 2.47 / 11.07 px | **1.30 / 3.52 px** |
| longest `est_h` park | 70 frames | 39 frames |
| `resp00_over_peak` max | 0.082 | 0.082 (training-target fix holds) |

**Centre error fell 3.2× from a size-only change** — independent confirmation that position error
was downstream of the scale error, not a separate defect.

**The new `scale_idx` column earned itself back on its first run.** Over 199 frames the detector
proposed **only −1, 0 or +1** (174 / 13 / 12) — never ±2, at either step size. That was the sim's
central finding and it is now direct hardware evidence rather than an inference from
`log(est_h/64)/log(a)`. The decisions overlap heavily in error (idx 0 spans −5.3%..+9.6%, idx −1
spans −1.5%..+7.6%), so near zero the detector is a noisy estimator, not a dead zone with a clean
threshold.

**Calibration honesty: the sim predicted a=1.04 well and a=1.02 badly.** Sim said 12.6% → 8.6%;
hardware gave 31.4% → 9.6%. The a=1.04 number matched; the a=1.02 number was 2.5× optimistic, and
that discrepancy is still unexplained (position error and background pan were both tested and
ruled out). Trust the ORDERING; do not quote the sim's absolute magnitudes as the board's.

**Where it stops.** `SCALE_ETA` does not help now that the stall is gone (sim at a=1.04:
8.6% / 10.3% / 9.2% at eta 0.025 / 0.05 / 0.1). `SCALE_N=65` at a=1.04 gives 7.0% against 8.6%
for **3.9× the scale-filter cost** — not worth it. ~8-10% box error is the practical floor of
this filter as configured; the next real gain would be a different scale estimator, not a tune.

### Blocked `unpack_spectrum` — ON HARDWARE: 29.61 → 28.64 ms, 33.77 → 34.92 FPS

`runs/run_0821_1635.log`, wall 28.54 vs instrument 28.64. **Tracking bit-identical to
`run_0821_1452`**, which a pure loop reordering must be — and was proven byte-identical offline
first, on random, all-zero, single-spike and tie-heavy inputs.

| | `run_0821_1452` | **`run_0821_1635`** |
|---|---|---|
| `unpack F_ch` | 2.902 (181.3 µs/call) | **1.780 (111.3 µs/call)** |
| APU | 16.328 | **15.307** |
| GMIO (control) | 11.093 | 11.142 |
| frame | 29.61 | **28.64** |

**x86 said 4.66×; the A72 gave 1.63×, and the prediction band was still too optimistic.** I
predicted 1.4-2.2 ms and got 1.12 — under by 20%. The caution was right in direction (do not
transfer a cache-boundary result between these machines) and still under-called the gap.

**AND THE REMAINING 1.78 ms IS PROBABLY NOT MEMORY ANY MORE.** 111.3 µs / 16384 elements =
6.8 ns/element ≈ **9.5 cycles at 1.4 GHz**, for two int16 loads, two int→float converts and one
8-byte store. That is in the range of the arithmetic itself, not of cache misses — the naive
form's 11 ns/element had roughly 4 ns of miss penalty in it and blocking removed that. **Further
blocking will not help**; the next lever on this slot would be NEON-vectorising the int16→float
conversion (`SSHLL` + `SCVTF`, 4-8 elements at a time), worth maybe 1 ms, and it is the only
remaining idea for it.

### `CMUL_ACCUM_MEMTILE` IS A NET LOSS — 29.61 → 29.97 ms. REVERTED. (2026-08-21)

`runs/run_0821_1531.log`. **The change did exactly what it was designed to do and saved
nothing.** Tracking bit-identical to `run_0821_1452`, so this is purely a cost result.

| | `run_0821_1452` | `run_0821_1531` |
|---|---|---|
| `gmio_accum_out` tx/frame | 256 | **16** ✅ as designed |
| `gmio_accum_out` µs/tx | 16.83 | **280.62** |
| `gmio_accum_out` ms/frame | 4.310 | **4.490** |
| `gmio_fft_col_out` (control) | 4.355 | 4.550 |
| **pair total** | **8.665** | **9.039** |
| frame | 29.61 | **29.97** |

**16× FEWER TRANSACTIONS CHANGED THE COST BY NOTHING. That retires a whole class of ideas about
this port.** `gmio_accum_out`'s 4.31 ms was never per-barrier overhead — it is the host waiting
for the AIE to PRODUCE the accumulator. The pair `fft_col_out + accum_out` ≈ 8.7 ms *is* the
col-FFT + cmul pipeline's production time for 16 channels, and no regrouping, windowing or
transaction-count change can touch it. The extra 0.37 ms is the memtile hop the data now takes
on its way to DDR.

**THE PREMISE WAS A NUMBER THIS FILE ALREADY WARNED AGAINST USING, AND I USED IT ANYWAY.** The
item was justified by "at `FFT_COL_WS=32` `gmio_accum_out` WON (4.42 → 1.25)". But the WS=32
entry three sections down says, in bold: *"IT IS NOT ATTRIBUTION SHUFFLING BETWEEN THE TWO
SIBLINGS... The pair total refutes that: 9.00 → 18.32."* At WS=32 the accumulator looked cheap
only because `fft_col_out` — waited FIRST in the interleaved loop — had absorbed the production
wait. The single-port number was an artifact of wait ordering, exactly as recorded, and it was
still taken as evidence that the port had per-barrier cost to recover.

**THE RULE, NOW STATED AS A RULE: NEVER SIZE A CHANGE FROM ONE MEMBER OF AN INTERLEAVED
async/wait GROUP. SUM THE GROUP.** The same effect has now appeared four times — `gmio_fft_row_out`
absorbing the weights feed, the `fft_col_out`/`accum_out` pair under `FFT_COL_WS`,
`gmio_cmul_in` absorbing both inputs after the port split, and here. Three of those are recorded
above. It is the single most repeated measurement error in this design.

**What the negative result buys.** The ~8.7 ms pair is AIE production, so the only lever left on
it is overlap, not transfer efficiency — software-pipeline the CHANNEL loop the way `roi_crop`
was pipelined, so channel k's drain overlaps channel k−1's ~0.4 ms of APU work. The memtile
ping-pong already permits one channel of lookahead in the graph; it is the host that serialises.
That is the same manoeuvre that turned 5.196 ms into 1.020 on `roi_crop`.

**Reverted**: `CMUL_ACCUM_MEMTILE ?= 0`, with the measurement recorded at the variable. The
graph code stays behind the flag — it is correct, it simply does not pay.

### Accumulator: the ON-TILE version needs a DELAY LINE### Accumulator: the ON-TILE version needs a DELAY LINE, not a shared_buffer — split ports done instead (2026-08-21)

**THE STATED PLAN DOES NOT WORK AS STATED, AND THE REASON IS STRUCTURAL.** "Accumulator in a
memory tile" was sized at ~6 ms and listed as "same mechanism as the transpose, no new API".
It is not the same mechanism. The transpose is a feed-forward hand-off between two *different*
kernels; the accumulator is a **read-modify-write by ONE kernel across invocations**
(`accum_next = accum_prev + F⊙conj(H)`), which as a `shared_buffer` is a graph CYCLE
`cmul → memtile → cmul`.

Worse than the cycle, and this is the part that kills the naive version: the required delay is
**not one invocation, it is `CMUL_N_CHUNKS` (16)**. Invocation (ch, c) needs `accum[ch-1][c]`,
not what invocation (ch, c-1) just wrote, because cmul walks chunks inner and channels outer. A
1-chunk ping-pong buffer would feed it the wrong chunk. Expressing that needs a 16-deep delay
line in the memtile with lock semantics that admit a full pass of lag — a dataflow problem, not
a re-connection.

The other route — keep the accumulator in cmul's own tile memory — needs all 16 chunk
accumulators resident (16 × 4 KB = 64 KB, the whole tile) on top of the existing port buffers.
It does not fit, and AIE-ML kernels cannot address a memory tile as random-access scratch.

**So `gmio_accum_out`'s 4.31 ms/frame stays for now.** Recorded rather than quietly dropped,
because the ~6 ms figure is still sitting in the next-steps list of anyone reading this later.
One cheap lead survives: at `FFT_COL_WS=32`, `gmio_accum_out` went **4.42 → 1.25 ms — it WON**;
only `gmio_fft_col_out` lost (4.57 → 17.07). The two are coupled solely because cmul's window
size is inherited from the col-FFT's. Decoupling the F_ch tap's drain granularity from the
accumulator's would let the accumulator take the win alone.

### `CMUL_SPLIT_ACCUM` — ON HARDWARE: 31.48 → 29.61 ms, 31.81 → 33.77 FPS

What *is* deliverable, and it is most of the win: give `accum_prev` its own kernel port
(`cmul.in[2]`, fed by a new `gmio_accum_in`) instead of packing it behind H in `cmul_in`.
Graph compiles clean at 128×128 ch16 (0 errors, 0 critical warnings); `cmul` lands at (14,0)
with `pi2` local to the core and every buffer 4 KB, so there is none of the tile pressure the
`FFT_COL_WS=32` attempt hit.

**This deletes the `cmul packing` slot outright, and the arithmetic says what that slot really
was**: 16 channels × 128 KB = 2 MB/frame of BO→BO memcpy at the startup probe's 696 MB/s
uncached-read rate = **3.01 ms predicted, 2.871 measured**. It was never a copy that happened to
be slow — it WAS the uncached read. The host now feeds H straight from `filter_bo` and the
running sum straight from `accum_bo`. Expected frame **31.48 → ~28.6 ms, ~35 FPS**.

**MEASURED, `runs/run_0821_1452.log`** (wall 29.51 vs instrument 29.61). **Tracking
BIT-IDENTICAL to `run_0821_1402` on all 200 frames** — the kernel reads the same bytes from a
different port, and that is exactly what the frames say.

| slot | `run_0821_1402` | **`run_0821_1452`** |
|---|---|---|
| `cmul packing` | 2.871 (16 calls) | **0.009 (1 call — ch0's memset)** |
| `gmio_cmul_in` | 0.347 @ 21.71 µs/tx | **1.001 @ 62.55 µs/tx** |
| `gmio_accum_in` | — | **0.266 @ 16.63 µs/tx** |
| APU | 19.063 | **16.328** |
| GMIO | 10.240 | **11.093** |
| **frame** | **31.48** | **29.61** |

**THE PACKING PREDICTION WAS EXACT AND THE NET WAS 1 ms SHORT, BECAUSE THE SECOND PORT IS NOT
FREE.** `cmul packing` went 2.871 → 0.009, the full predicted saving. But `gmio_cmul_in` got
**2.9× more expensive per transaction (21.71 → 62.55 µs) despite its payload HALVING**, and
`gmio_accum_in` added 0.266. Net −1.94 ms against a predicted −2.87.

The pair total is the honest number: `cmul_in + accum_in` = **1.267 ms against the old
`cmul_in`'s 0.347**, so splitting the port cost +0.92 ms of DMA to save 2.86 ms of uncached
memcpy. Only ~0.32 ms of that is the extra per-barrier fixed cost (32 tx/frame instead of 16);
the rest is cmul acquiring two GMIO-backed input buffers per invocation instead of one, with
`cmul_in`'s wait — issued first — absorbing both. **That is the same "the first wait in an
interleaved pair absorbs the sync" effect already recorded for `gmio_fft_row_out` and for the
`fft_col_out`/`accum_out` pair. Sum an interleaved group before believing any member of it.**

**In-place on `accum_bo` is safe, reasoned rather than discovered:** cmul reads chunk c and
later writes chunk c; chunks are distinct addresses and the read of c always precedes the write
of c, so no chunk is read after being updated this channel. A double buffer would reintroduce
the copy this change exists to remove. ch0 gets a `memset` of `accum_bo` in place of the packed
path's zero-fill of the accum half.

**The single-port design was never a hardware constraint.** `cmul_accum_kernel.h` says so
outright: it works around a Vitis 2025.2 *cycle-approximate aiesim* deadlock, and "on real
AIE-ML hardware both approaches are equivalent". **`make aiesim` therefore needs
`CMUL_SPLIT_ACCUM=0`**; hardware does not.

**A `static_assert` now ties the `DMA_*` enum to its name table.** Inserting `DMA_ACCUM_IN`
mid-enum without updating the table would have silently RENAMED every port after it in the
report — the AP_* slots already had this guard and the DMA ones did not.

**`MEMTILE_TRANSPOSE` is now the Makefile default (1)**, since it is proven on hardware and a
one-sided flag is a board deadlock rather than a compile error.

### Software-pipelined roi_crop — ON HARDWARE: 35.58 → 31.48 ms, 28.14 → 31.81 FPS

`runs/run_0821_1402.log`, 200 frames, wall clock 31.44 ms against the instrument's 31.48.
**Tracking is BIT-IDENTICAL to `run_0821_1348` on all 200 frames** — as it must be, since this
changes only *when* work is issued, not what is computed.

| slot | `run_0821_1348` | **`run_0821_1402`** |
|---|---|---|
| **`roi_crop launch`** | 5.196 | **1.020** |
| GMIO (control) | 10.124 | 10.240 |
| APU (control) | 19.101 | 19.063 |
| **frame** | **35.58** | **31.48** |

**THE TIME WAS ELIMINATED, NOT MOVED, AND THAT WAS THE CHECK THAT MATTERED.** The memtile run
had just taught the opposite lesson — its frame total matched the prediction while the
attribution was wrong, with conv2d's production reappearing in `roi_crop launch`. So the test
here was whether GMIO or APU would absorb the 4.2 ms. Neither did: GMIO moved +0.116, APU
−0.038. The crop genuinely now runs underneath the host's work.

**THE 1.02 ms RESIDUAL IS EXACTLY CHANNEL 0, AND THE TIMELINE NAMES IT.** Frame 199:

```
  ch   start()     drain      poll      wait    wait#2 | drain->poll
   0     0.001     0.083     1.046     1.046     1.046 |       0.963
   1     0.001     1.042     1.043     1.043     1.044 |       0.001
   2     0.001     1.112     1.113     1.113     1.114 |       0.001
```

Channel 0 spins for **0.963 ms**; channels 1-15 exit on their first read. Channel 0 is the
`recompute=1` pass — the full Stage A (bilinear resample, log, zero-mean, unit-L2, int8
quantize) over 16384 output pixels — and it is launched *before* the channel loop, so it has
nothing to overlap with. Channels 1-15 only re-stream the cached patch and are now completely
hidden. The magnitude corroborates independently: CLAUDE.md measured roi_crop at **245.5 µs for
a 64×64 recompute**, and 128×128 is 4× the pixels ≈ 1 ms.

**Removing that last millisecond needs the FRAME boundary restructured, not the channel loop.**
The obvious move — launch frame f+1's ch0 crop at the end of frame f, behind the ~9 ms APU tail
— does not work as stated: roi_crop reads `frame_buf`, and frame f+1's scene is not generated
or pushed until the top of frame f+1, so it would crop the previous frame. Hiding it means
generating and pushing the scene one frame early. That is 3.2% of the frame for a reordering
that touches the scene/tracking boundary; not worth it against the items below.

**Balance now: APU 60.5%, GMIO 32.5%, roi_crop 3.2%, unattributed 3.7%.**

### MEMORY-TILE TRANSPOSE — ON HARDWARE### MEMORY-TILE TRANSPOSE — ON HARDWARE: 45.60 → 35.58 ms, 21.93 → 28.14 FPS (2026-08-21)

`runs/run_0821_1348.log`, 200 frames. Wall clock agrees with the instrument: 35.54 vs 35.58.
Both transposes (forward and inverse) now happen in AIE-ML memory tiles; the four transpose
GMIOs and both APU transposes are gone. **Cumulative for the day: 62.71 → 35.58 ms, 1.76×.**

| | `run_0821_1109` | predicted | **`run_0821_1348`** |
|---|---|---|---|
| frame | 45.60 | ~35 | **35.58** |
| FPS | 21.93 | ~28 | **28.14** |
| GMIO | 20.98 | ~15 | **10.12** |
| APU | 23.37 | ~18.9 | **19.10** |
| roi_crop launch | 0.067 | *not predicted* | **5.196** |
| tx/frame | 628 | 577 | **577** ✓ |

**TRACKING IS EFFECTIVELY IDENTICAL, AND THAT IS BETTER THAN PREDICTED.** The pre-registered
warning said this build could not be scored bit-identically because `mean_now` is now derived
from F_ch. Measured: **every displacement on all 200 frames is identical**, every IoU identical,
every gate verdict identical, mean IoU 0.9188 / worst 0.8353 / centre 1.37/3.52 px / final box
64×64 — all unchanged. Six frames of 200 differ, and only in the last printed digit of PSR
(118→119) or `resp00_over_peak` (0.00→0.01). PSR min 25.75 identical, max 127.10→127.08. So the
F_ch derivation of Stage B1's mean and Stage B3's energy reproduces the datapath to well under
one response LSB. The caution was right to state in advance; the derivation was simply more
accurate than the caution assumed.

**THE FRAME TOTAL MATCHED THE PREDICTION AND THE ATTRIBUTION DID NOT — read this before sizing
the next item.** The prediction was "`gmio_fft_col_out` must grow to ~9-10 ms absorbing conv2d's
~5.3 ms/frame of production, GMIO ~15". Measured: `gmio_fft_col_out` **did not move**
(4.573 → 4.345) and GMIO came in at 10.12, better than predicted. conv2d's production went
somewhere else entirely:

| port / slot | before | after |
|---|---|---|
| `roi_crop launch` | 0.067 | **5.196** |
| `gmio_ifft_row_in` | 0.021 | **0.304** (304 µs/tx) |
| `gmio_weights` | 0.463 | 0.546 |
| `gmio_fft_col_out` | 4.573 | 4.345 |
| `gmio_accum_out` | 4.423 | 4.276 |

**AND CLAUDE.md PREDICTED THE roi_crop ONE, IN WRITING, MONTHS OF WORK EARLIER.** The
`ROI_CROP_USER_MANAGED` entry says: *"the spin exits on its first read every time — the CU
finishes inside the drain loop. **If the drain ever shrinks (`FFT_ROW_WS` 8→16 halves it) the
spin will start spinning for real**."* The drain did not shrink, it was deleted, and the spin now
spins for real: 5.196 ms/frame = **325 µs/channel of roi_crop's own PL execution**, previously
hidden entirely behind the row-FFT drain. `gmio_ifft_row_in` absorbed the IFFT chain's
production the same way, for the same reason.

**So the 10 ms this item was sized at was real, but ~5 ms of it was uncovering a cost that was
always being paid and never visible.** That is not a disappointment — it is the next item,
already named, now with a number: software-pipeline `roi_crop` one channel ahead (launch k+1
before polling k) and most of the 5.2 ms should hide behind the APU work again. Frame ~31 ms,
~32 FPS.

**Per-firing weights async/wait costs 0.083 ms/frame** (0.463 → 0.546), confirming that pairing
them after the abort did not reintroduce the lock-step — the wait is a 64-byte host→AIE
transfer, not an AIE round trip.

**The APU table confirms the change reached the binary**: the `transpose` row is GONE (zero
calls, so it does not print), `BO↔heap stage` is 18 calls/frame against 52, and `window
mean+energy` is 0.374 ms on the F_ch path against 0.421 on the row-FFT path.

**Balance now: APU 53.7%, GMIO 28.5%, roi_crop 14.6%.** GMIO has gone from the dominant cost to
under a third; the APU is the frame again, and `roi_crop` is suddenly the third-largest item.

**FIRST HARDWARE ATTEMPT ABORTED AT FRAME 0 — `runs/run_0821_1342.log`, and the constraint was
already in this file.**

```
terminate called after throwing an instance of 'xrt_core::error'
  what(): Asynchronous operation is already initiated.
          Multiple 'async' calls are not supported: Invalid argument
```

The rewritten weights feed queued all `CONV_INVOCATIONS` asyncs on `gmio_weights` and waited
once. **XRT GMIO allows ONE outstanding async per port** — the identical error the depth-2 drain
probe hit in `runs/run_0820_1629.log`. Removing the drain loop does not relax it: the constraint
is XRT's, not the loop's. Fixed by pairing async with wait per firing. **The general lesson: an
existing constraint does not become inapplicable because the code around it changed.** Every
async site in `mosse_tracker.cpp` was audited afterwards — 11 pairs, one outstanding per port.

### How it was de-risked, in order, and what each step cost

Worth keeping as a template: the whole item cost ~1 h of machine time and ONE wasted board run.

1. `make graph MEMTILE_TRANSPOSE=1` — 5 min. Maps clean, GMIO 10 → 6, and the generated buffer
   descriptors showed `{1,1}` tiling compiles to **ONE 2-D BD** (`length 16384, stepsize {128,1},
   wrap {128,128}`), not per-element transactions. That retired the item's biggest stated risk.
2. `make aiesim_plio` at 64×64 ch1 — 20 min. **OVERALL: PASS**, F_ch correlation with golden
   0.9986, peak within 1 px. Proves the transpose is *correct*, which the descriptors cannot.
   The `fft_col_in.bin` bypass is unreachable with the memtile, so this ran the whole real chain.
3. Host restructure, then `make graph` again at the REAL geometry — 5 min, and it **failed**,
   catching a break introduced after step 2. The stale `Map_Report.csv` in the build dir still
   showed the previous run's healthy 6-port table; only the compiler's `ERROR:1` line was true.
4. Board run — aborted on the async constraint above.
5. Board run — the result at the top of this entry.

### `FFT_COL_WS` 8→32 IS A NET LOSS### `FFT_COL_WS` 8→32 IS A NET LOSS — 45.60 → 55.17 ms. REVERTED. (2026-08-21)

`runs/run_0821_1152.log`. **The default is back at 8** and the Makefile carries a warning at
the variable. Tracking was **bit-identical across all 200 frames** to `run_0821_1109`, so the
datapath and the windowing are correct — this is purely a cost result.

| | WS=8 (`run_0821_1109`) | WS=32 (`run_0821_1152`) |
|---|---|---|
| frame | **45.60 ms, 21.93 FPS** | 55.17 ms, 18.13 FPS |
| GMIO | 20.98 (46.0%) | **30.58 (55.4%)** |
| APU (control) | 23.373 | 23.398 |
| tx/frame | 628 | 232 |

**THE PRE-REGISTERED PREDICTIONS WERE 2 OF 3 RIGHT AND CATASTROPHICALLY WRONG ON THE ONE THAT
MATTERED**, which is the whole value of having written them down first:

| port | tx/frame | predicted | measured | |
|---|---|---|---|---|
| `gmio_accum_out` | 256 → 64 | ~1.2, wins | **1.252 ms @ 19.57 µs/tx** | ✅ exact |
| `gmio_response` | 16 → 4 | ~0.5-0.6, loses | **0.532 ms @ 133.04 µs/tx** | ✅ exact |
| `gmio_fft_col_out` | 256 → 64 | ~1.2, wins | **17.065 ms @ 266.63 µs/tx** | ❌ **14.9× worse per tx** |

**IT IS NOT ATTRIBUTION SHUFFLING BETWEEN THE TWO SIBLINGS, AND THAT IS THE KEY READING.**
They are drained in one interleaved loop with `fft_col_out` waited first, so the naive story is
that it simply absorbed what `accum_out` used to pay. The pair total refutes that:
**9.00 → 18.32 ms.** Roughly 9.3 ms of genuinely new cost appeared. Always sum an interleaved
pair before believing either half of it.

**And "it is absorbing AIE production latency" does not survive arithmetic either.** The col
FFT is ~34 µs/channel of scheduled work and cmul ~8 µs, so production is ~42 µs/channel against
the 1066 µs/channel now being paid on this port — 25×. 266 µs for a 16 KB transfer is 61 MB/s,
against 229 MB/s for the same port at WS=8 and 5.76 GB/s for the largest transfer the DMA probe
ever measured. Something got slower; it is not a barrier count and it is not the transfer.

**The leading suspect is the tile spill, flagged as a constraint BEFORE the run and not taken
seriously enough as a cost.** `Map_Report.csv` at WS=32 puts `cmul` at (24,1) spanning three
tiles, with (24,0) holding `pi0`+`po0` ping-pong at **65536/65536 B** and (24,1) holding `pi1`
at **65536/65536**. The core therefore reaches across a tile boundary for its input and output
windows while the shim DMAs the same banks. At WS=8 those buffers total 32 KB and are local.
**Unverified** — the mapper report proves the placement, not that the placement is what costs
9.3 ms.

**Cheapest discriminator if this is ever reopened: `FFT_COL_WS=16`.** Its buffers come to
exactly 64 KB (`pi0`/`po0` 16 KB each, `pi1` 32 KB), so `make graph` alone — free, 5 minutes —
says whether they land in ONE tile. If they do and a run behaves like WS=8, the mechanism is
the spill; if WS=16 also loses, the knob is simply dead on this port pair. Expected upside is
only ~4.5 ms (the pair halving), against ~12 ms for the memory-tile transpose, so **weigh that
before spending another build-flash-run cycle on it.**

**THE GENERAL LESSON: A KNOB THAT WON 7× ON ONE PORT LOST 4× ON ANOTHER, AND THE DESIGN GAVE NO
WARNING.** `FFT_ROW_WS` took `gmio_fft_row_out` 73.22 → 9.93 ms across four points with per-tx
cost pinned at 286-310 µs. The identical change on `gmio_fft_col_out` — same kind of port, same
drain structure, same payload sizes — went the other way. The "overhead-dominated vs
production-dominated" discriminator this was built on called `accum_out` and `response` exactly
right and still failed, because it has no term for where the mapper puts the buffers. **Do not
generalise a windowing result from one port to another without a hardware run.**

### Result 2026-08-21: 62.71 → 45.60 ms, 15.95 → 21.93 FPS. Host-only, one build.

`runs/run_0821_1109.log`, 200 frames, same config as `run_0820_1807` (`ITER_CNT=200
TRAJECTORY=1 SCALE_TRAJ=1 VERBOSITY=0 DUMP_BUFFERS=0`, 4-4-4, `SCALE_STEP=1.04`). Wall clock
from the per-frame markers agrees with the instrument: 45.50 ms against 45.60.

| slot | 0820_1807 | predicted | **0821_1109** |
|---|---|---|---|
| `scale extract` | 9.436 (2.0 calls) | 1.5-2.5 | **2.139 (1.0 call)** |
| `filter update` → `filter upd+quant` | 10.177 | ~13 combined | **4.741** |
| `publish filter` → `publish (pack)` | 5.595 | (combined) | **1.905** |
| `diag scan (rails)` | 1.192 (4.0) | ~0.2 | **0.230 (3.0)** |
| `BO<->heap stage` | 3.398 (51) | +0.1 | 3.499 (52) |
| **APU subtotal** | **40.456** | ~28 | **23.373** |
| GMIO (control — host-only change) | 21.016 | unchanged | 20.980 |
| UNATTRIBUTED (control) | 1.176 | unchanged | 1.180 |
| **frame** | **62.71** | 50-53 | **45.60** |

**APU 51.3%, GMIO 46.0%** — the two are nearly equal again, so neither alone gets much past
another 2×. Every call count confirms the change reached the binary: `scale extract` 1.0,
`diag scan` 3.0, `BO<->heap` 52.

**THE BATCH BEAT ITS PREDICTION BY 5-7 ms AND ALL THE EXCESS IS IN ONE PLACE.** Items 3 and 6
landed on their estimates; the filter update came in at 4.741 against ~10 predicted. Since
`publish (pack)` is now pack+sync alone at 1.905, the baseline's quantiser cost
5.595 − 1.905 = **3.690 ms**, so the comparable work was 10.177 + 3.690 = **13.867 ms** and is
now **4.741**. The fusion itself can only account for the scan's re-read of A and half the
divides, ~1.8 ms of that. **The remaining ~7 ms is the channel-major B accumulation** — the
incidental half of the change, flagged in advance as "could be another win or nothing".

**So the standing claim needs refining, and this is the second time this slot has been
mis-attributed.** CLAUDE.md said `filter update` was "memory-bound on ~8 MB/frame of heap, so
the lever is TRAFFIC not arithmetic". Memory-bound was right; TRAFFIC was not. Flipping the B
loop to channel-major moves **exactly the same bytes** — it changes only the *stream count*,
from `st.channels` strided readers of `F_all` 128 KB apart to one sequential reader. On an A72
whose prefetcher tracks a handful of streams, sixteen of them cost ~7 ms/frame. **The lever
was ACCESS PATTERN.** A traffic-volume model would never have found it, and did not: it is
the reason the prediction was 5 ms out.

**Item 3's 3.11× transferred exactly.** `scale extract` went 4730 → 2138 µs/call while its
bilinear sampling was untouched. Solving `S + D = 4730`, `S + D/3.11 = 2138` gives the
transform at **D = 3819 µs (81% of the call)**, now 1228. That is the figure "Settled
questions" measured offline for this exact d·S² transform, confirmed on the board.

**TRACKING: BIT-IDENTICAL FOR FRAMES 0-126, THEN 71 OF 200 FRAMES DIFFER. That is the
predicted signature of item 2 and not a defect.**

| | 0820_1807 | 0821_1109 |
|---|---|---|
| mean / worst IoU | 0.9174 / 0.8326 | **0.9188 / 0.8353** |
| mean / worst centre err | 1.30 / 3.52 px | 1.37 / 3.52 px |
| final box (truth 63×63) | 62×62 | 64×64 |
| PSR min / mean / max | 25.75 / 83.04 / 127.10 | 25.75 / **83.74** / 127.10 |
| overlap precision @0.5 | 100.0% | 100.0% |
| rails, gate holds, scale holds | 0 / 0 / 0 | 0 / 0 / 0 |

**Why 127 identical frames is the STRONG result, not the weak one.** The box size is a
function of the *integer* idx sequence alone (`box.h *= a^idx`), so the two paths stay
bitwise identical as long as they make the same decisions — **including on frames where
idx ≠ 0**. And idx ≠ 0 happened constantly before f127: `SCALE_TRAJ_AMP=0.30` takes truth
from 64 to 83.2 px by f50, and IoU there is 0.94-0.95, which a box parked at 64 could not
produce (it would read (64/83)² ≈ 0.59). So the shifted-target update agreed with a real
re-extraction on every decision for 126 consecutive frames of a live scale envelope. That is
much better evidence than the `idx = 0` identity, which is true by construction.

**What happened at f127**: the native test measures the trained A differing by 2.0e-4 and B by
1.1e-5 (the `|idx|` edge levels the shift cannot see). Near zero the detector is a noisy
estimator with heavily overlapping decisions — CLAUDE.md already records idx 0 spanning
−5.3%..+9.6% and idx −1 spanning −1.5%..+7.6% — so a 2e-4 perturbation eventually flips one
near-tie argmax. From there the box-size sequences differ by one level and everything
downstream differs. The runs end one scale level apart (61.5 vs 64.0 against a truth of 63.4),
which is why mean IoU moved 0.0014 in the new run's favour and centre error 0.07 px against
it. Both are inside this filter's own ~8-10% box-error floor.

**The acceptance criterion stated before the run was met**: item 5 and item 6 are bit-exact by
construction and *nothing* diverged in the first 126 frames, so no position difference on an
identical-decision frame — which is what a bug in either would have produced.

### Result 2026-08-20: 880 → 60.7 ms, 1.14 → 16.48 FPS (14.5×)

`runs/run_0820_1620.log`. Tracking **bit-identical across all seven instrumented runs** — mean IoU
0.9174, worst 0.8326, centre 1.30/3.52 px, box 62×62, PSR 25.75/83.04/127.10. Every optimisation
below was accepted on that test, and none of them changed a number the tracker produces.

| step | frame | FPS |
|---|---|---|
| baseline (console at 115200) | 880 | 1.14 |
| console gating (`VERBOSITY`) | 180.6 | 5.54 |
| BO copy pattern + int64 energy | 143.3 | 6.98 |
| scene on the host | 134.6 | 7.43 |
| `-O3 -mcpu=cortex-a72` | 132.2 | 7.56 |
| hypot fix + `-fcx-limited-range` | 127.7 | 7.83 |
| `FFT_ROW_WS` 8→16 (AIE rebuild) | 89.5 | 11.17 |
| `FFT_ROW_WS` 16→32 | 70.9 | 14.10 |
| `FFT_ROW_WS` 32→64 | **60.7** | **16.48** |

**GMIO is now 67.2% of the frame** (87.0 ms) against APU 31.8% (41.1 ms) and 1.0% unattributed.
The APU is no longer where the frame is.

**The hypot fix over-delivered, `-fcx-limited-range` under-delivered, and the second one is a
methodology lesson:**

| stage | before | after | predicted |
|---|---|---|---|
| publish filter (hypot fix) | 9.827 | **5.611** | ~2-3 ms saved; got 4.2 |
| filter update (`-fcx-limited-range`) | 11.306 | **10.824** | ~4.2 ms saved; **got 0.48** |

The native x86 benchmark measured `filter_update` at 0.915 → 0.635 ms with the flag (1.6×) and
that did not transfer at all. **Because the bottleneck is not the same on the two machines.**
`filter_update` streams ~8 MB/frame of heap (A is 16×16384 cfloat = 2 MB, read and written;
`g_F_all` another 2 MB) — that fits in a desktop L3, so x86 is compute-bound and the flag helps;
on the A72 it does not fit, so the function is memory-bound and vectorising the arithmetic buys
nothing. The A72 is 12× slower than x86 here (10.8 vs 0.9 ms) against ~3-5× on scalar fp, which is
the tell.
**CLAUDE.md already said "the offline model's ratios and orderings are sound, its absolute
magnitudes are patch-specific". This is worse than that: the ORDERING did not transfer either,
because the working set crossed a cache boundary between the two machines.** Benchmark a
host-side change on the host, or expect to be wrong about which fix matters.

**Next, in order. APU 19.1 ms, GMIO 10.2 ms, roi_crop 1.0 ms of a 31.5 ms frame. The APU is
60% of the frame and is now a FLAT TAIL — biggest single item 4.7 ms — so there is no dominant
slot left to attack; the remaining wins are structural:**
-1. ~~**`CMUL_ACCUM_MEMTILE=1`**~~ — **TRIED AND REVERTED, a 0.36 ms LOSS.** The accumulator
   port's cost is AIE production, not DMA overhead; see the entry below.
-3. ~~**`unpack F_ch` blocked transpose**~~ — **DONE, 29.61 → 28.64 ms.** See below.
-0. **Software-pipeline the CHANNEL loop — the only remaining lever on the 8.7 ms
   `fft_col_out`+`accum_out` pair.** That pair is the col-FFT + cmul production time for 16
   channels, proven immune to transaction count. Overlap it with the host's ~0.4 ms/channel of
   APU work, exactly as `roi_crop` was pipelined (5.196 → 1.020 ms). The graph already permits
   one channel of lookahead; the host serialises it.
-2. ~~**Hermitian symmetry in the host filter**~~ — **REFUTED AND RETIRED**, see "Settled
   questions". The premise is false in fixed point and would inject its worst error exactly
   where H is largest.

0. ~~**Software-pipelined `roi_crop`**~~ — **DONE, 35.58 → 31.48 ms.** Residual 1.02 ms is
   channel 0's Stage A pass, structurally exposed at the frame boundary; see the entry below.

1. ~~**`FFT_COL_WS` 8→32**~~ — **TRIED AND REVERTED, a 9.57 ms LOSS.** See the entry above.
   The cheap-knob era is over: this was the last Makefile variable with a plausible win, and
   it went the wrong way. Everything below is structural.
2. ~~**Memory-tile transpose**~~ — **DONE on hardware 2026-08-21, 45.60 → 35.58 ms.** See the
   entry above. The `{1,1}` throughput question resolved to one 2-D buffer descriptor.
3. **The idle A72 core.** The boot log reads `SMP: Total of 2 processors activated` at
   1.4 GHz and the host is single-threaded, so one core is idle for the whole frame. Safe
   tier: the post-response tail splits into two independent halves — translation
   update+publish and the scale filter touch disjoint state — for ~10 ms. Aggressive tier:
   pipeline the tail against the next frame's GMIO, which makes H one frame stale and is
   therefore an accuracy change to be scored on IoU, not asserted.
4. **Hermitian symmetry in the host filter, ~7-8 ms** — `conv2d` emits cint16 with `imag = 0`,
   so `F_ch` should be conjugate-symmetric and A/B half-redundant. **Verify the premise on a
   `DUMP_BUFFERS=1` dump before building anything on it**: fixed-point rounding in the DSPLib
   FFT does not guarantee the symmetry is exact, and this project has twice let an offline
   model move a calibrated constant on an unverified premise.

### `FFT_ROW_WS` 32→64 — 70.9 → 60.7 ms, 16.48 FPS. The knob is now exhausted.

`runs/run_0820_1807.log`. Tracking bit-identical (tenth consecutive run).

| WS | tx/channel | µs/tx | `gmio_fft_row_out` ms/frame |
|---|---|---|---|
| 8 | 16 | 286.10 | 73.22 |
| 16 | 8 | 288.16 | 36.89 |
| 32 | 4 | 293.38 | 18.78 |
| **64** | **2** | **310.28** | **9.93** |

**8× payload range for 8.5% cost growth — but the growth is accelerating** (0.7%, 1.8%, **5.8%**
per doubling). That is the AIE's own production time surfacing, which it had to: the design spends
~6.4 ms/frame actually computing (conv2d 4.1 + FFT/IFFT ~2.2), and that is now a large fraction of
the 9.93 ms. **WS=128 is not worth it** — one chunk per channel, a 64 KB window with 128 KB
ping-pong, for at most 3-4 ms before hitting the compute floor.
`gmio_ifft_row_out` did NOT regress further (0.596 → 0.570 ms); the WS=32 anomaly stayed contained.

**`FFT_ROW_WS` alone took `gmio_fft_row_out` 73.22 → 9.93 ms — 63 ms, from a Makefile variable.**

**GMIO 21.0 ms (33.5%), APU 40.5 ms (64.5%).** GMIO is no longer the problem.

**A 64 KB ping-pong DOES fit on a 64 KB tile.** The prediction that it would not was wrong —
AIE-ML cores address neighbouring tiles' memory, so a window need not live entirely in the
producer's tile. Cost of being wrong: 3 minutes, because `make graph` was run alone before
committing to the full `sd_card` build. **Test an AIE placement question with `make graph`, not
with a 25-minute package.**

**Two structural projects shrank while a Makefile knob did the work** — the weights RTP (was the
headline plan; `gmio_weights` is now 0.47 ms) and the memory-tile transpose (was ~39 ms this
morning, now ~10). Exhaust the cheap knobs and re-measure before opening the graph.

### `FFT_ROW_WS` 16→32 — 89.5 → 70.9 ms, 14.10 FPS, and the fixed-cost model holds over 4×

`runs/run_0820_1739.log`. Tracking bit-identical (ninth consecutive run).

| payload/tx | 4096 B (WS=8) | 8192 B (WS=16) | 16384 B (WS=32) |
|---|---|---|---|
| `gmio_fft_row_out` µs/tx | 286.10 | 288.16 | **293.38** |
| ms/frame | 73.22 | 36.89 | **18.78** |

**4× the payload, 2.5% more per transaction.** GMIO 49.0 → 30.5 ms/frame (694 tx). APU is now
**56.4%** of the frame and GMIO 41.8% — the balance has flipped back.

**BUT THE RULE IS NOT UNIVERSAL, and one port got worse.** `gmio_ifft_row_out` went
**0.148 → 0.596 ms/frame** (8 tx @ 18.5 µs → 4 tx @ 149 µs). It was already running at the fast
sibling rate, and fewer/larger chunks just made each `wait()` absorb more AIE latency. So: raising
WS wins for a port whose per-tx cost is latency-dominated *regardless of size* (row_out, pinned at
~290 µs across 4×), and loses for one already at the ~18 µs floor. Check the per-port table, not
just the total, before raising a WS.

**WS=64 is probably the ceiling**: `CONV_OUT_CHUNK` would be 32 KB, ping-pong 64 KB = the entire
tile. If it fits, row_out → ~9 ms; the real floor beneath it is the ~6.4 ms/frame the AIE actually
spends computing.

### THE ROW-FFT DRAIN IS A FIXED PER-BARRIER COST — `FFT_ROW_WS` 8→16, 127.7 → 89.5 ms, 11.17 FPS

`runs/run_0820_1716.log`. Tracking bit-identical for the eighth run running (mean IoU 0.9174,
worst 0.8326, centre 1.30/3.52, box 62×62) — and this is the **first AIE-graph change** of the
sequence, so that also rules out a windowing bug.

**The measurement settles the question outright:**

| | WS=8 | WS=16 |
|---|---|---|
| `gmio_fft_row_out` | 256 tx, 73.22 ms, **286.10 µs/tx** (4096 B) | 128 tx, 37.00 ms, **289.09 µs/tx** (8192 B) |
| `gmio_weights` | 256 tx, 3.67 ms | 128 tx, 1.88 ms |
| GMIO total | 1090 tx, 87.0 ms | **826 tx, 49.0 ms** |
| frame | 127.7 ms, 7.83 FPS | **89.5 ms, 11.17 FPS** |

**Double the payload, identical cost per transaction (286 → 289 µs).** The drain is a fixed
per-barrier cost and nothing to do with bandwidth — consistent with the earlier probe finding that
a 64 B transfer costs 14.4 µs and a 128 KB one 22.8 µs. Halving the barrier count halved the time,
exactly.

**Next, and it is the same lever again.** `gmio_fft_row_out` is still **75% of GMIO** (36.9 of
49.0 ms). `FFT_ROW_WS=32` gives `CONV_OUT_CHUNK` = 4096 cint16 = 16 KB, ping-pong 32 KB/port —
under the 64 KB tile limit, `CONV_INVOCATIONS` = `ROW_CHUNKS` = 4 so the `static_assert` holds,
and the DSPLib window 4096 is 32× the 128-pt FFT. Expect row_out ~18.5 ms, GMIO ~30 ms, frame
**~71 ms ≈ 14 FPS**. WS=64 would need 64 KB ping-pong and is the likely ceiling.
Separately, `FFT_COL_WS` 8→16 halves `gmio_fft_col_out` (4.61 ms) and `gmio_accum_out` (4.50 ms)
for another ~4.5 ms — those are 256 tx/frame each at ~18 µs, so they are barrier-bound too.

**The weights RTP is now a small item, not the big one.** `gmio_weights` is 1.88 ms/frame. Its
value was always dissolving the interleave that made row_out cost 15× its siblings — and
`FFT_ROW_WS` buys the same thing by halving the barrier count, with no graph surgery.

### Memory-tile transpose — VALIDATED AGAINST DSPLib, not yet built (2026-08-20)

**AMD already ships this.** `Vitis_Libraries/dsp/L2/include/aie/fft_ifft_2d_graph.hpp` puts the
row→col transpose of a 2D FFT in an AIE-ML memory tile, so it is a port of a working reference,
not a research task:

```cpp
adf::shared_buffer<TT_DATA_D2> memTile1;                       // "performs the transpose"
memTile1 = adf::shared_buffer<TT_DATA_D2>::create({D1, D2}, 1, 1);
num_buffers(memTile1) = 2;                                     // ping-pong
write_access(memTile1.in[0]) = tiling({.tiling_dimension = {D1, D2}, .offset = {0,0}});
read_access (memTile1.out[0]) = tiling({.tiling_dimension = {1,1}, .offset = {0,0},
    .tile_traversal = {{.dimension=1,.stride=1,.wrap=D2},{.dimension=0,.stride=1,.wrap=D1}}});
connect<>(frontFFT.out[0], memTile1.in[0]);  connect<>(memTile1.out[0], backFFT.in[0]);
```

Write contiguously, read with the traversal dimensions swapped — `buffer_dimension[0]` is the
contiguous one, so walking dimension 1 in the inner loop IS the transpose. API confirmed present
in the installed 2025.2 toolchain (`shared_buffer`, `tiling_parameters{dimension,stride,wrap}`,
`write_access`/`read_access`, `num_buffers`) and marked **AIE-ML only** — which the VEK280 is.
128×128 cint16 = 64 KB, 128 KB with ping-pong, against a 512 KB tile; the graph uses **zero**
`shared_buffer` today.

**Worth ~39 ms of an 89.5 ms frame**: deletes `gmio_fft_row_out` (36.9 ms, the largest single
item in the design), `gmio_fft_col_in`, and the APU transpose.

**Bonus found in the same file: DSPLib halves the memory tile for REAL input**
(`kD1SizeMemTile = kIsRealDataD1 ? TP_POINT_SIZE_D1/2 : TP_POINT_SIZE_D1`, Hermitian symmetry).
`conv2d` emits cint16 with **imag = 0** — we run a complex FFT on real data and pay 2× in the
forward transform and in every byte that moves with it. Possibly larger than the transpose win.
Same observation the fDSST entry makes about the scale filter; it applies to the main pipeline too.

**Two things to settle before committing:**
- The transpose read is `tiling_dimension = {1,1}` — one element per step with a large stride.
  Supported, but **no throughput figure for 128×128 cint16 at that granularity**. Measure it in
  aiesim; it could be fast or it could reintroduce a per-element cost.
- **THE STATED BLOCKER WAS WRONG — CORRECTED 2026-08-21.** This entry claimed "our dataflow is not
  a plain 2D FFT — per-channel `cmul_accum` sits between the row and column passes". It does not.
  `mosse_graph.h` wires `fft2d.fft_col_out → cmul.in[0]`, i.e. **cmul is DOWNSTREAM of the column
  FFT**, and the Map_Report clusters agree (fft_rows (15,1) → fft_cols (29,0) → cmul (24,1)).
  Between the row and column passes there is nothing but the DDR round trip. **Both transposes —
  forward and inverse — are therefore textbook 2D-FFT transposes and the DSPLib pattern applies to
  each directly.** Still borrow the `shared_buffer` pattern rather than the whole graph, but for a
  different and much weaker reason: we need the `gmio_fft_col_out` tap on the column FFT's output
  and our own `cmul` downstream of it. Verify a claimed structural blocker against the wiring
  before letting it defer a 10 ms item for a week.
- (The accumulator is the second memory-tile candidate: CLAUDE.md's own "on-tile would need
  a Memory Tile" note, worth another ~7.7 ms.)

### XRT GMIO allows ONE outstanding async per port — the depth-2 probe is retired

`runs/run_0820_1629.log`. The sweep ran **40 clean frames at depth 1** (`gmio_fft_row_out`
73.06-73.22 ms, 256 tx, remarkably stable) and then **aborted** the instant it tried depth 2:

```
terminate called after throwing an instance of 'xrt_core::error'
  what():  Asynchronous operation is already initiated.
           Multiple 'async' calls are not supported: Invalid argument
```

**The per-firing barrier is imposed by XRT, not by our loop.** There is no pipeline to deepen.
This retired the planned fix in **11 seconds of board time**, before anything was built on it —
the same value the `poll`/`wait` split delivered in the 503 ms hunt, and the reason to build
probes that can fail loudly. The ascending sweep is why the depth-1 data survived the abort.

**WHAT IT DOES NOT KILL, and this is now the leading explanation.** The sibling output ports
drain the same 4096 B chunks under the *same* one-async-at-a-time rule at **17.7-19.6 µs/tx**.
So the API constraint is not what makes `gmio_fft_row_out` cost 286 µs/tx — 15× its siblings.
The difference is that its `wait()` is interleaved with the weights feed, so each of the 256
firings a frame pays a full host→AIE→host round trip that cannot start until its weights buffer
lands. Remove `gmio_weights` from the loop and what is left should be a plain drain like its
siblings: **73 ms → ~5 ms**.

**So the two remaining candidates both need an AIE rebuild:**
- **Weights as an RTP / async parameter** (already on record as "the proper fix, not done"). The
  host currently sends **the same 64 bytes 16 times per channel** — the `async` offset is
  `ch * WEIGHT_CH_BYTES`, with no `k` dependence — purely because ADF acquires an `input_buffer`
  per firing. Removes `gmio_weights` (256 tx/frame → 0, 3.67 ms) and dissolves the interleave.
- **`FFT_ROW_WS` 8→16** halves `CONV_INVOCATIONS` and therefore the round-trip count outright:
  256 tx → 128. A Makefile knob, no source change. Halves the drain if the cost is per-barrier,
  which is now the only surviving hypothesis.

Note the probe also confirmed the drain is **not** noise-limited: 40 consecutive frames at
73.06-73.22 ms is a 0.2% spread, so any change above ~0.5 ms will be unambiguous.

### -O3 and the hypot in filter_quantize_q15 (2026-08-20)

**`-O3 -mcpu=cortex-a72` is bit-identical and worth 2.4 ms** (`runs/run_0820_1610.log`, 134.6 →
132.2 ms, 7.56 FPS). Now the `HOST_OPT` default. But **all 2.4 ms is one slot**:

| stage | -O2 | -O3 | Δ |
|---|---|---|---|
| publish filter | 12.164 | 9.827 | **−2.34** |
| transpose | 2.215 | 1.756 | −0.46 |
| **filter update** | 11.040 | 11.306 | **+0.27** |
| **scale extract** | 9.356 | 9.428 | **+0.07** |

**The two biggest compute blocks did not move, and the reason is `std::complex<float>`.** C99
Annex G forces GCC to call the libgcc `__mulsc3` helper for complex multiply, which blocks
vectorisation. Verified by cross-compiling `mosse_filter.cpp`: `-fcx-limited-range` removes the
`__mulsc3` call and raises NEON fp ops 10 → 17, and a native benchmark puts `filter_update` at
0.915 → **0.635 ms (1.6×)**. It only drops Inf/NaN range handling in complex mul/div — nothing
here goes near those — unlike `-ffast-math`, which buys the same time by making every float
operation in the file unsafe.

**`std::abs()` ON A COMPLEX IS `hypot()`, AND IT WAS RUNNING 262144 TIMES PER FRAME.**
`filter_quantize_q15`'s max-|H| scan called it once per bin per channel. Native x86 benchmark of
that function alone: **9.63 ms at -O2, 8.71 at -O3, 8.70 with -fcx-limited-range, 2.01 with
-ffast-math** — i.e. no compiler flag short of unsafe maths touched it, because the cost was a
libm call, not codegen.
**Fixed in source, exactly rather than approximately**: scan on `re²+im²`, which is monotone in
`abs()` so it selects the same element, then take **one** square root of that element. 8.71 →
**1.88 ms, 4.6×**, matching what `-ffast-math` achieved — which is the evidence that the
transcendental was the whole cost. `make test_host` still passes, including the NumPy golden that
checks `filter_quantize_q15`'s actual output.
**The general trap: a per-element `std::abs`/`std::norm` on a complex is a transcendental call.**
Grep for it before profiling anything else in this file.

### Scene on the host — MEASURED 2026-08-20: 143.3 → 134.6 ms, 7.43 FPS

`runs/run_0820_1604.log`. Tracking bit-identical for the fifth run running (mean IoU 0.9174,
worst 0.8326, centre 1.30/3.52, box 62×62, PSR 25.75/83.04/127.10).

`g_frame_host` (2 MB heap) is now the authority for the scene and the host **pushes** it to
`frame_bo` once per frame; `scale_extract`'s two calls read the heap copy. The direction is the
point — the probe measured BO writes at 3470 MB/s against reads at 696, so pushing 2 MB costs
0.405 ms where pulling it would cost ~2.9 ms. **INVARIANT: `g_frame_host` is the authority,
`frame_bo` is a copy.** Anything writing `frame_bo` directly (`rc_control_cu_probe`'s zero-fill)
must run before the first push.

| | before | after |
|---|---|---|
| scale extract | 13.43 | **9.36** |
| UNATTRIBUTED | 5.94 | **1.31** |
| APU subtotal | 51.73 | **47.84** |
| frame | 143.3 | **134.6** |

**THE PREDICTION WAS WRONG AND THE ERROR IS THE USEFUL PART.** I predicted `scale extract`
13.43 → 1-2 ms; it went to 9.36. So the uncached read was only ~30% of it, not ~90%. The mistake
was reading the earlier split too hard: `scale extract 13.43` vs `scale detect+update 0.52` says
**where** the time is, and I took it as saying **why**. The remaining 9.36 ms is compute inside
`scale_extract` — dominated by the transform along the scale axis, `d`×`S²` = 484×33² = **527k
complex MACs**, which no memory fix touches.

**That reopens the real-input DFT with a much better number.** "Settled questions" measured it at
**3.11×** on this exact transform. Against 9.36 ms/frame that is worth **~6 ms**, not the 0.5 ms
the entry assumed — and the features are real by construction, so it is justified anyway.

**GMIO is now 63.9% of the frame** (87.07 ms) against APU 35.1% (47.84). Remaining APU compute:
`publish filter` 12.16, `filter update` 11.04, `scale extract` 9.36 — **32.6 ms of pure heap
arithmetic**, which is what `HOST_OPT="-O3 -mcpu=cortex-a72"` is being tested against next.

### APU copy pattern — MEASURED 2026-08-20: 181 → 143 ms, 5.54 → 6.98 FPS

`runs/run_0820_1554.log`. **Tracking is bit-identical to the three runs before it** — mean IoU
0.9174, worst 0.8326, centre 1.30/3.52 px, final box 62×62, PSR min 25.75 / mean 83.04 / max
127.10. That was the acceptance test: this change touches the datapath, so anything other than
an exact match would have meant a wrong copy size, ordering or a stale buffer, not a tuning
difference.

| stage | before | after | Δ | predicted |
|---|---|---|---|---|
| window mean + energy | 14.41 | **0.417** | −14.0 | ~2.1 (beaten 5×) |
| unpack F_ch | 16.52 | **2.945** | −13.6 | ~2.5 ✓ |
| transpose | 9.31 | **2.258** | −7.1 | ~2.4 ✓ |
| PSR scan | 3.79 | **0.529** | −3.3 | ~0.5 ✓ |
| BO↔heap stage (the overhead the fix ADDS) | — | +3.395 | +3.4 | — |
| **APU subtotal** | **84.82** | **51.73** | **−33.1** | ~45 |
| **frame** | **180.6** | **143.3** | **−37.3** | 142-147 ✓ |

`filter update` went 11.08 → 11.17 ms, i.e. **unchanged, as predicted** — it is pure heap and the
copy pattern cannot touch it. That null result is what validates the model behind the other rows.

`window mean + energy` beat its estimate by 5× (901 → 26.1 µs/call, **34×**): the estimate assumed
the probe's 38 µs heap-int-sum, but a clean int64 MAC loop on cached memory vectorises to
1.6 ns/element. **The int64 switch is bit-exact, not an approximation** — every term is a product
of two int16 so the sum is ≤ 3.5e13, below 2^53, meaning the old `double` was already carrying
exact integers.

**The two splits both paid.** `scale extract` is **13.43 ms** against `scale detect+update` at
**0.52 ms** — 96% of the scale filter's cost is inside `scale_extract`, and none of it is the
`scale_detect`/`scale_update` maths. (**Careful with that sentence**: it locates the cost, it does
not explain it. Reading it as "so it is the `frame_bo` reads" produced a 12 ms prediction that
came in at 4 — see "Scene on the host". Only ~30% was the mapping; the rest is the d·S² DFT.) And `diag scan (rails)` surfaced **1.21 ms** that had
been sitting in UNATTRIBUTED (`report_cint16` scans 64 KB four times a frame at *every* verbosity,
because rails detection is the point).

**The frame is now GMIO-dominated for the first time**: GMIO 87.2 ms = **60.2%**, APU 51.7 ms =
35.7%, unattributed 5.9 ms = 4.1%.

**Next, in order, all host-only:**
1. **`scale extract`, 13.43 ms** — the same copy pattern against `frame_bo`. The 33 crops span at
   most ~1.87× the box, so the touched window is ~116×116 px ≈ 13.5 KB; copy that window once per
   call rather than reading it scattered out of a write-combining 2 MB mapping. ~12 ms.
2. **`-O3 -mcpu=cortex-a72`** (`GCC_FLAGS` is still `-O2` with no `-mcpu`) — the only lever left
   for `publish filter` (12.17 ms) and `filter update` (11.17 ms), which are pure heap compute:
   ~262k complex ops each. Size the gain from a run, not from this sentence.
3. `cmul packing` (2.91 ms) was deliberately left alone: only ~0.7 of its ms is reachable without
   restructuring, and it sits in the deadlock-sensitive cmul path.

After (1) and (2) the APU would be ~30 ms against 87 ms of GMIO, at which point **`gmio_fft_row_out`
and the weights-RTP work are the only remaining lever** — see "Frame time".

### APU per-frame cost — the measurement that found it (2026-08-20), and the attribution was wrong

`runs/run_0820_1528.log`, 200 frames, `VERBOSITY=0 DUMP_BUFFERS=0`, mean frame body
**180.81 ms**. The instrument accounts for **95.2%** of it — unattributed 8.7 ms (4.8%), of which
console is 3.1 ms.

| stage | calls/fr | ms/frame | µs/call | share |
|---|---|---|---|---|
| **unpack F_ch** (conversion only; the `sync` is 4.4 µs, split out 2026-08-20) | 16 | **16.52** | 1032 | 9.1% |
| **window mean + Parseval energy** | 16 | **14.42** | 901 | 8.0% |
| **scale filter** (extract ×2, detect, update) | 2 | **13.95** | 6992 | 7.7% |
| **publish filter** (quantize + pack + sync) | 1 | **12.16** | 12160 | 6.7% |
| **filter update** | 1 | **11.06** | 11058 | 6.1% |
| transpose (17 × 64 KB) | 17 | 9.30 | 547 | 5.1% |
| PSR scan | 1 | 3.79 | 3806 | 2.1% |
| cmul packing (~2 MB memcpy) | 16 | 2.91 | 182 | 1.6% |
| scene gen | 1 | 0.61 | 610 | 0.3% |
| `frame_bo.sync` (2 MB) | 1 | 0.10 | 100 | 0.1% |
| B2 correction | 1 | 0.002 | 2.3 | 0.0% |
| **APU subtotal** | | **84.89** | | **46.9%** |
| GMIO (DMA_T) | | 87.12 | | 48.2% |
| roi_crop launch | | 0.068 | | 0.0% |
| **UNATTRIBUTED** | | **8.73** | | **4.8%** |

**THE ATTRIBUTION IN THIS FILE WAS WRONG, AND IN A SPECIFIC DIRECTION.** The ~90 ms was
repeatedly described as "transposes + the ~2 MB/frame packing memcpy + filter update". Measured:
transposes are 5.1% and **the packing memcpy is 1.6%** — the two costs that were named are the
small ones. The largest two, `unpack F_ch` and `window mean + energy`, were never mentioned at
all. Both are per-channel loops over 16384 elements on XRT BO mappings.

**`SCALE_ETA`'s cost note is wrong by 9×.** "1.54 ms/frame on x86" is recorded under the DSST
entry; on the A72 the scale filter is **13.95 ms/frame**. The "Settled questions" dismissal of
fDSST's real-input DFT — *"optimising a non-bottleneck: 1.5 ms against a 13-22 ms frame"* — has
both halves wrong (the frame is 181 ms). At 3.11× the real-input DFT is now worth ~9 ms, not
0.5 ms. Reopen it.

**ROOT-CAUSED 2026-08-20 (`runs/run_0820_1539.log`): THE BO MAPPINGS ARE WRITE-COMBINING, AND
fp64 IS A SECOND, INDEPENDENT 4×.** The startup probe, 64 KB × 64 reps:

| access | µs | MB/s | vs heap |
|---|---|---|---|
| memcpy heap → heap | 8.9 | 7359 | 1.0× |
| **memcpy BO → heap (read)** | **94.2** | **696** | **10.6×** |
| memcpy heap → BO (write) | 18.9 | 3470 | 2.1× |
| sum int16, heap | 38.4 | 1707 | 1.0× |
| **sum int16, BO mapping** | **222.6** | **294** | **5.8×** |
| **sum fp64, BO mapping** | **895.3** | **73** | **4.02× the int sum on the SAME buffer** |

Reads 10.6× slower than writes at 2.1× is the write-combining signature: stores buffer and burst,
loads go straight to DRAM uncached. **Both candidate mechanisms are real and they multiply** —
the probe was built to be able to kill either one and instead confirmed both, which is why it
benchmarked fp64 and int on the *same* buffer.

**The energy loop IS the probe.** `window mean+energy` measures 900.8 µs/call; "sum fp64, BO
mapping" measures 895.3 µs on the same 64 KB. That is not a correlation, it is the same loop.

**And the split settled the top row: the transfer is free, the conversion is everything.**
`fcol_bo.sync` is **4.4 µs/call (0.071 ms/frame)**; `unpack F_ch` is **1032 µs/call
(16.5 ms/frame)**. Before the split those were one 16.59 ms number that could have been read
either way.

**THE FIX IS ONE PATTERN AT FIVE SITES: never compute on a BO mapping.** A bulk `memcpy` out
(94 µs for 64 KB) amortises the uncached read; compute on the heap copy; `memcpy` back only if
needed (writes are only 2.1×). Estimated, using the probe's own rates:

| stage | now | reads a BO? | after | saving |
|---|---|---|---|---|
| unpack F_ch | 16.52 | yes | ~2.5 | ~14.0 |
| window mean+energy | 14.41 | yes, **and fp64** | ~2.1 | ~12.3 |
| transpose (in place) | 9.31 | yes, r+w | ~2.4 | ~6.9 |
| PSR scan | 3.79 | yes (`resp_bo`) | ~0.5 | ~3.3 |
| cmul packing | 2.92 | BO→BO | ~1.0 | ~1.9 |
| **total** | **46.95** | | **~8.5** | **~38** |

That is APU 84.8 → ~45 ms and the frame 181 → **~142 ms, 7.0 FPS**, host-only. Two independent
sub-fixes worth calling out separately: **the energy accumulator should be int64, not fp64** (sum
of 16384 int16 squares peaks at 1.8e13 — exact in int64, and 4× faster on the same memory), and
`filter update` (11.1 ms) is **pure heap already**, so the BO fix does nothing for it — that one
needs `-O3 -mcpu=cortex-a72` (`GCC_FLAGS` is still `-O2` with no `-mcpu`).

`scale filter` (13.86 ms) reads `frame_bo` directly for 33 crops per call. The crops span at most
~1.87× the box, so the touched window is ~13 KB — a targeted copy, not the whole 2 MB.

*Superseded lead, kept because the reasoning was right and the conclusion was half:*
**The per-element rates are the lead, and they all point the same way.** unpack 63 ns/element,
window+energy 55 ns/element, transpose 33 ns/element — that is 71 MB/s on a 64 KB buffer, roughly
50-100× below what an A72 does on cached DRAM. All three run **directly on `xrt::bo` mappings**
(`row_bo.map()`, `fcol_bo.map()`). Leading hypothesis: those mappings are uncached or
write-combining, so every access is a DRAM round trip. Untested.
**Discriminator, one run:** time a plain `memcpy` of 64 KB out of the BO into a heap buffer. If
it costs ~500 µs it is the mapping (fix: copy once, compute on the heap copy); if it costs ~10 µs
the mapping is fine and the loops are the problem (fix: `-O3 -mcpu=cortex-a72`, and drop the
energy accumulator from fp64). `GCC_FLAGS` is currently `-O2` with no `-mcpu`.

**Instrumentation caveat, now resolved:** the `unpack F_ch` slot originally timed
`fcol_bo.sync(FROM_DEVICE)` **and** `unpack_spectrum` together — the same mistake as `DMA_T`
timing `async` and `wait` together. Split 2026-08-20: sync 0.071 ms/frame, conversion 16.5. The
`AP_*` enum and its name table are now tied by a `static_assert` so they cannot drift.

**Ceiling:** if APU work went to zero the frame would be ~96 ms (87 GMIO + 8.7 residual), i.e.
10.4 FPS against today's 5.5. APU and GMIO are now almost exactly half the frame each, so neither
alone gets past ~2×.

Read the **CUMULATIVE** table, not the per-frame ones: frame 0 runs `filter_init` (13.5 ms) rather
than `filter_update` (11.0 ms), has no PSR scan, and calls the scale filter once instead of twice.

### hw_emu frame times — MEASURED, and they do not scale from ch1

```
                 ch1        ch16
per channel    ~50 min    ~43 min
per FRAME      ~50 min    ~11.5 h
ITER_CNT=2      ~1.7 h      ~23 h
ITER_CNT=3      ~2.5 h      ~34 h
```

**hw_emu wall clock does not track AIE compute.** The echo-mode ch16 run, where conv2d did no
MAC work at all, still took ~14 h/frame; the emulator is simulating the PL and the DMA/NoC
traffic. Vectorizing a kernel speeds up the *design*, not the emulation of it. Always size runs
from measured wall clock at the same `N_CHANNELS`.

### Next, in order

0. **The training target is fixed in the host and needs a hardware run to score it.** Nothing
   else in this list can be scored until the tracker holds lock. Read `resp00_over_peak` in
   `track.csv`: it must stay flat, not climb at the learning rate.
1. **One build carrying the remaining defects, then a long run.** `IFFT_ROW_SHIFT=4
   IFFT_COL_SHIFT=4` (AIE rebuild), the `conf` gate + structural vetoes on the scale update
   (host, written + native-tested), and `BG_PAN=1` (host, written, not yet on hardware). Then
   `ITER_CNT=200 TRAJECTORY=1 SCALE_TRAJ=1 DUMP_BUFFERS=0` — ~80 s at the new frame time, and
   long enough for the scale filter to settle (~120 frames). Each effect has its own instrument
   in the same log: `rails`/peak for the budget, `resp00_over_peak` for the pan, the `[scale]`
   trace for the gate.
2. **Console gating: DONE in the host, native-verified, not yet on hardware** — `VERBOSITY`,
   see the parameter table and "Validated / done". Run the long run at `VERBOSITY=0`. Raising
   the baud is complementary but the weaker half: deleting the bytes beats sending them faster.
3. Then `gmio_fft_row_out` (73 of the 87 ms). **Run the depth-2 probe first**: arm firings *k*
   and *k+1*, wait on *k*. Host-only, one run, and it can KILL the lock-step hypothesis — if
   286 µs is pipeline latency it collapses, if it is throughput it does not move. Only then
   choose between making weights an **RTP/async parameter** (removes `gmio_weights` outright and
   dissolves the deadlock constraint that forces the interleave; ~73 → ~5 ms, GMIO → ~19 ms,
   ~50 FPS ceiling) and `FFT_ROW_WS/COL_WS` 8→16 (1090 → 578 tx, helps only if the cost is
   per-window fixed). Either halves the per-channel drain, which is what currently hides the
   `ap_done` spin — see "Frame time".
4. Structural, once the floor is ~20 ms: kill the DDR transpose round-trips (4 GMIO ports +
   17 × 64 KB of cache-hostile APU shuffling; 75 of 76 memory tiles are free and a 128×128
   cint16 plane is 64 KB). Then `-O3 -mcpu=cortex-a72` (`GCC_FLAGS` is `-O2` with no `-mcpu`),
   and skipping the `gmio_fft_col_out` F_ch tap (4.7 ms) on PSR-gated frames.
5. Affine perturbations for init (Bolme §3.4) — currently the N=1 case; Fig. 3 puts
   second-frame PSR at ~4 for N=1 vs ~19 for N=8.
6. Video decode loop in `mosse_tracker.cpp` (OpenCV or V4L2).
7. RGB features — see above.

**Done:** the 503 ms KDS completion latency (was item 1) — root-caused to an undelivered CU
interrupt and fixed by `ROI_CROP_USER_MANAGED=1`, 8.26 s → 0.88 s. µs/tx is measured —
80.3 µs/tx, 88 ms/frame, per-transaction overhead rather than bandwidth.

Two principles that have repeatedly earned their keep: **instruments before changes**, and
**never move two magnitudes at once**.

### Validated / done

- **roi_crop as a user-managed CU (`ROI_CROP_USER_MANAGED=1`), hardware 2026-08-20.** 8.26 s →
  **0.88 s/frame**; the launch path went 8055 ms → 0.085 ms and `drain → poll` 503.4 ms →
  0.020 ms. `CropIp` (`mosse_tracker.cpp`) wraps `xrt::ip`, writes the AXI-Lite argument
  registers and polls the CU's own `ap_done`. Details that mattered: `xrt::ip` in 2025.2 imposes
  **no control-protocol restriction** (`xrt_ip.cpp`'s ctor needs only IP_LAYOUT presence, a base
  address and a range — the `ap_ctrl_chain` worry was unfounded); it takes an **exclusive** CU
  context, so the `xrt::kernel` for roi_crop must not be constructed at the same time
  (`Runtime.rw_shared=true` relaxes this if ever needed); `frame_buf` is an `m_axi` pointer and
  needs `bo.address()`, which `set_arg(0, bo)` used to supply implicitly; the register map came
  from `cu_info` on the board, not from guesswork. Both paths compile and print the same
  RC_*/timeline tables, so one log compares them. **Correctness verified by fingerprint**: `F_ch
  max|.|=2010`, `H(q15) max|.|=9363 at (6,6)`, `Q1.15 scale 6.982e+08` — identical to the KDS
  runs, so the offsets and the handshake are right.
- **Console gating (`VERBOSITY`), NOT YET ON HARDWARE.** Three levels; the per-frame
  instrumentation tables now print on the **first and last** frame only. Details that mattered:
  **(1)** `VERBOSITY` is a compile-time constant and the `VP1`/`VP2` macros are `if (VERBOSITY
  >= n)`, so the format strings are **dead-code-eliminated, not merely skipped**. Verified
  statically without hardware — `strings` on the ELF at each level: `V=0` holds only the compact
  line plus the anomaly strings, `V=1` adds the per-frame block, `V=2` adds the 96 per-channel
  lines. That is a stronger check than reading a log, and it costs one `make application`.
  **(2)** `dma_accumulate_frame()` had to be split out of `dma_report_frame()`. The printer was
  also the accumulator, so gating the print to two frames would have silently turned the
  CUMULATIVE report at exit into a two-frame report — a number that still looks plausible after
  it stops meaning what it says.
  **(3) Anomalies print at every level**: a railed bin (`report_cint16` prints regardless when
  `rails > 0`, and `rails` is the one number `track.csv` does not carry), a PSR or scale HOLD, a
  peak-definition disagreement, a negative peak. Silencing those to save console is how a
  shift-budget hunt goes wrong.
  **(4) `VERBOSITY=0` still prints one line per frame, deliberately.** Gating to literally
  nothing would delete the instrument that produced the measurement — `picocom | ts` needs a
  per-frame marker to time. ~4 ms against an ~87 ms floor is the right price.
  **(5)** The `expected displacement` hint was gated to level 1 rather than left unconditional:
  its `ok` criterion derives from `IMPULSE_DR`, so under `TRAJECTORY=1` it fires on healthy
  frames and would have been most of the console on exactly the long run this exists for.
  Also guarded the unconditional `NOTE: hw_emu wall time…` line behind `HW_EMU_BUILD` — it
  printed on `TARGET=hw` and had already caused genuine hardware numbers to be discounted.
- **Training target re-centred on the measured displacement, native-tested, NOT YET ON
  HARDWARE.** `filter_update()` now gets a per-frame G built at `(psr_abs.dr, psr_abs.dc)`;
  `filter_init()` keeps the centred one. Two closed-loop regression tests, both hardware-free
  (`scripts/mosse_loop_sim.py`, `run_training_target_tests()` in `make test_host`) — a single
  update cannot see this defect, which is why `gen_filter_golden.py` passed throughout. See the
  Correctness traps entry for the derivation and the numbers.
- **Scale-update gate (`scale_gate()`), native-tested, NOT YET ON HARDWARE.** Three vetoes
  reported separately for the same reason `GateReason` is an enum — "size held" alone does not
  say whether the filter was uncertain, structurally wrong, or clamped. `AT_SEARCH_RAIL` is
  checked *before* `LOW_CONF` because it is the more specific finding when both fire, which is
  the usual case. **The update skip is the load-bearing half**, not the box hold. Guarded so a
  degenerate filter cannot veto everything (`n_scales > 2`) and so "the gate never ran" (frame 0,
  occluded, PSR-gated) is distinguished from "the gate said no". 21 assertions in
  `make test_host`, driven by the values hardware actually produced (idx −12/conf 1.22,
  idx +16/conf 1.57) rather than invented ones, plus both rails, the `SCALE_TRAJ` envelope
  bounds, and `SCALE_CONF_MIN=0` still vetoing structurally.
- Full chain on hardware: roi_crop → PatchIn → conv2d → B1 → row FFT → transpose → col FFT →
  cmul(H_SHIFT) → B2 → IFFT rows → transpose → IFFT cols → response → PSR gate → filter update.
- **`mean_prev` seeding** (`mosse_tracker.cpp`, before the first `weights_bo.sync`): seed
  `mean_prev = bias_acc >> out_shift` for every channel at startup, since Stage A delivers a
  zero-mean patch. Without it Stage B1 is inert on frame 0 — the one frame the filter is trained
  from — and the ch16 response rails flat (ratio 1.00). Fixing it took F_ch from 32768 (railed,
  11 bins) to 53, accum 5264 → 70, and peak/sidelobe 3.59 → 23.04.
- **PSR gating (Bolme §3.5).** Four veto reasons, reported separately because "HOLD" alone is
  unactionable at these frame times: `ZERO_RESPONSE` (peak 0 — pipeline produced nothing),
  `FLAT_SIDELOBE` (sdev 0 — PSR undefined, not infinite), `NEGATIVE_PEAK` (anti-correlation),
  `LOW_PSR` (< 7.0). Only `LOW_PSR` is disabled by `PSR_GATE_MIN=0`. On a gated frame the host
  **holds the position** (moving to a noise peak walks the ROI off the target permanently) and
  skips `publish_filter` — required, not an optimization, because `filter_quantize_q15` reads
  `g_energy`, which the per-channel loop has already overwritten with the occluder's energies.
  Proven on hardware: PSR 3.90 → HOLD → reacquire with `err=0 px`, and the frozen filter is
  byte-identical to the ungated run's.
- **conv2d and cmul_accum vectorized, bit-identical.** conv2d: 16 px/iteration via `aie::mac`;
  use `aie::downshift` (arithmetic/floor, matching C++ `>>`) never `srs` (rounds to nearest),
  and `aie::unpack` not `cast_to` to widen the Hann table. cmul: `from_vector(acc, S)` seeds the
  accumulator exactly, so the accumulator must be folded in BEFORE the shift — converting the
  product to cint16 first clamps twice. `rounding_mode::positive_inf` and
  `saturation_mode::saturate` are load-bearing. `alignas(32)` on the tile-local buffers is
  required: **x86sim does not enforce alignment**, so omitting it passes every bit-exactness
  check and then misbehaves on hardware.
- **DMA 4258 → 1090 tx/frame** via `FFT_ROW_WS`/`FFT_COL_WS` 2→8 (chunks 1024→4096 B). 96% of
  the traffic was the four per-invocation-chunked ports. Counts match the formula exactly on
  hardware.
- **x86sim bit-exactness harness** (`make x86sim_check`). conv2d s6, cmul s7 and cmul_stress all
  16384/16384 identical. This verifies `simulate_conv2d`, on which the whole offline
  `phase1_sweep.py` methodology rests.
- **`roi_crop` bit-exact, 17 cases, zero tolerance** (`make test_roi_crop`, golden from
  `scripts/roi_crop_ref.py`, which `phase1_sweep.py` also uses). 11 cases execute the bilinear
  interpolator for the first time — every build to date runs `roi_h == patch_rows`, which makes
  the entire datapath collapse to a copy. **Bit-exactness says nothing about timing**: this
  suite passed throughout the period when `crop_run.wait()` was costing 98% of the frame.
- **`roi_crop` cycle counts, measured (hw_emu VCD, 64×64, `recompute=1`).** `ap_start` →
  `ap_done` = **245.5 µs = 76,725 cycles** at 312.5 MHz, and `ap_done` asserts at the *same
  instant* as `TLAST`. Per pass: PASS1 44,600 cyc, NORM 4,100, PASS2 27,900. **Two source
  comments in `roi_crop.cpp` are wrong**: PASS2 achieves **27.2 cycles/beat, not II=1** (real
  AIE backpressure — `TREADY` gates the stream at ~87 ns/beat against conv2d's scheduled
  ~35 ns per 4-px beat, so the AIE consumes ~2.5× slower than its compiler schedule), and PASS1
  achieves **10.9 cycles/output-px, not II=4** (m_axi latency on the four scattered bilinear
  taps). Both are real and both are small — 89 µs and 143 µs against the 505 ms of host overhead
  that dominated until 2026-08-20. Now that the launch path costs 0.085 ms/frame they are
  visible in principle, but they sit inside the 4.8 ms per-channel drain and are still ~5% of it.
- **Bounding-box state, ROI padding, σ anchoring.** State is a `TargetBox`;
  `roi = box × TARGET_PADDING`. `TARGET_H/W=64` with padding 2 gives roi=128 — exactly the old
  geometry, so adopting the box was a single-variable change. `patch_dr_to_frame` /
  `frame_dr_to_patch` are unit-tested (33 assertions incl. an anisotropic box), because while
  `roi_h == patch_rows` the two were accidentally the same number and padding breaks both by the
  resample ratio — a tracker that localises confidently and *drifts*, invisible to `err=0 px`.
- **DSST 1-D scale filter** — the existing filter at `rows = 1`; `filter_init`/`filter_update`
  read geometry from state and `gaussian_target_spectrum(G,1,S,σ,0,0)` degenerates cleanly.
  Defaults from §6.1: S=33, a=1.02, η=0.025, σ_s=S/16, template capped at 512 px. 19 analytic
  assertions; a single application under-corrects (+3 where +5 is exact) because the response is
  a discrete peak smoothed by a σ=S/16 target — the property to assert is that repeated
  application converges monotonically, which it does to 0.5%. **Cost 1.54 ms/frame on x86 but
  13.95 ms/frame ON THE A72** — 9×, measured 2026-08-20, and 7.7% of the frame. Do not size
  anything from the x86 figure.
- **Test-sequence generation for long runs**: static background generated once with dirty-rect
  restore; `TRAJECTORY=1` closed elliptical path (peak step 9.42 px/frame, ROI in-frame over
  2000 frames); `SCALE_TRAJ=1` sinusoidal size (0.99%/frame peak vs the filter's 2%/frame step);
  per-frame IoU and centre error with an OTB-style run summary. All defaults reproduce the
  previous behaviour exactly. **The legacy scheme plants the target at the tracker's own
  estimate plus a constant, so ground truth follows the tracker and `err=0 px` is nearly
  self-fulfilling**; `TRAJECTORY=1` makes drift real and measurable.
- **`OCCLUDE_START` is a warm-up.** Default 30 ≈ 4 time constants at `MOSSE_ETA=0.125`. The
  scale filter is 5× slower (~120 frames to settle), so occluding a converged *size* estimate
  wants `OCCLUDE_START=120`.
- Earlier: cmul_accum saturation fix, conv2d hang root-cause (weights starvation), rootfs fix,
  uint8→int8 contract fix, `transpose_inplace`, filter init/update + Q1.15 export, H_SHIFT,
  scenarios s6/s7, hw_emu PLIO smoke test, N_CHANNELS=16 in aiesim (accum 7728 = 24% of cint16 —
  the cint16 DDR accumulator is sufficient).

## Settled questions — do not reopen

- **`eps_rel = 1e-3` is optimal.** The response has a closed form `R = G·B/(B+ε)`. Sweeping the
  integer pipeline: ratio 12.90 / 13.96 / **16.15** / 10.62 / 4.01 at ε = 1e-5…1e-1. Two
  mechanisms pull opposite ways. Bolme Fig. 4's flat curve does not transfer — his ε is absolute
  on the denominator, ours is relative to `mean(B)`.
- **ReLU off beats ReLU on, by ~3×, and the `bias_acc` "fix" alone makes things worse.** Held-out
  peak/max-sidelobe: base(ReLU) 12.82, bias-corrected(ReLU) 3.92, bias-corrected(no ReLU) 16.25.
  `base` only looks decent because its oversized `bias_acc` makes ReLU a no-op on 11 of 16
  channels — the bank is accidentally almost linear. A DCF is linear in feature space; a
  half-wave rectifier throws away half the signal and the filter cannot undo it. Caveat: one
  patch (s6), held out by circular shift, and it diverges from Danelljan §3.3.
- **Padding ≥2; recommend 2.0 — but the 1.5 verdict is REOPENED.** Held-out at target 64,
  budget 4-2-2, ch16: padding 1.5 gives PSR 18.4 / ratio 2.59 / 0.75 px error; 2.0 gives
  45.7 / 4.97 / 0 px; 2.5 and 3.0 edge it out but trigger the aliasing detector (bilinear has no
  prefilter) and 3.0 clips 3.57% of samples. 2.0 is the only value where the resample stays 1:1,
  which is still a good reason to keep it. **But that holdout used a STATIC scene, where
  background lock costs nothing** — and background lock is precisely what more padding buys you
  more of (at padding 2.0 the target is 27% of ROI area; at 1.5 it is 44%). The measurement was
  blind to the failure mode, so it cannot arbitrate between them. Re-test under `TRAJECTORY=1`
  with `resp00_over_peak` as the metric before treating 1.5 as settled.
- **σ stays 2.0; PSR cannot select σ.** Bolme PSR is monotone decreasing in σ all the way to
  sub-pixel (80.3 at σ=0.75 down to 19.3 at σ=5.33), so it just rewards a sharper peak. The
  tempting Stage-B2 explanation was tested with `--no-b2` and refuted — the dependence is
  intrinsic to the metric. σ needs real video, or a holdout with scale/appearance change, to
  arbitrate. The DSST rule is available as `SIGMA_FROM_TARGET=1`.
- **fDSST's PCA compression is not worth it here.** Measured per frame (d=484, S=33): today
  1.136 ms; real-input DFT + Hermitian symmetry **0.365 ms (3.11×)**; full PCA 0.713 ms (1.59×),
  of which the QR is 0.46 ms and the compressed DFTs 0.012 ms. The QR is O(d·S²) — the same order
  as the DFT it eliminates. **Do the real-input DFT instead** (features are real by
  construction). After that the real levers are d and S, not PCA. But this is optimising a
  non-bottleneck: 1.5 ms against a 13-22 ms frame.
  **REOPENED 2026-08-20: both halves of that premise are measured wrong.** On hardware the scale
  filter is **13.95 ms/frame**, not 1.5, and the frame is **181 ms**, not 13-22. The real-input
  DFT's 3.11× is therefore worth ~9 ms/frame — 5% of the frame, for a change that is already
  justified on its own terms (the features are real by construction). The PCA verdict stands;
  the "non-bottleneck" verdict does not.
- **HALVING THE HOST FILTER ON HERMITIAN SYMMETRY DOES NOT WORK. Measured 2026-08-21, aiesim,
  and the premise is false in fixed point.** `conv2d` emits cint16 with `imag = 0`, so the 2-D
  spectrum of a real input is Hermitian *in exact arithmetic* and A/B look half-redundant —
  worth ~4 ms across `filter upd+quant`, `publish` and `unpack F_ch`. Measured on the real
  chain (64×64 ch1, s6, `make aiesim_plio`), the F_ch tap gives:

  ```
  max|residual| = 12 LSB at bin (0,1), mean 1.379, 3924/4096 bins asymmetric, max|F| = 444
  ```

  **95.8% of bins differ from their conjugate partner.** DSPLib's DIT butterflies do not compute
  a conjugate pair by symmetric operations and every stage rounds, so the symmetry is destroyed
  at the ~1 LSB level everywhere.

  **THE CONTROL IS WHAT MAKES THIS CONCLUSIVE.** The float golden for the same scenario,
  quantised to int16, is Hermitian to **0 LSB across all 4096 bins**. So the index convention is
  right, conv2d's output really is real, and int16 storage alone preserves the symmetry — the
  asymmetry is attributable to the fixed-point FFT and to nothing else. Without that control a
  12 LSB residual could just as easily have been a wrong conjugate index.

  **AND THE ERROR LANDS IN THE WORST POSSIBLE PLACE.** The residual is additive rounding noise
  (mean 1.4 LSB, and this file already establishes that DSPLib's cint16 loss is "additive, not a
  gain factor"), so at 128×128 where max|F| ≈ 2010 it would be ~0.6% rather than 2.7%. Still
  >> the ~0.003% that one int16 LSB of H represents — but the decisive point is *which* bins
  suffer. **max|H| sits where |F| is SMALLEST**, because that is where the regularised inverse
  peaks (recorded under H's quantization ceiling). A fixed ~1.4 LSB asymmetry is proportionally
  most damaging exactly at the low-|F| bins, i.e. mirroring would inject its largest errors into
  the largest filter coefficients.

  Cost of finding out: one 20-minute aiesim run and no board time. The check is now permanent in
  the harness (`F_ch Hermitian check`), so any future FFT change re-tests it for free.
  **What is NOT refuted**: the same symmetry argument for the AIE's own forward transform —
  DSPLib's `fft_ifft_2d_graph` halves its memory tile for real input via a different kernel
  (`fft_dit_2ch_real_graph`), which computes only the independent half rather than computing
  both and discarding one. That saves work instead of reconstructing it, so per-stage rounding
  never enters the argument.
- **Channel pruning is moot** with ReLU off — no structurally dead channels remain. (The
  grayscale collapse does leave ch0/ch9/ch14 collinear up to sign, i.e. 14 independent filters,
  and collinear channels add exactly coherently in the accumulator. The real fix is RGB.)

## Known issues and traps

### Measurement / methodology

- **`runs/.last_cfg` IS NOT AUTHORITATIVE. THE FLAGSTAMPS ARE.** For `run_0820_1418.log`,
  `.last_cfg` recorded `ITER_CNT=500` and `IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3`. The run actually
  executed **200 frames at 4-4-4** — settled from `build/hw/.../app.flagstamp` (`DITER_CNT=200`)
  and `aie.flagstamp` (`FFT_2D_TP_IFFT_ROW_SHIFT=4`), corroborated by the cumulative DMA report
  (218000 tx / 1090 = 200). The stamps are written by the recipe that runs the compiler, so they
  cannot disagree with the binary; `.last_cfg` is written separately and drifts.
  **Read the flagstamp, and diff it against `AIE_FLAGS` before an expensive run:**
  ```bash
  printf 'printvar:\n\t@echo "$(AIE_FLAGS)"\n' > /tmp/pv.mk
  make -f Makefile -f /tmp/pv.mk printvar TARGET=hw | tail -1 > /tmp/aie_now
  diff <(tr ' ' '\n' < /tmp/aie_now) <(tr ' ' '\n' < build/hw/.../aie.flagstamp)
  ```
  Quoting differs (the shell strips it); only the VALUES matter. Identical values ⇒ no graph
  rebuild. `make -n` cannot answer this — the stamp is a `FORCE` prerequisite, so `-n`
  conservatively shows the dependent recipe whether or not it would actually run.
  The startup banner now prints `ITER_CNT`, `VERBOSITY`, `DUMP_BUFFERS` and the shift budget, so
  new logs are self-describing; the shifts reach the host as report-only `*_CFG` defines from the
  same Makefile variables the graph gets.
- **VERIFY THE CARD, NOT THE IMAGE. Three consecutive board runs executed a stale ELF.** The
  multi-slot USB reader keeps its device node whether or not a card is inserted: with a card it
  is 59.5 G, without one the empty slot reports a **phantom 2 TB**, same model string
  (`STORAGE DEVICE`), same `removable=1`, same `/dev/sdb`. So `dd of=/dev/sdb` looks correct at
  the prompt either way and silently writes to nothing. **Size is the only discriminator.**
  After flashing, verify the *card*:
  ```bash
  sudo cmp -n $(stat -c %s "$IMG") "$IMG" /dev/sdX && echo "CARD MATCHES IMAGE"
  ls -1 /media/karolina/*/ | wc -l     # 10 on a fresh image; more means it was never written
  strings -a /media/karolina/*/mosse_tracker.elf | grep -c '<a string only the new build has>'
  ```
  A freshly `dd`'d card holds exactly 10 files (9 before `xrt.ini` was added to
  `make package` on 2026-08-18). If it holds hundreds of `*.bin`, those were
  written by a running board and the flash did not land. There are also **two physical SD
  cards** in circulation — label them; "I reflashed it" and "the board booted it" are different
  claims. Cheapest in-run check: put a unique string in every build and grep the boot log for it
  before waiting on results.
- **hw_emu packaging stalls on `udevadm settle` when the reader is plugged in.** Repeated
  "Timed out for waiting the udev queue being empty", ~120 s each, turning a 45 s package into
  10 min. Not a failure — v++ completes and the image is correct. Unplug the reader.
- **Check `CONV2D_MODE` before every expensive run.** The Makefile default was `1` (echo) until
  2026-08-14 and nothing in the build output says so; it cost a ~28 h ch16 baseline whose numbers
  all had to be requalified. In echo mode conv2d returns at the top of the function: no 3×3 MAC,
  no ReLU, no B1, **no Hanning window** (so B2's 9-bin identity does not hold), and all 16
  channels are bit-identical (so the accumulator sums 16 perfectly coherent copies and every
  amplitude and PSR figure is inflated). `roi_crop`/Stage A is unaffected.
- **hw_emu wall-clock timings are not hardware timings** — but **hw_emu SIMULATED PL CYCLES
  are**, and that distinction is what made the roi_crop diagnosis possible without hardware.
  The host runs on QEMU through `Runtime.hw_em_driver`, so any host-side latency measured there
  is meaningless (`gmio_fft_row_out` reported 210,925,994 µs/tx in one run). The PL is simulated
  at RTL, so `ap_start`→`ap_done` cycle counts transfer directly. **Use `make debug_sim &&
  make probe_emu`** — the probe defaults to `PROBE_CU=roi_crop_0 PROBE_PORT=patch_out` and
  captures 26 signals including `ap_start`/`ap_done`/`ap_idle`, both ends of the AXIS handshake,
  the `m_axi` handshake and per-sub-loop `ap_start`/`ap_done`. A 64×64 `N_CHANNELS=2
  ITER_CNT=1` run costs ~1-2 h and exercises both `recompute=1` (ch0) and `recompute=0` (ch1).
  The VCD is written incrementally, so a killed run still yields a parseable file. This answers
  PL-side questions outright and host-side ones by elimination.
  Note `ap_int/ext/str_blocking_n` do not resolve in this build; the same information comes from
  sub-loop `ap_done` plus the handshakes.
- **The `NOTE: hw_emu wall time is not real hardware time` line printed with every DMA report is
  UNCONDITIONAL** (`mosse_tracker.cpp`, no `TARGET` guard). It is not evidence a run was
  emulated, and it caused a set of genuine hardware numbers to be discounted.
- **`[MISMATCH vs injected offset]` is meaningless under `TRAJECTORY=1`.** The criterion derives
  `exp_dr` from `IMPULSE_DR`, so it fires on healthy frames; the companion print
  `"target at pos+(%d,%d)"` also reports the legacy offset rather than the trajectory position.
  IoU and centre error are the valid scores.
- **The offline model's ratios and orderings are sound; its absolute magnitudes are
  patch-specific.** `phase1_sweep.py` runs on the s6 patch; hw_emu injects a synthetic target
  through real `roi_crop`, which after log/z-score is far hotter — it predicted accum 162 where
  hardware gave 5264. With `mean_prev` seeded the model agrees with hardware to 3-11%.
- **Two different statistics are both called PSR.** The aiesim `snr_ratio_pct` is
  `|peak| / max|sidelobe|`; Bolme's is `(g_max − µ_sl) / σ_sl`. Same 11×11 circular exclusion,
  different statistic — they differ by several times and neither's thresholds transfer.
  `report_psr()` prints both, labelled. Where they disagree (5.11 vs 33 in one run, 17.4 vs 125
  in another), Bolme's is the meaningful one for occlusion and the ratio is contaminated by
  mainlobe width.
- **PSR must exclude the mainlobe or it asserts nothing** — a neighbour of a σ=2 peak sits at
  0.88 of it. Use Bolme's 11×11 exclusion with *circular* distance, since the map wraps.
- **s7's PSR threshold is geometry- and budget-dependent.** 15× was calibrated at 64×64 / 3-0-6 /
  ch1 (accum 466); at 128×128 / 4-3-3 / ch1 the accumulator reaches 284 and the measured ratio is
  11.8, which the offline model predicted (10.69). A FAIL from that threshold alone is not
  evidence of a defect until it is re-derived. **ch1 is the documented worst case for PSR** —
  channels add coherently, their quantization noise does not.
- **`err=0 px` is a weak pass criterion.** It cannot see mainlobe width, drift, a DC pedestal, or
  a gated frame (where a mismatch is a *pass*). Report the response profile and both PSR
  statistics too.
- **PSR is a weak pass criterion too, and in the same direction.** A tracker 179 px off target,
  confidently locked to a patch of background, reported **PSR 33**. Bolme's statistic measures
  how *peaked* the response is, not whether the peak is in the right place. **IoU is the only
  metric in the harness that can fail a confidently-wrong tracker** — which is why `track.csv`
  exists and why every run should be read from it rather than from the console.
- **A centred test impulse cannot validate localisation** — `peak_detect_sw`'s old scan returned
  index 0 on an all-zero response, i.e. displacement (0,0), the right answer produced without
  reading the data. The impulse is injected at `pos + (IMPULSE_DR, IMPULSE_DC)` = (10,−7),
  asymmetric and opposite-signed so a transpose and a sign flip are both caught.
- **`SIM_WALL_TIMEOUT` scales with patch area** (`SIM_PATCH_SCALE`); a timeout looks exactly like
  a deadlock. Check for `Error 124` before concluding "deadlock". Also check the right process:
  `ps -C aiesimulator` shows only bash wrappers at 0% CPU; the simulator is `aie2simmsm`
  (~190% CPU, ~2.5 GB RSS). A quiet log is normal — 10+ minutes between prints.
- **Verify flags reached the build.** `%.flagstamp` prerequisites force a rebuild now, but still
  grep `aiecompiler.log`. Order matters for `:=` in the Makefile — a stamp variable defined above
  its inputs expanded to `/aie.flagstamp` with an empty flag list and silently disarmed a test.
  See [[feedback-verify-the-build-ran]].
- **Test vectors can sit below the fixed-point floor** — s1's amplitude-1 impulse quantized to
  almost nothing (20/4096 bins non-zero). Now `GEN_IMPULSE_AMP=100`; s0/s2/s3/s4 are still
  amplitude 1. See [[aiesim-quantization-floor]].
- **`gen_aiesim_vectors.py`'s float Stage A differs from the kernel on 40.9% of samples**, by up
  to 2 LSB (rms 0.65 on a signal of std 32). Small, but ~2% relative.

### Correctness traps

- **THE FILTER IS TRAINED AGAINST A CENTRED G ON A PRE-UPDATE CROP. This is the root cause of
  the growing (0,0) peak, and "background lock" was the wrong explanation for it.**
  **FIXED IN THE HOST 2026-08-20, native-tested, NOT YET ON HARDWARE.**
  The defect: `mosse_tracker.cpp` passed `g_target` — the Gaussian centred at (0,0), built
  once — to `filter_update()`, while `g_F_all` is the patch cropped at the **pre-update**
  position, where the target sits at the measured displacement `(dr,dc)`. So every accepted
  frame teaches "a patch with the target at (dr,dc) peaks at (0,0)", and the bias compounds at
  `eta=0.125` until the zero-shift peak wins. Two independent confirmations:
  **(1)** `resp00_over_peak` follows `1-(1-eta)^k` — 0.125/0.234/0.330/0.414/0.487/0.551 against
  measured 0.08/0.29/0.42/0.38/0.65/0.69 — i.e. it is governed by the LEARNING RATE, not the
  scene, and frame 1 is near zero because frame 0 trains on a genuinely centred target.
  **(2)** `BG_PAN=1` cut background zero-shift correlation 6.6x (0.60 → 0.09, offline sweep) and
  changed the curve **not at all**: 0.08/0.22/0.30/0.40/0.57/0.78 static vs
  0.08/0.29/0.42/0.38/0.65/0.69 panned, collapsing at frame 7 and frame 6 respectively. Run
  `mean IoU 0.1708`, 9.0% overlap precision, 181 of 199 frames lost.
  **THE SIGN, derived rather than guessed, because it is easy to get backwards.** Let `Q_t` be
  the patch cropped exactly ON the object, so the patch actually held is `P_t = Q_t` shifted by
  `+d_t`. Correlation is shift equivariant, so `resp(P_t) = resp(Q_t)` shifted by `+d_t`. An
  on-target patch must peak at 0, so `resp(P_t)` must peak at `d_t` — **G is centred at `+d_t`,
  the SAME sign as the detected peak.** Training at 0 instead teaches `resp(Q_t) → peak at
  −d_t`; the next frame's patch is `Q_{t+1}` shifted by `d_{t+1}`, so that contribution lands at
  `d_{t+1} − d_t`, which under near-constant motion is exactly `(0,0)`. **That derivation
  predicts the learning-rate law in (1) analytically** — it was not only fitted to the data.
  **Fix applied**: `filter_update()` is now handed a per-frame `g_target_shift` built by
  `gaussian_target_spectrum(..., sigma_r, sigma_c, psr_abs.dr, psr_abs.dc)`. By the shift theorem
  this IS "re-crop at the new position", exactly and for free. `filter_init()` on frame 0 keeps
  the centred `g_target`, which is correct there — that crop really is centred on the object.
  The exact alternative is a real re-crop at the updated position (Bolme/DSST do this), which is
  16 more roi_crop launches ≈ 77 ms/frame — affordable now, but the G shift is free and the only
  difference is that the Hann window stays centred where the crop was taken.
  **Regression tests, both hardware-free:**
  `scripts/mosse_loop_sim.py` — full-size 128×128 closed loop, both arms, 60 frames: centred
  reaches `resp00/peak` 0.764 and ends 39.7 px off with 23.7% of frames on target; shifted holds
  0.207 flat, 0.00 px, 100%. `run_training_target_tests()` in `make test_host` — a 32×32 closed
  loop plus a direct check that `gaussian_target_spectrum(...,dr,dc)` peaks at spatial `(dr,dc)`.
  **A SINGLE UPDATE CANNOT SEE THIS DEFECT** — `gen_filter_golden.py`'s one-shot check passed
  throughout — which is why both new tests are closed loops.
  **Assert the shape, not an absolute level.** In the 32×32 test both arms start at 0.39: that is
  that scene's own zero-shift autocorrelation, not a defect, and CLAUDE.md's 0.3 healthy ceiling
  is calibrated for the 128×128 ch16 hardware geometry and does **not** transfer — the same trap
  as s7's PSR threshold. What is geometry-independent is that the defect makes the ratio GROW at
  the learning rate (0.39/0.52/0.64/0.77/0.85) while the fix leaves it flat or decaying
  (0.39/0.39/0.38/0.38/0.38/0.37).
  **`BG_PAN` is not wrong and should stay** — real video has camera motion — it just was never
  the fix. Watch it for one thing: on a white-noise background a rigid pan creates a correlation
  peak at exactly the pan offset (the toy locked to (-32,-47) for two frames). The real smooth
  texture does not do this on hardware, but higher-frequency texture could.

- **THE FRAME BUFFER WAS NEVER SEEDED WITH THE BACKGROUND. Fixed 2026-08-18.** `scene_init()`
  filled `g_background`, and `g_background` was read in exactly one place — `scene_restore()`,
  which copies only the *previous* frame's dirty rect. `g_dirty` starts empty, nothing ever
  copied the whole thing in, and the `camera_capture` zero-fill that used to initialise the
  buffer is commented out. So the frame the pipeline read was the BO as allocated, plus a
  target, plus whatever narrow rect a previous frame dirtied. Measured by replaying the host's
  own scene functions, fraction of the ROI never written: **frame 0 88.53%** (the frame the
  filter trains on), frames 2+ **6.92%** at `TRAJECTORY=0` / 3.1-4.7% at `TRAJECTORY=1`, and
  **55.68%** without `FRAME_NOISE` — that last being the configuration of
  `run_0_17-08-2026.txt`, whose frames 307-313 report `ratio 1.08x`, peak 7989 against max
  sidelobe 7379, identical to ±3 every frame, and 302 px error. **The cost was dynamic range,
  not false correlation.** The never-written band saturates the int8 rail (clipped count ==
  band count exactly), inflating Stage A's σ ~4.3× and dividing the real content by the same:
  signal std into conv2d 7.0-7.6 instead of 32.2-32.4, i.e. 77% lost normally and 88.7% on
  frame 0. **The tempting explanation — a patch-stationary band that locks the response to
  (0,0) — is WRONG and was tested:** the band sits at patch rows 123-127, where the Hann
  window is 177 against 32767, and those rows carry **7.3e-6** of the windowed patch energy.
  It cannot correlate with anything. This is a different defect from the entry below, which is
  real and separate. Fix is one 2 MB `memcpy` at startup, and it **must** come after
  `rc_control_cu_probe()` (which zero-fills `frame_bo` by design) — putting it next to
  `scene_init()`, where it naturally belongs, lets the probe erase it.
  **CORRECTION 2026-08-20, and it is the interesting part of this entry.** The measured
  "88.53% of the ROI never written on frame 0" was an *offline replay* of the host's scene
  functions, and it assumed the unwritten region held garbage that saturated Stage A's int8 rail
  and inflated σ ~4.3×. On hardware that region is **zeros** — the BO is allocated zeroed and
  `rc_control_cu_probe()` zero-fills it — so there is no rail and no σ inflation. Hardware
  before-and-after: `F_ch` frames 0 and 1 are **unchanged** (1854→2010, 1893→1894), frames 2+
  rise 2.55-3.10×. So the fix is real and worth keeping, but the predicted 9.2× response gain
  never existed, and the shift-budget change it justified (4-3-3 → 4-5-5) was wrong. See
  "Shift budget". **The lesson: an offline replay inherits every assumption the host makes about
  memory it did not write.** Verify the premise on hardware before letting a model move a
  calibrated constant.
- **A byte-identical static background makes the tracker lock onto it, and PSR cannot see it.**
  **SUPERSEDED as an explanation of the growing (0,0) peak — see the training-target trap
  above; `BG_PAN` refuted this one on hardware.** The measurements below stand; the
  mechanism they were attributed to does not.
  `fill_background()` is cached (it is ~0.6-1.2 s on the A72, so caching is not optional) and
  only the dirty rect is restored, so outside the target the frame repeated *to the LSB*. A DCF
  fed a perfectly repeating background correlates with it at exactly zero shift. Measured
  2026-08-17, ch16 4-3-3 `TRAJECTORY=1`: **two competing peaks every frame** — the true motion
  peak at ~(9,−3) and a static peak at (0,0)/(−1,0)/(−1,−1) worth **69-86% of it**. The static
  one won **21 of 48 frames**, and because MOSSE measures only RELATIVE displacement each win
  cost a permanent ~9.4 px offset: centre error stepped 1.35 → 9.56 → 87 → 292 px.
  **PSR read 24-35 throughout** — the response really was sharply peaked, just in the wrong
  place. Neither the PSR gate nor `err=0 px` can detect this; only IoU and `resp00_over_peak`
  can. Fixed by `FRAME_NOISE=2` (re-draw per frame the sensor-noise term that already existed
  in `fill_background()` but was frozen into the cache), applied over the ROI only —
  ~130×130 px, 122× cheaper than the full frame, and the existing dirty-rect machinery undoes
  it. **Watch `resp00_over_peak` in `track.csv`: 0.69-0.86 was broken, 0.07-0.23 is the fix,
  under ~0.3 is healthy.** Real video never does this; sensor noise and camera motion guarantee
  the background does not repeat.
- **`FRAME_NOISE=2` NO LONGER SUFFICES — seeding the frame buffer brought background lock back.**
  Measured 2026-08-20, ch16, `TRAJECTORY=1 SCALE_TRAJ=1`, 20 frames, with both the seeding fix
  and `FRAME_NOISE=2` active. `resp00_over_peak` per frame:

  | frame | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | resp00/peak | 0.08 | 0.22 | 0.30 | 0.40 | 0.57 | 0.78 | **0.86** | 0.04 | 0.25 | 0.63 | **0.82** |
  | peak/max-sidelobe | 3.14 | 2.76 | 2.04 | 1.54 | 1.21 | 1.07 | 1.37 | 2.12 | 2.31 | 1.52 | 1.18 |

  A **sawtooth**: the origin peak grows until it wins, the tracker jumps ~9 px, the ratio resets
  against the new (wrong) position, repeat. It crosses the 0.3 healthy ceiling at frame 4 and
  takes over at frame 7. **Why it came back: the two fixes interact.** Before seeding, the ROI
  was mostly an unwritten flat region with nothing to correlate against; seeding put the real
  static background there, and `FRAME_TEXTURE=1` (the default) makes it band-limited texture with
  genuine structure. The 2.55-3.10× rise in `F_ch` on frames 2+ *is* that background energy —
  the target did not change. 2 LSB peak noise cannot decorrelate it.
  **Discriminating background lock from a DC pedestal** (they look similar in a summary and have
  opposite fixes): frame 7's top-5 bins were `(1,−1) 1318, (0,−1) 1239, (1,0) 1220, (1,−2) 1167,
  (2,−1) 1136` — a *localised blob* at the origin — while the sidelobe mean was −2.9 ≈ 0. A
  pedestal lifts every bin uniformly; this is a genuine correlation peak. B2 was working
  normally (`max|removed|` 1967→4058).
- **`FRAME_NOISE` IS NOT THE FIX FOR BACKGROUND LOCK — and neither, it turns out, is
  `BG_PAN`; the defect was the training target.** Both arguments below are still correct
  about the background, and `BG_PAN` measurably decorrelates it; it just did not move the
  tracker, because the background was not what was broken. Independent additive noise
  cannot decorrelate a static pattern: the background still correlates with itself at exactly
  zero shift, and the added variance inflates the MOSSE numerator and the shared denominator
  alike. Byte-identity was never the mechanism; **high correlation** was. `FRAME_NOISE=2` did
  work on 2026-08-17 — but only because the frame buffer was not yet seeded and the ROI was
  mostly zeros, so 2 LSB of noise really was the dominant varying content. Seeding removed that
  accident and no amplitude recovers it; larger noise buries the target as fast as the
  background. **`BG_PAN` scrolls the cached background under the ROI**, which is what real
  camera motion does, and it is what makes the zero-shift peak impossible rather than merely
  noisy. Sampled from the cached field at a wrapped offset — two memcpys per row over the ROI —
  so `fill_background()` (0.6-1.2 s on the A72) does not re-run.
  **THE MAGNITUDE MUST BE SWEPT AGAINST THE TEXTURE'S WAVELENGTHS, NOT GUESSED IN PIXELS.**
  `scripts/bg_pan_sweep.py` (seconds, no hardware) reports the zero-shift correlation after
  Stage A + Hann + B2:

  | pan px/frame | 0,0 | 3,5 | 7,11 | 15,23 | 23,37 | **31,47** | 47,71 | 63,97 |
  |---|---|---|---|---|---|---|---|---|
  | corr@0shift | +0.60 | +0.61 | +0.64 | +0.54 | +0.31 | **+0.09** | −0.25 | −0.56 |

  The obvious first guess of 3-5 px/frame is **worthless** — the texture is six sinusoids of 1-6
  cycles per frame, so the shortest wavelength is 180 rows and a 5 px shift moves it under 3% of
  a period. Hence the 31/47 default. Re-run the sweep after any change to `fill_background()`,
  `FRAME_TEXTURE` or the ROI size.
  **`fill_background()` rounds its frequencies to WHOLE CYCLES per frame** so the pan's
  wraparound is seamless: with continuous frequencies the row wrap is a **8.10 LSB** mean
  discontinuity across the full frame width against 0.98 LSB between interior rows, a 128-px ROI
  straddles it ~12% of frames at this pan rate, and an artificial edge is exactly what a DCF
  locks onto. Rounded, the seam is 1.02 LSB — indistinguishable from the interior.
  31 and 47 are coprime to 1080 and 1920, so the offsets visit every row and column before
  repeating.
  **This fixes the TEST BENCH, not the tracker.** Real video never repeats to the LSB. Do not
  tune anything in the tracker against this artifact.
- **Conjugation: the stored filter is H, not Bolme's H\*.** `cmul_accum` conjugates itself, so the
  host stores `H = conj(G) ⊙ F / (B + ε)`. Storing Bolme's expression verbatim gives a
  phase-noise response peaking at an arbitrary bin — and this is **invisible whenever the target
  is centred**, since a centred real Gaussian has `conj(G) = G`. That is why s7's target is
  off-centre.
- **H's quantization ceiling is not `2^H_SHIFT`.** The ceiling sets H's *resolution* (always use
  all 15 bits); `H_SHIFT` sets the *product scale*. Coupling them traded one bit of filter
  precision per bit of accumulator gain, i.e. did nothing. **max|H| sits where |F| is smallest**,
  because that is where the regularized inverse peaks, so a spiky filter leaves every informative
  bin far below full scale — at `H_SHIFT=15` the accumulator reached 15 of 32767 and PSR
  collapsed to 5.2 while still localising exactly. The contract is encoded in four files:
  `mosse_filter.cpp`, `gen_filter_golden.py`, `test_mosse_filter.cpp`, s7 in
  `gen_aiesim_vectors.py`. Normalization is by complex *magnitude*, so the largest single int16
  component is typically below 32767 (a bin at 45° puts 32767/√2 in each part) — `rails=1` on one
  frame and `rails=0` on the next with `max|.|=32767` both times is not a bug.
- **The correlation response is SIGNED once Stage B1 is active** (s6 peaks at `{-417,0}`). The
  scan is `|real|`; both peak definitions are computed every frame and a disagreement is
  reported. A wrong-sign filter is caught by `NEGATIVE_PEAK` rather than silently followed.
- **B2's correction is not bit-exact.** The linearity argument is exact in real arithmetic, but
  conv2d's window multiply applies two `>>15` truncations. Measured residual ~1e-3 (vs 2.5e-2…9.9
  without). Fine for argmax; do not rely on it for exactness. With `mean_prev` seeded B2 is
  currently a no-op (`max|removed| = 0`), which is the desired state — revisit only if real video
  puts energy back in those bins.
- **DSPLib's cint16 FFT loss is additive, not a gain factor** — each pass subtracts ~21 from a
  summed DC bin, independent of amplitude. So `row_dc = PATCH_COLS*c − 21`,
  `accum0 = PATCH_ROWS*row_dc − 21`. Any "expected = N" calculation is wrong. An impulse loses
  only ~3. (A "2/3 gain" fits one data point by coincidence — don't fit a scaling law to one
  point.)
- **Preprocessing constants are coupled across engines with no compile-time check.**
  `hanning_*.h` must stay periodic; `mean_prev` at weights bytes `[18:22]` is duplicated in
  `conv2d_kernel.h`, `conv2d_kernel.cpp`, `export_weights.py`, `gen_aiesim_vectors.py`;
  `ROI_NORM_Q` in `roi_crop.h` sets the int8 scale `out_shift` was derived against;
  `FFT_ROW_WS`/`FFT_COL_WS` must reach **both** `AIE_FLAGS` and `GCC_FLAGS` (they didn't, and
  `mosse_tracker.cpp` silently defaults them to 2 — graph and host would have disagreed about
  every DMA chunk count and deadlocked). **General rule: any constant both the graph and the host
  derive from must be passed to both toolchains from one Makefile variable. A `#ifndef` default
  in the host is not a safety net — it is what makes the mismatch silent.** Audit `GCC_FLAGS`
  against `AIE_FLAGS` when adding a shared parameter.
- **`bias_acc`/`out_shift` are calibrated for the wrong input scale** (diagnosis stands; the fix
  is *not* to be applied alone — see the ReLU entry under Settled). `export_weights.py` assumes
  127 ≙ 1.0 while `roi_crop` emits a z-score at `ROI_NORM_Q = 32`, so `bias_acc` is ~4×
  oversized. Q3 verdicts: ch3 and ch15 structurally dead; ReLU provably a no-op on 11 of 16
  channels; signal resolution 7.6–13.0 of 15 bits. The bias also shifts the *signal* down to make
  room for a DC pedestal that Stage B1 subtracts two stages later. Also a semantic mismatch:
  weights quantized against ImageNet-normalized linear luminance, fed a z-score of the log.

### Infrastructure

- **THE CU COMPLETION INTERRUPT IS NEVER DELIVERED ON THIS PLATFORM. Every KDS launch costs
  ~503 ms.** Root cause found 2026-08-20 and it is a platform defect, not anything in this
  design. The CU side is healthy and armed — read straight off the board:
  ```
  devmem 0xa4010004  GIER = 0x1     global interrupt enable ON
  devmem 0xa4010008  IER  = 0x3     ap_done + ap_ready enabled
  devmem 0xa401000c  ISR  = 0x3     BOTH LATCHED, NEVER SERVICED
  ```
  The HLS ISR is toggle-on-write and only a handler clears it, so `ISR=0x3` standing after 100+
  launches proves **no handler has ever run** — an instrument independent of `/proc/interrupts`,
  which reads `0 0` on both `zocl_irq_intc` IRQs (51, 52) across every run. Clearing the ISR by
  hand (`devmem 0xa401000c w 0x3`, verified back to 0) and re-running puts it straight back to
  `0x3` with the counts still at 0 and the timing unchanged. So the CU raises its interrupt on
  every completion, re-latches immediately, and nothing upstream receives it. KDS then falls
  back to a ~500 ms timer. Consistent with the boot-time `zocl-drm zyxclmm_drm: error -ENXIO:
  IRQ index 32 not found`, which appears on every probe and survives reboots, and with IRQ 51
  being registered `Edge` while 52 is `Level` — two trigger types for two identical CUs.
  **Fix is `ROI_CROP_USER_MANAGED=1`** (drive the CU over AXI-Lite, poll its own `ap_done`);
  the interrupt wiring itself would need the base platform's device tree. Useful board-side
  checks, in the order that settles things fastest: `/proc/interrupts | grep zocl`, then the
  three `devmem` reads, then `cat /sys/bus/platform/devices/CU.N.auto/cu_stat` — its `sleep cnt`
  equalled the launch count **exactly** (32 sleeps over two 16-channel runs), i.e. every launch
  slept. `cu_info` on the same directory prints the base address, the control protocol and the
  full argument offset map, which is where `CropIp`'s register map came from.
- **An `xrt.ini` key set as a shell variable is not a test of that key.** `ert_polling=true` at
  the prompt sets an environment variable; XRT reads `Runtime.ert_polling` from `xrt.ini` in the
  process's working directory. One "null result" in the 2026-08-20 hunt was this and nothing
  else. The same file's header already warns that an unrecognised *key* is silently ignored —
  the same failure mode one level up.
- **conv2d weights are consumed per FIRING, not per patch.** `weights` is an `input_buffer` and
  ADF acquires it before every invocation, so the driver must supply `PATCH_ELEMS/CONV_OUT_CHUNK`
  buffers and must start the patch flowing *first* or it deadlocks. This caused every historical
  "PLIO hang". Proper fix, not done: make weights an RTP/async parameter.
  See [[hw_emu_conv2d_fft_hang]].
- **`aie2gm_nb()` transfers one kernel invocation per call, not the full N bytes.** One
  `async`/`wait` pair per invocation on all four output GMIOs, chunked by the producer's output
  window. **The drain loops must be ordered, not just chunked**: the row-FFT drain must
  *interleave* with the weights feed in one loop (1:1 by construction), and `gmio_accum_out` /
  `gmio_ifft_row_out` / `gmio_response` must be drained **before** waiting on the corresponding
  input GMIO. Symptoms of getting it wrong: stall after 6 of 64 weight buffers, DMA status frozen
  at `0x1a080010`, `roi_crop_0` at `ap_start=1, ap_done=0` forever.
  See [[feedback_aiesim_gmio]].
- **`v++ --package` corrupts the 2025.2 rootfs** (ext4 feature mismatch) — every hw_emu run
  panicked at boot. `make rootfs` builds a downgraded copy. See [[vpp_package_corrupts_rootfs]].
- **XRT AXIS ports consume a positional argument slot**, so scalars shift by one. Set args by
  explicit index. This was the real cause of "hw_emu PL→AIE PLIO delivers nothing".
  See [[plio-was-never-broken-xrt-arg-index]].
- **The host does not exit after the last frame.** `gr.end(0)` blocks forever on a free-running
  graph. Cosmetic — all data is already out — but `run_script.sh` never reports RC and the
  emulation must be killed by hand.
- **Probing PL↔AIE signals in hw_emu takes three non-obvious steps**, each of which silently
  defeats capture: `ai_engine_0.S00_AXIS` is a SystemC/TLM socket with no TVALID/TREADY wires
  (use the `VitisRegion/out_r_*` boundary port); v++ elaborates with `--debug off` so `log_vcd`
  aborts the run before Linux boots (`make smoke_debug_sim` re-elaborates); and `elaborate.sh`
  uses relative include paths that fail in the copied `package/` tree **while still printing
  "Built simulation snapshot"**. Probe with `make smoke_debug_sim && make smoke_probe_emu`.
  See [[hw-emu-signal-probing]].
- **The pre-computed FFT bypass in `make aiesim` skips the PatchIn→conv2d→row-FFT path.** Use
  `make aiesim_plio` to validate it. Buffer dumps: [[hw-emu-buffer-dumps]].
