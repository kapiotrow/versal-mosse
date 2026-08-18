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
| `IFFT_ROW_SHIFT` | `5` | Must stay non-zero at high channel counts |
| `IFFT_COL_SHIFT` | `5` | 4-5-5 (total 18) since the background-seed fix raised the signal 9.2×. 4-3-3 was the pre-fix hardware-validated budget and **rails now** |
| `FFT_ROW_WS` / `FFT_COL_WS` | `8` | Rows/cols per FFT invocation — **the DMA transaction-count knob** (2→8 gave 4258→1090 tx/frame) |
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
| `OCCLUDE_MASK` | `0` | Bitmask over frame index: bit *f* ⇒ frame *f* is occluded. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test |
| `OCCLUDE_SQUARE` / `OCCLUDE_START` / `OCCLUDE_PERIOD` / `OCCLUDE_LEN` | `8` / `30` / — / — | Occluder shape and schedule |
| `TRAJECTORY` / `TRAJ_AMP_R,C` / `TRAJ_PERIOD` | `0` | 1 = closed elliptical path (absolute ground truth) |
| `SCALE_TRAJ` / `SCALE_TRAJ_AMP` / `SCALE_TRAJ_PERIOD` | `0` | Sinusoidal size envelope |
| `FRAME_TEXTURE` | — | Band-limited background instead of a flat fill |
| `FRAME_NOISE` | `2` | Per-frame sensor noise, PEAK amplitude in LSB, applied over the ROI. **The one test-sequence default that does NOT reproduce previous behaviour** — see "background lock" below. `0` restores the pre-2026-08-17 (pathological) static background |
| `DUMP_BUFFERS` | `1` | Per-frame binary dumps of F_ch/accum/resp/H_q15. **1216 KB/frame**, ~2 s/frame on the board. Set `0` for any run measuring tracking or FPS |
| `CSV_LOG` | `1` | One row per frame to `track.csv` — gate verdict, both PSRs, peak, displacement, `resp00_over_peak`, both boxes, IoU, centre error. ~40 B/frame, flushed every row |

Artifacts land in `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

**Shift budget.** The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the
response scale, so weight moves freely between passes. Validated on hardware at ch1: **5-2-2**
(F_ch 53, accum 70, response 2534, ratio 23.0, nothing railed, correct σ=2 Gaussian).
**For ch16 at 128×128 use 4-5-5 (total 18) as of 2026-08-18. NOT YET ON HARDWARE.** It replaces
4-3-3 because the *scene* changed, not because 4-3-3 was wrong: the host never wrote the
generated background into the frame buffer (see "the frame buffer was never seeded" under
Correctness traps), and seeding it raises the response ~9.2×. 4-3-3 was genuinely validated on
hardware 2026-08-17 in the pre-fix scene — rails=0 every frame, response 14-26k (43-79% of
range), PSR 24-35 — and it **rails at ~400% of range** in the seeded scene. The +4 bits comes
from an offline re-sweep over the verified integer datapath, arms differing only in the frame
buffer: |feat| 4.3×, |F| 4.4×, but accum/response **8.3-10.5×**, the extra ~2.3× being H's
Q1.15 grid (73702 non-zero `Hq` bins before, 90122 after — `max|Hq|` is 32767 in both, so the
compressed spectrum simply lost bins below 1 LSB). Transfer of the measured 14-26k by 9.2×/2^k:
total 17 → 49-92% (8% from the rail), total 18 → 25-46%. 18 wins on the same argument that
retired 4-2-2. The split is free (invariant holds to 1.3% across all total-17 and total-18
splits); `FFT_SHIFT` stays 4 rather than 5 because that leaves the accumulator at ~1400 instead
of ~330 for the same response. **Read the first hardware run before trusting this**: scene and
budget moved together, which is unavoidable here but breaks "never move two magnitudes at once".

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
| cmul_accum | 0.13 | 30 → 2 cyc/element, now pipelined |
| FFT + IFFT chain | ~2.2 (band 1.3–3.5) | **trip counts inferred, not logged** — do not quote as measured |
| **total** | **~6.4** | from ~21.6 before vectorization |

These are the compiler's scheduled cycles — real cycles on an in-order VLIW core absent memory
stalls, trustworthy for sizing but not a profile. Tile map: `15_0` conv2d, `24_0` cmul_accum,
`15_1`/`29_0` forward FFT, `14_0`/`22_0` IFFT.

### Frame time — MEASURED ON HARDWARE 2026-08-17. The old estimate was wrong.

**Measured: 8.26 s/frame = 0.12 FPS at 128×128 ch16.** The previous table projected 36-75 FPS
from an assumed 2-10 µs/tx and concluded "the design clears 30 fps in both DMA regimes". That
conclusion was false, and it was false for a reason worth internalising: **the instrumentation
covered 1% of the frame.** `DMA_T` bracketed the GMIO calls and nothing else, so the 99% that
was not GMIO was invisible and got estimated instead of measured.

| stage | ms/frame | share | how |
|---|---|---|---|
| **`crop_run.wait()` — roi_crop completion** | **8082** | **98%** | measured, RC_* timers |
| console @115200 baud (8.3 KB/frame) | 720 | 8.8% | derived from log bytes |
| GMIO total (1090 tx @ 80.4 µs) | 88 | 1.1% | measured, DMA_T |
| APU transposes (18 × 64 KB) | ~104 | 1.3% | timestamped console |
| binary dumps (1216 KB/frame) | ~2000 | — | `DUMP_BUFFERS=0` removes it |
| everything else (psr/scale/gate/filter) | ≤130 | 1.6% | timestamped console |

(Shares exceed 100% because console output overlaps compute.)

**roi_crop is the frame.** Not the kernel — the *completion path*. Split four ways on hardware:

```
crop_run set_arg   16 x    0.023 ms     ctor (hoisted, once)  0.442 ms
crop_run start     16 x    0.244 ms
crop_run wait      16 x 8081.845 ms   <- 505 ms EACH
```

The CU is innocent, proven three ways: (1) ch0 does 9× the loop iterations (`recompute=1`:
36864 vs 4096) and is **1.8× faster** (277 vs 492 ms); (2) the weights/row-FFT drain loop
completes in 2-6 ms and cannot do so until conv2d has consumed all 16384 pixels, so roi_crop
has written every AXIS beat within ~5 ms; (3) **hw_emu VCD, 64×64 ch2: `ap_done` asserts at the
same instant as `TLAST`** — 245.5 µs total for the full `recompute=1` path, ~1 ms scaled to
128×128, against 277-505 ms measured. Zero completion latency in the PL.

So the 505 ms is host-side, in XRT's blocking `wait()`. **Not yet fixed** — the next hardware
run carries a `poll(state)` spin timed alongside `wait()` and prints the verdict itself. If
poll returns in ms while wait takes ~500 ms, the spin is the fix and the frame drops to ~0.5 s.

**DMA is not a bottleneck and the fabric is at spec.** 80.4 µs/tx average, but that is
per-transaction *overhead*, not bandwidth: a 64-byte transfer costs 14.4 µs and a 128 KB one
costs 22.8 µs — 2048× the size for 1.6× the time, i.e. latency-bound. The largest transfer
achieves **5.76 GB/s**, inside AMD's own measured GMIO range (5.08 GB/s at 1 port, 10.25 GB/s
at 32). Actual bytes moved are 6.27 MB/frame ≈ **1.1 ms of real transfer** — 1.2% of the 88 ms.
The rest is 1090 × ~14 µs of fixed cost. `FFT_ROW_WS/COL_WS` 8→16 would halve the four 256-tx
ports and save ~7 ms/frame; worth doing only after the `wait()` fix, when 7 ms is visible.

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
make test_host                     # native unit tests for filter/PSR/scale/conversions (seconds)
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
│   ├── mosse_tracker.cpp             # GMIO-driven XRT tracking loop
│   ├── mosse_filter.h/.cpp           # init/update/PSR/scale/Q1.15 export — NO XRT include
│   └── test/                         # native tests + NumPy goldens
├── system_configs/mosse_x1.cfg       # v++ linker
├── profiling_configs/, directives/
└── exec_scripts/run_script.sh
scripts/  export_weights.py, gen_aiesim_vectors.py, gen_filter_golden.py,
          check_collapse.py, check_kernel_bitexact.py, phase1_sweep.py, roi_crop_ref.py
```

## Current status (2026-08-17)

**The tracker runs end to end on real VEK280 hardware at 128×128 ch16 on the real conv path**,
with the box state, patch↔frame conversions, IoU and the DSST scale filter all executing for
the first time. At 4-3-3 with `FRAME_NOISE=2` it tracks correctly for ~20 frames (IoU 0.94-0.99,
centre error 0.00-1.27 px, PSR 24-35, rails=0) and then loses lock to a **scale runaway** —
see below. Earlier: hw_emu 128×128 ch1 `ITER_CNT=2` at 5-2-2 gives `err=0 px`,
peak/max-sidelobe 23.0, Bolme PSR 172, response matching `exp(-d²/8)` to within 4% at d=2, d=6.
PSR gating fires correctly under occlusion and reacquires.

**Two open defects, in order of severity:**

1. **`crop_run.wait()` costs 505 ms × 16 channels = 98% of the frame.** Host-side, not the CU.
   Diagnosis complete (see "Frame time"); fix built but untested — the next hardware run prints
   its own verdict from the `poll(state)` vs `wait()` split.
2. **The DSST scale filter runs away downward.** With the background lock fixed, the box tracks
   correctly to ~frame 12 (est 68 vs truth 71) and then collapses: level −16 (the rail of the
   ±16 search range) at f13, then −7/+8/−14, reaching 31 against a truth of 78 by f24. The
   target is *growing* throughout. **`conf` cleanly separates the two populations** — good
   frames 2.37-3.24, bad frames 0.72-1.85 — **and nothing gates on it.** Three fixes, cheapest
   first: gate the scale update on `conf` (~2.0) as `psr_gate` gates translation; reject
   `|sr.idx| == (SCALE_N-1)/2` as structurally invalid (an argmax on the search boundary cannot
   be right when the envelope moves 0.94%/frame against a 2%/frame filter step); tighten
   `SCALE_MIN_REL/MAX_REL` from 0.25/4.0, which is so loose it never fired.

**Scale collapse is a SYMPTOM, not a root cause, and was misdiagnosed as one.** Reading only
frames 307-313 of the first long run showed a collapsed box and suggested a scale clamp. The
real ordering is: static-background lock → missed frames → position drift → the scale filter
re-learns on off-target patches → collapse. Fixing the background made the scale defect
observable on its own terms for the first time.

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

1. **Confirm the `wait()` fix on hardware.** The build already carries `poll(state)` timed
   alongside `wait()`; one run prints the verdict. If poll returns in ms, replace `wait()` with
   the bounded spin — one line, ~16× on frame rate (8.26 s → ~0.5 s). `xrt.ini` sits next to
   the ELF on the SD card, so XRT config variants (`Runtime.thread_policy`, `cpu_affinity`,
   `kernel_channels`, `Debug.pl_deadlock_detection`) can be A/B'd **without a rebuild**.
2. **Gate the scale update on `conf`** — see the open defects above. Native-testable.
3. Once the frame is sub-second: console becomes the next bottleneck (0.72 s at 115200). Raise
   the baud or gate the 96 per-channel progress lines behind a verbosity flag.
4. Then `FFT_ROW_WS/COL_WS` 8→16: 1090 → 578 tx, ~7 ms/frame.
5. Affine perturbations for init (Bolme §3.4) — currently the N=1 case; Fig. 3 puts
   second-frame PSR at ~4 for N=1 vs ~19 for N=8.
6. Video decode loop in `mosse_tracker.cpp` (OpenCV or V4L2).
7. RGB features — see above.

**Done:** µs/tx is measured (was item 3) — 80.4 µs/tx, 88 ms/frame, and it is per-transaction
overhead rather than bandwidth.

Two principles that have repeatedly earned their keep: **instruments before changes**, and
**never move two magnitudes at once**.

### Validated / done

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
  taps). Both are real and both are small — 89 µs and 143 µs against 505 ms of host overhead.
  They only become visible after the `wait()` fix.
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
  application converges monotonically, which it does to 0.5%. Cost 1.54 ms/frame on x86.
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
- **Channel pruning is moot** with ReLU off — no structurally dead channels remain. (The
  grayscale collapse does leave ch0/ch9/ch14 collinear up to sign, i.e. 14 independent filters,
  and collinear channels add exactly coherently in the accumulator. The real fix is RGB.)

## Known issues and traps

### Measurement / methodology

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
  `scene_init()`, where it naturally belongs, lets the probe erase it. It forced the shift
  budget from 4-3-3 to 4-5-5; see "Shift budget".
- **A byte-identical static background makes the tracker lock onto it, and PSR cannot see it.**
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
