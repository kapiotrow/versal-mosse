# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MOSSE correlation filter tracker with CNN features on Versal VEK280.
Extends the AIE 2D-FFT tutorial (XD073) with a full object tracking pipeline.

## Environment setup

```bash
source setup_env.sh
```

This script sets all required env vars (same as tutorial):
- `PLATFORM_REPO_PATHS`, `XILINX_VITIS`, `COMMON_IMAGE_VERSAL`
- `DSPLIB_VITIS` — Vitis Libraries **root** (e.g. `.../Vitis_Libraries`), NOT the dsp subdirectory. The Makefile appends `/dsp` internally.
- `PLATFORM` — VEK280 XPFM path: `xilinx_vek280_base_202520_1.xpfm`

## Build parameters

| Parameter | Default | Notes |
|---|---|---|
| `TARGET` | `hw_emu` | `hw_emu` or `hw` |
| `PATCH_ROWS` | `128` | Must be a power of 2 (AIE FFT constraint) |
| `PATCH_COLS` | `128` | Must be a power of 2 |
| `N_CHANNELS` | `16` | Number of conv feature channels |
| `FFT_2D_DT` | `0` | 0=cint16, 1=cfloat |
| `ITER_CNT` | `1` | Frames to process in hw_emu. **Needs ≥2** — frame 0 initialises the filter |
| `PL_FREQ` | `312.5` | PL kernel frequency in MHz |
| `H_SHIFT` | `10` | cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget. (Was 15; lowered because a spiky filter left the accumulator at 15 of 32767 — see Known Issues) |
| `FFT_SHIFT` | `4` | Forward FFT shift, per pass. **See the budget note below — the ch16 design point wants 5.** |
| `IFFT_ROW_SHIFT` | `3` | Row IFFT shift. Must stay non-zero at high channel counts |
| `IFFT_COL_SHIFT` | `3` | Col IFFT shift |
| `FFT_ROW_WS` / `FFT_COL_WS` | `8` | Rows/cols per FFT invocation. **This is the DMA transaction-count knob** — raised 2→8 on 2026-08-14, 4258 → 1090 tx/frame |
| `CONV2D_MODE` | `0` | 0 = real 3×3 conv, 1 = echo passthrough, 2 = synthesize. **Default was 1 until 2026-08-14 and that cost a 28 h baseline** |
| `CONV_VECTORIZE` / `CMUL_VECTORIZE` | `1` | Vectorized kernels. Bit-identical to the scalar paths; 0 restores them for bisection |
| `CONV_RELU` | `0` | Half-wave rectifier after the output shift. **0 since 2026-08-14** — see the ReLU entry in Known Issues |
| `B2_NULL_BINS` | `1` | 1 = null the 9 low-frequency bins, 0 = subtract µ·W |
| `PSR_GATE_MIN` | `7.0` | Bolme §3.5 failure threshold. Below it the host HOLDS the position and skips both `filter_update` and `publish_filter`. **0 disables the threshold test** (structural failures still veto) — that is the A/B lever. Applies to Bolme's PSR only, never the ratio. Host-only |
| `OCCLUDE_MASK` | `0` | Occlusion injection, bitmask over frame index: bit *f* ⇒ frame *f* gets a checkerboard instead of the target. Bit 0 ignored. `ITER_CNT=3 OCCLUDE_MASK=0x2` is the occlude-then-reacquire test. Host-only |
| `OCCLUDE_SQUARE` | `8` | Checkerboard pitch for the occluder |

Build artifacts land under `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

**Shift budget, current understanding (2026-08-14).** The invariant
`2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so weight moves
freely between passes. Validated on hardware at ch1: **5-2-2** (run D — F_ch 53, accum 70,
response 2534, ratio 23.0, nothing railed, response a correct σ=2 Gaussian). Modelled for
ch16: **5-3-4** (accum 6.2%, response 32.6%, ratio 36.8). The defaults above are still 4-3-3
and should move to 5-3-4 when the ch16 run confirms it. **Do not size this budget against
railing before checking `mean_prev` is seeded** — the forward-FFT saturation that drove two
separate budget hunts was a frame-0 DC pedestal, not a scaling problem.

## Architecture overview

```
PS (A72) — mosse_tracker.cpp
  Drives all GMIO ports in the per-frame, per-channel loop.
  Runs peak_detect_sw(), transpose_inplace(), and the filter init/update in mosse_filter.cpp.

PL kernels (2 total)
  camera_capture : zero-fill DDR frame buffer (stub; TODO: MIPI RX)
  roi_crop       : DDR frame → MOSSE preprocessing → 32-bit AXIS (int8) → AIE PatchIn
                   Stage A (Bolme §3.1 + Danelljan §3.3): bilinear resample of an
                   arbitrary roi_h×roi_w to the fixed patch size with border clamping,
                   log transform, zero mean, unit L2 norm × ROI_NORM_Q, int8 quantize.
                   Two passes (mean/norm are global reductions) + a stream-out pass.
                   `recompute=1` on channel 0 only; channels 1..15 re-stream the cache.

AIE — single instances, serial per-channel processing (both custom kernels VECTORIZED
      2026-08-14; conv2d 4.1 ms/frame, cmul 0.13 ms, AIE total ~6.4 ms)
  conv2d_kernel      : int8 patch → 3×3 MAC → Hanning window → cint16 feature stream
                       (MobileNet-v3 Small layer 1, INT8-quantized, RGB collapsed to grayscale)
                       NOTE ReLU is OFF by default since 2026-08-14 (CONV_RELU=0) — it cost
                       ~25% of the peak/sidelobe ratio. Stage B1 subtracts mean_prev, which
                       the host now SEEDS before frame 0 (see Known Issues; leaving it zero
                       makes the ch16 response a flat railed map).
  fft2d (FFT2D_graph): PATCH_COLS-pt row FFT → GMIO → DDR; APU transposes; DDR → GMIO → PATCH_ROWS-pt col FFT
  cmul_accum_kernel  : col-FFT stream ⊙ H_ch* + accumulate (int32 intermediates,
                       cint16 accumulator in DDR — saturating; see headroom note)
  ifft2d (IFFT2D_graph): same DDR-transpose pattern as fft2d; PATCH_COLS-pt row IFFT + PATCH_ROWS-pt col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop PL kernel → conv2d AIE kernel (**32-bit**, int8 stream: one
int32 word = 4 packed int8 pixels).

**It is 32-bit, not 128-bit** — this line said 128 until 2026-08-12 and the wrong number
was load-bearing (it is the premise of a since-refuted argument against RGB features; see
Known Issues). `mosse_graph.h:121` creates it with `plio_32_bits` and lines 118-120 record
why: a 128-bit PLIO delivered one 128-bit beat per `readincr`, starving the kernel.
`roi_crop`'s port is `ap_axiu<32,0,0,0>`; conv2d reads `input_stream<int32>`.

This name must match between `mosse_graph.h` (`input_plio::create("PatchIn", ...)`) and
`mosse_x1.cfg` (`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

### GMIO ports (10 total: 5 input + 5 output)

| Name | Dir | Purpose |
|---|---|---|
| `gmio_weights` | DDR→AIE | conv2d INT8 weights per channel |
| `gmio_fft_row_out` | AIE→DDR | fft_rows output; APU transposes |
| `gmio_fft_col_in` | DDR→AIE | APU-transposed data → fft_cols |
| `gmio_fft_col_out` | AIE→DDR | **Broadcast tap** on fft_cols: F_ch for the PS filter update |
| `gmio_cmul_in` | DDR→AIE | [H_ch* \| prev_Σ] packed per chunk (replaces separate gmio_filter/gmio_accum_in) |
| `gmio_accum_out` | AIE→DDR | Updated partial sum after cmul_accum |
| `gmio_ifft_row_in` | DDR→AIE | Accumulated spectrum → ifft_rows |
| `gmio_ifft_row_out` | AIE→DDR | ifft_rows output; APU transposes |
| `gmio_ifft_col_in` | DDR→AIE | APU-transposed data → ifft_cols |
| `gmio_response` | AIE→DDR | Final correlation response → peak_detect_sw |

### Per-frame data flow (mosse_tracker.cpp)

```
camera_capture → DDR frame
For ch = 0..N_CHANNELS-1:
  roi_crop → PatchIn PLIO → conv2d → fft_rows → gmio_fft_row_out → DDR
  APU: transpose_inplace()
  DDR → gmio_fft_col_in → fft_cols → cmul_accum → gmio_accum_out → DDR
After all channels:
  DDR accum → gmio_ifft_row_in → ifft_rows → gmio_ifft_row_out → DDR
  APU: transpose_inplace()
  DDR → gmio_ifft_col_in → ifft_cols → gmio_response → DDR
  APU: peak_detect_sw() → update pos
  APU: filter_init() on frame 0, filter_update() thereafter (mosse_filter.cpp)
```

## Key design decisions

- **AIE-centric architecture**: All FFT/IFFT/conv/cmul compute runs on AIE. PL is only
  camera_capture + roi_crop. APU orchestrates via GMIO DDR round-trips.

- **Serial channel processing**: The single FFT2D and IFFT2D instance are reused across all
  N_CHANNELS (driven serially via GMIO). Minimal PL/PLIO count at the cost of throughput.

- **Transpose in DDR (APU)**: `transpose_inplace()` runs on A72 between row-FFT and col-FFT.
  ~64 KB memcpy + index reorder; acceptable at 30 fps for 128×128 patches.

- **Accumulator in DDR**: The partial accumulator (128×128 cint16 = 64 KB) lives in DDR
  (gmio_accum_out/gmio_accum_in). Fits easily; on-tile storage would require a Memory Tile.

- **Filter init/update on PS, and no FFT library**: `mosse_filter.{h,cpp}` implements
  Bolme eq. 10–12 with a *shared* denominator (Danelljan/DSST form — one reciprocal map
  per frame instead of 16, and better conditioned). It needs no FFT: `F_ch` arrives
  already transformed via `gmio_fft_col_out`, and the target spectrum `G` has a closed
  form (the DFT of a wrapped Gaussian is a Gaussian). KissFFT was removed from the plan
  entirely. Placement rationale: the update is ~2 M MAC/frame, roughly 5% of the
  pipeline's arithmetic, so it is not the bottleneck; PL would be the right home if the
  A72 saturates, and AIE would be the *worst* — the 2 MB of filter state does not fit
  on-tile, so it would pay the highest GMIO orchestration cost of the three while
  converting a problem that is trivial in fp32 into another fixed-point calibration.

- **The filter update runs AFTER peak detection.** Updating first leaks the current frame
  into its own detection and makes tracking look better than it is.

- **`mosse_filter.{h,cpp}` includes no XRT/ADF header** so `make test_host` can compile it
  with the system g++ and check it against a NumPy golden in seconds. The alternative for
  a sign error is a ~90 min hw_emu frame.

- **IFFT normalization**: Row IFFT shift = 0; col IFFT shift = 14 (= log2(128)+log2(128)).
  If aiesim response is 2^14× too large, set col shift to 0 and apply >>14 in APU after
  reading gmio_response.

- **Preprocessing split across PL / AIE / APU**: Bolme's chain does not live in one place.
  Intensity-domain steps (resample, log, zero-mean, unit-norm, int8 quantize) run in PL
  inside `roi_crop` — the data is already streaming through it and PL is nearly empty.
  Feature-map mean removal runs on AIE inside `conv2d` as a scalar subtract, using the
  *previous* frame's mean so no channel buffer is needed (a full channel is 64 KB, which
  exceeds an AIE-ML tile). The residual is cancelled on the APU with a 9-bin
  frequency-domain correction, and per-channel energy normalization folds into `H_ch*`
  for free. Total added cost: <2% of the pipeline's arithmetic, no new AIE tiles.

- **Periodic Hann, not symmetric**: `hanning_*.h` uses `sin²(πi/N)`, not `sin²(πi/(N-1))`.
  The periodic form's 2D DFT has exactly 9 non-zero bins, which is what makes the APU's
  mean correction possible. Measured DC/worst-leaked-bin ratio at N=128 in Q1.15:
  periodic 2.2e5, symmetric 373 — with the symmetric window the correction is not exact.
  Do not "fix" this back.

- **Conv layer implementation**: `conv2d_kernel.cpp` implements 3×3 sliding-window convolution with:
  - Per-channel INT8 weights from MobileNet-v3 Small (pretrained on ImageNet, quantized via Brevitas)
  - ReLU activation applied post-convolution (clamp negatives to 0)
  - Separable Hanning window folded into MAC loop (reduces spectral leakage in FFT)
  - Per-channel quantization: weights loaded via gmio_weights, bias and out_shift precomputed
  - Generate weights with: `make weights` (runs scripts/export_weights.py)

## VEK280 resource budget (measured 2026-08-12)

Device `xcve2802-vsvh1760-2MP-e-S`, 12 GB LPDDR4. From `platforminfo -p "$PLATFORM"` and
the `Work/reports/` mapper output of the 128×128 ch1 build — not from datasheet recall.

| Resource | Available | Used now | % |
|---|---|---|---|
| AIE-ML cores | **304** (38 cols × 8 rows) | 6 (`14_0,15_0,15_1,22_0,24_0,29_0`) | **2%** |
| AIE-ML memory tiles | 76 (38 cols × 2 rows) | 1 | 1% |
| BRAM18 | 1200 | 10 (`roi_crop`) | 0.8% |
| DSP | 1312 | 44 | 3.4% |
| LUT | 520704 | 7694 | 1.5% |
| FF | 1041408 | 7539 | 0.7% |

AIE core clock 1000 MHz (`directives/post_sys_link.tcl`). PL at `PL_FREQ=312.5`; the
platform also offers 625 / 156.25 / 100 / 78.125 MHz.

**The design uses 2% of the AIE array.** Every "we can't afford it on AIE" argument should be
checked against that number first — the binding constraints in this project have been tile
*memory* (64 KB/tile) and host DMA orchestration, never core count.

**`runtime<ratio>` is not utilization.** The mapper report's `Utilization` column shows 0.900
for conv2d/cmul and 0.800 for the FFT kernels; those are the declared budgets from
`mosse_graph.h:142-143` and the DSPLib graphs, not measured occupancy. Do not read
"conv2d is 90% full" from it.

### AIE compute cost per frame — derived from aiecompiler.log (2026-08-14)

**No simulation was needed for any of this.** `aiecompiler` schedules every loop and logs the
result; on an in-order VLIW core those are real cycles absent memory stalls. Source:
`build/hw_emu/128x128/ch1/aiecompiler.log` — **the ch1 build, because it is the one compiled
`CONV2D_ECHO_TEST=0`**; the ch16 log is echo mode and its conv2d figures describe a passthrough.

Tile → kernel mapping, recovered by matching the mangled template args in each `<tile>.ll`
compile section (`Lj128E`=point size, next `Lj0/1E`=NIFFT, next=TP_SHIFT):

| tile | kernel |
|---|---|
| `15_0` | conv2d |
| `24_0` | cmul_accum |
| `15_1`, `29_0` | forward FFT (nifft=1, shift=4) — fft_rows, fft_cols |
| `14_0`, `22_0` | IFFT (nifft=0, shift=2) — ifft_rows, ifft_cols |

**All four FFT/IFFT kernels compile to byte-identical loop schedules** (same point size, same
256-sample window), which is why one set of numbers covers the whole chain.

Loop schedules, as `II after folding` (the log's `-> after folding: N (folded over K
iterations)` line, not the pre-pipelining schedule length):

```
conv2d_kernel.cpp:215  MAC loop        II=37   trip=PATCH_COLS=128   (not further pipelined)
conv2d_kernel.cpp:195  stream read     II=28   trip=PATCH_COLS/4=32
cmul_accum_kernel.cpp:116  arith loop  II=30   trip=CMUL_N=256       (NOT pipelined - see below)
cmul_accum_kernel.cpp:102  vector copy II=2    trip=CMUL_N/8=32
fft_dit.hpp:127  radix stage loops     II=4, 8, 10, 12  (four stages; trip counts NOT in the log)
```

Per frame at 128×128 / ch16 / 1 GHz:

| kernel | derivation | ms/frame | confidence |
|---|---|---|---|
| **conv2d** | (16384px×37 + 128×32×28) × 16 ch | **11.5** | trip counts fixed by `chess_loop_range` |
| **cmul_accum** | (256×30 + 32×2) × 64 inv × 16 ch | **7.9** | trip counts fixed by `CMUL_N`, `COL_CHUNKS` |
| FFT + IFFT chain | ~500 cyc/transform × 128 × 2 passes × 16 ch, + IFFT | **~2.2** (band 1.3-3.5) | **trip counts inferred, not logged** |
| **AIE total** | | **~21.6 ms** | |

**SUPERSEDED 2026-08-14 — both scalar kernels are now vectorized.** The table above is the
*before* picture, kept because the RGB and FPS analysis below is written against it. Current:

| kernel | ms/frame | change |
|---|---|---|
| conv2d | **4.1** | 37 → 8.75 cyc/px; the stream-read loop is now 44% of what remains |
| cmul_accum | **0.13** | 30 → 2 cyc/element, and now pipelined |
| FFT + IFFT chain | ~2.2 | untouched |
| **AIE total** | **~6.4 ms** | from ~21.6 |

**conv2d and cmul are 19.4 of the 21.6 ms, and BOTH are scalar loops on a core whose int8 /
cint16 vector datapath is entirely unused.** That is the headline. Two specifics:

- conv2d: nine scalar `int32×int8` MACs, no `aie::vector`/`aie::mac`/MMUL. AIE-ML does
  **256 int8 MAC/cycle**.
- cmul: 30 cycles per complex element, for four int32 multiplies and two saturating adds.
  It carries `chess_prepare_for_pipelining` on the *copy* loop but deliberately **not** on the
  arithmetic loop — the comment at `cmul_accum_kernel.cpp:111` records that the pragma made the
  assembler OOM. So the design's second-largest kernel is running unpipelined, at ~30× the
  cost of a vectorized complex MAC. **That OOM workaround has never been revisited and is worth
  ~7.6 ms/frame.**

**Vectorizing conv2d and cmul is worth ~19 ms/frame — larger than the entire DMA uncertainty
in the optimistic regime.** With 298 of 304 cores free, this is the largest single lever in the
design, and unlike every other roadmap item it perturbs no calibrated constant: same weights,
same `bias_acc`, same shift budget, same DMA count, same numerics if done correctly.

**Caveat on the FFT row.** conv2d and cmul are solid — their trip counts are fixed by
`chess_loop_range`/`CMUL_N` and visible in the source. The FFT row is NOT: DSPLib's stage loop
bound is `block_size = FFT::block_size(n)`, computed inside the template, and the log does not
print it. The ~2.2 ms is inferred from four radix stages × 8-lane cint16 vectors. **The
conclusion that survives the whole 1.3-3.5 ms band is that the FFT chain is not the
bottleneck** — conv2d alone is 3-9× larger. Do not quote the 2.2 as measured.

### RGB features: the blocker was a miscalculation

`export_weights.py` justified grayscale by claiming RGB "does not divide evenly into 16-byte
beats at 128×128". Both halves are false: the PLIO is 32-bit (see the PLIO section), and RGB
divides exactly anyway — 128·128·3 = 49152 B = 3072 × 16 B = 12288 × 4 B, one row = 384 B =
24 beats = 96 words, also exact at 64×64 and 256×256. **There is no alignment obstacle.** The
design was constrained to grayscale, and thereby to 14 effective feature channels, by a
divisibility check that had no divisibility problem.

What going RGB would actually cost:

- **`conv2d` is fully scalar** (`conv2d_kernel.cpp:222-231`: nine explicit scalar `int32×int8`
  MACs, no `aie::vector`/`aie::mac`/MMUL). The AIE-ML int8 vector datapath is entirely
  untapped. RGB takes 9→27 MACs/pixel, and since the per-pixel body is load/multiply-bound the
  cycle cost is **~2.1×, not 3×**.

  **conv2d's cost IS measured — ~11.5 ms/frame, not the "~2.6-4 ms" this entry claimed until
  2026-08-14.** It was never a mystery: `aiecompiler` schedules the loops and logs the result.
  From `build/hw_emu/128x128/ch1/aiecompiler.log` (the real-conv build, `CONV2D_ECHO_TEST=0` —
  the ch16 log is echo mode and its conv2d figures are meaningless):
  ```
  conv2d_kernel.cpp:215  MAC loop,    1 output px/iter  : 37-45 cycles
  conv2d_kernel.cpp:195  stream read, 4 px/iter         : 28-31 cycles
  ```
  At 128×128 × 16 channels, 1 GHz core clock:
  ```
  MAC loop   16384 px x 37 cyc = 606k cyc/channel
  read loop  128 rows x 32 it x 28 cyc = 115k cyc/channel
  total      ~721k cyc/channel  ->  ~0.72 ms  ->  x16 = ~11.5 ms/frame   (13.6 at 45 cyc)
  ```
  Cross-check: conv2d fires 1024×/frame producing `CONV_OUT_CHUNK`=256 px each, so
  11.5 ms / 1024 / 256 = 45.7 cyc/px — consistent.

  **The old estimate assumed ~15 ops ≈ 15 cycles/pixel and was 3-4× optimistic.** Nine
  `int32×int8` scalar multiplies, nine unaligned byte loads from three pointers, two Hann
  multiplies and four clamp selects do not have the ILP to issue in 15 cycles on the scalar
  unit. **The 2.x ratio for RGB survives the correction; only the absolute moved.** So conv2d
  is already **~35% of a 33 ms budget** and the single largest AIE cost in the design — not
  the 8-12% the old figure implied.

  RGB scalar: ~37 → ~77 cyc/px plus a 3× longer read loop ⇒ **~25.6 ms/frame, +14.1 ms**,
  i.e. 45% of a 33 ms budget added on top of a stage that is already the biggest one.

  **Two escapes, both cheap with 298 cores free**: vectorize the 3×3×3 MAC, or split conv2d
  across 3 cores summing per input plane. The AIE-ML int8 datapath does **256 MAC/cycle**, so
  even at a fraction of theoretical throughput a vectorized 3×3 lands near ~0.3 ms/frame
  grayscale and ~0.9 ms RGB.

  **DONE 2026-08-14, and the estimate above was optimistic by ~14×.** Both kernels are now
  vectorized: conv2d 11.5 → **4.1 ms** (not 0.3) and cmul 7.9 → **0.13 ms**. The conv2d miss is
  instructive — the MAC loop did drop 37 → 8.75 cyc/px, but **the stream-read loop was left
  untouched and is now 44% of the kernel**, so the whole-kernel speedup is 2.8× rather than the
  ~40× the MAC loop alone would suggest. Amdahl, in a file that had already measured both
  loops. RGB's remaining cost is therefore ~4.9 ms, not ~0.6.

  *Caveat on provenance: these are the compiler's scheduled cycle counts, which on an in-order
  VLIW core are real cycles absent memory stalls — trustworthy for sizing, but not a profile.
  The other per-stage figures in the FPS table below are estimates, not measurements.*
- **Interleaved layout is forced.** conv2d keeps a 3-row sliding window
  (`conv2d_kernel.cpp:188`). Pixel-interleaved R,G,B feeds it unchanged (buffer 3×130 B →
  3×386 B, trivial against 64 KB). Planar would need two whole planes resident (32 KB) to
  produce one output pixel. Interleaved is also what makes the per-row 384 B = 24 beats work.
- **`roi_crop` scales fine but Pass 1's II degrades.** Patch cache 16 KB → 48 KB (~27 of 1200
  BRAM18); DSP 44 → ~132 of 1312. Both nothing. But Pass 1 is `II=4` because bilinear needs 4
  scattered reads through a *single* `m_axi` port (`roi_crop.cpp:125-129`); RGB makes that 12
  reads/pixel. ~630 µs/frame, still <2% of budget — this is where that comment's "wider/split
  AXI port" finally earns its keep. **Treat II 4→12 as a floor, not a ceiling**:
  `ROI_MAX_PATCH_ELEMS` triples (16384 → 49152) and HLS may re-bank the BRAM array rather than
  scaling it linearly. Cheap to check before committing — `make kernels` is minutes.
- **Stage A's normalization must become JOINT across the three planes — correctness trap.**
  Zero-mean and unit-L2 are global reductions. Normalizing each plane independently equalizes
  them and **destroys the chromatic information RGB was for**: a red patch and a blue patch
  would normalize to the same tensor. One joint mean and one joint L2, log applied per-plane
  before the reduction. Silent and self-defeating if done wrong.
- **Weight-buffer layout collision — it hits EVERY field, not just `mean_prev`.** 27 weights +
  shift + bias + dequant + mean_prev = 40 B, so it still fits the 64 B `CONV_WEIGHT_BYTES_PAD`.
  But RGB weights occupy `[0:27]`, which runs over `out_shift`[9], `bias_acc`[10:14],
  `dequant_scale`[14:18] **and** `mean_prev`[18:22] (`conv2d_kernel.h:18`). All four move; this
  entry used to name only `mean_prev`. The layout is duplicated in four files (see the
  preprocessing-coupling note in Known Issues), so it is a coordinated edit across all of them.

What does **not** change: `N_CHANNELS` is conv2d's *output* count and stays 16, so the entire
FFT → cmul → IFFT loop, the shift budget, `H_SHIFT`, the 1 MB filter state, and the **4258**
host DMA transactions/frame (measured 2026-08-13; the "~6500" this line used to carry was the
pre-measurement estimate) are all untouched. Frame buffer 2.07 → 6.22 MB of 12 GB. PLIO
traffic 256 → 768 KB/frame against ~41 MB/frame deliverable at 32-bit × 312.5 MHz.

**Frame-rate impact — UPDATED 2026-08-14, post-vectorization.** Both scalar kernels are now
vectorized and the DMA count is 4× lower, so the numbers below are the *current* baseline, not
the pre-vectorization one. Treated as serial, because the GMIO orchestration blocks the APU.

| stage | gray (now) | RGB | source |
|---|---|---|---|
| conv2d | **4.1** | ~9.0 | compiler schedule × source trip counts, vectorized |
| cmul_accum | **0.13** | 0.13 | compiler schedule × source trip counts, vectorized |
| FFT + IFFT chain | ~2.2 | ~2.2 | compiler schedule, trip counts inferred |
| APU transposes | ~1 | ~1 | estimate |
| filter update | ~3 | ~3 | estimate |
| roi_crop (PL) | ~0.7 | ~1.3 | partly measured (`ap_done` 766 µs, ch1) |
| DMA orchestration | **2.2-10.9** | same | **1090 tx/frame**, µs/tx still needs `TARGET=hw` |

| DMA regime | now (vectorized gray) | vectorized RGB |
|---|---|---|
| 2 µs/tx (2.2 ms) | ~13.3 ms → **75 FPS** | ~18.8 ms → **53 FPS** |
| 10 µs/tx (10.9 ms) | ~22.0 ms → **45 FPS** | ~27.5 ms → **36 FPS** |

**The design now clears 30 fps in both DMA regimes, with or without RGB.** That is a different
situation from the pre-vectorization estimate, where the pessimistic regime gave 14 FPS and RGB
made it 12. Two changes did it: conv2d + cmul 19.4 → 4.2 ms, and the WS=8 chunking cutting the
DMA count 4×, which shrinks the *uncertainty* as well as the cost.

RGB now costs ~5 FPS in the pessimistic regime and ~22 in the optimistic one — still real, but
no longer the difference between shipping and not. The remaining conv2d cost is dominated by
the **stream-read loop** (44% of the kernel), untouched; see the conv2d entry in Completed.

**Sequencing — three constraints, in order of how much they cost to get wrong:**

1. **RGB collides with Phase 1's calibration.** Phase 1 re-derives `bias_acc` against
   `ROI_NORM_Q`. RGB forces Stage A's normalization to become JOINT across three planes, which
   changes the per-plane σ and therefore the effective input scale — so `bias_acc` needs
   deriving a second time and the shift budget re-sweeping a second time. Doing RGB first
   throws away Phase 1's calibration; doing it after means redoing it. **Consider folding the
   joint-normalization change into Phase 1 rather than deferring it to Phase 5.**
2. **The accuracy premise is weaker than this section used to imply.** Danelljan Fig. 3's
   "conv layer-1 AUC 52.1 vs 37.0 for raw intensity" is *CNN features vs raw intensity* — this
   design already has CNN features, so that gain is already banked. The RGB-vs-grayscale-INPUT
   delta is a different, unquantified number. The nearest proxy in the same figure is CN
   (colour names) at 49.7, which is **below** conv layer-1's 52.1. The cost of RGB is now
   measured; the benefit is not established by the cited figure.
3. RGB does nothing about a tracker that trains through occlusions, and nothing about the DMA
   count. Do PSR gating and the DMA-cost measurement first — the latter could force an
   architectural change that would invalidate front-end work done before it.

## Weight export (MobileNetV3-Small layer 1)

Export and quantize via `make weights`:
1. Extracts first conv layer of torchvision.models.mobilenet_v3_small (pretrained)
2. Folds BatchNorm into weights/bias
3. Collapses RGB → grayscale using ITU-R BT.601 luminance: 0.2989×R + 0.5870×G + 0.1140×B
   — **deliberately NOT Danelljan's unweighted sum; do not "fix" it.** See the Known
   Issues entry and reproduce with `uv run --extra weights python3 scripts/check_collapse.py`
4. Symmetric INT8 quantization per output channel
5. Outputs:
   - `design/aie_src/weights/layer0_weights.bin` — 16 × 64 B (16 channels)
   - `design/aie_src/weights/layer0.h` — shift/scale metadata
   - `design/aie_src/hanning_128.h` — precomputed Q1.15 Hanning window

`scripts/check_collapse.py` is the front-end diagnostic — **four checks, no hardware, seconds**,
where the alternative for every one of them is an aiesim or hw_emu run. Re-run it after ANY
change to `export_weights.py`, `ROI_NORM_Q`, or the collapse.

| | Check | Needs |
|---|---|---|
| Q1 | LUM vs SUM collapse convention | torch |
| Q2 | linear diversity of the exported kernels (collinear = wasted pipeline passes) | `.bin` |
| Q3 | **`bias_acc`/`out_shift` sanity — input-independent**: structurally dead channels, whether ReLU can ever fire, bits of signal resolution actually used | `.bin` |
| Q4 | post-ReLU feature maps through conv2d's exact integer datapath on a real Stage-A patch: DC/AC ratio, post-nonlinearity redundancy | `.bin` + an s6/s7 scenario |

Q3 is the one to trust — its verdicts follow from `bias_acc` vs `127·Σ|w|` and hold for any
input at any patch size. Q4 is patch-specific and its rank figure is partly a property of the
patch. Run with `--skip-torch` to get Q2-Q4 without a torch install.

## Build commands

```bash
make weights                       # export MobileNet-v3 Small layer 1 (INT8 weights + hanning table)
make gen_vectors                   # generate aiesim test vectors
make graph                         # compile AIE graph only
make test_host                     # native unit test for the filter math (seconds, no hardware)
make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0   # BIT-EXACT kernel diff (seconds)
make x86sim_check KUT=cmul   SCENARIO=s7                 # ...same for cmul_accum
make x86sim_check KUT=cmul   SCENARIO=cmul_stress        # ...exercising sat16's rails
make aiesim                        # run AIE simulator — NOTE: bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but deletes fft_col_in.bin to force the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine one)
make kernels                       # compile camera_capture + roi_crop PL kernels
make xsa                           # link kernels + graph → XSA file
make application                   # cross-compile host ELF (aarch64)
make sd_card                       # full build: kernels → graph → xsa → application → package
make sd_card CONV2D_MODE=0         # ...with the REAL conv2d. THE DEFAULT IS ECHO (see below)
make sd_card TARGET=hw             # hardware build
make run_emu LAUNCH_HW_EMU_EXEC=1  # launch hw emulation
make cleanall
```

**`CONV2D_MODE ?= 1` — the default is ECHO, not the real convolution.** In echo mode
`conv2d_kernel.cpp` returns at the top of the function and passes the int8 patch straight
through as cint16: no 3×3 MAC, no ReLU, no Stage B1, no Hanning window, and the weight buffer
is never read. Nothing in the build output says so. This cost the 2026-08-13 ch16 baseline —
~28 h of emulation whose numbers had to be requalified. **Check before every expensive run:**

```bash
grep -o 'CONV2D_ECHO_TEST=[0-9]' build/$TARGET/${PATCH_ROWS}x${PATCH_COLS}/ch$N_CHANNELS/aiecompiler.log
```

## Directory layout

```
design/
├── aie_src/
│   ├── fft_graph.h            # FFT2D_graph (single instance, GMIO row→col)
│   ├── ifft_graph.h           # IFFT2D_graph (single instance, same pattern)
│   ├── conv2d_kernel.h/.cpp   # MobileNet-v3 layer 1: 3×3 MAC + ReLU + Hanning window
│   ├── cmul_accum_kernel.h/.cpp # col-FFT ⊙ H_ch* + accumulate (cint16 accum, saturating)
│   ├── mosse_graph.h          # Top-level: PLIO + 10 GMIO + 2 custom kernels + FFT2D + IFFT2D
│   ├── mosse_graph.cpp        # Instantiation + aiesim smoke test main()
│   ├── constraints.aiecst     # PatchIn PLIO shim placement
│   ├── hanning_128.h          # Precomputed Q1.15 Hanning window (auto-generated)
│   ├── weights/               # MobileNet-v3 Small layer 1 (INT8, auto-generated)
│   │   ├── layer0_weights.bin # 16 channels × 64 B
│   │   ├── layer0.h           # Shift/scale metadata per channel
│   │   └── layer0_meta.npz    # Validation data
│   └── aiesim_data/
│       └── s*/                # Test scenarios, written by `make gen_vectors`:
│                              #   s0-s4  raw patches — CONV2D_MODE=1 (echo) only
│                              #   s6     Stage-A preprocessed, H=unity — real conv path
│                              #   s7     s6 + a REAL MOSSE filter (per-bin complex H,
│                              #          off-centre target). The only scenario that
│                              #          exercises H_SHIFT with a non-identity filter,
│                              #          asserts a PSR, and checks the F_ch tap.
├── pl_src/
│   ├── camera_capture/        # Zero-fill frame buffer stub
│   └── roi_crop/              # Extract patch, stream to PatchIn PLIO
├── host_app_src/
│   ├── mosse_tracker.cpp      # GMIO-driven XRT tracking loop
│   ├── mosse_filter.h/.cpp    # MOSSE init/update/Q1.15 export — NO XRT include,
│   │                          # so `make test_host` builds it with the system g++
│   └── test/
│       ├── test_mosse_filter.cpp  # native unit test
│       └── golden/                # NumPy reference (scripts/gen_filter_golden.py)
├── system_configs/
│   └── mosse_x1.cfg           # v++ linker: camera_capture + roi_crop + PatchIn
├── profiling_configs/         # xrt.ini (trace settings)
├── directives/                # post_sys_link.tcl (AIE clock = 1 GHz)
└── exec_scripts/
    └── run_script.sh          # Board execution (mosse_tracker.elf a.xclbin)
```

## Current status (as of 2026-08-14)

### Where things stand — read this first

**The tracker works on the real conv path at ch1, with a correct σ=2 correlation surface.**
Run D (hw_emu, 128×128, ch1, ITER_CNT=2, 5-2-2): `err=0 px`, peak/max-sidelobe 23.0,
Bolme PSR 172, nothing railed at any stage, and the response profile matches
`exp(-d²/8)` to within 4% at both d=2 and d=6.

| | before 2026-08-14 | now |
|---|---|---|
| AIE compute | 21.6 ms/frame | **6.4 ms** (conv2d 11.5→4.1, cmul 7.9→0.13) |
| DMA transactions | 4258/frame | **1090** |
| peak/max-sidelobe (ch1) | 5.11 | **23.04** |
| response vs ideal σ=2 at d=6 | 0.154 | **0.011** (ideal 0.0111) |
| frame time (hw_emu ch1) | ~90 min | **~26 min** (but see below) |

**FRAME-TIME CORRECTION 2026-08-15: the ~26 min figure does not hold on the current machine.**
The PSR-gating regression run (same 128×128/ch1/5-2-2 build as run D, XSA reused unchanged)
measured **~50 min per frame**, ~2× the number recorded above. The ~26 min came from run B's
wall clock; nothing about the design changed between them, so treat this as machine/load
variance rather than a regression. **Budget ~50 min/frame at ch1**, i.e. ~1.7 h for
`ITER_CNT=2` and ~2.5 h for `ITER_CNT=3`.

**SECOND FRAME-TIME CORRECTION 2026-08-16 — AND THIS ONE IS AN ORDER OF MAGNITUDE.
EVERY FIGURE ABOVE IS ch1. DO NOT APPLY THEM TO ch16.**

Measured on a ch16 run (128×128, `CONV2D_MODE=0`, `ITER_CNT=3`, 5-2-2), started 11:35:41 UTC
and killed by hand 172 minutes later having completed ch0-ch3 and started ch4 —
`runs/occlusion_0816_1321.log`:

```
                        ch1 (measured)   ch16 (measured)
per channel                    ~50 min          ~43 min
per FRAME                      ~50 min        ~11.5 h     (16 channels)
ITER_CNT=2                      ~1.7 h          ~23 h
ITER_CNT=3                      ~2.5 h          ~34 h
```

The per-channel cost is almost the same at both; the frame cost is not, because the
per-channel loop runs 16×. **The "~1.7 h for ITER_CNT=2" this file carried for the ch16 run
was a ch1 number applied to a ch16 plan — wrong by ~13×.**

**The reasoning that produced it is also wrong, and that matters more than the number.**
This file attributed the speedup to "3.5× faster kernels and 4× fewer DMA transactions". But
**hw_emu wall-clock does not track AIE compute**: the 2026-08-13 echo-mode ch16 run, in which
`conv2d` returned at the top of the function and did *no MAC work whatsoever*, still took
~14 h/frame. The emulator is simulating the PL and the DMA/NoC traffic, and that cost is
almost independent of what the AIE cores do. Measured 11.5 h now against 14 h then is a
**1.2×** improvement — consistent with the DMA count falling 4×, and flatly inconsistent with
kernel vectorization mattering at all to emulation.

So: **vectorizing a kernel speeds up the DESIGN, not the emulation of it.** Size hw_emu runs
from measured wall clock at the same `N_CHANNELS`, never by scaling a ch1 measurement and
never by reasoning from AIE cycle counts.

**Four findings drove that, and all four were invisible to `err=0 px`** — which passed in
every single run, before and after:
1. The 2026-08-13 ch16 baseline ran conv2d in **echo mode**; several conclusions built on it
   were wrong (the "99.69% of the rail" headroom crisis, the σ=2 response claim, B2's "33% of
   the peak").
2. **ReLU costs ~25% of the peak/sidelobe ratio**, and the planned `bias_acc` fix would have
   made tracking 3× *worse* by switching ReLU back on.
3. **`mean_prev = 0` on frame 0** left Stage B1 inert on the frame the filter trains from —
   fatal at ch16 (flat railed response, ratio 1.00), and the cause of a forward-FFT railing
   that sent two separate shift-budget investigations chasing a phantom.
4. **Both custom kernels were scalar**, worth ~19 ms/frame; cmul's arithmetic loop was not
   even pipelined, behind a stale assembler-OOM workaround.

**Remaining expensive item:** the ch16 run — **~23 h for `ITER_CNT=2`, measured, not the
~1.7 h this file used to claim** — at budget **4-2-1**, not 5-3-4. Run ch1 (~1.7 h) first;
it exercises all the new host code and 5-3-4 no longer suits the geometry. See the ch16
entry under In Progress. **Largest functional gap:** PSR
gating (Bolme §3.5) — **DONE 2026-08-15**: 34 native assertions, a bit-exact hardware
regression vs run D, and an occlusion run in which the gate fired (`LOW_PSR`, PSR 3.90),
held the position, froze the filter provably bit-exactly, and reacquired the target on the
next frame with `err=0 px`.

### Completed
- [x] **TEST-SEQUENCE GENERATION, FOR RUNS OF HUNDREDS OF FRAMES (2026-08-16) — host-only.**
      Three things about the shipped test data do not survive contact with real hardware,
      where a frame costs ~20 ms instead of ~11.5 h and a run can afford hundreds of frames
      rather than two. All defaults reproduce the previous behaviour exactly, so existing
      hw_emu comparisons stay valid.

      **1. The background was regenerated every frame — ~0.6-1.2 s on the A72.**
      `fill_background` runs six sinusoids per pixel over 1080×1920: **12.4 M `sin()` calls per
      frame**, measured at 193 ms on an x86 host. Free when a frame took 11.5 h; on hardware it
      is **30-90× the entire pipeline**, so the FPS number — one of the two things a board run
      exists to produce — would have been a measurement of the test harness. The background is
      static by construction (fixed LCG seed, no frame dependence), so it is now generated once
      and only the dirty rectangle is restored. No flag: the old behaviour was simply wrong.
      The occluder dirties the whole frame, which is why the rect has to be able to grow to
      full size rather than being assumed target-sized.

      **2. The target walked off the frame at about frame 48.** The legacy scheme injects at
      `pos + (IMPULSE_DR, IMPULSE_DC)` and `pos` then moves by the detected displacement, i.e.
      constant velocity (10,−7)/frame from row 540 on a 1080-row frame. `TRAJECTORY=1` puts it
      on a closed ellipse instead — validated over 2000 frames: row 360..720, col 600..960, ROI
      inside the frame on every frame, peak step **9.42 px/frame** against the 12.2 the
      pipeline is already proven at, and it starts exactly at the initial centre so frame 0
      still trains on a centred target.

      **AND IT CHANGES SOMETHING MORE IMPORTANT THAN RUN LENGTH.** Under the legacy scheme the
      target is planted at the tracker's OWN ESTIMATE plus a constant — the ground truth
      follows the tracker wherever it goes, so it cannot drift and `err=0 px` is close to
      self-fulfilling. On an absolute scripted path the truth is independent of the estimate,
      so drift is real and measurable. The pass/fail expectation is now derived from where the
      target was actually DRAWN rather than from a constant; at `TRAJECTORY=0` that reduces to
      the old constant exactly.

      **3. The target never changed size, so the scale filter had nothing to track.**
      `inject_target_frame` drew at the fixed `TARGET_H/W` macros, so a hardware run could show
      the DSST filter does not crash and nothing more. `SCALE_TRAJ=1` adds a sinusoidal
      envelope, 0.70×..1.30× over 200 frames — **0.99%/frame peak against the filter's
      2%/frame step size**, which is the rate limit that actually matters (`SCALE_ETA=0.025`
      makes the model adapt slowly, so the envelope must move far slower than the single-frame
      range).

      **Also fixed:** `OCCLUDE_MASK` is a 32-bit mask indexed by frame number, so it cannot
      express anything past frame 31 and `mask >> frame` is undefined at frame ≥ 32 — now
      guarded, with `OCCLUDE_PERIOD`/`OCCLUDE_LEN`/`OCCLUDE_START` as the periodic alternative.

      **`OCCLUDE_START` is a warm-up and it matters more than it looks.** Without it the first
      occlusion lands on frame 1 — the filter occluded immediately after being initialised
      from a single patch, which tests the gate against a filter that has not converged and
      conflates two failures if it goes wrong. Default 30, i.e. ~4 time constants at
      `MOSSE_ETA=0.125`. **The scale filter is 5× slower** (`SCALE_ETA=0.025` ⇒ ~40 frames per
      constant, ~120 to settle), so a run intended to occlude a converged SIZE estimate rather
      than a converged position wants `OCCLUDE_START=120`. The startup line prints the warm-up
      in time constants for both filters so the choice is visible in the log.

      **And the run now produces a CURVE, not a verdict:** per-frame IoU and centre error
      against the drawn box, with a run summary reporting overlap precision at PASCAL's 0.5
      threshold, mean/worst IoU and mean/worst centre error. That is the OTB-style metric both
      papers report and which was uncomputable before the target box existed — and it is
      uncomputable in hw_emu for a different reason: two frames is not a curve.

      New host-only make variables: `TRAJECTORY`, `TRAJ_AMP_R/C`, `TRAJ_PERIOD`, `SCALE_TRAJ`,
      `SCALE_TRAJ_AMP/PERIOD`, `OCCLUDE_PERIOD`, `OCCLUDE_LEN`.

- [x] **DSST 1-D SCALE FILTER (2026-08-16) — host only, natively tested, and it CONVERGES.**
      `docs/1609.06141v1.pdf` §5.1. A separate 1-D correlation filter over scale rather than
      the ICCV'15 paper's exhaustive multi-resolution search, for a reason that is stronger
      here than in the paper: DSST Table 1 already beats exhaustive on **both** axes (OP 67.7
      vs 65.2, 25.4 vs 16.9 FPS), and on this hardware an exhaustive search would push patches
      resampled by ±30% through `roi_crop → conv2d → FFT` every frame — moving `|F|` and
      therefore the shift budget, the coupling that has forced two budget hunts — while
      spending exactly the 30 fps headroom that vectorizing conv2d and cmul bought back.

      **The filter is the existing filter at `rows = 1`.** DSST §3 states it in prose ("the
      same approach can be used to learn 1-dimensional scale estimation filters… accomplished
      by only adapting the feature extraction step"); it holds in this code because
      `filter_init`/`filter_update`/`FilterState::resize` read geometry from the state, and
      `gaussian_target_spectrum(G, 1, S, σ, 0, 0)` degenerates cleanly since
      `signed_freq(0,1) == 0`. So the reused surface is large and the new code is the feature
      extraction, a DFT and the detect loop. Same conjugation convention as the translation
      path — one convention in the design, because the header already records how silent a
      mix-up is.

      **Defaults from §6.1: S=33, a=1.02, η=0.025 (NOT the translation filter's 0.125),
      σ_s = S/16, template capped at 512 px.** `SCALE_N=1` disables it completely and
      reproduces the pre-scale behaviour — the `CONV_VECTORIZE=0` bisection lever, asserted.

      **19 analytic assertions, no new goldens** (the 1-D update is the same code already
      golden-tested at 2-D, so a golden would re-test tested code). The two that earn their
      keep:
      ```
      under-sized box -> level +3 (factor 1.061)   over-sized -> level -4 (0.924)
      convergence: 51.2 -> 63.66 against a truth of 64, error 12.80 -> 0.34 px, MONOTONE
      ```
      A single application UNDER-corrects — +3 where +5 would be exact — because the scale
      response is a correlation peak on a discrete grid smoothed by a σ=S/16 target. That is
      not a bug and the convergence test is what shows it: DSST iterates across frames, so the
      property to assert is that repeated application walks toward the truth and never away,
      which it does, to 0.5%.

      **MEASURED COST: 1.54 ms/frame on x86 (d=484, S=33, detect + update).** Not free, and
      that matches the paper — DSST Table 1 has the scale filter costing *more* than the
      translation filter (17.5 → 39.4 ms/frame). Expect ~3-5 ms on the A72. Still far below a
      3-scale exhaustive search's ~16 ms of extra AIE + DMA, so the choice stands, but do not
      describe it as free.

      **fDSST's PCA COMPRESSION IS NOT THE RIGHT NEXT OPTIMISATION HERE — MEASURED
      2026-08-16, and this entry said the opposite an hour earlier.** The retracted claim was
      "~28× fewer DFTs, i.e. ~0.06 ms". The DFT saving is real and even larger than that
      (**45.9× measured**), but it counted only what compression saves and not what it costs.
      Measured per frame, d=484, S=33, on the same machine:
      ```
      today: complex DFT, d transforms per sample                1.136 ms   1.00x
      (b) real-input DFT + Hermitian symmetry, NO PCA            0.365 ms   3.11x
      (c) full fDSST PCA: 2 QR + 1 projection + 3 compressed DFT 0.713 ms   1.59x
            of which  MGS/QR of one 484x33 matrix   0.231 ms  (x2)
                      Q^T z projection              0.214 ms
                      the compressed DFTs themselves 0.012 ms
      ```
      **PCA is beaten roughly 2:1 by simply not doing complex arithmetic on real data**, and
      the reason is structural rather than a tuning accident: **the QR is O(d·S²) — the same
      order as the DFT it eliminates.** Compression converts a complex O(dS²) transform into a
      real O(dS²) factorisation, which is a constant-factor win (real MACs are ~3× cheaper),
      not an asymptotic one. The compressed DFTs it enables cost 0.012 ms; the machinery to
      enable them costs 0.68 ms.

      The paper wins where we do not because its parameters differ: d≈1000 with S=17 gives a
      59× compression against our 14.7×, its QR is ~1.8× cheaper (dS² = 289k vs our 527k), it
      *also* halves S (33→17, interpolating the output back), and most of fDSST's headline 2×
      comes from the TRANSLATION filter's 32→18 reduction on a 2-D grid, not from the scale
      filter at all.

      **Do the real-input DFT instead**: the features are real by construction
      (`scale_extract` writes `cfloat(v, 0)`), so half the multiplies are against a zero
      imaginary part, and Hermitian symmetry makes only S/2+1 of the S outputs independent.
      3.11× for no algorithmic change and no new failure mode, and it is checkable against the
      current path to float precision.

      After that the real levers are **d and S, not PCA**: the template is 22×22=484 and the
      rank argument says at most S=33 dimensions carry information anyway, so a 16×16 template
      halves the cost outright; and S=33→17 with interpolation (what fDSST actually does for
      the scale filter) halves both the extraction and the transform.

      **Perspective: this is optimising a non-bottleneck.** 1.39 ms sits against a 13-22 ms
      frame; conv2d (4.1 ms) and the DMA (2.2-10.9 ms) are both larger. Do not spend
      correctness risk here before the ch16 run.

      **One cost trap found and fixed while writing the tests:** the obvious direct DFT
      evaluates `sin`/`cos` in the inner loop, i.e. n² transcendental calls per transform.
      With d≈484 transforms per sample and two samples per frame that is **~2M sin/cos per
      frame** — tens of milliseconds, which would have made the scale filter the most
      expensive thing in a design whose entire AIE budget is 6.4 ms. The twiddle table is
      built once per frame instead.

- [x] **BOUNDING-BOX STATE, ROI PADDING AND σ ANCHORING (2026-08-16) — host only, no AIE
      recompile, no XSA relink, no PL re-synthesis.** The tracker's state was `pos_row`/
      `pos_col`, two ints. It is now a `TargetBox` (centre + size, frame px) with
      `roi = box × TARGET_PADDING`, so the filter finally sees background context — the thing
      both papers use a larger-than-object window *for* (Bolme §3.1, Danelljan §3.1).
      `roi_crop` takes all geometry as runtime AXI-Lite scalars, which is why this costs no
      rebuild of anything but the host ELF.

      **THE TWO CONVERSIONS THAT WERE INVISIBLE — the real content of this change.** The peak
      is located in PATCH bins; the position lives in FRAME pixels. While `roi_h ==
      patch_rows` those are the same number, so `pos_row += dr` (`:1480`) and
      `dr == IMPULSE_DR` (`:1482`) were both accidentally correct. Padding breaks both by the
      resample ratio, and the symptom is a tracker that localises confidently and **drifts** —
      invisible to `err=0 px`, which is the failure mode this file already records four times.
      Now `patch_dr_to_frame` / `frame_dr_to_patch` in `mosse_filter.{h,cpp}`, i.e. under
      `make test_host`, with 33 analytic assertions including a `frame→patch→frame` round-trip
      over four paddings (worst error 0.5 frame px = exactly half a bin, the real quantisation)
      and an anisotropic box where a single shared ratio would pass every square-target test
      and fail.

      **`TARGET_H/W=64` with `TARGET_PADDING=2` gives `roi = 128` — EXACTLY today's geometry.**
      So the resample stays 1:1, the interpolator stays dormant, both conversions are
      identities, and the expected displacement is still literally `(IMPULSE_DR, IMPULSE_DC)`.
      Adopting the box is therefore a single-variable change and any later padding move is a
      separate one.

      **The test harness had to be rebuilt, and that is not incidental.** `inject_target_frame`
      drew an ~11×11 object inside a 128×128 ROI — an effective padding of **~11.6**, not 1 —
      so a σ anchored to a declared 64 px box would have been anchored to a fiction. The shape
      is now scaled by `TARGET_H/W` (reproducing the original exactly at 11×11, asymmetries
      intact) and the background is a **band-limited texture** rather than a flat fill.
      That second part is load-bearing: padding exists so the filter can learn
      target-vs-background, so against `memset(BACKGROUND=40)` more padding is strictly less
      target and *any* padding comparison is decided before it runs.

      **IoU reporting lands with it** (`box_iou`, PASCAL 0.5 flagged) — the metric both papers
      report and which was previously impossible to compute, since there was no box.

      **σ stays at 2.0 and the DSST rule is a build flag, not the default** — see the padding
      sweep entry below for why the evidence does not support switching. `SIGMA_FROM_TARGET=1`
      gives σ=4 at padding 2 and is asserted in `test_host` under both settings.

      New host-only make variables, all in `GCC_FLAGS` and deliberately **not** `AIE_FLAGS`
      (the AIE never sees the ROI): `TARGET_H`, `TARGET_W`, `TARGET_PADDING`, `MOSSE_SIGMA`,
      `SIGMA_FACTOR`, `SIGMA_FROM_TARGET`, `MOSSE_ETA`, `FRAME_TEXTURE`. `MOSSE_SIGMA` and
      `MOSSE_ETA` also give `DEFAULT_SIGMA`/`DEFAULT_ETA` the `#ifndef` escape they lacked —
      they were hard `constexpr`s, and a calibration constant that can only be changed by
      editing a header does not get swept.

- [~] **ROI PADDING SWEEP (2026-08-16) — padding SETTLED at ≥2; σ NOT settled, and the
      reason is that PSR cannot settle it.** `phase1_sweep.py --roi-model synth` now
      synthesizes a frame, crops an ROI of `target × padding`, and runs the bit-exact Stage A
      (`roi_crop_ref.py`), so padding finally has somewhere to act — the old `--roi-model file`
      path starts from a patch already Stage-A'd at 1:1 and has no notion of an ROI at all.
      Held out properly: `--frame-shift` moves the target in FRAME pixels and re-crops, so the
      evaluation patch differs by resample phase, border content and its own mean/σ.
      (`--holdout`'s `np.roll` is a circular shift of the Stage-A *output*, which cannot
      physically occur; the combination is now rejected rather than silently weaker.)

      **Padding, at target 64, budget 4-2-2, ch16, held out (10,−7) frame px:**
      ```
      padding  best PSR(B)   best ratio   locerr   notes
        1.5       18.4          2.59       0.75px  upsamples 1.33x; WORST on both metrics
        2.0       45.7          4.97       0.00px  roi 128 -> resample is 1:1
        2.5       51.5          5.68       0.00px  aliasing: 128 of 160 src rows read
        3.0       49.8          5.90       0.00px  aliasing + 3.57% of samples clipped
      ```
      **Padding 1.5 is clearly bad and ≥2 is clearly right** — consistent with DSST's 2 and
      fDSST's 3. 2.5-3.0 edge out 2.0 on both metrics but trigger the aliasing detector
      (bilinear has no prefilter, so beyond ~2× decimation source rows are skipped outright)
      and 3.0 clips 3.57% of samples. **Recommend padding 2.0**: it is the only value where
      the resample stays 1:1 at target 64, which makes it the clean single-variable step.

      **σ IS A DIFFERENT STORY, AND THE PLANNED CHANGE IS NOT SUPPORTED.** The plan was to
      move σ 2 → 4 per DSST §6.1's "σ = target size / 16" (at padding 2 the target is 64 patch
      px, so σ=4). Measured, at padding 2:
      ```
      sigma    0.75    1.00    1.50    2.00    2.50    3.20    4.00    5.33
      PSR(B)   80.3    71.8    55.2    45.7    41.2    37.5    30.5    19.3
      ratio     2.58    3.78    4.97    4.47    4.23    4.09    3.50    1.85
      ```
      **Bolme PSR is MONOTONE DECREASING in σ, all the way down to sub-pixel.** So PSR does
      not select σ — it just rewards a sharper peak, and a delta target would maximise it
      trivially. σ=4 scores worse than σ=2 on both metrics, but that is not evidence σ=4 is
      wrong; it is evidence the metric cannot arbitrate. What σ actually buys is robustness to
      appearance change, and a single-frame translation-only holdout does not exercise that.

      **A tempting mechanistic explanation was tested and REFUTED.** `b2_removed` rises 433 →
      3018 from σ=2 to σ=4, so Stage B2 — which nulls the 9 lowest bins, exactly where a wide
      Gaussian keeps its energy — looked like the cause. Re-run with `--no-b2` (new flag,
      models `B2_NULL_BINS=0`): PSR goes 45.7 → 46.5 at σ=2 and 30.5 → 32.9 at σ=4. B2 costs
      ~2 PSR points at large σ; the 45.7 → 30.5 drop survives with B2 off. **The σ dependence
      is intrinsic to the metric, not a Stage-B2 artifact.**

      **Consequence for the plan: keep σ=2.0 as the default and make it a build parameter.**
      Switching to the paper rule on the strength of a metric that is monotone in σ would be
      exactly the "plan built on unvalidated measurement" failure this file already records
      three times. σ needs real video, or at minimum a holdout with scale/appearance change,
      to arbitrate. The DSST rule stays available as `SIGMA_FACTOR`.

      **Two further things the sweep now reports that no PSR number reveals:**
      - **`px/bin` = `roi_h/patch_rows`, the localisation quantum in FRAME pixels.** At
        padding 3 on an 85 px target one response bin is 2 frame px, so the tracker cannot
        resolve better than 2 px however good its PSR looks.
      - **The centring bias is constant at 0.5 PATCH px but grows in FRAME px** with the
        resample ratio (0.375 → 0.75 source px over padding 1.5 → 3.0). It does not cancel in
        a closed loop, because `pos_row` is updated from the peak every frame.

- [x] **RUN D — the `mean_prev` seed CONFIRMED ON HARDWARE (2026-08-14).** ch1, 128×128,
      `ITER_CNT=2`, budget 5-2-2 — identical to run C, so the seed is the only variable.
      ```
      run   config                    F_ch rails  accum  response   d=6     ratio  Bolme PSR
      B     4-3-3, no seed                 11      5264     14076   0.154    5.11     33.23
      C     5-2-2, no seed                  5      2121     19668   0.269    3.59     33.25
      D     5-2-2, mean_prev SEEDED         0        70      2534   0.011   23.04    172.41
      model predicted for D                 0        72      2840  -0.020   29.89
      ```
      **F_ch fell from 32768 (railed) to 53 — 618× — and the railing vanished.** The forward-
      FFT saturation was ENTIRELY the frame-0 DC pedestal, not the shift budget. `FFT_SHIFT=4`
      was never the problem, which is why raising it to 5 in run C only halved the rail count:
      a factor of 4 against a pedestal 618× too large.

      **The response is now a correct σ=2 Gaussian, on real conv2d features:**
      ```
      d=2  measured/peak 0.633, 0.611   ideal 0.6065
      d=6  measured/peak 0.011, 0.014   ideal 0.0111
      ```
      This is the claim the 2026-08-13 entry made from an echo-mode run; it is now true for
      the real path. Peak/max-sidelobe 3.59 → **23.04** and Bolme PSR 33 → **172**, with
      `err=0 px` throughout and nothing railed at any stage.

      **The offline model is validated against hardware**: accum 72 predicted vs 70 measured
      (3%), response 2840 vs 2534 (11%), rails 0 vs 0. The ratio came in 23% below prediction
      (29.9 vs 23.0), consistent with the model's float-FFT approximation and with F_ch now
      being small (49) so the spectrum carries few bits.

      **Note the response DROPPED 7.8× (19668 → 2534) while quality improved 6.4×.** A peak
      shrinking while the peak-to-sidelobe ratio climbs is the signature of removing a
      pedestal rather than losing signal — and it is exactly what `err=0 px` cannot see, since
      localisation was correct in all three runs.

- [x] **ROOT CAUSE OF THE BROAD PEAK: `mean_prev = 0` ON FRAME 0, so Stage B1 is INERT on
      the one frame the filter is trained from. FIXED 2026-08-14.**
      `layer0_weights.bin` has bytes [18:22] zeroed, so frame 0's conv2d emitted its full DC
      pedestal (`bias_acc >> out_shift` ≈ 24689 for ch0). `filter_init` therefore learned from
      a DC-dominated spectrum, and frame 1 — where B1 *is* active — applies a filter matched
      to features that no longer exist.

      **At ch16 this is fatal, not merely degrading.** Modelled on the actual
      `inject_target_frame` patch, budget 5-2-2:
      ```
                          d=2      d=6     accum    resp    ratio   peak
      mean_prev=0  ch1    0.796   0.214     1320   20263    3.89   (10,121)  fft rails
      mean_prev=0  ch16   1.000   1.000     4721   32767    1.00   ( 8,120)  RESPONSE RAILED
      seeded       ch1    0.600  -0.020       72    2840   29.89   (10,121)  no rails
      seeded       ch16   0.599  -0.009     2035   10675   36.81   (10,121)  no rails (5-3-4)
      ```
      `ratio 1.00` with the peak displaced to (8,120) is a flat saturated map — **the ch16 run
      would have produced garbage.** The 16 channels' DC pedestals sum coherently and swamp
      everything; at ch1 it only degrades.

      **Fix (`mosse_tracker.cpp`, one block before the first `weights_bo.sync`): seed
      `mean_prev = bias_acc >> out_shift` for every channel at startup.** Stage A delivers a
      zero-mean patch, so the post-conv mean is the bias term alone. Predicted 24689 vs a
      converged 24686 on ch0 — accurate to 0.01%.

      With the fix the response becomes textbook: d=2 **0.5994** and d=6 **−0.0087** against an
      ideal σ=2 Gaussian's 0.6065 / 0.0111.

      **How it was found, and what it says about the offline model.** The model disagreed with
      hardware by 29× on the accumulator. Ruled out in turn: Stage A (hand-checked through
      `roi_crop`'s fixed-point path — LOG_LUT normalised to fill uint16, `inv_q = 32·65536/σ`
      uncapped at σ≈1292 — and it reproduces exactly), and lagged B1 (the frame-to-frame mean
      differs by 2 LSB). Modelling `mean_prev=0` on frame 0 closed the gap: **response 20263 vs
      19668 measured, ratio 3.89 vs 3.59, and it predicts the forward-FFT railing hardware
      showed.** The model is now validated against hardware on the real test patch.

- [x] **`eps_rel` IS ALREADY OPTIMAL AT 1e-3 — the open item suggesting it be RAISED is
      refuted (2026-08-14).** The response has a closed form: substituting
      `H = conj(G)F/(B+ε)` into `Σ F ⊙ conj(H)` gives **`R = G · B/(B+ε)`** — the target
      Gaussian times a Wiener gain, bin by bin. Sweeping it through the full integer pipeline:
      ```
      eps_rel   1e-5    1e-4    1e-3    1e-2    1e-1
      d=6      0.0037  0.0031  0.0155  0.0531  0.1625
      ratio    12.90   13.96   16.15   10.62    4.01
      ```
      The optimum is the current default, because two mechanisms pull opposite ways: lowering
      ε sharpens the peak analytically but collapses its amplitude (775 → 271) so quantization
      noise wins, while raising ε suppresses exactly the low-`B` high-frequency bins that make
      the peak sharp. Bolme Fig. 4's flat curve does not transfer — his ε is absolute on the
      denominator, ours is relative to `mean(B)`.
      **And ε was never the cause of the broad peak**: at 1e-3 it widens d=6 from 0.011 only to
      0.027, where hardware showed 0.269. See the `mean_prev` entry above for the real cause.

- [x] **RUN C — ch1 at `FFT_SHIFT=5` (budget 5-2-2). MY RAILING HYPOTHESIS WAS WRONG, and
      what it uncovered is more important than the budget (2026-08-14).**
      Same total budget as run B (14), so the response scale is held fixed and `FFT_SHIFT` is
      the only variable.
      ```
                        run B (4-3-3)   run C (5-2-2)
      F_ch rails             11               5
      accum             5264 (16.1%)    2121 ( 6.5%)
      response         14076 (43.0%)   19668 (60.0%)
      max sidelobe          2756            5486
      ratio                 5.11x           3.59x     <-- WORSE
      PSR (Bolme)          33.23           33.25      <-- unchanged
      displacement       (10,-7) OK      (10,-7) OK
      ```
      **Predicted: less railing → ridge collapses → ratio improves. Observed: the ratio got
      worse.** The hypothesis that the two railed `(±2,0)` bins caused the ridge is refuted.

      **What is actually happening: the ratio metric is measuring MAINLOBE WIDTH, not
      sidelobes.** The response is far broader than the σ=2 target implies:
      ```
      d(px)   ideal exp(-d²/8)   run B     run C
        2         0.6065        0.739     0.823
        6         0.0111        0.154     0.269
      ```
      Bolme's 11×11 exclusion is ±5 px, sized for a σ=2 mainlobe. This mainlobe extends well
      past ±5, so what the ratio calls "max sidelobe" is the mainlobe's own skirt at d=6.
      **Clipping was accidentally acting as a high-pass filter and SHARPENING the peak** —
      removing it preserved more low-frequency content, widened the peak, and raised the
      skirt. Run B looked better on this metric for the wrong reason.

      Bolme PSR is insensitive to it (33.23 vs 33.25) because it uses µ and σ over the whole
      excluded region, which is dominated by the many small values rather than the few large
      skirt ones. **Where the two metrics disagree like this, Bolme's is the meaningful one
      and the ratio is contaminated by mainlobe width.** Another reason to report both.

      **This is NOT a shift-budget problem — do not tune the budget against it.** The
      accumulator's peak sits at bin (0,−6), a LOW frequency: the correlation output is
      dominated by low-frequency content. That is a filter/regularization question (`eps_rel`,
      Phase 3) or a residual-DC question, not a scaling one.

      **It also retires one more echo-mode figure.** The 2026-08-13 entry celebrates a
      near-perfect σ=2 response (d=1: 0.893 vs 0.8825; d=2: 0.614 vs 0.6065). That was 16
      bit-identical passthrough channels. With real conv2d the response is 10-25× too wide at
      d=6. **The correct-σ finding was an echo-mode artifact too.**

- [x] **RUN B — hw_emu 128×128 ch1 ITER_CNT=2, the new configuration end to end
      (2026-08-14). Tracks correctly; found a REAL budget problem.**
      `CONV2D_MODE=0`, `CONV_RELU=0`, both kernels vectorized, WS=8, budget 4-3-3.
      ~26 min per frame at **ch1** — down from the ~90 min reference. *(The "consistent with
      3.5× faster kernels" gloss that used to follow is WRONG and was retracted 2026-08-16:
      hw_emu wall clock is dominated by simulating the PL and the DMA/NoC, not by AIE cycles —
      the echo-mode run did no MAC work at all and still took ~14 h/frame. See the second
      frame-time correction under Current status.)*
      ```
      Frame 1: displacement (10,-7) -> pos (550,953)  peak|re|=14076  [OK: matches injected offset]
      F_ch     max|.| = 32768   rails=11      <-- RAILED
      accum    max|.| =  5264   rails=0   (16.1%)
      response max|.| = 14076   rails=0   (43.0%)
      PSR 33.23 (Bolme, "inside normal 20-60")  |  ratio 5.11x (aiesim metric)
      DMA: 100 tx/frame   (formula predicts exactly 100; would have been 388 at WS=2)
      ```

      **What this validates:** all three chunked drain loops at `16 × 4096 B` on the real XRT
      host — the deadlock risk that justified landing WS=8 separately is cleared. roi_crop
      feeding the vectorized conv2d through the real PLIO. End-to-end localisation with real
      Stage A, `err=0 px`. And the DMA count matches the formula **exactly**, confirming the
      4258 → 1090 reduction will hold at ch16.

      **THE FORWARD COLUMN FFT RAILS AT `FFT_SHIFT=4` ON THIS TARGET — 11 bins.** Identical
      signature to the 64×64 `FFT_SHIFT=3` finding recorded in the Makefile: 11 bins, being
      the nine `{0,±1}×{0,±1}` that B2 touches plus `(±2,0)`. With `B2_NULL_BINS=1` the nine
      nulled bins' clipping is harmless — they are zeroed — but **the remaining two railed
      bins are not, and they produce a ridge**:
      ```
      resp at injected (10,-7) = 14076  |  (0,0) = 2415, (10,0) = 2447, (0,-7) = 2164
      max sidelobe 2756 = 20% of the peak
      ```
      That ridge is what holds the peak/max-sidelobe ratio to **5.11×** while Bolme PSR still
      reads a healthy 33.23. **The two statistics disagree by 6.5× here, and the ratio is the
      one telling the truth** — more evidence for reporting both.

      **Not caused by anything changed today.** `FFT_SHIFT` was 4 under the old 4-2-2 default
      too, so F_ch would have railed identically; and this is ch1, which drives `weights_ch0`,
      a channel where ReLU never fires, so `CONV_RELU=0` is a no-op here. It is a
      **pre-existing condition that was never visible**: the ch16 baseline was echo mode
      (F_ch max 90) and the ch1 runs that would have caught it were 64×64 at 3-0-6.

      **The offline model missed it, and the reason matters.** `phase1_sweep.py` predicted
      accum 162 with no rails; hardware gave 5264, **32× higher**. The arithmetic is right —
      it is bit-exact-verified — but its INPUT is the s6 scenario patch, whereas hw_emu
      injects a synthetic target on an empty frame through real `roi_crop`, which after the
      log/z-score is far hotter. **Treat the sweep's ratios and orderings as sound and its
      absolute magnitudes as patch-specific.**

- [~] **RUN A — aiesim s7, 128×128 ch1, the new configuration end to end (2026-08-14).
      Integration VALIDATED; `OVERALL: FAIL` on one stale threshold.**
      `make aiesim_plio SCENARIO=s7 PATCH_ROWS=128 PATCH_COLS=128 N_CHANNELS=1` with the new
      defaults (`CONV2D_MODE=0`, `CONV_RELU=0`, both kernels vectorized, WS=8, budget 4-3-3).
      ~16 min of a 4800 s budget; no `Error 124`.
      ```
      step 2: fft_row_out done          16 x 4096 B  [fallback -> REAL PatchIn path]
      accum   max|.| = 284   rails=0
      ifftrow max|.| = 407   rails=0
      response max|.|= 841   16318/16384 non-zero
      Peak {838,-3} @ idx 1401 (r=10,c=121)  expected 1401   err=0 px      OK
      F_ch tap correlation with golden = 0.9995 (required 0.95)            OK
      PSR 841/71 = 11.8x, required 15.00x                                  FAIL
      location OK  normalization OK  imag OK  SNR OK  fcol OK  accum0 OK
      accum_sat OK  resp_sat OK  ifftrow_sat OK
      ```
      **Everything the run existed to test passed.** `16 × 4096 B` is the WS=8 chunking live
      in the real drain path; the `[fallback]` tag confirms it took the PatchIn → conv2d →
      row-FFT route rather than the bypass; no stage railed; the F_ch tap still matches.

      **The PSR threshold does not apply to this configuration.** 15× was calibrated at
      **64×64, budget 3-0-6, ch1**, where the accumulator reached 466. Here it reaches 284 —
      fewer effective bits, so proportionally more quantization noise on the sidelobe, which
      is the exact mechanism the H_SHIFT entry already documents for the s7 PSR ceiling.
      Two independent checks that this is a stale threshold and not a regression:
      - **The offline model predicted it.** `phase1_sweep.py` at ch1/4-3-3 predicts ratio
        **10.69**; measured **11.8** — within the model's ~10% and on the good side.
      - **`CONV_RELU=0` cannot have caused it.** s7 drives `weights_ch0`, and ch0 is one of
        the 12 channels where ReLU never fires; the sweep gives byte-identical numbers for
        `base` and `base_nr` at ch1. Run A is blind to the ReLU change by construction.

      **ch1 is the documented WORST case for PSR** — channels add coherently, their
      quantization noise does not. Predicted at ch16/4-3-3: ratio **15.90**, accumulator 3.0%,
      response 20.1%, nothing railed.

      **Recorded reference for this configuration: 11.8x at 128×128 / 4-3-3 / ch1.** Not
      changing s7's 15× threshold on the strength of one measurement — see the
      "don't fit a scaling law to one data point" note under the DSPLib FFT loss entry. But
      **s7's threshold is now known to be geometry- and budget-dependent, so a FAIL from it
      alone is not evidence of a defect** until it is re-derived per configuration.

- [x] **DMA transactions 4258 → 1090/frame via `FFT_ROW_WS`/`FFT_COL_WS` 2 → 8
      (2026-08-14).** These set the DMA chunk size for the four ports carrying 96% of the
      host's GMIO traffic, so they are the transaction-count knob. Chunks go 1024 B → 4096 B;
      `CONV_INVOCATIONS`, `ROW_CHUNKS`, `COL_CHUNKS` all 64 → 16.
      ```
      gmio_weights 1024->256   fft_row_out 1024->256   fft_col_out 1024->256
      accum_out    1024->256   ifft_row_out  64-> 16   response      64-> 16
      total 4258 -> 1090 tx/frame  (3.9x)
      ```
      That matters because per-transaction driver cost is the dominant unmeasured risk
      (2-10 µs/tx ⇒ 8.5-42.6 ms/frame); this shrinks it by ~4x in whichever regime turns out
      to be real, without needing to know which.

      **Numerically neutral, verified rather than assumed:** both kernels re-checked bit-exact
      at WS=8 (`conv2d` ch11 s6, `cmul` s7 and cmul_stress) — 16384/16384 identical in every
      case, with the drain logs confirming 16 × 4096 B. conv2d's stateful row buffer takes a
      genuinely different path here (8 output rows per firing instead of 2), so this was not a
      formality. Kernel schedules unchanged (conv2d 140 cyc/16 px, cmul 2 cyc): only the
      chunking moved.

      Cost: `total_memory_size` 50616 → 130532 B across 6 cores; still 6 cores, 16 tiles.

- [x] **FOUND AND FIXED: the host never received `FFT_ROW_WS`/`FFT_COL_WS` (2026-08-14).**
      `GCC_FLAGS` passed `PATCH_ROWS`, `PATCH_COLS`, `N_CHANNELS`, `ITER_CNT`, `CMUL_H_SHIFT`
      — but not the window sizes, while `AIE_FLAGS` did. `mosse_tracker.cpp:67-71` defaults
      them to 2 in its own `#ifndef`, so **`make sd_card FFT_ROW_WS=8` would have built an AIE
      graph expecting 1024-sample windows against a host chunking the same DMAs in 256-sample
      pieces.** Every chunk count derives from these, and a mismatch deadlocks the drain loops
      — presenting identically to the historical `aie2gm_nb` hang, which took days to diagnose
      the first time. Caught before any hardware run because the flag was checked rather than
      assumed. This was the fifth coupled constant with no compile-time check; see the
      preprocessing-coupling entry in Known Issues.

- [x] **conv2d VECTORIZED — 37 → 8.75 cycles/pixel, ~11.5 ms → ~4.1 ms per frame,
      BIT-IDENTICAL (2026-08-14).** `CONV_VECTORIZE=1` (default; `=0` restores the scalar
      loop). 16 output pixels per iteration via `aie::mac` over nine `load_unaligned_v` taps.
      ```
      before:  conv2d_kernel.cpp:215  MAC loop   37 cyc/pixel
      after :  aie.hpp:0 loop#81      MAC loop  140 cyc / 16 px = 8.75 cyc/pixel
      per channel  606k -> 143k cycles (MAC) ;  per frame  11.5 ms -> ~4.1 ms
      ```
      **Bit-exactness is by construction and checked both ways:** every shift is
      `aie::downshift` (ARITHMETIC/floor, matching signed C++ `>>`), never `srs` — srs rounds
      to nearest, which would move every pixel by up to 1 LSB and invalidate B2's documented
      ~1e-3 residual, the shift budget and every s6/s7 expected value. `aie::unpack`, not
      `cast_to`, widens the Hann table int16→int32: `cast_to` reinterprets bits and silently
      halves the lane count (it compiled and failed loudly only because the lane counts
      disagreed — a subtler case would not have).

      **THE 2.8x IS NOT THE ~40x ESTIMATED EARLIER IN THIS FILE, and the reason is the
      remaining bottleneck: the STREAM READ loop, untouched, is now 44% of the kernel.**
      `conv2d_kernel.cpp:229` still reads one int32 and does four scalar byte stores at 28
      cycles per 4 pixels (7 cyc/px) — comparable to the 8.75 the MAC loop now costs.
      `readincr_v` exists and would fix it, **but it pulls 128 bits per call against a 32-bit
      PLIO**, which is exactly the width mismatch `mosse_graph.h:118-120` records as having
      starved the kernel 4:1 once already. Worth ~1.8 ms/frame; do it deliberately, with the
      harness, not as a drive-by.

      Per-frame AIE compute is now **~6.4 ms** (conv2d 4.1 + cmul 0.13 + FFT/IFFT ~2.2),
      down from ~21.6 ms.

- [x] **`CONV_RELU` flag added, DEFAULT 1 (unchanged behaviour) — this is the Phase 1
      decision knob (2026-08-14).** `CONV_RELU=0` removes the half-wave rectifier and
      saturates only. Held-out measurement says 0 is the better setting by a wide margin (see
      the ReLU entry in Known Issues); it is off by default because unlike `CONV_VECTORIZE`
      it CHANGES NUMERICS, so it needs its own before/after run rather than riding along with
      a rewrite that is bit-identical by design. Both settings compile clean and are checked.

- [x] **HARNESS COVERAGE GAP FOUND AND CLOSED: channel 0 cannot test ReLU (2026-08-14).**
      The conv2d harness loaded `weights_ch0.bin` unconditionally, and a `CONV_RELU=0`
      negative control PASSED against the ReLU golden — which looked like a broken harness
      and was actually a true statement about ch0. Measured on the s6 patch, `acc >> shift`
      per channel:
      ```
      ReLU never fires (min > 0)          : ch 0,1,2,4,5,6,8,9,10,12,13,14   (12 of 16)
      ReLU clamps EVERY pixel             : ch 3, 7, 15
      ReLU clamps SOME but not all pixels : ch 11 only  (12434 of 16384)
      ```
      So ch0 — and 11 other channels — are blind to the ReLU flag entirely. `make gen_vectors`
      now writes `weights_ch<0..15>.bin` into every scenario dir (each with its own
      window-weighted `mean_prev`), and `KUT_CH` selects one. **`KUT_CH=11` is the channel to
      use for any ReLU question.** With it, toggling `CONV_RELU` moves 10693 of 16384 samples
      (65%) — the harness discriminates, as verified by running the kernel at `CONV_RELU=0`
      against the `--relu 1` golden and confirming it FAILS.

      That table is also independent corroboration of the ReLU finding: on the real design
      point ReLU either does nothing (12 channels) or annihilates the channel outright (3),
      with exactly one channel doing anything in between.

- [x] **cmul_accum VECTORIZED — 30 → 2 cycles/iteration, ~7.9 ms → ~0.13 ms per frame,
      BIT-IDENTICAL (2026-08-14).** `CMUL_VECTORIZE=1` (default; `=0` restores the scalar loop
      for bisection). Measured schedule from `aiecompiler.log`, 128×128 hw target:
      ```
      before:  cmul_accum_kernel.cpp:116  arith loop  30 cycles  x 256 iter  (NOT pipelined)
      after :  cmul_accum_kernel.cpp:167  arith loop   2 cycles  x  32 iter  (pipelined)
      per invocation  7744 -> ~128 cycles ;  per frame (64 inv x 16 ch)  7.9 ms -> ~0.13 ms
      ```
      **The documented assembler OOM did not recur** — it was triggered by
      `chess_prepare_for_pipelining` on *scalar int32* arithmetic, and the vector path is a
      different code path, so the loop is now pipelined as well as vectorized. That OOM
      workaround had been costing ~7.8 ms/frame.

      **Bit-identical, and that is checked rather than argued:** `x86sim_check KUT=cmul` on
      both s7 (real complex filter) and `cmul_stress` (41.7% of outputs railed) reports
      16384/16384 samples identical on both real and imaginary parts.

      The identity that makes it exact — worth keeping if this is ever touched again:
      `from_vector(acc, S)` seeds the accumulator with `acc*2^S` exactly, so
      `srs_S(acc*2^S + prod) = sat16(acc + ((prod + 2^(S-1)) >> S))`, because an exact
      multiple of `2^S` passes through the floor untouched. The accumulator must therefore be
      folded in BEFORE the shift; converting the product to cint16 first and then adding
      clamps twice, which is wrong whenever the product overflows int16 but the sum does not.
      Two mode settings are load-bearing: `rounding_mode::positive_inf` (ties toward +inf) is
      what `(x + CMUL_RND) >> S` does — `conv_even` would differ on exact ties — and
      `saturation_mode::saturate` reproduces `sat16()`.

      `alignas(32)` on `flt_local`/`acc_local` is required, not cosmetic: `aie::load_v` is an
      aligned access and a `cint16_t` array carries only 4-byte alignment. **x86sim does not
      enforce alignment**, so omitting it passes every bit-exactness check and then misbehaves
      on hardware.

      Remaining easy win, not taken: the tile-local copy loop is now HALF the kernel's cost
      (64 of ~128 cycles). The vectorized arithmetic reads over the vector bus anyway, so the
      copy's original justification (avoiding ~100K-cycle scalar reads from the memory tile)
      no longer applies. Worth ~0.065 ms/frame — left alone to keep this change attributable.

- [x] **x86sim BIT-EXACTNESS HARNESS — and `simulate_conv2d` is now VERIFIED faithful
      (2026-08-14).** `make x86sim_check KUT=conv2d|cmul` isolates one kernel, dumps its raw
      output, and diffs it against the Python model **with no tolerance**, in seconds. Files:
      `design/aie_src/kernel_only_graph.{h,cpp}` (built on the existing `fft_only_graph`
      pattern), `design/aie_src/aiesim_scenario_io.h` (the scenario loader, lifted out of
      `mosse_graph.cpp` so the two harnesses cannot drift), `scripts/check_kernel_bitexact.py`,
      and `simulate_cmul()` in `gen_aiesim_vectors.py`.

      **Results against the UNMODIFIED scalar kernels — all three PASS:**
      ```
      conv2d  s6           16384/16384 identical   (out_shift=7 bias_acc=3160208 mean_prev=24580)
      cmul    s7           16384/16384 identical   both re and im, real complex filter
      cmul    cmul_stress  16384/16384 identical   41.7% of outputs at a rail
      ```
      conv2d's run had **`mean_prev = 24580`, i.e. Stage B1 was genuinely active**, so the
      match covers `bias_acc`, `out_shift`, ReLU, the B1 subtract and both `>>15` Hann
      truncations.

      **Why this matters beyond the vectorization it gates:** `gen_aiesim_vectors.py:205`
      claimed `simulate_conv2d` "replicates the integer arithmetic in conv2d_kernel.cpp
      exactly" and that had never been checked against the kernel — while the entire offline
      Phase 1 plan (choosing the `bias_acc` variant and sweeping the shift budget in Python
      instead of in hardware) rests on it. **It is now verified, so that offline work is
      trustworthy.**

      **A coverage gap was found and closed in the process: NO scenario exercised `sat16`.**
      Across s0-s7 only s0 has a non-zero `accum_prev` at all (max 1024) and none approaches
      the rail, so cmul's saturating add was dead code under every existing test — the exact
      path whose predecessor **wrapped**, flipping the accumulated spectrum's sign and sending
      the argmax to a garbage index. `write_cmul_stress_scenario()` now emits an
      `aiesim_data/cmul_stress` directory covering both rails, near-rail values that must NOT
      clamp, and full-scale operands; it rails 41.7% of outputs and prints that percentage so a
      future change that stops stressing saturation is visible.

- [x] `conv2d_kernel.cpp`: 3×3 MAC + ReLU + Hanning window (MobileNet-v3 Small layer 1)
- [x] Weight export: MobileNet-v3 Small layer 1 (INT8-quantized, RGB→grayscale collapsed)
- [x] `cmul_accum_kernel.cpp`: element-wise cmul_conj + accumulate (int32 intermediates)
- [x] cmul_accum saturation bug fixed (2026-07-30) — the int32→int16 accumulate cast
      **wrapped** instead of clamping, flipping the accumulated spectrum's sign on
      overflow and sending peak detection to a garbage index. Now saturates.
- [x] aiesim N_CHANNELS=1: PASS (real arithmetic with ISS workarounds for GMIO/PLIO bugs)
- [x] Two-channel aiesim support added (N_CHANNELS=2 option in sim)
- [x] Pre-computed FFT bypass for aiesim — **now obsolete**; it was working around a
      misdiagnosis (see Known Issues). It SKIPS the PatchIn→conv2d→row-FFT path, so
      `make aiesim` does not validate it. Use `make aiesim_plio`.
- [x] conv2d hang root-caused (2026-07-30): weights-buffer starvation, not the PLIO
- [x] `v++ --package` rootfs corruption fixed — `make rootfs` (every hw_emu run died at boot)
- [x] aiesim: full conv2d → FFT → cmul → IFFT → response chain drains end to end
- [x] aiesim scenarios s0–s4 PASS in echo mode (`CONV2D_MODE=1`); s6 PASSES on the real
      path (`CONV2D_MODE=0`). s0–s4 feed RAW patches, so they are only meaningful with
      echo — s6 is the only scenario whose input has been through Stage A.
- [x] Localisation verified exact (`err=0 px`) on a realistic smooth patch — transposes and
      IFFT row/col indexing are correct, no systematic bias
- [x] FFT/IFFT shifts are build-parameterizable (`FFT_SHIFT`, `IFFT_ROW_SHIFT`,
      `IFFT_COL_SHIFT`), fed from ONE Makefile variable to both the AIE compile and
      `gen_aiesim_vectors.py` so the graph and the vectors cannot disagree
- [x] **MOSSE preprocessing implemented (2026-07-31)** — it was entirely absent before.
      Stage A in `roi_crop` (bilinear resample + border clamp + log + zero-mean + unit-L2 +
      int8 quantize), Stage B1 in `conv2d` (prev-frame window-weighted mean subtraction),
      Stage B2 on the APU (9-bin frequency correction), Stage B3 hooked into the filter
      update — which was then the `filter_update_kissfft` stub and is now
      `filter_quantize_q15()` in `mosse_filter.cpp` (2026-08-05; KissFFT was never needed,
      see the Key design decisions entry). `hanning_*.h` switched to the periodic Hann.
      Cost: 44 DSP / 10 BRAM18 / 7694 LUT in PL, ~472 µs/frame (1.4% of a 33 ms budget),
      no new AIE tiles.
- [x] **roi_crop verified bit-exact — FOR REAL THIS TIME (2026-08-16). `make test_roi_crop`,
      17 cases, 0 failures.** This entry previously claimed a 6-case native C simulation
      against a NumPy reference. **That harness never existed** — no testbench, no `csim`
      target, no reference, and `git log --all --diff-filter=A -- '*roi*'` returns only the
      original kernel commit `5b9a40f`. The claim was load-bearing in two places: the scale
      estimation entry below said the resample "hardware is already built and tested", and
      `gen_aiesim_vectors.py:926` justified its float Stage A by citing it.

      **And the interpolator had never executed.** Every build to date runs
      `roi_h == patch_rows`, which makes `step_y` exactly 256, hence `fy == fx == 0`, hence
      `top = p00<<8`, `val = p00<<16`, `pix = p00` — the entire bilinear datapath
      (`ap_uint<18> top`, `ap_uint<27> val`, the `>>16`) collapses to a copy.

      **11 of the 17 cases execute it for the first time, and all 17 pass bit-exact.**
      Coverage: 1:1, 2× up (first nonzero `fy` in the project's history), 2× down, paddings
      1.5/2.5/3.0, an anisotropic case with `step_y ≠ step_x` (the only one that would catch
      an x/y transposition), negative `roi_row`/`roi_col`, both edge-clamp paths, whole frame,
      two `var == 0` paths, the `ROI_INV_Q_MAX` cap plus a negative control just below it, a
      case where the kernel's two independent floors annihilate a real variance to 0, a
      non-square patch, the `recompute=0` cached path, and the AXIS framing contract.

      The golden comes from **`scripts/roi_crop_ref.py`**, which `phase1_sweep.py` also uses
      to model the ROI — one reference, two consumers, so the sweep's padding numbers inherit
      what this target proves. Tolerance is **zero**, unlike `test_host`'s 1-LSB slack: this
      is an integer datapath and any difference is a bug.

      **Measured while validating it: the float Stage A in `gen_aiesim_vectors.py:928` differs
      from the kernel on 40.9% of samples, by at most 2 LSB** (rms 0.65 on a signal of
      std 32). Small, so nothing already recorded is invalidated — but it is ~2% relative,
      comparable to the model's own hardware agreement, and it had never been checked.
- [x] **aiesim s6 PASS** — the first scenario driven by a realistic Stage-A-preprocessed
      patch through the real conv2d path (`make aiesim_plio SCENARIO=s6 CONV2D_MODE=0
      FFT_SHIFT=3 IFFT_COL_SHIFT=6`). `err=0 px`, `rails=0`, all seven checks OK.
      Note it needs FFT_SHIFT=3 — see the known issue below.
- [x] The uint8→int8 contract violation is fixed — `roi_crop` used to emit *unsigned*
      0..255 into a port `conv2d_kernel.cpp:105-108` reads as `(int8_t)`, so every pixel
      ≥128 silently wrapped negative

- [x] **Filter init + update implemented (2026-08-05)** — `design/host_app_src/mosse_filter.{h,cpp}`.
      Bolme eq. 10–12, shared denominator, Stage B3 energy folding, Q1.15 export.
      `filter_init` on frame 0, `filter_update` (η=0.125) thereafter, both driven from
      `mosse_tracker.cpp`. No FFT library: `F_ch` comes from the new `gmio_fft_col_out`
      tap and `G` is closed-form. Verified by `make test_host` against a NumPy golden
      (`scripts/gen_filter_golden.py`) — all six comparisons at ≤1.5e-7 relative,
      quantized filter bit-exact (0 of 8192 int16 differ).
- [x] **`cmul_accum` H_SHIFT added (2026-08-05)** — the kernel multiplied F by H with no
      shift, and every scenario passed a literal H=1, so the whole shift budget was
      calibrated at a filter gain of one. H is now Q1.15 with a rounding `>>H_SHIFT`.
- [x] **aiesim scenario s7 (2026-08-05)** — first scenario with a real per-bin complex
      filter and an off-centre target, asserting location, sign, a PSR of ≥20× (11×11
      mainlobe excluded, Bolme §3.5) and the `gmio_fft_col_out` tap's correlation.

- [x] **s6 regression PASSES with H_SHIFT + the F_ch tap (2026-08-05)** —
      `make aiesim_plio SCENARIO=s6 CONV2D_MODE=0 PATCH_ROWS=64 PATCH_COLS=64
      FFT_SHIFT=3 IFFT_ROW_SHIFT=0 IFFT_COL_SHIFT=6 SIM_WALL_TIMEOUT=5400`.
      `OVERALL: PASS`, `err=0 px`, every sub-check OK.
      The numbers are **bit-identical to the pre-change baseline**: `max|accum|=1929`
      and peak `{-417,0}`, both exactly as recorded before H_SHIFT existed. That is the
      evidence that Q1.15 unity (`H=32767`, `>>15`) reproduces the old literal `H=1`
      and leaves the shift budget untouched.
      `F_ch tap: correlation with golden = 0.9998` — `gmio_fft_col_out` delivers the
      right spectrum in the right order.

- [x] **s7 proved a real MOSSE filter localises exactly on hardware (2026-08-05)** —
      `err=0 px` at the off-centre target (10,−7), flat index 697, through
      Stage A → conv2d → B1 → FFT → cmul(H_SHIFT) → IFFT. That validates the
      conjugation convention, the Q1.15 export and the tap end to end.
      It also FAILED its PSR check at `H_SHIFT=15`, which is what exposed the
      quantization-ceiling bug above. At `H_SHIFT=10`: **`OVERALL: PASS`** —
      accum 466, row IFFT 5817, response 785, PSR 19.6×, `err=0 px`, `rails=0`
      at every stage, `F_ch tap` correlation 0.9998, all ten sub-checks OK.
      Every stage matched the linear prediction to within 6%.

      **s7's PSR threshold is measured, not derived.** It was first set to
      `0.7 × golden` on the untested assumption that fixed point retains 70% of the
      float ratio; it retains ~51% (golden 38× → measured 19.6×). The gap is
      spectral quantization: the accumulator sits at 466/32767, so the spectrum
      carries ~9 bits and its rounding noise lifts the sidelobe to 40 where the
      true value is ~20. More gain fixes it (`H_SHIFT=8` quadruples the signal
      while the noise stays absolute) but rails the accumulator at 16 channels.
      **ch1 is the WORST case for PSR** — channels add coherently, their
      quantization noise does not. Threshold now 15×, which still fails a ~25%
      regression.

      **CORRECTION 2026-08-12 — the last clause of this entry used to read "and sits
      well above Bolme's §3.5 failure indicator of ~7". That comparison is invalid.**
      What `gen_aiesim_vectors.py` computes and calls PSR is
      `|peak| / max|sidelobe|` (`snr_ratio_pct`); Bolme's PSR is
      `(g_max − µ_sl) / σ_sl`. Same 11×11 circular exclusion, **different
      statistic** — for a noise-like sidelobe `max ≈ µ + 3..4σ`, so the two differ by
      several times and neither one's thresholds transfer to the other. The "19.6×"
      and "golden 38×" figures throughout this file are peak-to-max-sidelobe ratios,
      **not** comparable to Bolme's 20-60 normal range or his ~7 failure mark.
      `report_psr()` in `mosse_tracker.cpp` therefore prints BOTH, labelled, so the
      hw_emu baseline can be read against the aiesim scenarios and against the paper
      without conflating them.

- [~] **THE TRACKER WORKS END TO END ON HARDWARE — but NOT at the real design point
      (2026-08-13, qualified 2026-08-14).**
      **conv2d ran in ECHO MODE (`CONV2D_ECHO_TEST=1`) in this build**, so the chain listed
      below is wrong on one link: `conv2d → B1` did not run. No 3×3 MAC, no ReLU, no B1 mean
      subtract, no Hanning window; `roi_crop`/Stage A and everything downstream of the row FFT
      are as described. See the echo-mode entry at the top of Known Issues for what that
      qualifies and what it leaves standing. **The result below is real and worth keeping —
      it is just not the ch16 conv design point, and the numbers in it must not be used to
      calibrate anything until a `CONV2D_MODE=0` run reproduces them.**
      hw_emu, 128×128, `N_CHANNELS=16`, `ITER_CNT=2`, budget 4-2-2, `H_SHIFT=10`.
      Frame 0 initialises the filter, frame 1 tracks:
      `Frame 1: displacement (10,-7) → pos (550,953) peak|re|=32666 [OK: matches injected
      offset]` — **`err=0 px`**. This closes the long-open "hw_emu localisation is NOT yet
      validated" item. The whole chain is now hardware-validated: `roi_crop` → PatchIn PLIO →
      conv2d → B1 → row FFT → transpose → col FFT → cmul(H_SHIFT) → B2 → IFFT rows →
      transpose → IFFT cols → response → peak detect → filter update.

      **The response is the RIGHT FUNCTION, not just the right argmax.** The printed profiles
      match an ideal σ=2 Gaussian `exp(-d²/8)` at both radii, within ~2%:
      ```
      d=1px  measured/peak 0.893, 0.897   ideal 0.8825   err +1.2%, +1.7%
      d=2px  measured/peak 0.614, 0.620   ideal 0.6065   err +1.3%, +2.1%
      ```
      That is much stronger evidence than localisation alone — it validates the transposes,
      both IFFT passes, the conjugation convention and the Q1.15 export as producing the
      correct correlation surface, not merely a peak in the right cell.
      **This paragraph SURVIVES the echo-mode qualification**: every step it validates is a
      property of the filter maths and the FFT chain, independent of what features feed them.

      Other numbers from the run — **all ECHO MODE, and the PSR figures are further inflated
      by the 16 bit-identical channels (echo-mode consequence 2)**:
      `PSR 124.99` (Bolme) / `ratio 17.37x` (aiesim metric) — a
      **7.2× divergence between the two statistics**, which is why both are reported; peak
      definitions agreed and the peak was positive, so the `|real|`-vs-signed-max concern did
      not bite; `F_ch max|.| = 90` on both frames; `H(q15)` peaked at full scale as designed.
      **PSR 125 is above Bolme's 20-60 because the test target is synthetic on an empty
      frame** — a near-empty sidelobe inflates it. Treat PSR here as a RELATIVE metric for
      comparing builds, not an absolute one, until real video feeds it.

      One explained oddity: `H(q15)` shows `rails=1` on frame 0 but `rails=0` on frame 1 with
      `max|.|=32767` both times. That is the complex-MAGNITUDE normalization — when the peak
      bin's phase is near 45°, `32767/√2` lands in each component and neither rails
      individually. Not a bug.

### In Progress / Next

#### Fix order — REWRITTEN 2026-08-14 after the plan was overtaken by events

The two principles still hold and both earned their keep: **instruments before changes** (the
x86sim harness and the offline model paid for themselves several times over) and **never move
two magnitudes at once** (every isolating run below settled something a combined run could
not).

**What the original plan got wrong, recorded because the pattern is instructive.** It ordered
the work around `bias_acc` on the strength of an echo-mode baseline. In the event:

| planned | actual |
|---|---|
| Phase 1 = fix `bias_acc`, variant (a) first | variants (a)/(b) make tracking **3× worse**; the fix was `CONV_RELU=0` |
| the shift budget total "must GROW" from 12 | premise was an echo-mode artifact; the real problem was elsewhere |
| Phase 3: try raising `eps_rel` | refuted — 1e-3 is already the optimum |
| Phase 5 (last): vectorize | should have been first; it was the largest measured win in the file |
| — | **the actual top bug, `mean_prev=0` on frame 0, was not on the list at all** |

Three of the five phases were pointing at the wrong thing, and the one item that mattered most
was invisible until the offline model was validated against hardware. The generalisable lesson
is in the echo-mode and `mean_prev` entries: **a plan built on unvalidated measurements
inherits their errors, and `err=0 px` passed through every one of these failures.**

##### Current state

- **DONE:** x86sim bit-exactness harness; conv2d and cmul vectorized (21.6 → 6.4 ms AIE
  compute); DMA transactions 4258 → 1090; `CONV_RELU=0`; `mean_prev` seeding; offline model
  validated against hardware to 3-11% on the real test patch.
- **NEXT: a ch1 run at the new geometry (~1.7 h), THEN one ch16 run (~23 h) at budget
  4-2-1.** ch16 is the only thing that covers multi-channel accumulation with real conv2d, but
  at ~11.5 h/frame it is 13× more expensive than this file used to say, and the Phase D/E host
  code (box state, the patch↔frame conversions, IoU, the scale filter) is compiled but has
  never executed — ch1 exercises all of it for 1/13th the machine time. Budget 5-3-4 is
  retired: on the post-2026-08-16 geometry it puts the response at 0.4% of range.
- **PSR gating COMPLETE 2026-08-15** — native tests, a bit-exact ch1 regression vs run D,
  and an occlusion run proving the gate fires (PSR 3.90 → `LOW_PSR`), holds position, freezes
  the filter bit-exactly, and reacquires with `err=0 px`. Largest functional gap: closed.
- **THEN, in this order:** target size / σ
  anchoring; `TARGET=hw` for the µs/tx number; scale search and RGB.
- **DROPPED:** `bias_acc` variants (a)/(b) — harmful; raising `eps_rel` — refuted; channel
  pruning — moot, ReLU-off leaves no structurally dead channels.

Original plan follows for reference.

- **Phase 0 — instrument, then take the ch16 baseline.**
  **[x] Instrumentation LANDED 2026-08-12** (`mosse_tracker.cpp`, host-ELF only, not one byte
  entering the AIE changed; builds clean at 128×128/ch16/ITER_CNT=2, `make test_host` still
  PASS): `report_psr()` prints Bolme §3.5 PSR *and* the aiesim `|peak|/max|sidelobe|` ratio at
  both peak definitions (argmax|re| and argmax re), flags disagreement and negative peaks, and
  classifies against Bolme's ranges — **report only, nothing gates on it yet**. `DMA_TX`/`DMA_T`
  macros bracket every GMIO `async`/`wait` with a timer (call sites left textually in place,
  because the ordering is load-bearing), giving a per-frame and cumulative per-port transaction
  count and host time.
  **[~] Baseline run DONE 2026-08-13, but it ran conv2d in ECHO MODE — REDO REQUIRED
  (found 2026-08-14).** 128×128 / ch16 / ITER_CNT=2, ~14 h per frame, ~28 h total.
  **`err=0 px`, `OK: matches injected offset`.** But the build carried
  `CONV2D_ECHO_TEST=1`, so this is not the ch16 conv design point — see the echo-mode entry
  at the top of Known Issues. **Phase 0 is therefore NOT complete: the instrumentation
  landed and works, but the baseline it exists to produce has not been taken.** Re-run at
  `CONV2D_MODE=0` before Phase 1 starts; Phase 1's premise (below) is derived entirely from
  echo-mode numbers. The blocking consequence recorded from it — **the response at 99.69% of
  the cint16 rail** — is a measurement of a pipeline without conv2d, ReLU, B1 or the Hanning
  window, and with 16 bit-identical channels.
  **PSR is the instrument, not just a fix.** The current pass/fail is `err=0 px`, which is
  documented blind to exactly the bug class Phase 1 addresses (s7 localised *exactly* while
  its PSR collapsed to 5.2×). Without PSR, Phase 1's result is not interpretable.
- **Phase 1 — the `bias_acc`/`out_shift` scale fix, PLUS a shift-budget increase in the same
  change.** Highest measured payoff (5.5 bits, 2 dead channels) and the highest blast radius:
  it invalidates the shift budget and every s6/s7 expected value. Its own phase, judged by
  Phase 0's PSR. Start with variant (a); see the Known Issues entry for the three variants and
  the open question about keeping ReLU at all.
  **BLOCKED on redoing Phase 0's baseline at `CONV2D_MODE=0` (2026-08-14).** Everything below
  about the budget total is derived from echo-mode numbers, and the echo path has neither the
  ReLU/bias DC that variant (a) and (b) act on nor the Hanning window. The `4/3/4` starting
  point and the "~15 for (a), ~17.5 for (b)" sizing are therefore un-anchored until the real
  baseline exists. **The direction of the argument still holds** — recovering conv-output bits
  scales the response — but the magnitudes do not.

  **The budget increase is NOT optional — Phase 0 measured the response at 99.69% of the rail
  (101 LSB of margin).** `accum ∝ |F|` and `response ∝ accum`, because H is renormalized to
  full scale regardless of `|F|` (H ≈ conj(G)F/(B+ε), B ≈ |F|², ε relative to mean(B), so
  `H_q15`'s shape is invariant to a uniform scaling of F). Phase 1 raising F therefore raises
  the response by the same factor, and variant (a)'s ~2 bits alone would drive it to ~130000 —
  clipping by 4×, which flattens the peak and corrupts the argmax.
  Redistribution does not help: any budget summing to 12 leaves the response scale unchanged.
  The total must GROW. Sweep from `FFT_SHIFT=4 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=4` (total 15) —
  `FFT_SHIFT=4` puts accum at a healthy ~32% post-Phase-1 and the extra attenuation goes on the
  IFFT side, landing the response near 50%. **The required total depends on which variant is
  chosen** — ~15 for (a)'s 2 bits, ~17.5 for (b)'s full 5.5 — so settle the variant question
  before sweeping. `IFFT_ROW_SHIFT` must stay non-zero (at 16 channels its input is the
  accumulated spectrum and it hit ~101000 unattenuated).
- **Phase 2 — re-measure the feature bank, then prune.** Re-run `check_collapse.py` Q2/Q4 and
  drop whatever is *still* structurally dead or collinear, batched into one `libadf.a` rebuild
  (`N_CHANNELS` is a compile-time define). Provably free: a zero channel contributes zero to
  `A`, zero to `B`, and cannot affect the global `max|H|` normalization. Verify nothing assumes
  `N_CHANNELS` is a power of two.
- **Phase 3 — host-only numerics, strictly serialized.** `eps_rel` first (one constant, ~35×
  accumulator effect), then target size / σ anchoring. Together they are uninterpretable.
- **Phase 4 — tracking behaviour.** Flip PSR gating on; add affine init. Both need ITER_CNT≥3
  and a target that actually moves, so they want the video path or a scripted trajectory first.
- **Phase 5 — gated on Phase 0's DMA number.** Scale search and RGB. If the 4258 transactions
  land at 10 µs/tx (42.6 ms), both are infeasible until the transposes move off the APU — an
  architectural change that would invalidate front-end work done before it. Note this gate
  needs `TARGET=hw`, NOT the hw_emu baseline — hw_emu cannot answer it (see the timing caveat
  in Known Issues), so this phase is not actually blocked behind the ch16 re-run and can
  proceed in parallel given board access.
  **Vectorizing conv2d AND cmul_accum belongs here and should come FIRST within the phase** —
  arguably it should be its own phase ahead of everything else. Together they are **19.4 of the
  21.6 ms of AIE compute per frame** (see the AIE compute-cost section), both scalar, and cmul's
  arithmetic loop is not even pipelined — it carries a `chess_prepare_for_pipelining` omission
  worked around an assembler OOM that has never been revisited. The payoff is ~19 ms/frame,
  larger than the entire DMA uncertainty in the optimistic regime, and it is what turns RGB from
  a 31% frame-rate loss into a free one. Unlike every other roadmap item it perturbs no
  calibrated constant — same weights, same `bias_acc`, same shift budget, same DMA count, same
  numerics if done correctly. Large payoff, no coupling.

#### Tracker-level gaps vs the two source papers (audited 2026-08-12)

The signal-processing core of both papers is implemented and validated. What is missing is
the *tracker* around it. `docs/` holds both PDFs; section numbers below refer to them.

- [x] **PSR failure detection and update gating (Bolme §3.5) — DONE 2026-08-15. Implemented,
      regression-validated, and PROVEN FIRING under occlusion on hardware.** This was the
      largest functional gap in the tracker; it is closed.

      **Regression run (hw_emu, 128×128, ch1, `ITER_CNT=2`, budget 5-2-2) reproduces run D
      BIT-EXACTLY.** The XSA and `libadf.a` were not relinked — `aie.flagstamp` was byte-
      identical to run D's, so make reused them and **the gate was the only variable**:
      ```
                     run D    this run
      F_ch max         49        49
      accum max        70        70
      response max   2534      2534
      Bolme PSR    172.41    172.41
      ratio         23.04x    23.04x
      displacement (10,-7)   (10,-7)   err=0 px, rails=0 everywhere
      DMA           100 tx    100 tx   (the gate adds no transactions when accepting)
      ```
      That is the evidence that moving `compute_psr` into `mosse_filter`, deleting
      `peak_detect_sw` and restructuring the loop are all behaviour-preserving.
      ```
      [gate] frame 1: ACCEPT  reason=ACCEPT  psr=172.41  threshold=7.00
      [gate] SUMMARY over 2 frame(s): 1 evaluated, 1 accepted, 0 gated
      ```
      **Frame 0 correctly bypassed the gate** (`[INIT] response not evaluated`, no gate line,
      `filter: INITIALISED`) — the branch ordering that protects the bootstrap holds on
      hardware, not just in the unit tests.

      **THE NEGATIVE-PEAK RISK IS RETIRED — measured, not argued.** The concern was that
      Stage B1 makes the response bipolar (aiesim s6 peaks at `{-417,0}`), so a legitimately
      negative hardware peak would trip `NEGATIVE_PEAK` and freeze everything — and unlike
      `LOW_PSR`, that condition is deliberately NOT switched off by `PSR_GATE_MIN=0`.
      Hardware says the peak is **+2534** and that both peak definitions agree:
      ```
      [psr] at argmax|re| (10,-7): peak 2534 ...
      [psr] at argmax re  (10,-7): peak 2534 ...
      [psr] peak definitions agree.
      ```
      So `NEGATIVE_PEAK` does not need to become overridable. Note this took ONE run to
      settle only because `report_psr` prints both peak definitions and the SIGNED peak —
      the pre-gating log printed `peak|re|` (a magnitude), from which the sign is not
      recoverable. Worth remembering the next time a diagnostic looks redundant.

      Original entry follows. `filter_update` no longer runs unconditionally. Below
      `PSR_GATE_MIN` (default 7.0, Bolme's number) the host **holds the tracked position and
      skips both `filter_update` and `publish_filter`**, so an occluded frame cannot train
      the filter on background at η=0.125.

      **Holding the position is what makes reacquisition work with no reacquisition code**:
      the ROI stays on the last known good location, the filter keeps being applied there,
      and PSR recovers when the target reappears. Moving to a gated frame's peak — which is
      noise — walks the ROI off the target, after which it can never re-enter the search
      window. It also keeps the test meaningful: the recovery frame's expected displacement
      is still exactly `(IMPULSE_DR, IMPULSE_DC)` because `pos` never moved.

      **Skipping `publish_filter` is a correctness requirement, not an optimization.**
      `filter_quantize_q15` reads `g_energy`, which the per-channel loop overwrites with the
      *current* frame's energies before the response exists — so publishing on a gated frame
      would re-scale the *old* A/B by the *occluder's* per-channel norms. Skipping leaves
      `filter_bo` holding the H from the last accepted frame, which is what "frozen" must
      mean. The `H(q15)` report is guarded on the same flag, because `filter_scratch` still
      holds the previous frame's bytes and printing them as current is exactly the kind of
      plausible-looking output that costs a day.

      **Four failure conditions, not one** — the reason is reported, because "HOLD" alone is
      unactionable at ~50 min/frame at ch1 and ~11.5 h/frame at ch16:
      ```
      ZERO_RESPONSE   peak == 0     the pipeline produced nothing — shift budget/filter scale
      FLAT_SIDELOBE   sdev == 0     constant map; PSR undefined, not infinite
      NEGATIVE_PEAK   peak < 0      anti-correlation; g is a positive Gaussian, so not a detection
      LOW_PSR         psr < 7.0     Bolme's occlusion / failure indicator
      ```
      `NEGATIVE_PEAK` discharges the open item in the SIGNED-response entry under Known
      Issues, which asked for exactly this once gating landed. The structural three are kept
      separate from `LOW_PSR` on purpose: `compute_psr` reports `psr = 0` for a degenerate
      sidelobe, which would read as "occluded" when the pipeline in fact produced nothing.

      **`PsrResult`/`compute_psr` moved to `mosse_filter.{h,cpp}`**, which includes no XRT
      header — so the gate is under `make test_host` and runs in seconds. That matters more
      here than anywhere else in the design: **the gate CANNOT fire on the current synthetic
      test data** (PSR 172 against a threshold of 7), so the native tests are the only place
      its failure paths are exercised at all. 34 assertions, all passing:
      ```
      clean sigma=2 Gaussian at (10,-7)   psr 536.61  ratio 88.76x   ACCEPT
      pure noise                          psr   1.77                 LOW_PSR
      all zeros / constant map                                       ZERO_RESPONSE / FLAT_SIDELOBE
      sign-flipped clean peak             psr high                   NEGATIVE_PEAK
      circular-exclusion invariance       centred vs corner psr equal to 1.2e-14
      ```
      The circular-exclusion case is the one that earns its keep — the same peak centred and
      wrapped into a corner must score identically, and a linear exclusion window fails it by
      a wide margin. Tests are built **analytically in C++, with no NumPy golden**: a golden
      would be a second implementation of the logic under test, in another language, to be
      kept in sync forever — the failure mode this file already records twice, not a
      safeguard. NumPy earns its keep for `filter_update`, which is genuinely non-obvious.

      **The peak is now scanned ONCE per frame.** `peak_detect_sw` is deleted: it performed a
      bit-identical `|real|` scan to `compute_psr(use_abs=true)`, so keeping both meant two
      independent scans that could disagree about which peak the tracker acted on — one
      driving the position, the other driving the gate.

      **THE GATE HAS NOW FIRED ON HARDWARE — occlusion run 2026-08-15, `ITER_CNT=3
      OCCLUDE_MASK=0x2` (init → occluded → reacquire). FULL PASS.**
      ```
      Frame 1 [OCCLUDED, checkerboard]
        [psr]  peak 1566  sidelobe mu -2.7 sd 402.1 max 1552   PSR 3.90   ratio 1.01x
        [gate] frame 1: HOLD  reason=LOW_PSR  psr=3.90  threshold=7.00
        Frame 1: displacement (9,-3) HELD, pos stays pos (540,960)
        filter: FROZEN — update and publish both skipped (LOW_PSR)
      Frame 2 [target returns]
        [psr]  peak 2534  PSR 172.41  ratio 23.04x
        [gate] frame 2: ACCEPT   REACQUIRED after 1 gated frame(s)
        Frame 2: displacement (10,-7) → pos (550,953)   err=0 px
      [gate] SUMMARY: 2 evaluated, 1 accepted, 1 gated;  PSR min 3.90 / max 172.41
      ```
      **`ratio 1.01x` is the number that says "occluded"**: the largest sidelobe (1552) is
      within 1% of the "peak" (1566), i.e. there is no peak at all, and the sidelobe σ is 402
      against 14.7 on a good frame. Bolme's PSR reads 3.90 — squarely inside his "around 7.0
      indicates occlusion" band, on the first attempt, with no threshold tuning.

      **THE FREEZE IS PROVEN BIT-EXACTLY, by an argument stronger than the one planned.**
      The intended assertion — `cmp` the frame N and N+1 `H_q15` dumps — turned out to be
      *impossible by construction*: a frozen frame skips `publish_filter`, and the H dump is
      guarded on that, so no `H_q15_f1.bin` is ever written. Its **absence** from the SD image
      is itself positive evidence (`F_ch_f1`/`accum_f1`/`resp_f1` are all present; only
      `H_q15_f1` is missing). But the decisive check is a cross-run comparison:
      ```
      occlusion run frame 2  H_q15  ==  regression run frame 1  H_q15   BYTE-IDENTICAL
      (and H_q15_f0 is byte-identical across both runs — determinism confirmed)
      ```
      In the regression, frame 1 detected using the filter straight from `filter_init`. In the
      occlusion run, frame 2 detected using that same filter **after passing through a gated
      frame**. Identical bytes therefore prove the occluded frame perturbed *nothing* — and
      because the comparison is of the H produced by the post-detection update, it covers the
      host-side `g_filter.A`/`B` as well, not just the device buffer `filter_bo`. The planned
      `cmp` would only have covered the latter. Every downstream number agrees too: F_ch 49,
      accum 70, response 2534, PSR 172.41, ratio 23.04 — identical across the two runs.

      **`OCCLUDE_MASK` semantics: bit *f* occludes frame *f*, so `0x2` occludes frame 1.**
      Earlier drafts of this file described `0x2` as occluding frame 2; the prose was wrong,
      the code (`mosse_tracker.cpp`) was always right. Frame-1 occlusion is the better test
      anyway — it ends on a *recovery* frame, so it exercises reacquisition, which a
      frame-2 occlusion (`0x4`) cannot because the run ends there.

      **Risk worth tracking: `err=0 px` is no longer a sufficient pass criterion.** The
      occluded frame will legitimately report a mismatch — there is no target to find — and
      that is a *pass*. And CLAUDE.md's own Bolme Fig. 3 citation puts second-frame PSR at
      **≈4 for N=1**, below the threshold: on *real video*, with affine init (§3.4) still
      unimplemented, the gate could fire on frame 1 and freeze a filter that then never gets
      a second training frame. The margin on this harness is 17-24×, so it cannot happen
      here, and `PSR_GATE_MIN=0` disables the threshold test in one make variable if it
      does. A `PSR_GRACE_FRAMES` that skips the first N evaluated frames is the cheap
      stand-in for §3.4 — design it if real video lands first, and **default it to 0**.
- [x] **Target size / bounding box (both papers) — DONE 2026-08-16.** See the bounding-box
      entry in Completed. Tracker state is now a `TargetBox`, `roi = box × TARGET_PADDING`
      gives real background context, IoU is reported so the tracker is finally scoreable, and
      the patch↔frame conversions the box makes necessary are unit-tested. σ is anchored to
      the box behind `SIGMA_FROM_TARGET`, defaulting OFF — the sweep showed Bolme PSR is
      monotone in σ and therefore cannot arbitrate the DSST rule; that needs real video.
      *(Was: "Tracker state is pos_row/pos_col only… there is currently no way to score this
      tracker.")*
- [x] **Scale estimation — DONE 2026-08-16 via the DSST 1-D filter, not multi-resolution.**
      See the DSST entry in Completed. Danelljan §3.1's exhaustive form ("apply the learned
      filter at multiple resolutions… finding the maximum correlation score over all evaluated
      locations and scales") was NOT taken: DSST beats it on both accuracy and speed
      (Table 1), and here it would additionally perturb the shift budget every frame and spend
      the 30 fps headroom vectorization recovered. The 1-D filter runs entirely on the APU,
      touches no AIE, no PL and no calibrated constant, and converges to 0.5% in the native
      test. Bolme's log-polar suggestion (§5 conclusion) remains unexplored and would need a
      nonlinear warp roi_crop cannot do.
- [ ] **Affine perturbations for init (Bolme §3.4)** — already listed below, but note the
      papers *quantify* it: Fig. 3 puts second-frame PSR at ~4 for N=1 vs ~19 for N=8. That
      compounds with the missing PSR gating — a weak initial filter with no failure detection.
- [x] **~~Try raising `eps_rel`~~ — REFUTED 2026-08-14, 1e-3 is already the optimum; raising
      it monotonically worsens the peak/sidelobe ratio (16.15 → 10.62 → 4.01). See the
      `eps_rel` entry in Completed for the closed form `R = G·B/(B+ε)` and the sweep.**
      Original entry follows. `DEFAULT_EPS_REL = 1e-3`. Bolme Fig. 4 shows the MOSSE curve
      **flat at PSR ≈ 19 from ε=0.0001 to ~0.2**, then dropping — MOSSE is far more ε-tolerant
      than ASEF/UMACE. The existing measurement (`eps_rel` 0.001→3 moves the float accumulator
      674→23386 with `err=0 px` throughout) means this is ~an order of magnitude of free
      accumulator headroom against the quantization ceiling that forced `H_SHIFT=10`. Caveat:
      Bolme's ε is absolute on the denominator, ours is relative to mean(B), so the decades are
      not interchangeable — but the curve shape says it is cheap to try.
- [ ] **RGB features** — see the RGB section above. Blocker was a miscalculation; sequence
      after PSR gating and the DMA measurement.

#### Pipeline / validation

- [ ] **s7 at N_CHANNELS=16** — the real design point, and the case `H_SHIFT=10` was
      chosen for. Predicted accumulator 7456 (23% of range); PSR should be markedly
      better than ch1's 19.6× since channels add coherently and quantization noise
      does not. Note the harness reuses one spectrum for all channels, so it is the
      coherent best case for SNR and the coherent worst case for headroom.
      Wall clock ~5 h at 64×64 ch16 (`SIM_WALL_TIMEOUT` scales with N_CHANNELS).
- [ ] Re-run s6 at `H_SHIFT=10` — it passed at 15; unity-gain scenarios are
      H_SHIFT-independent by construction (`H_UNITY = 2^H_SHIFT`) but confirm it
- [ ] **THE ch16 RUN AT `CONV2D_MODE=0`, budget 5-3-4 — the one remaining expensive item.**
      Everything cheap that could be done first has been (runs A/B/C/D, the offline model,
      the x86sim harness). **The only thing it uniquely tests is multi-channel accumulation
      with real conv2d**; every other subsystem is now validated at ch1 or in x86sim.
      Confirm `CONV2D_ECHO_TEST=0` and `CONV_RELU=0` reach `aiecompiler.log` before starting.

      **COST: ~11.5 h PER FRAME, i.e. ~23 h for `ITER_CNT=2`. MEASURED 2026-08-16**
      (`runs/occlusion_0816_1321.log`: 172 min got through 4 of 16 channels at ch16,
      ~43 min/channel). This entry previously said "~1 h" and then "~1.7 h"; both were **ch1**
      numbers applied to a ch16 plan, wrong by ~13×. See the second frame-time correction
      under Current status.

      **Consequence: the Phase F ladder does not fit.** Four ch16 runs would be ~4 days of
      machine time. Run **ch1 first** — the box/conversion/IoU/scale-filter host code is
      compiled but has never executed, and ch1 exercises all of it at ~1.7 h instead of ~23 h.
      Then spend the single ch16 run on the one question ch1 cannot answer.

      Predicted (OLD, from the s6 patch): accum 6.2%, response 32.6%, ratio 36.8, peak (10,−7)
      exact, nothing railed — at budget 5-3-4. **That budget does not transfer to the
      post-2026-08-16 geometry**: re-swept on the synthetic textured source at target 64 /
      padding 2, 5-3-4 puts the response at **0.4%** of range while the accumulator sits at
      0.8%, so the attenuation is in the wrong place. Use **4-2-1** (response 23.9%, accum
      0.8%, PSR 45.7, peak exact, nothing railed). `IFFT_ROW_SHIFT=0` remains unsafe at 16
      channels, which rules out the 4-2-0 / 4-0-2 pair that also reach ~48%.
- [x] **Shift budget SETTLED 2026-08-14 — see the budget note under Build parameters.**
      The 99.69% figure that reopened it was echo mode; the railing that reopened it again was
      `mean_prev=0`. Validated at ch1: 5-2-2, F_ch 53, accum 70, response 2534 (7.7%), ratio
      23.0, nothing railed, response a correct σ=2 Gaussian. `H_SHIFT=10` is fine and the
      total never needed to grow.
- [x] **hw_emu with `ITER_CNT=2` DONE 2026-08-13** — frame 0 init → frame 1 track, `err=0 px`.
      Localisation flipped VOID → **OK**. See the Completed entry above. (The two-frame
      init→track *mechanism* is validated regardless of conv2d mode; only the numerics carry
      the echo-mode qualification.)
- [ ] aiesim N_CHANNELS=16: test multi-channel accumulation
- [ ] mosse_tracker.cpp: add video decode loop (OpenCV or V4L2)
- [ ] Affine perturbations for init (Bolme §3.4) — currently the N=1 case
- [x] `transpose_inplace()` implemented (`mosse_tracker.cpp:245`) — scratch-buffer transpose,
      not a stub. This item sat unchecked long after the fact; the 128×128 ch16 run depended
      on it, and localisation came back `err=0 px`, which a broken transpose would not give.
- [~] **Per-`async`/`wait` driver cost — COUNT now 1090/frame after WS=8, µs/tx still open.**
      **UPDATE 2026-08-14:** raising `FFT_ROW_WS`/`FFT_COL_WS` 2→8 cut the count
      **4258 → 1090** at ch16 (measured 388 → **100** at ch1, matching the formula exactly on
      hardware in runs B/C/D). That shrinks the *uncertainty* as well as the cost: the
      2-10 µs/tx band now spans 2.2-10.9 ms/frame instead of 8.5-42.6, so the design clears
      30 fps in both regimes rather than only the optimistic one. **The µs/tx figure itself is
      still unmeasured and still needs `TARGET=hw`** — hw_emu cannot answer it. Historical
      breakdown at WS=2 follows.
      The count was **4258 transactions per frame** at 128×128/ch16, not the ~6500 previously
      estimated (the estimate was ~50% high), and it is identical on both frames. Breakdown:
      ```
      gmio_weights 1024   gmio_fft_row_out 1024   gmio_fft_col_out 1024   gmio_accum_out 1024
      gmio_ifft_row_out 64   gmio_response 64   fft_col_in/cmul_in 16 each   ifft_row/col_in 1 each
      ```
      **96% of the traffic (4096 tx) is the four per-invocation-chunked ports at 1024 each**;
      the other six contribute 162 combined. That is the whole optimization target.
      Every count matches the formula exactly (`CONV_INVOCATIONS×16`, `ROW_CHUNKS×16`, …), which
      also validates the instrument.
      **Still open: the real per-transaction cost.** At 2 µs/tx the total is 8.5 ms/frame (26%
      of a 33 ms budget — fine); at 10 µs it is 42.6 ms (over budget). Still genuinely
      undecided, but the range narrowed by a third. **Needs `TARGET=hw`** — see the hw_emu
      timing caveat in Known Issues.
- [ ] Validate IFFT normalization shift (col_shift=12 empirically tuned; see ifft_graph.h)
- [x] **N_CHANNELS=16 validated end to end (2026-08-01)** — s6, `CONV2D_MODE=0`, budget
      `FFT_SHIFT=4 IFFT_ROW_SHIFT=2 IFFT_COL_SHIFT=2`, `OVERALL: PASS`, `err=0 px`.
      Accumulator peaks at 7728 (24% of cint16), row IFFT 8805, response 6692, `rails=0`
      everywhere. **The cint16 DDR accumulator is sufficient — no int32 widening needed.**
      Accumulation is exactly linear across channels. The aiesim harness now has a channel
      loop; it reuses one spectrum per channel, so this is the coherent worst case.
- [x] **hw_emu PLIO smoke test PASSES on the REAL stream path (2026-08-01)** —
      `SMOKE_SKIP_STREAM=0`, so `smoke_passthrough` actually calls `readincr()`.
      `[smoke] PASS — all 256 words correct`, and the waveform confirms **256/256 beats**
      at the shim boundary. This closes the "PL→AIE PLIO delivers nothing" issue: the
      PLIO mechanism works in hw_emu, the bug was the host's argument indexing.
- [x] **hw_emu single-channel end-to-end PASSES (2026-08-01)** — `N_CHANNELS=1 ITER_CNT=1`.
      The whole chain runs: `roi_crop` → PatchIn PLIO → conv2d → row FFT → transpose →
      col FFT → cmul_accum → IFFT rows → transpose → IFFT cols → response → peak detect.
      All four output GMIOs drained 64 × 1024 B. Both blocking bugs are fixed and
      validated on hardware: the XRT arg-index bug and the `aie2gm_nb` chunking/ordering
      bug. Timing reference: `roi_crop` `ap_done` at **766 µs** of simulated time; one
      frame is ~90 min wall at 128×128 with `--debug typical`.
- [x] **hw_emu localisation VALIDATED 2026-08-13** — `err=0 px` at 128×128/ch16/ITER_CNT=2,
      with the response verified as a correct σ=2 Gaussian. See the Completed entry above.
      *(conv2d ran in echo mode, which does not weaken this item: localisation and the
      response shape are properties of the FFT/filter chain. It does mean the run did not
      also validate conv2d, B1 or the window — see the echo-mode Known Issues entry.)*
      *(Was: "NOT yet validated". Frame 0 is the init pass and its response is deliberately
      not evaluated, which is why this needed `ITER_CNT=2`. Frame 0 duly came back
      `accum max|.|=0`, `response max|.|=0` — correct and expected, since the filter is zero
      by construction until `filter_init` runs at the END of frame 0.)*
      Historical description of the original defect follows.
      `H_ch*` used to be `memset` to zero
      (`mosse_tracker.cpp`, "initialize filters from first frame"), so `cmul_accum`
      multiplies by zero and the response map is identically zero. Confirmed empirically:
      `peak|re|=0`. The test now injects the impulse OFF-CENTRE at
      `pos + (IMPULSE_DR, IMPULSE_DC)` = (10,-7) and prints `[VOID/OK/MISMATCH]`, so this
      degenerate case reports honestly instead of printing a plausible position.
      Unblocked by first-frame filter init; the cheap interim is to set `H_ch*` to the
      conjugate spectrum of the same patch (matched filter → genuine peak at the offset).

### Known Issues

- **THE 2026-08-13 ch16 BASELINE RAN WITH conv2d IN ECHO MODE. Every number from it is
  qualified. Discovered 2026-08-14.**
  ```
  build/hw_emu/128x128/ch16/aie.flagstamp:  --Xpreproc="-DCONV2D_ECHO_TEST=1"
  ```
  `CONV2D_MODE ?= 1` is the Makefile default (`Makefile:192`) and nothing in a `make sd_card`
  run announces it. That build's `libadf.a` and XSA date from Aug 12 18:27-18:35 carrying the
  flag; only `mosse_tracker.elf` was rebuilt on Aug 13 21:57, so the ~28 h baseline ran that
  graph. The 128×128/**ch1** build and every 64×64 build carry `=0` — it is specifically the
  ch16 design point that has never run the real conv path.

  **What echo mode does and does not skip.** `conv2d_kernel.cpp` returns at the top of the
  function: it unpacks the int8 stream to cint16 and emits it. `roi_crop` is UNAFFECTED, so
  the FFT still saw a resampled, log-transformed, zero-mean, unit-L2 int8 patch — this is
  **not** the "raw patch through `CONV2D_MODE=0`" trap described in the shift-budget entry
  below. What is missing is the 3×3 MAC, ReLU, the Stage B1 mean subtract, and **the Hanning
  window**. The weight buffer is never read, so `bias_acc`, `out_shift` and `mean_prev` were
  all inert (the host still wrote `mean_prev` into bytes [18:22] every frame; conv2d ignored
  it).

  **Three consequences, in descending order of how much they move a conclusion:**

  1. **B2's premise does not hold without the window.** B2 corrects the residual
     *window-weighted* mean, and its 9-bin identity is a property of `DFT(w⊗w)` for the
     periodic Hann (see the Hann entry in Key design decisions). With no window applied those
     9 bins are not the window's DFT, so `max|removed|=859` measured something else. Do not
     act on that run's "reconsider `B2_NULL_BINS=0`" hint.
  2. **All 16 channels were bit-identical**, so the accumulator summed 16 perfectly coherent
     copies — the most coherent case physically possible, not merely the harness artifact
     noted for the aiesim s6 ch16 run. That inflates the response scale, and is a second
     candidate explanation for "response 5× higher than s6 predicted" alongside the
     spiky-filter argument. With 16 genuinely different conv channels the coherence, and
     therefore the response, would be lower.
  3. **The DC pedestal was absent.** Stage A already zero-means the patch and there is no
     ReLU in echo mode, so the non-negative-feature-map DC that B1 and B2 exist to fight was
     never generated. Any DC-related reading from that run describes a pipeline without the
     mechanism that produces the problem.

  **What survives unqualified:** the σ=2 Gaussian profile match — it validates the transposes,
  both IFFT passes, the conjugation convention and the Q1.15 export, all properties of the
  filter maths and the FFT chain independent of what features feed them — and the DMA
  transaction counts, which are structural and formula-verified.

  **The real conv path is not unvalidated.** aiesim s6 and s7 both PASS at `CONV2D_MODE=0`,
  and a 128×128/ch1 build exists with `=0`. It is the ch16 *combination* that is missing.

  This is precisely the failure mode the "flag-only make changes / stale `libadf.a`" entry at
  the bottom of this file warns about, and the check it prescribes is the one that catches it:
  grep `aiecompiler.log` or `aie.flagstamp` for the flag you intended, **every time**.

- **~~The response sits at 99.69% of the cint16 rail — the budget total of 12 is too small~~
  FULLY RETRACTED 2026-08-14. Both the measurement and the conclusion were wrong.**
  The measurement was echo mode (16 bit-identical passthrough channels summing coherently).
  Re-measured on the real conv path at ch1, the response sits at **7.7%** of the rail
  (run D: 2534 of 32767), and the modelled ch16 figure at 5-3-4 is **32.6%**. There was never
  a headroom crisis, and **the budget total did not need to grow** — 4-3-3 and 5-2-2 both have
  total 14 and both work. What *did* need fixing was `mean_prev=0` on frame 0, which is not a
  scaling problem at all. Original entry follows for the history of how the error propagated.
  ```
  accum     max|.| =  2621   ( 8.0% of range)   rails=0
  response max|.| = 32666   (99.69% of range)  rails=0   <-- 101 LSB of margin
  ```
  Nothing saturated, but there is no headroom. Two things this overturns:

  **The recorded aiesim s6 ch16 figures do not transfer.** Those were accum 7728 / response
  6692 — s6 uses a UNITY filter. A real MOSSE filter is spiky (max|H| lands where |F| is
  smallest), which redistributes the scale entirely: accum came in 3× *lower* and the response
  5× *higher* than s6 predicted. Never calibrate headroom on a unity-filter scenario again.

  **The budget is also badly unbalanced**: accum uses 8% while the response uses 99.69%, so
  ~3.6 bits of accumulator range sit idle while the response is one LSB from clipping.

  **Consequence for Phase 1, and it is a hard constraint:** `accum ∝ |F|` and
  `response ∝ accum`, because H is renormalized to full scale regardless of `|F|`
  (H ≈ conj(G)F/(B+ε) with B ≈ |F|² and ε relative to mean(B), so `H_q15`'s shape is invariant
  to a uniform scaling of F). Any fix that recovers conv-output bits therefore scales the
  response by the same factor — variant (a)'s ~2 bits alone would put it at ~130000, clipping
  by 4× and flattening the peak. **Redistribution cannot fix it: any budget summing to 12
  leaves the response scale unchanged. The total must GROW** — roughly 15 for variant (a),
  ~17.5 for variant (b). Sweep from `FFT_SHIFT=4 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=4`.

- **hw_emu wall-clock timings are NOT hardware timings, and the per-port split is not a driver-
  cost proxy either.**
  *(Taken from the echo-mode run, but unaffected by it: transaction counts are structural and
  the timing caveat is about the emulator, not about conv2d.)*
  The `[dma]` instrumentation reported `11188193 us/tx` and
  "144361599.5% of a 33 ms frame" — that is emulator time, and the printed NOTE says so. Less
  obvious: the *relative* split is also misleading. `gmio_accum_out` showed 72% of the frame's
  total, but that mostly reflects blocking on col-FFT + cmul **compute** latency, not DMA
  overhead. Only the transaction COUNT (4258/frame, exact and formula-verified) and the
  per-port count split are trustworthy from hw_emu. Real µs/tx needs `TARGET=hw`.

- **~~B2 is doing real work now, and `B2_NULL_BINS=1` discards it.~~ RETRACTED 2026-08-14 —
  and the real-path measurements say the opposite.** The 859 / "33% of the peak" figure was
  echo mode, where no Hanning window was applied at all, so those 9 bins were not `DFT(w⊗w)`.
  On the real conv path, `max|removed|` tracks the frame-0 DC pedestal and nothing else:
  ```
  run B (no seed)   max|removed| = 2752   accum 5264   B2 discarding the pedestal
  run C (no seed)   max|removed| = 1728   accum 2121
  run D (SEEDED)    max|removed| =    0   accum   70   nothing left to remove
  ```
  **With `mean_prev` seeded there is no low-frequency energy for B2 to null**, and the response
  is a correct σ=2 Gaussian without it doing anything. The log's standing hint to "reconsider
  `B2_NULL_BINS=0`" is therefore uninformative on the fixed pipeline — B2 is now a no-op, which
  is the desired state. Revisit only if a real video stream puts energy back in those bins.
  Frame 1 measured
  `[B2] nulled 9 low-frequency bins, max|removed|=859` against an accum peak of 2621 — the
  nulled energy is **33% of the peak**. With `B2_NULL_BINS=1` those 9 bins are *zeroed* rather
  than corrected, so genuine low-frequency target energy is thrown away. The log's own hint
  (`not railed — reconsider B2_NULL_BINS=0`) is now worth acting on: on frame 0 it was
  uninformative (accum was zero, `max|removed|=0`), but frame 1 gives it a real measurement.

- **The grayscale collapse degenerates the feature bank — 14 independent filters, not 16.**
  Measured 2026-08-12 on the shipped `layer0_weights.bin`; reproduce with
  `uv run --extra weights python3 scripts/check_collapse.py`. Correlating the 16 int8
  kernels pairwise:
  ```
  ch 0 / ch 9   cos = +0.9993
  ch 0 / ch14   cos = -0.9941
  ch 9 / ch14   cos = -0.9956
  ```
  ch0, ch9 and ch14 are the same 3×3 filter up to sign — visible by eye in the packed bytes
  (ch0 `[-107,-127,-116,-104,-126,-99,-8,-15,-8]`, ch9 `[-106,-127,-111,-101,-126,-95,-6,-8,-3]`).
  ch0 and ch9 are colour-opponent (cos(R,G) = −0.9997 and cos(R,B) = −0.98); on a colourless
  input they collapse onto ch14, an achromatic filter the bank already contained. The SVD of
  the normalized kernel matrix puts 99% of the energy in 8 of 9 dimensions.
  **Consequences:** two of sixteen serial FFT→cmul→IFFT round trips per frame compute a
  linear duplicate the DCF cannot exploit (Danelljan eq. 1 optimizes jointly over channels,
  so collinear channels are one degree of freedom between them) — ~12% of the dominant
  per-frame cost. And collinear channels add *exactly* coherently in the accumulator, so
  ch0/ch9/ch14 are a real instance of the "coherent worst case" the N_CHANNELS=16 headroom
  note assumed was only a harness artifact.
  **This is the LINEAR picture only — do not act on it yet.** Two channels with collinear
  kernels but different `bias_acc`/`out_shift` are two different *nonlinear* readouts of the
  same projection (ch0 has shift 7/bias 3160208, ch9 shift 5/bias 563624), so ReLU could in
  principle separate them. Measured post-ReLU on s6 (`check_collapse.py` Q4) it does **not** —
  ch0/ch9 correlate at +1.0000 and 28 of 120 channel pairs exceed |corr| 0.95 — but that
  measurement is **confounded with the `bias_acc` defect below**: every channel is dominated by
  the same bias-driven DC pedestal, which is exactly what makes them all look alike. The Q4
  effective rank of 5 of 13 live channels is also partly a property of s6's smooth patch, since
  a 3×3 bank on a smooth input excites few spatial modes.

  **Cheapest fix:** drop ch0 and ch9 (keep ch14) and reclaim two channel iterations, at no
  accuracy cost *if* they are still redundant. **But sequence it after the `bias_acc` fix** —
  that fix changes the DC that is driving this correlation, and it revives ch3, so re-run Q2/Q4
  and prune whatever is still degenerate rather than pruning now. The real fix is RGB — see the
  RGB section above.

- **Do NOT change the RGB→gray collapse to Danelljan's unweighted sum.** §3.3 says "set the
  R, G and B-channels equal to the grayscale intensities", which makes the paper-exact
  collapse `Σ_ic w[ic]`, and `export_weights.py:136` instead uses luminance weights. The
  deviation is deliberate and the paper-literal form is **worse here**: measured 2026-08-12,
  the unweighted sum annihilates the four colour-opponent channels (0, 2, 9, 10) to 2.5-5% of
  their per-plane norm, because those have w_R ≈ −w_G and cancel on summation. Per-channel
  int8 quantization then divides by `max|w_gray[oc]|` and amplifies the cancellation residue
  back to full ±127 scale. The other 11 channels are achromatic and agree between conventions
  to cos > 0.99 (int8 L1 diff ≤ 48 of ~1140), so the convention *only* matters where
  luminance wins. Magnitude is a non-issue either way — the LUM form is ~2.2× smaller but
  `quantize_weights()` normalizes per channel, absorbing it into `scales[oc]`.

- **PHASE 1's PLANNED FIX MAKES TRACKING ~3x WORSE. THE PROBLEM IS RELU, NOT `bias_acc`.
  Measured offline 2026-08-14 with `scripts/phase1_sweep.py` — seconds, no hardware.**

  The entry below diagnoses `bias_acc` correctly and then draws the wrong conclusion from it
  ("very likely the upstream cause of the documented PSR ceiling"). Variants (a) and (b) do
  exactly what that entry predicts — they revive ch3/ch15, cut the ReLU-never count 11→7 and
  lift resolution from 11.4 to 14.5 bits — **and tracking quality collapses anyway.**

  Held-out evaluation (filter trained on the s6 patch, applied to a circular shift of
  (10,−7); peak lands correctly in every row, so `err=0 px` cannot see any of this):

  | variant | ReLU | bits | PSR (Bolme) | peak/max-sidelobe |
  |---|---|---|---|---|
  | `base` (ships today) | on | 11.4 | 58.9 | 12.82 |
  | **`a`** (the planned fix) | on | 12.3 | **19.5** | **3.92** |
  | **`b`** | on | 14.5 | **20.5** | **3.71** |
  | `c` (zero bias, no ReLU) | off | 14.5 | 59.8 | 15.02 |
  | **`a_nr`** (a, ReLU removed) | off | 12.3 | **62.4** | **16.25** |
  | `b_nr` (b, ReLU removed) | off | 14.5 | 59.7 | 15.12 |

  **Every ReLU-off variant beats every ReLU-on variant, and the bias scale barely matters
  next to it.** The isolating pairs are the proof: `a` vs `a_nr` differ ONLY in ReLU and
  score 3.92 vs 16.25.

  **Why `base` looks decent today is an accident.** Its `bias_acc` is so oversized that ReLU
  is a no-op on 11 of 16 channels — the bank is *accidentally almost linear*. Variant (a)
  "fixes" the bias, which **switches ReLU back on** for those channels, and rectification is
  what destroys the correlation surface. The planned fix works by removing the accident that
  was protecting the design.

  This is the internal inconsistency the entry below already flags, now settled with a
  measurement: `export_weights.py` drops Hardswish arguing "no activation is needed at this
  stage — the MOSSE correlation filter is linear in feature space", and that argument is
  correct and applies to ReLU too. **A DCF is linear in the features; a half-wave rectifier
  throws away half the signal and the linear filter cannot undo it.**

  **Recommendation: `a_nr` — the ROI_NORM_Q bias correction PLUS removing ReLU — at budget
  `FFT_SHIFT=4 IFFT_ROW_SHIFT=3 IFFT_COL_SHIFT=3`.** Response lands at 32.7% of range,
  accumulator 4.7%, nothing railed, peak exact. `c` (zero bias, no ReLU) is within noise of
  it but needs `FFT_SHIFT=5` because its larger conv output rails the column FFT at 4.

  **A SECOND CONCLUSION IS OVERTURNED: the shift budget does NOT need to grow.** The
  "response at 99.69% of the rail, the total must GROW" finding came from the echo-mode run.
  With the REAL conv2d and 16 genuinely different channels, `base` at 4-2-2 puts the response
  at **19.6%** — nearly five bits of headroom, not 101 LSB. Sixteen bit-identical passthrough
  channels summing coherently is what filled the rail, not the pipeline's gain.

  **Caveats, stated plainly.** One patch (s6). "Held out" is a circular shift of that same
  patch, which is better than self-evaluation — it deflated the ReLU-off variants from ~201
  to ~60, so the check earned its keep — but it is still not a second frame. And removing
  ReLU is a real divergence from Danelljan §3.3, which uses post-activation conv features.
  Confirm on more patches before treating the absolute numbers as settled; the *ordering* is
  robust across both evaluation modes and every budget tested.

- **`bias_acc`/`out_shift` are calibrated for the wrong input scale — MEASURED 2026-08-12,
  and it costs 5.5 bits and kills 2 channels outright.**
  **[Diagnosis correct, CONCLUSION SUPERSEDED — see the ReLU entry above. Fixing the bias
  without also removing ReLU makes tracking ~3x worse. Do not implement (a) or (b) alone.]** This entry began as a hypothesis
  about "~2 bits"; the measurement is much worse. Reproduce: `scripts/check_collapse.py` Q3
  (input-independent) and Q4 (on s6's real patch).

  `export_weights.py` declares its input contract as full scale 127 ≙ 1.0
  (`x_int8 = clip(round(x_gray_float * 127))`) and derives `bias_acc = b_fold * 127/scale`
  and `ACC_MAX_THEORY = 9·127·127` from it. But `roi_crop` emits a per-patch z-score scaled by
  `ROI_NORM_Q = 32` — ±1σ at ±32, clipping at ±3.97σ. So `bias_acc` is ~4× oversized relative
  to the activations it meets.

  **Q3 verdicts are provable from `bias_acc` vs the maximum possible AC swing `127·Σ|w|`, so
  they hold for ANY int8 input at any patch size:**
  ```
  structurally dead (bias + maxAC <= 0, output identically zero always) : ch3, ch15
  ReLU provably NEVER active (bias - maxAC >= 0, so it is a no-op)      : 11 of 16 channels
  ReLU genuinely active                                                 : ch1, ch7, ch11 only
  signal resolution log2(maxAC >> out_shift)                            : 7.6 .. 13.0 of 15 bits
      worst: ch10 7.6, ch14 9.3, ch0 9.5, ch13 10.3   best: ch1/ch5/ch11 13.0
  ```
  **Why the low end is low:** `out_shift` is derived from `|bias_acc| + ACC_MAX_THEORY`, so a
  large bias shifts the *signal* down to make room for a DC pedestal that **Stage B1 subtracts
  away two stages later**. Up to 256× of attenuation to preserve a constant that is discarded.
  Q4 on s6's patch shows the result: ch0 post-ReLU mean 24687 with std 170 (DC/AC = 145),
  ch10 27871 ± 32 (DC/AC = 879, i.e. essentially a constant), and `frac>0 = 1.000` on 12 of 16
  channels — ReLU never fires.

  **This is very likely the upstream cause of the documented PSR ceiling.** The s7 notes
  attribute PSR retaining only ~51% of the float ratio to spectral quantization "carrying ~9
  bits"; that is exactly the range these channels deliver, and this defect sits upstream of
  the entire FFT chain.

  **Also a semantic mismatch, not only a scale one:** the weights were quantized against
  ImageNet-normalized *linear luminance* and they receive a per-patch z-score of the *log*
  intensity. And note the fidelity consequence — with ReLU a no-op on 11 of 16 channels, the
  "CNN features" are in practice a **linear** filter bank, not Danelljan §3.3's post-ReLU
  activations. `export_weights.py` separately argues "no activation is needed at this stage"
  as its reason for dropping Hardswish, so the design is internally inconsistent about whether
  it wants a nonlinearity at all.

  **Fix, in increasing order of blast radius:**
  (a) `bias_acc = b_fold * ROI_NORM_Q/scale` — unambiguously a contract bug, data-only change
      (the kernel reads `out_shift` from `wb[9]` at runtime, so no AIE recompile), recovers
      ~2 bits, and **revives ch3** (bias −9321 vs maxAC 25527);
  (b) additionally move `>>out_shift` after B1's mean subtraction — recovers the full 5.5 bits
      but needs an AIE kernel edit and rebuild;
  (c) drop ReLU and zero the bias — most consistent with the linearity argument, but a real
      divergence from Danelljan §3.3.
  In every case **the shift budget must be re-swept**: the signal gets *larger*, so 4-2-2 is
  no longer an upper bound. This is the fourth coupled constant with no compile-time check —
  see the preprocessing-coupling note below.

  **ORDERING: this fix is upstream of every feature-bank question.** Do not prune channels
  before it lands — (a) alone revives ch3, and the post-ReLU redundancy in Q4 is confounded
  with the DC pedestal, so the collinearity may separate once the DC is fixed.

- **`SIM_WALL_TIMEOUT` did not scale with patch area, and a timeout looks exactly like a
  deadlock.** It was `1200 × N_CHANNELS` — the 64×64 budget applied unchanged to a
  128×128 build carrying 4× the data. Measured 2026-08-05: `aiesim_plio SCENARIO=s6` at
  128×128 ch1 was killed by `timeout` (make reports `Error 124`) while still printing
  `step 2: waiting for fft_row_out`, which is byte-identical to what a genuine deadlock
  looks like. **Every `aiesim_plio` PASS recorded above before that date is a 64×64 run** —
  the 128×128 PLIO path had never been given enough wall clock to finish, so "it hangs at
  128×128" was never a real finding. Now scaled by `PATCH_ROWS*PATCH_COLS/4096`
  (`SIM_PATCH_SCALE`): 4800 s at 128×128 ch1. Before concluding "deadlock", check for
  `Error 124` and for the simulator's own cycle-timeout message.

  **Check the right process before calling something hung.** `ps -C aiesimulator` shows
  only the two bash wrapper scripts, which sit at 0.0% CPU for the whole run. The actual
  simulator is `aie2simmsm` (`aietools/bin/unwrapped/lnx64.o/`), and a healthy run holds
  ~190% CPU and ~2.5 GB RSS. A quiet log is normal — step 4/5 can go 10+ minutes between
  prints because the ISS models cross-tile vector loads at ~20k cycles each.

- **The `gmio_fft_col_out` broadcast changes the tile mapping but does NOT break the
  aiesim lock ordering.** Adding the tap moves `cmul.pi0` from a direct tile-to-tile
  buffer (`fft_cols...po0 → cmul.pi0`, tile 22,0) to a DMA path
  (`mosse_graph._dma[0].po0 → cmul.pi0`, tile 24,0), with fft_cols relocated to (29,0);
  same at both geometries. Since `cmul_accum_kernel.h` documents the ISS as delivering
  the `cmul_in` GMIO lock only after the tile-to-tile lock fires, this looked fatal.
  Measured 2026-08-05: it is not — a 64×64 s6 run with the tap in place reached **step 6**
  (row-IFFT), so the col FFT, the tap drain and cmul all completed.

- **The shift budget was calibrated at a filter gain of ONE — FIXED 2026-08-05, but read
  this before trusting any pre-2026-08 measurement.** `cmul_accum` multiplied F by H with
  no shift, justified by a comment saying "PS pre-scales H", a contract nothing
  implemented (the host `memset` the filter to zero). Every aiesim scenario passed
  `H_re=ones_re`, i.e. a literal integer 1. So `accum peak 7728` and the 4/2/2 budget
  measured what the pipeline does with *no filter at all*. At that scaling a real filter
  would have needed `|H| ≲ 4` to stay off the rails — two bits of resolution.

  H is now Q1.15 (host normalizes `max|H| → 32767` across all channels) and the kernel
  applies a round-to-nearest `>>H_SHIFT`. **The old budget survives unchanged as an upper
  bound**, because `|H|/2^15 ≤ 1` makes the product no larger than the old H=1 case — so
  4/2/2 needs no re-sweep for *safety*. The risk now runs the other way: a spiky filter
  leaves most bins far below the rails, so the host prints the Q1.15 scale and `max|H|`
  every frame, and `test_host` asserts the quantized filter actually reaches full scale.
  Note the normalization is by complex MAGNITUDE, so the largest single int16 component
  is typically below 32767 (a bin at 45° of phase puts 32767/√2 in each part).

- **H's quantization ceiling is NOT `2^H_SHIFT` — and a MOSSE filter is spiky.**
  `filter_quantize_q15()` originally derived the ceiling from `CMUL_H_SHIFT`, which
  looked right only because `(1<<15)-1 == 32767`. They are independent:
  the ceiling sets H's *resolution* (always use all 15 bits), `H_SHIFT` sets the
  *product scale*. Coupled, the knob traded one bit of filter precision for each bit
  of accumulator gain — i.e. it did nothing. Fixed in all four places that encode the
  contract: `mosse_filter.cpp`, `gen_filter_golden.py`, `test_mosse_filter.cpp`, and
  s7 in `gen_aiesim_vectors.py`.

  The reason it matters: **max|H| sits where |F| is SMALLEST**, because that is where
  the regularized inverse `conj(G)F/(|F|²+ε)` peaks. Normalizing that bin to full scale
  leaves every informative bin far below it. Measured at `H_SHIFT=15`, s7, 64×64,
  `FFT_SHIFT=3/0/6`, ch1: accumulator reached 15 of 32767 and the response was 21 LSB.
  It still localised **exactly** (`err=0 px`) — the failure was PSR collapsing to 5.2×
  against a golden 38× — which is why a location-only test would have called this a pass.
  `H_SHIFT` default is now **10**; see the calibration table in the Makefile.
  ε is a second lever (measured: `eps_rel` 0.001→3 moves the float accumulator
  674→23386 with `err=0 px` throughout) and is still at its default.

- **Conjugation: the stored filter is H, not Bolme's H*.** Bolme writes
  `H* = G ⊙ conj(F) / (F ⊙ conj(F))` because the correlation he forms is `F ⊙ H*`.
  `cmul_accum` applies the conjugation itself, so what the host stores is
  `H = conj(G) ⊙ F / (B + ε)`. Storing Bolme's expression verbatim gives
  `F ⊙ conj(H) = conj(G)·F/conj(F)`, whose phase is noise — the response then peaks at an
  arbitrary bin. **This is invisible whenever the target is centred**, because a centred
  real Gaussian has `conj(G) = G`; that is precisely why s7 puts its target off-centre,
  and the s7 generator's own assertion is what caught the mistake during implementation.

- **PSR must exclude the mainlobe or it asserts nothing.** A "largest non-peak element"
  check on a smooth σ=2 Gaussian response finds the peak's immediate neighbour at
  `exp(-1/8) = 0.88` of the peak, giving a ratio of 1.13 that passes on any blurry blob.
  Bolme §3.5 excludes an 11×11 window; with that exclusion s7's golden ratio is 28.9.
  The harness's `snr_ratio_pct` check does the exclusion with circular distance, since
  the response map wraps and a peak near an edge has its mainlobe split.

- **conv2d weights are consumed per FIRING, not per patch.** `weights` is an `input_buffer`,
  and ADF acquires every input buffer before every invocation. conv2d fires
  `PATCH_ELEMS / CONV_OUT_CHUNK` times per patch (32 at 64×64), so the driver must supply that
  many 64-byte buffers, and must start the patch flowing *first* or it deadlocks. This was the
  cause of every historical "PLIO hang". **Proper fix (not yet done): make weights an RTP /
  async parameter**, whose value persists across invocations.

- **RETRACTED — "PLIO→stream→window delivers wrong data for non-zero positions."** Disproved
  2026-07-30: with weights fed correctly, the s1 impulse's energy lands in row 17, exactly where
  it belongs. PatchIn delivers correctly in aiesim, off-centre impulses included. See
  [[feedback-aiesim-gmio]].

- **aie2gm_nb() transfers only one kernel invocation per call** (not the full N bytes) — still
  real, and **it bit for real in hw_emu on 2026-08-01**. `mosse_tracker.cpp` was arming a
  single `async()` for the whole 64 KB of `gmio_fft_row_out`. Symptoms: the design stalled
  after **6 of 64** weight buffers, the AIE DMA status register froze at `0x1a080010` for
  243k consecutive polls, and `roi_crop_0` sat at `ap_start=1, ap_done=0` forever — because
  fft_rows filled its one armed output window, blocked conv2d, and the backpressure ran all
  the way back up the PLIO. Fixed: one `async`/`wait` pair per invocation on all four output
  GMIOs, chunked by the producer's output window
  (`ROW_CHUNK_BYTES = PATCH_ROWS*FFT_ROW_WS*4`, `COL_CHUNK_BYTES = PATCH_COLS*FFT_COL_WS*4`;
  64 chunks each at 128×128).

  **The drain loops must be ORDERED correctly, not just chunked.** Two rules, both learned
  from deadlocks:
  - The row-FFT drain must **interleave** with the weights feed in one loop (they are 1:1 by
    construction — `static_assert(ROW_CHUNKS == CONV_INVOCATIONS)`). Draining after the
    weights loop is the 6-of-64 stall above; draining before it hangs waiting for data
    conv2d has not been fed.
  - `gmio_accum_out`, `gmio_ifft_row_out` and `gmio_response` must be drained **before**
    waiting on the corresponding input GMIO. Otherwise the consumer stalls on a full output
    window, which stalls the FFT feeding it, which stalls the very input DMA being waited on.

- **`v++ --package` corrupts the 2025.2 rootfs** (ext4 feature mismatch) — every hw_emu run
  panicked at boot until fixed. `make rootfs` builds a feature-downgraded copy;
  `package`/`smoke_package` depend on it. See [[vpp_package_corrupts_rootfs]].

- **hw_emu PL→AIE PLIO delivers nothing — SOLVED 2026-08-01. It was never the PLIO.**
  The PL kernel's scalar arguments were going to the wrong registers, so it emitted
  nothing and the AIE blocked forever on an empty stream.

  `xrt::kernel::operator()` assigns its parameters **positionally starting at index 0**,
  and an AXIS port still occupies an argument index even though it is not a settable
  register. `kernel.xml` for `roi_crop`:

  ```
  id=0  frame_buf (m_axi)   id=1  patch_out (AXIS)  <-- consumes a positional slot
  id=2  frame_rows          id=3  frame_cols
  id=4  roi_row             id=5  roi_col
  id=6  roi_h               id=7  roi_w
  id=8  patch_rows          id=9  patch_cols        id=10 recompute
  ```

  `crop(frame_bo, FRAME_ROWS, ...)` therefore shifted every scalar down by one:
  `patch_cols` received `(ch==0)?1:0` and `recompute` was **never written at all**.
  Result: a near-empty stream with Stage A permanently skipped — i.e. `0.00 MBps`.
  **Always set arguments on stream-bearing kernels by explicit index**
  (`xrt::run r(k); r.set_arg(i, v); r.start();`), never positionally.
  `camera_capture` is unaffected (no stream port ⇒ contiguous indices).

  Proven on `plio_smoke`, which had the same bug in miniature (`n` is index 1, `out`
  is index 0). Waveform at the shim boundary:
  `out_r_TREADY` rose at 2.688 µs and stayed high; `ap_start` at 67.485 µs;
  `ap_done` 3 clock cycles later at 67.495 µs; **`out_r_TVALID` never asserted**.
  A zero-trip loop — the AIE side was ready and waiting the whole time.
  xsim also gave `stream_src_0/out_r_TVALID` and `VitisRegion/out_r_tvalid` the SAME
  VCD identifier, proving they are one net with no FIFO in between that could have
  hidden the data. See [[plio-was-never-broken-xrt-arg-index]].

- **A centred test impulse cannot validate localisation.** `peak_detect_sw` starts at
  `max_val = -1`, so on an all-zero response only `i=0` satisfies `mag > max_val`; the
  index stays 0 and the wrap maps it to displacement `(0,0)` — the correct answer for a
  centred target, produced without reading the data. With `H_ch*` zeroed the response IS
  identically zero, so the old centred test printed a plausible position that was
  independent of its input. The impulse is now injected at `pos + (IMPULSE_DR, IMPULSE_DC)`
  (default 10,-7 — asymmetric and opposite-signed so a row/col transpose and a sign flip
  are both caught), and `peak_detect_sw` returns the peak magnitude so `peak == 0` is
  reported as `VOID` rather than as a pass. Note the detector scans `|real|`, so a
  legitimately negative peak (which Stage B1 makes possible) still classifies OK.

- **The host does not exit after the last frame.** `gr.run()` starts the graph
  free-running and `gr.end(0)` is documented as "block until graph completes", which never
  happens — the app prints every result, then sits in teardown forever. Cosmetic (all data
  is already out) but it means `run_script.sh` never reports RC and the emulation must be
  killed by hand.

- **Probing PL↔AIE signals in hw_emu takes three non-obvious steps.**
  1. `ai_engine_0.S00_AXIS` is a **SystemC/TLM socket**, not RTL — `vitis_design.protoinst`
     declares it as `"AXIS_SOCKET": "S00_AXIS_tlm_axis_socket"`. There are no TVALID/TREADY
     wires to probe there. Use the `VitisRegion/out_r_*` boundary port; its TREADY is
     driven by the TLM adapter, so it *is* the AIE-side backpressure signal.
  2. v++ generates `xelab --incr --debug off`, so `log_vcd` fails with
     `[Simulator 45-10] ... compiled without trace information` — and because
     `simulate.sh` passes `-onerror quit`, that error **aborts the emulation before Linux
     boots**. `make smoke_debug_sim` re-elaborates with `--debug typical`.
  3. `elaborate.sh` uses **relative** include paths (`../../../../prj.ip_user_files/...`)
     that resolve only in the Vivado project tree, not in the copied `package/` tree.
     Re-elaborating there fails to compile the SystemC interface *while still printing
     "Built simulation snapshot"* — a broken snapshot that looks successful. The make
     target symlinks the `prj.*` dirs and greps the log to fail loudly instead.

  Probe with `make smoke_debug_sim && make smoke_probe_emu`
  (`design/exec_scripts/plio_probe.tcl` + `scripts/analyze_plio_vcd.py`).

- **Shift budget — SETTLED 2026-08-14 at total 14 (`5-2-2` validated on hardware, `5-3-4`
  modelled for ch16). See the budget note under Build parameters.** The two reopenings below
  were both chasing the same phantom: the forward-FFT railing that drove them was the frame-0
  DC pedestal (`mean_prev=0`), not a scaling problem, and it disappeared entirely once seeded
  (F_ch 32768 railed → 53, rails 11 → 0). **Before sizing this budget against railing, check
  that `mean_prev` is seeded.** Historical entries follow.

  Was "RESOLVED at `FFT_SHIFT=4 / IFFT_ROW_SHIFT=2 / IFFT_COL_SHIFT=2`"
  (total 12). REOPENED 2026-08-13: it is not safe with a real filter.
  **Qualified 2026-08-14: the evidence for reopening is ECHO MODE.** Hardware measured the
  response at 99.69% of the rail — see the response-headroom entry at the top of Known Issues
  for the numbers and the consequence for Phase 1, and the echo-mode entry above it for why
  they need re-measuring. So the budget is **open**, on weaker evidence than this entry
  originally claimed: the 4-2-2 figures below were taken with a UNITY filter and the 99.69%
  figure was taken without conv2d. Neither describes the real design point. The
  `IFFT_ROW_SHIFT=0` warning and
  the "any total of 12 leaves the response scale unchanged" invariant below both still hold —
  the latter is precisely why redistribution cannot fix the headroom problem and the total has
  to grow. Original entry follows.

  Validated at the full design point (s6, `CONV2D_MODE=0`, N_CHANNELS=16): accum 7728,
  row IFFT 8805, response 6692, `err=0 px`, no stage above 27% of range.
  **`IFFT_ROW_SHIFT=0` is not safe at 16 channels.** The row IFFT's input is the
  *accumulated* spectrum, so with no attenuation it hit ~101000 vs 32767 under the old
  3/0/6 budget — while the accumulator and response both looked clean. The only symptoms
  were the response scaling 8.8× instead of 16× and the peak drifting 6 px. Check every
  stage, not just the endpoints; `mosse_graph.cpp` now reports `ifftrow_sat`.
  Any budget summing to 12 leaves the response scale unchanged, so weight can be moved
  between passes without recalibrating expected values (verified: 6692 vs 6688 modelled).
  **Calibrate scaling only on Stage-A-preprocessed input.** A raw patch through
  `CONV2D_MODE=0` produces a ~24.5M DC bin and appears to demand `FFT_SHIFT=6` — that is
  an artifact of bypassing Stage A, not a property of the design. The old `IFFT_COL=12`
  default was calibrated on raw/echo data and zeroes a pure-DC spectrum
  (`2731 >> 12 == 0`). Sweep table and history in [[ifft-col-shift-narrowband]].
  **Update 2026-07-31:** preprocessing changes the premise. The reason real patches were
  DC-dominated is that ReLU output is non-negative and nothing removed its mean; Stage B1
  now drops the DC/AC ratio from 18.6 to 5.4 bits. Re-run the sweep before choosing —
  the old table was measured on spectra that no longer exist.

- **Preprocessing is split across three engines, and the pieces are coupled.** Changing any
  one of them silently breaks the others:
  `hanning_*.h` must stay periodic (B2's 9-bin identity depends on it);
  `mean_prev` at weights bytes `[18:22]` is written by the host and read by conv2d — the
  layout is duplicated in `conv2d_kernel.h`, `conv2d_kernel.cpp`, `export_weights.py` and
  `gen_aiesim_vectors.py`;
  `ROI_NORM_Q` in `roi_crop.h` sets the int8 scale that conv2d's `out_shift` was derived
  against. There is no compile-time check tying these together.

  **A fifth member of this family bit on 2026-08-14 and is now fixed:** `FFT_ROW_WS` /
  `FFT_COL_WS` were passed to `AIE_FLAGS` but **not** to `GCC_FLAGS`, and
  `mosse_tracker.cpp:67-71` silently defaults them to 2. Graph and host would have disagreed
  about every DMA chunk count, deadlocking the drain loops in a way indistinguishable from
  the `aie2gm_nb` hang. **The general lesson: any constant the graph and the host both derive
  from must be passed to BOTH toolchains from one Makefile variable — a `#ifndef` default in
  the host is not a safety net, it is what makes the mismatch silent.** Audit `GCC_FLAGS`
  against `AIE_FLAGS` when adding any new shared parameter.

- **`FFT_SHIFT=0` overflows on a realistic patch — MEASURED 2026-07-31.** s6 (the first
  scenario fed a real Stage-A-preprocessed patch through conv2d) saturates the cint16
  accumulator at **channel 0 of 1**: row-FFT output peaks at 2575, then the column FFT sums
  64 rows of that and reaches `max|.|=52869` against a 32767 rail, with 2 elements pinned.
  The peak lands 15 px off as a result. `gen_aiesim_vectors.py` already predicted this in
  its shift-budget comment ("real conv2d output overflows the cint16 FFT at FFT_SHIFT=0"),
  but no scenario had ever exercised it — s0–s4 are raw impulses/constants run in echo mode.
  Move the budget onto the forward pass
  (`FFT_SHIFT=3 IFFT_COL_SHIFT=6` keeps `2*FFT_SHIFT + IFFT_ROW + IFFT_COL = 12`, so no
  expected value needs recalibrating). **This is a separate problem from the N_CHANNELS=16
  headroom question** — it saturates with a single channel.
  **CONFIRMED FIXED:** at `FFT_SHIFT=3 IFFT_COL_SHIFT=6`, s6 gives `max|accum|=1929`,
  `rails=0`, and localises **exactly** (`err=0 px`) through the full
  PatchIn→conv2d→B1→FFT→cmul→IFFT path. `FFT_SHIFT=0` is not a safe default for real data;
  treat 3/0/6 as the working budget until the N_CHANNELS=16 sweep picks a final one.

- **The correlation response is SIGNED once Stage B1 is active.** ReLU used to leave the
  feature map non-negative, so every scenario could assume a positive peak
  (`peak_re_lo=1`). Subtracting the mean makes the map bipolar, and the largest-magnitude
  point is as likely to be a trough as a crest — s6's peak is `{-417,0}`. `peak_detect_sw`
  in `mosse_tracker.cpp` therefore now scans `|real|` rather than `real` — a signed max
  would walk past a negative peak and return the largest positive sidelobe instead. s6's
  generator derives its sign bounds from the golden model, so the scenario asserts the sign
  rather than assuming it.

  **Caveat noted 2026-08-12 — `|real|` is a deviation from both papers and may be masking
  something.** Bolme §3.5 takes `g_max` as a plain maximum, and the desired output `g` is a
  *positive* Gaussian by construction, so a correctly trained filter should produce a
  positive peak; a negative peak means anti-correlation, which is a mismatch rather than a
  detection. The `|real|` scan was adopted because the degenerate zero-filter era made signed
  maxima useless, and it is defensible while H is still being calibrated — but it will happily
  report a confident position for a filter of the wrong sign. Once PSR gating lands, revisit:
  prefer a signed max, and treat "the largest magnitude is negative" as a failure signal.

  **DISCHARGED 2026-08-15, the second half of it.** "The largest magnitude is negative" is now
  `GateReason::NegativePeak` and vetoes the update — see the PSR gating entry above. The scan
  itself is still `|real|`, deliberately: it is what locates the peak the tracker acts on, and
  BOTH peak definitions are computed every frame (`compute_psr(use_abs=true/false)`) so a
  disagreement between them is reported rather than hidden. Switching the scan to a signed max
  is now unnecessary — a wrong-sign filter is caught by the gate instead of being silently
  followed. The disagreement is deliberately NOT a gate condition: it would double-count
  `NegativePeak` while adding a veto with no support in either paper.

- **B2's frequency correction is NOT bit-exact.** The linearity argument (`DFT(w*(f-µ)) =
  DFT(w*f) - µ*DFT(w)`) is exact in real arithmetic, but conv2d's window multiply applies two
  `>>15` truncations, which are nonlinear. Measured relative error after correction is ~1e-3,
  versus 2.5e-2 .. 9.9 without. Good enough for argmax; do not rely on it for anything
  requiring exactness.

- **DSPLib's cint16 FFT loss is additive, not a gain factor.** Each pass subtracts ~21 from a
  summed DC bin, independent of amplitude (measured at 64×64: ideal 64→43 and 448→427; ideal
  2752→2731 and 27328→27307). So `row_dc = PATCH_COLS*c - 21`, `accum0 = PATCH_ROWS*row_dc - 21`.
  Any "expected = N" calculation is wrong. The loss scales with how much *summation* the input
  causes — an impulse loses only ~3, since its DC bin has one non-zero term.
  (A "2/3 gain" fits the const=1 point by coincidence and is refuted at const=7 — don't fit a
  scaling law to one data point.)

- **Test vectors sat below the fixed-point floor.** s1 used an impulse of amplitude 1, which
  quantizes to nothing (20/4096 bins non-zero after the row FFT). Now `GEN_IMPULSE_AMP=100`.
  s0/s2/s3/s4 are still amplitude-1. See [[aiesim-quantization-floor]].

- **Flag-only make changes used to reuse a stale `libadf.a`** and produce convincing false
  results; `%.flagstamp` prerequisites now force a rebuild. Still worth confirming
  `aiecompiler.log` carries the flag you intended. See [[feedback-verify-the-build-ran]].
  **The smoke variant of this guard was itself broken until 2026-08-01**:
  `SMOKE_FLAGS_STAMP := $(SMOKE_BUILD)/aie.flagstamp` sat ~160 lines *above* where
  `SMOKE_BUILD` and `SMOKE_AIE_FLAGS` are defined, and `:=` expands immediately, so it
  resolved to `/aie.flagstamp` with an empty flag list. The recipe died with
  `Permission denied` at the filesystem root and `SMOKE_SKIP_STREAM` never armed.
  Order matters for `:=` — verify the flag reaches `aiecompiler.log`, every time.
