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
| `H_SHIFT` | `15` | cmul_accum filter-product shift; H is Q1.15. Independent of the FFT budget |

Build artifacts land under `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

## Architecture overview

```
PS (A72) — mosse_tracker.cpp
  Drives all GMIO ports in the per-frame, per-channel loop.
  Runs peak_detect_sw() and filter_update_kissfft() (stubs).

PL kernels (2 total)
  camera_capture : zero-fill DDR frame buffer (stub; TODO: MIPI RX)
  roi_crop       : DDR frame → MOSSE preprocessing → 32-bit AXIS (int8) → AIE PatchIn
                   Stage A (Bolme §3.1 + Danelljan §3.3): bilinear resample of an
                   arbitrary roi_h×roi_w to the fixed patch size with border clamping,
                   log transform, zero mean, unit L2 norm × ROI_NORM_Q, int8 quantize.
                   Two passes (mean/norm are global reductions) + a stream-out pass.
                   `recompute=1` on channel 0 only; channels 1..15 re-stream the cache.

AIE — single instances, serial per-channel processing
  conv2d_kernel      : int8 patch → 3×3 MAC + ReLU + Hanning window → cint16 feature stream
                       (MobileNet-v3 Small layer 1, INT8-quantized, RGB collapsed to grayscale)
  fft2d (FFT2D_graph): PATCH_COLS-pt row FFT → GMIO → DDR; APU transposes; DDR → GMIO → PATCH_ROWS-pt col FFT
  cmul_accum_kernel  : col-FFT stream ⊙ H_ch* + accumulate (int32 intermediates,
                       cint16 accumulator in DDR — saturating; see headroom note)
  ifft2d (IFFT2D_graph): same DDR-transpose pattern as fft2d; PATCH_COLS-pt row IFFT + PATCH_ROWS-pt col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop PL kernel → conv2d AIE kernel (128-bit, int8 stream).

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
  APU: filter_update_kissfft() (stub)
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

## Weight export (MobileNetV3-Small layer 1)

Export and quantize via `make weights`:
1. Extracts first conv layer of torchvision.models.mobilenet_v3_small (pretrained)
2. Folds BatchNorm into weights/bias
3. Collapses RGB → grayscale using ITU-R BT.601 luminance: 0.2989×R + 0.5870×G + 0.1140×B
4. Symmetric INT8 quantization per output channel
5. Outputs:
   - `design/aie_src/weights/layer0_weights.bin` — 16 × 64 B (16 channels)
   - `design/aie_src/weights/layer0.h` — shift/scale metadata
   - `design/aie_src/hanning_128.h` — precomputed Q1.15 Hanning window

## Build commands

```bash
make weights                       # export MobileNet-v3 Small layer 1 (INT8 weights + hanning table)
make gen_vectors                   # generate aiesim test vectors
make graph                         # compile AIE graph only
make test_host                     # native unit test for the filter math (seconds, no hardware)
make aiesim                        # run AIE simulator — NOTE: bypasses PatchIn→conv2d→row-FFT
make aiesim_plio                   # same, but deletes fft_col_in.bin to force the REAL PatchIn path
make aiesim_plio CONV2D_MODE=2     # bisect: conv2d synthesizes output, never reads the stream
make rootfs                        # feature-downgraded rootfs copy (v++ corrupts the pristine one)
make kernels                       # compile camera_capture + roi_crop PL kernels
make xsa                           # link kernels + graph → XSA file
make application                   # cross-compile host ELF (aarch64)
make sd_card                       # full build: kernels → graph → xsa → application → package
make sd_card TARGET=hw             # hardware build
make run_emu LAUNCH_HW_EMU_EXEC=1  # launch hw emulation
make cleanall
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

## Current status (as of 2026-06-17)

### Completed
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
      Stage B2 on the APU (9-bin frequency correction), Stage B3 hook ready in
      `filter_update_kissfft`. `hanning_*.h` switched to the periodic Hann.
      Cost: 44 DSP / 10 BRAM18 / 7694 LUT in PL, ~472 µs/frame (1.4% of a 33 ms budget),
      no new AIE tiles.
- [x] roi_crop verified bit-exact against a NumPy reference in native C simulation
      (6 cases: 1:1, 2× up, 2× down, both edge-clamp paths, whole frame)
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
      regression and sits well above Bolme's §3.5 failure indicator of ~7.

### In Progress / Next
- [ ] **s7 at N_CHANNELS=16** — the real design point, and the case `H_SHIFT=10` was
      chosen for. Predicted accumulator 7456 (23% of range); PSR should be markedly
      better than ch1's 19.6× since channels add coherently and quantization noise
      does not. Note the harness reuses one spectrum for all channels, so it is the
      coherent best case for SNR and the coherent worst case for headroom.
      Wall clock ~5 h at 64×64 ch16 (`SIM_WALL_TIMEOUT` scales with N_CHANNELS).
- [ ] Re-run s6 at `H_SHIFT=10` — it passed at 15; unity-gain scenarios are
      H_SHIFT-independent by construction (`H_UNITY = 2^H_SHIFT`) but confirm it
- [ ] Re-validate the shift budget at the 128×128 / 4-2-2 design point. `H_SHIFT=10`
      was calibrated at 64×64 / 3-0-6 / ch1. The response scale transfers (same total
      of 12) and `IFFT_ROW_SHIFT=2` makes the row IFFT 4× safer, but that is an
      inference, not a measurement.
- [ ] **hw_emu with `ITER_CNT=2`** — the only test that exercises the tracker's runtime
      behaviour (frame 0 init → frame 1 track). Everything in `mosse_tracker.cpp` is
      compile-verified only. Needs a full xsa + package; hours.
- [ ] hw_emu end-to-end with `ITER_CNT=2` to confirm localisation flips VOID → OK
- [ ] aiesim N_CHANNELS=16: test multi-channel accumulation
- [ ] mosse_tracker.cpp: add video decode loop (OpenCV or V4L2)
- [ ] Affine perturbations for init (Bolme §3.4) — currently the N=1 case
- [ ] mosse_tracker.cpp: implement `transpose_inplace()` (currently stub)
- [ ] **Measure real per-`async`/`wait` driver cost on hardware.** ~6500 host DMA
      transactions per frame at 16 channels; at 2–10 µs each that is 13–65 ms, which
      would dominate the 33 ms budget and outweigh every compute-placement decision.
      This measurement gates whether the transposes should move off the APU.
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
- [ ] **hw_emu localisation is NOT yet validated.** *(Updated 2026-08-05: the cause below
      is fixed — the filter is now computed — but the hw_emu run confirming it has not
      been done. Frame 0 is the init pass and its response is deliberately not evaluated,
      so this needs `ITER_CNT=2`, ~3 h.)* Historical description follows.
      `H_ch*` used to be `memset` to zero
      (`mosse_tracker.cpp`, "initialize filters from first frame"), so `cmul_accum`
      multiplies by zero and the response map is identically zero. Confirmed empirically:
      `peak|re|=0`. The test now injects the impulse OFF-CENTRE at
      `pos + (IMPULSE_DR, IMPULSE_DC)` = (10,-7) and prints `[VOID/OK/MISMATCH]`, so this
      degenerate case reports honestly instead of printing a plausible position.
      Unblocked by first-frame filter init; the cheap interim is to set `H_ch*` to the
      conjugate spectrum of the same patch (matched filter → genuine peak at the offset).

### Known Issues

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

- **Shift budget — RESOLVED at `FFT_SHIFT=4 / IFFT_ROW_SHIFT=2 / IFFT_COL_SHIFT=2`** (total 12).
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
