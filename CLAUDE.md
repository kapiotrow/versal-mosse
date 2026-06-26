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
| `ITER_CNT` | `1` | Frames to process in hw_emu |
| `PL_FREQ` | `312.5` | PL kernel frequency in MHz |

Build artifacts land under `build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)/`.

## Architecture overview

```
PS (A72) — mosse_tracker.cpp
  Drives all GMIO ports in the per-frame, per-channel loop.
  Runs peak_detect_sw() and filter_update_kissfft() (stubs).

PL kernels (2 total)
  camera_capture : zero-fill DDR frame buffer (stub; TODO: MIPI RX)
  roi_crop       : DDR frame → PATCH_ROWS×PATCH_COLS patch → 128-bit AXIS → AIE PatchIn

AIE — single instances, serial per-channel processing
  conv2d_kernel      : int8 patch → 3×3 MAC + ReLU + Hanning window → cint16 feature stream
                       (MobileNet-v3 Small layer 1, INT8-quantized, RGB collapsed to grayscale)
  fft2d (FFT2D_graph): PATCH_COLS-pt row FFT → GMIO → DDR; APU transposes; DDR → GMIO → PATCH_ROWS-pt col FFT
  cmul_accum_kernel  : col-FFT stream ⊙ H_ch* + accumulate (implemented with int32 accumulator)
  ifft2d (IFFT2D_graph): same DDR-transpose pattern as fft2d; PATCH_COLS-pt row IFFT + PATCH_ROWS-pt col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop PL kernel → conv2d AIE kernel (128-bit, int8 stream).

This name must match between `mosse_graph.h` (`input_plio::create("PatchIn", ...)`) and
`mosse_x1.cfg` (`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

### GMIO ports (9 total: 5 input + 4 output)

| Name | Dir | Purpose |
|---|---|---|
| `gmio_weights` | DDR→AIE | conv2d INT8 weights per channel |
| `gmio_fft_row_out` | AIE→DDR | fft_rows output; APU transposes |
| `gmio_fft_col_in` | DDR→AIE | APU-transposed data → fft_cols |
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

- **Filter update on PS**: A_ch[], B[], H_ch* computation runs on the A72 using KissFFT.
  Move to AIE if PS becomes a bottleneck.

- **IFFT normalization**: Row IFFT shift = 0; col IFFT shift = 14 (= log2(128)+log2(128)).
  If aiesim response is 2^14× too large, set col shift to 0 and apply >>14 in APU after
  reading gmio_response.

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
make aiesim                        # run AIE simulator (cycle-approx ISS with GMIO/PLIO workarounds)
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
│   ├── cmul_accum_kernel.h/.cpp # col-FFT ⊙ H_ch* + accumulate (int32 accumulator)
│   ├── mosse_graph.h          # Top-level: PLIO + 10 GMIO + 2 custom kernels + FFT2D + IFFT2D
│   ├── mosse_graph.cpp        # Instantiation + aiesim smoke test main()
│   ├── constraints.aiecst     # PatchIn PLIO shim placement
│   ├── hanning_128.h          # Precomputed Q1.15 Hanning window (auto-generated)
│   ├── weights/               # MobileNet-v3 Small layer 1 (INT8, auto-generated)
│   │   ├── layer0_weights.bin # 16 channels × 64 B
│   │   ├── layer0.h           # Shift/scale metadata per channel
│   │   └── layer0_meta.npz    # Validation data
│   └── aiesim_data/
│       └── s*/                # Test scenarios: s0, s1a, s1b, etc. (impulse at various positions)
├── pl_src/
│   ├── camera_capture/        # Zero-fill frame buffer stub
│   └── roi_crop/              # Extract patch, stream to PatchIn PLIO
├── host_app_src/
│   └── mosse_tracker.cpp      # GMIO-driven XRT tracking loop
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
- [x] `cmul_accum_kernel.cpp`: element-wise cmul_conj + int32 accumulate
- [x] aiesim N_CHANNELS=1: PASS (real arithmetic with ISS workarounds for GMIO/PLIO bugs)
- [x] Two-channel aiesim support added (N_CHANNELS=2 option in sim)
- [x] Pre-computed FFT bypass for aiesim (Bug 2/3: PLIO→stream→window delivery broken in cycle-approx ISS)

### In Progress / Next
- [ ] aiesim N_CHANNELS=16: test multi-channel accumulation
- [ ] mosse_tracker.cpp: add video decode loop (OpenCV or V4L2)
- [ ] mosse_tracker.cpp: implement first-frame filter initialization
- [ ] mosse_tracker.cpp: implement PS-side filter update (KissFFT for A_ch, B, H_ch*)
- [ ] mosse_tracker.cpp: implement `transpose_inplace()` (currently stub)
- [ ] Validate IFFT normalization shift (col_shift=12 empirically tuned; see ifft_graph.h)
- [ ] Validate cmul_accum fixed-point precision for N_CHANNELS=16 accumulation
- [ ] hw_emu: verify single-channel end-to-end (`make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1`)

### Known Issues
- **Cycle-approx aiesim GMIO/PLIO bugs** (Vitis 2025.2): See [[feedback-aiesim-gmio]]
  - aie2gm_nb() transfers only one kernel invocation per call (not full N bytes)
  - PLIO→stream→window adapter delivers wrong data for non-zero positions
  - **Workaround**: Loop per-invocation on output GMIO, pre-compute FFT outputs for aiesim
