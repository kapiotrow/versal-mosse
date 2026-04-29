# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MOSSE correlation filter tracker with CNN features on Versal VEK280.
Extends the AIE 2D-FFT tutorial (XD073) with a full object tracking pipeline.

## Environment setup

```bash
source sample_env_setup.sh
```

Required env vars (same as tutorial):
- `PLATFORM_REPO_PATHS`, `XILINX_VITIS`, `COMMON_IMAGE_VERSAL`
- `DSPLIB_VITIS` — set to Vitis Libraries **root** (e.g. `.../Vitis_Libraries`), NOT the dsp subdirectory. The Makefile appends `/dsp` internally.
- `PLATFORM` — set to VEK280 XPFM path

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
  conv2d_kernel      : int8 patch stream + weights → cint16 feature stream (stub: pass-through cast)
  fft2d (FFT2D_graph): PATCH_COLS-pt row FFT → GMIO → DDR; APU transposes; DDR → GMIO → PATCH_ROWS-pt col FFT
  cmul_accum_kernel  : col-FFT stream ⊙ H_ch* + accumulate (stub: pass-through)
  ifft2d (IFFT2D_graph): same DDR-transpose pattern as fft2d; PATCH_COLS-pt row IFFT + PATCH_ROWS-pt col IFFT
```

### PLIO (1 port)

`PatchIn` — roi_crop PL kernel → conv2d AIE kernel (128-bit, int8 stream).

This name must match between `mosse_graph.h` (`input_plio::create("PatchIn", ...)`) and
`mosse_x1.cfg` (`stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`).

### GMIO ports (10 total: 6 input + 4 output)

| Name | Dir | Purpose |
|---|---|---|
| `gmio_weights` | DDR→AIE | conv2d INT8 weights per channel |
| `gmio_fft_row_out` | AIE→DDR | fft_rows output; APU transposes |
| `gmio_fft_col_in` | DDR→AIE | APU-transposed data → fft_cols |
| `gmio_filter` | DDR→AIE | H_ch* per channel → cmul_accum |
| `gmio_accum_in` | DDR→AIE | Previous partial sum (skipped on ch=0) |
| `gmio_accum_out` | AIE→DDR | Updated partial sum |
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

- **Conv layer weights**: `conv2d_kernel.cpp` has pass-through stub. Replace with:
  - Pretrained INT8 weights exported from Brevitas (simplest path), or
  - FINN-generated RTL kernel (Versal support limited as of 2025.2).

## FINN / Brevitas notes

FINN's Versal support is limited as of 2025.2. Recommended approach:
1. Train and quantize the conv layer in Brevitas (PyTorch QAT).
2. Export weights as INT8 numpy arrays.
3. Hand-implement the conv in the AIE `conv2d_kernel.cpp` using those weights.
4. Monitor FINN Versal support for future migration to generated IP.

## Build commands

```bash
make graph                         # compile AIE graph only
make aiesim                        # run AIE simulator (uses aiesim_data/patch_in.txt)
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
│   ├── fft_graph.h            # FFT2D_graph (single instance, GMIO-broken row→col)
│   ├── ifft_graph.h           # IFFT2D_graph (single instance, same pattern)
│   ├── conv2d_kernel.h/.cpp   # int8 patch → cint16 feature (stub: cast only)
│   ├── cmul_accum_kernel.h/.cpp # col-FFT ⊙ H_ch* + accumulate (stub: pass-through)
│   ├── mosse_graph.h          # Top-level: PLIO + 10 GMIO + 2 custom kernels + FFT2D + IFFT2D
│   ├── mosse_graph.cpp        # Instantiation + aiesim smoke test main()
│   ├── constraints.aiecst     # PatchIn PLIO shim placement
│   └── aiesim_data/
│       └── patch_in.txt       # 16384 zeroed int8 samples for aiesim smoke test
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

## Current status / TODOs

- [ ] `conv2d_kernel.cpp`: implement sliding-window 3×3 MAC + Hanning window + load weights
- [ ] `cmul_accum_kernel.cpp`: implement element-wise cmul_conj + accumulate
- [ ] `mosse_tracker.cpp`: add video decode loop (OpenCV or V4L2)
- [ ] `mosse_tracker.cpp`: implement first-frame filter initialization
- [ ] `mosse_tracker.cpp`: implement PS-side filter update (KissFFT for A_ch, B, H_ch*)
- [ ] `mosse_tracker.cpp`: implement `transpose_inplace()` (currently stub)
- [ ] Validate IFFT normalization shift (see ifft_graph.h R7 note)
- [ ] Validate cmul_accum fixed-point precision (cint16 accumulator overflow risk for N_CHANNELS=16)
- [ ] aiesim: verify smoke test passes without deadlock (`make aiesim N_CHANNELS=1 ITER_CNT=1`)
- [ ] hw_emu: verify single-channel end-to-end (`make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1`)
