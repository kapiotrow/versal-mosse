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
PS (A72)
  video frame → patch extraction → filter update (A_ch, B, H_ch*)

PL kernels
  conv_layer   : RGB patch → INT8 feature maps (N_CHANNELS channels)
  fmt_adapter  : INT8 feature map + Hanning window → cint16 stream → AIE
  cmul_accum   : transpose buffer + H_ch* multiply + Σ accumulate + IFFT transpose
  peak_detect  : argmax on response map → (disp_row, disp_col)

AIE
  fft2d[0..N_CHANNELS-1]  : PATCH_COLS-point row FFT + PATCH_ROWS-point col FFT
  ifft2d                   : PATCH_COLS-point row IFFT + PATCH_ROWS-point col IFFT
```

### PLIO naming

Forward FFT channel `ch` (0-based): `FFTRowIn{ch}`, `FFTRowOut{ch}`, `FFTColIn{ch}`, `FFTColOut{ch}`
Inverse FFT (single instance): `IFFTRowIn0`, `IFFTRowOut0`, `IFFTColIn0`, `IFFTColOut0`

These names must match between `fft_graph.h` / `ifft_graph.h` and `mosse_x1.cfg` stream_connect entries.

### cmul_accum data flow (per tracking frame)

For channels 0 … N_CHANNELS-1 (serial):
1. Drain AIE row-FFT output → DDR `transpose_buf` (row-major store, col-major read-back)
2. Re-stream columns → AIE col-FFT input
3. Read col-FFT output, multiply by conjugate filter `H_ch*`, accumulate into `accum_buf`

On the last channel only: flush `accum_buf` → IFFT row input; handle IFFT transpose the same way.

### aiesim iteration count

`mosse_graph.cpp` runs the graph for `ITER_CNT * PATCH_ROWS` iterations (one iteration processes two rows: `FFT_ROW_WS=2`).

## Key design decisions

- **Serial channel processing**: N_CHANNELS FFT2D instances are driven one at a time
  by fmt_adapter + cmul_accum. This minimises PLIO count at the cost of throughput.
  Increasing parallelism later requires adding PLIO ports and updating mosse_x1.cfg.

- **PATCH_ROWS == PATCH_COLS**: Both row and column FFT use the same point size.
  This differs from the tutorial which always had MAT_ROWS = MAT_COLS/2.

- **Transpose in DDR**: The row-FFT output is transposed by cmul_accum via a DDR
  scratch buffer (`transpose_buf`, 64 KB for 128×128 cint16). Consider binding to
  URAM for lower latency once placement is confirmed.

- **Filter update on PS**: A_ch[], B[], H_ch* computation runs on the A72 using a
  software FFT library (KissFFT recommended). Move to PL if PS becomes a bottleneck.

- **Conv layer weights**: `conv_layer.cpp` has placeholder zero weights. Replace with:
  - Pretrained INT8 weights exported from Brevitas (simplest path), or
  - FINN-generated RTL kernel (see notes below on FINN/Versal compatibility).

## FINN / Brevitas notes

FINN's Versal support is limited as of 2025.2. Recommended approach:
1. Train and quantize the conv layer in Brevitas (PyTorch QAT).
2. Export weights as INT8 numpy arrays.
3. Hand-implement the conv in HLS (`conv_layer.cpp`) using those weights.
4. Monitor FINN Versal support for future migration to generated IP.

Alternatively, the FINN-generated IP can target the PL fabric only (treating
VEK280 PL as UltraScale+ class) and be packaged as an RTL kernel via `package_xo`.

## Build commands

```bash
make graph                         # compile AIE graph only
make aiesim                        # run AIE simulator
make kernels                       # compile all PL HLS kernels
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
│   ├── fft_graph.h       # FFT2D graph (N_CHANNELS instances, forward FFT)
│   ├── ifft_graph.h      # IFFT2D graph (single instance, inverse FFT)
│   ├── mosse_graph.h     # Top-level: FFT2D[] + IFFT2D
│   ├── mosse_graph.cpp   # Instantiation + aiesim main
│   └── constraints.aiecst
├── pl_src/
│   ├── conv_layer/       # Quantized 3×3 conv, INT8 weights
│   ├── fmt_adapter/      # INT8→cint16, Hanning window, → AIE
│   ├── cmul_accum/       # Transpose + H_ch* multiply + accumulate + IFFT feed
│   └── peak_detect/      # argmax → displacement output
├── host_app_src/
│   └── mosse_tracker.cpp # XRT tracking loop, filter update
├── system_configs/
│   └── mosse_x1.cfg      # v++ linker: kernel instances + stream wiring
├── profiling_configs/    # xrt.ini (trace settings)
├── directives/           # post_sys_link.tcl (AIE clock = 1 GHz)
└── exec_scripts/         # run_script.sh (board execution — ELF name is stale, update to mosse_tracker.elf)
```

## Current status / TODOs

- [ ] `conv_layer.cpp`: implement sliding-window 3×3 MAC + fill weights
- [ ] `mosse_tracker.cpp`: add video decode loop (OpenCV or V4L2)
- [ ] `mosse_tracker.cpp`: implement first-frame filter initialization
- [ ] `mosse_tracker.cpp`: implement PS-side filter update (KissFFT)
- [ ] `mosse_x1.cfg`: expand stream_connect for all N_CHANNELS PLIO ports (currently only ch=0)
- [ ] Prepare aiesim test vectors under `design/aie_src/aiesim_data/`
- [ ] Validate IFFT normalization shift (`FFT_2D_TP_IFFT_SHIFT` in `ifft_graph.h`; should be `log2(PATCH_COLS) + log2(PATCH_ROWS)`)
- [ ] Validate cmul_accum fixed-point shift in `cmul_cint16()`
- [ ] Bind `transpose_buf` to URAM in `cmul_accum.cpp`
- [ ] Update `exec_scripts/run_script.sh` — currently references old ELF name `fft_2d_aie_xrt.elf`
