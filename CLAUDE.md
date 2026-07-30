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
  cmul_accum_kernel  : col-FFT stream ⊙ H_ch* + accumulate (int32 intermediates,
                       cint16 accumulator in DDR — saturating; see headroom note)
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

### In Progress / Next
- [ ] aiesim N_CHANNELS=16: test multi-channel accumulation
- [ ] mosse_tracker.cpp: add video decode loop (OpenCV or V4L2)
- [ ] mosse_tracker.cpp: implement first-frame filter initialization
- [ ] mosse_tracker.cpp: implement PS-side filter update (KissFFT for A_ch, B, H_ch*)
- [ ] mosse_tracker.cpp: implement `transpose_inplace()` (currently stub)
- [ ] Validate IFFT normalization shift (col_shift=12 empirically tuned; see ifft_graph.h)
- [ ] Validate cmul_accum headroom for N_CHANNELS=16. The DDR accumulator is **cint16**,
      so `16 × per-channel magnitude` must fit 32767 or the sum clips. Saturation now makes
      that benign rather than catastrophic, but not absent — widening the accumulator to
      int32 (GMIO sizes + host) is the fix if real magnitudes don't fit.
      Note: the aiesim harness has **no channel loop** — 16-ch testing needs one added.
- [ ] hw_emu: verify single-channel end-to-end (`make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1`)

### Known Issues

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
  real. Loop one `aie2gm_nb`/`wait` pair per invocation on every output GMIO.

- **`v++ --package` corrupts the 2025.2 rootfs** (ext4 feature mismatch) — every hw_emu run
  panicked at boot until fixed. `make rootfs` builds a feature-downgraded copy;
  `package`/`smoke_package` depend on it. See [[vpp_package_corrupts_rootfs]].

- **hw_emu PL→AIE PLIO delivers nothing — OPEN.** The weights-free smoke graph reports
  `plio | S00_AXIS | IN | 0.00 MBps` and hangs, while the same graph with the stream read
  removed passes. Since aiesim proves the AIE-side construct is sound, the fault is in the
  PL→shim path. Unresolved.

- **IFFT col shift = 12 destroys narrowband spectra — OPEN.** It was calibrated so a
  broadband (impulse) spectrum round-trips to 1, which assumes ~64× summation gain in the IFFT.
  A DC-concentrated spectrum gets no such gain: s2's response came back **identically zero**
  (`2731 >> 12 == 0`) while s1 works fine. **Real image patches are DC-dominated**, so this may
  affect actual tracking. Settle before trusting hardware results.
  See [[ifft-col-shift-narrowband]].

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
