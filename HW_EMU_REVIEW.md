# Hardware Emulation (hw_emu) Setup Review

**Date:** 2026-06-17  
**Status:** Mostly ready with 2 issues to fix

---

## ✅ Components Verified & Working

### 1. Environment Setup (`setup_env.sh`)
- ✅ PLATFORM_REPO_PATHS configured correctly
- ✅ XILINX_VITIS pointing to 2025.2 installation
- ✅ COMMON_IMAGE_VERSAL path set
- ✅ DSPLIB_VITIS pointing to Vitis_Libraries
- ✅ PLATFORM path to VEK280 XPFM correct
- ✅ SDKTARGETSYSROOT now properly set (after fix)
- ✅ CXX cross-compiler properly configured (after fix)

### 2. Build Infrastructure (Makefile)
- ✅ Default TARGET=hw_emu set
- ✅ All build phases defined: kernels → graph → xsa → application → package
- ✅ Correct build directory structure: `build/hw_emu/128x128/ch16/`
- ✅ AIE compiler flags complete (--target=hw for hw_emu)
- ✅ v++ compiler flags include -t $(TARGET) for hw_emu
- ✅ Dependencies correctly ordered: `sd_card: kernels graph xsa application package`

### 3. PL Kernels (camera_capture, roi_crop)
- ✅ Both HLS kernels with proper HLS pragmas
- ✅ camera_capture: m_axi interface for DDR, AXI-Lite control
- ✅ roi_crop: AXIS stream output (128-bit plio_128_bits) to PLIO
- ✅ Both use II=1 pipelining for hw_emu compatibility
- ✅ Correct bundle names (control, gmem0)

### 4. AIE Graph (mosse_graph.h)
- ✅ PLIO defined: `input_plio::create("PatchIn", plio_128_bits, "patch_in.txt")`
- ✅ All 9 GMIO ports correctly defined (6 input + 3 output = 9; note: 10 total in CLAUDE.md)
  - gmio_weights (input)
  - gmio_fft_row_out (output)
  - gmio_fft_col_in (input)
  - gmio_cmul_in (input) — carries [H_ch* | accum] packed
  - gmio_accum_out (output)
  - gmio_ifft_row_in (input)
  - gmio_ifft_row_out (output)
  - gmio_ifft_col_in (input)
  - gmio_response (output)
- ✅ Stream connections properly declared with adf::connect<>()
- ✅ Window dimensions set with adf::dimensions()
- ✅ GMIO bandwidth and burst length configured (64 bytes, 1000 MB/s estimate)
- ✅ All kernel connections validated

### 5. System Configuration (mosse_x1.cfg)
- ✅ Kernel instantiation: `nk=camera_capture:1:camera_capture_0` and roi_crop
- ✅ PLIO stream connection: `stream_connect=roi_crop_0.patch_out:ai_engine_0.PatchIn`
- ✅ DDR memory bank assignments for both kernels
- ✅ hw_emu.enableProfiling=false (good for emulation speed)
- ✅ Post-sys-link TCL directive for AIE clock = 1 GHz

### 6. Host Application (mosse_tracker.cpp)
- ✅ XRT device and xclbin loading
- ✅ Graph instantiation and streaming mode (gr.run(-1))
- ✅ All 9 GMIO handles created with correct names
- ✅ XRT buffer objects (BO) allocated for all data: frame, row-FFT, accum, cmul, filter, weights, response
- ✅ PL kernel handles for camera_capture and roi_crop
- ✅ Per-frame tracking loop structure
- ✅ GMIO transactions: gm2aie_nb() and aie2gm_nb() with proper wait()
- ✅ Transpose_inplace() helper for row↔col FFT scratch
- ✅ Peak detection on real part of IFFT response (stride-2 for cint16)

### 7. AIE Compiler Setup
- ✅ --target=hw flag (compiles for both hw and hw_emu)
- ✅ --platform set to VEK280 XPFM
- ✅ All include paths for DSPLib (L1, L2)
- ✅ Preprocessor defines for PATCH_ROWS, PATCH_COLS, N_CHANNELS, FFT parameters
- ✅ Verbose logging for debugging

### 8. Packaging & Emulation
- ✅ v++ packaging step includes:
  - rootfs.ext4 from COMMON_IMAGE_VERSAL
  - kernel Image
  - Boot mode: sd
  - Output directory structure
  - Image format: ext4
  - All required files (.elf, .xsa, libadf.a, run_script.sh)
  - --package.defer_aie_run (correct for hw_emu)
- ✅ run_emu target invokes launch_hw_emu.sh with `-noc-ddr-only 1`
- ✅ run_script.sh has correct command: `./mosse_tracker.elf a.xclbin`

---

## ⚠️ Issues Found & Fixes Required

### Issue 1: **Missing Constraints File in Main AIE Compilation** ❌

**Severity:** HIGH  
**Location:** Makefile, lines 65-88 (AIE_FLAGS definition)

**Problem:**  
The main AIE graph compilation does NOT include the constraints file (`constraints.aiecst`), which defines the PLIO shim placement:
```json
{
  "NodeConstraints": {
    "PatchIn": {
      "shim": {"column": 15, "channel": 0}
    }
  }
}
```

Without this constraint, the AIE compiler may place the PLIO shim at an arbitrary location, causing:
1. Linker failure when trying to connect `roi_crop_0.patch_out:ai_engine_0.PatchIn`
2. Inconsistent placement across builds

**Current behavior:**  
Only the FFT-only test uses `--constraints $(AIE_SRC_REPO)/fft_only_constraints.aiecst` (line 246).

**Fix:**  
Add to AIE_FLAGS in Makefile (around line 88):
```makefile
AIE_FLAGS  += --constraints $(AIE_SRC_REPO)/constraints.aiecst
```

---

### Issue 2: **Missing PLIO Input File for aiesim** ⚠️

**Severity:** MEDIUM (only affects aiesim, not hw_emu)  
**Location:** mosse_graph.h, line 103

**Problem:**  
The PLIO creation references `patch_in.txt`:
```cpp
patch_in = input_plio::create("PatchIn", plio_128_bits, "patch_in.txt");
```

This file doesn't exist. For hw_emu, this is not an issue because:
- The PLIO receives data from roi_crop PL kernel (live stream)
- The file is only used in aiesim simulation for testing

However, running `make aiesim` will fail if `patch_in.txt` is missing.

**Impact:**  
- hw_emu: ✅ No impact (roi_crop feeds data)
- aiesim: ❌ Will fail without test vector

**Fix:**  
The file should be generated by `make gen_vectors` (scripts/gen_aiesim_vectors.py).  
**Verify:** Run `make gen_vectors` to generate aiesim test data.

---

## Minor Items (Best Practice, Not Blocking)

### 1. XRT profiling config not packaged
**File:** design/profiling_configs/xrt.ini  
**Issue:** The xrt.ini configuration is defined but not included in the `v++ --package` command.  
**Impact:** Optional; xrt.ini can still be provided at runtime.  
**Recommendation:** Add to package if detailed tracing is needed:
```makefile
--package.sd_file $(PROFILING_REPO)/xrt.ini \
```

### 2. XLC_EMULATION_MODE commented out in run_script.sh
**File:** design/exec_scripts/run_script.sh, line 10  
**Issue:** `export XLC_EMULATION_MODE=hw_emu` is commented  
**Impact:** XRT may still detect hw_emu mode via `a.xclbin`, but explicit setting is clearer.  
**Recommendation:** Uncomment line 10 for clarity.

### 3. GMIO port count mismatch in CLAUDE.md
**File:** CLAUDE.md vs. actual code  
**Issue:** CLAUDE.md says "10 total (6 input + 4 output)" but actual count is 9 (6 input + 3 output)  
**Analysis:** Counting gmio_cmul_in (input), there are actually:
- Inputs: gmio_weights, gmio_fft_col_in, gmio_cmul_in, gmio_ifft_row_in, gmio_ifft_col_in = 5
- Outputs: gmio_fft_row_out, gmio_accum_out, gmio_ifft_row_out, gmio_response = 4
- **Total: 9 ports (5 input + 4 output)**
**Impact:** None (documentation only)  
**Recommendation:** Update CLAUDE.md to say "9 total (5 input + 4 output)"

---

## Build & Test Checklist

### Before first hw_emu run:
- [ ] Apply Issue #1 fix (add --constraints to AIE_FLAGS)
- [ ] Verify environment: `source setup_env.sh`
- [ ] Generate test vectors: `make gen_vectors` (verifies patch_in.txt generation)
- [ ] Verify weights exist: `make weights`
- [ ] Quick AIE test: `make aiesim SCENARIO=s0` (optional, but recommended)

### Build hw_emu:
```bash
source setup_env.sh
make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1
```

### Launch emulation:
```bash
cd build/hw_emu/128x128/ch1/package
./launch_hw_emu.sh -noc-ddr-only 1 -run-app ../../../../../design/exec_scripts/run_script.sh
```

---

## Verdict

**Status:** ✅ **Ready for hw_emu with 1 critical fix**

The architecture is well-designed. After applying **Issue #1 fix** (add constraints file), the project is ready for:
1. ✅ AIE graph compilation (hw_emu target)
2. ✅ PL kernel compilation (hw_emu mode)
3. ✅ System linking with PLIO+GMIO connectivity
4. ✅ Host application cross-compilation
5. ✅ SD card packaging
6. ✅ Hardware emulation execution

All GMIO port names are consistent, PLIO shim placement is defined, and the XRT API usage is correct.
