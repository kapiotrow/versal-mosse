# Data Injection Strategy for hw_emu Testing

**Status:** Ready for hw_emu validation  
**Created:** 2026-06-17

---

## Overview

For hardware emulation testing, the MOSSE tracker pipeline needs to consume frame data. Since hw_emu is a simulator without real video input, we inject synthetic test data from the host application.

## Architecture

```
Host App (mosse_tracker.cpp)
    ↓
camera_capture kernel (zeros frame buffer)
    ↓
[Host injects synthetic test data via XRT BO map]
    ↓
roi_crop kernel (reads frame, crops ROI, streams patch)
    ↓
PLIO → AIE (conv2d + FFT2D + cmul_accum + IFFT2D)
    ↓
Response map → Host (peak detection)
```

## Implementation Details

### 1. camera_capture Kernel

**File:** `design/pl_src/camera_capture/camera_capture.cpp`

**Current behavior (hw_emu):**
- Zeros the entire frame buffer
- Minimal overhead; allows host to inject test data afterward

**Future behavior (hardware):**
- TODO: Add MIPI CSI-2 RX interface or V4L2 stream input
- Read from real video source

**Signature:**
```cpp
void camera_capture(ap_uint<8> *frame_buf, int frame_rows, int frame_cols)
```

### 2. roi_crop Kernel

**File:** `design/pl_src/roi_crop/roi_crop.cpp`

**Fully implemented:** ✅

**Behavior:**
- Reads frame data from DDR (via m_axi gmem0)
- Extracts ROI: `[roi_row..roi_row+patch_rows) × [roi_col..roi_col+patch_cols)`
- Packs 16 consecutive pixels per 128-bit AXIS beat
- Streams to PatchIn PLIO (128-bit plio_128_bits)

**Packing (128-bit word = 16 bytes):**
```
beat[0].data[7:0]     = pixel[0]
beat[0].data[15:8]    = pixel[1]
...
beat[0].data[127:120] = pixel[15]
```

**Performance:**
- II=1 pipelining (one beat per cycle)
- Total beats = (PATCH_ROWS × PATCH_COLS) / 16
- For 128×128 patch: 1024 beats at 312.5 MHz = ~3.3 µs

**Signature:**
```cpp
void roi_crop(
    const ap_uint<8>                  *frame_buf,
    hls::stream<ap_axiu<128,0,0,0>>  &patch_out,
    int  frame_cols,
    int  roi_row,
    int  roi_col,
    int  patch_rows,
    int  patch_cols)
```

### 3. Host App Test Data Injection

**File:** `design/host_app_src/mosse_tracker.cpp`

**New helper functions:**

```cpp
inject_impulse_frame(frame_buf, rows, cols, impulse_row, impulse_col, value)
  → Zeros entire frame, places single bright pixel at (impulse_row, impulse_col)
  → Use case: Validate spatial localization and peak detection

inject_gradient_frame(frame_buf, rows, cols)
  → Creates ramp pattern (0 at top-left, 255 at bottom-right)
  → Use case: Test edge detection and feature response

inject_checkerboard_frame(frame_buf, rows, cols, square_size)
  → Alternating black/white squares
  → Use case: Test high-frequency response and aliasing
```

**Usage in main loop:**

```cpp
for (int frame = 0; frame < ITER_CNT; ++frame) {
    // 1. Camera capture zeros the buffer
    auto run = cam(frame_bo, FRAME_ROWS, FRAME_COLS);
    run.wait();

    // 1b. Inject test data
    uint8_t *frame_ptr = frame_bo.map<uint8_t *>();
    inject_impulse_frame(frame_ptr, FRAME_ROWS, FRAME_COLS, pos_row, pos_col, 200);
    frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // Flush host → device

    // 2. Process channels (roi_crop reads frame_bo with injected data)
    for (int ch = 0; ch < N_CHANNELS; ++ch) {
        auto crop_run = crop(frame_bo, FRAME_COLS, roi_row, roi_col, PATCH_ROWS, PATCH_COLS);
        // ... GMIO transactions ...
    }
    
    // ... IFFT and peak detection ...
}
```

**Key point:** `frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)` flushes host memory to device DDR before the roi_crop kernel reads it.

---

## Test Scenarios for hw_emu

### Scenario 1: Impulse at Tracked Position
**Test:** Inject impulse at (pos_row, pos_col)
**Expected:** Response map should peak at center (0, 0) after DC removal
**Validates:** 
- Frame I/O pipeline
- ROI cropping correctness
- Spatial localization accuracy

### Scenario 2: Impulse at Off-Center
**Test:** Inject impulse at (pos_row + 10, pos_col + 20)
**Expected:** Response peaks at (10, 20)
**Validates:**
- Peak detection accuracy
- Tracking update logic

### Scenario 3: Gradient Pattern
**Test:** Inject gradient frame
**Expected:** Smooth response map (no aliasing)
**Validates:**
- Convolution kernel correctness
- FFT/IFFT symmetry

### Scenario 4: Zeros
**Test:** No injection (all zeros frame)
**Expected:** All-zero response
**Validates:**
- Baseline case; should produce minimal output

---

## Data Flow Diagram

```
Frame Buffer (DDR, 1080×1920 uint8)
    ↑
    │ camera_capture zeros it
    │
    Host App injects synthetic data
    │
    frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)  ← Flush to device
    ↓
roi_crop kernel (reads via m_axi gmem0)
    │
    Extracts ROI patch (128×128 or PATCH_ROWS×PATCH_COLS)
    │
    Packs into 128-bit beats
    ↓
PatchIn PLIO (128-bit stream)
    ↓
conv2d kernel (int8 patch → cint16 features)
    ↓
FFT2D (row-wise FFT)
    ↓
gmio_fft_row_out (GMIO output to DDR)
    ↓
Host app transpose in DDR
    ↓
gmio_fft_col_in (GMIO input from DDR)
    ↓
FFT2D (column-wise FFT)
    ↓
cmul_accum_kernel (H_ch* ⊙ F_ch + accumulate)
    ↓
... (IFFT, peak detection)
```

---

## Performance Notes

### Host Data Injection Overhead
- `frame_bo.map<uint8_t*>()`: ~1 µs (mapping overhead)
- `inject_impulse_frame()`: O(FRAME_ROWS × FRAME_COLS) = O(2.1M) = ~10 ms at CPU speed
- `frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)`: ~10-100 µs (depends on DMA controller)
- **Total injection overhead: ~10-15 ms per frame**

This is negligible for hw_emu emulation (which runs at ~100 MHz equivalent, so full pipeline ~100s).

### roi_crop Kernel Performance
- Total cycles ≈ (PATCH_ROWS × PATCH_COLS) / 16 + overhead
- For 128×128: 1024 beats × II=1 + overhead ≈ 1500 cycles
- At 312.5 MHz: ~5 µs actual execution

---

## Future Enhancements

### Phase 1 (current): hw_emu validation ✅
- Synthetic test data via host injection
- Validates full pipeline end-to-end
- Ready for functional correctness testing

### Phase 2: Real video input
- Implement camera_capture to read from file or V4L2 stream
- On hardware: MIPI CSI-2 RX interface
- On host simulation: Pre-recorded video files

### Phase 3: Online tracking
- Add ROI selection (initial bounding box)
- Implement filter update (KissFFT on A72)
- Real-time video playback with tracking overlay

---

## Testing Commands

### Build hw_emu:
```bash
source setup_env.sh
make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1
```

### Run emulation:
```bash
cd build/hw_emu/128x128/ch1/package
./launch_hw_emu.sh -noc-ddr-only 1 -run-app ../../../../../design/exec_scripts/run_script.sh 2>&1 | tee emu.log
```

### Verify output:
```bash
tail -50 emu.log
# Look for: "Frame 0: displacement (...) → pos (...)"
```

---

## Troubleshooting

**Issue:** roi_crop produces zeros
- Check: Is frame_bo.sync() being called?
- Check: Are synthetic data values > 0? (uint8, not signed)

**Issue:** Kernel hangs during roi_crop
- Check: Are frame_rows, frame_cols, roi_row, roi_col, patch_rows, patch_cols correct?
- Check: ROI must fit within frame bounds: roi_row + patch_rows < frame_rows

**Issue:** PLIO stream disconnects
- Check: Is PatchIn PLIO name correct? (mosse_graph.h vs mosse_x1.cfg)
- Check: Are PLIO shim placement constraints applied? (--constraints in Makefile ✓)

---

## Files Modified

- ✅ `design/pl_src/roi_crop/roi_crop.cpp` — Full implementation of cropping logic
- ✅ `design/pl_src/camera_capture/camera_capture.cpp` — Simplified stub with TODO for hw
- ✅ `design/host_app_src/mosse_tracker.cpp` — Added test data injection helpers + main loop integration
- ✅ `Makefile` — Added --constraints for PLIO placement

