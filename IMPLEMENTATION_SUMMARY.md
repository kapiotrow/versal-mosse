# Implementation Summary: roi_crop + Data Injection

**Date:** 2026-06-17  
**Status:** Ready for hw_emu testing

---

## What Was Done

### 1. ✅ Implemented roi_crop HLS Kernel

**File:** `design/pl_src/roi_crop/roi_crop.cpp`

**Key features:**
- Reads frame buffer from DDR via m_axi interface
- Extracts ROI patch (PATCH_ROWS × PATCH_COLS) at (roi_row, roi_col)
- Packs 16 uint8 pixels per 128-bit AXIS beat
- Streams via PatchIn PLIO with proper AXIS handshaking
- II=1 pipelining for efficient streaming
- Fully documented with address mapping logic

**Code structure:**
```cpp
// Iterate over 128-bit beats
for (int beat = 0; beat < total_beats; ++beat) {
    // Pack 16 pixels per beat
    for (int i = 0; i < 16; ++i) {
        // Map linear pixel index to frame address
        int linear_idx = beat * 16 + i;
        int patch_r = linear_idx / patch_cols;
        int patch_c = linear_idx % patch_cols;
        int frame_idx = (roi_row + patch_r) * frame_cols + (roi_col + patch_c);
        
        // Read and pack into 128-bit word
        ap_uint<8> pix = frame_buf[frame_idx];
        word.data.range(8*i + 7, 8*i) = pix;
    }
    patch_out.write(word);
}
```

### 2. ✅ Simplified camera_capture Kernel

**File:** `design/pl_src/camera_capture/camera_capture.cpp`

**Key changes:**
- Reduced to minimal stub: zeros the frame buffer
- Fixed frame size: now `frame_rows × frame_cols` (not ×3 for RGB)
- Added TODO comments for hardware phase (MIPI RX / V4L2)
- Updated documentation to explain data injection approach

### 3. ✅ Added Test Data Injection to Host App

**File:** `design/host_app_src/mosse_tracker.cpp`

**New helper functions:**
```cpp
inject_impulse_frame()     → Single bright pixel at specified location
inject_gradient_frame()    → Ramp pattern for edge/feature testing
inject_checkerboard_frame() → Checkerboard for high-freq response
```

**Main loop integration:**
```cpp
for (int frame = 0; frame < ITER_CNT; ++frame) {
    // 1. Camera capture (zeros buffer)
    camera_capture_kernel();
    
    // 1b. Inject synthetic test data
    inject_impulse_frame(frame_ptr, FRAME_ROWS, FRAME_COLS, pos_row, pos_col, 200);
    frame_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);  // Flush to device
    
    // 2. Process channels (roi_crop reads injected data)
    for (int ch = 0; ch < N_CHANNELS; ++ch) {
        roi_crop_kernel(frame_bo, ...);  // Reads injected frame data
        // ... AIE processing ...
    }
}
```

---

## Architecture: How Data Flows Through hw_emu

```
┌─────────────────────────────────┐
│  Host App (A72)                 │
│  ┌─────────────────────────────┐│
│  │ 1. frame_bo = XRT allocate  ││
│  │ 2. camera_capture zeros it  ││
│  │ 3. inject_impulse_frame()   ││
│  │ 4. frame_bo.sync() → device ││
│  └─────────────────────────────┘│
└────────────┬────────────────────┘
             │ frame_bo.map()
             ↓
       ┌─────────────┐
       │ DDR Buffer  │ (FRAME_ROWS × FRAME_COLS bytes)
       │ [synthetic] │
       └────┬────────┘
            │
            │ (via m_axi gmem0)
            ↓
       ┌──────────────────┐
       │  roi_crop PL     │
       │  kernel          │
       │  (reads ROI)     │
       └────┬─────────────┘
            │
            │ (128-bit beats)
            ↓
       ┌──────────────────────┐
       │ PatchIn PLIO → AIE   │
       │ conv2d kernel        │
       │ FFT2D, cmul_accum,   │
       │ IFFT2D               │
       └────┬─────────────────┘
            │
            │ (response map)
            ↓
       ┌──────────────────┐
       │ GMIO → DDR       │
       │ gmio_response    │
       └────┬─────────────┘
            │
            ↓
       ┌──────────────────┐
       │ Host reads peak  │
       │ displacement     │
       └──────────────────┘
```

---

## Data Types & Sizes

| Component | Type | Size | Notes |
|-----------|------|------|-------|
| Frame buffer | uint8 | 1080×1920 | Linear grayscale (1 byte/pixel) |
| ROI patch | uint8 | 128×128 | Cropped from frame |
| PLIO beat | ap_axiu<128> | 16 bytes | 16 pixels packed per beat |
| Total PLIO beats | — | 1024 | (128×128) / 16 |
| AIE feature maps | cint16 | 128×128 | Per channel (64 KB) |
| Response map | cint16 | 128×128 | Final correlation (64 KB) |

---

## Ready for Testing

The implementation is complete and hw_emu-ready. Key validation points:

✅ **roi_crop correctness:**
- Test with known ROI position
- Verify patch data matches frame crop
- Check PLIO stream integrity

✅ **Data injection correctness:**
- Impulse should appear in response map at correct location
- Gradient should produce smooth response
- Zeros should produce all-zero output

✅ **End-to-end pipeline:**
- Frame → roi_crop → PLIO → AIE → response → peak detection
- Displacement should match injected impulse offset

---

## Next Steps

### To run hw_emu for the first time:

```bash
cd /home/karolina/studia/MGR/versal-mosse

# 1. Set up environment
source setup_env.sh

# 2. Generate weights (required for conv2d)
make weights

# 3. Build hw_emu (single channel for speed)
make sd_card TARGET=hw_emu N_CHANNELS=1 ITER_CNT=1

# 4. Launch emulation
cd build/hw_emu/128x128/ch1/package
./launch_hw_emu.sh -noc-ddr-only 1 -run-app ../../../../../../design/exec_scripts/run_script.sh

# 5. Check output
tail -50 run_emu.log
```

### Expected output on success:
```
Aiecompiler:  (compiles AIE graph)
vitis-model:  (emulation starts)
...
Frame 0: displacement (0,0) → pos (540,960)
INFO: TEST PASSED, RC=0
```

---

## Known Limitations

1. **Single-channel grayscale:** Current implementation assumes uint8 grayscale pixels. RGB → grayscale conversion handled by conv2d_kernel separately.

2. **Test data in host memory:** Synthetic data is generated on A72 (slow). For large frames, consider pre-generating and loading from file in future.

3. **No video loop yet:** camera_capture doesn't read actual video. Phase 2 will add OpenCV/V4L2 support.

4. **Fixed ROI:** Tracked position moves by peak detection, but initial position is hard-coded. Phase 2 will add interactive ROI selection.

---

## Files Changed

```
design/pl_src/roi_crop/roi_crop.cpp
  → Full implementation of ROI cropping + packing logic

design/pl_src/camera_capture/camera_capture.cpp  
  → Simplified stub; frame size fix (removed ×3)

design/host_app_src/mosse_tracker.cpp
  → Added inject_impulse_frame(), inject_gradient_frame(), etc.
  → Added test data injection step in main loop
  → frame_bo.sync() after injection

Makefile
  → Added --constraints $(AIE_SRC_REPO)/constraints.aiecst

design/exec_scripts/run_script.sh
  → Uncommented export XLC_EMULATION_MODE=hw_emu

CLAUDE.md
  → Updated GMIO port count and names (9 ports, not 10)
```

---

## References

- **HW_EMU_REVIEW.md** — Complete hw_emu setup verification
- **DATA_INJECTION_STRATEGY.md** — Detailed data flow and test scenarios
- **CLAUDE.md** — Updated architecture documentation

