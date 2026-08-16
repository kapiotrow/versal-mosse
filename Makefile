# Makefile — versal-mosse MOSSE correlation filter tracker
# Targets VEK280 (xilinx_vek280_base_202520_1) with AIE + PL + PS
#
# Usage:
#   make kernels    — compile PL HLS kernels → .xo files
#   make graph      — compile AIE graph → libadf.a
#   make aiesim     — run AIE simulator (graph only, no PL/PS)
#   make xsa        — link kernels + AIE graph → .xsa
#   make application — cross-compile A72 host app → ELF
#   make package    — package SD card image
#   make run_emu    — launch hw_emu (set LAUNCH_HW_EMU_EXEC=1)
#   make sd_card    — kernels + graph + xsa + application + package
#   make cleanall   — remove all build outputs

# Fail recipes when any command in a `... | tee` pipeline fails (otherwise
# tee's exit code masks v++/aiecompiler failures and the build marches on).
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# =========================================================
# Build parameters
# =========================================================
TARGET         := hw_emu
PATCH_ROWS     := 128
PATCH_COLS     := 128
N_CHANNELS     := 16
FFT_2D_DT      := 0          # 0=cint16, 1=cfloat
ITER_CNT       := 1
PL_FREQ        := 312.5
EN_TRACE       := 0
LAUNCH_HW_EMU_EXEC := 0

# FFT cascade lengths (increase for cfloat or large point sizes)
FFT_ROW_CASCADE_LEN := 1
FFT_COL_CASCADE_LEN := 1
# Rows/cols per FFT kernel invocation. These set the DMA CHUNK SIZE for the four
# ports that carry 96% of the host's GMIO traffic, so they are the transaction-
# count knob, not a DSP parameter: every chunk count derives from them
# (ROW_CHUNKS, COL_CHUNKS, CONV_INVOCATIONS, CMUL_N_CHUNKS).
#
# RAISED 2 -> 8 on 2026-08-14. Transactions per frame 4258 -> 1090 (3.9x) at
# 128x128/ch16, which matters because the per-transaction driver cost is the
# dominant unmeasured risk in the design (2-10 us/tx => 8.5-42.6 ms/frame).
# NUMERICALLY NEUTRAL — same arithmetic, same shifts, same results, verified
# bit-exact for both kernels at WS=8 via x86sim_check. Cost is tile memory:
# total_memory_size 50616 -> 130532 B across 6 cores, and the kernel schedules
# are unchanged (conv2d 140 cyc/16px, cmul 2 cyc).
#
# Constraints if you change these: PATCH_ROWS % FFT_ROW_WS == 0; the DSPLib
# window PATCH_ROWS*FFT_ROW_WS must stay a whole multiple of the FFT point size;
# and ROW_CHUNKS must equal CONV_INVOCATIONS (static_assert in mosse_tracker.cpp)
# or the interleaved weights/drain loop deadlocks.
FFT_ROW_WS          := 8
FFT_COL_WS          := 8

# FFT/IFFT output shifts. The invariant is
#     2*FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT = 12
# which fixes the response scale, so weight can be moved between the forward and
# inverse passes without recalibrating any expected value.
#
# HISTORICAL: 4/2/2 was the default until 2026-08-14, validated with s6 at
# CONV2D_MODE=0 and N_CHANNELS=16 (accum 7728, row IFFT 8805, response 6692,
# err=0 px). That was with ReLU ON and a UNITY filter; see the current default
# below. Do not set IFFT_ROW_SHIFT=0 at high channel counts — the row IFFT takes
# the accumulated spectrum and overflows (~101000) with no attenuation.
#
# ECHO-MODE SCENARIOS (s0-s4, CONV2D_MODE=1) need 0/0/12 instead: their magnitudes
# are ~100, and a forward shift of 4 divides the spectrum by 256 and crushes them to
# zero. Run them as:
#   make aiesim_plio CONV2D_MODE=1 SCENARIO=s1 FFT_SHIFT=0 IFFT_ROW_SHIFT=0 IFFT_COL_SHIFT=12
#
# FFT_SHIFT=3 SATURATES THE FORWARD FFT ON A REALISTIC TARGET AT ONE CHANNEL.
# Measured 2026-08-11 in hw_emu, 64x64 ch1, 3/0/6, H_SHIFT=10, against the
# asymmetric structured target (inject_target_frame, not the old impulse):
# gmio_fft_col_out came back with 11 bins railed on BOTH frames, and they are
# exactly the nine bins {0,+-1}x{0,+-1} that apply_dc_correction() operates on,
# plus (+-2,0). So Stage B2 was computing its correction from clipped values —
# it cannot work as designed while the forward pass rails. The accumulator's DC
# bin was railed too. It still localised exactly (err=0 px), so LOCATION ALONE
# DOES NOT DETECT THIS: the tell was PSR 8.4 against Bolme's ~7 failure
# indicator (s7 measures 19.6), and a (10,0) ridge only 1.43x below the true
# peak. Zeroing row 0 and column 0 of the measured accumulator offline dropped
# that ridge from 1.08 to 0.35 of the peak, confirming residual DC as the
# dominant contaminant. 3/0/6 is an aiesim-era setting; do not use it.
# 4-3-3 as of 2026-08-14, up from 4-2-2, BECAUSE CONV_RELU=0 raises the signal.
# Predicted held-out at 128x128/ch16 (scripts/phase1_sweep.py): 4-2-2 -> response
# at 80.4% of rail, 4-3-3 -> 20.1%, accumulator 3.0%, nothing railed. The model
# tracks hardware to ~6%, so 80% is not a margin worth taking.
FFT_SHIFT           ?= 4
IFFT_ROW_SHIFT      ?= 3
IFFT_COL_SHIFT      ?= 3

# cmul_accum filter-product shift. INDEPENDENT of the invariant above.
#
# cmul_accum used to multiply F by H with no shift, on the strength of a comment
# saying "PS pre-scales H" — a contract nothing implemented. Every scenario passed
# a literal H = 1, so the whole budget above was calibrated at a filter gain of
# ONE and says nothing about a real filter. At that scaling a real H would need
# |H| <= 4 to keep the accumulator off the rails: two bits of resolution.
#
# H is now Q1.15 (the host normalizes max|H| -> 32767) and the kernel shifts the
# product right by H_SHIFT. Because |H|/2^15 <= 1, the product can only be SMALLER
# than the old H=1 case, so 4/2/2 above remains valid as an upper bound and needs
# no re-sweep for safety. The new risk runs the other way: a spiky H leaves most
# bins far below the rails, which is why the host reports max|response|.
#
# H_SHIFT is NOT the filter's quantization ceiling. H is always normalized to the
# full int16 range (32767) for maximum resolution; H_SHIFT only decides where the
# product F*H lands in the cint16 accumulator. Tying the two together throws away
# one bit of filter precision per bit of gain — see filter_quantize_q15().
#
# DEFAULT 10, calibrated 2026-08-05 against aiesim s7 (the first scenario with a
# REAL MOSSE filter). 15 was the naive choice — "H is Q1.15" — and it is wrong,
# because a MOSSE filter is spiky: max|H| sits where |F| is SMALLEST, since that is
# where the regularized inverse peaks, so normalizing the peak bin to full scale
# leaves every informative bin far below it. Measured at H_SHIFT=15, 64x64,
# FFT_SHIFT=3/0/6, ch1: accum 15/32767, response 21 LSB. It still localised
# exactly (err=0 px) but PSR collapsed to 5.2x against a golden 38x.
#
# Scaling from those measurements (every stage is linear in 2^-H_SHIFT):
#   H_SHIFT   accum(ch1)  accum(ch16)  rowIFFT  response
#      15          15          240        173        21   response at the floor
#      12         120         1920       1384       168   response at the floor
#      10         480         7680       5536       672   <-- 4x margin everywhere
#       8        1920        30719      22143      2688   accum ~94% of rail at ch16
# The ch16 column at H_SHIFT=10 (7680) lands on the 7728 already recorded as the
# validated 16-channel accumulator, which is a useful independent check.
#
# SINGLE SOURCE OF TRUTH — reaches the AIE kernel, the host app and the vector
# generator from this one line.
H_SHIFT             ?= 10

# =========================================================
# Paths
# =========================================================
RELATIVE_PROJECT_DIR := ./
PROJECT_REPO   := $(shell readlink -f $(RELATIVE_PROJECT_DIR))
DESIGN_REPO    := $(PROJECT_REPO)/design
AIE_SRC_REPO   := $(DESIGN_REPO)/aie_src
PL_SRC_REPO    := $(DESIGN_REPO)/pl_src
HOST_APP_SRC   := $(DESIGN_REPO)/host_app_src
SYS_CONFIGS    := $(DESIGN_REPO)/system_configs
PROFILING_REPO := $(DESIGN_REPO)/profiling_configs
DIRECTIVES     := $(DESIGN_REPO)/directives
EXEC_SCRIPTS   := $(DESIGN_REPO)/exec_scripts

DSPLIB_ROOT    := $(DSPLIB_VITIS)/dsp

# =========================================================
# Root filesystem for packaging
# =========================================================
# v++ --package must NOT be pointed at $(COMMON_IMAGE_VERSAL)/rootfs.ext4 directly.
# That image uses ext4 features (orphan_file, metadata_csum_seed, metadata_csum)
# that the ext4 writer bundled with v++ does not understand.  It injects the boot
# files anyway and silently shreds the filesystem — /usr/sbin drops from 582
# entries to 6, /usr/sbin/init disappears, and the journal is left invalid, so the
# target kernel panics at boot:
#
#     EXT4-fs (mmcblk0p2): Could not load journal inode
#     Kernel panic - not syncing: VFS: Unable to mount root fs on "/dev/mmcblk0p2"
#     Kernel panic - not syncing: No working init found.
#
# `make rootfs` produces a feature-downgraded copy that v++ can write safely.
# The pristine image is never modified.
ROOTFS_DIR     := build/rootfs
ROOTFS         := $(ROOTFS_DIR)/rootfs_compat.ext4

# =========================================================
# Build output directories
# =========================================================
BUILD_DIR      := build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/ch$(N_CHANNELS)
WORK_DIR       := $(BUILD_DIR)/Work

# =========================================================
# Output file names
# =========================================================
LIBADF_A       := $(BUILD_DIR)/libadf.a
APP_ELF        := mosse_tracker.elf
XSA            := versal_mosse.$(TARGET).xsa

# =========================================================
# AIE compiler flags
# =========================================================
AIE_FLAGS  := --target=hw
AIE_FLAGS  += --platform=$(PLATFORM)
AIE_FLAGS  += -include=$(AIE_SRC_REPO)
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L1/include/aie
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L1/src/aie
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L1/tests/aie/inc
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L1/tests/aie/src
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L2/include/aie
AIE_FLAGS  += -include=$(DSPLIB_ROOT)/L2/tests/aie/common/inc
AIE_FLAGS  += --Xpreproc="-DPATCH_ROWS=$(PATCH_ROWS)"
AIE_FLAGS  += --Xpreproc="-DPATCH_COLS=$(PATCH_COLS)"
AIE_FLAGS  += --Xpreproc="-DN_CHANNELS=$(N_CHANNELS)"
AIE_FLAGS  += --Xpreproc="-DFFT_2D_DT=$(FFT_2D_DT)"
AIE_FLAGS  += --Xpreproc="-DFFT_ROW_CASCADE_LEN=$(FFT_ROW_CASCADE_LEN)"
AIE_FLAGS  += --Xpreproc="-DFFT_COL_CASCADE_LEN=$(FFT_COL_CASCADE_LEN)"
AIE_FLAGS  += --Xpreproc="-DFFT_ROW_WS=$(FFT_ROW_WS)"
AIE_FLAGS  += --Xpreproc="-DFFT_COL_WS=$(FFT_COL_WS)"
# ITER_CNT is deliberately NOT here. It appears in NO file under design/aie_src —
# the frame count is purely a host loop bound — but while it was in AIE_FLAGS it
# landed in aie.flagstamp, so changing the number of frames forced a libadf.a
# rebuild and an XSA relink. On TARGET=hw that also means a full re-place-and-route:
# hours of implementation for a define nothing reads. Removed 2026-08-16, which
# makes ITER_CNT a host-only variable — changing it now rebuilds only the ELF.
# (GCC_FLAGS still carries it; see the app.flagstamp note.)
# FFT/IFFT normalization shifts. IFFT_COL_SHIFT is the consequential one: it was
# calibrated for BROADBAND spectra and destroys DC-concentrated ones (see the
# warning in ifft_graph.h). SINGLE SOURCE OF TRUTH — the same values are passed to
# gen_aiesim_vectors.py in the gen_vectors target, because the expected peak values
# scale by 2^(12-IFFT_COL_SHIFT). If the graph and the vectors ever disagree about
# the shift, every expected value is silently wrong.
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_SHIFT=$(FFT_SHIFT)"
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_IFFT_ROW_SHIFT=$(IFFT_ROW_SHIFT)"
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_IFFT_COL_SHIFT=$(IFFT_COL_SHIFT)"
AIE_FLAGS  += --Xpreproc="-DCMUL_H_SHIFT=$(H_SHIFT)"
# conv2d build mode: 0=real conv, 1=echo stream, 2=synthesize without reading the
# stream (bisect for "is conv2d blocked on readincr?"). See conv2d_kernel.cpp.
# DEFAULT CHANGED 1 -> 0 on 2026-08-14. Echo mode is a bisection tool, and having
# it as the default cost a ~28 h ch16 baseline whose numbers all had to be
# requalified (see the echo-mode entry in CLAUDE.md): nothing in a `make sd_card`
# run announces that conv2d is a passthrough. The scenarios that need echo mode
# (s0-s4, raw patches) already pass CONV2D_MODE=1 explicitly in their documented
# commands, so this costs them nothing.
CONV2D_MODE ?= 0
AIE_FLAGS  += --Xpreproc="-DCONV2D_ECHO_TEST=$(CONV2D_MODE)"
# cmul_accum arithmetic: 1 = vectorized aie::mac (default), 0 = the original
# scalar loop. BIT-IDENTICAL by construction and checked by
#   make x86sim_check KUT=cmul SCENARIO=s7
#   make x86sim_check KUT=cmul SCENARIO=cmul_stress
# so 0 is for bisection and cycle comparison, not for correctness fallback.
CMUL_VECTORIZE ?= 1
AIE_FLAGS  += --Xpreproc="-DCMUL_VECTORIZE=$(CMUL_VECTORIZE)"
# conv2d MAC loop: 1 = vectorized aie::mac (default), 0 = the original scalar
# loop. BIT-IDENTICAL, checked by
#   make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0
CONV_VECTORIZE ?= 1
AIE_FLAGS  += --Xpreproc="-DCONV_VECTORIZE=$(CONV_VECTORIZE)"
# Half-wave rectifier after the output shift. 1 = as shipped, 0 = saturate only.
#
# THIS IS THE PHASE 1 DECISION KNOB, and 0 is the better setting — measured
# offline (scripts/phase1_sweep.py), held out, at 128x128/ch16:
#   ReLU on , bias as shipped  : peak/max-sidelobe 12.8
#   ReLU on , bias "fixed" (a) : peak/max-sidelobe  3.9   <-- the planned fix, 3x WORSE
#   ReLU OFF, bias "fixed"     : peak/max-sidelobe 16.3   <-- best
# Unlike CONV_VECTORIZE this CHANGES NUMERICS, so it needs its own before/after
# run and its own shift-budget check. See the ReLU entry in CLAUDE.md.
# DEFAULT CHANGED 1 -> 0 on 2026-08-14 (user decision, on the sweep evidence).
# Held out at 128x128/ch16, peak/max-sidelobe: ReLU on 12.8, ReLU off 15.9.
# NOTE the bias_acc contract fix is NOT applied — with ReLU off it is worth
# almost nothing (15.9 vs 16.3), so export_weights.py is left alone and the
# whole Phase 1 change is this one flag.
# COUPLED: this changes numerics, so the shift budget moves with it (4-2-2 put
# the ch16 response at 80% of rail; 4-3-3 puts it at 20%) and the aiesim goldens
# must be regenerated — gen_aiesim_vectors.py takes GEN_CONV_RELU below.
CONV_RELU ?= 0
AIE_FLAGS  += --Xpreproc="-DCONV_RELU=$(CONV_RELU)"

# Snapshot of the PORTABLE part of the flags — platform, includes and every -D —
# taken here, before the hw-specific tail (--constraints/--Xchess/--Xelfgen/
# --workdir) is appended. The x86sim harness builds on this so it cannot drift
# out of sync with the real build's defines: a bit-exactness test compiled with
# different PATCH_ROWS or H_SHIFT than production would be worse than no test.
# Assignment only, never appended to, so AIE_FLAGS itself is unaffected.
AIE_FLAGS_COMMON := $(AIE_FLAGS)

AIE_FLAGS  += --verbose
AIE_FLAGS  += --log-level=5
AIE_FLAGS  += --pl-freq=$(PL_FREQ)
AIE_FLAGS  += --constraints $(AIE_SRC_REPO)/constraints.aiecst
AIE_FLAGS  += --Xchess="main:bridge.llibs=softfloat m"
AIE_FLAGS  += --Xelfgen="-j2"
AIE_FLAGS  += --workdir=$(WORK_DIR)

GRAPH_SRC_CPP := $(AIE_SRC_REPO)/mosse_graph.cpp

# Test scenario — selects aiesim_data/<SCENARIO>/ for PLIO and GMIO data.
# Scenarios: s0 (baseline), s1 (off-centre impulse), s2 (constant/DC), s3 (imag filter),
#            s4 (Gaussian filter), s6 (FULL preprocessing path — Stage A -> conv2d -> B1),
#            s7 (s6 + a REAL MOSSE filter: non-uniform complex H, off-centre target)
# s0-s4 are raw-patch scenarios: run them with CONV2D_MODE=1 (echo). s6 and s7 feed a
# Stage-A-preprocessed patch, so they are the only valid scenarios for CONV2D_MODE=0.
# s7 is the only scenario that exercises H_SHIFT with a filter that is not identity.
SCENARIO         ?= s0
SCENARIO_DATA_DIR = $(AIE_SRC_REPO)/aiesim_data/$(SCENARIO)

# Aiesimulator flags
AIE_SIM_FLAGS := --pkg-dir $(WORK_DIR)/
AIE_SIM_FLAGS += -i=$(SCENARIO_DATA_DIR)
# Safety net: abort if the simulation freezes.
# cmul_accum_kernel does 64 invocations × 64 v8cint16 vector loads from memory
# tile 13_0 = 4096 loads.  Cycle-approximate ISS models cross-tile vector loads
# at ~20K cycles each → ~82M cycles total.  500M gives a 6× safety margin and
# won't fire before the 1200s wall-clock kills a genuinely hung simulation.
# Both timeouts scale with N_CHANNELS: the harness now loops cmul_accum once per
# channel, and cmul is the dominant cost (~82M cycles for 64 invocations). At 16
# channels that is ~1.3B cycles, which would blow through a fixed 500M cap and the
# fixed 1200s wall clock and look like a hang rather than a slow run.
# aiesimulator parses --simulation-cycle-timeout as a 32-bit int, so the value must
# stay under INT32_MAX (2147483647). 500M x 16 channels = 8e9 is rejected outright:
#   "the argument ('8000000000') for option '--simulation-cycle-timeout' is invalid"
# Clamp to 2e9. Estimated need at 16 channels is ~1.3B cycles (~82M per channel), so
# that leaves ~1.5x margin — thinner than the 6x the single-channel default had. If a
# long run dies, check the log for the simulator's cycle-timeout message before
# assuming a hang.
SIM_CYCLE_MAX     := 2000000000
SIM_CYCLE_TIMEOUT ?= $(shell v=$$(expr 500000000 \* $(N_CHANNELS)); \
                             if [ $$v -gt $(SIM_CYCLE_MAX) ]; then echo $(SIM_CYCLE_MAX); else echo $$v; fi)
# The wall clock must scale with PATCH AREA as well as with the channel count.
# It used to be 1200 x N_CHANNELS only, which is the 64x64 budget applied
# unchanged to a 128x128 build carrying 4x the data — conv2d alone goes from 32 to
# 64 invocations over 4x the pixels. Measured 2026-08-05: s6 at 128x128 ch1 was
# killed by `timeout` (exit 124) while still in step 2, and the log looked
# identical to a deadlock. Every aiesim_plio PASS recorded in CLAUDE.md before that
# date is a 64x64 run, so the 128x128 path had simply never been given time to
# finish. Baseline 1200 s at 64x64 = 4096 elements.
SIM_PATCH_SCALE   := $(shell expr \( $(PATCH_ROWS) \* $(PATCH_COLS) + 4095 \) / 4096)
SIM_WALL_TIMEOUT  ?= $(shell expr 1200 \* $(N_CHANNELS) \* $(SIM_PATCH_SCALE))
AIE_SIM_FLAGS += --simulation-cycle-timeout=$(SIM_CYCLE_TIMEOUT)

# =========================================================
# v++ common flags
# =========================================================
HZ_UNIT      := 1000000
VPP_CLOCK_FREQ := $(shell printf "%.0f" `echo "$(PL_FREQ) * $(HZ_UNIT)" | bc`)

VPP_FLAGS  := -t $(TARGET)
VPP_FLAGS  += --platform $(PLATFORM)
VPP_FLAGS  += --save-temps
VPP_FLAGS  += --temp_dir $(BUILD_DIR)/_x

# =========================================================
# PL kernel compile flags (per kernel)
# =========================================================
CAM_VPP_FLAGS   := --hls.clock $(VPP_CLOCK_FREQ):camera_capture

CROP_VPP_FLAGS  := --hls.clock $(VPP_CLOCK_FREQ):roi_crop

# =========================================================
# Host application compiler flags
# =========================================================
GCC_FLAGS  := -O2 -std=c++17 -D__linux__ -D__PS_ENABLE_AIE__
GCC_FLAGS  += -DPATCH_ROWS=$(PATCH_ROWS)
GCC_FLAGS  += -DPATCH_COLS=$(PATCH_COLS)
GCC_FLAGS  += -DN_CHANNELS=$(N_CHANNELS)
GCC_FLAGS  += -DITER_CNT=$(ITER_CNT)
# THE HOST MUST AGREE WITH THE GRAPH ABOUT WINDOW SIZE. These were missing until
# 2026-08-14, and the failure mode was silent and expensive: mosse_tracker.cpp
# defaults FFT_ROW_WS/FFT_COL_WS to 2 in its own `#ifndef`, so a build with
# FFT_ROW_WS=8 produced an AIE graph expecting 1024-sample windows and a host
# chunking the same DMAs in 256-sample pieces. Every chunk count
# (ROW_CHUNKS, COL_CHUNKS, CONV_INVOCATIONS, CMUL_N_CHUNKS) derives from these,
# and a mismatch deadlocks the drain loops — which looks identical to the
# historical aie2gm_nb hang, i.e. a multi-day misdiagnosis waiting to happen.
# Same single-source-of-truth rule as the FFT/IFFT shifts above.
GCC_FLAGS  += -DFFT_ROW_WS=$(FFT_ROW_WS)
GCC_FLAGS  += -DFFT_COL_WS=$(FFT_COL_WS)
# Offset of the synthetic test impulse from the tracked position. A correct
# pipeline must report exactly this displacement; (0,0) would be untestable
# because it is also what a zero response yields. Sweep to check other offsets:
#   make application IMPULSE_DR=-20 IMPULSE_DC=31
IMPULSE_DR ?= 10
IMPULSE_DC ?= -7
GCC_FLAGS  += -DIMPULSE_DR=$(IMPULSE_DR)
GCC_FLAGS  += "-DIMPULSE_DC=($(IMPULSE_DC))"
GCC_FLAGS  += -DFRAME_ROWS=1080
GCC_FLAGS  += -DFRAME_COLS=1920
# The host builds H in Q1.15 to match the shift cmul_accum applies to the product.
# Same value as the AIE_FLAGS line above — see the H_SHIFT comment block.
GCC_FLAGS  += -DCMUL_H_SHIFT=$(H_SHIFT)
# Stage B2 mode. 1 (default) = NULL the 9 low-frequency bins; 0 = subtract µ*W,
# the original design. Nulling was made the default 2026-08-11 because those bins
# RAIL, so the subtraction operates on already-clipped values and the residual DC
# swamps the response (measured: peak/pedestal 1.43x at 3/0/6, and at 4/2/2 the
# pedestal wins outright). Nulling restores exact localisation at PSR 22.7.
# Set to 0 to reproduce the old behaviour for comparison — the two modes are the
# before/after pair for this finding. See apply_dc_correction() in mosse_tracker.cpp.
B2_NULL_BINS ?= 1
GCC_FLAGS  += -DB2_NULL_BINS=$(B2_NULL_BINS)

# ---------------------------------------------------------------------------
# PSR update gating — Bolme §3.5. HOST-ONLY: nothing in the AIE graph changes,
# so these deliberately do NOT appear in AIE_FLAGS. (Contrast FFT_ROW_WS above,
# which the graph and the host BOTH derive from and which therefore must go to
# both toolchains.)
# ---------------------------------------------------------------------------
# "when PSR drops to around 7.0 it is an indication that the object is occluded
# or tracking has failed" — below this the host HOLDS the tracked position and
# skips BOTH filter_update and publish_filter, so an occluded frame cannot train
# the filter on background at eta=0.125.
#
# SET TO 0 TO DISABLE the threshold test (report-only, the pre-2026-08-15
# behaviour). Structural failures — zero response, flat sidelobe, negative peak —
# still veto regardless; only the "weak peak" test is switched off. That makes the
# A/B lever one make variable rather than an #if.
#
# Applies to Bolme's PSR (g_max-mu)/sigma ONLY, never to the |peak|/max|sidelobe|
# ratio that gen_aiesim_vectors.py calls PSR — different statistics, and they
# differ by several times. No `f` suffix here: mosse_filter.h casts it, so 7, 7.0
# and 7.5 all work.
PSR_GATE_MIN ?= 7.0
GCC_FLAGS  += -DPSR_GATE_MIN=$(PSR_GATE_MIN)

# ---------------------------------------------------------------------------
# Target box, ROI padding and sigma — Bolme §3.1/§3.2, Danelljan §3.1,
# DSST (docs/1609.06141v1.pdf) §6.1. HOST-ONLY, same rule as PSR_GATE_MIN above:
# the AIE never sees the ROI, only the fixed patch, and roi_crop takes all of its
# geometry as runtime AXI-Lite scalars — so none of this reaches AIE_FLAGS and
# none of it needs a PL re-synthesis or a libadf.a relink.
# ---------------------------------------------------------------------------
# Until 2026-08-16 the tracker's whole state was pos_row/pos_col. That left sigma
# with no defined relation to the object, gave the filter no background context
# (roi_h was pinned to PATCH_ROWS, so the ROI *was* the patch), and made an IoU —
# the metric both papers report — impossible to compute.
#
# TARGET_H/W = 64 with TARGET_PADDING = 2 gives roi = 128, i.e. EXACTLY the
# geometry that shipped before the box existed, so the resample stays 1:1 and
# roi_crop's bilinear interpolator stays dormant. Adopting the box is therefore a
# single-variable change; moving padding afterwards is a second, separate one.
# TARGET_H/W also drive the shape inject_target_frame draws, so the declared box
# and the drawn object agree by construction (they did not before: an ~11x11
# object sat in a 128x128 ROI, an effective padding of ~11.6).
TARGET_H       ?= 64
TARGET_W       ?= 64
GCC_FLAGS  += -DTARGET_H=$(TARGET_H) -DTARGET_W=$(TARGET_W)

# roi = target * TARGET_PADDING. DSST §6.1 uses 2, fDSST 3. The offline sweep
# (scripts/phase1_sweep.py --roi-model synth) settles this at >= 2: padding 1.5 is
# worst on both metrics and shows a real 0.75 px localisation error, while 2.5/3.0
# edge ahead but trigger aliasing — roi_crop's bilinear has no prefilter, so
# beyond ~2x decimation source rows are skipped outright — and 3.0 clips 3.6% of
# samples. >2x decimation is sampling an aliased signal and no shift budget fixes
# that, so treat 2.5-3.0 as needing evidence rather than as a free upgrade.
TARGET_PADDING ?= 2.0
GCC_FLAGS  += -DTARGET_PADDING=$(TARGET_PADDING)

# Target Gaussian width, in PATCH pixels.
#
# MOSSE_SIGMA is used literally when SIGMA_FROM_TARGET=0 (the default). Setting
# SIGMA_FROM_TARGET=1 switches to DSST §6.1's rule, sigma = target_in_patch_px /
# SIGMA_FACTOR, which at padding 2 gives 4.0.
#
# THE DEFAULT IS DELIBERATELY STILL 2.0. The sweep does not support switching,
# and the reason is that the metric cannot arbitrate: Bolme PSR is MONOTONE
# DECREASING in sigma all the way to sub-pixel (80.3 at 0.75, 45.7 at 2.0, 30.5 at
# 4.0), so it rewards a sharp peak rather than selecting a width — a delta target
# would maximise it. What sigma buys is robustness to appearance change, which a
# single-frame translation-only holdout does not exercise. The obvious
# alternative explanation, Stage B2 nulling the 9 low bins where a wide Gaussian
# keeps its energy, was tested with B2 off and REFUTED (the drop survives).
# Decide it when real video can measure what sigma is actually for.
MOSSE_SIGMA       ?= 2.0
SIGMA_FACTOR      ?= 16.0
SIGMA_FROM_TARGET ?= 0
GCC_FLAGS  += -DMOSSE_SIGMA=$(MOSSE_SIGMA) -DSIGMA_FACTOR=$(SIGMA_FACTOR)
GCC_FLAGS  += -DSIGMA_FROM_TARGET=$(SIGMA_FROM_TARGET)

# Learning rate. Bolme §3.3 uses 0.125; DSST §6.1 uses 0.025 for both filters.
MOSSE_ETA      ?= 0.125
GCC_FLAGS  += -DMOSSE_ETA=$(MOSSE_ETA)

# Background for the synthetic test frame: 1 = band-limited texture, 0 = the
# pre-2026-08-16 flat fill. NOT cosmetic — padding exists so the filter can learn
# target-vs-background, so against a flat fill more padding is strictly less
# target and any padding comparison is decided before it runs.
FRAME_TEXTURE  ?= 1
GCC_FLAGS  += -DFRAME_TEXTURE=$(FRAME_TEXTURE)

# ---------------------------------------------------------------------------
# DSST 1-D scale filter — docs/1609.06141v1.pdf §5.1. HOST-ONLY.
# ---------------------------------------------------------------------------
# A SEPARATE 1-D filter over scale, not an exhaustive multi-resolution search.
# DSST Table 1 beats exhaustive on both axes (OP 67.7 vs 65.2, 25.4 vs 16.9 FPS),
# and on this hardware the gap is wider still: an exhaustive search would push
# patches resampled by +/-30% through roi_crop -> conv2d -> FFT every frame, which
# moves |F| and hence the shift budget, and it would spend exactly the 30 fps
# headroom that vectorizing conv2d and cmul bought back. The 1-D filter runs
# entirely on the APU at ~0.5M complex MACs/frame against the translation
# update's ~2M, and touches no AIE, no PL and no shift budget.
#
# SCALE_N=1 DISABLES the filter completely and reproduces the pre-scale
# behaviour — the same bisection lever CONV_VECTORIZE=0 provides, and it is
# asserted in test_host.
#
# NOT implemented yet, deliberately: fDSST's PCA compression (§5.2.3) and
# sub-grid interpolation (§5.2.1). The compression is PROVABLY LOSSLESS for the
# scale filter (rank <= S), so it can be added later as a pure optimisation with
# a bit-exactness test against this path.
SCALE_N            ?= 33
SCALE_STEP         ?= 1.02
SCALE_ETA          ?= 0.025
SCALE_SIGMA_FACTOR ?= 16.0
SCALE_TMPL_AREA    ?= 512
GCC_FLAGS  += -DSCALE_N=$(SCALE_N) -DSCALE_STEP=$(SCALE_STEP)
GCC_FLAGS  += -DSCALE_ETA=$(SCALE_ETA) -DSCALE_SIGMA_FACTOR=$(SCALE_SIGMA_FACTOR)
GCC_FLAGS  += -DSCALE_TMPL_AREA=$(SCALE_TMPL_AREA)

# Occlusion injection, to prove the gate actually FIRES — it cannot on the normal
# synthetic target, which measures PSR ~172 against a threshold of 7. Bitmask over
# frame index: bit f set => frame f gets a checkerboard instead of the target.
# Bit 0 is ignored (frame 0 trains the filter). The occlude-then-reacquire test:
#   make sd_card ITER_CNT=3 OCCLUDE_MASK=0x2
OCCLUDE_MASK   ?= 0
OCCLUDE_SQUARE ?= 8
GCC_FLAGS  += -DOCCLUDE_MASK=$(OCCLUDE_MASK)
GCC_FLAGS  += -DOCCLUDE_SQUARE=$(OCCLUDE_SQUARE)

# ---------------------------------------------------------------------------
# Test-sequence generation — HOST-ONLY. Added 2026-08-16 for the board runs.
# ---------------------------------------------------------------------------
# Every default below reproduces the pre-2026-08-16 behaviour exactly, so the
# existing hw_emu comparisons stay valid. They exist because a board run can
# afford hundreds of frames where hw_emu could afford two, and the shipped test
# data does not survive that:
#
#   * the background was regenerated every frame (12.4 M sin() calls, ~0.6-1.2 s
#     on the A72) — 30-90x the whole pipeline, which would have made the FPS
#     measurement a measurement of the test harness. Now generated ONCE with a
#     dirty-rect restore; no flag, it is simply correct.
#   * the target walked off a 1080-row frame at about frame 48.
#   * the target never changed size, so the scale filter had nothing to track.
#
# TRAJECTORY=1 also changes something subtler and more important: the target moves
# on an ABSOLUTE scripted path instead of being planted at the tracker's own
# estimate plus a constant. Under the legacy scheme the ground truth follows the
# estimate, so the tracker cannot drift and `err=0 px` is close to self-fulfilling.
# On an absolute path drift is real, measurable, and reported as IoU / centre
# error against the box actually drawn.
TRAJECTORY        ?= 0
TRAJ_AMP_R        ?= 180.0
TRAJ_AMP_C        ?= 180.0
TRAJ_PERIOD       ?= 120.0
GCC_FLAGS  += -DTRAJECTORY=$(TRAJECTORY) -DTRAJ_AMP_R=$(TRAJ_AMP_R)
GCC_FLAGS  += -DTRAJ_AMP_C=$(TRAJ_AMP_C) -DTRAJ_PERIOD=$(TRAJ_PERIOD)

# Size envelope for the scale filter. The rate limit is what matters: SCALE_STEP
# bounds one frame's correction and SCALE_ETA makes the model adapt slowly, so the
# envelope must change far more slowly than the filter's single-frame range.
# 0.30 over 200 frames peaks at 0.94%/frame against a 2%/frame step size.
SCALE_TRAJ        ?= 0
SCALE_TRAJ_AMP    ?= 0.30
SCALE_TRAJ_PERIOD ?= 200.0
GCC_FLAGS  += -DSCALE_TRAJ=$(SCALE_TRAJ) -DSCALE_TRAJ_AMP=$(SCALE_TRAJ_AMP)
GCC_FLAGS  += -DSCALE_TRAJ_PERIOD=$(SCALE_TRAJ_PERIOD)

# Periodic occlusion: occlude when (frame % OCCLUDE_PERIOD) < OCCLUDE_LEN.
# 0 keeps the legacy OCCLUDE_MASK, which is a 32-bit mask indexed by frame number
# and therefore cannot express anything past frame 31 (and shifted by >= 32, which
# is undefined — now guarded).
# OCCLUDE_START is a warm-up: no occlusion before this frame, and the period runs
# from there. Without it the first occlusion is frame 1 — the filter occluded
# immediately after being initialised from a single patch, which tests the gate
# against a filter that has not converged. 30 frames is ~4 time constants at
# MOSSE_ETA=0.125. The SCALE filter is much slower (SCALE_ETA=0.025, ~40 frames
# per constant), so a run meant to occlude a settled SIZE estimate wants ~120.
OCCLUDE_PERIOD    ?= 0
OCCLUDE_LEN       ?= 1
OCCLUDE_START     ?= 30
GCC_FLAGS  += -DOCCLUDE_PERIOD=$(OCCLUDE_PERIOD) -DOCCLUDE_LEN=$(OCCLUDE_LEN)
GCC_FLAGS  += -DOCCLUDE_START=$(OCCLUDE_START)

GCC_INC    := -I$(SDKTARGETSYSROOT)/usr/include/xrt
GCC_INC    += -I$(XILINX_VITIS)/aietools/include/
GCC_INC    += -I$(SDKTARGETSYSROOT)/usr/include
GCC_INC    += -I$(AIE_SRC_REPO)
GCC_INC    += -I$(HOST_APP_SRC)
GCC_INC    += -I$(DSPLIB_ROOT)/L2/include/aie
# No FFT library is needed on the host. The filter update consumes F_ch straight
# from the AIE column FFT (gmio_fft_col_out), and the Gaussian target spectrum has
# a closed form — see gaussian_target_spectrum() in mosse_filter.h.

GCC_LIBS   := -L$(SDKTARGETSYSROOT)/usr/lib
GCC_LIBS   += -L$(XILINX_VITIS)/aietools/lib/aarch64.o
GCC_LIBS   += -ladf_api_xrt -lxrt_coreutil

# =========================================================
# Link config
# =========================================================
VPP_LINK_FLAGS  := --vivado.synth.jobs 8
VPP_LINK_FLAGS  += --config $(SYS_CONFIGS)/mosse_x1.cfg
VPP_LINK_FLAGS  += --clock.freqHz $(VPP_CLOCK_FREQ):camera_capture_0
VPP_LINK_FLAGS  += --clock.freqHz $(VPP_CLOCK_FREQ):roi_crop_0

# =========================================================
# Kernel XO targets
# =========================================================
CAM_XO  := $(BUILD_DIR)/camera_capture.$(TARGET).xo
CROP_XO := $(BUILD_DIR)/roi_crop.$(TARGET).xo

KERNEL_XOS := $(CAM_XO) $(CROP_XO)

# =========================================================
# Rules
# =========================================================
.PHONY: help kernels graph gen_vectors aiesim graph_fft aiesim_fft xsa application package sd_card run_emu weights test_host test_roi_crop cleanall

help:
	@echo ""
	@echo "versal-mosse build targets:"
	@echo "  make kernels      — compile PL HLS kernels"
	@echo "  make graph        — compile AIE graph"
	@echo "  make weights      — export MobileNetV3-Small INT8 weights for conv2d_kernel"
	@echo "  make gen_vectors  — generate aiesim test vectors (impulse input)"
	@echo "  make aiesim       — run AIE simulator (round-trip FFT test)"
	@echo "  make graph_fft    — compile FFT-only smoke-test graph"
	@echo "  make aiesim_fft   — run FFT-only aiesim (2-GMIO, no PLIO)"
	@echo "  make xsa          — link → .xsa"
	@echo "  make application  — cross-compile host ELF"
	@echo "  make test_host    — native unit test for the filter init/update math"
	@echo "  make test_roi_crop — native bit-exact test for roi_crop resample + Stage A"
	@echo "  make package      — package SD card image"
	@echo "  make sd_card      — kernels + graph + xsa + application + package"
	@echo "  make run_emu      — launch hw emulator"
	@echo "  make cleanall     — remove all build outputs"
	@echo ""
	@echo "Key parameters (pass on command line):"
	@echo "  TARGET=$(TARGET)  PATCH_ROWS=$(PATCH_ROWS)  PATCH_COLS=$(PATCH_COLS)"
	@echo "  N_CHANNELS=$(N_CHANNELS)  FFT_2D_DT=$(FFT_2D_DT)  ITER_CNT=$(ITER_CNT)"
	@echo "  FFT_SHIFT=$(FFT_SHIFT)  IFFT_ROW_SHIFT=$(IFFT_ROW_SHIFT)  IFFT_COL_SHIFT=$(IFFT_COL_SHIFT)  H_SHIFT=$(H_SHIFT)"

print-%: ; @echo $* = $($*)

# -------------------------------------------------------
# PL kernels
# -------------------------------------------------------
kernels: $(KERNEL_XOS)

$(CAM_XO): $(PL_SRC_REPO)/camera_capture/camera_capture.cpp
	mkdir -p $(BUILD_DIR)
	v++ $(VPP_FLAGS) $(CAM_VPP_FLAGS) -c -k camera_capture $< -o $@

$(CROP_XO): $(PL_SRC_REPO)/roi_crop/roi_crop.cpp
	mkdir -p $(BUILD_DIR)
	v++ $(VPP_FLAGS) $(CROP_VPP_FLAGS) -c -k roi_crop $< -o $@

# -------------------------------------------------------
# AIE graph
# -------------------------------------------------------
# Flag stamps: rebuild when the compiler FLAGS change, not just the sources.
#
# CONV2D_MODE / SMOKE_SKIP_STREAM / etc. are make variables — changing one touches
# no file, so a source-only prerequisite list happily reuses a stale libadf.a and
# the simulation reports results from the PREVIOUS build's binary. That silently
# cost a debug cycle: a CONV2D_MODE=1 run produced output byte-identical to the
# CONV2D_MODE=2 build it had actually reused.
#
# The stamp is rewritten only when its contents change, so this does not force a
# rebuild on every invocation.
.PHONY: FORCE
FORCE:

%.flagstamp: FORCE
	@mkdir -p $(dir $@)
	@echo '$(FLAGS_FOR_STAMP)' | cmp -s - $@ || echo '$(FLAGS_FOR_STAMP)' > $@

AIE_FLAGS_STAMP   := $(BUILD_DIR)/aie.flagstamp
$(AIE_FLAGS_STAMP):   FLAGS_FOR_STAMP := $(AIE_FLAGS)
# NOTE: SMOKE_FLAGS_STAMP lives in the PLIO smoke section below — it cannot be
# defined here. SMOKE_BUILD/SMOKE_AIE_FLAGS are declared ~160 lines further down,
# and `:=` expands immediately, so defining it here silently produced
# `/aie.flagstamp` with an empty flag list (the recipe then died on "Permission
# denied" at the filesystem root, and the SMOKE_SKIP_STREAM guard never armed).

graph: $(LIBADF_A)

$(LIBADF_A): $(AIE_FLAGS_STAMP)                \
             $(AIE_SRC_REPO)/mosse_graph.cpp  \
             $(AIE_SRC_REPO)/mosse_graph.h     \
             $(AIE_SRC_REPO)/fft_graph.h       \
             $(AIE_SRC_REPO)/ifft_graph.h      \
             $(AIE_SRC_REPO)/conv2d_kernel.h   \
             $(AIE_SRC_REPO)/conv2d_kernel.cpp \
             $(AIE_SRC_REPO)/cmul_accum_kernel.h \
             $(AIE_SRC_REPO)/cmul_accum_kernel.cpp
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && aiecompiler $(AIE_FLAGS) $(GRAPH_SRC_CPP) 2>&1 | tee aiecompiler.log

weights:
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= uv run --extra weights python3 scripts/export_weights.py $(AIE_SRC_REPO)/weights $(PATCH_COLS)

# int8 samples per PatchIn beat. mosse_graph.h creates PatchIn as plio_32_bits,
# so 4; change to 16 if the PLIO ever goes back to plio_128_bits.
PLIO_BEAT_SAMPLES ?= 4

gen_vectors:
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    GEN_PATCH_ROWS=$(PATCH_ROWS) \
	    GEN_PATCH_COLS=$(PATCH_COLS) \
	    GEN_PLIO_BEAT_SAMPLES=$(PLIO_BEAT_SAMPLES) \
	    GEN_IFFT_COL_SHIFT=$(IFFT_COL_SHIFT) \
	    GEN_IFFT_ROW_SHIFT=$(IFFT_ROW_SHIFT) \
	    GEN_FFT_SHIFT=$(FFT_SHIFT) \
	    GEN_H_SHIFT=$(H_SHIFT) \
	    GEN_CONV_RELU=$(CONV_RELU) \
	    uv run python3 scripts/gen_aiesim_vectors.py $(AIE_SRC_REPO)/aiesim_data

aiesim: graph gen_vectors
	-cd $(BUILD_DIR) && AIESIM_SCENARIO_DIR=$(SCENARIO_DATA_DIR) \
	    timeout $(SIM_WALL_TIMEOUT) aiesimulator $(AIE_SIM_FLAGS) 2>&1 | tee aiesim.log
	@echo "--- aiesim done (SCENARIO=$(SCENARIO)); check $(BUILD_DIR)/aiesim.log for PASS/FAIL ---"

# Decisive PatchIn test: does the PLIO deliver into conv2d on the AIE side, with
# no PL, no v++, no shim involved?
#
# Plain `make aiesim` cannot answer this. When fft_col_in.bin exists the harness
# in mosse_graph.cpp bypasses PatchIn -> conv2d -> row-FFT entirely and feeds the
# col-FFT from that file, so the run completes even if conv2d is blocked forever
# on readincr — which is exactly why aiesim has "passed" historically.
#
# Removing fft_col_in.bin forces the fallback branch, which drains
# gmio_fft_row_out and therefore requires conv2d to have consumed the PLIO:
#   "step 2: fft_row_out done" + sane values -> AIE-side PLIO works; the hw_emu
#                                               fault is in the PL->shim link.
#   hangs at "waiting for fft_row_out"       -> AIE-side stream path is broken,
#                                               independent of any PL.
.PHONY: aiesim_plio
aiesim_plio: graph gen_vectors
	rm -f $(SCENARIO_DATA_DIR)/fft_col_in.bin
	-cd $(BUILD_DIR) && AIESIM_SCENARIO_DIR=$(SCENARIO_DATA_DIR) \
	    timeout $(SIM_WALL_TIMEOUT) aiesimulator $(AIE_SIM_FLAGS) 2>&1 | tee aiesim_plio.log
	@echo "--- aiesim_plio done (SCENARIO=$(SCENARIO)); look for 'step 2: fft_row_out done' in $(BUILD_DIR)/aiesim_plio.log ---"

# -------------------------------------------------------
# FFT-only aiesim — isolated DSPLib row-FFT smoke test
# -------------------------------------------------------
FFT_ONLY_BUILD_DIR := build/$(TARGET)/$(PATCH_ROWS)x$(PATCH_COLS)/fft_only
FFT_ONLY_WORK_DIR  := $(FFT_ONLY_BUILD_DIR)/Work
FFT_ONLY_SRC       := $(AIE_SRC_REPO)/fft_only_graph.cpp

# Reuse AIE_FLAGS but swap workdir and supply empty constraints file
# to prevent aiecompiler picking up the PatchIn PLIO constraint.
FFT_ONLY_AIE_FLAGS  := $(filter-out --workdir=$(WORK_DIR),$(AIE_FLAGS))
FFT_ONLY_AIE_FLAGS  += --workdir=$(FFT_ONLY_WORK_DIR)
FFT_ONLY_AIE_FLAGS  += --constraints $(AIE_SRC_REPO)/fft_only_constraints.aiecst

FFT_ONLY_SIM_FLAGS  := --pkg-dir $(FFT_ONLY_WORK_DIR)/
FFT_ONLY_SIM_FLAGS  += -i=$(AIE_SRC_REPO)/aiesim_data
FFT_ONLY_SIM_FLAGS  += --simulation-cycle-timeout=100000

graph_fft:
	mkdir -p $(FFT_ONLY_BUILD_DIR)
	cd $(FFT_ONLY_BUILD_DIR) && aiecompiler $(FFT_ONLY_AIE_FLAGS) \
	    $(FFT_ONLY_SRC) 2>&1 | tee aiecompiler_fft.log

aiesim_fft: graph_fft
	-cd $(FFT_ONLY_BUILD_DIR) && timeout 120 aiesimulator $(FFT_ONLY_SIM_FLAGS) 2>&1 | tee aiesim_fft.log
	@echo "--- aiesim_fft done; check $(FFT_ONLY_BUILD_DIR)/aiesim_fft.log ---"

# -------------------------------------------------------
# x86sim kernel bit-exactness harness
# -------------------------------------------------------
# Isolates ONE kernel, dumps its raw output, and diffs that against the Python
# model in gen_aiesim_vectors.py. Seconds per run, versus hours for aiesim and
# ~24 h for an hw_emu frame at ch16.
#
# This is the gate for any change to conv2d or cmul_accum. It is also the only
# check that gen_aiesim_vectors.py's simulate_conv2d() actually matches the
# kernel it claims to replicate — a docstring assertion the offline shift-budget
# work depends on entirely.
#
#   make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0
#   make x86sim_check KUT=cmul   SCENARIO=s7
#
# NOTE conv2d MUST be built CONV2D_MODE=0 to test the convolution; the default is
# echo, and echo mode compares a passthrough. The harness prints a warning.
KUT              ?= conv2d
ifeq ($(KUT),conv2d)
  KUT_ID := 0
else ifeq ($(KUT),cmul)
  KUT_ID := 1
else
  $(error KUT must be conv2d or cmul, got '$(KUT)')
endif

X86_BUILD_DIR := build/x86sim/$(PATCH_ROWS)x$(PATCH_COLS)/$(KUT)
# ABSOLUTE. The recipes `cd $(X86_BUILD_DIR)` first, so a relative --workdir
# resolves against it and buries Work at
#   build/x86sim/.../conv2d/build/x86sim/.../conv2d/Work
# which is what the other targets in this file do and it is needlessly confusing.
X86_WORK_DIR  := $(PROJECT_REPO)/$(X86_BUILD_DIR)/Work
X86_SRC       := $(AIE_SRC_REPO)/kernel_only_graph.cpp

# Built from AIE_FLAGS_COMMON (see its definition above) so every -D matches the
# real build. --target=hw is filtered out rather than overridden; the empty
# constraints file is reused from the fft_only harness since x86sim does not
# place shims.
X86_AIE_FLAGS := $(filter-out --target=hw,$(AIE_FLAGS_COMMON))
X86_AIE_FLAGS += --target=x86sim
X86_AIE_FLAGS += --Xpreproc="-DKERNEL_UNDER_TEST=$(KUT_ID)"
X86_AIE_FLAGS += --constraints $(AIE_SRC_REPO)/fft_only_constraints.aiecst
X86_AIE_FLAGS += --workdir=$(X86_WORK_DIR)

# Flag guard, same purpose as AIE_FLAGS_STAMP: a KUT= or CONV2D_MODE= change with
# no source edit must not silently reuse the previous kernel's build. Defined
# HERE, next to the flags it stamps — `:=` expands immediately, and defining a
# stamp above its variables is exactly the bug that broke SMOKE_FLAGS_STAMP.
X86_FLAGS_STAMP := $(X86_BUILD_DIR)/aie.flagstamp
$(X86_FLAGS_STAMP): FLAGS_FOR_STAMP := $(X86_AIE_FLAGS)

X86_SIM_FLAGS := --pkg-dir=$(X86_WORK_DIR)
X86_SIM_FLAGS += -i=$(SCENARIO_DATA_DIR)
X86_SIM_FLAGS += --output-dir=$(PROJECT_REPO)/$(X86_BUILD_DIR)

.PHONY: x86sim_graph x86sim_check
x86sim_graph: $(X86_FLAGS_STAMP) $(X86_SRC) \
              $(AIE_SRC_REPO)/kernel_only_graph.h \
              $(AIE_SRC_REPO)/aiesim_scenario_io.h \
              $(AIE_SRC_REPO)/conv2d_kernel.cpp \
              $(AIE_SRC_REPO)/cmul_accum_kernel.cpp
	mkdir -p $(X86_BUILD_DIR)
	cd $(X86_BUILD_DIR) && aiecompiler $(X86_AIE_FLAGS) $(X86_SRC) 2>&1 \
	    | tee aiecompiler_x86.log
	@grep -o 'KERNEL_UNDER_TEST=[0-9]' $(X86_BUILD_DIR)/aiecompiler_x86.log | tail -1
	@grep -o 'CONV2D_ECHO_TEST=[0-9]'  $(X86_BUILD_DIR)/aiecompiler_x86.log | tail -1

# conv2d weight channel under test. NOT a free choice: on the s6 patch ReLU never
# fires for ch0 — nor for 12 of the 16 channels — so KUT_CH=0 cannot tell
# CONV_RELU=1 from CONV_RELU=0. Use KUT_CH=11 for that: it is the only channel
# where ReLU clamps SOME but not all pixels (12434 of 16384).
KUT_CH ?= 0

x86sim_check: x86sim_graph gen_vectors
	cd $(X86_BUILD_DIR) && AIESIM_SCENARIO_DIR=$(SCENARIO_DATA_DIR) \
	    KERNEL_OUT_DIR=$(PROJECT_REPO)/$(X86_BUILD_DIR) \
	    KUT_CH=$(KUT_CH) \
	    timeout 600 x86simulator $(X86_SIM_FLAGS) 2>&1 | tee x86sim.log
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    GEN_PATCH_ROWS=$(PATCH_ROWS) \
	    GEN_PATCH_COLS=$(PATCH_COLS) \
	    GEN_H_SHIFT=$(H_SHIFT) \
	    uv run python3 scripts/check_kernel_bitexact.py \
	        --kernel $(KUT) \
	        --scenario $(SCENARIO_DATA_DIR) \
	        --ch $(KUT_CH) --relu $(CONV_RELU) \
	        --actual $(X86_BUILD_DIR)/kernel_out.bin

# -------------------------------------------------------
# System link
# -------------------------------------------------------
xsa: $(BUILD_DIR)/$(XSA)

$(BUILD_DIR)/$(XSA): $(KERNEL_XOS) $(LIBADF_A)
	v++ $(VPP_FLAGS) $(VPP_LINK_FLAGS) -l \
	    $(KERNEL_XOS) $(LIBADF_A) \
	    -o $(BUILD_DIR)/$(XSA) 2>&1 | tee $(BUILD_DIR)/vpp_link.log

# -------------------------------------------------------
# Host application
# -------------------------------------------------------
application: $(BUILD_DIR)/$(APP_ELF)

# The ELF gets a flag stamp for the same reason libadf.a does: N_CHANNELS,
# ITER_CNT, IMPULSE_DR/DC and now CMUL_H_SHIFT are make variables that touch no
# source file, so a source-only prerequisite list silently reuses the previous
# build's binary. That trap already existed here; adding CMUL_H_SHIFT (which the
# host uses to scale H) makes it load-bearing.
APP_FLAGS_STAMP := $(BUILD_DIR)/app.flagstamp
$(APP_FLAGS_STAMP): FLAGS_FOR_STAMP := $(GCC_FLAGS)

$(BUILD_DIR)/$(APP_ELF): $(APP_FLAGS_STAMP)                 \
                         $(HOST_APP_SRC)/mosse_tracker.cpp  \
                         $(HOST_APP_SRC)/mosse_filter.cpp   \
                         $(HOST_APP_SRC)/mosse_filter.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(GCC_FLAGS) $(GCC_INC) \
	    $(HOST_APP_SRC)/mosse_tracker.cpp $(HOST_APP_SRC)/mosse_filter.cpp \
	    $(GCC_LIBS) -o $@

# -------------------------------------------------------
# Native host-side unit test
# -------------------------------------------------------
# The filter init/update math is the one part of the host app that can be tested
# without hardware: mosse_filter.{h,cpp} deliberately includes NO XRT or ADF
# header, so it builds with the system g++ and runs in seconds. The alternative
# is a ~90 min hw_emu frame or a multi-hour aiesim, which is no way to debug
# arithmetic.
#
# The golden data is regenerated every run so the reference and the code cannot
# drift apart the way the shift budget and the vector generator once did.
TEST_HOST_DIR := $(HOST_APP_SRC)/test

.PHONY: test_host
test_host:
	mkdir -p $(BUILD_DIR)
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    GEN_H_SHIFT=$(H_SHIFT) \
	    uv run python3 scripts/gen_filter_golden.py $(TEST_HOST_DIR)/golden
	g++ -O2 -std=c++17 -Wall -Wextra -I$(HOST_APP_SRC) \
	    -DCMUL_H_SHIFT=$(H_SHIFT) -DPSR_GATE_MIN=$(PSR_GATE_MIN) \
	    -DMOSSE_SIGMA=$(MOSSE_SIGMA) -DSIGMA_FACTOR=$(SIGMA_FACTOR) \
	    -DSIGMA_FROM_TARGET=$(SIGMA_FROM_TARGET) -DMOSSE_ETA=$(MOSSE_ETA) \
	    -DTARGET_PADDING=$(TARGET_PADDING) \
	    -DSCALE_N=$(SCALE_N) -DSCALE_STEP=$(SCALE_STEP) -DSCALE_ETA=$(SCALE_ETA) \
	    -DSCALE_SIGMA_FACTOR=$(SCALE_SIGMA_FACTOR) -DSCALE_TMPL_AREA=$(SCALE_TMPL_AREA) \
	    $(HOST_APP_SRC)/mosse_filter.cpp $(TEST_HOST_DIR)/test_mosse_filter.cpp \
	    -o $(BUILD_DIR)/test_host
	$(BUILD_DIR)/test_host $(TEST_HOST_DIR)/golden

# -------------------------------------------------------
# Native roi_crop test — the RESAMPLE path
# -------------------------------------------------------
# roi_crop.cpp includes only ap_int.h / hls_stream.h / ap_axi_sdata.h and uses
# std::sqrt rather than hls::rsqrtf specifically so it stays C-simulatable, and
# the 2025.2 HLS headers are header-only and live under $(XILINX_VITIS)/include.
# So the kernel builds with the system g++ and runs in seconds.
#
# WHY THIS TARGET EXISTS. Until 2026-08-16 CLAUDE.md claimed roi_crop had been
# "verified bit-exact against a NumPy reference in native C simulation (6 cases)".
# No such harness was ever in the repo — no testbench, no csim target, no
# reference, nothing in git history. And every build to date runs
# roi_h == patch_rows, which makes step_y exactly 256, hence fy == fx == 0, hence
# the whole bilinear datapath collapses to `pix = p00`. The interpolator had never
# executed. 11 of the 17 cases here are its first execution.
#
# The golden comes from scripts/roi_crop_ref.py, which scripts/phase1_sweep.py
# also uses to model the ROI — one reference, two consumers, so the sweep's
# padding numbers inherit whatever this target proves. Regenerated every run, same
# anti-drift reason as test_host above.
TEST_PL_DIR := $(PL_SRC_REPO)/test

.PHONY: test_roi_crop
test_roi_crop:
	mkdir -p $(BUILD_DIR)
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    uv run python3 scripts/gen_roi_crop_golden.py $(TEST_PL_DIR)/golden
	g++ -O2 -std=c++17 -Wall -Wextra -Wno-comment -Wno-unknown-pragmas \
	    -I$(XILINX_VITIS)/include -I$(PL_SRC_REPO)/roi_crop \
	    $(PL_SRC_REPO)/roi_crop/roi_crop.cpp $(TEST_PL_DIR)/test_roi_crop.cpp \
	    -o $(BUILD_DIR)/test_roi_crop
	$(BUILD_DIR)/test_roi_crop $(TEST_PL_DIR)/golden

# -------------------------------------------------------
# Package
# -------------------------------------------------------
EMBEDDED_PACKAGE_OUT := $(BUILD_DIR)/package

# Feature-downgraded rootfs copy that v++ can package without corrupting it.
# See the ROOTFS comment near the top of this file for why this is necessary.
.PHONY: rootfs
rootfs: $(ROOTFS)
$(ROOTFS): $(COMMON_IMAGE_VERSAL)/rootfs.ext4
	mkdir -p $(ROOTFS_DIR)
	cp $< $@
	tune2fs -O ^orphan_file,^metadata_csum_seed,^metadata_csum $@
	e2fsck -fy $@ || test $$? -lt 4
	@echo "rootfs: $@ ready for packaging"

# The on-target run script is GENERATED, not copied, because XCL_EMULATION_MODE
# must be set for hw_emu and must NOT be set on real hardware. Packaging the
# template verbatim (as this did until 2026-08-16) bakes "hw_emu" into a board
# image, where it makes XRT open the emulation driver instead of the device.
$(BUILD_DIR)/run_script.sh: $(EXEC_SCRIPTS)/run_script.sh
	mkdir -p $(BUILD_DIR)
	sed 's|@EMU_MODE@|$(if $(filter hw_emu,$(TARGET)),hw_emu,)|' $< > $@
	chmod +x $@

package: $(BUILD_DIR)/$(APP_ELF) $(BUILD_DIR)/$(XSA) $(LIBADF_A) $(ROOTFS) \
         $(BUILD_DIR)/run_script.sh
	v++ --package $(VPP_FLAGS) \
	    --package.rootfs $(ROOTFS) \
	    --package.kernel_image $(COMMON_IMAGE_VERSAL)/Image \
	    --package.boot_mode=sd \
	    --package.out_dir $(EMBEDDED_PACKAGE_OUT) \
	    --package.image_format=ext4 \
	    --package.sd_file $(BUILD_DIR)/$(APP_ELF) \
	    --package.sd_file $(BUILD_DIR)/$(XSA) \
	    --package.sd_file $(LIBADF_A) \
	    --package.sd_file $(AIE_SRC_REPO)/weights/layer0_weights.bin \
	    --package.sd_file $(BUILD_DIR)/run_script.sh \
	    --package.defer_aie_run \
	    $(BUILD_DIR)/$(XSA) $(LIBADF_A)

sd_card: kernels graph xsa application package

# -------------------------------------------------------
# Emulation
# -------------------------------------------------------
run_emu:
ifeq ($(LAUNCH_HW_EMU_EXEC),1)
	cd $(EMBEDDED_PACKAGE_OUT) && ./launch_hw_emu.sh -no-reboot -run-app run_script.sh 2>&1 | tee run_emu.log
else
	@echo "Set LAUNCH_HW_EMU_EXEC=1 to auto-launch emulation"
	@echo "Or run manually: cd $(EMBEDDED_PACKAGE_OUT) && ./launch_hw_emu.sh"
endif

# -------------------------------------------------------
# Waveform probing for the MAIN design
#
#   make debug_sim                 # re-elaborate this package with trace enabled
#   make probe_emu                 # run it with the roi_crop AXIS probe armed
#
# `debug_sim` must be re-run after every `make package` — packaging regenerates
# elaborate.sh with --debug off and a fresh (untraced) snapshot.
#
# PROBE_CU/PROBE_PORT default to roi_crop_0 / patch_out, i.e. the PL->AIE PLIO
# link. Override to probe a different kernel:
#   make probe_emu PROBE_CU=camera_capture_0 PROBE_PORT=<axis port>
# -------------------------------------------------------
XSIM_DIR   := $(EMBEDDED_PACKAGE_OUT)/sim/behav_waveform/xsim
PROBE_CU   ?= roi_crop_0
PROBE_PORT ?= patch_out
# roi_crop packs 4 int8 pixels per 32-bit AXIS word (conv2d reads int32), so a
# full patch is PATCH_ROWS*PATCH_COLS/4 beats — 4096 at 128x128.
PROBE_BEATS := $(shell echo $$(( $(PATCH_ROWS) * $(PATCH_COLS) / 4 )))

.PHONY: debug_sim probe_emu probe_report
debug_sim:
	$(call REELABORATE_WITH_DEBUG,$(XSIM_DIR),$(BUILD_DIR))

probe_emu:
	rm -f $(XSIM_DIR)/plio_probe.vcd
	@grep -q -- '--debug typical' $(XSIM_DIR)/elaborate.sh 2>/dev/null || \
	    { echo "ERROR: snapshot has no trace info — run 'make debug_sim' first"; exit 1; }
	cd $(EMBEDDED_PACKAGE_OUT) && USER_PRE_SIM_SCRIPT=$(EXEC_SCRIPTS)/plio_probe.tcl \
	    PROBE_CU=$(PROBE_CU) PROBE_PORT=$(PROBE_PORT) \
	    ./launch_hw_emu.sh -no-reboot -run-app run_script.sh 2>&1 | tee run_emu.log
	$(MAKE) probe_report

# Analyse whatever the probe captured. Safe to run on a truncated VCD from a
# killed emulation — that is the normal case when chasing a hang.
probe_report:
	python3 $(PROJECT_REPO)/scripts/analyze_plio_vcd.py \
	    $(XSIM_DIR)/plio_probe.vcd $(PROBE_BEATS)

# =========================================================
# PLIO smoke test — minimal PL --AXIS--> PLIO --> AIE --> GMIO --> DDR
#
# Answers one question: can a PLIO deliver data from PL into an AIE kernel at
# all in this toolchain/platform? The full design hangs because conv2d never
# receives a word from its PatchIn PLIO. This strips everything else away.
#
#   make smoke_sd_card TARGET=hw_emu
#   make smoke_run_emu LAUNCH_HW_EMU_EXEC=1
# =========================================================
SMOKE_N        := 256
SMOKE_BUILD    := build/$(TARGET)/plio_smoke
SMOKE_WORK     := $(SMOKE_BUILD)/Work
SMOKE_XO       := $(SMOKE_BUILD)/stream_src.$(TARGET).xo
SMOKE_LIBADF   := $(SMOKE_BUILD)/libadf.a
SMOKE_XSA      := plio_smoke.$(TARGET).xsa
SMOKE_ELF      := plio_smoke.elf
SMOKE_PKG      := $(SMOKE_BUILD)/package

# AIE flags: reuse the main ones but swap workdir, drop the PatchIn constraint
# (SmokeIn is intentionally left unconstrained so the compiler places it).
SMOKE_AIE_FLAGS := $(filter-out --workdir=$(WORK_DIR),$(AIE_FLAGS))
SMOKE_AIE_FLAGS := $(filter-out --constraints $(AIE_SRC_REPO)/constraints.aiecst,$(SMOKE_AIE_FLAGS))
SMOKE_AIE_FLAGS += --Xpreproc="-DSMOKE_N=$(SMOKE_N)"
# SMOKE_SKIP_STREAM=1 builds the bisect variant: the kernel emits the expected
# pattern without reading the PLIO stream. Distinguishes "PLIO never delivers"
# from "the AIE core never runs" — see smoke_passthrough.cpp.
SMOKE_SKIP_STREAM ?= 0
SMOKE_AIE_FLAGS += --Xpreproc="-DSMOKE_SKIP_STREAM=$(SMOKE_SKIP_STREAM)"
SMOKE_AIE_FLAGS += --workdir=$(SMOKE_WORK)

# Flag stamp for the smoke graph. Must be defined HERE, after SMOKE_BUILD and
# SMOKE_AIE_FLAGS exist — see the note next to AIE_FLAGS_STAMP.
SMOKE_FLAGS_STAMP := $(SMOKE_BUILD)/aie.flagstamp
$(SMOKE_FLAGS_STAMP): FLAGS_FOR_STAMP := $(SMOKE_AIE_FLAGS)

SMOKE_VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) --save-temps --temp_dir $(SMOKE_BUILD)/_x

.PHONY: smoke_kernels smoke_graph smoke_xsa smoke_app smoke_package smoke_sd_card smoke_run_emu

smoke_kernels: $(SMOKE_XO)
$(SMOKE_XO): $(PL_SRC_REPO)/stream_src/stream_src.cpp
	mkdir -p $(SMOKE_BUILD)
	v++ $(SMOKE_VPP_FLAGS) --hls.clock $(VPP_CLOCK_FREQ):stream_src -c -k stream_src $< -o $@

smoke_graph: $(SMOKE_LIBADF)
$(SMOKE_LIBADF): $(SMOKE_FLAGS_STAMP)                 \
                 $(AIE_SRC_REPO)/plio_smoke_graph.cpp \
                 $(AIE_SRC_REPO)/plio_smoke_graph.h   \
                 $(AIE_SRC_REPO)/smoke_passthrough.cpp \
                 $(AIE_SRC_REPO)/smoke_passthrough.h
	mkdir -p $(SMOKE_BUILD)
	cd $(SMOKE_BUILD) && aiecompiler $(SMOKE_AIE_FLAGS) \
	    $(AIE_SRC_REPO)/plio_smoke_graph.cpp 2>&1 | tee aiecompiler.log

smoke_xsa: $(SMOKE_BUILD)/$(SMOKE_XSA)
$(SMOKE_BUILD)/$(SMOKE_XSA): $(SMOKE_XO) $(SMOKE_LIBADF)
	v++ $(SMOKE_VPP_FLAGS) --vivado.synth.jobs 8 \
	    --config $(SYS_CONFIGS)/plio_smoke.cfg \
	    --clock.freqHz $(VPP_CLOCK_FREQ):stream_src_0 \
	    -l $(SMOKE_XO) $(SMOKE_LIBADF) \
	    -o $(SMOKE_BUILD)/$(SMOKE_XSA) 2>&1 | tee $(SMOKE_BUILD)/vpp_link.log

smoke_app: $(SMOKE_BUILD)/$(SMOKE_ELF)
$(SMOKE_BUILD)/$(SMOKE_ELF): $(HOST_APP_SRC)/plio_smoke_host.cpp
	mkdir -p $(SMOKE_BUILD)
	$(CXX) -O2 -std=c++17 -D__linux__ -D__PS_ENABLE_AIE__ -DSMOKE_N=$(SMOKE_N) \
	    -I$(SDKTARGETSYSROOT)/usr/include/xrt -I$(XILINX_VITIS)/aietools/include/ \
	    -I$(SDKTARGETSYSROOT)/usr/include -I$(AIE_SRC_REPO) \
	    $< -L$(SDKTARGETSYSROOT)/usr/lib -L$(XILINX_VITIS)/aietools/lib/aarch64.o \
	    -ladf_api_xrt -lxrt_coreutil -o $@

smoke_package: $(SMOKE_BUILD)/$(SMOKE_ELF) $(SMOKE_BUILD)/$(SMOKE_XSA) $(SMOKE_LIBADF) $(ROOTFS)
	v++ --package $(SMOKE_VPP_FLAGS) \
	    --package.rootfs $(ROOTFS) \
	    --package.kernel_image $(COMMON_IMAGE_VERSAL)/Image \
	    --package.boot_mode=sd \
	    --package.out_dir $(SMOKE_PKG) \
	    --package.image_format=ext4 \
	    --package.sd_file $(SMOKE_BUILD)/$(SMOKE_ELF) \
	    --package.sd_file $(EXEC_SCRIPTS)/run_smoke.sh \
	    --package.defer_aie_run \
	    $(SMOKE_BUILD)/$(SMOKE_XSA) $(SMOKE_LIBADF)

smoke_sd_card: smoke_kernels smoke_graph smoke_xsa smoke_app smoke_package

SMOKE_XSIM_DIR := $(SMOKE_PKG)/sim/behav_waveform/xsim

# =========================================================
# Waveform-debug re-elaboration (shared by both designs)
#
# v++ --package generates elaborate.sh with `xelab --incr --debug off`, so xsim
# has NO trace information: open_vcd/log_vcd fail with
#   [Simulator 45-10] The current simulation was compiled without trace information
# and since simulate.sh passes `-onerror quit`, that error aborts the emulation
# before Linux even boots. Signal probing is impossible without this step.
#
# `--incr` is dropped deliberately: incremental reuse across a changed --debug
# setting is what produces a snapshot that still lacks trace data.
#
# Also: elaborate.sh carries RELATIVE include paths (../../../../prj.ip_user_files/...)
# that resolve only in the Vivado project tree where v++ originally ran it. In the
# copied package/ tree they dangle, and xelab then fails to compile the SystemC
# interface ("vitis_design_CIPS_0_0.h: No such file or directory") while STILL
# printing "Built simulation snapshot" — a broken snapshot that looks successful.
# From <xsim dir>, ../../../../ is the build dir, so the prj.* trees are linked there.
#
# Must run AFTER the package step (packaging regenerates elaborate.sh, dropping
# the patch and the fresh snapshot).
#
#   $(1) = xsim directory      $(2) = build directory holding _x/
# =========================================================
define REELABORATE_WITH_DEBUG
	@test -f $(1)/elaborate.sh || \
	    { echo "ERROR: $(1)/elaborate.sh missing — run the package step first"; exit 1; }
	cp -f $(1)/elaborate.sh $(1)/elaborate.sh.orig
	sed -i 's/xelab --incr --debug off /xelab --debug typical /g' $(1)/elaborate.sh
	@grep -q -- '--debug typical' $(1)/elaborate.sh || \
	    { echo "ERROR: elaborate.sh did not match the expected xelab flags — inspect it by hand"; exit 1; }
	ln -sfn _x/link/vivado/vpl/prj/prj.ip_user_files $(2)/prj.ip_user_files
	ln -sfn _x/link/vivado/vpl/prj/prj.gen           $(2)/prj.gen
	rm -rf $(1)/xsim.dir/tb_behav
	cd $(1) && ./elaborate.sh 2>&1 | tee elaborate_debug.log
	@grep -qi "No such file or directory" $(1)/elaborate_debug.log && \
	    { echo "ERROR: elaboration still has unresolved includes — snapshot is not trustworthy"; exit 1; } || true
endef

.PHONY: smoke_debug_sim smoke_probe_emu
smoke_debug_sim:
	$(call REELABORATE_WITH_DEBUG,$(SMOKE_XSIM_DIR),$(SMOKE_BUILD))

# Run the smoke emulation with the PL->AIE AXIS handshake probe armed.
# Requires smoke_debug_sim to have run against the current package.
smoke_probe_emu:
	rm -f $(SMOKE_XSIM_DIR)/plio_probe.vcd
	cd $(SMOKE_PKG) && USER_PRE_SIM_SCRIPT=$(EXEC_SCRIPTS)/plio_probe.tcl \
	    PROBE_CU=stream_src_0 PROBE_PORT=out_r \
	    ./launch_hw_emu.sh -no-reboot -run-app run_smoke.sh 2>&1 | tee run_smoke.log
	python3 $(PROJECT_REPO)/scripts/analyze_plio_vcd.py \
	    $(SMOKE_XSIM_DIR)/plio_probe.vcd $(SMOKE_N)

smoke_run_emu:
ifeq ($(LAUNCH_HW_EMU_EXEC),1)
	cd $(SMOKE_PKG) && ./launch_hw_emu.sh -no-reboot -run-app run_smoke.sh 2>&1 | tee run_smoke.log
else
	@echo "Set LAUNCH_HW_EMU_EXEC=1 to auto-launch the smoke emulation"
endif

# -------------------------------------------------------
# Clean
# -------------------------------------------------------
cleanall:
	rm -rf build/
