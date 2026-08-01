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
FFT_ROW_WS          := 2
FFT_COL_WS          := 2

# FFT/IFFT output shifts. The invariant is
#     2*FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT = 12
# which fixes the response scale, so weight can be moved between the forward and
# inverse passes without recalibrating any expected value.
#
# DEFAULT 4/2/2 is the REAL design point: validated with s6 at CONV2D_MODE=0 and
# N_CHANNELS=16 (accum 7728, row IFFT 8805, response 6692, err=0 px, no stage above
# 27% of cint16). Do not set IFFT_ROW_SHIFT=0 at high channel counts — the row IFFT
# takes the accumulated spectrum and overflows (~101000) with no attenuation.
#
# ECHO-MODE SCENARIOS (s0-s4, CONV2D_MODE=1) need 0/0/12 instead: their magnitudes
# are ~100, and a forward shift of 4 divides the spectrum by 256 and crushes them to
# zero. Run them as:
#   make aiesim_plio CONV2D_MODE=1 SCENARIO=s1 FFT_SHIFT=0 IFFT_ROW_SHIFT=0 IFFT_COL_SHIFT=12
FFT_SHIFT           ?= 4
IFFT_ROW_SHIFT      ?= 2
IFFT_COL_SHIFT      ?= 2

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
AIE_FLAGS  += --Xpreproc="-DITER_CNT=$(ITER_CNT)"
# FFT/IFFT normalization shifts. IFFT_COL_SHIFT is the consequential one: it was
# calibrated for BROADBAND spectra and destroys DC-concentrated ones (see the
# warning in ifft_graph.h). SINGLE SOURCE OF TRUTH — the same values are passed to
# gen_aiesim_vectors.py in the gen_vectors target, because the expected peak values
# scale by 2^(12-IFFT_COL_SHIFT). If the graph and the vectors ever disagree about
# the shift, every expected value is silently wrong.
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_SHIFT=$(FFT_SHIFT)"
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_IFFT_ROW_SHIFT=$(IFFT_ROW_SHIFT)"
AIE_FLAGS  += --Xpreproc="-DFFT_2D_TP_IFFT_COL_SHIFT=$(IFFT_COL_SHIFT)"
# conv2d build mode: 0=real conv, 1=echo stream, 2=synthesize without reading the
# stream (bisect for "is conv2d blocked on readincr?"). See conv2d_kernel.cpp.
CONV2D_MODE ?= 1
AIE_FLAGS  += --Xpreproc="-DCONV2D_ECHO_TEST=$(CONV2D_MODE)"
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
#            s4 (Gaussian filter), s6 (FULL preprocessing path — Stage A -> conv2d -> B1)
# s0-s4 are raw-patch scenarios: run them with CONV2D_MODE=1 (echo). Only s6 feeds a
# Stage-A-preprocessed patch, so it is the only valid scenario for CONV2D_MODE=0.
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
SIM_WALL_TIMEOUT  ?= $(shell expr 1200 \* $(N_CHANNELS))
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

GCC_INC    := -I$(SDKTARGETSYSROOT)/usr/include/xrt
GCC_INC    += -I$(XILINX_VITIS)/aietools/include/
GCC_INC    += -I$(SDKTARGETSYSROOT)/usr/include
GCC_INC    += -I$(AIE_SRC_REPO)
GCC_INC    += -I$(HOST_APP_SRC)
GCC_INC    += -I$(DSPLIB_ROOT)/L2/include/aie
# TODO: add KissFFT include path for PS-side filter update
# GCC_INC  += -I$(KISSFFT_ROOT)

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
.PHONY: help kernels graph gen_vectors aiesim graph_fft aiesim_fft xsa application package sd_card run_emu weights cleanall

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
	@echo "  make package      — package SD card image"
	@echo "  make sd_card      — kernels + graph + xsa + application + package"
	@echo "  make run_emu      — launch hw emulator"
	@echo "  make cleanall     — remove all build outputs"
	@echo ""
	@echo "Key parameters (pass on command line):"
	@echo "  TARGET=$(TARGET)  PATCH_ROWS=$(PATCH_ROWS)  PATCH_COLS=$(PATCH_COLS)"
	@echo "  N_CHANNELS=$(N_CHANNELS)  FFT_2D_DT=$(FFT_2D_DT)  ITER_CNT=$(ITER_CNT)"

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

$(BUILD_DIR)/$(APP_ELF): $(HOST_APP_SRC)/mosse_tracker.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(GCC_FLAGS) $(GCC_INC) $< $(GCC_LIBS) -o $@

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

package: $(BUILD_DIR)/$(APP_ELF) $(BUILD_DIR)/$(XSA) $(LIBADF_A) $(ROOTFS)
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
	    --package.sd_file $(EXEC_SCRIPTS)/run_script.sh \
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
