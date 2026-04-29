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
AIE_FLAGS  += --verbose
AIE_FLAGS  += --log-level=5
AIE_FLAGS  += --pl-freq=$(PL_FREQ)
AIE_FLAGS  += --Xchess="main:bridge.llibs=softfloat m"
AIE_FLAGS  += --workdir=$(WORK_DIR)

GRAPH_SRC_CPP := $(AIE_SRC_REPO)/mosse_graph.cpp

# Aiesimulator flags
AIE_SIM_FLAGS := --pkg-dir $(WORK_DIR)/
# TODO: add -i= pointing to aiesim input data once test vectors are prepared
# AIE_SIM_FLAGS += -i=$(AIE_SRC_REPO)/aiesim_data

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
CROP_VPP_FLAGS  += --Xpreproc="-DPATCH_ROWS=$(PATCH_ROWS)"
CROP_VPP_FLAGS  += --Xpreproc="-DPATCH_COLS=$(PATCH_COLS)"

# =========================================================
# Host application compiler flags
# =========================================================
GCC_FLAGS  := -O2 -std=c++17 -D__linux__ -D__PS_ENABLE_AIE__
GCC_FLAGS  += -DPATCH_ROWS=$(PATCH_ROWS)
GCC_FLAGS  += -DPATCH_COLS=$(PATCH_COLS)
GCC_FLAGS  += -DN_CHANNELS=$(N_CHANNELS)
GCC_FLAGS  += -DITER_CNT=$(ITER_CNT)
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
.PHONY: help kernels graph aiesim xsa application package sd_card run_emu cleanall

help:
	@echo ""
	@echo "versal-mosse build targets:"
	@echo "  make kernels      — compile PL HLS kernels"
	@echo "  make graph        — compile AIE graph"
	@echo "  make aiesim       — run AIE simulator"
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
graph: $(LIBADF_A)

$(LIBADF_A): $(AIE_SRC_REPO)/mosse_graph.cpp  \
             $(AIE_SRC_REPO)/mosse_graph.h     \
             $(AIE_SRC_REPO)/fft_graph.h       \
             $(AIE_SRC_REPO)/ifft_graph.h      \
             $(AIE_SRC_REPO)/conv2d_kernel.h   \
             $(AIE_SRC_REPO)/conv2d_kernel.cpp \
             $(AIE_SRC_REPO)/cmul_accum_kernel.h \
             $(AIE_SRC_REPO)/cmul_accum_kernel.cpp
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && aiecompiler $(AIE_FLAGS) $(GRAPH_SRC_CPP) 2>&1 | tee aiecompiler.log

aiesim: graph
	cd $(BUILD_DIR) && aiesimulator $(AIE_SIM_FLAGS) \
	    -i=$(AIE_SRC_REPO)/aiesim_data 2>&1 | tee aiesim.log

# -------------------------------------------------------
# System link
# -------------------------------------------------------
xsa: $(BUILD_DIR)/$(XSA)

$(BUILD_DIR)/$(XSA): $(KERNEL_XOS) $(LIBADF_A)
	cd $(BUILD_DIR) && \
	v++ $(VPP_FLAGS) $(VPP_LINK_FLAGS) -l \
	    $(KERNEL_XOS) $(LIBADF_A) \
	    -o $(XSA) 2>&1 | tee vpp_link.log

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

package: $(BUILD_DIR)/$(APP_ELF) $(BUILD_DIR)/$(XSA) $(LIBADF_A)
	v++ --package $(VPP_FLAGS) \
	    --package.rootfs $(COMMON_IMAGE_VERSAL)/rootfs.ext4 \
	    --package.kernel_image $(COMMON_IMAGE_VERSAL)/Image \
	    --package.boot_mode=sd \
	    --package.out_dir $(EMBEDDED_PACKAGE_OUT) \
	    --package.image_format=ext4 \
	    --package.sd_file $(BUILD_DIR)/$(APP_ELF) \
	    --package.sd_file $(BUILD_DIR)/$(XSA) \
	    --package.sd_file $(LIBADF_A) \
	    --package.sd_file $(EXEC_SCRIPTS)/run_script.sh \
	    --package.defer_aie_run \
	    $(BUILD_DIR)/$(XSA) $(LIBADF_A)

sd_card: kernels graph xsa application package

# -------------------------------------------------------
# Emulation
# -------------------------------------------------------
run_emu:
ifeq ($(LAUNCH_HW_EMU_EXEC),1)
	cd $(EMBEDDED_PACKAGE_OUT) && ./launch_hw_emu.sh -noc-ddr-only 1 -run-app $(EXEC_SCRIPTS)/run_script.sh 2>&1 | tee run_emu.log
else
	@echo "Set LAUNCH_HW_EMU_EXEC=1 to auto-launch emulation"
	@echo "Or run manually: cd $(EMBEDDED_PACKAGE_OUT) && ./launch_hw_emu.sh"
endif

# -------------------------------------------------------
# Clean
# -------------------------------------------------------
cleanall:
	rm -rf build/
