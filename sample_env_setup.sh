#!/bin/bash

#Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# =======================================================
# Set Platform ,Vitis and Versal Image repo
# =======================================================
export PLATFORM_REPO_PATHS=/home/karolina/Xilinx/2025.2/Vitis/base_platforms
export XILINX_VITIS=/home/karolina/Xilinx/2025.2/Vitis
export COMMON_IMAGE_VERSAL=/home/karolina/studia/MGR/xilinx-versal-common-v2025.2
# ====================================================
# Source Versal Image ,Vitis and Aietools
# ====================================================
# Run the below command to setup environment and CXX
source /opt/petalinux/2025.2/environment-setup-cortexa72-cortexa53-amd-linux
source $XILINX_VITIS/settings64.sh
# ========================================================
# Set DSP Library for Vitis
# ========================================================
export DSPLIB_VITIS=/home/karolina/studia/MGR/Vitis_Libraries
# =========================================================
# Platform Selection...
# =========================================================
tgt_plat=xilinx_vek280_base_202520_1
export PLATFORM=$PLATFORM_REPO_PATHS/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm
# ==========================================================
# Validating Tool Installation
# ==========================================================
echo ""
echo "Aiecompiler:"
which aiecompiler
echo ""
echo "Vivado:"
which vivado
echo ""
echo "Vitis:"
which vitis
echo ""
echo "Vitis HLS:"
which vitis_hls
echo ""
echo ""
echo "DSPLIBS"
echo "$DSPLIB_VITIS"
echo ""
