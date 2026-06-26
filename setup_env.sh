#!/bin/bash

# =======================================================
# Set Platform, Vitis and Versal Image repo
# =======================================================
export PLATFORM_REPO_PATHS=/home/karolina/Xilinx/2025.2/Vitis/base_platforms
export XILINX_VITIS=/home/karolina/Xilinx/2025.2/Vitis
export COMMON_IMAGE_VERSAL=/home/karolina/studia/MGR/xilinx-versal-common-v2025.2
# ========================================================
# Set DSP Library for Vitis
# ========================================================
export DSPLIB_VITIS=/home/karolina/studia/MGR/Vitis_Libraries
# =========================================================
# Platform Selection...
# =========================================================
tgt_plat=xilinx_vek280_base_202520_1
export PLATFORM=$PLATFORM_REPO_PATHS/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm

# ====================================================
# Source PetaLinux environment (sets SDKTARGETSYSROOT, CXX, etc.)
# ====================================================
# CRITICAL: unset LD_LIBRARY_PATH before sourcing PetaLinux script
# (PetaLinux safety guard exits early if LD_LIBRARY_PATH is set)
unset LD_LIBRARY_PATH
source /opt/petalinux/2025.2/environment-setup-cortexa72-cortexa53-amd-linux
if [ -z "$SDKTARGETSYSROOT" ]; then
    echo "ERROR: Failed to source PetaLinux environment script"
    echo "  Check: /opt/petalinux/2025.2/environment-setup-cortexa72-cortexa53-amd-linux"
    return 1
fi

# ====================================================
# Source Vitis settings
# ====================================================
source $XILINX_VITIS/settings64.sh

# ==========================================================
# Validate all required variables are set
# ==========================================================
missing_vars=""
for var in PLATFORM_REPO_PATHS XILINX_VITIS COMMON_IMAGE_VERSAL DSPLIB_VITIS PLATFORM SDKTARGETSYSROOT CXX; do
    if [ -z "${!var}" ]; then
        missing_vars="$missing_vars $var"
    fi
done

if [ -n "$missing_vars" ]; then
    echo "ERROR: Missing required environment variables:$missing_vars"
    return 1
fi

# ==========================================================
# Validating Tool Installation
# ==========================================================
echo ""
echo "=== Versal MOSSE Build Environment ==="
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
echo "=== Environment Variables ==="
echo "PLATFORM_REPO_PATHS: $PLATFORM_REPO_PATHS"
echo "PLATFORM: $PLATFORM"
echo "XILINX_VITIS: $XILINX_VITIS"
echo "DSPLIB_VITIS: $DSPLIB_VITIS"
echo "COMMON_IMAGE_VERSAL: $COMMON_IMAGE_VERSAL"
echo "SDKTARGETSYSROOT: $SDKTARGETSYSROOT"
echo "CXX: $CXX"
echo ""
