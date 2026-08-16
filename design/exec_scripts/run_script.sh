#!/bin/bash
#Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

echo ""
date
echo ""
SECONDS=0

# Emulation mode. @EMU_MODE@ is substituted by the Makefile at package time:
# "hw_emu" for TARGET=hw_emu, EMPTY for TARGET=hw.
#
# This MUST NOT be set on real hardware — it tells XRT to open the emulation
# driver instead of the device. The line used to read
#     export XLC_EMULATION_MODE=hw_emu
# unconditionally, which was wrong twice over: it would have broken a board run,
# and XLC_ is a typo for XRT's XCL_. The misspelling is why hw_emu worked anyway
# — the variable as written was inert, and XRT inferred emulation from the
# packaged xclbin. Both halves are fixed here; if a hw_emu run now behaves
# differently, this correctly-spelled variable is the thing that changed.
EMU_MODE="@EMU_MODE@"
if [ -n "$EMU_MODE" ]; then
    export XCL_EMULATION_MODE=$EMU_MODE
    echo "INFO: XCL_EMULATION_MODE=$XCL_EMULATION_MODE"
else
    unset XCL_EMULATION_MODE
    echo "INFO: real hardware — XCL_EMULATION_MODE not set"
fi
export XILINX_XRT=/usr

# Executing the elf...
./mosse_tracker.elf a.xclbin

return_code=$?

if [ $return_code -ne 0 ]; then
        echo "ERROR: Embedded host run failed, RC=$return_code"
else
        echo "INFO: TEST PASSED, RC=0"
fi

duration=$SECONDS

echo ""
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo ""
date
echo ""
echo "INFO: Embedded host run completed."
echo ""

exit $return_code
