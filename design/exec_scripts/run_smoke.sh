#!/bin/bash

echo ""
date
echo ""

export XLC_EMULATION_MODE=hw_emu
export XILINX_XRT=/usr

./plio_smoke.elf a.xclbin

return_code=$?

if [ $return_code -ne 0 ]; then
        echo "ERROR: PLIO smoke test failed, RC=$return_code"
else
        echo "INFO: PLIO SMOKE TEST PASSED, RC=0"
fi

echo ""
date
echo ""
echo "INFO: PLIO smoke run completed."
echo ""

exit $return_code
