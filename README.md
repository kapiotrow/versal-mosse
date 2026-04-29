# MOSSE correlation filter tracker on convolutional features on AMD versal
This project aims to accelerate a correlarion-filter based visual tracker on AMD Versal, taking advantage of the AI Engines for FFT/IFFT calculation and the PL for the conv layer and correlation calculation.

## Use 
**This project is under construction and may not work.**

Download the Common Versal Image, extract and run `./sdk.sh`. Wait for the script to finish. Note where the `environment-setup-cortexa72-cortexa53-amd-linux` script was written to. Clone the Vitis Libraries repository. Locate the base platforms (should be inside your Vivado/Vitis install directory under `/Xilinx/2025.2/Vitis/base_platforms`). Set the proper paths inside `setup_env.sh` (`PLATFORM_REPO_PATHS`, `XILINX_VITIS`, `COMMON_IMAGE_VERSAL`, setup script created byt `sdk.sh`, `DSPLIB_VITIS`).

Reopen the repo in Vitis (version used for development is 2025.2, older verisons may not work). Open a new terminal and run
```
. setup_env.sh
```
Check if the script outputs correct paths.
To run the AI Engine simulaiton, run
```
make aieism
```
The simulation takes a while to complete.