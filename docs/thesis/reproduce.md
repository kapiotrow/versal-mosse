# Reproduction appendix

What a reader needs to rebuild and re-run each arm. Draft — the `TODO`s are the parts only you
can supply; everything else is transcribed from `CLAUDE.md` and `scripts/vot_sweep.sh` and is
believed correct as of 2026-08-30.

## 1. Environment

| item | value |
|---|---|
| Vitis / Vivado | 2025.2 (older versions may not work) |
| platform | `xilinx_vek280_base_202520_1.xpfm` |
| board | VEK280, `xcve2802-vsvh1760-2MP-e-S`, AIE core clock 1 GHz, PL 312.5 MHz |
| Linux on board | Common Versal Image 2025.2, rootfs feature-downgraded by `make rootfs` |
| host toolchain | `environment-setup-cortexa72-cortexa53-amd-linux` from `sdk.sh` |
| DSP library | Vitis Libraries **root** (the Makefile appends `/dsp`) |
| Python | `uv sync`; `uv sync --extra weights` for torch/torchvision |

```bash
source setup_env.sh     # PLATFORM_REPO_PATHS, XILINX_VITIS, COMMON_IMAGE_VERSAL, PLATFORM, DSPLIB_VITIS
```

The Vitis environment **masks the venv**: `PYTHONHOME`/`PYTHONPATH` point python at Vivado's
build, which has no `_ctypes`. Every offline script runs as
`env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python …`.

TODO: pin exact Vitis patch level, DSPLib commit, and the Common Image release used for the
results in `results/arms.csv`.

## 2. Dataset

- VOT-STb2022, 62 sequences, obtained via the VOT toolkit. `$VOT_ROOT` points at it.
- Converted to board blobs + manifests by `scripts/vot_prepare.py`, which also carries **the**
  groundtruth reduction (`reduce_box`); every other reader imports it rather than
  re-implementing the polygon rule.
- The sequence list and order: `runs/vot/seqs62.txt`.
- TODO: record the toolkit version and the dataset download date/checksum.

## 3. Building an arm

The defaults **are** the shipping configuration (since 2026-08-28):
`CONV_IN_CH=3  H_SHIFT=15  budget 4-4-4  MOSSE_ETA=0.05  PSR_GATE_MIN=5.0
TARGET_PADDING=2.0  SCALE_STEP=1.04  SCALE_MAX_STEP=2  HOLD_COAST=0`.

```bash
make weights                        # RGB, 27 taps, BIAS_SCALE=roi
ARM=rgb scripts/calib_build.sh      # hardware build + flagstamp verification
```

`calib_build.sh` checks the flagstamps against the intended config and refuses to declare a
build good when they disagree. Use it for anything worth hours — `runs/.last_cfg` is **not**
authoritative and has already recorded a config the binary did not have.

**Host-only vs. card-swap.** Only `H_SHIFT` and `CONV_IN_CH` reach `AIE_FLAGS`. Every arm in
`results/arms.csv` after `rgb_h15` is host-only: an `scp` of the ELF, not a re-flash.

## 4. Running a sweep

```bash
scripts/vot_sweep.sh --arm <name> --seqs car1,tiger --ingest
scripts/vot_sweep.sh --arm <name> --dry-run          # prints every remote command, runs none
```

It mounts the exports, pushes the ELF, guards the build (compares the board's `a.xclbin` md5
against the package tree), runs, collects, and ingests into a VOT workspace. It writes
`<run>/config/` with the flagstamps, the ELF md5, the xclbin md5, the weights md5 and the git
HEAD; **a missing flagstamp is fatal**, because an unstamped run cannot be stamped afterwards.

Scoring: `scripts/vot_ingest.py` → `vot analysis`. It re-derives every run name from the
sequence's anchors and checks each trajectory's length — `scan()` only notices a *missing* file,
and a wrong-length run is otherwise scored without complaint.

**Frame time is not measurable over ssh** in a way comparable to the historical runs. Quote FPS
only from a serial-console run (`picocom … | ts '%H:%M:%.S' | tee log`).

## 5. Offline benches (no hardware, seconds to minutes)

```bash
make test_host          # filter/PSR/scale/gate/training-target. Runs TWICE, the second time
                        #   with -ffp-contract=fast -- that build caught bugs -O2 missed
make test_roi_crop      # bit-exact; ROI_IN_CH is compile-time, so RUN BOTH ARMS
make test_scene ; make test_vot_source ; make test_vot_format ; make scale_sim
make x86sim_check KUT=conv2d SCENARIO=s6rgb

env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py --sequence tiger
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_ar_offline.py   # the PROXY -- see claims.md rule 2
```

## 6. Regenerating the figures

```bash
make figures     # or: env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/figs/fig_*.py
```

## 7. Known-unreproducible

- `runs/vot/` directories before `0825_1919-smoke` predate the `config/` convention and several
  are gitignored. Their numbers are not in `results/arms.csv` and should not be quoted.
- The 503 ms KDS launch cost is a **platform** defect (the CU completion interrupt is never
  delivered); reproducing it needs this base platform's device tree, and the workaround
  (`ROI_CROP_USER_MANAGED=1`) is the default.
