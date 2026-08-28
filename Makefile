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
# 8 -> 16 ON HARDWARE 2026-08-20 (runs/run_0820_1716.log): frame 127.7 -> 89.5 ms,
# 7.83 -> 11.17 FPS, tracking bit-identical. The row-FFT drain is a FIXED PER-
# BARRIER cost, proved by the per-tx figure not moving when the payload doubled
# (286.10 us/tx at 4096 B -> 289.09 us/tx at 8192 B), so halving the barrier count
# halves the time. gmio_fft_row_out 73.22 -> 37.00 ms, gmio_weights 3.67 -> 1.88.
# Next step is 32 (CONV_OUT_CHUNK 16 KB, 32 KB ping-pong, still under the 64 KB
# tile limit); 64 would need 64 KB ping-pong and is the likely ceiling.
# 16 -> 32 ON HARDWARE 2026-08-20 (runs/run_0820_1739.log): 89.5 -> 70.9 ms,
# 11.17 -> 14.10 FPS, tracking bit-identical. us/tx held at 293.38 for a 16384 B
# payload against 286.10 at 4096 B, so the fixed-per-barrier model holds over 4x.
# 64 would need a 64 KB ping-pong (the whole tile) and is the likely ceiling.
# NOTE it is not a free knob for every port: gmio_ifft_row_out got 4x WORSE
# (0.148 -> 0.596 ms/frame) because it was already at the ~18 us/tx floor.
# 32 -> 64 ON HARDWARE 2026-08-20 (runs/run_0820_1807.log): 70.9 -> 60.7 ms,
# 14.10 -> 16.48 FPS, tracking bit-identical. us/tx 293.38 -> 310.28, i.e. the
# flat per-barrier cost is finally bending (growth per doubling 0.7%, 1.8%, 5.8%)
# as the AIE's own ~6.4 ms/frame of compute surfaces. THIS KNOB IS EXHAUSTED:
# WS=128 means a 64 KB window / 128 KB ping-pong for at most 3-4 ms.
# A 64 KB ping-pong DOES fit a 64 KB tile — AIE-ML cores address neighbouring
# tiles' memory. Test placement with `make graph` (3 min), not `make sd_card`.
FFT_ROW_WS          := 64
# 8, and DO NOT RAISE IT WITHOUT READING "FFT_COL_WS 8->32 IS A NET LOSS" in
# CLAUDE.md. 32 was tried on hardware 2026-08-21 and cost 9.57 ms/frame: the pair
# gmio_fft_col_out + gmio_accum_out went 9.00 -> 18.32 ms even though the
# transaction count fell 4x, because gmio_fft_col_out's per-tx cost exploded
# 17.86 -> 266.63 us. The knob that took gmio_fft_row_out 73 -> 9.9 ms does NOT
# transfer to this port pair, and the mechanism is not yet understood.
FFT_COL_WS          := 8

# Do the row->col transposes in AIE-ML memory tiles instead of the DDR round trip
# through the APU. 0 = the DDR path every recorded run used. EXPERIMENTAL: the
# graph side is written and mapped, the host side is NOT — with this at 1 the
# four transpose GMIOs cease to exist and mosse_tracker.cpp will not link.
# `make graph MEMTILE_TRANSPOSE=1` is the placement check; see CLAUDE.md.
# 1 since 2026-08-21: proven on hardware (45.60 -> 35.58 ms, tracking unchanged).
# Promoted to the default so it cannot be forgotten on one of the two toolchains
# — a mismatch is a board deadlock, not a compile error. Set 0 for the DDR path.
MEMTILE_TRANSPOSE   ?= 1

# Launch roi_crop one channel ahead so its ~325 us/channel of PL execution hides
# behind the host's APU work. Host-only, and inert unless MEMTILE_TRANSPOSE=1 —
# on the DDR path the row-FFT drain already covers the CU. Set 0 to bisect.
ROI_CROP_PIPELINE   ?= 1

# Give cmul_accum's accum_prev its own input port instead of packing it behind H.
# Deletes the host's 2 MB/frame packing memcpy, which IS an uncached BO read
# (2.871 ms measured). NOTE: `make aiesim` needs 0 — the single-port design works
# around a cycle-approximate ISS deadlock that does not exist on hardware.
CMUL_SPLIT_ACCUM    ?= 1

# Regroup cmul's accumulator output through a memory tile so gmio_accum_out is
# drained once per CHANNEL (16 tx/frame) instead of once per chunk (256 tx).
# Its cost is per-transaction, not per-byte — the DMA probe measured 14.4 us for
# 64 B against 22.8 us for 128 KB — so this is ~4.31 -> ~0.3 ms. It also
# decouples the accumulator from gmio_fft_col_out, which is what made
# FFT_COL_WS=32 a net loss even though accum_out won there.
# 0 — TRIED AND REVERTED 2026-08-21. It did exactly what it was designed to do
# (gmio_accum_out 256 tx -> 16) and saved NOTHING: per-tx cost went 16.83 ->
# 280.62 us, total 4.310 -> 4.490 ms, frame 29.61 -> 29.97. That port's cost was
# never per-barrier overhead — it is the host waiting for the AIE to produce the
# accumulator. See "CMUL_ACCUM_MEMTILE IS A NET LOSS" in CLAUDE.md before
# reaching for this again.
CMUL_ACCUM_MEMTILE  ?= 0

# Run the translation-filter update on the SECOND A72 core while the main thread
# does the scale filter. They are independent after the PSR — disjoint state, no
# shared buffers — so this is pure overlap with no accuracy change. Worth ~2.6 ms
# of the 9.25 ms frame tail. The board has two cores (boot log: "SMP: Total of 2
# processors activated") and the host has used one of them all along.
# NOTE: publish (pack + sync) stays on the MAIN thread deliberately — it is the
# only part of the tail that touches XRT, and keeping XRT single-threaded is a
# discipline this design should not give up for 1.9 ms.
TAIL_PARALLEL       ?= 1



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
#
# 4-5-5 (TOTAL 18) as of 2026-08-18, up from 4-3-3 (total 14), BECAUSE THE SCENE
# CHANGED. The host never wrote the generated background into the frame buffer,
# so every patch the pipeline has ever seen carried a saturated never-written
# band; that band inflated Stage A's sigma and compressed the real content by
# 4.3x. Seeding the background (mosse_tracker.cpp, after the control-CU probe)
# removes it and raises the signal, so the budget MUST move with it — applying the
# seed at 4-3-3 rails the response outright.
#
# The +4 bits is measured, not guessed. Re-sweep 2026-08-18 over the SAME integer
# datapath, on patches replayed from the host's own scene functions and cropped by
# roi_crop_ref's bit-exact Stage A, broken arm vs seeded arm, frames 2/8/15:
#
#   |feat|max  4.3x    (Stage A sigma no longer set by the band)
#   |F|max     4.4x    (linear, as expected)
#   accum      8.3-10.4x   <- NOT 4.3x, and this is the part that surprises
#   response   8.6-10.5x
#
# The extra ~2.3x is H's Q1.15 quantization. max|Hq| is 32767 in BOTH arms (the
# host always normalises to full scale), but the compressed spectrum drops far
# more filter bins below 1 LSB: 73702 non-zero Hq bins broken vs 90122 seeded.
# So the broken scene lost signal TWICE — once in Stage A, again in the filter
# grid — and only the first loss is visible in the patch.
#
# Transfer to hardware uses the measured converged range rather than the model's
# absolute level, per CLAUDE.md ("ratios and orderings are sound; absolute
# magnitudes are patch-specific"). Hardware at 4-3-3, broken scene, filter
# converged over ~20 frames: 14000-26000 = 43-79% of range. Scaling by the
# measured 9.2x and dividing by 2^k:
#
#   total 14  ->  395-735% of range   RAILS
#   total 16  ->   99-184%            RAILS
#   total 17  ->   49- 92%            top end 8% from the rail
#   total 18  ->   25- 46%            <- chosen
#
# Total 17 lands nearest the previously validated band, and is the fallback if
# hardware comes back small. It is not the default because the response GROWS as
# the filter converges — that is exactly how 4-2-2 passed at frame 1 (56%) and
# railed by frame 15 — and 8% of margin against a model accurate to ~6% is not
# margin. Verified offline at total 18: rails=0 at every stage (row/col FFT,
# accumulator, both IFFTs) on frames 0/1/2/8/15, |F|max a stable 32-33% of scale.
#
# THE SPLIT IS FREE, MEASURED: in the seeded scene the invariant holds to 1.3%
# across (4,4,5)/(4,5,4)/(5,3,4)/(5,4,3) and (4,5,5)/(5,3,5)/(5,4,4). FFT_SHIFT
# stays at 4 rather than 5 because that leaves the accumulator at ~1400 instead
# of ~330 — same response, 4x the accumulator resolution.
#
# READ THE FIRST HARDWARE RUN BEFORE TRUSTING ANY OF THIS. The scene and the
# budget move together here, which breaks the project's own "never move two
# magnitudes at once" rule; it is unavoidable because the budget change is
# DERIVED from the scene change. DUMP_BUFFERS=1 gives per-frame F_ch/accum/resp,
# so check rails and response %FS first and adjust k by whole bits.
# 4-4-4 SINCE 2026-08-20, down from 4-5-5, AND THIS ONE IS FROM HARDWARE.
# runs/run_0820_1418.log: 200 frames, ch16, TRAJECTORY=1 SCALE_TRAJ=1, built at
# 4-4-4 (see build/hw/.../aie.flagstamp, which is the authority — runs/.last_cfg
# claimed 4-3-3 and is stale). rails=0 on EVERY frame, and the response peak ran
# 16157..20994 = 49..64% of int16 range at the converged end. That is the healthy
# band 4-5-5 undershot by 6-11x (it gave 1.1-4.5%).
#
# Leaving the default at the known-wrong 4-5-5 was also an active trap: the shifts
# are AIE flags, so a host-only experiment invoked as `make sd_card TARGET=hw
# SCALE_STEP=1.04` would have silently triggered a multi-hour graph rebuild AT A
# DIFFERENT SHIFT BUDGET, i.e. moved two magnitudes at once without saying so.
FFT_SHIFT           ?= 4
IFFT_ROW_SHIFT      ?= 4
IFFT_COL_SHIFT      ?= 4

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
#      10         480         7680       5536       672   4x margin at BIAS_SCALE=127
#       8        1920        30719      22143      2688   accum ~94% of rail at ch16
# The ch16 column at H_SHIFT=10 (7680) lands on the 7728 already recorded as the
# validated 16-channel accumulator, which is a useful independent check.
#
# RAISED 10 -> 11 on 2026-08-24, because BIAS_SCALE=roi returned ~2.5x of signal
# and consumed that margin. At 10 the corrected build railed (runs/run_calib.log:
# accum 33952 = 104% of int16 on f173, response 32153 = 98% on f187, and the
# response was STILL GROWING at f200). H_SHIFT is the only knob upstream of BOTH
# the accumulator and the response — IFFT_* reaches only the response, and
# FFT_SHIFT moves it two bits at once because it applies to the row AND the
# column pass — and both needed exactly one bit.
#
# Validated on hardware over 200 frames (runs/run_0824_1354.log, gray /
# BIAS_SCALE=roi / 4-4-4 / TRAJECTORY=1 SCALE_TRAJ=1):
#   rails 0 on every frame; accum max 52.1%, response max 49.0% of int16
#   TRACKING BIT-IDENTICAL to the H_SHIFT=10 run on all 199 evaluated frames
#   (every IoU, centre error and box string), mean IoU 0.9188, worst 0.8353
#   PSR 25.92 / 84.06 / 127.36 against 25.92 / 84.08 / 127.41 at H_SHIFT=10
# That PSR agreement is the load-bearing check, not a nicety: the response now
# sits at ~28% of int16 typically, BELOW the 49-64% band the 4-4-4 budget was
# validated in, and PSR is the metric that would show a quantization floor. It
# did not move. The band was measured on a distribution with a 1.30x spread; the
# corrected build spreads 2.07x, so centring the TYPICAL frame in the band puts
# the TAIL on the rail. Size against the tail.
#
# SINGLE SOURCE OF TRUTH — reaches the AIE kernel, the host app and the vector
# generator from this one line.
#
# 15 is the SHIPPING value and it is deliberately OVER-shifted: it is what the
# flashed a.xclbin was built with and what the whole VOT-STb2022 benchmark ran
# on, with rails = 0 over 101,564 frames. The tight-but-safe RGB budget measured
# from the uncensored distribution is 13 (12 rails); 15 sits at 44% of ceiling
# and calib_report.py will call that UNDERSHOOT, which is advisory and intended.
# Moving to 13 buys two bits of nothing and costs a reflash plus a full re-run of
# both arms. Gray's equivalent over-shift arm is 14. Was 11 until 2026-08-28.
#
# THIS IS THE ONLY KNOB HERE THAT IS NOT HOST-ONLY — it reaches AIE_FLAGS, so
# changing it needs a graph rebuild, re-package and re-flash, and vot_sweep.sh's
# xclbin guard will correctly refuse until the card is updated.
H_SHIFT             ?= 15

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

# XRT runtime configuration, packaged onto the SD card NEXT TO THE ELF so XRT
# picks it up from the working directory at startup.
#
# It was not packaged before, despite CLAUDE.md claiming it was. That mattered:
# the whole point of this file is that XRT settings (scheduler mode, thread
# policy, CPU affinity) can be A/B'd on the board WITHOUT a rebuild, and none of
# them were reaching the board at all.
#
# Overridable so a variant can be packaged without editing the tracked file:
#     make package XRT_INI=design/profiling_configs/xrt_ert_off.ini
# Faster still, once a card is flashed: mount the FAT partition and edit the file
# in place. Nothing in the image depends on its contents.
XRT_INI ?= $(PROFILING_REPO)/xrt.ini
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
AIE_FLAGS  += --Xpreproc="-DMEMTILE_TRANSPOSE=$(MEMTILE_TRANSPOSE)"
AIE_FLAGS  += --Xpreproc="-DCMUL_SPLIT_ACCUM=$(CMUL_SPLIT_ACCUM)"
AIE_FLAGS  += --Xpreproc="-DCMUL_ACCUM_MEMTILE=$(CMUL_ACCUM_MEMTILE)"
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
# conv2d input planes. 1 = grayscale (what ships), 3 = RGB.
#
# CONV_IN_CH REACHES BOTH TOOLCHAINS (AIE_FLAGS and GCC_FLAGS) FROM THIS ONE
# VARIABLE. It has to: it picks the conv2d weight-buffer layout, which the graph
# READS and the host WRITES (mean_prev) every frame, and the two offsets differ
# (18 vs 36). A #ifndef default on the host side is not a safety net — it is what
# would make the mismatch silent, exactly as FFT_ROW_WS/FFT_COL_WS once did.
# The exporter stamps the layout into byte 63 of every channel buffer and the
# host checks it at startup, so a stale layer0_weights.bin fails loudly.
#
# Wiring status at CONV_IN_CH=3: export_weights.py YES, conv2d_kernel.cpp YES
# and VECTORIZED (27 aie::mac over CONV_VEC lanes; the static_assert on grayscale
# guards the SEPARATE gray vectorized block, which RGB never reaches), roi_crop
# YES, host YES. Remaining gap: gen_aiesim_vectors.py, and no hardware run.
# 3 (RGB) is the SHIPPING arm: it wins accuracy, robustness and EAO on the full
# 62-sequence benchmark and survives 12.8% more frames. Was 1 until 2026-08-28.
# Grayscale is still fully supported and tested — pass CONV_IN_CH=1. Note the
# grayscale AIESIM SCENARIOS need it explicitly (s6, not s6rgb).
CONV_IN_CH ?= 3
AIE_FLAGS  += --Xpreproc="-DCONV_IN_CH=$(CONV_IN_CH)"
# conv2d's AIE stack in BYTES, applied only at CONV_IN_CH=3. The 27-tap MAC
# chain measured 1344 against the 1024-byte default and the mapper REFUSED to
# produce a libadf.a. See the stack_size() note in mosse_graph.h.
CONV2D_STACK ?= 2048
AIE_FLAGS  += --Xpreproc="-DCONV2D_STACK=$(CONV2D_STACK)"
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

# ROI_IN_CH comes from CONV_IN_CH — ONE knob for both engines. roi_crop decides
# what the AXIS wire carries and conv2d decides how to unpack it; a disagreement
# is not a compile error at either end, it is a graph that runs and produces a
# wrong feature map. See CLAUDE.md's rule on constants both engines derive from.
CROP_VPP_FLAGS  := --hls.clock $(VPP_CLOCK_FREQ):roi_crop
CROP_VPP_FLAGS  += -D ROI_IN_CH=$(CONV_IN_CH)

# =========================================================
# Host application compiler flags
# =========================================================
# Host optimisation level. SEPARATE VARIABLE so it can be swept without editing
# the flag list, and so a run's app.flagstamp records which one it used.
#
# -O2 with no -mcpu is what every run up to 2026-08-20 used. After the BO copy
# pattern the frame is 143 ms of which `publish filter` (12.17 ms) and `filter
# update` (11.17 ms) are pure heap compute — ~262k complex ops each, and the only
# remaining lever on them is codegen.
#
# CAUTION, and this is why it is a separate build: -O3 can change floating-point
# contraction (FMA), so the filter maths is not guaranteed bit-identical. The
# acceptance test is the tracking summary — mean IoU 0.9174, worst 0.8326, centre
# 1.30/3.52 px, final box 62x62. If those move, -O3 changed the arithmetic and
# that is a decision to make deliberately, not a speedup to bank.
#
# -O3 -mcpu=cortex-a72 VERIFIED ON HARDWARE 2026-08-20 (runs/run_0820_1610.log):
# tracking bit-identical, frame 134.6 -> 132.2 ms. It is worth 2.4 ms and ALL of
# that is one slot — `publish filter` 12.16 -> 9.83. `filter update` got 0.27 ms
# SLOWER and `scale extract` did not move, because both are std::complex<float>
# arithmetic that GCC will not vectorise: C99 Annex G forces the libgcc __mulsc3
# helper for complex multiply. Confirmed by cross-compiling mosse_filter.cpp:
# -fcx-limited-range removes the __mulsc3 call and raises NEON fp ops 10 -> 17,
# and a native benchmark puts filter_update at 0.915 -> 0.635 ms (1.6x).
#
#   make sd_card TARGET=hw HOST_OPT="-O3 -mcpu=cortex-a72 -fcx-limited-range" ...
#
# -fcx-limited-range only drops the Inf/NaN range handling in complex mul/div;
# no value in this pipeline is near those. NOT -ffast-math, which buys the same
# time by making every float operation in the file unsafe.
HOST_OPT   ?= -O3 -mcpu=cortex-a72
GCC_FLAGS  := $(HOST_OPT) -std=c++17 -D__linux__ -D__PS_ENABLE_AIE__
GCC_FLAGS  += -DPATCH_ROWS=$(PATCH_ROWS)
GCC_FLAGS  += -DPATCH_COLS=$(PATCH_COLS)
GCC_FLAGS  += -DN_CHANNELS=$(N_CHANNELS)
# Selects the conv2d weight-buffer layout the host writes mean_prev into. MUST
# match the AIE build and the exported .bin — see the CONV_IN_CH note above.
GCC_FLAGS  += -DCONV_IN_CH=$(CONV_IN_CH)
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
# Must reach BOTH toolchains from this one variable: at 1 the graph deletes four
# GMIO ports and the host must stop driving them. A mismatch is not a compile
# error on either side alone — it is a deadlock on the board.
GCC_FLAGS  += -DMEMTILE_TRANSPOSE=$(MEMTILE_TRANSPOSE)
# Host-only: no AIE counterpart, so it does NOT need to reach AIE_FLAGS.
GCC_FLAGS  += -DROI_CROP_PIPELINE=$(ROI_CROP_PIPELINE)
# Shared by both toolchains: at 1 the graph gains gmio_accum_in and the host must
# stop packing. A mismatch deadlocks rather than failing to compile.
GCC_FLAGS  += -DCMUL_SPLIT_ACCUM=$(CMUL_SPLIT_ACCUM)
# Shared by both toolchains: at 1 the host drains accum_out once per CHANNEL
# instead of once per chunk. A mismatch is a stalled drain, not a compile error.
GCC_FLAGS  += -DCMUL_ACCUM_MEMTILE=$(CMUL_ACCUM_MEMTILE)
# Host-only, and it needs -pthread in BOTH the compile and the link.
GCC_FLAGS  += -DTAIL_PARALLEL=$(TAIL_PARALLEL) -pthread
# Offset of the synthetic test impulse from the tracked position. A correct
# pipeline must report exactly this displacement; (0,0) would be untestable
# because it is also what a zero response yields. Sweep to check other offsets:
#   make application IMPULSE_DR=-20 IMPULSE_DC=31
IMPULSE_DR ?= 10
IMPULSE_DC ?= -7
GCC_FLAGS  += -DIMPULSE_DR=$(IMPULSE_DR)
GCC_FLAGS  += "-DIMPULSE_DC=($(IMPULSE_DC))"
# THE MAXIMUM frame geometry, which is what frame_bo is allocated at. It is no
# longer the geometry the pipeline runs: at FRAME_SOURCE=vot every sequence
# brings its own rows/cols from its manifest, and the host writes them to
# roi_crop's AXI-Lite registers per sequence. Phase 1 checked all 62 stb2022
# sequences against this bound -- none exceeds it, and birds2/zebrafish1/frisbee
# sit EXACTLY at it, so the margin is zero and a new dataset needs the check
# re-run rather than assumed (runs/vot/phase1.md).
GCC_FLAGS  += -DFRAME_ROWS=1080
GCC_FLAGS  += -DFRAME_COLS=1920
# WHERE FRAMES COME FROM.
#   synth  the generated scene -- background, scripted trajectory, injected
#          target, occluder. Reproduces today's behaviour EXACTLY; every
#          existing result and every knob above still means what it meant.
#   vot    frames memcpy'd out of a converted VOT blob, geometry and init box
#          from its manifest. The synthetic scene generator, TRAJECTORY,
#          OCCLUDE_MASK, BG_PAN and FRAME_NOISE are all inert -- the frames are
#          whatever the dataset holds -- and IoU is scored against the
#          manifest's groundtruth instead of against what the host drew.
# The board still does no protocol: no failure detection, no anchor logic, no
# reset policy. See runs/vot/ and the plan artifact.
FRAME_SOURCE ?= synth
ifeq ($(FRAME_SOURCE),vot)
  GCC_FLAGS += -DFRAME_SOURCE_VOT=1
  # Linked on this arm ONLY. At synth it would be dead code in the ELF -- a
  # manifest parser and a trajectory writer that nothing calls -- and the two
  # arms should differ structurally, not just by a #define.
  VOT_SRC := $(HOST_APP_SRC)/vot_source.cpp
else ifeq ($(FRAME_SOURCE),synth)
  GCC_FLAGS += -DFRAME_SOURCE_VOT=0
  VOT_SRC :=
else
  $(error FRAME_SOURCE must be 'synth' or 'vot', got '$(FRAME_SOURCE)')
endif
# Defaults for the VOT run, all overridable on the board's command line
# (--vot-data / --vot-results / --vot-seq / --vot-job) so picking a different
# sequence or anchor costs neither a rebuild nor a re-flash. They are compiled in
# as defaults only, and the ELF prints what it actually used.
VOT_DATA_DIR    ?= /mnt/vot
VOT_RESULTS_DIR ?= /mnt/vot-results
VOT_SEQUENCE    ?= car1
VOT_JOB         ?= 0
# Deliberately break one item of run_reset(), so the determinism test's ability
# to FAIL is demonstrated rather than assumed. 0 = none (the shipping build).
#   1 mean_prev   2 filter_bo   3 g_filter   4 coast   5 scale reconfigure
# Every non-zero value prints a banner and invalidates the run's tracking output.
RESET_MUTANT    ?= 0
# STREAMING FRAME SOURCE, for the sequences that do not fit in heap.
#
# The board maps 2 GB of the VEK280's 12 GB and 512 MB of that is CMA, so usable
# heap is ~0.9-1.2 GB -- not the 12 GB the resource table records, which is the
# PART's capacity (runs/vot/TODO_board_memory.md). At CONV_IN_CH=3 five stb2022
# sequences exceed it and the 2026-08-26 full-62 RGB sweep lost exactly those
# five to std::bad_alloc and the OOM killer.
#
# VOT_RESIDENT_MAX_MB is the blob+sidecar size above which the run streams from
# the NFS mount through a prefetched ring instead of staging into heap. 700 MB
# leaves room for frame_bo, the scene buffers and the filter state against the
# ~950-1200 MB actually available -- it is a ceiling on the DATA, not on the
# process. --vot-stream on the board's command line overrides it without a
# rebuild, which is what the mode-equivalence test uses.
#
# VOT_STREAM_RING is the ring depth in FRAMES. It must be >= 2 (at(k) holds slot
# k while the prefetcher fills ahead) and the reader refuses 1 rather than
# clamping it. 8 frames is 63 MB at stb2022's largest geometry (1080x1920 RGB
# plus the luma plane) and under 10 MB for most of the dataset.
VOT_RESIDENT_MAX_MB ?= 700
VOT_STREAM_RING     ?= 8
GCC_FLAGS  += -DVOT_RESIDENT_MAX_MB=$(VOT_RESIDENT_MAX_MB)
GCC_FLAGS  += -DVOT_STREAM_RING=$(VOT_STREAM_RING)
GCC_FLAGS  += -DVOT_DATA_DIR='"$(VOT_DATA_DIR)"'
GCC_FLAGS  += -DVOT_RESULTS_DIR='"$(VOT_RESULTS_DIR)"'
GCC_FLAGS  += -DVOT_SEQUENCE='"$(VOT_SEQUENCE)"'
GCC_FLAGS  += -DVOT_JOB=$(VOT_JOB)
GCC_FLAGS  += -DRESET_MUTANT=$(RESET_MUTANT)
# What "colour" means for the SYNTHETIC scene at CONV_IN_CH=3. 1 = per-plane
# tint with a warmer target (real chroma for the joint normalisation to carry);
# 0 = replicate luma into all three planes, which is the hardware analogue of the
# offline `rgb-lum` control — same 27 taps and same plumbing, no colour. Inert at
# CONV_IN_CH=1, and irrelevant once a real frame source supplies its own colour.
FRAME_RGB_MODE ?= 1
GCC_FLAGS  += -DFRAME_RGB_MODE=$(FRAME_RGB_MODE)
# Re-colourise the WHOLE frame each push and abort on a mismatch, with
# coordinates. The incremental colourise is correct only if every luma write
# reached scene_touch(); miss one and the device reads last frame's colour
# there, which looks like a slightly worse tracking result rather than a bug.
# O(frame)/frame, so it is opt-in: run it for a handful of frames after any
# change to a scene function, then turn it off. Inert at CONV_IN_CH=1.
#
# PLUMBED 2026-08-24. It was a bare `#ifndef SCENE_VERIFY` default in
# mosse_tracker.cpp and nothing passed it, so `make sd_card SCENE_VERIFY=1`
# silently built the instrument DISABLED — the "a #ifndef default in the host is
# not a safety net, it is what makes the mismatch silent" trap, on the one
# instrument written for the colour path. Verify it took with:
#   strings $(BUILD_DIR)/mosse_tracker.elf | grep 'incremental colourise'
# which is present only when the check is compiled in.
SCENE_VERIFY ?= 0
GCC_FLAGS  += -DSCENE_VERIFY=$(SCENE_VERIFY)
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
# 5.0 is the SHIPPING value: +0.0134 robustness on the full benchmark against
# 7.0, on the eta=0.05 filter. Was 7.0 until 2026-08-28. ITS WORTH DEPENDS ON THE
# PSR SCALE — a slower filter or a different feature geometry moves PSR and
# re-opens this. 0 disables the LOW_PSR test only; it measured as a null.
PSR_GATE_MIN ?= 5.0
GCC_FLAGS  += -DPSR_GATE_MIN=$(PSR_GATE_MIN)

# ---------------------------------------------------------------------------
# Run instrumentation. HOST-ONLY, like PSR_GATE_MIN above.
# ---------------------------------------------------------------------------
# DUMP_BUFFERS writes F_ch / accum / resp / H_q15 as raw binaries next to the
# ELF, one file per tag per frame. Measured on the 2026-08-17 board run:
#
#   F_ch      64 KB     accum   64 KB     resp    64 KB
#   H_q15   1024 KB   <- 86% of the volume, written on every ACCEPTED frame
#            -------
#           1216 KB per frame
#
# through fopen/fwrite/fclose onto FAT32, into a directory that accumulates one
# file per tag per frame across runs (so every fopen("wb") does a linear
# directory scan over ~1000+ entries). That run measured 10.2 s/frame end to end
# against an 88 ms instrumented pipeline and 0.72 s of 115200-baud console — the
# dumps are the residual, i.e. the frame rate of a board run is set by this knob
# and not by the design.
#
# So: 1 for a short diagnostic run where you intend to open the binaries in
# NumPy, 0 for any run whose purpose is tracking behaviour or FPS. The CSV below
# is the thing you actually want for the latter and costs ~40 B/frame.
#
# Not just a #define in mosse_tracker.cpp: it was hardcoded to 1 with no way to
# turn it off from the build, which is how a 500-frame run ends up spending 99%
# of its wall clock on diagnostics nobody reads.
DUMP_BUFFERS ?= 1
# Console verbosity. THIS IS A FRAME-RATE PARAMETER, not a cosmetic one.
#
# Measured 2026-08-20 over 198 frames of runs/run_0820_1244.log at
# DUMP_BUFFERS=0: regressing frame period against console bytes gives slope
# 92.5 us/byte (115200 8N1 is 86.8 us/byte) and an INTERCEPT OF ZERO. The frame
# time WAS the UART. The 87 ms of GMIO, the 17 APU transposes, the ~2 MB/frame
# of cmul packing memcpy and the filter update were all already hidden behind
# the tty drain, and 79% of the ~10 KB/frame was instrumentation for problems
# that are now closed (the 503 ms KDS latency, the roi_crop launch path).
#
#   0  one compact line per frame (~45 B, ~4 ms). USE FOR ANY LONG RUN.
#   1  human-readable per-frame block; roi_crop/DMA tables on first+last frame
#      only. Default.
#   2  everything, including the 96 per-channel progress lines — the behaviour
#      of every run before 2026-08-20.
#
# Anomalies print at EVERY level: a railed bin, a PSR/scale HOLD, a peak-
# definition disagreement, a negative peak. Silencing those to save console is
# how a shift-budget hunt goes wrong.
#
# ESTIMATED from the same run's byte categories (measure it, do not trust this):
# at 128x128/ch16, DUMP_BUFFERS=0, level 2 is the measured ~880 ms; level 1
# drops to ~1.2 KB/frame => ~105 ms; level 0 to ~45 B => the ~87 ms GMIO floor,
# ~11 FPS. Level 1 is still console-bound; level 0 is the first configuration
# where something other than the UART sets the frame time.
VERBOSITY  := 1
GCC_FLAGS  += -DVERBOSITY=$(VERBOSITY)
# How often the LEVEL-0 progress line prints, in frames. 1 = every frame =
# byte-identical to every run before 2026-08-25.
#
# The 4%-of-the-frame justification for that line was written against an ~87 ms
# floor. The floor is now 26.29 ms, so the same ~45 B is 15% of it -- and on a
# gate-heavy sequence the console is worse still: correlation(gated%,
# unattributed frame time) = 0.963 across the 8-sequence sweep, and `animal`
# spends 58% of its frame printing. Thinning the line is the fix; SILENCING it
# is not, because `picocom | ts` needs a per-frame marker to time and that
# marker is how the frame time was measured in the first place.
#
# Frame 0 and the last frame always print, whatever N -- a run whose last line
# is missing looks exactly like a run that hung. Host-only.
PROGRESS_EVERY ?= 1
GCC_FLAGS  += -DPROGRESS_EVERY=$(PROGRESS_EVERY)
# Row-FFT drain pipeline depth. 1 = the historical per-firing barrier; 0 = sweep
# 1,2,4,8,16 in 40-frame blocks inside one run. See the note in mosse_tracker.cpp:
# GMIO is 67% of the frame and gmio_fft_row_out is 73 of its 87 ms, so this is the
# remaining question. Host-only.
FFT_DRAIN_DEPTH := 1
GCC_FLAGS  += -DFFT_DRAIN_DEPTH=$(FFT_DRAIN_DEPTH)
# REPORT-ONLY on the host: the shift budget is an AIE parameter and the host does
# no arithmetic with it. Passed anyway so the startup banner can state it, from
# the SAME Makefile variables the graph gets — a log that cannot name its own
# shift budget is how "4-5-5 is wrong and the Makefile still defaults to it"
# survived as long as it did. Same one-variable-to-both-toolchains rule as
# FFT_ROW_WS/FFT_COL_WS, for the same reason.
GCC_FLAGS  += -DFFT_SHIFT_CFG=$(FFT_SHIFT)
GCC_FLAGS  += -DIFFT_ROW_SHIFT_CFG=$(IFFT_ROW_SHIFT)
GCC_FLAGS  += -DIFFT_COL_SHIFT_CFG=$(IFFT_COL_SHIFT)
ifeq ($(TARGET),hw_emu)
GCC_FLAGS  += -DHW_EMU_BUILD=1
endif
GCC_FLAGS  += -DDUMP_BUFFERS=$(DUMP_BUFFERS)

# One CSV row per frame — frame, gate verdict, both PSR statistics, peak, the
# displacement in bins, the zero-displacement/peak ratio, and both boxes with
# IoU and centre error. ~40 B/frame, so ~20 KB for a 500-frame run, fflush'd
# every row so a power cut costs at most the frame in progress.
#
# This is the run's actual product: the summary block at the end of the run
# reports means, and `err=0 px` cannot see drift, mainlobe width or a gated
# frame. Everything plotted in the thesis comes from here.
CSV_LOG ?= 1
GCC_FLAGS  += -DCSV_LOG=$(CSV_LOG)
# How often track.csv is flushed, in rows. 1 = every row = the previous
# behaviour, and that stays the default deliberately: per-row flushing was
# justified by surviving a power cut, and a power cut really did take out arm B's
# car1 run on 2026-08-25. At DUMP_BUFFERS=0 there are no per-frame binaries to
# hide behind, so this is a filesystem sync in the timed path once per frame.
#
# A RAILED row flushes regardless of N -- it invalidates the shift budget and is
# the one row worth interrupting a sweep over. A gate veto deliberately does not,
# because vetoes are commonest on exactly the runs this knob exists for (`animal`
# gates 76% of frames), so flushing on them would save nothing there.
#
# The whole 62-sequence dataset is ~4.6 MB of rows, so buffering a full run costs
# nothing. Host-only.
CSV_FLUSH_EVERY ?= 1
GCC_FLAGS  += -DCSV_FLUSH_EVERY=$(CSV_FLUSH_EVERY)

# ---------------------------------------------------------------------------
# Launch-path diagnosis knobs (roi_crop's 505 ms crop_run.wait()). HOST-ONLY —
# none of this reaches AIE_FLAGS or the PL, so changing any of it costs an ELF
# rebuild and a repackage, not a re-synthesis.
# ---------------------------------------------------------------------------

# PL clock, MHz. Passed to the HOST as well as to the AIE compiler because the
# host now converts PL cycles to ms when it reports the control CU's expected
# datapath cost. Same variable, both toolchains — a second literal in the host
# would be the exact failure mode CLAUDE.md records for FFT_ROW_WS.
GCC_FLAGS  += -DPL_FREQ_MHZ=$(PL_FREQ)

# Frames for which every individual launch-path call is printed, rather than only
# the frame mean. A mean cannot distinguish a cost quantized to a scheduler tick
# from one that tracks data, and that distinction is the whole diagnosis. ~240
# short lines per frame, so keep it small.
RC_TRACE_FRAMES ?= 3
GCC_FLAGS  += -DRC_TRACE_FRAMES=$(RC_TRACE_FRAMES)

# Control-CU probe: camera_capture launched CONTROL_CU_RUNS times at startup with
# the same start/poll/wait timing as roi_crop, alternating a 1-row (~6 us) and a
# full-frame (~6.6 ms) datapath. It answers the question no roi_crop measurement
# can: does ANY CU completion cost ~500 ms on this stack?
#
# Defaults to 0 under hw_emu, where camera_capture's II=1 zero-fill of 2 M bytes
# is simulated at RTL and would cost hours to produce a host-side number that is
# meaningless there anyway (the host runs on QEMU).
CONTROL_CU_RUNS ?= $(if $(filter hw,$(TARGET)),8,0)
GCC_FLAGS  += -DCONTROL_CU_RUNS=$(CONTROL_CU_RUNS)

# roi_crop launch path. 1 = user-managed CU via xrt::ip, the host writes the
# AXI-Lite control registers and polls the CU's own ap_done. 0 = the original
# KDS path (xrt::kernel/xrt::run + start/wait).
#
# WHY 1 IS THE DEFAULT. Measured on hardware 2026-08-20: the CU finishes in
# 4.8 ms and KDS reports it 503.4 ms later, because the CU's completion
# interrupt is never delivered (/proc/interrupts reads 0 on both zocl IRQs
# while the CU's own ISR reads 0x3, latched and unserviced). That is a platform
# interrupt-wiring defect — poll_threshold, a hand-cleared ISR and
# Runtime.ert_polling all left the number unchanged — so the host stops asking
# KDS and reads the status register itself.
#
# HOST-ONLY: no AIE flag, no PL re-synthesis, no libadf.a relink. Both modes
# print the same RC_*/timeline tables, so a single log compares them directly.
ROI_CROP_USER_MANAGED ?= 1
GCC_FLAGS  += -DROI_CROP_USER_MANAGED=$(ROI_CROP_USER_MANAGED)

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
#
# 0.05 is the SHIPPING value: robustness 0.3065 -> 0.3283 (+7.1%), EAO +8.5%,
# 13.9% more frames on the full benchmark. Was 0.125 until 2026-08-28. The sweep
# is NOT monotone (0.025 is much worse), so this is a shallow optimum, not a
# trend. Two of three mechanism falsifiers fired — the gain is real, the drift
# explanation for it is not established. See runs/vot/eta05.md.
MOSSE_ETA      ?= 0.05
GCC_FLAGS  += -DMOSSE_ETA=$(MOSSE_ETA)

# Background for the synthetic test frame: 1 = band-limited texture, 0 = the
# pre-2026-08-16 flat fill. NOT cosmetic — padding exists so the filter can learn
# target-vs-background, so against a flat fill more padding is strictly less
# target and any padding comparison is decided before it runs.
FRAME_TEXTURE  ?= 1
GCC_FLAGS  += -DFRAME_TEXTURE=$(FRAME_TEXTURE)

# Per-frame sensor noise, PEAK amplitude in LSB. 0 = the pre-2026-08-17 behaviour.
#
# The background was generated once and cached (fill_background() is ~0.6-1.2 s on
# the A72, so caching is not optional), which made it byte-identical every frame
# outside the dirty rect. A DCF correlates with a perfectly repeating background
# at exactly zero shift: the 2026-08-17 run carried a static peak at (0,0) worth
# 69-86% of the true motion peak, it won 21 of 48 frames, and each win cost a
# permanent ~9.4 px offset because MOSSE measures only relative displacement.
# PSR read 24-35 throughout — the response was sharply peaked, at the wrong place.
#
# Only the noise term is re-drawn per frame, and only over the ROI (~130x130, the
# window the pipeline actually reads), so the sinusoid field stays cached. Watch
# the resp00_over_peak column in the CSV: 0.69-0.86 was the broken run, under
# ~0.3 is healthy.
#
# THE ONE TEST-SEQUENCE DEFAULT THAT DOES NOT REPRODUCE PREVIOUS BEHAVIOUR.
# Deliberate: the old default is the pathological case. FRAME_NOISE=0 restores it.
FRAME_NOISE    ?= 2
GCC_FLAGS  += -DFRAME_NOISE=$(FRAME_NOISE)

# Camera pan over the cached background, px/frame. HOST-ONLY. 0 = the static
# background, i.e. the pre-2026-08-20 behaviour.
#
# THIS, NOT FRAME_NOISE, IS THE FIX FOR BACKGROUND LOCK. Independent additive
# noise cannot decorrelate a static pattern — the background still correlates
# with itself at exactly zero shift, and the extra variance inflates the MOSSE
# numerator and the shared denominator alike. FRAME_NOISE=2 appeared to work on
# 2026-08-17 only because the frame buffer was not yet seeded and the ROI was
# mostly zeros, so the noise was the dominant varying content. Once seeding
# landed, resp00_over_peak went straight back to 0.86 (measured 2026-08-20) and
# raising the amplitude cannot recover it — it would only bury the target too.
#
# Panning is what real video actually provides: the tracking window sees a
# different piece of the world every frame because the camera moves. Sampled from
# the cached field at a wrapped offset (two memcpys per row over the ROI), so it
# does NOT re-run fill_background(), which is ~0.6-1.2 s on the A72.
#
# Constant velocity and independent of TRAJECTORY on purpose: a pan that tracked
# the target would hold the background still in ROI coordinates and rebuild the
# bug, and a periodic one could resonate with TRAJ_PERIOD.
#
# THE VALUES ARE SWEPT, NOT GUESSED — scripts/bg_pan_sweep.py, seconds, no
# hardware. It replicates fill_background() and Stage A and reports the normalised
# zero-shift correlation between consecutive ROI patches after B2:
#
#   pan/frame     0,0    3,5    7,11   15,23  23,37  31,47  47,71  63,97
#   corr@0shift  +0.60  +0.61  +0.64  +0.54  +0.31  +0.09  -0.25  -0.56
#
# **A SMALL PAN DOES NOTHING.** The texture is six sinusoids of 1-6 cycles per
# frame, i.e. wavelengths of 180-1080 rows, so a 3-5 px shift moves it by under 2%
# of the shortest period and the first guess at 3/5 was worthless (0.61 vs 0.60 at
# no pan at all). |corr| is minimised near 31/47; beyond that it just
# anti-correlates. Re-run the sweep if FRAME_TEXTURE or fill_background() changes
# — the right pan is a property of the texture's spectrum, not a universal number.
#
# 31 and 47 are coprime to 1080 and 1920, so the offsets cycle through every row
# and column before repeating. fill_background() rounds its frequencies to whole
# cycles per frame precisely so this wraparound is seamless.
BG_PAN         ?= 1
BG_PAN_R       ?= 31
BG_PAN_C       ?= 47
GCC_FLAGS  += -DBG_PAN=$(BG_PAN) -DBG_PAN_R=$(BG_PAN_R) -DBG_PAN_C=$(BG_PAN_C)

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
# DSST 6.1's a. THE SEARCH RANGE a^((S-1)/2) IS WHAT MATTERS, not the resolution
# — measured offline 2026-08-20 with `make scale_sim`, which reproduces the board's
# f130 scale stall (runs/run_0820_1418.log). Sweeping a and S on the same envelope:
#
#   a=1.02 S=33  range 1.373  moving 12.6%  step 37.3%   1.00x cost   <- default
#   a=1.03 S=33  range 1.605  moving  9.5%  step 42.9%   1.00x
#   a=1.04 S=33  range 1.873  moving  8.6%  step  4.4%   1.00x        <- free win
#   a=1.02 S=49  range 1.608  moving  8.3%  step 42.9%   2.20x
#   a=1.02 S=65  range 1.885  moving  6.1%  step  6.1%   3.88x        <- best
#
# CONFIRMED ON HARDWARE 2026-08-20 (runs/run_0820_1513.log, 200 frames, identical to
# run_0820_1418 in every other respect): 1.02 -> 1.04 took mean IoU 0.807 -> 0.917,
# worst IoU 0.579 -> 0.833, max box error 31.4% -> 9.6%, mean centre error 2.47 ->
# 1.30 px, worst 11.07 -> 3.52 px. Free: S is unchanged, so d*S^2 is unchanged.
# NOW THE DEFAULT, deliberately deviating from DSST 6.1's 1.02.
SCALE_STEP         ?= 1.04
SCALE_ETA          ?= 0.025
SCALE_SIGMA_FACTOR ?= 16.0
SCALE_TMPL_AREA    ?= 512
GCC_FLAGS  += -DSCALE_N=$(SCALE_N) -DSCALE_STEP=$(SCALE_STEP)
GCC_FLAGS  += -DSCALE_ETA=$(SCALE_ETA) -DSCALE_SIGMA_FACTOR=$(SCALE_SIGMA_FACTOR)
GCC_FLAGS  += -DSCALE_TMPL_AREA=$(SCALE_TMPL_AREA)

# Scale-update gate — the size-axis analogue of PSR_GATE_MIN. HOST-ONLY.
#
# WHY IT EXISTS. Measured on hardware 2026-08-20 with position tracking EXACT
# (IoU 1.0000, centre error 0.00 px through frame 5): the scale filter jumped to
# level -12 on frame 6 and took the box 64.0 -> 50.5 in one step, which shrank the
# ROI, which then broke position tracking. The collapse was previously recorded as
# a SYMPTOM of background lock -> position drift; it is not, it fires first.
#
# SCALE_CONF_MIN is on ScaleResult::psr, which separates the two populations
# cleanly on three independent hardware runs — healthy 2.24-3.31, collapsed
# 0.72-1.87 — so 2.0 sits in the gap every time. It is NOT a universal constant:
# re-derive it from the [scale] SUMMARY conf range after any change to the
# geometry, the template size or the feature scale. 0 disables the threshold test
# only; the structural vetoes (argmax on the search rail, proposed box outside the
# absolute bounds) still apply, exactly as with PSR_GATE_MIN=0.
#
# SCALE_MIN_REL/MAX_REL tightened from 0.25/4.0, which was so loose it never fired
# in any run on record. They are a DRIFT backstop, not a per-frame plausibility
# test — the per-frame change is already bounded by the filter's own range
# (step^±(S-1)/2 = ±37% at the defaults) — so they must still admit the test
# sequence's own envelope (SCALE_TRAJ_AMP=0.30 => 0.70x..1.30x). 0.5/2.0 leaves
# margin on both sides of that.
SCALE_CONF_MIN     ?= 2.0
SCALE_MIN_REL      ?= 0.5
SCALE_MAX_REL      ?= 2.0
GCC_FLAGS  += -DSCALE_CONF_MIN=$(SCALE_CONF_MIN)
GCC_FLAGS  += -DSCALE_MIN_REL=$(SCALE_MIN_REL) -DSCALE_MAX_REL=$(SCALE_MAX_REL)
# Largest |idx| ONE frame may move the box -- a rate limit, where MIN_REL/MAX_REL
# are a drift bound. 0 disables the test.
#
# DEFAULT 2, AND 1 WAS MEASURED AND REJECTED. car1's hardware run argues for 1:
# all seven proposals with |idx| >= 2 landed on frames whose IoU was 0.000,
# including frame 490's +9 (a 1.42x inflation while the tracker was 227 px off).
# But `make scale_sim` shows 1 parks the NORMAL smooth-envelope arm for 123 of
# 200 frames and ends 28.0% wrong, against 1.0% unlimited -- the sim's detector
# really does use |idx| = 2 there. 2 costs the smooth arm nothing and still
# vetoes three of car1's seven. Sweep it with:
#     make scale_sim && build/.../scale_loop_sim --max-step N
SCALE_MAX_STEP     ?= 2
# COASTING THROUGH A HOLD. On a gate veto the host holds position, which assumes
# the target stays put while the filter is frozen -- an assumption stb2022
# violates on most sequences. HOLD_COAST=1 moves the search window at the last
# measured velocity instead, decayed by COAST_DECAY on each successive held
# frame, so total drift over a hold run is bounded by v/(1-decay) = 2v and a long
# hold fades back to a freeze. HOLD_COAST=0 restores the old freeze exactly.
#
# COAST_DECAY=0.5 chosen by sweeping all 62 stb2022 sequences offline
# (scripts/vot_hold_budget.py --policy both --coast-decay D); PURE constant
# velocity (1.0) is worse than 0.5 and is worse than freezing on slow sequences,
# because a near-stationary target's measured velocity is mostly noise.
# DEFAULT WAS FLIPPED TO 1 ON 2026-08-25 AND REVERTED TO 0 THE SAME DAY, WHEN
# THE SAME 54 TRAJECTORY PAIRS WERE SCORED BY THE TOOLKIT INSTEAD OF BY MEAN IoU.
# The two metrics disagree, and both readings are of the same runs:
#
#   mean IoU (evidence_arm_ab.md)   0.2709 -> 0.3005 frame-weighted   1 WINS
#   vot AR / EAO (evidence_ar.md)   A 0.638 -> 0.616, R 0.309 -> 0.288,
#                                   EAO 0.208 -> 0.194                0 WINS
#
# AR is the metric of record, so the default follows it. The mechanism is
# understood: vot fails a run on 10 CONSECUTIVE frames at overlap <= 0.1, and on
# a direction change the coast carries the box the old way while the target
# reverses -- car1 anchor 741 drops out for 13 frames coasting against 7 frames
# freezing, so the coast trips the grace where the freeze does not. The run then
# REACQUIRES and tracks ~470 more frames at overlap 0.82, all of which the rule
# discards. Failure counts barely move (48 of 54 runs vs 49); only their timing
# does, which is exactly what a mean cannot see.
#
# So this is not "coasting is bad": it wins on many short holds (car1 job 0's
# permanent loss disappears, 73 accept->hold transitions, ~5 coasted frames
# each) and loses on one long hold that crosses a turn. The untested option is a
# CAP on consecutive coasted frames -- k from hold_policy.md's per-sequence
# budget (median 6, car1 4) -- which would plausibly keep the win and delete the
# loss. It is closed loop, so it needs a board run, not more analysis.
#
# The earlier note stands on its own terms: the offline model predicted the
# opposite for car1 and was wrong because it was OPEN LOOP -- it treated the
# observed 29-frame gated run as fixed, when coasting turns frames that would
# have been gated into ACCEPTED ones and every accept restarts the coast. Read
# hold_policy.md's budgets as a bound on one uninterrupted hold, never as a
# prediction of outcome.
#
# HOLD_COAST=1 reproduces the coast1 arm; every result recorded before
# 2026-08-25 is reproducible at 0, which is again the default.
HOLD_COAST         ?= 0
COAST_DECAY        ?= 0.5
GCC_FLAGS  += -DHOLD_COAST=$(HOLD_COAST) -DCOAST_DECAY=$(COAST_DECAY)
GCC_FLAGS  += -DSCALE_MAX_STEP=$(SCALE_MAX_STEP)

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
.PHONY: help kernels graph gen_vectors aiesim graph_fft aiesim_fft xsa application package sd_card run_emu weights test_host test_roi_crop test_scene cleanall

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

# The .xo needs a flag stamp for the same reason libadf.a and the ELF do:
# CONV_IN_CH touches no source file, so a source-only prerequisite list would
# happily reuse a grayscale roi_crop.xo in an RGB build — and the failure is a
# wrong feature map, not a build error.
CROP_FLAGS_STAMP := $(BUILD_DIR)/crop.flagstamp
$(CROP_FLAGS_STAMP): FLAGS_FOR_STAMP := $(CROP_VPP_FLAGS)

$(CROP_XO): $(CROP_FLAGS_STAMP) $(PL_SRC_REPO)/roi_crop/roi_crop.cpp \
            $(PL_SRC_REPO)/roi_crop/roi_crop.h
	mkdir -p $(BUILD_DIR)
	v++ $(VPP_FLAGS) $(CROP_VPP_FLAGS) -c -k roi_crop \
	    $(PL_SRC_REPO)/roi_crop/roi_crop.cpp -o $@

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

# BIAS_SCALE: `roi` (default since 2026-08-23) derives bias_acc against
# ROI_NORM_Q = 32, the scale roi_crop actually emits. `127` restores the
# historical int8-full-scale assumption, under which bias_acc was ~4x oversized
# and out_shift pushed the SIGNAL down to make room for it.
#
# THE DEFAULT CHANGED, AND IT INVALIDATES THE SHIFT BUDGET. Every hardware
# measurement recorded in CLAUDE.md up to 2026-08-23 — the 4-4-4 budget, mean
# IoU 0.9188, the 38.04 FPS run — was taken at 127. The correction moves the
# effective input scale, so 4-4-4 must be re-swept over >= 20 frames before any
# tracking number from a `roi` build is trusted. It only pays at CONV_RELU=0
# (held-out peak/max-sidelobe: 12.82 base+ReLU, 3.92 corrected+ReLU, 16.25
# corrected+no-ReLU), which is the shipping configuration.
#
# `make weights BIAS_SCALE=127` reverts, bit-for-bit apart from the layout tag.
BIAS_SCALE ?= roi

weights:
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= uv run --extra weights python3 scripts/export_weights.py $(AIE_SRC_REPO)/weights $(PATCH_COLS) --in-ch $(CONV_IN_CH) --bias-scale $(BIAS_SCALE)

# int8 samples per PatchIn beat. mosse_graph.h creates PatchIn as plio_32_bits,
# so 4; change to 16 if the PLIO ever goes back to plio_128_bits.
PLIO_BEAT_SAMPLES ?= 4

gen_vectors:
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    GEN_PATCH_ROWS=$(PATCH_ROWS) \
	    GEN_PATCH_COLS=$(PATCH_COLS) \
	    GEN_CONV_IN_CH=$(CONV_IN_CH) \
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
	    GEN_CONV_IN_CH=$(CONV_IN_CH) \
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
                         $(HOST_APP_SRC)/mosse_filter.h     \
                         $(HOST_APP_SRC)/scene_colour.cpp   \
                         $(HOST_APP_SRC)/scene_colour.h     \
                         $(VOT_SRC)                         \
                         $(HOST_APP_SRC)/vot_source.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(GCC_FLAGS) $(GCC_INC) \
	    $(HOST_APP_SRC)/mosse_tracker.cpp $(HOST_APP_SRC)/mosse_filter.cpp \
	    $(HOST_APP_SRC)/scene_colour.cpp $(VOT_SRC) \
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

# scene_colour.{h,cpp} follows the same rule as mosse_filter: no XRT header, so
# the luma -> interleaved-RGB pass is testable in seconds. It needs to be,
# because BOTH of its failure modes are silent on hardware — a wrong interleave
# or gain hands conv2d a plausible feature map, and a luma write that skipped
# scene_touch() hands it the PREVIOUS frame's colour over one rect. Neither
# shows up as anything but a slightly worse IoU.
#
# The missed-touch case is reproduced deliberately in the harness, so the
# verifier is known to fire rather than assumed to.
.PHONY: test_scene
test_scene:
	mkdir -p $(BUILD_DIR)
	g++ -O2 -std=c++17 -Wall -Wextra -Werror -I$(HOST_APP_SRC) \
	    $(HOST_APP_SRC)/scene_colour.cpp \
	    $(TEST_HOST_DIR)/test_scene_colour.cpp \
	    -o $(BUILD_DIR)/test_scene
	$(BUILD_DIR)/test_scene

# vot_source.{h,cpp} follows the same rule again: no XRT header, so the manifest
# parser, the blob offsets, the run-order convention and the trajectory writer
# are testable in seconds. They need to be, because NOTHING in that file
# computes anything -- every failure mode is bookkeeping that produces a
# complete, plausible, invalid AR report instead of an error. Phase 1 lost every
# groundtruth box in 62 manifests to exactly that shape of bug.
#
# The suite is mutation-tested: 19 mutants, each of which must be REJECTED. If
# $VOT_ROOT is exported it additionally parses every real manifest in
# $VOT_ROOT/data, which is the only check that the converter's output and this
# parser agree.
.PHONY: test_vot_source
test_vot_source:
	mkdir -p $(BUILD_DIR)
	g++ -O2 -std=c++17 -Wall -Wextra -Werror -pthread -I$(HOST_APP_SRC) \
	    $(HOST_APP_SRC)/vot_source.cpp \
	    $(TEST_HOST_DIR)/test_vot_source.cpp \
	    -o $(BUILD_DIR)/test_vot_source
	$(BUILD_DIR)/test_vot_source

# The one piece of arithmetic in the whole result path -- the tracker's CENTRE
# box becoming the toolkit's top-left x,y,w,h -- checked against the toolkit's
# OWN reader rather than against a second copy of our format rules. Phase 0b
# round-tripped the format using the toolkit's WRITER, which says nothing about
# the printf that will actually run on the board.
#
# The C++ side emits its INPUT (centre boxes) next to the trajectory and the
# conversion is re-derived in Python, so a wrong conversion disagrees instead of
# agreeing with itself. Both a transposed pair and a missing centre offset were
# confirmed to FAIL this check.
#
# Needs the venv (vot-toolkit), and the Vitis environment MASKS python -- hence
# the env -u, the same one every offline script in scripts/ carries.
.PHONY: test_vot_format
test_vot_format: test_vot_source
	rm -rf $(BUILD_DIR)/vot_format
	mkdir -p $(BUILD_DIR)/vot_format
	$(BUILD_DIR)/test_vot_source $(BUILD_DIR)/vot_format
	cd $(PROJECT_REPO) && env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python \
	    scripts/vot_check_trajectory.py $(BUILD_DIR)/vot_format

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
	@echo ""
	@echo "=== second build: FMA contraction enabled ==============================="
	@echo "The board's compiler contracts mul+add into FMA by default"
	@echo "(-ffp-contract=fast is GCC's default on aarch64); this dev host at -O2"
	@echo "does not, so a bit-exactness claim proven only above is proven on the"
	@echo "wrong machine. filter_update_quantize()'s 'bitwise identical' checks"
	@echo "FAILED here on the first cut and passed at -O2 — see the two-loop note"
	@echo "in mosse_filter.cpp. Cheap to keep, and it is the only thing standing"
	@echo "between a fused fast path and a silently different H."
	g++ -O3 -march=native -ffp-contract=fast -fcx-limited-range \
	    -std=c++17 -Wall -Wextra -I$(HOST_APP_SRC) \
	    -DCMUL_H_SHIFT=$(H_SHIFT) -DPSR_GATE_MIN=$(PSR_GATE_MIN) \
	    -DMOSSE_SIGMA=$(MOSSE_SIGMA) -DSIGMA_FACTOR=$(SIGMA_FACTOR) \
	    -DSIGMA_FROM_TARGET=$(SIGMA_FROM_TARGET) -DMOSSE_ETA=$(MOSSE_ETA) \
	    -DTARGET_PADDING=$(TARGET_PADDING) \
	    -DSCALE_N=$(SCALE_N) -DSCALE_STEP=$(SCALE_STEP) -DSCALE_ETA=$(SCALE_ETA) \
	    -DSCALE_SIGMA_FACTOR=$(SCALE_SIGMA_FACTOR) -DSCALE_TMPL_AREA=$(SCALE_TMPL_AREA) \
	    $(HOST_APP_SRC)/mosse_filter.cpp $(TEST_HOST_DIR)/test_mosse_filter.cpp \
	    -o $(BUILD_DIR)/test_host_fma
	$(BUILD_DIR)/test_host_fma $(TEST_HOST_DIR)/golden

# -------------------------------------------------------
# Closed-loop DSST scale simulation — no hardware
# -------------------------------------------------------
# Explains the frozen scale estimate in runs/run_0820_1418.log (est_h stuck at
# 59.13 for frames 130..199 while truth went 48 -> 45 -> 63). Drives the REAL
# scale_extract/scale_detect/scale_gate/scale_update through a closed loop with
# the position held, which the hardware run could not do. Seconds, native g++.
.PHONY: scale_sim
scale_sim: $(BUILD_DIR)/scale_loop_sim
	$(BUILD_DIR)/scale_loop_sim

# Flagstamp prerequisite, for the reason CLAUDE.md records twice already: SCALE_N,
# SCALE_ETA and SCALE_CONF_MIN are -D values that touch no source file, so a
# source-only prerequisite list silently reuses the previous binary. Caught in the
# act on 2026-08-20 — changing the SCALE_STEP default left the sim reporting
# step=1.020 from a stale ELF.
SCALE_SIM_STAMP := $(BUILD_DIR)/scale_sim.flagstamp
$(SCALE_SIM_STAMP): FLAGS_FOR_STAMP := $(SCALE_N)/$(SCALE_STEP)/$(SCALE_ETA)/$(SCALE_SIGMA_FACTOR)/$(SCALE_TMPL_AREA)/$(SCALE_CONF_MIN)/$(SCALE_MIN_REL)/$(SCALE_MAX_REL)/$(SCALE_MAX_STEP)

$(BUILD_DIR)/scale_loop_sim: $(SCALE_SIM_STAMP)                \
                             $(HOST_APP_SRC)/mosse_filter.cpp \
                             $(HOST_APP_SRC)/mosse_filter.h   \
                             $(TEST_HOST_DIR)/scale_loop_sim.cpp
	mkdir -p $(BUILD_DIR)
	g++ -O2 -std=c++17 -Wall -Wextra -I$(HOST_APP_SRC) \
	    -DCMUL_H_SHIFT=$(H_SHIFT) -DPSR_GATE_MIN=$(PSR_GATE_MIN) \
	    -DMOSSE_SIGMA=$(MOSSE_SIGMA) -DSIGMA_FACTOR=$(SIGMA_FACTOR) \
	    -DSIGMA_FROM_TARGET=$(SIGMA_FROM_TARGET) -DMOSSE_ETA=$(MOSSE_ETA) \
	    -DTARGET_PADDING=$(TARGET_PADDING) \
	    -DSCALE_N=$(SCALE_N) -DSCALE_STEP=$(SCALE_STEP) -DSCALE_ETA=$(SCALE_ETA) \
	    -DSCALE_SIGMA_FACTOR=$(SCALE_SIGMA_FACTOR) -DSCALE_TMPL_AREA=$(SCALE_TMPL_AREA) \
	    -DSCALE_CONF_MIN=$(SCALE_CONF_MIN) \
	    -DSCALE_MIN_REL=$(SCALE_MIN_REL) -DSCALE_MAX_REL=$(SCALE_MAX_REL) \
	    -DSCALE_MAX_STEP=$(SCALE_MAX_STEP) \
	    $(HOST_APP_SRC)/mosse_filter.cpp $(TEST_HOST_DIR)/scale_loop_sim.cpp \
	    -o $@

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
# Runs BOTH arms. ROI_IN_CH is compile-time (it sizes the scratch buffer and the
# interleave stride), so one binary cannot cover both; the golden directory holds
# the cases for both and each binary runs the ones tagged for it. A build that
# matches no case exits 2 rather than reporting a vacuous pass.
test_roi_crop:
	mkdir -p $(BUILD_DIR)
	cd $(PROJECT_REPO) && env PYTHONHOME= PYTHONPATH= \
	    uv run python3 scripts/gen_roi_crop_golden.py $(TEST_PL_DIR)/golden
	for ch in 1 3; do \
	  g++ -O2 -std=c++17 -Wall -Wextra -Wno-comment -Wno-unknown-pragmas \
	      -Wno-deprecated-copy -DROI_IN_CH=$$ch \
	      -I$(XILINX_VITIS)/include -I$(PL_SRC_REPO)/roi_crop \
	      $(PL_SRC_REPO)/roi_crop/roi_crop.cpp $(TEST_PL_DIR)/test_roi_crop.cpp \
	      -o $(BUILD_DIR)/test_roi_crop_ch$$ch || exit 1; \
	  $(BUILD_DIR)/test_roi_crop_ch$$ch $(TEST_PL_DIR)/golden || exit 1; \
	done

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
	$(MAKE) --no-print-directory board_provision ROOTFS_IMG=$@
	@echo "rootfs: $@ ready for packaging"

# Board provisioning: a static address on end0 and an authorized key for root,
# written into the rootfs so the board comes up as an ssh target with nothing
# typed at the console. sshd is already enabled in the stock image; see
# scripts/board_provision.sh for what was verified and what was missing.
#
# BOARD_KEY=none opts out EXPLICITLY. There is deliberately no silent skip when
# the key is absent: a rootfs quietly built without it boots unreachable, and
# that is indistinguishable from a cable fault at the moment it costs most.
BOARD_KEY   ?= $(HOME)/.ssh/id_ed25519.pub
BOARD_IP    ?= 192.168.10.2/24
BOARD_IFACE ?= end0
ROOTFS_IMG  ?= $(ROOTFS)

.PHONY: board_provision
board_provision:
ifeq ($(BOARD_KEY),none)
	@echo "board_provision: SKIPPED (BOARD_KEY=none) -- this image has no ssh key"
else
	@test -f $(BOARD_KEY) || { \
	  echo "ERROR: BOARD_KEY=$(BOARD_KEY) not found."; \
	  echo "       ssh-keygen -t ed25519    to make one, or"; \
	  echo "       make ... BOARD_KEY=none  to build an image with no ssh access."; \
	  exit 1; }
	scripts/board_provision.sh $(ROOTFS_IMG) \
	    --key $(BOARD_KEY) --ip $(BOARD_IP) --iface $(BOARD_IFACE)
endif

# The on-target run script is GENERATED, not copied, because XCL_EMULATION_MODE
# must be set for hw_emu and must NOT be set on real hardware. Packaging the
# template verbatim (as this did until 2026-08-16) bakes "hw_emu" into a board
# image, where it makes XRT open the emulation driver instead of the device.
$(BUILD_DIR)/run_script.sh: $(EXEC_SCRIPTS)/run_script.sh
	mkdir -p $(BUILD_DIR)
	sed 's|@EMU_MODE@|$(if $(filter hw_emu,$(TARGET)),hw_emu,)|' $< > $@
	chmod +x $@

package: $(BUILD_DIR)/$(APP_ELF) $(BUILD_DIR)/$(XSA) $(LIBADF_A) $(ROOTFS) \
         $(BUILD_DIR)/run_script.sh $(XRT_INI)
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
	    --package.sd_file $(XRT_INI) \
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
