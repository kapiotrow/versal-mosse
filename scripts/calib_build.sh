#!/usr/bin/env bash
#
# calib_build.sh — build a REAL HARDWARE SD card for a shift-budget calibration
# run, with the pre-flight checks that this project has paid for.
#
# WHY A SCRIPT AND NOT "make sd_card TARGET=hw"
# --------------------------------------------
# A hardware build is hours. Every entry in CLAUDE.md's "Measurement / build
# hygiene" section is about a run that produced convincing numbers from the wrong
# binary: a stale libadf.a reused after a flag-only change, CONV2D_MODE left at
# echo for a ~28 h baseline, runs/.last_cfg recording a configuration the run did
# not execute. The checks below are cheap; the run is not.
#
# WHAT IS BEING CALIBRATED
# ------------------------
# The 4-4-4 budget was validated on hardware over 200 frames, but BEFORE the
# 2026-08-23 bias_acc correction (BIAS_SCALE=roi). That correction moves the
# effective input scale, so the budget is formally unvalidated even though the
# numbers themselves have not changed. This build re-establishes it.
#
# ARMS, and why gray goes first
#   gray (default)  CONV_IN_CH=1, BIAS_SCALE=roi. The known-good comparator is
#                   now runs/run_0824_1354.log (H_SHIFT=11, mean IoU 0.9188,
#                   rails 0, response max 49% of int16) — same geometry,
#                   trajectory and frame count. It superseded run_0821_1725.log,
#                   which predates the bias_acc correction, and run_calib.log,
#                   which is the same build at H_SHIFT=10 and RAILED on f173.
#   rgb             CONV_IN_CH=3. Do NOT run this first: it moves the bias scale
#                   AND the feature bank AND the input scale at once, and a bad
#                   result would be unattributable. See "never move two
#                   magnitudes at once" in CLAUDE.md.
#
# USAGE
#   scripts/calib_build.sh                 # gray, 4-4-4, 200 frames
#   ARM=rgb scripts/calib_build.sh         # after gray has been validated
#   FFT_SHIFT=4 IFFT_ROW_SHIFT=5 IFFT_COL_SHIFT=5 scripts/calib_build.sh
#   H_SHIFT=11 scripts/calib_build.sh      # accumulator headroom, FFT budget unchanged
#   DRY_RUN=1 scripts/calib_build.sh       # pre-flight only, build nothing
#   COMPARATOR=runs/run_calib.log H_SHIFT=11 scripts/calib_build.sh
#   PATCH_ROWS=64 PATCH_COLS=64 ARM=rgb FFT_SHIFT=4 IFFT_ROW_SHIFT=3 \
#     IFFT_COL_SHIFT=3 H_SHIFT=13 scripts/calib_build.sh   # the 64x64 arm
#

# @thesis subsec:narzedziaBudowanie | B-07,M-06 | The hardware build with pre-flight checks: it
#   verifies the FLAGSTAMPS against the intended config, because a flag-only change has silently
#   reused a stale libadf.a.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${ARM:-gray}
case "$ARM" in
  gray)   CONV_IN_CH=1; CONV_KSIZE=3; CONV_STRIDE=1; CONV_RELU=0; WEIGHT_BANK=mobilenet; ACC_BOUND=loose ;;
  rgb)    CONV_IN_CH=3; CONV_KSIZE=3; CONV_STRIDE=1; CONV_RELU=0; WEIGHT_BANK=mobilenet; ACC_BOUND=loose ;;
  # The pre-registered Layer-1 arm: 7x7 stride 2, resnet18-PCA bank, ReLU ON,
  # a 128x128 crop into a 64x64 feature map.
  # docs/thesis/evidence/arm_l1relu.md, claim N-16 / O-04.
  #
  # IT CARRIES ITS OWN GEOMETRY AND ITS OWN NONLINEARITY, so it cannot be spelled
  # as `ARM=rgb` plus overrides: CONV_RELU and CONV_KSIZE reach AIE_FLAGS, and
  # this script would otherwise print BUILD VERIFIED on a build that silently
  # omitted both -- exactly what CLAUDE.md records about the FILTER_MASK arms.
  # PATCH_ROWS/PATCH_COLS/N_CHANNELS still come from the environment or the
  # Makefile, so pass PATCH_ROWS=64 PATCH_COLS=64 N_CHANNELS=32 with it.
  l1relu) CONV_IN_CH=3; CONV_KSIZE=7; CONV_STRIDE=2; CONV_RELU=1; WEIGHT_BANK=l1resnet; ACC_BOUND=l1 ;;
  # The MECHANISM CHECK for the arm above -- the same bank and geometry with the
  # rectifier off. arm_l1relu.md's falsifier calls for it only if the
  # ReLU arm wins; if the twin wins too, the gain is the bank, not the
  # nonlinearity, and the thesis argument is void.
  l1lin)  CONV_IN_CH=3; CONV_KSIZE=7; CONV_STRIDE=2; CONV_RELU=0; WEIGHT_BANK=l1resnet; ACC_BOUND=l1 ;;
  *) echo "ARM must be gray, rgb, l1relu or l1lin, got '$ARM'" >&2; exit 2 ;;
esac

# Both are OVERRIDABLE, but only deliberately: they are the two knobs that make
# an arm a different KERNEL rather than a different constant.
CONV_KSIZE=${CONV_KSIZE_OVERRIDE:-$CONV_KSIZE}
CONV_STRIDE=${CONV_STRIDE_OVERRIDE:-$CONV_STRIDE}
CROP_ROWS_EXP=0   # filled in once PATCH_ROWS is known

# The budget under test. Defaults are DERIVED FROM THE MAKEFILE via its print-%
# target, never copied. This script used to hardcode `H_SHIFT=${H_SHIFT:-10}`,
# and the moment the Makefile default moved to 11 that copy would have silently
# rebuilt the RAILING configuration while the banner and calib_cfg.txt reported
# 10 in good faith. Same class as the constants that must reach both toolchains
# from one variable: a second copy of a default is a second source of truth.
# Override any of them on the command line to sweep.
mk() { make --no-print-directory -s "print-$1" | sed 's/^[^=]*= *//'; }
FFT_SHIFT=${FFT_SHIFT:-$(mk FFT_SHIFT)}
IFFT_ROW_SHIFT=${IFFT_ROW_SHIFT:-$(mk IFFT_ROW_SHIFT)}
IFFT_COL_SHIFT=${IFFT_COL_SHIFT:-$(mk IFFT_COL_SHIFT)}
BIAS_SCALE=${BIAS_SCALE:-$(mk BIAS_SCALE)}

# H_SHIFT is NOT part of the FFT budget and is swept separately, because it is
# the only knob upstream of BOTH the cmul accumulator and the response:
#   accum ~ F * 2^-H_SHIFT ,  response ~ accum * 2^-(IFFT_ROW+IFFT_COL)
# so IFFT_* can only fix the response, and FFT_SHIFT moves the response by two
# bits at once (it applies to the row AND the column pass). When both the
# accumulator and the response need the same correction — run_calib.log, where
# accum hit 104% of int16 on f173 and the response 98% on f187 — this is the
# only single knob that delivers it.
#
# It reaches make through VARS below, is CHECKED against both flagstamps after
# the build, and is recorded in calib_cfg.txt. It used to be none of those: an
# env-var H_SHIFT did reach make (an exported variable beats `?=`), so the build
# would have been correct while nothing verified it and the config record still
# said only "budget=4-4-4" — a run whose one variable under test was invisible
# in its own provenance.
H_SHIFT=${H_SHIFT:-$(mk H_SHIFT)}

# FEATURE-MAP GEOMETRY. Derived from the Makefile like every other default, and
# passed to make EXPLICITLY, because BUILD_DIR is keyed on it: a script that
# hardcoded 128x128 would build build/hw/64x64/ch16 and then verify the stamps
# of the 128x128 build sitting next to it — reporting BUILD VERIFIED for a
# binary it never looked at. That is the same class as the stale libadf.a.
#
# THE 64x64 ARM (docs/thesis/evidence/arm_res64.md, claim N-03b) is
# built with PATCH_ROWS=64 PATCH_COLS=64. It needs its OWN shift budget — the
# transform gain falls with the point size on the forward AND the inverse pass —
# so do not reuse 4-4-4/H_SHIFT=15 there without reading sec.3 of that file.
PATCH_ROWS=${PATCH_ROWS:-$(mk PATCH_ROWS)}
PATCH_COLS=${PATCH_COLS:-$(mk PATCH_COLS)}
N_CHANNELS=${N_CHANNELS:-$(mk N_CHANNELS)}

# Re-colourise the whole frame each push and abort on a mismatch. O(frame)/frame,
# so it belongs on a short BRING-UP run, never on a 200-frame budget run.
SCENE_VERIFY=${SCENE_VERIFY:-$(mk SCENE_VERIFY)}

# What "colour" means for the SYNTHETIC scene at CONV_IN_CH=3. 1 = per-plane
# tint; 0 = replicate luma into all three planes, which is the hardware analogue
# of the offline colour-free control arm — same 27 taps, same bias and
# quantization grid, no colour. Inert at CONV_IN_CH=1.
#
# This is THE variable under test for a control run, so it is recorded and
# stamp-checked like any other. It reached make by environment inheritance
# before, which builds the right thing but leaves the run's own provenance
# silent about the one knob that was moved.
FRAME_RGB_MODE=${FRAME_RGB_MODE:-$(mk FRAME_RGB_MODE)}

# What KIND of run this image is for. The two have genuinely different
# requirements and conflating them is how a 5-frame result gets quoted as a
# budget validation:
#   budget  (default) a shift-budget / tracking measurement. >=20 frames, hard.
#   bringup           a correctness gate for a path that has never run on this
#                     hardware — short on purpose, and its amplitudes prove
#                     NOTHING about the budget (the response grows as the filter
#                     converges; the retired 4-2-2 point peaked at 56% on frame 1
#                     and railed from frame 15).
# calib_cfg.txt records which one, so the artifacts carry the distinction too.
MODE=${MODE:-budget}
case "$MODE" in budget|bringup) ;; *) echo "MODE must be budget or bringup, got '$MODE'" >&2; exit 2 ;; esac

# The known-good run this build is one variable away from. Recorded next to the
# artifacts so the A/B is legible months later; override it when the comparator
# moves (after a successful H_SHIFT sweep it becomes runs/run_calib.log).
#
# IT IS GEOMETRY-SPECIFIC. The stored comparator is a 128x128 run, and at any
# other patch size the transform gain changes on all four passes, so its
# amplitudes are not the yardstick. Recording it anyway would put a number in
# calib_cfg.txt that a later reader would compare against in good faith — the
# same failure mode as runs/.last_cfg describing a run that did not happen.
if [ "$PATCH_ROWS" = 128 ] && [ "$PATCH_COLS" = 128 ]; then
  COMPARATOR=${COMPARATOR:-"runs/run_0824_1354.log (gray/roi/4-4-4/H_SHIFT=11: mean IoU 0.9188, rails 0, accum max 52%, response max 49%)"}
else
  COMPARATOR=${COMPARATOR:-"NONE — ${PATCH_ROWS}x${PATCH_COLS} has no comparator run; the stored 128x128 one does NOT apply (different transform gain on all four passes). This build establishes its own budget."}
fi

# Run shape. 200 frames because THE RESPONSE GROWS AS THE FILTER CONVERGES — a
# budget validated at ITER_CNT=2 is not validated, and the retired 4-2-2 point
# peaked at 56% on frame 1 and railed from frame 15.
ITER_CNT=${ITER_CNT:-200}
# VERBOSITY=1 so the per-frame block carries `rails`, which track.csv does NOT.
# This run is NOT an FPS measurement and its frame time is not comparable to the
# 38.04 FPS figure, which was taken at VERBOSITY=0 (docs/thesis/results/perf.csv,
# run_0821_1725, claim P-01).
VERBOSITY=${VERBOSITY:-1}
# 1216 KB and ~2 s per frame. Never on for a 200-frame run.
DUMP_BUFFERS=${DUMP_BUFFERS:-0}

TOTAL=$(( 2 * FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT ))

VARS=(TARGET=hw
      PATCH_ROWS=$PATCH_ROWS
      PATCH_COLS=$PATCH_COLS
      N_CHANNELS=$N_CHANNELS
      CONV_IN_CH=$CONV_IN_CH
      CONV_KSIZE=$CONV_KSIZE
      CONV_STRIDE=$CONV_STRIDE
      CONV_RELU=$CONV_RELU
      WEIGHT_BANK=$WEIGHT_BANK
      ACC_BOUND=$ACC_BOUND
      BIAS_SCALE=$BIAS_SCALE
      FFT_SHIFT=$FFT_SHIFT
      IFFT_ROW_SHIFT=$IFFT_ROW_SHIFT
      IFFT_COL_SHIFT=$IFFT_COL_SHIFT
      H_SHIFT=$H_SHIFT
      SCENE_VERIFY=$SCENE_VERIFY
      FRAME_RGB_MODE=$FRAME_RGB_MODE
      ITER_CNT=$ITER_CNT
      VERBOSITY=$VERBOSITY
      DUMP_BUFFERS=$DUMP_BUFFERS
      CSV_LOG=1
      TRAJECTORY=1
      SCALE_TRAJ=1
      CONV2D_MODE=0)

BUILD_DIR=build/hw/${PATCH_ROWS}x${PATCH_COLS}/ch${N_CHANNELS}

echo "=================================================================="
echo " shift-budget calibration build"
echo "=================================================================="
printf '  arm            %s (CONV_IN_CH=%d, CONV_KSIZE=%d, CONV_STRIDE=%d, CONV_RELU=%d, bank %s)\n' \
       "$ARM" "$CONV_IN_CH" "$CONV_KSIZE" "$CONV_STRIDE" "$CONV_RELU" "$WEIGHT_BANK"
CROP_ROWS_EXP=$(( PATCH_ROWS * CONV_STRIDE ))
CROP_COLS_EXP=$(( PATCH_COLS * CONV_STRIDE ))
printf '  geometry       crop %dx%d -> feature map %dx%d  ch%d   -> %s\n' \
       "$CROP_ROWS_EXP" "$CROP_COLS_EXP" \
       "$PATCH_ROWS" "$PATCH_COLS" "$N_CHANNELS" \
       "build/hw/${PATCH_ROWS}x${PATCH_COLS}/ch${N_CHANNELS}"
printf '  budget         %d-%d-%d   (total 2*%d+%d+%d = %d)\n' \
       "$FFT_SHIFT" "$IFFT_ROW_SHIFT" "$IFFT_COL_SHIFT" \
       "$FFT_SHIFT" "$IFFT_ROW_SHIFT" "$IFFT_COL_SHIFT" "$TOTAL"
printf '  H_SHIFT        %d   (cmul filter-product shift, independent of the budget)\n' "$H_SHIFT"
printf '  bias scale     %s\n' "$BIAS_SCALE"
printf '  frames         %d   verbosity %d   dumps %d\n' \
       "$ITER_CNT" "$VERBOSITY" "$DUMP_BUFFERS"
printf '  mode           %s%s\n' "$MODE" \
       "$([ "$MODE" = bringup ] && echo '   <-- CORRECTNESS GATE. Its amplitudes are NOT a budget result.' || true)"
printf '  scene verify   %d%s\n' "$SCENE_VERIFY" \
       "$([ "$SCENE_VERIFY" = 1 ] && echo '   (O(frame)/frame — short runs only)' || true)"
if [ "$CONV_IN_CH" = 3 ]; then
  printf '  frame rgb mode %d   %s\n' "$FRAME_RGB_MODE" \
    "$([ "$FRAME_RGB_MODE" = 0 ] && echo 'REPLICATED LUMA — this is the COLOUR-FREE CONTROL arm' || echo 'per-plane tint')"
fi
echo

fail=0
note() { printf '  %-42s %s\n' "$1" "$2"; }
bad()  { printf '  %-42s FAIL — %s\n' "$1" "$2"; fail=1; }

# ---- 1. frame count -------------------------------------------------------
# Rule 1 of the shift budget: size it from frames 1-20, not from frame 1.
if [ "$ITER_CNT" -lt 20 ]; then
    if [ "$MODE" = bringup ]; then
        note "frames >= 20" "WAIVED — MODE=bringup ($ITER_CNT frames)"
        note "  -> this image CANNOT validate a shift budget" "correctness gate only"
    else
        bad "frames >= 20" "ITER_CNT=$ITER_CNT cannot show convergence growth (MODE=bringup waives this)"
    fi
else
    note "frames >= 20" "OK ($ITER_CNT)"
fi

# SCENE_VERIFY on a long run is a mistake, not a choice: it re-expands the whole
# frame every push. Catch it here rather than in a run that takes all afternoon.
if [ "$SCENE_VERIFY" = 1 ] && [ "$ITER_CNT" -gt 20 ]; then
    bad "SCENE_VERIFY vs frame count" "SCENE_VERIFY=1 with ITER_CNT=$ITER_CNT is O(frame)/frame — use a short MODE=bringup run"
else
    note "SCENE_VERIFY vs frame count" "OK (verify=$SCENE_VERIFY, frames=$ITER_CNT)"
fi

# ---- 1b. the geometry must be buildable -----------------------------------
# Three things fail SILENTLY at a new patch size, so each is checked here rather
# than after an hours-long build:
#   * conv2d_kernel.cpp and mosse_tracker.cpp both select hanning_<PATCH_COLS>.h.
#     A missing table is an #error, but it fires deep in an aiecompiler log.
#   * roi_crop's BRAM scratch is sized by ROI_MAX_PATCH_*; the patch must fit.
#   * the AIE FFT wants a power of two, and the DSPLib window rule needs
#     PATCH_ROWS % FFT_ROW_WS == 0.
if [ "$PATCH_ROWS" != "$PATCH_COLS" ]; then
    bad "patch is square" "${PATCH_ROWS}x${PATCH_COLS} — the Hanning selection and both graphs assume PATCH_ROWS==PATCH_COLS"
elif [ $(( PATCH_ROWS & (PATCH_ROWS - 1) )) != 0 ]; then
    bad "patch is a power of 2" "$PATCH_ROWS — the AIE FFT point size must be"
else
    note "patch geometry" "OK (${PATCH_ROWS}x${PATCH_COLS})"
fi

HTAB_H=design/aie_src/hanning_${PATCH_COLS}.h
if [ -f "$HTAB_H" ]; then
    note "hanning table" "OK ($HTAB_H)"
else
    bad "hanning table" "$HTAB_H missing — run: make weights PATCH_COLS=$PATCH_COLS CONV_IN_CH=$CONV_IN_CH"
fi

# THE CROP, NOT THE FEATURE MAP. At CONV_STRIDE=2 roi_crop must produce twice
# the feature map on each axis, so a 64x64 map is a 128x128 crop and it is the
# CROP that has to fit the BRAM scratch. Checking PATCH_ROWS here would have
# passed a 128x128 map at stride 2, i.e. a 256x256 crop, silently.
ROI_MAX=$(grep -oP 'define ROI_MAX_PATCH_ROWS\s+\K[0-9]+' design/pl_src/roi_crop/roi_crop.h)
if [ -n "$ROI_MAX" ] && [ "$CROP_ROWS_EXP" -gt "$ROI_MAX" ]; then
    bad "crop fits roi_crop scratch" "crop is ${CROP_ROWS_EXP}x${CROP_COLS_EXP} (PATCH_ROWS=$PATCH_ROWS x CONV_STRIDE=$CONV_STRIDE) > ROI_MAX_PATCH_ROWS=$ROI_MAX"
else
    note "patch fits roi_crop scratch" "OK (max $ROI_MAX)"
fi

# The stored comparator is a 128x128 run. At any other geometry the transform
# gain changes on BOTH the forward and the inverse pass, so its amplitudes are
# not the yardstick and rails=0 has to be re-established from scratch.
if [ "$PATCH_ROWS" != 128 ] || [ "$PATCH_COLS" != 128 ]; then
    note "comparator applies" "NO — ${PATCH_ROWS}x${PATCH_COLS} is not the comparator's geometry"
    note "  -> this is a NEW shift budget" "read arm_res64.md sec.3 before quoting it"
fi

# ---- 2. the weights file must match the arm -------------------------------
# The layout tag makes this checkable. Without it, a CONV_IN_CH=3 build fed a
# grayscale export reads out_shift out of the G plane and tracks nothing.
WBIN=design/aie_src/weights/layer0_weights.bin
if [ ! -f "$WBIN" ]; then
    bad "weights present" "$WBIN missing — run: make weights CONV_IN_CH=$CONV_IN_CH"
else
    # The buffer is CONV_WEIGHT_BYTES_PAD, which is CONV_KSIZE-dependent: 64 B
    # for every 3x3 bank, 192 for 7x7 RGB. The tags are its LAST TWO bytes, so
    # the offset has to be computed rather than hardcoded at 63.
    RAW=$(( CONV_IN_CH * CONV_KSIZE * CONV_KSIZE ))
    WBUF=$(( ((RAW + 15 + 63) / 64) * 64 ))
    TAG=$(od -An -tu1 -j $(( WBUF - 1 )) -N 1 "$WBIN" | tr -d ' ')
    TAGK=$(od -An -tu1 -j $(( WBUF - 2 )) -N 1 "$WBIN" | tr -d ' ')
    [ "$TAG"  = "0" ] && TAG=1       # pre-tag exports were all grayscale
    [ "$TAGK" = "0" ] && TAGK=3      # ...and all 3x3
    WSZ=$(stat -c%s "$WBIN")
    if [ "$TAG" != "$CONV_IN_CH" ] || [ "$TAGK" != "$CONV_KSIZE" ]; then
        bad "weights layout tag" "file is CONV_IN_CH=$TAG CONV_KSIZE=$TAGK, build is $CONV_IN_CH / $CONV_KSIZE — run: make weights CONV_IN_CH=$CONV_IN_CH CONV_KSIZE=$CONV_KSIZE WEIGHT_BANK=$WEIGHT_BANK N_CHANNELS=$N_CHANNELS BIAS_SCALE=$BIAS_SCALE"
    elif [ "$WSZ" != "$(( N_CHANNELS * WBUF ))" ]; then
        # A right-tagged file of the wrong LENGTH is a channel-count mismatch,
        # which the tags cannot see: the host would read past the end of a
        # 16-channel file on a 32-channel build.
        bad "weights channel count" "$WBIN is $WSZ B, build wants N_CHANNELS=$N_CHANNELS x $WBUF = $(( N_CHANNELS * WBUF )) B — run: make weights N_CHANNELS=$N_CHANNELS WEIGHT_BANK=$WEIGHT_BANK CONV_KSIZE=$CONV_KSIZE"
    else
        note "weights layout tag" "OK (CONV_IN_CH=$TAG, CONV_KSIZE=$TAGK, $N_CHANNELS x $WBUF B)"
    fi

    # ---- 2b. and the out_shifts must be the ones ACC_BOUND asks for --------
    # ACC_BOUND does not reach either toolchain -- it only changes the DATA in
    # this file -- so the flagstamps below cannot see it and a stale file would
    # pass every check above. Recompute both bounds from the file's own taps and
    # require the file to agree with the arm. This is the cross-implementation
    # check CLAUDE.md asks for: the shell reads the bytes, the exporter wrote
    # them, and neither trusts the other.
    ACC_MSG=$(env PYTHONHOME= PYTHONPATH= python3 - "$WBIN" "$ACC_BOUND" "$CONV_IN_CH" "$CONV_KSIZE" <<'PYEOF'
import sys
sys.path.insert(0, 'scripts')
import conv_weight_layout as C
path, want, n_in, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
loose = n_in * k * k * 127 * 127
def shift_of(bound, bias):
    v = abs(bias) + bound
    n = 0
    while (v >> n) > 32767:
        n += 1
    return n
mism = []
for i, (taps, sh, bias, dq, mp, lay) in enumerate(C.load_bin(path)):
    exact = 127 * sum(abs(t) for t in taps)
    want_sh = shift_of(exact if want == 'l1' else loose, bias)
    if want_sh != sh:
        mism.append(f"ch{i} file={sh} {want}={want_sh}")
print("OK" if not mism else "MISMATCH " + " ".join(mism[:4]))
PYEOF
)
    case "$ACC_MSG" in
        OK) note "weights acc bound" "OK (out_shift matches ACC_BOUND=$ACC_BOUND on all $N_CHANNELS channels)" ;;
        *)  bad  "weights acc bound" "$ACC_MSG — run: make weights ACC_BOUND=$ACC_BOUND WEIGHT_BANK=$WEIGHT_BANK CONV_IN_CH=$CONV_IN_CH CONV_KSIZE=$CONV_KSIZE N_CHANNELS=$N_CHANNELS" ;;
    esac
fi

# ---- 3. the weights must be at the intended bias scale --------------------
H=design/aie_src/weights/layer0.h
if [ -f "$H" ] && grep -q LAYER0_BIAS_SCALE "$H"; then
    GOT=$(grep LAYER0_BIAS_SCALE "$H" | awk '{print $3}' | tr -d 'f')
    WANT=$([ "$BIAS_SCALE" = roi ] && echo 32.0 || echo 127.0)
    if [ "$GOT" != "$WANT" ]; then
        bad "weights bias scale" "exported at $GOT, build wants $WANT — re-run make weights"
    else
        note "weights bias scale" "OK ($GOT)"
    fi
else
    note "weights bias scale" "unknown (layer0.h predates the field)"
fi

# ---- 4. rootfs ------------------------------------------------------------
# v++ --package corrupts the pristine 2025.2 rootfs; every hw_emu run then
# panics at boot. make rootfs builds the downgraded copy.
if make -n rootfs >/dev/null 2>&1; then note "rootfs target" "available"; fi

# ---- 5. echo mode ---------------------------------------------------------
# In echo mode conv2d does no MAC, no Hanning and all 16 channels are identical,
# so every amplitude and PSR figure is inflated. It cost a ~28 h baseline once.
note "CONV2D_MODE" "0 (real convolution) — passed explicitly below"

echo
if [ "$fail" != 0 ]; then
    echo "PRE-FLIGHT FAILED — nothing built."
    exit 1
fi
echo "pre-flight OK"
echo

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "DRY_RUN=1 — stopping before the build."
    printf '  would run: make sd_card %s\n' "${VARS[*]}"
    exit 0
fi

# ---- build ----------------------------------------------------------------
echo "building (hours) ..."
set -x
make sd_card "${VARS[@]}"
set +x

# ---- 6. THE STAMPS ARE THE AUTHORITY --------------------------------------
# runs/.last_cfg has recorded a configuration the run did not execute. The
# flagstamps are written by the recipe that runs the compiler, so they cannot
# disagree with the binary. Verify AFTER the build, against the binary that
# exists, and record the result next to it.
echo
echo "=== post-build verification (the stamps, not .last_cfg) ==="
# A plain substring grep passes on the WRONG value: "ITER_CNT=200" matches
# "ITER_CNT=2000", and "CMUL_H_SHIFT=1" matches "CMUL_H_SHIFT=10". In a stamp
# every value is terminated by a quote or by whitespace, so require that
# boundary. The check exists to catch a build that did not take the flag; one
# that silently accepts a longer value is the same silent pass in a new place.
stamp_has() {   # stamp_has <file> <FLAG=VALUE>
    grep -qE -- "${2}(\"|[[:space:]]|\$)" "$1"
}
for want in "PATCH_ROWS=$PATCH_ROWS" \
            "PATCH_COLS=$PATCH_COLS" \
            "N_CHANNELS=$N_CHANNELS" \
            "FFT_2D_TP_SHIFT=$FFT_SHIFT" \
            "FFT_2D_TP_IFFT_ROW_SHIFT=$IFFT_ROW_SHIFT" \
            "FFT_2D_TP_IFFT_COL_SHIFT=$IFFT_COL_SHIFT" \
            "CMUL_H_SHIFT=$H_SHIFT" \
            "CONV_IN_CH=$CONV_IN_CH" \
            "CONV_KSIZE=$CONV_KSIZE" \
            "CONV_STRIDE=$CONV_STRIDE" \
            "CONV_RELU=$CONV_RELU" \
            "CONV2D_ECHO_TEST=0"; do
    if stamp_has "$BUILD_DIR/aie.flagstamp" "$want"; then
        printf '  aie.flagstamp  %-34s OK\n' "$want"
    else
        printf '  aie.flagstamp  %-34s MISSING\n' "$want"; fail=1
    fi
done
for want in "-DITER_CNT=$ITER_CNT" "-DCONV_IN_CH=$CONV_IN_CH" \
            "-DCONV_KSIZE=$CONV_KSIZE" "-DCONV_STRIDE=$CONV_STRIDE" \
            "-DCROP_ROWS=$CROP_ROWS_EXP" "-DCROP_COLS=$CROP_COLS_EXP" \
            "-DPATCH_ROWS=$PATCH_ROWS" "-DPATCH_COLS=$PATCH_COLS" \
            "-DN_CHANNELS=$N_CHANNELS" \
            "-DFFT_SHIFT_CFG=$FFT_SHIFT" "-DVERBOSITY=$VERBOSITY" \
            "-DCMUL_H_SHIFT=$H_SHIFT" "-DSCENE_VERIFY=$SCENE_VERIFY" \
            "-DFRAME_RGB_MODE=$FRAME_RGB_MODE" \
            "-DDUMP_BUFFERS=$DUMP_BUFFERS" "-DTRAJECTORY=1"; do
    if stamp_has "$BUILD_DIR/app.flagstamp" "$want"; then
        printf '  app.flagstamp  %-34s OK\n' "$want"
    else
        printf '  app.flagstamp  %-34s MISSING\n' "$want"; fail=1
    fi
done

CFG="$BUILD_DIR/calib_cfg.txt"
{
    date -Is
    echo "arm=$ARM CONV_IN_CH=$CONV_IN_CH BIAS_SCALE=$BIAS_SCALE"
    # The four knobs that MAKE an arm what it is. Without them this record
    # cannot distinguish l1relu from l1lin, or either from a 3x3 build at the
    # same geometry -- the same defect the H_SHIFT note above describes: a run
    # whose variable under test is invisible in its own provenance.
    echo "conv=${CONV_KSIZE}x${CONV_KSIZE} stride=$CONV_STRIDE CONV_RELU=$CONV_RELU bank=$WEIGHT_BANK acc_bound=$ACC_BOUND"
    echo "geometry=crop ${CROP_ROWS_EXP}x${CROP_COLS_EXP} -> map ${PATCH_ROWS}x${PATCH_COLS} channels=$N_CHANNELS"
    echo "budget=${FFT_SHIFT}-${IFFT_ROW_SHIFT}-${IFFT_COL_SHIFT} total=$TOTAL H_SHIFT=$H_SHIFT"
    echo "ITER_CNT=$ITER_CNT VERBOSITY=$VERBOSITY DUMP_BUFFERS=$DUMP_BUFFERS"
    echo "mode=$MODE SCENE_VERIFY=$SCENE_VERIFY FRAME_RGB_MODE=$FRAME_RGB_MODE"
    echo "weights_md5=$(md5sum "$WBIN" | cut -d' ' -f1)"
    echo "comparator=$COMPARATOR"
} > "$CFG"
echo
echo "recorded: $CFG"

[ "$fail" = 0 ] && echo "BUILD VERIFIED" || { echo "BUILD FLAGS DO NOT MATCH — do not run this"; exit 1; }
