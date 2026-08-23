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
#   gray (default)  CONV_IN_CH=1, BIAS_SCALE=roi. ONE variable moves against
#                   runs/run_0821_1725.log (mean IoU 0.9188, rails 0, peak
#                   49-64% of int16). That log is the known-good comparator and
#                   it used this exact geometry, trajectory and frame count.
#   rgb             CONV_IN_CH=3. Do NOT run this first: it moves the bias scale
#                   AND the feature bank AND the input scale at once, and a bad
#                   result would be unattributable. See "never move two
#                   magnitudes at once" in CLAUDE.md.
#
# USAGE
#   scripts/calib_build.sh                 # gray, 4-4-4, 200 frames
#   ARM=rgb scripts/calib_build.sh         # after gray has been validated
#   FFT_SHIFT=4 IFFT_ROW_SHIFT=5 IFFT_COL_SHIFT=5 scripts/calib_build.sh
#   DRY_RUN=1 scripts/calib_build.sh       # pre-flight only, build nothing
#
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${ARM:-gray}
case "$ARM" in
  gray) CONV_IN_CH=1 ;;
  rgb)  CONV_IN_CH=3 ;;
  *) echo "ARM must be gray or rgb, got '$ARM'" >&2; exit 2 ;;
esac

# The budget under test. Defaults are the incumbent; change these to sweep.
FFT_SHIFT=${FFT_SHIFT:-4}
IFFT_ROW_SHIFT=${IFFT_ROW_SHIFT:-4}
IFFT_COL_SHIFT=${IFFT_COL_SHIFT:-4}
BIAS_SCALE=${BIAS_SCALE:-roi}

# Run shape. 200 frames because THE RESPONSE GROWS AS THE FILTER CONVERGES — a
# budget validated at ITER_CNT=2 is not validated, and the retired 4-2-2 point
# peaked at 56% on frame 1 and railed from frame 15.
ITER_CNT=${ITER_CNT:-200}
# VERBOSITY=1 so the per-frame block carries `rails`, which track.csv does NOT.
# This run is NOT an FPS measurement and its frame time is not comparable to the
# 38.04 FPS figure, which was taken at VERBOSITY=0.
VERBOSITY=${VERBOSITY:-1}
# 1216 KB and ~2 s per frame. Never on for a 200-frame run.
DUMP_BUFFERS=${DUMP_BUFFERS:-0}

TOTAL=$(( 2 * FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT ))

VARS=(TARGET=hw
      CONV_IN_CH=$CONV_IN_CH
      BIAS_SCALE=$BIAS_SCALE
      FFT_SHIFT=$FFT_SHIFT
      IFFT_ROW_SHIFT=$IFFT_ROW_SHIFT
      IFFT_COL_SHIFT=$IFFT_COL_SHIFT
      ITER_CNT=$ITER_CNT
      VERBOSITY=$VERBOSITY
      DUMP_BUFFERS=$DUMP_BUFFERS
      CSV_LOG=1
      TRAJECTORY=1
      SCALE_TRAJ=1
      CONV2D_MODE=0)

BUILD_DIR=build/hw/128x128/ch16

echo "=================================================================="
echo " shift-budget calibration build"
echo "=================================================================="
printf '  arm            %s (CONV_IN_CH=%d)\n' "$ARM" "$CONV_IN_CH"
printf '  budget         %d-%d-%d   (total 2*%d+%d+%d = %d)\n' \
       "$FFT_SHIFT" "$IFFT_ROW_SHIFT" "$IFFT_COL_SHIFT" \
       "$FFT_SHIFT" "$IFFT_ROW_SHIFT" "$IFFT_COL_SHIFT" "$TOTAL"
printf '  bias scale     %s\n' "$BIAS_SCALE"
printf '  frames         %d   verbosity %d   dumps %d\n' \
       "$ITER_CNT" "$VERBOSITY" "$DUMP_BUFFERS"
echo

fail=0
note() { printf '  %-42s %s\n' "$1" "$2"; }
bad()  { printf '  %-42s FAIL — %s\n' "$1" "$2"; fail=1; }

# ---- 1. frame count -------------------------------------------------------
# Rule 1 of the shift budget: size it from frames 1-20, not from frame 1.
if [ "$ITER_CNT" -lt 20 ]; then
    bad "frames >= 20" "ITER_CNT=$ITER_CNT cannot show convergence growth"
else
    note "frames >= 20" "OK ($ITER_CNT)"
fi

# ---- 2. the weights file must match the arm -------------------------------
# The layout tag makes this checkable. Without it, a CONV_IN_CH=3 build fed a
# grayscale export reads out_shift out of the G plane and tracks nothing.
WBIN=design/aie_src/weights/layer0_weights.bin
if [ ! -f "$WBIN" ]; then
    bad "weights present" "$WBIN missing — run: make weights CONV_IN_CH=$CONV_IN_CH"
else
    TAG=$(od -An -tu1 -j 63 -N 1 "$WBIN" | tr -d ' ')
    [ "$TAG" = "0" ] && TAG=1        # pre-tag exports were all grayscale
    if [ "$TAG" != "$CONV_IN_CH" ]; then
        bad "weights layout tag" "file is CONV_IN_CH=$TAG, build is $CONV_IN_CH — run: make weights CONV_IN_CH=$CONV_IN_CH BIAS_SCALE=$BIAS_SCALE"
    else
        note "weights layout tag" "OK (CONV_IN_CH=$TAG)"
    fi
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
for want in "FFT_2D_TP_SHIFT=$FFT_SHIFT" \
            "FFT_2D_TP_IFFT_ROW_SHIFT=$IFFT_ROW_SHIFT" \
            "FFT_2D_TP_IFFT_COL_SHIFT=$IFFT_COL_SHIFT" \
            "CONV_IN_CH=$CONV_IN_CH" \
            "CONV2D_ECHO_TEST=0"; do
    if grep -q -- "$want" "$BUILD_DIR/aie.flagstamp"; then
        printf '  aie.flagstamp  %-34s OK\n' "$want"
    else
        printf '  aie.flagstamp  %-34s MISSING\n' "$want"; fail=1
    fi
done
for want in "-DITER_CNT=$ITER_CNT" "-DCONV_IN_CH=$CONV_IN_CH" \
            "-DFFT_SHIFT_CFG=$FFT_SHIFT" "-DVERBOSITY=$VERBOSITY" \
            "-DDUMP_BUFFERS=$DUMP_BUFFERS" "-DTRAJECTORY=1"; do
    if grep -q -- "$want" "$BUILD_DIR/app.flagstamp"; then
        printf '  app.flagstamp  %-34s OK\n' "$want"
    else
        printf '  app.flagstamp  %-34s MISSING\n' "$want"; fail=1
    fi
done

CFG="$BUILD_DIR/calib_cfg.txt"
{
    date -Is
    echo "arm=$ARM CONV_IN_CH=$CONV_IN_CH BIAS_SCALE=$BIAS_SCALE"
    echo "budget=${FFT_SHIFT}-${IFFT_ROW_SHIFT}-${IFFT_COL_SHIFT} total=$TOTAL"
    echo "ITER_CNT=$ITER_CNT VERBOSITY=$VERBOSITY DUMP_BUFFERS=$DUMP_BUFFERS"
    echo "weights_md5=$(md5sum "$WBIN" | cut -d' ' -f1)"
    echo "comparator=runs/run_0821_1725.log (mean IoU 0.9188, rails 0)"
} > "$CFG"
echo
echo "recorded: $CFG"

[ "$fail" = 0 ] && echo "BUILD VERIFIED" || { echo "BUILD FLAGS DO NOT MATCH — do not run this"; exit 1; }
