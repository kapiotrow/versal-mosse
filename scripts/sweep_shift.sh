#!/usr/bin/env bash
#
# sweep_shift.sh — sweep FFT_SHIFT along the constant-total-budget line and
# tabulate accumulator headroom, response health and localisation.
#
# WHY A CONSTANT TOTAL
#   The response magnitude is set by the TOTAL budget
#       total = 2*FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT
#   (FFT_SHIFT lands on BOTH forward passes, hence the 2x). Holding the total
#   fixed and moving budget from the IFFT onto the forward pass buys intermediate
#   headroom at ZERO cost to the final response — it costs only spectrum
#   precision. So this is a 1-D search along a constraint line, not a 3-D sweep.
#   gen_aiesim_vectors.py rescales every expected value for the configured budget
#   (IFFT_REF_TOTAL / scale_peak / scale_accum), so the scenarios stay valid.
#
# WHY YOU CAN SWEEP AT ch1 AND PROJECT TO ch16
#   The aiesim harness reuses ONE spectrum and ONE filter for every channel
#   (mosse_graph.cpp, "SCOPE" comment). The accumulator is therefore EXACTLY
#   (k+1)x the single-channel value after channel k — the closed form the harness
#   itself documents. So max|accum| at N channels = N * (the ch1 value), and the
#   ch1 sweep predicts ch16 exactly. This matters because cmul is the dominant
#   simulation cost and the harness loops it once per channel, so a ch16 point
#   costs ~16x a ch1 point (see the SIM_*_TIMEOUT comments in the Makefile).
#   Use ch1 to FIND the budget; use one ch16 run to CONFIRM it.
#
#   It is also the coherent worst case: real per-channel filters differ and
#   partially cancel, so a budget that survives here survives in practice.
#
# USAGE
#   scripts/sweep_shift.sh                      # defaults: s6, 64x64, ch1, shifts 2..5
#   SHIFTS="3 4" NCH=16 PR=128 PC=128 scripts/sweep_shift.sh
#
# Environment knobs (all optional):
#   SHIFTS   space-separated FFT_SHIFT values      (default "2 3 4 5")
#   TOTAL    total shift budget to hold constant   (default 12)
#   SCENARIO aiesim scenario                       (default s6)
#   NCH      N_CHANNELS                            (default 1)
#   PR / PC  PATCH_ROWS / PATCH_COLS               (default 64 / 64)
#   MODE     CONV2D_MODE                           (default 0 = real convolution)
#   WALL     per-point wall-clock cap, seconds     (default 3600)
#   OUT      output table path                     (default sweep/<auto>.md)

set -uo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD

SHIFTS=${SHIFTS:-"2 3 4 5"}
TOTAL=${TOTAL:-12}
SCENARIO=${SCENARIO:-s6}
NCH=${NCH:-1}
PR=${PR:-64}
PC=${PC:-64}
MODE=${MODE:-0}
# Cap the wall clock per point. The Makefile default is 1200*N_CHANNELS, which is
# 5.3 HOURS at ch16 — one hung point would eat the whole sweep before failing.
WALL=${WALL:-3600}
OUT=${OUT:-sweep/shift_${SCENARIO}_${PR}x${PC}_ch${NCH}.md}

mkdir -p sweep

# shellcheck disable=SC1091
source ./setup_env.sh >/dev/null 2>&1 || {
    echo "ERROR: setup_env.sh failed — cannot reach the Vitis toolchain." >&2
    exit 1
}

BUILD_DIR="build/hw_emu/${PR}x${PC}/ch${NCH}"
LOG="${BUILD_DIR}/aiesim_plio.log"

{
    echo "# Shift budget sweep"
    echo
    echo "- scenario: \`${SCENARIO}\`  geometry: ${PR}x${PC}  N_CHANNELS: ${NCH}  CONV2D_MODE: ${MODE}"
    echo "- total budget held at ${TOTAL}: \`IFFT_COL = ${TOTAL} - 2*FFT_SHIFT\`, IFFT_ROW = 0"
    echo "- generated $(date -Is) by \`scripts/sweep_shift.sh\`"
    echo
    echo "| FFT_SHIFT | IFFT_COL | first_sat_ch | accum max {re,im} | proj @ch16 | resp nz | resp max | peak err | accum_sat | resp_sat | OVERALL |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|"
} > "$OUT"

echo "sweep -> $OUT"

for FS in $SHIFTS; do
    COL=$(( TOTAL - 2 * FS ))
    if [ "$COL" -lt 0 ]; then
        echo "skip FFT_SHIFT=$FS (IFFT_COL would be $COL < 0)"
        continue
    fi

    echo "=== FFT_SHIFT=$FS IFFT_COL_SHIFT=$COL ==="
    make aiesim_plio \
        SCENARIO="$SCENARIO" CONV2D_MODE="$MODE" \
        N_CHANNELS="$NCH" PATCH_ROWS="$PR" PATCH_COLS="$PC" \
        FFT_SHIFT="$FS" IFFT_ROW_SHIFT=0 IFFT_COL_SHIFT="$COL" \
        SIM_WALL_TIMEOUT="$WALL" \
        > "sweep/run_${SCENARIO}_${PR}x${PC}_ch${NCH}_fs${FS}.log" 2>&1

    # Keep the full aiesim log for this point — the table is a summary, and the
    # per-channel rails progression only exists in the raw log.
    [ -f "$LOG" ] && cp "$LOG" "sweep/aiesim_${SCENARIO}_${PR}x${PC}_ch${NCH}_fs${FS}.log"

    # --- verify the build actually used these flags ---------------------------
    # A flag-only change reusing a stale libadf.a has produced a convincing false
    # result on this project before. %.flagstamp should force the rebuild; check
    # rather than trust.
    STAMP="${BUILD_DIR}/aie.flagstamp"
    if ! grep -q -- "-DFFT_2D_TP_SHIFT=$FS" "$STAMP" 2>/dev/null \
       || ! grep -q -- "-DFFT_2D_TP_IFFT_COL_SHIFT=$COL" "$STAMP" 2>/dev/null; then
        echo "  !! flagstamp does not carry FFT_SHIFT=$FS / IFFT_COL=$COL — build may be stale"
        FLAGWARN=" **(stale?)**"
    else
        FLAGWARN=""
    fi

    # --- parse ---------------------------------------------------------------
    P="sweep/aiesim_${SCENARIO}_${PR}x${PC}_ch${NCH}_fs${FS}.log"

    firstsat=$(grep -oP 'first saturation at channel \K[0-9]+' "$P" 2>/dev/null | head -1)
    [ -z "$firstsat" ] && firstsat="none"

    amax=$(grep -oP 'accum_out max: \K\{[^}]*\}' "$P" 2>/dev/null | head -1)
    [ -z "$amax" ] && amax="-"

    # Project the worst component to ch16. Exact for this harness (identical
    # channels ⇒ accumulator linear in channel index); flag it if it would rail.
    proj="-"
    if [ "$amax" != "-" ]; then
        re=$(echo "$amax" | tr -d '{}' | cut -d, -f1)
        im=$(echo "$amax" | tr -d '{}' | cut -d, -f2)
        worst=$(python3 -c "print(max(abs($re),abs($im)))" 2>/dev/null)
        if [ -n "$worst" ]; then
            p16=$(( worst * 16 / NCH ))
            if [ "$p16" -gt 32767 ]; then proj="${p16} RAILS"; else
                pct=$(( p16 * 100 / 32767 ))
                proj="${p16} (${pct}%)"
            fi
        fi
    fi

    rnz=$(grep -oP 'response range: \K[0-9]+/[0-9]+' "$P" 2>/dev/null | head -1)
    [ -z "$rnz" ] && rnz="-"
    rmax=$(grep -oP 'response range:.*max\|\.\|=\K[0-9]+' "$P" 2>/dev/null | head -1)
    [ -z "$rmax" ] && rmax="-"

    perr=$(grep -oP 'err=\K[0-9]+ px' "$P" 2>/dev/null | head -1)
    [ -z "$perr" ] && perr="-"

    asat=$(grep -oP 'accum_sat=\K[A-Z]+' "$P" 2>/dev/null | head -1); [ -z "$asat" ] && asat="-"
    rsat=$(grep -oP 'resp_sat=\K[A-Z]+'  "$P" 2>/dev/null | head -1); [ -z "$rsat" ] && rsat="-"
    over=$(grep -oP 'OVERALL: \K[A-Z]+'  "$P" 2>/dev/null | head -1); [ -z "$over" ] && over="NO-RESULT"

    echo "| $FS | $COL | $firstsat | \`$amax\` | $proj | $rnz | $rmax | $perr | $asat | $rsat | ${over}${FLAGWARN} |" >> "$OUT"
    echo "  -> $over  first_sat_ch=$firstsat  accum max=$amax  proj@ch16=$proj  err=$perr"
done

{
    echo
    echo "## Decision rule"
    echo
    echo "1. \`first_sat_ch\` must be \`none\` — zero rails at EVERY channel, not just the"
    echo "   final state. \`HEADROOM EXCEEDED\` and \`accum_sat\` are different checks: a"
    echo "   channel can rail and be pulled back off the rail by a later one, which the"
    echo "   final-state scan misses."
    echo "2. \`resp_sat\` OK and the response not crushed (\`resp nz\` high, \`resp max\` >> 1)."
    echo "   Rails mean the col shift is too low; \`nz = 0\` means it is too high."
    echo "3. \`peak err\` = 0 px."
    echo "4. Among survivors take the SMALLEST FFT_SHIFT — every extra bit of forward"
    echo "   shift is spectrum precision thrown away permanently."
    echo
    echo "\`proj @ch16\` scales the measured worst component to 16 channels. Exact for this"
    echo "harness (identical channels ⇒ linear accumulator), and the coherent worst case."
    echo "Confirm the winner with a real ch16 run before shipping it."
} >> "$OUT"

echo
echo "=== table ==="
cat "$OUT"
