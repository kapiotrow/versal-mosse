#!/bin/sh
# power_probe.sh -- sample whatever power/thermal channels a host exposes, as CSV.
#
# @thesis subsec:metrykiSystemowe | P-12 | The board-side sampler behind the energy-per-frame
#   measurement: it discovers its channels instead of hardcoding them, and refuses to emit an
#   empty CSV.
#
# WHY THIS EXISTS, AND WHY IT IS A SEPARATE PROCESS
# ------------------------------------------------
# The thesis promises energy per frame (`subsec:metrykiSystemowe`) and nothing in
# `docs/thesis/results/` measures power. This is half of paying that debt; the other
# half is scripts/power_measure.py, which drives the protocol and does the arithmetic.
#
# It runs BESIDE the tracker, never inside it. The frame budget is the project's most
# re-measured quantity (results/perf.csv is 15 rows of it) and an instrument that adds
# a syscall to the frame loop would move the very number the energy figure divides by.
# For the sc_app backend it does not even run on the same chip.
#
# WHERE THE POWER ACTUALLY COMES FROM
# -----------------------------------
# Measured on the board 2026-09-03: the APU can see NO current sensor. `hwmon0` is
# `versal_thermal` (a temperature and nothing else), `iio:device0` is `xlnx,versal-sysmon`
# (seven REGULATED voltages, flat to +/-2 mV over a second, plus die temperature), the
# device tree declares no ina*/ir38*/irps* node, PMC `i2c@f1000000` is `disabled`, and
# `xrt-smi examine -r electrical` answers "No report generator found" on the edge shell.
#
# So watts come from the VEK280 System Controller, which owns the INA226 rails and
# exposes them through `sc_app`. That is the `sc_app` backend, and it runs ON THE SC.
# The `sysmon` backend runs on the board, yields no watts at all, and exists for the
# die-temperature channel and as a control: a run where both backends are sampled shows
# whether a power delta and a thermal delta appear together.
#
# OUTPUT IS LONG FORMAT
# ---------------------
#   t_remote,channel,unit,value
# One row per channel per sample. Long rather than wide because the channel SET is
# discovered at startup and differs per board revision and per backend; a wide header
# would have to be guessed, and a guessed header is how a renamed rail becomes a silently
# missing column. power_measure.py pivots.
#
# A PARSER THAT FINDS NOTHING MUST NOT LOOK LIKE A CLEAN RUN
# ----------------------------------------------------------
# That trap has been paid for in this repo already (claim M-09: a statistic read 0.0000
# for a whole sweep beside five green unit tests). So:
#   * discovery that yields zero channels EXITS 3 before sampling anything;
#   * a channel that parses on one sample and not the next emits `nan`, never a dropped
#     row -- the sample count per channel is then a check the driver can apply;
#   * `--list` dumps the backend's RAW output and exits, so the parser can be tightened
#     against what the instrument really prints rather than against what it should.
#
# Usage:
#   power_probe.sh --backend sc_app|sysmon [--rails "A B C"|all]
#                  [--period-ms 500] [--duration-s N] [--list]
#
# Intended invocation is over a single ssh with the script on stdin, so nothing has to be
# installed on the sampling host:
#   ssh root@<sc> sh -s -- --backend sc_app --period-ms 500 < scripts/power_probe.sh

BACKEND=""
# The rails that matter for THIS design; "--rails all" takes every rail listpower reports.
# VCCINT is the PL+AIE core rail, VCC_PSFP/VCC_PSLP_CPM5 the APU, VCC1V1_LP4 the LPDDR4.
RAILS="VCCINT VCC_SOC VCC_PMC VCC_PSFP VCC_PSLP_CPM5 VCC1V1_LP4 VCCAUX VCCAUX_PMC"
PERIOD_MS=500
DURATION_S=0          # 0 = until killed
LIST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)    BACKEND="$2"; shift 2 ;;
        --rails)      RAILS="$2"; shift 2 ;;
        --period-ms)  PERIOD_MS="$2"; shift 2 ;;
        --duration-s) DURATION_S="$2"; shift 2 ;;
        --list)       LIST=1; shift ;;
        -h|--help)    sed -n '2,60p' "$0" 2>/dev/null; exit 0 ;;
        # A typo must not fall back to a default -- the same rule vot_sweep.sh applies.
        *) echo "power_probe: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

case "$BACKEND" in
    sc_app|sysmon) ;;
    *) echo "power_probe: --backend must be sc_app|sysmon, got '$BACKEND'" >&2; exit 2 ;;
esac

SYSMON=/sys/bus/iio/devices/iio:device0
THERMAL=/sys/class/hwmon/hwmon0

# ----------------------------------------------------------------------------
# Discovery. Prints one "channel unit source" triple per line on stdout.
# ----------------------------------------------------------------------------
discover_sysmon() {
    for f in "$SYSMON"/in_voltage*_input; do
        [ -r "$f" ] || continue
        b=$(basename "$f")
        # in_voltage5_vcc_soc_input -> vcc_soc
        n=${b#in_voltage}; n=${n#*_}; n=${n%_input}
        echo "$n mV $f"
    done
    [ -r "$SYSMON/in_temp160_temp_input" ] && echo "die_temp mC $SYSMON/in_temp160_temp_input"
    [ -r "$THERMAL/temp1_input" ]          && echo "versal_thermal mC $THERMAL/temp1_input"
    return 0
}

# CONFIRMED against the real System Controller 2026-09-03 (image
# xilinx-versal-system-controller-20222, sc_app in /usr/bin), which is why this is
# `listpower` and not `listpowerdomain`: the domain commands answer
# "ERROR: power domain operation is not supported" on this image. One query costs 12 ms
# and returns all three quantities at once:
#
#     $ sc_app -c getpower -t VCCINT
#     Voltage(V):     0.8050
#     Current(A):     4.9271
#     Power(W):       1.7091
#
# ALL THREE ARE EMITTED, per rail, and that is a control rather than completeness.
# `Power(W)` is a separate INA226 register, not V*I computed: over 30 idle samples it
# tracked V*I to -0.57% +/- 2.63%, but ONE sample in 30 was 13% low and an earlier one
# read 1.7091 W where V*I said 3.97 -- a 55% glitch. With V rock-steady at 0.1%, the
# product is the cross-check that catches those, and a glitch is worth catching: at 30
# samples one of them moves the mean by 2%.
discover_sc_app() {
    command -v sc_app >/dev/null 2>&1 || {
        echo "power_probe: sc_app not found on this host -- is this the System Controller?" >&2
        return 1
    }
    avail=$(sc_app -c listpower 2>/dev/null | tr -d '\r' \
            | sed -n 's/^[[:space:]]*\([A-Za-z0-9_.-]\{1,\}\)[[:space:]]*$/\1/p')
    [ -n "$avail" ] || return 1
    if [ "$RAILS" = "all" ]; then
        want="$avail"
    else
        want="$RAILS"
    fi
    for r in $want; do
        # A NAMED RAIL THAT THE BOARD DOES NOT HAVE IS FATAL, not skipped: silently
        # dropping it would produce a narrower measurement that still looks complete.
        if ! echo "$avail" | grep -qx "$r"; then
            echo "power_probe: rail '$r' is not on this board. Available:" >&2
            echo "$avail" | tr '\n' ' ' >&2; echo >&2
            return 1
        fi
        echo "$r W $r"
    done
    return 0
}

if [ "$LIST" -eq 1 ]; then
    echo "# power_probe --list, backend=$BACKEND, host=$(uname -n 2>/dev/null)"
    if [ "$BACKEND" = sc_app ]; then
        echo "--- sc_app -c listpower ---";   sc_app -c listpower   2>&1
        echo "--- sc_app -c listvoltage ---"; sc_app -c listvoltage 2>&1
        # One live read, so the per-sample parser can be checked against real text.
        first=$(discover_sc_app 2>/dev/null | head -1 | cut -d' ' -f1)
        if [ -n "$first" ]; then
            echo "--- sc_app -c getpower -t $first ---"
            sc_app -c getpower -t "$first" 2>&1
        fi
        echo "--- rails selected ---"; discover_sc_app 2>&1
    else
        echo "--- $SYSMON ---"; ls "$SYSMON" 2>&1
        discover_sysmon
    fi
    exit 0
fi

CHANNELS=$(if [ "$BACKEND" = sc_app ]; then discover_sc_app; else discover_sysmon; fi)
N=$(echo "$CHANNELS" | grep -c '[^[:space:]]')

if [ -z "$CHANNELS" ] || [ "$N" -eq 0 ]; then
    echo "power_probe: backend '$BACKEND' discovered ZERO channels -- refusing to emit an" >&2
    echo "  empty CSV that would read as a clean run. Try --list to see what it saw." >&2
    exit 3
fi

echo "# power_probe backend=$BACKEND host=$(uname -n 2>/dev/null) channels=$N period_ms=$PERIOD_MS"
echo "$CHANNELS" | while read -r name unit src; do
    [ -n "$name" ] && echo "# channel $name $unit $src"
done
echo "t_remote,channel,unit,value"

# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
# BusyBox sleep takes a fraction, and so does coreutils'. A period below ~200 ms is not
# useful anyway: every one of these sensors is an I2C or ADC read averaging over many
# frames at 24 ms/frame, which is why this measures ENERGY PER FRAME over a sustained
# run and can never attribute power to a pipeline stage.
PERIOD_S=$(awk "BEGIN{printf \"%.3f\", $PERIOD_MS/1000.0}")
DEADLINE=0
[ "$DURATION_S" -gt 0 ] && DEADLINE=$(( $(date +%s) + DURATION_S ))

# BusyBox date has no %N: it returns the literal string, which would put "1748553727.%N"
# in every row and quietly destroy sub-second resolution. Probed once here rather than
# assumed -- the board's own date did exactly this on the first run of this script.
# The board clock is ALSO unset (it read 2025-05-29 while the PC read 2026-09-03), which
# is why t_remote is a diagnostic and power_measure.py stamps arrival time itself.
if date +%s.%N 2>/dev/null | grep -q '%N'; then HAVE_NS=0; else HAVE_NS=1; fi

trap 'exit 0' TERM INT

while :; do
    if [ "$HAVE_NS" -eq 1 ]; then T=$(date +%s.%N); else T=$(date +%s); fi
    echo "$CHANNELS" | while read -r name unit src; do
        [ -n "$name" ] || continue
        if [ "$BACKEND" = sysmon ]; then
            v=$(cat "$src" 2>/dev/null)
            case "$v" in
                ''|*[!0-9.eE+-]*) v=nan ;;
            esac
            echo "$T,$name,$unit,$v"
        else
            # ONE query, three rows. A rail that stops parsing yields nan on all three
            # rather than vanishing, so the per-channel sample count stays a check.
            sc_app -c getpower -t "$src" 2>/dev/null | tr -d '\r' \
              | awk -v t="$T" -v n="$name" '
                  /^[Vv]oltage/ { for (i=NF;i>=1;i--) if ($i ~ /^-?[0-9.]+$/) { v=$i; break } }
                  /^[Cc]urrent/ { for (i=NF;i>=1;i--) if ($i ~ /^-?[0-9.]+$/) { c=$i; break } }
                  /^[Pp]ower/   { for (i=NF;i>=1;i--) if ($i ~ /^-?[0-9.]+$/) { p=$i; break } }
                  END {
                      printf "%s,%s_V,V,%s\n", t, n, (v == "" ? "nan" : v)
                      printf "%s,%s_I,A,%s\n", t, n, (c == "" ? "nan" : c)
                      printf "%s,%s,W,%s\n",   t, n, (p == "" ? "nan" : p)
                  }'
        fi
    done
    [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ] && break
    sleep "$PERIOD_S"
done
exit 0
