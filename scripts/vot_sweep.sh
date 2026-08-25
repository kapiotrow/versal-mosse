#!/bin/bash
# vot_sweep.sh — drive a whole VOT sweep on the board from the PC.
#
# What it replaces
# ----------------
# Per session: `ip addr add`, two `mount -t nfs`, then one hand-typed ELF
# invocation per sequence, watched through picocom. Per arm: a card swap. The
# 8-sequence coast A/B was 54 runs driven that way, and it produced two real
# defects that had nothing to do with the tracker — an arm staged with the wrong
# build flags, and trajectories written to the export root where the next arm
# would have silently overwritten them.
#
# So this script's job is not only convenience. Three of its steps exist because
# the manual flow has already failed at exactly that point:
#
#   * it REFUSES to start when the arm's results directory is non-empty
#     (--resume to continue an interrupted sweep), because the overwrite is
#     silent: same filenames, a successful write message, nothing in either log;
#   * it compares the board's a.xclbin against the one in the PC's package tree
#     and refuses on a mismatch, because the ELF it pushes is only meaningful
#     against the bitstream it was built with;
#   * it records the flagstamps, the ELF md5 and the full build config beside
#     the results, because `runs/.last_cfg` once recorded a configuration the
#     run did not execute.
#
# The ELF is PUSHED, not flashed. It is 163 KB and every host-only knob
# (HOLD_COAST, PSR_GATE_MIN, sub-bin interpolation, PROGRESS_EVERY) leaves the
# xclbin untouched, so an arm change is an scp and not a card swap. The weights
# and xrt.ini go with it: weights carry a layout tag the host checks at runtime,
# and xrt.ini is read from the process's CWD — a run from a directory without it
# silently loses Runtime.rw_shared.
#
# Usage
# -----
#   scripts/vot_sweep.sh --arm coast0 --seqs car1,tiger,nature
#   scripts/vot_sweep.sh --arm subbin1 --seqs @runs/vot/seqs8.txt --ingest
#   scripts/vot_sweep.sh --arm x --seqs car1 --dry-run     # print, run nothing
#
#   --arm NAME       results subdirectory, and the name the ingest scores under
#   --seqs LIST      comma-separated, or @file with one per line
#   --jobs SPEC      passed to --vot-jobs        (default: all)
#   --elf PATH       host ELF to push           (default: the hw build's)
#   --out DIR        logs + config              (default: runs/vot/<date>-<arm>)
#   --board HOST     (default 192.168.10.2)
#   --resume         keep existing results, skip sequences already complete
#   --ingest         run scripts/vot_ingest.py over the results when done
#   --dry-run        print every remote command instead of running it

set -euo pipefail
cd "$(dirname "$0")/.."

ARM=""; SEQS=""; JOBS="all"; ELF=""; OUT=""; BOARD="192.168.10.2"
RESUME=0; INGEST=0; DRY=0
DATA_MNT="/mnt/vot"; RES_MNT="/mnt/vot-results"
PC_DATA="/srv/vot/data"; PC_RESULTS="/srv/vot/results"; PC_IP="192.168.10.1"
WORK="/tmp/mosse"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --seqs) SEQS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --elf) ELF="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --ingest) INGEST=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1' -- a typo must not fall back to a default" >&2; exit 1 ;;
    esac
done

[[ -n "$ARM"  ]] || { echo "ERROR: --arm is required" >&2; exit 1; }
[[ -n "$SEQS" ]] || { echo "ERROR: --seqs is required" >&2; exit 1; }
[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: --arm must be [A-Za-z0-9._-]" >&2; exit 1; }

if [[ "$SEQS" == @* ]]; then
    LIST=$(grep -vE '^\s*(#|$)' "${SEQS#@}")
else
    LIST=${SEQS//,/$'\n'}
fi
[[ -n "$LIST" ]] || { echo "ERROR: sequence list is empty" >&2; exit 1; }

ELF=${ELF:-build/hw/128x128/ch16/mosse_tracker.elf}
BUILD_DIR=$(dirname "$ELF")
CARD_SRC="$BUILD_DIR/package/sd_card"
OUT=${OUT:-runs/vot/$(date +%m%d_%H%M)-$ARM}

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@$BOARD"
# No -t on purpose: a pty would put bare \r into the logs, and picocom's \r is
# what made line-number ranges disagree between grep and python during the
# coast A/B analysis. Without a pty the stream is clean \n.

say()  { printf '\n=== %s\n' "$*"; }
rsh()  { if [[ $DRY -eq 1 ]]; then echo "  [board] $*"; else $SSH "$*"; fi; }
rshq() { $SSH "$*"; }          # always real: used for probes, never for effects

# --- PC-side preconditions -------------------------------------------------
say "preflight"
[[ -f "$ELF" ]] || { echo "ERROR: no ELF at $ELF (make application TARGET=hw)" >&2; exit 1; }
[[ -f "$CARD_SRC/a.xclbin" ]] || { echo "ERROR: no packaged xclbin at $CARD_SRC/a.xclbin" >&2; exit 1; }
[[ -d "$PC_DATA" ]] || { echo "ERROR: $PC_DATA missing -- run scripts/vot_prepare.py first" >&2; exit 1; }

for s in $LIST; do
    [[ -f "$PC_DATA/$s.json" && -f "$PC_DATA/$s.raw" ]] || {
        echo "ERROR: sequence '$s' has no blob/manifest in $PC_DATA" >&2; exit 1; }
done

RESDIR="$PC_RESULTS/$ARM"
if [[ -d "$RESDIR" ]] && compgen -G "$RESDIR/*.txt" >/dev/null; then
    if [[ $RESUME -eq 0 ]]; then
        echo "ERROR: $RESDIR already holds trajectories." >&2
        echo "       Pass --resume to continue that sweep, or pick another --arm." >&2
        echo "       (An arm silently overwriting another is a real, already-observed" >&2
        echo "        failure -- see runs/vot/evidence_arm0.md.)" >&2
        exit 1
    fi
    echo "  resuming into $RESDIR ($(ls "$RESDIR"/*.txt 2>/dev/null | wc -l) trajectories present)"
fi

echo "  arm       $ARM"
echo "  sequences $(echo "$LIST" | tr '\n' ' ')"
echo "  elf       $ELF ($(md5sum "$ELF" | cut -c1-12))"
echo "  out       $OUT"
[[ $DRY -eq 1 ]] && echo "  DRY RUN — no remote command will be executed"

mkdir -p "$OUT/config"

# --- reachability ----------------------------------------------------------
say "board"
if [[ $DRY -eq 0 ]]; then
    rshq true || { echo "ERROR: cannot ssh to root@$BOARD." >&2
        echo "       Was the card flashed from a provisioned image?  make board_provision" >&2
        echo "       Check the link first:  ping -c1 $BOARD" >&2; exit 1; }
    echo "  $(rshq 'uname -srm')"
    CARD=$(rshq 'for d in /run/media/*/ /media/*/ /mnt/sd*/; do [ -f "$d/a.xclbin" ] && echo "$d" && break; done')
    [[ -n "$CARD" ]] || { echo "ERROR: no directory with a.xclbin found on the board" >&2; exit 1; }
    echo "  card      $CARD"
else
    CARD="/run/media/mmcblk0p1/"
fi

# --- the guard that makes pushing an ELF safe ------------------------------
# The ELF is deliberately allowed to differ from the card's. The BITSTREAM is
# not: a host built for a different geometry links against a graph that is not
# there, and the failure would be a plausible tracking result rather than an
# error.
say "build agreement"
PC_XCLBIN_MD5=$(md5sum "$CARD_SRC/a.xclbin" | awk '{print $1}')
if [[ $DRY -eq 0 ]]; then
    BOARD_XCLBIN_MD5=$(rshq "md5sum ${CARD}a.xclbin" | awk '{print $1}')
    if [[ "$PC_XCLBIN_MD5" != "$BOARD_XCLBIN_MD5" ]]; then
        echo "ERROR: the card's a.xclbin is not the one this build produced." >&2
        echo "       PC    $PC_XCLBIN_MD5  ($CARD_SRC/a.xclbin)" >&2
        echo "       board $BOARD_XCLBIN_MD5  (${CARD}a.xclbin)" >&2
        echo "       Re-flash the card, or point --elf at the build that matches it." >&2
        exit 1
    fi
    echo "  a.xclbin matches: ${PC_XCLBIN_MD5:0:12}"
else
    echo "  [board] md5sum ${CARD}a.xclbin  (compared against ${PC_XCLBIN_MD5:0:12})"
fi

# --- record what is about to run -------------------------------------------
# Beside the results, not in a shared file that a later run can rewrite.
for f in aie.flagstamp app.flagstamp crop.flagstamp calib_cfg.txt; do
    [[ -f "$BUILD_DIR/$f" ]] && cp "$BUILD_DIR/$f" "$OUT/config/"
done
{
    echo "date       $(date -Is)"
    echo "arm        $ARM"
    echo "jobs       $JOBS"
    echo "sequences  $(echo "$LIST" | tr '\n' ' ')"
    echo "elf        $ELF"
    echo "elf_md5    $(md5sum "$ELF" | awk '{print $1}')"
    echo "xclbin_md5 $PC_XCLBIN_MD5"
    echo "weights    $(md5sum design/aie_src/weights/layer0_weights.bin 2>/dev/null | awk '{print $1}')"
    echo "git        $(git rev-parse HEAD 2>/dev/null) $(git diff --quiet 2>/dev/null && echo clean || echo DIRTY)"
} > "$OUT/config/sweep.txt"
echo "  recorded $OUT/config/sweep.txt"

# --- mounts, idempotent ----------------------------------------------------
say "mounts"
# The results export is mounted at RES_MNT and the ARM is a directory INSIDE it.
# Mounting the export AT .../<arm> is what put arm A's 54 trajectories in the
# export root, one board run away from being overwritten.
rsh "mkdir -p $DATA_MNT $RES_MNT $WORK"
rsh "grep -q ' $DATA_MNT ' /proc/mounts || mount -t nfs -o vers=3,nolock,ro,rsize=1048576,proto=tcp $PC_IP:$PC_DATA $DATA_MNT"
rsh "grep -q ' $RES_MNT ' /proc/mounts || mount -t nfs -o vers=3,nolock,rw,rsize=1048576,proto=tcp $PC_IP:$PC_RESULTS $RES_MNT"
rsh "mkdir -p $RES_MNT/$ARM"
if [[ $DRY -eq 0 ]]; then
    rshq "test -f $DATA_MNT/$(echo "$LIST" | head -1).json" || {
        echo "ERROR: $DATA_MNT is mounted but the first sequence's manifest is not there" >&2; exit 1; }
    echo "  data + results mounted, $RES_MNT/$ARM ready"
fi

# --- push the host-side artifacts ------------------------------------------
say "push"
if [[ $DRY -eq 0 ]]; then
    scp -q -o BatchMode=yes "$ELF" root@"$BOARD":"$WORK/mosse_tracker.elf"
    scp -q -o BatchMode=yes design/aie_src/weights/layer0_weights.bin root@"$BOARD":"$WORK/"
    [[ -f "$CARD_SRC/xrt.ini" ]] && scp -q -o BatchMode=yes "$CARD_SRC/xrt.ini" root@"$BOARD":"$WORK/"
    rshq "chmod +x $WORK/mosse_tracker.elf; ln -sf ${CARD}a.xclbin $WORK/a.xclbin"
    PUSHED=$(rshq "md5sum $WORK/mosse_tracker.elf" | awk '{print $1}')
    [[ "$PUSHED" == "$(md5sum "$ELF" | awk '{print $1}')" ]] || {
        echo "ERROR: the ELF on the board does not match the one sent" >&2; exit 1; }
    echo "  elf + weights + xrt.ini in $WORK, a.xclbin symlinked, md5 verified"
else
    echo "  [scp] $ELF -> $WORK/  (+ weights, xrt.ini; a.xclbin symlinked)"
fi

# --- run -------------------------------------------------------------------
say "sweep"
ok=0; skipped=0; failed=0
for s in $LIST; do
    if [[ $RESUME -eq 1 && -n $(compgen -G "$RESDIR/${s}_*.txt" 2>/dev/null || true) ]]; then
        echo "  $s: already has trajectories, skipping"
        skipped=$((skipped + 1)); continue
    fi
    log="$OUT/$s.log"
    cmd="cd $WORK && XILINX_XRT=/usr ./mosse_tracker.elf ./a.xclbin \
--vot-data $DATA_MNT --vot-results $RES_MNT/$ARM --vot-seq $s --vot-jobs $JOBS"
    if [[ $DRY -eq 1 ]]; then
        echo "  [board] $cmd"
        continue
    fi
    printf '  %-10s -> %s\n' "$s" "$log"
    # Timestamps come from the PC side of the pipe. Over ssh they are good to
    # about a second -- fine for locating a stall, NOT a frame-time instrument.
    # Frame time comes from the run's own AP_* slots and track.csv.
    if $SSH "$cmd" 2>&1 | ts '%H:%M:%.S' > "$log"; then
        n=$(ls "$RESDIR/${s}_"*.txt 2>/dev/null | wc -l)
        echo "      done, $n trajectories"
        ok=$((ok + 1))
    else
        echo "      FAILED (see $log)"; tail -3 "$log" | sed 's/^/        /'
        failed=$((failed + 1))
    fi
done

[[ $DRY -eq 1 ]] && { echo; echo "dry run complete — nothing was executed"; exit 0; }

# --- collect the per-sequence CSVs -----------------------------------------
say "collect"
scp -q -o BatchMode=yes root@"$BOARD":"$WORK/track_*.csv" "$OUT/" 2>/dev/null \
    && echo "  $(ls "$OUT"/track_*.csv 2>/dev/null | wc -l) track_*.csv" \
    || echo "  no track_*.csv on the board (CSV_LOG=0?)"

say "summary"
echo "  $ok ran, $skipped skipped, $failed failed"
echo "  trajectories: $(ls "$RESDIR"/*.txt 2>/dev/null | wc -l) in $RESDIR"
echo "  logs + config: $OUT"
[[ $failed -eq 0 ]] || exit 1

if [[ $INGEST -eq 1 ]]; then
    say "ingest"
    env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_ingest.py \
        --results "$PC_RESULTS" --out "${VOT_ROOT:-$HOME/vot}/analysis/$(basename "$OUT")"
fi
