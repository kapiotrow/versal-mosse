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
#   --stream MODE    passed to --vot-stream: auto|always|never (default: auto).
#                    auto streams a sequence whose blob exceeds the ELF's
#                    VOT_RESIDENT_MAX_MB and stages the rest in heap, which is
#                    what the five oversized RGB sequences need. always/never
#                    force one mode for every sequence in the sweep -- that is
#                    the MODE-EQUIVALENCE TEST: streaming changes no arithmetic,
#                    so the same sequence run both ways must come back with
#                    IDENTICAL run-state digests. See docs/thesis/evidence/TODO_board_memory.md
#   --elf PATH       host ELF to push           (default: the hw build's)
#   --out DIR        logs + config              (default: runs/vot/<date>-<arm>)
#   --board HOST     (default 192.168.10.2)
#   --data DIR       PC-side data export (default /srv/vot/data). The RGB arm's
#                    blobs are a SEPARATE export: <seq>.raw is the same filename
#                    for both channel counts, so they cannot share a directory
#   --resume         keep existing results, skip sequences already complete
#   --ingest         run scripts/vot_ingest.py over the results when done
#   --dry-run        print every remote command instead of running it

# @thesis sec:metodykaBadan | B-07,R-09 | One sweep, one command: mount, push, guard the build
#   against the board's xclbin, run, collect, ingest -- and record the flagstamps beside the
#   results.

set -euo pipefail
cd "$(dirname "$0")/.."

ARM=""; SEQS=""; JOBS="all"; STREAM="auto"; ELF=""; OUT=""; BOARD="192.168.10.2"
RESUME=0; INGEST=0; DRY=0
RUN_TIMEOUT=3600
DATA_MNT="/mnt/vot"; RES_MNT="/mnt/vot-results"
PC_DATA="/srv/vot/data"; PC_RESULTS="/srv/vot/results"; PC_IP="192.168.10.1"
WORK="/tmp/mosse"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --seqs) SEQS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --stream) STREAM="$2"; shift 2 ;;
        --elf) ELF="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --data) PC_DATA="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --ingest) INGEST=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1' -- a typo must not fall back to a default" >&2; exit 1 ;;
    esac
done

# Validated HERE as well as in the ELF. The ELF's check is the one that matters,
# but a typo caught after mounting, pushing and starting a 62-sequence sweep is
# 62 failed runs where this is one line.
case "$STREAM" in
    auto|always|never) ;;
    *) echo "ERROR: --stream must be auto|always|never, got '$STREAM'" >&2; exit 1 ;;
esac

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
        echo "        failure -- see docs/thesis/evidence/evidence_arm0.md.)" >&2
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
# An unstamped run cannot be stamped after the fact: its numbers are unciteable
# and the only repair is to sweep it again. So a missing flagstamp is FATAL here,
# not a silent skip. calib_cfg.txt is advisory -- it only exists for builds made
# through calib_build.sh -- so that one warns instead.
for f in aie.flagstamp app.flagstamp crop.flagstamp; do
    if [[ ! -f "$BUILD_DIR/$f" ]]; then
        echo "FATAL: $BUILD_DIR/$f is missing -- this run would be unciteable." >&2
        echo "       Rebuild through scripts/calib_build.sh, or delete $OUT and" >&2
        echo "       accept that the arm cannot go in the thesis." >&2
        exit 1
    fi
    cp "$BUILD_DIR/$f" "$OUT/config/"
done
if [[ -f "$BUILD_DIR/calib_cfg.txt" ]]; then
    cp "$BUILD_DIR/calib_cfg.txt" "$OUT/config/"
else
    echo "  WARNING: no calib_cfg.txt (build did not come through calib_build.sh)"
fi
{
    echo "date       $(date -Is)"
    echo "arm        $ARM"
    echo "jobs       $JOBS"
    echo "stream     $STREAM"
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
# MOUNTED IS NOT ENOUGH -- THE SOURCE HAS TO MATCH.
#
# This used to be `mounted? then skip`, which is correct while there is only one
# data export and silently wrong the moment there are two. The gray blobs live
# in /srv/vot/data and the RGB ones in /srv/vot/data-rgb; after a gray sweep
# leaves $DATA_MNT mounted, an RGB sweep with --data would have skipped the
# mount and run the whole arm against GRAY blobs.
#
# That particular case fails loudly downstream -- the host refuses a manifest
# whose `channels` disagrees with CONV_IN_CH -- but it fails after the push and
# the staging, and only because a second guard happens to cover it. Two arms of
# the same channel count would not have been caught at all. So: remount when the
# source differs, and say so.
rsh "cur=\$(awk -v t=$DATA_MNT '\$2==t {print \$1}' /proc/mounts); \
     want=$PC_IP:$PC_DATA; \
     if [ -n \"\$cur\" ] && [ \"\$cur\" != \"\$want\" ]; then \
       echo \"  [mount] $DATA_MNT is \$cur, want \$want -- remounting\"; \
       umount $DATA_MNT || exit 1; cur=; \
     fi; \
     [ -n \"\$cur\" ] || mount -t nfs -o vers=3,nolock,ro,rsize=1048576,proto=tcp \$want $DATA_MNT"
rsh "grep -q ' $RES_MNT ' /proc/mounts || mount -t nfs -o vers=3,nolock,rw,rsize=1048576,proto=tcp $PC_IP:$PC_RESULTS $RES_MNT"
rsh "mkdir -p $RES_MNT/$ARM"
if [[ $DRY -eq 0 ]]; then
    rshq "test -f $DATA_MNT/$(echo "$LIST" | head -1).json" || {
        echo "ERROR: $DATA_MNT is mounted but the first sequence's manifest is not there" >&2; exit 1; }
    echo "  data + results mounted, $RES_MNT/$ARM ready"
fi

# --- push the host-side artifacts ------------------------------------------
# --- clear the previous run before touching the ELF -----------------------
# THE HOST DOES NOT EXIT AFTER THE LAST FRAME. `gr.end(0)` blocks forever on a
# free-running graph -- a known, documented, cosmetic-looking defect that stops
# being cosmetic the moment a second run follows a first: the finished process
# still holds the ELF open (so the push fails ETXTBSY, reported by scp as the
# uninformative `dest open ...: Failure`) and still holds the XRT device
# context. It sleeps at 0% CPU, so it perturbs nothing and shows up as nothing.
#
# Found on this script's SECOND run, by the push failing. Killing it is exactly
# what the manual flow did by hand; the run's own work is already complete when
# it gets here -- trajectories are written per sequence, before the block.
say "clear"
if [[ $DRY -eq 0 ]]; then
    left=$(rshq "ps w 2>/dev/null | grep -v grep | grep mosse_tracker.elf | wc -l")
    if [[ "${left:-0}" -gt 0 ]]; then
        echo "  $left leftover mosse_tracker.elf still resident (gr.end(0) never returns) — killing"
        rshq "for p in \$(ps w | grep -v grep | grep mosse_tracker.elf | awk '{print \$1}'); do kill \$p; done" || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            n=$(rshq "ps w 2>/dev/null | grep -v grep | grep mosse_tracker.elf | wc -l")
            [[ "${n:-0}" -eq 0 ]] && break
            sleep 1
        done
        n=$(rshq "ps w 2>/dev/null | grep -v grep | grep mosse_tracker.elf | wc -l")
        if [[ "${n:-0}" -gt 0 ]]; then
            rshq "for p in \$(ps w | grep -v grep | grep mosse_tracker.elf | awk '{print \$1}'); do kill -9 \$p; done" || true
            sleep 2
        fi
        n=$(rshq "ps w 2>/dev/null | grep -v grep | grep mosse_tracker.elf | wc -l")
        [[ "${n:-0}" -eq 0 ]] || { echo "ERROR: could not clear the previous run (still $n)" >&2; exit 1; }
        echo "  cleared"
    else
        echo "  nothing resident"
    fi
else
    echo "  [board] kill any resident mosse_tracker.elf"
fi

say "push"
if [[ $DRY -eq 0 ]]; then
    scp -q -o BatchMode=yes "$ELF" root@"$BOARD":"$WORK/mosse_tracker.elf"
    scp -q -o BatchMode=yes scripts/board_run.sh root@"$BOARD":"$WORK/"
    scp -q -o BatchMode=yes design/aie_src/weights/layer0_weights.bin root@"$BOARD":"$WORK/"
    [[ -f "$CARD_SRC/xrt.ini" ]] && scp -q -o BatchMode=yes "$CARD_SRC/xrt.ini" root@"$BOARD":"$WORK/"
    rshq "chmod +x $WORK/mosse_tracker.elf $WORK/board_run.sh; ln -sf ${CARD}a.xclbin $WORK/a.xclbin"
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
    # COMPLETE MEANS THE RIGHT NUMBER, NOT "SOME". A sequence writes one
    # trajectory per anchor, so an interrupted sequence leaves a PARTIAL set --
    # frisbee came back 4 of 6 after the board lost power on 2026-08-26. The old
    # test was `any .txt exists`, which would have skipped it and left the sweep
    # permanently two runs short. `vot_ingest.py` does check for a missing file,
    # so it would eventually have been caught, but only after the board was
    # packed up and only if someone read the warning.
    #
    # The job count comes from the manifest, which is the same authority the
    # board uses to decide how many runs to do.
    if [[ $RESUME -eq 1 ]]; then
        # `|| have=0` for the same pipefail reason as `n=` below: compgen -G
        # exits 1 when nothing matches, the substitution inherits it, and set -e
        # kills the sweep. Zero matches is the NORMAL case for an unrun
        # sequence, so this line runs 38 times a sweep and must not be fatal.
        have=$(compgen -G "$RESDIR/${s}_*.txt" 2>/dev/null | wc -l) || have=0
        want=$(env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python -c \
               "import json,sys;print(len(json.load(open(sys.argv[1]))['jobs']))" \
               "$PC_DATA/$s.json" 2>/dev/null || echo 0)
        if [[ "$want" -gt 0 && "$have" -eq "$want" ]]; then
            echo "  $s: complete ($have/$want), skipping"
            skipped=$((skipped + 1)); continue
        elif [[ "$have" -gt 0 ]]; then
            # Re-run and overwrite. A partial set cannot be topped up: the board
            # runs a whole sequence per invocation and rewrites every anchor.
            echo "  $s: PARTIAL ($have/$want) -- re-running the whole sequence"
            # DELETED FROM THE BOARD, NOT THE PC. The export is rw for the board
            # and root_squash maps its root to `nobody`, so the trajectories are
            # owned by nobody and the PC user cannot unlink them. Removing them
            # is not strictly required -- the re-run rewrites every anchor -- but
            # a stale file from a FAILED attempt is indistinguishable from a
            # fresh one, and that is exactly the state this power cut produced.
            rsh "rm -f $RES_MNT/$ARM/${s}_*.txt $RES_MNT/$ARM/${s}_*_time.value"
        fi
    fi
    log="$OUT/$s.log"
    # Driven through board_run.sh, which returns when the run has printed its
    # summary. Running the ELF directly would hang the sweep forever: gr.end(0)
    # never returns. RUN_TIMEOUT bounds a genuinely stuck run and is reported as
    # a timeout, never as a clean finish.
    cmd="sh $WORK/board_run.sh $RUN_TIMEOUT ./a.xclbin \
--vot-data $DATA_MNT --vot-results $RES_MNT/$ARM --vot-seq $s --vot-jobs $JOBS \
--vot-stream $STREAM"
    if [[ $DRY -eq 1 ]]; then
        echo "  [board] $cmd"
        continue
    fi
    printf '  %-10s -> %s\n' "$s" "$log"
    # Timestamps come from the PC side of the pipe. Over ssh they are good to
    # about a second -- fine for locating a stall, NOT a frame-time instrument.
    # Frame time comes from the run's own AP_* slots and track.csv.
    rc=0
    $SSH "$cmd" 2>&1 | ts '%H:%M:%.S' > "$log" || rc=$?
    # `|| n=0` IS LOAD-BEARING. Under `set -o pipefail` a command substitution
    # inherits the pipeline's status, so when a sequence produces NO
    # trajectories the `ls` fails, the substitution fails, and `set -e` kills
    # the whole sweep -- one bad sequence silently abandoning every sequence
    # after it. That happened on 2026-08-26: gray was pushed RGB weights, the
    # ELF refused them at startup (correctly), and the sweep stopped dead with
    # 37 sequences unrun and nothing in the log saying so.
    n=$(ls "$RESDIR/${s}_"*.txt 2>/dev/null | wc -l) || n=0
    # The board names it track_<sequence>.csv at FRAME_SOURCE=vot. Resolved by
    # asking the board rather than by composing the name here, so a change to
    # csv_open()'s sanitisation cannot silently make this fetch nothing.
    # NAMED EXACTLY, not `| tail -1`: the CSVs accumulate in $WORK across the
    # sweep, so "the last one listed" is the alphabetically last sequence, not
    # the one that just ran. `ls` on the exact name also verifies it exists.
    SEQ_CSV=$(rshq "ls $WORK/track_${s}.csv 2>/dev/null" || true)
    # ZERO TRAJECTORIES IS A FAILURE, not a run that happened to write nothing.
    # Counting it as `ok` is how a sweep reports success over an empty result.
    if [[ $rc -eq 0 && $n -gt 0 ]]; then
        echo "      done, $n trajectories"
        ok=$((ok + 1))
        # FETCH THIS SEQUENCE'S CSV NOW, not at the end of the sweep.
        #
        # track.csv is written to the board's CWD (/tmp/mosse) because putting
        # it on the NFS mount would move a filesystem sync into the timed path.
        # /tmp is tmpfs, so a reboot erases every CSV the sweep has not yet
        # collected -- and the end-of-sweep collect never runs if the sweep is
        # interrupted. Two power cuts on 2026-08-26 destroyed 61 of 62 CSVs that
        # way while every trajectory survived, because trajectories go to the
        # results mount per sequence and the CSVs did not.
        #
        # The amplitudes in that CSV ARE the survey; the trajectories are not a
        # substitute for them. One scp per sequence is ~500 KB and costs
        # nothing next to a 4-minute run.
        if [[ -n "$SEQ_CSV" ]]; then
            scp -q -o BatchMode=yes root@"$BOARD":"$SEQ_CSV" "$OUT/" 2>/dev/null \
                || echo "      WARNING: could not fetch $SEQ_CSV"
        else
            # Silence here would be the whole failure mode again: a sweep that
            # tracks perfectly and collects no amplitudes.
            echo "      WARNING: no track_${s}.csv on the board (CSV_LOG=0?)"
        fi
    else
        if [[ $rc -eq 0 ]]; then
            echo "      FAILED: run returned cleanly but wrote NO trajectories (see $log)"
        else
            echo "      FAILED rc=$rc (see $log)"
        fi
        tail -3 "$log" | sed 's/^/        /'
        failed=$((failed + 1))
    fi
done

[[ $DRY -eq 1 ]] && { echo; echo "dry run complete — nothing was executed"; exit 0; }

# --- collect the per-sequence CSVs -----------------------------------------
say "collect"
# ENUMERATE, THEN FETCH BY NAME. Do NOT pass a remote glob to scp: since
# OpenSSH 9 scp speaks SFTP by default, and a remote wildcard then transfers
# NOTHING AND SAYS NOTHING -- no error text, just no file. It cost this script
# its first real run's CSVs, and the failure message blamed CSV_LOG, which was
# not the cause. `scp -O` (legacy protocol) also works; enumerating works on
# both, so it does not depend on which scp is installed.
csvs=$(rshq "ls $WORK/track_*.csv 2>/dev/null" || true)
if [[ -n "$csvs" ]]; then
    n=0
    while read -r f; do
        [[ -z "$f" ]] && continue
        # stderr is NOT discarded here: a silent collect failure is exactly what
        # this block exists to prevent.
        if scp -q -o BatchMode=yes root@"$BOARD":"$f" "$OUT/"; then
            n=$((n + 1))
        else
            echo "  WARNING: could not fetch $f"
        fi
    done <<< "$csvs"
    echo "  $n track_*.csv -> $OUT"
else
    echo "  no track_*.csv on the board — CSV_LOG=0, or the run never opened one"
fi

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
