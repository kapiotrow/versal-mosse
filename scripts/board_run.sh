#!/bin/sh
# board_run.sh — run the tracker on the board and RETURN when it is finished.
#
# WHY THIS EXISTS
# ---------------
# The host does not exit after the last frame: `gr.end(0)` blocks forever on a
# free-running graph. That has been documented as cosmetic since the manual flow
# killed it by hand. It stops being cosmetic the moment runs are automated:
#
#   * `ssh <board> ./mosse_tracker.elf ...` never returns, so a sweep can drive
#     exactly ONE sequence and then hangs;
#   * the finished process keeps the ELF open, so the next arm's push fails with
#     scp's uninformative `dest open ...: Failure` (ETXTBSY);
#   * it keeps the XRT device context.
#
# It sleeps at 0% CPU while doing all that, so nothing about it is visible in a
# log or a load average. Both symptoms were found on the sweep script's second
# real use, one by the push failing and one by inspecting the board.
#
# So: run detached, stream the output, and return once the run has printed its
# process-wide summary — then kill the corpse. The marker is the LAST thing a
# completed run prints, and the wrapper reports which way it ended so a timeout
# can never be mistaken for a clean finish.
#
# Usage:  board_run.sh <timeout_s> <elf args...>

TIMEOUT="$1"; shift
DIR=/tmp/mosse
LOG=$DIR/run.out
MARKER='== UNATTRIBUTED'      # last line of the [apu] CUMULATIVE block

cd "$DIR" || exit 90
: > "$LOG"

XILINX_XRT=/usr ./mosse_tracker.elf "$@" > "$LOG" 2>&1 &
PID=$!

tail -f "$LOG" &
TAILPID=$!

rc=91
i=0
while [ "$i" -lt "$TIMEOUT" ]; do
    # EXITED ON ITS OWN IS NOT THE SAME AS FINISHED. An ELF that aborts at
    # startup -- a weights layout-tag mismatch, a missing manifest, a refused
    # mount -- exits in under a second, and reporting that as rc=0 tells the
    # sweep the sequence ran. Check for the completion marker before believing
    # it. (2026-08-26: gray was pushed RGB weights and this returned success.)
    if ! kill -0 "$PID" 2>/dev/null; then
        if grep -q "$MARKER" "$LOG" 2>/dev/null; then rc=0; else rc=93; fi
        break
    fi
    # The cumulative block is process-wide and prints once, after the last run.
    if grep -q "$MARKER" "$LOG" 2>/dev/null && \
       grep -q 'CUMULATIVE over' "$LOG" 2>/dev/null; then
        sleep 2                                                    # let the tail catch up
        rc=0
        break
    fi
    sleep 1
    i=$((i + 1))
done

sleep 1
kill "$TAILPID" 2>/dev/null

if kill -0 "$PID" 2>/dev/null; then
    if [ "$rc" -eq 0 ]; then
        echo "[board_run] run complete; killing the process that gr.end(0) will not release"
    else
        echo "[board_run] TIMEOUT after ${TIMEOUT}s with no completion marker — the run did NOT finish"
    fi
    kill "$PID" 2>/dev/null
    sleep 2
    kill -9 "$PID" 2>/dev/null
fi
exit "$rc"
