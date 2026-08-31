#!/usr/bin/env bash
# offline_sweep_par.sh -- run rgb_vs_gray_loop.py over all 62 sequences in
# PARALLEL, one process per sequence, then merge.
#
# WHY ONE JSON PER WORKER AND A MERGE AT THE END. rgb_vs_gray_loop.py's --json
# does read-modify-write on a single file. Serially that is fine; in parallel it
# is a lost-update race AND a torn read -- observed directly while the mask-power
# sweep was running, as `json.decoder.JSONDecodeError: Expecting ',' delimiter`
# from a reader that caught the file mid-write. So each worker owns a private
# file and the merge is a separate, serial step.
#
# Usage:
#   scripts/offline_sweep_par.sh <out.json> <jobs> <arm> [arm ...] -- --eta E --psr-min G [...]
#   SEQS=/path/to/list.txt scripts/offline_sweep_par.sh ...     # smoke-test subset

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="$1"; shift
JOBS="$1"; shift
ARMS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then shift; EXTRA=("$@"); break; fi
    ARMS+=("$1"); shift
done
[[ ${#ARMS[@]} -gt 0 ]] || { echo "no arms given" >&2; exit 2; }

# THE BENCH'S MODULE DEFAULTS ARE NOT THE SHIPPING CONFIG, AND OMITTING THEM
# SILENTLY SCORES A DIFFERENT TRACKER. rgb_vs_gray_loop.py carries ETA = 0.125
# and PSR_GATE_MIN = 7.0 -- the pre-2026-08-28 values -- while the shipping arm
# is eta 0.05 / gate 5.0, and every stored sweep passed them explicitly. A sweep
# run without them moved the baseline off the recorded board-form control
# (docs/thesis/evidence/pooled_features.md, A 0.5394 / R 0.2910 / 5792 tracked)
# to 0.5277 / 0.3014 / 5998. THE BASELINE CONTROL CAUGHT IT, not
# the arm -- which is the whole reason a baseline is re-run in every invocation.
# Refuse rather than score an arm against the wrong control.
EXTRA_STR=" ${EXTRA[*]:-} "
case "$EXTRA_STR" in
    *" --eta "*) ;;
    *) echo "ERROR: pass --eta after -- (shipping 0.05; bench default 0.125)" >&2; exit 2 ;;
esac
case "$EXTRA_STR" in
    *" --psr-min "*) ;;
    *) echo "ERROR: pass --psr-min after -- (shipping 5.0; bench default 7.0)" >&2; exit 2 ;;
esac

# Overridable so the driver can be smoke-tested on a handful of sequences IN
# PLACE. Copying the script to /tmp instead breaks its own
# `cd "$(dirname "$0")/.."`, which resolves to / there.
SEQS="${SEQS:-runs/vot/seqs62.txt}"

PARTS=$(mktemp -d)
trap 'rm -rf "$PARTS"' EXIT

echo "arms: ${ARMS[*]}"
echo "jobs: $JOBS   sequences: $(wc -l < "$SEQS")   extra: ${EXTRA[*]}"

# Each sequence is independent and CPU-bound, so the sequence is the unit of
# work. Workers are silenced individually; a failure is reported by exit status.
xargs -a "$SEQS" -I{} -P "$JOBS" \
    env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py \
        --sequence {} --arms "${ARMS[@]}" "${EXTRA[@]}" --json "$PARTS/{}.json" \
    > /dev/null 2>&1 || { echo "at least one worker failed" >&2; exit 1; }

env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python - "$PARTS" "$OUT" <<'PY'
import glob, json, os, sys
parts, out = sys.argv[1], sys.argv[2]
merged = {}
files = sorted(glob.glob(os.path.join(parts, '*.json')))
for f in files:
    merged.update(json.load(open(f)))
json.dump(merged, open(out, 'w'))
print(f"merged {len(files)} sequence files -> {out}  ({len(merged)} seq|arm records)")
PY
