#!/usr/bin/env python3
"""
scripts/vot_ingest.py  --  ingest board trajectories into a VOT workspace and
run `vot analysis`.

WHY THIS EXISTS
---------------
The board writes one trajectory per run onto the NFS results export. Those files
are the run's whole output, but they are not yet a RESULT: AR numbers come out of
the toolkit's own analysis over a workspace whose layout, anchor set and run
lengths all have to agree with the dataset. This script builds that workspace
from an arbitrary number of ARMS (one results directory each), checks the
agreement before the analysis runs, and prints the AR table.

The check is the point. `MultiStartExperiment.scan()` reports a run as missing
only if the FILE is absent; a trajectory of the wrong LENGTH -- the signature of
a reversed backward run, or of a truncated `--vot-max-frames` bring-up -- is
read, scored and reported without complaint. So before any analysis runs, every
run name is re-derived from the sequence's own anchor values and every
trajectory's length is compared against what the multistart order demands:

    forward anchor  i  ->  len(sequence) - i frames
    backward anchor i  ->  i + 1 frames

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_ingest.py \
      --results /srv/vot/results --out $VOT_ROOT/analysis/0825

@thesis sec:metodykaBadan | R-04,M-03 | Board trajectories into a VOT workspace for `vot
  analysis`. It re-derives every run name from the anchors and checks each trajectory's LENGTH.
"""

import argparse
import json
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RUN_RE = re.compile(r"^(?P<seq>.+)_(?P<anchor>\d{8})\.txt$")


def discover(root: Path):
    """arm -> {sequence: {anchor: Path}}, from <root>/<arm>/<seq>_<anchor>.txt."""
    arms = {}
    for armdir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(armdir.glob("*.txt")):
            m = RUN_RE.match(f.name)
            if not m:
                continue
            runs.setdefault(m["seq"], {})[int(m["anchor"])] = f
        if runs:
            arms[armdir.name] = runs
    stray = [f.name for f in root.glob("*.txt")]
    if stray:
        raise SystemExit(
            f"{len(stray)} trajectory file(s) sit at the export ROOT, not in an "
            f"arm directory ({stray[:3]} ...). Move them into a per-arm "
            f"subdirectory first -- an unseparated arm is exactly what gets "
            f"silently overwritten by the next run.")
    return arms


def build_workspace(ws: Path, seq_src: Path, sequences, arms):
    """Workspace skeleton: sequences (symlinked), stack, config, trackers.ini."""
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "results").mkdir(parents=True)
    seqdir = ws / "sequences"
    seqdir.mkdir()
    for s in sequences:
        src = seq_src / s
        if not src.is_dir():
            raise SystemExit(f"sequence '{s}' not found under {seq_src}")
        (seqdir / s).symlink_to(src)
    # list.txt is the DATASET. It carries only the sequences we have results
    # for, so a partial sweep analyses cleanly instead of failing on the rest.
    (seqdir / "list.txt").write_text("\n".join(sequences) + "\n")

    # The published stb2022 stack defines baseline, realtime AND unsupervised,
    # and `vot analysis` runs all three, failing on any without results. The
    # board produces multistart baseline runs only, so pin a local stack with
    # just that experiment -- and add the per-SEQUENCE AR alongside the
    # aggregate, because one number over eight sequences hides that `car1` and
    # the other seven are different populations.
    (ws / "stack.yaml").write_text(
        "title: MOSSE VEK280 multistart (baseline only)\n"
        "experiments:\n"
        "  baseline:\n"
        "    type: multistart\n"
        "    analyses:\n"
        "      - type: multistart_ar\n"
        "        name: arseq\n"
        "      - type: multistart_average_ar\n"
        "        name: ar\n"
        "      - type: multistart_eao_score\n"
        "        name: eaoscore\n"
        "        low: 115\n"
        "        high: 755\n")
    (ws / "config.yaml").write_text(
        "stack: stack.yaml\nsequences: sequences\nregistry:\n- ./trackers.ini\n")
    (ws / "trackers.ini").write_text("".join(
        f"[{a}]\nlabel = {a}\nprotocol = traxpython\ncommand = noop\n\n"
        for a in arms))


def stage(ws: Path, arms):
    """Copy trajectories + their .value sidecars into results/<arm>/baseline/<seq>/."""
    n = 0
    for arm, runs in arms.items():
        for seq, byanchor in runs.items():
            dst = ws / "results" / arm / "baseline" / seq
            dst.mkdir(parents=True, exist_ok=True)
            for anchor, src in byanchor.items():
                shutil.copy2(src, dst / src.name)
                for v in src.parent.glob(f"{src.stem}_*.value"):
                    shutil.copy2(v, dst / v.name)
                n += 1
    return n


def verify(ws: Path, arms):
    """Re-derive every run from the sequence's own anchors and check LENGTH.

    Returns a list of problem strings; empty means the workspace agrees with the
    dataset. A wrong length is the failure this exists for -- the toolkit reads
    it and scores it rather than reporting it.
    """
    from vot.dataset import load_sequence
    from vot.experiment.multistart import find_anchors
    from vot.tracker import Trajectory
    from vot.tracker.results import Results
    from vot.workspace import LocalStorage

    problems = []
    seqdir = ws / "sequences"
    plan = {}
    for s in sorted({s for runs in arms.values() for s in runs}):
        seq = load_sequence(str(seqdir / s))
        fwd, bwd = find_anchors(seq)
        plan[s] = ({i: len(seq) - i for i in fwd}, {i: i + 1 for i in bwd}, len(seq))

    for arm, runs in arms.items():
        storage = LocalStorage(str(ws)).substorage("results").substorage(arm) \
                                       .substorage("baseline")
        for s, byanchor in sorted(runs.items()):
            fwd, bwd, _ = plan[s]
            want = {**fwd, **bwd}
            have = set(byanchor)
            for extra in sorted(have - set(want)):
                problems.append(f"{arm}/{s}: anchor {extra} is not an anchor of "
                                f"this sequence")
            for miss in sorted(set(want) - have):
                problems.append(f"{arm}/{s}: anchor {miss} "
                                f"({'forward' if miss in fwd else 'backward'}) "
                                f"has no trajectory")
            results = Results(storage.substorage(s))
            for anchor in sorted(have & set(want)):
                name = f"{s}_{anchor:08d}"
                t = Trajectory.read(results, name)
                if len(t) != want[anchor]:
                    kind = "forward" if anchor in fwd else "backward"
                    problems.append(
                        f"{arm}/{name}: {len(t)} regions, the {kind} run needs "
                        f"{want[anchor]}")
    return problems, plan


def analyse(ws: Path, arms):
    vot = Path(sys.executable).parent / "vot"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    r = subprocess.run([str(vot), "analysis", "--workspace", ".",
                        *arms, "--format", "json"],
                       cwd=str(ws), env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    for line in tail:
        print("  vot analysis:", line)
    if r.returncode != 0:
        raise SystemExit("vot analysis failed")
    return json.load(open(sorted(glob.glob(str(ws / "analysis" / "*.json")))[-1]))


def report(d, arms):
    trackers = list(d["trackers"])
    base = d["results"]["baseline"]
    names = [a["name"] for a in base["parameters"]["analyses"]]
    sequences = list(d["sequences"])

    def rows(which):
        return base["results"][names.index(which)]

    ar = {trackers[i]: r for i, r in enumerate(rows("ar"))}
    eao = {trackers[i]: r for i, r in enumerate(rows("eaoscore"))}

    print(f"\n  {'arm':<10}{'accuracy':>10}{'robustness':>12}{'EAO':>9}"
          f"{'frames':>9}")
    for a in arms:
        print(f"  {a:<10}{ar[a][0]:10.4f}{ar[a][1]:12.4f}{eao[a][0]:9.4f}"
              f"{ar[a][3]:9d}")

    # per-sequence: `multistart_ar` is a SeparableAnalysis, so its grid is
    # flattened tracker-major -- index i*len(sequences)+j, NOT [i][j].
    seqrows = rows("arseq")
    ns = len(sequences)
    print(f"\n  per sequence ({ns}), A = accuracy, R = robustness:\n")
    head = "  {:<10}".format("sequence")
    for a in arms:
        head += f"{a + ' A':>11}{a + ' R':>11}"
    if len(arms) == 2:
        head += f"{'dA':>9}{'dR':>9}"
    print(head)
    for j, s in enumerate(sequences):
        cells = [seqrows[trackers.index(a) * ns + j] for a in arms]
        line = f"  {s:<10}"
        for c in cells:
            line += f"{c[0]:11.4f}{c[1]:11.4f}"
        if len(arms) == 2:
            line += f"{cells[1][0] - cells[0][0]:+9.4f}{cells[1][1] - cells[0][1]:+9.4f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True,
                    help="directory holding one subdirectory per ARM")
    ap.add_argument('--out', default=None, help="workspace to build")
    ap.add_argument('--sequences', default=None,
                    help="dataset directory (default $VOT_ROOT/workspace/sequences)")
    args = ap.parse_args()

    root = os.environ.get('VOT_ROOT')
    seq_src = Path(args.sequences) if args.sequences else (
        Path(root) / "workspace" / "sequences" if root else None)
    if seq_src is None:
        raise SystemExit("set VOT_ROOT or pass --sequences")
    ws = Path(args.out) if args.out else Path(root) / "analysis" / "ingest"

    arms = discover(Path(args.results))
    if not arms:
        raise SystemExit(f"no arm directories with trajectories under {args.results}")
    sequences = sorted({s for runs in arms.values() for s in runs})

    print(f"workspace: {ws}\nsequences: {seq_src}\n")
    for a, runs in arms.items():
        print(f"  arm {a:<10} {sum(len(v) for v in runs.values()):3d} runs over "
              f"{len(runs)} sequences")

    build_workspace(ws, seq_src, sequences, arms)
    n = stage(ws, arms)
    print(f"\n  staged {n} trajectories")

    problems, _plan = verify(ws, arms)
    if problems:
        for p in problems[:40]:
            print("   ", p)
        if len(problems) > 40:
            print(f"    ... {len(problems)-40} more")
        raise SystemExit(f"workspace does NOT agree with the dataset: "
                         f"{len(problems)} problem(s)")
    print("  verified: every run name and length matches the dataset's anchors")

    print()
    d = analyse(ws, list(arms))
    report(d, list(arms))
    print(f"\nworkspace kept at {ws}")


if __name__ == '__main__':
    main()
