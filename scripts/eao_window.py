#!/usr/bin/env python3
"""Re-score EXISTING board trajectories under a different EAO window.

WHY THIS EXISTS. The EAO window is a property of the CHALLENGE, not of the tracker:
VOT-STb2022 integrates the expected-overlap curve over [115, 755], VOT2015 over
[108, 371] (`vot/stack/vot2015/rgb.yaml`). Papers from the two eras are therefore
not comparable on EAO even before the metric definitions are considered (claim
M-17). This script measures how much of that gap is the WINDOW by scoring one set
of trajectories under both, so the term is a number rather than an argument.

IT IS NOT A VOT2015 RUN. Only the integration window changes. The dataset, the
protocol (anchor-based multistart), the failure rule and the ground-truth format all
stay VOT-STb2022's, and the two remaining breaks in M-17 -- R's definition and
polygon-vs-mask ground truth -- are untouched and NOT measured here. A number from
this script is an upper bound on how much of the apparent EAO agreement with a
VOT2015-era paper is a window artefact; it is not a comparable EAO.

METHOD. The workspace is copied (sequences are symlinks, so this is cheap), its
stack.yaml is rewritten to carry an `multistart_eao_score` at EACH window plus the
`multistart_eao_curve`, and `vot analysis` runs once. The scores are THE TOOLKIT'S
OWN, not a reimplementation -- `EAOScore.compute` is `mean(curve[low:high+1])` and
the curve is read back so the sub-window decomposition uses the same array the
scores were integrated from. The [115, 755] column is the CONTROL: it must
reproduce the published row in results/arms.csv, and the script says so per arm.

@thesis sec:porownanieReferencyjne | M-17 | re-score existing trajectories under a
second EAO window, to price the window term in a cross-era comparison
"""
import argparse, glob, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

CURVE_HIGH = 755          # the longest window we integrate; the curve is this long
PUBLISHED = (115, 755)    # VOT-STb2022, and this project's own stack


def build(src: Path, dst: Path, windows):
    """Copy a workspace and pin a stack carrying a score per window plus the curve."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("analysis", "cache"))
    scores = "".join(
        f"      - type: multistart_eao_score\n"
        f"        name: eao_{lo}_{hi}\n"
        f"        low: {lo}\n"
        f"        high: {hi}\n"
        for lo, hi in windows)
    (dst / "stack.yaml").write_text(
        "title: EAO window re-analysis (baseline only)\n"
        "experiments:\n"
        "  baseline:\n"
        "    type: multistart\n"
        "    analyses:\n"
        "      - type: multistart_average_ar\n"
        "        name: ar\n"
        f"{scores}"
        "      - type: multistart_eao_curve\n"
        "        name: eaocurve\n"
        f"        high: {CURVE_HIGH}\n")


def analyse(ws: Path, arms):
    """Run the toolkit. PYTHONPATH/PYTHONHOME are stripped: the Vitis environment
    points python at Vivado's build, which has no _ctypes."""
    vot = Path(sys.executable).parent / "vot"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    r = subprocess.run([str(vot), "analysis", "--workspace", ".", *arms,
                        "--format", "json"],
                       cwd=str(ws), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stdout + r.stderr)[-4000:])
        raise SystemExit("vot analysis failed")
    return json.load(open(sorted(glob.glob(str(ws / "analysis" / "*.json")))[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", type=Path, help="an ingested vot workspace")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="default: every arm the workspace holds")
    ap.add_argument("--window", action="append", default=None,
                    metavar="LOW:HIGH",
                    help="repeatable; default 115:755 (STb2022) and 108:371 (VOT2015)")
    ap.add_argument("--out", type=Path, default=None, help="scratch workspace")
    ap.add_argument("--csv", type=Path, default=None, help="write a long-format CSV")
    a = ap.parse_args()

    windows = [tuple(int(x) for x in w.split(":")) for w in a.window] if a.window \
        else [PUBLISHED, (108, 371)]
    for lo, hi in windows:
        if hi > CURVE_HIGH:
            raise SystemExit(f"window high {hi} exceeds the curve length {CURVE_HIGH}")

    src = a.workspace
    arms = a.arms or sorted(p.name for p in (src / "results").iterdir() if p.is_dir())
    dst = a.out or Path("/tmp") / f"eaowin_{src.name}"
    print(f"workspace {src}  arms {' '.join(arms)}")
    print(f"windows   {'  '.join(f'[{lo}, {hi}]' for lo, hi in windows)}\n")

    build(src, dst, windows)
    d = analyse(dst, arms)

    trackers = list(d["trackers"])
    base = d["results"]["baseline"]
    names = [x["name"] for x in base["parameters"]["analyses"]]

    def rows(which):
        return base["results"][names.index(which)]

    ar = {trackers[i]: r for i, r in enumerate(rows("ar"))}
    curve = {trackers[i]: np.asarray(r[0], dtype=float)
             for i, r in enumerate(rows("eaocurve"))}
    eao = {(lo, hi): {trackers[i]: r[0]
                      for i, r in enumerate(rows(f"eao_{lo}_{hi}"))}
           for lo, hi in windows}

    hdr = f"  {'arm':<12}{'A':>9}{'R':>9}"
    for lo, hi in windows:
        hdr += f"{f'EAO[{lo},{hi}]':>16}"
    if len(windows) == 2:
        hdr += f"{'dEAO':>10}{'ratio':>8}"
    print(hdr)
    for arm in arms:
        line = f"  {arm:<12}{ar[arm][0]:9.4f}{ar[arm][1]:9.4f}"
        vals = [eao[w][arm] for w in windows]
        for v in vals:
            line += f"{v:16.4f}"
        if len(windows) == 2:
            line += f"{vals[1] - vals[0]:+10.4f}{vals[1] / vals[0]:8.3f}"
        print(line)

    # The control. EAOScore is written as mean(curve[low:high+1]) -- but the curve
    # it integrates has LENGTH `high`, so index `high` does not exist and the slice
    # silently returns [low, high-1]. THAT OFF-BY-ONE IS THE TOOLKIT'S PUBLISHED
    # BEHAVIOUR: it is in the [115, 755] number in results/arms.csv exactly as it is
    # in the [108, 371] one, so it cancels in any comparison and is NOT corrected
    # here. Reproducing it (curve[low:high], not [low:high+1]) is what makes this a
    # control rather than a second opinion -- reading the docstring instead cost
    # 3.1e-04 of unexplained residual, three orders below the effect but enough to
    # make the control fire, which is what it is for.
    print("\n  control -- toolkit score vs mean of the returned curve:")
    worst = 0.0
    for arm in arms:
        for lo, hi in windows:
            mine = float(np.mean(curve[arm][lo:hi]))
            worst = max(worst, abs(mine - eao[(lo, hi)][arm]))
    print(f"    max |toolkit - curve mean| = {worst:.2e}"
          f"   {'OK' if worst < 1e-9 else 'MISMATCH -- do not trust the decomposition'}")

    if len(arms) == 2:
        x, y = arms
        print(f"\n  where the two arms differ, by horizon ({y} - {x}):")
        print(f"    {'sub-window':<16}{'share of [115,755]':>20}{'dEO':>10}")
        for lo, hi in [(115, 301), (301, 755), (108, 371), (371, 755)]:
            share = (hi - lo) / (PUBLISHED[1] - PUBLISHED[0])
            d_ = float(np.mean(curve[y][lo:hi] - curve[x][lo:hi]))
            print(f"    [{lo}, {hi-1}]{'':<6}{share:19.0%}{d_:+10.4f}")

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("workspace,arm,low,high,eao,A,R\n")
            for arm in arms:
                for lo, hi in windows:
                    f.write(f"{src.name},{arm},{lo},{hi},{eao[(lo,hi)][arm]:.4f},"
                            f"{ar[arm][0]:.4f},{ar[arm][1]:.4f}\n")
        print(f"\n  wrote {a.csv}")


if __name__ == "__main__":
    main()
