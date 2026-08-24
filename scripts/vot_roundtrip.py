#!/usr/bin/env python3
"""
scripts/vot_roundtrip.py  --  VOT Phase 0b: toolkit result-format round-trip.

WHY THIS EXISTS
---------------
The board will emit trajectories that the VOT toolkit has to ingest. If the
format is subtly wrong the toolkit does not complain -- it analyses the wrong
thing and produces plausible AR numbers. That is the failure mode this spike is
designed to eliminate, so nothing here is derived from documentation: every
artefact is written with the toolkit's OWN writers and read back with its OWN
readers, over a fabricated workspace whose expected answers are known in
advance.

Three fake trackers are scored, and the ORDERING of their results is the
assertion:

  perfect   groundtruth copied verbatim        -> accuracy ~1, 0 failures
  jittered  groundtruth + a few px of noise    -> accuracy slightly below 1
  lost      jumps far off target midway        -> failures > 0

A round-trip that cannot distinguish those three would pass on a broken format.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_roundtrip.py \
      --out $VOT_ROOT/roundtrip
  ... then, as the script prints:
  cd $VOT_ROOT/roundtrip && vot analysis --workspace . perfect jittered lost
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

STACK = "vot2022/shorttermbox"       # = "VOT-ST2022 bounding-box challenge"
SEQ_LEN = 60
SEQ_SIZE = (160, 120)                # (w, h) -- small on purpose, this is a format test
BOX = (24, 18)                       # (w, h)
N_RUNS = 5           # seqa has 3 anchors, seqb 2 -- one trajectory each


# ---------------------------------------------------------------------------
# fabricate a dataset
# ---------------------------------------------------------------------------

def make_sequence(root, name, anchors, seed):
    """One VOT-format sequence: color/%08d.jpg, groundtruth.txt, sequence, *.value.

    `anchors` maps frame index -> +1 (forward run) or -1 (backward run). The
    toolkit's find_anchors() reads exactly this, and the SIGN is the direction,
    so a frame is a forward anchor or a backward anchor, never both.
    """
    from PIL import Image
    from vot.region import Rectangle
    from vot.region.io import write_trajectory
    from vot.utilities import write_properties

    rng = np.random.default_rng(seed)
    d = root / name
    (d / "color").mkdir(parents=True, exist_ok=True)

    w, h = SEQ_SIZE
    bw, bh = BOX
    gt = []
    for i in range(SEQ_LEN):
        # a target on a smooth path, so overlap is meaningful frame to frame
        t = i / SEQ_LEN
        x = 10 + (w - bw - 20) * t
        y = h / 2 - bh / 2 + 20 * np.sin(2 * np.pi * t)
        img = rng.normal(20, 4, (h, w, 3)).clip(0, 255).astype(np.uint8)
        img[int(y):int(y) + bh, int(x):int(x) + bw] = 200
        Image.fromarray(img).save(d / "color" / f"{i+1:08d}.jpg", quality=95)
        gt.append(Rectangle(float(x), float(y), float(bw), float(bh)))

    write_trajectory(str(d / "groundtruth.txt"), gt)

    # Every frame needs a value; float(line) is unconditional in the reader, so
    # a blank line raises rather than meaning "no anchor". 0 = not an anchor.
    (d / "anchor.value").write_text(
        "\n".join(str(anchors.get(i, 0)) for i in range(SEQ_LEN)) + "\n")

    write_properties(str(d / "sequence"),
                     {"name": name, "fps": 30, "format": "default",
                      "channel.default": "color",
                      "channels.color": "color/%08d.jpg"})
    return gt


# ---------------------------------------------------------------------------
# fabricate results, using the toolkit's own writer
# ---------------------------------------------------------------------------

def write_results(ws, tracker, seq_name, gt, anchors, arm, seed):
    """Write one trajectory per anchor, via Trajectory.write().

    The run order is the MULTISTART order, not sequence order: a forward anchor
    at i covers frames [i .. end], a backward anchor covers [i .. 0]. Index 0 of
    the trajectory is always the anchor frame itself, and it is written as
    Special(INITIALIZATION) -- NOT as the init box.
    """
    from vot.region import Rectangle, Special
    from vot.tracker import Trajectory
    from vot.workspace import LocalStorage

    rng = np.random.default_rng(seed)
    storage = LocalStorage(str(ws)).substorage("results") \
                                   .substorage(tracker) \
                                   .substorage("baseline") \
                                   .substorage(seq_name)
    from vot.tracker.results import Results
    results = Results(storage)

    written = []
    for anchor, direction in sorted(anchors.items()):
        if direction == 0:
            continue
        order = (list(range(anchor, len(gt))) if direction > 0
                 else list(reversed(range(0, anchor + 1))))
        traj = Trajectory(len(order))
        traj.set(0, Special(Trajectory.INITIALIZATION), {"time": 26.29})
        for k, f in enumerate(order[1:], start=1):
            g = gt[f]
            if arm == 'perfect':
                r = Rectangle(g.x, g.y, g.width, g.height)
            elif arm == 'jittered':
                r = Rectangle(g.x + rng.normal(0, 1.0), g.y + rng.normal(0, 1.0),
                              g.width, g.height)
            else:                                  # 'lost' -- leaves the target
                off = 0 if k < len(order) // 2 else 90
                r = Rectangle(g.x + off, g.y + off, g.width, g.height)
            traj.set(k, r, {"time": 26.29})
        name = f"{seq_name}_{anchor:08d}"
        traj.write(results, name)
        written.append((name, len(order)))
    return written


def read_back(ws, tracker, seq_name, name):
    from vot.tracker import Trajectory
    from vot.tracker.results import Results
    from vot.workspace import LocalStorage
    storage = LocalStorage(str(ws)).substorage("results") \
                                   .substorage(tracker) \
                                   .substorage("baseline") \
                                   .substorage(seq_name)
    return Trajectory.read(Results(storage), name)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--keep', action='store_true',
                    help="do not wipe an existing workspace first")
    args = ap.parse_args()

    out = args.out or os.environ.get('VOT_ROOT')
    if not out:
        raise SystemExit("set VOT_ROOT or pass --out")
    ws = Path(out) if args.out else Path(out) / "roundtrip"
    if ws.exists() and not args.keep:
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)

    # -- workspace skeleton --------------------------------------------------
    (ws / "results").mkdir(exist_ok=True)
    seqdir = ws / "sequences"
    seqdir.mkdir(exist_ok=True)
    # `registry` is REQUIRED in config.yaml -- omitting it fails the analysis
    # with "Missing arguments: registry", which reads like a CLI problem.
    # The stb2022 stack ("vot2022/shorttermbox") defines THREE experiments --
    # baseline, realtime and unsupervised -- and `vot analysis` runs all of
    # them, failing on any without results. The board produces multistart runs
    # only, so pin a local stack carrying just that experiment. Phase 5 needs
    # this file, or it needs to fabricate the other two.
    (ws / "stack.yaml").write_text(
        "title: MOSSE VEK280 multistart (baseline only)\n"
        "experiments:\n"
        "  baseline:\n"
        "    type: multistart\n"
        "    analyses:\n"
        "      - type: multistart_average_ar\n"
        "        name: ar\n"
        "      - type: multistart_eao_score\n"
        "        name: eaoscore\n"
        "        low: 115\n"
        "        high: 755\n")
    # `registry` is REQUIRED in config.yaml -- omitting it fails the analysis
    # with "Missing arguments: registry", which reads like a CLI problem.
    (ws / "config.yaml").write_text(
        "stack: stack.yaml\nsequences: sequences\nregistry:\n- ./trackers.ini\n")

    # trackers.ini is generated here, so the workspace and any in-repo shim have
    # ONE source for the cross-reference.
    (ws / "trackers.ini").write_text("".join(
        f"[{t}]\nlabel = {t}\nprotocol = traxpython\ncommand = noop\n\n"
        for t in ("perfect", "jittered", "lost")))

    # -- dataset -------------------------------------------------------------
    # Anchors: frame 0 forward, 25 BACKWARD, 40 forward. Both directions are
    # exercised, and the sign convention is the thing under test.
    plans = {
        "seqa": {0: 1, 25: -1, 40: 1},
        "seqb": {0: 1, 30: -1},
    }
    print(f"workspace: {ws}\nstack:     {STACK}\n")
    gts = {}
    for i, (name, anchors) in enumerate(plans.items()):
        gts[name] = make_sequence(seqdir, name, anchors, seed=1 + i)
        print(f"  sequence {name}: {SEQ_LEN} frames {SEQ_SIZE[0]}x{SEQ_SIZE[1]}, "
              f"anchors {anchors}")
    (seqdir / "list.txt").write_text("\n".join(plans) + "\n")

    # -- confirm the toolkit finds the anchors we planted --------------------
    from vot.dataset import load_sequence
    from vot.experiment.multistart import find_anchors
    print()
    ok = True
    for name, anchors in plans.items():
        seq = load_sequence(str(seqdir / name))
        fwd, bwd = find_anchors(seq)
        want_f = sorted(k for k, v in anchors.items() if v > 0)
        want_b = sorted(k for k, v in anchors.items() if v < 0)
        good = (fwd == want_f and bwd == want_b)
        ok &= good
        print(f"  find_anchors({name}) -> forward {fwd} backward {bwd}   "
              f"{'ok' if good else 'MISMATCH want %s / %s' % (want_f, want_b)}")
    if not ok:
        raise SystemExit("anchor round-trip failed")

    # -- write + read back ---------------------------------------------------
    print()
    fails = []
    for arm, seed in (("perfect", 10), ("jittered", 11), ("lost", 12)):
        for name, anchors in plans.items():
            for wname, length in write_results(ws, arm, name, gts[name],
                                               anchors, arm, seed):
                t = read_back(ws, arm, name, wname)
                if len(t) != length:
                    fails.append(f"{arm}/{wname}: read {len(t)} != {length}")
                if "time" not in t.properties():
                    fails.append(f"{arm}/{wname}: no 'time' property sidecar")
                r0 = t.region(0)
                from vot.region import is_special
                if not is_special(r0):
                    fails.append(f"{arm}/{wname}: frame 0 is {r0}, "
                                 f"expected Special(INITIALIZATION)")
        print(f"  {arm:<9} written and read back")

    if fails:
        for f in fails:
            print("   ", f)
        raise SystemExit(f"round-trip FAILED: {len(fails)} problem(s)")

    # -- show what landed on disk -------------------------------------------
    print("\nresult layout:")
    base = ws / "results" / "perfect" / "baseline" / "seqa"
    for p in sorted(base.iterdir()):
        print(f"  results/perfect/baseline/seqa/{p.name}   ({p.stat().st_size} B)")
    # The toolkit writes .bin unless config.results_binary is False. Both are
    # readable; the DEFAULT is binary, which is worth knowing before anyone
    # hand-rolls a text trajectory and wonders why it is ignored.
    from vot import config
    print(f"\nconfig.results_binary = {config.results_binary}  "
          f"-> trajectories are {'BINARY (.bin)' if config.results_binary else 'TEXT (.txt)'}")

    b = sorted(base.glob("seqa_00000000.bin"))
    if b:
        raw = b[0].read_bytes()
        import struct
        ver, n = struct.unpack("<hI", raw[:6])
        # header 6 B; Special = <BI = 5 B; Rectangle = <Bffff = 17 B
        expect = 6 + 5 + (n - 1) * 17
        print(f"  header: version={ver} regions={n}")
        print(f"  size {len(raw)} B, predicted 6 + 5 + {n-1}*17 = {expect} B  "
              f"{'ok' if expect == len(raw) else 'MISMATCH'}")
        code0, = struct.unpack("<B", raw[6:7])
        sp, = struct.unpack("<I", raw[7:11])
        print(f"  region 0: type={code0} (0=Special) code={sp} "
              f"(1=INITIALIZATION)")
        c1, x, y, w, h = struct.unpack("<Bffff", raw[11:28])
        print(f"  region 1: type={c1} (1=Rectangle) "
              f"x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f}")
    tv = sorted(base.glob("*_time.value"))[0]
    print(f"\n  {tv.name} first 2 lines: "
          f"{tv.read_text().splitlines()[:2]}")

    print("\nROUND-TRIP OK -- running the analysis\n")
    run_analysis(ws)


def run_analysis(ws):
    """Run `vot analysis` and ASSERT the three arms score in the expected order.

    A round-trip that produces numbers is not evidence; a round-trip that
    produces numbers which move the right way when the input is deliberately
    degraded is.
    """
    import glob
    import json
    import subprocess

    vot = Path(sys.executable).parent / "vot"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    r = subprocess.run([str(vot), "analysis", "--workspace", ".",
                        "perfect", "jittered", "lost", "--format", "json"],
                       cwd=str(ws), env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
    print("  vot analysis:", *tail)
    if r.returncode != 0:
        raise SystemExit("vot analysis failed")

    d = json.load(open(sorted(glob.glob(str(ws / "analysis" / "*.json")))[-1]))
    tr = list(d["trackers"])
    base = d["results"]["baseline"]
    ai = [a["name"] for a in base["parameters"]["analyses"]].index("ar")
    ar = {tr[i]: row for i, row in enumerate(base["results"][ai])}

    print(f"\n  {'arm':<10}{'accuracy':>10}{'robustness':>12}{'frames':>8}")
    for k, v in ar.items():
        print(f"  {k:<10}{v[0]:10.4f}{v[1]:12.4f}{v[3]:8d}")

    total = ar["perfect"][3]                 # frames tracked with no failure
    n_runs = N_RUNS
    checks = [
        ("perfect robustness == 1", ar["perfect"][1] == 1.0),
        ("jittered robustness == 1", ar["jittered"][1] == 1.0),
        ("lost robustness < 0.6", ar["lost"][1] < 0.6),
        ("jittered accuracy < perfect", ar["jittered"][0] < ar["perfect"][0]),
        # THE INIT FRAME COUNTS IN THE ACCURACY DENOMINATOR WITH OVERLAP 0.
        # A perfect tracker therefore scores exactly (total - runs) / total,
        # never 1.0. Pinned here so a toolkit upgrade that changes it is loud.
        (f"perfect accuracy == ({total}-{n_runs})/{total}",
         abs(ar["perfect"][0] - (total - n_runs) / total) < 1e-12),
    ]
    print()
    bad = 0
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += not ok
    if bad:
        raise SystemExit(f"{bad} assertion(s) failed")
    print("\nPHASE 0b PASS: format round-trip + analysis discriminate correctly.")


if __name__ == '__main__':
    main()
