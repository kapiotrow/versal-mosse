#!/usr/bin/env python3
"""Run a HOST-SIDE tracker under the BOARD's multistart protocol, and write the
board's trajectory format so `vot_ingest.py` scores it with the toolkit.

WHY THIS EXISTS. Two gaps, one mechanism.

  1. THE SCORING PATH HAS NEVER BEEN CHECKED AGAINST A PUBLISHED NUMBER. The
     pinned stack is config-identical to the toolkit's own vot2022/shorttermbox
     baseline (verified 2026-09-03), but the sequence conversion, the anchor set,
     the trajectory format and the staging have only ever been checked against
     THEMSELVES. Every row of results/arms.csv rests on that path. traps.md:
     "a cross-implementation check is the only thing that catches a contract
     mismatch". Running a tracker whose VOT-STb2022 numbers are PUBLISHED through
     this path is that check.
  2. OFFLINE SCREENING IS NOT QUOTABLE. `vot_ar_offline.py` applies VOT's failure
     rule to SINGLE-START runs; its resolution is ~0.02 in R and it has now
     inverted twice (the mask's sign on a new bank, then MOSSE_ETA). Trajectories
     written here go through the SAME ingest and the SAME toolkit analysis as a
     board sweep, so an arm scored this way is scored by the same arithmetic.

WHAT THIS IS NOT. It is not a board run and never substitutes for one: an arm
that reaches AIE_FLAGS cannot be screened here at all, and a host-only arm scored
here is still a different implementation (float, no shift budget, no int8) of the
same algorithm. It answers "is this algorithm class capable of X", not "does the
board do X".

THE CONTROLS ARE THE POINT, and they run in seconds:

  oracle       returns the groundtruth box. MUST score A/R/EAO ~ 1.0. If it does
               not, this harness has a frame-indexing, run-order or format bug and
               NOTHING else it produces means anything. Run it first, every time.
  oracle-lag1  returns the PREVIOUS frame's groundtruth -- a deliberate defect
               that must score measurably WORSE than oracle. A suite with no
               failing mutant is worth nothing until one has been shown.
  static       returns the init box forever. The floor: a tracker that does not
               track, scored by the same path.

@thesis sec:metodykaBadan | M-17 | host-side trackers under the board's multistart
protocol, scored by the toolkit, so a published tracker can validate the path
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Tracker backends. A backend is init(image, box) -> None and update(image) ->
# box, with box = (x, y, w, h) TOP-LEFT, the toolkit's convention and the one
# the trajectory file carries. Anything stateful lives in the instance: one
# instance is built per RUN, never reused across anchors.
# --------------------------------------------------------------------------
class Oracle:
    """The ceiling, and the harness's own self-test."""
    needs_groundtruth = True

    def __init__(self, lag=0):
        self.lag, self.hist = lag, []

    def init(self, image, box, gt=None):
        self.hist = [box]

    def update(self, image, gt=None):
        self.hist.append(gt if gt is not None else self.hist[-1])
        i = max(0, len(self.hist) - 1 - self.lag)
        return self.hist[i]


class Static:
    """The floor: never moves off the init box."""
    needs_groundtruth = False

    def init(self, image, box, gt=None):
        self.box = box

    def update(self, image, gt=None):
        return self.box


class OpenCV:
    """cv2's own trackers. CSRT is CSRDCF and KCF is KCF -- both are in VOT2022
    Table 12, which is what makes them the validation instrument.

    NOT byte-identical to the authors' VOT submissions, so treat the published
    row as a BAND to land in, never an equality to hit.
    """
    needs_groundtruth = False

    # THE TOOLKIT HANDS OUT RGB; cv2's TRACKERS EXPECT BGR. Both CSRT and KCF use
    # Color-Names features, so feeding RGB scrambles them. A real bug, and the
    # oracle control could never have found it -- the oracle never reads a pixel.
    #
    # BUT IT IS NOT WHAT AILS opencv-kcf, and this comment used to say it was.
    # Correcting the channel order is worth +0.0065 R and -0.0036 EAO on KCF: a
    # null. KCF's real gap to its published row is that cv2's TrackerKCF is not
    # the KCF submitted to VOT. Both figures and the refutation are in
    # results/reference_trackers.csv (claim R-12, evidence/harness_validation.md).
    # `opencv:kcf-rgb` keeps the broken arm as the negative control that settled
    # it.
    def __init__(self, kind, bgr=True):
        import cv2
        self.cv2, self.kind, self.bgr = cv2, kind, bgr
        factories = {
            "csrt": ("TrackerCSRT_create", "TrackerCSRT"),
            "kcf":  ("TrackerKCF_create", "TrackerKCF"),
            "mil":  ("TrackerMIL_create", "TrackerMIL"),
        }
        if kind not in factories:
            raise SystemExit(f"unknown opencv tracker '{kind}'")
        self.make = None
        for name in factories[kind]:
            f = getattr(cv2, name, None) or getattr(getattr(cv2, "legacy", None), name, None)
            if f is not None:
                self.make = (lambda f: (lambda: f.create() if hasattr(f, "create") else f()))(f)
                break
        if self.make is None:
            raise SystemExit(
                f"cv2 {cv2.__version__} has no {kind.upper()} tracker.\n"
                f"CSRT and KCF live in opencv-contrib-python, not opencv-python "
                f"(which vot-toolkit pulls in). Install the contrib build into "
                f"the venv -- it REPLACES opencv-python and is a superset, so "
                f"the toolkit keeps working:\n"
                f"    uv pip install opencv-contrib-python\n"
                f"Then re-run. `--tracker oracle` needs none of this and is what "
                f"validates the harness itself.")

    def _prep(self, image):
        return image[:, :, ::-1] if self.bgr else image

    def init(self, image, box, gt=None):
        image = self._prep(image)
        self.t = self.make()
        # cv2 5.0 takes an INTEGER Rect and rejects floats outright. The rounding
        # is the tracker's property, not this harness's: an opencv arm therefore
        # cannot reach the oracle's accuracy ceiling, and the sub-pixel init the
        # board gets is not available to it. Say so when reporting an A.
        x, y, w, h = (int(round(v)) for v in box)
        self.t.init(image, (x, y, max(1, w), max(1, h)))
        self.last = box

    def update(self, image, gt=None):
        ok, b = self.t.update(self._prep(image))
        if ok:
            self.last = tuple(float(v) for v in b)
        # else: hold. See below.
        # A tracker that reports failure still owes the protocol a box: the
        # multistart experiment scores every frame and reads a missing one as a
        # short trajectory, which vot_ingest.verify rejects outright. Holding the
        # last box is the same thing the board does on a gate veto.
        return self.last


def make_tracker(spec):
    if spec == "oracle":
        return Oracle(lag=0)
    if spec == "oracle-lag1":
        return Oracle(lag=1)
    if spec == "static":
        return Static()
    if spec.startswith("opencv:"):
        kind = spec.split(":", 1)[1]
        if kind.endswith("-rgb"):          # the colour-order mutant
            return OpenCV(kind[:-4], bgr=False)
        return OpenCV(kind, bgr=True)
    raise SystemExit(f"unknown tracker '{spec}'")


# --------------------------------------------------------------------------
def run_order(anchor, length_seq, backward):
    """Frame indices IN RUN ORDER. Mirrors vot_source.cpp's run_order():
    forward [anchor .. n-1], backward [anchor .. 0]. The board writes its
    trajectory in this order and the toolkit reads it in this order via
    FrameMapSequence, so a reversed backward run scores as a tracker that
    instantly fails rather than as a bug."""
    return list(range(anchor, length_seq)) if backward is False else \
        list(range(anchor, -1, -1))


def box_of(region):
    """A region -> (x, y, w, h) top-left. STb2022 groundtruth is Rectangle; the
    bounds() fallback covers a polygon or mask without silently producing a
    plausible-but-wrong box (vot_prepare.py's reduce_box made exactly that
    mistake once, on 62 sequences at a time)."""
    if region is None or region.is_empty():
        return None
    x0, y0, x1, y1 = region.bounds()
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def run_one(seq, anchor, backward, spec):
    """One anchored run. Returns (lines, times_ms)."""
    idx = run_order(anchor, len(seq), backward)
    tracker = make_tracker(spec)
    init_box = box_of(seq.groundtruth(idx[0]))
    if init_box is None:
        raise RuntimeError(f"anchor {anchor} has an empty groundtruth box")

    t0 = time.perf_counter()
    tracker.init(seq.frame(idx[0]).image(), init_box,
                 gt=init_box if getattr(tracker, "needs_groundtruth", False) else None)
    lines, times = ["1"], [(time.perf_counter() - t0) * 1e3]

    for j in idx[1:]:
        gt = box_of(seq.groundtruth(j)) if getattr(tracker, "needs_groundtruth", False) else None
        t0 = time.perf_counter()
        b = tracker.update(seq.frame(j).image(), gt=gt)
        times.append((time.perf_counter() - t0) * 1e3)
        lines.append("%.4f,%.4f,%.4f,%.4f" % tuple(b))

    want = (len(seq) - anchor) if backward is False else (anchor + 1)
    if len(lines) != want:
        raise RuntimeError(f"anchor {anchor}: wrote {len(lines)} regions, "
                           f"the job needs {want}")
    return lines, times


def plan_sequence(seqdir, name):
    """(name, anchor, backward, length) for every run of one sequence."""
    from vot.dataset import load_sequence
    from vot.experiment.multistart import find_anchors
    seq = load_sequence(str(Path(seqdir) / name))
    fwd, bwd = find_anchors(seq)
    n = len(seq)
    return ([(name, a, False, n - a) for a in fwd] +
            [(name, a, True, a + 1) for a in bwd])


def run_task(args):
    """ONE anchored run. The pool's unit of work is a RUN, not a sequence.

    Parallelising per SEQUENCE looks natural and is badly imbalanced: `girl`
    (1500 frames x 31 anchors) and `flamingo1` (1377 x 28) are 36% of the whole
    180,544-frame workload between them, so a per-sequence pool finishes 60
    sequences in three minutes and then runs two cores for another forty.
    Per-run tasks cost one extra load_sequence() each -- the images are lazy, so
    that is a small config read -- and flatten the tail.
    """
    seqdir, name, anchor, backward, want, spec, outdir = args
    from vot.dataset import load_sequence

    # ONE THREAD PER WORKER, and it is not optional. OpenCV's trackers are
    # internally multithreaded, so a pool of N processes becomes N x ncores
    # threads: measured load average 776 on 32 cores, far SLOWER than serial.
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

    stem = Path(outdir) / f"{name}_{anchor:08d}"
    traj = stem.with_suffix(".txt")
    # RESUME, and it doubles as an integrity check: a file of the wrong length is
    # rewritten rather than trusted. vot_ingest.verify() would catch a short
    # trajectory later, but only after the run that produced it is gone.
    if traj.exists() and sum(1 for _ in traj.open()) == want:
        return name, want, True

    seq = load_sequence(str(Path(seqdir) / name))
    lines, times = run_one(seq, anchor, backward, spec)
    traj.write_text("\n".join(lines) + "\n")
    (Path(outdir) / f"{name}_{anchor:08d}_time.value").write_text(
        "".join(f"{t:.3f}\n" for t in times))
    return name, want, False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", required=True,
                    help="oracle | oracle-lag1 | static | opencv:csrt | opencv:kcf | opencv:mil; append -rgb to an opencv kind for the colour-order mutant")
    ap.add_argument("--arm", default=None, help="output arm name (default: the tracker spec)")
    ap.add_argument("--out", type=Path, default=None,
                    help="results root; one subdirectory per arm (default $VOT_ROOT/results-offline)")
    ap.add_argument("--sequences", type=Path, default=None,
                    help="dataset dir (default $VOT_ROOT/workspace/sequences)")
    ap.add_argument("--seqs", default=None, help="comma-separated subset; default all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    # Pin the thread pools BEFORE numpy/cv2 are imported in any worker.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    root = os.environ.get("VOT_ROOT")
    seqdir = a.sequences or (Path(root) / "workspace" / "sequences" if root else None)
    if seqdir is None:
        raise SystemExit("set VOT_ROOT or pass --sequences")
    outroot = a.out or (Path(root) / "results-offline" if root else None)
    if outroot is None:
        raise SystemExit("set VOT_ROOT or pass --out")

    arm = a.arm or a.tracker.replace(":", "-")
    outdir = Path(outroot) / arm
    outdir.mkdir(parents=True, exist_ok=True)

    names = [s.strip() for s in a.seqs.split(",")] if a.seqs else \
        sorted(p.name for p in Path(seqdir).iterdir()
               if (p / "sequence").exists())
    # Fail before the work, not during it: an unknown sequence name is a typo and
    # a partial arm is what vot_ingest silently excludes later.
    for n in names:
        if not (Path(seqdir) / n / "sequence").exists():
            raise SystemExit(f"no sequence '{n}' under {seqdir}")

    print(f"tracker   {a.tracker}\narm       {arm}\nout       {outdir}")
    print(f"sequences {len(names)} from {seqdir}\njobs      {a.jobs}\n")

    make_tracker(a.tracker)          # fail fast on a missing cv2 backend

    tasks = []
    for n in names:
        for name, anchor, backward, want in plan_sequence(seqdir, n):
            tasks.append((str(seqdir), name, anchor, backward, want,
                          a.tracker, str(outdir)))
    # LONGEST FIRST. With a fixed pool the makespan is set by the longest task,
    # so a 1500-frame run must not be picked up last.
    tasks.sort(key=lambda t: -t[4])
    total_frames = sum(t[4] for t in tasks)
    print(f"  {len(tasks)} runs, {total_frames} frames\n")

    t0 = time.perf_counter()
    runs = skipped = 0
    done_frames = 0

    def note(res):
        nonlocal runs, skipped, done_frames
        _, want, was_skipped = res
        runs += 1; done_frames += want
        skipped += int(was_skipped)
        if runs % 25 == 0 or runs == len(tasks):
            el = time.perf_counter() - t0
            eta = el * (total_frames - done_frames) / max(done_frames, 1)
            print(f"  {runs:3d}/{len(tasks)} runs  {done_frames:6d}/{total_frames} frames"
                  f"  {el:5.0f}s elapsed  ~{eta:4.0f}s left")

    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for res in pool.imap_unordered(run_task, tasks, chunksize=1):
                note(res)
    else:
        for t in tasks:
            note(run_task(t))

    dt = time.perf_counter() - t0
    print(f"\n  {runs} trajectories over {len(names)} sequences in {dt:.1f} s"
          + (f" ({skipped} already present, skipped)" if skipped else ""))
    print(f"\n  score it:\n    env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python "
          f"scripts/vot_ingest.py \\\n      --results {outroot} --arms {arm} "
          f"--out $VOT_ROOT/analysis/offline_{arm}")
    if a.tracker == "oracle":
        print("\n  ORACLE: A, R and EAO must all come back at ~1.0. Anything else "
              "is a bug\n  in THIS harness, not a tracking result.")


if __name__ == "__main__":
    main()
