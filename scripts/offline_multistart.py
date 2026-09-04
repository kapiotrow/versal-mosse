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

# Danilowicz & Kryjak's checkout. Gitignored and PINNED -- see the DeepDCF
# block below for the commit and why it is not vendored into this repo.
DEEP_ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent / "external" / "deep_mosse")


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


# --------------------------------------------------------------------------
# DANILOWICZ & KRYJAK'S deepDCF -- THEIR PUBLISHED CODE, on THIS benchmark.
#
# WHY THIS EXISTS. Their tracker is the architecturally nearest neighbour in the
# literature (conv1 features into a multichannel MOSSE, quantised, on an
# embedded SoC-FPGA, at this project's own 128x128 ROI -> 64x64 filter
# geometry), but their numbers are VOT2015 supervised with an inverted R, a
# [108,371] EAO window and polygon ground truth -- M-17, and the window term
# alone is 1.39x this project's whole arm ladder. NONE of that has to be
# reconciled if their tracker is run HERE instead: same 62 sequences, same 419
# anchored runs, same vot_ingest -> `vot analysis` path that R-12 validated
# against published CSRDCF. Bring their tracker to our benchmark, never the
# reverse -- the reverse needs a VOT2015 conversion AND the supervised/reset
# protocol this harness does not implement, i.e. a second unvalidated path.
#
# THE CODE IS UNMODIFIED. external/deep_mosse is a pinned clone of
# github.com/mdanilow/MOSSE_fpga @ ee0f93ab (branch deep_features, MIT), and
# nothing in it is patched: their float path builds torchvision vgg11
# features[:3] (conv 3x3 -> ReLU -> 2x2 maxpool) and runs as published. Anything
# that ever DOES need patching belongs in scripts/patches/ as a reviewable diff,
# never as an edit in place -- the whole value of this comparison is that the
# algorithm is theirs.
#
# WHAT CANNOT BE REPRODUCED, and it must be said in the write-up: their 4-bit
# QUANTIZED arm -- the one that actually ran on the ZCU104 -- needs
# `savegame_0_15000.pth.tar`, referenced by an absolute path on the author's
# machine and NOT in the repo. So these arms are their SOFTWARE MODEL. Their
# own section 6 licenses that stand-in: the hardware "yielded the same results
# on sequences from the VOT2015 set" as the software model.
#
# THE SIGMA CONFOUND, found while reading _get_gauss_response and worth as much
# as the comparison itself. Their response is exp(-r^2 / (2*sigma)) with r in
# FEATURE-MAP BINS and the map at ROI_SIZE/stride, while `sigma` is a single
# global config value with no geometry term. So their Table 1 row that reports
# +0.024 EAO for the 224/112 geometry over 128/64 moves the map AND the mainlobe
# width together -- which is precisely the confound R-11 caught in this
# project's own 64x64 arm, where the gain turned out to be the width the arm
# carried by accident and the resolution term was a null (R-14). Their ordering
# is the last external support for a 128x128 Layer-1 arm, so the `hw32w` preset
# below exists to separate the two terms IN THEIR TRACKER: same map as `hw32`,
# sigma rescaled to hold the width. sigma is a VARIANCE in this parameterisation
# (std = sqrt(sigma)), so holding std/target across a 2x map change scales sigma
# by 4: 7 -> 1.75.
DEEP_PRESETS = {
    # their configs/config.json verbatim -- Table 1's best software row
    # (EAO 0.207, results/embedded_baselines.csv)
    "best":  dict(ROI_SIZE=224, num_scales=5, channels=32, sigma=7),
    # the geometry that went to the ZCU104 (Table 1's hardware row, EAO 0.183,
    # results/embedded_baselines.csv)
    "hw":    dict(ROI_SIZE=128, num_scales=3, channels=8,  sigma=7),
    # ...at THIS project's channel count, so only the algorithm differs
    "hw32":  dict(ROI_SIZE=128, num_scales=3, channels=32, sigma=7),
    # ...and at matched MAINLOBE WIDTH. The R-11 control on their own tracker.
    "hw32w": dict(ROI_SIZE=128, num_scales=3, channels=32, sigma=1.75),
}

_DEEP = {}


def _deep_setup(root):
    """Per-WORKER setup, cached: import their package, memoize the backbone.

    DeepMosse.__init__ calls get_VGG_backbone() per instance and this harness
    builds one tracker per RUN (419 of them), so an unmemoized backbone is 419
    vgg11 constructions. The memo wraps THEIR function from outside rather than
    editing it -- see the note above about keeping their code unmodified.
    """
    key = str(root)
    if key not in _DEEP:
        import sys, json
        rp = Path(root)
        if not (rp / "deep_mosse.py").exists():
            raise SystemExit(
                f"no deepDCF checkout at {rp}.\n"
                f"  /usr/bin/git clone https://github.com/mdanilow/MOSSE_fpga "
                f"{rp}\n"
                f"  cd {rp} && /usr/bin/git checkout ee0f93ab183d8b2f712de039f1ec6d4776847fb2\n"
                f"NOTE /usr/bin/git explicitly: Vivado puts its own git 2.50.0 "
                f"first on PATH and that build has no https remote helper.\n"
                f"Deps: uv pip install easydict imutils fxpmath brevitas")
        # THEIR ROOT GOES ON sys.path ONLY FOR THE DURATION OF THE IMPORT, and
        # this is not tidiness. Their repo ships a top-level `vot.py` -- the VOT
        # toolkit's own tracker-integration stub -- which SHADOWS the installed
        # `vot` package the moment their directory is on the path. The symptom
        # is `ModuleNotFoundError: No module named 'vot.dataset'; 'vot' is not a
        # package` from plan_sequence, i.e. this harness losing the toolkit
        # underneath itself. Their `utils` and `finnmodels` are equally generic.
        # Import under priority, then restore, so only sys.modules keeps them.
        saved = list(sys.path)
        sys.path.insert(0, str(rp))
        try:
            import deep_mosse as DM
            import utils as DU
        finally:
            sys.path[:] = saved
        for mod in (DM, DU):
            got = Path(mod.__file__).resolve().parent
            if got != rp.resolve():
                raise SystemExit(
                    f"{mod.__name__} resolved to {got}, not the pinned "
                    f"checkout {rp.resolve()} -- a name collision, not a "
                    f"tracking result. Rename the shadowing module.")
        import importlib
        if importlib.import_module("vot").__file__ and \
                Path(importlib.import_module("vot").__file__).resolve().parent == rp.resolve():
            raise SystemExit(
                "the installed `vot` toolkit is shadowed by their vot.py; "
                "the restore above failed and no score from this run is valid.")
        memo = {}
        orig = DU.get_VGG_backbone

        def cached(*a, **k):
            if "b" not in memo:
                memo["b"] = orig(*a, **k)
            return memo["b"]

        DM.get_VGG_backbone = cached
        base = json.loads((rp / "configs" / "config.json").read_text())
        _DEEP[key] = (DM, base)
    return _DEEP[key]


class DeepDCF:
    """Danilowicz & Kryjak's DeepMosse behind this harness's init/update API.

    Their tracker takes BGR (it is driven by cv2.imread in their own
    vot_integration.py) and the toolkit hands out RGB, so the flip here is the
    same one the OpenCV arms need -- and `opencv-kcf-rgb` is the standing
    negative control proving the flip is not free to get wrong.
    """
    def __init__(self, preset, root):
        if preset not in DEEP_PRESETS:
            raise SystemExit(f"unknown deepdcf preset '{preset}'; "
                             f"expected one of {sorted(DEEP_PRESETS)}")
        self.preset, self.root = preset, root
        _deep_setup(root)          # fail fast, in the parent

    def _prep(self, image):
        # ascontiguousarray, not a bare ::-1 view: their crop path goes straight
        # into cv2.copyMakeBorder/cv2.resize, which want a real buffer.
        import numpy as np
        return np.ascontiguousarray(image[:, :, ::-1])

    def init(self, image, box, gt=None):
        DM, base = _deep_setup(self.root)
        cfg = dict(base)
        cfg.update(DEEP_PRESETS[self.preset])
        # The 4-bit checkpoint is not in the repo (see above), so every arm here
        # is the float software model. Pinned explicitly rather than inherited
        # from their config file, which is what a reader will check first.
        cfg["deep"], cfg["quantized"] = True, False
        self.t = DM.DeepMosse(self._prep(image), [float(v) for v in box], cfg)
        self.last = tuple(float(v) for v in box)

    def update(self, image, gt=None):
        b = self.t.track(self._prep(image))
        # Their track() returns ints and holds the last box once target_lost is
        # set. Both are the TRACKER's properties, not this harness's: an integer
        # box cannot reach the oracle accuracy ceiling, exactly as for the cv2
        # arms, and a hold is what the board does on a gate veto. Say so when
        # reporting an A.
        if b is not None and len(b) == 4 and b[2] > 0 and b[3] > 0:
            self.last = tuple(float(v) for v in b)
        return self.last


def make_tracker(spec):
    if spec == "oracle":
        return Oracle(lag=0)
    if spec == "oracle-lag1":
        return Oracle(lag=1)
    if spec == "static":
        return Static()
    if spec.startswith("mosse:"):
        return None          # batch path; run_task dispatches, see run_mosse
    if spec.startswith("deepdcf:"):
        rest = spec.split(":", 1)[1]
        preset, _, root = rest.partition("@")
        return DeepDCF(preset, root or DEEP_ROOT_DEFAULT)
    if spec.startswith("opencv:"):
        kind = spec.split(":", 1)[1]
        if kind.endswith("-rgb"):          # the colour-order mutant
            return OpenCV(kind[:-4], bgr=False)
        return OpenCV(kind, bgr=True)
    raise SystemExit(f"unknown tracker '{spec}'")



# --------------------------------------------------------------------------
# THIS PROJECT'S OWN TRACKER, in float -- the twin.
#
# rgb_vs_gray_loop.run_arm is a WHOLE-RUN function, not an incremental
# init/update tracker, so it does not fit the class interface above and is not
# forced into one: refactoring it into a stepping object would touch every
# screening arm that file carries (the mask, eta, pooling, warp and bank
# screens) and invite exactly the drift `resolve_arm` was extracted to prevent.
# Instead the pool's task path calls it directly, once per anchored run.
#
# WHAT THIS TWIN IS FOR. It is the same algorithm as the board -- same Stage A,
# same bank via l1_banks, same shifted training target, same PSR gate, same
# 128x128 crop -> 64x64 map -- in float64 downstream of the features. The gap
# between it and the board arm prices the embedded implementation; the gap
# between it and CSRDCF (R-12) prices the algorithm. Neither question has a row
# yet without it.
#
# WHAT IT DELIBERATELY IS NOT: it has NO DSST SCALE FILTER. Box size is held at
# its init value (SCALE_N=1 equivalent) or taken from groundtruth
# (--oracle-scale), and the two BRACKET the question. That is a priced decision,
# not an oversight: scale_oracle_bound.py measures a PERFECT scale filter at
# +0.0023 R on the shipping arm, and the board's own filter is frozen on ~90% of
# frames with detector gain -0.003. State the assumption when reporting a number
# from this twin.
_MOSSE = {}


def _mosse_setup(arm):
    """Per-WORKER setup, cached: torch import, folded BN weights, PCA bank.

    ~2.6 s, so it must not run per task. Keyed by arm because one invocation
    scores one arm, but the dict costs nothing and makes that explicit.
    """
    if arm not in _MOSSE:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rgb_vs_gray_loop as RL
        import rgb_vs_gray_holdout as RH
        w_rgb, b_fold = RH.folded_weights()
        w_gray = (w_rgb * RH.LUM[None, :, None, None]).sum(axis=1, keepdims=True)
        W = {'gray': RH.quantize(w_gray, b_fold),
             'rgb': RH.quantize(w_rgb, b_fold),
             'rgb-lum': RH.quantize(w_rgb, b_fold),
             'gray-float': RH.quantize(w_gray, b_fold),
             'rgb-float': RH.quantize(w_rgb, b_fold)}
        floatw = {'gray-float': (w_gray, b_fold), 'rgb-float': (w_rgb, b_fold)}
        # ONE dispatch, shared with the bench's main(). See resolve_arm.
        res = RL.resolve_arm(arm, W, w_rgb, w_gray, b_fold, quiet=True)
        _MOSSE[arm] = (RL, RH, res, floatw)
    return _MOSSE[arm]


def run_mosse(seqdir, name, frames, spec, opts):
    """One anchored run of the float twin. Returns boxes as (x, y, w, h)."""
    arm = spec.split(":", 1)[1]
    RL, RH, res, floatw = _mosse_setup(arm)

    # set_sequence()/load_gt() drive MODULE GLOBALS in rgb_vs_gray_holdout, so
    # they must be re-pointed for every task -- a worker handles many sequences.
    #
    # IT RESOLVES THE PATH ITSELF, from $VOT_ROOT, and does NOT know about this
    # harness's --sequences. If those two disagree the twin would track a
    # different dataset than the anchors were derived from and still produce a
    # full, plausible, wrong trajectory -- so assert they are the same directory
    # rather than trusting it. main() checks this once up front too; this is the
    # per-task backstop, because the failure is silent.
    RH.set_sequence(name)
    resolved = Path(RH.SEQ).resolve().parent
    wanted = (Path(seqdir) / name).resolve()
    if resolved != wanted:
        raise SystemExit(
            f"{name}: rgb_vs_gray_holdout resolved {resolved} but this run's "
            f"dataset is {wanted}. set_sequence() reads $VOT_ROOT; either "
            f"export VOT_ROOT to match --sequences, or drop --sequences.")
    gt = RH.load_gt()

    rec = RL.run_arm(res['stem'], *res['bank'], gt, len(gt),
                     opts.get('oracle_scale', False), False,
                     frames=frames,
                     float_conv=floatw.get(res['base']),
                     sigma=opts.get('sigma') or RL.SIGMA,
                     eta=opts.get('eta', RL.ETA),
                     psr_min=opts.get('psr_min', RL.PSR_GATE_MIN),
                     eps_rel=opts.get('eps_rel', RL.EPS_REL),
                     n_warps=res['n_warps'], mask_plateau=res['mask_plateau'],
                     rect=res['rect'], chrel_gamma=res['chrel_gamma'],
                     stride=res['stride'], blur_n=res['blur_n'],
                     ceta_lo=res['ceta_lo'], pool_n=res['pool_n'],
                     pool_mode_ov=res['pool_mode_ov'],
                     # THE BOARD'S FILTER QUANTIZATION (`-hq`, `-hq<pct>`).
                     # Forwarded explicitly: an arm whose suffix parsed but
                     # never reached run_arm would score as the plain twin and
                     # look like a null, which is the failure this whole file
                     # exists to make impossible.
                     quant_h=res['quant_h'], smem=res['smem'],
                     padding=opts.get('padding'))
    # run_arm works in CENTRE convention (row, col, h, w); the trajectory file
    # is TOP-LEFT (x, y, w, h). Same conversion as vot_source.cpp's as_text().
    return [(c - w / 2.0, r - h / 2.0, w, h) for r, c, h, w in rec['box']]


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
    seqdir, name, anchor, backward, want, spec, outdir, opts = args
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

    if spec.startswith("mosse:"):
        # BATCH PATH: run_arm is a whole-run function. It reads its own frames
        # and groundtruth, so the toolkit Sequence is needed only for its length.
        seq = load_sequence(str(Path(seqdir) / name))
        idx = [i + 1 for i in run_order(anchor, len(seq), backward)]   # 1-based
        t0 = time.perf_counter()
        boxes = run_mosse(seqdir, name, idx, spec, opts)
        dt = (time.perf_counter() - t0) * 1e3 / max(len(boxes), 1)
        lines = ["1"] + ["%.4f,%.4f,%.4f,%.4f" % b for b in boxes[1:]]
        times = [dt] * len(lines)
    else:
        seq = load_sequence(str(Path(seqdir) / name))
        lines, times = run_one(seq, anchor, backward, spec)
    if len(lines) != want:
        raise RuntimeError(f"{name} anchor {anchor}: {len(lines)} regions, "
                           f"the job needs {want}")
    traj.write_text("\n".join(lines) + "\n")
    (Path(outdir) / f"{name}_{anchor:08d}_time.value").write_text(
        "".join(f"{t:.3f}\n" for t in times))
    return name, want, False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", required=True,
                    help="oracle | oracle-lag1 | static | opencv:csrt | opencv:kcf | "
                         "opencv:mil (append -rgb for the colour-order mutant) | "
                         "mosse:<arm>, e.g. mosse:rgb-l1relu | "
                         "deepdcf:<preset> (best|hw|hw32|hw32w), optionally "
                         "deepdcf:<preset>@<path-to-checkout>")
    ap.add_argument("--arm", default=None, help="output arm name (default: the tracker spec)")
    ap.add_argument("--out", type=Path, default=None,
                    help="results root; one subdirectory per arm (default $VOT_ROOT/results-offline)")
    ap.add_argument("--sequences", type=Path, default=None,
                    help="dataset dir (default $VOT_ROOT/workspace/sequences)")
    ap.add_argument("--seqs", default=None, help="comma-separated subset; default all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    # --- the float twin's knobs. Defaults are the SHIPPING arm's, not the
    # bench's: rgb_vs_gray_loop defaults to eta 0.125 / gate 7.0, which are the
    # values from before eta05 and gate5 shipped. A twin left on those defaults
    # would be a twin of a config the board stopped running on 2026-08-27.
    ap.add_argument("--eta", type=float, default=0.05, help="MOSSE_ETA")
    ap.add_argument("--psr-min", type=float, default=5.0, help="PSR_GATE_MIN")
    ap.add_argument("--sigma", type=float, default=None,
                    help="MOSSE_SIGMA in BINS; default 2.0 = sigma/target 1/16")
    ap.add_argument("--padding", type=float, default=None,
                    help="TARGET_PADDING. COUPLED TO --sigma: the target spans "
                         "map/padding bins, so sigma/target = sigma*padding/map "
                         "and 1/16 is the measured optimum (R-11). Padding 3.0 "
                         "at the shipping sigma 2.0 is 1/10.7, not 1/16")
    ap.add_argument("--oracle-scale", action="store_true",
                    help="box size from groundtruth. The twin has NO DSST scale "
                         "filter; this and the default BRACKET what scale is worth")
    a = ap.parse_args()
    opts = dict(eta=a.eta, psr_min=a.psr_min, sigma=a.sigma,
                padding=a.padding, oracle_scale=a.oracle_scale)

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

    if a.tracker.startswith("mosse:"):
        # The twin resolves its own dataset from $VOT_ROOT. Fail here, once,
        # rather than 419 times inside the workers.
        vr = os.environ.get("VOT_ROOT")
        want = Path(seqdir).resolve()
        got = (Path(vr) / "workspace" / "sequences").resolve() if vr else None
        if got != want:
            raise SystemExit(
                f"the float twin resolves sequences from $VOT_ROOT "
                f"({got}), but --sequences is {want}. Export VOT_ROOT to match.")
        print(f"  twin knobs: eta {a.eta}, gate {a.psr_min}, "
              f"sigma {a.sigma or 'default 2.0'}, "
              f"scale {'ORACLE' if a.oracle_scale else 'HELD FIXED (no DSST filter)'}")

    if a.tracker.startswith("deepdcf:"):
        preset = a.tracker.split(":", 1)[1].partition("@")[0]
        cfg = DEEP_PRESETS.get(preset, {})
        print(f"  deepDCF preset {preset}: " +
              ", ".join(f"{k} {v}" for k, v in cfg.items()) +
              "\n  FLOAT software model (the 4-bit checkpoint is not in their "
              "repo); their code UNMODIFIED at the pinned commit.")
        if a.jobs > 4:
            print(f"  NOTE --jobs {a.jobs}: the backbone runs on the GPU and "
                  f"every worker opens its own CUDA context. Time a single "
                  f"sequence before scaling this up.")

    print(f"tracker   {a.tracker}\narm       {arm}\nout       {outdir}")
    print(f"sequences {len(names)} from {seqdir}\njobs      {a.jobs}\n")

    make_tracker(a.tracker)          # fail fast on a missing cv2 backend

    tasks = []
    for n in names:
        for name, anchor, backward, want in plan_sequence(seqdir, n):
            tasks.append((str(seqdir), name, anchor, backward, want,
                          a.tracker, str(outdir), opts))
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
        # SPAWN, not the default fork, whenever a worker will touch CUDA:
        # "Cannot re-initialize CUDA in forked subprocess". Spawn re-imports the
        # module per worker (a few seconds each, once) and the pool is
        # long-lived, so the cost is amortised over hundreds of runs. Every
        # other backend keeps fork, which is cheaper to start.
        ctx = mp.get_context("spawn") if a.tracker.startswith("deepdcf:") else mp
        with ctx.Pool(a.jobs) as pool:
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
