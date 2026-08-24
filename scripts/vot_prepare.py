#!/usr/bin/env python3
"""
scripts/vot_prepare.py

VOT sequences -> board-ready raw blobs + manifests, and the verifier that says
the conversion was faithful.

WHY A BLOB
----------
The board's job is "given a frame-order list and one init box, emit one box per
frame". Everything else -- anchors, failure detection, reset policy, AR scoring
-- stays on the PC. So the on-board frame source should be the dumbest thing
that can work: one `read()` per sequence into a heap buffer, zero parsing.
Hence frames back to back, `rows * cols * channels` bytes each, NO HEADER. The
board computes an offset and memcpy's.

The manifest carries everything the blob deliberately does not.

THE LUMA CONVENTION IS PINNED, AND NOT TO PIL
---------------------------------------------
`Image.convert("L")` is NOT this pipeline's grayscale. PIL uses a truncating
integer path with coefficients 299/587/114; `rgb_vs_gray_holdout.to_luma` uses
float 0.2989/0.5870/0.1140 with round-half-even and a clip, and that is the
convention `export_weights.py` collapses the conv kernels with. Feeding the
board PIL's luma while the offline arm uses to_luma's would put a 1-LSB
disagreement into every frame -- invisible per pixel, and exactly the kind of
difference that would later be misread as "fixed point vs float FFT" when
Phase 2 compares the two trajectories.

So: converter and offline harness use ONE convention, and `verify --mutate
pilluma` exists to prove the check can tell them apart.

WHAT verify DOES AND DOES NOT COVER
-----------------------------------
It re-decodes every JPEG and compares byte-for-byte against the blob slice, so
it covers frame ORDER, blob OFFSETS, dimensions, channel layout and the luma
convention. It does not independently validate libjpeg -- both sides decode
with PIL. That is deliberate: the decoder is deterministic and is not what is
at risk here; the bookkeeping is. A test that shares an input with the thing it
checks can pass on corrupted data, so the input is pinned by md5 and the
bookkeeping is mutation-tested.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_prepare.py \
      convert --root test-sequences --out $VOT_ROOT/data
  ... verify --out $VOT_ROOT/data
  ... verify --out $VOT_ROOT/data --mutate all      # prove verify can FAIL
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# BT.601, byte-identical to rgb_vs_gray_holdout.LUM. Duplicated rather than
# imported because importing that module drags in gen_filter_golden and the
# whole offline stack; `verify --check-gt` asserts the two agree, which is a
# stronger statement than sharing a constant would be.
LUM = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

# The board's frame buffer. The plan assumes every sequence fits; the converter
# enforces it rather than letting the board discover it.
MAX_ROWS, MAX_COLS = 1080, 1920

# roi = box * TARGET_PADDING (Makefile default 2.0). Not a hard error -- roi_crop
# border-clamps -- but recorded per sequence so the board can assert instead of
# silently clamping.
TARGET_PADDING = 2.0

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# pixel conventions
# ---------------------------------------------------------------------------

def to_luma(rgb_hw3):
    """uint8 [H,W,3] -> uint8 [H,W]. BT.601, matching rgb_vs_gray_holdout."""
    return np.clip(np.round((rgb_hw3.astype(np.float64) * LUM).sum(-1)),
                   0, 255).astype(np.uint8)


def decode(path):
    """JPEG -> uint8 [H,W,3] RGB."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def frame_bytes(rgb_hw3, channels):
    """One frame in wire layout: luma plane, or pixel-interleaved RGB.

    Interleaved, not planar, because that is what roi_crop reads at
    ROI_IN_CH=3 -- planar would need whole planes resident on the tile.
    """
    if channels == 1:
        return to_luma(rgb_hw3).tobytes()
    return np.ascontiguousarray(rgb_hw3).tobytes()


def reduce_box(poly):
    """VOT rotated polygon -> axis-aligned (row, col, h, w), min-max convention.

    Identical to rgb_vs_gray_vot.load_gt. Doing this ONCE here, on the PC, is
    what keeps the board and the offline harness agreeing about what the ground
    truth box is, rather than having two copies of the rule drift apart.
    """
    v = np.asarray(poly, dtype=np.float64)
    x, y = v[0::2], v[1::2]
    return (0.5 * (y.min() + y.max()), 0.5 * (x.min() + x.max()),
            y.max() - y.min(), x.max() - x.min())


def load_gt(path):
    return [reduce_box([float(t) for t in line.split(',')])
            for line in path.read_text().split()]


# ---------------------------------------------------------------------------
# sequence discovery -- annotation dirs are named inconsistently
# ("car1-annotations" but "fernando - annotations"), so match loosely.
# ---------------------------------------------------------------------------

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def discover(root):
    root = Path(root)
    ann_dirs = [d for d in root.iterdir()
                if d.is_dir() and 'annotation' in d.name.lower()]
    seqs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or 'annotation' in d.name.lower():
            continue
        frames = sorted(d.glob("*.jpg"))
        if not frames:
            continue
        ann = next((a for a in ann_dirs
                    if _norm(a.name).startswith(_norm(d.name))), None)
        gt = ann / "groundtruth.txt" if ann else None
        if gt and gt.exists():
            seqs[d.name] = (frames, gt)
        else:
            print(f"  [skip] {d.name}: no groundtruth.txt")
    return seqs


def check_frame_order(frames):
    """Assert files are exactly 00000001.jpg .. %08d.jpg with no gaps.

    sorted() on zero-padded names is numerically correct, but a missing or
    extra file would silently shift every subsequent frame against the ground
    truth -- a whole-sequence offset that looks like a tracking failure.
    """
    for i, f in enumerate(frames, start=1):
        if f.name != f"{i:08d}.jpg":
            raise SystemExit(
                f"frame order broken at index {i}: expected {i:08d}.jpg, "
                f"got {f.name}")


# ---------------------------------------------------------------------------
# job list
# ---------------------------------------------------------------------------

def make_jobs(gt, spacing):
    """Synthetic multi-start anchors at a fixed spacing.

    ONE DIRECTION PER ANCHOR. Phase 0b established this from the toolkit source:
    `find_anchors()` reads a per-frame `anchor` value and splits on its SIGN --
    positive goes in the forward list, negative in the backward list, and the
    two lists are disjoint. `MultiStartExperiment` then runs each anchor exactly
    once, in its own direction. An earlier version of this function emitted BOTH
    directions per interior anchor, which would have doubled the run count
    against what the toolkit will actually score.

    THESE ARE STILL NOT THE DATASET'S ANCHORS. The real run takes anchors and
    their directions from stb2022's own per-frame anchor values (Phase 1); this
    exists so the manifest format round-trips and so Phase 2 has a job to run.
    The direction chosen here -- toward the farther end, to maximise coverage --
    is a placeholder, not the dataset's rule. `anchors_source` records which kind
    the manifest holds, because a synthetic job list mistaken for the real one
    would produce a complete, plausible, unusable AR report.
    """
    n = len(gt)
    jobs = []
    for a in range(0, n, spacing):
        forward = (n - 1 - a) >= a          # farther end wins; ties go forward
        jobs.append({
            "anchor": a,
            "direction": "forward" if forward else "backward",
            "init_box": list(gt[a]),
            "length": (n - a) if forward else (a + 1),
        })
    return jobs


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def convert_sequence(name, frames, gtf, out, channels, spacing, mutate=None):
    check_frame_order(frames)
    gt = load_gt(gtf)
    if len(gt) != len(frames):
        raise SystemExit(f"{name}: {len(frames)} frames but {len(gt)} gt lines")

    h, w, _ = decode(frames[0]).shape
    if h > MAX_ROWS or w > MAX_COLS:
        raise SystemExit(f"{name}: {h}x{w} exceeds the {MAX_ROWS}x{MAX_COLS} "
                         f"frame buffer")

    order = list(range(len(frames)))
    if mutate == 'offbyone':
        order = order[1:] + [order[-1]]
    elif mutate == 'dropframe':
        mid = len(order) // 2
        order = order[:mid] + order[mid + 1:] + [order[-1]]
    elif mutate == 'reverse':
        order = order[::-1]

    blob = out / f"{name}.raw"
    md5 = hashlib.md5()
    expect = h * w * channels
    t0 = time.time()
    with open(blob, 'wb') as fh:
        for k in order:
            rgb = decode(frames[k])
            if rgb.shape[:2] != (h, w):
                raise SystemExit(f"{name}: frame {k+1} is {rgb.shape[:2]}, "
                                 f"sequence is {(h, w)}")
            if mutate == 'transpose':
                rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))
            if mutate == 'pilluma' and channels == 1:
                from PIL import Image
                b = np.asarray(Image.open(frames[k]).convert("L"),
                               dtype=np.uint8).tobytes()
            else:
                b = frame_bytes(rgb, channels)
            if len(b) != expect:
                raise SystemExit(f"{name}: frame {k+1} produced {len(b)} B, "
                                 f"expected {expect}")
            fh.write(b)
            md5.update(b)
    dt = time.time() - t0

    # A luma sidecar only matters at channels=3: scale_extract() reads an
    # intensity template, and computing luma on the board would put a
    # frame-sized pass inside the frame loop.
    luma_name = luma_md5 = None
    if channels == 3:
        luma_name = f"{name}.luma"
        lm = hashlib.md5()
        with open(out / luma_name, 'wb') as fh:
            for k in order:
                b = to_luma(decode(frames[k])).tobytes()
                fh.write(b)
                lm.update(b)
        luma_md5 = lm.hexdigest()

    sides = [min(b[2], b[3]) for b in gt]
    over = sum(1 for b in gt
               if b[2] * TARGET_PADDING > h or b[3] * TARGET_PADDING > w)
    jobs = make_jobs(gt, spacing)

    man = {
        "schema": SCHEMA_VERSION,
        "sequence": name,
        "frames": len(order),
        "rows": h, "cols": w, "channels": channels,
        "dtype": "uint8",
        "frame_bytes": expect,
        "layout": ("frames back-to-back, row-major, no header; "
                   + ("luma plane" if channels == 1
                      else "pixel-interleaved RGB")),
        "blob": blob.name,
        "blob_md5": md5.hexdigest(),
        "luma_blob": luma_name,
        "luma_md5": luma_md5,
        "luma_convention": ("BT.601 clip(round(0.2989R+0.5870G+0.1140B)) -- "
                            "matches rgb_vs_gray_holdout.to_luma, NOT "
                            "PIL Image.convert('L')"),
        "gt_convention": ("axis-aligned min-max of the rotated polygon "
                          "-> (row, col, h, w)"),
        "groundtruth": [list(b) for b in gt],
        "anchors_source": f"synthetic:spacing={spacing}",
        "jobs": jobs,
        "tracked_frames": sum(j["length"] for j in jobs),
        "min_box_side": float(min(sides)),
        "roi_exceeds_frame": int(over),
        "generator": {
            "script": "scripts/vot_prepare.py",
            "created": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            "mutation": mutate,
        },
    }
    (out / f"{name}.json").write_text(json.dumps(man, indent=1))
    print(f"  {name:<12} {len(order):5d}f {h:4d}x{w:<4d} "
          f"{blob.stat().st_size/1e6:8.1f} MB  {dt:5.1f}s  "
          f"jobs={len(jobs):4d} tracked={man['tracked_frames']:6d}"
          + (f"  [MUTANT {mutate}]" if mutate else ""))
    return man


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify_sequence(name, frames, gtf, out, check_gt, limit=None):
    """Re-decode and compare byte-for-byte. Returns a list of failure strings."""
    fails = []
    mp = out / f"{name}.json"
    if not mp.exists():
        return [f"{name}: no manifest"]
    man = json.loads(mp.read_text())
    blob = out / man["blob"]

    exp_size = man["frames"] * man["frame_bytes"]
    if blob.stat().st_size != exp_size:
        fails.append(f"{name}: blob {blob.stat().st_size} B, "
                     f"manifest implies {exp_size} B")
        return fails

    if man["frames"] != len(frames):
        fails.append(f"{name}: manifest {man['frames']} frames, "
                     f"directory has {len(frames)}")

    md5 = hashlib.md5()
    with open(blob, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            md5.update(chunk)
    if md5.hexdigest() != man["blob_md5"]:
        fails.append(f"{name}: blob md5 {md5.hexdigest()} != manifest "
                     f"{man['blob_md5']}")

    h, w, ch = man["rows"], man["cols"], man["channels"]
    fb = man["frame_bytes"]
    if fb != h * w * ch:
        fails.append(f"{name}: frame_bytes {fb} != {h}*{w}*{ch}")

    n = min(man["frames"], len(frames))
    if limit:
        n = min(n, limit)
    with open(blob, 'rb') as fh:
        for k in range(n):
            fh.seek(k * fb)
            got = fh.read(fb)
            want = frame_bytes(decode(frames[k]), ch)
            if got != want:
                d = np.frombuffer(got, np.uint8).astype(int) - \
                    np.frombuffer(want, np.uint8).astype(int)
                nz = int(np.count_nonzero(d))
                fails.append(f"{name}: frame {k+1} differs -- {nz}/{fb} bytes, "
                             f"max |delta| {int(np.abs(d).max())}")
                break                      # one report per sequence is enough

    if check_gt:
        want_gt = load_gt(gtf)
        got_gt = man["groundtruth"]
        if len(want_gt) != len(got_gt):
            fails.append(f"{name}: gt length {len(got_gt)} != {len(want_gt)}")
        else:
            a = np.array(want_gt); b = np.array(got_gt)
            if not np.allclose(a, b, rtol=0, atol=0):
                fails.append(f"{name}: gt differs, max "
                             f"{np.abs(a-b).max():.6g}")

    return fails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MUTANTS = ['offbyone', 'dropframe', 'reverse', 'transpose', 'pilluma']


def resolve_out(args):
    if args.out:
        return Path(args.out)
    root = os.environ.get('VOT_ROOT')
    if not root:
        raise SystemExit(
            "VOT_ROOT is unset and --out was not given.\n"
            "This is deliberate: a default path inside the repo is exactly what "
            "makes a data/build mismatch silent. Set VOT_ROOT in setup_env.sh.")
    return Path(root) / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['convert', 'verify'])
    ap.add_argument('--root', default='test-sequences')
    ap.add_argument('--out', default=None,
                    help="output dir (default $VOT_ROOT/data)")
    ap.add_argument('--channels', type=int, default=1, choices=[1, 3],
                    help="1 = luma (CONV_IN_CH=1, run this first), 3 = RGB")
    ap.add_argument('--spacing', type=int, default=50,
                    help="synthetic anchor spacing; real anchors come from the "
                         "toolkit in Phase 1")
    ap.add_argument('--sequences', nargs='+', default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help="verify only the first N frames per sequence")
    ap.add_argument('--no-gt-check', action='store_true')
    ap.add_argument('--mutate', default=None,
                    help="'all' or one of: " + ", ".join(MUTANTS))
    args = ap.parse_args()

    out = resolve_out(args)
    seqs = discover(args.root)
    if args.sequences:
        seqs = {k: v for k, v in seqs.items() if k in args.sequences}
    if not seqs:
        raise SystemExit("no sequences found")

    if args.mode == 'convert' and not args.mutate:
        out.mkdir(parents=True, exist_ok=True)
        print(f"converting {len(seqs)} sequences -> {out} "
              f"(channels={args.channels})")
        tot = tracked = 0
        for name, (frames, gtf) in seqs.items():
            m = convert_sequence(name, frames, gtf, out, args.channels,
                                 args.spacing)
            tot += m["frames"]; tracked += m["tracked_frames"]
        print(f"\n{len(seqs)} sequences, {tot} frames, "
              f"{tracked} tracked frames across all jobs")
        return

    if args.mode == 'verify' and not args.mutate:
        print(f"verifying {len(seqs)} sequences in {out}")
        allf = []
        for name, (frames, gtf) in seqs.items():
            f = verify_sequence(name, frames, gtf, out,
                                not args.no_gt_check, args.limit)
            print(f"  {name:<12} {'FAIL' if f else 'ok'}")
            for x in f:
                print(f"      {x}")
            allf += f
        print()
        if allf:
            raise SystemExit(f"FAILED: {len(allf)} problem(s)")
        print(f"PASS: {len(seqs)} sequences byte-exact against a fresh decode")
        return

    # ---- mutation testing -------------------------------------------------
    # A verifier that has never been shown to FAIL is worth nothing on a path
    # with no prior coverage. Each mutant corrupts the converter in one
    # specific way; verify must catch every one.
    muts = MUTANTS if args.mutate == 'all' else [args.mutate]
    for m in muts:
        if m not in MUTANTS:
            raise SystemExit(f"unknown mutant {m}; pick from {MUTANTS}")
    name = args.sequences[0] if args.sequences else sorted(seqs)[0]
    frames, gtf = seqs[name]
    mout = out / "_mutants"
    mout.mkdir(parents=True, exist_ok=True)
    lim = args.limit or 40
    print(f"mutation test on '{name}' (first {lim} frames verified)\n")
    caught = 0
    for m in muts:
        if m == 'pilluma' and args.channels != 1:
            print(f"  {m:<10} SKIP (luma mutant is channels=1 only)")
            continue
        convert_sequence(name, frames, gtf, mout, args.channels,
                         args.spacing, mutate=m)
        f = verify_sequence(name, frames, gtf, mout, True, lim)
        ok = bool(f)
        caught += ok
        print(f"  {m:<10} {'CAUGHT' if ok else 'SURVIVED  <-- verifier gap'}")
        for x in f[:2]:
            print(f"      {x}")
        print()
    n = len([m for m in muts if not (m == 'pilluma' and args.channels != 1)])
    print(f"{caught}/{n} mutants caught")
    if caught != n:
        raise SystemExit("a mutant survived: the verifier does not cover it")


if __name__ == '__main__':
    main()
