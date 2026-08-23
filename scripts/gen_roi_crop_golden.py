#!/usr/bin/env python3
"""
gen_roi_crop_golden.py — golden data for the native roi_crop harness.

Contains NO arithmetic of its own: every expected value comes from
scripts/roi_crop_ref.py, which is the same module scripts/phase1_sweep.py uses to
model the ROI. One reference, two consumers — see roi_crop_ref's docstring for why
that matters more than it looks.

Run: uv run python3 scripts/gen_roi_crop_golden.py <output_dir>

WHAT THIS EXISTS TO CATCH
-------------------------
roi_crop's bilinear interpolator has never executed. Every build to date runs
roi_h == patch_rows, which makes step_y exactly 256, hence fy == fx == 0, hence
the whole datapath collapses to `pix = p00`. CLAUDE.md claims a 6-case bit-exact
verification that is not in the repository. So the cases below are not a
regression net over known-good behaviour; several of them are the first execution
of the code they cover, and failures are expected on the first run.

CASE SELECTION
--------------
Each case earns its place by covering something no other case does. The
interesting ones:

  up_2x           first nonzero fy in the project's history; fy=128 is the
                  maximum-weight interpolation point
  pad_2p5_aniso   step_y != step_x — the only case that would catch an x/y
                  transposition in the sampler
  clamp_topleft   negative roi_row/roi_col: arithmetic-shift floor (not trunc)
                  and the two's-complement residue for `sy & 255`. The host
                  produces negative coordinates and padding makes it common.
  clamp_botright  y1 clamps one row BEFORE y0 does — a real off-by-one trap
  cap_hit/miss    var=1 vs var=5 either side of ROI_INV_Q_MAX. Reaching the cap
                  needs a contrived input (see below), which is itself the
                  finding: the cap is not a padding hazard.
  var_floor_zero  a patch with genuine variance that the kernel's TWO
                  independent floors annihilate to var=0. np.var would give
                  0.27 here and the reference would be wrong in the one place
                  the cap branch is decided.
  nonsquare       patch_rows != patch_cols: independent n_elems, total_beats and
                  row-major indexing

THE CAP CONSTRUCTION, BECAUSE IT IS NOT OBVIOUS
-----------------------------------------------
Reaching ROI_INV_Q_MAX needs var <= 4 in LOG_LUT^2 units. That is far harder than
it sounds, because var = floor(ex2) - floor(mean)^2 uses two INDEPENDENT floors:
if the true mean has fractional part f, the computed variance picks up ~2*mean*f,
which at mean ~65535 swamps any small true variance. A single differing pixel
gives var ~130701, not ~0.13.

So the mean must be an EXACT integer. LOG_LUT[249..251] happens to have equal
gaps of 47 on both sides, so k pixels at 249 and k at 251 around a base of 250
cancel exactly. var_true is then 2*k*47^2/16384 = 0.2697*k, and the floor of ex2
turns that into the integer the kernel sees:

    k=1  -> var 0  (annihilated by the floor)   -> inv_q 0, all-zero patch
    k=4  -> var 1                               -> inv_q capped at 2^20
    k=20 -> var 5                               -> inv_q 937874, NOT capped

which gives a hit, a miss and the floor path from one family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roi_crop_ref as R  # noqa: E402
import synth_frame as SF  # noqa: E402

FRAME_ROWS, FRAME_COLS = 1080, 1920


def _tex(rows: int, cols: int, seed: int, target=(64, 64), at=None) -> np.ndarray:
    """A textured frame with a target. HARNESS_NOISE, not SWEEP_NOISE: bilinear
    between two EQUAL neighbours returns the same value for any fraction, so every
    locally flat pair is a place an interpolator bug could hide."""
    if at is None:
        at = (rows / 2.0, cols / 2.0)
    return SF.make_frame(rows, cols, target[0], target[1], at[0], at[1],
                         kind="bars", background="texture",
                         noise=SF.HARNESS_NOISE, seed=seed)


def _tex_rgb(rows: int, cols: int, seed: int, target=(64, 64), at=None) -> np.ndarray:
    """Three GENUINELY DIFFERENT planes, returned planar [3, rows, cols].

    Not three copies of one texture: with identical planes the joint reduction
    is arithmetically indistinguishable from the per-plane one (mean and ex2 both
    scale by 3), so every RGB case built that way would pass with a per-plane
    kernel. The planes here are decorrelated by construction — a different seed
    per plane, plus an inversion on B — so a kernel that normalizes per plane
    fails these outright.

    The one case that DOES use identical planes is rgb_gray_control, where the
    equivalence is the property under test.
    """
    return np.stack([
        _tex(rows, cols, seed, target=target, at=at),
        _tex(rows, cols, seed + 100, target=target, at=at),
        255 - _tex(rows, cols, seed + 200, target=target, at=at),
    ])


def _cap_frame(k: int) -> np.ndarray:
    """128x128 frame whose LOG_LUT mean is an exact integer — see the docstring."""
    p = np.full(128 * 128, 250, dtype=np.uint8)
    p[:k] = 249
    p[k:2 * k] = 251
    return p.reshape(128, 128)


def build_cases() -> list[dict]:
    C: list[dict] = []

    def add(name, frame, roi_row, roi_col, roi_h, roi_w,
            patch_rows=128, patch_cols=128, twice=False, why=""):
        # in_ch follows the frame's shape: 2-D is grayscale, [3,r,c] is RGB.
        # The harness runs only the cases matching the ROI_IN_CH it was built
        # with, so both arms share one golden directory and one manifest.
        C.append(dict(name=name, frame=frame, roi_row=roi_row, roi_col=roi_col,
                      roi_h=roi_h, roi_w=roi_w, patch_rows=patch_rows,
                      patch_cols=patch_cols, twice=twice, why=why,
                      in_ch=(3 if frame.ndim == 3 else 1)))

    add("identity_1to1", _tex(160, 160, 1, target=(40, 40)), 16, 16, 128, 128,
        why="the only geometry ever built; interpolator provably dead")
    add("down_2x", _tex(320, 320, 2, target=(80, 80)), 20, 20, 256, 256,
        why="integer decimation; exactly half the source rows read")
    add("up_2x", _tex(200, 200, 3, target=(30, 30)), 30, 30, 64, 64,
        why="FIRST nonzero fy ever; fy=128 is max-weight interpolation")
    add("pad_1p5", _tex(FRAME_ROWS, FRAME_COLS, 4, target=(85, 85), at=(463, 763)),
        400, 700, 127, 127, why="128 distinct fy phases")
    add("pad_2p0_small", _tex(FRAME_ROWS, FRAME_COLS, 5, target=(11, 11), at=(545, 965)),
        534, 954, 22, 22, why="5.8x upsample; smallest ROI")
    add("pad_2p5_aniso", _tex(FRAME_ROWS, FRAME_COLS, 6, target=(51, 51), at=(363, 364)),
        300, 300, 127, 129, why="step_y != step_x — catches an x/y transposition")
    add("pad_3p0", _tex(FRAME_ROWS, FRAME_COLS, 7, target=(43, 43), at=(164, 164)),
        100, 100, 129, 129, why="slow phase drift; adjacent to 1:1 in ratio")
    add("clamp_topleft", _tex(FRAME_ROWS, FRAME_COLS, 8, target=(64, 64), at=(60, 40)),
        -40, -60, 200, 200, why="negative coords: floor-shift and &255 residue")
    add("clamp_botright", _tex(FRAME_ROWS, FRAME_COLS, 9, target=(64, 64), at=(1060, 1900)),
        1000, 1850, 200, 200, why="y1 clamps one row before y0")
    add("outside", _tex(FRAME_ROWS, FRAME_COLS, 10), -500, -500, 100, 100,
        why="every sample clamps -> var 0 via the clamp path")
    add("whole_frame", _tex(FRAME_ROWS, FRAME_COLS, 11, target=(200, 200)),
        0, 0, FRAME_ROWS, FRAME_COLS,
        why="max step; 8.4x/15x decimation with no prefilter")
    add("flat", np.full((256, 256), 77, dtype=np.uint8), 0, 0, 128, 128,
        why="legitimately uniform image -> var 0, distinct from the clamp path")
    add("cap_hit", _cap_frame(4), 0, 0, 128, 128,
        why="var=1 -> inv_q clamped at ROI_INV_Q_MAX")
    add("cap_miss", _cap_frame(20), 0, 0, 128, 128,
        why="var=5 -> just below the cap; must NOT clamp")
    add("var_floor_zero", _cap_frame(1), 0, 0, 128, 128,
        why="real variance annihilated by the two floors; np.var would differ")
    add("nonsquare", _tex(FRAME_ROWS, FRAME_COLS, 16, target=(64, 32)),
        200, 200, 192, 96, patch_rows=96, patch_cols=128,
        why="patch_rows != patch_cols")
    add("recompute_cached", _tex(FRAME_ROWS, FRAME_COLS, 4, target=(85, 85), at=(463, 763)),
        400, 700, 127, 127, twice=True,
        why="second call must re-stream the static buffer byte-identically")

    # ---------------------------------------------------------------
    # ROI_IN_CH=3. Run by the RGB build of the harness only.
    # ---------------------------------------------------------------
    add("rgb_identity_1to1", _tex_rgb(160, 160, 21, target=(40, 40)), 16, 16, 128, 128,
        why="interleaved store + joint stats with the interpolator dead")
    add("rgb_up_2x", _tex_rgb(200, 200, 22, target=(30, 30)), 30, 30, 64, 64,
        why="interpolator active on all three planes, shared fy/fx")
    add("rgb_clamp_topleft", _tex_rgb(400, 400, 23, target=(64, 64), at=(60, 40)),
        -40, -60, 200, 200,
        why="negative coords THROUGH the *3 interleave — the likeliest indexing bug")
    add("rgb_aniso", _tex_rgb(400, 400, 24, target=(51, 51), at=(200, 200)),
        150, 150, 127, 129,
        why="step_y != step_x with 3 planes; catches an x/y or plane transposition")
    add("rgb_nonsquare", _tex_rgb(400, 400, 25, target=(64, 32)),
        100, 100, 192, 96, patch_rows=96, patch_cols=128,
        why="row_bytes = 3*patch_cols stride, and patch_rows != patch_cols")
    add("rgb_flat", np.stack([np.full((256, 256), 77, dtype=np.uint8)] * 3),
        0, 0, 128, 128, why="var 0 across all planes -> inv_q 0 -> all zeros")
    add("rgb_gray_control", np.stack([_tex(160, 160, 1, target=(40, 40))] * 3),
        16, 16, 128, 128,
        why="THE CONTROL: three identical planes must reproduce identity_1to1 "
            "exactly, plane for plane — same frame, same ROI, same geometry")
    add("rgb_recompute_cached", _tex_rgb(400, 400, 26, target=(85, 85), at=(200, 200)),
        150, 150, 127, 127, twice=True,
        why="the 48 KB static buffer must re-stream byte-identically")
    return C


def main(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()

    print(f"roi_crop golden -> {outdir}")
    print(f"{'case':<18} {'roi':>18} {'patch':>9} {'step y/x':>10} "
          f"{'fy nz':>6} {'var':>10} {'inv_q':>8} {'clip':>6}")
    print("-" * 96)

    manifest = [str(len(cases))]
    for c in cases:
        patch, d = R.stage_a(c["frame"], c["roi_row"], c["roi_col"],
                             c["roi_h"], c["roi_w"],
                             c["patch_rows"], c["patch_cols"], with_diag=True)

        # The kernel reads frame_buf INTERLEAVED, so planar [3,r,c] is written
        # as [r,c,3]. stage_a() already returns the patch interleaved.
        frame_out = (np.moveaxis(c["frame"], 0, -1).copy()
                     if c["in_ch"] == 3 else c["frame"])
        frame_out.tofile(outdir / f"{c['name']}_frame.bin")
        patch.tofile(outdir / f"{c['name']}_patch.bin")

        fr, fc = (int(x) for x in c["frame"].shape[-2:])
        with open(outdir / f"{c['name']}.txt", "w") as f:
            f.write(f"name {c['name']}\n")
            f.write(f"frame_rows {fr}\nframe_cols {fc}\n")
            f.write(f"roi_row {c['roi_row']}\nroi_col {c['roi_col']}\n")
            f.write(f"roi_h {c['roi_h']}\nroi_w {c['roi_w']}\n")
            f.write(f"patch_rows {c['patch_rows']}\npatch_cols {c['patch_cols']}\n")
            f.write(f"twice {1 if c['twice'] else 0}\n")
            f.write(f"in_ch {c['in_ch']}\n")
            # Diagnostics: not asserted end-to-end, but printed by the harness on
            # failure so a mismatch localises without a second run.
            f.write(f"exp_mean {d['mean']}\nexp_var {d['var']}\n")
            f.write(f"exp_inv_q {d['inv_q']}\n")
            f.write(f"exp_beats "
                    f"{(c['patch_rows'] * c['patch_cols'] * c['in_ch']) // 4}\n")
            f.write(f"step_y {d['step_y']}\nstep_x {d['step_x']}\n")
            f.write(f"fy_nonzero {d['fy_nonzero']}\nfx_nonzero {d['fx_nonzero']}\n")
            f.write(f"why {c['why']}\n")

        manifest.append(c["name"])
        print(f"{c['name']:<18} {str(c['roi_h'])+'x'+str(c['roi_w']):>18} "
              f"{str(c['patch_rows'])+'x'+str(c['patch_cols']):>9} "
              f"{str(d['step_y'])+'/'+str(d['step_x']):>10} "
              f"{d['fy_nonzero']:>6} {d['var']:>10} {d['inv_q']:>8} "
              f"{d['clipped']:>6}")

    (outdir / "manifest.txt").write_text("\n".join(manifest) + "\n")

    # Coverage assertions: if a future edit makes a case degenerate, say so here
    # rather than letting the harness pass vacuously.
    diags = {}
    for c in cases:
        _, d = R.stage_a(c["frame"], c["roi_row"], c["roi_col"], c["roi_h"],
                         c["roi_w"], c["patch_rows"], c["patch_cols"], with_diag=True)
        diags[c["name"]] = d

    problems = []
    if diags["up_2x"]["fy_nonzero"] == 0:
        problems.append("up_2x no longer exercises the interpolator")
    if diags["pad_2p5_aniso"]["step_y"] == diags["pad_2p5_aniso"]["step_x"]:
        problems.append("pad_2p5_aniso no longer has step_y != step_x")
    if not diags["cap_hit"]["inv_q_capped"]:
        problems.append("cap_hit no longer reaches ROI_INV_Q_MAX")
    if diags["cap_miss"]["inv_q_capped"]:
        problems.append("cap_miss now reaches the cap — it is the negative control")
    if diags["var_floor_zero"]["var"] != 0:
        problems.append("var_floor_zero no longer exercises the floor annihilation")
    if diags["flat"]["inv_q"] != 0 or diags["outside"]["inv_q"] != 0:
        problems.append("a var==0 case no longer yields inv_q 0")
    n_interp = sum(1 for d in diags.values() if d["fy_nonzero"] or d["fx_nonzero"])
    if n_interp < 8:
        problems.append(f"only {n_interp} cases exercise the interpolator")

    print()
    if problems:
        for p in problems:
            print(f"  COVERAGE PROBLEM: {p}")
        return 1
    print(f"  {len(cases)} cases, {n_interp} exercise the bilinear interpolator "
          f"(shipped geometry: 0)")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("golden")
    raise SystemExit(main(out))
