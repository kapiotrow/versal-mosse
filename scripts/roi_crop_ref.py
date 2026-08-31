#!/usr/bin/env python3
"""
roi_crop_ref.py — bit-exact NumPy model of design/pl_src/roi_crop/roi_crop.cpp.

ONE model, TWO consumers, on purpose:

  * scripts/gen_roi_crop_golden.py  — the golden for the native harness that
    checks the KERNEL against this reference.
  * scripts/phase1_sweep.py         — the offline sweep, which needs to know what
    patch a given (target, padding) would actually deliver.

Writing a float model for the sweep and an exact one for the harness would be two
implementations of the same fixed-point arithmetic, drifting apart forever. This
repo has already recorded that failure twice (the aiesim generator drifting from
the shift budget; s7's PSR check asserting 0.7x of a golden it did not resemble).

The reference is EXACT INTEGER, not float, and that is not fastidiousness. The
sweep's output variable IS a fixed-point scale — it reads `accum %rail` and
`response %rail` against 32767 — and that depends on the int8 patch's contrast
after the +/-127 clip and after LOG_LUT's compression. A float log gets the bright
end wrong (the LUT's top gap is 46/65535, vs a continuous log1p) and skipping the
clip inflates the tails. Both bias exactly the quantity being swept.

-----------------------------------------------------------------------------
STATUS OF THE THING THIS MODELS — READ BEFORE TRUSTING ANY OUTPUT

The kernel's bilinear interpolator HAS NEVER EXECUTED. Every build to date runs
roi_h == patch_rows, which makes step_y exactly 256, hence fy == fx == 0, hence

    top = p00 << 8 ;  val = p00 << 16 ;  pix = p00

i.e. the entire bilinear datapath (ap_uint<18> top, ap_uint<27> val, the >>16)
collapses to a copy. CLAUDE.md's claim that roi_crop was "verified bit-exact
against a NumPy reference in native C simulation (6 cases: 1:1, 2x up, 2x down,
both edge-clamp paths, whole frame)" is NOT backed by anything in the repository:
no testbench, no csim target, no reference, and nothing in git history. Treat the
resample path as unverified until `make test_roi_crop` passes.

-----------------------------------------------------------------------------
SEMANTICS A NAIVE NumPy PORT GETS WRONG

Each row is a real trap, not a hypothetical. Line numbers are roi_crop.cpp.

  :133 sample grid    Corner-aligned and HALF-OPEN: sy = roi_row*256 + r*step_y,
                      so samples span [roi_row, roi_row + roi_h - step/256).
                      np.linspace(..., endpoint=True) is wrong by up to a full
                      source pixel at the bottom edge. There is NO +0.5 pixel-
                      centre term — the OpenCV INTER_LINEAR convention would
                      shift everything by half a source pixel.
  :135 y0 = sy >> 8   Arithmetic shift = FLOOR. int(sy/256) and np.trunc round
                      toward zero and are off by one for EVERY negative
                      coordinate. This is the single most likely bug in a
                      hand-written reference, and the host does produce negative
                      roi_row (pos_row - roi_h/2).
  :134 fy = sy & 255  Two's-complement AND gives the POSITIVE residue for
                      negative sy. abs(sy) % 256 and math.fmod are both wrong.
  :139 clamp order    y0 and y1 are computed FIRST (y1 = y0 + 1), then clamped
                      INDEPENDENTLY. Clamping sy first, or deriving y1 as
                      min(y0+1, max_y) after clamping y0, differs when y0 < 0.
  :165 pix = val>>16  TRUNCATES. The kernel truncates here and rounds later
                      (:212) — an inconsistency worth preserving exactly, not
                      tidying up.
  :178 mean           Integer floor division, and :183 ex2 likewise. var is then
                      floor(ex2) - floor(mean)^2, which is NOT floor(true var):
                      the two independent floors can shift it by O(1). np.var is
                      wrong by exactly the amount that decides the
                      ROI_INV_Q_MAX branch.
  :193 invsig         float32 THROUGHOUT, evaluated left to right:
                      (1/sqrtf(var) * 32.0f) * 65536.0f. float64, or
                      reassociating the product, moves the last bits and (int)
                      truncation at :195 can then land either side.
  :195 inv_q          (int) truncates toward zero. Not round.
  :212 rnd            (prod + (1<<15)) >> 16 on a signed value is arithmetic
                      shift = ROUND-HALF-TOWARD-+INF. np.round is banker's
                      rounding — a deviation phase1_sweep.py:39 already documents
                      as a known model error. Do not import it here.
  :215 clip           SYMMETRIC +/-127. Not -128. An np.int8 cast or
                      np.clip(-128, 127) silently admits a value the kernel
                      cannot emit.
  :188 var == 0       Explicit branch to inv_q = 0 (all-zero patch). Dividing
                      through would give inf/nan.

All integer arithmetic is done in int64 / Python int. The kernel's ap_uint widths
are asserted rather than emulated, so a geometry that would overflow them fails
loudly here instead of wrapping silently there.

@thesis subsec:weryfikacja | A-05 | The bit-exact NumPy model of roi_crop, and the source
  of the 25 golden cases. THE groundtruth for Stage A.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Constants — mirrored from roi_crop.h, asserted against it below.
# --------------------------------------------------------------------------
ROI_FRAC_BITS = 8
ROI_FRAC_ONE = 1 << ROI_FRAC_BITS
ROI_NORM_Q = 32
ROI_INV_Q_MAX = 1 << 20
ROI_MAX_PATCH_ROWS = 128
ROI_MAX_PATCH_COLS = 128
ROI_MAX_PATCH_ELEMS = ROI_MAX_PATCH_ROWS * ROI_MAX_PATCH_COLS

_HERE = Path(__file__).resolve().parent
ROI_CROP_CPP = _HERE.parent / "design" / "pl_src" / "roi_crop" / "roi_crop.cpp"
ROI_CROP_H = _HERE.parent / "design" / "pl_src" / "roi_crop" / "roi_crop.h"


# --------------------------------------------------------------------------
# LOG_LUT — parsed from the kernel, then cross-checked against the closed form.
# --------------------------------------------------------------------------
def _parse_log_lut(cpp_path: Path) -> np.ndarray:
    """Extract LOG_LUT from roi_crop.cpp.

    PARSED rather than recomputed so the reference cannot drift from the kernel:
    if someone regenerates the table, this model follows automatically. The
    closed-form assert below then acts as an INDEPENDENT derivation, so a
    corrupted parse is caught too. Belt and braces, because every other check in
    this file is downstream of the table being right.
    """
    text = cpp_path.read_text()
    start = text.index("static const ap_uint<16> LOG_LUT[256]")
    end = text.index("};", start)
    vals = [int(m) for m in re.findall(r"L\(\s*(\d+)\s*\)", text[start:end])]
    if len(vals) != 256:
        raise ValueError(
            f"expected 256 LOG_LUT entries in {cpp_path}, parsed {len(vals)}"
        )
    return np.array(vals, dtype=np.int64)


def _closed_form_log_lut() -> np.ndarray:
    """round(65535 * log1p(v) / log1p(255)), round-half-up.

    floor(x + 0.5), not np.round: np.round is banker's rounding and would differ
    on any exact .5 (v=15 lands on 32767.5). They happen to agree for this table,
    but relying on that is how the next table becomes subtly wrong.
    """
    v = np.arange(256, dtype=np.float64)
    exact = 65535.0 * np.log1p(v) / np.log1p(255.0)
    return np.floor(exact + 0.5).astype(np.int64)


LOG_LUT = _parse_log_lut(ROI_CROP_CPP)

_lut_closed = _closed_form_log_lut()
if not np.array_equal(LOG_LUT, _lut_closed):
    _bad = np.nonzero(LOG_LUT != _lut_closed)[0]
    raise AssertionError(
        "roi_crop.cpp LOG_LUT no longer matches round(65535*log1p(v)/log1p(255)); "
        f"{_bad.size} entries differ, first at v={_bad[0]} "
        f"(kernel {LOG_LUT[_bad[0]]}, closed form {_lut_closed[_bad[0]]})"
    )


def _check_header_constants() -> None:
    """Fail loudly if roi_crop.h's constants drift away from the mirror above."""
    text = ROI_CROP_H.read_text()
    for name, expect in (
        ("ROI_FRAC_BITS", ROI_FRAC_BITS),
        ("ROI_NORM_Q", ROI_NORM_Q),
        ("ROI_MAX_PATCH_ROWS", ROI_MAX_PATCH_ROWS),
        ("ROI_MAX_PATCH_COLS", ROI_MAX_PATCH_COLS),
    ):
        m = re.search(rf"#define\s+{name}\s+(\S+)", text)
        if m is None:
            raise AssertionError(f"{name} not found in {ROI_CROP_H}")
        got = int(m.group(1), 0)
        if got != expect:
            raise AssertionError(
                f"{name}: roi_crop.h says {got}, roi_crop_ref.py mirrors {expect}"
            )
    m = re.search(r"#define\s+ROI_INV_Q_MAX\s+\(1\s*<<\s*(\d+)\)", text)
    if m is None or (1 << int(m.group(1))) != ROI_INV_Q_MAX:
        raise AssertionError("ROI_INV_Q_MAX drifted from roi_crop.h")


_check_header_constants()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def q8_step(extent: int, out_n: int) -> int:
    """step = (extent << 8) / out_n, exactly as roi_crop.cpp:109-110.

    C++ integer division on non-negative operands truncates, which matches
    Python's // here. Guarded against out_n <= 0, which the kernel does NOT
    guard and where it would divide by zero.
    """
    if out_n <= 0:
        raise ValueError(f"patch dimension must be positive, got {out_n}")
    return (int(extent) * ROI_FRAC_ONE) // int(out_n)


def step_is_exact(extent: int, out_n: int) -> bool:
    """Does the Q8 step divide evenly?

    It always does at supported geometries: out_n is a power of two <= 256
    (ROI_MAX_PATCH_ROWS is 128), so 256/out_n is an integer and
    step = extent * (256/out_n) is exact for every integer extent. I had assumed
    truncation here would accumulate into sub-pixel drift over 128 rows; it
    cannot. This predicate exists so that raising ROI_MAX_PATCH_ROWS above 256
    fails a test rather than silently introducing the drift.
    """
    return (int(extent) * ROI_FRAC_ONE) % int(out_n) == 0


def centring_bias_px(extent: int, out_n: int) -> float:
    """Sub-pixel offset between the patch centre and the true ROI centre, in
    SOURCE pixels.

    Samples span [roi, roi + extent - step/256), so their mean position is
    roi + extent/2 - step/512 — the patch sits step/512 source pixels above and
    left of centre. That is 0.5 px at the 1:1 geometry shipped today and grows
    with the resample ratio.

    It matters because the tracker is CLOSED LOOP: pos_row is updated from the
    response peak every frame (mosse_tracker.cpp:1480), so a constant bias in the
    crop does not average out — it pushes the ROI the same direction every frame.
    """
    return q8_step(extent, out_n) / (2.0 * ROI_FRAC_ONE)


# --------------------------------------------------------------------------
# Pass 1 — bilinear resample (roi_crop.cpp:131-173)
# --------------------------------------------------------------------------
def resample_q8(
    frame: np.ndarray,
    roi_row: int,
    roi_col: int,
    roi_h: int,
    roi_w: int,
    patch_rows: int,
    patch_cols: int,
) -> np.ndarray:
    """Q8 bilinear resample with border replication. Returns uint8 [pr, pc].

    `frame` is uint8 [frame_rows, frame_cols], row-major, matching the kernel's
    linear frame_buf.
    """
    if frame.dtype != np.uint8:
        raise TypeError(f"frame must be uint8, got {frame.dtype}")
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2-D, got shape {frame.shape}")

    frame_rows, frame_cols = (int(x) for x in frame.shape)
    max_y = frame_rows - 1
    max_x = frame_cols - 1

    step_y = q8_step(roi_h, patch_rows)
    step_x = q8_step(roi_w, patch_cols)

    # roi_row * ROI_FRAC_ONE, never roi_row << ROI_FRAC_BITS: the kernel's shift
    # is UB for negative roi_row before C++20 and the host does produce those.
    r = np.arange(patch_rows, dtype=np.int64)
    c = np.arange(patch_cols, dtype=np.int64)
    sy = np.int64(roi_row) * ROI_FRAC_ONE + r * step_y
    sx = np.int64(roi_col) * ROI_FRAC_ONE + c * step_x

    # & on two's complement gives the positive residue for negative sy;
    # >> on a signed int64 is arithmetic, i.e. floor. Both are what the kernel
    # does and both differ from the "obvious" float formulations.
    fy = sy & (ROI_FRAC_ONE - 1)
    fx = sx & (ROI_FRAC_ONE - 1)
    y0 = sy >> ROI_FRAC_BITS
    x0 = sx >> ROI_FRAC_BITS
    y1 = y0 + 1
    x1 = x0 + 1

    # Clamp AFTER the increment, each independently — roi_crop.cpp:139-140.
    y0 = np.clip(y0, 0, max_y)
    y1 = np.clip(y1, 0, max_y)
    x0 = np.clip(x0, 0, max_x)
    x1 = np.clip(x1, 0, max_x)

    f = frame.astype(np.int64)
    p00 = f[np.ix_(y0, x0)]
    p01 = f[np.ix_(y0, x1)]
    p10 = f[np.ix_(y1, x0)]
    p11 = f[np.ix_(y1, x1)]

    fxr = fx[None, :]
    fyr = fy[:, None]
    top = p00 * (ROI_FRAC_ONE - fxr) + p01 * fxr
    bot = p10 * (ROI_FRAC_ONE - fxr) + p11 * fxr
    val = top * (ROI_FRAC_ONE - fyr) + bot * fyr

    # The kernel declares top/bot as ap_uint<18> and val as ap_uint<27>. Assert
    # rather than emulate: a geometry that would overflow those should fail here
    # loudly, not wrap silently in hardware.
    if top.size and int(max(top.max(), bot.max())) >= (1 << 18):
        raise AssertionError("top/bot exceeded the kernel's ap_uint<18>")
    if val.size and int(val.max()) >= (1 << 27):
        raise AssertionError("val exceeded the kernel's ap_uint<27>")

    pix = val >> (2 * ROI_FRAC_BITS)  # TRUNCATES; the rounding happens later
    return pix.astype(np.uint8)


# --------------------------------------------------------------------------
# Statistics (roi_crop.cpp:175-196)
# --------------------------------------------------------------------------
def stats_q(patch_u8: np.ndarray) -> tuple[int, int, int]:
    """Return (mean, var, inv_q) exactly as the kernel computes them.

    All three are integers. var is floor(ex2) - floor(mean)^2 with two
    INDEPENDENT floors, which is not floor(true variance); do not "fix" it.
    """
    n_elems = int(patch_u8.size)
    if n_elems == 0:
        raise ValueError("empty patch")

    lv = LOG_LUT[patch_u8.astype(np.int64)]
    sum_x = int(lv.sum())
    sum_x2 = int((lv * lv).sum())

    if sum_x2 >= (1 << 48):
        raise AssertionError("sum_x2 exceeded the kernel's ap_uint<48>")

    mean = sum_x // n_elems
    ex2 = sum_x2 // n_elems
    mean_sq = mean * mean
    var = ex2 - mean_sq if ex2 > mean_sq else 0

    if var == 0:
        inv_q = 0
    else:
        # float32 throughout, left-to-right, matching roi_crop.cpp:193-195.
        invsig = np.float32(1.0) / np.sqrt(np.float32(var))
        scaled = np.float32(invsig * np.float32(ROI_NORM_Q)) * np.float32(65536.0)
        inv_q = (
            ROI_INV_Q_MAX
            if scaled >= np.float32(ROI_INV_Q_MAX)
            else int(scaled)  # truncates toward zero, not round
        )
    return mean, var, inv_q


# --------------------------------------------------------------------------
# Pass 1b — normalize in place (roi_crop.cpp:198-220)
# --------------------------------------------------------------------------
def normalize_q(patch_u8: np.ndarray, mean: int, inv_q: int) -> np.ndarray:
    """q = clip(round((LOG_LUT[x] - mean) * inv_q >> 16), +/-127). Returns int8."""
    lv = LOG_LUT[patch_u8.astype(np.int64)]
    dev = lv - np.int64(mean)
    prod = dev * np.int64(inv_q)

    if prod.size and int(np.abs(prod).max()) >= (1 << 47):
        raise AssertionError("prod exceeded the kernel's ap_int<48>")

    # Arithmetic shift on a signed value: round-half-toward-+inf, NOT banker's.
    rnd = (prod + (1 << 15)) >> 16
    return np.clip(rnd, -127, 127).astype(np.int8)  # symmetric: never -128


# --------------------------------------------------------------------------
# The whole chain
# --------------------------------------------------------------------------
def stage_a(
    frame: np.ndarray,
    roi_row: int,
    roi_col: int,
    roi_h: int,
    roi_w: int,
    patch_rows: int,
    patch_cols: int,
    with_diag: bool = False,
):
    """Full Stage A. Returns the int8 patch PatchIn actually carries.

    `frame` is either 2-D [rows, cols] (grayscale, ROI_IN_CH=1) or 3-D
    [3, rows, cols] PLANAR (RGB, ROI_IN_CH=3). The return follows: [pr, pc] for
    grayscale, [pr, pc, 3] PIXEL-INTERLEAVED for RGB — so `.tofile()` on the
    result is byte-for-byte the AXIS wire order the kernel emits and conv2d
    unpacks. Planar in, interleaved out, deliberately: the planes are what the
    resampler and the host think in, the interleave is what the wire carries.

    NORMALIZATION IS JOINT ACROSS PLANES. One mean and one inv_q over all
    3*pr*pc samples, applied to all three. Per-plane statistics would equalize
    the planes and delete the chromatic contrast RGB exists for — the failure is
    silent, since the output looks entirely reasonable. `stats_q` is
    shape-agnostic (it reduces over `.size`), so the joint statistic is what you
    get by handing it the stacked array and nothing else changes.

    With `with_diag`, returns (patch_i8, diag) where diag carries what the sweep
    needs to tell a bad (target, padding) apart from a good one — see the fields
    below, each of which surfaces a failure mode no PSR number reveals.
    """
    planar = frame.ndim == 3
    planes = frame if planar else frame[None]
    if planes.shape[0] not in (1, 3):
        raise ValueError(f"expected 1 or 3 planes, got {planes.shape[0]}")

    resampled = np.stack([
        resample_q8(pl, roi_row, roi_col, roi_h, roi_w, patch_rows, patch_cols)
        for pl in planes
    ])                                            # [P, pr, pc]

    mean, var, inv_q = stats_q(resampled)         # JOINT over all P*pr*pc

    q = np.stack([normalize_q(resampled[i], mean, inv_q)
                  for i in range(planes.shape[0])])
    # [P, pr, pc] -> [pr, pc, P], i.e. R0 G0 B0 R1 G1 B1 ... in memory order.
    patch = q[0] if not planar else np.moveaxis(q, 0, -1).copy()
    if not with_diag:
        return patch

    step_y = q8_step(roi_h, patch_rows)
    step_x = q8_step(roi_w, patch_cols)
    r = np.arange(patch_rows, dtype=np.int64)
    c = np.arange(patch_cols, dtype=np.int64)
    sy = np.int64(roi_row) * ROI_FRAC_ONE + r * step_y
    sx = np.int64(roi_col) * ROI_FRAC_ONE + c * step_x

    diag = {
        "step_y": step_y,
        "step_x": step_x,
        # roi_h/patch_rows: the localisation quantum in FRAME pixels. At 2.0 the
        # tracker cannot resolve better than 2 px however good its PSR is.
        "px_per_bin_y": step_y / ROI_FRAC_ONE,
        "px_per_bin_x": step_x / ROI_FRAC_ONE,
        # How many rows/cols actually exercise the interpolator. 0 means this
        # geometry still runs the degenerate copy path that shipped.
        "fy_nonzero": int(np.count_nonzero(sy & (ROI_FRAC_ONE - 1))),
        "fx_nonzero": int(np.count_nonzero(sx & (ROI_FRAC_ONE - 1))),
        # Distinct source rows/cols touched. Much less than roi_h means the
        # 2x2-neighbourhood sampler is skipping source rows outright — there is
        # no prefilter, so beyond 2x decimation the signal is aliased and no
        # shift budget can repair it.
        "src_rows_touched": int(np.unique(np.clip(sy >> ROI_FRAC_BITS, 0, frame.shape[0] - 1)).size),
        "src_cols_touched": int(np.unique(np.clip(sx >> ROI_FRAC_BITS, 0, frame.shape[1] - 1)).size),
        "centring_bias_y": centring_bias_px(roi_h, patch_rows),
        "centring_bias_x": centring_bias_px(roi_w, patch_cols),
        "step_exact": step_is_exact(roi_h, patch_rows) and step_is_exact(roi_w, patch_cols),
        "in_ch": int(planes.shape[0]),
        "mean": mean,
        "var": var,
        "inv_q": inv_q,
        "inv_q_capped": inv_q == ROI_INV_Q_MAX,
        "clipped": int(np.count_nonzero(np.abs(patch.astype(np.int64)) == 127)),
        # n_elems counts SAMPLES (all planes), which is what the kernel reduces
        # over and what decides mean/var. Not pixels.
        "n_elems": int(patch.size),
        "patch_std": float(patch.astype(np.float64).std()),
    }
    return patch, diag


# --------------------------------------------------------------------------
# AXIS packing — the convention asserted in three places and checked in none
# --------------------------------------------------------------------------
def stage_a_rgb(planes_u8, roi_row, roi_col, roi_h, roi_w,
                patch_rows, patch_cols, with_diag=False):
    """Explicit RGB entry point. `planes_u8` is [3, rows, cols] planar.

    Thin alias for stage_a — the 3-plane path is not a separate implementation,
    which is the whole point of this module having one model.
    """
    planes = np.asarray(planes_u8)
    if planes.ndim != 3 or planes.shape[0] != 3:
        raise ValueError(f"expected [3, rows, cols], got {planes.shape}")
    return stage_a(planes, roi_row, roi_col, roi_h, roi_w,
                   patch_rows, patch_cols, with_diag)


def pack_axis_words(patch_i8: np.ndarray) -> np.ndarray:
    """Pack the int8 patch into 32-bit AXIS beats, 4 pixels per beat, LSB first.

    Mirrors roi_crop.cpp:239-243. The same convention is independently restated
    in roi_crop.h:26-30, phase1_sweep.py's unpack, and gen_aiesim_vectors.py's
    write_plio_txt — three assertions of one contract, with nothing checking that
    they agree. This function is what lets the harness check it.
    """
    if patch_i8.size % 4 != 0:
        raise ValueError(
            "patch_cols must be a multiple of 4: the kernel's PASS2_COL steps by "
            "4 and total_beats = n_elems >> 2, so a non-multiple both reads past "
            "the row and never asserts word.last, stalling the PLIO"
        )
    flat = patch_i8.reshape(-1).astype(np.uint8).astype(np.uint32)
    q = flat.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (q[:, 3] << 24)).astype(np.uint32)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
def _self_check() -> int:
    failures = 0

    def check(what: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {what:<34} {'OK  ' if cond else 'FAIL'}{' — ' + detail if detail else ''}")
        if not cond:
            failures += 1

    print("roi_crop_ref self-check\n")

    check("LOG_LUT parsed", LOG_LUT.size == 256, f"{LOG_LUT.size} entries")
    check("LOG_LUT matches closed form", np.array_equal(LOG_LUT, _lut_closed))
    check("LOG_LUT monotone", bool(np.all(np.diff(LOG_LUT) > 0)))
    check("LOG_LUT spans uint16", LOG_LUT[0] == 0 and LOG_LUT[255] == 65535)

    # The withdrawn drift hazard, encoded as a test.
    exact = all(
        step_is_exact(h, p)
        for p in (16, 32, 64, 128)
        for h in range(1, 4097)
    )
    check("Q8 step exact, all supported dims", exact,
          "raising ROI_MAX_PATCH_ROWS past 256 breaks this")

    # Negative coordinates: floor vs truncate, and the positive residue.
    sy = np.array([-1, -256, -257], dtype=np.int64)
    check("negative >> is floor", list(sy >> 8) == [-1, -1, -2], str(list(sy >> 8)))
    check("negative & 255 is positive", list(sy & 255) == [255, 0, 255], str(list(sy & 255)))

    # 1:1 geometry must be an exact crop — this is the only path ever built, and
    # it is the case where the interpolator is provably dead.
    rng = np.random.default_rng(20260816)
    frame = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)
    got = resample_q8(frame, 16, 16, 128, 128, 128, 128)
    check("1:1 resample is an exact crop",
          np.array_equal(got, frame[16:144, 16:144]))
    _, d = stage_a(frame, 16, 16, 128, 128, 128, 128, with_diag=True)
    check("1:1 leaves interpolator dead",
          d["fy_nonzero"] == 0 and d["fx_nonzero"] == 0,
          f"fy {d['fy_nonzero']}, fx {d['fx_nonzero']}")

    # 2x upsample: the first geometry with a nonzero fraction, ever.
    _, d2 = stage_a(frame, 30, 30, 64, 64, 128, 128, with_diag=True)
    check("2x up exercises interpolator",
          d2["fy_nonzero"] == 64 and d2["fx_nonzero"] == 64,
          f"fy {d2['fy_nonzero']}/128, step {d2['step_y']}")

    # 2x downsample reads half the source rows — the aliasing signature.
    _, d3 = stage_a(rng.integers(0, 256, (320, 320), dtype=np.uint8),
                    20, 20, 256, 256, 128, 128, with_diag=True)
    check("2x down touches 128 src rows", d3["src_rows_touched"] == 128,
          f"{d3['src_rows_touched']} of 256")

    # Flat frame -> var 0 -> inv_q 0 -> all zeros (not a division by zero).
    flat = np.full((160, 160), 77, dtype=np.uint8)
    p_flat, d_flat = stage_a(flat, 0, 0, 128, 128, 128, 128, with_diag=True)
    check("flat frame gives var 0", d_flat["var"] == 0 and d_flat["inv_q"] == 0)
    check("flat frame gives all zeros", bool(np.all(p_flat == 0)))

    # Symmetric clip: -128 must never be emitted.
    bimodal = np.where(rng.random((160, 160)) < 0.5, 0, 255).astype(np.uint8)
    p_bi = stage_a(bimodal, 0, 0, 128, 128, 128, 128)
    check("clip is symmetric, never -128", int(p_bi.min()) >= -127,
          f"min {int(p_bi.min())}, max {int(p_bi.max())}")

    # Centring bias, the hazard that does not cancel in a closed loop.
    check("centring bias 0.5 px at 1:1", centring_bias_px(128, 128) == 0.5)
    check("centring bias grows with ratio", centring_bias_px(256, 128) == 1.0)

    # ---------------------------------------------------------------
    # RGB (ROI_IN_CH=3)
    # ---------------------------------------------------------------
    lum = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)

    # THE CONTROL, and the strongest statement available here: three IDENTICAL
    # planes must reproduce the grayscale result exactly, plane for plane.
    # It holds by construction — with P copies, sum_x and n both scale by P, so
    # mean is unchanged, and likewise ex2 — which is what makes it a real test of
    # the JOINT reduction rather than a tautology: a per-plane or mis-weighted
    # reduction would still pass a "looks reasonable" eyeball and fail this.
    g_out = stage_a(lum, 12, 20, 100, 100, 128, 128)
    rgb_same = stage_a_rgb(np.stack([lum] * 3), 12, 20, 100, 100, 128, 128)
    check("RGB shape is interleaved", rgb_same.shape == (128, 128, 3),
          str(rgb_same.shape))
    check("3 identical planes reproduce gray",
          all(np.array_equal(rgb_same[:, :, k], g_out) for k in range(3)))

    # ...and the discriminator: with DIFFERENT planes, the joint statistic must
    # NOT equal what per-plane normalization would give. If these ever agree the
    # test frame has become degenerate and the check above proves nothing.
    planes = np.stack([lum,
                       np.clip(lum.astype(np.int16) // 2 + 40, 0, 255).astype(np.uint8),
                       (255 - lum).astype(np.uint8)])
    joint = stage_a_rgb(planes, 12, 20, 100, 100, 128, 128)
    per_plane = np.stack([stage_a(pl, 12, 20, 100, 100, 128, 128)
                          for pl in planes], axis=-1)
    check("joint != per-plane on real colour",
          not np.array_equal(joint, per_plane),
          f"{int(np.count_nonzero(joint != per_plane))} of {joint.size} samples differ")

    # The per-plane MEANS must survive. Under per-plane normalization every
    # plane is centred on 0 by construction; under joint they are not, and that
    # offset IS the chromatic information.
    pm = [float(joint[:, :, k].mean()) for k in range(3)]
    check("joint preserves inter-plane offsets",
          max(abs(m) for m in pm) > 1.0,
          "plane means " + ", ".join(f"{m:+.1f}" for m in pm))

    # ap_uint<48> headroom on sum_x2 at 3 planes — the kernel's widest reduction.
    worst = 3 * 128 * 128 * 65535 * 65535
    check("sum_x2 fits ap_uint<48> at 3 planes", worst < (1 << 48),
          f"{worst:.3e} vs {float(1 << 48):.3e} ({100.0 * worst / (1 << 48):.0f}%)")

    # A row must be a whole number of AXIS beats at 3 bytes per pixel.
    check("RGB row is beat-aligned", (128 * 3) % 4 == 0, "384 bytes = 96 beats")

    # AXIS packing round-trip.
    words = pack_axis_words(p_bi)
    unpacked = np.stack(
        [((words >> (8 * i)) & 0xFF).astype(np.uint8) for i in range(4)], axis=1
    ).reshape(-1).view(np.int8)
    check("AXIS pack round-trips", np.array_equal(unpacked, p_bi.reshape(-1)))
    check("AXIS beat count", words.size == p_bi.size // 4, f"{words.size} beats")

    print(f"\n  OVERALL: {'FAIL' if failures else 'PASS'} ({failures} failure"
          f"{'' if failures == 1 else 's'})\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
