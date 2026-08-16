#!/usr/bin/env python3
"""
synth_frame.py — synthetic source FRAMES for the ROI sweep and the roi_crop harness.

Frames, not patches. That distinction is the whole point: every existing generator
in this repo synthesizes at PATCH resolution, which silently assumes the ROI and
the patch are the same thing. Padding is exactly the change that breaks the
assumption, so the source has to exist at frame resolution and be cropped.

-----------------------------------------------------------------------------
WHY THE BACKGROUND DEFAULTS TO TEXTURE, AND WHY THAT IS NOT COSMETIC

`inject_target_frame` (mosse_tracker.cpp:903) does

    memset(frame_buf, BACKGROUND=40, rows*cols)

then draws an ~11x11 target. Padding's entire justification in both papers is
that the window must be larger than the object SO THE FILTER LEARNS
TARGET-VS-BACKGROUND (Bolme 3.1, Danelljan 3.1). Against a flat background there
is nothing to learn: more padding is strictly less target, so a padding sweep run
against that frame will report a monotone decline and "prove" padding is pure
loss. That is a correct answer to the wrong question, and it would be a very
convincing one.

So `background='texture'` is the default here, and `--bg-contrast` exists so that
"how much does the answer depend on the background I invented?" is itself a
measurable axis rather than a hidden assumption.

-----------------------------------------------------------------------------
WHY THE TEXTURE IS A CONTINUOUS FUNCTION, NOT PER-PIXEL NOISE

The texture is a band-limited 2-D Fourier series evaluated at pixel coordinates,
plus a fixed-seed noise floor. Band-limited matters: the sweep compares different
padding factors, i.e. different resample ratios, and roi_crop's bilinear sampler
has NO prefilter. White noise would alias differently at every ratio, so the sweep
would be measuring aliasing of the test signal rather than the effect of padding.
A band-limited field resamples predictably and keeps the comparison fair.

-----------------------------------------------------------------------------
THE TWO CONSUMERS WANT OPPOSITE THINGS FROM THE NOISE FLOOR

  * The SWEEP wants the field as band-limited as possible, for the reason above.
    SWEEP_NOISE = 2.0.
  * The HARNESS wants a large gradient between EVERY adjacent pixel pair, because
    bilinear interpolation between two equal neighbours returns the same value for
    any fraction — so an interpolator bug is invisible wherever the source is
    locally flat. HARNESS_NOISE = 6.0.

Measured flat-neighbour fraction on a 256x256 texture: 16.8% at noise 0, 10.6% at
2.0, 4.5% at 6.0. Using one setting for both would either blind ~1 pixel in 6 of
the harness or contaminate the sweep with aliasing, so the two constants below are
deliberately different and each consumer names the one it needs.
"""

from __future__ import annotations

import numpy as np

# The original constants from inject_target_frame (mosse_tracker.cpp:899-901),
# kept so `kind='bars'` at 11x11 reproduces the shipped target exactly.
BACKGROUND = 40
BAR_VALUE = 220
SPUR_VALUE = 150

# Nominal target box the shipped shape was drawn in: |dr| <= 5 and dc in [-2, 8]
# is 11 rows x 11 cols.
_NOMINAL = 11.0

# See the module docstring: the sweep and the harness need different noise floors
# for opposite and equally good reasons. Named rather than passed as magic numbers
# so a future change has to confront the trade-off.
SWEEP_NOISE = 2.0
HARNESS_NOISE = 6.0


def _texture(rows: int, cols: int, contrast: float, seed: int) -> np.ndarray:
    """Band-limited background in [-1, 1], as a function of pixel coordinates."""
    rng = np.random.default_rng(seed)
    y = np.arange(rows, dtype=np.float64).reshape(-1, 1)
    x = np.arange(cols, dtype=np.float64).reshape(1, -1)
    field = np.zeros((rows, cols), dtype=np.float64)

    # A handful of low-order components: enough structure that no region is flat,
    # few enough that the field is smooth relative to any resample ratio swept.
    for _ in range(6):
        fy = rng.uniform(1.0, 6.0) / max(rows, 1)
        fx = rng.uniform(1.0, 6.0) / max(cols, 1)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.4, 1.0)
        field += amp * np.sin(2.0 * np.pi * (fy * y + fx * x) + ph)

    peak = np.abs(field).max()
    if peak > 0:
        field /= peak
    # Small noise floor so the field has some high-frequency content too — but at
    # a level that cannot dominate the band-limited part.
    field += 0.05 * rng.standard_normal((rows, cols))
    return np.clip(field * contrast, -1.0, 1.0)


def _background(rows: int, cols: int, kind: str, contrast: float, seed: int) -> np.ndarray:
    if kind == "flat":
        return np.full((rows, cols), float(BACKGROUND), dtype=np.float64)
    if kind == "gradient":
        y = np.arange(rows, dtype=np.float64).reshape(-1, 1) / max(rows - 1, 1)
        x = np.arange(cols, dtype=np.float64).reshape(1, -1) / max(cols - 1, 1)
        return 30.0 + 120.0 * 0.5 * (y + x)
    if kind == "texture":
        # Centred on a mid-grey so the target still stands out at any contrast.
        return 110.0 + 90.0 * _texture(rows, cols, contrast, seed)
    raise ValueError(f"unknown background {kind!r}")


def _draw_bars(img: np.ndarray, tr: float, tc: float,
               target_h: float, target_w: float) -> None:
    """The shipped bar-plus-spur, scaled to an arbitrary target box.

    Its three asymmetries are deliberate and load-bearing (mosse_tracker.cpp:
    1115-1117): different extents in r and c catch a transpose, the one-sided spur
    catches a reflection, and neither is symmetric about the centre so a sign flip
    shows up. Scaling must preserve all three, so every extent below is a fraction
    of the target box rather than an absolute pixel count.
    """
    rows, cols = img.shape
    sh = target_h / _NOMINAL
    sw = target_w / _NOMINAL

    r = np.arange(rows, dtype=np.float64).reshape(-1, 1) - tr
    c = np.arange(cols, dtype=np.float64).reshape(1, -1) - tc

    in_rows = np.abs(r) <= 5.0 * sh
    bar = in_rows & (np.abs(c) <= 2.0 * sw)
    spur = in_rows & (r >= 2.0 * sh) & (c >= 3.0 * sw) & (c <= 8.0 * sw)

    img[spur] = float(SPUR_VALUE)
    img[bar] = float(BAR_VALUE)  # bar last: it wins where they overlap, as in the original


def _draw_blob(img: np.ndarray, tr: float, tc: float,
               target_h: float, target_w: float) -> None:
    """Gaussian blob, generalising the s6 scenario target.

    s6 uses sigma = PATCH_COLS/9 for a target implicitly filling a quarter of the
    patch; anchoring to the target instead gives sigma = target_w/2.25, which
    recovers PATCH_COLS/9 exactly when target_w = PATCH_COLS/4. So s6's shape
    stays reachable and comparable.
    """
    rows, cols = img.shape
    sy = max(target_h / 2.25, 1e-6)
    sx = max(target_w / 2.25, 1e-6)
    r = np.arange(rows, dtype=np.float64).reshape(-1, 1) - tr
    c = np.arange(cols, dtype=np.float64).reshape(1, -1) - tc
    img += 180.0 * np.exp(-((r / sy) ** 2 + (c / sx) ** 2) / 2.0)


def make_frame(
    frame_rows: int,
    frame_cols: int,
    target_h: float,
    target_w: float,
    tr: float,
    tc: float,
    kind: str = "bars",
    background: str = "texture",
    bg_contrast: float = 0.35,
    noise: float = 2.0,
    seed: int = 20260816,
) -> np.ndarray:
    """Build a uint8 frame with a target of the given box size centred at (tr, tc).

    `tr`/`tc` may be fractional — the target lands on a sub-pixel position, which
    is what makes a held-out evaluation exercise resample phase rather than just
    a circular shift.
    """
    if kind not in ("bars", "blob", "flat", "checker"):
        raise ValueError(f"unknown kind {kind!r}")

    img = _background(frame_rows, frame_cols, background, bg_contrast, seed)

    if kind == "bars":
        _draw_bars(img, tr, tc, target_h, target_w)
    elif kind == "blob":
        _draw_blob(img, tr, tc, target_h, target_w)
    elif kind == "checker":
        sq = max(int(round(min(target_h, target_w) / 2.0)), 1)
        y = (np.arange(frame_rows) // sq).reshape(-1, 1)
        x = (np.arange(frame_cols) // sq).reshape(1, -1)
        img = np.where((y + x) & 1, 255.0, 0.0)

    if noise > 0.0:
        img = img + noise * np.random.default_rng(seed + 1).standard_normal(img.shape)

    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def target_box(target_h: float, target_w: float, tr: float, tc: float):
    """(top, left, height, width) of the target's bounding box, for IoU scoring."""
    return (tr - target_h / 2.0, tc - target_w / 2.0, float(target_h), float(target_w))


def _self_check() -> int:
    failures = 0

    def check(what: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {what:<38} {'OK  ' if cond else 'FAIL'}{' — ' + detail if detail else ''}")
        if not cond:
            failures += 1

    print("synth_frame self-check\n")

    # The shipped target, reproduced. inject_target_frame draws an 11x11 bar+spur
    # on a flat BACKGROUND; kind='bars' at 11x11 with background='flat' must match.
    rows, cols, tr, tc = 200, 200, 100, 100
    got = make_frame(rows, cols, 11, 11, tr, tc, kind="bars",
                     background="flat", noise=0.0)
    ref = np.full((rows, cols), BACKGROUND, dtype=np.uint8)
    for r in range(rows):
        dr = r - tr
        if dr < -5 or dr > 5:
            continue
        for c in range(cols):
            dc = c - tc
            v = 0
            if -2 <= dc <= 2:
                v = BAR_VALUE
            elif dr >= 2 and 3 <= dc <= 8:
                v = SPUR_VALUE
            if v:
                ref[r, c] = v
    check("bars@11x11 == inject_target_frame", np.array_equal(got, ref),
          f"{int(np.count_nonzero(got != ref))} px differ")

    # Scaling must preserve the asymmetries the shape exists for.
    big = make_frame(400, 400, 64, 64, 200, 200, kind="bars",
                     background="flat", noise=0.0)
    ys, xs = np.nonzero(big != BACKGROUND)
    check("scaled target grows with the box",
          (ys.max() - ys.min()) > 50, f"height {ys.max()-ys.min()+1} px")
    check("scaled target stays asymmetric in c",
          abs((xs.max() - 200) - (200 - xs.min())) > 5,
          f"right {xs.max()-200}, left {200-xs.min()}")
    check("spur survives scaling", np.count_nonzero(big == SPUR_VALUE) > 0,
          f"{int(np.count_nonzero(big == SPUR_VALUE))} px")

    # At HARNESS_NOISE an interpolation error must not be able to hide: bilinear
    # between two EQUAL neighbours returns the same value for any fraction, so
    # every flat neighbour pair is a blind spot for the harness.
    tex = make_frame(256, 256, 32, 32, 128, 128, kind="bars",
                     background="texture", noise=HARNESS_NOISE)
    gx = np.abs(np.diff(tex.astype(np.int64), axis=1))
    gy = np.abs(np.diff(tex.astype(np.int64), axis=0))
    flat = max(np.count_nonzero(gx == 0) / gx.size,
               np.count_nonzero(gy == 0) / gy.size)
    check("harness texture: few blind pairs", flat < 0.05,
          f"{100.0*flat:.2f}% flat neighbours at noise={HARNESS_NOISE}")

    # At SWEEP_NOISE the BACKGROUND must stay smooth enough that different
    # resample ratios compare padding, not aliasing of the test signal. Measured
    # on the background alone: the target is a hard-edged shape and is broadband
    # by construction, so including it would test the wrong claim. Frequencies
    # are signed, because a real image's low-frequency energy sits near index 0
    # AND near index n-1.
    n = 256
    bg = _background(n, n, "texture", 0.35, seed=20260816)
    bg = bg + SWEEP_NOISE * np.random.default_rng(20260817).standard_normal((n, n))
    spec = np.abs(np.fft.fft2(bg - bg.mean())) ** 2
    k = np.minimum(np.arange(n), n - np.arange(n))
    lowband = (k[:, None] < n // 8) & (k[None, :] < n // 8)
    frac = spec[lowband].sum() / spec.sum()
    check("sweep background is band-limited", frac > 0.80,
          f"{100.0*frac:.1f}% of energy below n/8")

    # Determinism, and independence from frame size at fixed seed.
    a = make_frame(256, 256, 32, 32, 128, 128, seed=7)
    b = make_frame(256, 256, 32, 32, 128, 128, seed=7)
    check("deterministic at fixed seed", np.array_equal(a, b))

    print(f"\n  OVERALL: {'FAIL' if failures else 'PASS'} ({failures} failure"
          f"{'' if failures == 1 else 's'})\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
