#!/usr/bin/env python3
"""
scripts/rgb_vs_gray_loop.py

Closed-loop MOSSE on real video, gray vs RGB vs a colour-free control.

rgb_vs_gray_holdout.py measures RESPONSE QUALITY with the filter frozen. It
found RGB worth 1.63x on Bolme PSR and nothing measurable on localisation, and
it could not tell whether that was RGB's fault or the protocol's: a filter
trained on one frame and evaluated ten frames later is mostly just failing.

This closes the loop. The filter updates every frame at MOSSE_ETA, the position
is the tracker's own, and errors compound — which is the only regime in which
"does RGB track better" is a meaningful question. Same three arms, same exact
integer Stage A / int8 / conv datapath, same float FFT (see the holdout's
docstring for why the FFT is float).

Faithful to the hardware in the ways that matter
------------------------------------------------
  * THE FILTER TRAINS AGAINST G CENTRED AT THE MEASURED DISPLACEMENT, not at
    (0,0). This is the defect fixed 2026-08-20 and it is invisible to any
    single-update test; a closed loop is the only thing that sees it. See the
    training-target entry in CLAUDE.md.
  * Stage B1 subtracts the PREVIOUS frame's per-channel mean.
  * The PSR gate holds position AND skips the filter update, both.
  * Position updates by peak * roi/patch — the resample ratio, not 1:1.

Deliberately NOT modelled: the DSST scale filter. The box size is held at its
initial value (equivalent to SCALE_N=1). That caps IoU on any sequence whose
target changes size — car1's width goes 122 -> 83 px — but it caps all three
arms identically, and adding a second estimator would confound the comparison.
--oracle-scale takes the box size from ground truth instead, which isolates
localisation from scale; run both and read the pair.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_loop.py
  ... --oracle-scale --frames 400
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('GEN_PATCH_ROWS', '128')
os.environ.setdefault('GEN_PATCH_COLS', '128')

import gen_filter_golden as FG                      # noqa: E402
from rgb_vs_gray_holdout import (                   # noqa: E402
    LUM, N_OUT, PADDING, SIGMA, EPS_REL, R, C,
    load_gt, load_frame_rgb, to_luma, set_sequence,
    stage_a_gray, stage_a_rgb, folded_weights, quantize, conv_features_float,
    conv_features, metrics, wrap,
)

ETA = 0.125            # MOSSE_ETA
PSR_GATE_MIN = 7.0     # Bolme 3.5


# ---------------------------------------------------------------------------
# POOLED FEATURES
#
# The feature bank is 16 channels of 3x3 conv at stride 1, so one feature pixel
# sees a 3-pixel neighbourhood -- on a ~73 px target that is an edge detector
# with no tolerance for the target changing shape underneath it. HOG, which the
# published classical baselines use, is gradient histograms over 4x4 or 8x8
# CELLS: the same fine measurement, aggregated over a neighbourhood, which is
# what makes it survive deformation. `runs/vot/detector_gain.md` says that is
# where this tracker's remaining deficit is: on targets that genuinely
# translate the detector already recovers 93% of the annotated motion, and 69%
# of on-target frames are targets whose appearance the filter cannot follow by
# translation at all.
#
# Three arms, and the third one is the control that makes the other two
# readable:
#
#   pool<N>  average-pool the conv output N x N.  This is the HOG-style
#            aggregation: full-resolution conv, then cell averaging.  On
#            hardware it is a 2x2 average at the end of conv2d, which leaves
#            conv2d's cost unchanged and makes every downstream stage (FFT,
#            cmul, IFFT, the whole APU filter tail) 4x cheaper.
#   dec<N>   SUBSAMPLE the conv output N x N -- take every Nth pixel, no
#            averaging.  Identical resolution, identical bin size, identical
#            sigma in frame pixels, and NO aggregation.
#   (none)   the shipped 128x128.
#
# dec<N> vs pool<N> isolates POOLING.  dec<N> vs the baseline isolates the
# RESOLUTION CHANGE.  Without the dec arm a pool win is unattributable: coarser
# bins alone change the sub-bin quantisation, the effective sigma in frame
# pixels and the PSR sidelobe statistics, and any of those could carry the
# result.  This project has twice accepted a mechanism it did not separate.
# ---------------------------------------------------------------------------

def split_mask(name):
    """'rgb-mask50' -> ('rgb', 0.50).  'rgb' -> ('rgb', None).

    The number is the PLATEAU width as a percent of the patch; 50 is the target
    box at TARGET_PADDING=2.
    """
    m = re.match(r'^(.*?)-mask(\d+)$', name)
    return (m.group(1), int(m.group(2)) / 100.0) if m else (name, None)


def split_warp(name):
    """'rgb-warp8' -> ('rgb', 8).  'rgb' -> ('rgb', 1).

    A SUFFIX, not a global flag, so one invocation runs the arm and its control
    against the same frames -- the shape every other arm in this file uses.
    """
    m = re.match(r'^(.*?)-warp(\d+)$', name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


def parse_arm(name):
    """'rgb-pool2' -> ('rgb', 2, 'avg').  'gray' -> ('gray', 1, 'avg')."""
    m = re.match(r'^(.*?)-(relupool|relublur|relu|pool|blur|dec)(\d+)?$', name)
    if not m:
        return name, 1, 'avg'
    kind = m.group(2)
    n = int(m.group(3)) if m.group(3) else 1
    return m.group(1), n, {'pool': 'avg', 'dec': 'dec', 'blur': 'blur',
                           'relupool': 'reluavg', 'relu': 'reluavg',
                           'relublur': 'relublur'}[kind]


def pool_features(ft, n, mode):
    """[16, R, C] -> [16, R//n, C//n]."""
    if n == 1:
        return ft
    # STRIDE-1 AGGREGATION. pool/dec both shrink the map, so they change the
    # receptive field AND the resolution, and the first sweep showed the
    # resolution loss costs more than the aggregation gains (gray 0.2533 ->
    # dec2 0.2390 is resolution alone). `blur` is the arm that separates them:
    # an n x n box average at stride 1, so the map stays 128x128, the bin size,
    # sigma-in-frame-pixels and sub-bin quantisation are all UNCHANGED, and the
    # only thing that moves is how far a feature response is smeared. That is
    # the deformation-tolerance hypothesis on its own.
    #
    # Circular, to match the circular correlation the FFT performs.
    if mode in ('blur', 'relublur'):
        acc = np.zeros_like(ft)
        for dr in range(n):
            for dc in range(n):
                acc += np.roll(np.roll(ft, -dr, axis=1), -dc, axis=2)
        return acc / (n * n)
    ch, r, c = ft.shape
    if mode == 'dec':
        return ft[:, ::n, ::n]
    return ft.reshape(ch, r // n, n, c // n, n).mean(axis=(2, 4))


def arm_doc(name):
    base, n, mode = parse_arm(name)
    return (f"{base}, "
            + ("no pooling" if n == 1 and mode == 'avg' else
               f"{'ReLU + ' if mode.startswith('relu') else ''}"
               + ('ReLU only' if n == 1 else
                  ('subsample' if mode == 'dec' else
                   'box blur (stride 1)' if mode.endswith('blur') else 'average')
                  + f" {n}x{n} -> "
                  + (f"{R}x{C}" if mode.endswith('blur') else f"{R//n}x{C//n}"))))


def metrics_rc(resp, rows, cols, excl):
    """metrics() from the holdout, but at an arbitrary map size.

    The holdout's version closes over the module-level R, C and a hardcoded
    11x11 sidelobe exclusion. Both are wrong for a pooled map: the exclusion
    must stay the same number of SIGMAS, not the same number of bins, or a
    pooled arm's PSR is computed with a mainlobe left inside the sidelobe set
    and is not comparable to the baseline's. `excl` is that radius in bins.
    """
    idx = np.unravel_index(np.argmax(np.abs(resp)), resp.shape)
    peak = resp[idx]
    rr = np.minimum((np.arange(rows) - idx[0]) % rows, (idx[0] - np.arange(rows)) % rows)
    cc = np.minimum((np.arange(cols) - idx[1]) % cols, (idx[1] - np.arange(cols)) % cols)
    sl = resp[~((rr[:, None] <= excl) & (cc[None, :] <= excl))]
    if sl.size == 0 or sl.std() == 0:
        return idx, peak, 0.0, 0.0
    return idx, peak, (peak - sl.mean()) / sl.std(), abs(peak) / max(np.abs(sl).max(), 1e-9)


def box_iou(a, b):
    """a, b = (row, col, h, w) centre-form. Axis-aligned overlap."""
    ar0, ac0, ar1, ac1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    br0, bc0, br1, bc1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ih = max(0.0, min(ar1, br1) - max(ar0, br0))
    iw = max(0.0, min(ac1, bc1) - max(ac0, bc0))
    inter = ih * iw
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# REGULARISED INITIALISATION -- Bolme 3.4, the N>1 case
#
# filter_init() is filter_update() at eta = 1 against a zeroed state: the exact
# closed form for ONE training image. Bolme regularises it with eight random
# affine perturbations of the first frame, and `mosse_tracker.cpp:31` has
# carried that as a TODO since the tracker was written.
#
# It matters here far more than in a single-start benchmark. The board runs the
# ANCHORED multi-start protocol -- 419 runs per arm, so 419 inits -- and
# `scripts/vot_init_anatomy.py` measures 61 of the 373 losing runs (16%) failing
# within 10 frames of init, already at median IoU 0.571 and PSR 7.35 one frame
# after filter_init() against 0.915 and 36.73 for every other run. Those runs do
# not drift off the target; they never acquire it.
#
# THE WARP SET IS RESTRICTED TO WHAT roi_crop CAN ACTUALLY PRODUCE, and that is
# the load-bearing constraint on this whole experiment. roi_crop resamples an
# AXIS-ALIGNED ROI with runtime geometry, so translation and scale are free (and
# a scale warp is the first thing that would ever exercise its bilinear
# interpolator on hardware) -- but there is NO rotation. An offline arm that
# jittered rotation would be a result about hardware that does not exist. Bolme
# uses rotation; this is deliberately a weaker perturbation set than his, and if
# the win turns out to live in rotation specifically, this arm cannot show it and
# the conclusion is "not implementable", not "does not work".
#
# sigma stays FIXED IN BINS across warps, because the shipped tracker's sigma is
# a constant (MOSSE_SIGMA=2.0, SIGMA_FROM_TARGET=0) and does not track box size.
# Scaling it per warp would be a second change riding along inside the first.
def warp_set(n, shift_frac, scale_frac, aspect_frac=0.0, rot_deg=0.0,
             seed=1234):
    """n-1 perturbations plus the identity, as (shift_r, shift_c, sr, sc, ang).

    Shifts are FRACTIONS OF THE ROI applied to the crop centre; sr/sc multiply
    the ROI extent per axis; ang is a rotation in degrees about the crop centre.
    Deterministic from a fixed seed so two arms see the same warps and a rerun
    reproduces byte-for-byte.

    THE ASPECT AND ROTATION DRAWS ARE SKIPPED WHEN THEIR MAGNITUDE IS ZERO, and
    that is not a micro-optimisation: drawing them unconditionally would shift
    the RNG sequence and silently change the already-measured shift+scale arm,
    making the 62-sequence result on record irreproducible.

    Cost of each axis on the BOARD, which is why they are separate knobs:
      shift, isotropic scale  -- free, roi_crop's runtime AXI-Lite geometry
      aspect (sr != sc)       -- free, roi_h and roi_w are separate registers
      rotation                -- NOT free: roi_crop resamples an axis-aligned
                                 ROI. Cheapest hardware route is the host
                                 pre-rotating the ROI region into frame_bo
                                 before the launch (no PL change, ~2 ms/warp at
                                 init only); doing it inside roi_crop is a
                                 kernel rebuild, re-package and reflash.
    """
    out = [(0.0, 0.0, 1.0, 1.0, 0.0)]
    if n <= 1:
        return out
    rng = np.random.RandomState(seed)
    for _ in range(n - 1):
        dr = float(rng.uniform(-shift_frac, shift_frac))
        dc = float(rng.uniform(-shift_frac, shift_frac))
        sc = float(np.exp(rng.uniform(-scale_frac, scale_frac)))
        sr = sc
        if aspect_frac > 0.0:
            # Anisotropic: one axis stretched against the other, area held.
            a = float(np.exp(rng.uniform(-aspect_frac, aspect_frac)))
            sr, sc = sc * a, sc / a
        ang = float(rng.uniform(-rot_deg, rot_deg)) if rot_deg > 0.0 else 0.0
        out.append((dr, dc, sr, sc, ang))
    return out


def rotate_window(planes, cy, cx, ang_deg, r0, c0, h, w):
    """Rotate the frame about (cy, cx) and return the [r0:r0+h, c0:c0+w] window.

    THIS IS THE HOST PRE-ROTATION ROUTE, modelled exactly: the frame region is
    rotated and then cropped AXIS-ALIGNED, which is what roi_crop would do to a
    frame_bo the host had already rotated. It is deliberately NOT a rotated
    sampler inside the crop -- that is the other, expensive route, and modelling
    it here would measure hardware that does not exist.

    Bilinear with border clamping, matching roi_crop's own tap and clamp rule.
    """
    t = np.deg2rad(ang_deg)
    ct, stt = np.cos(t), np.sin(t)
    yy = np.arange(r0, r0 + h, dtype=np.float64)[:, None]
    xx = np.arange(c0, c0 + w, dtype=np.float64)[None, :]
    dy, dx = yy - cy, xx - cx
    sy = cy + ct * dy + stt * dx
    sx = cx - stt * dy + ct * dx
    H, W = planes.shape[-2], planes.shape[-1]
    y0 = np.clip(np.floor(sy).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(sx).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    fy = np.clip(sy - np.floor(sy), 0, 1)
    fx = np.clip(sx - np.floor(sx), 0, 1)
    p = planes.astype(np.float64)
    top = p[:, y0, x0] * (1 - fx) + p[:, y0, x1] * fx
    bot = p[:, y1, x0] * (1 - fx) + p[:, y1, x1] * fx
    return np.clip(np.round(top * (1 - fy) + bot * fy), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# SPATIAL RELIABILITY -- a mask on the FILTER, as a one-shot projection
#
# CSR-DCF's contribution, and the literature's highest-priced item on the
# robustness list: their ablation puts "removing the mask entirely -- reduces the
# tracker to a standard DCF with a large receptive field" at >50% EAO, and that
# sentence describes this design. At TARGET_PADDING=2 the target is 27% of the
# ROI area and nothing masks the rest, so the filter is free to spend its energy
# on background. MEASURED at init: a centred 64x64 box -- exactly the target box
# at padding 2 -- holds only 51.6% (car1) / 54.9% (tiger) of sum|h|^2. Half the
# filter is background.
#
# The projection is h <- m*h, i.e. constrain the filter's support to the object.
# It is applied to H at DETECTION time and A/B are untouched, so it is a one-shot
# projection, NOT CSR-DCF's ADMM (theirs iterates, with a mask estimated per
# frame from colour segmentation). If the projection wins, the estimated mask is
# the follow-up, not the entry point.
#
# WHERE THE MASK GOES IS MEASURED, NOT ASSUMED. resp[0,0] is "no displacement",
# so the RESPONSE origin is index 0 -- but h's energy is centred at the PATCH
# CENTRE (peak at (64,64); a corner-wrapped 64x64 box holds only 8-12%). Masking
# at the origin would delete the filter and read as "masking hurts".
#
# ON THE BOARD this is host-only and needs NO host FFT: the mask is SEPARABLE and
# a raised cosine has a compact DFT, so h <- m*h is a sparse 2-D convolution on
# the published H -- the same identity Stage B2 is built on. That is why the
# window below is separable even though numpy would not care.
def spatial_mask(rows, cols, plateau_frac, taper_frac):
    """Separable raised-cosine mask, centred at the PATCH CENTRE.

    plateau_frac: width of the flat top as a fraction of the patch. 0.5 is
                  exactly the target box at TARGET_PADDING=2.
    taper_frac:   width of the cosine roll-off, same units. 0 gives a hard box,
                  whose DFT is a sinc and therefore NOT sparse -- it is available
                  for offline comparison but is not the board-implementable one.
    """
    def axis(n):
        x = np.abs(np.arange(n) - (n - 1) / 2.0) / n      # 0 at centre, 0.5 at edge
        p, t = plateau_frac / 2.0, taper_frac / 2.0
        if t <= 0:
            return (x <= p).astype(np.float64)
        r = np.clip((x - p) / t, 0.0, 1.0)
        return np.cos(0.5 * np.pi * r) ** 2
    return np.outer(axis(rows), axis(cols))


def make_patch(arm, planes, lum, roi_row, roi_col, roi_h, roi_w):
    arm, _, _ = parse_arm(arm)
    if arm in ('gray', 'gray-float'):
        return stage_a_gray(lum, roi_row, roi_col, roi_h, roi_w)[None]
    if arm in ('rgb', 'rgb-float'):
        return stage_a_rgb(planes, roi_row, roi_col, roi_h, roi_w)
    return stage_a_rgb(np.stack([lum] * 3), roi_row, roi_col, roi_h, roi_w)


def run_arm(arm, wq, bias, shift, gt, n_frames, oracle_scale, verbose,
            detect_iters=1, detect_gain=1.0, float_conv=None, sigma=SIGMA,
            eta=ETA, psr_min=PSR_GATE_MIN, n_warps=1,
            warp_shift=0.05, warp_scale=0.05, warp_mutant='none',
            warp_aspect=0.0, warp_rot=0.0, eps_rel=EPS_REL,
            mask_plateau=None, mask_taper=0.25):
    # float_conv = (w_float, b_fold) runs the UNQUANTIZED conv instead of the
    # int8 one, everything else identical. See conv_features_float's docstring:
    # this arm exists to answer whether quantization causes the tracker's poor
    # robustness, and it is the second half of that question -- the first half
    # (the cint16/Q1.15 correlation pipeline) is already answered by this whole
    # model being float64 downstream of the features.
    """One full pass over the sequence. Returns a per-frame record."""
    # Feature-map geometry. SIGMA stays in BINS across arms, so a pooled arm's
    # target is the same shape on its own map -- the "same tracker at a lower
    # resolution" reading, and the one hardware would take. Its sigma in FRAME
    # PIXELS therefore doubles at pool2, which is exactly why the dec2 control
    # exists: dec2 carries the identical geometry change with no pooling.
    _base, pool, pool_mode = parse_arm(arm)
    fr, fc = ((R, C) if pool_mode in ('blur', 'relublur')
              else (R // pool, C // pool))
    excl = 5                       # sidelobe exclusion, BINS (11x11 at pool1)
    row, col, bh, bw = gt[0]
    A = B = None
    mean_prev = None
    # `step` records the measurement itself, per frame, in BINS and in frame px,
    # because the failure under investigation is a detector that reports (0,0)
    # while the target moves several bins -- invisible in IoU or PSR, which both
    # look healthy while it happens.
    rec = {'iou': [], 'cerr': [], 'psr': [], 'holds': 0, 'lost_at': None,
           'step': [], 'resp00': []}
    # Built once: the mask is fixed in PATCH coordinates, so it does not move
    # with the box. Its fixedness is the whole reason it is host-only and free.
    mask = (None if mask_plateau is None
            else spatial_mask(fr, fc, mask_plateau, mask_taper))

    for f in range(1, n_frames + 1):
        if oracle_scale:
            bh, bw = gt[f - 1][2], gt[f - 1][3]
        roi_h, roi_w = int(round(bh * PADDING)), int(round(bw * PADDING))
        roi_row = int(round(row - roi_h / 2.0))
        roi_col = int(round(col - roi_w / 2.0))

        planes = load_frame_rgb(f)
        lum = to_luma(planes)

        def crop_features(rr, rc, mp, rh=None, rw=None, pl=None, lm=None):
            # rh/rw default to this frame's ROI; the warped init passes its own,
            # which is the whole point of roi_crop's geometry being runtime.
            patch = make_patch(arm,
                               planes if pl is None else pl,
                               lum if lm is None else lm, rr, rc,
                               roi_h if rh is None else rh,
                               roi_w if rw is None else rw)
            if float_conv is not None:
                ft, om = conv_features_float(patch, float_conv[0], float_conv[1],
                                             mean_prev=mp)
            else:
                ft, om = conv_features(patch, wq, bias, shift, mean_prev=mp,
                                       relu=pool_mode in ('reluavg', 'relublur'))
            ft = pool_features(ft.astype(np.float64), pool, pool_mode)
            return np.fft.fft2(ft, axes=(1, 2)), om

        # The warp crops must see the SAME Stage B1 state the canonical crop
        # saw, not the one it produced -- they are alternative views of the same
        # frame, not later frames.
        mp_in = mean_prev
        F, own_mean = crop_features(roi_row, roi_col, mean_prev)
        mean_prev = own_mean

        if A is None:
            # filter_init. Frame 1's crop really is centred on the target, so G
            # is centred here and only here.
            Gt = FG.gaussian_target_spectrum(fr, fc, sigma, 0, 0)
            A = np.conj(Gt)[None] * F
            B = np.sum(np.abs(F)**2, axis=0)
            if n_warps > 1:
                # Bolme's N-sample closed form: A = SUM conj(G_i) F_i,
                # B = SUM |F_i|^2, accumulated on top of the identity warp
                # already formed above -- so the n_warps == 1 path above is
                # TEXTUALLY the shipped one and stays bit-identical.
                for sr, sc, s_r, s_c, ang in warp_set(
                        n_warps, warp_shift, warp_scale,
                        warp_aspect, warp_rot)[1:]:
                    wh = int(round(roi_h * s_r))
                    ww = int(round(roi_w * s_c))
                    # Crop centre moves by (sr, sc) ROI-fractions, so the target
                    # sits at MINUS that offset inside the warped patch, and G
                    # must be centred there -- the same sign rule as the
                    # measured-displacement training target (CLAUDE.md).
                    dsr, dsc = sr * roi_h, sc * roi_w
                    wr = int(round(row + dsr - wh / 2.0))
                    wc = int(round(col + dsc - ww / 2.0))
                    if ang != 0.0:
                        # Rotate the frame about the TARGET centre, then crop
                        # axis-aligned -- the host pre-rotation route. Only the
                        # crop window is produced, which is all frame_bo would
                        # need rewritten.
                        pw = rotate_window(planes, row, col, ang, wr, wc, wh, ww)
                        Fw, _ = crop_features(0, 0, mp_in, wh, ww,
                                              pl=pw, lm=to_luma(pw))
                    else:
                        Fw, _ = crop_features(wr, wc, mp_in, wh, ww)
                    # MUTANTS. A warped init that helps must be able to HURT
                    # when its geometry is wrong, or the arm is measuring the
                    # extra samples and not the warps. 'gsign' centres G at the
                    # crop offset instead of minus it -- the one error that is
                    # invisible by inspection and looks like ordinary noise.
                    # 'noshift' trains every warp as if it were centred, which
                    # is what forgetting the correction entirely does.
                    sgn = 1.0 if warp_mutant == 'gsign' else -1.0
                    if warp_mutant == 'noshift':
                        sgn = 0.0
                    Gw = FG.gaussian_target_spectrum(fr, fc, sigma,
                                                     sgn * dsr * fr / wh,
                                                     sgn * dsc * fc / ww)
                    A = A + np.conj(Gw)[None] * Fw
                    B = B + np.sum(np.abs(Fw)**2, axis=0)
                # Normalise back to ONE sample's scale. H = A/(B + eps*mean(B))
                # is scale-invariant, so this does not change frame 1's filter --
                # it keeps the first filter_update's eta weighting the new frame
                # against a state of the same magnitude, which an unnormalised
                # sum would not.
                A /= n_warps
                B /= n_warps
            rec['iou'].append(box_iou((row, col, bh, bw), gt[f - 1]))
            rec['cerr'].append(float(np.hypot(row - gt[f-1][0], col - gt[f-1][1])))
            rec['psr'].append(float('nan'))
            continue

        # detect
        energy = np.sum(np.abs(F)**2, axis=(1, 2))
        chscale = np.where(energy > 0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
        H = A * chscale[:, None, None] / (B + eps_rel * B.mean())[None]
        if mask is not None:
            # h <- m*h. EXACT FFTs here on purpose: offline answers "does the
            # projection help at all", and the 9-bin sparse-spectrum form is a
            # BOARD implementation detail whose approximation error is a separate
            # question. Testing the approximation before the idea would confound
            # the two.
            H = np.fft.fft2(np.fft.ifft2(H, axes=(1, 2)) * mask[None], axes=(1, 2))
        resp = np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))
        idx, peak, bolme, _ratio = metrics_rc(resp, fr, fc, excl)
        dr, dc = wrap(idx[0], fr), wrap(idx[1], fc)

        # ITERATED LOCALISATION. A windowed correlation systematically reports
        # LESS than the true displacement: the patch is Hann-weighted, so a
        # target that has moved far from the ROI centre is attenuated and the
        # peak is pulled back toward zero. On `tiger` the report is roughly half
        # of what was needed, every frame, which accumulates into a standing lag
        # rather than a loss. Re-cropping at the updated position and detecting
        # again attacks exactly that: each pass shrinks the residual, because the
        # second crop has the target much closer to the centre.
        #
        # detect_iters=1 is the shipped behaviour, evaluated identically.
        it_dr, it_dc = dr, dc
        for _ in range(detect_iters - 1):
            if not ((peak > 0) and (bolme >= psr_min)) or (it_dr == 0 and it_dc == 0):
                break
            row += it_dr * roi_h / fr
            col += it_dc * roi_w / fc
            rr = int(round(row - roi_h / 2.0))
            rc = int(round(col - roi_w / 2.0))
            F, _ = crop_features(rr, rc, mean_prev)
            en2 = np.sum(np.abs(F)**2, axis=(1, 2))
            ch2 = np.where(en2 > 0, 1.0 / np.sqrt(np.maximum(en2, 1e-300)), 0.0)
            resp = np.real(np.fft.ifft2(np.sum(F * np.conj(A * ch2[:, None, None]
                                               / (B + eps_rel * B.mean())[None]), axis=0)))
            idx, peak, bolme, _ratio = metrics_rc(resp, fr, fc, excl)
            it_dr, it_dc = wrap(idx[0], fr), wrap(idx[1], fc)
        dr, dc = it_dr, it_dc

        # resp00/peak: how much of the peak sits at ZERO SHIFT. The discriminator
        # for origin lock -- a value near 1 means the response is peaked at the
        # origin no matter where the target went.
        rec['resp00'].append(float(abs(resp[0, 0]) / abs(peak)) if peak else 0.0)
        # The displacement the detector REPORTED, and the one it NEEDED to report
        # to land on the target -- (groundtruth now) minus (where the tracker was
        # when it cropped). Comparing the report against the TARGET'S motion
        # instead is a different and much weaker question: it is only the same
        # number while the tracker is exactly on target, and it made a healthy
        # car1 look like a 20%-correct detector.
        need_r = (gt[f-1][0] - row) / (roi_h / fr)
        need_c = (gt[f-1][1] - col) / (roi_w / fc)
        rec['step'].append((dr, dc, roi_h / R, roi_w / C, need_r, need_c))

        gate_ok = (peak > 0) and (bolme >= psr_min)
        if gate_ok:
            # Patch bins -> frame pixels by the resample ratio.
            row += dr * detect_gain * roi_h / fr
            col += dc * detect_gain * roi_w / fc
            # THE TRAINING TARGET IS SHIFTED BY THE MEASURED DISPLACEMENT.
            # g_F_all was cropped at the PRE-update position, where the object
            # sits at (dr,dc); training against a centred G teaches "target at
            # (dr,dc) peaks at 0" and compounds at ETA until zero-shift wins.
            Gt = FG.gaussian_target_spectrum(fr, fc, sigma, dr, dc)
            A, B = FG.filter_update(A, B, F, Gt, eta)
        else:
            rec['holds'] += 1        # hold position AND skip the update, both

        iou = box_iou((row, col, bh, bw), gt[f - 1])
        rec['iou'].append(iou)
        rec['cerr'].append(float(np.hypot(row - gt[f-1][0], col - gt[f-1][1])))
        rec['psr'].append(bolme)
        if verbose and f % 50 == 0:
            print(f"    {arm:<8} f{f:<4} IoU {iou:.3f}  PSR {bolme:6.2f}  "
                  f"cerr {rec['cerr'][-1]:6.1f}px", flush=True)

    # First frame after which IoU never recovers above 0.5 — "permanently lost".
    iou = np.array(rec['iou'])
    below = iou < 0.5
    if below.any():
        k = len(iou)
        while k > 0 and below[k - 1]:
            k -= 1
        if k < len(iou):
            rec['lost_at'] = k + 1
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=0, help='0 = whole sequence')
    ap.add_argument('--arms', nargs='+', default=['gray', 'rgb', 'rgb-lum'])
    ap.add_argument('--oracle-scale', action='store_true',
                    help='take box size from ground truth (isolates localisation)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--psr-min', type=float, default=PSR_GATE_MIN,
                    help='PSR_GATE_MIN (default %g). The best RECORDED hardware\n'
                         'arm is 5.0, not 7.0, and it interacts with anything\n'
                         'that moves the PSR scale -- a coarser feature map\n'
                         'halves PSR, so a fixed threshold is not a fixed\n'
                         'gate.' % PSR_GATE_MIN)
    ap.add_argument('--eta', type=float, default=ETA,
                    help='MOSSE_ETA (default %g). 0.05 is the POSITIVE\n'
                         'CONTROL for this bench: it is the one change\n'
                         'already known to move it up, 0.2533 -> 0.2599\n'
                         'frame-weighted over the 8-sequence set.' % ETA)
    ap.add_argument('--sigma', type=float, default=None,
                    help='target sigma in BINS (default %g for every arm). '
                         'Pass 1.0 with a pool2 arm to hold sigma constant in '
                         'FRAME PIXELS instead -- the other half of the '
                         'geometry confound the dec arm controls for.' % SIGMA)
    ap.add_argument('--warp-shift', type=float, default=0.05,
                    help='init warp translation jitter, as a fraction of the\n'
                         'ROI (default 0.05). Used only by a -warp<N> arm.')
    ap.add_argument('--warp-scale', type=float, default=0.05,
                    help='init warp log-scale jitter (default 0.05, i.e.\n'
                         '+-5%% of the ROI extent). roi_crop can deliver this;\n'
                         'it CANNOT deliver rotation, so there is no knob for\n'
                         'one -- see warp_set().')
    ap.add_argument('--mask-taper', type=float, default=0.25,
                    help='cosine roll-off width of the -mask<N> spatial mask, as\n'
                         'a fraction of the patch (default 0.25). 0 is a HARD\n'
                         'box, whose DFT is a sinc and therefore not sparse --\n'
                         'offline-only, not board-implementable.')
    ap.add_argument('--eps-rel', type=float, default=EPS_REL,
                    help='filter regularizer, RELATIVE to mean(B) (default\n'
                         '%g). BOLME 3.3 PRESENTS REGULARIZATION AND INIT\n'
                         'PERTURBATIONS AS ALTERNATIVE CURES FOR THE SAME\n'
                         'defect -- low-energy denominator bins -- and his\n'
                         'Figure 3 (the perturbation curve) is captioned\n'
                         '"without regularization". Lower this to reach his\n'
                         'regime and the warp arms become testable against\n'
                         'their actual mechanism.' % EPS_REL)
    ap.add_argument('--warp-aspect', type=float, default=0.0,
                    help='init warp ANISOTROPIC log-scale jitter: one ROI axis\n'
                         'stretched against the other, area held. FREE on the\n'
                         'board -- roi_h and roi_w are separate registers.')
    ap.add_argument('--warp-rot', type=float, default=0.0,
                    help='init warp rotation, DEGREES. Modelled as the host\n'
                         'pre-rotating the ROI region into frame_bo; roi_crop\n'
                         'itself resamples an axis-aligned ROI and cannot\n'
                         'rotate.')
    ap.add_argument('--warp-mutant', default='none',
                    choices=('none', 'gsign', 'noshift'),
                    help='deliberately break the warped init so the arm is\n'
                         'known to be able to fail: gsign flips the sign of\n'
                         "the G centring, noshift drops it. Non-'none'\n"
                         'invalidates the run as a result.')
    ap.add_argument('--json', default=None,
                    help='append this run to a JSON file keyed\n'
                         '"<sequence>|<arm>" -> {"iou": [...]}, the input\n'
                         'format scripts/vot_ar_offline.py reads. Merges into\n'
                         'an existing file so a shell loop over sequences\n'
                         'builds one scorable set.')
    ap.add_argument('--sequence', default=None,
                    help='sequence name; stb2022 under $VOT_ROOT is preferred, '
                         'then test-sequences/. Default: the built-in car1.')
    args = ap.parse_args()

    navail = set_sequence(args.sequence)
    gt = load_gt()
    n = min(navail, len(gt)) if args.frames == 0 else min(args.frames, navail, len(gt))
    w_rgb, b_fold = folded_weights()
    w_gray = (w_rgb * LUM[None, :, None, None]).sum(axis=1, keepdims=True)
    W = {'gray': quantize(w_gray, b_fold),
         'rgb': quantize(w_rgb, b_fold),
         'rgb-lum': quantize(w_rgb, b_fold),
         'gray-float': quantize(w_gray, b_fold),   # unused, keeps the call shape
         'rgb-float': quantize(w_rgb, b_fold)}
    # The unquantized counterparts. Same folded BN weights, same bias, no int8
    # grid, no out_shift, no int16 clips, no integer Hann.
    FLOATW = {'gray-float': (w_gray, b_fold), 'rgb-float': (w_rgb, b_fold)}

    print(f"{args.sequence or 'car1'}, {n} frames, closed loop: eta {ETA}, padding {PADDING}, "
          f"sigma {SIGMA}, PSR gate {args.psr_min}")
    print(f"scale: {'ORACLE (from ground truth)' if args.oracle_scale else 'HELD FIXED (SCALE_N=1 equivalent)'}")
    if args.warp_mutant != 'none':
        print(f"*** WARP MUTANT '{args.warp_mutant}' ACTIVE -- this run is a "
              f"negative control, not a result ***")
    print()

    out = {}
    for a in args.arms:
        print(f"  running {a} ...", flush=True)
        stem, n_warps = split_warp(a)
        stem, mask_plateau = split_mask(stem)
        base, pool, mode = parse_arm(stem)
        if base not in W:
            sys.exit(f"unknown arm '{a}' (base '{base}'): "
                     f"expected one of {sorted(W)} with an optional "
                     f"-pool<N> / -dec<N> / -warp<N> suffix")
        if pool > 1 and (R % pool or C % pool):
            sys.exit(f"{a}: pool {pool} does not divide the {R}x{C} patch")
        out[a] = run_arm(stem, *W[base], gt, n, args.oracle_scale, args.verbose,
                         float_conv=FLOATW.get(base),
                         sigma=args.sigma if args.sigma else SIGMA,
                         eta=args.eta, psr_min=args.psr_min, n_warps=n_warps,
                         warp_shift=args.warp_shift, warp_scale=args.warp_scale,
                         warp_mutant=args.warp_mutant,
                         warp_aspect=args.warp_aspect, warp_rot=args.warp_rot,
                         eps_rel=args.eps_rel, mask_plateau=mask_plateau,
                         mask_taper=args.mask_taper)

    print()
    print(f"{'arm':<13} {'frozen, truth>=1bin':>20} {'frozen, <1bin':>14} "
          f"{'moved':>7} {'resp00/peak':>12}")
    print("-" * 86)
    for a in args.arms:
        st = out[a]['step']
        big = small = moved = 0
        # need_r/need_c COME FROM THE RECORD, they are not recomputed here.
        # This used to unpack 4 fields from a 6-field record (a ValueError since
        # need_* were added) and recompute the truth motion as a groundtruth
        # DIFFERENCE -- which is the weaker question run_arm's own comment warns
        # about: the target's motion only equals the required displacement while
        # the tracker is exactly on target, and reading it that way "made a
        # healthy car1 look like a 20%-correct detector".
        for k, (dr, dc, _br, _bc, tdr, tdc) in enumerate(st, start=2):
            if k >= len(gt):
                break
            if dr == 0 and dc == 0:
                if max(abs(tdr), abs(tdc)) >= 1.0:
                    big += 1
                else:
                    small += 1
            else:
                moved += 1
        tot = max(1, big + small + moved)
        print(f"{a:<13} {100*big/tot:19.1f}% {100*small/tot:13.1f}% "
              f"{100*moved/tot:6.1f}% {np.mean(out[a]['resp00']):12.3f}")

    print()
    print(f"{'arm':<13} {'mean IoU':>9} {'worst':>7} {'>=0.5':>7} "
          f"{'cerr mean':>10} {'cerr max':>9} {'PSR mean':>9} {'holds':>6} {'lost at':>8}")
    print("-" * 86)
    for a in args.arms:
        r = out[a]
        iou = np.array(r['iou']); ce = np.array(r['cerr'])
        psr = np.array(r['psr'])[1:]
        print(f"{a:<13} {iou.mean():9.4f} {iou.min():7.4f} "
              f"{100*np.mean(iou >= 0.5):6.1f}% "
              f"{ce.mean():10.2f} {ce.max():9.2f} {np.nanmean(psr):9.2f} "
              f"{r['holds']:6d} "
              f"{(str(r['lost_at']) if r['lost_at'] else 'never'):>8}")

    if args.json:
        import json
        seq = args.sequence or 'car1'
        blob = {}
        if os.path.exists(args.json):
            with open(args.json) as fh:
                blob = json.load(fh)
        for a in args.arms:
            blob[f"{seq}|{a}"] = {'iou': [float(x) for x in out[a]['iou']]}
        with open(args.json, 'w') as fh:
            json.dump(blob, fh)
        print(f"\nwrote {len(args.arms)} run(s) for '{seq}' to {args.json}")

    if all(a in out for a in ('gray', 'rgb', 'rgb-lum')):
        print()
        print("=" * 82)
        g, r, l = (np.array(out[a]['iou']) for a in ('gray', 'rgb', 'rgb-lum'))
        print(f"mean IoU   gray {g.mean():.4f}   rgb {r.mean():.4f} "
              f"({r.mean()-g.mean():+.4f})   rgb-lum {l.mean():.4f} "
              f"({l.mean()-g.mean():+.4f})")
        print(f"per-frame  rgb better than gray on {100*np.mean(r>g+1e-9):.1f}% of frames, "
              f"worse on {100*np.mean(r<g-1e-9):.1f}%")
        print(f"           rgb better than rgb-lum on {100*np.mean(r>l+1e-9):.1f}%, "
              f"worse on {100*np.mean(r<l-1e-9):.1f}%")
        print()
        print("The control carries RGB's taps, bias and quantization grid and NO")
        print("colour. An rgb win that rgb-lum matches is not a colour result.")


if __name__ == '__main__':
    main()
