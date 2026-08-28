#!/usr/bin/env python3
"""
scripts/rgb_vs_gray_holdout.py

Does RGB input buy discriminability over the shipped luminance collapse?

Companion to check_collapse.py (which answers "what is wrong with the feature
bank") and phase1_sweep.py (which answers "what should the bias/shift budget
be"). This answers the one question neither can: does the collapse cost
TRACKING, and by how much.

Why this holdout and not phase1_sweep's
---------------------------------------
phase1_sweep evaluates on the training patch (or a synthetic shift of it), on
ONE patch (s6). Its own docstring says the absolute numbers are optimistic and
only the ORDERING is meaningful. Here train and eval are DIFFERENT FRAMES of a
real sequence: the eval patch differs by real object motion, real appearance
change, real background, real illumination — everything a circular shift cannot
produce. That makes the absolute ratios comparable to nothing published here,
but it makes the gray-vs-RGB DELTA mean something.

The three arms, and why the third one exists
--------------------------------------------
  gray     what ships: BT.601 luminance collapse, 9-tap int8 kernels
  rgb      27-tap int8 kernels on a jointly-normalised 3-plane patch
  rgb-lum  CONTROL. The 27-tap kernels, fed three IDENTICAL luminance planes.

The control is the point. `rgb` differs from `gray` in three ways at once — more
taps, a different quantization grid, and actual colour. `rgb-lum` has the first
two and none of the third. If rgb beats rgb-lum, colour is the cause. If
rgb == rgb-lum, the win is bookkeeping and RGB is not worth 5 ms/frame.

What is modelled exactly, and what is not
-----------------------------------------
EXACT: Stage A (roi_crop_ref's integer datapath, bit-exact against the kernel),
the int8 weight quantization, the integer conv/shift/clip, Stage B1, and the
separable Hann with both >>15 truncations.

FLOAT: the FFT, the filter, and the response. Deliberately. The shift budget is
calibrated for gray's |F| and RGB's accumulator range is 3x larger, so running
both arms through a gray-tuned fixed-point budget would measure the budget, not
the features. The fixed-point cost of RGB is a separate question and it is
flagged in CLAUDE.md as forcing a re-sweep.

Both arms use the CORRECTED bias (ROI_NORM_Q, not 127) and ReLU OFF, which is
the settled better configuration — so neither the bias bug nor the ReLU
question is a confound here.

Usage
-----
  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/rgb_vs_gray_holdout.py
  ... --step 20 --deltas 1 5 10 20
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('GEN_PATCH_ROWS', '128')
os.environ.setdefault('GEN_PATCH_COLS', '128')

import roi_crop_ref as RC          # noqa: E402
from vot_prepare import reduce_box  # noqa: E402
import gen_aiesim_vectors as G     # noqa: E402

LUM = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
KSIZE, N_OUT = 3, 16
ROI_NORM_Q = RC.ROI_NORM_Q
PADDING = 2.0
SIGMA = 2.0
EPS_REL = 1e-3
R, C = G.PATCH_ROWS, G.PATCH_COLS
HANN = G.HANNING.astype(np.int64)

SEQ = Path("test-sequences/car1")
GT = Path("test-sequences/car1-annotations/groundtruth.txt")
_FRAMES = None          # set by set_sequence(); None = the %08d.jpg pattern above


def set_sequence(name):
    """Point the harness at another sequence, by name.

    TWO LAYOUTS, because the project holds the same data twice:

      test-sequences/<name>/*.jpg          + <name>-annotations/groundtruth.txt
      $VOT_ROOT/workspace/sequences/<name>/color/*.jpg + groundtruth.txt

    The annotation directories in test-sequences/ are named inconsistently
    ("car1-annotations" but "fernando - annotations"), so the match is loose --
    the same rule rgb_vs_gray_vot.discover() uses.

    stb2022 is preferred when both exist: it is the dataset the board runs, its
    groundtruth is the 4-column rectangle format, and the two are NOT the same
    annotations even where the frames match.
    """
    global SEQ, GT, _FRAMES
    if name is None:
        return len(load_gt())

    root = Path(os.environ.get('VOT_ROOT', str(Path.home() / 'vot')))
    cand = root / "workspace" / "sequences" / name
    if (cand / "color").is_dir() and (cand / "groundtruth.txt").is_file():
        SEQ, GT = cand / "color", cand / "groundtruth.txt"
    else:
        local = Path("test-sequences") / name
        if not local.is_dir():
            raise SystemExit(
                f"sequence '{name}' not found in {cand} or {local}")
        ann = None
        key = re.sub(r'[^a-z0-9]', '', name.lower())
        for d in Path("test-sequences").iterdir():
            if d.is_dir() and 'annotation' in d.name.lower() and \
                    re.sub(r'[^a-z0-9]', '', d.name.lower()).startswith(key):
                ann = d
                break
        if ann is None or not (ann / "groundtruth.txt").is_file():
            raise SystemExit(f"no annotations found for '{name}'")
        SEQ, GT = local, ann / "groundtruth.txt"

    _FRAMES = sorted(SEQ.glob("*.jpg"))
    if not _FRAMES:
        raise SystemExit(f"no frames in {SEQ}")
    gt = load_gt()
    if len(_FRAMES) != len(gt):
        # Not fatal on its own, but it means one of the two is not what it
        # claims -- say so rather than silently tracking against the shorter.
        print(f"  WARNING: {name} has {len(_FRAMES)} frames and "
              f"{len(gt)} groundtruth lines")
    return min(len(_FRAMES), len(gt))


# ---------------------------------------------------------------------------
# sequence + ground truth
# ---------------------------------------------------------------------------

def load_gt():
    """Groundtruth -> axis-aligned (row, col, h, w) per frame.

    SINGLE-SOURCED FROM vot_prepare.reduce_box. This function used to carry its
    own polygon-only copy, which is correct for test-sequences/ and silently
    wrong for stb2022's 4-column rectangles -- reading (440,229,198,230) as a
    polygon gives a 1.0 x 242.0 sliver. That copy was harmless only because this
    harness had never been pointed at stb2022; set_sequence() is exactly what
    points it there. Same fix rgb_vs_gray_vot.py already carries.
    """
    return [reduce_box([float(t) for t in line.split(',')])
            for line in GT.read_text().split()]


def load_frame_rgb(idx):
    """1-based frame index -> uint8 [3, H, W].

    Indexes a sorted listing when set_sequence() has run, rather than formatting
    a name: stb2022 numbers from 00000001.jpg but nothing guarantees that for
    every sequence, and an off-by-one here would look like a tracking result.
    """
    from PIL import Image
    path = SEQ / f"{idx:08d}.jpg" if _FRAMES is None else _FRAMES[idx - 1]
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def to_luma(rgb):
    """BT.601, the same convention export_weights.py collapses the kernels with."""
    return np.clip(np.round((rgb.astype(np.float64) * LUM[:, None, None]).sum(0)),
                   0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage A — gray is the shipped path; RGB adds the JOINT normalization
# ---------------------------------------------------------------------------

def stage_a_gray(frame_u8, roi_row, roi_col, roi_h, roi_w):
    return RC.stage_a(frame_u8, roi_row, roi_col, roi_h, roi_w, R, C).astype(np.int64)


def stage_a_rgb(planes_u8, roi_row, roi_col, roi_h, roi_w):
    """Three planes resampled independently, normalised JOINTLY.

    Joint is not a detail. Normalising each plane on its own mean and sigma
    equalises the three and destroys exactly the chromatic information RGB is
    for — silent, and self-defeating. So: one LOG_LUT mean and one inv_q over
    all 3*R*C samples, applied to all three planes.

    DELEGATES to roi_crop_ref, which is now what the PL kernel is tested
    against (make test_roi_crop, ROI_IN_CH=3). This used to compose the three
    primitives here, which was correct but was a SECOND copy of the joint
    reduction — and the whole point of roi_crop_ref is that there is one. The
    reference returns interleaved [pr, pc, 3] because that is the AXIS wire
    order; the conv model downstream wants planes, hence the moveaxis.
    """
    out = RC.stage_a_rgb(np.asarray(planes_u8), roi_row, roi_col,
                         roi_h, roi_w, R, C)
    return np.moveaxis(out, -1, 0).astype(np.int64)


# ---------------------------------------------------------------------------
# weights — same derivation for both arms, only the tap count differs
# ---------------------------------------------------------------------------

def folded_weights():
    import torchvision.models as models
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    conv, bn = m.features[0][0], m.features[0][1]
    cw = conv.weight.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    beta = bn.bias.detach().numpy().astype(np.float64)
    mu = bn.running_mean.detach().numpy().astype(np.float64)
    s = g / np.sqrt(var + bn.eps)
    return cw * s[:, None, None, None], s * (0.0 - mu) + beta


def quantize(w_float, b_fold):
    """export_weights.py's rule, with the bias fixed to the real input scale.

    bias_acc uses ROI_NORM_Q (32), not export_weights' 127 — see the bias_acc
    entry in CLAUDE.md. out_shift then follows from the tap count, which is the
    only place the two arms legitimately diverge: 9 taps vs 27.
    """
    n_in = w_float.shape[1]
    acc_max_theory = n_in * KSIZE * KSIZE * 127 * 127
    flat = w_float.reshape(N_OUT, -1)
    scale = np.abs(flat).max(axis=1) / 127.0
    wq = np.clip(np.round(w_float / scale[:, None, None, None]), -127, 127)
    bias = np.round(b_fold * ROI_NORM_Q / scale).astype(np.int64)
    shift = np.array([int(np.ceil(np.log2(max(
        (abs(int(bias[oc])) + acc_max_theory) / 32767.0, 1.0))))
        for oc in range(N_OUT)], dtype=np.int64)
    return wq.astype(np.int64), bias, shift


# ---------------------------------------------------------------------------
# conv2d — the integer datapath, ReLU off, Stage B1, separable Hann
# ---------------------------------------------------------------------------

# Saturation accounting. --match-shift forces a 27-tap accumulator through a
# 9-tap out_shift, which CAN overflow int16; np.clip saturates silently and a
# saturation-driven result would look exactly like a real one. Counted here so
# the caller can print it and disqualify the run if it is significant.
SAT = {'clipped': 0, 'total': 0, 'clipped_b1': 0, 'clipped_hann': 0}


def sat_reset():
    for k in SAT:
        SAT[k] = 0


def sat_frac():
    return SAT['clipped'] / SAT['total'] if SAT['total'] else 0.0


def conv_features(patch, wq, bias, shift, mean_prev=None, relu=False):
    """patch: [n_in, R, C] int64. Returns (features [16,R,C] int64, per-ch mean).

    mean_prev=None means "this is the frame that supplies it" — B1 then uses the
    frame's own mean, which is what the host seeds on frame 0. On the eval frame
    the caller passes the TRAINING frame's means, mirroring the hardware, where
    Stage B1 always subtracts the previous frame's mean.
    """
    n_in = patch.shape[0]
    pad = np.zeros((n_in, R + 2, C + 2), dtype=np.int64)
    pad[:, 1:-1, 1:-1] = patch

    acc = np.zeros((N_OUT, R, C), dtype=np.int64)
    for kr in range(KSIZE):
        for kc in range(KSIZE):
            tile = pad[:, kr:kr + R, kc:kc + C]                # [n_in, R, C]
            acc += np.einsum('oi,irc->orc', wq[:, :, kr, kc], tile)
    acc += bias[:, None, None]

    # >> out_shift is an ARITHMETIC shift (floor), matching the kernel's
    # aie::downshift and C++ >>. No ReLU: saturate only.
    sh = acc >> shift[:, None, None]
    SAT['clipped'] += int(np.count_nonzero((sh > 32767) | (sh < -32768)))
    SAT['total'] += int(sh.size)
    sh = np.clip(sh, -32768, 32767)

    # CONV_RELU. Off in the shipping build, and the settled entry in CLAUDE.md
    # says why: a DCF is linear in feature space, so a half-wave rectifier alone
    # throws away half the signal and costs ~3x the peak/sidelobe ratio. It is
    # exposed here because that argument does NOT survive adding pooling: HOG's
    # deformation tolerance comes from aggregating a RECTIFIED signal, and
    # averaging a signed edge map over a cell cancels the very lobes it is
    # meant to summarise. relu+pool is the combination worth measuring; relu
    # alone is the one already measured and rejected.
    #
    # Order matches the kernel exactly (conv2d_kernel.cpp): downshift -> ReLU
    # -> subtract mean_prev -> Hann. B1 re-centres the rectified map, so the DC
    # pedestal a rectifier introduces is removed by machinery that is already
    # there.
    if relu:
        sh = np.maximum(sh, 0)

    own_mean = (sh.sum(axis=(1, 2)) // (R * C)).astype(np.int64)
    mp = own_mean if mean_prev is None else mean_prev
    b1 = sh - mp[:, None, None]
    SAT['clipped_b1'] += int(np.count_nonzero((b1 > 32767) | (b1 < -32768)))
    cen = np.clip(b1, -32768, 32767)

    w1 = (cen * HANN[None, :, None]) >> 15
    h2 = (w1 * HANN[None, None, :]) >> 15
    SAT['clipped_hann'] += int(np.count_nonzero((h2 > 32767) | (h2 < -32768)))
    out = np.clip(h2, -32768, 32767)
    return out, own_mean


def conv_features_float(patch, w_float, b_fold, mean_prev=None):
    """The SAME conv as conv_features, in float64, with UNQUANTIZED weights.

    WHY: "is the tracker's poor robustness caused by quantization?" is a
    question about a counterfactual, and the closed-loop model already answers
    half of it -- its FFT, filter and response are float64, so the cint16 /
    Q1.15 / H_SHIFT pipeline is not in the loop and the board's failures are
    reproduced anyway. What that model still shares with the board is the INT8
    FEATURE path: per-channel weight quantization, out_shift, the int16 clips
    and the integer Hann multiplies. This function removes exactly those and
    nothing else, so the pair brackets the whole quantization question.

    NOT removed, deliberately: Stage A's int8 patch. That is roi_crop's output
    and a separate design choice with its own scale (ROI_NORM_Q); mixing it in
    would make a difference unattributable. The input scale convention is the
    same one quantize() uses -- ROI_NORM_Q counts as 1.0 -- which is why the
    folded bias enters as b_fold * ROI_NORM_Q here and as
    round(b_fold * ROI_NORM_Q / scale) there.
    """
    n_in = patch.shape[0]
    pad = np.zeros((n_in, R + 2, C + 2), dtype=np.float64)
    pad[:, 1:-1, 1:-1] = patch.astype(np.float64)

    acc = np.zeros((N_OUT, R, C), dtype=np.float64)
    for kr in range(KSIZE):
        for kc in range(KSIZE):
            tile = pad[:, kr:kr + R, kc:kc + C]
            acc += np.einsum('oi,irc->orc', w_float[:, :, kr, kc], tile)
    acc += (b_fold * ROI_NORM_Q)[:, None, None]

    # No out_shift and no int16 clip: those exist only to fit a fixed-point
    # container. B1 and the Hann window stay, because they are algorithm, not
    # arithmetic -- dropping them would change what is being compared.
    own_mean = acc.mean(axis=(1, 2))
    mp = own_mean if mean_prev is None else mean_prev
    cen = acc - mp[:, None, None]
    hf = HANN.astype(np.float64) / 32768.0
    out = cen * hf[None, :, None] * hf[None, None, :]
    return out, own_mean


# ---------------------------------------------------------------------------
# filter + response  (float; see the module docstring)
# ---------------------------------------------------------------------------

def gaussian_target_spectrum(rows, cols, sigma):
    k = lambda n: np.where(np.arange(n) > n // 2, np.arange(n) - n, np.arange(n))
    u = k(rows).reshape(-1, 1).astype(np.float64)
    v = k(cols).reshape(1, -1).astype(np.float64)
    return np.exp(-2.0 * np.pi**2 * sigma**2 * (u**2 / rows**2 + v**2 / cols**2))


def train(feats):
    """One-shot filter_init: A = conj(G)*F, B = sum|F|^2, H = A*chscale/(B+eps)."""
    F = np.fft.fft2(feats.astype(np.float64), axes=(1, 2))
    Gt = gaussian_target_spectrum(R, C, SIGMA)
    A = np.conj(Gt)[None] * F
    B = np.sum(np.abs(F)**2, axis=0)
    energy = np.sum(np.abs(F)**2, axis=(1, 2))          # Stage B3
    chscale = np.where(energy > 0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
    return A * chscale[:, None, None] / (B + EPS_REL * B.mean())[None]


def respond(H, feats):
    """cmul_accum conjugates the stored filter — resp = IFFT(sum F * conj(H))."""
    F = np.fft.fft2(feats.astype(np.float64), axes=(1, 2))
    return np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))


def metrics(resp):
    idx = np.unravel_index(np.argmax(np.abs(resp)), resp.shape)
    peak = resp[idx]
    rr = np.minimum((np.arange(R) - idx[0]) % R, (idx[0] - np.arange(R)) % R)
    cc = np.minimum((np.arange(C) - idx[1]) % C, (idx[1] - np.arange(C)) % C)
    sl = resp[~((rr[:, None] <= 5) & (cc[None, :] <= 5))]
    if sl.size == 0 or sl.std() == 0:
        return idx, peak, 0.0, 0.0
    return idx, peak, (peak - sl.mean()) / sl.std(), abs(peak) / max(np.abs(sl).max(), 1e-9)


def wrap(d, n):
    return d - n if d > n // 2 else d


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=40, help='frames between anchors')
    ap.add_argument('--deltas', type=int, nargs='+', default=[1, 5, 10],
                    help='holdout gaps, in frames')
    ap.add_argument('--arms', nargs='+', default=['gray', 'rgb', 'rgb-lum'])
    args = ap.parse_args()

    gt = load_gt()
    n = len(gt)
    w_rgb, b_fold = folded_weights()
    w_gray = (w_rgb * LUM[None, :, None, None]).sum(axis=1, keepdims=True)

    W = {'gray': quantize(w_gray, b_fold),
         'rgb': quantize(w_rgb, b_fold),
         'rgb-lum': quantize(w_rgb, b_fold)}

    print("int8 derivation per arm (the tap count is the only legitimate divergence):")
    for a in args.arms:
        wq, bias, shift = W[a]
        print(f"  {a:<8} taps {wq.shape[1]*9:>2}   out_shift {shift.min()}-{shift.max()} "
              f"(mean {shift.mean():.2f})   |bias| median {np.median(np.abs(bias)):.0f}")
    print("  NOTE: more taps => larger ACC_MAX_THEORY => larger out_shift => RGB")
    print("  carries FEWER signal bits per channel. Any RGB win here is a floor.")
    print()

    anchors = list(range(1, n - max(args.deltas), args.step))
    print(f"sequence car1: {n} frames, {len(anchors)} anchors, "
          f"deltas {args.deltas}, padding {PADDING}, sigma {SIGMA}")
    print(f"arms: {', '.join(args.arms)}   (bias=ROI_NORM_Q, ReLU off, float FFT)")
    print()

    res = {a: {d: {'ratio': [], 'bolme': [], 'err': []} for d in args.deltas}
           for a in args.arms}

    cache = {}

    def planes(i):
        if i not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[i] = load_frame_rgb(i)
        return cache[i]

    for t in anchors:
        row, col, bh, bw = gt[t - 1]
        roi_h, roi_w = int(round(bh * PADDING)), int(round(bw * PADDING))
        roi_row = int(round(row - roi_h / 2.0))
        roi_col = int(round(col - roi_w / 2.0))

        p_tr = planes(t)
        lum_tr = to_luma(p_tr)

        patches_tr = {
            'gray': stage_a_gray(lum_tr, roi_row, roi_col, roi_h, roi_w)[None],
            'rgb': stage_a_rgb(p_tr, roi_row, roi_col, roi_h, roi_w),
            'rgb-lum': stage_a_rgb(np.stack([lum_tr] * 3), roi_row, roi_col,
                                   roi_h, roi_w),
        }

        state = {}
        for a in args.arms:
            wq, bias, shift = W[a]
            f, mprev = conv_features(patches_tr[a], wq, bias, shift)
            state[a] = (train(f), mprev)

        for d in args.deltas:
            e = t + d
            r2, c2, _, _ = gt[e - 1]
            # Expected peak in PATCH bins: a frame-pixel displacement scales by
            # the resample ratio. Same conversion the host needs once roi != patch.
            exp = (int(round((r2 - row) * R / roi_h)), int(round((c2 - col) * C / roi_w)))

            p_ev = planes(e)
            lum_ev = to_luma(p_ev)
            patches_ev = {
                'gray': stage_a_gray(lum_ev, roi_row, roi_col, roi_h, roi_w)[None],
                'rgb': stage_a_rgb(p_ev, roi_row, roi_col, roi_h, roi_w),
                'rgb-lum': stage_a_rgb(np.stack([lum_ev] * 3), roi_row, roi_col,
                                       roi_h, roi_w),
            }
            for a in args.arms:
                wq, bias, shift = W[a]
                H, mprev = state[a]
                f, _ = conv_features(patches_ev[a], wq, bias, shift, mean_prev=mprev)
                idx, _pk, bolme, ratio = metrics(respond(H, f))
                dr, dc = wrap(idx[0], R), wrap(idx[1], C)
                err = float(np.hypot(dr - exp[0], dc - exp[1]))
                res[a][d]['ratio'].append(ratio)
                res[a][d]['bolme'].append(bolme)
                res[a][d]['err'].append(err)

    print(f"{'arm':<9} {'dt':>3}  {'peak/maxSL':>18}  {'Bolme PSR':>16}  "
          f"{'loc err (bins)':>18}  {'<=2 bins':>8}")
    print(f"{'':<9} {'':>3}  {'mean   median':>18}  {'mean  median':>16}  "
          f"{'mean  median':>18}")
    print("-" * 84)
    for a in args.arms:
        for d in args.deltas:
            r = np.array(res[a][d]['ratio']); b = np.array(res[a][d]['bolme'])
            e = np.array(res[a][d]['err'])
            print(f"{a:<9} {d:>3}  {r.mean():8.2f} {np.median(r):8.2f}  "
                  f"{b.mean():7.2f} {np.median(b):7.2f}  "
                  f"{e.mean():8.2f} {np.median(e):8.2f}  "
                  f"{100*np.mean(e <= 2):7.1f}%")
        print()

    if all(a in res for a in ('gray', 'rgb', 'rgb-lum')):
        print("=" * 84)
        allof = lambda a, k: np.concatenate([res[a][d][k] for d in args.deltas])
        for k, lbl in (('ratio', 'peak/max-sidelobe'), ('bolme', 'Bolme PSR')):
            g, r, l = (allof('gray', k).mean(), allof('rgb', k).mean(),
                       allof('rgb-lum', k).mean())
            print(f"{lbl:<20} gray {g:7.2f}   rgb {r:7.2f} ({r/g:+.2f}x)   "
                  f"rgb-lum {l:7.2f} ({l/g:+.2f}x)")
        for k, lbl in (('err', 'loc err (bins)'),):
            g, r, l = (allof('gray', k), allof('rgb', k), allof('rgb-lum', k))
            print(f"{lbl:<20} gray {g.mean():7.2f}   rgb {r.mean():7.2f}          "
                  f"   rgb-lum {l.mean():7.2f}")
            print(f"{'within 2 bins':<20} gray {100*np.mean(g<=2):6.1f}%   "
                  f"rgb {100*np.mean(r<=2):6.1f}%            "
                  f"rgb-lum {100*np.mean(l<=2):6.1f}%")
        print()
        print("PAIRED, per evaluation (n = anchors x deltas) — means hide outliers:")
        for k, lbl, better in (('ratio', 'peak/max-sidelobe', 'gt'),
                               ('bolme', 'Bolme PSR', 'gt'),
                               ('err', 'loc err', 'lt')):
            g, r, l = allof('gray', k), allof('rgb', k), allof('rgb-lum', k)
            cmp = (lambda a, b: a > b) if better == 'gt' else (lambda a, b: a < b)
            print(f"  {lbl:<20} rgb beats gray {100*np.mean(cmp(r,g)):5.1f}% of {len(g)}"
                  f"   |   rgb-lum beats gray {100*np.mean(cmp(l,g)):5.1f}%"
                  f"   |   rgb beats rgb-lum {100*np.mean(cmp(r,l)):5.1f}%")
            if better == 'lt':
                for nm, x in (('rgb vs gray', (r, g)), ('rgb vs rgb-lum', (r, l))):
                    a, b = x
                    print(f"  {'':<20} {nm:<16} win {100*np.mean(a<b-1e-9):5.1f}%  "
                          f"tie {100*np.mean(np.abs(a-b)<=1e-9):5.1f}%  "
                          f"loss {100*np.mean(a>b+1e-9):5.1f}%")
            if better == 'gt':
                print(f"  {'':<20} median per-eval ratio rgb/gray {np.median(r/np.maximum(g,1e-9)):.3f}"
                      f"   rgb-lum/gray {np.median(l/np.maximum(g,1e-9)):.3f}")
        print()
        print("READ THE CONTROL: rgb beating gray means nothing unless rgb also")
        print("beats rgb-lum. rgb-lum has RGB's taps and quantization grid and")
        print("NO colour information; whatever it gains is bookkeeping, not colour.")


if __name__ == '__main__':
    main()
