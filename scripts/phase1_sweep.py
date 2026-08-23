#!/usr/bin/env python3
"""
scripts/phase1_sweep.py

Phase 1 offline decision tool: pick the bias_acc/out_shift variant AND the shift
budget, in seconds, without a single simulation.

Companion to check_collapse.py, which answers "what is wrong with the feature
bank" (Q1-Q4). This answers "what should we change it to, and what does the shift
budget have to become as a result" — the two are coupled, because every variant
makes the conv output LARGER and the response scales with it.

Why this can be trusted
-----------------------
It runs the same integer datapath the kernel runs. As of 2026-08-14 that model is
VERIFIED, not assumed: `make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0`
diffs gen_aiesim_vectors.simulate_conv2d against the real kernel under x86sim and
reports 16384/16384 samples identical, with Stage B1 active. Before that check
existed this script would have been guesswork dressed as measurement.

What it still approximates — read before trusting a number to better than ~10%:
  * DSPLib's cint16 FFT is modelled as an exact float FFT with a rounding
    quantization per pass. The real kernel also loses ~21 LSB additively on a
    summed DC bin (FFT_DC_TRUNC in gen_aiesim_vectors.py).
  * Rounding is np.round (banker's) rather than the hardware's round-half-away.
  * The filter is trained and evaluated on the SAME patch, so absolute PSR is
    optimistic. Budget COMPARISONS are unaffected, which is what this is for.
  The recorded model-vs-hardware agreement is ~6% (CLAUDE.md, s7). Size the
  budget with headroom rather than trying to land the response on the rail.

Variants (see the bias_acc entry in CLAUDE.md Known Issues)
------------------------------------------------------------
  base  what ships today: bias_acc = b_fold*127/scale
  a     bias_acc = b_fold*ROI_NORM_Q/scale  -- data-only, no AIE rebuild
  b     (a) + >>out_shift moved AFTER B1's mean subtraction -- kernel edit
  c     drop ReLU and zero the bias -- diverges from Danelljan §3.3

Usage
-----
  uv run python3 scripts/phase1_sweep.py                  # all variants, default grid
  uv run python3 scripts/phase1_sweep.py --variant a b    # subset
  uv run python3 scripts/phase1_sweep.py --budgets 4,2,2 4,3,4
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# gen_aiesim_vectors reads geometry from the environment at import time. Force
# the design point unless the caller has already chosen one, so a bare run
# describes the real 128x128 build rather than whatever was last exported.
os.environ.setdefault('GEN_PATCH_ROWS', '128')
os.environ.setdefault('GEN_PATCH_COLS', '128')
os.environ.setdefault('GEN_H_SHIFT', '10')

import gen_aiesim_vectors as G          # noqa: E402
import gen_filter_golden as FG          # noqa: E402
import roi_crop_ref as RC               # noqa: E402
import synth_frame as SF                # noqa: E402

KSIZE = 3
N_OUT = 16
ROI_NORM_Q = 32          # roi_crop.h — the int8 scale conv2d actually receives
FULL_SCALE = 127         # what export_weights.py assumed instead
ACC_MAX_THEORY = KSIZE * KSIZE * 127 * 127

WEIGHTS_BIN = Path("design/aie_src/weights/layer0_weights.bin")
SCENARIO = Path("design/aie_src/aiesim_data/s6")

HANN = G.HANNING.astype(np.int64)
R, C = G.PATCH_ROWS, G.PATCH_COLS
H_SHIFT = G.H_SHIFT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def q16(z):
    """Quantize a complex array to cint16 with saturation. Returns (arr, n_railed)."""
    re = np.round(z.real)
    im = np.round(z.imag)
    railed = int(np.sum((re > 32767) | (re < -32768) | (im > 32767) | (im < -32768)))
    return np.clip(re, -32768, 32767) + 1j * np.clip(im, -32768, 32767), railed


def wmean(f):
    """Window-weighted mean, the constant that actually zeros the DC bin.

    Bolme's text says plain mean, but the plain mean does not zero a
    window-weighted DC sum — see the Stage B1 note in conv2d_kernel.h. This is
    the fixed point mean_prev converges to, so using it here models the steady
    state rather than frame 0.
    """
    w = HANN[:, None] * HANN[None, :]
    return float(np.sum(w * f)) / float(np.sum(HANN)) ** 2


def out_shift_for(max_val):
    """export_weights.compute_acc_params' rule, factored out."""
    return int(np.ceil(np.log2(max(max_val / 32767.0, 1.0))))


def load_channels():
    b = WEIGHTS_BIN.read_bytes()
    # Grayscale-only: the offline model convolves a single-plane Stage-A patch.
    # The layout tag turns a CONV_IN_CH=3 export into an error instead of a
    # plausible-looking model built from R-plane taps.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conv_weight_layout as CWL
    if CWL.detect(b[:CWL.BUF_BYTES]) != 1:
        raise SystemExit(f"{WEIGHTS_BIN} is an RGB export; phase1_sweep is "
                         f"grayscale-only. Re-run: make weights CONV_IN_CH=1")
    chans = []
    for oc in range(len(b) // 64):
        r = b[oc * 64:(oc + 1) * 64]
        w = np.frombuffer(r[0:9], dtype=np.int8).astype(np.int64).reshape(3, 3)
        chans.append((w, int(r[9]), struct.unpack("<i", r[10:14])[0]))
    return chans


def load_patch():
    """The Stage-A preprocessed int8 patch, decoded from the PLIO stimulus."""
    words = np.loadtxt(SCENARIO / "patch_in.txt", dtype=np.int64)[: (R * C) // 4]
    u = words.astype(np.uint32)
    px = np.zeros(R * C, dtype=np.int8)
    for k in range(4):
        px[k::4] = ((u >> (8 * k)) & 0xFF).astype(np.uint8).view(np.int8)
    return px.astype(np.int64).reshape(R, C)


def build_patch_synth(target_h, target_w, padding, frame_rows, frame_cols,
                      kind, background, bg_contrast, seed, shift=None):
    """Synthesize a FRAME, crop an ROI of target*padding, run the real Stage A.

    This is what `--roi-model file` cannot do: load_patch() reads a patch that was
    already Stage-A'd at 1:1, so it carries no notion of an ROI and padding has
    nowhere to act. Here the source exists at frame resolution and the padding
    factor decides how much of it the fixed 128x128 patch has to represent.

    Stage A is the BIT-EXACT integer model (roi_crop_ref), not a float stand-in.
    That matters because the sweep's output variable is a fixed-point scale — it
    reads accum/response against 32767 — and the int8 patch's contrast after the
    +/-127 clip and the LOG_LUT's compression is precisely what sets it. Measured
    on s6, the float Stage A shipped in gen_aiesim_vectors.py differs from the
    kernel on 40.9% of samples (max 2 LSB, rms 0.65 on a signal of std 32).

    `shift` gives a REAL held-out evaluation: the target moves by (dr, dc) FRAME
    pixels and the same ROI is re-cropped, so the evaluation patch differs by
    resample phase, border content and its own mean/sigma. --holdout's np.roll is
    a circular shift of the Stage-A OUTPUT, which cannot physically happen.
    """
    roi_h = int(round(target_h * padding))
    roi_w = int(round(target_w * padding))
    tr, tc = frame_rows / 2.0, frame_cols / 2.0
    # The host computes roi_row = pos_row - roi_h/2; mirror it exactly, including
    # the fact that it goes negative for a target near an edge.
    roi_row = int(round(tr - roi_h / 2.0))
    roi_col = int(round(tc - roi_w / 2.0))

    def frame_at(dr, dc):
        return SF.make_frame(frame_rows, frame_cols, target_h, target_w,
                             tr + dr, tc + dc, kind=kind, background=background,
                             bg_contrast=bg_contrast, noise=SF.SWEEP_NOISE,
                             seed=seed)

    patch, diag = RC.stage_a(frame_at(0.0, 0.0), roi_row, roi_col, roi_h, roi_w,
                             R, C, with_diag=True)
    diag['roi_h'], diag['roi_w'] = roi_h, roi_w
    diag['roi_row'], diag['roi_col'] = roi_row, roi_col

    patch_ev, expect = None, (0, 0)
    if shift is not None:
        dr, dc = shift
        patch_ev = RC.stage_a(frame_at(float(dr), float(dc)),
                              roi_row, roi_col, roi_h, roi_w, R, C)
        patch_ev = patch_ev.astype(np.int64)
        # A frame-pixel displacement maps to patch bins by the resample ratio.
        # This is the SAME conversion mosse_tracker.cpp needs at :1480 and :1482
        # once roi_h != patch_rows, and getting it wrong there gives a tracker
        # that localises confidently and drifts.
        expect = (int(round(dr * R / roi_h)) % R, int(round(dc * C / roi_w)) % C)
    return patch.astype(np.int64), patch_ev, expect, diag


def window(v):
    """conv2d's separable Hann, with BOTH >>15 truncations. Not srs rounding."""
    wnd = (v * HANN[:, None]) >> 15
    wnd = (wnd * HANN[None, :]) >> 15
    return np.clip(wnd, -32768, 32767)


# ---------------------------------------------------------------------------
# the four conv2d variants
# ---------------------------------------------------------------------------

# A variant is three independent choices, not a single knob. Naming them
# separately is what makes the bias question separable from the ReLU question —
# and that separation turned out to matter: see the a-vs-a_nr pair.
#   bias : 127 (as shipped) | 32 (ROI_NORM_Q, the contract fix) | 0
#   relu : on | off
#   late : shift AFTER the B1 mean subtraction (recovers the pedestal's bits)
VARIANTS = {
    'base':  dict(bias=127, relu=True,  late=False),   # what ships today
    'a':     dict(bias=32,  relu=True,  late=False),   # contract fix, data-only
    'b':     dict(bias=32,  relu=True,  late=True),    # a + kernel edit
    'c':     dict(bias=0,   relu=False, late=False),   # drop ReLU, zero bias
    # Isolating runs. 'a' and 'c' differ in TWO things at once (bias scale and
    # ReLU), so neither can be blamed without these.
    'a_nr':  dict(bias=32,  relu=False, late=False),   # a, ReLU removed
    'b_nr':  dict(bias=32,  relu=False, late=True),    # b, ReLU removed
    'base_nr': dict(bias=127, relu=False, late=False), # base, ReLU removed
}


def variant_params(w, shift, bias, variant):
    """Return (bias, out_shift) for the variant, plus the max AC swing."""
    spec = VARIANTS[variant]
    max_ac = 127 * int(np.abs(w).sum())

    if spec['bias'] == 127:
        b = bias
    elif spec['bias'] == 32:
        b = int(round(bias * ROI_NORM_Q / FULL_SCALE))
    else:
        b = 0

    if variant == 'base':
        return b, shift, max_ac          # keep the shipped shift exactly
    if spec['late'] or spec['bias'] == 0:
        # The DC pedestal is gone before the shift (or never existed), so the
        # shift only has to accommodate the AC swing. This is where the bits are.
        return b, out_shift_for(max_ac), max_ac
    return b, out_shift_for(abs(b) + ACC_MAX_THEORY), max_ac


def conv_variant(patch, w, shift, bias, variant):
    """One channel through conv2d, returning the windowed int16 feature map.

    'base' is byte-for-byte the datapath in conv2d_kernel.cpp (and is checked
    against it by check_kernel_bitexact.py). The others differ only where the
    variant says they do.
    """
    spec = VARIANTS[variant]
    xp = np.pad(patch, 1)
    acc = np.full((R, C), bias, dtype=np.int64)
    for kr in range(KSIZE):
        for kc in range(KSIZE):
            acc += w[kr, kc] * xp[kr:kr + R, kc:kc + C]

    if spec['late']:
        # Nonlinearity (if any) and mean removal both in the accumulator domain,
        # shift last.
        v = np.maximum(acc, 0) if spec['relu'] else acc
        mp = int(round(wmean(v)))
        out = np.clip((v - mp) >> shift, -32768, 32767)
        return window(out), mp

    sh = acc >> shift
    if spec['relu']:
        out16 = np.where(sh > 32767, 32767, np.where(sh <= 0, 0, sh))
    else:
        out16 = np.clip(sh, -32768, 32767)

    mp = int(round(wmean(out16)))
    centred = np.clip(out16 - mp, -32768, 32767)
    return window(centred), mp


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def forward_fft(feats, fft_shift):
    """conv output -> row FFT -> col FFT, cint16 after each pass."""
    F_all, energy = [], []
    row_rail = col_rail = 0
    for f in feats:
        A, r1 = q16(np.fft.fft(f.astype(np.float64), axis=1) / (1 << fft_shift))
        row_rail += r1
        energy.append(float(np.mean(np.abs(A) ** 2)))       # Stage B3, from row FFT
        Fc, r2 = q16(np.fft.fft(A, axis=0) / (1 << fft_shift))
        col_rail += r2
        F_all.append(Fc)
    return np.array(F_all), np.array(energy), row_rail, col_rail


def run_pipeline(feats, fft_shift, ifft_row_shift, ifft_col_shift,
                 feats_eval=None, sigma=None, b2_null=True):
    """conv output -> FFT -> MOSSE filter -> cmul -> B2 -> IFFT -> response.

    feats trains the filter. feats_eval, if given, is what the filter is then
    applied to — a HELD-OUT evaluation. Without it the filter is applied to its
    own training patch, which makes a matched filter exactly optimal and
    therefore flatters a linear pipeline specifically. Since the headline result
    here is "remove the nonlinearity", that self-evaluation is the objection the
    finding has to survive, so it is worth running both ways.

    sigma is the target Gaussian's width. None keeps FG.SIGMA, so every result
    recorded before this parameter existed reproduces byte for byte.

    NOTE there are four independent copies of "sigma = 2" in the tree and only
    ONE of them is a design parameter:
        mosse_filter.h:92 DEFAULT_SIGMA        <- the shipped tracker value
        gen_filter_golden.py:41 SIGMA          <- test_host golden fixture
        gen_aiesim_vectors.py:998 s7_sigma     <- aiesim s7 fixture
        test_mosse_filter.cpp:246 exp(-d2/8)   <- self-contained PSR fixture
    A sweep result lands in the first. Unifying the other three would silently
    invalidate three goldens.
    """
    n_ch = len(feats)
    rails = {}

    F_all, energy, row_rail, col_rail = forward_fft(feats, fft_shift)
    rails['fft_row'], rails['fft_col'] = row_rail, col_rail

    if feats_eval is None:
        F_eval = F_all
    else:
        F_eval, _, r1, r2 = forward_fft(feats_eval, fft_shift)
        rails['fft_row'] += r1
        rails['fft_col'] += r2

    # Filter: init (eta=1 against a zeroed state), centred G, exactly as the
    # tracker does on frame 0.
    Gt = FG.gaussian_target_spectrum(
        R, C, FG.SIGMA if sigma is None else float(sigma), 0, 0)
    A_f, B_f = FG.filter_update(np.zeros_like(F_all), np.zeros((R, C)),
                                F_all, Gt, 1.0)
    hq, _, _ = FG.quantize_q15(A_f, B_f, energy, FG.EPS_REL)
    Hq = (hq[0::2].astype(np.int64).reshape(n_ch, R, C)
          + 1j * hq[1::2].astype(np.int64).reshape(n_ch, R, C))

    # cmul: saturating accumulate, channel by channel — the same order and the
    # same clamp the kernel applies. simulate_cmul is the verified model.
    acc_re = np.zeros((R, C), dtype=np.int64)
    acc_im = np.zeros((R, C), dtype=np.int64)
    for ch in range(n_ch):
        acc_re, acc_im = G.simulate_cmul(
            F_eval[ch].real.astype(np.int64), F_eval[ch].imag.astype(np.int64),
            Hq[ch].real.astype(np.int64), Hq[ch].imag.astype(np.int64),
            acc_re, acc_im)
    accum = acc_re + 1j * acc_im
    rails['accum'] = int(np.sum((np.abs(acc_re) >= 32767) | (np.abs(acc_im) >= 32767)))

    # Stage B2, B2_NULL_BINS=1: zero the 9 low-frequency bins.
    #
    # b2_null=False models B2_NULL_BINS=0. It is a parameter because B2 is not
    # neutral with respect to sigma: a wider target Gaussian puts more of its
    # energy in low frequencies, which is exactly what these 9 bins are, so
    # nulling them penalises large sigma specifically. Sweeping sigma with B2
    # hardcoded would attribute a Stage-B2 cost to the sigma choice.
    b2_removed = 0.0
    for a in (0, 1, R - 1):
        for b in (0, 1, C - 1):
            b2_removed = max(b2_removed, abs(accum[a, b]))
            if b2_null:
                accum[a, b] = 0

    # Inverse: row IFFT then col IFFT. *N undoes numpy's 1/N, matching DSPLib.
    Yr, r3 = q16(np.fft.ifft(accum, axis=1) * C / (1 << ifft_row_shift))
    rails['ifft_row'] = r3
    Y, r4 = q16(np.fft.ifft(Yr, axis=0) * R / (1 << ifft_col_shift))
    rails['response'] = r4

    return Y.real, accum, np.abs(Yr).max(), rails, b2_removed


def psr(resp):
    """Both statistics, because neither one's thresholds transfer to the other.

    Bolme §3.5 is (g_max - mu_sl)/sigma_sl; the aiesim harness reports
    |peak|/max|sidelobe|. Same 11x11 circular exclusion, different numbers —
    measured 7.2x apart on real data.
    """
    idx = np.unravel_index(np.argmax(np.abs(resp)), resp.shape)
    peak = resp[idx]
    rr = np.minimum((np.arange(R) - idx[0]) % R, (idx[0] - np.arange(R)) % R)
    cc = np.minimum((np.arange(C) - idx[1]) % C, (idx[1] - np.arange(C)) % C)
    mask = (rr[:, None] <= 5) & (cc[None, :] <= 5)
    sl = resp[~mask]
    if sl.size == 0 or sl.std() == 0:
        return idx, peak, 0.0, 0.0
    bolme = (peak - sl.mean()) / sl.std()
    ratio = abs(peak) / max(np.abs(sl).max(), 1e-9)
    return idx, peak, bolme, ratio


# ---------------------------------------------------------------------------

def _feats_for(patch, chans, variant):
    return [conv_variant(patch, w, sh, b, variant)[0]
            for (w, shift, bias) in chans
            for (b, sh, _mx) in [variant_params(w, shift, bias, variant)]]


def run_synth_sweep(args, chans, budgets):
    """Joint sweep over padding x target x sigma x budget.

    JOINT, not one axis at a time, because the couplings are specific:

      * padding sets the RESAMPLE RATIO, and the same padding upsamples at an
        11 px target and downsamples at 85 px — opposite spectral effects. So
        target size is a third axis, not a fixture.
      * sigma sets G's bandwidth and hence how much of |F| survives into the
        accumulator, which interacts directly with the low-pass that upsampling
        introduces.
      * Stage B2 nulls the 9 lowest bins, and upsampling pushes energy INTO
        them, so B2's loss is itself padding-dependent. b2_removed was already
        computed and thrown away; it is a reported column now.

    Sweeping sigma at fixed padding, or padding at a fixed budget, would
    reproduce exactly the coupling error that forced two shift-budget hunts.
    """
    fr, fc = args.frame
    summary = []

    print(f"geometry {R}x{C}  channels {len(chans)}  H_SHIFT {H_SHIFT}  "
          f"eps_rel {FG.EPS_REL}")
    print(f"source: synth/{args.frame_kind} {fr}x{fc}  bg {args.background} "
          f"c={args.bg_contrast}  seed {args.seed}  Stage A: bit-exact "
          f"(roi_crop_ref)")
    if args.background == 'flat':
        print("  WARNING: a flat background makes padding INFORMATION-FREE by "
              "construction —")
        print("           more padding is strictly less target, so this run will "
              "'prove' padding")
        print("           is pure loss. That is a correct answer to the wrong "
              "question.")
    if args.frame_shift:
        print(f"HELD-OUT eval: target moved {tuple(args.frame_shift)} FRAME px, "
              f"same ROI re-cropped, Stage A re-run")
    print()

    for variant in args.variant:
        for target in args.target:
            for padding in args.padding:
                patch, patch_ev, expect, d = build_patch_synth(
                    target, target, padding, fr, fc, args.frame_kind,
                    args.background, args.bg_contrast, args.seed,
                    shift=tuple(args.frame_shift) if args.frame_shift else None)
                roi_h, roi_w = d['roi_h'], d['roi_w']
                ratio_y = roi_h / R

                print("=" * 96)
                print(f"VARIANT {variant}   TARGET {target}x{target}   "
                      f"PADDING {padding}   -> roi {roi_h}x{roi_w} @"
                      f"({d['roi_row']},{d['roi_col']})")
                print("=" * 96)
                print(f"  step {d['step_y']}/{d['step_x']} Q8   "
                      f"{d['px_per_bin_y']:.3f} px/bin   "
                      f"{'UP' if ratio_y < 1 else 'DOWN'} {max(ratio_y, 1/ratio_y):.2f}x")
                print(f"  interpolator: fy nonzero on {d['fy_nonzero']}/{R} rows, "
                      f"fx on {d['fx_nonzero']}/{C} cols"
                      + ("   <-- DEGENERATE: still the shipped copy path"
                         if d['fy_nonzero'] == 0 and d['fx_nonzero'] == 0 else ""))
                print(f"  src rows touched {d['src_rows_touched']} of {roi_h}"
                      + ("   <-- ALIASING: bilinear has no prefilter, source rows "
                         "are being skipped" if d['src_rows_touched'] < roi_h * 0.9
                         and roi_h > R else ""))
                print(f"  patch: std {d['patch_std']:.1f}  clipped "
                      f"{d['clipped']}/{d['n_elems']} "
                      f"({100.0*d['clipped']/d['n_elems']:.2f}%)  inv_q {d['inv_q']}"
                      + ("  <-- CAPPED" if d['inv_q_capped'] else ""))
                print(f"  centring bias {d['centring_bias_y']:.3f} src px "
                      f"({d['centring_bias_y']/ratio_y:.2f} patch px) — does NOT "
                      f"cancel in a closed loop")

                feats = _feats_for(patch, chans, variant)
                feats_ev = _feats_for(patch_ev, chans, variant) if patch_ev is not None else None

                print()
                print("   sigma  budget      accum   %rail    b2rm   response   "
                      "%rail  PSR(B)   ratio  peak@        locerr")
                print("  " + "-" * 94)

                for sigma_spec in args.sigma:
                    sigma = (R / (16.0 * padding) if str(sigma_spec) == 'auto'
                             else float(sigma_spec))
                    for (fs, irs, ics) in budgets:
                        resp, accum, ifftrow, rails, b2 = run_pipeline(
                            feats, fs, irs, ics, feats_ev, sigma=sigma,
                            b2_null=not args.no_b2)
                        amax = float(np.abs(accum).max())
                        rmax = float(np.abs(resp).max())
                        idx, peak, bol, rat = psr(resp)
                        i0, i1 = int(idx[0]), int(idx[1])
                        dr = min((i0 - expect[0]) % R, (expect[0] - i0) % R)
                        dc = min((i1 - expect[1]) % C, (expect[1] - i1) % C)
                        locerr = float(np.hypot(dr * roi_h / R, dc * roi_w / C))
                        flag = ""
                        bad = [k for k in rails if rails[k]]
                        if bad:
                            flag = f"  <-- RAILED: {','.join(bad)}"
                        print(f"  {sigma:6.2f}  {fs}-{irs}-{ics}   {amax:9.0f}  "
                              f"{100*amax/32767:5.1f}%  {b2:6.0f}  {rmax:9.0f}  "
                              f"{100*rmax/32767:5.1f}%  {bol:6.1f}  {rat:6.2f}  "
                              f"({i0:3d},{i1:3d})  {locerr:6.2f} px{flag}")
                        summary.append(dict(
                            variant=variant, target=target, padding=padding,
                            sigma=sigma, budget=(fs, irs, ics), ratio=rat,
                            bolme=bol, resp_pct=100 * rmax / 32767,
                            locerr=locerr, railed=bool(bad),
                            px_per_bin=d['px_per_bin_y']))
                print()

    print("=" * 96)
    print("SUMMARY — best budget per (target, padding, sigma), ranked by ratio")
    print("=" * 96)
    print("  target  padding  sigma  budget   ratio  PSR(B)   resp%  locerr   flags")
    best = {}
    for row in summary:
        if row['railed']:
            continue
        k = (row['target'], row['padding'], row['sigma'])
        if k not in best or row['ratio'] > best[k]['ratio']:
            best[k] = row
    for row in sorted(best.values(), key=lambda r: -r['ratio']):
        flags = []
        if row['px_per_bin'] >= 2.0:
            flags.append(f"px/bin {row['px_per_bin']:.1f} — resolution-limited")
        if row['locerr'] > 1.0:
            flags.append("localisation error")
        b = "-".join(str(x) for x in row['budget'])
        print(f"  {row['target']:6d}  {row['padding']:7.1f}  {row['sigma']:5.2f}  "
              f"{b:>6}  {row['ratio']:6.2f}  {row['bolme']:6.1f}  "
              f"{row['resp_pct']:5.1f}%  {row['locerr']:5.2f}px  {'; '.join(flags)}")
    if not best:
        print("  (every combination railed — widen the budget grid)")
    print()
    print("=" * 96)
    print("How to read this")
    print("=" * 96)
    print("Pick the RESPONSE near 50% of range with nothing railed, then check the")
    print("section header: px/bin is the localisation quantum in FRAME pixels, so a")
    print("padding that wins on ratio while showing px/bin 2.0 cannot resolve better")
    print("than 2 px however good its PSR looks.")
    print()
    print("The model agrees with hardware to ~6% on the pipeline. The ROI model adds")
    print("an error that is now MEASURED, not assumed: make test_roi_crop diffs")
    print("roi_crop_ref against the kernel over 17 cases, 11 of which execute the")
    print("bilinear interpolator that no build has ever run. Do not commit a padding")
    print("to hardware while that target is failing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', nargs='+', default=['base', 'a', 'b', 'c'])
    ap.add_argument('--budgets', nargs='+', default=None,
                    help='comma triples FFT,IFFT_ROW,IFFT_COL, e.g. 4,2,2')
    ap.add_argument('--channels', type=int, default=None,
                    help='use only the first N channels. The accumulator and the '
                         'response do NOT simply scale with this — H is renormalised '
                         'to full scale for whatever set of channels it is built '
                         'from — so a ch1 prediction has to be computed, not divided.')
    ap.add_argument('--holdout', nargs=2, type=int, metavar=('DR', 'DC'),
                    default=None,
                    help='train on the patch, evaluate on a circular shift of it '
                         '(the peak must then land at DR,DC). Guards against the '
                         'self-evaluation bias that flatters a linear pipeline.')
    # --- ROI model: everything below defaults to the pre-existing behaviour ---
    ap.add_argument('--roi-model', choices=['file', 'synth'], default='file',
                    help="file (default) = the Stage-A patch from s6/patch_in.txt, "
                         "byte-identical to every result recorded before this "
                         "existed. synth = synthesize a frame, crop an ROI of "
                         "target*padding, run the bit-exact Stage A.")
    ap.add_argument('--padding', nargs='+', type=float, default=[2.0],
                    help='ROI padding factors. DSST 6.1 uses 2, fDSST 3.')
    ap.add_argument('--target', nargs='+', type=int, default=[64],
                    help='target size in FRAME pixels. 11 is what '
                         'inject_target_frame actually draws today.')
    ap.add_argument('--sigma', nargs='+', default=['auto'],
                    help="target Gaussian width. 'auto' = patch/(16*padding), "
                         "DSST 6.1's rule; numbers are used literally.")
    ap.add_argument('--frame-shift', nargs=2, type=int, metavar=('DR', 'DC'),
                    default=None,
                    help='REAL held-out eval (synth only): move the target by '
                         '(DR,DC) FRAME px and re-crop the same ROI, so the eval '
                         'patch differs by resample phase and border content.')
    ap.add_argument('--frame', nargs=2, type=int, metavar=('ROWS', 'COLS'),
                    default=[1080, 1920])
    ap.add_argument('--frame-kind', choices=['bars', 'blob'], default='bars')
    ap.add_argument('--background', choices=['texture', 'flat', 'gradient'],
                    default='texture',
                    help="texture (default). 'flat' reproduces "
                         "inject_target_frame, against which padding is "
                         "information-free BY CONSTRUCTION — see below.")
    ap.add_argument('--bg-contrast', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=20260816)
    ap.add_argument('--no-b2', action='store_true',
                    help='model B2_NULL_BINS=0. B2 nulls the 9 lowest bins, which is where a wide target Gaussian keeps its energy, so it penalises large sigma specifically.')
    args = ap.parse_args()

    if args.roi_model == 'synth' and args.holdout:
        ap.error("--holdout does an np.roll on the Stage-A OUTPUT, which cannot "
                 "physically occur once an ROI exists. Use --frame-shift, which "
                 "moves the target in the frame and re-crops.")

    if args.budgets:
        budgets = [tuple(int(x) for x in b.split(',')) for b in args.budgets]
    else:
        budgets = [(4, 2, 2), (4, 3, 3), (4, 3, 4), (5, 3, 4), (5, 4, 4), (5, 4, 5)]

    chans = load_channels()
    if args.channels:
        chans = chans[:args.channels]

    if args.roi_model == 'synth':
        run_synth_sweep(args, chans, budgets)
        return

    patch = load_patch()
    patch_ev = None
    expect = (0, 0)
    if args.holdout:
        dr, dc = args.holdout
        patch_ev = np.roll(patch, (dr, dc), axis=(0, 1))
        expect = (dr % R, dc % C)

    print(f"geometry {R}x{C}  channels {len(chans)}  H_SHIFT {H_SHIFT}  "
          f"eps_rel {FG.EPS_REL}")
    print(f"patch: {SCENARIO}  range [{patch.min()},{patch.max()}]  "
          f"std {patch.std():.1f}")
    if patch_ev is not None:
        print(f"HELD-OUT eval: filter trained on the patch, applied to a "
              f"circular shift of {tuple(args.holdout)} -> peak must land at {expect}")
    print()

    for variant in args.variant:
        print("=" * 78)
        print(f"VARIANT {variant}")
        print("=" * 78)

        feats, feats_ev, dead, relu_never, bits = [], [], [], [], []
        for oc, (w, shift, bias) in enumerate(chans):
            b, sh, max_ac = variant_params(w, shift, bias, variant)
            if VARIANTS[variant]['relu'] and b + max_ac <= 0:
                dead.append(oc)
            if VARIANTS[variant]['relu'] and b - max_ac >= 0:
                relu_never.append(oc)
            bits.append(np.log2(max(max_ac >> sh, 1)))
            f, _ = conv_variant(patch, w, sh, b, variant)
            feats.append(f)
            if patch_ev is not None:
                fe, _ = conv_variant(patch_ev, w, sh, b, variant)
                feats_ev.append(fe)

        bits = np.array(bits)
        print(f"  structurally dead : {dead if dead else 'none'}")
        print(f"  ReLU never active : {len(relu_never)} of {len(chans)} "
              f"{relu_never if relu_never else ''}")
        print(f"  signal resolution : {bits.min():.1f}-{bits.max():.1f} bits of 15 "
              f"(mean {bits.mean():.1f})")
        fm = np.array([np.abs(f).max() for f in feats])
        print(f"  conv output max|.|: {fm.min():.0f}-{fm.max():.0f} "
              f"(mean {fm.mean():.0f})")
        print()
        print("  budget      accum      %rail   ifftrow    response    %rail  "
              "PSR(Bolme)  ratio   peak@")
        print("  " + "-" * 88)

        for (fs, irs, ics) in budgets:
            resp, accum, ifftrow, rails, b2 = run_pipeline(
                feats, fs, irs, ics, feats_ev if patch_ev is not None else None)
            amax = float(np.abs(accum).max())
            rmax = float(np.abs(resp).max())
            idx, peak, bol, rat = psr(resp)
            flag = ""
            if any(rails[k] for k in ('fft_row', 'fft_col', 'accum',
                                      'ifft_row', 'response')):
                bad = [k for k in rails if rails[k]]
                flag = f"  <-- RAILED: {','.join(bad)}"
            if tuple(int(v) for v in idx) != expect:
                flag += f"  <-- PEAK MISPLACED (want {expect})"
            print(f"  {fs}-{irs}-{ics}   {amax:9.0f}  {100*amax/32767:6.1f}%  "
                  f"{ifftrow:9.0f}  {rmax:9.0f}  {100*rmax/32767:6.1f}%  "
                  f"{bol:9.1f}  {rat:6.2f}  {idx}{flag}")
        print()

    print("=" * 78)
    print("How to read this")
    print("=" * 78)
    print("Pick the budget whose RESPONSE lands near 50% of range with nothing railed.")
    print("The model agrees with hardware to ~6%, and hw_emu measured the response at")
    print("99.69% of the rail under the old 4-2-2 budget, so a target of 'just under")
    print("100%' is not a target — it is a clipped peak one modelling error away.")
    print()
    print("PSR here is trained and evaluated on the same patch, so its ABSOLUTE value")
    print("is optimistic. Compare budgets against each other, not against Bolme's 20-60.")


if __name__ == '__main__':
    main()
