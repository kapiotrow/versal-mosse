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

@thesis sec:metodykaBadan | N-01,N-02,N-03 | The closed-loop offline bench: float64 downstream
  of the features, 3.5 s per 100 frames, and the home of the quantization, pooling and
  init-perturbation arms.
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
# what makes it survive deformation. `docs/thesis/evidence/detector_gain.md` says that is
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


def split_rand(name):
    """'rgb-rand0' -> ('rgb', 0).  'rgb' -> ('rgb', None).  The number is a SEED.

    THE NEGATIVE CONTROL FOR THE FEATURE BANK. Replaces the pretrained conv1
    weights with random Gaussian taps of the SAME per-channel row norms -- so
    the int8 grid, out_shift and bias_acc all see a comparable scale and the
    only thing that changed is which 16 directions of the 27-dim tap space the
    bank spans.

    WHY IT IS WORTH RUNNING. "Would a better network help?" is answered by how
    much the PRETRAINING is worth, and at layer 1 that is measurable directly.
    Held-out Bolme PSR (scripts, 2026-08-29, 4 sequences, RGB path) says random
    matches pretrained -- car1 36.09 vs 37.25 at dt=1, tiger 31.22 vs 30.99,
    nature 83.07 vs 75.51, bolt1 19.29 vs 18.89 -- but held-out PSR is not AR,
    and this project decides feature questions on AR. This arm is that check.

    It doubles as a control on the BENCH: a bench that cannot tell a pretrained
    bank from noise cannot rank two pretrained banks either, so a null here is
    informative in one direction only and a large loss is what validates the
    instrument.

    NOT the same as `participation ratio`, which is why that statistic is not
    the objective: random Gaussian 16x27 scores PR 10.69 against the shipping
    bank's 7.43, and a random ORTHONORMAL basis scores the theoretical maximum
    16.00. PR is maximised by noise, in weight space and (1.99 vs 1.43) on real
    activations too.
    """
    m = re.match(r'^(.*?)-rand(\d+)$', name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def apce(resp):
    """Average Peak-to-Correlation Energy (LMCF, Wang et al. 2017 3.2).

        APCE = (F_max - F_min)^2 / mean( (F - F_min)^2 )

    THE SECOND CONFIDENCE STATISTIC, and the reason it exists here. PSR is
    ALREADY what `PSR_GATE_MIN` tests, so modulating eta by PSR is one
    instrument used twice -- and this project's own rule is that two
    INDEPENDENT instruments beat one instrument twice. APCE reads the whole
    response's shape (how peaked it is against its own floor) where PSR reads
    the peak against a sidelobe set with an 11x11 hole in it, so the two fail
    differently: PSR is blind to a broad pedestal inside the exclusion, APCE is
    blind to where the peak sits.

    NOT PRE-SCREENABLE FROM THE BOARD LOGS. `track.csv` carries psr_bolme and
    no response map, so the PSR-vs-APCE comparison could only ever be made
    here. Its within-run separation is an ASSUMPTION until this bench measures
    it -- PSR's is 0.618 (scratch analysis of runs/vot/0902_1413-l1relu,
    n=230 losing runs).

    ON THE BOARD it is one pass over a 16 KB host-resident response (~22 us,
    0.09% of a 24 ms frame) and folds into compute_psr's existing peak scan,
    because min/max/sum/sum2 all come off one traversal.
    """
    r = np.asarray(resp, dtype=np.float64)
    lo = r.min()
    d = r - lo
    den = float(np.mean(d * d))
    return float((r.max() - lo) ** 2 / den) if den > 0 else 0.0


def split_ceta(name):
    """'rgb-ceta6' -> ('rgb', 0.6).  'rgb-cetaneg' -> ('rgb', -1.0).  else (name, None).

    CONFIDENCE-MODULATED LEARNING RATE -- LMCF's high-confidence update, as a
    SOFT law rather than their hard veto:

        eta_eff = eta * clamp(conf / median(conf over this run's PAST frames),
                              lo, 1.0)

    The returned value IS `lo`, the clamp floor; `-ceta6` is lo = 0.6.

    ONE-SIDED ON PURPOSE (hi = 1.0, eta is only ever REDUCED). The upward half
    walks into measured-bad territory: MOSSE_ETA=0.1 was REJECTED on hardware
    (EAO 0.1960 -> 0.1817, arm_l1relu.md 13) and 0.025 is "much worse". So
    there is no version of this arm that raises eta and is not already refuted.

    WHY A WARM-UP IS REQUIRED, AND WHY IT IS NOT A HACK. Measured on the
    shipping arm (419 runs, 180,125 evaluated frames):

      * the running median is BIASED HIGH early -- median(psr[:k]) reads 1.86x
        the run's settled level at k=1, 1.05x by k=12, 0.98x by k=20. Seeding
        it from frame 1 would cut eta across the board for a spurious reason.
      * and the relative statistic does not SEPARATE early anyway:
        P[doomed < healthy] is 0.608 at frame 1, 0.529 at frame 3 and 0.461 by
        frame 12 -- i.e. worse than useless. ABSOLUTE psr separates those same
        runs at 0.82-0.85, because dividing by a doomed run's own depressed
        median removes exactly the evidence.

    So the early population is NOT addressable by this law. It is also not
    addressable by eta at all: those are INIT failures (61 runs broken one
    frame after filter_init, f1 IoU 0.571 -- robustness_proposals 1), and
    lowering eta on a bad init PRESERVES the bad init. That route is closed
    separately as N-02.

    The warm-up therefore EXEMPTS them, and the exemption is priced rather than
    hidden: N=12 leaves ~17% of all losses untouched, N=20 leaves 24.3%.

    WHAT THE LAW IS WORTH, from the same data -- the within-run dip is real and
    is slightly STRONGER on this arm than on the one robustness_proposals 4
    measured (0.892 vs 1.004 there):

        window                median   frac < 0.6x
        pre-loss (-5..0)       0.808         29.6%
        control (-20..-15)     0.978         11.7%
        P[pre-loss < control] = 0.618, paired 147/230

    That is a WEAK discriminator. At lo = 0.6 the law acts on 29.6% of pre-loss
    frames and 11.7% of control frames; expect a small effect or none.

    `-cetaneg` is the MUTANT and must LOSE: it inverts the law, raising eta
    when confidence is LOW. If it does not lose, the statistic is inert and any
    gain is "perturbing eta", not confidence -- the role `-chrelneg` played for
    channel reliability, and the reason that null was informative.
    """
    m = re.match(r'^(.*?)-ceta(neg|\d+)$', name)
    if not m:
        return name, None
    tok = m.group(2)
    return m.group(1), (-1.0 if tok == 'neg' else int(tok) / 10.0)


def split_smem(name):
    """'rgb-smem8' -> ('rgb', (8, None)).  'rgb-smem8p20' -> ('rgb', (8, 0.20)).

    THE TRAINING-SAMPLE MEMORY -- R-06, and the last untested algorithmic
    candidate after the 2026-09-03 closures.

    THE MECHANISM IT ATTACKS. This tracker keeps ONE exponentially-weighted
    running average (A, B <- (1-eta)*old + eta*new). At eta 0.05 its effective
    memory is ~20 frames, so after ~20 drifting frames the filter is built
    ENTIRELY from drifted crops and nothing anchors it to the target it was
    initialised on. That is exactly the measured failure: ACCEPT 82.0% at median
    PSR 18.83 in the 5 frames before a loss, box moving 1.88 px/frame -- it
    walks off the target CONFIDENTLY (R-06, evidence/robustness_gap.md).
    SRDCF/CSRDCF keep weighted sample SETS instead, and CSRDCF still beats this
    tracker's own float twin by +0.0144 R trim-5 (P=0.018), a gap no arithmetic
    explains.

    THIS IS A MIXTURE, NOT A SELECTOR, and that distinction is what keeps it
    open. The two-filter ensemble is CLOSED (N-22, N-24, O-03), but everything
    closed there used a second filter as a CONFIDENCE signal -- a validator or a
    per-frame selection rule -- and the pure-observer probe refuted the premise
    that peak disagreement predicts a loss (AUC 0.461 frozen / 0.555 slow).
    Nothing there touches combining several samples into ONE filter, which is
    why `roadmap.md` still lists R-06 as untested. Do not re-derive the
    selector result here.

    THE CONTROL IS BUILT IN, and it must be run first. With no pin and M large,
    a weighted sample set IS the running average -- A = SUM w_i a_i with
    w_i <- (1-eta) w_i and w_new = eta is exactly the exponential recursion. So
    `-smem<M>` with a generous M must REPRODUCE the baseline arm, and if it does
    not, the implementation is wrong and no pinned result from it means
    anything.

    `p<pct>` PINS THE INIT SAMPLE at that fraction of total weight. That is the
    anchor: the one sample guaranteed to be on the target, held at a floor the
    exponential decay cannot erode, so a run of drifted frames can no longer
    take the whole memory. It is the treatment.
    """
    m = re.match(r'^(.*?)-smem(\d+)(?:p(\d+))?$', name)
    if not m:
        return name, None
    pin = (float(m.group(3)) / 100.0) if m.group(3) else None
    return m.group(1), (int(m.group(2)), pin)


class SampleMemory:
    """A weighted set of training samples, replacing the single running average.

    Each sample holds the SAME quantities the running average accumulates --
    a_i = conj(G_i) * F_i and b_i = SUM_ch |F_i|^2 -- so the filter built from
    the set is the same estimator, differing only in how weight is distributed
    over history. See split_smem() for why that makes the no-pin case a control.

    COST, stated because it decides whether this can ever reach the board: one
    sample is `channels x rows x cols` complex plus `rows x cols` real -- 2 MB
    per sample at 32ch/64x64 in complex128, against the 2 MB of filter state the
    board already carries. M=8 is 16 MB on the host. Halving to complex64 is
    available if the board ever wants it; it is not done here because fidelity
    matters more than footprint while the question is still "does it work".
    """

    def __init__(self, m, pin, a0, b0):
        self.m, self.pin = m, pin
        self.a = [a0.copy()]
        self.b = [b0.copy()]
        self.w = [1.0]

    def update(self, F, Gt, eta):
        a_new = np.conj(Gt)[None, :, :] * F
        b_new = np.sum(np.abs(F) ** 2, axis=0)
        self.w = [x * (1.0 - eta) for x in self.w]
        self.a.append(a_new); self.b.append(b_new); self.w.append(eta)
        if len(self.w) > self.m:
            # EVICT BY MERGING, NEVER BY DROPPING, and this is the whole design.
            #
            # Dropping the lowest weight DESTROYS MASS: at eta 0.05 a set of M
            # samples retains only 1-(1-eta)^M of the total (34% at M=8), and
            # renormalising what survives silently turns the arm into a much
            # SHORTER memory -- an effective eta near 0.125, which is a measured
            # WORSE setting (0.05 beats 0.125 by +0.0218 R). Screened that way,
            # M=8 lost basketball at frame 21 against never, and the result was
            # about the memory LENGTH, not about keeping a sample set at all.
            #
            # Merging the two lowest weights preserves SUM(w) exactly, so the
            # effective memory length is unchanged and M becomes a pure
            # RESOLUTION knob on old history -- which is the quantity R-06 is
            # actually about. Weights are monotone in age, so this coarsens the
            # oldest samples first, exactly as intended.
            lo = 1 if self.pin is not None else 0
            order = sorted(range(lo, len(self.w)), key=lambda i: self.w[i])
            j, k = order[0], order[1]
            wj, wk = self.w[j], self.w[k]
            tot = wj + wk
            if tot > 0:
                self.a[k] = (wj * self.a[j] + wk * self.a[k]) / tot
                self.b[k] = (wj * self.b[j] + wk * self.b[k]) / tot
            self.w[k] = tot
            del self.a[j]; del self.b[j]; del self.w[j]

    def filter(self):
        w = np.asarray(self.w, dtype=np.float64)
        if self.pin is not None and len(w) > 1:
            # Renormalise the tail to (1-pin) and hold the anchor at pin.
            tail = w[1:].sum()
            w = np.concatenate(([self.pin],
                                w[1:] * ((1.0 - self.pin) / tail) if tail > 0
                                else np.zeros(len(w) - 1)))
        tot = w.sum()
        if tot <= 0:
            return self.a[0], self.b[0]
        w = w / tot
        A = np.zeros_like(self.a[0])
        B = np.zeros_like(self.b[0])
        for wi, ai, bi in zip(w, self.a, self.b):
            if wi == 0.0:
                continue
            A += wi * ai
            B += wi * bi
        return A, B


def split_hq(name):
    """'rgb-hq' -> ('rgb', 'max').  'rgb-hq99.9' -> ('rgb', 99.9).  else (name, None).

    THE BOARD'S FILTER QUANTIZATION, which the float twin does not have and
    which no instrument on the board can see.

    WHY THIS ARM EXISTS. The matched comparison of 2026-09-04 (board SCALE_N=1
    against the float twin, both without a scale filter) puts the fixed-point
    implementation at +0.0102 R trim-5, P(dR<=0)=0.002 -- the largest
    unattributed term left. The RESPONSE path is not the cause: over 180,125
    evaluated frames of runs/vot/0904_1225-l1relu_s1 the response sits at a
    median 15.4% of full scale, i.e. ~12.3 bits of 15, so its quantization
    noise is ~2^-12 of peak.

    H IS THE REMAINING CANDIDATE, and it is invisible to every board
    instrument. `mosse_filter.cpp:publish_packed` normalises H to Q1.15 against
    a SINGLE GLOBAL MAX over all channels and all bins, and its own comment
    records why that is dangerous: a MOSSE filter is SPIKY -- max|H| sits where
    |F| is SMALLEST, because that is where the regularized inverse peaks -- so
    normalizing the peak bin to full scale leaves every informative bin far
    below it. `rails`, `accum_max` and `resp_max` all measure the RESPONSE and
    are blind to this, which is why the shift-budget work could not have found
    it.

    THE CLIPPED VARIANT IS THE CANDIDATE FIX, and it is HOST-ONLY. Normalising
    to a percentile of |H| instead of its max buys every informative bin the
    bits the outlier was hoarding, at the price of saturating the few bins above
    the percentile. It changes only the DATA written to `gmio_cmul_in` -- no
    AIE_FLAGS, no rebuild, no re-flash. A per-CHANNEL scale is NOT available:
    cmul_accum applies ONE shift across the whole accumulation.
    """
    m = re.match(r'^(.*?)-hq([0-9.]+)?$', name)
    if not m:
        return name, None
    return m.group(1), (float(m.group(2)) if m.group(2) else 'max')


def quantize_h_board(H, mode='max'):
    """H -> Q1.15 exactly as mosse_filter.cpp:publish_packed does it.

    Global scale from ONE max over every channel and bin, `nearbyint`, then a
    PER-COMPONENT clamp to +/-32767 (never -32768: cmul_accum computes
    in.re*flt.re + in.im*flt.im in int32, and all four operands at -32768 is
    exactly 2^31, one past INT32_MAX).

    Returns H in LSB units. That is a UNIFORM rescale of the filter, which
    moves neither the argmax nor PSR -- both are scale-free -- so the only
    thing this arm changes is the quantization itself. `mode` is 'max' for the
    board's rule or a percentile of |H| for the clipped candidate.
    """
    mag = np.abs(H)
    peak = float(mag.max())
    if peak <= 0.0:
        return H, (0.0, 0.0, 0.0)
    ref = peak if mode == 'max' else float(np.percentile(mag, mode))
    if not (ref > 0.0):
        ref = peak
    scale = 32767.0 / ref
    # NOT named `re`/`im`: `re` is the regex module this file parses arm names with.
    qre = np.clip(np.rint(H.real * scale), -32767.0, 32767.0)
    qim = np.clip(np.rint(H.imag * scale), -32767.0, 32767.0)
    # DIAGNOSTICS, and they are the point of the first run: how many bits the
    # MEDIAN bin actually receives, the spikiness that decides it, and how much
    # was clipped to buy them.
    med_bits = float(np.log2(max(float(np.median(mag)) * scale, 1e-12)))
    spike = float(peak / max(float(np.median(mag)), 1e-300))
    sat = float(np.mean((np.abs(qre) >= 32767.0) | (np.abs(qim) >= 32767.0)))
    return qre + 1j * qim, (med_bits, spike, sat)


def split_chrel(name):
    """'rgb-chrel10' -> ('rgb', 1.0).  'rgb-chrelneg' -> ('rgb', -1.0).  else (name, None).

    CHANNEL RELIABILITY IN STAGE B3 -- CSR-DCF's third contribution, and the one
    `baselines.md` prices at -12% in their ablation while this tracker has never
    tested it. B3 today normalises each channel to UNIT ENERGY
    (`chscale = 1/sqrt(sum|F|^2)`), i.e. by how LOUD a channel is and not by how
    well it discriminates. A channel whose response is diffuse or peaks in the
    wrong place is weighted identically to one that spikes on the target.

    THE STATISTIC, and why it is free. Per channel, with P = F .* conj(H):

        r_c(0)   = (1/N) SUM_k P_c(k)                 response AT the target
        ||r_c||^2 = (1/N) SUM_k |P_c(k)|^2            total response energy   (Parseval)
        rho_c    = |SUM P|^2 / (N * SUM |P|^2)        fraction of energy on target

    Both are single accumulations over a spectrum the loop ALREADY walks -- one
    complex sum and one magnitude-squared sum per bin, no inverse FFT. That is
    what "free by Parseval" means in the roadmap: the per-channel inverse FFT
    the direct measurement would need is 30.1 ms (the FILTER_MASK_STAT note).

    rho is SCALE-INVARIANT in H, so weighting by rho^gamma does not change rho:
    there is no fixed point to solve and no circularity.

    The weights are normalised to mean 1 so the arm does not also move the global
    response scale -- that would shift PSR and the gate with it, and a confounded
    arm is not an arm. gamma = 0 is exactly today's behaviour, i.e. the null is
    built in; `-chrelneg` (gamma = -1) is the ANTI-RELIABILITY MUTANT and must
    LOSE, or the statistic is inert and the weighting is doing something else.
    """
    m = re.match(r'^(.*?)-chrel(neg|\d+)$', name)
    if not m:
        return name, None
    tok = m.group(2)
    return m.group(1), (-1.0 if tok == 'neg' else int(tok) / 10.0)


L1_KINDS = ('l1lin', 'l1relu', 'l1lin16', 'l1relu16',
            'l5lin', 'l5relu', 'l5lin16', 'l5relu16',
            'danlin', 'danrelu',
            'gablin', 'gabrelu', 'gabrelublur',
            # AGGREGATION ON THE SHIPPING BANK, and its NEGATIVE CONTROL.
            # settled.md refutes aggregation with an ARGUMENT, not only a
            # measurement: "a box average of a linear map is another linear map
            # with the same span, it CANNOT do anything -- a fact about
            # LINEARITY, not about aggregation". `CONV_RELU=1` shipped
            # 2026-09-02, so that argument no longer applies to the map this
            # tracker actually computes. The pair tests exactly that: `l1relublur`
            # must GAIN and `l1linblur` must reproduce the `blur2` null
            # (-0.0010/-0.0012). If both gain, the linearity explanation was
            # wrong; if neither does, aggregation is refuted on its own terms.
            'l1relublur', 'l1linblur')


def split_l1(name):
    """'rgb-l1relu' -> ('rgb', 'l1relu').  else (name, None).

    DANELLJAN'S LAYER 1, on this datapath. His Table 1 fixes the geometry --
    224x224 -> 109x109 at 96 channels is a 7x7 kernel at STRIDE 2 -- and 3.3
    says the activations are taken AFTER the ReLU. Figure 2 puts Layer 0 (the
    raw preprocessed RGB image) at ~45% OP against Layer 1's ~61%.

    WHY THIS IS THE ARM. With CONV_RELU=0 and 16 channels over a 27-dim tap
    space, this tracker IS his Layer 0: the conv is a linear lift the online
    filter absorbs (see feature_bank.md, and the `-eye` control that ties it).
    Pointwise ReLU on the 3x3 bank is refuted, but a 3x3 has no orientation
    selectivity to rectify -- the nonlinearity only has something to encode once
    the kernel is large enough to be oriented. So kernel, stride, channel count
    and the rectifier have to move TOGETHER, and each arm here has its own
    linear twin so the rectifier is isolated AT the new geometry.

    THE CONTROL THAT MATTERS IS `rgb-dec2`, not `rgb`. MOSSE_SIGMA is in BINS
    and the target spans patch/padding bins, so a stride-2 map halves the bin
    count and DOUBLES sigma/target onto the 1/16 optimum -- the free win that
    turned out to be all of the res64 result (sec.25 of arm_res64.md).
    Scoring these against the 128x128 baseline would hand them that win a second
    time. `dec2` is the geometry-matched Layer 0.
    """
    m = re.match(r'^(.*?)-(' + '|'.join(L1_KINDS) + r')$', name)
    return (m.group(1), m.group(2)) if m else (name, None)


def split_bank(name):
    """'rgb-eye' -> ('rgb', 'eye').  'rgb-orth3' -> ('rgb', 'orth3').  else (name, None).

    THE LINEARITY CONTROLS. `CONV_RELU=0` ships, so conv2d is
    `saturate(acc >> out_shift)` and nothing else: the bank is a FIXED LINEAR
    LIFT of the 27-dim (3x3x3) tap space, and the detector the tracker can
    express is

        y = sum_c h_c * (w . sum_k W_ck u_k) = sum_k (sum_c W_ck h_c) * (w . u_k)

    i.e. an element of the ROW SPACE of W and nothing else. The online MOSSE
    filter does the discriminative learning and absorbs any change of basis
    within that span. That predicts `-rand<N>` matching pretrained (measured,
    feature_bank.md) and it makes two sharper predictions these arms test:

      -eye    ONE-HOT taps: no network at all, just 16 of the 27 raw lifted
              planes. If this matches, the conv layer contributes only a choice
              of basis and "CNN features" overstates what the datapath does.
      -orth<N> random ORTHONORMAL rows: the BEST-CONDITIONED lift of the same
              dimension. Basis-invariance is broken only by conditioning terms
              -- the DSST shared denominator, eps_rel, Stage B3's per-channel
              energy normalisation, and per-channel int8 quantization -- so if
              anything beats pretrained it should be this.

    Both match the pretrained PER-CHANNEL ROW NORMS, for the same reason
    `-rand<N>` does: otherwise the arm also moves out_shift and bias_acc and a
    difference is unattributable.

    NOTE the one-hot bank quantizes EXACTLY (a single tap per channel goes to
    +-127), so it carries no quantization penalty. That favours it, and it is
    the right way round: this arm is trying to show the network is unnecessary.
    """
    m = re.match(r'^(.*?)-(eye|orth\d+|crelu|half8|abs)$', name)
    return (m.group(1), m.group(2)) if m else (name, None)


def split_warp(name):
    """'rgb-warp8' -> ('rgb', 8).  'rgb' -> ('rgb', 1).

    A SUFFIX, not a global flag, so one invocation runs the arm and its control
    against the same frames -- the shape every other arm in this file uses.
    """
    m = re.match(r'^(.*?)-warp(\d+)$', name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


def parse_arm(name):
    """'rgb-pool2' -> ('rgb', 2, 'avg').  'gray' -> ('gray', 1, 'avg').

    MAX pooling (`-mpool`, `-relumpool`) was added 2026-08-31 after reading
    Danilowicz & Kryjak 2022: their deepDCF stem is VGG11 conv1 INCLUDING ReLU
    AND 2x2 MAXPOOL, i.e. the aggregation the DCF literature actually ships. Every
    pooling arm in this file before then was an AVERAGE, and averaging a SIGNED
    edge map cancels the lobes (pool2 PSR 30.4 -> 13.4) -- which is a property of
    the average, not of pooling. Max on a rectified map cannot cancel, so
    `relumpool` is the arm that tests the literature's actual choice.
    """
    m = re.match(r'^(.*?)-(relumpool|relupool|relublur|relu|mpool|pool|blur|dec)(\d+)?$', name)
    if not m:
        return name, 1, 'avg'
    kind = m.group(2)
    n = int(m.group(3)) if m.group(3) else 1
    return m.group(1), n, {'pool': 'avg', 'dec': 'dec', 'blur': 'blur',
                           'relupool': 'reluavg', 'relu': 'reluavg',
                           'relublur': 'relublur',
                           'mpool': 'max', 'relumpool': 'relumax'}[kind]


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
    if mode in ('max', 'relumax'):
        # MAX over the cell. On a SIGNED map this is not a symmetric operation --
        # it keeps positive lobes and discards negative ones, which is why it is
        # only meaningful paired with the rectification the literature applies
        # first (relumax). `mpool` without ReLU is kept as the control that
        # separates "max" from "max of a rectified map".
        return ft.reshape(ch, r // n, n, c // n, n).max(axis=(2, 4))
    return ft.reshape(ch, r // n, n, c // n, n).mean(axis=(2, 4))


def arm_doc(name):
    base, n, mode = parse_arm(name)
    return (f"{base}, "
            + ("no pooling" if n == 1 and mode == 'avg' else
               f"{'ReLU + ' if mode.startswith('relu') else ''}"
               + ('ReLU only' if n == 1 else
                  ('subsample' if mode == 'dec' else
                   'box blur (stride 1)' if mode.endswith('blur') else
                   'MAX' if mode.endswith('max') else 'average')
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
# (claim N-02, docs/thesis/evidence/) within 10 frames of init, already at median
# IoU 0.571 and PSR 7.35 one frame
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
def spatial_mask(rows, cols, plateau_frac, taper_frac, centre='board'):
    """Separable raised-cosine mask, centred at the PATCH CENTRE.

    plateau_frac: width of the flat top as a fraction of the patch. 0.5 is
                  exactly the target box at TARGET_PADDING=2.
    taper_frac:   width of the cosine roll-off, same units. 0 gives a hard box,
                  whose DFT is a sinc and therefore NOT sparse -- it is available
                  for offline comparison but is not the board-implementable one.
                  **1.0 IS THE ONLY EXACTLY-9-BIN CASE and is NOT the default of
                  --mask-taper**; at the 0.25 default `mask0` has 99 non-zero
                  bins per axis and is not board-implementable.
    centre:       'board' centres the axis at n/2, giving EXACTLY the periodic
                  Hann sin^2(pi i / n) -- hanning_<N>.h's table, whose DFT is
                  {n/2, -n/4, -n/4} and REAL. 'bench' centres at (n-1)/2, which
                  is what this function did before 2026-08-29 and what
                  runs/vot/0828_offline-mask/mask62_hann9bin.json was swept with.

    WHY THE CENTRING IS AN OPTION AND NOT A DETAIL. Half a sample, max|dm| =
    0.0123 -- and NOT benign per sequence: on `tiger` the bench centring gives
    mean IoU 0.1715 (lost f107) against 0.2813 (lost f360). Over all 62 the
    aggregate holds (dR +0.0718 bench, +0.0601 board) but individual sequences
    are chaotically sensitive to it. 'board' is the DEFAULT because scoring a
    window the hardware cannot produce is how a proxy stops transferring.
    """
    if centre not in ('board', 'bench'):
        raise SystemExit(f"--mask-center must be 'board' or 'bench', got {centre!r}")

    def axis(n):
        if centre == 'board' and plateau_frac == 0.0 and taper_frac >= 1.0:
            # The exact periodic Hann. Written as the closed form rather than
            # as a limit of the general expression below so it is provably the
            # same table conv2d and Stage B2 use.
            return np.sin(np.pi * np.arange(n) / n) ** 2
        off = (n / 2.0) if centre == 'board' else ((n - 1) / 2.0)
        x = np.abs(np.arange(n) - off) / n                # 0 at centre, 0.5 at edge
        p, t = plateau_frac / 2.0, taper_frac / 2.0
        if t <= 0:
            return (x <= p).astype(np.float64)
        r = np.clip((x - p) / t, 0.0, 1.0)
        return np.cos(0.5 * np.pi * r) ** 2
    return np.outer(axis(rows), axis(cols))


def box_energy_fraction(H, box_rows, box_cols):
    """Fraction of SUM|h|^2 inside a CENTRED box of box_rows x box_cols BINS.

    THE MECHANISM CHECK for the spatial mask (arm_mask.md 4): "the
    fraction of the filter's energy inside the target box should rise from the
    measured 51.6% (car1) / 54.9% (tiger). If EAO moves while that does not, the
    gain is not the mask." Those two numbers lived only in a COMMENT above --
    nothing computed them -- so the falsifier had no instrument on either side.

    This is the OFFLINE half; the board's is filter_box_energy_fraction() in
    mosse_filter.cpp, and the pair is the point: it gives the hardware run a
    PREDICTED value to hit rather than only a direction.

    Centred at the PATCH CENTRE, matching the mask and matching the board. The
    filter's energy peaks there, not at resp[0,0]; scoring at the response
    origin reads 8-12% and looks like a destroyed filter.

    H is (channels, rows, cols) in the FREQUENCY domain -- the same array the
    detector uses -- so it is transformed here rather than by the caller.
    """
    h = np.fft.ifft2(H, axes=(1, 2))
    e = np.abs(h) ** 2
    rows, cols = e.shape[1], e.shape[2]
    br = max(0, min(int(round(box_rows)), rows))
    bc = max(0, min(int(round(box_cols)), cols))
    total = float(e.sum())
    if total <= 0.0 or br == 0 or bc == 0:
        return 0.0
    r0, c0 = rows // 2 - br // 2, cols // 2 - bc // 2
    return float(e[:, r0:r0 + br, c0:c0 + bc].sum()) / total


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
            mask_plateau=None, mask_taper=0.25, mask_centre='board',
            mask_power=1, rect=None, chrel_gamma=None, stride=1, blur_n=1,
            pool_n=1, pool_mode_ov=None,
            ceta_lo=None, ceta_stat='psr', ceta_warmup=12, lt_eta=None,
            quant_h=None, smem=None, padding=None, frames=None):
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
    if stride > 1:                       # a strided conv shrinks the map
        fr, fc = fr // stride, fc // stride
    if blur_n > 1:                       # aggregation at stride 1: size unchanged
        pool, pool_mode = blur_n, 'blur'
    if pool_mode_ov is not None:         # a real pool: it shrinks the map again
        pool, pool_mode = pool_n, pool_mode_ov
        fr, fc = fr // pool_n, fc // pool_n
    excl = 5                       # sidelobe exclusion, BINS (11x11 at pool1)
    # FRAME ORDER. `frames` is 1-based ABSOLUTE indices in RUN order; None means
    # the single-start bench, [1 .. n_frames], which is what every screening arm
    # passes and is why the default reproduces those runs exactly.
    #
    # The multistart protocol needs the other two shapes: a forward run
    # [anchor .. n] and a BACKWARD run [anchor .. 1], the second of which visits
    # the sequence in reverse. Everything downstream indexes gt and the image
    # loader by the absolute f, so neither shape needs special-casing -- only
    # the INIT box moves, from gt[0] to the first frame actually visited.
    if frames is None:
        frames = list(range(1, n_frames + 1))
    row, col, bh, bw = gt[frames[0] - 1]
    A = B = None
    mean_prev = None
    # `step` records the measurement itself, per frame, in BINS and in frame px,
    # because the failure under investigation is a detector that reports (0,0)
    # while the target moves several bins -- invisible in IoU or PSR, which both
    # look healthy while it happens.
    rec = {'iou': [], 'cerr': [], 'psr': [], 'holds': 0, 'lost_at': None,
           'step': [], 'resp00': [], 'ebox': [], 'etascale': [], 'ltdiv': [],
           'hq': [], 'bdyn': [],
           # `box` is the TRAJECTORY: (row, col, h, w) per visited frame, in run
           # order, init frame included. The bench scores itself from `iou` and
           # never needed it; the multistart protocol scores the boxes and
           # nothing else. One append per iteration, both branches.
           'box': []}
    # Causal history for the confidence law: appended AFTER each frame's
    # scale is computed, so median() never sees the current frame.
    conf_hist = []
    # LONG-TERM FILTER PROBE -- a PURE OBSERVER. A_lt/B_lt ride the live
    # trajectory (same crops, same shifted training target) and differ only in
    # TEMPORAL SUPPORT: lt_eta = 0 freezes them at the init state. Their peak is
    # compared with the live peak and RECORDED; nothing here ever feeds back
    # into the trajectory, which is what makes inertness the control -- a probe
    # arm's IoU must be bit-identical to the baseline's.
    #
    # WHY THIS PROBE EXISTS (M-13, the prior question). Confidence-derived
    # per-frame statistics are closed as a class (N-22/N-23): they read the
    # CURRENT response map, and pre-loss this tracker looks confident and moving.
    # Disagreement between two models with different memory is the one candidate
    # signal that is NOT inside a single response map. If the peaks do not
    # diverge before the loss, the two-filter ensemble (O-03) is dead too.
    A_lt = B_lt = None
    mem = None                 # the training-sample memory (R-06), see split_smem
    # Built once: the mask is fixed in PATCH coordinates, so it does not move
    # with the box. Its fixedness is the whole reason it is host-only and free.
    # mask_power k: the mask is m**k. THIS IS A REAL, BOARD-IMPLEMENTABLE WIDTH
    # KNOB, and the docs said there was none. A single periodic Hann is forced
    # (only it has an exactly sparse spectrum), but m**k is ALSO exactly sparse
    # -- 2k+1 bins per axis, since sin^2 raised to k is a cosine polynomial of
    # degree k -- and multiplying by m**k is EXACTLY applying the board's
    # existing projection k times. Verified against an exact FFT to 4.9e-16, so
    # k costs 8k complex adds per bin, no multiplies, and NO NEW BOARD CODE.
    # Effective width, fraction of mask energy in the centred 64x64 box:
    # k=1 0.6695, k=2 0.8544, k=3 0.9347.
    mask = (None if mask_plateau is None
            else spatial_mask(fr, fc, mask_plateau, mask_taper, mask_centre)
                 ** mask_power)

    for f in frames:
        if oracle_scale:
            # AN EMPTY GROUNDTRUTH CARRIES NO SIZE, so hold the previous one.
            # stb2022 has 41 zero-size annotations over 12 sequences (agility,
            # girl, tennis, soldier ...). Adopting the zero collapses the ROI,
            # Stage A then divides by zero, and the NaN surfaces hundreds of
            # frames later as "cannot convert float NaN to integer" -- nowhere
            # near its cause. The bench never hit this because car1, its default,
            # has none; the multistart harness runs all 62 and found it.
            # VOT's own failure rule skips empty-groundtruth frames too.
            g = gt[f - 1]
            if g[2] > 0 and g[3] > 0:
                bh, bw = g[2], g[3]
        # PADDING IS A KNOB, NOT A CONSTANT, and it is coupled to MOSSE_SIGMA.
        # The target spans `map / padding` BINS, so sigma/target = sigma*padding
        # / map: moving padding silently moves the MAINLOBE WIDTH, which R-11
        # identifies as THE axis with an optimum at 1/16. The 2026-08-28
        # hardware refutation of padding 3.0 ran at sigma 2.0, i.e.
        # sigma/target = 1/10.7 -- 1.5x too wide -- so it moved two magnitudes
        # at once. Pass `padding` and `sigma` together or the arm is not a
        # padding arm. See split_smem()'s note on M-14 for the general rule.
        pad = PADDING if padding is None else padding
        roi_h, roi_w = int(round(bh * pad)), int(round(bw * pad))
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
                                       relu=pool_mode in ('reluavg', 'relublur', 'relumax'),
                                       rect=rect, stride=stride)
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
            # The observer starts from the SAME init. With lt_eta = 0 it stays
            # there for the whole run: the filter the tracker had at frame 1,
            # carried forward untouched.
            if lt_eta is not None:
                A_lt, B_lt = A.copy(), B.copy()
            if smem is not None:
                # Seeded from the init state, AFTER the warp normalisation, so
                # the anchor is exactly the filter the tracker had at frame 1 --
                # the one sample guaranteed to sit on the target.
                mem = SampleMemory(smem[0], smem[1], A, B)
            rec['ltdiv'].append(float('nan'))     # no detection on the init frame
            rec['iou'].append(box_iou((row, col, bh, bw), gt[f - 1]))
            rec['cerr'].append(float(np.hypot(row - gt[f-1][0], col - gt[f-1][1])))
            rec['psr'].append(float('nan'))
            rec['box'].append((row, col, bh, bw))
            continue

        # detect
        energy = np.sum(np.abs(F)**2, axis=(1, 2))
        chscale = np.where(energy > 0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 0.0)
        H = A * chscale[:, None, None] / (B + eps_rel * B.mean())[None]
        if chrel_gamma is not None:
            # Stage B3 channel reliability. See split_chrel(): rho is the
            # fraction of each channel's response energy sitting AT the target,
            # both terms Parseval sums over the spectrum this loop already has.
            P = F * np.conj(H)
            num = np.abs(P.sum(axis=(1, 2))) ** 2
            den = (np.abs(P) ** 2).sum(axis=(1, 2)) * float(P[0].size)
            rho = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
            w = np.power(np.maximum(rho, 1e-12), chrel_gamma)
            w /= w.mean()          # keep the global response scale, and the gate with it
            H = H * w[:, None, None]
        if mask is not None:
            # h <- m*h. EXACT FFTs here on purpose: offline answers "does the
            # projection help at all", and the 9-bin sparse-spectrum form is a
            # BOARD implementation detail whose approximation error is a separate
            # question. Testing the approximation before the idea would confound
            # the two.
            H = np.fft.fft2(np.fft.ifft2(H, axes=(1, 2)) * mask[None], axes=(1, 2))
        # The mechanism check, on the SAME H the detector is about to use -- so
        # it reflects the mask if one is applied and the baseline if not. The
        # box is measured in bins from THIS frame's resample ratio, so it tracks
        # the box size instead of assuming the initial one.
        # DENOMINATOR CONDITIONING -- the quantity Bolme 3.3's perturbation
        # argument turns on. The 2026-08-28 refutation of init perturbations
        # rests on "bins below 1e-6*mean(B) are 0.00%", measured on the 3x3
        # mobilenet bank at 16 channels. Both the bank and the channel count
        # moved on 2026-09-02, and a RECTIFIED 7x7/2 map is far more low-pass,
        # so its spectrum is DC-heavy and its high bins are relatively smaller.
        # Logged per frame so the claim can be re-checked on any bank.
        _bm = float(np.mean(B))
        if _bm > 0:
            rec['bdyn'].append((float(np.min(B)) / _bm,
                                float(np.percentile(B, 1)) / _bm,
                                float(np.mean(B < 1e-6 * _bm)),
                                float(np.mean(B < 1e-3 * _bm))))

        if quant_h is not None:
            # LAST, after chrel and the mask: on the board cmul_accum only ever
            # sees the PUBLISHED H, so everything downstream here -- ebox and
            # the response alike -- must see the quantized one too. A and B stay
            # float, as they do on the board.
            H, hstat = quantize_h_board(H, quant_h)
            rec['hq'].append(hstat)

        rec['ebox'].append(box_energy_fraction(H, bh * fr / roi_h, bw * fc / roi_w))

        resp = np.real(np.fft.ifft2(np.sum(F * np.conj(H), axis=0)))
        idx, peak, bolme, _ratio = metrics_rc(resp, fr, fc, excl)
        dr, dc = wrap(idx[0], fr), wrap(idx[1], fc)

        # LONG-TERM PROBE (observer only -- see A_lt above). Same F, same
        # chscale, same regularizer: the ONLY difference is how much history the
        # filter carries. Distance is CIRCULAR in bins, because a response map
        # wraps and a naive |a-b| would call a 1-bin disagreement across the
        # wrap a 63-bin one.
        if lt_eta is not None and A_lt is not None:
            H_lt = A_lt * chscale[:, None, None] / (B_lt + eps_rel * B_lt.mean())[None]
            resp_lt = np.real(np.fft.ifft2(np.sum(F * np.conj(H_lt), axis=0)))
            i_lt = np.unravel_index(np.argmax(np.abs(resp_lt)), resp_lt.shape)
            ddr = wrap(i_lt[0], fr) - dr
            ddc = wrap(i_lt[1], fc) - dc
            rec['ltdiv'].append(float(np.hypot(ddr, ddc)))
        else:
            rec['ltdiv'].append(float('nan'))

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
            # CONFIDENCE-MODULATED LEARNING RATE -- see split_ceta(). The scale
            # is computed from the run's OWN PAST frames only; conf_hist is
            # appended after, so the current frame can never normalise itself.
            eta_eff = eta
            if ceta_lo is not None:
                conf = bolme if ceta_stat == 'psr' else apce(resp)
                if len(conf_hist) >= ceta_warmup:
                    med = float(np.median(conf_hist))
                    if med > 0:
                        r_conf = conf / med
                        if ceta_lo < 0:      # -cetaneg MUTANT: invert the law
                            eta_eff = eta * min(max(1.0 / max(r_conf, 1e-6),
                                                    0.6), 1.0)
                        else:
                            eta_eff = eta * min(max(r_conf, ceta_lo), 1.0)
                conf_hist.append(conf)
            rec['etascale'].append(eta_eff / eta if eta else 1.0)
            # The observer rides the live trajectory: same F, same shifted Gt.
            # lt_eta = 0 leaves it frozen at the init state (filter_update with
            # eta 0 is the identity, so this is written out rather than special-
            # cased -- one code path, no branch that could differ).
            if lt_eta is not None and A_lt is not None:
                A_lt, B_lt = FG.filter_update(A_lt, B_lt, F, Gt, lt_eta)
            if mem is not None:
                # THE SAMPLE SET REPLACES THE RUNNING AVERAGE. With no pin and a
                # generous M this is arithmetically the same recursion, which is
                # what makes `-smem<M>` the control -- see split_smem().
                mem.update(F, Gt, eta_eff)
                A, B = mem.filter()
            else:
                A, B = FG.filter_update(A, B, F, Gt, eta_eff)
        else:
            rec['holds'] += 1        # hold position AND skip the update, both
            # A vetoed frame does not update, so it does not enter the history
            # either -- the median must describe frames the filter LEARNED from.

        iou = box_iou((row, col, bh, bw), gt[f - 1])
        rec['box'].append((row, col, bh, bw))
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


# ---------------------------------------------------------------------------
# THE ARM DISPATCH -- ONE COPY, and it has to stay that way.
#
# main() and scripts/offline_multistart.py both resolve arm names through this.
# A second copy is how `rgb-l1relu` on the bench and `rgb-l1relu` under the
# multistart protocol quietly become two different configurations -- the exact
# failure l1_banks.py exists to prevent between the bench and `make weights`.
# Extracted verbatim from main() 2026-09-04 and checked bit-identical on
# rgb / rgb-l1relu / rgb-eye / rgb-dec2.
# ---------------------------------------------------------------------------
def resolve_arm(a, W, w_rgb, w_gray, b_fold, quiet=False):
    """Arm name -> everything run_arm needs that is not a CLI knob.

    Returns a dict with the run_arm keyword names, plus `stem` (the positional
    arm string), `base` (for the FLOATW lookup) and `bank` (the (wq, bias,
    shift) triple run_arm takes splatted).
    """
    _p = (lambda *x, **k: None) if quiet else print
    stem, n_warps = split_warp(a)
    stem, mask_plateau = split_mask(stem)
    stem, rand_seed = split_rand(stem)
    stem, bank_kind = split_bank(stem)
    stem, smem = split_smem(stem)
    stem, quant_h = split_hq(stem)
    stem, chrel_gamma = split_chrel(stem)
    stem, ceta_lo = split_ceta(stem)
    stem, l1_kind = split_l1(stem)
    rect = None
    stride, blur_n = 1, 1
    pool_n, pool_mode_ov = 1, None
    base, pool, mode = parse_arm(stem)
    if base not in W:
        sys.exit(f"unknown arm '{a}' (base '{base}'): "
                 f"expected one of {sorted(W)} with an optional "
                 f"-pool<N> / -dec<N> / -warp<N> suffix")
    if pool > 1 and (R % pool or C % pool):
        sys.exit(f"{a}: pool {pool} does not divide the {R}x{C} patch")
    # The random bank is built HERE, per arm, so the pretrained arm in the
    # same invocation is untouched and both see identical frames.
    if l1_kind is not None:
        import l1_banks
        if l1_kind.startswith('dan'):
            # DANILOWICZ & KRYJAK's stem: 3x3 conv, ReLU, 2x2 MAXPOOL --
            # NOT a 7x7. Their Table 1 (docs/papers/danilowicz2022_embedded_dcf.pdf,
            # VOT2015 -- NOT comparable to this project's STb2022 numbers)
            # reaches EAO 0.184 with it at exactly
            # this project's 128x128 ROI / 64x64 map, and 8 channels ties
            # 32. So the kernel does not have to grow for the nonlinearity
            # to have something to act on; the maxpool is what makes the
            # rectified map useful, and an AVERAGE over a signed map (this
            # project's `blur2`) is the one aggregation that cannot work.
            fw, bf = l1_banks.vgg16_conv1_pca(16)
            label = 'vgg16 conv1 3x3 -> 16ch (VGG11 stand-in) + 2x2 MAXPOOL'
            stride, pool_n, pool_mode_ov = 1, 2, 'max'
        elif l1_kind.startswith('l5'):
            # 5x5 STRIDE 1: the map stays 128x128, the geometry hardware
            # prefers. Needs MOSSE_SIGMA=4 to hold sigma/target at 1/16 --
            # pass --sigma 4, or the arm is scored at half the mainlobe and
            # the comparison is against the wrong control.
            n_ch = 16 if l1_kind.endswith('16') else 32
            fw, bf = l1_banks.resnet18_conv1_5x5(n_ch)
            label = f'resnet18 conv1 -> 5x5 CENTRE CROP, stride 1, {n_ch}ch'
            stride, pool_n, pool_mode_ov = 1, 1, None
        elif l1_kind.startswith('l1'):
            n_ch = 16 if l1_kind.endswith('16') else 32
            fw, bf = l1_banks.resnet18_conv1_pca(n_ch)
            label = f'resnet18 conv1 7x7/2, 64 -> {n_ch} by weight PCA'
            stride, pool_n, pool_mode_ov = l1_banks.STRIDE, 1, None
        else:
            fw, bf = l1_banks.gabor_bank()
            label = ('analytic GABOR 7x7/2, 16 filters + negations = 32 ch'
                     ' -- DIAGNOSTIC ONLY, hand-crafted taps are outside'
                     ' this project\'s conv-feature requirement')
            stride, pool_n, pool_mode_ov = l1_banks.STRIDE, 1, None
        rect = 'relu' if 'relu' in l1_kind else None
        blur_n = 2 if l1_kind.endswith('blur') else 1
        bank = quantize(fw, bf)
        _p(f"    *** LAYER-1 BANK: {label}, stride {stride}, "
              f"rect={rect}, blur={blur_n} ***")
    elif bank_kind is not None:
        src = w_gray if base.startswith('gray') else w_rgb
        n_src = np.linalg.norm(src.reshape(src.shape[0], -1), axis=1)
        nout, ntap = src.shape[0], int(np.prod(src.shape[1:]))
        if bank_kind in ('crelu', 'half8', 'abs'):
            # THE NONLINEARITY ARMS. `-abs` keeps the pretrained bank and
            # swaps the rectifier. `-crelu` is 8 filters AND THEIR
            # NEGATIONS, so the existing half-wave rectifier emits both
            # halves and the map spans {linear, |.|} at 8 directions
            # instead of 16 -- and it is board-implementable as a WEIGHTS
            # file plus CONV_RELU=1, no graph change. `-half8` is its
            # control: the same 8 directions, DUPLICATED rather than
            # negated, and no rectifier, so it prices the span 16->8 loss
            # on its own. Without it a crelu result is unreadable between
            # "the nonlinearity helped" and "halving the bank hurt".
            if bank_kind == 'abs':
                fw, bf, rect = src.copy(), b_fold.copy(), 'abs'
                label = 'pretrained bank, FULL-WAVE rectifier (abs)'
            else:
                h = nout // 2
                sgn = -1.0 if bank_kind == 'crelu' else 1.0
                fw = np.concatenate([src[:h], sgn * src[:h]], axis=0)
                bf = np.concatenate([b_fold[:h], sgn * b_fold[:h]])
                rect = 'relu' if bank_kind == 'crelu' else None
                label = ('8 filters + THEIR NEGATIONS, ReLU on (CReLU)'
                         if bank_kind == 'crelu' else
                         '8 filters DUPLICATED, no rectifier (span-8 control)')
            bank = quantize(fw, bf)
            _p(f"    *** BANK REPLACED: {label} ***")
        elif bank_kind == 'eye':
            # Cycle the colour planes fastest so the 16 chosen coordinates
            # are not all one plane: tap index = spatial*in_ch + plane.
            flat = np.zeros((nout, ntap))
            order = [sp * src.shape[1] + pl
                     for sp in range(src.shape[2] * src.shape[3])
                     for pl in range(src.shape[1])]
            for i in range(nout):
                flat[i, order[i % ntap]] = 1.0
            label = 'ONE-HOT (identity lift, no network)'
        else:
            seed = int(bank_kind[4:])
            q, _ = np.linalg.qr(np.random.default_rng(seed)
                                .standard_normal((ntap, nout)))
            flat = q.T                      # orthonormal ROWS, 16 x 27
            label = f'random ORTHONORMAL, seed {seed}'
        if bank_kind not in ('crelu', 'half8', 'abs'):
            flat *= n_src[:, None] / np.linalg.norm(flat, axis=1)[:, None]
            bank = quantize(flat.reshape(src.shape), b_fold)
            _p(f"    *** BANK REPLACED: {label}, per-channel row norms "
                  f"matched to the pretrained bank ***")
    elif rand_seed is None:
        bank = W[base]
    else:
        src = w_gray if base.startswith('gray') else w_rgb
        rng = np.random.default_rng(rand_seed)
        rw = rng.standard_normal(src.shape)
        # MATCH THE PER-CHANNEL ROW NORMS. Without this the arm also moves
        # out_shift and bias_acc, and a loss would be unattributable
        # between "the weights are random" and "the fixed-point scale moved".
        n_src = np.linalg.norm(src.reshape(src.shape[0], -1), axis=1)
        n_rw = np.linalg.norm(rw.reshape(rw.shape[0], -1), axis=1)
        rw *= (n_src / n_rw)[:, None, None, None]
        bank = quantize(rw, b_fold)
        _p(f"    *** RANDOM BANK, seed {rand_seed}: pretrained conv1 "
              f"REPLACED by Gaussian taps of matched row norms ***")
    return dict(stem=stem, base=base, bank=bank, n_warps=n_warps,
                mask_plateau=mask_plateau, chrel_gamma=chrel_gamma,
                ceta_lo=ceta_lo, rect=rect, stride=stride, blur_n=blur_n,
                pool_n=pool_n, pool_mode_ov=pool_mode_ov, quant_h=quant_h,
                smem=smem)


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
    ap.add_argument('--mask-power', type=int, default=1,
                    help='apply the mask m**k, i.e. the board projection k times.\n'
                         'Exactly sparse for every k (2k+1 bins per axis), so k is\n'
                         'a board-implementable WIDTH knob needing no new code.')
    ap.add_argument('--lt-probe', type=float, default=None, metavar='ETA',
                    help='LONG-TERM FILTER PROBE, a PURE OBSERVER: keep a second\n'
                         'filter at this eta (0 = frozen at init) riding the\n'
                         'live trajectory, and record the CIRCULAR distance in\n'
                         'bins between its peak and the live one. It never feeds\n'
                         'back, so a probe run must be BIT-IDENTICAL to the same\n'
                         'arm without it -- that is the control. Answers the M-13\n'
                         'prior question for the two-filter ensemble (O-03):\n'
                         'confidence statistics are closed as a class (N-22/N-23)\n'
                         'because they read one response map, and disagreement\n'
                         'between two memories is the one signal that is not in\n'
                         'one. Costs one extra inverse FFT per frame.')
    ap.add_argument('--ceta-stat', default='psr', choices=['psr', 'apce'],
                    help="confidence statistic for a -ceta<N> arm (default\n"
                         "psr). apce is the INDEPENDENT one -- psr is already\n"
                         "what PSR_GATE_MIN tests, so modulating by it is one\n"
                         "instrument used twice. APCE's separation is an\n"
                         "ASSUMPTION until measured here; it is not in\n"
                         "track.csv, so no board log can pre-screen it.")
    ap.add_argument('--ceta-warmup', type=int, default=12,
                    help='frames of causal history before a -ceta<N> arm\n'
                         'modulates (default 12). Measured on the shipping\n'
                         'arm: median(psr[:k])/settled is 1.86 at k=1 and\n'
                         '1.05 by k=12, and the relative statistic does not\n'
                         'separate doomed from healthy runs early anyway\n'
                         '(0.608 at f1, 0.461 by f12). N=12 exempts ~17% of\n'
                         'all losses, N=20 exempts 24.3%. Those are INIT\n'
                         'failures, which eta cannot fix -- see split_ceta().')
    ap.add_argument('--mask-center', default='board', choices=['board', 'bench'],
                    help="where the -mask<N> axis is centred (default board).\n"
                         "board = n/2, the EXACT periodic Hann and the only\n"
                         "window the hardware can apply; bench = (n-1)/2, what\n"
                         "this script did before 2026-08-29 and what\n"
                         "mask62_hann9bin.json was swept with. Worth mean IoU\n"
                         "0.1715 vs 0.2813 on `tiger` -- see spatial_mask().")
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
        R_ = resolve_arm(a, W, w_rgb, w_gray, b_fold)
        stem, base, bank = R_['stem'], R_['base'], R_['bank']
        n_warps, mask_plateau = R_['n_warps'], R_['mask_plateau']
        chrel_gamma, ceta_lo = R_['chrel_gamma'], R_['ceta_lo']
        rect, stride, blur_n = R_['rect'], R_['stride'], R_['blur_n']
        pool_n, pool_mode_ov = R_['pool_n'], R_['pool_mode_ov']
        out[a] = run_arm(stem, *bank, gt, n, args.oracle_scale, args.verbose,
                         float_conv=FLOATW.get(base),
                         sigma=args.sigma if args.sigma else SIGMA,
                         eta=args.eta, psr_min=args.psr_min, n_warps=n_warps,
                         warp_shift=args.warp_shift, warp_scale=args.warp_scale,
                         warp_mutant=args.warp_mutant,
                         warp_aspect=args.warp_aspect, warp_rot=args.warp_rot,
                         eps_rel=args.eps_rel, mask_plateau=mask_plateau,
                         mask_taper=args.mask_taper,
                         mask_centre=args.mask_center,
                         mask_power=args.mask_power, rect=rect,
                         chrel_gamma=chrel_gamma, stride=stride, blur_n=blur_n,
                         ceta_lo=ceta_lo, ceta_stat=args.ceta_stat,
                         ceta_warmup=args.ceta_warmup, lt_eta=args.lt_probe,
                         pool_n=pool_n, pool_mode_ov=pool_mode_ov)

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
          f"{'cerr mean':>10} {'cerr max':>9} {'PSR mean':>9} {'holds':>6} {'lost at':>8}"
          f" {'e_box':>7}")
    print("-" * 94)
    for a in args.arms:
        r = out[a]
        iou = np.array(r['iou']); ce = np.array(r['cerr'])
        psr = np.array(r['psr'])[1:]
        print(f"{a:<13} {iou.mean():9.4f} {iou.min():7.4f} "
              f"{100*np.mean(iou >= 0.5):6.1f}% "
              f"{ce.mean():10.2f} {ce.max():9.2f} {np.nanmean(psr):9.2f} "
              f"{r['holds']:6d} "
              f"{(str(r['lost_at']) if r['lost_at'] else 'never'):>8}"
              # e_box: filter energy inside the target box. The MECHANISM CHECK
              # for a -mask arm -- it must RISE, or the gain is not the mask.
              f" {np.mean(r['ebox']) if r['ebox'] else float('nan'):7.4f}")

    if args.json:
        import json
        seq = args.sequence or 'car1'
        blob = {}
        if os.path.exists(args.json):
            with open(args.json) as fh:
                blob = json.load(fh)
        for a in args.arms:
            # 'ebox' is ADDITIVE -- every existing reader selects 'iou' by key,
            # and the stored sweeps that predate it simply lack it. It is the
            # mask's mechanism check and has to be persistable, or the board run
            # has a direction to hit and no value.
            blob[f"{seq}|{a}"] = {'iou': [float(x) for x in out[a]['iou']],
                                  'ebox': [float(x) for x in out[a]['ebox']],
                                  # THE MECHANISM CHECK for a -ceta<N> arm: the
                                  # per-frame eta multiplier. An arm that moves
                                  # AR while this is flat did not move it by
                                  # modulating eta -- the rule mask_ebox exists
                                  # for (spatial_mask.md), applied here.
                                  'etascale': [float(x) for x in out[a]['etascale']],
                                  'ltdiv': [float(x) for x in out[a]['ltdiv']]}
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
