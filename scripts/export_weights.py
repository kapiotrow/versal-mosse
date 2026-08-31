#!/usr/bin/env python3
"""
scripts/export_weights.py

Extract and INT8-quantize the first convolutional layer of MobileNetV3-Small
for the AIE conv2d_kernel (versal-mosse project).

Model:  torchvision.models.mobilenet_v3_small  (pretrained on ImageNet)
Layer:  features[0] = Conv2dNormActivation(Conv2d(3→16, 3×3), BN, Hardswish)

Processing steps
----------------
1. Fold BatchNorm into conv weights/bias.
2. --in-ch 1 (default): collapse RGB (3 input channels) → grayscale (1 channel)
   via ITU-R BT.601 luminance coefficients LUM = [0.2989, 0.5870, 0.1140].
   --in-ch 3: no collapse — export all 27 taps for the RGB conv2d path.
3. Per-output-channel symmetric INT8 weight quantization (over ALL taps of the
   channel, so the three planes share one scale — see --in-ch below).
4. Compute per-channel integer bias and output right-shift for the AIE accumulator.
5. Pack into 64-byte buffers and write outputs.
6. Write hanning_128.h into design/aie_src/ (precomputed Q1.15 window table).

--in-ch 3 — the RGB export
--------------------------
RGB is a ROBUSTNESS change, not an accuracy one: measured offline over 16 VOT
sequences it takes supervised failures 51 → 42 (−18%) with accuracy moving only
+0.016, and a colour-free control (the same 27 taps fed three identical
luminance planes) reproduces the grayscale arm, so the win is colour and not
bookkeeping. See the RGB section of CLAUDE.md for the full table and costs.

The quantization scale is per OUTPUT channel, taken over all 27 taps at once —
NOT per plane. A per-plane scale would renormalize the three planes against each
other and destroy the chromatic contrast the export exists to preserve, which is
the same trap Stage A's normalization has to avoid in roi_crop.

ACC_MAX_THEORY triples with the tap count, so out_shift rises by ~0.6 bits on
average and RGB carries slightly fewer signal bits per channel. That is real but
it is NOT the reason RGB underperformed in one offline arm: forcing gray's
shifts onto RGB made it worse (42 → 53 failures) at 0.0000% saturation. Do not
"fix" the shift.

--bias-scale — a KNOWN DEFECT, kept as the default on purpose
------------------------------------------------------------
bias_acc converts the folded float bias into accumulator units, and that needs
the scale of the activations the bias actually meets. This script has always
used 127 (i.e. "int8 full scale ≙ 1.0"), but roi_crop emits a z-score at
ROI_NORM_Q = 32 per sigma, so the bias is ~4x oversized relative to the
activations — and since out_shift is derived from |bias_acc| + ACC_MAX_THEORY,
an oversized bias shifts the SIGNAL down to make room for it. Measured cost:
7.6–13.0 of 15 bits of signal resolution (scripts/check_collapse.py Q3).

  --bias-scale 127   (default)  what has shipped and what every hardware
                                measurement in CLAUDE.md was taken with
  --bias-scale roi              b_fold * ROI_NORM_Q / scale, the corrected form

The correction is NOT a free win and must not be applied alone: held-out
peak/max-sidelobe is 12.82 for base(ReLU), 3.92 for bias-corrected(ReLU) and
16.25 for bias-corrected(no ReLU). It only pays with CONV_RELU=0, which is the
shipping configuration. The offline RGB harness (scripts/rgb_vs_gray_*.py) uses
the CORRECTED bias, so a board run meant to reproduce those numbers wants
--bias-scale roi on BOTH arms; a board run meant to be compared against the
existing hardware baseline wants the default on both. Do not mix the two across
arms — that is two magnitudes moving at once.

Changing the bias scale changes the effective input scale, so it obliges a
shift-budget re-sweep over >= 20 hardware frames. See CLAUDE.md.

The Hardswish activation is omitted.  The MOSSE correlation filter is linear in
feature space — no activation is needed at this stage; H_ch* adapts to any scale.

Grayscale collapse rationale — the original one was WRONG (corrected 2026-08-12)
--------------------------------------------------------------------------------
This docstring used to claim: "The PLIO stream carries exactly
PATCH_ROWS×PATCH_COLS int8 samples per channel invocation (128-bit beats, 16
samples/beat). An RGB layout (3 bytes/pixel) does not divide evenly into 16-byte
beats at 128×128 resolution."  Both halves are false:

  * The PLIO is 32-BIT, not 128-bit.  mosse_graph.h:121 creates it with
    `plio_32_bits`, and lines 118-120 record why it was changed away from 128
    bits ("a 128-bit PLIO delivered one 128-bit beat per readincr, starving the
    kernel").  roi_crop's port is `ap_axiu<32,0,0,0>`.  "16 samples/beat" is
    stale.
  * RGB divides EXACTLY anyway.  128*128*3 = 49152 B = 3072 x 16 B = 12288 x 4 B,
    and one row is 384 B = 24 beats = 96 words.  Also exact at 64x64 and 256x256.

So there is no alignment obstacle to RGB.  The real reasons to stay grayscale are
conv2d's scalar 3x3 MAC loop (9 -> 27 MACs/pixel, ~2.2x its cycles) and the fact
that Stage A's mean/L2 reductions would have to become JOINT across the three
planes.  Both are tractable; see the RGB feasibility section in CLAUDE.md.
Do not re-add the divisibility argument.

The cost of staying grayscale is measured and non-trivial: the collapse
degenerates the 16-channel bank to 14 independent filters.  See the LUM-vs-SUM
note below and the Known Issues entry in CLAUDE.md.

Why LUMINANCE weights and not the unweighted sum
-------------------------------------------------
Danelljan §3.3 says "for grayscale images, we simply set the R, G and B-channels
equal to the grayscale intensities".  With x_R = x_G = x_B = g the network
computes  y = Σ_k g[k] * Σ_ic w[ic,k], so the paper-exact collapse is the
UNWEIGHTED sum Σ_ic w[ic].  This code deliberately does NOT do that.

Measured on the real pretrained weights (2026-08-12): the unweighted sum
ANNIHILATES the colour-opponent channels, because those have w_R ≈ -w_G and
cancel on summation.  Collapsed norm as a fraction of the mean per-plane norm:

    ch     SUM     LUM     cos(sum,lum)   cos(R,G)
     0    0.049   0.318      -0.967       -0.9997
     2    0.029   0.599      -0.018       -0.996
     9    0.049   0.634      -0.634       (cos(R,B) -0.98)
    10    0.025   0.037      +0.594       -0.994

Per-channel int8 quantization then divides by max|w_gray[oc]|, so SUM would
amplify a 2.5-5% cancellation residue back to full ±127 scale.  The other 11
channels are achromatic and agree between the two conventions to cos > 0.99
(int8 L1 difference <= 48 of a possible ~1140), so the convention only matters
for the opponent channels — and on those luminance is strictly better.

Magnitude is NOT a consideration either way: the LUM form is ~2.2x smaller, but
quantize_weights() normalizes per output channel, so the scale is absorbed
entirely into scales[oc].

DO NOT "fix" this to the paper-literal sum.  Reproduce the measurement first:
scripts/check_collapse.py.

Input contract for the AIE kernel
----------------------------------
  x_int8  in [-127, 127]  representing  x_gray_float = x_int8 / 127
  where x_gray_float is the ImageNet-normalized luminance:
    x_gray_float = (0.2989*R/255 + 0.5870*G/255 + 0.1140*B/255 - lum_mean) / lum_std
    lum_mean = LUM @ IMAGENET_MEAN   lum_std = LUM @ IMAGENET_STD  (approx)

  roi_crop / host should quantize as:
    x_int8 = clip(round(x_gray_float * 127), -127, 127)

AIE kernel computation model
-----------------------------
  acc       = Σ_{kr,kc} w_int8[kr][kc] * x_int8[kr][kc]   (int32, 9 terms)
  full_acc  = acc + bias_acc                                (int32)
  out_int16 = saturate_int16(full_acc >> out_shift)
  cint16    = {real=out_int16, imag=0}

Float reconstruction (for validation):
  y_approx  = out_int16 * dequant_scale * (1 << out_shift)

where  dequant_scale = scale / 127  and  scale = max(|w_gray[oc]|) / 127.

64-byte weight buffer layout
----------------------------
NOT defined here. The layout lives in design/aie_src/conv_weight_layout.h and is
mirrored, formula for formula, in scripts/conv_weight_layout.py, which this
script packs through. It is derived from the tap count, so:

  gray (9 taps)   taps [0:9)   shift 9   bias 10  dequant 14  mean_prev 18
  RGB  (27 taps)  taps [0:27)  shift 27  bias 28  dequant 32  mean_prev 36

and byte 63 carries the layout tag (= CONV_IN_CH) so a reader can assert rather
than guess. RGB's 27 taps overrun ALL FOUR grayscale fields, silently, which is
why the offsets stopped being hardcoded.

mean_prev is written by the HOST, not here: it is seeded from
bias_acc >> out_shift before frame 0 and rewritten every frame after.

Outputs (under design/aie_src/weights/ by default)
-----------------------------------------------------
  layer0_weights.bin   16 × 64 = 1024 bytes — load into weights_bo
  layer0_meta.npz      numpy archive for validation
  layer0.h             C header (shift per channel, dequant scales)

Also writes:
  design/aie_src/hanning_128.h   precomputed Q1.15 Hanning window (PATCH_SIZE × 2 bytes)

Usage
-----
  uv run --extra weights python3 scripts/export_weights.py [output_dir]

Dependencies (install once):
  uv add --optional weights torch torchvision
  or:  pip install torch torchvision

@thesis subsec:wyborSieci | A-07,B-08 | Donor network and layer choice: torchvision
  mobilenet_v3_small conv1, BatchNorm folded, then either the BT.601 luminance collapse or all
  27 RGB taps.
"""

import argparse
import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_weight_layout as CWL          # noqa: E402

try:
    import torch
    import torchvision.models as models
except ImportError:
    sys.exit(
        "torch / torchvision not found.\n"
        "  Install: uv add --optional weights torch torchvision\n"
        "  or:      pip install torch torchvision"
    )

# ---------------------------------------------------------------------------
# Constants matching conv2d_kernel.h / hanning_128.h
# ---------------------------------------------------------------------------
N_OUT       = 16
N_IN        = 3        # RGB channels in the pretrained model
KSIZE       = CWL.KSIZE
BUF_BYTES   = CWL.BUF_BYTES    # CONV_WEIGHT_BYTES_PAD
PATCH_SIZE  = 128      # PATCH_ROWS = PATCH_COLS (for hanning_128.h generation)

# roi_crop emits (x-µ)/σ scaled by this — design/pl_src/roi_crop/roi_crop.h.
# Duplicated here rather than parsed: one number, and --bias-scale prints it.
ROI_NORM_Q  = 32

# ImageNet normalization used during MobileNetV3-Small pretraining
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# Luminance coefficients for RGB → grayscale weight collapse (ITU-R BT.601)
LUM = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

def acc_max_theory(n_in: int) -> int:
    """Worst-case |acc| before bias:  n_in * KSIZE^2 * 127 * 127.

    145 161 at 9 taps, 435 483 at 27. out_shift is sized against this plus
    |bias_acc|, so the tap count costs ~log2(3) = 1.6 bits of headroom, of which
    ~0.6 shows up in the mean shift once the bias term is included.
    """
    return n_in * KSIZE * KSIZE * 127 * 127


# ---------------------------------------------------------------------------
# Step 1 — BN fold + grayscale collapse
# ---------------------------------------------------------------------------

def fold_bn(conv_w: np.ndarray, conv_b, gamma, beta, mu, var, eps: float):
    """Fold BatchNorm into conv weights/bias. NO collapse — 3 input planes.

    Returns:
        w_fold  [N_OUT, 3, KSIZE, KSIZE]  BN-folded RGB conv weights (float64)
        b_fold  [N_OUT]                   folded bias (float64)
    """
    s      = gamma / np.sqrt(var + eps)                      # [N_OUT]
    w_fold = conv_w * s[:, None, None, None]                  # [N_OUT, 3, K, K]
    b_in   = conv_b if conv_b is not None else np.zeros(N_OUT, dtype=np.float64)
    b_fold = s * (b_in - mu) + beta                          # [N_OUT]
    return w_fold, b_fold


def collapse_lum(w_fold: np.ndarray) -> np.ndarray:
    """RGB → grayscale via ITU-R BT.601 luminance. [N_OUT,3,K,K] → [N_OUT,1,K,K].

        w_gray[oc, 0, kr, kc] = Σ_ic LUM[ic] * w_fold[oc, ic, kr, kc]

    Deliberately NOT Danelljan's unweighted sum — see the long note above.
    """
    return np.sum(w_fold * LUM[None, :, None, None], axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Step 2 — per-channel symmetric INT8 quantization
# ---------------------------------------------------------------------------

def quantize_weights(w: np.ndarray):
    """Symmetric per-output-channel INT8 quantization.

    w shape: [N_OUT, n_in, KSIZE, KSIZE]   (n_in = 1 gray, 3 RGB)

    scale[oc]  = max(|w[oc]|) / 127          — over ALL taps of the channel
    w_int8[oc] = clip(round(w[oc] / scale[oc]), -127, 127)

    ONE SCALE PER OUTPUT CHANNEL, NOT PER INPUT PLANE. At n_in=3 a per-plane
    scale would equalize the three planes against each other and delete the
    chromatic contrast that is the entire point of the RGB export — the same
    failure mode as normalizing the three planes independently in Stage A.

    Returns w_int8 (int8) and scales (float64), both indexed by output channel.
    """
    flat   = w.reshape(N_OUT, -1)                           # [N_OUT, 9 or 27]
    maxabs = np.abs(flat).max(axis=1).clip(min=1e-12)
    scales = maxabs / 127.0
    w_q    = np.clip(np.round(w / scales[:, None, None, None]), -127, 127)
    return w_q.astype(np.int8), scales


# ---------------------------------------------------------------------------
# Step 3 — integer bias and per-channel output shift
# ---------------------------------------------------------------------------

def compute_acc_params(b_fold: np.ndarray, scales: np.ndarray,
                       n_in: int, bias_input_scale: float):
    """Derive per-channel bias_acc and out_shift for the AIE kernel.

    The accumulator holds  acc ≈ (y_float - b_fold) * q / scale, where q is the
    scale of the int8 activations the kernel actually receives. Adding
    bias_acc = round(b_fold * q / scale) gives full_acc ≈ y_float * q / scale.

    `bias_input_scale` IS that q, and it is the whole --bias-scale question:
    127 assumes int8 full scale ≙ 1.0, ROI_NORM_Q (32) matches what roi_crop
    emits. See the --bias-scale section of the module docstring.

    out_shift is chosen so |full_acc| >> out_shift fits in int16 [-32767, 32767],
    sized against the worst case |bias_acc| + acc_max_theory(n_in).

    Returns:
        bias_acc  int32[N_OUT]
        shifts    int8[N_OUT]    (out_shift per channel)
    """
    bias_acc_f = b_fold * float(bias_input_scale) / scales
    bias_acc   = np.clip(np.round(bias_acc_f), -(2**31), 2**31 - 1).astype(np.int32)

    max_full   = np.abs(bias_acc).astype(np.float64) + acc_max_theory(n_in)
    log2_over  = np.log2(np.maximum(max_full / 32767.0, 1.0))
    shifts     = np.ceil(log2_over).astype(np.int8).clip(0, 30)

    return bias_acc, shifts


# ---------------------------------------------------------------------------
# Step 4 — pack into 64-byte buffer
# ---------------------------------------------------------------------------

def pack_channel(lay: CWL.Layout, w_int8_oc: np.ndarray, shift: int,
                 bias_acc: int, scale: float) -> bytes:
    """Pack one output channel into BUF_BYTES bytes.

    The layout comes from conv_weight_layout, which mirrors
    design/aie_src/conv_weight_layout.h. Taps are flattened [n_in][kr][kc], so
    RGB lands PLANAR: [0:9] R, [9:18] G, [18:27] B.
    """
    return CWL.pack(lay, w_int8_oc.flatten().tolist(), shift, bias_acc,
                    scale / 127.0)


# ---------------------------------------------------------------------------
# Hanning window header generation
# ---------------------------------------------------------------------------

def _gen_hanning_h(out_path: Path, n: int) -> None:
    """Write a precomputed Q1.15 Hanning window header for an n-point patch.

    PERIODIC window (denominator n), not the symmetric one (denominator n-1).
    This is deliberate and load-bearing, not a style choice.

    A Hann window is a 3-term cosine sum, so on the DFT grid the periodic form
    has EXACTLY three non-zero bins: W[0] = n/2, W[±1] = -n/4, zero elsewhere.
    Separable in 2D, that is exactly 9 non-zero bins at (r,c) in {0,±1}^2.

    The host relies on that identity to cancel the pre-window feature mean in
    the frequency domain: DFT(w*(f-µ)) = DFT(w*f) - µ*DFT(w), so removing µ
    costs 9 complex subtractions instead of a spatial pass over the whole map.

    The symmetric form (n-1) is aperiodic on the DFT grid and leaks across all
    bins. Measured at n=128 in Q1.15, worst leaked bin sits only ~8.5 bits below
    DC (ratio 373) versus ~18 bits (ratio 2.2e5) for the periodic form — so with
    the symmetric window the 9-bin correction is NOT exact. Symmetric is the
    right choice for filter design; periodic is the right choice for FFT work.
    """
    vals = [round(math.sin(math.pi * i / n) ** 2 * 32767) for i in range(n)]

    lines = [
        "/*",
        f" * hanning_{n}.h",
        f" * Precomputed separable Hanning window in Q1.15 for {n}-point patches.",
        " *",
        f" * HANNING_{n}[i] = round(sin(π*i/{n})^2 * 32767)   for i = 0..{n-1}",
        " *",
        " * PERIODIC window (denominator n). Its 2D DFT has exactly 9 non-zero bins,",
        " * which is what lets the host cancel the pre-window mean in the frequency",
        " * domain. Do not switch to the symmetric (n-1) form — see _gen_hanning_h",
        " * in scripts/export_weights.py.",
        " *",
        f" * Usage in conv2d_kernel: out_windowed = (out * HANNING_{n}[r] / 32768)",
        f" *                                        * HANNING_{n}[c] / 32768",
        " *",
        " * Regenerate with:  make weights        (runs scripts/export_weights.py)",
        f" * Guard: kernel will fail to compile if PATCH_ROWS or PATCH_COLS != {n}.",
        " */",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        "/* Guard: only enforce when building with AIE pre-processor defines. */",
        "#if defined(PATCH_ROWS) && defined(PATCH_COLS)",
        f"#  if PATCH_ROWS != {n} || PATCH_COLS != {n}",
        f'#    error "hanning_{n}.h is generated for PATCH_ROWS={n}, PATCH_COLS={n}. Re-run make weights."',
        "#  endif",
        "#endif",
        "",
        f"static const int16_t HANNING_{n}[{n}] = {{",
    ]

    per_row = 8
    for row_start in range(0, n, per_row):
        chunk = vals[row_start : row_start + per_row]
        lines.append("    " + ", ".join(f"{v:6d}" for v in chunk) + ",")

    lines += ["};", ""]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _run_float(w: np.ndarray, b_fold: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    """Float-exact reference: BN-folded conv on one n_in×3×3 patch."""
    x_f = x_int8.astype(np.float64) / 127.0   # [n_in, K, K]
    return np.array(
        [float(np.sum(w[oc] * x_f)) + b_fold[oc] for oc in range(N_OUT)]
    )


def validate(w, b_fold, w_int8, scales, bias_acc, shifts, n_in, seed=42):
    """Compare quantized integer simulation against float reference.

    PyTorch comparison is intentionally omitted: at --in-ch 1 the grayscale
    collapse means the kernel no longer computes the same function as the
    3-channel model, so a direct numerical comparison would not be meaningful.

    NOTE this validates the QUANTIZATION only — it feeds a patch at int8 full
    scale (127 ≙ 1.0), which is the assumption --bias-scale 127 encodes. Under
    --bias-scale roi the reconstruction error here will be LARGER by design,
    because the bias is then sized for activations at ROI_NORM_Q. That is not a
    regression; the arbiter for the corrected bias is roi_crop's real output,
    not this synthetic patch.
    """
    rng    = np.random.default_rng(seed)
    x_int8 = rng.integers(-127, 127, (n_in, KSIZE, KSIZE), dtype=np.int8)

    y_float = _run_float(w, b_fold, x_int8)

    print(f"\n--- Kernel simulation vs float reference "
          f"(random {n_in}×{KSIZE}×{KSIZE} patch, int8 full scale) ---")
    print(f"\n  {'ch':>4}  {'float_ref':>10}  {'int_acc':>10}  "
          f"{'>>shft':>8}  {'recon':>10}  {'abs_err':>9}")
    print(f"  {'-'*66}")

    max_err = 0.0
    for oc in range(N_OUT):
        acc  = int(np.sum(w_int8[oc].astype(np.int32) * x_int8.astype(np.int32)))
        full = acc + int(bias_acc[oc])
        sh   = int(shifts[oc])
        out16 = max(-32768, min(32767, full >> sh))

        dq     = float(scales[oc]) / 127.0
        y_recon = out16 * dq * (1 << sh)

        err    = abs(y_float[oc] - y_recon)
        max_err = max(max_err, err)
        flag   = " !" if err > 0.1 * (abs(y_float[oc]) + 1e-6) else "  "

        print(f"  {oc:>4}  {y_float[oc]:>10.4f}  {full:>10d}  "
              f"{out16:>8d}  {y_recon:>10.4f}  {err:>9.5f}{flag}")

    print(f"\n  Float vs int recon max abs error: {max_err:.5f}")
    if max_err > 0.5:
        print("  WARNING: large reconstruction error — check scale or shift logic")
    return max_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir  = Path(__file__).resolve().parent
    repo_root   = script_dir.parent
    default_out = repo_root / "design" / "aie_src" / "weights"

    # The two positionals are the historical `make weights` interface
    # (out_dir, then PATCH_COLS) and must keep working unchanged.
    ap = argparse.ArgumentParser(
        description="Export MobileNetV3-Small conv1 as INT8 weights for conv2d_kernel.")
    ap.add_argument('out_dir', nargs='?', default=str(default_out))
    ap.add_argument('patch_size', nargs='?', type=int, default=PATCH_SIZE,
                    help="Hanning table length (= PATCH_COLS)")
    ap.add_argument('--in-ch', type=int, default=1, choices=(1, 3),
                    help="1 = BT.601 luminance collapse (default, what ships); "
                         "3 = RGB, all 27 taps")
    ap.add_argument('--bias-scale', default='127', choices=('127', 'roi'),
                    help="activation scale bias_acc is derived against. "
                         "127 = int8 full scale (default, what every hardware "
                         "measurement used); roi = ROI_NORM_Q, what roi_crop "
                         "actually emits. See the module docstring.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_size = args.patch_size

    n_in = args.in_ch
    lay  = CWL.Layout(n_in)
    bias_input_scale = 127.0 if args.bias_scale == '127' else float(ROI_NORM_Q)

    print(f"CONV_IN_CH = {n_in}  ({'RGB, 27 taps' if n_in == 3 else 'grayscale, 9 taps'})")
    print(f"  {lay}")
    print(f"  bias_acc scale: {bias_input_scale:g} "
          f"({'int8 full scale — the shipped default' if args.bias_scale == '127' else 'ROI_NORM_Q — corrected; needs CONV_RELU=0 and a shift-budget re-sweep'})")
    print(f"  acc_max_theory: {acc_max_theory(n_in)}")

    # ------------------------------------------------------------------
    # 1. Load pretrained model
    # ------------------------------------------------------------------
    print("Loading MobileNetV3-Small (pretrained on ImageNet)...")
    try:
        weights_enum = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m = models.mobilenet_v3_small(weights=weights_enum)
    except AttributeError:
        m = models.mobilenet_v3_small(pretrained=True)  # type: ignore[call-arg]
    m.eval()

    seq  = m.features[0]          # Conv2dNormActivation
    conv = seq[0]                  # Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
    bn   = seq[1]                  # BatchNorm2d(16)
    print(f"  {conv}")
    print(f"  {bn}")
    assert tuple(conv.weight.shape) == (N_OUT, N_IN, KSIZE, KSIZE), \
        f"Unexpected conv weight shape: {conv.weight.shape}"

    # ------------------------------------------------------------------
    # 2. Extract parameters (float64 to preserve precision through fold)
    # ------------------------------------------------------------------
    conv_w = conv.weight.detach().cpu().numpy().astype(np.float64)
    conv_b = conv.bias.detach().cpu().numpy().astype(np.float64) \
             if conv.bias is not None else None
    gamma  = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta   = bn.bias.detach().cpu().numpy().astype(np.float64)
    mu     = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var_   = bn.running_var.detach().cpu().numpy().astype(np.float64)
    eps    = float(bn.eps)

    # ------------------------------------------------------------------
    # 3. Fold BN + grayscale collapse
    # ------------------------------------------------------------------
    w_fold, b_fold = fold_bn(conv_w, conv_b, gamma, beta, mu, var_, eps)
    if n_in == 1:
        print("Folding BatchNorm + collapsing RGB → grayscale (ITU-R BT.601)...")
        w_taps = collapse_lum(w_fold)
    else:
        print("Folding BatchNorm, keeping all three input planes (RGB)...")
        w_taps = w_fold
    print(f"  w_taps  shape: {w_taps.shape}  "
          f"range: [{w_taps.min():.5f}, {w_taps.max():.5f}]")
    print(f"  b_fold  range: [{b_fold.min():.5f}, {b_fold.max():.5f}]")

    # ------------------------------------------------------------------
    # 4. Quantize weights
    # ------------------------------------------------------------------
    print("Quantizing weights (symmetric per-output-channel INT8)...")
    w_int8, scales = quantize_weights(w_taps)
    w_dq    = w_int8.astype(np.float64) * scales[:, None, None, None]
    quant_err = np.abs(w_taps - w_dq)
    print(f"  Weight quant error — max: {quant_err.max():.2e}  mean: {quant_err.mean():.2e}")
    print(f"  Scale range: [{scales.min():.4e}, {scales.max():.4e}]")

    # ------------------------------------------------------------------
    # 5. Bias and output shift
    # ------------------------------------------------------------------
    print("Computing integer bias and per-channel output shift...")
    bias_acc, shifts = compute_acc_params(b_fold, scales, n_in, bias_input_scale)
    print(f"  out_shifts:  {shifts.tolist()}")
    print(f"  bias_acc range: [{bias_acc.min()}, {bias_acc.max()}]")

    # ------------------------------------------------------------------
    # 6. Validation
    # ------------------------------------------------------------------
    max_err = validate(w_taps, b_fold, w_int8, scales, bias_acc, shifts, n_in)

    # ------------------------------------------------------------------
    # 7. Pack and write outputs
    # ------------------------------------------------------------------
    flat = bytearray()
    for oc in range(N_OUT):
        flat += pack_channel(lay, w_int8[oc], shifts[oc], bias_acc[oc], scales[oc])

    bin_path = out_dir / "layer0_weights.bin"
    bin_path.write_bytes(flat)
    print(f"\nWrote {bin_path}  ({len(flat)} bytes = {N_OUT}×{BUF_BYTES}, "
          f"layout tag {n_in})")

    # Read it straight back through the SHARED unpacker. This is cheap and it
    # closes the loop the tag byte exists for: if the header and the Python
    # mirror ever disagree about an offset, the exporter itself says so instead
    # of shipping a file the kernel will misread.
    for oc, (taps, sh, ba, dq, mp, _l) in enumerate(CWL.load_bin(bin_path, n_in)):
        assert taps == w_int8[oc].flatten().tolist(), f"tap round-trip failed on ch{oc}"
        assert sh == int(shifts[oc]) and ba == int(bias_acc[oc]), \
            f"shift/bias round-trip failed on ch{oc}"
        assert mp == 0, f"mean_prev must ship as 0 (the host seeds it), ch{oc}"
    print(f"  round-trip through conv_weight_layout: OK ({N_OUT} channels)")

    # Numpy archive (for debugging / aiesim scenarios)
    npz_path = out_dir / "layer0_meta.npz"
    np.savez(str(npz_path),
             w_taps=w_taps.astype(np.float32),
             # w_gray kept for readers that predate --in-ch; at --in-ch 3 it is
             # the luminance collapse of the SAME folded weights, i.e. what the
             # grayscale export would have produced, not what was shipped.
             w_gray=collapse_lum(w_fold).astype(np.float32),
             n_in=np.int32(n_in),
             bias_input_scale=np.float32(bias_input_scale),
             w_int8=w_int8,
             b_float=b_fold.astype(np.float32),
             bias_acc=bias_acc,
             shifts=shifts,
             scales=scales.astype(np.float32),
             imagenet_mean=IMAGENET_MEAN.astype(np.float32),
             imagenet_std=IMAGENET_STD.astype(np.float32),
             lum=LUM.astype(np.float32))
    print(f"Wrote {npz_path}  (validation metadata)")

    # C header for host app / AIE kernel
    h_path = out_dir / "layer0.h"
    with open(h_path, 'w') as f:
        f.write("/* Auto-generated by scripts/export_weights.py — do not edit */\n")
        f.write("#pragma once\n\n")
        f.write('/* Load into weights_bo: fread(ptr, 1, LAYER0_TOTAL_BYTES, f) */\n')
        f.write('#define LAYER0_WEIGHTS_FILE  "layer0_weights.bin"\n')
        f.write(f"#define LAYER0_N_OUT_CH      {N_OUT}\n")
        f.write(f"#define LAYER0_BUF_BYTES     {BUF_BYTES}\n")
        f.write(f"#define LAYER0_TOTAL_BYTES   {N_OUT * BUF_BYTES}\n\n")
        f.write("/* What this file was exported as. The BUILD must agree:\n"
                " * a CONV_IN_CH=1 graph fed a CONV_IN_CH=3 weights file reads\n"
                " * taps [0:9] as a 3x3 kernel and slices out_shift out of the\n"
                " * G plane. Byte 63 of every channel buffer carries the same\n"
                " * number so a reader can assert at runtime. */\n")
        f.write(f"#define LAYER0_IN_CH         {n_in}\n")
        f.write(f"#define LAYER0_N_TAPS        {lay.raw}\n")
        f.write(f"#define LAYER0_BIAS_SCALE    {bias_input_scale:.1f}f\n\n")
        # Bare include, not "../conv_weight_layout.h": design/aie_src is already on
        # the include path for both toolchains (GCC_INC and the aiecompiler
        # --include list), and a relative path breaks the moment layer0.h is
        # generated anywhere but its usual directory.
        f.write("#include \"conv_weight_layout.h\"\n")
        f.write("#if LAYER0_IN_CH != CONV_IN_CH\n"
                "#  error \"layer0_weights.bin was exported for a different "
                "CONV_IN_CH. Re-run: make weights CONV_IN_CH=<n>\"\n"
                "#endif\n\n")
        f.write("/* Per-channel dequantization scale: y_float ≈ out_int16 * scale * (1<<shift) */\n")
        f.write(f"static const float layer0_dequant_scales[{N_OUT}] = {{\n")
        for s in scales:
            f.write(f"    {s / 127.0:.10e}f,\n")
        f.write("};\n\n")
        f.write("/* Per-channel output right-shift stored at byte 9 of each 64-byte buffer */\n")
        f.write(f"static const int layer0_out_shifts[{N_OUT}] = {{ ")
        f.write(", ".join(str(int(s)) for s in shifts))
        f.write(" };\n")
    print(f"Wrote {h_path}  (C header)")

    # ------------------------------------------------------------------
    # 8. Generate hanning_<patch_size>.h
    # ------------------------------------------------------------------
    hanning_path = repo_root / "design" / "aie_src" / f"hanning_{patch_size}.h"
    _gen_hanning_h(hanning_path, patch_size)
    print(f"Wrote {hanning_path}  (Q1.15 Hanning window, {patch_size} points)")

    print(f"\nDone. Max reconstruction error: {max_err:.5f}")


if __name__ == "__main__":
    main()
