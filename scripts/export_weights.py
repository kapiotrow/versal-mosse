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
2. Collapse RGB (3 input channels) → grayscale (1 channel) via ITU-R BT.601
   luminance coefficients: LUM = [0.2989, 0.5870, 0.1140].
3. Per-output-channel symmetric INT8 weight quantization.
4. Compute per-channel integer bias and output right-shift for the AIE accumulator.
5. Pack into 64-byte buffers and write outputs.
6. Write hanning_128.h into design/aie_src/ (precomputed Q1.15 window table).

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

64-byte weight buffer layout (CONV_WEIGHT_BYTES_PAD in conv2d_kernel.h)
------------------------------------------------------------------------
  [ 0:  9]  int8[3][3]   grayscale conv weights w[kr][kc] (row-major)
  [ 9]      int8         out_shift   (right-shift: int32 → int16)
  [10: 14]  int32 LE     bias_acc    = round(b_fold * 127 / scale)
  [14: 18]  float32 LE   dequant_scale = scale / 127   (host validation only)
  [18: 64]  zero padding

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
"""

import sys
import math
import struct
import numpy as np
from pathlib import Path

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
N_IN_GRAY   = 1        # grayscale: what the AIE kernel uses (CONV_IN_CH=1)
KSIZE       = 3
BUF_BYTES   = 64       # CONV_WEIGHT_BYTES_PAD
PATCH_SIZE  = 128      # PATCH_ROWS = PATCH_COLS (for hanning_128.h generation)

# ImageNet normalization used during MobileNetV3-Small pretraining
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# Luminance coefficients for RGB → grayscale weight collapse (ITU-R BT.601)
LUM = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

# Theoretical worst-case int32 accumulator magnitude before bias:
#   N_IN_GRAY * KSIZE^2 * 127 * 127  = 9 * 127 * 127 = 144 963
ACC_MAX_THEORY = N_IN_GRAY * KSIZE * KSIZE * 127 * 127


# ---------------------------------------------------------------------------
# Step 1 — BN fold + grayscale collapse
# ---------------------------------------------------------------------------

def fold_bn(conv_w: np.ndarray, conv_b, gamma, beta, mu, var, eps: float):
    """Fold BatchNorm into conv weights/bias, then collapse RGB → grayscale.

    Returns:
        w_gray  [N_OUT, 1, KSIZE, KSIZE]  grayscale conv weights (float64)
        b_fold  [N_OUT]                   folded bias (float64)
    """
    s      = gamma / np.sqrt(var + eps)                      # [N_OUT]
    w_fold = conv_w * s[:, None, None, None]                  # [N_OUT, 3, K, K]
    b_in   = conv_b if conv_b is not None else np.zeros(N_OUT, dtype=np.float64)
    b_fold = s * (b_in - mu) + beta                          # [N_OUT]

    # Collapse RGB input channels → single grayscale channel via luminance weights.
    # w_gray[oc, 0, kr, kc] = Σ_ic LUM[ic] * w_fold[oc, ic, kr, kc]
    w_gray = np.sum(w_fold * LUM[None, :, None, None], axis=1, keepdims=True)
    # w_gray: [N_OUT, 1, K, K]
    return w_gray, b_fold


# ---------------------------------------------------------------------------
# Step 2 — per-channel symmetric INT8 quantization
# ---------------------------------------------------------------------------

def quantize_weights(w_gray: np.ndarray):
    """Symmetric per-output-channel INT8 quantization.

    w_gray shape: [N_OUT, 1, KSIZE, KSIZE]

    scale[oc]  = max(|w_gray[oc]|) / 127
    w_int8[oc] = clip(round(w_gray[oc] / scale[oc]), -127, 127)

    Returns w_int8 (int8) and scales (float64), both indexed by output channel.
    """
    flat   = w_gray.reshape(N_OUT, -1)                           # [N_OUT, 9]
    maxabs = np.abs(flat).max(axis=1).clip(min=1e-12)
    scales = maxabs / 127.0
    w_q    = np.clip(np.round(w_gray / scales[:, None, None, None]), -127, 127)
    return w_q.astype(np.int8), scales


# ---------------------------------------------------------------------------
# Step 3 — integer bias and per-channel output shift
# ---------------------------------------------------------------------------

def compute_acc_params(b_fold: np.ndarray, scales: np.ndarray):
    """Derive per-channel bias_acc and out_shift for the AIE kernel.

    The accumulator holds  acc ≈ (y_float - b_fold) * 127 / scale.
    Adding  bias_acc = round(b_fold * 127 / scale)  gives:
        full_acc ≈ y_float * 127 / scale

    out_shift is chosen so that |full_acc| >> out_shift fits in int16 [-32767,32767].

    Returns:
        bias_acc  int32[N_OUT]
        shifts    int8[N_OUT]    (out_shift per channel)
    """
    bias_acc_f = b_fold * 127.0 / scales
    bias_acc   = np.clip(np.round(bias_acc_f), -(2**31), 2**31 - 1).astype(np.int32)

    max_full   = np.abs(bias_acc).astype(np.float64) + ACC_MAX_THEORY
    log2_over  = np.log2(np.maximum(max_full / 32767.0, 1.0))
    shifts     = np.ceil(log2_over).astype(np.int8).clip(0, 30)

    return bias_acc, shifts


# ---------------------------------------------------------------------------
# Step 4 — pack into 64-byte buffer
# ---------------------------------------------------------------------------

def pack_channel(w_int8_oc: np.ndarray, shift: int, bias_acc: int, scale: float) -> bytes:
    """Pack one output channel into BUF_BYTES bytes.

    Layout (must match conv2d_kernel.h):
      [ 0: 9]  int8[3][3]   grayscale weights w[kr][kc] row-major
      [ 9]     int8         out_shift
      [10:14]  int32 LE     bias_acc
      [14:18]  float32 LE   dequant_scale = scale / 127
      [18:64]  zero padding
    """
    buf      = bytearray(BUF_BYTES)
    buf[0:9] = w_int8_oc.flatten().tobytes()           # 9 bytes
    buf[9]   = int(shift) & 0xFF
    struct.pack_into('<i', buf, 10, int(bias_acc))
    struct.pack_into('<f', buf, 14, float(scale / 127.0))
    return bytes(buf)


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

def _run_float(w_gray: np.ndarray, b_fold: np.ndarray, x_gray_int8: np.ndarray) -> np.ndarray:
    """Float-exact reference: grayscale BN-folded conv on a 1×3×3 patch."""
    x_f = x_gray_int8.astype(np.float64) / 127.0   # [1, K, K]
    return np.array(
        [float(np.sum(w_gray[oc] * x_f)) + b_fold[oc] for oc in range(N_OUT)]
    )


def validate(w_gray, b_fold, w_int8, scales, bias_acc, shifts, seed=42):
    """Compare quantized integer simulation against float reference.

    PyTorch comparison is intentionally omitted: after the grayscale collapse
    the kernel no longer computes the same function as the 3-channel model,
    so a direct numerical comparison would not be meaningful.
    """
    rng         = np.random.default_rng(seed)
    x_gray_int8 = rng.integers(-127, 127, (N_IN_GRAY, KSIZE, KSIZE), dtype=np.int8)

    y_float = _run_float(w_gray, b_fold, x_gray_int8)

    print(f"\n--- Kernel simulation vs float reference (random 1×{KSIZE}×{KSIZE} grayscale patch) ---")
    print(f"\n  {'ch':>4}  {'float_ref':>10}  {'int_acc':>10}  "
          f"{'>>shft':>8}  {'recon':>10}  {'abs_err':>9}")
    print(f"  {'-'*66}")

    max_err = 0.0
    for oc in range(N_OUT):
        acc  = int(np.sum(w_int8[oc].astype(np.int32) * x_gray_int8.astype(np.int32)))
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

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch size for the Hanning table (square patch: PATCH_ROWS == PATCH_COLS).
    # Passed as argv[2] by `make weights` (= $(PATCH_COLS)); defaults to 128.
    patch_size = int(sys.argv[2]) if len(sys.argv) > 2 else PATCH_SIZE

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
    print("Folding BatchNorm + collapsing RGB → grayscale (ITU-R BT.601)...")
    w_gray, b_fold = fold_bn(conv_w, conv_b, gamma, beta, mu, var_, eps)
    print(f"  w_gray  shape: {w_gray.shape}  "
          f"range: [{w_gray.min():.5f}, {w_gray.max():.5f}]")
    print(f"  b_fold  range: [{b_fold.min():.5f}, {b_fold.max():.5f}]")

    # ------------------------------------------------------------------
    # 4. Quantize weights
    # ------------------------------------------------------------------
    print("Quantizing weights (symmetric per-output-channel INT8)...")
    w_int8, scales = quantize_weights(w_gray)
    w_dq    = w_int8.astype(np.float64) * scales[:, None, None, None]
    quant_err = np.abs(w_gray - w_dq)
    print(f"  Weight quant error — max: {quant_err.max():.2e}  mean: {quant_err.mean():.2e}")
    print(f"  Scale range: [{scales.min():.4e}, {scales.max():.4e}]")

    # ------------------------------------------------------------------
    # 5. Bias and output shift
    # ------------------------------------------------------------------
    print("Computing integer bias and per-channel output shift...")
    bias_acc, shifts = compute_acc_params(b_fold, scales)
    print(f"  out_shifts:  {shifts.tolist()}")
    print(f"  bias_acc range: [{bias_acc.min()}, {bias_acc.max()}]")

    # ------------------------------------------------------------------
    # 6. Validation
    # ------------------------------------------------------------------
    max_err = validate(w_gray, b_fold, w_int8, scales, bias_acc, shifts)

    # ------------------------------------------------------------------
    # 7. Pack and write outputs
    # ------------------------------------------------------------------
    flat = bytearray()
    for oc in range(N_OUT):
        flat += pack_channel(w_int8[oc], shifts[oc], bias_acc[oc], scales[oc])

    bin_path = out_dir / "layer0_weights.bin"
    bin_path.write_bytes(flat)
    print(f"\nWrote {bin_path}  ({len(flat)} bytes = {N_OUT}×{BUF_BYTES})")

    # Numpy archive (for debugging / aiesim scenarios)
    npz_path = out_dir / "layer0_meta.npz"
    np.savez(str(npz_path),
             w_gray=w_gray.astype(np.float32),
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
