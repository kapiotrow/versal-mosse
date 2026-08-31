#!/usr/bin/env python3
"""
scripts/check_collapse.py

What the conv2d front end actually produces — four checks on the exported
INT8 feature bank, none of which need hardware and all of which run in seconds.
The expensive alternative for every one of these is an aiesim or hw_emu run.

  Q1  LUM vs SUM — should the RGB->gray collapse be the luminance-weighted sum
      (what export_weights.py does) or the unweighted sum (what Danelljan §3.3
      implies)? Per-channel int8 quantization absorbs any magnitude difference,
      so only the DIRECTION of the collapsed 9-vector matters. Needs torch.

  Q2  Linear diversity — how many of the exported 3x3 kernels are independent?
      Collinear channels are one degree of freedom to the DCF (Danelljan eq. 1
      optimizes jointly over channels), so each duplicate costs a full serial
      FFT->cmul->IFFT round trip and returns nothing.

  Q3  Bias / out_shift sanity — INPUT-INDEPENDENT, and the most important check
      here. Derived purely from bias_acc vs the maximum possible AC swing
      127*sum|w|, so its verdicts hold for ANY int8 input at any patch size:
        * bias + maxAC <= 0  =>  the channel is identically zero, always
        * bias - maxAC >= 0  =>  ReLU can never fire; it is a no-op
        * log2(maxAC >> out_shift) => bits of signal resolution actually used
      out_shift is derived from |bias_acc| + ACC_MAX_THEORY, so a large bias
      shifts the SIGNAL down to make room for a DC pedestal that Stage B1
      subtracts away downstream. That trade is pure loss.

  Q4  Post-ReLU feature maps on a real Stage-A-preprocessed patch, through
      conv2d's exact integer datapath. Patch-specific, unlike Q3. Shows the
      DC/AC ratio each channel actually delivers and which channels are
      redundant *after* the nonlinearity (two collinear kernels with different
      bias and out_shift are two different nonlinear readouts of the same
      projection, so Q2 does not settle this).

Re-run this after ANY change to export_weights.py, ROI_NORM_Q, or the collapse.

Usage
-----
  uv run --extra weights python3 scripts/check_collapse.py        # all four
  python3 scripts/check_collapse.py                               # Q2-Q4 only

Q1 needs torch/torchvision; Q2-Q4 need only design/aie_src/weights/layer0_weights.bin
(run `make weights`). Q4 additionally needs an aiesim scenario directory.

@thesis subsec:wyborSieci | A-07,N-15,N-16 | The four offline checks that decide the feature
  bank: collapse convention, linear diversity (the structural rank-9 cap), bias_acc/out_shift
  sanity, and the post-ReLU map.
"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_weight_layout as CWL

N_OUT, KSIZE = 16, 3
LUM = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
WEIGHTS_BIN = Path("design/aie_src/weights/layer0_weights.bin")
# Q4 input: any scenario whose patch has been through Stage A. s0-s4 are raw
# impulses/constants and are meaningless here; s6/s7 are preprocessed.
SCENARIO = Path("design/aie_src/aiesim_data/s6")
# Must match export_weights.py's acc_max_theory(): n_in * KSIZE^2 * 127 * 127.
def acc_max_theory(n_in):
    return n_in * KSIZE * KSIZE * 127 * 127


# ---------------------------------------------------------------------------
# shared loaders
# ---------------------------------------------------------------------------

def load_channels():
    """Unpack layer0_weights.bin -> (list of (w[n_in,3,3], out_shift, bias), n_in).

    Goes through conv_weight_layout so the offsets cannot drift from the kernel's,
    and so the layout tag decides n_in instead of this file assuming it. Reading
    an RGB export with the grayscale offsets is the failure this guards: it does
    not throw, it reports sixteen healthy channels built from R-plane taps with a
    bias sliced out of the G plane.
    """
    if not WEIGHTS_BIN.exists():
        raise SystemExit(f"{WEIGHTS_BIN} absent — run `make weights` first.")
    chans = CWL.load_bin(WEIGHTS_BIN)
    n_in = chans[0][5].n_in
    out = []
    for taps, shift, bias, _dq, _mean, lay in chans:
        w = np.array(taps, dtype=np.int32).reshape(lay.n_in, 3, 3)
        out.append((w, shift, bias))
    return out, n_in


def q8(w):
    """Per-channel symmetric int8, exactly as quantize_weights() does."""
    maxabs = np.abs(w.reshape(len(w), -1)).max(axis=1).clip(min=1e-12)
    sc = maxabs / 127.0
    return np.clip(np.round(w / sc[:, None, None]), -127, 127).astype(np.int8), sc


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else float("nan")


def hdr(n, title):
    print()
    print("=" * 76)
    print(f"Q{n}  {title}")
    print("=" * 76)


# ---------------------------------------------------------------------------
# Q1 — which collapse convention
# ---------------------------------------------------------------------------

def q1_lum_vs_sum():
    hdr(1, "LUM vs SUM collapse convention")
    try:
        import torchvision.models as models
    except ImportError:
        print("torch/torchvision not installed — skipped.")
        print("  Install: uv add --optional weights torch torchvision")
        return

    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    conv, bn = m.features[0][0], m.features[0][1]
    cw = conv.weight.detach().numpy().astype(np.float64)
    gamma = bn.weight.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    w_fold = cw * (gamma / np.sqrt(var + bn.eps))[:, None, None, None]

    w_sum = w_fold.sum(axis=1)                                # Danelljan-exact
    w_lum = (w_fold * LUM[None, :, None, None]).sum(axis=1)   # what we ship
    q_sum, _ = q8(w_sum)
    q_lum, _ = q8(w_lum)
    # A colour-opponent filter cancels on collapse. SUM sums 3 planes so its
    # neutral reference is 3x the per-plane norm; LUM's coefficients sum to 1.
    plane = np.linalg.norm(w_fold.reshape(N_OUT, 3, -1), axis=2).mean(axis=1)

    print(" ch  cos(sum,lum)  int8 L1 diff  SUM keeps  LUM keeps  cos(R,G)")
    print(" " + "-" * 70)
    dead, differ = [], []
    for oc in range(N_OUT):
        a, b = w_sum[oc].ravel(), w_lum[oc].ravel()
        c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        l1 = int(np.abs(q_sum[oc].astype(int) - q_lum[oc].astype(int)).sum())
        rs, rl = np.linalg.norm(a) / (3 * plane[oc]), np.linalg.norm(b) / plane[oc]
        R, G = w_fold[oc, 0].ravel(), w_fold[oc, 1].ravel()
        crg = R @ G / (np.linalg.norm(R) * np.linalg.norm(G) + 1e-18)
        if rs < 0.10:
            dead.append(oc)
        if c < 0.99:
            differ.append(oc)
        print(f" {oc:2d}    {c:+.4f}       {l1:5d}       {rs:.4f}     "
              f"{rl:.4f}    {crg:+.4f}{'   <-- opponent' if rs < 0.10 else ''}")

    print()
    print(f"UNWEIGHTED sum annihilates (<10% of per-plane norm): {dead}")
    print(f"conventions disagree (cos < 0.99):                   {differ}")
    print()
    print("VERDICT: luminance is correct for this design. The unweighted sum is")
    print("paper-exact but cancels the opponent channels, and per-channel int8")
    print("normalization then amplifies the residue to full scale. Do not switch.")


# ---------------------------------------------------------------------------
# Q2 — linear diversity of the shipped kernels
# ---------------------------------------------------------------------------

def q2_linear_diversity(chans, n_in=1):
    hdr(2, f"linear diversity of the shipped int8 kernels (CONV_IN_CH={n_in})")
    if n_in == 1:
        print("A 3x3 grayscale kernel lives in 9 dimensions, so 16 channels CANNOT")
        print("be independent — the rank cap is structural, not a property of the")
        print("pretrained weights. At CONV_IN_CH=3 the space is 27-dimensional and")
        print("the cap lifts. See the RGB section of CLAUDE.md.")
    W = np.stack([w.astype(float).ravel() for w, _, _ in chans])
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    C = Wn @ Wn.T
    n = len(Wn)

    pairs = [(i, j, C[i, j]) for i in range(n) for j in range(i + 1, n)
             if abs(C[i, j]) > 0.95]
    print("collinear kernel pairs (|cos| > 0.95) — redundant *linear* filters:")
    for i, j, c in pairs:
        print(f"  ch{i:2d} / ch{j:2d}   cos = {c:+.4f}")
    if not pairs:
        print("  none")

    groups, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        g = {i} | {j for j in range(n) if j != i and abs(C[i, j]) > 0.95}
        seen |= g
        groups.append(sorted(g))
    ev = np.linalg.svd(Wn, compute_uv=False)
    ambient = Wn.shape[1]                       # 9 for a 3x3 grayscale kernel
    num_rank = int((ev > ev[0] * 1e-2).sum())
    energy = ev**2 / (ev**2).sum()
    pr = (ev**2).sum()**2 / (ev**4).sum()       # participation ratio
    d95 = int(np.searchsorted(np.cumsum(energy), 0.95) + 1)

    print()
    print("singular values: " + "  ".join(f"{v:.3f}" for v in ev))
    planes = ambient // (KSIZE * KSIZE)
    print(f"ambient dimension: {ambient} (a {KSIZE}x{KSIZE} kernel over "
          f"{planes} input plane{'s' if planes != 1 else ''}) — "
          f"{n} channels CANNOT exceed it")
    print(f"numerical rank (sv > 1% of max): {num_rank} of {n} channels")
    print(f"effective rank (participation ratio): {pr:.2f}   "
          f"directions for 95% of kernel energy: {d95}")
    print(f"collinear GROUPS at |cos|>0.95: {len(groups)} "
          f"({n - len(groups)} channel(s) absorbed into another group)")
    print()
    print("READ THE RANK, NOT THE GROUP COUNT. The group count only merges pairs")
    print("that are nearly parallel; it said '14 of 16' here for a bank whose rank")
    print("is capped at 9 by the collapse itself, and understated the problem for")
    print("months. The luminance collapse projects 27-dim RGB kernels onto 9 dims,")
    print("so redundancy is structural, not a property of the pretrained weights:")
    print("in RGB the same kernels have rank 16 and participation ratio 7.43, and")
    print("the three collinear channels here (0/9/14, within 2-6 degrees) are")
    print("59-72 degrees apart. Measured by scripts/rgb_vs_gray_holdout.py.")
    print()
    print("NOTE: this is the LINEAR picture only. Channels with the same kernel but")
    print("different bias/out_shift are different nonlinear readouts — see Q4.")


# ---------------------------------------------------------------------------
# Q3 — bias / out_shift sanity (input-independent)
# ---------------------------------------------------------------------------

def exported_bias_scale():
    """The --bias-scale the .bin was exported with, from layer0.h beside it.

    Returns None if the header is missing or predates LAYER0_BIAS_SCALE, in
    which case the export used the historical 127 by definition. This exists so
    Q3 does not print a root-cause paragraph about a defect that a corrected
    export has already fixed.
    """
    h = WEIGHTS_BIN.parent / "layer0.h"
    if not h.exists():
        return None
    for line in h.read_text().splitlines():
        if line.startswith("#define LAYER0_BIAS_SCALE"):
            return float(line.split()[2].rstrip('f'))
    return None


def q3_bias_shift(chans, n_in=1):
    hdr(3, "bias / out_shift sanity — INPUT-INDEPENDENT")
    print(" ch  shift    bias_acc     maxAC   verdict                       bits/15")
    print(" " + "-" * 74)
    always_zero, relu_never, bits = [], [], []
    for oc, (w, shift, bias) in enumerate(chans):
        ac = 127 * int(np.abs(w).sum())      # max possible |acc - bias|
        if bias + ac <= 0:
            v = "ALWAYS ZERO (structurally dead)"
            always_zero.append(oc)
        elif bias - ac >= 0:
            v = "ReLU provably NEVER active"
            relu_never.append(oc)
        else:
            v = "ReLU active for some inputs"
        nb = np.log2(max(ac >> shift, 1))
        bits.append(nb)
        print(f" {oc:2d}   {shift:3d}  {bias:10d}  {ac:8d}   {v:31s}  {nb:4.1f}")

    bits = np.array(bits)
    print()
    print(f"structurally dead channels : {always_zero}"
          f"   ({len(always_zero)} wasted pipeline passes/frame, guaranteed)")
    print(f"ReLU is a no-op on         : {relu_never}  ({len(relu_never)} of {len(chans)})")
    print(f"signal resolution          : {bits.min():.1f} to {bits.max():.1f} bits "
          f"of 15 available (spread {bits.max()-bits.min():.1f} bits)")
    print()
    print("WHY the low end is low: out_shift comes from |bias_acc| + ACC_MAX_THEORY")
    print(f"(={acc_max_theory(n_in)}), so a large bias shifts the SIGNAL down to make room for a")
    print("DC pedestal that Stage B1 subtracts away downstream. Pure loss.")
    bs = exported_bias_scale()
    if bs is None or abs(bs - 127.0) < 1e-6:
        print("Root cause: export_weights.py derived bias_acc as b_fold*127/scale, i.e.")
        print("for an input scale of 127 = 1.0, but roi_crop emits ROI_NORM_Q = 32 per")
        print("sigma. The bias is ~4x oversized relative to the activations it meets, and")
        print("it sits BEFORE ReLU, so it also decides which activations survive at all.")
        print("Fix: make weights BIAS_SCALE=roi — but ONLY with CONV_RELU=0, and it")
        print("obliges a shift-budget re-sweep. See export_weights.py's --bias-scale.")
    else:
        print(f"bias_acc was exported at scale {bs:g} (the ROI_NORM_Q correction), so")
        print("the classic ~4x-oversized-bias loss does NOT apply to this file. Any")
        print("remaining spread is the tap count and the per-channel weight scales.")


# ---------------------------------------------------------------------------
# Q4 — post-ReLU feature maps on a real preprocessed patch
# ---------------------------------------------------------------------------

def load_patch(scenario):
    """Reconstruct the int8 patch from a scenario's PLIO text stream.

    patch_in.txt is one int32 per line, 4 packed int8 (byte order matches
    conv2d_kernel.cpp's unpack). N is taken from a scenario .bin (cint16, 4 B
    per sample) rather than guessed, and the stream may repeat the patch.
    """
    ptxt, ref = scenario / "patch_in.txt", scenario / "fft_col_in.bin"
    if not ptxt.exists() or not ref.exists():
        return None, 0
    n2 = ref.stat().st_size // 4
    N = int(round(n2 ** 0.5))
    if N * N != n2:
        return None, 0
    words = np.loadtxt(ptxt, dtype=np.int64)
    u = words.astype(np.uint32)
    px = np.zeros(len(words) * 4, dtype=np.int8)
    for k in range(4):
        px[k::4] = ((u >> (8 * k)) & 0xFF).astype(np.uint8).view(np.int8)
    if len(px) < N * N:
        return None, 0
    return px[:N * N].astype(np.int32).reshape(N, N), N


def q4_post_relu(chans, scenario=SCENARIO, n_in=1):
    hdr(4, f"post-ReLU feature maps on a real patch ({scenario})")
    if n_in != 1:
        print(f"CONV_IN_CH={n_in}: skipped. This check runs the kernel's integer")
        print("datapath over a single-plane Stage-A patch, and the scenario data")
        print("is grayscale. An RGB verdict needs 3-plane vectors from")
        print("gen_aiesim_vectors.py, which do not exist yet.")
        return
    patch, N = load_patch(scenario)
    if patch is None:
        print(f"no usable preprocessed patch in {scenario} — skipped.")
        print("  Generate with: make gen_vectors SCENARIO=s6")
        return
    print(f"patch {N}x{N}: min {patch.min()} max {patch.max()} "
          f"mean {patch.mean():.2f}")
    print("(Stage-A preprocessed. s0-s4 are RAW and would be meaningless here.)")

    pad = np.zeros((N + 2, N + 2), dtype=np.int32)
    pad[1:-1, 1:-1] = patch
    maps = []
    for w, shift, bias in chans:
        acc = np.zeros((N, N), dtype=np.int64) + bias
        for kr in range(3):
            for kc in range(3):
                acc += int(w[0, kr, kc]) * pad[kr:kr + N, kc:kc + N]
        maps.append(np.clip(acc >> shift, 0, 32767).astype(np.float64))
    maps = np.stack(maps)

    print()
    print(" ch   post-ReLU mean       std   DC/AC   frac>0   frac railed")
    print(" " + "-" * 60)
    for oc, m in enumerate(maps):
        ratio = m.mean() / m.std() if m.std() > 1e-9 else float("inf")
        rs = "  inf" if ratio == float("inf") else f"{ratio:6.0f}"
        print(f" {oc:2d}   {m.mean():13.1f}  {m.std():8.1f}  {rs}   "
              f"{(m > 0).mean():.3f}    {(m >= 32767).mean():.3f}")

    zero = [oc for oc, m in enumerate(maps) if m.std() < 1e-9 and m.mean() < 1e-9]
    print()
    print(f"identically zero on this patch: {zero}")
    n = len(maps)
    hits = [(i, j, corr(maps[i], maps[j])) for i in range(n) for j in range(i + 1, n)
            if abs(corr(maps[i], maps[j])) > 0.95]
    print(f"post-ReLU redundant pairs (|corr| > 0.95): {len(hits)} of "
          f"{n*(n-1)//2} pairs")
    for i, j, c in hits[:12]:
        print(f"  ch{i:2d} / ch{j:2d}   corr = {c:+.4f}")
    if len(hits) > 12:
        print(f"  ... and {len(hits)-12} more")

    flat = maps.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)   # B1 removes the DC pedestal
    live = np.linalg.norm(flat, axis=1) > 1e-9
    if live.sum():
        fn = flat[live] / np.linalg.norm(flat[live], axis=1, keepdims=True)
        ev = np.linalg.svd(fn, compute_uv=False)
        cum = np.cumsum(ev ** 2) / np.sum(ev ** 2)
        print()
        print(f"effective rank after mean removal: "
              f"{int(np.searchsorted(cum, 0.99) + 1)} of {int(live.sum())} live channels")
        print("CAVEAT: rank is partly a property of this patch — a smooth input excites")
        print("few spatial modes through a 3x3 bank. Q3's verdicts are the robust ones.")


if __name__ == "__main__":
    chans, n_in = load_channels()
    print(f"layer0_weights.bin: {len(chans)} channels, CONV_IN_CH={n_in} "
          f"({'RGB' if n_in == 3 else 'grayscale'}), {chans[0][0].size} taps/channel")
    if "--skip-torch" not in sys.argv:
        q1_lum_vs_sum()
    q2_linear_diversity(chans, n_in)
    q3_bias_shift(chans, n_in)
    q4_post_relu(chans, n_in=n_in)
