#!/usr/bin/env python3
"""
check_kernel_bitexact.py — diff a kernel's real output against the Python model.

Reads the raw dump produced by design/aie_src/kernel_only_graph.cpp under x86sim
and compares it, SAMPLE BY SAMPLE WITH NO TOLERANCE, against the model in
gen_aiesim_vectors.py.

Why no tolerance: this exists to gate rewrites of conv2d and cmul_accum (the
vectorization work). The question a tolerance-based check cannot answer is "did
this rewrite change any sample?", and that is the only question that matters when
the stated goal is a bit-identical rewrite. aiesim's end-to-end scenarios already
cover the tolerant case.

It doubles as the first real test of the model itself. gen_aiesim_vectors.py
claims simulate_conv2d() "replicates the integer arithmetic in conv2d_kernel.cpp
exactly" — never verified against the kernel, and the offline shift-budget work
rests on it. A FAIL against the UNMODIFIED scalar kernel means the model is
wrong, not the kernel.

Usage (normally via `make x86sim_check`):
    python3 scripts/check_kernel_bitexact.py \
        --kernel conv2d \
        --scenario design/aie_src/aiesim_data/s6 \
        --actual build/x86sim/128x128/conv2d/kernel_out.bin
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# gen_aiesim_vectors reads PATCH_ROWS/PATCH_COLS/H_SHIFT etc. from the
# environment at import time. The Makefile target passes the same values the
# graph was compiled with; a mismatch here would compare against a differently
# shaped golden, so fail loudly rather than guess.
import gen_aiesim_vectors as G

MAX_REPORT = 10


def load_patch_int8(scenario: str) -> np.ndarray:
    """Recover the int8 patch from the PLIO stimulus the kernel actually read.

    Deliberately decoded from patch_in.txt rather than regenerated: that file is
    literally what was streamed into the kernel, so a packing bug in the
    generator shows up as a diff instead of cancelling out on both sides.

    Format (write_plio_txt): one signed int32 decimal per line, 4 int8 packed
    little-endian, followed by PLIO_PADDING_FRAMES frames of zero padding which
    we drop.
    """
    path = os.path.join(scenario, 'patch_in.txt')
    words = np.loadtxt(path, dtype=np.int64)
    need = G.N // 4
    if words.size < need:
        sys.exit(f"ERROR: {path} has {words.size} words, need at least {need}. "
                 f"Wrong PATCH_ROWS/PATCH_COLS, or stale vectors — run `make gen_vectors`.")
    words = words[:need].astype(np.uint32)
    b = np.empty(G.N, dtype=np.uint8)
    b[0::4] = (words) & 0xFF
    b[1::4] = (words >> 8) & 0xFF
    b[2::4] = (words >> 16) & 0xFF
    b[3::4] = (words >> 24) & 0xFF
    return b.view(np.int8)


def load_cint16(path: str):
    """Read a flat int16 LE cint16 file as (re, im) int64 arrays."""
    raw = np.fromfile(path, dtype='<i2')
    if raw.size != G.N * 2:
        sys.exit(f"ERROR: {path} has {raw.size} int16, expected {G.N * 2}")
    return raw[0::2].astype(np.int64), raw[1::2].astype(np.int64)


def report(name, exp, act):
    """Exact diff. Returns True on match."""
    exp = np.asarray(exp, dtype=np.int64).ravel()
    act = np.asarray(act, dtype=np.int64).ravel()
    if exp.size != act.size:
        print(f"  {name}: SIZE MISMATCH expected {exp.size}, got {act.size}")
        return False

    bad = np.nonzero(exp != act)[0]
    if bad.size == 0:
        print(f"  {name}: OK — {exp.size}/{exp.size} samples identical "
              f"(max|.|={np.abs(exp).max()})")
        return True

    print(f"  {name}: {bad.size} of {exp.size} samples differ "
          f"({100.0 * bad.size / exp.size:.2f}%)")
    delta = act[bad] - exp[bad]
    print(f"    delta: min={delta.min()} max={delta.max()} "
          f"mean={delta.mean():.3f}  |delta|<=1 for {np.sum(np.abs(delta) <= 1)} of {bad.size}")
    # A uniform +/-1 delta is the signature of a rounding-mode change (truncate
    # vs round-to-nearest), not a logic error. Worth naming, because the two have
    # very different fixes.
    if np.all(np.abs(delta) <= 1):
        print("    NOTE: every delta is within 1 LSB — this is the signature of a "
              "rounding-mode change (floor-shift vs srs), not a logic error.")
    print(f"    first {min(MAX_REPORT, bad.size)} mismatches (index: expected -> actual):")
    for i in bad[:MAX_REPORT]:
        print(f"      [{i}] {exp[i]} -> {act[i]}")
    return False


def check_conv2d(scenario: str, actual_path: str, ch: int, relu: int) -> bool:
    patch = load_patch_int8(scenario)

    wpath = os.path.join(scenario, f'weights_ch{ch}.bin')
    with open(wpath, 'rb') as f:
        weights = f.read()
    if len(weights) < 22:
        sys.exit(f"ERROR: {wpath} is {len(weights)} bytes, expected >= 22")

    # mean_prev lives at bytes [18:22] and the KERNEL reads it from there
    # (conv2d_kernel.cpp:135-139). Passing 0 instead would silently compare
    # against a Stage-B1-free golden. This layout is duplicated across four
    # files — see the preprocessing-coupling note in CLAUDE.md.
    mean_prev = struct.unpack_from('<i', weights, 18)[0]
    out_shift = weights[9]
    bias_acc = struct.unpack_from('<i', weights, 10)[0]
    print(f"  weights: out_shift={out_shift} bias_acc={bias_acc} mean_prev={mean_prev}")

    # relu is passed EXPLICITLY rather than inherited from G's GEN_CONV_RELU, so
    # this check states which datapath it is comparing against instead of
    # depending on an environment variable being set correctly.
    golden = G.simulate_conv2d(patch, weights, mean_prev,
                               relu=bool(relu)).astype(np.int64).ravel()

    act_re, act_im = load_cint16(actual_path)

    ok = report("conv2d real", golden, act_re)
    # The kernel writes imag = 0 unconditionally. If that ever changes, the whole
    # downstream FFT calibration changes with it.
    if np.any(act_im != 0):
        n = int(np.count_nonzero(act_im))
        print(f"  conv2d imag: FAIL — {n} non-zero imaginary parts, kernel must emit 0")
        ok = False
    else:
        print(f"  conv2d imag: OK — all zero")
    return ok


def check_cmul(scenario: str, actual_path: str) -> bool:
    in_re, in_im = load_cint16(os.path.join(scenario, 'fft_col_in.bin'))
    f_re, f_im = load_cint16(os.path.join(scenario, 'cmul_filter.bin'))
    a_re, a_im = load_cint16(os.path.join(scenario, 'cmul_accum.bin'))

    print(f"  H_SHIFT={G.H_SHIFT}  max|F|={np.abs(in_re).max()} "
          f"max|H|={np.abs(f_re).max()} max|acc_prev|={np.abs(a_re).max()}")

    exp_re, exp_im = G.simulate_cmul(in_re, in_im, f_re, f_im, a_re, a_im)
    act_re, act_im = load_cint16(actual_path)

    ok_re = report("cmul real", exp_re, act_re)
    ok_im = report("cmul imag", exp_im, act_im)
    return ok_re and ok_im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', required=True, choices=['conv2d', 'cmul'])
    ap.add_argument('--scenario', required=True,
                    help='scenario directory, e.g. design/aie_src/aiesim_data/s6')
    ap.add_argument('--actual', required=True,
                    help='kernel_out.bin dumped by kernel_only_graph.cpp')
    ap.add_argument('--ch', type=int, default=0,
                    help='conv2d weight channel under test. NOT interchangeable: '
                         'on s6, ReLU never fires for ch0 (nor 12 of 16 channels), '
                         'so ch0 cannot distinguish CONV_RELU=1 from 0. ch11 clamps '
                         'some but not all pixels.')
    ap.add_argument('--relu', type=int, default=1,
                    help='model ReLU (1) or not (0) — must match the CONV_RELU the '
                         'kernel was built with, or the diff is meaningless.')
    args = ap.parse_args()

    if not os.path.exists(args.actual):
        sys.exit(f"ERROR: {args.actual} does not exist — the x86sim run did not "
                 f"produce a dump. Check x86sim.log.")

    print(f"[bitexact] kernel={args.kernel} scenario={args.scenario} ch={args.ch} relu={args.relu} "
          f"geometry={G.PATCH_ROWS}x{G.PATCH_COLS}")

    if args.kernel == 'conv2d':
        ok = check_conv2d(args.scenario, args.actual, args.ch, args.relu)
    else:
        ok = check_cmul(args.scenario, args.actual)

    print(f"\n=== BIT-EXACT: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        print("A FAIL against the UNMODIFIED scalar kernel means the Python model is\n"
              "wrong, not the kernel — and every offline number derived from it is\n"
              "suspect. A FAIL after a kernel rewrite means the rewrite changed the\n"
              "arithmetic.")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
