#!/usr/bin/env python3
"""
scripts/check_ebox_crosscheck.py -- do the BOARD's and the OFFLINE bench's
`mask_ebox` compute the same statistic?

THE DEFECT THIS EXISTS FOR. `filter_box_energy_fraction()` (C++, runs on the
board) and `box_energy_fraction()` (Python, rgb_vs_gray_loop.py) are described
as "written against the same definition", and the whole value of the mask arm's
mechanism check rests on that: the offline half supplies the PREDICTED value
(0.51-0.55 at init unmasked, ~0.91 masked) that the board run is scored against.
They were not the same. The Python half transforms H to the spatial domain
first; the C++ half squared H directly. Both were self-consistent, both had
green tests, and the disagreement surfaced only as `mask_ebox = 0.0000` on every
frame of a hardware sweep.

A unit test cannot catch that -- each side agrees with itself. The only
instrument that can is this one: run BOTH implementations on the SAME H and
compare. That is also what makes the MUTANTS below meaningful; a cross-check
that has never been shown to fail is not evidence.

Usage
-----
  python3 scripts/check_ebox_crosscheck.py            # build, compare, verdict
  python3 scripts/check_ebox_crosscheck.py --keep     # leave the artifacts
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'design', 'host_app_src')
TOL = 1e-5

# (channels, rows, cols, box_rows, box_cols, seed). Powers of two on both axes,
# since the board refuses anything else; a rectangular case because a row/col
# swap in a 2-D transform is invisible on a square one.
CASES = [
    (1, 32, 32, 16, 16, 2026),
    (3, 32, 32, 12, 20, 7),
    (2, 64, 32, 24, 8, 99),
    (16, 16, 16, 8, 8, 4242),
]


def build(exe, extra_defs=()):
    cmd = ['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-I', SRC]
    cmd += list(extra_defs)
    cmd += [os.path.join(SRC, 'mosse_filter.cpp'),
            os.path.join(SRC, 'test', 'ebox_crosscheck.cpp'), '-o', exe]
    subprocess.run(cmd, check=True)


def board_value(exe, blob, case):
    ch, r, c, br, bc, seed = case
    out = subprocess.run([exe, blob, *map(str, (ch, r, c, br, bc, seed))],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def offline_value(blob, case):
    """The OFFLINE implementation, imported rather than reimplemented.

    Importing is the point: a second copy transcribed into this file would drift
    from the bench exactly the way the C++ half drifted from it, and the check
    would go on passing. rgb_vs_gray_loop imports numpy only at module level, so
    this costs nothing.
    """
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from rgb_vs_gray_loop import box_energy_fraction

    ch, r, c, br, bc, _ = case
    raw = np.fromfile(blob, dtype=np.float32).reshape(ch, r, c, 2)
    H = raw[..., 0] + 1j * raw[..., 1]
    return float(box_energy_fraction(H, br, bc))


def run_cases(exe, tmp, label, expect_agree=True):
    ok = True
    for case in CASES:
        blob = os.path.join(tmp, 'H.bin')
        got = board_value(exe, blob, case)
        ref = offline_value(blob, case)
        agree = abs(got - ref) <= TOL * max(1.0, abs(ref))
        if expect_agree:
            ok &= agree
            flag = 'OK  ' if agree else 'FAIL'
        else:
            ok &= not agree          # a mutant MUST disagree
            flag = 'caught' if not agree else 'MISSED'
        print(f"  {label:<34} ch{case[0]:<3} {case[1]}x{case[2]} "
              f"box {case[3]}x{case[4]}   board {got:.6f}  offline {ref:.6f}   {flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='ebox_')
    exe = os.path.join(tmp, 'ebox')
    failures = 0

    print("=== the shipping implementation, against the offline bench ===")
    build(exe)
    if not run_cases(exe, tmp, 'agreement'):
        failures += 1

    # MUTANTS. Each disables one property of the C++ side; each must make the
    # cross-check FAIL. A mutant that survives marks a property this check does
    # not actually test.
    #
    # They are injected through -D rather than by editing the source, so the
    # shipping file is never left modified by a failed run.
    print()
    print("=== mutants (each must be CAUGHT) ===")
    mutants = [
        ('no inverse transform (the shipped bug)', '-DEBOX_MUTANT=1'),
        ('box at the origin, not the centre',      '-DEBOX_MUTANT=2'),
        ('column pass skipped (1-D only)',       '-DEBOX_MUTANT=3'),
        ('forward transform instead of inverse',   '-DEBOX_MUTANT=4'),
        ('box extent off by one',                  '-DEBOX_MUTANT=5'),
    ]
    for name, define in mutants:
        mexe = os.path.join(tmp, 'ebox_m')
        build(mexe, [define])
        if not run_cases(mexe, tmp, name, expect_agree=False):
            failures += 1

    print()
    if failures:
        print(f"CROSS-CHECK FAILED ({failures} group(s))")
    else:
        print("CROSS-CHECK PASSED — the two implementations agree, and all "
              "5 mutants are caught")
    if args.keep:
        print(f"artifacts kept in {tmp}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
