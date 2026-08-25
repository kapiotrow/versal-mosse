#!/usr/bin/env python3
"""
scripts/vot_check_trajectory.py -- does the BOARD's trajectory writer produce a
file the VOT toolkit reads back as the boxes the board meant?

WHY THIS EXISTS, GIVEN PHASE 0b ALREADY ROUND-TRIPPED THE FORMAT
----------------------------------------------------------------
Phase 0b wrote its trajectories with the toolkit's OWN Trajectory.write(), which
proves the format and the analysis pipeline, and proves nothing about the writer
that will actually run on the board. That writer is a printf in
design/host_app_src/vot_source.cpp and it performs the one piece of arithmetic
in the whole result path: the tracker's CENTRE box (row, col, h, w) becomes the
toolkit's top-left (x, y, w, h). A transposed pair or a sign there produces a
perfectly well-formed file, and the toolkit scores it without complaint.

WHAT MAKES THIS AN INDEPENDENT CHECK
------------------------------------
The C++ side emits `expect.txt` carrying the boxes it was HANDED, in centre
convention -- its INPUT, not its output. The conversion is re-derived here, and
the file is parsed by the toolkit's own reader rather than by a second copy of
our format rules. So a wrong conversion in the writer disagrees with this script
even though both sides come from the same source tree, whereas an expectation
file produced by the writer itself would agree no matter what it did. That is
the trap `generate_scenario`'s corrupted weights_ch0.bin taught this project:
zero tolerance is not enough when both sides share an input.

Usage:  make test_vot_format
        (or) vot_check_trajectory.py <dir-written-by 'test_vot_source <dir>'>
"""

import sys
from pathlib import Path

from vot.region import Special
from vot.region.io import read_trajectory
from vot.region.shapes import Rectangle

TOL = 1e-4          # the writer prints %.4f; nothing here needs more


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    d = Path(sys.argv[1])

    expect = []
    for line in (d / "expect.txt").read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        row, col, h, w = (float(v) for v in line.split(','))
        # centre -> top-left, derived here and not read from the writer
        expect.append((col - w / 2.0, row - h / 2.0, w, h))

    traj = sorted(d.glob("*_[0-9]*.txt"))
    if len(traj) != 1:
        raise SystemExit(f"expected exactly one trajectory in {d}, found {len(traj)}")
    regions = read_trajectory(str(traj[0]))

    checks = []
    checks.append(("filename is {sequence}_{anchor:08d}.txt",
                   traj[0].stem.split('_')[-1].isdigit() and
                   len(traj[0].stem.split('_')[-1]) == 8))
    checks.append(("region count == 1 init + rectangles",
                   len(regions) == len(expect) + 1))
    checks.append(("index 0 is Special(INITIALIZATION), not a box",
                   isinstance(regions[0], Special) and regions[0].code == 1))
    ok_boxes = len(regions) == len(expect) + 1
    if ok_boxes:
        for r, e in zip(regions[1:], expect):
            if not isinstance(r, Rectangle):
                ok_boxes = False
                break
            got = (r.x, r.y, r.width, r.height)
            if any(abs(a - b) > TOL for a, b in zip(got, e)):
                print(f"  got {got}  expected {e}")
                ok_boxes = False
    checks.append(("every box reads back as the centre box converted", ok_boxes))

    times = d / (traj[0].stem + "_time.value")
    lines = times.read_text().splitlines() if times.exists() else []
    checks.append(("time sidecar exists with one float per region",
                   len(lines) == len(regions) and
                   all(line.strip() and float(line) >= 0 for line in lines)))

    bad = 0
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += not ok
    if bad:
        raise SystemExit(f"{bad} check(s) failed")
    print("\nboard writer -> toolkit reader: EXACT")


if __name__ == '__main__':
    main()
