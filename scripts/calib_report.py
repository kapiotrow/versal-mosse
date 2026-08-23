#!/usr/bin/env python3
"""
calib_report.py — turn a calibration run into a shift-budget VERDICT.

    scripts/calib_report.py <console.log> [track.csv]

WHY THIS EXISTS
---------------
CLAUDE.md's measurement rules include "instrument the two candidate mechanisms in
ONE run and let the log print the verdict" and "measure the total and print the
residual". A shift-budget run produces 200 frames of console plus a CSV; reading
either by eye is how a budget gets accepted on frame 1 and rails from frame 15.

The three rules this checks, from the shift-budget section:
  1. THE RESPONSE GROWS AS THE FILTER CONVERGES. Frame 1 proves nothing, so the
     early window and the converged window are reported SEPARATELY.
  2. rails=0 on every frame, or the budget is wrong. rails is the one number
     track.csv does not carry, so it is parsed from the console.
  3. A budget that undershoots is as wrong as one that rails: the retired 4-5-5
     point sat at 1.1-4.5% of int16 range. Target is 49-64%, from the validated
     4-4-4 run.

It reports; it does not decide for you. A FAIL here is a reason to look, not
proof of a defect — s7's PSR threshold entry is the cautionary tale.
"""

import csv
import re
import sys
from pathlib import Path

INT16_MAX = 32767.0
# The band the validated 4-4-4 run occupied at the converged end.
TARGET_LO, TARGET_HI = 0.49, 0.64

RAILS_RE = re.compile(r'\[diag\]\s+(\S+)\s+max\|\.\|=\s*(\d+)\s+.*rails=(\d+)')
# NOT anchored at line start. Board logs are captured with
#   picocom ... | ts '%H:%M:%.S' | tee log
# which is the documented way to timestamp a run without rebuilding, so every
# line carries a time prefix. An anchored ^Frame matched nothing and the report
# silently claimed one frame — found by running this against an existing log
# instead of against the run it was written for.
FRAME_RE = re.compile(r'(?:^|\s)Frame\s+(\d+)\s*:')


def parse_log(path):
    """Per-frame {tag: (max, rails)} plus the frames where anything railed."""
    frames, cur, idx = {}, {}, -1
    for line in Path(path).read_text(errors='replace').splitlines():
        m = FRAME_RE.search(line)
        if m:
            if cur:
                frames[idx] = cur
            idx, cur = int(m.group(1)), {}
            continue
        m = RAILS_RE.search(line)
        if m:
            cur[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    if cur:
        frames[idx] = cur
    return frames


def parse_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: r[k] for k in r})
            except KeyError:
                pass
    return rows


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return dict(n=n, lo=s[0], hi=s[-1], med=s[n // 2],
                mean=sum(s) / n)


def band(label, vals, verdict_lo=None, verdict_hi=None):
    st = stats(vals)
    if not st:
        print(f"  {label:<26} (no data)")
        return None
    frac_lo, frac_hi = st['lo'] / INT16_MAX, st['hi'] / INT16_MAX
    mark = ""
    if verdict_lo is not None:
        if frac_hi < verdict_lo:
            mark = "   <-- UNDERSHOOT"
        elif frac_lo > verdict_hi:
            mark = "   <-- HOT"
    print(f"  {label:<26} n={st['n']:<4} min {st['lo']:>8.0f} ({frac_lo*100:4.1f}%)"
          f"  med {st['med']:>8.0f}  max {st['hi']:>8.0f} ({frac_hi*100:4.1f}%){mark}")
    return st


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    log = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else 'track.csv'

    print(f"\ncalibration report — log {log}")
    print("=" * 78)

    frames = parse_log(log)
    print(f"\nframes with [diag] scans: {len(frames)}")

    # ---- rails: the hard gate --------------------------------------------
    railed = {f: {t: v for t, v in d.items() if v[1] > 0}
              for f, d in frames.items()}
    railed = {f: d for f, d in railed.items() if d}
    print("\nRAILS (must be zero on EVERY frame)")
    if not railed:
        print("  none   OK")
    else:
        print(f"  {len(railed)} frame(s) railed   <-- BUDGET IS WRONG")
        for f in sorted(railed)[:10]:
            for tag, (mx, n) in railed[f].items():
                print(f"    frame {f:>4}  {tag:<10} rails={n} max={mx:.0f}")
        if len(railed) > 10:
            print(f"    ... and {len(railed) - 10} more")
        first = min(railed)
        print(f"  FIRST railed frame: {first}"
              + ("   (frame 1 is fine but LATE frames rail — this is the"
                 " convergence-growth trap)" if first > 5 else ""))

    # ---- response amplitude, early vs converged --------------------------
    tags = sorted({t for d in frames.values() for t in d})
    print("\nAMPLITUDE by buffer  (target for `response`: 49-64% of int16 at the"
          " converged end)")
    for tag in tags:
        early = [d[tag][0] for f, d in frames.items() if 1 <= f <= 20 and tag in d]
        late = [d[tag][0] for f, d in frames.items() if f > 20 and tag in d]
        lo, hi = (TARGET_LO, TARGET_HI) if tag.startswith('resp') else (None, None)
        band(f"{tag}  frames 1-20", early)
        band(f"{tag}  frames 21+", late, lo, hi)

    # ---- tracking, from the CSV ------------------------------------------
    if Path(csv_path).exists():
        rows = parse_csv(csv_path)
        ious = [float(r['iou']) for r in rows if r.get('iou')]
        cerr = [float(r['centre_err']) for r in rows if r.get('centre_err')]
        peaks = [float(r['peak']) for r in rows if r.get('peak')]
        holds = [r for r in rows if r.get('accept') == '0' and r.get('evaluated') == '1']
        print(f"\nTRACKING  ({csv_path}, {len(rows)} rows)")
        if ious:
            print(f"  mean IoU {sum(ious)/len(ious):.4f}   worst {min(ious):.4f}"
                  f"   (comparator: 0.9188 / 0.8353)")
        if cerr:
            print(f"  centre err mean {sum(cerr)/len(cerr):.2f} px"
                  f"   worst {max(cerr):.2f}   (comparator: 1.37 / 3.52)")
        if peaks:
            band("csv peak frames 21+",
                 [float(r['peak']) for r in rows
                  if r.get('peak') and int(r['frame']) > 20],
                 TARGET_LO, TARGET_HI)
        print(f"  gate holds: {len(holds)}   (comparator: 0)")
    else:
        print(f"\nTRACKING: {csv_path} not found — IoU is the only metric that can"
              " fail a confidently-wrong tracker, so fetch it before concluding.")

    print("\n" + "=" * 78)
    print("REMINDERS")
    print("  * PSR cannot fail a confidently-wrong tracker (33 at 179 px off).")
    print("    Read IoU.")
    print("  * A budget validated at ITER_CNT=2 is not validated.")
    print("  * runs/.last_cfg is not authoritative; build/hw/.../calib_cfg.txt and")
    print("    the flagstamps are.")
    print()


if __name__ == '__main__':
    main()
