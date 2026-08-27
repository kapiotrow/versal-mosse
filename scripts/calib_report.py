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
# Matches the frame HEADER only. `Frame N:` alone is not enough: the per-frame
# block also carries `Frame N: displacement ...` AFTER the F_ch/accum/response
# scans and BEFORE the H(q15) scan, so a loose pattern starts a second, empty
# record for the same index mid-frame. That is not a cosmetic parse loss — the
# report then scans only H(q15) and prints "RAILS: none OK" over a run that
# railed the accumulator (run_calib.log frame 173, accum rails=2). Same failure
# class as the anchoring bug above: a parser that finds nothing looks exactly
# like a clean run. Belt and braces — parse_log() also MERGES repeated indices,
# so neither half alone can lose a scan.
FRAME_RE = re.compile(r'(?:^|\s)Frame\s+(\d+):\s+(?:target at|\[INIT\])')


def parse_log(path):
    """Per-frame {tag: (max, rails)}. Repeated indices MERGE, never overwrite."""
    frames, idx = {}, -1
    for line in Path(path).read_text(errors='replace').splitlines():
        m = FRAME_RE.search(line)
        if m:
            idx = int(m.group(1))
            frames.setdefault(idx, {})
            continue
        m = RAILS_RE.search(line)
        if m and idx >= 0:
            frames[idx][m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return {f: d for f, d in frames.items() if d}


def fnum(key):
    """The frame INDEX out of a frames{} key.

    Keys are a bare int at FRAME_SOURCE=synth and a (job, frame) tuple on a
    multi-start CSV. Everything that asks "is this an early frame?" means the
    frame index within its own run, not a position in the file -- on a
    multi-start sweep frame 5 of job 12 is as early as frame 5 of job 0, and
    the convergence-growth trap this report exists to catch is per RUN.
    """
    return key[1] if isinstance(key, tuple) else key


def fname(key):
    """How a frames{} key prints."""
    return f"job {key[0]} frame {key[1]}" if isinstance(key, tuple) else f"frame {key}"


def check_coverage(frames):
    """Warn when frames disagree about WHICH scans they carry.

    The whole point: an under-parse is silent. Every frame emits the same set
    of [diag] tags, so a frame missing one means the parser dropped it, not
    that the run skipped it.
    """
    if not frames:
        print("\n  WARNING: no [diag] scans parsed at all — check the log format"
              " before reading anything below.")
        return
    full = max((set(d) for d in frames.values()), key=len)
    short = {f: sorted(full - set(d)) for f, d in frames.items() if set(d) != full}
    if short:
        print(f"\n  WARNING: {len(short)} of {len(frames)} frame(s) are missing"
              f" scans that other frames carry — the parser may be dropping them.")
        for f in sorted(short)[:5]:
            print(f"    frame {f:>4}  missing {', '.join(short[f])}")


def parse_csv_frames(rows):
    """parse_log()'s shape, rebuilt from track.csv's diag columns.

    Those columns (rails, accum_max, fch0_max, h_max) were added 2026-08-24 so a
    VERBOSITY=0 run — the only kind that can also be an FPS measurement — stays
    diagnosable. Before that the amplitudes existed ONLY in the console, and at
    VERBOSITY=0 the [diag] lines print only when something rails, so a healthy
    run yielded nothing at all.

    Returns (frames, rails_by_frame, have_cols). have_cols is False for a CSV
    that predates the columns, so the caller can report "no data" rather than
    the much more dangerous "no rails".
    """
    if not rows or 'rails' not in rows[0]:
        return {}, {}, False
    frames, rails = {}, {}
    # amplitude column, and the PER-BUFFER rails column when the CSV has it.
    #
    # `resp_max` (2026-08-26+) is the response scan's own maximum. Before it,
    # `peak` stood in -- and it is a good stand-in but not the same quantity:
    # `peak` is the SIGNED REAL PART at the argmax of |real|, so it is blind to
    # a bin saturating in the imaginary part alone. Measured agreement on the
    # runs that had both: 199/199 synthetic, 739/741 on car1, the exceptions at
    # amplitude ~25 of 32767. Prefer resp_max, fall back to peak, and say which.
    col = [('response', 'resp_max', 'rails_resp'),
           ('accum',    'accum_max', 'rails_accum'),
           ('F_ch',     'fch0_max',  'rails_fch'),
           ('H(q15)',   'h_max',     'rails_h')]
    if 'resp_max' not in rows[0]:
        col[0] = ('response', 'peak', 'rails_resp')
    per_buffer = 'rails_resp' in rows[0]
    # KEY BY (job, frame), NOT frame. A multi-start track.csv carries every
    # anchor of a sequence in one file, so a bare frame index collides across
    # runs and the last job silently overwrites the other fourteen. Measured on
    # runs/vot/0825_1919-smoke/track_car1.csv: 8434 rows collapsed to 742
    # frames and 266 railed frames were reported as 4 -- the tool that is
    # supposed to be the rails gate under-reported them by 66x, and it did so
    # while printing a confident "BUDGET IS WRONG" that happened to be right for
    # the wrong reason. A composite key costs nothing at FRAME_SOURCE=synth,
    # where `job` is absent and this degenerates to the old behaviour exactly.
    multi = 'job' in rows[0]
    for r in rows:
        try:
            f = int(r['frame'])
        except (KeyError, ValueError):
            continue
        if multi and r.get('job') not in (None, ''):
            f = (int(r['job']), f)
        d = {}
        for tag, c, rc in col:
            if r.get(c) in (None, ''):
                continue
            # Per-buffer rails when the CSV carries them; 0 otherwise, with the
            # frame TOTAL still reported separately. An unattributed total is
            # what made the 2026-08-25 car1 evidence read as an accumulator
            # problem when the console said the response railed twice as often.
            n = int(r[rc] or 0) if per_buffer and r.get(rc) not in (None, '') else 0
            d[tag] = (abs(float(r[c])), n)
        if d:
            frames[f] = d
            rails[f] = int(r['rails'] or 0)
    return frames, rails, True


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
    csv_rows = parse_csv(csv_path) if Path(csv_path).exists() else []
    csv_frames, csv_rails, csv_has_cols = parse_csv_frames(csv_rows)
    source = "console [diag]"
    if not frames and csv_frames:
        # VERBOSITY=0 run: the console has nothing, the CSV has everything.
        frames, source = csv_frames, f"{csv_path} (VERBOSITY=0 run)"
    tags_seen = sorted({t for d in frames.values() for t in d})
    print(f"\nframes with amplitude data: {len(frames)}   source: {source}"
          f"\n  buffers: {', '.join(tags_seen) if tags_seen else '(none)'}")
    check_coverage(frames)

    # ---- rails: the hard gate --------------------------------------------
    railed = {f: {t: v for t, v in d.items() if v[1] > 0}
              for f, d in frames.items()}
    railed = {f: d for f, d in railed.items() if d}
    if not railed and csv_rails:
        # Only reachable on a PRE-2026-08-26 CSV, which carries the per-frame
        # total and no attribution. Kept so an old run still reports its rails
        # rather than reporting none -- but the buffer genuinely is unknown
        # there, and the label has to say so rather than pick a plausible one.
        railed = {f: {'(unattributed)': (0, n)} for f, n in csv_rails.items() if n}
    print("\nRAILS (must be zero on EVERY frame)")
    if not frames and not csv_has_cols:
        # "none" and "never looked" are different answers and must not print
        # the same word. This log has no [diag] scans at all (VERBOSITY=0
        # prints them only when something rails), so the run is UNCHECKED.
        print("  NOT CHECKED — no [diag] scans in this log and no rails column"
              " in the CSV.\n  Re-run with VERBOSITY=1, or rebuild so track.csv"
              " carries rails (2026-08-24+).")
    elif not railed:
        print("  none   OK")
    else:
        print(f"  {len(railed)} frame(s) railed   <-- BUDGET IS WRONG")
        for f in sorted(railed)[:10]:
            for tag, (mx, n) in railed[f].items():
                print(f"    {fname(f):<20} {tag:<14} rails={n} max={mx:.0f}")
        if len(railed) > 10:
            print(f"    ... and {len(railed) - 10} more")
        # Earliest by frame index WITHIN a run, not by file order: the trap
        # this line names is a filter converging, which restarts every run.
        first = min(railed, key=fnum)
        # Per-buffer totals. WHICH buffer rails picks the knob: H_SHIFT is
        # upstream of both the accumulator and the response, IFFT_ROW_SHIFT /
        # IFFT_COL_SHIFT reach only the response. An unattributed count cannot
        # choose between them.
        by_tag = {}
        for d in railed.values():
            for tag, (_, n) in d.items():
                by_tag[tag] = by_tag.get(tag, 0) + (n if n else 1)
        print("  by buffer: " + ", ".join(f"{t}={n}" for t, n in
                                          sorted(by_tag.items(), key=lambda x: -x[1])))
        print(f"  FIRST railed: {fname(first)}"
              + ("   (frame 1 is fine but LATE frames rail — this is the"
                 " convergence-growth trap)" if fnum(first) > 5 else ""))

    # ---- response amplitude, early vs converged --------------------------
    tags = tags_seen
    print("\nAMPLITUDE by buffer  (target for `response`: 49-64% of int16 at the"
          " converged end)")
    for tag in tags:
        early = [d[tag][0] for f, d in frames.items() if 1 <= fnum(f) <= 20 and tag in d]
        late = [d[tag][0] for f, d in frames.items() if fnum(f) > 20 and tag in d]
        lo, hi = (TARGET_LO, TARGET_HI) if tag.startswith('resp') else (None, None)
        band(f"{tag}  frames 1-20", early)
        band(f"{tag}  frames 21+", late, lo, hi)

    # ---- tracking, from the CSV ------------------------------------------
    if csv_rows:
        rows = csv_rows
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
