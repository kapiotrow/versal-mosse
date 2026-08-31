#!/usr/bin/env python3
"""
scripts/vot_mask_stat.py -- read the `mask_ebox` column: DID THE MASK DO WHAT IT
IS THEORISED TO DO?

`FILTER_MASK_STAT=1` logs, per frame, the fraction of the filter's energy
(sum |h|^2 over all 16 channels) that sits inside a centred box the size of the
target. It is the mechanism check for `FILTER_MASK`, and it is deliberately
INDEPENDENT of it so the baseline arm is measurable with the same instrument --
see docs/thesis/evidence/proposed_build_mask.md sec.4.

The check the evidence doc asks for: `mask_ebox` should sit near 0.91 at init
and 0.93-0.99 thereafter on a masked arm, against a baseline that starts near
0.52-0.55 and CLIMBS. If EAO moves while the two arms do not separate here, the
gain is not the mask and the result is unattributable.

THREE WAYS THIS COLUMN IS EASY TO READ WRONG, all of them guarded below.

  1. `-1` IS NOT ZERO. It means the value was not measured on that frame:
     frame 0's filter_init() path, every held frame, and -- since the statistic
     costs an inverse FFT per channel -- every frame outside the sampling
     schedule (FILTER_MASK_STAT_WARM / _EVERY). Averaging it in as a zero
     measures a mixture of the hold rate and the schedule and calls it a filter
     statistic. Those rows are excluded here and counted separately.

  2. THE FRACTION RISES AS THE FILTER CONVERGES. car1 goes 0.514 -> 0.741
     unmasked over 39 frames. So the 51.6% / 54.9% figures quoted in the docs
     are AT-INIT values, and comparing a run's MEAN against them would
     "confirm" the mechanism on an unmasked arm. Compare like with like: this
     reports at-init and per-frame-index profiles, and only then a mean.

  3. A BARE FRAME INDEX IS NOT A KEY. A FRAME_SOURCE=vot CSV holds every anchor
     of a sequence in one file. Keying on frame alone collapsed 8434 rows to
     742 once and under-reported rails by 66x. Everything here is keyed on
     (sequence, job, frame).

The offline half of the same statistic is `rgb_vs_gray_loop.py`'s `e_box`
column. They were once merely DESCRIBED as sharing a definition and did not --
the board half omitted the inverse transform and read 0.0000 on every frame of a
whole sweep. `scripts/check_ebox_crosscheck.py` now runs both implementations on
the same H and is mutation-tested, so the agreement is enforced rather than
asserted.

Usage
-----
  python3 scripts/vot_mask_stat.py runs/vot/<dir>                 # one arm
  python3 scripts/vot_mask_stat.py runs/vot/<base> runs/vot/<mask>  # paired
  python3 scripts/vot_mask_stat.py <base> <mask> --per-sequence
"""

# @thesis sec:metodykaBadan | O-01 | The mechanism check for the spatial mask: mask_ebox read
#   at-init and per frame index, never as a pooled mean, with the -1 rows excluded rather than
#   averaged in as zeros.

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

COL = 'mask_ebox'

# Frame indices profiled, matching the table in proposed_build_mask.md sec.4 so
# the board's numbers land next to the offline ones they are predicted from.
PROFILE_FRAMES = (1, 5, 20, 40)

# The offline prediction, quoted so a reader does not have to go and find it.
# NOT a pass criterion -- EAO is the arbiter for this arm; this is attribution.
PREDICT = {'baseline_init': (0.514, 0.550), 'mask_init': (0.913, 0.913)}


def load(run_dir):
    """{(sequence, job): {frame: value}} plus the row counts.

    Values are floats; -1 rows are kept out of the map and counted, because a
    caller that receives them cannot tell them from a filter whose energy
    really is at -1 -- which is to say, cannot tell them from nothing.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, 'track_*.csv')))
    if not paths:
        raise SystemExit(f"no track_*.csv in {run_dir}")

    runs, n_rows, n_held = defaultdict(dict), 0, 0
    # Every run key seen, INCLUDING those that never yielded a sample. Without
    # this the loader silently reports a smaller run count than the sweep wrote
    # -- 412 against 419 on the first full arm -- and a reader cannot tell a
    # missing arm from runs that simply never re-formed H.
    all_keys = set()
    seen_column = False
    for path in paths:
        seq = os.path.basename(path)[len('track_'):-len('.csv')]
        with open(path) as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames is None or COL not in rd.fieldnames:
                # A missing column and a column of zeros must not print the same
                # word. This run was built without FILTER_MASK_STAT=1 and there
                # is nothing here to read.
                raise SystemExit(
                    f"{path} has no `{COL}` column -- this arm was built without\n"
                    f"FILTER_MASK_STAT=1, so the mechanism check cannot be run on it.\n"
                    f"Rebuild with FILTER_MASK_STAT=1 and re-sweep; the flag is\n"
                    f"independent of FILTER_MASK precisely so a BASELINE can carry it.")
            seen_column = True
            for r in rd:
                n_rows += 1
                all_keys.add((seq, r['job']))
                v = float(r[COL])
                if v < 0.0:
                    n_held += 1
                    continue
                runs[(seq, r['job'])][int(r['frame'])] = v

    if not seen_column or not runs:
        raise SystemExit(f"{run_dir}: `{COL}` is present but never valid -- every row is -1")
    return runs, n_rows, n_held, len(all_keys)


def at_init(frames):
    """The value on the FIRST frame that re-formed H, not on frame 0.

    Frame 0 takes the filter_init() path and logs -1, so the first valid row is
    normally frame 1. Taking `frames[1]` directly would silently drop any run
    whose frame 1 was held.
    """
    return frames[min(frames)] if frames else None


def quantiles(x):
    s = sorted(x)
    return (st.median(s), s[int(0.25 * len(s))], s[int(0.75 * len(s))])


def describe(runs, n_rows, n_held, n_all_runs, label):
    inits = [v for v in (at_init(f) for f in runs.values()) if v is not None]
    med, p25, p75 = quantiles(inits)

    empty = n_all_runs - len(runs)
    print(f"{label}: {len({k[0] for k in runs})} sequences, {n_all_runs} runs, "
          f"{n_rows} rows")
    if empty:
        # A run whose gate vetoed every SAMPLED frame never re-formed H, so it
        # contributes nothing here. Reporting only the runs that did would
        # understate the arm and hide a difference between arms.
        print(f"  {empty} run(s) yielded NO sample at all (gate vetoed every "
              f"sampled frame); {len(runs)} runs contribute below")
    print(f"  {COL} valid on {n_rows - n_held} rows ({100*(n_rows-n_held)/n_rows:.1f}%); "
          f"{n_held} rows are -1 and are EXCLUDED")
    print("    -1 = NOT MEASURED: frame 0's filter_init path, a held frame, or a")
    print("    frame outside the sampling schedule (FILTER_MASK_STAT_WARM/_EVERY).")
    print("    Since the schedule dominates that count, it is NOT a hold rate --")
    print("    read holds from the gate columns, not from here.")
    print()
    print(f"  at init (first re-formed frame of each run)")
    print(f"    median {med:.4f}   p25 {p25:.4f}   p75 {p75:.4f}   n={len(inits)}")
    print(f"  by frame index")
    for j in PROFILE_FRAMES:
        vals = [f[j] for f in runs.values() if j in f]
        if vals:
            print(f"    f{j:<4} median {st.median(vals):.4f}   n={len(vals)}")
    means = [st.mean(f.values()) for f in runs.values() if f]
    print(f"  per-run mean over valid rows      median {st.median(means):.4f}")
    print()
    print("  THE FRACTION RISES AS THE FILTER CONVERGES. Compare at-init against")
    print("  at-init; a mean held up against the 0.516/0.549 at-init figures in")
    print("  the docs would 'confirm' the mechanism on an unmasked arm.")
    return inits


def paired(a, b, label_a, label_b, per_sequence=False):
    """Compare two arms on the runs, and then the FRAMES, they share.

    The two arms hold on different frames, so their valid-row sets differ. A
    pooled mean per arm is therefore taken over different frames -- the same
    structural point this project already makes about accuracy being averaged
    over tracked frames. The headline number here is restricted to the frames
    valid in BOTH arms; the pooled one is printed beside it so the size of the
    effect is visible rather than assumed.
    """
    keys = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(f"paired comparison   A = {label_a}   B = {label_b}")
    print(f"  {len(keys)} runs in both", end='')
    if only_a or only_b:
        print(f"; {len(only_a)} only in A, {len(only_b)} only in B "
              f"-- THE ARMS ARE NOT THE SAME RUN SET")
    else:
        print(" (identical run sets)")
    if not keys:
        raise SystemExit("no runs in common -- are these two arms of the same sweep?")

    ia = [v for v in (at_init(a[k]) for k in keys) if v is not None]
    ib = [v for v in (at_init(b[k]) for k in keys) if v is not None]
    ma, qa25, qa75 = quantiles(ia)
    mb, qb25, qb75 = quantiles(ib)
    print()
    print(f"  {'':<10} {'A':>8} {'B':>8} {'B - A':>8}")
    print(f"  {'at init':<10} {ma:8.4f} {mb:8.4f} {mb-ma:+8.4f}"
          f"    (A p25/p75 {qa25:.3f}/{qa75:.3f}, B {qb25:.3f}/{qb75:.3f})")
    for j in PROFILE_FRAMES:
        va = [a[k][j] for k in keys if j in a[k]]
        vb = [b[k][j] for k in keys if j in b[k]]
        if va and vb:
            print(f"  {'f%d' % j:<10} {st.median(va):8.4f} {st.median(vb):8.4f} "
                  f"{st.median(vb)-st.median(va):+8.4f}")

    ca, cb = [], []
    for k in keys:
        for f in a[k].keys() & b[k].keys():
            ca.append(a[k][f])
            cb.append(b[k][f])
    print(f"  {'common':<10} {st.mean(ca):8.4f} {st.mean(cb):8.4f} "
          f"{st.mean(cb)-st.mean(ca):+8.4f}    over {len(ca)} frames valid in BOTH arms")
    pa = [v for k in keys for v in a[k].values()]
    pb = [v for k in keys for v in b[k].values()]
    print(f"  {'pooled':<10} {st.mean(pa):8.4f} {st.mean(pb):8.4f} "
          f"{st.mean(pb)-st.mean(pa):+8.4f}    over {len(pa)} / {len(pb)} rows "
          f"-- DIFFERENT frame sets")

    print()
    d = mb - ma
    separated = d > 0.10 and qb25 > qa75
    print(f"  offline prediction: baseline {PREDICT['baseline_init'][0]:.3f}-"
          f"{PREDICT['baseline_init'][1]:.3f} at init, masked "
          f"{PREDICT['mask_init'][0]:.3f}")
    if separated:
        print(f"  VERDICT: the arms SEPARATE at init (delta {d:+.4f}, and B's p25 "
              f"{qb25:.3f} clears A's p75 {qa75:.3f}).")
        print("  The instrument sees the mask. An EAO move is now ATTRIBUTABLE to it.")
    else:
        print(f"  VERDICT: the arms DO NOT separate at init (delta {d:+.4f}"
              f"{'' if qb25 > qa75 else ', distributions overlap'}).")
        print("  Any EAO move is UNATTRIBUTABLE -- do not report it as the mask's.")
    print()
    print("  This is ATTRIBUTION, not acceptance. EAO on `vot analysis` is the")
    print("  arbiter for an A/R trade; see proposed_build_mask.md sec.4.")

    if per_sequence:
        print()
        print(f"  at-init by sequence   {'A':>8} {'B':>8} {'B - A':>8}   runs")
        per = defaultdict(lambda: ([], []))
        for k in keys:
            va, vb = at_init(a[k]), at_init(b[k])
            if va is not None and vb is not None:
                per[k[0]][0].append(va)
                per[k[0]][1].append(vb)
        for seq in sorted(per, key=lambda s: st.median(per[s][1]) - st.median(per[s][0])):
            xa, xb = per[seq]
            print(f"    {seq:<18} {st.median(xa):8.4f} {st.median(xb):8.4f} "
                  f"{st.median(xb)-st.median(xa):+8.4f}   {len(xa)}")


def main():
    ap = argparse.ArgumentParser(
        description="read the mask_ebox column -- the FILTER_MASK mechanism check")
    ap.add_argument('run_dir', help='the arm to read (the BASELINE when two are given)')
    ap.add_argument('mask_dir', nargs='?',
                    help='the masked arm; given, the two are compared pairwise')
    ap.add_argument('--per-sequence', action='store_true',
                    help='paired mode: at-init per sequence, worst separation first')
    args = ap.parse_args()

    a, na, ha, ka = load(args.run_dir)
    describe(a, na, ha, ka, args.run_dir)

    if args.mask_dir:
        print()
        b, nb, hb, kb = load(args.mask_dir)
        describe(b, nb, hb, kb, args.mask_dir)
        print()
        paired(a, b, args.run_dir, args.mask_dir, args.per_sequence)


if __name__ == '__main__':
    main()
