#!/usr/bin/env python3
"""Merge one-arm-per-file grid cells into a single JSON with distinct arm names.

`offline_sweep_par.sh` runs one (sigma, eta) cell per invocation, and every cell
carries the SAME arm name `rgb` because sigma and eta are global flags in
rgb_vs_gray_loop.py, not per-arm suffixes. vot_ar_offline.py compares two arms
WITHIN one file, so a cross-cell comparison needs the arm renamed per cell first.

  scripts/merge_grid.py <out.json> <cell.json> [cell.json ...]

The new arm name is the cell's file stem, so s4_e005.json -> arm `s4_e005`.
"""
import json, sys
from pathlib import Path

out, cells = sys.argv[1], sys.argv[2:]
merged, seqs = {}, None
for c in cells:
    name = Path(c).stem
    d = json.load(open(c))
    got = set()
    for k, v in d.items():
        seq, arm = k.rsplit('|', 1)
        if arm != 'rgb':
            raise SystemExit(f"{c}: expected arm 'rgb', found {arm!r}")
        merged[f"{seq}|{name}"] = v
        got.add(seq)
    # Every cell must cover the SAME sequences, or a pooled comparison silently
    # averages different sets -- the defect vot_ingest.py's length check exists for.
    if seqs is None:
        seqs = got
    elif got != seqs:
        raise SystemExit(f"{c}: covers {len(got)} sequences, first cell had {len(seqs)}")
json.dump(merged, open(out, 'w'))
print(f"merged {len(cells)} cells x {len(seqs)} sequences -> {out}")
