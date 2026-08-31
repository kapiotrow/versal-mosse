#!/usr/bin/env python3
"""
scripts/check_doc_numbers.py -- keep measured numbers in comments traceable.

@thesis sec:metodykaBadan | M-09 | The lint that keeps a measured figure in a comment
  traceable to results/*.csv, so a re-swept arm cannot leave a stale number behind in
  the source.

THE RULE

A comment may state any number it likes. What it may NOT do is state a number that the
thesis also quotes, without saying where that number comes from -- because the day an
arm is re-swept, `results/*.csv` changes and the comment silently becomes a lie. Prose
in CLAUDE.md had this problem, the CSVs fixed it there, and comments are the third copy.

So this script reports two things, and only two:

  DUPLICATE   a decimal in a comment that appears VERBATIM in docs/thesis/results/*.csv,
              in a comment that cites nothing. This is the drift risk. Fix by adding a
              citation -- a claim id, a results/ path, or the run log it came from. The
              number itself usually SHOULD stay: deleting "8.71 -> 1.88 ms, 4.6x" from
              the comment that explains the hypot fix makes the code worse, not better.

  UNRECORDED  a frame-time (ms/frame, FPS) or a tracking figure (EAO, IoU, A =, R =)
              that appears in NO csv AND cites nothing. Not a drift risk -- the
              opposite. It is a measurement that exists nowhere else, and if the
              thesis wants it, it needs a CSV row. Citing the log it was read off
              also resolves it: the point is traceability, not bookkeeping.

Deliberately NOT reported: ratios and factors (2.0x per tap, a 0.70x..1.30x envelope,
a box inflating 1.42x). Those are explanations, they are local to the code, and no
thesis table quotes them. An earlier, looser version of this check flagged 272 lines,
which is a check nobody runs twice.

WHAT COUNTS AS A CITATION

Anywhere in the same comment block: a claim id (`P-01`, `N-04`, ...), a `results/` or
other `docs/thesis/` path, a `runs/...log` path, or an `@thesis` tag. An evidence note
is provenance as much as a CSV is.

Usage:
    python3 scripts/check_doc_numbers.py            # report
    python3 scripts/check_doc_numbers.py --strict   # exit 1 if any DUPLICATE is uncited
"""

import argparse
import collections
import itertools
import csv
import glob
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/thesis/results"
SEARCH = ["design", "scripts"]
EXTS = (".cpp", ".h", ".hpp", ".c", ".py", ".sh")
SKIP_DIRS = {"build", "Work", ".git", "__pycache__", ".sweep_iso", "golden"}

COMMENT = re.compile(r"^\s*(//|#|\*(?!/))")
DECIMAL = re.compile(r"\d+\.\d+")
CITATION = re.compile(r"\b[ABPRNMO]-\d\d\b|results/|docs/|[\w./-]+\.log\b|@thesis")

# UNRECORDED: a headline measurement, not a ratio.
HEADLINE = re.compile(
    r"(\d+\.\d+)\s*(?:ms/frame|ms per frame|FPS|fps)"
    r"|\b(?:EAO|IoU|mean IoU)\b[^0-9\n]{0,14}(\d\.\d{3,})"
    r"|\b(?:A|R)\s*=\s*(\d\.\d{3,})"
)

Finding = collections.namedtuple("Finding", "kind path line value text")


def csv_values():
    """Every decimal that appears in a results CSV, as the exact string written there."""
    vals = set()
    for p in sorted(RESULTS.glob("*.csv")):
        with open(p, encoding="utf-8") as fh:
            for row in csv.reader(l for l in fh if not l.lstrip().startswith("#")):
                for cell in row:
                    cell = cell.strip().lstrip("+-")
                    if (re.fullmatch(r"\d+\.\d+", cell) and len(cell) >= 4
                            and set(cell.split(".")[1]) != {"0"}):
                        # An all-zero fraction (0.000, 1.0000) is not distinctive:
                        # it collides with unrelated comments and reported two
                        # false positives the first time this ran.
                        vals.add(cell)
    return vals


def blocks(lines):
    """Yield (start, end) of each run of consecutive comment lines, 0-based inclusive."""
    i = 0
    while i < len(lines):
        if COMMENT.match(lines[i]):
            j = i
            while j + 1 < len(lines) and COMMENT.match(lines[j + 1]):
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


def sources():
    for top in SEARCH:
        for dirpath, dirnames, filenames in os.walk(ROOT / top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(EXTS):
                    yield pathlib.Path(dirpath) / fn


def scan(vals):
    found = []
    for path in sources():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "scripts/check_doc_numbers.py":
            continue
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except (UnicodeDecodeError, OSError):
            continue
        for a, b in blocks(lines):
            block = "\n".join(lines[a:b + 1])
            cited = bool(CITATION.search(block))
            for k in range(a, b + 1):
                line = lines[k]
                for m in DECIMAL.finditer(line):
                    if m.group(0) in vals and not cited:
                        found.append(Finding("DUPLICATE", rel, k + 1, m.group(0),
                                             line.strip()))
                if cited:
                    # Both classes obey ONE rule: say where the number came from.
                    # A figure sourced to the log or the schedule it was read off
                    # is traceable, whether or not a CSV happens to hold it.
                    continue
                for m in HEADLINE.finditer(line):
                    v = next(g for g in m.groups() if g)
                    if v not in vals and set(v.split(".")[1]) != {"0"}:
                        found.append(Finding("UNRECORDED", rel, k + 1, v, line.strip()))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any DUPLICATE is uncited")
    ap.add_argument("--kind", choices=["DUPLICATE", "UNRECORDED"],
                    help="report only one class")
    args = ap.parse_args()

    vals = csv_values()
    all_found = scan(vals)
    # The summary always reports BOTH totals: --kind filters what is LISTED, and a
    # header whose numbers move with a display flag is a header that lies.
    by_kind = collections.Counter(f.kind for f in all_found)
    found = [f for f in all_found if not args.kind or f.kind == args.kind]
    print(f"{len(vals)} distinct decimals in docs/thesis/results/*.csv")
    print(f"DUPLICATE (uncited, duplicates a CSV value): {by_kind['DUPLICATE']}")
    print(f"UNRECORDED (measurement no CSV holds):       {by_kind['UNRECORDED']}")

    for kind in ("DUPLICATE", "UNRECORDED"):
        rows = [f for f in found if f.kind == kind]
        if not rows:
            continue
        print(f"\n--- {kind} ---")
        for path, group in itertools.groupby(rows, key=lambda f: f.path):
            group = list(group)
            print(f"\n{path}")
            for f in group:
                print(f"  {f.line:5d}  {f.value:>8s}  {f.text[:96]}")

    if args.strict and by_kind["DUPLICATE"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
