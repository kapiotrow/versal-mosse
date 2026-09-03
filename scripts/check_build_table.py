#!/usr/bin/env python3
"""
scripts/check_build_table.py -- the build-parameter tables must agree with the Makefile.

@thesis subsec:narzedziaBudowanie | M-06 | The lint that keeps the documented build defaults
  equal to the Makefile's own, so a documented arm is always an arm the repo can build.

THE RULE

Two documents carry a defaults column: `CLAUDE.md` (a digest of the knobs touched
routinely) and `docs/engineering/build_params.md` (all of them, one paragraph each).
Neither is the source of truth -- the Makefile is -- and a hand-maintained default is a
number that drifts the first time an arm ships. It already did: on 2026-09-02 eight
defaults moved, CLAUDE.md followed and build_params.md did not, so for a day the file
CLAUDE.md sends you to before moving a knob described a config the repo no longer built.

So this script reports five things, and only five:

  MISMATCH    a documented default that is not what `make print-<KNOB>` returns. This is
              the drift. Fix the DOCUMENT -- the Makefile is the authority.
  UNKNOWN     a knob documented in a table that is not a Makefile variable at all
              (renamed, removed, or a typo). `make print-` returns empty for it.
  MISSING     a knob CLAUDE.md's digest carries that build_params.md does not. The
              digest is a SUBSET of the full table by construction; a knob that is only
              in the digest has no prose home.
  UNPARSED    a defaults cell this script could not read. Reported so that a cell whose
              formatting drifted cannot silently stop being checked -- a checker that
              quietly skips a row looks exactly like a row that passes.

  COUNT       the digest's own sentence "The other N knobs are not listed here" disagreeing
              with the number of knobs the full table holds and the digest does not. It is
              the one hand-written number left in either table, so it is checked too.

Cells that document a DERIVED default (the literal `derived` anywhere in the cell)
are skipped deliberately and counted: `CROP_ROWS` follows the geometry and `ACC_BOUND`
follows `WEIGHT_BANK`, so there is no literal for either and the prose is the answer.

CELL FORMAT, which is what makes the check cheap

  | `KNOB`              | `value`            |  one knob, one value
  | `KNOB_A`/`KNOB_B`   | `v1`/`v2`          |  n knobs, n values, positionally
  | `TARGET_H`/`TARGET_W` | `64`             |  n knobs, ONE value, applied to all
  | `CONTROL_CU_RUNS`   | `0` (`8` on `hw`)  |  n knobs, the FIRST n values count

Values are compared numerically when both sides parse as a number (so `2.0` and `2.00`
agree), and as strings otherwise. Defaults are read in ONE `make print-...` invocation
at the Makefile's own defaults, which is why a conditional default documents the
unconditional value first.

Exit status is 1 on MISMATCH, UNKNOWN, MISSING or COUNT -- this check has no advisory
class, because every one of them makes a documented build wrong. UNPARSED is reported
but does not fail on its own: it means the check went blind, not that the doc is wrong.
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = [
    pathlib.Path("CLAUDE.md"),
    pathlib.Path("docs/engineering/build_params.md"),
]
DIGEST, FULL = DOCS

KNOB = re.compile(r"`([A-Z][A-Z0-9_]*)`")
BACKTICKED = re.compile(r"`([^`]+)`")


class Row:
    def __init__(self, path, line, knobs, cell):
        self.path, self.line, self.knobs, self.cell = path, line, knobs, cell


def parse_tables(path):
    """Every row of every table whose first header cell is `Parameter`."""
    rows, in_table = [], False
    for n, line in enumerate(path.read_text().splitlines(), 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        if cells and cells[0].lower() in ("parameter", "knob"):
            in_table = True
            continue
        if not in_table or len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        knobs = KNOB.findall(cells[0])
        # A row whose first cell names no knob is a continuation or a note, not a knob.
        if knobs:
            rows.append(Row(path, n, knobs, cells[1]))
    return rows


def cell_values(row):
    """The documented defaults of `row`, one per knob, or None if unreadable."""
    cell = row.cell.replace("**", "").strip()
    if "derived" in cell.lower():
        return None  # deliberate: counted as DERIVED by the caller
    vals = BACKTICKED.findall(cell)
    if not vals:
        bare = cell.strip()
        vals = [bare] if bare and " " not in bare else []
    if len(vals) == 1 and len(row.knobs) > 1:
        return vals * len(row.knobs)          # one value shared by every knob
    if len(vals) >= len(row.knobs):
        return vals[: len(row.knobs)]         # positional; trailing prose ignored
    return []


def make_defaults(knobs):
    """`make print-K` for every knob, in one invocation. Empty means: not a variable."""
    out = subprocess.run(
        ["make"] + [f"print-{k}" for k in sorted(knobs)],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    got = {}
    for line in out.splitlines():
        if " = " in line or line.rstrip().endswith(" ="):
            name, _, val = line.partition("=")
            name = name.strip()
            if name in knobs:
                got[name] = val.strip()
    return got


def equal(doc, mk):
    try:
        return float(doc) == float(mk)
    except ValueError:
        return doc == mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args()

    tables = {p: parse_tables(ROOT / p) for p in DOCS}
    knobs = {k for rows in tables.values() for r in rows for k in r.knobs}
    mk = make_defaults(knobs)

    mismatch, unknown, unparsed, derived, checked = [], [], [], 0, 0
    for path, rows in tables.items():
        for r in rows:
            vals = cell_values(r)
            if vals is None:
                derived += 1
                continue
            if not vals:
                unparsed.append((path, r.line, ",".join(r.knobs), r.cell))
                continue
            for knob, doc in zip(r.knobs, vals):
                if not mk.get(knob):
                    unknown.append((path, r.line, knob, doc))
                    continue
                checked += 1
                if not equal(doc, mk[knob]):
                    mismatch.append((path, r.line, knob, doc, mk[knob]))

    digest_knobs = {k for r in tables[DIGEST] for k in r.knobs}
    full_knobs = {k for r in tables[FULL] for k in r.knobs}
    missing = sorted(digest_knobs - full_knobs)

    # The digest says how many knobs it leaves to the full table. That sentence is the
    # only hand-written count left in either document, so it is checked like a default.
    stated = re.search(r"The other (\d+) knobs are not listed here",
                       (ROOT / DIGEST).read_text())
    undocumented = len(full_knobs - digest_knobs)
    count_off = (stated and int(stated.group(1)) != undocumented)

    if not args.quiet:
        print(f"{len(knobs)} knobs documented, {checked} defaults checked against the Makefile")
        print(f"  {len(digest_knobs)} in {DIGEST} (digest), {len(full_knobs)} in {FULL} (full)")
        print(f"  {derived} rows documented as derived (skipped deliberately)")
    print(f"MISMATCH (doc disagrees with the Makefile): {len(mismatch)}")
    print(f"UNKNOWN  (not a Makefile variable):         {len(unknown)}")
    print(f"MISSING  (in the digest, not in the full table): {len(missing)}")
    print(f"UNPARSED (defaults cell unreadable):        {len(unparsed)}")
    if count_off:
        print(f"\nCOUNT: {DIGEST} says it omits {stated.group(1)} knobs; "
              f"{FULL} holds {undocumented} the digest does not")

    if mismatch:
        print("\n--- MISMATCH ---")
        for path, line, knob, doc, real in mismatch:
            print(f"  {path}:{line}  {knob:<24s} doc {doc!r}  make {real!r}")
    if unknown:
        print("\n--- UNKNOWN ---")
        for path, line, knob, doc in unknown:
            print(f"  {path}:{line}  {knob:<24s} documented as {doc!r}")
    if missing:
        print("\n--- MISSING from " + str(FULL) + " ---")
        for knob in missing:
            print(f"  {knob}")
    if unparsed:
        print("\n--- UNPARSED ---")
        for path, line, knobs, cell in unparsed:
            print(f"  {path}:{line}  {knobs:<24s} cell {cell!r}")

    return 1 if (mismatch or unknown or missing or count_off) else 0


if __name__ == "__main__":
    sys.exit(main())
