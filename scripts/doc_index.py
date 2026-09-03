#!/usr/bin/env python3
"""
scripts/doc_index.py -- generate docs/thesis/evidence/README.md, and check the headers.

@thesis subsec:narzedziaBudowanie | M-06 | Generates the evidence index from each note's own
  status header and from claims.md, so the index cannot disagree with the notes it lists.

WHY GENERATED

`claims.md` maps a QUESTION to its note. Nothing mapped the other way: given a note, is it
current, was it superseded, and which claims rest on it? That answer was only obtainable by
opening 29 files, and an index maintained by hand would be one more thing to forget after a
sweep. So every doc carries one header line

    **Status:** <state> · **Updated:** <YYYY-MM-DD> · **Scope:** <one line>

and this script reads them. The states are fixed, exactly as `claims.md` fixes its verdicts:

    current      the live home for a topic -- if it disagrees with a note, this wins
    closed       a question that was answered; the finding stands, no work is owed
    superseded   another document carries this now, named in the note's own text
    generated    written by a script; do not edit by hand

CHECKS (fatal, so `make check-docs` fails on them)

  HEADER      a doc under docs/ with no header line, or an unknown state.
  DANGLING    claims.md citing `evidence/<name>.md` that does not exist. `check_doc_links.py`
              cannot see these -- they are written relative to docs/thesis/, not as repo paths.
  DUPLICATE   two claim rows sharing an id (claims.md rule 7). Two rows were both `N-17` until
              2026-09-03, and the `@thesis` tags pointed at one of them -- so the ledger and the
              code disagreed about what N-17 meant, silently.
  UNSORTED    a claims table not in id order. The id is the index; an out-of-order row is how a
              duplicate hides.

REPORTED, NOT FATAL

  UNCITED     an evidence note no claim in claims.md points at. Usually a bring-up record
              rather than a finding, which is legitimate -- but it is also what a note whose
              claim row was never written looks like, and those are worth seeing.
  LONG        a claim cell over 450 characters (claims.md rule 8). A cell that long has stopped
              being an index entry and started restating its evidence note.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EVIDENCE = DOCS / "thesis" / "evidence"
CLAIMS = DOCS / "thesis" / "claims.md"
OUT = EVIDENCE / "README.md"

STATES = ("current", "closed", "superseded", "generated")
HEADER = re.compile(
    r"^\*\*Status:\*\* (?P<status>[\w ]+?) · \*\*Updated:\*\* (?P<updated>[^·]+?) · "
    r"\*\*Scope:\*\* (?P<scope>.+)$", re.M)


def header(path):
    m = HEADER.search(path.read_text())
    return m.groupdict() if m else None


ROW = re.compile(r"^\| (?P<id>[ABPRNMO]-\d+[a-z]?) \|(?P<claim>[^|]*)\|")


def claim_rows():
    """(id, table letter, claim cell, line number) for every row of every claims table."""
    out, table = [], None
    for n, line in enumerate(CLAIMS.read_text().splitlines(), 1):
        m = re.match(r"^## ([ABPRNMO]) — ", line)
        if m:
            table = m.group(1)
            continue
        m = ROW.match(line)
        if m:
            out.append((m.group("id"), table, m.group("claim").strip(), n))
    return out


def sort_key(cid):
    letter, num = cid.split("-")
    return letter, int(re.match(r"\d+", num).group()), num[len(re.match(r"\d+", num).group()):]


def claim_map():
    """claim id -> notes it cites, and note -> claims that cite it."""
    by_note = {}
    for line in CLAIMS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cid = re.fullmatch(r"`?([ABPRNMO]-\d+[a-z]?)`?", cells[0]) if cells else None
        if not cid:
            continue
        for note in re.findall(r"evidence/([\w.\-]+\.md)", line):
            by_note.setdefault(note, []).append(cid.group(1))
    return by_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not rewrite the index")
    args = ap.parse_args()

    bad_header, dangling = [], []
    for d in sorted(DOCS.rglob("*.md")):
        if d == OUT:
            continue
        h = header(d)
        if not h or h["status"].strip() not in STATES:
            bad_header.append((d.relative_to(ROOT), (h or {}).get("status", "no header")))

    ids, duplicate, unsorted_, long_cells = {}, [], [], []
    per_table = {}
    for cid, table, claim, n in claim_rows():
        if cid in ids:
            duplicate.append((cid, ids[cid], n))
        ids[cid] = n
        per_table.setdefault(table, []).append((cid, n))
        if len(claim) > 450:
            long_cells.append((cid, len(claim)))
    for table, rows_ in per_table.items():
        got = [c for c, _ in rows_]
        if got != sorted(got, key=sort_key):
            unsorted_.append(table)

    by_note = claim_map()
    for note in sorted(by_note):
        if not (EVIDENCE / note).exists():
            dangling.append(note)

    notes = sorted(p for p in EVIDENCE.glob("*.md") if p.name not in ("README.md", "TEMPLATE.md"))
    uncited = [p.name for p in notes if p.name not in by_note]

    if not args.check and not bad_header and not dangling:
        rows = []
        for p in notes:
            h = header(p) or {"status": "?", "updated": "?", "scope": ""}
            claims = ", ".join(f"`{c}`" for c in by_note.get(p.name, [])) or "—"
            rows.append(f"| [`{p.name}`]({p.name}) | {h['status']} | {h['updated'].strip()} "
                        f"| {claims} | {h['scope']} |")
        OUT.write_text("""# Evidence notes — the index

**Status:** generated · **Updated:** by `make doc-index` · **Scope:** every evidence note, its \
state, and the claims that rest on it

**Generated by `scripts/doc_index.py`. Do not edit.** Regenerate with `make doc-index`; it reads
each note's own `**Status:**` header and the citations in [`../claims.md`](../claims.md).

[`claims.md`](../claims.md) goes from a QUESTION to its note and is the citation target for the
thesis. This table goes the other way: given a note, what state is it in and what rests on it.

**A note is append-only.** Where one carries dated sections, the LATEST wins, and the notes whose
verdict changed after they were written say so in a `WHERE THIS ENDED UP` block under the header.
New notes start from [`TEMPLATE.md`](TEMPLATE.md); name them for their TOPIC, never for their
status (`TODO_`, `proposed_`), because the status changes and the filename does not.

| note | status | updated | claims | scope |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n" + f"""
{len(notes)} notes; {len(uncited)} cited by no claim
({', '.join('`%s`' % u for u in uncited) if uncited else 'none'}) — those are bring-up records
rather than findings, and a note that should carry a claim looks exactly the same, so check.
""")
        print(f"wrote {OUT.relative_to(ROOT)}: {len(notes)} notes, {len(by_note)} cited")

    print(f"HEADER   (missing or unknown state):        {len(bad_header)}")
    print(f"DUPLICATE(two claim rows sharing an id):    {len(duplicate)}")
    print(f"UNSORTED (claims table not in id order):    {len(unsorted_)}")
    print(f"DANGLING (claims.md cites a missing note):  {len(dangling)}")
    print(f"UNCITED  (note no claim points at):         {len(uncited)}")
    print(f"LONG     (claim cell over 450 chars):       {len(long_cells)}")
    for rel, got in bad_header:
        print(f"  HEADER   {rel}  ({got})")
    for n in dangling:
        print(f"  DANGLING evidence/{n}")
    for cid, first, second in duplicate:
        print(f"  DUPLICATE claims.md:{second}  {cid} already defined at line {first}")
    for t in unsorted_:
        print(f"  UNSORTED  claims table {t}")
    if uncited:
        print("  UNCITED  " + ", ".join(uncited))
    if long_cells:
        print("  LONG     " + ", ".join(f"{c} ({n})" for c, n in long_cells))

    return 1 if (bad_header or dangling or duplicate or unsorted_) else 0


if __name__ == "__main__":
    sys.exit(main())
