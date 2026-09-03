#!/usr/bin/env python3
"""
scripts/check_doc_links.py -- the documentation must describe THIS repository.

@thesis subsec:narzedziaBudowanie | M-06 | The lint that keeps CLAUDE.md's file map and
  every documented path pointing at files that exist, so a rename cannot leave the docs
  describing a tree the repo no longer has.

THE RULE

CLAUDE.md is read as if it were the repository. A path in it that no longer exists, or a
source file it never mentions, both fail the same way: an agent (or a reader) reasons
about a tree that is not there, and nothing complains. `check_build_table.py` does this
for build DEFAULTS; this does it for the FILE MAP.

Four classes, all fatal:

  LINK        a relative markdown link, in any doc, whose target does not exist.
  PATH        a backticked repo path (`scripts/...`, `design/...`, `docs/...`) that does
              not exist. Prose paths inside runs/ and build/ are deliberately NOT checked
              -- those are run artifacts, not tracked files.
  TARGET      a backticked `make <target>` that the Makefile does not define. Only
              backticked forms count, so English ("make it possible") is not a target.
  UNMAPPED    a source file under scripts/ or design/ that CLAUDE.md's "Directory layout"
              block never names. This is the direction that actually rots: a new script
              is written, used once, and is invisible to the next reader.

The layout block is matched on the file's STEM, so one entry can cover a pair
(`mosse_graph.h/.cpp`, `mosse_filter.{h,cpp}`) and a family entry that globs
(`hanning_<N>.h`, `aiesim_data/s*/`) covers what it globs. A DIRECTORY entry does not
cover its contents: that is the loophole a new script slips through, and closing it is
the point. The check is that a file is MENTIONED, never how.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLAUDE = ROOT / "CLAUDE.md"

# Prefixes that name TRACKED files. runs/ and build/ hold run artifacts: a doc naming a
# run directory is citing provenance, not a file the repo must contain.
TRACKED = ("design/", "scripts/", "docs/", "test-sequences/", "_ide/")
SRC_EXT = {".py", ".sh", ".cpp", ".h", ".cfg", ".aiecst"}

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
PATH = re.compile(r"`((?:%s)[\w./<>\-]+)`" % "|".join(p.rstrip("/") + "/" for p in TRACKED))
TARGET = re.compile(r"`make ([a-z_][\w\-]*)")


def docs():
    return sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))


def layout_block():
    """CLAUDE.md's Directory layout fence, as one lowercase blob."""
    text = CLAUDE.read_text()
    start = text.index("## Directory layout")
    fence = text.index("```", start)
    return text[fence : text.index("```", fence + 3)].lower()


def mapped(path, blob):
    """Is `path` mentioned in the layout block?

    Matched on the STEM, so one entry covers a pair: `mosse_graph.h/.cpp`,
    `mosse_filter.{h,cpp}` and a bare `roi_crop` under `pl_src/` all count. A directory
    entry does NOT cover its contents -- that is the loophole through which a new script
    stays invisible, and it is exactly what this check is for.
    """
    name, stem = path.name.lower(), path.stem.lower()
    if name in blob or stem in blob:
        return True
    # A family entry documents a family: `hanning_<N>.h` covers hanning_64.h. The
    # placeholder must sit inside a literal, or the entry names no family in particular --
    # `<adf.h>` is an include, not a glob, and must not match every file in the tree.
    for tok in re.findall(r"[\w./<>*\-]*[<*][\w./<>*\-]*", blob):
        literal = re.sub(r"<[^>]*>|\*", "", tok)
        if len(literal) < 4:
            continue
        pat = re.sub(r"<[^>]*>", "[^/]*", re.escape(tok).replace(r"\*", "[^/]*"))
        try:
            if re.fullmatch(pat, name):
                return True
        except re.error:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args()

    mk = (ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^([a-zA-Z_][\w\-]*)\s*:", mk, re.M))
    # Pattern rules (`print-%:`) define a family; a doc may cite any member, or the stem.
    patterns = [re.compile(re.escape(t).replace("%", ".*").replace(r"\%", ".*") + "$")
                for t in re.findall(r"^([a-zA-Z_][\w\-]*%[\w\-]*)\s*:", mk, re.M)]
    blob = layout_block()

    link, path, target = [], [], []
    for d in docs():
        rel = d.relative_to(ROOT)
        for n, line in enumerate(d.read_text().splitlines(), 1):
            for m in LINK.finditer(line):
                t = m.group(1).split("#")[0].strip()
                if not t or t.startswith(("http", "mailto:")):
                    continue
                if not (d.parent / t).exists():
                    link.append((rel, n, t))
            for m in PATH.finditer(line):
                t = m.group(1)
                # a glob or a placeholder documents a FAMILY of files, not one file
                if "<" in t or "*" in t or (ROOT / t).exists():
                    continue
                path.append((rel, n, t))
            for m in TARGET.finditer(line):
                t = m.group(1)
                if t not in targets and not any(p.match(t) for p in patterns):
                    target.append((rel, n, t))

    unmapped = []
    for base in ("scripts", "design"):
        for f in sorted((ROOT / base).rglob("*")):
            if not f.is_file() or f.suffix not in SRC_EXT or "__pycache__" in f.parts:
                continue
            if not mapped(f, blob):
                unmapped.append(f.relative_to(ROOT))

    if not args.quiet:
        print(f"{len(docs())} documents, {len(targets)} make targets, layout block "
              f"{len(blob.splitlines())} lines")
    print(f"LINK     (broken relative markdown link):   {len(link)}")
    print(f"PATH     (backticked repo path not found):  {len(path)}")
    print(f"TARGET   (`make X` with no such target):    {len(target)}")
    print(f"UNMAPPED (source file CLAUDE.md never names): {len(unmapped)}")

    for name, rows in (("LINK", link), ("PATH", path), ("TARGET", target)):
        if rows:
            print(f"\n--- {name} ---")
            for rel, n, t in rows:
                print(f"  {rel}:{n}  {t}")
    if unmapped:
        print("\n--- UNMAPPED ---")
        for f in unmapped:
            print(f"  {f}")

    return 1 if (link or path or target or unmapped) else 0


if __name__ == "__main__":
    sys.exit(main())
