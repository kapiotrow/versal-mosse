#!/usr/bin/env python3
"""
scripts/thesis_index.py -- build docs/thesis/code_map.md from the @thesis tags
in the source tree.

WHAT A TAG IS

    // @thesis <label> | <claim>[,<claim>...] | <one line on what this site contributes>
    //   <continuation lines, indented, same comment prefix>

`<label>` is a LaTeX \\label from the thesis itself, so the map is greppable from
either repository. `<claim>` ids come from docs/thesis/claims.md. Both are
checked, and this script is the only thing that checks them: a tag naming a label
the thesis does not define, or a claim the ledger does not list, is reported
rather than silently written into the map.

WHY THIS EXISTS

The code was already 27-68% comments before a single tag was placed, and the
comments were good. What was missing was an ADDRESS: no way to ask "which code
does subsec:arytmetyka describe?", and no way to learn, after editing a kernel,
which thesis sections now describe something that changed. That is the whole job
of this file.

A tag naming a label that does not exist yet is NOT an error -- it is the map
telling you the thesis still needs that section. Those are listed separately.

Usage:
    python3 scripts/thesis_index.py                  # write docs/thesis/code_map.md
    python3 scripts/thesis_index.py --check          # exit 1 on a bad claim id
    python3 scripts/thesis_index.py --thesis <dir>   # override the thesis checkout
"""

import argparse
import collections
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_THESIS = pathlib.Path("/home/karolina/studia/MGR/6a7e0cfd038fe83922d37077")
OUT = ROOT / "docs/thesis/code_map.md"
CLAIMS = ROOT / "docs/thesis/claims.md"

SEARCH = ["design", "scripts"]
EXTS = {".cpp", ".h", ".hpp", ".c", ".py", ".sh", ".cfg", ".aiecst"}
SKIP_DIRS = {"build", "Work", ".git", "__pycache__", ".sweep_iso", "golden"}

TAG = re.compile(r"@thesis\s+([A-Za-z]+:[A-Za-z0-9]+)\s*\|\s*([^|]+?)\s*\|\s*(.*)$")
# Continuation of a tag's text. In C-like and shell files the comment prefix is
# still there; inside a Python module docstring there is none, so a bare indented
# line continues the tag -- but ONLY in .py, where a tag can only ever sit in a
# docstring. Allowing it everywhere would swallow the indented line of CODE that
# a tag sits above.
CONT = re.compile(r"^\s*(?://|#|\*)\s{2,}(\S.*)$")
CONT_BARE = re.compile(r"^\s{2,}(\S.*)$")

Tag = collections.namedtuple("Tag", "label claims text path line")


def strip_comment(line):
    return re.sub(r"^\s*(?://+|#+|\*|/\*)\s?", "", line).rstrip()


def scan_file(path):
    rel = path.relative_to(ROOT).as_posix()
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (UnicodeDecodeError, OSError):
        return []
    out = []
    for i, line in enumerate(lines):
        m = TAG.search(line)
        if not m:
            continue
        label, claims, text = m.group(1), m.group(2), m.group(3)
        # absorb indented continuation lines
        bare_ok = path.suffix == ".py"
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if "@thesis" in lines[j] or stripped in ('"""', "'''", ""):
                break
            c = CONT.match(lines[j]) or (CONT_BARE.match(lines[j]) if bare_ok else None)
            if not c:
                break
            text += " " + c.group(1).strip()
            j += 1
        claim_ids = [c.strip() for c in claims.split(",") if c.strip()]
        out.append(Tag(label, claim_ids, " ".join(text.split()), rel, i + 1))
    return out


def collect_tags():
    tags = []
    for top in SEARCH:
        for dirpath, dirnames, filenames in os.walk(ROOT / top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = pathlib.Path(dirpath) / fn
                if p.suffix in EXTS:
                    tags.extend(scan_file(p))
    return sorted(tags, key=lambda t: (t.label, t.path, t.line))


def thesis_labels(thesis_dir):
    """label -> [files it is defined in]. A label in two files is a real bug."""
    defs = collections.defaultdict(list)
    if not thesis_dir.is_dir():
        return defs
    for tex in sorted(thesis_dir.glob("*.tex")):
        for m in re.finditer(r"\\label\{([A-Za-z]+:[A-Za-z0-9]+)\}", tex.read_text(encoding="utf-8")):
            defs[m.group(1)].append(tex.name)
    return defs


def thesis_nesting(thesis_dir):
    r"""subsection label -> the section label it sits under.

    Needed so the gap report does not accuse a PARENT section of having no code
    when every one of its subsections is tagged. \section and \subsection are
    read in document order, and the label is the one that follows the heading.
    """
    parent = {}
    if not thesis_dir.is_dir():
        return parent
    head = re.compile(r"\\(section|subsection)\{|\\label\{([A-Za-z]+:[A-Za-z0-9]+)\}")
    for tex in sorted(thesis_dir.glob("*.tex")):
        level = None
        current_section = None
        for m in head.finditer(tex.read_text(encoding="utf-8")):
            if m.group(1):
                level = m.group(1)
            elif m.group(2) and level:
                if level == "section":
                    current_section = m.group(2)
                elif current_section:
                    parent[m.group(2)] = current_section
                level = None
    return parent


def thesis_declared_claims(thesis_dir):
    r"""label -> {claim ids the thesis's own stub declares for it}.

    The chapter stubs carry `% Roszczenia: A-05, A-06.` lines. That is the AUTHOR's
    mapping, so it outranks any judgement made while placing a tag -- and it keeps
    outranking it after the chapter is restructured, which has already happened
    twice. Ranges of the form `B-01...B-08` are expanded.
    """
    out = collections.defaultdict(set)
    if not thesis_dir.is_dir():
        return out
    head = re.compile(r"\\(?:sub)?section\{|\\label\{([A-Za-z]+:[A-Za-z0-9]+)\}"
                      r"|%\s*Roszczeni[ae]:(.*)")
    rng = re.compile(r"([ABPRNMO])-(\d\d)\s*\.\.\.\s*([ABPRNMO])-(\d\d)")
    for tex in sorted(thesis_dir.glob("*.tex")):
        current = None
        for m in head.finditer(tex.read_text(encoding="utf-8")):
            if m.group(1):
                current = m.group(1)
            elif m.group(2) is not None and current:
                text = m.group(2)
                for r in rng.finditer(text):
                    if r.group(1) == r.group(3):
                        for n in range(int(r.group(2)), int(r.group(4)) + 1):
                            out[current].add(f"{r.group(1)}-{n:02d}")
                text = rng.sub("", text)
                out[current].update(re.findall(r"\b[ABPRNMO]-\d\d\b", text))
    return out


def claim_ids():
    if not CLAIMS.is_file():
        return set()
    return set(re.findall(r"^\|\s*\**([ABPRNMO]-\d\d)\**\s*\|",
                          CLAIMS.read_text(encoding="utf-8"), re.M))


def claims_md_labels():
    r"""Every thesis label mentioned in claims.md, with its line number.

    Matched WITHOUT requiring a closing backtick, because `subsec:kwantyzacja (projekt)`
    once hid a stale section reference from a check that demanded one. A label is a
    label wherever it appears.
    """
    out = []
    if not CLAIMS.is_file():
        return out
    fenced = False
    for n, line in enumerate(CLAIMS.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced          # a fenced block is illustrative, not a
            continue                     # reference: that is where renamed and
        if fenced:                       # removed labels are recorded on purpose
            continue
        for m in re.finditer(r"\b((?:cha|sec|subsec):[A-Za-z0-9]+)", line):
            out.append((m.group(1), n, line.strip()))
    return out


def claim_summary():
    """claim id -> its one-line claim text, for the map's second column."""
    out = {}
    if not CLAIMS.is_file():
        return out
    for line in CLAIMS.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\|\s*\**([ABPRNMO]-\d\d)\**\s*\|\s*(.+?)\s*\|", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", type=pathlib.Path, default=DEFAULT_THESIS)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a tag names a claim the ledger does not define")
    args = ap.parse_args()

    tags = collect_tags()
    labels = thesis_labels(args.thesis)
    known = claim_ids()
    summaries = claim_summary()

    bad_claims = sorted({c for t in tags for c in t.claims if known and c not in known})
    missing = sorted({t.label for t in tags if labels and t.label not in labels})
    dupes = sorted(l for l, files in labels.items() if len(files) > 1)
    declared = thesis_declared_claims(args.thesis)
    tagged_claims = collections.defaultdict(set)
    for t in tags:
        tagged_claims[t.label].update(t.claims)
    undelivered = {lab: sorted(c for c in cl if c not in tagged_claims.get(lab, set()))
                   for lab, cl in declared.items()}
    undelivered = {k: v for k, v in undelivered.items() if v}
    declared_anywhere = set().union(*declared.values()) if declared else set()
    homeless = sorted({c for t in tags for c in t.claims} - declared_anywhere) if declared else []
    unexpected = {lab: sorted(c for c in cl if c not in declared[lab])
                  for lab, cl in tagged_claims.items() if lab in declared}
    unexpected = {k: v for k, v in unexpected.items() if v}

    tagged = {t.label for t in tags}
    nesting = thesis_nesting(args.thesis)
    # A section whose subsections carry the tags is COVERED, not a gap.
    covered = {parent for child, parent in nesting.items() if child in tagged}
    # claims.md's own section references must resolve too. A stale one there is the
    # same defect as a stale tag, and it is not covered by anything else.
    ledger_bad = sorted({(lab, n, txt) for lab, n, txt in claims_md_labels()
                         if labels and lab not in labels},
                        key=lambda x: x[1])

    unused = sorted(l for l in labels
                    if l.startswith(("sec:", "subsec:"))
                    and l not in tagged and l not in covered)

    by_label = collections.OrderedDict()
    for t in tags:
        by_label.setdefault(t.label, []).append(t)

    lines = []
    A = lines.append
    A("# Code map — thesis section → source")
    A("")
    # Every doc under docs/ carries this one-line header; scripts/doc_index.py checks it.
    A("**Status:** generated · **Updated:** by `make code-map` · **Scope:** thesis section -> "
      "the source that implements it, from the `@thesis` tags")
    A("")
    A("**Generated by `scripts/thesis_index.py`. Do not edit.** Regenerate with")
    A("`python3 scripts/thesis_index.py` after adding or moving a `@thesis` tag.")
    A("")
    A(f"{len(tags)} tags over {len({t.path for t in tags})} files, "
      f"{len(by_label)} thesis sections.")
    A("")
    A("A tag reads `@thesis <label> | <claims> | <what this site contributes>`. The label is the")
    A("thesis's own `\\label`, so `grep -rn 'subsec:arytmetyka'` finds both the prose and the code.")
    A("")

    if missing:
        A("## Sections the thesis does not define yet")
        A("")
        A("Tagged code points at these, and `\\label` does not exist in the thesis. Each is a")
        A("section that needs writing (or a tag that needs correcting).")
        A("")
        for l in missing:
            A(f"- `{l}` — {len(by_label[l])} code site(s)")
        A("")
    if dupes:
        A("## Duplicate labels in the thesis")
        A("")
        for l in dupes:
            A(f"- `{l}` is defined in {', '.join(labels[l])} — `\\ref` silently resolves to the last one")
        A("")
    if undelivered:
        A("## Claims the thesis declares, with no tagged code")
        A("")
        A("The chapter stub says `% Roszczenia: X`, and no `@thesis` tag at that label carries X.")
        A("Either the code site is untagged, or the claim is one the prose argues without")
        A("pointing at an implementation -- which is fine, and worth knowing which.")
        A("")
        for lab in sorted(undelivered):
            A(f"- `{lab}` — declares {', '.join('`%s`' % c for c in undelivered[lab])}")
        A("")
    if unexpected:
        A("## Tags carrying a claim the thesis does not declare there")
        A("")
        A("Advisory: a code site may legitimately contribute more than the stub lists. But a")
        A("claim appearing here and nowhere else in the thesis has no section to be reported in.")
        A("")
        for lab in sorted(unexpected):
            A(f"- `{lab}` — tags add {', '.join('`%s`' % c for c in unexpected[lab])}")
        A("")
    if homeless:
        A("## Claims with tagged code but no section to be reported in")
        A("")
        A("No `% Roszczenia:` line anywhere in the thesis names these. That is expected while a")
        A("chapter is unannotated -- `ewaluacja.tex` and `podsumowanie.tex` carry no such lines")
        A("yet, which is where most `M-*`, `N-*` and `R-*` claims belong. It becomes a real gap")
        A("once those chapters declare their claims and one of these is still missing.")
        A("")
        A(", ".join(f"`{c}`" for c in homeless))
        A("")
    if ledger_bad:
        A("## `claims.md` references a section the thesis does not define")
        A("")
        for lab, n, txt in ledger_bad:
            A(f"- `claims.md:{n}` — `{lab}`")
        A("")
    if bad_claims:
        A("## Tags naming a claim the ledger does not define")
        A("")
        for c in bad_claims:
            A(f"- `{c}`")
        A("")

    A("## The map")
    A("")
    for label, items in by_label.items():
        where = labels.get(label)
        note = f" — `{where[0]}`" if where else "  **(section not in the thesis yet)**"
        A(f"### `{label}`{note}")
        A("")
        A("| source | claims | contributes |")
        A("|---|---|---|")
        for t in items:
            cl = ", ".join(f"`{c}`" for c in t.claims)
            A(f"| `{t.path}:{t.line}` | {cl} | {t.text} |")
        A("")
        used = sorted({c for t in items for c in t.claims})
        if used and summaries:
            A("<details><summary>claims cited here</summary>")
            A("")
            for c in used:
                if c in summaries:
                    A(f"- **{c}** — {summaries[c]}")
            A("")
            A("</details>")
            A("")

    if unused:
        A("## Thesis sections with no tagged code")
        A("")
        A("Grouped by the chapter file that defines them. `teoria.tex` and `przeglad.tex` describe")
        A("other people's work and are expected here; so are the pure results sections of")
        A("`ewaluacja.tex`, which report numbers rather than describe code. A section of")
        A("`projekt.tex` in this list is a genuine gap: implemented, not addressable.")
        A("A section whose SUBSECTIONS are tagged is counted as covered and is not listed.")
        A("")
        per_file = collections.defaultdict(list)
        for l in unused:
            per_file[labels[l][0]].append(l)
        for tex in sorted(per_file):
            A(f"- **`{tex}`** — " + ", ".join(f"`{l}`" for l in sorted(per_file[tex])))
        A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(tags)} tags, {len(by_label)} sections")
    if missing:
        print(f"  {len(missing)} label(s) not in the thesis: {', '.join(missing)}")
    if dupes:
        print(f"  {len(dupes)} duplicate label(s) in the thesis: {', '.join(dupes)}")
    if undelivered:
        print(f"  {sum(len(v) for v in undelivered.values())} declared claim(s) with no tagged "
              f"code, over {len(undelivered)} section(s)")
    if unexpected:
        print(f"  {sum(len(v) for v in unexpected.values())} tagged claim(s) the thesis does not "
              f"declare there, over {len(unexpected)} section(s)")
    if homeless:
        print(f"  {len(homeless)} tagged claim(s) no section declares "
              f"(expected while ewaluacja.tex is unannotated)")
    if ledger_bad:
        print(f"  {len(ledger_bad)} stale section reference(s) in claims.md: "
              + ", ".join(f"line {n} ({lab})" for lab, n, _ in ledger_bad))
    if bad_claims:
        print(f"  {len(bad_claims)} unknown claim id(s): {', '.join(bad_claims)}")
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
