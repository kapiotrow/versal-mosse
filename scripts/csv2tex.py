#!/usr/bin/env python3
"""
scripts/csv2tex.py -- docs/thesis/results/*.csv  ->  booktabs `tabular` bodies.

@thesis sec:metodykaBadan | M-09 | Generates the thesis's result tables from the CSVs, so a
  number cannot be right in results/ and stale in the .tex. The thesis \\input's the
  generated file; the caption and the label stay hand-written.

WHAT IT EMITS, AND WHAT IT DELIBERATELY DOES NOT

Each output file is a bare `tabular` environment -- column spec, rules, rows. The
chapter owns `\\begin{table}`, the placement, the `\\caption` and the `\\label`:

    \\begin{table}[htbp]
    \\centering\\small
    \\caption{Wyniki na zbiorze VOT-STb2022...}    <- yours, Polish, argumentative
    \\label{tab:wynikiVot}                         <- yours
    \\input{tables/arms}                          <- generated
    \\end{table}

A caption in this thesis is prose, not a description, and a script has no business
generating it or owning a label the text cites.

POLISH LIVES IN THE CSV, NOT HERE

Every table has Polish row labels, and they are `*_pl` COLUMNS in the CSV. Putting
them in this file would mean a regeneration silently reverts a wording fix; putting
them in the .tex would mean a regeneration overwrites it. The CSV is the one place
that survives both. Column HEADERS are in TABLE_SPEC below -- there are a dozen of
them and they come straight from docs/thesis/glossary.md table A.

Decimal marker is `{,}`, matching the prose (`0{,}1` in subsec:vot).

Usage:
    python3 scripts/csv2tex.py                    # -> docs/thesis/tables/*.tex
    python3 scripts/csv2tex.py --overleaf         # ...and copy into the thesis repo
    python3 scripts/csv2tex.py --overleaf PATH    # ...at an explicit path
    python3 scripts/csv2tex.py --check            # exit 1 if anything is out of date

--overleaf is OFF by default and only ever WRITES FILES. It does not add, commit,
push or pull: you review the diff in the thesis repo and commit it there.
"""

import argparse
import csv
import io
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/thesis/results"
OUTDIR = ROOT / "docs/thesis/tables"
DEFAULT_OVERLEAF = pathlib.Path("/home/karolina/studia/MGR/6a7e0cfd038fe83922d37077")
OVERLEAF_SUBDIR = "tables"

# name -> (csv, [(csv column, Polish header, alignment, format)], row filter)
#   format: "text" | "num" (decimal comma) | "int" (thin-space thousands)
TABLE_SPEC = {
    "arms": ("arms.csv", [
        ("label_pl", "Wariant", "l", "text"),
        ("A", "Dokładność $A$", "r", "num"),
        ("R", "Odporność $R$", "r", "num"),
        ("EAO", "EAO", "r", "num"),
        ("frames", "Klatki", "r", "int"),
    ], lambda r: bool(r.get("A"))),

    "baselines": ("baselines.csv", [
        ("tracker", "Rozwiązanie", "l", "text"),
        ("class_pl", "Zasada działania", "l", "text"),
        ("EAO", "EAO", "r", "num"),
        ("A", "$A$", "r", "num"),
        ("R", "$R$", "r", "num"),
    ], lambda r: True),

    "perf": ("perf.csv", [
        ("change_pl", "Zmiana", "l", "text"),
        ("frame_ms", "Czas klatki [ms]", "r", "num"),
        ("fps", "FPS", "r", "num"),
    ], lambda r: bool(r.get("frame_ms"))),

    # Three arms; l1relu is the SHIPPING one and the column the thesis should quote.
    # See the CSV header: the stage lists are not the same experiment, so a row-wise
    # difference between columns is not a measurement of anything.
    "frame_budget": ("frame_budget.csv", [
        ("stage_pl", "Etap", "l", "text"),
        ("gray_ms", "Skala szarości [ms]", "r", "num"),
        ("rgb_ms", "RGB [ms]", "r", "num"),
        ("l1relu_ms", "Arm docelowy [ms]", "r", "num"),
    ], lambda r: True),

    "apu_stages": ("apu_stages.csv", [
        ("stage_pl", "Etap", "l", "text"),
        ("ms_per_frame", "Czas [ms/klatkę]", "r", "num"),
        ("status", "Stan", "l", "text"),
    ], lambda r: True),

    # Only the rails that MOVED plus VCCINT: the null rails are the point of the note,
    # not of the table, and a bound of "< 0.006 W" is not a number a booktabs column can
    # carry honestly. VCCINT is kept BECAUSE it is null -- that is the result.
    "power": ("power.csv", [
        ("arm_pl", "Szyna zasilania", "l", "text"),
        ("p_static_w", "Spoczynek [W]", "r", "num"),
        ("p_work_w", "Przyrost [W]", "r", "num"),
        ("se_work_w", "Bł. std. [W]", "r", "num"),
        ("mj_per_frame", "Energia [mJ/klatkę]", "r", "num"),
    ], lambda r: r.get("channel") in
       ("VCC_PSFP", "VCC1V1_LP4", "VCC_SOC", "VCCINT")),

    # SHIPPING ARM ONLY. resources.csv also carries the superseded roi_crop_hls_128ch1 row
    # set (an HLS estimate, not a utilisation) -- see that file's header. A table mixing the
    # two would put an estimate and a routed number in the same column.
    "resources": ("resources.csv", [
        ("resource_pl", "Zasób", "l", "text"),
        ("available", "Dostępne", "r", "int"),
        ("used", "Wykorzystane", "r", "int"),
        ("percent", "Udział [\\%]", "r", "num"),
    ], lambda r: r.get("build") == "rgb_l1relu"),

    "aie_compute": ("aie_compute.csv", [
        ("kernel", "Jądro", "l", "text"),
        ("ms_per_frame", "Czas [ms/klatkę]", "r", "num"),
    ], lambda r: True),
}

STATUS_PL = {"current": "aktualny", "superseded": "zastąpiony", "removed": "usunięty",
             "historic": "historyczny"}

SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(text):
    """Escape LaTeX specials, but leave an already-escaped `\\%` and math alone."""
    if "$" in text and text.count("$") % 2 == 0:
        return text                      # a cell that is deliberately math
    if "\\" in text:
        return text                      # already hand-escaped in the CSV
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def fmt(value, kind):
    value = (value or "").strip()
    if not value:
        return "---"
    if kind == "num":
        return value.replace(",", "{,}") if "," in value else value.replace(".", "{,}")
    if kind == "int":
        try:
            return f"{int(value):,}".replace(",", "\\,")
        except ValueError:
            return esc(value)
    return esc(STATUS_PL.get(value, value))


def read_rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))


def build(name):
    csv_name, cols, keep = TABLE_SPEC[name]
    rows = [r for r in read_rows(RESULTS / csv_name) if keep(r)]
    align = "".join(c[2] for c in cols)
    out = [
        f"% GENERATED by scripts/csv2tex.py from docs/thesis/results/{csv_name}",
        "% DO NOT EDIT -- edit the CSV and re-run `make thesis-tables`.",
        "% Polish labels live in that CSV's *_pl column, so they survive regeneration.",
        "% This file is a bare tabular: the caption, the label and the placement",
        "% belong to the chapter that \\input's it.",
        f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        "\\toprule",
        " & ".join(c[1] for c in cols) + r" \\",
        "\\midrule",
    ]
    for r in rows:
        out.append(" & ".join(fmt(r.get(c[0], ""), c[3]) for c in cols) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overleaf", nargs="?", const=str(DEFAULT_OVERLEAF), default=None,
                    metavar="PATH",
                    help="also copy into <PATH>/tables/ (writes files only; never commits)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any generated file is missing or out of date")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for name in sorted(TABLE_SPEC):
        text = build(name)
        dest = OUTDIR / f"{name}.tex"
        current = dest.read_text(encoding="utf-8") if dest.exists() else None
        if current != text:
            stale.append(dest.relative_to(ROOT).as_posix())
            if not args.check:
                dest.write_text(text, encoding="utf-8")
        rows = text.count(r"\\") - 1
        if args.check:
            state = "STALE " if current != text else "ok    "
        else:
            state = "wrote "          # --check writes nothing; do not say it did
        print(f"  {state}{dest.relative_to(ROOT).as_posix():34s} {rows} rows")

    if args.check:
        if stale:
            print(f"\n{len(stale)} table(s) out of date: " + ", ".join(stale))
            return 1
        print("\nall tables up to date")
        return 0

    if args.overleaf:
        target = pathlib.Path(args.overleaf).expanduser() / OVERLEAF_SUBDIR
        if not target.parent.is_dir():
            print(f"\nERROR: {target.parent} is not a directory", file=sys.stderr)
            return 1
        target.mkdir(exist_ok=True)
        for name in sorted(TABLE_SPEC):
            shutil.copy2(OUTDIR / f"{name}.tex", target / f"{name}.tex")
        print(f"\ncopied {len(TABLE_SPEC)} tables -> {target}")
        print("Nothing was committed. Review the diff in the thesis repo and commit there.")
        print(r"Chapters reference them as \input{tables/<name>} inside their own table float.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
