# `docs/thesis/` — the thesis's evidence base

The repo already contained a thesis's worth of measurement; it was spread across an 1839-line
agent-facing config file and 25 notes under `runs/vot/`. This directory is where that becomes
citeable. Scaffolded 2026-08-30.

```
claims.md            THE INDEX. One row per question answered, with a verdict, an evidence
                     note, a run directory and the thesis \label it belongs under. It also
                     carries a reverse index (section -> claims) and the list of forward
                     promises teoria.tex and przeglad.tex already make. Start here.
code_map.md          GENERATED. Thesis section -> the source that implements it, from the
                     @thesis tags in the code. `make code-map`.
results/*.csv        Every number the thesis quotes. Prose drifts; a CSV does not.
tables/              GENERATED booktabs bodies, one per results CSV. `make thesis-tables`.
evidence/*.md        The notes, moved unchanged from runs/vot/. TEMPLATE.md is the skeleton.
figures/             Generated PDFs (gitignored). Sources are scripts/figs/.
glossary.md          Code English -> thesis Polish. teoria/przeglad already FIXED the
                     terminology; projekt/ewaluacja must reuse it, not re-translate it.
methodology.md       Index of the M-* claims: the rules this project paid to learn.
reproduce.md         The appendix: exact commands, per arm.
```

## How to use it while still doing the work

- Finishing a sweep: append a row to `results/arms.csv`, add or update a row in `claims.md`,
  write the evidence note from `evidence/TEMPLATE.md`. Twenty minutes, once, versus
  reconstructing provenance in the final week.
- Writing a section of the thesis at
  `/home/karolina/studia/MGR/6a7e0cfd038fe83922d37077/`: look the `\label` up in the reverse
  index in `claims.md`, then follow each claim to its evidence note and its CSV row.
- Describing the implementation: open `code_map.md` at the section you are writing. It lists
  every tagged source site with what it contributes and which claims it carries. Its
  "sections with no tagged code" list, grouped by chapter file, is the gap report.
- A claim whose evidence column says `— GAP` has the measurement but not the write-up. Those
  are the notes to write, and several of them (`P-03`, `N-01`, `R-01`, `P-07`, `P-10`) carry
  the project's strongest material.

## Getting a table into the thesis

`make thesis-tables` writes `tables/*.tex` here. Copy the one you need into the thesis repo's
`tables/` folder (or opt in with `python3 scripts/csv2tex.py --overleaf`, which writes the
files and nothing else -- no add, no commit, no push). Then, in the chapter:

```latex
\begin{table}[htbp]
\centering\small
\caption{Wyniki na zbiorze VOT-STb2022; kazdy wiersz zmienia jeden parametr.}
\label{tab:wynikiVot}
\input{tables/arms}
\end{table}
```

The caption, the label and the placement are yours. Only the `tabular` is generated, so
re-running the script can never touch your prose. Polish row labels live in the CSV's `*_pl`
column for the same reason: in the `.tex` a regeneration would overwrite them, in the script a
wording fix would be reverted by the next edit to the data.

All seven tables were compile-tested at the thesis's own geometry (`lmargin=30mm rmargin=20mm`,
`\small`) with no overfull boxes.

## Language

**Everything in this repository is English** — identifiers, comments, `@thesis` tags, evidence
notes, CSVs, this file. The thesis is Polish. `glossary.md` is the only place the two meet, and
it runs one way: it tells you which Polish phrase the thesis has already committed to for a
given English term. It never licenses Polish in the code.

## What did not move

`CLAUDE.md` keeps the operational half — environment, build parameters, traps, commands — and
that is what it is good at. It should shrink as findings migrate into `claims.md` and
`evidence/`, but nothing depends on that happening today; every path reference has already been
updated to the new locations.
