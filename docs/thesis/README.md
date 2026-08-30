# `docs/thesis/` — the thesis's evidence base

The repo already contained a thesis's worth of measurement; it was spread across an 1839-line
agent-facing config file and 25 notes under `runs/vot/`. This directory is where that becomes
citeable. Scaffolded 2026-08-30.

```
claims.md            THE INDEX. One row per question answered, with a verdict, an evidence
                     note, a run directory and the thesis \label it belongs under. It also
                     carries a reverse index (section -> claims) and the list of forward
                     promises teoria.tex and przeglad.tex already make. Start here.
results/*.csv        Every number the thesis quotes. Prose drifts; a CSV does not.
evidence/*.md        The notes, moved unchanged from runs/vot/. TEMPLATE.md is the skeleton.
figures/             Generated PDFs (gitignored). Sources are scripts/figs/.
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
- A claim whose evidence column says `— GAP` has the measurement but not the write-up. Those
  are the notes to write, and several of them (`P-03`, `N-01`, `R-01`, `P-07`, `P-10`) carry
  the project's strongest material.

## What did not move

`CLAUDE.md` keeps the operational half — environment, build parameters, traps, commands — and
that is what it is good at. It should shrink as findings migrate into `claims.md` and
`evidence/`, but nothing depends on that happening today; every path reference has already been
updated to the new locations.
