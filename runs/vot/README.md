# `runs/vot/` — raw sweep output only

The **evidence notes that used to live here moved to `docs/thesis/evidence/`** (2026-08-30).
Nothing was rewritten; only the path changed. Every reference in `CLAUDE.md`, the scripts and
the host sources was updated in the same commit.

What stays here:

| kind | example | what it is |
|---|---|---|
| a sweep's output | `0827_1441-eta05/` | one arm: `track_<seq>.csv`, `<seq>.log`, and `config/` |
| the sequence list | `seqs62.txt` | the 62 stb2022 sequences, in the order the sweep runs them |

**A run directory is only citeable if it has `config/`** — `aie.flagstamp`, `app.flagstamp`,
`crop.flagstamp`, `sweep.txt`. A directory without it cannot be stamped after the fact, and its
numbers cannot go in the thesis. `0827_1441-eta05/config/` is the model; see
`docs/thesis/results/README.md`.

The naming convention is `MMDD_HHMM-<arm>` for hardware sweeps and `MMDD_offline-<arm>` for
offline bench output. It is the project's real chronology — the git history is not.
