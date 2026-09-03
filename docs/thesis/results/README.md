# `results/` — the only place a thesis number may come from

**Status:** current · **Updated:** — · **Scope:** the only place a thesis number may come from, and the provenance rules

Prose drifts; a CSV does not. When an arm is re-swept you **append a row**, you do not edit five
paragraphs in three files.

| file | contents | quotable? |
|---|---|---|
| `arms.csv` | hardware VOT-STb2022 arms, `vot analysis` output | **yes** — this is the result |
| `arms_offline.csv` | `vot_ar_offline.py` single-start proxy | **no** — decides board time only |
| `baselines.csv` | the 41 published VOT2022 trackers, and this work's row | yes |
| `perf.csv` | frame-time history, one row per accepted optimisation | yes |
| `frame_budget.csv`, `frame_budget_rgb_delta.csv` | where the frame goes | yes |
| `resources.csv`, `aie_compute.csv` | device utilisation and scheduled AIE cycles | yes |
| `apu_stages.csv` | per-stage APU cost and what each measurement motivated | yes, per row |

## The `provenance` column in `arms.csv`

- `config` — the run directory carries `config/` with `aie.flagstamp`, `app.flagstamp`,
  `crop.flagstamp` and `sweep.txt`. The number is defensible.
- `VERIFY` — **not yet checked.** Either confirm `config/` is present and change this to
  `config`, or mark the row `unstamped` and do not put it in the thesis. This cannot be
  repaired after the fact; an unstamped run has to be re-swept.

Empty cells are genuinely unrecorded, not zero. Fill them from the run's `vot analysis` output
rather than guessing.

## Rules

1. `runs/.last_cfg` is **not** authoritative and is not a provenance source. The flagstamps are.
2. Offline and hardware numbers never appear in the same table. `R = 0.3435` (offline mutant)
   and `R = 0.3417` (shipping, hardware) agree to three decimals by coincidence and mean
   opposite things.
3. `apu_stages.csv` is **not one arm** — every row carries its own date and source, because
   they were measured on different builds as the frame shrank from 880 ms to 26. Summing a
   column would be the interleaved-group error of claim M-05.
4. `make check-docs` keeps the source comments honest about these values: a comment that
   repeats a number from this directory has to say so, or it goes stale the next time an arm
   is re-swept.
5. A 57-sequence subset is not quotable: the five missing sequences change the subsequence-length
   distribution, so EAO does not degrade gracefully.
