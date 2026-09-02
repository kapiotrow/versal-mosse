# Engineering notes

The operational detail that used to live in `CLAUDE.md`. CLAUDE.md keeps the
one-line version of everything here and links out; nothing was deleted in the move
(2026-08-31), so if a rule reads as an unexplained assertion there, its derivation,
measurement and cost are in one of these files.

| file | what it holds |
|---|---|
| [`build_params.md`](build_params.md) | the unabridged build-parameter table, one paragraph per knob |
| [`shift_budget.md`](shift_budget.md) | the shift budget (3-3-3 at 64x64 SHIPPING, 4-4-4 at 128x128) and `H_SHIFT`: how each was closed, the four rules they cost, the real-video arm |
| [`scale_filter.md`](scale_filter.md) | the DSST scale filter. **It is FROZEN on ~90% of real frames** and that governs 71% of the EAO window |
| [`performance.md`](performance.md) | resource use, per-frame AIE compute, the optimisation history, the frame breakdown, hw_emu wall times, the abandoned parallel-for |
| [`roadmap.md`](roadmap.md) | what to try next for robustness and for frame time, ranked, with the evidence behind each rank |
| [`scale_filter.md`](scale_filter.md) | the DSST scale filter: root cause, `SCALE_STEP=1.04`, where it stops |
| [`measurement.md`](measurement.md) | the measurement rules that were paid for in board time |
| [`traps.md`](traps.md) | build hygiene, metrics that cannot fail a broken tracker, correctness / AIE / infrastructure traps |
| [`settled.md`](settled.md) | validated facts and closed questions — do not reopen |
| [`baselines.md`](baselines.md) | where this tracker sits against the published VOT-STb2022 table, and the attributed loss mechanism |
| [`rgb.md`](rgb.md) | the `CONV_IN_CH=3` datapath, its testing, and what it costs |

Thesis-facing material stays in [`../thesis/`](../thesis/): claims index, evidence
notes, generated tables, glossary.
