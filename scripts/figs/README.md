# `scripts/figs/` — one script per thesis figure

Rules that keep this honest:

1. **A figure script reads a CSV from `docs/thesis/results/`.** No number is typed into a plot
   script. If a figure needs a number that is not in a CSV, the number goes in the CSV first.
2. **Output goes to `docs/thesis/figures/` and is gitignored.** The script is the source; the
   PDF is a build product. `make figures` regenerates all of them.
3. **Runnable with no arguments**, so a reviewer's "add the mask arm" is a re-run, not an
   evening in a vector editor.

The Vitis environment masks the venv (`PYTHONHOME`/`PYTHONPATH` point python at Vivado's build,
which has no `_ctypes`). Run figures as:

```
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/figs/fig_perf_history.py
```

## Figures to write

| script | source | claim |
|---|---|---|
| `fig_perf_history.py` | `perf.csv` | P-01 — **written** |
| `fig_frame_budget.py` | `frame_budget.csv` | P-02 — stacked bar, gray vs RGB, with the residual shown |
| `fig_ar_scatter.py` | `baselines.csv` + `arms.csv` | R-05 — A on x, R on y, the 41 baselines as grey dots and this work labelled. The single most useful figure in the thesis: it shows the split |
| `fig_arm_ladder.py` | `arms.csv` | R-02/R-03 — R and EAO per arm, one knob per step |
| `fig_offline_vs_hw.py` | `arms_offline.csv` + `arms.csv` | M-01/N-04 — where the proxy transferred and where it did not |
