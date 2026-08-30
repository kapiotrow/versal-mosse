#!/usr/bin/env python3
"""Frame-time history figure: docs/thesis/results/perf.csv -> figures/perf_history.pdf.

The pattern every figure script in this directory follows:
  * read a CSV from docs/thesis/results/ -- never a number typed into this file;
  * write one PDF into docs/thesis/figures/;
  * be runnable with no arguments.

Run:  env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/figs/fig_perf_history.py
(The Vitis environment masks the venv -- PYTHONHOME points python at Vivado's build.)
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/thesis/results/perf.csv"
OUT = ROOT / "docs/thesis/figures/perf_history.pdf"


def load(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))]
    return [r for r in rows if r["frame_ms"]]


def main():
    rows = load(SRC)
    ms = [float(r["frame_ms"]) for r in rows]
    labels = [r["change"] for r in rows]
    x = range(len(rows))
    gray = [i for i, r in enumerate(rows) if r["arm"] != "rgb-synth"]
    rgb = [i for i, r in enumerate(rows) if r["arm"] == "rgb-synth"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(x, ms, "-", color="0.6", linewidth=1, zorder=1)
    ax.scatter([x for x in gray], [ms[i] for i in gray], s=26, zorder=2, label="grayscale")
    ax.scatter([x for x in rgb], [ms[i] for i in rgb], s=26, marker="s", zorder=2, label="RGB")
    ax.set_yscale("log")
    ax.set_ylabel("frame time (ms, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
    ax.grid(axis="y", which="both", color="0.9", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i in (0, len(rows) - 2):
        ax.annotate(f"{ms[i]:.6g} ms", (i, ms[i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} points)")


if __name__ == "__main__":
    main()
