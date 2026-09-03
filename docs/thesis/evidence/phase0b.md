# VOT Phase 0b — toolkit result-format round-trip

**Status:** closed · **Updated:** 2026-08-24 · **Scope:** VOT bring-up 0b: the toolkit result-format round-trip

**2026-08-24. CLOSED.** Format written and read back with the toolkit's own
writers/readers, `vot analysis` run end to end over a fabricated workspace, and
the three fake arms score in the expected order.

Tool: `scripts/vot_roundtrip.py` — one command, self-asserting.

```
find_anchors(seqa) -> forward [0, 40] backward [25]   ok
find_anchors(seqb) -> forward [0]     backward [30]   ok

arm         accuracy  robustness  frames
perfect       0.9746      1.0000     197
jittered      0.8547      1.0000     197
lost          0.9490      0.4973      98

ok   perfect robustness == 1        ok   jittered accuracy < perfect
ok   jittered robustness == 1       ok   perfect accuracy == (197-5)/197
ok   lost robustness < 0.6
```

Three arms, not one, because a round-trip that merely *produces* numbers is not
evidence. `perfect` copies groundtruth verbatim, `jittered` adds a few px of
noise, `lost` leaves the target midway. Numbers that move the right way when the
input is deliberately degraded are the assertion; a broken format would score
all three alike.

`vot-toolkit==0.8.1` installed into `.venv`. **The install needs
`env -u PYTHONPATH -u PYTHONHOME`** — otherwise the source build picks up
Vivado's Python 3.13 and dies in `typing`. Same masking `CLAUDE.md` already
records for the offline scripts, one level down.

## Format, confirmed from the toolkit's source and from bytes on disk

| | |
|---|---|
| Stack id | **`vot2022/shorttermbox`** — "VOT-ST2022 bounding-box challenge". *Not* `vot-stb2022`. |
| Dataset | `https://data.votchallenge.net/vot2022/stb/description.json` |
| Result path | `results/<tracker>/<experiment>/<sequence>/` |
| Per-run name | `{sequence}_{anchor:08d}` — **confirmed** |
| Extension | **`.bin`, not `.txt`** — `config.results_binary` defaults True. Both are readable; the default is binary. |
| Timing sidecar | `{name}_time.value`, one float per line, from the `time` property |

**Binary layout, verified byte-exactly** (`seqa_00000000.bin`, 60 regions,
1014 B = `6 + 5 + 59*17`):

```
header    <hI    version=1, count
Special   <BI    type=0, code   (code 1 = INITIALIZATION)
Rectangle <Bffff type=1, x, y, w, h
```

## Five things that would have gone wrong silently

1. **Trajectory index 0 is `Special(INITIALIZATION)`, not the init box.** The
   anchor frame is written as a special code, never as a rectangle.

2. **Trajectories are in RUN order, not sequence order.** A backward anchor at
   `i` covers `reversed(range(0, i+1))`, so index 0 is frame `i`, index 1 is
   frame `i-1`. Writing them in sequence order would be read without complaint
   and scored as a tracker that runs backwards.

3. **Each anchor has exactly ONE direction.** `find_anchors()` reads a per-frame
   `anchor` value and splits on its SIGN — positive forward, negative backward,
   the two lists disjoint. **This corrected `vot_prepare.py`**, whose synthetic
   job list emitted both directions per interior anchor: 85,020 tracked frames
   over the local 16 became **64,409**, a 24% overstatement of the run's cost.

4. **`config.yaml` must carry `registry:`.** Omit it and the analysis dies with
   `Missing arguments: registry`, which reads like a CLI usage error and is not.

5. **The stb2022 stack defines THREE experiments** — `baseline`, `realtime`,
   `unsupervised` — and `vot analysis` runs all of them, failing on any without
   results. The board produces multistart runs only. The workspace therefore
   pins a local `stack.yaml` with just `baseline`; Phase 5 needs that file, or
   it must fabricate the other two.

## Two scoring facts that change how Phase 5's numbers read

**A perfect tracker cannot score accuracy 1.0.** `AccuracyRobustness` computes
`accuracy = sum(overlaps[0:progress]) / robustness`, and `overlaps[0]` is the
`Special` init region against a real groundtruth box — overlap 0. So a verbatim
copy of groundtruth scores exactly `(total - n_runs) / total`; here
192/197 = 0.9746192893401016, matching the toolkit's output to the last digit.
The deficit scales with anchor density, so **accuracy is not comparable across
different anchor counts.** Pinned as an assertion so a toolkit upgrade that
changes it is loud rather than silent.

**`burnin` in `multistart_average_ar` is not a burn-in.** It is passed as the
third positional argument of `calculate_overlaps(first, second, bounds, ignore)`
— i.e. `self.burnin` only toggles whether overlaps are computed *bounded to the
image*. No frames are excluded after a (re)initialisation.

That differs from `scripts/rgb_vs_gray_vot.py`, which applies a real `BURNIN=10`
exclusion. **The two accuracy numbers are therefore different statistics and
must not be quoted against each other.** Phase 2's gate compares *trajectories*,
which is unaffected — but any later temptation to compare the toolkit's A/R
against the offline harness's A/R is a mistake.

The failure rule itself matches the plan exactly: `overlap <= 0.1` against
non-empty groundtruth, grace 10, `progress = j + 1 - grace`.

## Not exercised

**EAO.** `multistart_eao_score` runs with `low=115, high=755`, a window longer
than the 60-frame fabricated sequences, so it reports 0.0 for both healthy arms
and 0.055 for the broken one. That is an artifact of the test's scale, not a
result. AR is the meaningful analysis here; EAO gets its first real exercise on
stb2022.

## Dataset format, for the Phase 1 workspace

A sequence directory needs `color/%08d.jpg`, `groundtruth.txt`, a `sequence`
properties file, and `<name>.value` files; the parent needs `list.txt`.
Value files are read with an unconditional `float(line)`, so **every frame needs
a number — a blank line raises rather than meaning "no anchor".** Write `0`.
