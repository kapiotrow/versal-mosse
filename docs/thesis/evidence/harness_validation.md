# The scoring path reproduces a published tracker: CSRDCF to within 0.008 EAO through this project's own ingest

**Status:** current · **Updated:** 2026-09-03 · **Scope:** validating the multistart scoring path against published numbers and against a known-answer geometry pair, and the offline multistart harness that made it possible

**2026-09-03.** No board time. `scripts/offline_multistart.py` runs a host-side tracker under
the BOARD's multistart protocol and writes the board's trajectory format; `scripts/vot_ingest.py`
then scores it exactly as it scores a sweep. 62 sequences, 419 anchored runs, 180,544 frames —
**the same totals the board produces**, which is itself the first agreement worth noting.
Numbers: `results/reference_trackers.csv`. Claim `R-12`.

## The prediction, written down first

Written before the runs. **CSRDCF through this path should land in the band of its published
VOT-STb2022 row** (A 0.519 / R 0.580 / EAO 0.251, `results/baselines.csv`). A *band*, not an
equality: `cv2`'s CSRT is not byte-identical to the authors' VOT submission.

**The falsifier:** if a published tracker cannot be reproduced through this path, then the path
is not measuring what the challenge measures, and **every row of `results/arms.csv` inherits the
defect**. Until today the path had only ever been checked against itself — the stack is
config-identical to the toolkit's own `vot2022/shorttermbox` baseline (verified the same day),
but the sequence conversion, the anchor set, the trajectory format and the staging had no
external reference at all.

## The result

| tracker | A | R | EAO | frames | published A / R / EAO |
|---|---|---|---|---|---|
| `oracle` — ceiling | 0.9963 | **1.0000** | 0.9994 | 180,544 | — |
| `oracle-lag1` — mutant | 0.8290 | 0.9851 | 0.8004 | 179,003 | — |
| `static` — floor | 0.3938 | 0.1996 | 0.0781 | 34,541 | — |
| **`opencv-csrt`** | **0.5134** | **0.5471** | **0.2432** | 113,147 | **0.519 / 0.580 / 0.251** |
| `opencv-kcf` | 0.5229 | 0.2797 | 0.1418 | 65,009 | 0.542 / 0.532 / 0.239 |
| `opencv-kcf-rgb` — colour mutant | 0.5315 | 0.2732 | 0.1454 | 61,195 | — |

**CSRDCF reproduces to A −0.0056, R −0.0329, EAO −0.0078.** The scoring path measures what the
challenge measures. This is the first external check the path has ever had and it passes by more
than it was designed to.

**KCF does not reproduce (R 0.280 against 0.532), and that is an IMPLEMENTATION difference, not
a harness one.** `cv2`'s `TrackerKCF` is a weak reimplementation — no scale adaptation by
default, different features — and is not the KCF submitted to VOT. The reasoning that makes this
attributable rather than a loose excuse: **CSRT and KCF traverse the identical path over the
identical 419 runs**, so a path defect cannot be selective between them. **Never quote
`opencv-kcf` as "KCF".**

## Did the mechanism hold?

- **Oracle R = 1.0000 exactly, over all 419 runs — HELD, and it is the load-bearing control.**
  It is what proves the anchor set, the forward/backward run order, the frame indexing and the
  trajectory format. Any of those wrong produces spurious failures and R < 1.
- **Oracle A = 0.9963 and EAO = 0.9994 are the DATASET'S CEILING, not a defect — checked.**
  `AccuracyRobustness` computes `accuracy += sum(overlaps[0:progress])`, which includes the init
  frame (written as the `1` special, so overlap 0 by construction) and every empty-groundtruth
  frame. **A = 1.0 is unreachable here and every board arm in `arms.csv` pays the same penalty.**
  This was chased as a suspected bug first; it is not one.
- **`oracle-lag1` FIRED as required**: a one-frame lag costs 0.199 EAO. The suite has a mutant
  that fails, so a pass means something.

## Controls

- **`static`** (0.3938 / 0.1996 / 0.0781) is the floor. On a 3-sequence subset it scored EAO
  0.1408, close to this project's early arms — **a warning about small subsets, not a result**;
  over the full 62 it drops to 0.0781.
- **`opencv-kcf-rgb` is the colour-order mutant, and it is in the ledger because the first
  reading of this run was WRONG.** The toolkit's `frame.image()` returns RGB while `cv2`'s
  trackers expect BGR, so the Color-Names features of both CSRT and KCF were scrambled. That is
  a genuine bug and it is fixed. **But it is NOT what ails KCF**: correcting it is worth
  **+0.0065 R and −0.0036 EAO — a null.** It was blamed for KCF's deficit before the control
  existed; the control refuted that within the hour.
- The oracle could never have found the colour bug — **the oracle never reads a pixel.** Only
  the comparison against a published number could, and only for a tracker whose features
  actually use colour. That is the argument for keeping a published reference in the loop
  permanently rather than treating today's check as one-off.

## 2026-09-04 — the known-answer geometry calibration, and a correction it forced

`results/geometry_calibration.csv`. Claim `R-14`.

**Why.** Screening the proposed 128x128 Layer-1 arm offline would mean screening a GEOMETRY
change on a bench that `arm_res64.md` sec.25.1 recorded **inverting on exactly that axis** — the
old single-start proxy gave +0.0263 R to the 64 map where hardware gave +0.0222 R to the 128 map.
So the bench was pointed at the pair whose answer hardware already knows, before being trusted
with the pair whose answer it does not. That is `traps.md`'s *test an analysis tool against an
OLD log first*, applied to a bench instead of a parser.

**The pair.** Matched `sigma/target` = 1/16, same 3x3 RGB bank, eta 0.05, gate 5.0: a 64x64 map
at sigma 2 against a 128x128 map at sigma 4 — hardware's `res64` and `sigma4`, run offline as
`mosse:rgb-dec2 --sigma 2` and `mosse:rgb --sigma 4`.

### The prediction, written down first

Written before the run: *if the harness reproduces hardware's sign it is licensed to screen the
128x128 L1 arm; if it also inverts, that arm goes straight to hardware with no offline opinion.*
Recorded caveat, also written first: **one pair is one point of evidence, not a general licence.**

### The result — two findings, and the second is the bigger one

| resolution term (128 map − 64 map) | pooled dR | paired mean | median | trim-5 | better/worse/tied | P(dR≤0) |
|---|---|---|---|---|---|---|
| **hardware** (`sigma4` − `res64`) | **+0.0222** | +0.0049 | 0.0000 | **−0.0040** | 26/25/11 | 0.205 |
| **offline twin** (this run) | **+0.0119** | −0.0009 | 0.0000 | **−0.0092** | 22/17/23 | 0.537 |
| old single-start proxy | −0.0263 | — | — | — | — | — |

**1. The harness is no longer inverted.** Pooled, it puts the 128 map ahead — hardware's sign,
where the old proxy pointed the other way. It **undercalls the magnitude by about half**.

**2. HARDWARE'S OWN RESOLUTION TERM IS NOT PAIRED-STABLE.** The +0.0222 is a **pooled,
frame-weighted** difference between two separately-run arms. Paired across 62 sequences it is
+0.0049 with a median of **exactly 0.0000**, 26 better / 25 worse / 11 tied, and **trim-5 goes
negative**. The harness reproduces that null faithfully. **So the harness agrees with hardware
about the verdict — "not separable from a null" — and the calibration passed in a way that also
invalidated the thing it was calibrating against.**

### The correction this forced

Until 2026-09-04, **five documents quoted +0.0222 R / +0.0082 EAO as an established effect**:
`claims.md` (`R-11`), `arm_res64.md` sec.25.1, `layer1_features.md` (twice), `feature_bank.md`,
`settled.md`, `baselines.md`, `embedded_comparison.md` and `scripts/l1_banks.py`. All now say
pooled-only. **The pooled figure is not wrong and stays in `arms.csv`** — it is the official
metric and what the challenge ranks on. What was wrong was treating it as a mechanism, when this
project's own rules (*never accept an arm on R alone*; trim before believing) apply to it like
any other.

**Consequence for the 128x128 Layer-1 arm: its geometry premise is thin.** The support reduces to
Danilowicz Table 1's +0.024 EAO (different dataset, protocol and metric, with no per-sequence data
to trim) plus a pooled-only in-house difference whose paired form is a null. **That does not
justify an AIE rebuild, re-package, re-flash and a fresh 200-frame shift-budget calibration.**
`SCALE_N=1` — one host-only flag, deciding `R-13`'s open attribution — is the better next run.

### Controls

- Both offline arms resolve through `resolve_arm`, the same dispatch the bench's own `main()`
  uses, so the twin's arms and hardware's arms cannot be two spellings of one config.
- The hardware paired figures are computed from workspace `0902_cmp`, whose pooled values
  reproduce `arms.csv` rows `rgb_res64` and `rgb_sigma4` exactly (0.3873 / 0.4095).
- `P(dR≤0)` is a 10,000-sample bootstrap of the paired mean, seeded, on both sources.

### What not to re-derive

- **Pooled and paired answer different questions and both are legitimate.** Pooled says which arm
  scored higher on the challenge's metric; paired says whether the difference survives across
  sequences. Reporting only the first is, in this project's own words, the statistic it has most
  often been misled by — and this is the case that proves it applies to HARDWARE numbers too, not
  just to the offline proxy.
- The offline twin's **run count is a bad progress indicator**: tasks are longest-first, so 54 of
  419 runs done meant 34% of frames done. Estimate from frames.
- 128-map arm cost 2335 s against the 64-map arm's 1680 s — 2.1x for 4x the map pixels, about
  what an N log N transform predicts.

## What this buys, and what it does not

- **It licenses offline screening.** `vot_ar_offline.py` is single-start, has a measured
  resolution of ~0.02 in R, and has inverted twice. Trajectories written here go through the
  same ingest and the same toolkit analysis as a board sweep. That matters for the one remaining
  robustness candidate — a training-sample memory (`R-06`) — which is host-only.
- **It gives the first same-ground comparison against a real baseline.** CSRDCF at R 0.5471
  against the shipping arm's 0.4279, on identical sequences, anchors and protocol. The published
  gap is now measured rather than quoted across papers.
- **It does NOT yet answer "is R below expectation for THIS algorithm class".** CSRDCF is a
  different algorithm (HOG/CN, spatial and channel reliability). The missing row is a FLOAT TWIN
  of this project's own tracker — same bank, same eta, same gate, no quantisation, no shift
  budget — and it is the next thing to build on this harness.
- It is not a board run and never substitutes for one. An arm that reaches `AIE_FLAGS` cannot be
  screened here at all.

## What not to re-derive

- **`pkill -f offline_multistart` kills the shell that runs it**, because that shell's own
  command line contains the pattern. It cost three restarts. Use `pkill -f '[o]ffline_multistart'`.
- **Pin the thread pools.** OpenCV's trackers are internally multithreaded, so a 30-process pool
  became ~780 threads on 32 cores: measured load average **776**, far slower than serial.
  `cv2.setNumThreads(1)` plus `OMP_NUM_THREADS=1` took KCF's full 62 sequences to **207 s**.
- **Parallelise per RUN, not per sequence.** `girl` (1500 frames x 31 anchors) and `flamingo1`
  (1377 x 28) are **36% of the whole 180,544-frame workload between them**, so a per-sequence
  pool finishes 60 sequences in three minutes and then runs two cores for forty more. Per-run
  tasks, longest-first, flatten the tail; the extra `load_sequence()` per task is a small config
  read because the images are lazy.
- **`cv2` 5.0 rejects a float bounding box** and takes an integer Rect. The rounding is the
  tracker's property, not the harness's — an opencv arm cannot reach the oracle's accuracy
  ceiling, and does not get the sub-pixel init the board gets.
- **Installing `opencv-contrib-python` needs `env -u PYTHONPATH -u PYTHONHOME`** like every other
  venv call here — the Vitis environment points `python` at Vivado's build. The contrib wheel
  replaces `opencv-python` at the same version and the toolkit keeps working; verified.
