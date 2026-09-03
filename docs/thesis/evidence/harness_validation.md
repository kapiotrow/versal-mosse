# The scoring path reproduces a published tracker: CSRDCF to within 0.008 EAO through this project's own ingest

**Status:** current · **Updated:** 2026-09-03 · **Scope:** validating the multistart scoring path against published numbers, and the offline multistart harness that made it possible

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
