# The frozen detector: two sequences, two completely different causes

**Status:** closed · **Updated:** 2026-08-25 · **Scope:** `nature` and `tiger` fail the same way for two completely different reasons

**2026-08-25.** `nature` and `tiger` both fail by reporting zero displacement on
most frames. They have nothing else in common, and one of them is not a tracker
defect at all.

Everything here is offline and takes seconds: `rgb_vs_gray_loop.py --sequence`
runs the real integer Stage A / int8 conv / shifted-G closed loop at **3.5 s per
100 frames**, and it reproduces the board.

| | hardware | offline model |
|---|---|---|
| `nature` frozen frames | 44.3% (truth ≥ 1 bin) | 38.4% |
| `tiger` frozen frames | 62.4% | 65.8% |

## `nature`: THE PREMISE WAS FALSE — the pixels do not move

Every previous diagnosis of `nature` assumed the target moves and the tracker
fails to follow. It does not move. Three independent measurements:

**1. The response is healthy and says "no translation".** At frame 2 — one
update after init, before anything can have compounded — the peak sits at (0,0)
with `resp00/peak = 1.0000`, PSR 33, sidelobe mean +0.0001 of peak, and a
mainlobe of 16 bins against the 13 an ideal σ=2 target would give. That is a
well-formed correlation peak, not a pedestal and not noise.

**2. Padding does not touch it.** At `padding = 1.0` the ROI *is* the box, with
no background in it whatsoever, and the detector still reports (0,0) on 98% of
frames with `resp00/peak = 0.999`. So it is not background lock either.

**3. Staying still explains the pixels BETTER than following the annotation.**
`scripts/vot_motion_check.py` takes frame f−1's box content and correlates it
against frame f at the position the annotation moved to, and at the position it
came from:

```
sequence    |gt step| px  NCC moved  NCC still  still wins  appearance f1 vs f50
nature              2.79      0.816      0.940         80%                 0.072
crabs1             17.13      0.558      0.581         56%                 0.346
book               15.44      0.361      0.217         30%                -0.045
ball3              34.93      0.529      0.236         26%                 0.376
tiger              11.83      0.798      0.510          8%                 0.382
soccer2            26.46      0.823      0.414          5%                 0.756
animal             31.66      0.518      0.062          4%                 0.116
car1               20.29      0.800      0.328          3%                 0.468
```

**On 80% of `nature`'s frames, not moving correlates better than moving.** The
target deforms in place — its box aspect swings 0.58 → 1.65 and its appearance
decorrelates to **0.072** by frame 50, the worst in the set — and the annotation
centre moves because it is a min–max reduction over a changing shape. A rigid
translation tracker reports no translation, which is the correct answer to the
image and the wrong answer to the benchmark.

**So `nature` is a task mismatch, and it is 46% of the frames in the 8-sequence
evidence set.** Any frame-weighted mean over that set is dominated by a sequence
this tracker cannot be right about. Read `docs/thesis/evidence/metric_ar_vs_iou.md`'s per-sequence
table, not the aggregate, and do not tune anything against `nature`.

**Phase correlation was the wrong instrument and nearly produced a fourth wrong
diagnosis.** It reported 0.00 px of motion on `car1` — a car crossing at 20
px/frame — because it returns the DOMINANT motion in a window and static
background fills the box. The NCC comparison above asks about the target's own
pixels at two named hypotheses and cannot be fooled that way.

## `tiger`: the motion is real and the detector still freezes

`tiger` is the opposite: still-wins 8%, mean step 11.83 px = **9.2 row bins**,
well inside the ROI and nowhere near the search limit, and the tracker reports
zero on 62-66% of frames with `resp00/peak = 0.77`. Nothing excuses that. **This
is the real defect, and it is now reproducible offline in 13 seconds.**

Padding does not fix it (65.8 / 69.8 / 69.3% frozen at 2.0 / 1.5 / 1.2), so the
next candidates are the ones that change what the filter remembers rather than
what it sees: `MOSSE_ETA` against an appearance that decorrelates to 0.38 in 50
frames, `SIGMA`, and `EPS_REL`.

## Three regimes, not one failure

| regime | sequences | evidence |
|---|---|---|
| wrong question — pixels do not move | `nature` 80%, `crabs1` 56% | still-wins |
| target outruns the search window | `ball3`, `animal`, `soccer2` (26-35 px/frame, budget 0) | `hold_policy.md` |
| **real motion, detector freezes** | **`tiger`** | still-wins 8%, 62% frozen |

## An incidental result: padding 2.0 beats 1.5 on real moving video

CLAUDE.md lists the 1.5-vs-2.0 padding question as REOPENED, because the holdout
that settled it used a static scene where background lock costs nothing. On
`car1`, 200 frames closed-loop: mean IoU **0.857 / 0.780 / 0.174** at padding
2.0 / 1.5 / 1.2. The shipping default is right, and the collapse at 1.2 is a
reminder that the ROI has to carry context as well as target.
