# `nature`: the tracker fails without the gate ever firing

> **CORRECTION, 2026-08-25 (same day). THE MECHANISM BELOW IS REFUTED. The
> OBSERVATIONS are all correct; the explanation built on them is not, and the
> fix it proposed would not have fixed this sequence.** Two independent benches,
> neither needing the board:
>
> **1. The closed-loop sim says quantisation cannot compound.**
> `python3 scripts/mosse_loop_sim.py --subbin` runs 9 configurations (resample
> ratio 1-3 x speed 0.1-1.5 bins/frame, 200 frames). The detector reports (0,0)
> on up to **86%** of frames, exactly as predicted — and the tracking error does
> not grow at all: worst late-half / early-half error ratio **1.00**, error
> bounded at ~half a bin throughout. The reason is structural and was missed
> here: **the detector measures the offset that exists NOW, not the increment.**
> Lag accumulates only until it crosses half a bin, at which point the next
> measurement is a whole bin and takes it back. The loop is self-correcting.
> The bench is known to be capable of showing compounding drift — its `centred`
> arm is exactly that and ends tens of pixels off.
>
> **2. The real trajectory says the frozen frames are not sub-bin frames.**
> `scripts/vot_traj_anatomy.py` compares the board's own `nature` trajectory
> against groundtruth in units of the tracker's bin:
>
> ```
>                       no motion reported     no motion reported     motion
>                       & truth >= 1 bin       & truth < 1 bin        reported
>   nature   (fails)          44.3%                 41.9%              13.8%
>   tiger    (fails)          62.4%                  1.4%              36.3%
>   car1     (best)           13.5%                  0.4%              86.1%
> ```
>
> **44% of `nature`'s frames report zero motion while the target moved MORE than
> a bin** — including frame 2 (+6.5 px = 4 row bins), frame 19 (−8.0 = 5 bins)
> and frame 23 (−10.0 = 6 bins). Sub-bin refinement cannot touch those: it
> adjusts the argmax by at most half a bin, and the argmax is wrong by six. On
> `tiger` the sub-bin case is 1.4% of frames and the failure is the same shape.
>
> **The arithmetic that convinced me was a conflation.** "The tracker captures
> ~83% of the motion" came from mean SPEED (|d| averaged), where the tracker's
> smooth path scores lower than a jittering groundtruth by construction. Mean
> DISPLACEMENT — the statistic that would show a systematic lag — matches truth
> or exceeds it: row truth +0.002 vs track +0.024 px/frame, col +0.022 vs
> −0.075. There is no systematic under-measurement to compound.
>
> **What survives:** every observation in the original note (86.1% of frames
> report (0,0); PSR rises 20 → 111 as IoU falls; the gate never fires; the aspect
> problem in the second section). What replaces the explanation: the detector is
> **pinned at the origin**, which is this project's already-documented background
> / zero-shift lock — "a localised blob at the origin, PSR 24-35 throughout, each
> win costing a permanent offset" (CLAUDE.md). `resp00_over_peak` is the
> instrument for it, it is already in `track.csv`, and the evidence run's CSVs
> were never collected off the board. `scripts/vot_sweep.sh` now collects them.
>
> **The general lesson is the one this project keeps paying for: a mechanism
> inferred from a correlation between two console statistics is a hypothesis.**
> This one was written as a finding, promoted into CLAUDE.md as a validated fact,
> and would have spent a hardware A/B on a fix for a mechanism that is not there.
> It cost one afternoon offline to refute and would have cost a board day to
> discover.

**2026-08-25, from the first arm of the multi-anchor evidence run
(`runs/run_0825_1546.log`, 14 runs, 10,604 frames, `HOLD_COAST=0`).**

`nature` was picked as the EASY end of the hold-budget spread — median budget 60
frames, 0.1% one-frame escapes, 93% of frames with budget >= 20. It is the worst
result on record.

| | `nature` (14 runs) | `car1` job 0 |
|---|---|---|
| mean of per-run mean IoU | **0.1535** | 0.5005 |
| gated frames | **18 of 10,604 (0.17%)** | 164 of 742 (22%) |
| runs with ZERO gated frames | **9 of 14** | — |
| PSR mean, per run | 34 - 100 | 25.7 |

**The gate is not involved.** Nine of fourteen runs never hold a single frame,
and the tracker is still 350-500 px off. Whatever is wrong here, the hold policy
— and therefore `HOLD_COAST`, and the whole hold-budget analysis in
[hold_policy.md](hold_policy.md) — has nothing to do with it. That analysis is
not wrong; it is answering a question this sequence does not ask.

## The mechanism: motion below one bin, reported as zero

Run 1 (anchor 0) decays monotonically from the first frame, with **PSR rising as
IoU falls**:

```
f1   IoU 0.85  PSR  32   d=(0,0)      f80   IoU 0.29  PSR  54   d=(-1,0)
f5   IoU 0.68  PSR  20   d=(0,0)      f200  IoU 0.30  PSR  37   d=(0,-1)
f30  IoU 0.48  PSR  24   d=(+1,0)     f400  IoU 0.00  PSR  97   d=(0,0)
f50  IoU 0.45  PSR  33   d=(0,+2)     f700  IoU 0.00  PSR 111   d=(0,0)
```

There is no loss EVENT. There is a drift, and the tracker gets more confident as
it goes — it is perfectly locked onto whatever it has drifted onto.

**86.1% of run 1's frames report a displacement of exactly (0,0)**, while the
target's true median motion is **2.06 px/frame**. One patch bin at this ROI is
**1.61 px (rows) x 2.78 px (cols)**, because the box is 103x178, the ROI is
206x356, and `roi_crop` resamples that to a fixed 128x128. So the true motion is
BELOW one bin on most frames.

`PsrResult::dr/dc` are `int`, `patch_dr_to_frame()` takes an `int`, and there is
no parabolic or sub-bin refinement anywhere in the host. The peak detector is a
pure integer argmax, so sub-bin motion is reported as no motion at all.

**And then the filter is trained on that answer.** `filter_update()` is fed a G
centred at the MEASURED displacement — which is correct, and is the 2026-08-20
fix — but "correct" there means "correct given the measurement". A quantised
measurement of (0,0) teaches the filter that an appearance which has actually
moved 2 px is centred. The next frame starts 2 px further behind, the lag
compounds, and nothing in the pipeline can see it because the filter matches what
it is looking at better and better. It is the training-target trap arriving
through a different door: not a sign error, a resolution error.

The arithmetic is consistent: mean centre error reaches 348 px over 999 frames =
**0.35 px/frame** of accumulated lag against 2.06 px/frame of true motion, i.e.
the tracker captures ~83% of the motion and loses the rest, every frame, forever.

## A second, independent problem on the same sequence

`nature`'s groundtruth swings ASPECT violently — 103x178 (aspect 1.73) at frame
0, 99x63 (0.64) at frame 100, 108x230 (2.13) at frame 500. The scale filter is
isotropic by construction (DSST's 1-D filter scales h and w by one factor), so it
cannot represent this at all, and the ROI it drives is resampled into a square
patch. This is the penalty the plan already accepted for axis-aligned boxes
against rotating polygons — but `nature` shows it is not a small one, and it is
independent of the sub-bin lag above.

## What this changes

1. **The dominant failure mode is not the one we have been working on.** `car1`
   loses to fast motion overrunning the search window; `nature` loses to slow
   motion falling below the bin. Those are opposite problems and the second one
   is invisible to every instrument that was watching the first — PSR RISES
   through it, and the gate never fires.
2. **The coast A/B will show nothing on `nature`.** No holds means the coast
   cannot fire; predict IDENTICAL digests between the two arms on this sequence.
   That is also the plumbing check: if `nature`'s digests differ between arms,
   something is firing that should not be.
3. **The candidate fix is sub-bin peak interpolation** — a 3-point parabola
   around the argmax in each axis, which is standard in MOSSE/DSST
   implementations and absent here. It is cheap, it is host-side, and it is
   testable offline: `scripts/mosse_loop_sim.py` should reproduce the drift with
   a target moving below one bin, which would make the mechanism falsifiable
   before any hardware time is spent on it.

**Do not tune anything on this sequence yet.** One sequence is a hypothesis. The
rest of the sweep (`car1`, `tiger`, `book`, `crabs1`, `ball3`, `soccer2`,
`animal`) will say whether sub-bin lag is a `nature` peculiarity or the thing
that has been limiting every result so far — and it costs no extra board time,
because those runs were already planned.

## Determinism, incidentally

The in-progress repeat of this sweep (`runs/run_0825_1604.log`) reproduces the
completed one **bit-for-bit across processes**: 9 of 9 overlapping runs have
identical state digests. `car1` job 0 in the 15:46 log also digests
`f0d43c096e9c6610`, the same value as `run_0825_1523`, so the `track_<seq>.csv`
rename changed nothing in the datapath — checked rather than assumed.
