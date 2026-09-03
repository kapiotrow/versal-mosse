# The HOLD-on-gate policy, measured against the dataset

**Status:** closed · **Updated:** 2026-08-25 · **Scope:** the HOLD-on-gate policy measured against the dataset

**2026-08-25.** `car1` on hardware (`runs/run_0825_1314.log`) held position for
53 consecutive frames and never recovered, losing 281 of 742 frames. This is the
characterisation of *why*, and of how far it generalises.

Tool: `scripts/vot_hold_budget.py` — groundtruth only, no tracker, no board, no
simulation, seconds to run over all 62 sequences.

## The bound, and why it needs no tracker

On a gated frame the host holds position and freezes the filter (Bolme §3.5).
`roi_crop` therefore keeps reading the same window: `roi = box × TARGET_PADDING`
centred on the frozen centre. **A correlation filter can only peak on something
inside that window.** Once the target's centre leaves it, later evidence cannot
bring the tracker back, because the response is computed over pixels the target
is not in.

So define the **hold budget** at frame *i*: the largest *k* such that the target
centre at frame *i+k* is still inside the window frozen at *i*. At the shipping
`TARGET_PADDING = 2.0` the window half-extent is exactly `box/2` — **the target
may drift half its own size before a hold is unrecoverable.**

This is an upper bound that a *perfect* tracker also obeys. It says nothing about
whether the gate was right to fire; only what a hold costs once it does.

## What the dataset says

```
across 62 sequences: median hold budget 6 frames, min 0 (animal), max 83 (nature)
30 of 62 sequences have a median budget <= 4 frames
```

| median budget | sequences |
|---|---|
| **0** | `animal`, `ball2`, `ball3`, `soccer2` |
| 1-3 | `agility`, `drone1`, `frisbee`, `hand2`, `leaves`, `marathon`, `rowing`, `wheel`, `birds1`, `drone_across`, `polo`, `ants1`, `book`, `crabs1`, `hand`, `handball2`, `matrix`, `rabbit2`, `surfing` |
| 4-9 | `bag`, **`car1`**, `handball1`, `kangaroo`, `lamb`, `tiger`, `zebrafish1`, `conduction1`, `rabbit`, `motocross1`, `monkey`, `gymnastics3`, `singer3`, `tennis` |
| ≥20 | `basketball`, `wiper`, `fish2`, `gymnastics1`, `flamingo1`, `iceskater1`, `girl`, `iceskater2`, `graduate`, `singer2`, `fernando`, `nature` |

**On four sequences the median budget is ZERO**, and on `ball3` the target
escapes the frozen window in a *single* frame on **75.7%** of frames. There, a
hold of even one frame is usually already fatal — not because the tracker is bad
but because the object has physically left the search window.

**`car1`'s budget is 4 and its longest hold was 53.** The policy overspent by
13×.

## What `car1`'s own run shows — and its sample size

Of 22 hold runs in that log, only **three began while the tracker was still on
target** (IoU ≥ 0.5 on the preceding frame); the other 19 are inside the
already-lost tail, where "recovery" is not a meaningful question.

| hold run | length | IoU before | best IoU in the 10 frames after | |
|---|---|---|---|---|
| 9-9 | 1 | 0.841 | 0.921 | RECOVERED |
| 374-377 | 4 | 0.663 | 0.759 | RECOVERED |
| 461-489 | **29** | 0.740 | 0.000 | LOST |

Consistent with a budget of 4: the two runs within budget recovered, the one 7×
over it did not. **n = 3.** An ordering inferred from one failing run is a
hypothesis, not a finding — this is the project's most expensive recurring
lesson, and it applies to this table. The DATASET bound above is the strong
result; this table is the thing that needs more anchors.

## What this does and does not license

**Does:** the hold policy cannot be correct as an unconditional rule. It is
justified for occlusion, where the target is behind something and *stays roughly
put*; stb2022 contains that case and also contains `ball3`, where the assumption
is violated on three frames in four.

**Does not:** it does not say the gate should stop firing. Moving to a noise peak
is what the gate exists to prevent, and that failure mode is documented on
hardware (background lock, PSR 24-35 while 87-292 px off). The question is what
to do DURING a hold, not whether to hold.

Three candidate policies, none implemented, in increasing cost:

1. **Bound the hold.** After *k* held frames, stop holding and do something else
   — the budget says *k* is scene-dependent and small (0-4 for half the dataset),
   which makes a fixed constant hard to defend.
2. **Coast on constant velocity** instead of freezing the position. Costs
   nothing (two floats of state) and directly attacks the mechanism: the window
   follows the target instead of being abandoned by it. It converts "hold" from
   "assume stationary" to "assume last-known motion", which is a strictly weaker
   and more often true assumption. It also cannot help when the gate fired
   *because* the motion changed.
3. **Widen the search during a hold** — raise the effective padding for held
   frames only, so the window covers where the target could have gone. This is
   the honest fix and the expensive one: it moves `roi_h/roi_w` at runtime, which
   changes the resample ratio, which moves `|F|`, which reopens the shift budget.

`TARGET_PADDING` itself is the cheapest lever on the budget and it is already
flagged as REOPENED in CLAUDE.md — the padding-1.5-vs-2.0 holdout was measured on
a STATIC scene, where a hold costs nothing and this whole effect is invisible.

## The measurement that would settle it

More anchors, chosen to SPAN the budget rather than to be convenient, with the
prediction written down first. `--vot-job N` is a runtime argument, so this costs
no rebuild:

| sequence | budget | prediction |
|---|---|---|
| `nature` or `girl` | 83 / 39 | holds are nearly free; loss should be rare and recoverable |
| `car1` (comparator) | 4 | reproduces `run_0825_1314` |
| `crabs1` or `book` | 3 | loss after any hold run longer than ~3 |
| `soccer2` or `ball3` | 0 | loss on essentially the first hold, whatever its length |

The falsifier: if the budget-0 sequences track fine, or `nature` loses lock after
short holds, the window-escape mechanism is not what is driving the losses and
this bound is a coincidence that happens to fit `car1`.

**Use ONE build for all of them**, and re-run `car1` job 0 on it as the
comparator — `SCALE_MAX_STEP` landed after that log, and while it only fired
after the loss there, a run compared against a differently-built comparator is
not a comparison.


## The coast — implemented 2026-08-25, DEFAULT ON since the same day

**Superseded in part: this section argued for shipping OFF and hardware
overturned it. Read [evidence_arm_ab.md](evidence_arm_ab.md) first.** What
survives here is the decay sweep and the mechanism; what does not is the
prediction that `car1` could not be rescued.

`HOLD_COAST=1` moves the search window at the last measured velocity during a
hold, decayed by `COAST_DECAY` each successive held frame; `HOLD_COAST=0` (the
default) is today's freeze, unchanged. The logic is `coast_observe()` /
`coast_step()` / `coast_drift_bound()` in `mosse_filter.{h,cpp}` — no XRT header,
so `make test_host` covers it natively — and the tracker just calls them.

**The decay is the safety property, not a tuning knob.** Total drift over one
hold run is `|v| / (1 - decay)` = 2·|v| at 0.5, so a long hold fades back to a
freeze instead of becoming a second way to lose the target. `make test_host`
asserts the bound is respected AND approached (a coast that does nothing would
satisfy "bounded" trivially), that an accepted frame restores full velocity, that
a zero measured velocity never coasts, and that a hold before any accepted frame
coasts by nothing rather than by an undefined velocity.

**`COAST_DECAY = 0.5`, and pure constant velocity is worse.** Swept over all 62
sequences, mean over sequences:

| policy | P(survive 1 held frame) | P(survive 3) | median budget |
|---|---|---|---|
| freeze (today) | 90.3% | 69.9% | 6 |
| coast decay 0.0 | 94.8% | 74.4% | 7 |
| **coast decay 0.5** | **94.9%** | **76.2%** | **8** |
| coast decay 1.0 | 94.9% | 74.6% | 6 |

At 1.0 the "velocity" of a near-stationary target is mostly detection noise:
`nature` goes 83 → 34 frames of budget and `girl` 39 → 21. At 0.5, 40 sequences
improve, 15 are unchanged, 7 are marginally worse (worst: `soccer1` 12 → 9), and
the mean per-sequence escape rate halves, 9.7% → 5.1% (`ball2` 55.3% → 13.5%).

### Why it shipped OFF first — and why that reasoning was wrong

**The budget metric measures TIME TO FIRST ESCAPE, not time to unrecoverable.**
Those differ whenever the motion oscillates, and `car1` proves it on this
project's own hardware log: the hold at frame 374 had a freeze budget of **0**
and the tracker **recovered** — the target left the window and came back while
the window sat still. A coast would have walked away from that return.

At `car1`'s three hold onsets the model gives:

| hold began | length | freeze budget | coast 0.5 |
|---|---|---|---|
| 9 | 1 | **10** | **1** |
| 374 | 4 | 0 | 1 |
| 461 | 29 | 0 | 1 |

Worse at the first, marginally better at the other two, and **saving neither of
the losses** — the 29-frame hold needed 29 frames of budget and no policy here
supplies that. So the aggregate is real, the mechanism is worth having, and
turning it on by default would set a shipping behaviour from an offline model
with a known counterexample in the same dataset. That is the shape of mistake
this project has recorded three times.

**AND THE CAUTION WAS RIGHT WHILE THE PREDICTION WAS WRONG.** Hardware rescued
`car1` outright. The table above is an OPEN-LOOP bound: it treats the 29-frame
gated run as an input, when coasting turns would-be-gated frames into accepted
ones and every accept restarts the coast, so that run never happens. Deferring
to hardware was correct; the number the model produced was not.

### The A/B, which is one flag

```bash
make sd_card TARGET=hw FRAME_SOURCE=vot HOLD_COAST=1 DUMP_BUFFERS=0
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-job 0
```

Predictions, written down first:

- **`car1` job 0 will NOT be saved.** Its fatal hold is 29 frames against a
  1-frame coast budget. If mean IoU improves materially on `car1`, the mechanism
  is doing something the model does not describe and the model needs revisiting
  before it is trusted anywhere else.
- **The frame-9 hold may get WORSE** (freeze budget 10, coast 1) — a coast that
  walks off a target that was about to stay put. One frame either way, but it is
  the specific harm this policy can do and it is visible in `[coast]` lines.
- **The sequences to test are the budget-0 ones**, `ball3` / `soccer2` /
  `animal`, where freeze is hopeless (escape on 51-76% of frames) and the coast
  roughly halves that. That is where the aggregate says the win is; `car1` is
  where the comparator is.

`[coast]` lines print at every verbosity, like the other anomalies, so a run of
them is visible in a `VERBOSITY=0` log — a coast moves the search window on no
evidence, and a run of them is the shape of a loss in progress.
