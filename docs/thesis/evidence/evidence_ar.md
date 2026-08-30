# The first AR numbers — and they REVERSE the coast verdict

**2026-08-25.** The 108 trajectories from the 8-sequence evidence sweep
(`evidence_arm0.md`, `evidence_arm_ab.md`) ingested into a toolkit workspace and
scored by `vot analysis`. No board time: this is the same data, read by the
metric the challenge actually reports.

Tool: `scripts/vot_ingest.py` — builds the workspace from one directory per arm,
verifies it against the dataset, runs the analysis, prints the table.

```
arm         accuracy  robustness      EAO   frames
coast0        0.6384      0.3087   0.2081     8938        HOLD_COAST=0
coast1        0.6161      0.2884   0.1941     8136        HOLD_COAST=1

sequence     coast0 A   coast0 R   coast1 A   coast1 R       dA       dR
animal         0.2027     0.0279     0.2027     0.0279  +0.0000  +0.0000
ball3          0.1170     0.0355     0.1170     0.0355  +0.0000  +0.0000
book           0.2645     0.1076     0.2603     0.0990  -0.0042  -0.0087
car1           0.7514     0.6898     0.7402     0.5783  -0.0112  -0.1116
crabs1         0.6456     0.1262     0.6391     0.1695  -0.0065  +0.0433
nature         0.4395     0.2431     0.4421     0.2438  +0.0026  +0.0007
soccer2        0.1659     0.0613     0.1427     0.0368  -0.0232  -0.0245
tiger          0.3674     0.1820     0.3730     0.2427  +0.0056  +0.0607
```

**`HOLD_COAST=1` won on mean IoU (+0.0296 frame-weighted) and LOSES on AR and
EAO**, and it loses hardest on `car1` — the sequence the coast was adopted for.
Both measurements are of the same 54 trajectory pairs. Neither is wrong.

## The mechanism, from `car1` anchor 741 (backward)

`vot`'s failure rule fires on **10 consecutive frames** at overlap ≤ 0.1
(grace 10, threshold 0.1) and discards everything after it. Per-run progress for
`car1`:

```
anchor    coast0        coast1
     0   462/742  ->   742/742     the coast's win: the run that never recovers, recovers
   250   212/492  ->   492/492     same
   550   551/551  ->    69/551     <- the coast's cost
   600   601/601  ->   119/601
   650   194/651  ->   169/651
   741   742/742  ->   260/742
```

Those four collapsed runs have **mean overlap 0.76-0.82 over the whole
trajectory** — they are not lost runs. Anchor 741 at the failure point:

```
 j=258  ov 0.818   the car turns
 j=259  ov 0.320   gate holds; the coast keeps the box going the OLD way
 j=260  ov 0.091   ...while the target turns back
 j=263  ov 0.000   box decaying to its geometric fixed point (COAST_DECAY=0.5)
 ...    13 consecutive frames <= 0.1
 j=273  ov 0.886   REACQUIRED, and it tracks the remaining ~470 frames
```

Arm A's freeze at the same turn drops out for **7 frames** — under the grace, so
no failure is declared and the run scores 742/742. **The coast converted a
recoverable 7-frame dropout into a 13-frame one by carrying the box away from a
target that was reversing.** The drift bound `v/(1-decay) = 2v` is a bound on
distance, not on direction: on a turn the coast is confidently wrong where the
freeze is merely stale.

Failure COUNTS barely move (48 of 54 runs vs 49). The whole AR difference is
*when* the failure lands, which is exactly what mean IoU cannot see: it averages
a 13-frame excursion against 470 good frames and calls the run excellent.

## What this means for the default

`HOLD_COAST=1` was made the default on 2026-08-25 on the IoU evidence, before any
AR number existed. The AR evidence points the other way, and AR is the metric of
record. **Neither result is a reason to run more hardware yet** — the interesting
option is neither 0 nor 1:

- The coast's benefit comes from MANY SHORT coasts (`car1` job 0: 73 accept→hold
  transitions, 379 coasted frames ≈ 5 frames each).
- Its cost comes from ONE LONG coast crossing a direction change.
- So a **cap on consecutive coasted frames** — freeze after k, with k from
  `hold_policy.md`'s per-sequence budget (median 6, `car1` 4) — plausibly keeps
  the win and deletes the loss. It is a one-constant change, and it is testable
  against these same trajectories only in ordering, not in outcome: coasting is
  closed loop, so it needs a board run to settle.

Do not quote a coast verdict without saying which metric produced it.

## Two things the ingest checks that the toolkit does not

`MultiStartExperiment.scan()` calls a run missing only when the FILE is absent. A
trajectory of the wrong LENGTH — a reversed backward run, or a truncated
`--vot-max-frames` bring-up — is read, scored and reported without complaint. So
`vot_ingest.py` re-derives every run name from the sequence's own anchor values
and checks each length against the multistart order (forward `i` ⇒ `len-i`,
backward `i` ⇒ `i+1`) before the analysis runs. All 108 passed.

It also refuses to run while trajectories sit at the export ROOT rather than in a
per-arm directory — the state arm B was left in, one board run away from being
silently overwritten.

## Caveat carried from Phase 0b

The init frame counts in the accuracy denominator with overlap 0, so even a
perfect tracker cannot score 1.0. The numbers above are directly comparable
between arms and to other VOT entries; they are not comparable to the console's
mean IoU, which excludes it.
