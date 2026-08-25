# Multi-anchor evidence run — arm A vs arm B (`HOLD_COAST` 0 vs 1)

**2026-08-25, both arms in `runs/run_0825_1604.log`** (the picocom session was
never restarted, so arm B is appended to arm A's file). 8 sequences, 54 runs,
23,297 frames per arm.

## Result

```
sequence   runs  digest differs   IoU A    IoU B     delta   acc->hold  coast lines
nature       14        4/14      0.1535   0.1518   -0.0018       5          18
car1         15       15/15      0.5105   0.5908   +0.0803      73         379
tiger         7        7/7       0.1293   0.1395   +0.0101      25         109
crabs1        4        4/4       0.0958   0.1177   +0.0219      16         100
book          4        2/4       0.0431   0.0393   -0.0038       7          33
soccer2       3        3/3       0.0107   0.0065   -0.0042      46          96
animal        2        1/2       0.0090   0.0090   +0.0000      31          87
ball3         4        0/4       0.0182   0.0182   +0.0000       0           0

frame-weighted mean IoU:  A 0.2709 -> B 0.3005   (+0.0296)
```

**`car1` job 0 stops losing.** Arm A drifts permanently from frame 461; arm B
tracks to the end — 456 frames above IoU 0.5 becomes **729 of 739**.

```
frame   461   465   470   500   550   600   700   741
arm A  0.22  0.00  0.00  0.00  0.00  0.00  0.00  0.00
arm B  0.58  0.60  0.68  0.55  0.62  0.65  0.69  0.69
```

## THE PREDICTION WAS WRONG, AND THE REASON IS THE INTERESTING PART

Written before the run (phase2/hold_policy): *"`car1` job 0 will NOT be saved.
Its fatal hold is 29 frames against a 1-frame coast budget. If mean IoU improves
materially on `car1`, the mechanism is doing something the model does not
describe."* It improved by 0.08 and the loss disappeared entirely.

**The hold-budget model is OPEN LOOP and the tracker is CLOSED LOOP.** The model
asks "does the target stay inside the window frozen at the hold onset", treating
the observed 29-frame gated run as a fixed quantity. But with coasting the window
stays near the target, so frames that would have been GATED are instead
ACCEPTED — and every accept resets the velocity and restarts the coast. The
29-frame hold never happens in arm B; it was a consequence of freezing, not an
input to the decision. A model that takes the failure it is trying to prevent as
a given cannot see the fix working.

That is the third time on this project a self-consistent offline model has been
overturned by hardware on its PREMISE rather than its arithmetic. The budget
numbers in `hold_policy.md` are still correct as a bound on a SINGLE
uninterrupted hold; they are not a prediction of tracking outcome.

## `ball3`: the coast cannot rescue a tracker that never acquires

`ball3` is 69% gated and the coast fired **zero** times; all four digests are
identical between arms. The reason is exact: **0 accept→hold transitions.** Every
hold run begins at the start of a run, before any frame has been accepted, so
`coast_observe()` has never run and `coast_step()` correctly refuses to coast on
a velocity it has never measured. `ball3`'s targets escape the window in the
first frame (budget 0, 75.7% one-frame escapes), so the tracker fails from frame
1 and there is nothing to coast from.

The prediction "the win will be on `ball3`/`animal`/`soccer2`" was wrong for the
right-sounding reason — those sequences hold the most, but holding is not the
same as *having something to coast on*. `animal` (31 transitions) and `soccer2`
(46) did coast and did not improve; `car1` (73) coasted and improved a lot.

## Two build flags were wrong in arm B, and the digest is what saves the result

**I staged an arm-B ELF built without `VERBOSITY=0 DUMP_BUFFERS=0`.** It ran at
`VERBOSITY=1 DUMP_BUFFERS=1` — the second being the one explicitly called
mandatory-off, because dump files are named by frame index and it writes
~1216 KB per frame to the FAT card. Consequences:

- **Frame times are NOT comparable between arms** (`car1`: ~36 ms/frame in arm A,
  ~114 ms in arm B). No FPS number from arm B means anything.
- ~23k frames x 1.2 MB of SD-card write traffic. **There was a power loss during
  arm B's `car1` run**, forcing a reboot; `car1` was re-run cleanly afterwards
  and the analysis above uses the re-run. Whether the card hammering contributed
  is unproven, but it is not a comfortable coincidence.

**The tracking comparison survives, and only because of the state digest.**
`ball3`'s four runs and ten of `nature`'s fourteen are **bit-identical across the
two arms** despite `VERBOSITY` and `DUMP_BUFFERS` differing — which proves those
two flags are datapath-neutral, so every IoU difference above is attributable to
`HOLD_COAST` alone. Without the digest this run would have had to be discarded
and repeated.

## An analysis error worth recording

Three intermediate numbers in this diagnosis were wrong before they were caught:
first a count of "rails" that was matching `AT_SEARCH_RAIL` (a scale veto), then
two line-range analyses that disagreed with `grep` about the same file. The cause
of the second: **picocom's log contains bare `\r`, so Python's `readlines()`
splits it into 448,562 lines where `wc -l` and `grep` see 448,195.** Line-number
ranges are therefore not portable between the tools, and the mismatch is silent —
it moves a "ball3" window into `animal`'s output and yields a plausible answer.
Attribute by CONTENT (the `RUN n/N: <seq>` banner), never by line number, or open
with `newline='\n'`.

## Where this leaves `HOLD_COAST`

The default is currently 0. The evidence now says 1: **+0.0296 frame-weighted
across 8 sequences and 54 runs**, a decisive win on the best-tracking sequence,
no sequence worse than -0.0042, and a mechanism that is understood on both the
sequences it helps and the ones it cannot. The counter-argument is that the win
is concentrated: remove `car1` and the delta is roughly zero.

Flipping the default is a one-line change and every prior result is reproducible
with `HOLD_COAST=0`.
