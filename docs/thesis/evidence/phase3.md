# VOT Phase 3 — per-run reset (multi-start)

**2026-08-25. GATE MET. Determinism passes on the clean build
(`runs/run_0825_1523.log`) and FAILS on the deliberately-broken one
(`runs/run_0825_1530.log`), in the predicted manner. The trajectories land on
the PC and ingest with the toolkit's own reader.** `runs/run_0825_1431.log`, `car1 --vot-jobs 0,1,0`, three
runs and 2176 frames in one process. See "The gate run" below. Two things are
outstanding: `RESET_MUTANT=1` has not been run, so the test is not yet known to
be capable of failing; and the trajectories were not written, because
`/mnt/vot-results` did not exist on the board.

## What multi-start needs

The toolkit's `MultiStartExperiment` spawns one run per anchor, each starting
from a fresh filter. Until now the ELF ran exactly one job per process, so
"fresh" was guaranteed by process startup. Running several jobs in one process
makes every piece of state that outlives a run a correctness hazard, and the
hazard is silent: a run that inherits the previous one's filter tracks, produces
a complete trajectory, and is wrong.

`--vot-jobs all` runs every anchor of a sequence; `--vot-jobs 0,3,0` runs a
specific list **and may repeat an index**, which is the determinism test.

## The reset surface

Each entry is here because leaving it out is invisible from the console.

| state | action | why it bites |
|---|---|---|
| `g_filter` | `= FilterState{}` | run B would start from A's trained filter |
| **`filter_bo`** | zero + sync | **the DEVICE holds H too.** Frame 0 of run B would correlate against A's filter — and frame 0 is the frame B trains FROM, so the damage lands in the initialisation |
| `scale` | `scale_filter_config()` afresh | the template size is DERIVED from the init box, which changes per anchor. (It also clears `sf.st` and `initialized`, so the scale MODEL resets with it) |
| `g_target`, `sigma_r/c` | recompute from the new box | G is built from the box; a new box needs a new G |
| `mean_prev` | re-seed `bias_acc >> out_shift`, re-sync `weights_bo` | without it Stage B1 is inert on the one frame the filter trains from, and the ch16 response rails flat |
| `g_energy` | zero | `filter_quantize_q15` reads it; stale values mis-scale H |
| `g_target_shift` | zero | the exact variable the first `TAIL_PARALLEL` attempt read stale, taking mean IoU 0.9188 → 0.4794 |
| `coast` | `= CoastState{}` | **new 2026-08-25**: run B would coast on A's velocity |
| box / `box_h0,w0` / pos / truth | from the job's init box | `box_h0/w0` are the drift bounds' reference and were `const` |
| gate / track / scale counters | zero | a run that inherits them reports a plausible summary for a run that did not happen |

The re-seed is idempotent by construction: `bias_acc` and `out_shift` are never
written during a run, so recomputing the seed from the weight buffer's own
fields reproduces exactly what startup produced.

## The determinism test, and its negative control

Run job A, job B, job A in one process. **A's two trajectories must be
byte-identical** — the only cheap instrument that can see state leaking across
`run_reset()`, and it needs no groundtruth to be meaningful.

It is built into the ELF: a repeated job index is compared **in memory** against
the first run's bytes (both would write the same filename), and the second run
is deliberately NOT written, so on a failure the file on disk is the first run's
— the one the comparison reports against. The comparison uses
`Trajectory::as_text()`, the same function `write()` emits from, because a
comparison carrying its own copy of the format proves only that two copies agree.

**`RESET_MUTANT` is the negative control, as a build flag rather than a code
edit** — repeatable, recorded in the log, and impossible to leave commented out
by accident:

```
RESET_MUTANT=0  none (the shipping build)
             1  skip the mean_prev re-seed        <- the plan's mutant
             2  skip zeroing filter_bo
             3  skip clearing g_filter
             4  skip clearing the coast velocity
             5  skip the scale reconfigure
```

Any non-zero value prints a banner saying the build is deliberately broken and
that its tracking output is not a result. Confirmed at build time: the three
binaries differ, and the banner string is absent at 0 (folded away) and present
otherwise.

## Reporting changes

- **Per RUN**: the `[track]`, `[scale]` and `[gate]` summaries, and the
  trajectory write. They already reset with the counters.
- **Process-wide**: `[dma]` and `[apu]` cumulative, now divided by
  `g_frames_total` rather than by the last run's length — which at multi-start
  would be a per-frame average over the wrong denominator, i.e. a plausible
  wrong number.
- **`track.csv` is named `track_<sequence>.csv` at `FRAME_SOURCE=vot`.** One
  evidence sweep is one ELF invocation PER SEQUENCE and the file opens in `"w"`
  mode, so a fixed name would leave eight sequences with one file — and the loss
  is silent, because the survivor looks like a perfectly good CSV. The ARM is not
  in the name: that separation comes from `--vot-results`, which already has to
  differ per arm because the trajectories collide the same way. The name is
  sanitised to `[A-Za-z0-9._-]` because the sequence string arrives from a
  manifest on a mount, i.e. it is input; all 62 stb2022 names pass through
  unchanged. The file stays on the SD card and NOT on the results mount —
  `csv_row()` still `fflush()`es every row, and that on NFS would put a
  filesystem sync in the timed path (Phase 4).
- **`track.csv` gains trailing `job,anchor` columns.** One file now carries
  several runs and their rows are otherwise indistinguishable; a mean IoU over
  the whole file would silently average unrelated runs. Trailing, so every
  reader that selects by header name is unaffected.

## `FRAME_SOURCE=synth` is untouched

`n_jobs` is 1, `run_reset()` is not even compiled, and the loop body is the
frame loop as it was. The only differences on that arm are the two new CSV
columns (`job=0, anchor=-1`) and the `g_frames_total` denominator, which equals
`ITER_CNT` when there is one run.

## The gate

```bash
make sd_card TARGET=hw FRAME_SOURCE=vot DUMP_BUFFERS=0 VERBOSITY=0
# on the board:
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-jobs 0,1,0     # must PASS
```

then the negative control, which must FAIL:

```bash
make sd_card TARGET=hw FRAME_SOURCE=vot DUMP_BUFFERS=0 RESET_MUTANT=1
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-jobs 0,1,0     # must FAIL
```

**Predictions, written first.** `RESET_MUTANT=1` should fail EARLY — `mean_prev`
feeds Stage B1 on frame 0, the frame the filter trains from, so run A's second
pass should diverge within the first few frames rather than drifting apart late.
`RESET_MUTANT=4` (coast) should fail only if job 1 ends with a hold and only at
`HOLD_COAST=1`; at the shipping `HOLD_COAST=0` it is expected to PASS, and that
is not a bug in the test — it is the coast being genuinely absent from the
shipping build.

**Note `DUMP_BUFFERS=0` is not optional here**: dump files are named by frame
index, so several runs in one process would overwrite each other's dumps and the
last run would silently win.

## Not in this phase

The batch over all 62 sequences (Phase 5), and `PROGRESS_EVERY` /
`CSV_FLUSH_EVERY` (Phase 4). Neither blocks the multi-anchor evidence run that
Phase 3 exists to make possible.


## The gate run — `car1 --vot-jobs 0,1,0` (`runs/run_0825_1431.log`)

```
[vot] 3 run(s) queued: 0,1,0
[vot] staged 217.4 MB in 2.12 s = 102.5 MB/s
RUN 1/3: car1 job 0, anchor 0  forward, 742 frames
RUN 2/3: car1 job 1, anchor 50 forward, 692 frames
RUN 3/3: car1 job 0, anchor 0  forward, 742 frames
[vot] DETERMINISM: job 0 re-run, trajectory BYTE-IDENTICAL (26748 B)
[vot] 3 run(s), 2176 frames total
```

**Runs 1 and 3 agree on every reported statistic** — 741 evaluated, 455/741
above 0.5, mean IoU 0.5005, mean centre error 170.18 px, final box 66×68 at
(−210.2, 2.1), gate 640 accepted / 101 gated, longest gated run 29, PSR
−27.36/27.33/87.89, scale 640 evaluated / 518 accepted / 122 held with an
identical veto breakdown — **and the trajectory is byte-identical at 26748 B**.
`run_reset()` leaks nothing that this instrument can see.

**THE TEST IS NOT YET KNOWN TO BE ABLE TO FAIL.** `RESET_MUTANT=1` has not been
run. Until it has, "byte-identical" is a passing test on a path with no prior
coverage, which this project has repeatedly established is worth nothing on its
own. That run is the remaining half of the gate.

### The finding this run produced, which is not about Phase 3

**Two anchors, 50 frames apart, lose the target at the SAME DATASET FRAME.**

| run | anchor | permanent loss at run frame | = dataset frame |
|---|---|---|---|
| 1 | 0 | 461 | **461** |
| 2 | 50 | 411 | **461** |

Job 1 starts 50 frames later, trains a different filter on a different frame,
and tracks 411 frames of its own history — and fails at the same instant. That
rules out accumulated drift and filter degradation as the cause and points at
the scene: dataset frames 457-461 move 31.0, 49.0, 29.4, 23.8, **48.0** px
against a box of ~93 px, i.e. roughly half the box per frame, twice, inside five
frames. `car1`'s hold budget is 4 frames (`docs/thesis/evidence/hold_policy.md`); this is
the event that spends it.

**One anchor was a hypothesis; two independent anchors failing at the same frame
is a finding.** It is also the first thing multi-start has bought that a single
run could not have.

### `SCALE_MAX_STEP=2` changed nothing while the tracker was tracking

Against the previous build (`run_0825_1314`, same sequence and anchor, no rate
limit): per-frame IoU over frames 1-460 differs by at most 0.0050, which is
exactly the rounding bound of the `VERBOSITY=0` line's `%.2f`. So the two are
identical to the precision the log prints, and the rate limit never fired in a
way that moved the tracker while it was on target.

It changed the tail, as predicted. Post-loss the box now shrinks to 66×68
instead of growing to 128×132, and mean centre error reads 170.18 px against
136.72 — **both statistics over frames whose IoU is 0.000 either way**, so
neither is evidence about tracking. Note IoU cannot see this at all: two
different wrong boxes both score 0.

The veto tally moved more than expected — `MAX_STEP x75` against a prediction of
3-7. Most of that is RECLASSIFICATION, not new vetoes: `MAX_STEP` is checked
before the range and confidence tests, so frames the old build reported as
`LOW_CONF` (30) or `OUT_OF_RANGE` (29) now report as `MAX_STEP`. Net new holds
are 61 → 122 across a run that is 281 frames lost. **A veto count is not a veto
effect** when the new veto sits ahead of the old ones in the chain.

### Frame time, measured properly for the first time

**27.92 ms/frame = 35.8 FPS**, over 2176 frames at `VERBOSITY=0 DUMP_BUFFERS=0`,
against grayscale synthetic's 26.29 ms. The +1.6 ms is the VOT seam and the
bilinear resample: `scene gen` (the blob memcpy) 0.068 ms, `frame push` 0.084 ms
for 307 KB. The 91.81 ms in `run_0825_1314` was 74.9% console and should never
be quoted.

### What did not work

**No trajectories were written.** `/mnt/vot-results` does not exist on the
board:

```
[vot] TRAJECTORY NOT WRITTEN: cannot write /mnt/vot-results/car1_00000000.txt:
      No such file or directory
```

The determinism test was unaffected — it compares in memory, deliberately,
because both runs of a repeated job would write the same filename. But the run
produced no ingestible result. The results export still needs mounting
(phase0a.md's last open item, and the command is in phase2.md). **The failure was
loud and per-run, which is the intended behaviour**, and it cost nothing here
because the run's purpose was the determinism test.


## The negative control — `RESET_MUTANT=1` (`runs/run_0825_1443.log`)

**The mutant was active, the reset was deliberately broken, and the determinism
test passed anyway.**

```
[vot] *** RESET_MUTANT=1 IS ACTIVE — run_reset() is DELIBERATELY BROKEN ***
[vot] DETERMINISM: job 0 re-run, trajectory BYTE-IDENTICAL (26748 B)
```

That is the negative control doing its job: it says the instrument was not
measuring what it claimed.

### The mutant was NOT inert — the test was too coarse

The build ran at `VERBOSITY=1`, so the per-frame diagnostics are in the log, and
run 1 and run 3 (both job 0) differ from **frame 1**:

| quantity | differing frames of 741 | first difference |
|---|---|---|
| `accum` max | 114 | frame 1: 1963 vs 1961 |
| `response` max | 106 | frame 1: 1556 vs 1553 |
| B2 `max|removed|` | 111 | frame 1: 667 vs 671 |
| `F_ch` (ch0) max | 25 | frame 4: 4696 vs 4695 |
| `H(q15)` max | 0 | — |

So state genuinely leaked: run 3 inherited run 2's `mean_prev` instead of the
re-seeded value, Stage B1 subtracted the wrong DC, and the difference is visible
in the accumulator, the response and in how much B2 then had to remove. It is
~0.1-0.2%, and **a trajectory cannot see it**: the box comes from an INTEGER
peak bin, so any difference that does not move an argmax is invisible.

**This is the exact failure mode CLAUDE.md already records** — "tracking would
come back *nearly* identical, the one outcome that makes the bit-identical
criterion useless", written about the parallel-for FMA experiment. The
determinism test walked into it, and only the negative control caught that.

### The fix: a run-state digest

The determinism key is now the trajectory **and** an FNV-1a digest accumulated
every frame over:

- the **full response buffer** (`RESP_BYTES`, from the heap copy) — the
  pipeline's output, and the last thing that exists before an argmax throws its
  precision away;
- `box.row/col/h/w`, `gate.psr`, `psr.peak` in full precision — the host's own
  continuous state.

Two runs that agree on this agree bit-for-bit on the datapath, not merely on
their conclusions. The verdict now distinguishes three cases, and the middle one
is what `RESET_MUTANT=1` should print:

```
trajectory AND state digest IDENTICAL      -> no leak
SAME trajectory, DIFFERENT state digest    -> leaked without moving a peak
different trajectory                       -> leaked and moved a peak
```

The digest is on **both** arms and printed at the end of every run. That is
worth more than the determinism test alone: this project's acceptance criterion
for every optimisation has been "tracking comes back bit-identical", checked by
eye across two logs, and the digest makes it one number to compare.

Cost is measured, not assumed: it has its own `AP_DET_HASH` slot, because an
instrument added to the timed path whose cost is invisible cannot be judged.
Expect ~64 KB of FNV per frame; the next run reports it.

### Both runs must be repeated

The clean run (`run_0825_1431`) passed a test now known to be too weak, so its
PASS carries only as much information as the test had. Re-run:

```bash
make sd_card TARGET=hw FRAME_SOURCE=vot DUMP_BUFFERS=0 VERBOSITY=0
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-jobs 0,1,0     # digest must MATCH

make sd_card TARGET=hw FRAME_SOURCE=vot DUMP_BUFFERS=0 RESET_MUTANT=1
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-jobs 0,1,0     # digest must DIFFER
```

**Prediction, written first**: with the digest, `RESET_MUTANT=1` fails with SAME
trajectory / DIFFERENT digest — not a different trajectory — because the leak is
already known to be sub-argmax on this sequence. If it reports a *different
trajectory*, something other than `mean_prev` is also leaking.

### Still not written: the trajectories

```
[vot] TRAJECTORY NOT WRITTEN: cannot write /mnt/vot-results/car1_00000000.txt:
      Read-only file system
```

The mount exists now but is READ-ONLY. Either it was mounted without `rw`, or it
is the `/srv/vot/data` export (which is exported `ro`) rather than
`/srv/vot/results`. The mount command is in phase2.md; check
`mount | grep vot` on the board and confirm the server path ends in `/results`.


## Re-run with the digest — clean build (`runs/run_0825_1523.log`)

```
RUN 1/3  car1 job 0  -> state digest f0d43c096e9c6610
RUN 2/3  car1 job 1  -> state digest 17e670688a0c8934
RUN 3/3  car1 job 0  -> state digest f0d43c096e9c6610
[vot] DETERMINISM: job 0 re-run, trajectory AND state digest IDENTICAL (26748 B)
```

**Runs 1 and 3 agree on the digest as well as the trajectory**, so `run_reset()`
is clean to the precision of the full response buffer, not merely to the
precision of an argmax. Job 1's digest differs, which is the control that says
the digest is not simply constant.

**The instrument did not change the tracker.** Every tracking statistic is
identical to `run_0825_1431`, which ran the same configuration without the
digest: mean IoU 0.5005, 455/741 above 0.5, centre error 170.18/609.88 for
job 0; 0.4963 and 401/691 for job 1.

**Cost, measured in its own slot**: `determinism hash  2.0 calls/fr  0.283
ms/frame  1.0%`. Mean frame body 28.13 ms against 27.92 without it — the 0.21 ms
difference matches the slot. **Subtract ~0.28 ms before quoting an FPS figure
from a digest-enabled build**, or make it a build flag if the batch run needs
the last percent; it buys nothing on a run with no repeated jobs.

## The results export works

```
[vot] wrote /mnt/vot-results/car1_00000000.txt  (742 regions, 18.1 ms)
[vot] wrote /mnt/vot-results/car1_00000050.txt  (692 regions, 12.6 ms)
```

phase0a.md's last open item is now closed: the `rw` export is exercised, and the
files are on the PC. Verified with the toolkit's own reader:

| file | regions | job length | index 0 | first tracked frame |
|---|---|---|---|---|
| `car1_00000000.txt` | 742 | 742 | `Special(1)` | dataset 1, IoU 0.958 |
| `car1_00000050.txt` | 692 | 692 | `Special(1)` | dataset 51, IoU 0.963 |

Every region after index 0 is a `Rectangle`, the counts match the manifest's job
lengths exactly, and the first tracked box reproduces the IoU the board printed.
`car1_00000000_time.value` carries 742 lines, one per region, opening at 38.549
ms (frame 0 includes initialisation) and settling to ~25.6-25.9.

**That is the whole result path proven end to end**: manifest -> blob -> board ->
NFS -> toolkit reader, with the numbers agreeing at both ends.

## What is left

`RESET_MUTANT=1` with the digest. The prediction stands and is now the only
thing untested: it must fail with **SAME trajectory, DIFFERENT digest**. A
different trajectory would mean something beyond `mean_prev` is also leaking.


## The negative control, with the digest (`runs/run_0825_1530.log`)

**It fails, and it fails in the way that was written down before the run.**

```
[vot] *** RESET_MUTANT=1 IS ACTIVE — run_reset() is DELIBERATELY BROKEN ***
RUN 1/3  car1 job 0  -> digest f0d43c096e9c6610
RUN 2/3  car1 job 1  -> digest 830a105de69617d0
RUN 3/3  car1 job 0  -> digest 5390f48b88678fcf
[vot] DETERMINISM FAILED: job 0 re-run has the SAME trajectory but a DIFFERENT
      state digest. State leaked across run_reset() without moving a peak.
[vot] DETERMINISM: 1 failure(s) across 3 run(s)
```

The prediction was "SAME trajectory, DIFFERENT digest — not a different
trajectory, because the leak is already known to be sub-argmax on this
sequence". That is exactly the verdict printed.

### The digest values tell the whole story on their own

| run | clean build | `RESET_MUTANT=1` | |
|---|---|---|---|
| 1 (job 0) | `f0d43c096e9c6610` | `f0d43c096e9c6610` | **identical** |
| 2 (job 1) | `17e670688a0c8934` | `830a105de69617d0` | differs |
| 3 (job 0) | `f0d43c096e9c6610` | `5390f48b88678fcf` | differs |

**Run 1 is bit-identical between the two builds**, and it must be: run 1 is
preceded by the STARTUP seeding, which the mutant does not touch. Runs 2 and 3
are preceded by `run_reset()`, which the mutant breaks, and both diverge. So the
mutant changes exactly one thing — the reset between runs — and the digest
localises the damage to precisely the runs that pass through it. Every tracking
statistic of run 1 is also unchanged (mean IoU 0.5005), which is the same
statement in a coarser instrument.

The per-frame diagnostics confirm the mechanism unchanged from the first mutant
run: `accum` differs on 114/741 frames (first at frame 1, 1963 vs 1961),
`response` on 106/741 (1556 vs 1553), `F_ch` on 25/741. Sub-argmax throughout,
which is why the trajectory never moved.

### What this closes, and what it cost

The determinism test is now **known to detect a real leak** rather than assumed
to. It took three hardware runs to get there:

1. `run_0825_1431` — clean build, trajectory-only test: PASS. **Meaningless**, as
   it turned out, because the test could not fail.
2. `run_0825_1443` — mutant, trajectory-only test: also PASS. The negative
   control earning its keep — the only run that could have revealed this.
3. `run_0825_1523` / `run_0825_1530` — clean and mutant with the digest: PASS
   and FAIL respectively.

**A test that has never been shown to fail is worth nothing on a path with no
prior coverage** — the rule every RGB suite was built to, applied to the one
place it had not been. The cost of learning it here was one hardware run; the
cost of not learning it would have been every multi-start result in the project
resting on an instrument that could not see a leak.
