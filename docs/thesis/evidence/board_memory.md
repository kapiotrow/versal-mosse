# The board has ~1.7 GB, not 12 GB, and RGB did not fit

**Status:** closed · **Updated:** 2026-08-27 · **Scope:** usable heap is ~0.9-1.2 GB, not 12 GB; the streaming reader that recovered the five oversized RGB sequences

**Status: CLOSED, 2026-08-27.** The streaming reader is built, tested,
validated by a mode-equivalence run on `car1`, and the five oversized sequences
have now RUN — `runs/vot/0827_1326-rgb15stream`, 5 ran / 0 failed / 88
trajectories. RGB covers **62 of 62**, the AR report over all 62 is in
`~/vot/analysis/full62`, and the comparability note at the bottom of this file
is retired. The gray arm is unaffected (it completed
62/62); this only bites at `CONV_IN_CH=3`, where a blob is 4x the size.

**Two things in the original text below did not survive the measurement, and are
kept as written so the correction is legible. See "2026-08-27" at the end.**
  1. The failure count was **5 of 62, not 8** — `ants1`, `drone1` and
     `conduction1` fit after all.
  2. **Step 1 therefore recovers ZERO sequences**, since those three are exactly
     the ones it was scoped to rescue. It is not a memory fix. It survives only
     as a BANDWIDTH fix under Step 2, on two sequences.

## What happened

The RGB full-62 sweep (`runs/vot/0826_1550-rgb15`) failed on
`flamingo1`, `frisbee`, `girl`, `nature` with `std::bad_alloc` and OOM kills,
and would have failed on `ants1`, `drone1`, `conduction1`, `zebrafish1` for the
same reason. `board_run.sh` reported every one as `rc=93` rather than as a
clean finish -- the guard added earlier the same day, which before that morning
would have counted them as "done, 0 trajectories".

## The premise that was wrong

The plan's NFS section says:

> peak RAM is one sequence, not the dataset: girl at 1500 frames of 640x480
> grayscale is 461 MB, comfortable against 12 GB of LPDDR4

The VEK280 does carry 12 GB. **Linux cannot reach it.** Measured on the board:

```
MemTotal      1799156 kB      MemAvailable  ~950000 kB
CmaTotal       524288 kB      <- half a GB of the usable bank is CMA for XRT

/proc/device-tree/memory@00000000     0x0_00000000 +0x0_80000000   (2 GB)
                                      0x8_00000000 +0x0_80000000   (2 GB)
/proc/device-tree/memory@50000000000  0x500_00000000 +0x2_00000000 (8 GB)

/proc/iomem:
  00000000-7fffffff : System RAM      <- the ONLY usable bank, 2 GB
  800000000-87fffffff : reserved      <- 2 GB reserved
  (the 8 GB bank at 0x500_00000000 never appears)
```

So the DT declares all 12 GB and Linux maps 2 GB, of which 512 MB is CMA.
**Do not repeat the 12 GB figure from CLAUDE.md's resource table as an
available-memory budget** -- it is the part's capacity, not the PS's map.

## Who does not fit

Blob = frames x rows x cols x channels, plus the `.luma` sidecar at
channels=3. Against ~950-1200 MB available (it moves with cache and CMA):

| sequence | RGB raw+luma | RGB raw only | gray |
|---|---|---|---|
| flamingo1 | 4841 MB | 3631 | 1210 |
| zebrafish1 | 3164 | 2373 | 791 |
| nature | 1976 | 1482 | 494 |
| frisbee | 1962 | 1471 | 490 |
| girl | 1758 | 1318 | 439 |
| ants1 | 1300 | 975 | 325 |
| drone1 | 1248 | 936 | 312 |
| conduction1 | 1213 | 910 | 303 |

8 of 62 on RGB; 1 of 62 on gray (`flamingo1`, which still fit).

## OPTION REJECTED: load per JOB instead of per sequence

Tempting, because a backward run from anchor `i` needs only frames `[0..i]`.
**It fixes nothing: every sequence has a FORWARD run from anchor 0, which needs
the whole blob.** Peak memory is unchanged. Written down because it is the first
idea everyone has.

## Step 1 (cheap, ~25%): derive luma on the board, drop the sidecar
## — SUPERSEDED 2026-08-27: recovers 0 sequences. See the end of this file.

RGB is 3x raw + 1x luma; dropping the sidecar saves a quarter of the heap AND a
quarter of the NFS staging. Recovers **ants1, drone1, conduction1** -> 57/62.

Cost: a full-frame BT.601 pass, ~307k px/frame, ~0.3-0.6 ms against a 25 ms
frame. `scale_extract` reads only a box-sized region, so a per-ROI derivation
would be cheaper still, but full-frame keeps `scene_luma()`'s contract intact.

**The convention is pinned and must be reproduced exactly.** `vot_prepare.py`
uses float 0.2989/0.5870/0.1140 with round-half-even and a clip -- explicitly
NOT PIL's truncating integer path. An on-board derivation that rounds
differently gives a 1-LSB-different intensity template: immaterial to tracking,
fatal to any bit-exact comparison against a sidecar run. Needs a native test
against the same golden `to_luma`, in `make test_vot_source`.

## Step 2 (the real fix): windowed blob reader — BUILT 2026-08-27

Required for the other five (girl, frisbee, nature, zebrafish1, flamingo1) and
it SUBSUMES step 1, so if all 62 is the goal, build this and treat step 1 as
optional.

Access is strictly sequential in run order -- forward `[anchor..end]` or reverse
`[anchor..0]` -- so a fixed ring of K frames with a prefetch thread is enough.
It is NOT `mmap`: the plan rejected mmap because demand paging moves I/O into
the frame loop as page faults and reports it as unattributed frame time. An
explicit prefetch keeps the cost in a named slot, which is the whole difference.

Budget: 921 KB/frame at the Phase 0a rate (117.2 MB/s) is **7.9 ms/frame** of
I/O against ~25 ms of compute, so it overlaps IF prefetched. A synchronous
refill every K frames does NOT hide it -- it amortises to the same 7.9 ms.
`TAIL_PARALLEL` already puts the filter tail on core 1, so this would be a third
thread on two cores; it should be mostly I/O-blocked rather than CPU-hungry, but
that is an assumption to measure, not to assert.

Report the staging slot separately from the frame body, the way `AP_*` already
does. A reader that quietly inflates frame time is worse than one that fails.

## Not the fix: recovering the other 10 GB

The 2 GB at `0x8_00000000` is reserved and the 8 GB bank is unmapped. Changing
that is a platform device-tree edit plus a boot-image rebuild -- a different
piece of work with its own risk, and it would invalidate every frame-time figure
taken so far if the memory map moves. Worth knowing it exists; not worth folding
into this.

## Comparability note for the AR report — RETIRED 2026-08-27

~~Gray covered 62/62, RGB will cover 54/62. Any gray-vs-RGB table must restrict
gray to the SAME 54.~~ **Both arms now cover 62 of 62 and the run sets are
IDENTICAL** — 419 trajectories each, same sequences, same anchors, checked by
diffing the two arms' run-name lists before the ingest. No subsetting is needed
and none should be applied.

**It was worth 5 sequences, not the 8 predicted — and reading the subset would
have been wrong in a way that is worth recording.** The 57-sequence figures were
A 0.5406 / R 0.3343 / EAO 0.1314; the full 62 are A 0.5043 / R 0.3065 / **EAO
0.1474**. A and R fell, as they should on five hard sequences, but **EAO ROSE**.
EAO pools over a subsequence-LENGTH curve and the five change the length
distribution (`girl` 1500, `flamingo1` 1377, `nature` 999), so a subset's EAO is
not a truncated version of the whole — it is a different number. A and R degrade
gracefully under subsetting; EAO does not. Quote only the full-62 EAO.

---

## 2026-08-27 — WHAT THE RUN ACTUALLY SHOWED, AND WHAT WAS BUILT

### The failure count was 5, not 8, and it moves Step 1 off the critical path

`runs/vot/0826_1550-rgb15` completed **57 of 62 sequences, 101,564 frames**. The
five that died are `flamingo1`, `zebrafish1`, `nature`, `frisbee`, `girl` — the
five whose RAW blob alone is 1318 MB or more. `ants1` (975 MB), `drone1` (936)
and `conduction1` (910) **completed with the sidecar loaded**, against a table
above that predicted all three would fail.

That matters for sequencing, not for pride: **Step 1 was scoped to recover
exactly those three**, and dropping 25% of the heap does not bring any of the
remaining five under a ~1.0-1.2 GB ceiling (`girl`, the smallest, is 1318 MB raw).
So Step 1 is worth **zero sequences** and Step 2 is not merely the better fix, it
is the only one.

**Where Step 1 does still pay is BANDWIDTH, not heap.** Under streaming the
binding constraint stops being the ceiling and becomes 117.2 MB/s of NFS:

| | MB/frame | I/O at 117.2 MB/s | vs ~28 ms of compute |
|---|---|---|---|
| girl | 1.17 | 10 ms | hides under prefetch |
| nature | 1.98 | 17 ms | hides |
| flamingo1 | 3.52 | 30 ms | borderline |
| frisbee, zebrafish1 (1080x1920) | 7.91 | **67 ms** | **I/O-bound, 2.4x** |

Dropping the sidecar takes the two 1080p sequences from 67 to 50 ms — still
I/O-bound. They are 743 and 1798 tracked frames, so it is ~3 minutes of wall
clock either way. **Do Step 1 only if their frame time needs to be comparable,
and note that it will not be comparable regardless.**

### What was built

`vot::StreamBlob` in `design/host_app_src/vot_source.{h,cpp}`, and it is a
HYBRID: the resident `Blob` is untouched and stays the default. That is
deliberate and load-bearing — **57 of 62 RGB sequences have already run on the
resident path, and their results are only comparable to the remaining five if
nothing about their frame delivery moved.**

- **A ring of K frames plus a prefetch thread**, not mmap (demand paging moves
  I/O into the frame loop and reports it as unattributed time) and not a
  synchronous refill every K frames (amortises to the same bytes on the same
  critical path and merely makes the cost lumpy).
- **The whole access pattern is known before the run starts**, because
  `job_order()` IS the list. So the producer walks a known list and the
  consumer's index is checked against it. **Out-of-order access is an ERROR, not
  a seek** — a silent seek would serialise the run against NFS and read as
  "streaming is slow" rather than as the misuse it is.
- **Selection**: automatic above `VOT_RESIDENT_MAX_MB` (default 700), overridable
  per run with `--vot-stream auto|always|never`, and **printed on both arms** —
  a run whose frame-delivery mechanism is not in its own log has a frame time
  that cannot be compared to anything.
- **The wait has its own AP slot** (`AP_VOT_STAGE`), separate from `AP_SCENE`,
  because the entire argument for a prefetched ring is that the wait is usually
  zero, and a claim like that is worth nothing unless the run prints the number.
  A second, independent accounting inside the reader is printed per job as a
  cross-check.
- **The sidecar follows the blob's mode.** Streaming one and staging the other
  would defeat the point on exactly the sequences that need it — `nature`'s
  494 MB of luma is most of the heap the streamed blob just freed.
- `AP_VOT_STAGE` is inside `#if FRAME_SOURCE_VOT`, so **nothing from this change
  reaches the synth arm**: preprocessing `mosse_tracker.cpp` at
  `FRAME_SOURCE_VOT=0` finds zero occurrences of every symbol it adds.

### Why the tests are worth something

`make test_vot_source` gained 20 checks, and they were **mutation-tested against
the implementation** rather than declared:

| mutant | caught? |
|---|---|
| producer writes slot `i+1` | FAIL (7) |
| `at()` releases its own slot (producer overwrites under the caller) | FAIL (5) |
| producer allowed one slot too far ahead | FAIL (4) |
| `begin_run` keeps the previous job's order | FAIL (2) |
| `begin_run` ignores the truncated length | FAIL (1) |
| `open_luma` uses `frame_bytes` (the 3x sidecar bug) | FAIL (4) |
| no length check at `open` | FAIL (3) |
| `at()` seeks instead of refusing out-of-order access | FAIL (2) |
| producer publishes a frame it never filled | FAIL (7) |
| producer fills half a frame per `pread` | **PASS — and correctly so** |

The last row is the positive control: the partial-read loop genuinely completes
the frame, so halving each request is a no-op rather than a defect. **A real bug
in the test itself fell out of this**: the truncation check named `at()` twice
inside one `check()` call, and `at()` is stateful, so with the detail argument
evaluated first the condition's call was rejected as out-of-order and the check
passed for the wrong reason. It is why the "ignores the truncated length" mutant
survived the first pass.

There is also a real-data rehearsal, skipped without `$VOT_ROOT`: it picks the
LARGEST blob present — which is `flamingo1`, one of the five — and compares a
forward slice and a backward slice, blob and sidecar, against an independent
`fopen`/`fread`. The unit fixtures are 40-byte frames, where a partial read is
impossible; `flamingo1`'s are 2.64 MB, where it is routine.

### THE BOARD-SIDE ACCEPTANCE TEST — RUN 2026-08-27, PASSED

Streaming changes no arithmetic, so the criterion is exact and needs no
groundtruth:

```bash
scripts/vot_sweep.sh --arm streamA --seqs car1 --stream never  --data /srv/vot/data-rgb
scripts/vot_sweep.sh --arm streamB --seqs car1 --stream always --data /srv/vot/data-rgb
```

`runs/vot/0827_1313-streamA` and `runs/vot/0827_1318-streamB`. RGB
`H_SHIFT=15` build, the same xclbin the full-62 sweep ran on
(`52235f49221e`), 15 anchors, 8434 frames each.

| criterion | result |
|---|---|
| run-state digests | **15 of 15 IDENTICAL** |
| trajectories | **15 of 15 byte-identical** |
| `track.csv`, all 36 columns x 8434 rows | **byte-identical** |
| mean frame body | 26.84 ms resident, **26.78 ms streamed** |
| `AP_VOT_STAGE` | 0.001 ms/frame resident, **0.037 ms/frame streamed (0.1%)** |
| heap held for frames | 870 MB resident, **9.3 MB of rings streamed** |
| up-front staging read | 8.5 s resident (652 MB at 102.5 MB/s), **none streamed** |

**`car1` was chosen because it is the one sequence that can settle this**: at
870 MB it FITS in heap, so it can be run both ways, and it is above the 700 MB
threshold, so `auto` would stream it. A sequence that only fits one way cannot
be compared.

**Streaming is FREE here, and the reason is in the numbers rather than in the
result**: the per-job wait is 0.011 ms/frame on the first job and 0.002-0.003
after, i.e. the ring is never empty when the frame loop asks. `car1` is
1.17 MB/frame = ~10 ms of I/O against ~27 ms of compute, which is exactly the
regime the budget table above predicts prefetching can hide. **Do not read this
as "streaming is free"** — the same table says `frisbee` and `zebrafish1` at
7.91 MB/frame cannot be hidden, and the honest 0.037 ms in `AP_VOT_STAGE` here
is what will become tens of milliseconds there. That slot existing is the point.

**One defect the run found, and it was in the log rather than in the run.** On
the streamed path the resident staging line printed `staged 0.0 MB in 0.00 s =
0.0 MB/s` -- indistinguishable from a blob that failed to load, on a run that
was in fact byte-perfect. Fixed to say there is no staging read. A correct run
whose log reads like a failure is the wrong way round, and it would have been
read as one on the first oversized sequence.

**The digest was the criterion, not the trajectory** — a byte-identical
trajectory is not a bit-identical run, which this project already learned from a
`RESET_MUTANT=1` build that passed the determinism test. Here all three agreed,
which is the outcome that makes the weaker two informative rather than the one
that makes them sufficient.

Next: `--stream auto` over the five, then re-ingest all 62 and
regenerate the AR table. The comparability note below then no longer applies —
gray covers 62 and RGB will too, so the comparison stops needing a subset.

## 2026-08-27 — THE FIVE RAN. THE READER'S COST, MEASURED

`runs/vot/0827_1326-rgb15stream`, `--stream auto`, 78,980 tracked frames,
5 ran / 0 skipped / 0 failed / 88 trajectories. `flamingo1` — 4841 MB of frames,
the sequence that died on `std::bad_alloc` — ran on **28 MB of rings** with
1486 MB still free on the board, and `auto` selected streaming without being
told.

| sequence | MB/frame | predicted I/O | measured wait | frame body |
|---|---|---|---|---|
| girl | 1.17 | 10 ms | **0.007 ms** | 27.15 ms |
| nature | 1.98 | 17 ms | **0.026 ms** | 28.99 ms |
| flamingo1 | 3.52 | 30 ms | 1.10 ms | 34.77 ms |
| frisbee | 7.91 | 67 ms | **24.43 ms** | 64.41 ms |
| zebrafish1 | 7.91 | 67 ms | **28.68 ms** | 70.33 ms |

**The prefetch hides the I/O completely up to ~2 MB/frame, mostly at 3.5, and at
7.91 it hides ~40 of the 67 ms and cannot hide the rest.** That is what the
budget table predicted and it is why the wait has its own slot instead of
landing in UNATTRIBUTED. **The two 1080p sequences' frame times are NOT
comparable to any other run** and must be quoted separately if quoted at all.

### A reading error worth recording, because the instrument caught it

An intermediate reading of `AP_VOT_STAGE` put `girl` at **10.976 ms/frame** —
absurd next to `car1` at 0.011 ms/frame for the identical 1.17 MB/frame — and a
page-cache story was already being assembled to explain it. It was a PARSE bug:
`tail -1` on a still-running log picks a per-frame breakdown row, where the first
ring fill legitimately dominates, not the cumulative row. The reader's own
per-job accounting reads a flat 0.007-0.013 ms/frame across all 31 anchors.
**Two independent instruments on the same quantity is what made the wrong one
obvious**; with only the AP table there was a plausible number and a plausible
mechanism, and nothing to contradict either.

### The three-way build equivalence that licensed combining the results

The 57 sequences came from the `rgb15` ELF and the 5 from a later one, so the
combined 62 is only one tracker if the builds agree. Checked, not assumed, on
`car1`'s 15 anchors:

```
rgb15 ELF (resident)  ==  new ELF (--stream never)  ==  new ELF (--stream always)
```

all 15 digests identical at every step. Same xclbin `52235f49221e`, same
weights. So all 419 RGB runs are one tracker regardless of which build emitted
them, and `vot_ingest.py` additionally re-derived every run name and checked
every trajectory length against the dataset's anchors before analysing.
