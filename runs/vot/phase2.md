# VOT Phase 2 — board frame source + runtime geometry

**2026-08-25. GATE MET on the datapath, FAILED on `rails=0`, and the tracker
LOSES REAL VIDEO.** `car1` anchor 0, all 742 frames, `runs/run_0825_1314.log` —
see "The gate run" at the bottom. The path is proven end to end; the tracking
result is a separate problem and a real one.

**Original status line, kept because the plan below was written before the run:** Everything testable without the board is tested and
mutation-tested; nothing here has seen silicon.

```
make test_vot_source     19 mutants, all REJECTED; 62/62 real manifests parse
make test_vot_format     board writer -> toolkit's own reader, EXACT
make application FRAME_SOURCE=vot|synth      both arms link, no warnings
```

## What was built

**`design/host_app_src/vot_source.{h,cpp}`** — manifest parse, blob load, run
order, trajectory accumulation and write. **No XRT and no ADF header**, the same
rule `mosse_filter` and `scene_colour` follow, so `make test_vot_source` runs it
natively in seconds. Nothing in the file computes anything; it is pure
bookkeeping, which is precisely why it needed a test suite before it needed a
board.

**`FRAME_SOURCE=synth|vot`** in the Makefile. `synth` is the default and is
compiled and linked exactly as before — `vot_source.cpp` is not even linked on
that arm, so the two differ structurally and not just by a `#define`. The flag
reaches `GCC_FLAGS` on **both** arms (`-DFRAME_SOURCE_VOT=0` or `=1`) rather
than only on one: a bare `#ifndef` default in the host is not a safety net, it
is what made `SCENE_VERIFY=1` silently build the instrument disabled.

**Runtime frame geometry.** `FRAME_ROWS`/`FRAME_COLS` are now the MAXIMUM — what
`frame_bo` is allocated at — and `g_frame_rows`/`g_frame_cols`/`g_frame_bytes`
are what the pipeline runs at, written once per sequence from the manifest.
Sites touched: the crop CU's static args and register writes, both
`scale_extract()` calls, the initial box, `scene_init()`, the scene functions,
the frame push and the colourise/verify pair. `roi_crop` already takes all of it
over AXI-Lite, so nothing needed re-synthesising.

**The frame-loop seam.** The synthetic scene block is replaced by two memcpys —
blob → `g_frame_host` → `frame_bo` — under `#if FRAME_SOURCE_VOT`. The first
copy is deliberate: repointing `scene_luma()` at the blob would save ~0.04 ms at
640×480 and would put the colourise path, the dirty-rect invariant and
`scale_extract`'s luma pointer on a buffer nothing else in the file expects.
Groundtruth for the IoU line comes from the manifest and **never steers the
tracker**.

**Run length is the JOB's, not `ITER_CNT`.** `g_run_frames` replaces `ITER_CNT`
everywhere it meant "frames this run executes" (the loop bound, the last-frame
trace predicates, every per-frame average in the cumulative reports). At
`FRAME_SOURCE=vot`, `ITER_CNT` is ignored, and the run banner now prints the
count the loop will actually run alongside the one the build was configured
with — the exact artifact `runs/.last_cfg` was.

## Two things that are checked rather than argued

**"synth reproduces today's behaviour exactly" is an assertion in the ELF.** The
runtime-geometry substitution is safe only if the runtime values ARE the macro
values on that arm, so `main()` checks all four and refuses to run otherwise. It
reads as obviously true right up until someone adds a knob that moves one.

Independently: the synth ELF was diffed against one built from `HEAD`. The only
strings that changed are the two deliberately reworded messages (the `ITER_CNT<2`
warning and the run banner), and **no vot_source symbol is linked into it**.
That is not a bit-identical-tracking test — only hardware can give that — but it
does rule out a dropped diagnostic or an accidentally-shipped parser.

**The trajectory writer is checked against the toolkit's own reader**
(`make test_vot_format`). Phase 0b round-tripped the format with the toolkit's
*writer*, which proves the format and the analysis and says nothing about the
`printf` that runs on the board. That printf performs the one piece of
arithmetic in the whole result path — centre `(row,col,h,w)` → top-left
`(x,y,w,h)` — and a transposed pair there produces a perfectly well-formed file
the toolkit scores without complaint. The C++ side emits its **input** (centre
boxes) beside the trajectory and the conversion is re-derived in Python, so a
wrong conversion disagrees instead of agreeing with itself. Confirmed to fail on
a transposed pair and on a missing centre offset.

## Format decisions

**Text (`.txt`), not the binary default.** `config.results_binary` governs the
toolkit's *writer*; `Trajectory.read()` accepts `name.txt` and `name.bin`
either way (`vot/tracker/results.py`). So the board writes text and Phase 5
ingests it by copying, with no board-side struct layout to keep in step.

Index 0 is `Special(INITIALIZATION)` = the line `1`, then one `x,y,w,h` per
tracked frame, plus `{name}_time.value` with one float per region. Trajectories
are accumulated in RAM and written **once** at the end of the run; nothing
touches the mount inside the frame loop.

## Deliberate refusals

- **`FRAME_SOURCE=vot` with `CONV_IN_CH=3` is a `#error`.** The RGB arm needs the
  manifest's `.luma` sidecar staged alongside the blob, because `scale_extract()`
  reads an intensity template out of `scene_luma()` and a VOT frame arrives
  interleaved. Building the pair today would give a colour datapath with a scale
  filter reading the R plane as luma — a tracker that works slightly worse, not
  one that fails. Grayscale first, per the plan.
- **An unrecognised command-line argument is fatal.** A typo in `--vot-seq` that
  fell back to the compiled-in default would produce a complete run attributed to
  the wrong sequence, and the console would not say so.
- **`--vot-max-frames N` truncates the run and then refuses to write the
  trajectory.** A short result file is read back without complaint and scored as
  a tracker that stopped early, so bring-up runs say so on the console instead.

## Assertions the manifest makes possible, and which are wired

`channels != CONV_IN_CH` (a blob converted for the other arm), geometry beyond
`FRAME_ROWS`/`FRAME_COLS`, job index out of range, an empty init box (41 frames
of stb2022 have empty groundtruth; Phase 1 checked no anchor lands on one, this
asserts it anyway), a blob whose length disagrees with `frames × frame_bytes` in
**either** direction. Advisory notes are printed for `roi_exceeds_frame` and for
an init box under 16 px, both with the numbers Phase 1 measured.

Empty-groundtruth frames encountered mid-run are counted and reported, because
the board's IoU line scores them 0 while the toolkit's failure rule ignores them
— two different statistics, and the PC-side one is the number of record.

## The gate, and how to run it

Not met. It needs one anchor of one sequence tracking end to end on hardware.

```bash
# grayscale, one sequence, one anchor
make sd_card TARGET=hw FRAME_SOURCE=vot DUMP_BUFFERS=0 VERBOSITY=1 CSV_LOG=1
# on the board, with /mnt/vot and /mnt/vot-results mounted per phase0a.md
./mosse_tracker.elf a.xclbin --vot-seq car1 --vot-job 0
```

**`a.xclbin`, not anything more descriptive** — it is `v++ --package`'s default
name and it is what `design/exec_scripts/run_script.sh` passes. An earlier draft
of this file said `mosse.xclbin` and cost a board run: the ELF stages the whole
sequence BEFORE opening the device (deliberately, so a bad sequence name is
cheap), so the xclbin error arrives 2 s in, after a completely healthy-looking
`[vot]` banner.

## Measured on the board, 2026-08-25

The staging slot, on a real converted blob rather than Phase 0a's
`/dev/urandom` probe — the one item phase0a.md left open:

```
[vot] car1  480x640 x1  742 frames  md5 f0f01e9612179e13d4df460d5d8f1e1a
[vot] job 0/15: anchor 0 forward, 742 frames  init box 114.0x117.0 at (221.0,300.5)
[vot] staged 217.4 MB in 2.15 s = 101.2 MB/s
```

**101.2 MB/s against the probe's 117.2**, i.e. 13.7% slower, and the difference
is very likely NOT the link: `Blob::load` resizes the heap buffer before reading
it, so the slot includes first-touch zeroing and page faults on 217 MB that the
`dd` probe never paid. Do not "correct" the figure to the probe's — **this one
is what a run actually costs**, which is what Phase 5 must budget against. If it
ever needs decomposing, time the `resize` separately rather than inferring.

Either way the conclusion is unchanged and has margin: 9.5 GB of blobs at
101 MB/s is ~1.6 min of staging against ~79 min of tracking.

**The mount itself was the trap, not the code.** `/srv/vot/data` held only
Phase 0a's `probe.bin`; the blobs live in `$VOT_ROOT/data`. Bind-mount rather
than copy — a copy is a second authority for 9.5 GB of regenerable output that
can drift from the manifests' own md5s — and note that exporting
`$VOT_ROOT/data` directly does NOT work: `/home/<user>` is `drwxr-x---` and
`root_squash` maps the board's root to `nobody`, which cannot traverse it. The
bind mount means the server never resolves that path at all.

```bash
sudo mount --bind $VOT_ROOT/data /srv/vot/data && sudo exportfs -ra
#   /etc/fstab:  /home/<user>/vot/data  /srv/vot/data  none  bind  0 0
```

The **results** export still needs mounting on the board; Phase 0a only ever
exercised the read side, and without it a run tracks perfectly and then fails at
the single write that makes it worth anything:

```bash
mkdir -p /mnt/vot-results
mount -t nfs -o vers=3,nolock,rw,wsize=1048576,proto=tcp \
      192.168.10.1:/srv/vot/results /mnt/vot-results
```

**Order the exposure deliberately** (phase0c.md): `car1` first — boxes 49–139 px,
resample near 1:1 — to establish the path works at all, *then* a sequence that
hits the bilinear interpolator hard. It has never run on hardware in any mode:
every build to date sets `roi_h == patch_rows`, collapsing the datapath to a
copy.

**What to assert, written down before the run:** `rails = 0`, `roi_crop launch`
in the ~1 ms band it occupies at 1:1, and the trajectory bounded against
`rgb_vs_gray_vot.py` on the same anchor. **Bounded centre error, NOT
bit-exactness** — the Python arm uses a float FFT by design. Say which one you
are asserting before running, because "close" is exactly the verdict that
absorbs a real 1-LSB convention error (phase0c.md's `pilluma` mutant).

`scripts/rgb_vs_gray_vot.py` can now be pointed at stb2022: its `load_gt` was
carrying Phase 1's polygon-only bug and now single-sources `reduce_box` from
`vot_prepare`, verified unchanged on all 17 polygon files in `test-sequences/`
and correct on the 4-column stb2022 format.

## Not in this phase

`DUMP_BUFFERS=0` is a run-configuration matter, but note that the default is
**1** — 1216 KB and ~2 s per frame, which would dominate any VOT run. Phase 4's
`PROGRESS_EVERY` and `CSV_FLUSH_EVERY` are not built yet; at `VERBOSITY=0` the
per-frame console line is ~15% of a 26.29 ms frame and `csv_row()` still
`fflush()`es every row, over NFS.

Per-run reset (Phase 3) does not exist: this arm runs ONE job and exits.


## The gate run — `car1` anchor 0, 742 frames (`runs/run_0825_1314.log`)

### What passed

| | |
|---|---|
| Frames tracked | 742 / 742, no stall, no crash |
| Trajectory | `/mnt/vot-results/car1_00000000.txt`, **742 regions, 19.8 ms**, one write |
| Toolkit read-back | 742 regions, index 0 `Special(1)` — **the `rw` export is now exercised**, closing phase0a.md's last open item |
| Board IoU vs PC recomputation | **agrees to 1e-4 on all 741 frames** |
| `roi_crop launch` | **0.990 ms** against synth's 1.013 |
| Frame seam | blob memcpy 0.068 ms + push 0.064 ms = **0.13 ms/frame** |

**The board's own IoU and a PC-side recomputation from the WRITTEN TRAJECTORY
agree to 1e-4 on every frame.** That is the strongest single result here: it
validates the frame indexing, the groundtruth alignment, the centre → top-left
conversion and the file itself in one measurement, using the number the board
printed and the number the toolkit's parser recovers. An off-by-one anywhere in
the seam would have shown up as a systematic offset, not as 1e-4.

**The bilinear interpolator ran on hardware for the first time and cost
nothing.** `car1`'s init box is 114×117, so `roi = 228×234` resampled DOWN to
128×128 — a 1.78× downsample, where every previous build set `roi_h ==
patch_rows` and collapsed the datapath to a copy. `roi_crop launch` came in at
0.990 ms against the synthetic arm's 1.013. That is the expected answer and it
is worth stating why: PASS1 is **output-driven** — four bilinear taps per OUTPUT
pixel regardless of the ratio — so the resample changes which addresses are read
and not how many. The risk register rated this High; the *timing* half of it is
now closed for downsampling. **Upsampling is still untested** — `car1`'s
smallest box is 76 px, and the dataset's extreme is `tennis` at 2.0 px.

### What failed

```
[track] SUMMARY over 741 evaluated frame(s):
  overlap precision @0.5 :  61.4%  (455 of 741; 286 lost)
  mean IoU               : 0.5005   (worst 0.0000)
  mean centre error      : 136.72 px (worst 490.59 px)
  final box              : 128x132 at (-84.9,-2.1)  truth 85x92 at (259.5,336.0)
[gate] 741 evaluated, 577 accepted, 164 gated
[gate]   reasons:  ACCEPT x577  NEGATIVE_PEAK x87  LOW_PSR x77
[gate]   longest gated run: 53
```

**Three loss episodes, and the third is permanent:** frame 9 (1 frame,
recovered), frames 374-377 (4 frames, recovered), **frames 461-741 (281 frames,
never recovers)**. The tracker held IoU 0.85-0.90 for the first ~350 frames and
0.66-0.78 from ~360 to 460, so this is not a bring-up defect — it tracked, then
lost.

**The mechanism is the HOLD-on-gate policy meeting real motion.** `car1`'s
median inter-frame motion is **15.1 px** with bursts to 90; frames 457-461 move
31, 49, 29, 24, 48 px. On a gated frame the host holds position by design
(Bolme §3.5, and correct for occlusion: moving to a noise peak walks the ROI off
target permanently). But a HOLD assumes the target does not go anywhere while
the filter is frozen. Here the target keeps moving at 15-50 px/frame, so each
held frame makes the next one worse — the longest gated run is 53 frames, by
which point the object is hundreds of px away and the ROI contains no target at
all. `NEGATIVE_PEAK x87` is the signature of that end state: the response peaks
*negative*, i.e. the filter is anti-correlated with whatever the ROI now holds.
**This policy has never been tested against a moving target before** — the
synthetic occlusion test (`OCCLUDE_MASK`) occludes a target that stays put.

**The scale filter is worse than its documented floor, and at one point made
things actively worse.** At frame 452 the truth box is 84×98 and the tracker's
is 110×113 — a ~25% linear error against the "~8-10% is this filter's practical
floor" recorded in CLAUDE.md, which was measured on a synthetic envelope moving
at 0.94%/frame. `car1` shrinks to 0.6 of its initial area over the run, with
real aspect change the isotropic filter cannot represent. Worse, at **frame 490,
with the tracker already 227 px off target, the scale gate ACCEPTED a +9-level
jump** (105→150 px, factor 1.42). `SCALE_CONF_MIN=2.0` did not veto it. This is
the failure CLAUDE.md predicts in words — "`conf` cannot distinguish a wrong
proposal from a big correct correction" — observed for the first time on
hardware, and it is on the wrong side: inflating the box while lost makes
reacquisition strictly less likely. **The scale filter is already slaved to
the position gate** — `if (scale.enabled() && gate.accept && scale.initialized)`,
and this run proves it holds: 577 position ACCEPTs, 577 scale evaluations, zero
on a held frame. An earlier reading of this log proposed *adding* that slaving,
inferred from frame 490's line without checking the counts. It was already
there. Frame 490 passed because the POSITION gate accepted it at PSR 7.87
against the 7.00 threshold while 227 px off target — the weak-PSR problem, not a
missing scale precondition.

**What was added instead is `SCALE_MAX_STEP`** (2026-08-25): a per-frame RATE
limit on the proposal, default 2, which vetoes three of this run's seven bad
proposals including frame 490's +9. The tempting default of 1 — which would have
caught all seven — was measured in `scale_loop_sim` and REJECTED: it parks the
normal smooth-envelope arm for 123 of 200 frames and ends it 28.0% wrong against
1.0% unlimited, because that detector legitimately uses ±2. Any limit at all
costs the abrupt-change arm (42.9% end error against 8.3% unlimited), the same
irreducible trade `SCALE_CONF_MIN` already carries. **None of this would have
prevented the loss at frame 461** — all seven bad proposals are after it. It
keeps the box sane while lost, which is a precondition for reacquisition, not
reacquisition itself.

**`rails = 0` FAILED: 8 frames railed** — `response` at 51-53 and at 234/244/
250/258/269, `accum` at 234 and at 526/543/548. All the early ones happened
while tracking was healthy (IoU 0.78-0.90), so the rails did not cause the loss,
but they are exactly what the risk register predicted: **4-4-4 at `H_SHIFT=11`
was validated against the synthetic scene, and real video moves `|F|`.** Note
`F_ch` (ch0) ran 2500-6900 here against ~7400 on the synthetic arm, while the
response still railed — the distribution is different in shape, not just scale,
so this needs re-deriving from a VOT run rather than adjusting by a ratio.

### Frame time — do not quote 91.81 ms

`[apu] CUMULATIVE ... mean frame body 91.81 ms` with **UNATTRIBUTED 68.78 ms
(74.9%)**, and the log prints its own warning that a breakdown with a residual
that size is not the frame. That residual is the `VERBOSITY=1` console at
115200 baud, and the per-frame slots say what the frame actually costs:

```
APU subtotal 15.055 + GMIO 11.094 + roi_crop 0.990 + unattributed 0.596 = 27.7 ms
```

against the synthetic arm's 26.29 — i.e. **VOT frames cost what synthetic frames
cost**, and the whole difference is the smaller frame push (0.064 ms for 307 KB
against 0.472 for 2 MB) offset by the run's own noise. This is the measurement
Phase 4 exists to make quotable: at `VERBOSITY=0` the per-frame line is still
~15% of the frame, and `csv_row()` still `fflush()`es every row.

*(The `frame push (2MB)` slot label is now wrong at `FRAME_SOURCE=vot` — it is
307 KB here. Cosmetic, but it is a label that states a number.)*

### What this does NOT tell us

Nothing here is an AR score. The board ran ONE anchor of ONE sequence with no
reset, so `vot analysis` has nothing to consume yet — and the toolkit's failure
rule would have called the first failure at frame 375, not 461, because it fires
at overlap ≤ 0.1 rather than at the 0.5 the board's console reports. **Those are
two different statistics and neither is the other's approximation.**
