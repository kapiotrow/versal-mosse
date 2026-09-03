# VOT Phase 4 — overhead removal: `PROGRESS_EVERY`, `CSV_FLUSH_EVERY`

**Status:** closed · **Updated:** 2026-08-25 · **Scope:** VOT bring-up 4: console overhead removal, and the transport that was the whole win

**WHERE THIS ENDED UP.** It ran: three `car1` runs on 2026-08-25 (all 15 anchors, 8434
frames each) put both knobs at 1.1% and the TRANSPORT (UART -> ssh, 3.79 ms/frame) at the
whole win — claim `P-09`. Both knobs are kept. The status below is as written before that.

**2026-08-25. BUILT AND VERIFIED IN THE BINARY. Not yet run on hardware** — the
gate is a before/after frame time on one sequence, which the next board session
provides.

Two per-frame costs were justified against a frame time that no longer exists.
Both justifications were correct when written; the floor moved underneath them.

## The two knobs

| | was | now | default |
|---|---|---|---|
| level-0 progress line | every frame, ~45 B, "4% of an ~87 ms floor" | every Nth frame | **1 = every frame** |
| `track.csv` flush | every row, "nothing against 1216 KB/frame of binaries" | every Nth row | **1 = every row** |

The floor is now **26.29 ms**, so that same line is **15%** of the frame — and on
a gate-heavy sequence far worse: `correlation(gated%, unattributed frame time) =
0.963` across the 8-sequence sweep, with `animal` spending **58%** of its frame
printing (`evidence_arm0.md`). At `DUMP_BUFFERS=0` there are no per-frame
binaries left for the CSV flush to hide behind either, so it is a filesystem sync
in the timed path once per frame.

**Neither default changed**, and that is deliberate in both cases:

- **The level-0 line is not silenced, only thinned.** Its per-frame marker is
  what `picocom | ts` times, and that instrument is how the frame time was
  measured in the first place. **Frame 0 and the last frame always print**,
  whatever N — a run whose final line is missing looks exactly like a run that
  hung.
- **Per-row flushing was justified by surviving a power cut, and a power cut is
  not hypothetical here** — one took out arm B's `car1` run on 2026-08-25.
  Raising `CSV_FLUSH_EVERY` trades the tail of a run for frame time, and a sweep
  makes that trade knowingly. A **railed** row flushes regardless of N: it
  invalidates the shift budget and is the one row worth interrupting a sweep
  over, and rails are rare (8 frames in 742 on the worst run recorded).

**A gate veto deliberately does NOT force a flush**, though it is the tempting
choice. Vetoes are a routine tracking outcome, and they are commonest on exactly
the runs this knob exists for — `animal` gates 76% of its frames, so flushing on
vetoes would flush 76% of rows and save nothing where the cost is worst.

## What was verified, and how

**1. The defaults are BYTE-IDENTICAL to the pre-knob binary.** Not argued from
`#if` — measured, `cmp` on the ELF against one built from the committed source.
Getting there took two corrections, and both are the point of doing it this way:

- an unconditional `static long csv_rows` counter changed codegen even at
  `CSV_FLUSH_EVERY=1`, so the default arm is now the ORIGINAL line textually,
  under `#if CSV_FLUSH_EVERY > 1 / #else`;
- an explicit `fflush()` added to `csv_close()` shifted every later offset
  (9958 differing bytes from one inserted call). `fclose()` flushes by
  definition, so it was removed rather than kept for symmetry.

**The control that makes the comparison mean anything:** building the same
source twice gives a byte-identical ELF, so this project's ELF diff is a valid
instrument and not a coin flip. Worth knowing independently — `phase2.md` used
an ELF diff to argue the `synth` arm was unchanged.

**2. THE FIRST `PROGRESS_EVERY` CHECK SAID THE KNOB WAS INERT, AND IT WAS RIGHT
TO.** `PROGRESS_EVERY=25` produced an ELF identical to the default — because the
default build is `VERBOSITY=1`, where the level-0 line does not exist and the
whole branch is dead-code-eliminated. Rebuilt at `VERBOSITY=0`, where the line
does exist, the ELF differs (49 bytes) at 10 and 25 and matches at 1.

That is the `SCENE_VERIFY` lesson arriving on schedule: **verify a feature flag
on a build that can actually exercise it.** Had the first check been read as
"the knob works" instead of "the knob is inert", a sweep would have run with an
unthinned console and the frame times would have looked like a regression in
something else.

## Use

```bash
scripts/vot_sweep.sh --arm x --seqs ...        # with the ELF built as:
make application TARGET=hw FRAME_SOURCE=vot \
     VERBOSITY=0 DUMP_BUFFERS=0 PROGRESS_EVERY=25 CSV_FLUSH_EVERY=200
```

Both arms of `FRAME_SOURCE` link clean with the knobs set.

## THE GATE, RUN — AND THE PREDICTION WAS WRONG BY 15x

**2026-08-25, three runs on `car1` (all 15 anchors, 8434 frames each), all 15
state digests identical across all three, so every difference below is console
and I/O and nothing else.**

```
configuration                                frame ms   unattributed
UART, progress every frame, flush every row      28.48         4.98     <- the old flow
ssh,  progress every frame, flush every row      24.69        0.953
ssh,  progress every frame, flush every 200      24.70        0.947
ssh,  progress every 25,    flush every 200      24.43        0.953
```

Decomposed, one variable at a time:

| change | worth |
|---|---|
| UART -> ssh (the automation) | **3.79 ms** |
| `CSV_FLUSH_EVERY` 1 -> 200 | **0.00 ms** |
| `PROGRESS_EVERY` 1 -> 25 | **0.27 ms** |

**The prediction written down before the run was ~4.0 ms for the console
thinning. It is 0.27 ms.** The two knobs Phase 4 exists for are together worth
1.1% of the frame, and the entire 4.05 ms improvement came from the transport
change that shipped alongside them.

**Why the model was wrong, and it is the same shape as every other wrong model
on this project: the premise, not the arithmetic.** 92.5 us/byte is the cost of a
byte *at 115200 baud*. The run it was predicting was launched over ssh, where a
byte costs essentially nothing, so thinning the bytes saves essentially nothing.
**The knob targeted a cost the automation had already deleted** — and the two
changes shipped in the same session, which is exactly the "never move two
magnitudes at once" trap, avoided here only because the control run was run.

`CSV_FLUSH_EVERY` is worth **literally zero** for a second reason worth
recording: the automation runs the ELF from `/tmp/mosse`, and `/tmp` on this
board is **tmpfs**. An `fflush` to tmpfs is a memcpy. In the old flow the CWD was
the SD card's FAT partition, where the same flush was a real filesystem sync.
The working-directory change deleted that cost too, silently.

**Both knobs are kept.** They are correct, they default to 1, they are proven
inert at that default, and they immediately become worth 15% again for anyone
driving a run over the serial console — which is still the fallback whenever ssh
is unavailable, and still the only transport whose timing is comparable to every
result recorded before 2026-08-25.

**The honest headline: `car1` runs at 24.43 ms/frame = 40.9 FPS**, the best frame
time on record — and it is NOT comparable to the 26.29 ms in the performance
history, which was measured over UART on the synthetic scene. On this evidence
the UART was costing 3.79 ms/frame at `VERBOSITY=0`, so like-for-like the two
numbers are much closer than they look.
