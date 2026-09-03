# Methodology: the rules this project paid to learn

**Status:** current · **Updated:** 2026-08-31 · **Scope:** which of the paid-for rules generalise past this project

Chapter 8's material. The rules themselves live in
[`../engineering/measurement.md`](../engineering/measurement.md) and
[`../engineering/traps.md`](../engineering/traps.md) (they left CLAUDE.md on 2026-08-31); the
indexed, thesis-facing versions are the `M-*` rows of [`claims.md`](claims.md). This file is the
bridge: it says which of those rules generalise past this project and are worth a chapter.

Two principles everything below is an instance of: **instruments before changes**, and **never
move two magnitudes at once**.

## The four that would generalise to any hardware-accelerator project

1. **A measurement that can only confirm your hypothesis is worth less than one that can also
   kill it.** The `poll(state)`/`wait()` split cost one hardware run and retired a planned fix
   before it was built on. Hardware access is the scarce resource. (M-05, and the `503 ms`
   story under P-03.)
2. **Two independent instruments beat one instrument twice.** `/proc/interrupts` reading 0 is
   consistent with "no interrupt raised" *and* "raised but never delivered"; the CU's own
   toggle-on-write `ISR=0x3` discriminates them outright.
3. **A test that has never failed is not known to work.** Every RGB suite here is
   mutation-tested; the multi-start determinism test *passed* with a deliberately broken reset
   until a negative control was run against it (M-04).
4. **A self-consistent offline model can be overturned by its premise, and was — three times.**
   The shift budget twice, and the "~30 FPS" RGB prediction once. In each case the arithmetic
   was fine and an assumption about memory or about where time was spent was not (B-09).

## The metric traps, which are specific to tracking and worth a section of their own

- Mean IoU and the toolkit's AR **order two arms oppositely on the same trajectories** (M-03).
- An offline R can be raised by **degrading** the filter — demonstrated, not argued (M-01).
- Accuracy is averaged over tracked frames, so a longer-surviving arm is scored on harder ones;
  score A on the common survived prefix (M-02).
- PSR is a weak pass criterion in a specific direction: a tracker 179 px off target, locked to
  background, reported PSR 33.

**Suggested framing for the chapter:** these are not lab-notebook anecdotes. Each one changed a
decision, and three of them reversed a decision that had already been made. That is the argument
for the chapter existing.
