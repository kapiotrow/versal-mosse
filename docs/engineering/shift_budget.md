# Shift budget

Moved out of CLAUDE.md 2026-08-31; content unchanged.

### Shift budget — SETTLED: 4-4-4, `H_SHIFT` 14 (gray) / 15 (RGB)

**FFT budget closed 2026-08-24 on five 200-frame runs; `H_SHIFT` closed 2026-08-27 on real
video (see "Shift budget on real video").** The FFT budget never moved — every fix has been
`H_SHIFT`, the only knob upstream of **both** the accumulator and the response (`IFFT_*`
reaches only the response, `FFT_SHIFT` moves it two bits at once).

The invariant `2·FFT_SHIFT + IFFT_ROW_SHIFT + IFFT_COL_SHIFT` fixes the response scale, so
weight moves freely between passes (holds to 1.3% across splits). `FFT_SHIFT` stays 4 rather
than 5 because that leaves the accumulator at ~1400 instead of ~330 for the same response.
Retired points: 4-5-5 (total 16) undershot 6-11×; 5-3-4 (17) gave 0.4%; 4-2-2 (12) peaks at 56%
on frame 1, then rails from frame 15 and sign-flips to −32768, holding forever on
`NEGATIVE_PEAK`; 4-2-1 was never validated past frame 1. `IFFT_ROW_SHIFT=0` is unsafe at ch16.

**Do not re-centre the response in the 49-64% band.** That band came from a distribution with a
1.30× spread; the corrected build spreads 2.07× at the converged end, so centring the TYPICAL
frame puts the TAIL on the rail. Size against the tail. The response sits at ~28% (gray) / 22%
(RGB) and `calib_report.py` calls that UNDERSHOOT; advisory, and PSR is the arbiter.

**Four rules this budget cost real time to learn:**
1. **The response GROWS as the filter converges** — a budget validated at `ITER_CNT=2` is not
   validated. The 08-24 rail appeared at f173 and the 98% peak at f187; use the full 200.
2. **Twice an offline model set this budget and hardware overturned it.** Both times the model
   was self-consistent and its *premise* was wrong (see frame-buffer seeding under Correctness
   traps).
3. **Never size this budget against railing before checking `mean_prev` is seeded** — two budget
   hunts chased a frame-0 DC pedestal, not a scaling problem.
4. **Do not size from early frames.** RGB's response reads ~1.03× of gray at f1-4 and 0.785×
   once converged; the weights-derived estimate (0.685–0.790×) was right.

A calibration run's criterion is `rails=0` plus BIT-IDENTICAL tracking (a uniform rescale cannot
move an argmax) plus PSR not moving — PSR is where a quantization floor would show, and `F_ch` /
`H(q15)` must be digit-for-digit unchanged since both are upstream of `H_SHIFT`.

`runs/.last_cfg` is **stale and not authoritative**; `build/hw/.../aie.flagstamp` is, and
`scripts/calib_build.sh` checks it for you.


### Shift budget on real video — CLOSED 2026-08-27

`docs/thesis/evidence/TODO_shift_budget.md`. `car1` railed on 266 of 8434 frames at `H_SHIFT=11`, attributed
to BOTH `accum` and `response` — so `H_SHIFT`, the only knob upstream of both, was the lever. Two
deliberately over-shifted arms returned the UNCENSORED distribution: over 101,564 RGB frames
`rails_accum = rails_resp = 0`, maxima 15.6% / 10.1% of ceiling. **`H_SHIFT=13` is the
tight-but-safe RGB budget; 12 rails.** The shipped arms stay at gray 14 / RGB 15 because
`rails = 0` is the only hard criterion and the whole benchmark is banked on them. **Stated
confound: the benchmark's two arms are at different `H_SHIFT`.**

Three things not to re-derive:
- **`accum_max = 46340 = 141.4%` IS NOT OVERSHOOT** — it is 32767·√2, the largest magnitude a
  non-saturated cint16 bin can hold (the rail is per COMPONENT, `accum_max` is a magnitude).
  **`rails` is the only saturation instrument.**
- **Readings are CENSORED at the rail**, so no budget is derivable from a clipped maximum. That
  is why the over-shifted arm had to be built.
- Rails do NOT correlate with tracking loss (`corr = −0.025`) and do not explain the
  `NEGATIVE_PEAK` vetoes. A budget defect, never a tracking fix.

**`H_SHIFT` is the one knob that is NOT host-only** — it reaches `AIE_FLAGS`, so it needs a graph
rebuild, re-package and re-flash, and the sweep's xclbin guard will refuse until the card is
updated. **Re-provision after packaging**: `v++ --package` takes `build/rootfs/rootfs_compat.ext4`,
which `make rootfs` regenerates only when the pristine rootfs changes, so a re-package inherits
whatever that copy holds — it did not hold the ssh key until 2026-08-25, which produces a board
that boots unreachable and reads as a cable fault.

