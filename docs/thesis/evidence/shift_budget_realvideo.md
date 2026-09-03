# Re-deriving the shift budget for REAL VIDEO

**Status:** closed · **Updated:** 2026-08-26 · **Scope:** the shift budget re-derived on real video: the instrument gap, the censoring, and why `H_SHIFT` is the only lever

**WHERE THIS ENDED UP.** The budget is SETTLED — 4-4-4 at 128x128 (closed on hardware
2026-08-24/27) and 3-3-3 at 64x64 (2026-09-02), with `H_SHIFT=15`; the current statement is
`../../engineering/shift_budget.md`. This file is how it was reached.
**Status at the time of writing: OPEN. Instrument gap CLOSED 2026-08-26 (host-only); the
budget change itself was still the next hardware task, and the LEVER CHANGED — read the
2026-08-26 section before the prediction below, which no longer stands.** Written down
because it has now been deferred three times: the risk register predicted it
before Phase 2, Phase 2 measured 8 railed frames and moved on, and the Phase 4
control session measured 266 and moved on again.

## The evidence

`car1`, 15 anchors, 8434 frames, `runs/vot/0825_1919-smoke/track_car1.csv`
(gray, 4-4-4, `H_SHIFT=11`, `BIAS_SCALE=roi` — the shipping build):

```
railed frames            266 of 8434   (3.2%)
total railed bins       2385
accum_max                46340 = 141.4% of the cint16 rail (32767)
accum_max, median         11691 on ACCEPTed frames
```

Per anchor the rails are lumpy — job 3 alone has 139 of the 266 — and **they do
not visibly cost tracking**: `corr(rail rate, mean IoU)` over the 15 jobs is
**−0.025**. They also do NOT explain the 738 `NEGATIVE_PEAK` vetoes: only 7% of
railed frames are negative-peak and only 3% of negative-peak frames railed, and
the accumulator amplitude is the same on both (median 12598 vs 11691). **Two
independent problems; this TODO is only about the rails.**

So the honest framing: this is a **budget defect with no measured tracking cost
yet**, and it is worth fixing because a saturating accumulator invalidates every
amplitude number the run produces, not because it is known to be losing frames.

## The lever, and why it is the only one

`H_SHIFT` 11 → 12. It is the only knob upstream of **both** the accumulator and
the response: `IFFT_*` reaches only the response, and `FFT_SHIFT` moves the
response two bits at once. This is exactly the reasoning that closed the
2026-08-24 calibration, where the fix was `H_SHIFT` 10 → 11 and the FFT budget
never moved.

## WRITE THIS PREDICTION DOWN BEFORE THE RUN

One bit halves the accumulator: 46340 → ~23170 = **71% of the rail**, so
**`rails` should go to 0** with margin. The response halves with it.

**The acceptance criterion is NOT the state digest.** A uniform rescale changes
every value in the response buffer, so the digest MUST differ — using it here
would report a false failure. What must hold instead:

1. **`rails = 0`** across all 15 anchors.
2. **The trajectory is bit-identical on every frame that did not rail.** A
   uniform rescale cannot move an argmax; only the 266 saturating frames may
   legitimately change.
3. **PSR does not move.** It is scale-invariant, so a drop means a quantization
   floor, which is where the risk of this change actually lives.

If PSR falls, the budget is now undershooting and `IFFT_*` must give a bit back.

## 2026-08-26 — THE INSTRUMENT GAP IS CLOSED, AND IT MOVED THE DIAGNOSIS

Three things came out of closing it. All three were derived from data already on
disk; none needed the board. **The framing at the top of this file — accumulator
at 141% of the rail — does not survive them, and is kept above as written so the
correction is legible.**

### 1. `accum_max = 46340 = 141.4% of the rail` IS NOT OVERSHOOT. It is √2.

`32767 × √2 = 46339.5`. `accum_max` is a complex MAGNITUDE (`sqrt(re²+im²)`)
while the rail is per COMPONENT, so 141.4% is exactly the largest magnitude a
non-saturated cint16 bin can hold, and a bin at 45° reaches it with neither part
railed. CLAUDE.md already says this in the H-quantization entry ("a bin at 45°
puts 32767/√2 in each part"); it was not carried across to reading `accum_max`.

Measured on the smoke CSV: **clean (`rails=0`) frames reach `accum_max` 42941 =
131% of 32767**, entirely legitimately. So a magnitude above 100% is not evidence
of anything. `rails` is the only saturation instrument, and it always was.

### 2. THE BUFFER THAT RAILS IS THE RESPONSE, NOT THE ACCUMULATOR.

Two independent reads, both from existing data:

| source | evidence |
|---|---|
| `runs/run_0825_1314.log` (car1 anchor 0, `VERBOSITY=1`, per-buffer console) | **8 response-railed frames vs 4 accum** |
| `0825_1919-smoke/track_car1.csv`, 8434 rows | **191 frames have `\|peak\| == 32767` exactly**, and all 191 are inside the 266-frame railed set ⇒ **72% of railed frames have a saturated response** |

The remaining 75 railed frames rail somewhere other than the response peak
(accum, F_ch, H, or a response bin saturating in the imaginary part, which
`peak` cannot see — see §4).

**This changes the lever.** `H_SHIFT` is the only knob upstream of BOTH, which is
why this file chose it — but that reasoning wanted the accumulator to be the
problem. If the response is, then `IFFT_COL_SHIFT` 4 → 5 buys the same one bit on
the response and **leaves the accumulator's headroom alone**, where `H_SHIFT`
would halve an accumulator that is already comfortable (clean median 35.8% of the
magnitude ceiling) and spend precision for nothing. Same rebuild cost either way.

### 3. THE READINGS ARE CENSORED, SO THE PREDICTION BELOW IS NOT DERIVABLE.

Once a bin saturates, the recorded value is the rail, not the true value. 266
frames — the entire top of the amplitude distribution — are censored, so
"46340 → ~23170 = 71% of the rail" halves a **clipped** number and concludes
nothing about where the true value was. It could be 1.1× or 10× the ceiling; the
data cannot say, and a lognormal fit to the surviving 96.8% puts the true tail
anywhere from 2× to 5× the ceiling after one bit (unreliable — it is fit to
truncated data, and quoted here only to show the sign of the uncertainty).

**So one bit may not be enough, and the current evidence cannot rule that in or
out.** The uncensored measurement is a deliberately OVER-shifted run: one arm at
`IFFT_COL_SHIFT=6` (or `H_SHIFT=13`) where nothing can rail, which returns the
true distribution, from which the correct shift follows by arithmetic instead of
by another guess. That is the difference between this being the last budget run
and being the fourth.

### 4. `peak` really was a good proxy for `resp_max` — and is now recorded anyway.

The header comment claimed "`response` is deliberately absent: `peak` above
already is it." **Tested rather than trusted:** identical on **199 of 199**
synthetic frames (`run_0824_1354`) and **739 of 741** car1 frames
(`run_0825_1314`), the two exceptions at amplitude ~25 of 32767. So the claim
held. It is still not the same quantity — `peak` is the SIGNED REAL PART at the
argmax of `|real|`, so it is blind to a bin saturating in the imaginary part
alone — and `resp_max` is now a column, at the cost of one `%.0f`.

### What was changed

- **`mosse_tracker.cpp`** (HOST-ONLY — no graph rebuild, no re-package, no
  re-flash; `vot_sweep.sh` pushes the ELF): `FrameDiag` gains `rails_fch`,
  `rails_accum`, `rails_resp`, `rails_h`; `track.csv` gains
  `resp_max,rails_fch,rails_accum,rails_resp,rails_h` as trailing columns. The
  total `rails` still accumulates for every tag including unnamed ones, so the
  per-buffer columns are an attribution of the total and never a replacement —
  if they stop summing to it, that gap is itself the finding.
- **`scripts/calib_report.py`**: reads `resp_max` when present and falls back to
  `peak` while saying which; reports rails BY BUFFER; labels a pre-2026-08-26
  CSV `(unattributed)` rather than guessing a buffer.
- **`scripts/calib_report.py` — A REAL BUG, and it was in the rails gate
  itself.** `parse_csv_frames` keyed by frame index, so on a multi-start CSV the
  last of 15 anchors overwrote the other fourteen: 8434 rows collapsed to 742 and
  **266 railed frames were reported as 4**. The gate under-reported by 66× while
  printing a confident "BUDGET IS WRONG" that was right for the wrong reason. Now
  keyed by `(job, frame)`; degenerates to the old behaviour at
  `FRAME_SOURCE=synth`, where there is no `job` column.
  Validated by reproducing a known answer: on `run_0824_1354` it still prints
  `rails=0`, accum max 52.1%, response max 49.0% — the three figures CLAUDE.md
  records for that run.

## 2026-08-26, MEASURED — `runs/vot/0826_1232-attrib`. STEP 1 IS DONE.

The instrumented ELF was pushed (host-only, no reflash) and `car1` re-run on the
UNCHANGED budget, `--vot-jobs all`, 8434 frames, 15 anchors.

**The run is provably a pure re-measurement.** All 15 run-state digests are
identical to the smoke run's, and 0 of 8434 rows differ on any column the two
CSVs share. So the instrumentation changed nothing, and the attribution below
describes the same run the 266 rails were originally found in — not a similar
one.

### The attribution

```
railed frames 266   total railed bins 2385
attribution sums to the frame total on all 8434 rows: True

buffer      bins    % bins   frames   % of railed frames
response    1330     55.8%      191                71.8%
accum       1055     44.2%      123                46.2%
F_ch           0      0.0%        0                 0.0%
H(q15)         0      0.0%        0                 0.0%

response only 143    accum only 75    BOTH 48
```

### THIS OVERTURNS §2 ABOVE, AND RESTORES `H_SHIFT` AS THE LEVER

§2 inferred "the response rails, not the accumulator" from the only attributed
sample that existed — 12 railed frames in one anchor's console — and proposed
`IFFT_COL_SHIFT` on the strength of it. **With all 15 anchors attributed, the
accumulator rails on 123 frames, and on 75 of them the response is fine.** An
`IFFT_*` fix reaches only the response and would have left those 75 railing.

So the original choice of `H_SHIFT` — the one knob upstream of both — was right,
and it was right for a reason this file had not yet established. §2 is left
standing above as written: it is a clean instance of this project's own rule
that **an ordering inferred from one failing run is a hypothesis, not a
finding**, and it cost nothing precisely because the cheap host-only run was
done before the expensive reflash rather than after it.

**Independent confirmation that the instrument is reading what it claims:** the
191 response-railed frames are exactly the 191 frames with `|peak| == 32767`,
predicted from the old CSV before this run.

### `resp_max` changed no conclusion, and that is the honest result

`resp_max` differs from `|peak|` on **1 of 8434 rows**, and that row is not
budget-relevant (amplitude well under 10000). The header comment's claim was
correct. The column is worth keeping because it retires an assumption rather
than because it revealed anything — `peak` is blind to an imaginary-part rail by
construction, and now nothing depends on that never happening.

### What is still censored

Per-buffer clean (`rails=0`) distributions, against the 46340 magnitude ceiling:

| | median | p99 | p100 | p100 after 1 bit |
|---|---|---|---|---|
| accum | 25.5% | 65.7% | 92.7% | 46.3% |
| response | 16.8% | 64.4% | 70.4% | 35.2% |

These are the SURVIVING values only. The 266 censored frames are still the top of
the distribution and still unmeasured, so **§3 stands: one bit cannot be shown
sufficient from this run either.** What this run does add is that the required
headroom must be sized against BOTH buffers, and the accumulator is the tighter
of the two (92.7% vs 70.4% at the clean maximum).

### The re-sequenced plan

The instrument change is **host-only**, and the budget change is not. So do not
pay for one rebuild-and-reflash to learn which knob to turn and a second to turn
it:

1. ~~**Push the instrumented ELF and re-run `car1` on the CURRENT budget.**~~
   **DONE 2026-08-26** — `runs/vot/0826_1232-attrib`. Verdict: `H_SHIFT`, not
   `IFFT_*`, because both buffers rail and only `H_SHIFT` reaches both.
2. **One deliberately OVER-shifted arm, for the uncensored distribution.**
   `H_SHIFT=14` (three bits). This is the first arm that needs the graph
   rebuild, re-package and re-flash, and the re-provisioning check below.
   Three bits rather than two because the point of this arm is that it CANNOT
   rail: at two bits it still can if the true tail exceeds 4x the ceiling, and
   a censored over-shift arm answers nothing and costs the same reflash.
   Undershoot is acceptable here and quantization loss is irrelevant — this arm
   is measuring a MAXIMUM, not tracking quality, and a uniform rescale moves the
   maximum exactly, so the true distribution at any shift follows by scaling.
   Expect `calib_report.py` to shout UNDERSHOOT; that is the intended state.
3. **Then the real budget**, computed from step 2's measured tail rather than
   predicted from a clipped one — sized against the ACCUMULATOR, which step 1
   showed is the tighter of the two buffers.

Step 1 is done and cost no reflash. Step 2 is the measurement this file has been
trying to skip for three sessions.

## Instrument gap to close FIRST (small, host-only) — CLOSED, see above

**`track.csv` does not carry the response maximum.** `g_fdiag.resp` is computed
by the same scan that fills `accum_max`, and then dropped —
`csv_row()` writes `rails, accum_max, fch0_max, h_max` and not `resp`. So at
`VERBOSITY=0` the response half of the budget is invisible, which is precisely
the trade CLAUDE.md claims was retired on 2026-08-24 when the other three were
added.

Check whether the existing `peak` column already suffices (it is the response
value at the argmax of |real|, which may or may not be the same as the scan's
maximum) and if not, add `resp_max`. Do this before the run, not after: sizing
the response against the tail needs the tail, and a second hardware run to get
one number is the outcome this project keeps paying for.

## Cost

The RUN is ~4 minutes — `scripts/vot_sweep.sh --arm hshift12 --seqs car1` drives
all 15 anchors in 3.5 min. **The BUILD is the cost.** `H_SHIFT` is NOT host-only:
it reaches `AIE_FLAGS` as `H_SHIFT` and `GCC_FLAGS` as `CMUL_H_SHIFT`, so this
needs a graph rebuild, a re-package and a **re-flash** — the only arm since the
automation landed that does. The sweep's build-agreement guard will refuse to run
until the card carries the new `a.xclbin`, which is the correct behaviour and
will look like a failure if it is not expected.

Verify the build with `scripts/calib_build.sh`, which checks the flagstamps
against the intended config rather than trusting `runs/.last_cfg`.

**Re-provision after packaging.** A fresh `v++ --package` produces an image from
`build/rootfs/rootfs_compat.ext4`, and `make rootfs` only regenerates that copy
when the pristine 2025.2 rootfs changes — so a re-package inherits whatever that
file already holds. It did NOT hold the ssh key until 2026-08-25 (the key had
been written into `sd_card.img` directly, downstream of it), which would have
produced a board that boots unreachable and looks like a cable fault. That is now
fixed at the source, but check it rather than assume:

```bash
debugfs -R "cat /root/.ssh/authorized_keys" build/rootfs/rootfs_compat.ext4
make board_provision ROOTFS_IMG=build/hw/128x128/ch16/package/sd_card.img   # belt and braces
```

## Rules from the last calibration that still apply

- **Do not adjust by a ratio.** Phase 2 already found the VOT distribution
  differs in SHAPE, not just scale: `F_ch` ran 2500-6900 against ~7400 on the
  synthetic arm while the response still railed.
- **Do not size from early frames.** The response GROWS as the filter converges;
  the 2026-08-24 rail appeared at frame 173 and peaked at frame 187.
- **Do not re-centre the response in the 49-64% band.** That band came from a
  distribution with a 1.30x spread; the corrected build spreads 2.07x, so
  centring the typical frame puts the tail on the rail.
- **`calib_report.py` will say UNDERSHOOT and it is advisory. PSR is the
  arbiter.**

## Then

Re-run the same `car1` sweep on the new budget, confirm the three criteria, and
only then let the 62-sequence Phase 5 run inherit it — results attributed to a
build with a saturating accumulator are not results.
