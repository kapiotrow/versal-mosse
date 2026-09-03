# `MOSSE_ETA=0.05` on hardware, all 62 — WINS ON AR, BUT NOT BY THE PREDICTED MECHANISM

**Status:** closed · **Updated:** 2026-08-27 · **Scope:** `MOSSE_ETA=0.05` on hardware: wins on AR, but not by the predicted mechanism

**2026-08-27.** `runs/vot/0827_1441-eta05`, 62 sequences, 419 runs, 180,125 frames.
Host-only A/B against the shipping RGB arm (`rgb15` + `rgb15stream`, same 419 runs):
one flag differs, `-DMOSSE_ETA=0.125` -> `0.05`, verified by `app.flagstamp` diff, and
the board's `a.xclbin` matched the package tree (`52235f49221e`) so no rebuild was involved.
Workspace `~/vot/analysis/eta05ab`.

## The result

| arm | accuracy | robustness | EAO | frames |
|---|---|---|---|---|
| `rgb_eta125` (shipping) | 0.5043 | 0.3065 | 0.1474 | 54,813 |
| **`rgb_eta05`** | **0.5100** | **0.3283** | **0.1600** | **62,436** |
| | +0.0057 | **+0.0218 (+7.1%)** | **+0.0126 (+8.5%)** | +13.9% |

**R moves 6x more than A and EAO follows R** — the one prediction that held, and the shape a
robustness change is supposed to have. 13.9% more frames survive.

**But per sequence it is nearly a coin flip: R better on 28, worse on 24, tied on 10; A better
on 28, worse on 28, tied on 6.** The pooled figures are frame-weighted, so long sequences carry
them. Largest swings both ways: `snake` +0.312, `butterfly` +0.228, `singer2` +0.202 against
`book` -0.332, `drone_across` -0.253, `monkey` -0.217. **This is NOT the uniform-in-sign result
the offline 8-sequence sweep showed** (6 better / 2 tied / 1 worse on mean IoU), and the
difference is not the metric alone — the offline set was 8 sequences of which `nature` was 46%.

## The mechanism prediction FAILED — two of three falsifiers fired

Written down in `robustness_gap.md` before the run, checked with
`scripts/vot_loss_anatomy.py`:

```
                                        eta=0.125     eta=0.05
mean hold rate                             23.70%       27.09%
vetoes: NEGATIVE_PEAK / LOW_PSR          29955/3912   29349/6533   <- LOW_PSR nearly DOUBLES
runs that never reach IoU<=0.1              6.68%        6.68%     <- identical
median first-loss position in run           0.143        0.128     <- EARLIER, not later
--- 5 frames before the first loss
  ACCEPT share                              82.0%        73.7%     <- PREDICTED TO STAY ~80%
  median PSR                                18.77        13.91
  median box motion                    1.85 px/fr    1.69 px/fr
```

- **"pre-loss ACCEPT share stays ~80%"** — it fell to 73.7%. The stated falsifier was that if
  this moved, something other than drift changed. It moved.
- **"first losses land later"** — the median lands marginally EARLIER, and the fraction of runs
  that never dip is identical to four digits.
- **"R moves more than A"** — held.

**What actually happened, as far as the instruments can say.** A slower filter is a weaker
filter: median PSR drops ~26%, so more frames fall under `PSR_GATE_MIN=7.0` and the hold rate
rises 3.4 points with `LOW_PSR` vetoes up 67%. First dips come no later. What improves is
**recovery**: 13.9% more frames are scored before the toolkit's 10-consecutive-frame rule
fires, i.e. dips that used to become terminal now do not. That is a plausible reading of a
slower filter — it is slower to lock onto the wrong thing after a bad frame — but it is a
*post hoc* reading, and the arm did not test it.

**Caveat on the instrument, stated rather than buried:** `vot_loss_anatomy.py`'s "first loss"
is a single frame at IoU <= 0.1, which is NOT the toolkit's failure rule (10 consecutive, after
a 10-frame burn-in, on bounded overlaps). A recomputation closer to the toolkit's rule gives
384 -> 376 runs failing and the median failure position unchanged (0.178 -> 0.179), which does
not reproduce the toolkit's +13.9% frames either. **Treat the anatomy as a mechanism probe, not
as a second scoring of the run;** `vot analysis` is the metric of record.

## Verdict

**Ship it, and do not build on the explanation.** The gain is real by the metric that will be
reported and it cost one host-only sweep. But two falsifiers fired, and this project's own
record says an arm accepted on a mechanism it did not test is how `HOLD_COAST` and the
frame-buffer seeding both went wrong. The honest statement is: *eta = 0.05 improves AR by 7%
on robustness; the drift model predicted that improvement and predicted the wrong reason for
it.*

**What this does NOT license:** treating drift as confirmed and moving to the spatial mask on
that basis. The mask is still the highest-priced item in the literature (CSR-DCF's >50% EAO
no-mask ablation), but it should be run as its own arm with its own falsifier, not as step 2
of a chain whose step 1 came back unattributed.

**Open, and cheap:** the pre-loss PSR fell to 13.91 against a gate at 7.00, and `LOW_PSR`
vetoes nearly doubled. `PSR_GATE_MIN` was calibrated for the eta=0.125 filter. An
`eta05 + PSR_GATE_MIN` re-tune is one more host-only sweep and is now a live question that
did not exist before this run.
