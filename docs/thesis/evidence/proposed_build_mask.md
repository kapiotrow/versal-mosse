# Proposed hardware build — the spatial mask (`FILTER_MASK`)

**SUPERSEDED 2026-08-31 — the arm was built, swept and ACCEPTED (EAO 0.1629 -> 0.1740,
+0.0110). The result, and everything that must not be re-derived, is in
`evidence/spatial_mask.md`. This file is kept as the PRE-REGISTRATION: its section 4 is the
falsifier written before the run, which is what makes the result readable. Read it as a record
of what was predicted, never as current status.**

**2026-08-28. Status: PROPOSED, not built.** Written so the decision to spend board time is
made against a falsifier and a cost, not against a hope. Evidence:
`docs/thesis/evidence/robustness_proposals.md` §2; trajectories in `runs/vot/0828_offline-mask/`.

## 1. Why this is worth board time — the BOARD-IMPLEMENTABLE variant is the one that wins

**The arm swept in `robustness_proposals.md` §2 (Tukey `mask50/70/35`) was a null, and it is
also NOT the arm this build proposes.** See §2: only a Hann-shaped mask has an exactly sparse
spectrum, so only it is implementable host-side without an approximation study. Swept
separately, all 62 sequences, same eta/gate:

```
arm                  A        R   tracked/19903   meanIoU        dA        dR
rgb             0.5394   0.2910            5792    0.1792
rgb-mask50      0.5413   0.3033            6037    0.1939   +0.0020   +0.0123
rgb-mask0       0.5048   0.3628            7221    0.2040   -0.0346   +0.0718
   (Hann-shaped, exactly 9 bins -- but see the WINDOW CONVENTION below)
```

**dR = +0.0718 — 3.6x the +0.02 bar and 3.6x the instrument's measured resolution.** It is the
first arm all day that is genuinely RESOLVABLE by this bench. It also **survives the symmetric
trim** (drop 3 sequences at each end: **+0.0480**), which nothing else this day did, and tracks
**24.7% more frames**. R better on 18 / worse on 9 / tied on 35.

**THE ARM ABOVE NEEDS `--mask-taper 1.0`, WHICH IS NOT THE DEFAULT.** `--mask-taper` defaults to
0.25, where `mask0` is a narrow raised cosine reaching zero at 12.5% of the patch: 99 non-zero
bins per axis, 47% of the energy in the 3 that survive truncation, and NOT board-implementable.
Confirmed 2026-08-29 by replaying `surfing` against the stored trajectories — taper 1.0
reproduces `mask62_hann9bin.json` with maxdiff 0.0, taper 0.25 gives mean IoU 0.2546 against
0.1087. A rerun that omits the flag scores a different arm and looks like a failed replication.

### THE WINDOW CONVENTION — measured 2026-08-29, and it moves the number

`spatial_mask()` centres its axis at `(n-1)/2`; the board's window is the periodic Hann already
in `hanning_128.h`, centred at `n/2`. Half a sample, `max|Δm| = 0.0123` — and NOT benign per
sequence: on `tiger` the bench window gives mean IoU 0.1715 (lost f107) and the exact Hann
0.2813 (lost f360). So the arm scored above is not the arm the board would run, and the whole
62 was re-swept in the BOARD form (`mask62_boardform.json`, baseline re-run in the same
invocation so both arms see identical frames):

```
arm                  A        R   tracked/19903   meanIoU        dA        dR
rgb             0.5394   0.2910            5792    0.1792                        <- control
rgb-maskbench   0.5048   0.3628            7221    0.2040   -0.0346   +0.0718    <- as recorded
rgb-mask0       0.5109   0.3512            6989    0.2016   -0.0284   +0.0601    <- BOARD FORM
```

| check | bench window | board window |
|---|---|---|
| symmetric trim (3 each end) | +0.0480 | **+0.0409** |
| drop top-3 gainers only | +0.0847 | +0.0711 |
| mean IoU | 0.1792 -> 0.2040 | 0.1792 -> **0.2016** (RISES) |
| dA on the COMMON survived prefix | −0.0084 (5563 fr) | **−0.0103** (5586 fr) |
| hold-rate change | +0.64% of frames | **+0.71%** |
| R better / worse / tied | 18 / 9 / 35 | **20 / 9 / 33** |

**The swap costs ~16% of the gain and the arm survives it**: +0.0601 is still 3.0x the
instrument's measured resolution, and every check that cleared the bench arm clears this one.
**The control is what makes it readable** — the baseline reproduces the recorded A 0.5394 /
R 0.2910 / 5792 tracked / 19,903 frames exactly, and the stored bench arm re-scores digit-for-
digit, so the window is the only thing that moved.

**Free finding, and it argues against a lucky draw:** the per-sequence composition SHUFFLES
between the two windows — `gymnastics1` (+0.536) and `motocross1` (−0.323) are top movers only
for the bench window, `monkey` (−0.297) only for the board one — while the aggregate barely
moves. Individual sequences are chaotically sensitive to a half-sample shift; the pooled result
is not.

**BUILD THE EXACT PERIODIC HANN.** It is the measured arm, its taps are REAL, and it is the
window the design already tabulates.

### The accuracy drop trips this project's own artifact rule — and three checks clear it

`dR > 0 with dA < −0.02` is the `gsign` mutant's signature and must be assumed to be the
failure-rule artifact until shown otherwise. Here `dA = −0.0346`. Shown otherwise, three ways:

1. **Mean IoU RISES** (0.1792 → 0.2040, the largest of any arm run this day). The mutant's FELL
   (0.1792 → 0.1683). That is the discriminator between "survives longer" and "degraded".
2. **It is not bought by freezing.** Hold rate moves by **+0.64% of frames**, far too little to
   account for +0.0718, and R-gain and R-loss sequences have the SAME median hold-rate change
   (0.0000). (`corr(dR, d hold-rate) = +0.183` — weak, and the opposite sign to `dec2`'s −0.198,
   so this check is supportive rather than conclusive on its own; the magnitude is what settles
   it.)
3. **MOST OF THE ACCURACY DROP IS A SELECTION EFFECT.** A is the mean overlap over TRACKED
   frames, so an arm surviving 24.7% longer is averaged over a HARDER set — the same structural
   point CLAUDE.md already makes about RGB. Scored on the frames BOTH arms survived (5563
   frames, identical set for both):

```
   A, common survived prefix     rgb 0.5539     rgb-mask0 0.5454     (-0.0085)
```

   **−0.0346 → −0.0085.** Three quarters of the apparent accuracy loss is the harder frames, not
   worse boxes.

So the arm's real profile is: **large robustness gain, 24.7% more frames survived, mean IoU up,
and a small genuine accuracy cost of about 0.009.** That is the shape a spatial-reliability
change is supposed to have.

### What still requires the board, and it is not a formality

- **EAO is not computable offline** and EAO is the arbiter for an A/R trade — that is the pad30
  lesson, where R rose +0.0077 and EAO FELL. This arm trades A for R, so it is exactly the case
  where EAO decides.
- **This is SINGLE-START; the toolkit is 419 ANCHORED runs.** That difference already produced
  one wrong prediction here (pad30: a big search region had hundreds of frames to earn back a
  drift that short anchored runs never gave it). A mask has a reason to be protocol-sensitive in
  the OPPOSITE direction — it constrains what the filter learns from frame 1, so its effect
  concentrates in the early life of a filter, and the anchored protocol is 419 early lives where
  single-start is 62 (`vot_init_anatomy.py`: 16% of losses land within 10 frames of an init).
  **Which way that cuts is unknown, and that is the point of running it.**
- **Median per-sequence dR is 0.0000** — 35 of 62 sequences are untouched. The gain is
  concentrated in 18, though unlike every other arm this day it survives a symmetric trim.

## 3. The build — IMPLEMENTED 2026-08-29, NOT YET SWEPT

**Host-only, and now verified so rather than argued: `aie.flagstamp` comes back byte-identical
across the FILTER_MASK=0 and FILTER_MASK=1 builds.** An scp, not a card swap.

```
FILTER_MASK=1                 # apply h <- m (.) h to H before publish_packed()
FILTER_MASK_STAT=1            # log the box-energy fraction -- the 4 falsifier
CONV_IN_CH=3  H_SHIFT=15  MOSSE_ETA=0.05  TARGET_PADDING=2.0  SCALE_STEP=1.04
SCALE_MAX_STEP=2  HOLD_COAST=0  FRAME_SOURCE=vot  VERBOSITY=0  DUMP_BUFFERS=0
```

`PSR_GATE_MIN` is the open one — see §5.

**`FILTER_MASK_TAPS` DOES NOT EXIST, and proposing it was a misreading.** There is no width or
tap-count to choose: only the periodic Hann has an exactly sparse spectrum, so the shape is
forced, and once it is forced the constants collapse. Writing the transform pair out:

```
DFT of a period-n raised cosine = {n/2, -n/4, -n/4} on bins {0,+1,-1}, REAL
separable, and every constant cancels against the pair's 1/(rows*cols):

    h <- m (.) h    ==    H <- D_row(D_col(H)) / 16,
                          D(X)[i] = 2X[i] - X[i-1] - X[i+1]   (CIRCULAR)
```

Verified against an exact FFT round-trip to **8e-16**. So the projection is **8 complex ADDS
per bin and one scaling — no multiplies at all**, against the 2.4 MMAC/frame §6 first costed it
at. The same {n/2, -n/4, -n/4} rule is already written down in `apply_dc_correction()`'s
subtract branch, which is what "reuse Stage B2's machinery" turns out to mean: the COEFFICIENT
RULE is shared, the code is not — B2 acts on the accumulated spectrum in the device's
TRANSPOSED layout, the mask acts on `H` in the host's row-major one.

**What was implemented** (`FILTER_MASK`, `FILTER_MASK_STAT` in the Makefile; `mosse_filter.{h,cpp}`;
`mosse_tracker.cpp`):
- `filter_mask_project()` — the projection above, in place, one channel at a time.
- **BOTH publish paths, and the second one is easy to miss.** `filter_update_quantize()` is the
  fused per-frame path; `filter_quantize_q15()` is **frame 0's**, taken after `filter_init()`.
  Under the anchored protocol that is 419 inits per arm, and `vot_init_anatomy.py` puts 16% of
  all losses within 10 frames of one — masking only the fused path would leave every anchor's
  first published filter unmasked, exactly where the mask is theorised to pay. Confirmed in the
  ELF: `bl filter_mask_project` appears in both symbols.
- **The projection runs BEFORE the max-|H| scan**, because it moves max|H| and max|H| sets the
  single global Q1.15 scale. Scanning first would scale the masked filter by a peak no longer
  in it.
- `filter_box_energy_fraction()` + the `mask_ebox` CSV column — see §4.

**Two traps hit while building it, both now guarded:**
- **An axis shorter than 3 bins must be SKIPPED, not wrapped.** At `rows == 1` the circular `D`
  reads `X[-1] == X[+1] == X[0]` and returns `2X - X - X = ZERO` — it would silently delete the
  filter. `FilterState` is also the DSST scale filter's type at `rows == 1`, so this is live.
- **The `#define` for a new host flag must precede every `#if` that reads it.** First cut put
  `FILTER_MASK_STAT`'s default *after* the CSV header block and *before* the row block, so the
  header would have omitted the column while the row still printed it — a silently misaligned
  CSV, which is the `parse_csv_frames` class of bug all over again.

### `cmp` CANNOT CHECK THIS KNOB'S INERTNESS, AND THAT IS WORTH KNOWING

`FILTER_MASK=0` is **not** byte-identical to the shipping arm, and expecting it to be
(PROGRESS_EVERY's rule) is a misreading: the two new functions have EXTERNAL LINKAGE, so they
are emitted uncalled and every later address relocates. The image grows 231688 -> 232384 bytes
with no executed instruction changed.

The check that does work, and was run: disassemble both, group by symbol, compare each
function's instruction stream with addresses and immediates normalised. **245 functions in both,
0 differing**; the additions are exactly `filter_mask_project`, `filter_box_energy_fraction` and
one PLT entry (`__cxa_thread_atexit`, for the `thread_local` scratch). Control first — the same
source built twice IS byte-identical, so the instrument is sound.

**Incidental, and it wants checking before the sweep:** the ELF sitting in `build/` on 2026-08-28
(`a661817379a1…`, the one the last sweep pushed) does **not** match a rebuild from HEAD — it
differs in `main` and `report_cint16` by more than relocation, while the build is reproducible.
So the pushed baseline ELF was not built from committed source. Rebuild the baseline from HEAD
before comparing arms, or the comparison has an unknown variable in it.

### Pre-flight — DONE

1. **`make test_host` covers the projection**, at every `FILTER_MASK` setting, because
   `filter_mask_project()` is compiled unconditionally and only its call sites are `#if`'d — a
   projection tested only in the arm that uses it is tested too late. The reference is an
   INDEPENDENT naive DFT written in the test file, not `mosse_filter.cpp`'s own transform, and
   the assertion is made in the SPATIAL domain where the claim lives: `ifft(project(H))` must
   equal `ifft(H) * m`. Max rel err **1.0e-07**.
2. **Six mutants, all caught** — because a passing test on a previously-uncovered path is worth
   nothing until it has been shown to fail:

   | mutant | caught by |
   |---|---|
   | constant 1/16 -> 1/8 | 4 checks |
   | sign flip on the row taps | 2 |
   | column wrap clamped, not circular | 3 |
   | row axis NOT skipped at rows < 3 | 2 |
   | energy box at the origin, not the centre | 1 |
   | energy box off-by-one extent | 1 |

3. **`make test_host FILTER_MASK=1` passes at BOTH `-O2` and `-ffp-contract=fast`**, including
   `fused H(q15) bitwise identical` — so the two publish paths still produce the same buffer with
   masking on, which is the property that keeps them from drifting. The one comparison that
   cannot hold there is `H_q15` against the NumPy golden (the golden is the unmasked filter); it
   prints SKIP rather than being silently dropped, and regenerating the golden against the mask
   was deliberately NOT done — that would make the golden and the implementation share the
   projection, which is the "self-consistent test on corrupted data" failure this project has
   already paid for once.
4. Still to do: `scripts/calib_build.sh`, then `scripts/vot_sweep.sh --arm mask --ingest`.

## 4. Falsifier, written BEFORE the run

- **Accept** only on `vot analysis`: EAO must rise. A/R may trade, but **EAO is the arbiter for
  an A/R trade** — that is the pad30 lesson, where R rose +0.0077 and EAO FELL.
- **dEAO ≥ +0.005** to ship. Below that it is inside the noise of a 419-run comparison and the
  arm is a null.
- **dR > 0 with dA < −0.02 is the failure-rule artifact, not a win** — demonstrated by the
  `gsign` mutant at dR +0.0525 / dA −0.0975. Require A not to fall, or price the fall. The board
  arm is expected at dA ≈ −0.03 pooled and ≈ −0.01 on the common survived prefix (§2); score the
  prefix before calling an accuracy drop real.

### The mechanism check now HAS an instrument, and a predicted value

`FILTER_MASK_STAT=1` logs `mask_ebox` — the fraction of `Σ|h|²` inside a centred box the size of
the target, per frame, as a trailing `track.csv` column and a `VERBOSITY>=1` line. The offline
half is `rgb_vs_gray_loop.py`'s `e_box` column. **Both were written against the same definition
and the offline one reproduces the numbers this file already quotes:**

```
                      AT INIT      f5      f20      f39
car1  rgb              0.5140  0.5832   0.6920   0.7408      <- doc says 51.6% at init
car1  rgb-mask0        0.9131  0.9151   0.9318   0.9377
tiger rgb              0.5498  0.6940   0.8431   0.8934      <- doc says 54.9% at init
tiger rgb-mask0        0.9128  0.9415   0.9731   0.9850
```

**THE 51.6% / 54.9% FIGURES ARE AT-INIT VALUES, AND THE FRACTION RISES AS THE FILTER
CONVERGES** — `car1` reaches 0.74 unmasked by frame 39, `tiger` 0.89. That was not stated
anywhere and it changes how this check must be read: **comparing a board run's MEAN `mask_ebox`
against 51.6% would "confirm" the mechanism on an unmasked arm.** Compare like with like —
at-init against at-init, or the two arms' means against each other.

So the check to apply: **`mask_ebox` should sit near 0.91 at init and 0.93-0.99 thereafter,
against a baseline that starts near 0.52-0.55 and climbs.** If EAO moves while `mask_ebox` does
not separate the arms, the gain is not the mask and the result is unattributable.

`mask_ebox` is **-1 on frames where H was not re-formed** — frame 0's `filter_init` path and
every held frame. Do not average that in as a zero; a reader that does is measuring the hold
rate.

## 5. The known coupling — budget a second sweep for it

**Masking moves the PSR scale**, measured offline on `car1`: mean PSR 48.2 / 42.8 / 38.3 / 31.0
at no mask / 70 / 50 / 35, and 36.7 for the Hann-shaped mask. `PSR_GATE_MIN`'s worth is
CONDITIONAL on the PSR scale — 7.0 → 5.0 was worth +0.0134 precisely because `MOSSE_ETA=0.05`
had moved PSR. So a mask arm at the inherited gate of 5.0 is a confounded arm.

Plan for it: run `FILTER_MASK=1` at gate 5.0 first (one variable), and if PSR moves as offline
predicts, run the gate re-tune as a SECOND host-only arm. Both are scp-only, so this is two
sweeps and no reflash. Do NOT move both at once — that is the rule this project has paid for
twice.

## 6. Cost

- Board: two ssh sweeps, 62 sequences each, ~25 min of run time per sweep plus ingest. No
  reflash, no re-package.
- Host frame time: 2.4 MMAC/frame against an APU that already spends ~5.2 ms in
  `filter_update_quantize`. Expect well under 1 ms; measure it from the `AP_*` slots rather than
  predicting it, and remember frame time must be quoted from a serial-console run, not from ssh.
- Risk: LOW. Host-only, revertible by a flag, and it cannot touch the flashed xclbin.

## 7. If it is a null

Then two independent cheap spatial-reliability stand-ins have failed on hardware (padding, and
this), and the conclusion for the write-up is specific and defensible: **the value in CSR-DCF is
the per-frame ESTIMATION of the mask, not the masking** — their own ablation prices a uniform box
mask at −21% EAO *relative to* their estimated one. That is a result worth reporting, and it
retires the item rather than leaving it as an untested "we could have tried a mask".
