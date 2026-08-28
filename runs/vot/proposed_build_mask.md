# Proposed hardware build — the spatial mask (`FILTER_MASK`)

**2026-08-28. Status: PROPOSED, not built.** Written so the decision to spend board time is
made against a falsifier and a cost, not against a hope. Evidence:
`runs/vot/robustness_proposals.md` §2; trajectories in `runs/vot/0828_offline-mask/`.

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
   (Hann-shaped, exactly 9 bins, EXACT on the board)
```

**dR = +0.0718 — 3.6x the +0.02 bar and 3.6x the instrument's measured resolution.** It is the
first arm all day that is genuinely RESOLVABLE by this bench. It also **survives the symmetric
trim** (drop 3 sequences at each end: **+0.0480**), which nothing else this day did, and tracks
**24.7% more frames**. R better on 18 / worse on 9 / tied on 35.

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

## 3. The build

**Host-only. No AIE change, no re-synthesis, no re-package, no reflash — an scp, not a card
swap.** `aie.flagstamp` must come back byte-identical and the xclbin guard must pass; if either
moves, something is wrong with the build, not with the tracker.

```
FILTER_MASK=1                 # new, host-only: apply h <- m*h to H before publish_packed()
FILTER_MASK_TAPS=9            # 3 bins per axis, the exact Hann-shaped case
CONV_IN_CH=3  H_SHIFT=15  MOSSE_ETA=0.05  TARGET_PADDING=2.0  SCALE_STEP=1.04
SCALE_MAX_STEP=2  HOLD_COAST=0  FRAME_SOURCE=vot  VERBOSITY=0  DUMP_BUFFERS=0
```

`PSR_GATE_MIN` is the open one — see §5.

Implementation notes:
- The mask is FIXED in patch coordinates and constant for the whole run, so its 9 coefficients
  are computed ONCE at startup, not per frame.
- It applies to `H` at publish time, so detection and the filter state stay consistent and
  nothing downstream of `cmul_accum` changes.
- **The mask is centred at the PATCH CENTRE, not at the response origin.** Measured: the peak of
  `Σ|h|²` is at (64,64) and a corner-wrapped 64×64 box holds only 8-12% of it. Getting this
  backwards deletes the filter and reads as "masking hurts".
- Reuse Stage B2's existing sparse-window machinery; do not write a second copy of the rule.

Pre-flight, before any board time:
1. `make test_host` — add a case asserting the 9-tap sparse convolution matches an exact FFT
   projection to a stated tolerance on a real `H`. **This is the one new thing that can be
   silently wrong**, and it is testable natively in seconds.
2. Confirm a full-width mask is the EXACT identity through the board path, the same control the
   offline bench uses.
3. `scripts/calib_build.sh` to verify the flagstamps, then `scripts/vot_sweep.sh --ingest`.

## 4. Falsifier, written BEFORE the run

- **Accept** only on `vot analysis`: EAO must rise. A/R may trade, but **EAO is the arbiter for
  an A/R trade** — that is the pad30 lesson, where R rose +0.0077 and EAO FELL.
- **dEAO ≥ +0.005** to ship. Below that it is inside the noise of a 419-run comparison and the
  arm is a null.
- **dR > 0 with dA < −0.02 is the failure-rule artifact, not a win** — demonstrated by the
  `gsign` mutant at dR +0.0525 / dA −0.0975. Require A not to fall, or price the fall.
- **The specific mechanism check:** the fraction of the filter's energy inside the target box
  should rise from the measured 51.6% (`car1`) / 54.9% (`tiger`). If EAO moves while that does
  not, the gain is not the mask and the result is unattributable.

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
