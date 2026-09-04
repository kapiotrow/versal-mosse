# Danilowicz & Kryjak's deepDCF, run on this project's benchmark

**Status:** current · **Updated:** 2026-09-04 · **Scope:** their published tracker under this
project's own scoring path, and the VOT2015 known-answer check that has to pass first

**2026-09-04.** `scripts/offline_multistart.py --tracker deepdcf:<preset>`, against a pinned
clone of `github.com/mdanilow/MOSSE_fpga` @ `ee0f93ab183d8b2f712de039f1ec6d4776847fb2`
(branch `deep_features`, MIT). **Their code is UNMODIFIED** — `scripts/patches/` is empty and
must stay that way; the whole value of this comparison is that the algorithm is theirs.
Claim id: `R-15` in `docs/thesis/claims.md`.

## Why this exists

`M-17` says their tracking numbers cannot be compared to this project's: VOT2015 supervised
against STb2022 multistart, an inverted `R`, a `[108, 371]` EAO window against `[115, 755]`
(worth **+0.0827** on the shipping arm, 1.39x this project's whole arm ladder), and rotated
polygon ground truth against mask-fitted boxes. **None of that has to be reconciled if their
tracker is run HERE instead.** Same 62 sequences, same 419 anchored runs, same
`vot_ingest.py` -> `vot analysis` path that `R-12` validated against published CSRDCF.

**Bring their tracker to our benchmark, never the reverse.** The reverse needs a VOT2015
conversion AND the supervised/reset protocol this harness does not implement — a second
unvalidated scoring path, to answer a question the first one already answers.

## The prediction, written down first

Written **before** the VOT2015 run, 2026-09-04.

**What is being checked.** That this project drives their tracker correctly. A port that is
subtly wrong produces a plausible-looking trajectory and an uninterpretable score, and this
project already owns the cautionary case: `opencv-kcf` lands at R 0.280 against a published
0.532, and the ONLY reason that reads as a cv2 implementation difference rather than a harness
defect is that CSRT validated the identical path on the identical runs (`R-12`).

**THE REFERENCE IS INEXACT, AND THIS IS THE LIMIT OF THE CHECK.** Every implementation row in
their Table 1 is **4-bit quantised**, and their 4-bit path loads a trained Brevitas checkpoint
(`savegame_0_15000.pth.tar`) that is referenced by an absolute path on the author's machine and
is **not in the repo**. The float path we can run builds torchvision `vgg11.features[:3]`
instead — a *differently trained bank*, not merely a different precision. So the float arms
reproduce **no row of their table exactly**, and this can only ever be a BAND check. Their own
only float row is the ORIGINAL deepDCF (Danelljan ICCVW15, 96 channels): A 0.48, R 1.75, no EAO.

Reference rows, their Table 1, VOT2015 supervised (`R` is failures/sequence, LOWER better):

| their arm (all 4-bit) | A | R | EAO |
|---|---|---|---|
| deepDCF 5 scales, 224 ROI -> 112 map, 32ch — `best` | 0.505 | 1.829 | **0.207** |
| deepDCF 3 scales, 128 ROI -> 64 map, 32ch — `hw32` | 0.494 | 1.92 | **0.184** |
| deepDCF 3 scales, 128 ROI -> 64 map, 8ch (THE FPGA ARM) | 0.491 | 2.082 | 0.183 |

**Two independent predictions, and BOTH must hold.**

1. **MAGNITUDE, a wide band: `best` lands at EAO in [0.15, 0.26] on `vot2015/rgb` supervised.**
   The band is deliberately wide because the bank is not theirs. It is wide enough to be a weak
   test of the ALGORITHM and narrow enough to be a decisive test of the INTEGRATION, which is
   what it is for: a broken port does not land at 0.19, it lands near the `static` floor
   (0.0781 on STb2022, `results/reference_trackers.csv`).
2. **ORDERING: `best` beats `hw32` on EAO**, reproducing the sign of their +0.023. This is an
   internal, paired comparison between two arms sharing one bank, one protocol and one dataset,
   so it is immune to the float/4-bit difference that makes prediction 1 weak.

**The falsifiers, by name:**

- **EAO < 0.12** — the port is broken, or float vgg11 is much worse than their trained 4-bit
  bank. Either way **no STb2022 number from this backend is quotable until it is explained.**
- **EAO > 0.26** — we are not running what they ran. Suspect the stack, the window or the
  protocol before believing a gain.
- **ORDERING INVERTS** — their geometry claim does not survive re-running in their own code on
  their own dataset. That is a finding, but a configuration error looks identical, so it must be
  chased before it is reported.

**A THIRD ARM RIDES ALONG, and it is not part of the known-answer check.** `hw32w` is `hw32`
with sigma rescaled 7 -> 1.75. Their response is `exp(-r^2 / (2*sigma))` with `r` in
FEATURE-MAP BINS and the map at `ROI_SIZE/stride`, while `sigma` is a single global config value
with **no geometry term** — so their Table 1 comparison of 224/112 against 128/64 moves the map
AND THE MAINLOBE WIDTH together. That is precisely the confound `R-11` caught in this project's
own 64x64 arm, where the gain turned out to be the width the arm carried by accident and the
resolution term was a null (`R-14`). Sigma is a VARIANCE in this parameterisation
(std = sqrt(sigma)), so holding std/target across a 2x map change scales sigma by 4.
**Their ordering is the last external support for a 128x128 Layer-1 arm** (`CLAUDE.md`, "What to
try next"), so `hw32` vs `hw32w` runs the `R-11` experiment on THEIR tracker. No prediction is
registered for it: it is a question, not a check.

## The result

*(pending — the VOT2015 run is what this section is waiting for)*

| arm | protocol | A | R | EAO |
|---|---|---|---|---|
| | | | | |

## Did the mechanism hold?

*(pending)*

## Controls

- **`oracle` on the STb2022 path returns R = 1.0000 exactly** over all 419 runs, and CSRDCF
  reproduces its published row to −0.0078 EAO (`R-12`, `evidence/harness_validation.md`). The
  path this backend feeds is validated; what the VOT2015 check adds is the BACKEND.
- **`opencv-kcf` is the standing negative example**: a reimplementation that is not the
  published tracker, landing 0.25 R below its own row on a validated path. It is why an inexact
  reference is stated as a band and not as a reproduction.
- **`opencv-kcf-rgb`** proves the RGB/BGR flip is not free to get wrong. Their tracker is driven
  by `cv2.imread` in their own integration, i.e. **BGR**; the toolkit hands out RGB, so
  `offline_multistart`'s backend flips and the VOT2015 integration script does not.

## What not to re-derive

- **Their repo ships a top-level `vot.py`** — the TraX integration stub — which SHADOWS the
  installed `vot` toolkit package the moment their directory is on `sys.path`. The symptom is
  `ModuleNotFoundError: No module named 'vot.dataset'; 'vot' is not a package` raised from
  `plan_sequence`, i.e. this harness losing the toolkit underneath itself. Their root is now on
  the path only for the duration of the import, with an assertion that `deep_mosse` and `utils`
  resolved to the pinned checkout. In the TRACKER process the shadowing is wanted — that stub is
  the TraX binding.
- **CUDA cannot be forked** (`Cannot re-initialize CUDA in forked subprocess`). The pool uses
  `spawn` for this backend only; every other backend keeps the cheaper `fork`.
- **Vivado puts its own `git 2.50.0` first on `PATH` and that build has no https remote helper**
  (`git: 'remote-https' is not a git command`). Use `/usr/bin/git`. A new instance of the
  Vitis-masking trap that `docs/engineering/traps.md` records for Python.
- **`torchvision` still accepts `pretrained=True`** through `**kwargs` (deprecation warning
  only), so no patch is needed for the removed argument. `brevitas` installs without moving
  torch, so their module-scope FINN imports resolve untouched.
- **Their `configs/config.json` IS the `best` preset** — 224 ROI, 5 scales, 32 channels — i.e.
  the config for Table 1's 0.207 row, except for the quantisation.
