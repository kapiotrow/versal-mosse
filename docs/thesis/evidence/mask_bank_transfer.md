# The spatial mask does NOT transfer across the feature bank — it INVERTS on the shipping Layer-1 arm

**Status:** closed · **Updated:** 2026-09-03 · **Scope:** the spatial mask and Stage B3 channel reliability re-screened on the SHIPPING Layer-1 bank; both refuted, and the mask's sign flips

**2026-09-03.** `runs/vot/0903_offline-l1mask/`, `scripts/rgb_vs_gray_loop.py`, 62 sequences /
19,903 frames, shipping `--eta 0.05 --psr-min 5.0`, scored with `vot_ar_offline.py` and
`scripts/grid_stats.py`. **Offline only — no board time was spent, and that is the point of the
note.** Claim ids: `N-21` (the mask), `M-14` (the methodological result); `N-20` is re-confirmed.

Every arm here rides the SHIPPING feature bank — `rgb-l1relu`, i.e. resnet18 conv1 7x7 stride 2
PCA'd to 32 channels with the rectifier on, at a 64x64 map, which is what `a.xclbin 9cea47ce`
runs. Every previously recorded mask number was measured on the OLD bank (3x3 mobilenet, 16
channels, 128x128 map, sigma 2). Nothing but the bank and the geometry differs between the two
screens: same script, same 62 sequences, same eta, same gate, same `--mask-taper 1.0`, same
`--mask-center board`.

## The prediction, written down first

From the 2026-09-03 build proposal, before the sweep ran. The mask was the PROPOSED hardware arm
for that day, on three grounds: it is the only host-only knob with a measured hardware EAO gain
(+0.0110), its mechanism was independently confirmed by `mask_ebox`, and its hardware effect was
on the TAIL (mean time-to-first-loss 119.4 -> 141.4, +18%) — the 301-755 sub-window that owns 71%
of the EAO window and that `l1relu` did not move (`arm_l1relu.md` sec.10.3).

**THE TRANSFER ASSUMPTION, NAMED BEFORE THE RUN — and it is the one that broke:**

> *that the mask's value is independent of the feature bank. It is not obviously so — a rectified
> 7x7 Layer-1 map is more spatially concentrated than a signed 3x3 one, so the filter may already
> self-concentrate and leave the mask less to remove. `mask_ebox` on the BASELINE arm answers this
> directly: if the l1relu baseline already sits well above the old 0.6049, expect a smaller gain.*

Falsifier for the mechanism: **if `e_box` fails to separate the arms, the screen is broken and
says nothing about the mask.** That is the check that makes a negative readable rather than
merely disappointing.

## The result

`vot_ar_offline.py` (pooled) and `grid_stats.py` (paired per-sequence, 62 sequences):

| arm | A | R | tracked / 19903 | dR mean | trim3 | trim5 | b/w/t | sign p | P(dR<=0) |
|---|---|---|---|---|---|---|---|---|---|
| `rgb-l1relu` (control) | 0.5077 | 0.4364 | 8686 | — | — | — | — | — | — |
| `rgb-l1relu-mask0` | 0.4998 | **0.4269** | 8497 | **−0.0127** | −0.0358 | −0.0441 | 14/13/35 | 1.000 | **0.706** |
| `rgb-l1relu-mask0k2` | 0.5487 | **0.4175** | 8309 | −0.0066 | −0.0480 | −0.0644 | 18/17/27 | 1.000 | 0.576 |
| `rgb-l1relu-chrel05` | 0.5220 | 0.4356 | 8669 | −0.0040 | −0.0378 | −0.0429 | 7/16/39 | 0.093 | 0.563 |

**The mask is negative on every column** — pooled −0.0095, paired −0.0127, both trims, bootstrap
at P(dR<=0) = 0.71 — with 35 of 62 sequences EXACTLY TIED. On the old bank the same instrument
read **+0.0601**.

| bank the mask was screened on | offline dR | what happened next |
|---|---|---|
| 3x3 mobilenet, 16ch, 128x128, sigma 2 (2026-08-29) | **+0.0601** | swept on hardware, delivered +0.0192 R / +0.0110 EAO |
| 7x7/2 resnet18-PCA, 32ch, 64x64, sigma 2 (2026-09-03) | **−0.0127** | no board time |

**This bench has over-called the mask 3x and inverted on `MOSSE_ETA`. It has never inverted the
mask, and it had a working positive control for this exact arm.** That is what makes the sign
flip attributable to the bank rather than to the instrument.

**Width is not the fix.** `mask0k2` — the projection applied twice, the only board-implementable
width knob (`filter_mask_project` called a second time, exactly sparse for every k) — is WORSE on
both trims. So this is directional, not a tuning miss. Its pooled accuracy gain (+0.0410) arrives
with 377 fewer tracked frames and is the selection effect `M-02` names, not a win.

## Did the mechanism hold?

**HELD — decisively, and that is what makes the negative trustworthy.** `e_box` at init, median
over all 62 sequences, quartiles in brackets:

| arm | e_box at init |
|---|---|
| `rgb-l1relu` | 0.6795 [0.6328, 0.7398] |
| `rgb-l1relu-mask0` | 0.9547 [0.9397, 0.9637] |
| `rgb-l1relu-mask0k2` | 0.9901 [0.9855, 0.9922] |

Non-overlapping quartiles, the same separation the hardware arm showed (0.6049 -> 0.9500). **The
projection does exactly what it claims and the tracking gets worse.** This is not a mis-set
taper, a wrong mask centre, or a dead code path.

**The predicted explanation is NOT established, and must not be written up as if it were.** The
proposal guessed that the Layer-1 filter self-concentrates and leaves the mask little to remove.
The baseline `e_box` is 0.6795 against the old bank's 0.6049 — higher, but not nearly enough to
carry a sign flip, and the mask still moves it by +0.275. **Where the removed energy goes was not
measured.** The finding is that the mask's value is bank-dependent; the reason is open.

## Controls

- **The control arm reproduces its recorded value digit-for-digit.** `rgb-l1relu` scores
  A 0.5077 / R 0.4364 / 8686 tracked, exactly the row in `layer1_features.md`'s 32-channel batch.
  A drifting baseline is how `offline_sweep_par.sh` catches a wrong `--eta`/`--psr-min`, and it
  is the reason the baseline is re-run inside every invocation rather than quoted from a file.
- **`--mask-taper 1.0` was passed explicitly.** At the 0.25 default `mask0` is a narrow raised
  cosine with 99 non-zero bins per axis and is NOT board-implementable; the invocation is part of
  the result (`robustness_proposals.md` §2).
- **`--mask-center` left at `board`** (`n/2`, the exact periodic Hann), the only window the
  hardware can apply.
- **Health check before the sweep**, one sequence, read rather than assumed: the Layer-1 banner
  fires on all three arms (`resnet18 conv1 7x7/2, 64 -> 32 by weight PCA, stride 2, rect=relu`),
  gate 5.0, sigma 2.0 bins.
- 186 + 62 records merged from 62 + 62 private per-worker files; no worker failed.

## Stage B3 channel reliability — the licensed re-open is ANSWERED and CLOSED

`channel_reliability.md` refused board time on 2026-09-01 and named **one** legitimate re-open:
the screen ran against the sigma-2 / 128x128 operating point, and the response shape has moved
twice since (median accepted PSR 30.9 -> 18.6), so `rho`'s distribution does not transfer by
assumption. `chrel05` rode this sweep to answer it at zero marginal cost.

**dR −0.0040 paired, trim-3 −0.0378, P(dR<=0) = 0.563, 39 of 62 sequences tied.** The re-open is
spent: `gamma = 0.5` was +0.0210 pooled / +0.0023 trimmed on the old operating point and is a
null-to-loss on this one. **`N-20` stands, and there is no second re-open** — the mechanism
result (the anti-reliability mutant loses 0.0396) is unaffected and remains the part worth
citing. Its +0.0143 accuracy comes with 17 fewer tracked frames, i.e. `M-02` again.

## What not to re-derive

- **A prior positive screen EXPIRES when the operating point moves** — this is `M-14` and it is
  the transferable result. `channel_reliability.md` wrote the rule down for `rho` and applied it
  to itself; the mask was proposed on 2026-09-03 WITHOUT applying it, and eight minutes of CPU
  caught what would have been a 30-minute sweep, an ingest and a card round-trip. The rule is not
  "re-screen when you suspect something", it is **re-screen whenever the bank, the geometry or
  the response shape has moved since the arm was scored.**
- **A mechanism check does not license the arm.** `e_box` separates on both banks; the tracking
  outcome is opposite. `spatial_mask.md` had already found this from the other side — the
  hardware mask's `NEGATIVE_PEAK` story inverted when split by on-target. Mechanism confirmation
  bounds attributability, never value.
- **Do not read `mask0k2`'s accuracy as a reason to keep looking at width.** Every mask arm in
  this project that gained accuracy while losing tracked frames has been the selection effect.
- The bench header prints the module constant (`eta 0.125`), not `args.eta`. The value actually
  used is `args.eta`, verified at the call site (`rgb_vs_gray_loop.py:1072`); the printed line is
  cosmetic and misleading. **A stored log from this script does not record its own operating
  point** — read the invocation, not the header.

## Reproduce

```bash
scripts/offline_sweep_par.sh runs/vot/0903_offline-l1mask/l1mask62.json 16 \
    rgb-l1relu rgb-l1relu-mask0 rgb-l1relu-chrel05 \
    -- --eta 0.05 --psr-min 5.0 --mask-taper 1.0
scripts/offline_sweep_par.sh runs/vot/0903_offline-l1mask/l1mask62_k2.json 16 \
    rgb-l1relu-mask0 \
    -- --eta 0.05 --psr-min 5.0 --mask-taper 1.0 --mask-power 2
# k=2 carries the same arm name, so it is merged under a distinct one before scoring
# (merge_grid.py does not apply here -- it requires every cell to be the bare arm `rgb`).
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/grid_stats.py \
    runs/vot/0903_offline-l1mask/l1mask62_all.json rgb-l1relu \
    rgb-l1relu-mask0 rgb-l1relu-mask0k2 rgb-l1relu-chrel05
```

8 min 20 s wall on 16 workers, both sweeps.
