# The float twin beats the board arm by +0.021 R paired — and the result is CONFOUNDED, because the twin also has no scale filter

**Status:** current · **Updated:** 2026-09-04 · **Scope:** this project's own tracker in float under the multistart protocol; the pre-registered prediction FIRED, its attribution is open, and the scale bracket is now COMPLETE

## WHERE THIS ENDED UP

**2026-09-04, later the same day — TWO CORRECTIONS, one of them to this note's central argument.**

1. **The confound is RESOLVED.** The board's `SCALE_N=1` run supplies the missing cell. The
   scale term is a NULL (dR trim-5 **-0.0210**, P(dR<=0)=0.858) and the deconfounded arithmetic
   term is LARGER than the contrast below (trim-5 **+0.0102**, P=0.001). The headline was NOT
   the scale filter. See `evidence/fixed_point_cost.md` and claim `R-16`.
2. **THIS NOTE'S TRIM FIGURES FOR `twin - board` WERE COMPUTED TWO-SIDED, AND THE ARGUMENT
   BUILT ON THEM IS WITHDRAWN.** `drop-top-5 +0.0222` is a five-off-EACH-END trim. Under
   `grid_stats.py`'s stated one-sided definition -- drop the 5 sequences most FAVOURABLE to the
   arm -- it is **+0.0018**, and the dA figure is **-0.0124**, not -0.0027. Dropping the most
   favourable sequences cannot RAISE a mean, so "larger than the untrimmed mean" was the tell
   and it was read as evidence instead. **`R-13` was not trim-stable.** Its direction survives
   (36/21/5) and the deconfounded `R-16` contrast is trim-stable where this one was not.
   **Every OTHER trim in this note was re-checked against the same trajectories and is
   one-sided and correct** -- oracle-twin -0.0032, oracle-board +0.0210, CSRDCF-twin +0.0144,
   oracle dA +0.0766 all reproduce exactly, as do all four arms' A/R/EAO
   (workspace `0904_trimcheck`).

**2026-09-04.** No board time. `scripts/offline_multistart.py --tracker mosse:rgb-l1relu`, 62
sequences, 419 anchored runs, 180,544 frames, scored by `vot_ingest.py` exactly as a board
sweep. Numbers: `results/float_twin.csv`. Claim `R-13`.

## The prediction, written down first

Written before the run, in the scoping message. **`settled.md` holds that quantization is not
the cause of the poor robustness — "removing it makes tracking WORSE" — so the twin was
predicted to land AT OR BELOW the board arm's R 0.4279.** The stated falsifier: *"if it comes
back substantially better, a settled claim is wrong — and that, not the comparison itself,
would be the finding."*

## The result

| arm | implementation | scale | A | R | EAO |
|---|---|---|---|---|---|
| `board-l1relu` — shipping | fixed-point on VEK280 | DSST 33-scale | 0.5129 | 0.4279 | 0.1960 |
| **`mosse-rgb-l1relu`** — the twin | float64 host | **none** (held fixed) | 0.5125 | **0.4559** | **0.2041** |

Paired per sequence, twin minus board, n = 62:

```
dR  mean +0.0213   median +0.0047   trim-5 +0.0018   36 better / 21 worse / 5 tied   P(dR<=0) 0.055
dA  mean -0.0038   median -0.0025   trim-5 -0.0124   29/32/1                          P(dA<=0) 0.709
```

**CORRECTED 2026-09-04** (see WHERE THIS ENDED UP): these two rows read `drop-top-5 +0.0222` and
`-0.0027`, which were TWO-SIDED trims. The one-sided figures are above.

**The prediction FIRED**, in DIRECTION: the twin is better, not worse, on 36 sequences against 21.
**The trim-stability claim this note originally made here is WITHDRAWN** — one-sided trim-5 is
+0.0018, so a handful of sequences carry most of the magnitude, and `P(dR<=0) = 0.055` is above
the 0.05 line with nothing stronger behind it. Accuracy is a clean null either way. The
deconfounded contrast in `R-16` is the one that is trim-stable (+0.0102, P = 0.001).

## Did the mechanism hold? NO — and the headline must not be reported as a quantization result

**The twin differs from the board in TWO ways, not one**, which is precisely what
`measurement.md` forbids ("never move two magnitudes at once"):

1. **Arithmetic** — float64 correlation pipeline against cint16/Q1.15. (The FEATURES are *not* a
   difference: `float_conv` is unpopulated for the L1 banks, so the twin runs the same int8 conv
   through the same `l1_banks` bank.)
2. **Scale handling** — the twin has NO DSST scale filter at all; the board runs a 33-scale one
   that is measured broken (frozen on ~90% of frames, detector gain **−0.003**).

So `+0.0213 R` has two candidate mechanisms and this run cannot separate them:

- **(a) the fixed-point pipeline costs robustness** — which would overturn the settled claim; or
- **(b) the board's broken scale filter costs robustness** — i.e. a broken estimator is worse
  than no estimator. Note `scale_oracle_bound.py` priced a *perfect* filter at only +0.0023 R;
  it never priced the *harm* of the broken one, and nothing in this project has.

**(b) is not the less likely branch.** `scale_filter.md` root-caused the freeze to a
self-confirming loop; a self-confirming size estimate that drifts is exactly the kind of thing
that costs survival while a held box does not.

**THE SEPARATING EXPERIMENT IS ONE HOST-ONLY FLAG.** A board run at **`SCALE_N=1`** gives
*fixed-point + no scale filter*, the matched comparison to the twin's *float + no scale filter*.
That isolates the arithmetic exactly, and it is an scp, not a card swap. **Do not write up
either branch until it has run.**

## The scale bracket, completed — perfect scale buys ACCURACY, not survival

| arm | scale | A | R | EAO |
|---|---|---|---|---|
| `board-l1relu` | DSST 33-scale (broken) | 0.5129 | 0.4279 | 0.1960 |
| `mosse-rgb-l1relu` — the twin | none (held fixed) | 0.5125 | 0.4559 | 0.2041 |
| `mosse-l1relu-oracle` | oracle (groundtruth size) | **0.6233** | 0.5005 | 0.2667 |
| `opencv-csrt` — CSRDCF reference | own multi-scale | 0.5134 | 0.5471 | 0.2432 |

Paired per sequence, n = 62:

```
oracle - fixed (twin)   dR  mean +0.0319  median +0.0035  trim-5 -0.0032  33/25/4  P(<=0) 0.059
oracle - fixed (twin)   dA  mean +0.0936  median +0.0827  trim-5 +0.0766  53/ 8/1  P(<=0) 0.000
oracle twin - board     dR  mean +0.0532  median +0.0252  trim-5 +0.0210  41/15/6  P(<=0) 0.004
CSRDCF - twin (fixed)   dR  mean +0.0580  median +0.0492  trim-5 +0.0144  40/19/3  P(<=0) 0.018
```

**Read the oracle row with its trim, and the pooled number is a trap.** Its R looks like a large
gain (+0.0446 pooled over the fixed twin) and **it is not trim-stable: drop-top-5 is −0.0032**,
essentially zero, so the pooled figure is carried by a handful of sequences. Its **accuracy** gain
is the real one — +0.0936 mean, +0.0766 trimmed, 53 better / 8 worse, `P(dA<=0) = 0.000`.

**PERFECT SCALE KNOWLEDGE BUYS ACCURACY AND NOT SURVIVAL.** That is exactly what
`scale_oracle_bound.py` found post-hoc (+0.0023 R against +0.054 mean IoU) — **now reproduced
closed-loop from the opposite direction.** The two methods are genuinely independent: the oracle
bound substituted ground-truth sizes into fixed board trajectories and so could not change where
the tracker went, whereas here a correct size changes the ROI crop, hence the features, hence
every detection. A feedback path that could have rescued survival was available and did not.
**`scale_filter.md`'s conclusion is now corroborated by a method that could have contradicted it.**

**The bracket does NOT bound the harm of a broken filter, and this is the limit of what it
settles.** It spans no-filter to perfect-filter; a self-confirming estimate that drifts can sit
*outside* that span, worse than either end. So branch (b) survives — but it now needs the board's
broken filter to be substantially worse than *both* ends of a bracket whose two endpoints are
within ~0 R of each other. **The `SCALE_N=1` board run remains the only thing that separates
(a) from (b), and it is still one host-only flag.**

## Against a real baseline, in float

**CSRDCF still beats the twin: +0.0580 R pooled, +0.0144 after drop-top-5, `P(dR<=0) = 0.018`.**
So removing the fixed-point arithmetic does not close the robustness gap to a classical DCF —
it narrows it from 0.119 (board) to 0.091 (twin) pooled, and the trimmed figure says the honest
remaining gap is smaller still but real. **The original question — "is R below expectation for
this algorithm class?" — now has a partial answer: not purely an embedded-implementation
artefact.** What CSRDCF has and this does not is the anti-drift machinery `robustness_gap.md`
already named, and that conclusion no longer rests on a cross-paper comparison.

## What this does to the settled claim

`settled.md`'s quantization entry is **not yet overturned, but it is out of date on its own
terms.** It was screened on the 3x3 mobilenet bank at a 128x128 map via offline SINGLE-START
mean IoU (`gray` vs `gray-float`). By `M-14` — *a prior positive screen expires when the
operating point moves* — the bank, the geometry and the response shape have all moved since,
and this is the first time the question has been asked at the shipping operating point with a
toolkit-scored multistart protocol. **Re-screening it was due regardless of which way this
run came out.**

## Controls

- **`board-l1relu` re-ingested in this workspace returns 0.5129 / 0.4279 / 0.1960** — its
  published `results/arms.csv` row exactly. The pairing is against the real arm, not a
  re-derivation of it.
- **A forward run from anchor 0 is byte-for-byte identical to the bench's own single-start run**
  of the same arm (car1, 741 boxes, formatted strings compared). That is what makes the
  frame-list plumbing trustworthy: the multistart harness did not change the tracker.
- **`resolve_arm` was extracted from `rgb_vs_gray_loop.main()` and checked bit-identical** on
  `rgb`, `rgb-l1relu`, `rgb-eye` and `rgb-dec2` before anything else was built on it, so the
  twin's arm and the bench's arm cannot be two spellings of one config.
- **The oracle-scale bracket COMPLETED** (it had stopped at 348/419; the resume recomputed only
  the 71 missing runs, which is also what proves the resume path skips on LENGTH rather than on
  mere existence). See section "The scale bracket" below. It does not settle (a) vs (b) — that
  still needs the board's `SCALE_N=1` run.

## What not to re-derive

- **`--oracle-scale` crashed on the full 62 and the fix is in `run_arm`.** stb2022 carries **41
  zero-size groundtruth boxes over 12 sequences** (agility, girl, tennis, soldier, handball1/2,
  soccer1/2, flamingo1, frisbee, kangaroo, wiper). Adopting a zero size collapses the ROI,
  Stage A divides by zero, and the NaN surfaces hundreds of frames later as *"cannot convert
  float NaN to integer"* — nowhere near its cause. An empty annotation carries no size, so the
  previous one is held; VOT's own failure rule skips those frames too. **The bench never hit
  this because `car1`, its default sequence, has none.**
- **The twin does not scale like the OpenCV arms.** 31 ms/frame at 4 workers but 189 ms/frame at
  30: the numpy FFT work is memory-bandwidth bound, so the full 62 takes ~13 min per bracket
  mode, not the ~3 min a linear extrapolation predicts. Budget accordingly.
- **WITHDRAWN 2026-09-04.** This bullet read: *do not treat `P(dR<=0) = 0.055` as nothing,
  because the trim-5 figure (+0.0222) exceeds the untrimmed mean.* That figure was a TWO-SIDED
  trim; one-sided it is +0.0018, and a trim that exceeds its own mean is arithmetically
  impossible under the one-sided definition, which is exactly the tell that was missed. The
  effect IS carried by a few sequences on magnitude. What survives is the direction (36/21/5)
  and, after deconfounding, `R-16`.
