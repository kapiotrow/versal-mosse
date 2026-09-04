# The board's broken scale filter costs nothing; the fixed-point CORRELATION pipeline costs +0.0102 R

**Status:** current · **Updated:** 2026-09-04 · **Scope:** the decomposition of `R-13` into a scale term and an arithmetic term, using the board's `SCALE_N=1` run

**2026-09-04.** `runs/vot/0904_1225-l1relu_s1` (62 sequences, 419 anchored runs, 87,333 frames)
against `runs/vot/0902_1413-l1relu` and the float twin, all three ingested into one workspace
(`~/vot/analysis/0904_twin_s1`). The new arm differs from the shipping arm on exactly one flag:
`app.flagstamp` diff is `-DSCALE_N=33` → `-DSCALE_N=1` and nothing else; `aie.flagstamp` and
`crop.flagstamp` are byte-identical to the 0902 sweep's; the weights file is md5-identical
(`d8a7d6ff`). Claim id: `R-16` in `docs/thesis/claims.md`; numbers in `results/scale_ablation.csv`
and `results/arms.csv` row `rgb_l1relu_s1`.

## The prediction, written down first

Written before the sweep, in the session that built the ELF:

> Branch **(a)**, arithmetic: `SCALE_N=1` lands at R ≈ 0.4279, the shipping arm — removing the
> scale filter changes nothing and the twin's +0.0213 is the fixed-point pipeline.
> Branch **(b)**, the broken estimator: `SCALE_N=1` moves toward the twin's 0.4559 — a broken
> scale filter is worse than none, and the arithmetic is exonerated.

The falsifier for (a) was a board R materially closer to 0.4559 than to 0.4279. The falsifier
for (b) was a null. Also pre-registered, on the frame budget: **no change**, because
`TAIL_PARALLEL=1` hides the ~2.89 ms scale path behind the ~4.80 ms filter update.

## The result

| arm | implementation | scale | A | R | EAO | frames |
|---|---|---|---|---|---|---|
| `board-l1relu` (shipping) | fixed-point on VEK280 | DSST 33-scale | 0.5129 | 0.4279 | 0.1960 | 82504 |
| **`l1relu_s1`** (this arm) | fixed-point on VEK280 | **none** (`SCALE_N=1`) | 0.5146 | **0.4334** | 0.2000 | 87333 |
| `mosse-rgb-l1relu` (twin) | float64 downstream of features | none (held fixed) | 0.5125 | 0.4559 | 0.2041 | 89107 |

Paired per sequence, `grid_stats.py`'s definitions:

| term | contrast | dR mean | median | trim3 | trim5 | b/w/t | sign p | P(dR≤0) |
|---|---|---|---|---|---|---|---|---|
| **SCALE** | `s1 − board` | −0.0099 | +0.0000 | −0.0171 | **−0.0210** | 19/29/14 | 0.193 | **0.858** |
| **ARITHMETIC** | `twin − s1` | +0.0312 | +0.0047 | +0.0177 | **+0.0102** | 35/12/15 | **0.001** | **0.001** |
| R-13 (confounded) | `twin − board` | +0.0213 | +0.0047 | +0.0079 | +0.0018 | 36/21/5 | 0.063 | 0.053 |

**Disabling the board's scale filter is a null, and the two mechanisms do not split the +0.0213
between them — the scale term is NEGATIVE.** Pooled R does say +0.0055 for `s1`, and that is the
trap: it is length-weighted, the paired median is exactly 0.0000, and more sequences lose (29)
than win (19). `birds2` alone contributes −0.3438. This is the same shape as `R-14`, and it is
the statistic `CLAUDE.md` names as the one this project has most often been misled by.

**With scale handling matched on both sides the arithmetic term is stronger than the confounded
contrast it came from**: `R-13` did not clear its own bar (trim-5 +0.0018, P = 0.053), and the
deconfounded term does, at trim-5 +0.0102 and P = 0.001 with 35 sequences better against 12.

## What "arithmetic" means here, and what it does NOT mean

**The twin runs the same int8 conv datapath as the board** — the same `l1_banks` bank, quantized
weights, `saturate(acc >> out_shift)`, int16 clips, integer Hann (`rgb_vs_gray_loop.resolve_arm`
calls `quantize(fw, bf)` for every `l1` arm; only the explicit `-float` arms bypass it). What
differs is everything DOWNSTREAM of the feature map: the cint16 FFT/IFFT chain and its 3-3-3
shift budget, the Q1.15 filter with `H_SHIFT=15`, and the fixed-point B2/B3 corrections.

So this measures the **fixed-point CORRELATION pipeline**, and it does **not** touch the int8
FEATURE path, which is common to both arms.

That distinction is what keeps this from colliding with `settled.md` as a whole. Of the three
suspects that entry bracketed:

| suspect | settled.md verdict | status after this run |
|---|---|---|
| cint16 / Q1.15 / `H_SHIFT` pipeline | exonerated — "the float64 bench reproduces the board's failures anyway" | **CONTRADICTED.** That exoneration was a qualitative reproduction argument, never a paired contrast. Measured here at +0.0102 trim-5, P = 0.001 |
| saturation / rails | exonerated (corr −0.025, zero rails) | untouched |
| the int8 FEATURE path — "it HELPS" | exonerated | **untouched, and out of scope.** Measured 2026-08-27 offline on 8 sequences with the GRAY mobilenet 3x3 bank, before the Layer-1 arm existed. Per `M-14` that screen has EXPIRED for the current operating point, but nothing here refutes it — the twin carries the same int8 features |

**Do not write this up as "quantization is the cause of the poor robustness."** It is one
quantized stage of several, priced at ~1/4 of the gap to CSRDCF, and it costs accuracy back.

## Did the mechanism hold?

- **Branch (b) falsifier FIRED.** The scale term is a null and trims negative. A broken
  estimator is not worse than none on this arm, and the `SCALE_N=1` direction buys nothing.
- **Branch (a) HELD, and sharpened.** See the caveat above about which stage.
- **The frame-budget prediction HELD.** 24.82 ms/frame against the 0903 shipping run's 25.16 ms
  (8162 frames, `car1` job 0 ×11), and energy 12.76 mJ/frame against 12.2 — no material change,
  which is what `TAIL_PARALLEL=1` requires. `results/power.csv` rows `l1relu_s1`, claim `P-12`.
  **This is not an FPS result**: the ~5% spread is cross-session and unbounded by this protocol.
- **ACCURACY MOVES THE OTHER WAY.** `twin − s1` on A is −0.0060 mean, −0.0133 trim-5. This is a
  robustness/accuracy trade, not "float is better". And EAO moves only 0.2000 → 0.2041 on an R
  gap of +0.0225 pooled — the [115, 755] window diluting a feature-side gain threefold, exactly
  as `M-17` describes.

## Controls

- **`board-l1relu` re-ingested to `0.5129 / 0.4279 / 0.1960`** — its published `arms.csv` row,
  digit for digit, from a results root it had never been ingested from. Its 838 files are
  md5-identical to `/srv/vot/results/l1relu`.
- **`mosse-rgb-l1relu` re-ingested to `0.5125 / 0.4559 / 0.2041`** — its `float_twin.csv` row,
  digit for digit.
- Both controls came from the same `vot analysis` invocation as the new arm, so a scoring-path
  drift would have moved them too.
- **The flag is live in the binary.** `strings` is inert here: both the enabled and the
  `scale: DISABLED` banners are compiled into every build, so it reports a false absence. The
  real discriminator is that a `SCALE_N=33` twin built from the same sources has a different
  md5 (`a31696c5` against `cd5850c0`), and `make test_host SCALE_N=1` asserts *S=1 disables the
  filter*, *disabled → invalid, factor 1*, *disabled → no training*.
- `vot_ingest.py` re-derived every run name from the anchors and checked every trajectory
  length: 1257 staged, all verified.

## What not to re-derive

- **The xclbin md5 does not match the 0902 sweep's and that is not an error.** `9cea47ce` was no
  longer on disk (the tree had been rebuilt for the `l1lin` arm at `CONV_RELU=0`), so the card
  was rebuilt and re-flashed; `v++` is not bit-reproducible and the new one is `65ed581a`. The
  `aie.flagstamp` is byte-identical, which is what establishes it is the same design. **This is
  a residual confound on the SCALE term only** — both arms of the ARITHMETIC term sit on the
  same side of it. It also means `roadmap.md`'s "one host-only flag, an scp not a card swap" was
  stale: `CONV_RELU` reaches `AIE_FLAGS`, so this cost a full AIE rebuild.
- **Pooled R and paired R disagree in SIGN on the scale term** (+0.0055 pooled, −0.0099 paired).
  Do not quote the pooled number.
- **The power ELF is not the sweep ELF, deliberately.** The sweep ELF carries
  `PROGRESS_EVERY=25` and `CSV_FLUSH_EVERY=200`; measuring power with it would have priced
  console I/O and per-frame flushes onto the APU rail, which carries 68% of the dynamic power.
  The power build restores `PROGRESS_EVERY=100000 CSV_FLUSH_EVERY=1` so its `app.flagstamp`
  differs from the 0903 power build on exactly `-DSCALE_N`.
- **`power_measure.py` crashed AFTER the 25-minute run completed** — `UnboundLocalError` at
  `settle_s = settle_s if settle is None else settle`, a self-reference that made the LIVE
  report path fail every time since `89aaa1c`; only `--reanalyse` (which passes `settle=0.0`)
  ever ran. Fixed to `args.settle_s`. Nothing was lost because `samples.csv` is written before
  `report()`, which is what `--reanalyse` exists for.
- **`--reanalyse`'s comment "settle already applied when samples.csv was binned" is FALSE.**
  `phase_of()` bins on the raw window bounds and keeps every sample, so reanalysis with
  `settle=0.0` includes the transients the live path discards. The 0904 numbers above came
  through that path, as the 0903 ones did, so the two are comparable to each other — but the
  0903/0904 pair should be reanalysed together if that behaviour is ever changed. NOT FIXED.
- **Three rails report `CONTROL FAILED` (graph_post vs graph) at −0.001, +0.005 and −0.037 W.**
  That is the quantization false-alarm `power.md` already documents: on a rail whose s.e.
  collapses, any difference clears "2 s.e.". `tail vs static` passed on every rail. Sound run,
  but the verdict line says FAILED and must not be quoted as a clean pass.
- The `l1relu_s1` entry under `~/vot/results-offline/` is a **symlink** to
  `/srv/vot/results/l1relu_s1`, not a copy — `vot_ingest.py` takes one results root.
- `/srv/vot/results/_stray_root_0903/` holds two `car1` trajectories the 0903 power run wrote to
  the export ROOT (it used the default `VOT_RESULTS_DIR`). `vot_ingest.py` refuses to run while
  an unseparated arm sits there, which is the guard working. Today's power run wrote to
  `power_s1/` instead.
