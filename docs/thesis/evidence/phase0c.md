# VOT Phase 0c — converter validated against the 16 local sequences

**2026-08-24. CLOSED. 16/16 byte-exact, 5/5 mutants caught.**

`test-sequences/` is a subset of the same dataset the stb2022 run will use, so the
converter can be validated before a single byte of stb2022 is downloaded. That is
the whole point of doing 0c before Phase 1.

Tool: `scripts/vot_prepare.py` (tracked in git; the blobs are not — the script is
what makes them regenerable).

## Result

```
16 sequences, 5971 frames, 1.9 GB, 52 s to convert, 49 s to verify
PASS: 16 sequences byte-exact against a fresh decode   (ALL frames, not sampled)
gt cross-check vs rgb_vs_gray_vot.load_gt: 16/16 exact
5/5 mutants caught
```

`girl` converts to 460.8 MB, which is the 461 MB the plan's staging budget assumed.

## What the converter emits

Per sequence, into `$VOT_ROOT/data`:

- **`<seq>.raw`** — frames back to back, `rows × cols × channels` bytes each, **no
  header**. The board computes an offset and `memcpy`s; it never parses anything.
- **`<seq>.json`** — dims, frame count, blob md5, per-frame groundtruth reduced to
  axis-aligned boxes, the job list, and the two convention strings spelled out.
- **`<seq>.luma`** — only at `--channels 3`.

## The luma convention is pinned, and NOT to PIL

`Image.convert("L")` is **not** this pipeline's grayscale. PIL truncates through an
integer path with coefficients 299/587/114; `rgb_vs_gray_holdout.to_luma` rounds a
float `0.2989R + 0.5870G + 0.1140B` and clips, and that is the convention
`export_weights.py` collapses the conv kernels with.

The `pilluma` mutant measures the gap: **37 differing bytes out of 885,760, max
delta 1.** Two consequences:

1. A tolerance-based check would not have seen it. Zero tolerance is load-bearing
   here for the same reason it is on `make test_roi_crop`.
2. Had it shipped, Phase 2's trajectory comparison would have shown board and
   Python disagreeing — and the ready explanation ("fixed-point vs float FFT, we
   expected close-not-identical") would have absorbed it silently. A 1-LSB
   convention error hiding behind an expected difference is the worst shape of bug
   this project keeps finding.

## Mutation testing — the verifier is known to fail, not assumed to

A passing test on a path with no prior coverage is worth nothing until it has been
shown to fail. Each mutant corrupts the converter one specific way:

| mutant | what it breaks | caught by |
|---|---|---|
| `offbyone` | blob starts at frame 2 | frame 1, 286961/885760 bytes differ |
| `dropframe` | one interior frame omitted | frame 39, 565878 bytes |
| `reverse` | frame order inverted | frame 1, 856136 bytes |
| `transpose` | rows/cols swapped | frame 1, 878434 bytes |
| `pilluma` | PIL's luma instead of BT.601 | frame 1, **37 bytes, max delta 1** |

```bash
env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python scripts/vot_prepare.py \
    verify --out $VOT_ROOT/data --sequences bmx --mutate all
```

**What verify covers**: frame order, blob offsets, dimensions, channel layout, the
luma convention, blob md5, and the box reduction. **What it does not**: libjpeg
itself — both sides decode with PIL. That is deliberate; the decoder is
deterministic and is not what is at risk. The bookkeeping is, and the bookkeeping
is what the mutants attack.

The groundtruth check is genuinely independent: the manifest boxes are compared
against `rgb_vs_gray_vot.load_gt`, a **separate implementation**, so 16/16 exact
means two implementations of the min-max reduction agree — not that one agrees
with itself. Phase 1 should collapse them into one by having the harness import
from `vot_prepare`, at which point this check becomes structural instead of
empirical.

## Findings that change Phase 2

**The padded ROI exceeds the frame on 5 of 16 sequences — this is not an
stb2022-extremes problem, it is live in data already on disk.** `roi = box × 2.0`:

| sequence | frames with ROI > frame | of |
|---|---|---|
| singer1 | **140** | 351 |
| singer3 | 55 | 131 |
| bmx | 22 | 76 |
| dinosaur | 19 | 326 |
| motocross1 | 3 | 164 |

The risk register rated this Medium and expected it only at the dataset's extremes.
`roi_crop` border-clamps, so it is not a crash — but the plan's position is to
**assert on the board rather than clamp silently**, and `roi_exceeds_frame` is in
every manifest so the board can.

**The local 16 exercise both ends of the bilinear interpolator**, which has never
run on hardware:

- **Upsampling**: `birds1` min box side 19.0 px, `book` 21.2, `sheep` 22.0 → a
  ~40 px ROI resampled UP to 128×128.
- **Downsampling**: `bmx` max side 524.8 px → a 1049 px ROI resampled DOWN by 8.2×
  with no prefilter. This is the aliasing case the risk register warns about.

So Phase 2's first hardware run can pick its exposure deliberately. Suggested
order: a mid-range sequence (`car1`, boxes 49–139 px, resample near 1:1) to
establish the path works at all, *then* `birds1` or `bmx` to hit the interpolator
hard — one variable at a time, rather than discovering both at once.

## The job list is SYNTHETIC and the manifest says so

`anchors_source` reads `synthetic:spacing=50`. Real anchors and their directions
come from stb2022's own per-frame anchor values in Phase 1. This placeholder exists
so the format round-trips and so Phase 2 has a job to run.

It is a named field rather than an assumption because a synthetic job list mistaken
for the real one would produce a complete, plausible, entirely invalid AR report.

At spacing 50, **one direction per anchor**, the 16 local sequences yield
**64,409 tracked frames ≈ 28 min at 26.29 ms/frame**. Useful as a dry-run cost;
**not** a prediction for stb2022, whose anchor scheme differs.

**Corrected by Phase 0b.** The first version of `make_jobs()` emitted both a
forward and a backward run per interior anchor, giving 85,020 — a 24%
overstatement. The toolkit's `find_anchors()` splits anchors on the SIGN of a
per-frame value into two disjoint lists, so each anchor runs exactly once, in
one direction. See [phase0b.md](phase0b.md).

## Still open

- `--channels 3` is implemented but unexercised. Grayscale first, per the plan.
- `VOT_ROOT` is **not** defaulted — the script exits with an explanation if it is
  unset and `--out` is absent. It still needs adding to `setup_env.sh`.
- The blobs currently live in `~/vot/data` (1.9 GB). Move if `VOT_ROOT` differs.
