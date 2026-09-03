# VOT Phase 1 — PC-side converter, all 62 stb2022 sequences

**Status:** closed · **Updated:** 2026-08-24 · **Scope:** VOT bring-up 1: the PC-side converter over all 62 stb2022 sequences

**2026-08-24. GATE MET.** 62/62 converted and byte-exact, md5s recorded, real
anchors, 6/6 mutants caught, frame-buffer assertion enforced.

```
62 sequences, 19903 frames, 180544 tracked frames across 419 runs
anchors: dataset
  at 26.29 ms/frame (gray): 79.1 min
  at 28.58 ms/frame (RGB):  86.0 min
blobs 10.13 GB -> 1.4 min staging at the Phase 0a rate (117.2 MB/s)
largest single blob 1269 MB = peak board heap

PASS: 62 sequences byte-exact against a fresh decode
job cross-check vs toolkit find_anchors: 62/62 sequences exact
```

Fetched with `vot initialize vot2022/shorttermbox` (1.7 GB) into
`$VOT_ROOT/workspace`; converted by `scripts/vot_prepare.py` into
`$VOT_ROOT/data`. `VOT_ROOT` is now exported from `setup_env.sh` with no
in-repo fallback; `vot-toolkit` is a declared `vot` extra in `pyproject.toml`.

## THE BUG THIS PHASE FOUND — every groundtruth box was wrong

**stb2022 groundtruth is 4-column axis-aligned `x,y,w,h`. The VOT2015-era
sequences in `test-sequences/` are 8-column rotated polygons. `reduce_box`
assumed polygons.**

Read with the polygon rule, a 4-value line gives `x = [x, w]`, `y = [y, h]`.
`fernando` frame 291 (`440,229,198,230`) reduced to a **1.0 x 242.0 sliver**
instead of 230 x 198. All 62 manifests were affected — plausible, complete, and
wrong in every box.

**Why nothing caught it.** `scripts/rgb_vs_gray_vot.py:load_gt` makes the *same*
polygon-only assumption, so Phase 0c's "independent" cross-check agreed —
two implementations of one wrong rule. On `test-sequences/` the rule is correct,
so every 0c check passed legitimately. The format changed under a validated tool,
which is the same shape as `calib_report.py` being written against one log format
and pointed at another.

**What it took to see it.** Not a check — a number that looked wrong. A 1 px-tall
init box in the anchor list is the only reason this surfaced before hardware.

**The fix, and the check that now covers it.** `reduce_box` dispatches on value
count (4 -> rectangle, even >= 6 -> polygon, anything else a hard error), and
`verify` cross-checks every box against the toolkit's own `parse_region()` — a
genuinely independent parser that handles both formats itself. A new `polyonly`
mutant reproduces the original bug exactly and is caught with a 229 px
disagreement.

**Still outstanding: `rgb_vs_gray_vot.py` carries the same latent bug.** It is
correct on `test-sequences/` and would silently produce garbage on stb2022. It
needs either the dispatch or a format guard before it is ever pointed at this
dataset. The plan's intent — single-source the reduction so the board and the
offline harness agree by construction — would close it properly.

## Corrections to the plan's figures

| | plan | measured |
|---|---|---|
| Sequences | 60 | **62** |
| Frames | 20,826 | **19,903** |
| Tracked frames | ~57k–115k bracket | **180,544** |
| Run time (gray) | 25–50 min | **79 min** (86 min RGB) |
| Converted blobs | ~6.5 GB | **10.13 GB** gray (30.4 GB if RGB) |
| Peak board heap | 461 MB (`girl`) | **1269 MB** (`flamingo1`) |
| Stack id | `vot-stb2022` | **`vot2022/shorttermbox`** |

**180,544 is 57% above the plan's upper bracket**, because stb2022 places each
anchor to cover the *farther* end. The plan flagged this figure as the one to
measure rather than carry forward, and it was right to.

Staging stays negligible: 1.4 min against 79 min of tracking, ~1.8%. Peak heap
of 1.27 GB is comfortable against 12 GB, but it is 2.75x the plan's estimate —
size the board-side buffer from the manifest, not from `girl`.

## Frame geometry

No sequence exceeds 1080x1920 — but `birds2`, `zebrafish1` and `frisbee` sit
**exactly** at it, so the assertion passes with zero margin. `frame_bo` allocated
at maximum geometry is exactly right, and any future dataset needs the check
re-run rather than assumed.

## What Phase 2 and 3 must handle

**Empty groundtruth exists.** 41 frames (0.21%) across 12 sequences are
`0,0,0,0`. The toolkit's failure rule already ignores them
(`overlap <= 0.1 and not gt.is_empty()`). **No anchor lands on one** — checked,
0 of 419 — so `run_reset()` never receives a degenerate init box from the
dataset. Worth an assert on the board anyway, since it is cheap and the
alternative is a silently degenerate filter.

**Tiny targets are the real interpolator stress, not large ones.** 22 of 419
anchors init from a box with a side under 16 px; `drone1` anchor 200 is 9 px.
The smallest non-empty box in the dataset is `tennis` at **2.0 px**, i.e. a 4 px
ROI resampled UP to 128x128 — a 32x upsample through an interpolator that has
never run on hardware in any mode. `roi_crop`'s bilinear path is bit-exact
natively across 25 cases, but this is the extreme end of it.

**ROI exceeds the frame on 11/62 sequences, 9.6% of frames.** Worst is
`iceskater1` (531 of 661 frames). `roi_crop` border-clamps; `roi_exceeds_frame`
is in every manifest so the board can assert rather than clamp silently.

*(An earlier draft of these numbers read 52/62 and 24.9% — computed from the
broken boxes, before the format fix. Superseded.)*

## Layout

```
$VOT_ROOT/workspace/      toolkit workspace + JPEGs   1.7 GB   not in git
$VOT_ROOT/data/           62 x {.raw,.json} + trackers.ini  10.13 GB   not in git
$VOT_ROOT/data-local16/   the Phase 0c blobs, kept as a polygon-format
                          regression (12 of 16 names collide with stb2022,
                          so they cannot share a directory)   1.8 GB
$VOT_ROOT/roundtrip/      Phase 0b fabricated workspace
```

`scripts/vot_prepare.py` also writes `trackers.ini` next to the blobs, so the
out-of-repo workspace and the in-repo shim cross-reference through one source.

## Reproduce

```bash
source setup_env.sh                      # exports VOT_ROOT
uv sync --extra vot                      # vot-toolkit==0.8.1
cd $VOT_ROOT/workspace && vot initialize vot2022/shorttermbox --workspace .
cd -
P=env -u PYTHONPATH -u PYTHONHOME ./.venv/bin/python   # Vitis masks python
$P scripts/vot_prepare.py convert --root $VOT_ROOT/workspace/sequences --out $VOT_ROOT/data
$P scripts/vot_prepare.py verify  --root $VOT_ROOT/workspace/sequences --out $VOT_ROOT/data
$P scripts/vot_prepare.py verify  --root $VOT_ROOT/workspace/sequences --out $VOT_ROOT/data \
     --sequences fernando --mutate all        # 6/6 must be CAUGHT
```

Convert ~5 min, verify ~5 min. **The install and every invocation need
`env -u PYTHONPATH -u PYTHONHOME`** — the Vitis environment points python at
Vivado's 3.13 build, which fails the toolkit's source build in `typing`.

## Still open

- `--channels 3` remains unexercised. Grayscale first, per the plan; RGB is a
  re-run of this phase at 30.4 GB, not a redesign.
- The mutation suite runs on one sequence at a time. `fernando` covers the
  rectangle format; `--root test-sequences --sequences bmx` covers polygons.
