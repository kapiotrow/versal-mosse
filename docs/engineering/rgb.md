# RGB features (`CONV_IN_CH=3`) — shipping

**Status:** current · **Updated:** 2026-09-02 · **Scope:** the `CONV_IN_CH=3` datapath, its testing, and what it costs

*(`CONV_IN_CH=3` still ships. The rest of this file describes the 3x3 mobilenet arm it was
measured on; since 2026-09-02 the shipping bank is resnet18 conv1 7x7/2 PCA'd to 32 channels at
a 64x64 map, so the ch16 conv2d figures below are that arm's, not today's — see
`../thesis/evidence/arm_l1relu.md`.)*

Split out of CLAUDE.md 2026-08-31 and **maintained here since** — this file, not
CLAUDE.md, is where this topic is kept current; CLAUDE.md carries only the one-line
version and a link.


`CONV_IN_CH=3` is the default. Build with `make weights CONV_IN_CH=3` then
`ARM=rgb PATCH_ROWS=128 PATCH_COLS=128 N_CHANNELS=16 scripts/calib_build.sh`. 4-4-4 carries both
gray and RGB at 128x128; no RGB-specific FFT budget.

### Why RGB — CLOSED ON HARDWARE 2026-08-27

`~/vot/analysis/full62`, 419 runs per arm, both verified run-name-and-length against the
dataset's anchors before analysis:

| arm | A | R | EAO | frames |
|---|---|---|---|---|
| gray `H_SHIFT=14` | 0.4890 | 0.2743 | 0.1367 | 48,603 |
| **RGB `H_SHIFT=15`** | **0.5043** | **0.3065** | **0.1474** | **54,813** |

R better on 37 / worse on 15 / tied on 10. Largest swings are all RGB gains and all in
ROBUSTNESS: `car1` 0.634 → **1.000**, `book` +0.311, `lamb` +0.272. **12.8% more frames
survive.** (Confound: the arms are at different `H_SHIFT`.)

**RGB WAS NEVER AN ACCURACY CHANGE, and the shape matters more than the number.** Offline mean
IoU over 8 stb2022 sequences is a TIE (+0.0011); the synthetic hardware arm moved the wrong way
(0.9188 → 0.9173). Its claim was always **failures**, −18% under the supervised protocol — a
robustness metric mean IoU structurally cannot express. **Decide a colour question on AR, never
on mean IoU.**

Supporting offline measurements (`scripts/rgb_vs_gray_*.py`):

| measurement | gray | RGB |
|---|---|---|
| feature-bank rank / participation ratio | 9 (hard cap) / 4.94 | 16 / 7.43 |
| held-out Bolme PSR (147 paired evals, car1) | 12.97 | 21.18 (**+1.63×**) |
| VOT supervised failures (16 seq, 5971 frames) | 51 | **42 (−18%)** |
| conv2d scheduled cycles/frame (ch16) | 4.60 ms | 9.19 ms (**2.00×**) |

**Why the collapse costs anything.** A 3×3 grayscale kernel lives in 9 dimensions, so 16
channels CANNOT be independent — the cap is structural, not a property of the pretrained
weights. BT.601 guts the four colour-opponent channels: 0/2/9/10 keep 0.32/0.60/0.63/**0.037**
of the per-plane norm against 1.24–1.39 for the achromatic ones, and per-channel int8 then
renormalises that residue to full scale. ch0/ch9/ch14 sit within 2–6° of one line in gray,
59–72° apart in RGB.

**The 08-24 validation, kept for its method.** Three 200-frame arms, one variable each
(`run_0824_1354`/`_1432`/`_1442`). The colour-free control — same 27 taps, bias, quantization
grid and joint normalization, fed three IDENTICAL luma planes — reproduced grayscale's
decisions **bit-for-bit on all 199 frames**, which validates the whole RGB datapath against a
known-good reference AND proves the 1.65× PSR floor is colour and not bookkeeping. ch0's
`F_ch` collapse to 0.084× of gray was PREDICTED at 0.09× from the exported weights alone —
the sharpest confirmation in this file that the weight-collapse model is right, and what makes
ch0's `F_ch` the cheap on-board colour-path test.

**THE SYNTHETIC SCENE IS A WEAK COLOUR STIMULUS.** `FRAME_RGB_MODE=1` tints one luma image per
plane, i.e. **rank-1 across the plane dimension**: no chromatic texture independent of luma, so
a colour-opponent channel sees a scaled copy of luma. ch0 lands at 0.52× instead of the
decorrelated 1.25×. **A good IoU on that scene is NOT evidence for the VOT result above.**

### The datapath, end to end

**The wire format is pixel-interleaved, the line buffer is planar.** roi_crop must send
interleaved (planar would need whole planes resident); conv2d de-interleaves into three 3-row
buffers as it unpacks, costing index arithmetic and ~1.2 KB. Without it `load_unaligned_v`
gathers `[R G B R G B…]` and the MAC loop needs shuffles. Three int32 words carry exactly four
RGB pixels: `R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3`.

**`roi_crop` Stage A** at `ROI_IN_CH=3`: all three planes share one geometry and one set of
bilinear weights — only the `+p` byte offset differs — and the frame is interleaved in DDR, so
one source pixel's three taps are contiguous. Normalization is **JOINT**: one mean and one
`inv_q` over all 3·pr·pc samples. Per-plane statistics would equalize the planes and delete
exactly the chromatic contrast RGB exists for — silent, and self-defeating. Scratch 16 → 48 KB.
`sum_x2` peaks at 2.111e14 against `ap_uint<48>`'s 2.815e14, so **a fourth plane would not fit**
and the reference asserts the width.

**The host keeps the scene in LUMA and colourises on the way out.** Every scene function and
`scale_extract`'s 33 crops are single-plane; one pass expands the touched rect into the
interleaved buffer. At `CONV_IN_CH=1` there is no second buffer and no copy. `scale_extract`
reads luma, so **the DSST scale filter needs no recalibration**. The invariant is that every
luma write reaches `scene_touch()`; miss one and the device reads last frame's colour there,
which looks like a slightly worse tracking result rather than a bug. `SCENE_VERIFY=1` catches it.

**The RGB conv2d stack.** `make graph CONV_IN_CH=3` used to fail the link-stage stack check
(1344 bytes against 1024) producing NO `libadf.a`, while the per-kernel *compile* succeeded
either way — which is why it went unnoticed. Fix: `stack_size(conv2d) = CONV2D_STACK` (2048),
applied **only** at `CONV_IN_CH=3`. No kernel arithmetic changed; the RGB build allocates
`MG(15,0) size: 0x800` while every other node stays `0x400`.

**The RGB branch is VECTORIZED, not scalar** — 27 `aie::mac` with `load_unaligned_v`. The
`static_assert(CONV_IN_CH == 1)` guards the SEPARATE grayscale block, which the RGB branch
returns before reaching. Reading its 27 hoisted `int8_t` *weight* scalars as a scalar datapath
makes the 219-cycle schedule look far worse than it is.

### Testing — and why each suite means anything

Every RGB suite is **mutation-tested**, because a passing test on a path with no prior coverage
is worth nothing until it has been shown to fail.

| suite | coverage | mutants caught |
|---|---|---|
| `make test_roi_crop` | 17 gray + 8 RGB cases, zero tolerance | per-plane mean, planar scratch store, dropped plane index: 6-7 of 8 RGB, 0 of 17 gray |
| `make test_scene` | interleave, Q8 gains, saturation, clipping, missed `scene_touch()` | 8 of 8 |
| `make x86sim_check … s6rgb` | the real 27-tap kernel vs `simulate_conv2d`, 16384/16384 | de-interleave 48.4%, dropped MAC 22.8% |

The survivors are informative: `rgb_flat` survives everything (var 0 → all zeros regardless),
and a dropped plane index survives `rgb_gray_control` because with three identical planes it
genuinely IS a no-op. **A suite built only from replicated-luma frames would have caught none of
that mutant** — hence decorrelated planes in the RGB test frames. `s6rgb` writes to its OWN
directory (overwriting `s6` would silently feed RGB vectors to a grayscale check) and its patch
comes from `roi_crop_ref.stage_a_rgb`, not s6's float shortcut. **Only ONE RGB scenario exists,
on purpose**: RGB changes conv2d and nothing downstream.

### Cost — RGB COSTS WHAT THE HOST PAYS, NOT WHAT conv2d COSTS

**conv2d 2.00×**, from the compiler's schedules, reproduced byte-for-byte by the build that
linked:

| loop | gray | RGB | ratio |
|---|---|---|---|
| stream read, per 4 px | 28–31 cyc | 84–87 cyc | **3.00×** |
| MAC + post, per 16 px | 163 cyc | 219 cyc | 1.34× |
| per frame at ch16, 1 GHz | 4.60 ms | 9.19 ms | **2.00×** |

**The stream read goes from 44% of conv2d to 61%** — RGB makes the already-dominant term more
dominant, because the patch is re-streamed once per output channel. The RGB MAC loop does NOT
software-pipeline (`219 (exceeds -k 64) → no folding`, critical cycle 200 against gray's 24),
so 219 is a give-up number a tuned variant could plausibly beat.

**Two traps in reading those schedules.** (1) `aiecompiler` reuses a cached per-kernel object
when the preprocessed source is unchanged, so a `CONV_IN_CH=1` baseline silently reports NO
conv2d schedule — `rm -rf $(BUILD_DIR)/Work $(BUILD_DIR)/libadf.a` first. (2) The "conv2d 140
cyc/16px" figure in the Makefile is the `main_` WRAPPER block, and does not move when the
arithmetic triples.

Measured `run_0824_1457` vs `run_0821_1725`: **28.58 ms vs 26.29**, so +2.29 ms — and conv2d's
+4.59 ms of AIE compute **does not appear in the frame at all** (GMIO total unchanged, 11.133 vs
11.134; blocking `wait()` unchanged, 4.55 vs 4.51). The whole +2.29 is host-side:

| stage | gray | RGB | delta |
|---|---|---|---|
| frame push (`frame_bo` 2 → 6 MB) | 0.472 | 1.385 | **+0.91** |
| roi_crop launch (3× bilinear taps) | 1.013 | 1.471 | +0.46 |
| colourise RGB (new pass) | — | 0.338 | +0.34 |
| `frame_bo.sync` | 0.116 | 0.304 | +0.19 |
| filter upd+quant | 4.806 | 5.215 | +0.41 |
| **frame** | **26.29** | **28.58** | **+2.29** |

**The offline "~30 FPS" prediction was arithmetically fine and its PREMISE was wrong** — it
costed conv2d as if AIE time were frame time. The frame is 84% CPU-bound, so the AIE had slack
and absorbed the doubling. Third time a self-consistent offline model has been overturned by its
premise here. **The lever for RGB speed is host memory traffic, not the 27 taps.**

**Retired — "RGB is handicapped by its larger `out_shift`."** 27 taps triple `ACC_MAX_THEORY`
(mean out_shift 3.69 → 4.25), but forcing gray's shifts onto RGB (`--match-shift`) makes it
**worse**, 42 → 53 failures, with 0.0000% saturation at all three clip sites. Not clipping.
**Caching the patch in conv2d's tile does not fit**: at `FFT_ROW_WS=64` the output window's
ping-pong is already the whole 64 KB tile. **No alignment obstacle** — the PLIO is 32-bit and
128·128·3 = 49152 B divides exactly. See [[verify-stated-blockers-arithmetically]].

