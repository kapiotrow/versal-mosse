# The shipping arm on the serial console: 25.82 ms, the transport term is 0.91 ms not 3.79, and roi_crop is now 29% of the frame

**Status:** current · **Updated:** 2026-09-04 · **Scope:** frame time and the per-stage frame budget for the shipping arm, the measured console-transport term, and why the synthetic scene cannot be used for either

**2026-09-04.** `runs/perf/0904_l1relu_console/`. No rebuild of anything but the host ELF: the
card already held the shipping `a.xclbin` (`65ed581a…`, md5-matched against the package tree).
Claim ids: **`P-01`**, **`P-02`**, **`P-14`** in `docs/thesis/claims.md`; numbers in
`results/perf.csv` and `results/frame_budget.csv`.

## Why this run existed

`perf.csv`'s headline was **26.29 ms / 38.04 FPS** — a 128x128 / ch16 / 3x3 **grayscale**
build on a synthetic scene, measured 2026-08-21. Every frame time for the shipping arm
(`rgb_l1relu`, 64x64 / ch32 / 7x7-s2 / RGB) was an **ssh** number: 24.07 ms on `agility`,
24.82 scale-free, 25.16 in the power run. So the thesis's best-FPS figure and its best-EAO
figure came from different builds, and `measurement.md`'s own rule — *quote FPS only from a
serial-console run* — made the shipping numbers unquotable as FPS.

## The result

Same ELF (`a31696c5…`, which differs from the 0904 `SCALE_N=1` sweep ELF on exactly
`-DSCALE_N`), same sequence, `car1` job 0, 742 frames, **741 accepted / 0 gated** in both.

| | serial console | ssh | delta |
|---|---|---|---|
| frame body | **25.82 ms** (38.7 FPS) | 24.91 ms (40.1 FPS) | **0.91 ms** |
| APU subtotal | 9.627 | 9.602 | +0.025 |
| GMIO | 9.960 | 9.876 | +0.084 |
| roi_crop launch | 7.422 | 7.528 | −0.106 |
| **unattributed** | **1.389** | **0.485** | **+0.904** |

**The transport term is 0.91 ms, not `phase4`'s 3.79 ms** (`P-09`). That is not a contradiction:
phase4 measured the UART at **progress-every-frame**, and the shipping build prints at
`PROGRESS_EVERY=25`. The cost is proportional to bytes on the wire, which is the thing
`embedded_comparison.md` sec.3.2 could only assume.

**The control validated itself.** The two runs agree stage-for-stage to under 0.15 ms on
everything, and the whole difference lands in `unattributed` — the bucket the console I/O is
supposed to occupy — to within 0.006 ms of the frame-time delta. A transport term that leaked
into any other stage would have meant the instrument was measuring something else.

## The re-ranking: roi_crop is 29% of the frame and 99% of it is a busy-wait

| stage | 128x128 ch16 gray | shipping | |
|---|---|---|---|
| roi_crop launch | 1.013 (3.9%) | **7.422 (28.7%)** | **7.3x** |
| APU subtotal | 15.639 | 9.627 | 1.6x smaller |
| GMIO | 11.134 | 9.960 | |
| filter upd+quant | 4.806 | 2.593 | |
| scale extract | 2.248 | 2.211 | invariant, as `arm_res64.md` sec.19.5 predicted |

`roi_crop`'s own breakdown says **7.495 of 7.569 ms is `crop_ip poll ap_done`** — 0.23 ms per
channel across 32 channels, 4386 poll iterations. So the largest single item in the shipping
frame is the host **spinning on the PL's completion flag**, not host work and not blocked in
`wait()`. It grew 7.3x because the crop is 128x128 (four times the output pixels of the ch16
arm's 64x64 crop — `roi_crop`'s cost is set by OUTPUT pixels, `arm_res64.md`) across twice as
many channels.

**This re-ranks the performance roadmap.** `roadmap.md`'s item 1 — software-pipeline the channel
loop — was argued from the `fft_col_out`/`accum_out` pair; the bigger and simpler target is now
overlapping this poll with the host's per-channel work, exactly as `roi_crop` itself was
pipelined on 2026-08-21 (5.196 -> 1.020 ms). **`scale extract` is NOT the head of the tail on
this arm**, which `arm_res64.md` sec.19.5 had concluded from the ch16 arm.

## `P-02` restated: still CPU-bound, but a third of it is now a spin

Host CPU = APU subtotal 9.602 + GMIO async 6.778 + roi_crop 7.528 + unattributed 0.485 =
**24.4 ms of a 24.91 ms frame**, and only **31%** of GMIO blocks (3.097 of 9.876 ms) against
41% on the old arm. The frame is *more* CPU-bound than the 84% `P-02` records — but the
character has changed: **7.5 ms of that "CPU" is a busy-wait on `roi_crop`**, which a sleeping
wait or a pipelined launch would return. `P-12`'s energy result (68% of dynamic power on the
APU rail) is unaffected in direction and is, if anything, better explained: a spin loop burns
the APU rail exactly like real work.

## THE SYNTHETIC SCENE CANNOT MEASURE THIS ARM — and it fails in both directions

The run was planned on the synthetic scene, to sit in `perf.csv`'s existing series. It cannot:
**the shipping arm does not track the synthetic scene.** 128 of 199 frames gate
`NEGATIVE_PEAK`, longest gated run 64, mean IoU 0.6946 against the gray arm's 0.9188 on the
same generator.

- **Output to a file: 23.23 ms — the tail is UNDER-counted.** `filter upd+quant`, `publish` and
  `scale extract` report **0.4 calls/frame**, because a gated frame skips all three. The number
  is ~2.6 ms low and describes a tracker that is mostly refusing to update.
- **Output to the UART: 58.53 ms — dominated by warning spam.** Each gated frame prints **six**
  lines (`[psr] WARNING`, `[psr] DISAGREE`, `[scale] FROZEN`, `[gate] position HELD`,
  `filter: FROZEN`, `H(q15): unchanged`) that `VERBOSITY=0` does not suppress. 128 frames x 6
  lines is ~35 ms/frame at 115200.

Both are in the run directory and **neither is quotable.** This is `CLAUDE.md`'s existing
warning — *the synthetic scene is a WEAK COLOUR STIMULUS, a good IoU there is not evidence for
the VOT result* — biting from the other end: a **bad** result there is not evidence either, and
a frame time taken there is not the arm's frame time.

**The rule this bought.** *Before quoting a frame time, read the run's accept/gate summary.*
A frame budget whose `calls/fr` column is not 1.0 on the once-per-frame stages is measuring a
tracker that is not working, and it fails SILENTLY — the number looks better, not worse.
It is `traps.md`'s "metrics that cannot fail a broken tracker" applied to performance rather
than to tracking.

## Controls and provenance

- **Two instruments, one difference.** The console/ssh pair differs in exactly one thing and the
  difference landed in exactly one bucket (above).
- **Same-run `P-03` re-confirmation.** The control-CU probe ran in the same process and again
  reported the user-managed path observing `ap_done` in ~ms while the KDS path pays ~503 ms.
- **The ELF is the shipping arm.** `a31696c5…`; `fixed_point_cost.md` already records it as the
  `SCALE_N=33` twin of the sweep ELF, built from the same sources.
- **No board state was changed.** Root login is locked by `board_provision.sh`'s key-only
  posture and the `petalinux` account demands a forced password change, so rather than set a
  credential the UART runs were launched over ssh with stdout redirected to the board's console
  tty `/dev/ttyAMA0`. stdout is still a tty (line-buffered) and every byte still crosses the same
  115200 UART, so the transport cost is paid identically; the redirect was verified end-to-end
  with a probe string before the first run.

## What was predicted and what happened

Written before the run:

| prediction | outcome |
|---|---|
| console run lands 24-26 ms | **HELD** — 25.82 |
| ssh within ~0.5 ms of the console, **falsifier: a 3.79 ms gap** | **HELD in direction, missed in size** — 0.91 ms. The falsifier did not fire |
| `scale extract` is the largest APU item | **WRONG** — `roi_crop`'s poll is, at 3.4x it |
| `roi_crop launch` ~0.37 ms | **BADLY WRONG** — 7.42 ms. The 0.373 came from the ch16 / 64x64-crop arm and `roi_crop` scales with OUTPUT pixels |
| shipping arm within ~1 ms of the old arm at matched console settings | **HELD, but not comparable** — 25.82 vs 26.29 on a different scene |
