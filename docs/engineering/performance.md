# Performance — resources, history, and where the frame goes

Moved out of CLAUDE.md 2026-08-31; content unchanged.

## Resources and cost

VEK280 `xcve2802-vsvh1760-2MP-e-S`, 12 GB LPDDR4 **of which Linux maps 2 GB, and
512 MB of that is CMA — usable heap is ~0.9-1.2 GB, NOT 12 GB** (see
`docs/thesis/evidence/TODO_board_memory.md`; it cost **5** of 62 sequences on the RGB VOT
arm — the predicted 8 included three that turned out to fit, which is why the
"derive luma on the board" step recovers zero sequences and the streaming reader
was the only fix. `VOT_RESIDENT_MAX_MB` above is that reader).
The device tree declares all three banks; `/proc/iomem` shows only
`00000000-7fffffff : System RAM`, with the 2 GB at `0x8_00000000` reserved and
the 8 GB at `0x500_00000000` absent. The figure below is the PART's capacity and
has been misread as an available-memory budget once already; AIE core clock 1 GHz
(`directives/post_sys_link.tcl`). Measured on the 128×128 ch1 build:

| Resource | Available | Used | % |
|---|---|---|---|
| AIE-ML cores | 304 | 6 | 2% |
| AIE-ML memory tiles | 76 | 1 | 1% |
| BRAM18 | 1200 | 10 | 0.8% |
| DSP | 1312 | 44 | 3.4% |
| LUT | 520704 | 7694 | 1.5% |
| FF | 1041408 | 7539 | 0.7% |

**The design uses 2% of the AIE array.** Check any "we can't afford it on AIE" claim against
that first — the binding constraints have always been tile memory (64 KB/tile) and host DMA
orchestration, never core count. **`runtime<ratio>` is not utilization**: the mapper's
`Utilization` column shows declared budgets, not measured occupancy.

### Per-frame AIE compute (128×128, ch16, from `aiecompiler.log` schedules, post-vectorization)

| kernel | ms/frame | note |
|---|---|---|
| conv2d | 4.1 | 37 → 8.75 cyc/px; the untouched **stream-read loop is now 44%** of it |
| cmul_accum | 0.13 | 30 → 2 cyc/element, now pipelined |
| FFT + IFFT chain | ~2.2 (band 1.3–3.5) | **trip counts inferred, not logged** |
| **total** | **~6.4** | from ~21.6 before vectorization |

These are the compiler's scheduled cycles — real cycles on an in-order VLIW core absent memory
stalls, trustworthy for sizing but not a profile.


### Performance history

| date | change | frame ms | FPS | log |
|---|---|---|---|---|
| — | baseline (console at 115200) | 880 | 1.14 | |
| 08-20 | console gating (`VERBOSITY`) | 180.6 | 5.54 | `run_0820_1528` |
| 08-20 | BO copy pattern + int64 energy | 143.3 | 6.98 | `run_0820_1554` |
| 08-20 | scene on the host | 134.6 | 7.43 | `run_0820_1604` |
| 08-20 | `-O3 -mcpu=cortex-a72` | 132.2 | 7.56 | `run_0820_1610` |
| 08-20 | hypot fix + `-fcx-limited-range` | 127.7 | 7.83 | |
| 08-20 | `FFT_ROW_WS` 8→16→32→64 | 89.5 → 70.9 → 60.7 | 16.48 | `run_0820_1807` |
| 08-21 | scale-extract real DFT, fused filter update+quantize, folded diag scan | 45.60 | 21.93 | `run_0821_1109` |
| 08-21 | memory-tile transpose | 35.58 | 28.14 | `run_0821_1348` |
| 08-21 | software-pipelined roi_crop | 31.48 | 31.81 | `run_0821_1402` |
| 08-21 | `CMUL_SPLIT_ACCUM` | 29.61 | 33.77 | `run_0821_1452` |
| 08-21 | blocked `unpack_spectrum` | 28.64 | 34.92 | `run_0821_1635` |
| 08-21 | tail split onto core 1 (`TAIL_PARALLEL`) | **26.23** | **38.15** | `run_0821_1712` |
| 08-24 | RGB (`CONV_IN_CH=3`) — cost is host memory, not conv2d | 28.58 | 34.99 | `run_0824_1457` |

**Every one of these was accepted on a bit-identical-tracking test**, and that criterion has now
caught two bugs it was not designed for (the `g_target_shift` race, the `FFT_COL_WS` datapath
check). A tolerance-based check would have shrugged at IoU 0.48.

### Where the frame goes — gray 26.29 ms, RGB 28.58 ms

```
                  gray    RGB      (VERBOSITY=0, run_0821_1725 / run_0824_1457)
APU subtotal    15.639  17.442
of which OVERLAP -2.762  -2.746   <- ran CONCURRENTLY on core 1
APU wall        12.876  14.696    <- what the frame actually spent
GMIO            11.134  11.133    (async 6.6 / wait 4.5 — UNCHANGED by RGB)
roi_crop launch  1.013   1.471    (= channel 0's Stage A, structurally exposed)
UNATTRIBUTED    +1.261  +1.284
```

**The frame is 84% CPU-BOUND**, not wait-bound: host CPU = APU 15.4 + GMIO async 6.6 + roi_crop
1.0 + unattributed 1.2 = 24.2 ms of a 28.7 ms frame (measured pre-threading). Only 41% of GMIO
blocks. **The second core's value is splitting 24 ms of work, not filling 4.5 ms of gaps**;
perfect two-core use floors at ≈12-15 ms, 65-80 FPS. That slack is also why RGB is nearly free
(see "RGB costs what the HOST pays").

**Overlap accounting is measured, not estimated.** With H the helper's elapsed time and W the
main thread's time blocked in `join()`, the region's WALL cost is (main's own work + W) while the
slots credit (main's own work + H), so the double-count is exactly **H − W**. It validated
itself: overlap 2.762 against a predicted `min(4.80, 2.89)` = 2.9, and the residual returned to
+1.261 against +1.17 before threading — a correction tuned merely to erase a negative number
would have had no reason to land there.


### Parallel-for inside `filter_update_quantize` — ATTEMPTED AND ABANDONED (2026-08-21)

**~0.96 ms (3.6%) was not worth what every formulation cost in bit-exactness.** The gain was
bounded before it started: the tail is `max(scale 2.89, filter 4.80) + publish 1.91 = 6.71 ms`,
and **`publish` cannot move to the other core** (it consumes `filter_scratch`/`q15_scale`), so
splitting the filter internally only helps on its last 1.91 ms.

Three formulations — split by element range, split by channel, and a `noinline` wrapper — were
each bit-exact under `-O2` and each 1 ulp off under `-ffp-contract=fast`, always the same
82/1024 elements of **A** at 1.49e-08. **Root cause: GCC's FMA contraction is sensitive to
INLINING CONTEXT, not just to the expression** — at `nw == 1` the worker body inlines, at
`nw > 1` it is emitted out-of-line, and the two contract `eta*conj(G[i])*f[i] + keep*a[i]`
differently. 1 ulp of A was disqualifying because A is carried frame to frame at eta = 0.125, so
it settles at ~8 ulps and flips occasional bins across 200 frames — tracking would come back
*nearly* identical, the one outcome that makes the bit-identical criterion useless.

**What survives for a next attempt.** (1) `make test_host`'s `-ffp-contract=fast` second build
found every one of these; `-O2` found none. (2) An element-range split does protect B's
channel-order sum. (3) The `|H|` max scan needs a lowest-global-index tie-break to be split at
all, since `std::abs()` is `hypot()`. (4) **Parallelise across FUNCTIONS, not inside them** — the
`TAIL_PARALLEL` split worked precisely because it moved a whole function and touched no
arithmetic.


### hw_emu frame times — MEASURED, and they do not scale from ch1

```
                 ch1        ch16
per channel    ~50 min    ~43 min
per FRAME      ~50 min    ~11.5 h
ITER_CNT=2      ~1.7 h      ~23 h
```

**hw_emu wall clock does not track AIE compute.** The echo-mode ch16 run, where conv2d did no MAC
work at all, still took ~14 h/frame; the emulator is simulating the PL and the DMA/NoC traffic.
Vectorizing a kernel speeds up the *design*, not the emulation of it. Always size runs from
measured wall clock at the same `N_CHANNELS`.

