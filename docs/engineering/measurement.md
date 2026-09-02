# Measurement methodology — the rules that were paid for

Moved out of CLAUDE.md 2026-08-31; content unchanged.


Two principles that have repeatedly earned their keep: **instruments before changes**, and
**never move two magnitudes at once**.

- **NEVER SIZE A CHANGE FROM ONE MEMBER OF AN INTERLEAVED async/wait GROUP. SUM THE GROUP.**
  The first `wait()` in an interleaved pair absorbs the other's production latency. This has
  appeared four times: `gmio_fft_row_out` absorbing the weights feed (286 µs/tx vs siblings'
  18 µs), the `fft_col_out`/`accum_out` pair under `FFT_COL_WS` (single ports looked like a win;
  pair total went 9.00 → 18.32), `gmio_cmul_in` absorbing both inputs after the port split, and
  `CMUL_ACCUM_MEMTILE`. **It is the single most repeated measurement error in this design**, and
  it caused a whole optimisation to be built on an artifact.
- **`DMA_TX`/`DMA_T` must time `async` and `wait` separately.** Fused, every port figure is a
  lie about mechanism. Splitting them revealed 6.6 ms/frame of host descriptor cost that was
  invisible.
- **Measure the total and print the residual.** A profiler that does not account for the whole
  frame lets you conclude confidently and wrongly — twice here.
- **Instrument the two candidate mechanisms in ONE run and let the log print the verdict.** The
  `poll(state)`/`wait()` split cost one hardware run and **retired the planned fix before it was
  built on**. Hardware access is the scarce resource; a measurement that can only confirm your
  hypothesis is worth less than one that can also kill it.
- **Two independent instruments beat one instrument twice.** `/proc/interrupts` reading 0 is
  consistent with "no interrupt raised" *and* "raised but never delivered". The CU's own
  `ISR=0x3` (toggle-on-write) discriminates them outright.
- **Keep a known-good comparator on the old path when you change a mechanism.** camera_capture —
  no AXIS port, ~6 µs of work — paying the same 512 ms turned "roi_crop is slow" into "any CU
  completion is slow"; after the fix, the *same* probe in the *same run* still paying 512 ms is
  what proves the fix is the fix.
- **A MULTI-START CSV COLLIDES ON FRAME INDEX, AND IT SILENTLY DISARMED THE RAILS GATE.**
  `calib_report.py`'s `parse_csv_frames` keyed by frame number, so one `track.csv` holding 15
  anchors collapsed 8434 rows to 742 and reported **4 railed frames where there were 266** — a
  66× under-report, printed next to a confident "BUDGET IS WRONG" that was right for the wrong
  reason. Fixed 2026-08-26 by keying on `(job, frame)`. Any per-frame reader of a
  `FRAME_SOURCE=vot` CSV has this bug until shown otherwise; `job` is a column for exactly this
  reason and a bare frame index is not a key.
- **AN OFFLINE R CAN BE RAISED BY DEGRADING THE FILTER — DEMONSTRATED, NOT ARGUED
  (2026-08-28).** A deliberately broken init (`--warp-mutant gsign`, G centred at `+`the crop
  offset instead of `−`) scored `vot_ar_offline` **dR +0.0525 — 6.5× the correct arm** — while
  tracking measurably worse by every direct measure: A −0.0975, mean IoU 0.1792 → 0.1683,
  frames above IoU 0.5 17.3% → 15.4%. A weak, under-confident filter reports smaller
  displacements and stays near where it started, which survives the 10-consecutive-frame rule
  while overlapping badly throughout. **Never accept an arm on R alone: require A not to fall,
  or price the fall.** dR > 0 with dA < −0.02 is this artifact until shown otherwise. (`dec2`
  at dR +0.0567 / dA −0.0086 passes; the mutant at +0.0525 / −0.0975 does not, and the two are
  otherwise the same shape.) It also did its intended job — the bench is demonstrably NOT blind
  to warp geometry, which is what makes the init null readable.
- **ACCURACY IS AVERAGED OVER TRACKED FRAMES, SO A LONGER-SURVIVING ARM IS SCORED ON HARDER
  ONES — score A on the COMMON survived prefix before calling an accuracy drop real.** The
  spatial-mask arm reads `dA = −0.0346` pooled, which trips the artifact test above; on the
  frames BOTH arms survived (identical 5563-frame set) it is **−0.0085**, so three quarters of
  the "loss" is the extra 24.7% of harder frames it reached. The same structural point is
  already made about RGB ("a tracker that survives longer is scored on harder frames"), but it
  was never turned into a control. It is one `min(progress(x), progress(y))` per sequence, and
  without it a robustness win looks like an accuracy regression.
- **`R = 0.3435` (the gsign mutant, OFFLINE) IS NOT `R = 0.3417` (the SHIPPING ARM).** They
  agree to three decimals by coincidence and mean opposite things. Every `vot_ar_offline`
  number is the toolkit's RULE on SINGLE-START runs, is biased ~0.02 high in R, and is
  comparable only between arms scored identically — so the offline baseline at R 0.2910 is
  **not** a regression from the hardware 0.3417.
- **Test an analysis tool against an OLD log before the run it was written for.**
  `calib_report.py` was written to parse a calibration run, then pointed at an existing board
  log: its frame regex was anchored at line start, but board logs are captured through
  `picocom … | ts`, so every line carries a timestamp. Anchored, it found ONE frame and reported
  "no rails" with a straight face. A parser that finds nothing looks exactly like a clean run.
- **Timestamp the console instead of adding timers.** `picocom … | ts '%H:%M:%.S' | tee log`
  needs no rebuild. That localised 7.66 s to a single interval in one run, after two wrong
  inferences from reasoning-by-elimination. Do this *before* writing any in-code profiler.
- **Benchmark a host-side change on the host.** `-fcx-limited-range` gave 1.6× on x86 and 0.48 ms
  on the board, because `filter_update` streams ~8 MB/frame — inside a desktop L3 (compute-bound)
  and outside the A72's (memory-bound). **The ORDERING did not transfer, not just the
  magnitude**, because the working set crossed a cache boundary between the machines. Same for
  the blocked `unpack_spectrum`: x86 said 4.66×, the A72 gave 1.63×.
- **Test an AIE placement question with `make graph`, not with a 25-minute package.** The
  prediction that a 64 KB ping-pong would not fit was wrong (AIE-ML cores address neighbouring
  tiles' memory) and cost 3 minutes to disprove.
- **A knob that won 7× on one port lost 4× on another with no warning.** `FFT_ROW_WS` took
  `gmio_fft_row_out` 73.22 → 9.93 ms; the identical change on `gmio_fft_col_out` went the other
  way. The "overhead- vs production-dominated" discriminator has no term for where the mapper
  puts the buffers. **Do not generalise a windowing result from one port to another without a
  hardware run.**
- **Write predictions down before the run.** The `FFT_COL_WS` sweep was 2 of 3 right and
  catastrophically wrong on the one that mattered; the memtile transpose matched its frame total
  while its attribution was completely wrong (conv2d's production reappeared in `roi_crop
  launch`, 0.067 → 5.196 ms). Checking *where* the time went, not just the total, is what turned
  that into the next item.
- **`make test_roi_crop` and the HLS report cannot see launch-path bugs.** Both are correct about
  the datapath. The 503 ms was between two `printf`s, untimed.
- **An existing constraint does not become inapplicable because the code around it changed.**
  The memtile rewrite queued all `CONV_INVOCATIONS` asyncs on `gmio_weights`; XRT allows **one
  outstanding async per port**, the identical error a depth-2 drain probe had already hit. Every
  async site in `mosse_tracker.cpp` is now audited — 11 pairs, one outstanding per port.
- **For a thread launch, disjoint state is not enough.** The right question is not "does the
  other path TOUCH anything this one touches?" but **"is everything this helper READS already
  written when it starts?"** The first `TAIL_PARALLEL` attempt passed the first question and
  silently corrupted tracking (mean IoU 0.9188 → 0.4794): `filter_update_quantize` consumes
  `g_target_shift`, which was filled ~200 lines later, so it trained on the previous frame's
  target. The instrument written to explain that bug then had the same bug — it sampled
  `g_ap_us[AP_FILTER]` before `join()`. Cost differed sharply: the first took a full
  build-flash-run to find; the second only broke a diagnostic and **failed silently rather than
  printing a plausible wrong number**. Failing loudly-or-not-at-all beats failing plausibly.



## The offline proxy bounds SAMPLING noise, not TRANSFER — 2026-09-02

`vot_ar_offline.py`'s ~0.02 resolution in R was already measured. This adds a sharper limit.

| axis | offline | hardware | |
|---|---|---|---|
| sigma 2 -> 4 | +0.0808 dR | +0.0678 | transferred, 84% |
| 64x64 map (`dec2`) | +0.1071, P(dR<=0)=0.000 | +0.0456 | transferred, 43% |
| Layer-1 + ReLU | borderline, P=0.041 | trim-5 stable, P=0.011 | UNDER-called |
| spatial mask | +0.0601 | +0.0192 | over-called 3x |
| `pad30` | large | ~0 | over-called ~11x |
| **`MOSSE_ETA` 0.05 -> 0.1** | **+0.0481, trim-3 +0.0097, P=0.021** | **−0.0030 R, EAO −0.0143** | **INVERTED** |

The eta cell was the ONLY one of 22 in its grid to survive a symmetric trim AND a bootstrap, and
it still inverted. **A trim says the result is not carried by three sequences. A bootstrap says
it is not sampling noise. NEITHER says the bench models the tracker.** `P(dR<=0)` is necessary
and never sufficient.

**The practice that made this cheap rather than mysterious:** the transfer assumption was written
down BEFORE the run (`proposed_build_l1relu.md` sec.11) — the grid ran at 128x128/sigma 4, the
arm at 64x64/sigma 2, and `sigma/target` was established to govern SIGMA and nothing else. When
the arm failed, the suspect was already named. **Do that for every arm screened at a geometry the
board does not run.**


## BOUND THE PRIZE BEFORE ATTRIBUTING THE DEFECT — 2026-09-02

A full day went into the scale filter: it is frozen on ~90% of real frames, its detector gain
against the correction actually warranted is **−0.003** where the position detector scores 0.93,
and the cause is a self-confirming loop fed by a scale-NORMALISING feature. Every step was
correctly measured and reproducible.

**Then the prior question was asked, and it took ten minutes on data already on disk.** Replace
the box SIZE with ground truth on the board's own trajectories, keep the tracker's centre,
re-apply VOT's rule (`scripts/scale_oracle_bound.py`):

| arm | R as tracked | R with a PERFECT scale filter | dR |
|---|---|---|---|
| `l1relu` (shipping) | 0.4536 | 0.4559 | **+0.0023** |
| `sigma4` | 0.4549 | 0.4460 | **−0.0089** |

Mean IoU rises **+0.054** on both and converts into nothing, because VOT fails a run on POSITION
(IoU <= 0.1 for 10 frames) and resizing a box that is not on the target rescues nothing.

**THE RULE: before attributing a defect, bound what removing it is worth.** An ORACLE over the
trajectories already collected is nearly always available, costs minutes, and is strictly cheaper
than the attribution it may retire. Attribution tells you WHY something is broken; an oracle tells
you WHETHER to care. Do the second one first.

**A corollary the same day paid for twice:** an instrument that cannot show the failure cannot
score its fix. `scale_loop_sim`'s raw-feature alpha is 0.174-0.83 where the board's is −0.003, so
it never reproduced the defect — yet it was proposed, and used, as the acceptance test for a
repair. This project had already written that rule down for `rgb_vs_gray_loop.py` having no scale
filter at all. **Check that the instrument reproduces the defect BEFORE it scores a candidate**,
one level up from where the rule was first learned.
