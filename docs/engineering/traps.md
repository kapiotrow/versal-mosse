# Known issues and traps

Moved out of CLAUDE.md 2026-08-31; content unchanged.


### Measurement / build hygiene

- **`runs/.last_cfg` IS NOT AUTHORITATIVE. THE FLAGSTAMPS ARE.** For `run_0820_1418.log`,
  `.last_cfg` recorded `ITER_CNT=500` and 4-3-3; the run executed **200 frames at 4-4-4**. The
  stamps are written by the recipe that runs the compiler, so they cannot disagree with the
  binary. **Diff the stamp against `AIE_FLAGS` before an expensive run:**
  ```bash
  printf 'printvar:\n\t@echo "$(AIE_FLAGS)"\n' > /tmp/pv.mk
  make -f Makefile -f /tmp/pv.mk printvar TARGET=hw | tail -1 > /tmp/aie_now
  diff <(tr ' ' '\n' < /tmp/aie_now) <(tr ' ' '\n' < build/hw/.../aie.flagstamp)
  ```
  Quoting differs; only VALUES matter. `make -n` cannot answer this — the stamp is a `FORCE`
  prerequisite. Also grep the compiler log directly:
  ```bash
  grep -o 'CONV2D_ECHO_TEST=[0-9]' build/$TARGET/${PATCH_ROWS}x${PATCH_COLS}/ch$N_CHANNELS/aiecompiler.log
  ```
  Flag-only changes have silently reused a stale `libadf.a` and produced convincing false
  results. Order matters for `:=` in the Makefile — a stamp variable defined above its inputs
  expanded to `/aie.flagstamp` and silently disarmed a test. See [[feedback-verify-the-build-ran]].
  **`scripts/calib_build.sh` does this check for you** and refuses to declare a build good when
  the stamps disagree, so prefer it for any run worth hours. Every `.xo` now carries a stamp too
  (`crop.flagstamp`): `CONV_IN_CH` touches no source file, so a source-only prerequisite list
  would have shipped a grayscale `roi_crop.xo` inside an RGB build.
- **A stale `Map_Report.csv` in the build dir survives a failed compile** and still shows the
  previous run's healthy table. Only the compiler's `ERROR:` line is true.
- **Check `CONV2D_MODE` before every expensive run.** The default was `1` (echo) until
  2026-08-14 and nothing in the build output says so; it cost a ~28 h ch16 baseline. In echo mode
  conv2d returns at the top: no MAC, no ReLU, no B1, **no Hanning window** (so B2's 9-bin
  identity does not hold), and all 16 channels are bit-identical (so the accumulator sums 16
  coherent copies and every amplitude and PSR figure is inflated).
- **NEVER KILL A BOARD RUN MID-SEQUENCE — THE NEXT PROCESS STALLS, AND KILLING THE LEFTOVER DOES
  NOT CLEAR IT.** Interrupting a sweep leaves the free-running AIE graph and the XRT device
  context inconsistent. The next run then hangs: both threads in `clock_nanosleep`, ~33% of one
  core, `roi_crop`'s CU reading `0x20E` (`ap_done=1 ap_idle=1`, so the CU is NOT the blocker),
  and ZERO frames produced for 14 minutes on a sequence that takes 14 SECONDS. `vot_sweep.sh`'s
  own leftover-kill ran first and the fresh process stalled anyway — **a reboot is the only
  clear**. Two things hide it and neither is a fault: the board's stdout is block-buffered at
  4 KB, and `CSV_FLUSH_EVERY=200` holds the rows, so a healthy long sequence and a hung one look
  identical from the log. **Watch the TRAJECTORY COUNT.** Cost: one aborted 62-sequence arm.
- **A CROSS-IMPLEMENTATION CHECK IS THE ONLY THING THAT CATCHES A CONTRACT MISMATCH.**
  `filter_box_energy_fraction()` (board) and `box_energy_fraction()` (offline) were described as
  "written against the same definition"; the board half omitted the inverse transform, so it
  measured a spatial statistic on a frequency-domain array and read **0.0000 on every frame of a
  whole sweep**. Five unit tests were green throughout, because they fed it SPATIAL arrays —
  self-consistent with the bug and agreeing with the caller about nothing.
  `scripts/check_ebox_crosscheck.py` runs BOTH implementations on the SAME H and is
  mutation-tested (5 mutants, all caught). Same family as the `generate_scenario` trap in this file:
  zero-tolerance comparison proves nothing when both sides share an assumption.
- **LAUNCHING OVER SSH CHANGES THE FRAME-TIME MEASUREMENT.** `scripts/vot_sweep.sh` drives the
  board over ssh, which moves the ELF's stdout off the 115200 console. That console is itself a
  distortion — 15% of the frame at `VERBOSITY=0`, **58% on `animal`** — so ssh frame times are
  more honest AND **not comparable to any run before 2026-08-25**, `run_0821_1725` included.
  `ts` on the PC side of a TCP stream is good to about a second: it locates a stall, it is not
  the instrument `picocom … | ts` was. Take frame time from the `AP_*` slots and `track.csv`, and
  quote FPS only from a serial-console run. Incidental gain: ssh without a pty emits clean `\n`,
  where picocom's bare `\r` made `readlines()` and `grep` disagree about line numbers.
  See `docs/thesis/evidence/automation.md`.
- **`debugfs`'s `mkdir` ALLOCATES THE INODE BEFORE it fails on an existing directory**, so
  re-running an image-provisioning script leaves an unconnected inode and a filesystem `e2fsck`
  calls dirty. Test-then-create. The read-back verification passed both times — only the closing
  `e2fsck -fn` caught it, which is the argument for ending image surgery with a filesystem check
  rather than with a content check.
- **hw_emu packaging stalls on `udevadm settle` when the card reader is plugged in** — ~120 s per
  repeat, turning a 45 s package into 10 min. Not a failure. Unplug the reader.
- **hw_emu wall-clock timings are not hardware timings — but hw_emu SIMULATED PL CYCLES are.**
  The host runs on QEMU (`Runtime.hw_em_driver`), so host-side latency there is meaningless
  (one run reported 210,925,994 µs/tx). The PL is simulated at RTL, so `ap_start`→`ap_done`
  counts transfer directly. Use `make debug_sim && make probe_emu` (defaults
  `PROBE_CU=roi_crop_0 PROBE_PORT=patch_out`, 26 signals). A 64×64 `N_CHANNELS=2 ITER_CNT=1` run
  costs ~1-2 h and exercises both `recompute=1` (ch0) and `recompute=0` (ch1). The VCD is written
  incrementally, so a killed run still parses. `ap_int/ext/str_blocking_n` do not resolve in this
  build; use sub-loop `ap_done` plus the handshakes.
- **The `NOTE: hw_emu wall time is not real hardware time` line is UNCONDITIONAL** (no `TARGET`
  guard) — it is not evidence a run was emulated, and it caused genuine hardware numbers to be
  discounted. Now guarded behind `HW_EMU_BUILD`.
- **`SIM_WALL_TIMEOUT` scales with patch area** (`SIM_PATCH_SCALE`); a timeout looks exactly like
  a deadlock. Check for `Error 124` before concluding "deadlock". Check the right process:
  `ps -C aiesimulator` shows only bash wrappers at 0% CPU; the simulator is `aie2simmsm`
  (~190% CPU, ~2.5 GB RSS). A quiet log is normal — 10+ minutes between prints.
- **The offline model's ratios and orderings are sound; its absolute magnitudes are
  patch-specific.** `phase1_sweep.py` runs on the s6 patch; hw_emu injects a synthetic target
  through real `roi_crop`, which after log/z-score is far hotter — it predicted accum 162 where
  hardware gave 5264. With `mean_prev` seeded the model agrees to 3-11%.

### Metrics that cannot fail a broken tracker

- **PSR is a weak pass criterion, and in a specific direction.** A tracker 179 px off target,
  confidently locked to background, reported **PSR 33**. One run: `19 evaluated, 19 accepted, 0
  gated`, PSR min 15.95 / mean 26.60 — while IoU fell to 0.0656. The response really was sharply
  peaked, just in the wrong place. **IoU is the only metric in the harness that can fail a
  confidently-wrong tracker.** Read `track.csv`, not the console.
- **`[diag] F_ch` IS CHANNEL 0 ONLY, not a bank maximum.** It is printed under `if (ch == 0)`
  (in `mosse_tracker.cpp`, under the per-channel `if (ch == 0)`), while `accum` and `response`
  are bank-wide. So "F_ch looks
  comfortable" says nothing about the other 15 channels, and a hot channel elsewhere is
  invisible. It matters most at `CONV_IN_CH=3`: **ch0 is one of the four colour-opponent
  channels** (0/2/9/10), so its amplitude is the one number in the console that discriminates a
  real colour path from a colour-free one. Predicted input-referred gain for ch0, from the
  exported weights: gray 8.72, RGB `FRAME_RGB_MODE=1` 10.87 (1.25×), RGB `FRAME_RGB_MODE=0`
  0.80 (**0.09×, near-dead**). If ch0's `F_ch` does not collapse ~10× between those two RGB
  arms, colour is not reaching conv2d.
- **`err=0 px` is a weak pass criterion too.** It cannot see mainlobe width, drift, a DC
  pedestal, or a gated frame (where a mismatch is a *pass*).
- **A centred test impulse cannot validate localisation** — `peak_detect_sw`'s old scan returned
  index 0 on an all-zero response, i.e. the right answer produced without reading the data. The
  impulse is injected at `pos + (IMPULSE_DR, IMPULSE_DC)` = (10,−7), asymmetric and
  opposite-signed so a transpose and a sign flip are both caught.
  See [[impulse-test-pattern-is-degenerate]].
- **`[MISMATCH vs injected offset]` is meaningless under `TRAJECTORY=1`** — the criterion derives
  `exp_dr` from `IMPULSE_DR`, so it fires on healthy frames.
- **Two different statistics are both called PSR.** The aiesim `snr_ratio_pct` is
  `|peak| / max|sidelobe|`; Bolme's is `(g_max − µ_sl) / σ_sl`. Same 11×11 circular exclusion,
  different statistic — they differ by several times and neither's thresholds transfer.
  `report_psr()` prints both, labelled. Bolme's is the meaningful one for occlusion.
- **PSR must exclude the mainlobe or it asserts nothing** — a neighbour of a σ=2 peak sits at
  0.88 of it. Use a *circular* distance, since the map wraps.
- **s7's PSR threshold is geometry- and budget-dependent.** 15× was calibrated at 64×64 / 3-0-6 /
  ch1; at 128×128 / 4-3-3 / ch1 the measured ratio is 11.8. A FAIL from that threshold alone is
  not evidence of a defect until it is re-derived. **ch1 is the documented worst case** —
  channels add coherently, their quantization noise does not.
- **An ordering inferred from one failing run is a hypothesis, not a finding.** This cost the
  most time on this project. Only a run where one of the two candidates is known-good can order
  them.

### Correctness traps

- **THE FILTER MUST BE TRAINED AGAINST A G CENTRED AT THE MEASURED DISPLACEMENT, NOT AT (0,0).**
  Fixed 2026-08-20. `filter_update()` was passed `g_target` — the Gaussian centred at (0,0) —
  while `g_F_all` is the patch cropped at the **pre-update** position, where the target sits at
  `(dr,dc)`. Every accepted frame taught "a patch with the target at (dr,dc) peaks at (0,0)",
  compounding at `eta` until the zero-shift peak won. Mean IoU 0.1708, 181 of 199 frames lost.
  **The sign, derived rather than guessed.** Let `Q_t` be the patch cropped exactly ON the
  object; the patch actually held is `P_t = Q_t` shifted by `+d_t`. Correlation is shift
  equivariant, so an on-target patch must peak at 0 ⇒ **G is centred at `+d_t`, the SAME sign as
  the detected peak.** That derivation **predicts analytically** the observed law
  `resp00_over_peak ≈ 1-(1-eta)^k` — i.e. it is governed by the LEARNING RATE, not the scene.
  **Fix**: a per-frame `g_target_shift` from `gaussian_target_spectrum(..., psr_abs.dr,
  psr_abs.dc)`. By the shift theorem this IS "re-crop at the new position", exactly and for free.
  `filter_init()` on frame 0 keeps the centred `g_target` — that crop really is centred. The
  exact alternative is a real re-crop at 16 more roi_crop launches ≈ 77 ms/frame.
  **A SINGLE UPDATE CANNOT SEE THIS DEFECT** — the one-shot golden check passed throughout —
  which is why both regression tests are closed loops (`scripts/mosse_loop_sim.py` and
  `run_training_target_tests()` in `make test_host`).
  **Assert the shape, not an absolute level.** In the 32×32 test both arms start at 0.39 — that
  scene's own zero-shift autocorrelation, not a defect. What is geometry-independent is that the
  defect makes the ratio GROW at the learning rate while the fix leaves it flat.
- **Background lock was the WRONG explanation for the above** — kept because the measurements
  are real and the mechanism exists. `fill_background()` is cached and only the dirty rect is
  restored, so outside the target the synthetic frame repeats to the LSB, and a DCF fed a
  perfectly repeating background correlates with it at exactly zero shift. Measured 2026-08-17:
  the static peak was worth 69-86% of the true one and won 21 of 48 frames, each win costing a
  permanent ~9.4 px offset (centre error 1.35 → 9.56 → 87 → 292 px) **while PSR read 24-35
  throughout**. **`FRAME_NOISE` is not the fix** (independent additive noise cannot decorrelate
  a static pattern). **`BG_PAN` is the right instrument** and measurably works — but changed the
  tracker not at all, which is what refuted this explanation. **Sweep its magnitude against the
  texture's wavelengths, not in pixels** (`scripts/bg_pan_sweep.py`): corr@0shift falls +0.60 →
  **+0.09** at 31,47 px/frame; the obvious guess of 3-5 px/frame is worthless, the texture's
  shortest wavelength being 180 rows. `fill_background()` rounds frequencies to WHOLE CYCLES per
  frame so the pan wraps seamlessly (else an **8.10 LSB** row-wrap discontinuity, and an
  artificial edge is exactly what a DCF locks onto); 31 and 47 are coprime to 1080 and 1920.
  **This fixes the TEST BENCH, not the tracker** — do not tune the tracker against it.
  **Discriminating background lock from a DC pedestal** (similar look, opposite fixes): a
  pedestal lifts every bin uniformly; background lock is a *localised blob* at the origin with
  sidelobe mean ≈ 0.
- **THE FRAME BUFFER WAS NEVER SEEDED WITH THE BACKGROUND. Fixed 2026-08-18** — one 2 MB
  `memcpy` at startup, which **must** come after `rc_control_cu_probe()` (which zero-fills
  `frame_bo` by design). **The interesting part is the correction.** The measured "88.53% of the
  ROI never written" came from an *offline replay*, which assumed the unwritten region held
  garbage that saturated Stage A's int8 rail and inflated σ 4.3×. On hardware that region is
  **zeros**. Hardware before/after: `F_ch` frames 0 and 1 are **unchanged** — including frame 0,
  the one the filter trains from — and frames 2+ rise 2.55-3.10×. The fix is real; the predicted
  9.2× response gain never existed, and the shift-budget change it justified was wrong.
  **The lesson: an offline replay inherits every assumption the host makes about memory it did
  not write.** Verify the premise on hardware before letting a model move a calibrated constant.
- **`mean_prev` seeding** (`mosse_tracker.cpp`, before the first `weights_bo.sync`): seed
  `mean_prev = bias_acc >> out_shift` for every channel at startup, since Stage A delivers a
  zero-mean patch. Without it Stage B1 is inert on frame 0 — the one frame the filter trains
  from — and the ch16 response rails flat. Fixing it took `F_ch` from 32768 (railed, 11 bins) to
  53, accum 5264 → 70, peak/sidelobe 3.59 → 23.04.
- **Conjugation: the stored filter is H, not Bolme's H\*.** `cmul_accum` conjugates itself, so
  the host stores `H = conj(G) ⊙ F / (B + ε)`. Storing Bolme's expression verbatim gives a
  phase-noise response peaking at an arbitrary bin — **invisible whenever the target is
  centred**, since a centred real Gaussian has `conj(G) = G`. That is why s7's target is
  off-centre.
- **H's quantization ceiling is not `2^H_SHIFT`.** The ceiling sets H's *resolution* (always use
  all 15 bits); `H_SHIFT` sets the *product scale*. Coupling them trades one bit of filter
  precision per bit of accumulator gain, i.e. does nothing. **max|H| sits where |F| is
  smallest**, because that is where the regularized inverse peaks — at `H_SHIFT=15` the
  accumulator reached 15 of 32767 and PSR collapsed to 5.2 while still localising exactly. The
  contract is encoded in four files: `mosse_filter.cpp`, `gen_filter_golden.py`,
  `test_mosse_filter.cpp`, s7 in `gen_aiesim_vectors.py`. Normalization is by complex
  *magnitude*, so `rails=1` on one frame and `rails=0` on the next with `max|.|=32767` both times
  is not a bug (a bin at 45° puts 32767/√2 in each part).
- **The correlation response is SIGNED once Stage B1 is active** (s6 peaks at `{-417,0}`). The
  scan is `|real|`; both peak definitions are computed every frame and a disagreement reported.
- **B2's correction is not bit-exact.** The linearity argument is exact in real arithmetic, but
  conv2d's window multiply applies two `>>15` truncations. Residual ~1e-3 (vs 2.5e-2…9.9
  without). Fine for argmax. With `mean_prev` seeded B2 is currently a no-op
  (`max|removed| = 0`), which is the desired state.
- **DSPLib's cint16 FFT loss is additive, not a gain factor** — each pass subtracts ~21 from a
  summed DC bin, independent of amplitude. So `row_dc = PATCH_COLS*c − 21`,
  `accum0 = PATCH_ROWS*row_dc − 21`. Any "expected = N" calculation is wrong. An impulse loses
  ~3. (A "2/3 gain" fits one data point by coincidence — don't fit a scaling law to one point.)
- **A per-element `std::abs`/`std::norm` on a complex is a TRANSCENDENTAL CALL.** `std::abs()` on
  a complex is `hypot()`, and `filter_quantize_q15`'s max scan ran it 262144×/frame. No compiler
  flag short of `-ffast-math` touched it. Fixed exactly: scan on `re²+im²` (monotone in `abs()`,
  so it selects the same element), then take **one** square root. 8.71 → 1.88 ms, 4.6×. Grep for
  it before profiling anything else in this file.
- **`std::complex<float>` blocks vectorisation** — C99 Annex G forces the libgcc `__mulsc3`
  helper. `-fcx-limited-range` removes it (NEON fp ops 10 → 17) and only drops Inf/NaN range
  handling in complex mul/div, unlike `-ffast-math` which makes every float op in the file
  unsafe. Worth 1.6× on x86 and 0.48 ms on the board — see the benchmarking rule above.
- **Never compute on a BO mapping.** The mappings are write-combining and the asymmetry is
  extreme: memcpy heap→heap 7359 MB/s, **BO→heap read 696**, heap→BO write 3470; summing int16
  on a BO mapping is 5.8× the heap rate, and **fp64 on the same buffer is another 4.02×**. Bulk
  `memcpy` out, compute on the heap copy, `memcpy` back only if needed. This was worth 33 ms/frame
  across five sites. Two independent sub-fixes: **the energy accumulator should be int64, not
  fp64** (sum of 16384 int16 squares peaks at 1.8e13 — exact in int64, and 4× faster on the same
  memory; the old `double` was already carrying exact integers, so the switch is bit-exact), and
  `filter_update` is **pure heap already**, so the BO fix does nothing for it.
- **Access PATTERN, not traffic volume, was the lever on `filter_update`.** Flipping the B loop
  to channel-major moves **exactly the same bytes** — it changes only the stream count, from 16
  strided readers 128 KB apart to one sequential reader. On an A72 whose prefetcher tracks a
  handful of streams, that was ~7 ms/frame. A traffic-volume model would never have found it,
  and did not.
- **A SELF-CONSISTENT TEST CAN PASS ON CORRUPTED DATA. Found 2026-08-23, and it is
  the best argument in this file for single-sourcing an offset.**
  `generate_scenario` patched `mean_prev` into the scenario's `weights_ch0.bin` at
  a hardcoded byte 18 — correct at 9 taps, and **inside the B plane at 27**. At
  `CONV_IN_CH=3` it therefore overwrote taps [18:22] with an int32 and left the
  real `mean_prev` field at 0, so Stage B1 was silently disabled AND three of the
  27 taps were garbage. **The bit-exactness check passed anyway**, 16384/16384,
  because the kernel and the model both read the same corrupted file. The only
  symptom was `mean_prev=0` printed where the generator had computed 4842 — a
  number nothing was asserting on. Zero-tolerance comparison is not enough when
  both sides share an input; the INPUT has to be pinned too.
- **THE CONV WEIGHT LAYOUT IS NOW SINGLE-SOURCED. The rest of this trap still stands.**
  Fixed 2026-08-23. The layout lives in `design/aie_src/conv_weight_layout.h`, derived from
  `CONV_IN_CH`, and is mirrored formula-for-formula in `scripts/conv_weight_layout.py`. It used
  to be four hardcoded copies of `[0:9]/[9]/[10:14]/[14:18]/[18:22]`.
  **RGB is what forced it: 27 taps overrun ALL FOUR grayscale fields**, so a `CONV_IN_CH=3` file
  read by a `CONV_IN_CH=1` reader takes `out_shift` out of the G plane and `bias_acc` out of G/B
  taps — no crash, sixteen plausible channels, a meaningless tracker. **Byte 63 of every channel
  buffer carries the layout tag** (= `CONV_IN_CH`; 0 in pre-tag files, read as grayscale), and
  three independent guards were claimed on a mismatch — and **one of the three is inert**.
  Corrected 2026-08-25: the `#error` in the generated `layer0.h` CANNOT fire, because no
  translation unit in either the AIE or the host build includes that header
  (`grep -rn layer0.h design/` returns nothing but the file itself). What actually fires is
  the **runtime tag check in the host**, before any weight byte is read, and `SystemExit` in
  `check_collapse.py` / `gen_aiesim_vectors.py` / `phase1_sweep.py`. The runtime check caught
  a real one on 2026-08-25: an RGB `layer0_weights.bin` (tag 3) on the SD card under a
  `CONV_IN_CH=1` ELF — `FATAL: layer0_weights.bin ch0 has layout tag 3`. **The dead guard went
  unnoticed precisely because a live guard covers the same case**, which is the argument for
  testing each guard separately rather than testing that the case is caught.
  **The weights are a RUNTIME data file**: nothing includes `layer0.h`, and `hanning_128.h`
  does not depend on `CONV_IN_CH`, so switching arms needs `make weights CONV_IN_CH=<n>` and a
  1 KB file copy onto the card — no re-synthesis, no re-package, no re-flash.
  Gray resolves to the historical offsets; **proven, not assumed** — preprocessing
  `conv2d_kernel.cpp` at `CONV_IN_CH=1` from `HEAD` and from the working tree and folding the 18
  constant index expressions gives identical text.
- **Preprocessing constants are coupled across engines with no compile-time check.**
  `hanning_*.h` must stay periodic;
  `ROI_NORM_Q` in `roi_crop.h` sets the int8 scale `out_shift` was derived against;
  `FFT_ROW_WS`/`FFT_COL_WS` must reach **both** `AIE_FLAGS` and `GCC_FLAGS` (they didn't, and
  `mosse_tracker.cpp` silently defaults them to 2 — graph and host would have disagreed about
  every DMA chunk count and deadlocked). **General rule: any constant both the graph and the host
  derive from must be passed to both toolchains from one Makefile variable. A `#ifndef` default
  in the host is not a safety net — it is what makes the mismatch silent.**
- **A `static_assert` ties each `AP_*`/`DMA_*` enum to its name table.** Inserting `DMA_ACCUM_IN`
  mid-enum without updating the table would silently RENAME every port after it.
- **The `bias_acc` correction — APPLIED 2026-08-23, and it re-opens the shift budget.**
  `export_weights.py` derived `bias_acc` for an input scale of 127 ≙ 1.0 while `roi_crop` emits
  a z-score at `ROI_NORM_Q = 32`, so `bias_acc` was ~4× oversized — and since `out_shift` comes
  from `|bias_acc| + ACC_MAX_THEORY`, an oversized bias shifts the SIGNAL down to make room for a
  DC pedestal Stage B1 subtracts away anyway. Pure loss. `make weights` now defaults to
  `BIAS_SCALE=roi`; `BIAS_SCALE=127` reverts bit-for-bit apart from the layout tag.
  Measured on the grayscale export (`check_collapse.py` Q3, input-independent):

  | | `127` | `roi` |
  |---|---|---|
  | structurally dead channels | ch3, ch15 | **none** |
  | ReLU provably a no-op on | 11 of 16 | 7 of 16 |
  | signal resolution | 7.6–13.0 of 15 bits | **9.6–13.4** (spread 5.5 → 3.8) |

  Rank and participation ratio do NOT move (9 / 4.94) — the collapse is a property of the
  weights, not the bias, so this fix and RGB are independent wins.
  **It only pays with `CONV_RELU=0`** — see the ReLU entry in [`settled.md`](settled.md); corrected+ReLU is the
  worst of the three arms measured. It also spent the accumulator margin, which is what forced
  `H_SHIFT` 10 → 11 on 2026-08-24; that is now closed on hardware for both arms.
  Still unfixed, and unrelated: the semantic mismatch of weights quantized against
  ImageNet-normalized linear luminance being fed a z-score of the log.
- **`gen_aiesim_vectors.py`'s float Stage A differs from the kernel on 40.9% of samples**, by up
  to 2 LSB (rms 0.65 on a signal of std 32).
- **Test vectors can sit below the fixed-point floor** — s1's amplitude-1 impulse quantized to
  almost nothing (20/4096 bins non-zero). Now `GEN_IMPULSE_AMP=100`.
  See [[aiesim-quantization-floor]].
- **The legacy test scheme plants the target at the tracker's own estimate plus a constant**, so
  ground truth follows the tracker and `err=0 px` is nearly self-fulfilling. `TRAJECTORY=1` makes
  drift real and measurable.

### AIE / kernel traps

- **Vectorization gotchas, all found the hard way.** conv2d: use `aie::downshift`
  (arithmetic/floor, matching C++ `>>`) never `srs` (rounds to nearest); `aie::unpack` not
  `cast_to` to widen the Hann table. cmul: `from_vector(acc, S)` seeds the accumulator exactly,
  so the accumulator must be folded in BEFORE the shift — converting the product to cint16 first
  clamps twice. `rounding_mode::positive_inf` and `saturation_mode::saturate` are load-bearing.
  **`alignas(32)` on tile-local buffers is required: x86sim does not enforce alignment**, so
  omitting it passes every bit-exactness check and then misbehaves on hardware.
- **conv2d weights are consumed per FIRING, not per patch.** `weights` is an `input_buffer` and
  ADF acquires it before every invocation, so the driver must supply
  `PATCH_ELEMS/CONV_OUT_CHUNK` buffers and must start the patch flowing *first* or it deadlocks.
  This caused every historical "PLIO hang". Proper fix, not done: make weights an RTP.
  Note the host currently sends the **same 64 bytes 16 times per channel** (the `async` offset
  has no `k` dependence). See [[hw_emu_conv2d_fft_hang]].
- **`aie2gm_nb()` transfers one kernel invocation per call, not the full N bytes.** One
  `async`/`wait` pair per invocation on every output GMIO, chunked by the producer's output
  window. **The drain loops must be ordered, not just chunked**: `gmio_accum_out` /
  `gmio_response` must be drained **before** waiting on the corresponding input GMIO. Symptoms of
  getting it wrong: stall after 6 of 64 weight buffers, DMA status frozen at `0x1a080010`,
  `roi_crop_0` at `ap_start=1, ap_done=0` forever. See [[feedback_aiesim_gmio]].
- **XRT GMIO allows ONE outstanding async per port.** There is no pipeline to deepen. A depth-2
  probe aborted in 11 seconds of board time with
  `Asynchronous operation is already initiated. Multiple 'async' calls are not supported`.

### Infrastructure

- **THE CU COMPLETION INTERRUPT IS NEVER DELIVERED ON THIS PLATFORM. Every KDS launch costs
  ~503 ms.** A platform defect, not anything in this design. The CU side is healthy and armed:
  ```
  devmem 0xa4010004  GIER = 0x1     global interrupt enable ON
  devmem 0xa4010008  IER  = 0x3     ap_done + ap_ready enabled
  devmem 0xa401000c  ISR  = 0x3     BOTH LATCHED, NEVER SERVICED
  ```
  The HLS ISR is toggle-on-write and only a handler clears it, so `ISR=0x3` standing after 100+
  launches proves **no handler has ever run** — independent of `/proc/interrupts`, which reads
  `0 0` on both `zocl_irq_intc` IRQs (51, 52). Clearing the ISR by hand puts it straight back to
  `0x3` with counts still 0. Consistent with the boot-time `zocl-drm: error -ENXIO: IRQ index 32
  not found` and with IRQ 51 registered `Edge` while 52 is `Level`.
  **Everything reachable from userspace was tried and none of it moved the number**:
  `poll_threshold=1000000`, a hand-cleared ISR, `Runtime.ert_polling` — all 503.40 ms. **Fix is
  `ROI_CROP_USER_MANAGED=1`**; the interrupt wiring itself needs the base platform's device tree.
  Board-side checks in the order that settles things fastest: `/proc/interrupts | grep zocl`, the
  three `devmem` reads, then `cat /sys/bus/platform/devices/CU.N.auto/cu_stat` — its `sleep cnt`
  equalled the launch count exactly. `cu_info` in the same directory prints the base address,
  control protocol and full argument offset map, which is where `CropIp`'s register map came from.
- **`xrt::ip` details that mattered** for `CropIp`: 2025.2 imposes **no control-protocol
  restriction** (the `ap_ctrl_chain` worry was unfounded); it takes an **exclusive** CU context,
  so the `xrt::kernel` for roi_crop must not be constructed at the same time
  (`Runtime.rw_shared=true` relaxes this); `frame_buf` is an `m_axi` pointer and needs
  `bo.address()`, which `set_arg(0, bo)` used to supply implicitly. After the fix `drain → poll`
  is 0.019-0.027 ms and the spin exits on its first read. **If the drain ever shrinks the spin
  will start spinning for real** — the 60 s bound is what keeps that safe. (It did: the memtile
  transpose deleted the drain and exposed 5.196 ms/frame, exactly as predicted here.)
- **An `xrt.ini` key set as a shell variable is not a test of that key.** `ert_polling=true` at
  the prompt sets an environment variable; XRT reads `Runtime.ert_polling` from `xrt.ini` in the
  process's working directory. One "null result" in the 503 ms hunt was this and nothing else.
  The same file's header warns that an unrecognised *key* is silently ignored — the same failure
  mode one level up.
- **`v++ --package` corrupts the 2025.2 rootfs** (ext4 feature mismatch) — every hw_emu run
  panicked at boot. `make rootfs` builds a downgraded copy. See [[vpp_package_corrupts_rootfs]].
- **XRT AXIS ports consume a positional argument slot**, so scalars shift by one. Set args by
  explicit index. This was the real cause of "hw_emu PL→AIE PLIO delivers nothing".
  See [[plio-was-never-broken-xrt-arg-index]].
- **The host does not exit after the last frame.** `gr.end(0)` blocks forever on a free-running
  graph. Cosmetic, but `run_script.sh` never reports RC and emulation must be killed by hand.
- **Probing PL↔AIE signals in hw_emu takes three non-obvious steps**, each of which silently
  defeats capture: `ai_engine_0.S00_AXIS` is a SystemC/TLM socket with no TVALID/TREADY wires
  (use the `VitisRegion/out_r_*` boundary port); v++ elaborates with `--debug off` so `log_vcd`
  aborts before Linux boots (`make smoke_debug_sim` re-elaborates); and `elaborate.sh` uses
  relative include paths that fail in the copied `package/` tree **while still printing "Built
  simulation snapshot"**. See [[hw-emu-signal-probing]].
- **The pre-computed FFT bypass in `make aiesim` skips the PatchIn→conv2d→row-FFT path.** Use
  `make aiesim_plio`. Buffer dumps: [[hw-emu-buffer-dumps]].

