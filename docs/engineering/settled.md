# Settled questions and validated facts — do not re-derive

**Status:** current · **Updated:** 2026-09-02 · **Scope:** validated facts and closed questions — do not reopen

Split out of CLAUDE.md 2026-08-31 and **maintained here since** — this file, not
CLAUDE.md, is where this topic is kept current; CLAUDE.md carries only the one-line
version and a link.

## Validated / done (facts worth not re-deriving)

- **`roi_crop` cycle counts, measured** (hw_emu VCD, 64×64, `recompute=1`): `ap_start` →
  `ap_done` = **245.5 µs = 76,725 cycles** at 312.5 MHz, and `ap_done` asserts at the same
  instant as `TLAST`. PASS1 44,600 cyc, NORM 4,100, PASS2 27,900. **Two source comments in
  `roi_crop.cpp` are wrong**: PASS2 achieves **27.2 cycles/beat, not II=1** (real AIE
  backpressure), and PASS1 achieves **10.9 cycles/output-px, not II=4** (m_axi latency on the
  four scattered bilinear taps). 128×128 recompute is 4× the pixels ≈ 1 ms, exactly the residual
  `ROI_CROP_PIPELINE` leaves on channel 0.
- **`roi_crop` bit-exact, 25 cases, zero tolerance** (`make test_roi_crop`, golden from
  `scripts/roi_crop_ref.py`): 17 grayscale + 8 RGB, run as two builds because `ROI_IN_CH` is
  compile-time. 16 cases execute the bilinear interpolator, which **has still never run on
  hardware** — every build to date sets `roi_h == patch_rows`, collapsing the datapath to a copy.
  Real ground-truth boxes (a VOT run, say) would activate it for the first time.
  **Bit-exactness says nothing about timing**: this suite passed throughout the period when the
  launch path cost 98% of the frame.
- **Bounding-box state, ROI padding, σ anchoring.** State is a `TargetBox`;
  `roi = box × TARGET_PADDING`. `patch_dr_to_frame` / `frame_dr_to_patch` are unit-tested (33
  assertions incl. an anisotropic box), because while `roi_h == patch_rows` the two are
  accidentally the same number and padding breaks both by the resample ratio — a tracker that
  localises confidently and *drifts*, invisible to `err=0 px`.
- **`nature` IS NOT A TRACKER DEFECT — ITS PIXELS DO NOT MOVE. Measured 2026-08-25, offline.**
  Three diagnoses (sub-bin lag, origin lock, background lock) all assumed the target moves and
  the tracker fails to follow. `scripts/vot_motion_check.py` tests the premise: **on 80% of
  `nature`'s frames NOT MOVING correlates better** (NCC 0.940 vs 0.816). The target deforms in
  place — aspect 0.58 → 1.65, appearance decorrelating to 0.072 by frame 50 — and the box centre
  moves because it is a min–max over a changing shape. Corroborated twice: the response is
  healthy and says zero translation (peak at (0,0), `resp00/peak` 1.0000, PSR 33), and at
  `padding = 1.0`, with no background in the ROI at all, it still reports (0,0) on 98% of frames
  — so not background lock either. **`nature` is 46% of the frames in the 8-sequence evidence
  set**, so read per-sequence tables, never a frame-weighted aggregate, and never tune against
  it. `docs/thesis/evidence/frozen_detector.md`.
- **`tiger`: eta / sigma / eps_rel SWEPT, and the freeze is a SYMPTOM, not the objective.**
  `SIGMA=1` and `EPS_REL=0.1` each unfreeze the detector completely (froze-while-needed 65.8% →
  0.5% / 0.0%) and tracking does not improve — IoU stays ~0.21, centre error gets worse. **Do
  not optimise the freeze rate.** Three mechanisms tested and killed: ATTENUATION (iterated
  re-cropping helps `car1`, cerr 5.6 → 4.7 px, and makes `tiger` worse); the ONLINE UPDATE (with
  the crop at groundtruth every frame the detector is still wrong by a median **17 bins**, and
  identically so at `eta = 0`, so the filter is wrong before learning touches it — `car1`: 2
  bins); and the filter INVENTING the offset (a plain NCC template search with no filter and no
  features puts the best match **(−6.7, −8.9) px** from the annotation centre). **So `tiger` is
  `nature`'s disease in a milder form**: the box centre is a min–max over a deforming object and
  drifts against the appearance, ~11 px ≈ 8 bins, which the filter amplifies ~1.7×. Trackable
  but permanently penalised. `docs/thesis/evidence/tiger.md`.
- **`MOSSE_ETA = 0.05` SHIPPED — hardware R 0.3065 → 0.3283 (+7.1%), EAO +8.5%, 13.9% more
  frames. But two of three mechanism falsifiers FIRED.** The drift model predicted the gain and
  predicted the wrong reason: pre-loss ACCEPT share was predicted to stay ~80% and fell to
  73.7%, and first losses land marginally EARLIER, not later. What improves is RECOVERY. Per
  sequence it is a coin flip (R better on 28 / worse on 24 / tied 10); the pooled figure is
  frame-weighted. **Do not build on the explanation** — and note the gain forced the
  `PSR_GATE_MIN` re-tune, because a slower filter drops median PSR ~26% into a gate calibrated
  for the faster one. `docs/thesis/evidence/eta05.md`.
- **PHASE CORRELATION IS THE WRONG INSTRUMENT FOR "did the target move".** It returns the
  DOMINANT motion in the window, so static background filling the box makes it read zero: it
  reported **0.00 px on `car1`**, a car crossing the frame at 20 px/frame. Ask about the
  target's own pixels at two named hypotheses instead (`vot_motion_check.py`).
- **`rgb_vs_gray_loop.py --sequence <name>` reproduces the board's tracking failures at 3.5 s
  per 100 frames** — `nature` 38.4% frozen against hardware's 44.3%, `tiger` 65.8% against
  62.4%. Resolves stb2022 under `$VOT_ROOT` first, then `test-sequences/`. **Its `load_gt` used
  to carry a polygon-only copy of the groundtruth rule**, correct for `test-sequences/` and
  silently wrong for every stb2022 rectangle; it now single-sources `vot_prepare.reduce_box`.
- **Padding 2.0 is CLOSED, three ways.** `car1` closed-loop mean IoU 0.857 / 0.780 / 0.174 at
  2.0 / 1.5 / 1.2; all 62 offline, every value below 2.0 is worse (R 0.2251 at 1.5 vs 0.2910);
  and on hardware 3.0 LOST EAO. The original 1.5-vs-2.0 holdout was on a static scene where
  background lock costs nothing — that objection was valid and the answer came out the same.
- **SUB-BIN INTERPOLATION IS A SMALL ACCURACY WIN, NOT A FIX — and the compounding-lag argument
  for it is wrong.** The peak detector is a pure integer argmax, and on `nature` one bin is
  1.61 × 2.78 frame px against 2.06 px/frame of true motion, so 86% of frames really do report
  (0,0). What does NOT follow is that the error compounds: **the detector measures the offset
  that exists now, not the increment**, so lag accumulates only until it crosses half a bin.
  `mosse_loop_sim.py --subbin` sweeps ratio 1-3 × 0.1-1.5 bins/frame: worst late/early error
  ratio **1.00**, error bounded at ~half a bin — and the same bench's `centred` arm DOES
  compound, so the flat result is a finding, not an insensitive instrument. Parabolic refinement
  cuts the bounded error at large resample ratios (2.68 → 1.25 px at ratio 3), worth well under
  1% of IoU on a 100 px box. **The arithmetic that sold the wrong story conflated mean SPEED
  with mean DISPLACEMENT.** `docs/thesis/evidence/subbin_lag.md`.
- **The HOLD on a gated frame is NOT unconditionally correct — measured 2026-08-25.** From
  stb2022 groundtruth alone (`scripts/vot_hold_budget.py`, no tracker, no board), the **hold
  budget** — frames before the target's centre leaves the frozen `box × padding` window, i.e.
  before recovery is impossible for ANY tracker — has a median of **6 frames**, is **≤ 4 on 30
  of 62 sequences**, and is **0 on four**. `car1`'s budget is 4 and its longest hold on hardware
  was **53**. The candidate fix is `HOLD_COAST` (`coast_observe()`/`coast_step()`), OFF by
  default because the two metrics disagree about it (see the mean-IoU-vs-AR entry in this file); over 8 sequences
  it is +0.0296 frame-weighted mean IoU. **THE OFFLINE BUDGET MODEL PREDICTED THE OPPOSITE FOR
  `car1`, BECAUSE IT WAS OPEN LOOP**: it treated the observed 29-frame gated run as fixed, when
  coasting turns gated frames into accepted ones and every accept restarts the coast. Read the
  budget as a bound on ONE uninterrupted hold, never as a prediction of outcome. The coast
  cannot rescue a tracker that never acquires — `ball3` gates on 69% of frames and coasted zero
  times. `docs/thesis/evidence/hold_policy.md`, `evidence_arm_ab.md`.
- **MEAN IoU AND THE TOOLKIT'S AR CAN ORDER TWO ARMS OPPOSITELY, ON THE SAME RUNS —
  2026-08-25.** The `HOLD_COAST` A/B was decided on frame-weighted mean IoU (+0.0296, arm B
  wins); ingesting the identical 54 trajectory pairs into a VOT workspace
  (`scripts/vot_ingest.py`) reversed it — accuracy 0.638 → 0.616, robustness 0.309 → 0.288,
  EAO 0.208 → 0.194, arm A wins. **The cause is that `vot` fails a run on 10 CONSECUTIVE frames
  at overlap ≤ 0.1 and discards everything after**, so a short excursion outweighs hundreds of
  good frames: `car1` anchor 741 coasts 13 frames the wrong way through a turn (freezing drops
  out for 7, under the grace), then reacquires and tracks ~470 more frames at overlap 0.82 that
  the rule throws away. Failure COUNTS barely moved (48 of 54 runs vs 49) — only their timing
  did. **Quote which metric produced a verdict, and prefer AR for anything that will be
  reported**; mean IoU cannot see the timing of a loss, and that is what the challenge scores.
  See `docs/thesis/evidence/metric_ar_vs_iou.md`.
- **PSR gating (Bolme §3.5)** — four veto reasons reported separately: `ZERO_RESPONSE`,
  `FLAT_SIDELOBE` (sdev 0 — PSR undefined, not infinite), `NEGATIVE_PEAK`, `LOW_PSR`. Only
  `LOW_PSR` is disabled by `PSR_GATE_MIN=0`. On a gated frame the host **holds the position**
  (moving to a noise peak walks the ROI off target permanently) and skips `publish_filter` —
  required, not an optimization, because `filter_quantize_q15` reads `g_energy`, which the
  per-channel loop has already overwritten with the occluder's energies. Proven on hardware:
  PSR 3.90 → HOLD → reacquire with `err=0 px`, frozen filter byte-identical to the ungated run's.
  **`HOLD_COAST=1` does NOT change the short occlusion regression**, and the reason is structural rather than lucky: `ITER_CNT=3 OCCLUDE_MASK=0x2`
  occludes frame 1, and frame 0 takes the `if (!evaluate)` branch without ever reaching the
  accept path, so no velocity has been measured and `coast_step()` refuses. **Longer occlusion
  runs DO change** — at `OCCLUDE_START=30` the frames before the occlusion set a velocity and the
  window coasts through it. Compare any such run against the shipping default (`0`) rather than
  against the 08-25 arm-B runs, and use the run-state digest to tell "unchanged" from "nearly
  unchanged".
- **`scale_gate()`** — three vetoes reported separately. `AT_SEARCH_RAIL` is checked *before*
  `LOW_CONF` because it is the more specific finding when both fire. **The update skip is the
  load-bearing half**, not the box hold. Guarded so a degenerate filter cannot veto everything
  (`n_scales > 2`) and so "the gate never ran" is distinguished from "the gate said no". 21
  assertions in `make test_host`, driven by values hardware actually produced.
- **DSST 1-D scale filter** = the existing filter at `rows = 1`; `gaussian_target_spectrum(G,1,S,
  σ,0,0)` degenerates cleanly. Defaults from §6.1 except the step: S=33, η=0.025, σ_s=S/16,
  template capped at 512 px. A single application under-corrects (+3 where +5 is exact) because
  the response is a discrete peak smoothed by a σ=S/16 target — the property to assert is that
  **repeated** application converges monotonically, which it does to 0.5%.
- **A BYTE-IDENTICAL TRAJECTORY IS NOT A BIT-IDENTICAL RUN — found 2026-08-25 BY A NEGATIVE
  CONTROL.** The multi-start determinism test (job A, B, A in one process; A's two trajectories
  must match) passed with `RESET_MUTANT=1` active, i.e. with `run_reset()`'s `mean_prev` re-seed
  deliberately skipped. The leak was real and visible from frame 1 — `accum` 1963 vs 1961,
  `response` 1556 vs 1553, B2 removing 667 vs 671, differing on ~15% of frames — but a box comes
  from an INTEGER peak bin, so a 0.1% difference in the response never reaches the trajectory.
  Same failure mode as the parallel-for FMA experiment: "nearly identical" is the outcome that
  makes a bit-identical criterion useless. **Fix: a per-frame FNV-1a digest over the full response
  buffer plus `box`/`psr` in full precision**, printed at the end of every run on BOTH arms and
  folded into the determinism key. It also replaces "diff two logs by eye" as this project's
  bit-identical check with a single number. **The lesson is about the CONTROL, not the test**: the
  test had already "passed" on hardware once, and only running the deliberately-broken build
  showed that the pass meant nothing.
- **`strings` ON THE ELF CAN REPORT A FALSE ABSENCE — found 2026-08-25.** The `strings` check recommended in [`traps.md`](traps.md)
  (and for `SCENE_VERIFY`) is sound, but it answers "did the compiler emit this?", which is
  not "did the build enable this?". The `[coast]` printf was absent from a `FRAME_SOURCE=synth
  ITER_CNT=1` ELF with the feature correctly enabled: `g_run_frames` is a `static int` that
  nothing writes on that arm, so GCC proved the frame loop runs ONCE, and any state carried
  ACROSS iterations (here the coast velocity, set only in the mutually-exclusive branch) is then
  provably dead. It reappears at `ITER_CNT=200` or at `FRAME_SOURCE=vot`, where the count is
  written at runtime. **Verify a feature flag on a build that can actually exercise it**, and
  when a `strings` check says a feature is missing, check the loop bounds before the flags. Cost
  here: an hour of bisecting a compiler that was right.
- **PHASE 4's TWO KNOBS ARE WORTH 1.1%, NOT 15% — THE TRANSPORT WAS THE WHOLE WIN
  (2026-08-25).** Three `car1` runs, 8434 frames, all 15 state digests identical: UART + every-
  frame progress + every-row flush **28.48 ms**; ssh **24.69**; ssh + `CSV_FLUSH_EVERY=200`
  **24.70**; ssh + `PROGRESS_EVERY=25` **24.43**. So UART→ssh is **3.79 ms**, `CSV_FLUSH_EVERY`
  **0.00**, `PROGRESS_EVERY` **0.27** — against a written-down prediction of ~4.0 ms for the
  thinning. Premise again: 92.5 µs/byte is the cost of a byte *at 115200*, and the run being
  predicted was launched over ssh where a byte is nearly free. `CSV_FLUSH_EVERY` is zero for a
  second reason: the sweep runs from `/tmp`, which is **tmpfs**. **`car1` at 24.43 ms = 40.9 FPS
  is the best frame time recorded and is NOT comparable to the 26.29 ms in the performance
  history** (UART, synthetic scene).
- **VERIFY A FEATURE FLAG ON A BUILD THAT CAN EXERCISE IT — the check caught its own case.**
  `PROGRESS_EVERY=25` produced an ELF byte-identical to the default, i.e. the knob was INERT —
  correctly, because the default build is `VERBOSITY=1` where the level-0 line does not exist
  and the branch is dead-code-eliminated. At `VERBOSITY=0` the ELF differs at 10 and 25 and
  matches at 1. Read the first result as "it works" and a sweep runs with an unthinned console.
  **Two corrections were needed to make the defaults byte-identical**, both worth repeating: an
  unconditional `static` counter changes codegen even when its value is never used, so the
  default arm must be the original line TEXTUALLY (`#if N > 1 / #else`); and one inserted
  `fflush()` shifted every later offset, producing 9958 differing bytes from a single redundant
  call. **The control:** building the same source twice gives a byte-identical ELF, so an ELF
  `cmp` is a valid instrument on this project.
- **Console gating (`VERBOSITY`) details that mattered.** (1) The `VP1`/`VP2` macros are
  `if (VERBOSITY >= n)` on a compile-time constant, so format strings are **dead-code-eliminated,
  not merely skipped** — verify with `strings` on the ELF. (2) `dma_accumulate_frame()` had to be
  split out of `dma_report_frame()`; the printer was also the accumulator, so gating the print
  would have silently turned the CUMULATIVE report into a two-frame report. (3) **Anomalies print
  at every level.** (4) `VERBOSITY=0` still prints one line per frame deliberately: gating to
  nothing would delete the instrument `picocom | ts` needs.
- **`track.csv` carries `rails,accum_max,fch0_max,h_max` since 2026-08-24, and that retired the
  "`VERBOSITY=1` is load-bearing" rule.** The scan always ran — `report_cint16` gates the PRINT,
  not `scan_cint16` — so the numbers were computed and discarded. **Validated by reproducing a
  known answer**: the `VERBOSITY=0` run's CSV gives `F_ch`/`accum`/`response`/`H(q15)`
  digit-for-digit identical to the `VERBOSITY=1` run's console, at 28.58 ms/frame instead of
  62.67. `calib_report.py` falls back to the CSV and says which source it used; it distinguishes
  "0 rails" from "no rails column", because those must never print the same word. `fch0_max` is
  named for ch0 because `F_ch` is scanned under `if (ch == 0)` — not a bank max.
- **DMA 4258 → 1090 tx/frame** via `FFT_ROW_WS`/`FFT_COL_WS` 2→8; 96% of the traffic was the
  four per-invocation-chunked ports. **DMA is not a bottleneck and the fabric is at spec**:
  80 µs/tx is per-transaction *overhead*, not bandwidth — 64 B costs 14.4 µs and 128 KB costs
  22.8 µs (2048× the size for 1.6× the time). The largest transfer achieves **5.76 GB/s**.
- **x86sim bit-exactness harness** (`make x86sim_check`) — conv2d s6, cmul s7 and cmul_stress all
  16384/16384 identical. This verifies `simulate_conv2d`, on which the whole offline
  `phase1_sweep.py` methodology rests.
- **Test-sequence generation**: static background generated once with dirty-rect restore;
  `TRAJECTORY=1` closed elliptical path (peak 9.42 px/frame, ROI in-frame over 2000 frames);
  `SCALE_TRAJ=1` sinusoidal size (0.99%/frame peak vs the filter's 2%/frame step); per-frame IoU
  and centre error with an OTB-style run summary. All defaults reproduce previous behaviour.
- Earlier: cmul_accum saturation fix, conv2d hang root-cause (weights starvation), rootfs fix,
  uint8→int8 contract fix, filter init/update + Q1.15 export, H_SHIFT, scenarios s6/s7, hw_emu
  PLIO smoke test, N_CHANNELS=16 in aiesim (accum 7728 = 24% of cint16 — the cint16 DDR
  accumulator is sufficient). See [[cmul-accum-wrap-bug]], [[weights-bo-never-populated]].


## Settled questions — do not reopen

- **`eps_rel = 1e-3` is optimal.** The response has a closed form `R = G·B/(B+ε)`. Sweeping the
  integer pipeline: ratio 12.90 / 13.96 / **16.15** / 10.62 / 4.01 at ε = 1e-5…1e-1. Bolme
  Fig. 4's flat curve does not transfer — his ε is absolute on the denominator, ours is relative
  to `mean(B)`.
- **ReLU — NO LONGER SETTLED. THE SCOPE IS THE BANK, and it left this file on 2026-09-02.**
  On the 3x3 mobilenet bank the refutation below stands and is confirmed on AR (62 sequences,
  `feature_bank.md`): pooled dR relu **−0.0332**, full-wave abs **−0.0232**, sign-paired CReLU
  **+0.0020** (a tie), all collapsing under a symmetric trim. **But on a LEARNED Layer-1 bank the
  rectifier beats its own linear twin four times offline, and on an ANALYTIC Gabor bank it
  LOSES** — so the property that matters is that the bank is learned, not that a nonlinearity is
  present (`layer1_features.md`). `CONV_RELU=1` SHIPS since 2026-09-02 on a 7x7/2 resnet18-PCA
  bank (EAO 0.1960). The mechanism reading below — "a DCF is linear in feature space, so the
  filter cannot undo a rectifier" — explains why it fails on a 3x3 signed edge map and is
  incomplete for a large-kernel learned one. **`ARM=l1lin` has NOT run on hardware**, so the
  arm's gain is not yet attributed to the rectifier itself.
- **ReLU off beats ReLU on, by ~3×, and the `bias_acc` fix must be paired with it.** Held-out
  peak/max-sidelobe: base(ReLU) 12.82, bias-corrected(ReLU) 3.92, bias-corrected(no ReLU) 16.25.
  `base` only looks decent because its oversized `bias_acc` makes ReLU a no-op on 11 of 16
  channels. **The shipping pair is now the best of the three** (`CONV_RELU=0`,
  `BIAS_SCALE=roi`, applied 2026-08-23 — superseded as the SHIPPING pair on 2026-09-02, see
  above); the middle column is what "apply the fix alone" means
  and is why the correction sat unapplied for months. A DCF is linear in feature space; a half-wave rectifier throws away half the signal
  and the filter cannot undo it. Caveat: one patch (s6), held out by circular shift, and it
  diverges from Danelljan §3.3.
- **Padding ≥2; recommend 2.0 — REOPENED 2026-08-24, and CLOSED 2026-08-28 IN FAVOUR OF 2.0.**
  See the feature-geometry entry in this file: on all 62 sequences, every padding under 2.0 is WORSE
  (R 0.2251 at 1.5 against 0.2910 at 2.0), and at two matched px/bin points padding 2.0
  wins both. The original static-scene objection stands and the answer came out the same.
  Historic detail: the original holdout (target 64, 4-2-2, ch16) gave PSR 18.4 at 1.5 against
  45.7 at 2.0, and 2.0 is the only value where the resample stays 1:1. **2.5 and 3.0 trip the
  aliasing detector (bilinear has no prefilter) and 3.0 clips 3.57% of samples** — which is the
  standing reason not to go up, independent of the tracking result.
- **σ stays 2.0; PSR cannot select σ.** Bolme PSR is monotone decreasing in σ all the way to
  sub-pixel (80.3 at σ=0.75 down to 19.3 at σ=5.33), so it just rewards a sharper peak. The
  tempting Stage-B2 explanation was tested with `--no-b2` and refuted — the dependence is
  intrinsic to the metric. σ needs real video, or a holdout with scale/appearance change.
- **fDSST's PCA compression is not worth it; the real-input DFT was, and is DONE.** Measured per
  frame (d=484, S=33): real-input DFT + Hermitian symmetry is **3.11×**; full PCA 1.59×, of which
  the QR is 0.46 ms — and the QR is O(d·S²), the same order as the DFT it eliminates. The 3.11×
  transferred exactly to hardware (`scale extract` 4730 → 2138 µs/call, isolating the transform
  at 3819 → 1228 µs). After that the real levers are d and S, not PCA.
- **HALVING THE HOST FILTER ON HERMITIAN SYMMETRY DOES NOT WORK — the premise is false in fixed
  point.** `conv2d` emits cint16 with `imag = 0`, so the 2-D spectrum is Hermitian *in exact
  arithmetic* and A/B look half-redundant (~4 ms). Measured on the real chain (64×64 ch1, s6,
  `make aiesim_plio`): `max|residual| = 12 LSB at bin (0,1), mean 1.379, 3924/4096 bins
  asymmetric, max|F| = 444` — **95.8% of bins differ from their conjugate partner**. DSPLib's DIT
  butterflies do not compute a conjugate pair by symmetric operations and every stage rounds.
  **The control is what makes this conclusive**: the float golden for the same scenario,
  quantised to int16, is Hermitian to **0 LSB across all 4096 bins**. So the index convention is
  right and int16 storage alone preserves the symmetry — the asymmetry is attributable to the
  fixed-point FFT and nothing else. Without that control a 12 LSB residual could just as easily
  have been a wrong conjugate index.
  **And the error lands in the worst possible place**: **max|H| sits where |F| is smallest**, so a
  fixed ~1.4 LSB asymmetry is proportionally most damaging exactly at the bins carrying the
  largest filter coefficients. The check is now permanent in the harness (`F_ch Hermitian
  check`), so any future FFT change re-tests it for free.
  **What is NOT refuted**: the same argument for the AIE's own forward transform — DSPLib's
  `fft_ifft_2d_graph` halves its memory tile for real input via `fft_dit_2ch_real_graph`, which
  computes only the independent half rather than computing both and discarding one. That saves
  work instead of reconstructing it, so per-stage rounding never enters the argument.
- **QUANTIZATION IS NOT THE CAUSE OF THE POOR ROBUSTNESS. Removing it makes tracking
  WORSE — measured 2026-08-27, offline, 8 sequences, 2841 frames.** The suspects are bracketed
  rather than argued:

  | suspect | verdict | what settles it |
  |---|---|---|
  | cint16 / Q1.15 / `H_SHIFT` pipeline | exonerated | `rgb_vs_gray_loop.py` is **float64 downstream of the features** and reproduces the board's failures anyway |
  | saturation / rails | exonerated | `corr(rail rate, mean IoU)` = **−0.025**; the shipping arm has **zero** rails over 101,564 frames and still scores R = 0.307 |
  | the int8 FEATURE path | **exonerated — it HELPS** | `--arms gray gray-float`: frame-weighted mean IoU **0.2533 → 0.2350**, **zero of eight sequences improve** |

  The control: the int8 arm reproduces the recorded per-sequence table to four decimal places,
  so the float column is the only thing that moved. **WHY float is worse** — on `tiger` it takes
  the frozen-detector rate from 74.7% to **1.4%** and IoU from 0.1696 to 0.0982. That is the
  third independent instance of the same rule: **the int8 grid acts as a DEADBAND**, suppressing
  sub-threshold responses a float pipeline follows faithfully to the wrong place. *Do not
  optimise the freeze rate*, and do not read "more precision" as "more accuracy" on a tracker
  whose features are the binding constraint.
  **So the robustness deficit is upstream of all arithmetic**: no pooling and a participation
  ratio of 4.94/7.43 against HOG's 31 dims; no spatial reliability, so the filter trains on 73%
  background; and a hold policy whose duration exceeds the recovery budget. **Corollary for the
  write-up: the fixed-point design costs nothing in accuracy or robustness, so the frame rate is
  bought at no algorithmic price.**
- **INIT PERTURBATIONS (Bolme §3.4) — CLOSED 2026-08-28. The 16-CHANNEL DENOMINATOR IS
  ALREADY THE CURE.** `mosse_tracker.cpp` carried "TODO: affine perturbations for
  initialisation; this is the N=1 case" from the start, and the board says it matters:
  `scripts/vot_init_anatomy.py` measures **61 of 373 losing runs (16%) failing within 10 frames
  of init**, already at median IoU 0.571 / PSR 7.35 one frame after `filter_init()` against
  0.915 / 36.73 for every other run. Those runs never acquire — a different mechanism from the
  drift in `robustness_gap.md`.
  Offline (`rgb_vs_gray_loop.py` arm suffix `-warp<N>`, 62 sequences, shipping eta/gate,
  scored with `vot_ar_offline.py`), **four perturbation axes × three regularizer settings, and
  nothing clears the +0.02 falsifier written down first**:

  | axis | board cost | dR |
  |---|---|---|
  | translation + isotropic scale | free (`roi_crop` runtime AXI-Lite) | +0.0081 |
  | aspect (`roi_h`/`roi_w` INDEPENDENTLY) | free | **−0.0322** |
  | rotation 10° (host pre-rotates into `frame_bo`; `roi_crop` cannot) | ~2 ms/warp at init | +0.0146 |

  Rotation is the best and is three sequences: dropping `rowing`/`singer2`/`birds2` gives
  −0.0020. It is the only arm moving A, R and mean IoU together, so it is the one variant that
  is not the failure-rule artifact described in [`measurement.md`](measurement.md).
  **WHY IT DOES NOT TRANSFER, and this is the part worth keeping.** Bolme §3.3 presents
  regularization and perturbations as ALTERNATIVE cures for one defect — low-energy denominator
  bins — and **his Figure 3, the whole empirical case for perturbations, is captioned "Results
  shown without regularization"**. This design has both other cures: `eps_rel=1e-3`, and a
  SHARED denominator summed over **16 channels** (Bolme's own "sum of the energies over more
  images" argument, applied across the bank). The prediction that the warp gain is a function
  of the regularizer HELD — dR +0.0081 → +0.0143 as eps falls 1e-3 → 1e-6 — and then saturates,
  because `B` never gets that small: bins below `1e-6·mean(B)` are **0.00%** on car1/tiger/nature.
  **Bolme's ε≈0 regime is unreachable here by lowering ε.** So: perturbations do exactly what
  Bolme says, and this design already bought that stability structurally; the ceiling is ~+0.014
  in R. It is a property of the FEATURE BANK, not of the fixed-point path.
  Two things not to re-derive: translation warps are NOT algebraically degenerate (they should
  be — `conj(G_δ)F_δ = conj(G)F` exactly — but the fixed Hann window and border inflow give
  `rel|ΔA|` 0.19/0.30/0.51 at 2/5/10% shift, so "the jitter was too timid" is refuted); and
  photometric warps are worthless because Stage A's log → zero-mean → unit-L2 annihilates gain
  and contrast, leaving 7–9 LSB of quantization residue against a patch σ of ~32.

- **FEATURE GEOMETRY — POOLING, RESOLUTION AND PADDING ALL TESTED, ALL REJECTED (2026-08-28).**
  Offline (8 seq, then all 62, `rgb_vs_gray_loop.py` arms `-pool<N>/-dec<N>/-blur<N>`), then
  hardware. `docs/thesis/evidence/pooled_features.md`.
  * **Aggregation is REFUTED on every bank and every operator tried — and RE-CONFIRMED on the
    RECTIFIED Layer-1 bank 2026-09-02, where it is a LOSS, not a null** (`l1relublur` vs
    `l1relu`: dR **−0.0242**, trim-5 −0.0269, P(dR<=0)=0.995, worse on 7 sequences and better on
    3). **The LINEARITY EXPLANATION this entry used to carry is WITHDRAWN.** It said a box
    average of a linear map is another linear map with the same span and so "CANNOT do anything";
    that predicts a NULL for the linear arm, and the 2026-09-02 negative control measures
    **−0.0180** on exactly that arm. The argument is arithmetically true and trackingwise
    irrelevant — the same fate as the signed-lobe hypothesis it replaced. What is left is a
    RESOLUTION reading: the map is already 64x64 from a 128x128 crop, aggregation and decimation
    agree to 0.001, and at matched `sigma/target` hardware prefers the FINER map **pooled;
    that preference is NOT paired-stable and is a null across sequences (`R-14`)**.
    `docs/thesis/evidence/pooled_features.md`.
    *(Original 2026-08-31 entry, verdict unchanged:)*
    `blur2` (2x2 box average at STRIDE 1) is −0.0010 gray / −0.0012 RGB, and `pool2 <= dec2`
    everywhere. Prompted by Danilowicz & Kryjak 2022, whose stem is VGG11 conv1 **with ReLU and
    2x2 MAXPOOL**, `-mpool`/`-relumpool` were added: `dec2` R 0.3981, `mpool2` 0.3971, `pool2`
    0.3970 — **max, average and NO aggregation agree to 0.001.** The operator is irrelevant; the
    hypothesis that averaging a SIGNED map cancelled the lobes is arithmetically true and
    trackingwise nil. `relumpool2` pins accuracy at baseline (0.5393 vs 0.5394) for a third of
    the robustness — a conservative tracker on an easier prefix.
  * **RESOLUTION IS NOT REFUTED, AND IT IS THE STRONGEST OFFLINE ARM MEASURED HERE.** A 64x64
    map at `TARGET_PADDING=2.0` (px/bin box/32) gives **dR +0.1071**, survives dropping the
    top-3 gainers at **+0.050** and a bootstrap CI excluding zero (P(dR<=0)=0.000) — against the
    shipped spatial mask's +0.0330 / +0.0101 on the same instrument. It is a RESOLUTION change,
    not a pooling one (`dec2` alone scores it), and it is the geometry Danilowicz ships. **It
    costs 0.25x the host work (~15 ms/frame) — robustness AND frame rate.** Standing discount:
    this bench over-predicted the mask 3x and pad30 ~11x; unlike pad30 it holds padding at 2.0,
    so the missing-scale-filter blindness that broke that prediction does not apply. Claim
    N-03b, `docs/thesis/evidence/pooled_features.md`. `blur2` is the arm that
    matters and it did not exist in the first sweep; without it the result reads "pooling loses",
    which is true and unattributable.
  * What moved was **px/bin**, and only on RGB. **Gray and RGB gave OPPOSITE signs for the
    resolution half; do not generalise a feature result across banks.**
  * **HARDWARE KILLED IT.** `runs/vot/0828_1451-pad30`, 62 seq / 419 trajectories, host-only:
    padding 3.0 vs 2.0 gives A 0.5100 → 0.5030, R 0.3417 → **0.3494 (+0.0077 against a predicted
    +0.088)**, and **EAO 0.1629 → 0.1570 — DOWN**. Below the +0.02 floor written before the run.
    `TARGET_PADDING=2.0` stands and the 64x64 reflash is OFF — it rested on the same proxy.
  * **Why the proxy failed:** the offline loop has NO DSST scale filter, and padding sets the
    ROI `scale_extract` draws from, so the whole scale axis was invisible by construction; and it
    is SINGLE-START, where a big search region has hundreds of frames to earn back a drift,
    against 419 short anchored runs. Both were named as caveats and neither was priced.
    **A caveat that is not priced is a hope.**
  * Free corrections: **padding costs NO frame time** (it resamples to a FIXED 128x128 patch, so
    cost is output px × 4 taps, independent of input extent), and more context makes the gate
    LOOSER (holds 15.51% → 10.94%). **Instrument bug:** `vot_detector_gain.py` hardcodes
    `box*2/128`, so on pad30 it reports alpha 0.442 where the corrected value is ~0.66.
  Still NOT refuted: a REPLACEMENT bank. That is a weights export plus the same offline loop.
- **Channel pruning is moot** (written when ReLU was off; the span argument is unchanged by the
  rectifier since it is about the LIFT, not the nonlinearity), and doubly so at `BIAS_SCALE=roi`, which retires
  the last two structurally dead channels (ch3, ch15) outright. The real
  redundancy is the collapse: it caps the bank at **rank 9** (participation ratio 4.94) and
  leaves ch0/ch9/ch14 collinear up to sign, and collinear channels add exactly coherently. The
  fix is RGB. `check_collapse.py` Q2 used to print "14 independent filters" here — that was a
  count of near-parallel GROUPS, not a rank, and it understated the problem for months.


---

## Closed 2026-09-02

- **`MOSSE_SIGMA`'s interior — CLOSED.** A 22-cell sigma x eta grid over 62 sequences
  (`runs/vot/0902_offline-sigmaeta/`, positive control reproduces `rgb-s4` digit for digit) puts
  sigma **3, 5, 6 and 8 all below 4** at `sigma/target = 1/16`, with sigma 8 the worst cell in
  the search (paired trim-3 **−0.0914**). The optimum is located, on the geometry that ships, and
  the turnover is now measured on THIS map rather than inferred from the 64x64 one.
  **The knob is `sigma/target`, not sigma** — 2.0 at 64x64 and 4.0 at 128x128 are the same point.
- **`SCALE_ETA` — CLOSED, and for a mechanical reason.** Inert from 0.025 to 0.3 in
  `scale_loop_sim` (a 43-frame terminal freeze at 0.3 is the same 41-frame freeze as at 0.025;
  0.5 falls apart at 41.5% error). It cannot be otherwise: on 838 real board runs `scale_idx` is
  already **0 on 84% of frames** and `est_h` is EXACTLY unchanged on **~90%**, so no learning
  rate can move a filter that is not asking to move. **The freeze is a DETECTION failure.**
  `scale_filter.md`.
- **"The scale estimate diffuses" — REFUTED the day it was proposed.** var(log est_h/truth_h)
  grows **1.95x** from t=25 to t=500 where a random walk predicts **20x**; the saturating fit
  beats the linear one. The growing IQR is SURVIVORSHIP (runs whose scale goes wrong fail and
  leave the population) plus many runs each parked at their own offset. **Recorded because the
  wrong reading was written into three files before it was tested** — a spread statistic on a
  shrinking population is not a process measurement.
