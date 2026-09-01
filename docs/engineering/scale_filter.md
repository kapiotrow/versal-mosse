# Scale filter (DSST) — findings

Moved out of CLAUDE.md 2026-08-31; content unchanged.

### Scale filter — root-caused offline, confirmed on hardware (2026-08-20)

`design/host_app_src/test/scale_loop_sim.cpp` (`make scale_sim`) drives the REAL
`scale_extract`/`scale_detect`/`scale_gate`/`scale_update` in a closed loop with position held.
Native g++, seconds per arm. It **reproduces the board** (the `moving` arm parks for 42 frames
from f131; hardware froze at f130) and refuses to report a verdict when the premise arm fails to
reproduce.

**`SCALE_STEP=1.04` confirmed on hardware** (`run_0820_1513`): mean/worst IoU 0.807/0.579 →
**0.917/0.833**, max box error 31.4% → **9.6%**, mean/worst centre error 2.47/11.07 px →
**1.30/3.52**. **Centre error fell 3.2× from a size-only change** — independent confirmation
that position error was downstream of the scale error.

**On hardware the detector proposed only −1, 0 or +1 over 199 frames** (174/13/12) — but **that
sentence describes ONE SYNTHETIC SCENE and failed twice on 2026-08-25.** On `car1` the detector
proposed ±2 or more seven times, up to **+9**, every one on a frame whose IoU was 0.000; and in
`scale_loop_sim` it uses ±2 *legitimately* on a smooth envelope — capping at 1 parks the
`moving` arm for 123 of 200 frames and ends it 28.0% wrong against 1.0% uncapped. Reading ±1 as
a property of the DETECTOR rather than of that scene is exactly what `SCALE_MAX_STEP=1` would
have been; the sim, not the hardware observation, is the bench that decides that parameter.

**The scale filter is ALREADY slaved to the position gate, and that is not sufficient.** The
whole block runs under `if (scale.enabled() && gate.accept && scale.initialized)`, verified on
hardware (`run_0825_1314`: 577 ACCEPTs, 577 scale evaluations, zero on a held frame). `car1`
frame 490 still got through because the POSITION gate accepted it at PSR 7.87 while the tracker
was 227 px off target. **Do not "add" that slaving; it is there.** `SCALE_MAX_STEP` exists
because the precondition is only ever as good as the PSR gate.

**`SCALE_CONF_MIN` blocks legitimate large corrections.** On the sim's `step` arm the detector
proposes the correct `idx=-14` and the gate vetoes it as `LOW_CONF` for four frames; the box
then walks at 2%/frame where bypassing the gate corrects it in ONE frame. **`conf` cannot
distinguish "wrong proposal" from "big correct correction"** — both match the model poorly, for
the same reason. Exonerated for smooth envelopes; it will bite on any abrupt scale change.

**Where it stops.** `SCALE_ETA` does not help (8.6/10.3/9.2% at 0.025/0.05/0.1); `SCALE_N=65`
gives 7.0% against 8.6% for **3.9× the cost**. ~8-10% box error is this filter's practical floor
— and it is what the worst IoU frames on every 2026-08-24 run are. The next gain needs a
different estimator, not a tuning change.

**Calibration honesty: the sim predicted a=1.04 well and a=1.02 badly** (12.6% → 8.6% predicted;
31.4% → 9.6% measured), and it recovers where hardware never did — **unexplained**; position
error and background pan were both ruled out. **Trust the ORDERING; do not quote the sim's
absolute magnitudes as the board's.** The truth rate matters and the test bench chose it:
`SCALE_TRAJ_AMP=0.30` over 200 frames shrinks the target at ~0.94%/frame, under half of one
scale level, so on most frames the correct proposal rounds to 0.

