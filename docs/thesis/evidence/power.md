# Energy per frame is 12.2 mJ, and 68% of it is the APU — the accelerator rail does not move

**Status:** current · **Updated:** 2026-09-03 · **Scope:** the energy-per-frame measurement, the instrument it needed, and why the answer is a difference

**2026-09-03.** Board probed live at `192.168.10.2` (PetaLinux 2025.2+release-S11151021,
kernel 6.12.40-xilinx, XRT 2.20.0, `Xilinx Versal vek280 Eval board revB`). No tracking run:
this note records the instrument survey, the tooling, and **the measurement itself**.
Claim id: `P-12` in `docs/thesis/claims.md`; numbers in `results/power.csv`.

## The prediction, written down first

Written before the probe: *the APU will expose per-rail current through hwmon, and
`xrt-smi examine -r electrical` will report it, as it does on Alveo.* The falsifier: any
result in which the only readable electrical quantities are voltages.

**The falsifier FIRED, completely.** Every path checked came back without current.

## The result — what the DUT can actually read

| path | what is there | current? |
|---|---|---|
| `/sys/class/hwmon/hwmon0` | `versal_thermal`, `temp1_input` only | no |
| `/sys/bus/iio/devices/iio:device0` | `xlnx,versal-sysmon`: 7 voltages + die temp | **no** |
| device tree | no `ina*` / `ir38*` / `irps*` / `max20751` node anywhere | no |
| PMC `i2c@f1000000` | **`status = disabled`** | no |
| `i2c-0`, `i2c-1` (Cadence) | one 24c128 EEPROM at `1-0054`, nothing else | no |
| `xrt-smi examine -r electrical` | listed in `--help`, answers `No report generator found` | no |

The seven SYSMON rails are `vccaux 1502`, `vccaux_pmc 1502`, `vcc_pmc 877`, `vcc_psfp 880`,
`vcc_pslp 879`, `vcc_soc 804`, `vp_vn 129` (mV). Sampled a second apart they move by ±2 mV —
they are regulated, so **voltage carries no load signal** and V alone cannot be turned into
power. Note also that no VCCINT or VCCINT_AIE rail appears at all: the two rails that would
matter most are not among the seven.

Die temperature reads 37.4–37.7 °C idle and is **the only load-correlated quantity the DUT
exposes**. It is a proxy, not a measurement, and belongs in a note like this one rather than
in `results/`.

Watts therefore come from the **VEK280 System Controller**, which owns the INA226 rails and
exposes them through `sc_app`. **Reaching it is the whole remaining blocker** — see the status
section at the end: the USB cable was plugged in later the same day and presents four FTDI
ports, but the account cannot open them.

## Why the answer is a difference and not a reading

The design uses **2% of the AIE array** and the frame is **84% CPU-bound**. On a development
board the fans, PHYs, the System Controller itself and regulator loss dominate anything this
design draws. A board-total wattage would be a true number answering the wrong question, and
would rank two arms by ambient temperature.

So the protocol is three states, and the figures are the differences:

| phase | state | what its difference gives |
|---|---|---|
| `static` | idle, xclbin not loaded | the board floor |
| `graph` | xclbin loaded, graph free-running, no frames | `P_design = P_graph − P_static` |
| `run` | frames processing | `P_work = P_run − P_graph` |
| `graph_post` | graph up again, after the last frame | **control 1**: the same board state as `graph`. Two equal readings mean the board was thermally settled; a higher second one means `P_work` carries warming, not workload |
| `tail` | idle again, process gone | **control 2**: the same test at the board level |

`J/frame = P_work × frame_time`. The `tail` phase is not decoration — without it a thermal
trend and a workload delta are indistinguishable.

## Controls

- **Two thermal-drift controls**, above: `graph_post` against `graph` (same state, after the
  work) and `tail` against `static` (same state, after the process). `power_measure.py` prints
  `CONTROL FAILED` when either gap exceeds twice the instrument noise. Two independent controls
  rather than one control twice — the repo's own rule, and here they fail differently: a
  `graph_post` gap indicts the die, a `tail` gap indicts the board.
- **A resolution gate.** A delta smaller than twice the instrument's own phase-to-phase
  scatter is reported as `NOT RESOLVED` and as a bound (`< X W`), never as a number. At 2%
  utilisation this outcome is genuinely likely, and it is a publishable result: *the design's
  marginal power is below X W* pays the thesis debt as honestly as a positive reading does.
  **The bar has to be set before the run, not after seeing the delta.**
- **Frame time taken twice**, from independent sources: the tracker's own
  `[apu] CUMULATIVE ... mean frame body`, and the run phase's wall duration over the frames
  it reported. They measure different spans and are not expected to agree exactly; a large
  gap means the phase boundaries do not bracket the work.
- **The probe refuses to produce an empty CSV.** Zero discovered channels exits 3. A channel
  that parses on one sample and not the next emits `nan` rather than a dropped row, so the
  per-channel sample count is itself a check. This is claim M-09's trap — a parser that finds
  nothing looks exactly like a clean run — and it was designed in rather than discovered.

## What not to re-derive

- **Do not look for INA226s on the APU's i2c buses.** They are not there, the PMC bus that
  would reach them is `disabled` in the device tree, and no driver is bound despite
  `ina2xx_adc`/`ina260-adc` being present in `/sys/bus/i2c/drivers`. The presence of the
  driver is what makes this look promising; it is not evidence of a device.
- **Do not expect `xrt-smi examine -r electrical` to work on the edge shell.** It appears in
  the report list and then fails at generation, which reads like a permissions or device
  problem and is neither.
- **Two bugs the first live run of `power_probe.sh` caught, both of which would have produced
  plausible wrong data rather than an error:** BusyBox `date` has no `%N`, so every row was
  stamped with the literal string `1748553727.%N`; and **the board's clock is unset** — it
  read 2025-05-29 against the PC's 2026-09-03. Together they are why samples are binned into
  phases on their LOCAL ARRIVAL time at the driver, with the remote stamp kept only as a
  drift diagnostic. Aligning phases on the remote clock would have mis-binned everything
  silently. *Running the tool against the real board on the day it was written is what found
  both.*
- **Transport is a confound here exactly as it is for FPS.** The ethernet PHY and ssh's CPU
  share sit inside the board total, and CLAUDE.md already restricts quotable FPS to
  serial-console runs for this reason. Compare like with like, and record the transport
  (`meta.txt` does).
- **No sampling rate available here attributes power to a pipeline stage.** Every sensor is an
  I2C/ADC read at a few Hz, averaging over ~100 frames at 24 ms/frame. This measures energy
  per frame over a sustained run and nothing finer. Do not design a per-stage power figure.

## What is still missing

1. **The instrument**: SC access, over its USB-UART or its own ethernet. Then
   `power_probe.sh --backend sc_app --list` against the real SC, which dumps `sc_app`'s raw
   output so the domain parser can be tightened against what it actually prints rather than
   what it was assumed to print.
2. ~~The `graph` phase needs a host-only `--power-pause`~~ — **BUILT 2026-09-03.**
   `mosse_tracker.cpp` holds twice, before frame 0 and after the last frame, printing
   `[power] PHASE <name> BEGIN|END`. Host-only (no flagstamp), rejects a hold under 5 s, and
   uses plain `printf` rather than `VP1` so a `VERBOSITY=0` power run keeps its boundaries.
   Both reject paths and the unknown-argument path were exercised on the board; none opens
   the device. The driver aligns on the markers' ARRIVAL rather than on an assumed duration —
   staging reads up to 1.27 GB over NFS before frame 0, and a window timed from the start of
   the run command would price that transfer as the design's resident power.

## 2026-09-03, second entry — THE PROTOCOL RAN. Thermal only, and it moved the method

`runs/power/0903_l1relu_sysmon/` (flagstamps in `config/`). Shipping arm, `DUMP_BUFFERS=0
VERBOSITY=0`, `car1` job 0 x 11 = **8162 frames**, 60 s holds either side, `sysmon` backend at
2 Hz, 520 s wall. **No watts** — the SC was still locked. What this run establishes is that the
INSTRUMENT and the PROTOCOL work, and it corrected the statistics.

**Every control that could fire, fired usefully:**

- **Workload constancy.** All 11 repeats returned one distinct run-state digest,
  `04442e3fe95b279e`. The workload was byte-identical across the whole run window, so a delta
  cannot be a drifting workload. This is `--vot-jobs`'s determinism test doing double duty.
- **Phase boundaries bracket the work.** Frame time 25.72 ms from the tracker's own
  `[apu] CUMULATIVE` against 25.74 ms from run-phase wall over frames — **0.02 ms apart**, from
  two independent sources. The `graph` marker landed at log line 51, i.e. AFTER the 652 MB NFS
  blob read, which is exactly what placing the hold below staging was for.

| channel | static | graph | run | graph_post | tail |
|---|---|---|---|---|---|
| `die_temp` (mC) | 37223.6 | 37236.0 | 37404.3 | 37177.2 | 37278.4 |
| `versal_thermal` (mC) | 37247.8 | 37270.1 | 37423.3 | 37217.9 | 37340.4 |

| quantity | die_temp | versal_thermal | verdict |
|---|---|---|---|
| work (run − graph) | **+168 ± 47 mC (3.6σ)** | **+153 ± 57 mC (2.7σ)** | **RESOLVED**, and the two independent sensors agree to 15 mC |
| design resident (graph − static) | +12 ± 25 mC (0.5σ) | +22 ± 34 mC (0.7σ) | **not resolved — bound it at < 60 mC** |
| CONTROL graph_post − graph | −59 ± 115 mC | −52 ± 99 mC | not resolved, but see the power warning below |
| CONTROL tail − static | +55 ± 26 mC (2.1σ) | **+93 ± 37 mC (2.5σ)** | **FAILED — the board warmed monotonically** |

**So: the workload is thermally visible at ~+0.16 °C; merely loading the xclbin and running the
graph is not.** At 2% AIE utilisation the resident design cost does not move the die.

### THE ERROR BAR IS THE RESULT, AND THE FIRST VERSION OF IT WAS WRONG

The resolution gate as first written compared a difference of MEANS against the SINGLE-SAMPLE
standard deviation. On this data that calls a 3.6σ effect **NOT RESOLVED**. Swinging to the
naive standard error calls the same effect **7.7σ**. Both are wrong, in opposite directions:
these are slow time series, and consecutive samples half a second apart are strongly correlated.

With a Bartlett autocorrelation correction the run phase's 383 samples are worth **n_eff = 31**,
and a 112-sample hold is worth **n_eff = 6**. `power_measure.py` now builds every error bar
that way (`neff()`, `resolved()`). **Do not revert this to an sd comparison because it looks
more conservative — conservative in the wrong denominator is just wrong.**

### AN UNDERPOWERED CONTROL IS NOT A PASSED CONTROL

The two controls appear to disagree: `graph_post − graph` says the die returned to its pre-run
state, `tail − static` says the board ended 55–93 mC warmer. They do not actually conflict —
the `graph_post` window has **n_eff = 6** and an error bar of ±115 mC, so it could not have
resolved a drift a third the size of the signal even if one were there. It is not evidence of
no drift; it is no evidence.

The real reading is the one that CAN fire: **the board warms monotonically over a 9-minute
protocol, by about a third of the work delta.** So +168 mC is an UPPER BOUND on the workload's
thermal contribution, not an estimate of it. This is claim M-16 again — match the test to how
much data the run actually gives.

**Consequences for the next run, all cheap:**
1. **Holds of 300 s, not 60**, and a `static` of 300 s. The thermal autocorrelation lag is
   15–20 s; a 60 s hold buys about six independent samples.
2. **`PROGRESS_EVERY` was left at 1**, so 8162 console lines went over ssh inside the run
   window. Console transport is worth 3.79 ms/frame (P-09) and it is in `p_work`. Set it to a
   number larger than the frame count. This also explains part of 25.72 ms against the 24.07 ms
   recorded on `agility` — the rest is the sequence.
3. **Let the board thermally settle before `static`**, or accept the drift as a stated bound.

### What the run does NOT establish

Nothing about watts, and **the thermal proxy must not be converted into one**. Die temperature
is a lagged, nonlinear function of power with the fan in the loop; the ratio between +0.16 °C
and joules is not known and is not derivable from anything in this repo. This section exists to
record that the protocol and its controls work, not to pay the thesis debt.

## THE MEASUREMENT — `runs/power/0903_l1relu_scapp/`

Shipping arm, `DUMP_BUFFERS=0 VERBOSITY=0 PROGRESS_EVERY=100000`, `car1` job 0 x 11 =
**8162 frames**, 300 s `static` / 300 s `graph` / 205 s `run` / 300 s `graph_post` / 180 s
`tail`, `sc_app` over `/dev/ttyUSB3` at 1.4 Hz, 497 samples per idle window. Flagstamps in
`config/`. Frame time **25.16 ms from both instruments** — the tracker's own `[apu] CUMULATIVE`
and run-phase wall over frames — agreeing to two decimal places.

| rail | what it supplies | static | run | **Δ (work)** | mJ/frame |
|---|---|---|---|---|---|
| `VCC_PSFP` | **APU (A72)** | 0.185 | 0.520 | **+0.328 ± 0.006 W** | **8.26** |
| `VCC1V1_LP4` | **LPDDR4** | 1.202 | 1.317 | **+0.114 ± 0.009 W** | **2.88** |
| `VCC_SOC` | NoC | 3.153 | 3.198 | **+0.044 ± 0.019 W** (2.3σ, marginal) | 1.11 |
| **`VCCINT`** | **PL + AIE-ML** | 3.855 | 3.845 | **−0.014 ± 0.016 W — NOT RESOLVED, < 0.033 W** | — |
| `VCC_PSLP_CPM5` | PS low-power | 1.719 | 1.718 | not resolved, < 0.009 W | — |
| `VCCAUX`, `VCC_PMC`, `VCCAUX_PMC` | aux/PMC | | | not resolved, < 0.006 W | — |

**Dynamic total 0.487 W over 8 rails, against 11.52 W of idle on the same rails.**
**Energy per frame = 12.2 mJ at 25.16 ms (39.7 FPS).**

### The result: the frame's energy is the CPU and the memory, not the accelerator

- **The APU rail is 67% of the dynamic power**, LPDDR4 23%, the NoC 9%.
- **The PL + AIE-ML core rail does not move at all**, bounded at **33 mW** — under 7% of the
  dynamic total — while carrying conv2d, both FFT/IFFT chains and `cmul_accum`.
- **`P_design` (graph − static) is not resolved on ANY rail.** Loading the xclbin and running
  the graph free-running, with no frame in flight, costs under ~15 mW. The design's cost is
  what it does per frame, not what it costs to be resident.

**This is an independent confirmation of P-02 in the energy domain.** P-02 says the frame is
84% CPU-bound in TIME, measured by host instrumentation; this says it is ~68% APU in POWER,
measured on a different chip by a different instrument that shares no code with it. It is also
the sharpest available evidence for `przeglad.tex:61` — that a heterogeneous system's cost is
decided by a stage's share, not by how fast one stage is made — and it is the same story as
**P-04**, where conv2d doubled and the frame did not move. Here conv2d runs and the rail that
carries it does not move either.

**CAVEAT TO CHECK BEFORE WRITING THIS UP:** `listpower` exposes no separate AI Engine rail
among its 20, so `VCCINT` is taken to supply both the PL fabric and the AIE-ML array. That is
consistent with the rail set but is NOT confirmed from the datasheet here. If the AIE-ML has a
supply folded into another rail, the "accelerator does not move" claim needs restating. The
measured facts — which rails move and by how much — are unaffected.

### What the run cost in method

- **All four controls passed.** `graph_post` vs `graph` within 2 mW on the rails that matter,
  `tail` vs `static` within 3 mW. The 300 s windows (sized from the measured autocorrelation,
  after the 60 s ones gave n_eff = 6) delivered n_eff ≈ 500.
- **Determinism**: one distinct run-state digest across 11 repeats, `04442e3fe95b279e` — the
  same digest as the thermal run, so both measurements watched the identical workload.
- **`PROGRESS_EVERY=100000`** removed the 8162 console lines that sat inside the previous run
  window; the frame came down 25.72 → 25.16 ms, and `run.out` went 10646 → 2506 lines.

### QUANTIZATION BROKE THE GATE, AND THE NULL RAILS ARE WHERE IT SHOWED

The rails are read through wildly different ADC steps — over this run `VCC_PSFP` resolved **149
distinct values** and `VCC_PMC` resolved **three**. On a 3-level rail nearly every sample lands
on one level, the sample sd collapses toward zero, the standard error collapses with it, and
then ANY difference clears "2 s.e." The first report of this run therefore printed
**`CONTROL FAILED ... -0.000 +/- 0.000 W`** on three null rails, and marked a −0.001 W nothing
as `RESOLVED`.

`resolved()` now floors the standard error at the quantization noise of the mean
(`q/sqrt(12·n_eff)`) and, below **5 distinct levels**, refuses to resolve anything finer than
one step. Every control then passed. **Note this does NOT invalidate the sub-LSB bound on
`VCCINT`**: its step is 0.122 W but the signal dithers across 13 levels with a large sd, which
is exactly the condition under which averaging legitimately beats one LSB. Dither is what
separates the two cases, and it has to be checked per rail, not assumed.

`--reanalyse DIR` exists because of this: the fix was arithmetic, and a 22-minute board run
must never be repeated to correct arithmetic.

## The System Controller interface, as it actually is (2026-09-03)

Image `xilinx-versal-system-controller-20222`, `sc_app` in `/usr/bin`, reached over
**`/dev/ttyUSB3`** (`/dev/ttyUSB1` is the Versal PS console). Login `petalinux`. Four things
here were assumed wrong before they were checked, and each would have produced a plausible
wrong result rather than an error:

1. **`listpowerdomain` / `getpowerdomain` DO NOT EXIST on this image** — they answer
   `ERROR: power domain operation is not supported`. The working commands are
   **`listpower`** (20 rails) and **`getpower -t <RAIL>`**. The first parser was written
   against the domain commands and would have discovered zero channels.
2. **One query costs 12 ms** (76 ms for eight rails), so the sampling rate is set by the
   console, not by `sc_app`. Measured 1.4 Hz for 8 rails at a 500 ms period.
3. **`Power(W)` is a separate INA226 register, not V×I.** Over 30 idle VCCINT samples it
   tracked V×I to **−0.57% ± 2.63%**, but one sample in 30 was 13% low and an earlier one
   read **1.7091 W where V×I said 3.97** — a 55% glitch. V itself is steady to 0.1%. So the
   probe emits **V, I and P per rail** from the one query, and the product is the control
   that catches a bad P. At n=30 a single such glitch moves the mean by 2%.
4. **Noise is NEGATIVELY autocorrelated** (lag-1 −0.28 on P, −0.41 on I), i.e. better than
   white, so averaging is unusually effective: single-sample sd is 7.7% of the reading, and
   2 s.e. reaches **0.034 W at 300 samples** — about 0.9% of an idle rail.

Idle rails, board powered and no xclbin running: **VCCINT 3.82 W** (PL + AIE core),
**VCC_SOC 3.15 W** (NoC), **VCC_PSLP_CPM5 1.72 W** (APU), VCCAUX ~1.14 W, VCC_PSFP 0.18 W,
VCC_PMC 0.13 W, VCCAUX_PMC 0.13 W.

### Two console traps, both of which defeated a guard that looked correct

- **A CONSOLE ECHOES WHAT YOU TYPE**, so a sentinel you grep for appears in the echo of the
  command that tests for it. `command -v base64 >/dev/null && echo B64_OK` **passed on an SC
  that has no `base64` at all**, because the echoed command line contains `B64_OK`. The
  `NOMD5` guard failed the same way. Every sentinel is now SPLIT (`echo SHELL"_OK"`) so the
  literal exists in the reply and never in the echo. **This is the console form of M-09: a
  check that cannot fail is not a check.**
- **The SC has no `base64`**, so the probe is carried across as a quoted heredoc — which is
  the natural encoding for a shell script typed at a shell — paced in 512-byte chunks because
  the link has no flow control, and verified by md5 afterwards. The first version chunked by
  LINE and re-joined with `"\n"`, which appended one byte to a file that already ended in a
  newline; the md5 guard caught it. Send exact bytes.

## The 2026-09-03 status: tooling done, instrument still the blocker

`--power-pause`, `power_probe.sh` and `power_measure.py` are built and exercised end to end.
The marker alignment was smoke-tested over the real ssh path with a synthetic run emitting the
exact markers: all five windows (`static`, `graph`, `run`, `graph_post`, `tail`) bin correctly.

**What is still missing is only an SC LOGIN.** `dialout` was fixed on 2026-09-03 and all four
FTDI ports open (`sg dialout -c` works without a re-login). The ports are identified:
**`/dev/ttyUSB1` is the Versal PS console** (shell prompt, already logged in) and
**`/dev/ttyUSB3` is the System Controller** — `xilinx-versal-system-controller-20222 login:`.
`root`/`root` is REJECTED. Nothing else was tried; the credential is the one remaining blocker,
and it is the user's to supply. Once in, `power_probe.sh --backend sc_app --list` over that port
dumps `sc_app`'s raw output so the domain parser can be finished against real text.
**Note that the probe's ssh transport does not fit a serial console** — a serial backend, or the
SC's own network, is the next piece of work.
