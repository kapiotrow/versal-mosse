# Claims ledger

**Status:** current · **Updated:** 2026-09-02 · **Scope:** THE INDEX — one row per question answered, with a verdict and an evidence note

**One row per question this project answered.** Seeded 2026-08-30 from `CLAUDE.md`'s *Current
status*, *Settled questions* and *Next, in order* sections; the section map was corrected the
same day against the actual thesis at
`/home/karolina/studia/MGR/6a7e0cfd038fe83922d37077/`. This file, not `CLAUDE.md`, is the
citation target: a thesis section cites a claim id, the claim cites an evidence note and a run
directory, and the run directory carries the flagstamps that make the number defensible.

**Rules.**
1. A claim's `verdict` uses exactly one of: `accepted-hw`, `accepted-offline`, `refuted`,
   `open`, `retired`. No other words.
2. `accepted-offline` may **not** be reported as a result. The offline AR proxy has a measured
   resolution of ~0.02 in R and is biased ~0.02 high; it decides what deserves board time, never
   what ships. See `evidence/pooled_features.md` for the one arm where it did not transfer.
3. Every numeric claim points at a row in `results/arms.csv`, not at prose.
4. A claim with no evidence note is a **gap**, marked `— GAP`. Those are the notes to write; the
   measurement already exists.
5. The `§` column holds the thesis's own `\label`, so it is greppable from either repo.
6. The code that implements a claim is found through `code_map.md`, generated from `@thesis`
   tags by `make code-map`. This ledger says what was established; that map says where it
   lives. Neither is edited by hand to agree with the other — the map is generated.
7. **An id is permanent and unique.** Two rows were numbered `N-17` until 2026-09-03; the
   code's `@thesis` tags bound it to `SCALE_MAX_STEP`, so the channel-reliability row became
   `N-20`. Never reuse an id, never renumber one that a tag or a note already cites, and keep
   each table sorted by id — the id IS the index.
8. **A cell states the claim and its verdict; the argument lives in the note.** Eight cells had
   grown into paragraphs restating their evidence note, which is a third copy of numbers
   `results/*.csv` already owns. Keep the decisive qualifier ("never write this up as a pass"),
   move the supporting figures out.
9. **The thesis's own `% Roszczenia:` lines outrank this column.** `make code-map` reads them
   out of the chapter stubs and reports three mismatches: a claim a stub declares with no
   tagged code, a tag carrying a claim its stub does not declare, and a tagged claim no section
   declares at all. `projekt.tex` carries those lines; `ewaluacja.tex` and `podsumowanie.tex`
   do not yet, which is why 19 claims currently have no declaring section.

## The thesis, as it actually stands

`praca.tex` includes six chapters. Prose is Polish; this ledger stays in English (it is the
working language of the evidence notes) and carries the Polish labels verbatim.

| chapter | file | state | claims land here |
|---|---|---|---|
| `cha:wstep` Wstęp | `wstep.tex` | written | — |
| `cha:teoria` Wprowadzenie teoretyczne | `teoria.tex` | **written, 322 lines** | a few design choices are *justified* here; do not report results |
| `cha:przeglad` Przegląd istniejących rozwiązań | `przeglad.tex` | **written, 140 lines** | `R-05` is the hypothesis this chapter already states; see the debts table |
| `cha:projekt` Projekt i implementacja systemu | `projekt.tex` | **219 lines: headings, scoped stubs and `% Roszczenia:` declarations** | `A-*`, `B-*`, and the design half of `P-*`, `N-*` |
| `cha:ewaluacja` Ewaluacja | `ewaluacja.tex` | **skeleton, headings only — no `% Roszczenia:` lines yet** | `R-*`, the measured half of `P-*`, `M-*`, `N-*` |
| `cha:podsumowanie` Podsumowanie | `podsumowanie.tex` | skeleton | `O-*` under `sec:dalszePrace` |

### Reverse index — what to pull when writing a given section

This is the direction you will actually use.

| thesis section | claims to write from |
|---|---|
| `sec:zalozeniaWymagania` | A-01, P-01 (the FPS target and what met it) |
| `sec:architekturaSystemu` | A-01, A-02, A-04 |
| `subsec:wyborSieci` | A-07, N-15, N-16, R-01 |
| `subsec:kwantyzacjaImpl` | B-08, N-01 |
| `subsec:przetwarzanieWstepne` | A-05, A-06 |
| `subsec:fftAie` | A-01, P-10 (why the host filter is not halved), P-05 |
| `subsec:operacjeCzestotliwosc` | A-04, A-05 |
| `subsec:arytmetyka` | **B-01…B-08** — the whole shift budget; B-06 is shared with `subsec:aktualizacjaFiltra` |
| `subsec:aktualizacjaFiltra` | A-03, B-06 (conjugation and the shared denominator), O-01, and the training-target derivation |
| `subsec:filtrSkali` | A-08, A-09, A-10, N-17, N-18, P-11 (the real-input DFT) |
| `sec:przeplywDanych` | A-04, P-06, P-05 |
| `subsec:petlaSterowania` | **P-03**, A-03, P-05 |
| `subsec:zrodlaObrazu` | R-08, R-09, N-12 (the synthetic scene and the pan) |
| `subsec:narzedziaBudowanie` | M-06, B-07 + `reproduce.md` |
| `subsec:weryfikacja` | B-07, M-04, M-06, R-09 |
| `subsec:kosztTransferow` | P-05, P-06, P-04 |
| `subsec:optymalizacjaOdrzucona` | P-07 — one case, by the thesis's own scoping |
| `subsec:modelObalony` | B-09 |
| `sec:metodykaBadan` | **M-01…M-10**, M-14, M-17, P-09 |
| `sec:zbioryTestowe` | R-08 (why streaming was needed), `runs/vot/seqs62.txt` |
| `sec:jakoscSledzenia` | **R-01, R-02, R-03, R-04**, R-06, R-07, A-09 |
| `sec:wydajnoscZasoby` | **P-01, P-02, P-04**, P-06, A-02, P-08, P-13 |
| `sec:porownanieReferencyjne` | **R-05**, **P-13**, **M-17** — and R-05 answers `przeglad`'s stated hypothesis |
| `sec:dyskusjaWynikow` | **N-01…N-11, N-14, N-19, N-20, N-21**, B-05, R-06, R-07, M-01, M-02, M-03 |
| `sec:wnioski` | R-04, R-05, N-01 |
| `sec:dalszePrace` | **O-01…O-05**, N-13 |

### Debts already written into `teoria` and `przeglad`

`teoria.tex` and `przeglad.tex` are finished prose, and they make **seventeen forward references
to `cha:ewaluacja` and `cha:projekt`**. Each is a promise that a specific measurement will
appear. All but one are already paid by a claim below — which is a strong position to be in, and
the reason to check them off explicitly rather than trust memory.

| where | what is promised | pays it | status |
|---|---|---|---|
| `teoria.tex:86` | *"Ilościowe potwierdzenie tego rozumowania"* — that the 16-channel shared denominator makes Bolme's init perturbations unnecessary | **N-02** | measured; the note is a GAP |
| `teoria.tex:133` | *"Przyjęte wartości $S$ oraz $a$, wraz z uzasadniającymi je pomiarami"* | **A-09** (a=1.04 on hardware), N-17 (the rate limit), and S=33 vs 65 | measured; GAP |
| `teoria.tex:241` | measurements justifying the three-channel (RGB) variant | **R-01** | measured; **the note is the top GAP** |
| `teoria.tex:265` | quantization's effect on tracking, *"przez porównanie potoku stałoprzecinkowego z przebiegiem odniesienia w arytmetyce zmiennoprzecinkowej"* | **N-01** — this is exactly the `gray`/`gray-float` arm | measured; GAP |
| `teoria.tex:272` | the same, restated | **N-01** | as above |
| `teoria.tex:279` | *"wyniki pomiarów... będą wielokrotnie wracać"* to the algorithm-to-resource mapping | **P-02, P-04** | measured |
| `przeglad.tex:15` | MOSSE chosen despite being the weaker tracker — *"zweryfikowana w rozdziale ewaluacja"* | **R-05** | measured |
| `przeglad.tex:61` | that in a heterogeneous system performance is decided by a stage's *share of frame time* and its transfer overhead, not by speeding one stage up | **P-04** — conv2d doubled and the frame did not move — and **P-02** | measured; **P-04 is the cleanest possible confirmation of this and should be written as such** |
| `przeglad.tex:72` | that the speedups came from fewer transfers, not fewer operations | **P-05, P-06** | measured, `results/perf.csv` |
| `przeglad.tex:136` | the stated hypothesis: *"dokładność porównywalną z metodami odniesienia przy niższej odporności"* | **R-05** | **confirmed exactly** — A 0.510 against CSRDCF's 0.519, R 0.342 against every one of the 41 |
| `przeglad.tex:36, 48, 57, 59, 81` | that `cha:ewaluacja` supplies the frame in which the reviewed work is to be read, and the limits of cross-dataset comparison | R-05, M-03 | verify each against its paragraph when writing |

The one debt with no claim behind it is **power** — see below.

### What the thesis still promises and the repo does not have

- **Power / energy per frame.** `subsec:metrykiSystemowe` promises energy per frame as the
  measure that makes embedded comparisons fair, and `sec:wydajnoscZasoby` lists `pobór mocy`.
  **PAID 2026-09-03.** `results/power.csv` carries the measurement and **P-12** the verdict:
  12.2 mJ/frame, 0.487 W dynamic, 68% of it the APU rail and the PL+AIE-ML rail not moving at
  all. Measured from the System Controller's INA226 rails (the APU exposes no current sensor)
  as a difference against an idle baseline, with four controls. `sec:wydajnoscZasoby`'s
  `pobór mocy` and `subsec:metrykiSystemowe`'s energy-per-frame are both now answerable from a
  CSV, and `docs/thesis/tables/power.tex` is generated.
- **`subsec:wymiarowoscBanku` presents the participation ratio as the proper measure of bank
  dimensionality — and claim N-07 refutes using it to *rank* banks** (random Gaussian scores
  10.69 against the shipping bank's 7.43). The theory's actual use of it, the structural rank-9
  argument, is unaffected and correct; but if `cha:ewaluacja` reports N-07, the theory needs one
  sentence saying PR bounds redundancy and does not rank quality.

**`projekt.tex` was restructured twice on 2026-08-30 and the tags followed both times.**
33 tags were remapped, each following the stub's `% Roszczenia:` line. The old labels are
recorded in the block below rather than in prose, because a dead label written as a live
reference is exactly what `make code-map` is there to catch:

```
subsec:kwantyzacja (in projekt)  ->  subsec:kwantyzacjaImpl   (duplicate with teoria resolved)
subsec:realizacja                ->  subsec:przetwarzanieWstepne
(new)                            ->  subsec:filtrSkali
sec:warstwaSterujaca             ->  + petlaSterowania, zrodlaObrazu
sec:srodowiskoProjektowe         ->  + narzedziaBudowanie, weryfikacja
subsec:kosztSterowania           ->  subsec:kosztTransferow
subsec:optymalizacjeOdrzucone    ->  subsec:optymalizacjaOdrzucona   (P-07 only now)
subsec:przewidywaniaObalone      ->  subsec:modelObalony             (B-09 only now)
```

`sec:problemyOptymalizacje` was re-scoped to **three cases only** — the ones where a result
changed a design decision. Everything else measured now belongs in the frame-time table of
`sec:wydajnoscZasoby`, which is where `P-08` went.

**Do not hand-verify this again.** `make code-map` now reads those lines and reports the three
ways code and thesis can disagree. That is what caught the second restructure.

### The chapter that does not exist

There is no chapter for refuted hypotheses or for methodology. Nineteen `N-*` rows and ten
`M-*` rows have to live in `sec:dyskusjaWynikow`, `sec:problemyOptymalizacje` and
`sec:metodykaBadan`, and that is a defensible home for them. **If any of it overflows** — N-01,
P-03, P-07 and P-10 are each big enough to be a section — the cheap fix is subsections under
`sec:dyskusjaWynikow`, not a seventh chapter, which would break the AGH structure the template
implies.

---

## A — Architecture and datapath

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| A-01 | The whole FFT/IFFT/conv/cmul chain runs on AIE; PL carries only `camera_capture` + `roi_crop`; the APU orchestrates via GMIO DDR round-trips | accepted-hw | GAP (`../engineering/performance.md`, `code_map.md`) | `runs/run_0821_1725.log` | `sec:architekturaSystemu` |
| A-02 | The design uses 2% of the AIE array; the binding constraints are tile memory (64 KB) and host DMA orchestration, never core count | accepted-hw | `results/resources.csv` | ch16 build report | `sec:wydajnoscZasoby` |
| A-03 | Filter init/update belongs on the PS with no FFT library — `F_ch` arrives transformed and `G` has a closed form | accepted-hw | GAP (`design/host_app_src/mosse_filter.cpp`, `make test_host`) | — | `subsec:aktualizacjaFiltra` |
| A-04 | The DDR accumulator is correct and the on-tile version is a graph cycle needing 16 invocations of delay, not 1 | refuted (on-tile) | GAP (`../engineering/performance.md` — `CMUL_ACCUM_MEMTILE` is the measured half) | `make graph` | `subsec:operacjeCzestotliwosc` |
| A-05 | Preprocessing splits across PL (Stage A) / AIE (B1) / APU (B2, B3) for <2% added arithmetic and no new AIE tiles | accepted-hw | GAP | — | `subsec:przetwarzanieWstepne` |
| A-06 | The periodic Hann (not symmetric) is what makes Stage B2's 9-bin correction exact | accepted-hw | GAP | measured DC/leak in Q1.15 | `subsec:przetwarzanieWstepne` |
| A-07 | Grayscale collapse must use BT.601 luminance, not Danelljan's unweighted sum, which annihilates four colour-opponent channels | accepted-offline | GAP (`scripts/check_collapse.py`) | — | `subsec:wyborSieci` |
| A-08 | Scale estimation is DSST's 1-D filter, not multi-resolution search | accepted-hw | GAP (`../engineering/scale_filter.md`, DSST Table 1) | `make scale_sim` | `subsec:filtrSkali` |
| A-09 | `SCALE_STEP=1.04` beats DSST §6.1's 1.02 on hardware (IoU 0.807 → 0.917) | accepted-hw | GAP | `runs/run_0820_1513.log` | `sec:jakoscSledzenia` |
| A-10 | A single scale correction under-corrects; the assertable property is that repeated application converges monotonically | accepted-offline | GAP (`make test_host`) | — | `subsec:filtrSkali` |

## B — Calibration and fixed-point budget

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| B-01 | The FFT shift budget is 4-4-4 and the invariant `2·FFT_SHIFT + IFFT_ROW + IFFT_COL` fixes the response scale (holds to 1.3% across splits) | accepted-hw | `evidence/shift_budget_realvideo.md` | five 200-frame runs, 08-24 | `subsec:arytmetyka` |
| B-02 | `H_SHIFT` is the only knob upstream of both the accumulator and the response, so every budget fix has been `H_SHIFT` | accepted-hw | `evidence/shift_budget_realvideo.md` | `0826_1311-hshift14` | `subsec:arytmetyka` |
| B-03 | `H_SHIFT=13` is the tight-but-safe RGB budget; 12 rails. Shipped arms stay over-shifted at gray 14 / RGB 15 with `rails=0` over 101,564 frames | accepted-hw | `evidence/shift_budget_realvideo.md` | `0826_1324-hs14survey` | `subsec:arytmetyka` |
| B-04 | `accum_max = 46340` is not overshoot — it is 32767·√2, and `rails` is the only saturation instrument | accepted-hw | `evidence/shift_budget_realvideo.md` | — | `subsec:arytmetyka` |
| B-05 | Rails do not correlate with tracking loss (corr = −0.025) — a budget defect, never a tracking fix | refuted (rails-as-cause) | `evidence/shift_budget_realvideo.md` | — | `sec:dyskusjaWynikow` |
| B-06 | Do not re-centre the response in the 49–64% band: the corrected build spreads 2.07×, so size against the tail | accepted-hw | GAP | `scripts/calib_report.py` | `subsec:arytmetyka` |
| B-07 | A calibration run's criterion is `rails=0` **plus** bit-identical tracking plus PSR not moving | accepted-hw | GAP | — | `subsec:weryfikacja` |
| B-08 | The `bias_acc` correction (`BIAS_SCALE=roi`) retires both structurally dead channels and widens signal resolution to 9.6–13.4 bits | accepted-offline | GAP (`check_collapse.py` Q3) | — | `subsec:kwantyzacjaImpl` |
| B-09 | Twice an offline model set this budget and hardware overturned it — both times the model was self-consistent and its premise was wrong | accepted-hw | `evidence/shift_budget_realvideo.md` | — | `subsec:modelObalony` |

## P — Performance

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| P-01 | 880 ms → 26.29 ms/frame (38.04 FPS), every step accepted on a bit-identical-tracking test. **The SHIPPING arm was measured on the serial console 2026-09-04: 25.82 ms = 38.7 FPS** on `car1`, 742 frames, 0 gated — a different arm on a different scene, so it extends the table and does not continue the ladder | accepted-hw | `results/perf.csv`, `evidence/frame_time_shipping.md` | `runs/run_0821_1725.log`, `runs/perf/0904_l1relu_console/` | `sec:wydajnoscZasoby` |
| P-02 | The frame is 84% CPU-bound, not wait-bound; only 41% of GMIO blocks. **RE-MEASURED on the shipping arm 2026-09-04: 24.4 of 24.91 ms, and only 31% of GMIO blocks — but 7.5 ms of that "CPU" is a BUSY-WAIT on `roi_crop`'s `ap_done`**, so the frame is more CPU-bound and less of it is work | accepted-hw | `results/frame_budget.csv`, `evidence/frame_time_shipping.md` | `run_0821_1725`, `runs/perf/0904_l1relu_console/` | `sec:wydajnoscZasoby` |
| P-03 | The CU completion interrupt is never delivered on this platform: every KDS launch costs ~503 ms. `ROI_CROP_USER_MANAGED=1` is worth 20.6× on frame rate | accepted-hw | GAP — **write this one, it is a self-contained chapter** | `ISR=0x3`, `cu_stat` | `subsec:petlaSterowania` |
| P-04 | RGB costs what the host pays (+2.29 ms), not what conv2d costs (+4.59 ms of AIE time that never appears in the frame) | accepted-hw | `results/frame_budget.csv` | `run_0824_1457` | `subsec:kosztTransferow` |
| P-05 | Memory-tile transpose, `CMUL_SPLIT_ACCUM`, `TAIL_PARALLEL`, blocked `unpack_spectrum`, pipelined `roi_crop` each measured | accepted-hw | `results/perf.csv` | 08-21 runs | `subsec:kosztTransferow` |
| P-06 | DMA is not a bottleneck: 80 µs/tx is per-transaction overhead; the largest transfer achieves 5.76 GB/s | accepted-hw | GAP | — | `sec:przeplywDanych` |
| P-07 | Parallel-for inside `filter_update_quantize` — ~0.96 ms, abandoned: FMA contraction is sensitive to inlining context, 1 ulp of A is disqualifying | refuted | GAP — **the FMA/inlining finding is publishable on its own** | — | `subsec:optymalizacjaOdrzucona` |
| P-08 | `FFT_COL_WS` 8→32 is a 9.57 ms loss; `CMUL_ACCUM_MEMTILE` alone is a 0.36 ms loss | refuted | GAP | `0821_colws32`, `0821_accmem` | `sec:wydajnoscZasoby` |
| P-09 | Phase 4's console knobs are worth 1.1%; the transport (UART→ssh, 3.79 ms) was the whole win | accepted-hw | `evidence/phase4.md` | three `car1` runs | `sec:metodykaBadan` |
| P-10 | Halving the host filter on Hermitian symmetry does not work — the premise is false in fixed point (95.8% of bins differ from their conjugate partner) | refuted | GAP — **has the ideal control: the float golden is Hermitian to 0 LSB** | `make aiesim_plio` | `subsec:fftAie` |
| P-11 | fDSST's PCA compression is not worth it; the real-input DFT was (3.11×, transferred exactly to hardware) | refuted (PCA) / accepted-hw (DFT) | GAP | `run_0821_1109` | `subsec:filtrSkali` |
| P-12 | **Energy per frame is 12.2 mJ (0.487 W dynamic at 25.16 ms).** 68% is the APU rail, 23% LPDDR4, 9% NoC; **the PL+AIE-ML rail does not move (< 33 mW)** and the design's RESIDENT cost is unresolved on every rail. Confirms P-02's CPU-bound frame in the energy domain, on a different chip and instrument | accepted-hw | `evidence/power.md` | `runs/power/0903_l1relu_scapp`, `results/power.csv` | `subsec:metrykiSystemowe` |
| P-13 | Against the nearest published embedded equivalent (Danilowicz & Kryjak 2022, deepDCF on ZCU104, same geometry): **14.9x fewer LUT, 25.2x fewer FF, 20.8x less BRAM, 8.6x fewer DSP**, at 32 channels and 33 scales against their 8 and 3. **The mechanism is the AIE array, not the filter.** Their FPS is a per-scale PL projection — never divide the two. **Re-measured on the ROUTED shipping build 2026-09-04; every ratio fell** (was 20.4/44.4/54.1/10.9x, an HLS estimate of one kernel on a gray ch1 build) | accepted-hw | `evidence/embedded_comparison.md` sec.6 | `results/resources.csv` (`build=rgb_l1relu`), `results/embedded_baselines.csv` | `sec:porownanieReferencyjne` |
| P-14 | **The console transport term is 0.91 ms on the shipping arm, not `P-09`'s 3.79 ms** — the cost is proportional to bytes on the wire and this build prints at `PROGRESS_EVERY=25`. Control: the console/ssh pair agrees stage-for-stage to <0.15 ms and the whole difference lands in `unattributed`, to within 0.006 ms of the frame delta. **Also: `roi_crop` is now 28.7% of the frame and 99% of that is the `ap_done` POLL** (7.495 of 7.569 ms), which re-ranks the performance roadmap. **And the SYNTHETIC SCENE CANNOT MEASURE THIS ARM** — it gates 128 of 199 frames, so the tail runs on 0.4 calls/frame | accepted-hw | `evidence/frame_time_shipping.md` | `runs/perf/0904_l1relu_console/`, `results/perf.csv`, `results/frame_budget.csv` | `sec:wydajnoscZasoby` |

## R — Tracking results

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| R-01 | RGB features beat the BT.601 collapse: R 0.2743 → 0.3065, EAO 0.1367 → 0.1474, 12.8% more frames survive | accepted-hw | GAP — **the highest-value missing note; the colour-free control belongs in it** | `0826_1336-hs14full`, `0826_1550-rgb15` | `sec:jakoscSledzenia` |
| R-02 | `MOSSE_ETA=0.05` ships: R +0.0218, EAO +8.5% — but two of three mechanism falsifiers fired | accepted-hw | `evidence/eta05.md` | `0827_1441-eta05` | `sec:jakoscSledzenia` |
| R-03 | `PSR_GATE_MIN=5.0` is worth +0.0134 R at 128x128, and **the "conditional on the PSR scale" half is REFUTED ON HARDWARE 2026-09-01**: at 64x64 the PSR scale fell and the gate fired 17x more often, but rescaling the threshold to 3.5 returned **an EAO null to four decimals**. The mis-scaling was cosmetic because 95.9% of vetoes fire AFTER the run is already lost | accepted-hw, mechanism refuted | `results/arms.csv` rows `rgb_eta05_gate5`, `rgb_res64_gate35` | `0827_1642-eta05_g5p0`, `0901_1442-res64_g35` | `sec:jakoscSledzenia` |
| R-04 | The shipping arm on 62 sequences / 419 trajectories: **A 0.5129 / R 0.4279 / EAO 0.1960** (`rgb_l1relu`, 2026-09-02). The arm history — every row one knob — is `results/arms.csv`, not this cell | accepted-hw | `results/arms.csv` row `rgb_l1relu`, `evidence/arm_l1relu.md` | workspace `0902_l1relu` | `sec:jakoscSledzenia` |
| R-05 | Accuracy is inside the classical-DCF band (A 0.510, under CSRDCF by 0.009); robustness is below all 41 published trackers | accepted-hw | `results/baselines.csv` | Kristan et al. 2022 Table 12 | `sec:porownanieReferencyjne` |
| R-06 | The loss mechanism is attributed: the tracker walks off target confidently (ACCEPT 82.0% at median PSR 18.83 in the 5 pre-loss frames); the gate is the aftermath, not the cause | accepted-offline | `evidence/robustness_gap.md` | CSVs only | `sec:dyskusjaWynikow` |
| R-07 | On targets that genuinely translate the detector recovers 93% of annotated motion — localisation is not the fault | accepted-offline | `evidence/detector_gain.md` | — | `sec:dyskusjaWynikow` |
| R-08 | The board's usable heap is ~0.9–1.2 GB, not 12 GB; streaming (`VOT_STREAM_RING`) recovered the five RGB sequences that died on `std::bad_alloc`, with identical digests both ways | accepted-hw | `evidence/board_memory.md` | `0827_1313-streamA/B` | `subsec:zrodlaObrazu` |
| R-09 | Multi-start determinism: two runs of the same job return byte-identical trajectories, and `RESET_MUTANT` proves the test can fail | accepted-hw | `evidence/phase3.md` | — | `subsec:weryfikacja` |
| R-10 | The spatial mask on the OLD bank: EAO 0.1629 → 0.1740 (+0.0110), A +0.0179 on the common survived prefix — but **not distinguishable from a null** (3 sequences of 62 carry it, median dR 0.0000, P(dR≤0)=0.22) and the mechanism is refuted. **It does not carry to the shipping arm: re-screened 2026-09-03 the sign INVERTS (N-21), so this row scores a configuration the board no longer runs** | **weak-hw** | `evidence/spatial_mask.md`, `evidence/mask_bank_transfer.md` | `0831_1528-mask` vs `0831_1340-base_stat` | `sec:jakoscSledzenia` |
| R-11 | The MAINLOBE WIDTH `sigma/target` is the axis, with an optimum at 1/16 (DSST's `target/16`). **The RATIO was challenged and UPHELD against `sigma/map` (`R-18`), with the one deviating column independently explained by aliasing.** **At matched width the finer map wins POOLED (+0.0222 R) — but that difference is NOT paired-stable and is a null across sequences (`R-14`), so it is not an established effect.** The sigma INTERIOR is closed (22-cell grid; 3, 5, 6, 8 all worse) | accepted-hw (width); **pooled-only (resolution)** | `evidence/arm_res64.md` sec.21, sec.25 | `results/arms.csv` row `rgb_sigma4`; `runs/vot/0902_offline-sigmaeta/` | `sec:jakoscSledzenia`, `sec:dyskusjaWynikow` |
| R-12 | **The scoring path reproduces a published tracker**: CSRDCF through `offline_multistart.py` -> `vot_ingest.py` lands at A 0.5134 / R 0.5471 / EAO 0.2432 against a published 0.519 / 0.580 / 0.251 (EAO −0.0078). The oracle control returns **R = 1.0000 exactly** over all 419 runs. `opencv-kcf` does NOT reproduce and that is a cv2 implementation difference, not a path defect — never quote it as KCF | accepted-hw | `evidence/harness_validation.md`, `results/reference_trackers.csv` |
| R-13 | **The float twin of this tracker beats the board arm: R 0.4559 against 0.4279, EAO 0.2041 against 0.1960, paired dR +0.0213 (trim-5 +0.0018, P(dR<=0)=0.053), accuracy a null.** The pre-registered prediction from `settled.md` FIRED. **The CONFOUND IS RESOLVED by `R-16`**: the board's `SCALE_N=1` run supplies the missing cell, the scale term is a null, and the deconfounded arithmetic term is LARGER than this contrast | accepted-hw | `evidence/float_twin.md`, `evidence/fixed_point_cost.md`, `results/float_twin.csv` | workspaces `offline_paired`, `0904_twin_s1` | `sec:dyskusjaWynikow` |
| R-14 | **A known-answer geometry calibration of the offline multistart harness, which passed on sign and invalidated its own reference.** The harness is no longer inverted (pooled +0.0119 R to the 128 map, hardware's sign, ~half the magnitude) — **but hardware's own +0.0222 R resolution term is POOLED-ONLY: paired it is +0.0049, median 0.0000, trim-5 −0.0040, P(dR<=0)=0.205.** Five documents quoted it as established; all corrected | accepted-hw | `evidence/harness_validation.md`, `results/geometry_calibration.csv` | workspaces `0902_cmp`, `offline_cal` | `sec:metodykaBadan` |
| R-15 | **Danilowicz & Kryjak's deepDCF run on THIS project's benchmark**, from a pinned unmodified clone (`ee0f93ab`, MIT) through `offline_multistart.py` -> `vot_ingest.py`. It removes `M-17`'s three incomparabilities at once by moving THEIR tracker to OUR protocol rather than the reverse. **Their arms here are the FLOAT software model — the 4-bit checkpoint of every Table 1 row is absent from their repo — so the VOT2015 known-answer check is a BAND check, never a reproduction.** Also tests, on their own tracker, whether their +0.024 EAO geometry ordering survives holding the MAINLOBE WIDTH fixed: their `sigma` is in feature-map bins with no geometry term, the same confound `R-11`/`R-14` caught here | open | `evidence/deepdcf_reproduction.md` | pinned checkout `external/deep_mosse` | `sec:porownanieReferencyjne` |
| R-16 | **`SCALE_N=1` on the board decides `R-13`: the ARITHMETIC, not the scale filter.** Disabling the DSST filter is a NULL (dR trim-5 **-0.0210**, P(dR<=0)=0.858; pooled inverts in sign). At matched scale handling, float64 vs fixed-point is dR trim-5 **+0.0102**, 35/12, P=**0.001**. **Scope: the fixed-point CORRELATION pipeline, NOT the int8 FEATURE path** | accepted-hw | `evidence/fixed_point_cost.md`, `results/scale_ablation.csv` | `runs/vot/0904_1225-l1relu_s1`, workspace `0904_twin_s1`; `arms.csv` row `rgb_l1relu_s1` | `sec:dyskusjaWynikow` |
| R-17 | **`M-14` run in reverse: a REFUTATION is scoped to its bank too.** Re-screening the pre-2026-09-02 refutations on the Layer-1 bank, **Bolme's init perturbations are refuted A FORTIORI**: the denominator floor is **173x higher** and the fraction of bins below `1e-3*mean(B)` falls **15.80% -> 0.000%**, so the low-energy-bin defect perturbations cure is ABSENT. **Corollary: the 16% of runs failing within 10 frames of init are NOT a conditioning problem**, closing that whole family of cures. **`eps_rel=1e-3` is EXPIRED and re-screened: INERT across 1e-5..3e-3** (trim-3 exactly 0.0000), and nothing above it is trim-stable (`3e-2` mean +0.0112, median 0.0000, trim-3 -0.0009). The value stands; the ORIGINAL optimality claim is now vacuous, the curve being flat | accepted-offline | `evidence/layer1_rescreen.md` | `sec:dyskusjaWynikow` |
| R-18 | **`R-11`'s invariant TESTED AND UPHELD.** A (padding x sigma) grid with all three columns BRACKETED: best sigma is **2.667 / 2.000 / 3.500** at padding 1.5 / 2.0 / 3.0. `sigma/target` predicts 2.667/2.000/1.333 and matches **two of three**; `sigma/map` predicts a flat 2.000 and matches only the degenerate middle column. The lone exception is padding 3.0, which `settled.md` independently records as tripping the ALIASING detector, and it deviates in the direction that defect predicts (noisier features favour a wider target). **CORRECTED from a first reading of the same grid taken before the sigma range was extended — with padding 3.0 unbracketed it looked as though `sigma/map` won.** Corroborated on Danilowicz's tracker, where width-controlling their geometry comparison made the effect LARGER (+0.0060 -> +0.0149 EAO), not smaller. **Padding is not a clean lever for width above 2.0** | accepted-offline | `evidence/layer1_rescreen.md`, `evidence/deepdcf_reproduction.md` | `sec:metodykaBadan` |

## N — Refuted, and the reason the obvious explanation was wrong

| id | claim tested | verdict | evidence | § |
|---|---|---|---|---|
| N-01 | Quantization causes the poor robustness | **refuted — removing it makes tracking worse** (mean IoU 0.2533 → 0.2350, 0 of 8 sequences improve) | GAP — **write this; it is the thesis's strongest single result** | `sec:dyskusjaWynikow` |
| N-02 | Bolme §3.4 init perturbations would help | refuted — the 16-channel shared denominator is already the cure; Bolme's Fig. 3 is captioned "without regularization" | GAP (`0828_offline-warp8`) | `sec:dyskusjaWynikow` |
| N-03 | POOLING would help | **refuted, and on the RECTIFIED bank it is a LOSS, not a null.** The aggregation OPERATOR is irrelevant (average, max and decimate agree to 0.001); the 2x2 screen lost on the rectified bank AND on its linear negative control, where the original LINEARITY argument predicted a null, so **that explanation is WITHDRAWN** — the operative reading is RESOLUTION. Does not speak to Danilowicz's 3x3/stride-1 stem, which pools down TO its working resolution | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| N-03b | LOWER FEATURE RESOLUTION would help | **confirmed on hardware 2026-09-01 and RE-ATTRIBUTED the same day**: the gain was the MAINLOBE WIDTH the arm carried by accident (see R-11), not the resolution, which is a small loss at matched width. The first pre-registered proposal here to pass its own falsifier; the offline proxy called the sign right and 43% of the magnitude | `evidence/arm_res64.md`, `evidence/pooled_features.md`; `results/arms.csv` row `rgb_res64` | `sec:jakoscSledzenia`, `sec:dyskusjaWynikow` |
| N-04 | `TARGET_PADDING=3.0` (predicted +0.088 R) | refuted on hardware — R +0.0077 and **EAO down**; the offline proxy had no scale filter and was single-start | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| N-05 | Padding below 2.0 | refuted — every value worse (R 0.2251 at 1.5) | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| N-06 | A better pretrained feature bank would help | **refuted, and since 2026-09-01 for a STRUCTURAL reason**: with `CONV_RELU=0` the bank is a LINEAR LIFT, so only its row space matters and the online filter absorbs any change of basis — a **one-hot identity lift with no network at all** ties the pretrained bank. Originally: pretraining at layer 1 is worth ~0.011-0.015 R, below the bench's resolution. What DOES pay is geometry, not weights (O-04) | `evidence/feature_bank.md` | `sec:dyskusjaWynikow` |
| N-07 | Participation ratio ranks feature banks | **refuted — PR is maximised by noise** (random Gaussian 10.69 vs the shipping bank's 7.43) | `evidence/feature_bank.md` | `sec:dyskusjaWynikow` |
| N-08 | `HOLD_COAST=1` improves tracking | refuted on the metric of record — wins mean IoU (+0.0296), loses A, R and EAO on the identical 54 pairs | `evidence/hold_policy.md`, `evidence/metric_ar_vs_iou.md` | `sec:dyskusjaWynikow` |
| N-09 | `nature` is a tracker defect | refuted — its pixels do not move; on 80% of frames *not moving* correlates better (NCC 0.940 vs 0.816) | `evidence/frozen_detector.md` | `sec:dyskusjaWynikow` |
| N-10 | `tiger` is a filter or learning-rate defect | refuted — a plain NCC template search with no filter puts the best match 11 px off the annotation; it is `nature`'s disease, milder | `evidence/tiger.md` | `sec:dyskusjaWynikow` |
| N-11 | Sub-bin quantisation lag compounds | refuted — the detector measures the offset that exists now, not the increment; error bounded at ~half a bin, worst late/early ratio 1.00 | `evidence/subbin_lag.md` | `sec:dyskusjaWynikow` |
| N-12 | Background lock explains the training-target failure | refuted as the explanation (the mechanism is real and measured); `BG_PAN` decorrelates 6.6× and changed the tracker not at all | GAP | `subsec:zrodlaObrazu` |
| N-13 | Re-detection / search-window expansion would help | retired by the protocol — VOT terminates 10 frames after failure, so recovery after a loss scores nothing | `evidence/robustness_proposals.md` | `sec:dalszePrace` |
| N-14 | Relaxing the gate would help | refuted — 88% of vetoes are `NEGATIVE_PEAK`, which `PSR_GATE_MIN` cannot disable, and 95.8% land after the run is already lost | `evidence/robustness_gap.md` | `sec:dyskusjaWynikow` |
| N-15 | Channel pruning is worth doing | retired — moot with ReLU off and `BIAS_SCALE=roi`; the real redundancy is the rank-9 collapse, whose fix is RGB | GAP | `subsec:wyborSieci` |
| N-16 | ReLU on (as in the donor network) | **mechanism CONFIRMED ON HARDWARE 2026-09-03.** The pre-registered linear twin `l1lin` — same bank (identical weights md5), same geometry, `app.flagstamp` byte-identical, `aie.flagstamp` differing on exactly `CONV_RELU` — lands at **EAO 0.1851 against 0.1960**, so the gain is the RECTIFIER and not the bank. Paired R **+0.0447 mean, +0.0274 after drop-top-5, sign p 0.018, P(dR<=0)=0.000**, with A moving the same way (no trade; the pooled A reversal is the `M-02` selection effect). **The shipping arm still did NOT pass its own dEAO>=+0.005 bar** (`arm_l1relu.md` sec.10.1/12) — that is a separate question and unchanged | `results/arms.csv` rows `rgb_l1relu`, `rgb_l1lin`; `evidence/arm_l1relu.md` sec.14 | `subsec:wyborSieci`, `sec:jakoscSledzenia` |
| N-17 | `SCALE_MAX_STEP=1` | refuted — parks the sim's smooth arm 123 of 200 frames and ends 28.0% wrong | GAP (`make scale_sim`) | `subsec:filtrSkali` |
| N-18 | `SCALE_CONF_MIN` distinguishes a wrong proposal from a big correct one | refuted — it cannot; both match the model poorly for the same reason | GAP | `subsec:filtrSkali` |
| N-19 | `eps_rel` and `MOSSE_SIGMA` need tuning | settled/retired — ε=1e-3 optimal (closed form `R = G·B/(B+ε)`); PSR is monotone in σ and so cannot select it | GAP | `sec:dyskusjaWynikow` |
| N-20 | Channel reliability in Stage B3 (CSR-DCF's third contribution) would help | refuted offline 2026-09-01 — **the mechanism HOLDS** (the anti-reliability mutant loses 0.0396 R, so the statistic carries real information) but the gain does not: `gamma=0.5` is +0.0210 pooled and **+0.0023 after drop-top-3** with 39 of 62 sequences exactly tied, `gamma=1` a null, `gamma=2` −0.058. No board time . **The one licensed re-open (a moved operating point) is SPENT: re-screened 2026-09-03 on the shipping Layer-1 arm at dR −0.0040, trim-3 −0.0378, 39 of 62 tied** | `evidence/channel_reliability.md`, `evidence/mask_bank_transfer.md` | `sec:dyskusjaWynikow` |
| N-21 | The spatial mask's value carries across the feature bank | **refuted offline 2026-09-03 — the sign INVERTS.** Same script, sequences, eta and gate: dR **+0.0601** on the old 3x3/16ch/128x128 bank against **−0.0127** paired (trim-3 −0.0358, P(dR≤0)=0.706, 35 of 62 tied) on the SHIPPING Layer-1 bank; the k=2 width knob is worse still. The mechanism HELD both times (`e_box` 0.6795 → 0.9547, non-overlapping quartiles), so a confirmed mechanism bounds attributability and never value. **Why it inverts is OPEN** — the "the Layer-1 filter already self-concentrates" guess is not supported by a baseline `e_box` of 0.6795 against 0.6049 | `evidence/mask_bank_transfer.md` | `sec:dyskusjaWynikow` |
| N-22 | A confidence-modulated learning rate (LMCF's high-confidence update) would help | **refuted offline 2026-09-03 on BOTH statistics, and the MUTANT does not lose** — every arm is negative (best −0.0108 against a +0.02 bar), more modulation is monotonically worse, and inverting the law is not worse than the correct law (PSR −0.0047 pooled; APCE's +0.0133 pooled dies at trim-3 −0.0153, sign test 1.000). So the statistic is INERT: unlike N-20 **no mechanism survives this null**. The within-run dip is real (P=0.618) and still not a usable control input | `evidence/confidence_eta.md` | `sec:dyskusjaWynikow` |
| N-23 | PSR (and APCE) rise and fall with tracking CORRECTNESS | **refuted — the relation is NON-MONOTONE, and the most confident band is the worst.** Fraction of frames already lost, per sequence then medianed: 70.2% at PSR 0-10, **48.4% at 20-30 (the optimum)**, 77.8% at 30-50, **96.9% above 50** (17 of 20 sequences over 50%). Mechanism: those frames are WELDED to static background — median box motion **0.00 px** while the truth moves 2.24 px/frame — so the filter self-correlates more sharply than on a deforming target. Generalises the PSR-33-at-179-px anecdote to 9,034 frames and identifies TWO lost populations, wandering and welded | `results/psr_correctness.csv`, `results/psr_welded_motion.csv`, `evidence/confidence_eta.md` | `sec:dyskusjaWynikow` |
| N-24 | A second filter with different temporal support sees drift that one does not (the two-filter ensemble premise) | **refuted offline 2026-09-03.** Long-term/short-term PEAK DISAGREEMENT does not predict an imminent loss: `P[healthy < doomed]` = **0.461** frozen and **0.555** slow, against PSR's 0.618 which is already closed as too weak. The filters do NOT agree (3.61 bins apart on healthy frames) — the disagreement is large and UNINFORMATIVE. With confidence closed by `N-23`, a per-frame SELECTION rule has no signal left to use, so the ensemble is closed in both its cheap and expensive forms | `results/lt_divergence.csv`, `evidence/confidence_eta.md` | `sec:dyskusjaWynikow` |

## M — Methodology claims (chapter 8)

| id | claim | evidence |
|---|---|---|
| M-01 | An offline R can be raised by **degrading** the filter — demonstrated with a deliberately broken init (dR +0.0525, 6.5× the correct arm, while tracking worse by every direct measure) | `evidence/arm_mask.md` |
| M-02 | Accuracy is averaged over tracked frames, so a longer-surviving arm is scored on harder ones — score A on the common survived prefix | `evidence/arm_mask.md` |
| M-03 | Mean IoU and the toolkit's AR order two arms **oppositely on the same runs** | `evidence/metric_ar_vs_iou.md` |
| M-04 | A byte-identical trajectory is not a bit-identical run — found by a negative control, fixed with a per-frame FNV-1a digest | `evidence/phase3.md` |
| M-05 | Never size a change from one member of an interleaved async/wait group — the single most repeated measurement error in this design (4 occurrences) | GAP |
| M-06 | Verify a feature flag on a build that can exercise it; `strings` on the ELF can report a false absence | `evidence/phase4.md` |
| M-07 | A multi-start CSV collides on frame index — keyed by frame alone it under-reported rails 66× next to a confident wrong verdict | GAP |
| M-08 | Write predictions and falsifiers down before the run; two of three fired on the arm that shipped anyway | `evidence/eta05.md` |
| M-09 | Benchmark a host-side change on the host — the *ordering* did not transfer, not just the magnitude, because the working set crossed a cache boundary | GAP |
| M-10 | A caveat that is not priced is a hope | `evidence/pooled_features.md` |
| M-11 | An aggregate rate can invert when split by the state it depends on: the mask's `NEGATIVE_PEAK` drop (15.4% → 10.2%) is a RISE on the frames that matter (3.07% → 3.87% on target) | `evidence/spatial_mask.md` |
| M-12 | **A detector that reports "no change" is not necessarily blind — separate the two with the NULL RATE, not the gain.** The scale filter's gain reads as blindness; its null rate (88.4% against 3.0% for a noise argmax) shows it is strongly LOCKED to the current size instead, and the same estimator reaches gain 0.93 in the sim once the target outruns the lock. The loop is self-confirming and the code is sound | `../engineering/scale_filter.md` |
| M-13 | **Bound the prize before attributing the defect.** The scale filter was root-caused before anyone asked what repairing it was worth; an ORACLE puts a PERFECT filter at +0.0023 R. Attribution says WHY, an oracle says WHETHER TO CARE. **Corroborated closed-loop 2026-09-04** by a method that could have contradicted it: accuracy +0.0766 trimmed (P=0.000), survival none (dR trim-5 −0.0032) | `../engineering/scale_filter.md`, `evidence/float_twin.md` |
| M-14 | **A prior positive screen expires when the operating point moves.** Re-screen whenever the bank, the geometry or the response shape has changed since the arm was scored — not only when something feels suspect. The spatial mask was proposed for hardware on 2026-09-03 on the strength of a screen run against a bank the board no longer runs; eight minutes of CPU inverted it and saved a sweep, an ingest and a card round-trip | `evidence/mask_bank_transfer.md` |
| M-15 | **Check a statistic's SHAPE against ground truth before building a control law on it.** A confidence-modulated eta was built, screened and refuted before anyone plotted PSR against IoU; the relation is U-shaped (`N-23`), so both the law and its inverted mutant were misspecified and the mutant's failure to lose — the one result that should have been diagnostic — was uninterpretable until the shape was known. Monotone law, non-monotone statistic | `evidence/confidence_eta.md` |
| M-16 | **Match the test to how much data the runs actually give.** The ensemble probe was first scored by comparing two 5-frame windows per run — n = 27 on a single-start bench, and it returned 0.631, which read as a signal. The per-frame form uses the SAME runs, has 7,032 samples instead of 27 pairs, and inverts the reading to 0.555. A window statistic borrowed from a 419-run board protocol is underpowered on a 62-run bench | `evidence/confidence_eta.md` |
| M-17 | Danilowicz & Kryjak's tracking numbers are not comparable to this project's in three independent ways: `R`'s definition and sign, the EAO window, and polygon vs mask-fitted ground truth. **The window term alone is +0.0827 on the shipping arm — 1.39x this project's entire measured arm ladder (0.0593)** — while leaving the ladder's ORDERING intact | accepted-hw | `evidence/embedded_comparison.md` sec.5, `results/eao_window.csv` |

## O — Open, at the time of writing

| id | item | state | evidence |
|---|---|---|---|
| O-01 | Spatial mask on the filter (`FILTER_MASK=1`) | **SWEPT 2026-08-31, NO LONGER OPEN — see R-10.** EAO rose +0.0110 but the arm is not separable from a null, the predicted `PSR_GATE_MIN` re-tune is unsupported (the gate's bite did not move), and the CSR-DCF mechanism is refuted here: it is a PROJECTION not a constraint, and the unmasked filter self-concentrates unaided. The id is kept because `@thesis` tags bind to it. **Closed twice over 2026-09-03: on the SHIPPING bank the offline sign inverts (N-21)** | `evidence/spatial_mask.md`, `evidence/arm_mask.md`, `evidence/mask_bank_transfer.md` |
| O-02 | Channel reliability in Stage B3 | **REFUTED offline 2026-09-01, and re-refuted at the new operating point 2026-09-03 — see N-20.** Was: untested; both halves available by Parseval inside the existing loop | `evidence/robustness_proposals.md`, `evidence/mask_bank_transfer.md` |
| O-03 | Two-filter temporal ensemble (short/long eta) | **CLOSED 2026-09-03, both halves.** The cheap half (a long-term filter as a confidence VALIDATOR feeding a modulated eta) is refuted by `N-22`; the premise underneath it — that a second memory sees drift a single filter does not — is refuted by `N-24`, peak disagreement at AUC 0.461/0.555. No signal remains for a per-frame selection rule, so the expensive form (a second AIE bank) has nothing to select on | `evidence/confidence_eta.md` |
| O-04 | Feature-bank **geometry** — not the weights | **CONFIRMED AND SHIPPED 2026-09-02.** It was the only feature axis left (the weights are a linear lift; one-hot ties the network) and it is the one that paid: 7x7 stride 2, 32 channels, resnet18-PCA bank, ReLU on, 64x64 map → EAO 0.1960. Geometry, not weights, is where feature work belongs | `evidence/feature_bank.md`, `evidence/arm_l1relu.md` |
| O-05 | Software-pipeline the channel loop (~8.7 ms exposed) | untested | `../engineering/roadmap.md` §PERFORMANCE |
