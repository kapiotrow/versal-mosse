# Claims ledger

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
7. **The thesis's own `% Roszczenia:` lines outrank this column.** `make code-map` reads them
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
| `sec:metodykaBadan` | **M-01…M-10**, P-09 |
| `sec:zbioryTestowe` | R-08 (why streaming was needed), `runs/vot/seqs62.txt` |
| `sec:jakoscSledzenia` | **R-01, R-02, R-03, R-04**, R-06, R-07, A-09 |
| `sec:wydajnoscZasoby` | **P-01, P-02, P-04**, P-06, A-02, P-08 |
| `sec:porownanieReferencyjne` | **R-05** — and it answers `przeglad`'s stated hypothesis |
| `sec:dyskusjaWynikow` | **N-01…N-11, N-14, N-19**, B-05, R-06, R-07, M-01, M-02, M-03 |
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
  **Nothing in `results/` measures power.** Either take the measurement or cut the promise.
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
| A-01 | The whole FFT/IFFT/conv/cmul chain runs on AIE; PL carries only `camera_capture` + `roi_crop`; the APU orchestrates via GMIO DDR round-trips | accepted-hw | `CLAUDE.md` §Architecture — GAP | `runs/run_0821_1725.log` | `sec:architekturaSystemu` |
| A-02 | The design uses 2% of the AIE array; the binding constraints are tile memory (64 KB) and host DMA orchestration, never core count | accepted-hw | `results/resources.csv` | ch16 build report | `sec:wydajnoscZasoby` |
| A-03 | Filter init/update belongs on the PS with no FFT library — `F_ch` arrives transformed and `G` has a closed form | accepted-hw | `CLAUDE.md` §Key design decisions — GAP | — | `subsec:aktualizacjaFiltra` |
| A-04 | The DDR accumulator is correct and the on-tile version is a graph cycle needing 16 invocations of delay, not 1 | refuted (on-tile) | `CLAUDE.md` §Key design decisions — GAP | `make graph` | `subsec:operacjeCzestotliwosc` |
| A-05 | Preprocessing splits across PL (Stage A) / AIE (B1) / APU (B2, B3) for <2% added arithmetic and no new AIE tiles | accepted-hw | GAP | — | `subsec:przetwarzanieWstepne` |
| A-06 | The periodic Hann (not symmetric) is what makes Stage B2's 9-bin correction exact | accepted-hw | GAP | measured DC/leak in Q1.15 | `subsec:przetwarzanieWstepne` |
| A-07 | Grayscale collapse must use BT.601 luminance, not Danelljan's unweighted sum, which annihilates four colour-opponent channels | accepted-offline | GAP (`scripts/check_collapse.py`) | — | `subsec:wyborSieci` |
| A-08 | Scale estimation is DSST's 1-D filter, not multi-resolution search | accepted-hw | `evidence/phase1.md` | `make scale_sim` | `subsec:filtrSkali` |
| A-09 | `SCALE_STEP=1.04` beats DSST §6.1's 1.02 on hardware (IoU 0.807 → 0.917) | accepted-hw | GAP | `runs/run_0820_1513.log` | `sec:jakoscSledzenia` |
| A-10 | A single scale correction under-corrects; the assertable property is that repeated application converges monotonically | accepted-offline | GAP (`make test_host`) | — | `subsec:filtrSkali` |

## B — Calibration and fixed-point budget

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| B-01 | The FFT shift budget is 4-4-4 and the invariant `2·FFT_SHIFT + IFFT_ROW + IFFT_COL` fixes the response scale (holds to 1.3% across splits) | accepted-hw | `evidence/TODO_shift_budget.md` | five 200-frame runs, 08-24 | `subsec:arytmetyka` |
| B-02 | `H_SHIFT` is the only knob upstream of both the accumulator and the response, so every budget fix has been `H_SHIFT` | accepted-hw | `evidence/TODO_shift_budget.md` | `0826_1311-hshift14` | `subsec:arytmetyka` |
| B-03 | `H_SHIFT=13` is the tight-but-safe RGB budget; 12 rails. Shipped arms stay over-shifted at gray 14 / RGB 15 with `rails=0` over 101,564 frames | accepted-hw | `evidence/TODO_shift_budget.md` | `0826_1324-hs14survey` | `subsec:arytmetyka` |
| B-04 | `accum_max = 46340` is not overshoot — it is 32767·√2, and `rails` is the only saturation instrument | accepted-hw | `evidence/TODO_shift_budget.md` | — | `subsec:arytmetyka` |
| B-05 | Rails do not correlate with tracking loss (corr = −0.025) — a budget defect, never a tracking fix | refuted (rails-as-cause) | `evidence/TODO_shift_budget.md` | — | `sec:dyskusjaWynikow` |
| B-06 | Do not re-centre the response in the 49–64% band: the corrected build spreads 2.07×, so size against the tail | accepted-hw | GAP | `scripts/calib_report.py` | `subsec:arytmetyka` |
| B-07 | A calibration run's criterion is `rails=0` **plus** bit-identical tracking plus PSR not moving | accepted-hw | GAP | — | `subsec:weryfikacja` |
| B-08 | The `bias_acc` correction (`BIAS_SCALE=roi`) retires both structurally dead channels and widens signal resolution to 9.6–13.4 bits | accepted-offline | GAP (`check_collapse.py` Q3) | — | `subsec:kwantyzacjaImpl` |
| B-09 | Twice an offline model set this budget and hardware overturned it — both times the model was self-consistent and its premise was wrong | accepted-hw | `evidence/TODO_shift_budget.md` | — | `subsec:modelObalony` |

## P — Performance

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| P-01 | 880 ms → 26.29 ms/frame (38.04 FPS), every step accepted on a bit-identical-tracking test | accepted-hw | `results/perf.csv` | `runs/run_0821_1725.log` | `sec:wydajnoscZasoby` |
| P-02 | The frame is 84% CPU-bound, not wait-bound; only 41% of GMIO blocks | accepted-hw | `results/frame_budget.csv` | `run_0821_1725` | `sec:wydajnoscZasoby` |
| P-03 | The CU completion interrupt is never delivered on this platform: every KDS launch costs ~503 ms. `ROI_CROP_USER_MANAGED=1` is worth 20.6× on frame rate | accepted-hw | GAP — **write this one, it is a self-contained chapter** | `ISR=0x3`, `cu_stat` | `subsec:petlaSterowania` |
| P-04 | RGB costs what the host pays (+2.29 ms), not what conv2d costs (+4.59 ms of AIE time that never appears in the frame) | accepted-hw | `results/frame_budget.csv` | `run_0824_1457` | `subsec:kosztTransferow` |
| P-05 | Memory-tile transpose, `CMUL_SPLIT_ACCUM`, `TAIL_PARALLEL`, blocked `unpack_spectrum`, pipelined `roi_crop` each measured | accepted-hw | `results/perf.csv` | 08-21 runs | `subsec:kosztTransferow` |
| P-06 | DMA is not a bottleneck: 80 µs/tx is per-transaction overhead; the largest transfer achieves 5.76 GB/s | accepted-hw | GAP | — | `sec:przeplywDanych` |
| P-07 | Parallel-for inside `filter_update_quantize` — ~0.96 ms, abandoned: FMA contraction is sensitive to inlining context, 1 ulp of A is disqualifying | refuted | GAP — **the FMA/inlining finding is publishable on its own** | — | `subsec:optymalizacjaOdrzucona` |
| P-08 | `FFT_COL_WS` 8→32 is a 9.57 ms loss; `CMUL_ACCUM_MEMTILE` alone is a 0.36 ms loss | refuted | GAP | `0821_colws32`, `0821_accmem` | `sec:wydajnoscZasoby` |
| P-09 | Phase 4's console knobs are worth 1.1%; the transport (UART→ssh, 3.79 ms) was the whole win | accepted-hw | `evidence/phase4.md` | three `car1` runs | `sec:metodykaBadan` |
| P-10 | Halving the host filter on Hermitian symmetry does not work — the premise is false in fixed point (95.8% of bins differ from their conjugate partner) | refuted | GAP — **has the ideal control: the float golden is Hermitian to 0 LSB** | `make aiesim_plio` | `subsec:fftAie` |
| P-11 | fDSST's PCA compression is not worth it; the real-input DFT was (3.11×, transferred exactly to hardware) | refuted (PCA) / accepted-hw (DFT) | GAP | `run_0821_1109` | `subsec:filtrSkali` |

## R — Tracking results

| id | claim | verdict | evidence | run / source | § |
|---|---|---|---|---|---|
| R-01 | RGB features beat the BT.601 collapse: R 0.2743 → 0.3065, EAO 0.1367 → 0.1474, 12.8% more frames survive | accepted-hw | GAP — **the highest-value missing note; the colour-free control belongs in it** | `0826_1336-hs14full`, `0826_1550-rgb15` | `sec:jakoscSledzenia` |
| R-02 | `MOSSE_ETA=0.05` ships: R +0.0218, EAO +8.5% — but two of three mechanism falsifiers fired | accepted-hw | `evidence/eta05.md` | `0827_1441-eta05` | `sec:jakoscSledzenia` |
| R-03 | `PSR_GATE_MIN=5.0` is worth +0.0134 R at 128x128. **The "conditional on the PSR scale" half is REFUTED ON HARDWARE 2026-09-01**: at 64x64 the PSR scale fell 0.66x and the gate fired 17x more often, but rescaling the threshold to 3.5 (two independent estimators, 3.51 and 3.30) returned **EAO 0.184867 against 0.184936 — a null to four decimals** (R +0.0063, A -0.0028, per-sequence drop-top-3 +0.0017). The mis-scaling was cosmetic because 95.9% of vetoes fire AFTER the run is already lost | accepted-hw, mechanism refuted | `results/arms.csv` | `0827_1642-eta05_g5p0`, `0901_1442-res64_g35` | `sec:jakoscSledzenia` |
| R-04 | Shipping arm: A 0.5100 / R 0.3417 / EAO 0.1629 on 62 sequences, 419 trajectories | accepted-hw | `results/arms.csv` | workspace `gate` | `sec:jakoscSledzenia` |
| R-05 | Accuracy is inside the classical-DCF band (A 0.510, under CSRDCF by 0.009); robustness is below all 41 published trackers | accepted-hw | `results/baselines.csv` | Kristan et al. 2022 Table 12 | `sec:porownanieReferencyjne` |
| R-06 | The loss mechanism is attributed: the tracker walks off target confidently (ACCEPT 82.0% at median PSR 18.83 in the 5 pre-loss frames); the gate is the aftermath, not the cause | accepted-offline | `evidence/robustness_gap.md` | CSVs only | `sec:dyskusjaWynikow` |
| R-07 | On targets that genuinely translate the detector recovers 93% of annotated motion — localisation is not the fault | accepted-offline | `evidence/detector_gain.md` | — | `sec:dyskusjaWynikow` |
| R-08 | The board's usable heap is ~0.9–1.2 GB, not 12 GB; streaming (`VOT_STREAM_RING`) recovered the five RGB sequences that died on `std::bad_alloc`, with identical digests both ways | accepted-hw | `evidence/TODO_board_memory.md` | `0827_1313-streamA/B` | `subsec:zrodlaObrazu` |
| R-09 | Multi-start determinism: two runs of the same job return byte-identical trajectories, and `RESET_MUTANT` proves the test can fail | accepted-hw | `evidence/phase3.md` | — | `subsec:weryfikacja` |
| R-10 | The spatial mask: EAO 0.1629 → 0.1740 (+0.0110) and A is **+0.0179** on the common survived prefix — but **the gain is carried by 3 sequences of 62 (top-3 = 133% of it), the per-sequence median dR is 0.0000 and the bootstrap CI includes zero (P(dR≤0)=0.22). NOT distinguishable from a null**, and the mechanism is refuted: init failures 61 → 60 despite the largest `mask_ebox` edge | **weak-hw** | `evidence/spatial_mask.md` | `0831_1528-mask` vs `0831_1340-base_stat` | `sec:jakoscSledzenia` |

## N — Refuted, and the reason the obvious explanation was wrong

| id | claim tested | verdict | evidence | § |
|---|---|---|---|---|
| N-01 | Quantization causes the poor robustness | **refuted — removing it makes tracking worse** (mean IoU 0.2533 → 0.2350, 0 of 8 sequences improve) | GAP — **write this; it is the thesis's strongest single result** | `sec:dyskusjaWynikow` |
| N-02 | Bolme §3.4 init perturbations would help | refuted — the 16-channel shared denominator is already the cure; Bolme's Fig. 3 is captioned "without regularization" | GAP (`0828_offline-warp8`) | `sec:dyskusjaWynikow` |
| N-03 | POOLING would help | **refuted, and RE-TESTED ON THE RECTIFIED BANK 2026-09-02 where it is a LOSS, not a null.** `blur2` is a null on both linear banks, and MAX pooling is the same null as average (`dec2` 0.3981 / `mpool2` 0.3971 / `pool2` 0.3970, agreeing to 0.001), so the aggregation OPERATOR is irrelevant. Re-opened because the refutation carried a LINEARITY argument that `CONV_RELU=1` voids; the 2x2 screen (blur x rectified/linear, 62 sequences) measured **−0.0242 rectified (trim-5 −0.0269, P(dR<=0)=0.995) and −0.0180 on the linear negative control**, where that argument predicted a null. **Both predictions failed, so the linearity explanation is WITHDRAWN** — the operative reading is RESOLUTION (the map is already 64x64; aggregation and decimation are the same knob). Does NOT speak to Danilowicz's 3x3/stride-1 stem, which pools DOWN TO its working resolution rather than below it | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| R-11 | The MAINLOBE WIDTH `sigma/target` is the axis, with an optimum at 1/16 (DSST's `target/16`) | **confirmed on hardware 2026-09-01, **: `MOSSE_SIGMA=4.0` at 128x128 gives A 0.5133 / R 0.4095 / EAO **0.1931**, dEAO +0.0301 (bar +0.005), drop-top-3 +0.0235, accuracy +0.0299 on the common survived prefix. HOST-ONLY. At matched sigma/target the finer map beats the coarser one (+0.0222 R), the OPPOSITE of the offline proxy **Superseded as best arm by `rgb_l1relu` 2026-09-02 (EAO 0.1960).** The sigma INTERIOR is now closed too: a 22-cell sigma x eta grid (`runs/vot/0902_offline-sigmaeta/`) puts sigma 3, 5, 6 and 8 all below 4, with sigma 8 the worst cell in the search (trim-3 −0.0914) | `evidence/proposed_build_res64.md` sec.21, sec.25 | `sec:jakoscSledzenia`, `sec:dyskusjaWynikow` |
| N-17 | Channel reliability in Stage B3 (CSR-DCF's third contribution) would help | refuted offline 2026-09-01 — **the mechanism HOLDS** (the anti-reliability mutant loses 0.0396 R, so the statistic carries real information) but the gain does not: `gamma=0.5` is +0.0210 pooled and **+0.0023 after drop-top-3** with 39 of 62 sequences exactly tied, `gamma=1` a null, `gamma=2` −0.058. No board time | `evidence/channel_reliability.md` | `sec:dyskusjaWynikow` |
| N-03b | LOWER FEATURE RESOLUTION would help — confirmed on hardware 2026-09-01 but **RE-ATTRIBUTED THE SAME DAY (sec.25): the gain was the MAINLOBE WIDTH the arm carried by accident, not the resolution, which is a small LOSS at matched width.** Original finding, the first proposal in this file to be pre-registered and then pass: 62 sequences, 419 trajectories, A 0.5100→0.5336, R 0.3417→0.3873, EAO 0.1629→**0.1849**. Passes the pre-registered two-part falsifier (dEAO +0.0220 ≥ +0.005; per-sequence dR +0.0323 mean, still +0.0187 after drop-top-3), 37 sequences better / 22 worse. The offline proxy predicted dR +0.1071 and hardware returned +0.0456 — right sign, 43% of the magnitude | confirmed | `evidence/proposed_build_res64.md`, `evidence/pooled_features.md` | `sec:jakoscSledzenia`, `sec:dyskusjaWynikow` |
| N-04 | `TARGET_PADDING=3.0` (predicted +0.088 R) | refuted on hardware — R +0.0077 and **EAO down**; the offline proxy had no scale filter and was single-start | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| N-05 | Padding below 2.0 | refuted — every value worse (R 0.2251 at 1.5) | `evidence/pooled_features.md` | `sec:dyskusjaWynikow` |
| N-06 | A better pretrained feature bank would help | refuted, and since 2026-09-01 for a STRUCTURAL reason: with `CONV_RELU=0` the bank is a LINEAR LIFT, so only its ROW SPACE matters and the online filter absorbs any change of basis. A **one-hot identity lift with no network at all** ties the pretrained bank (common-prefix IoU 0.5833 vs 0.5832 over 4724 frames, 34 of 62 sequences exactly tied); a random ORTHONORMAL bank is the best of the five tested, on accuracy only. Originally: at layer 1 pretraining is worth ~0.011–0.015 R, below the bench's resolution; a random bank of matched row norms ties it | `evidence/feature_bank.md` | `sec:dyskusjaWynikow` |
| N-07 | Participation ratio ranks feature banks | **refuted — PR is maximised by noise** (random Gaussian 10.69 vs the shipping bank's 7.43) | `evidence/feature_bank.md` | `sec:dyskusjaWynikow` |
| N-08 | `HOLD_COAST=1` improves tracking | refuted on the metric of record — wins mean IoU (+0.0296), loses A, R and EAO on the identical 54 pairs | `evidence/hold_policy.md`, `evidence/evidence_ar.md` | `sec:dyskusjaWynikow` |
| N-09 | `nature` is a tracker defect | refuted — its pixels do not move; on 80% of frames *not moving* correlates better (NCC 0.940 vs 0.816) | `evidence/frozen_detector.md` | `sec:dyskusjaWynikow` |
| N-10 | `tiger` is a filter or learning-rate defect | refuted — a plain NCC template search with no filter puts the best match 11 px off the annotation; it is `nature`'s disease, milder | `evidence/tiger.md` | `sec:dyskusjaWynikow` |
| N-11 | Sub-bin quantisation lag compounds | refuted — the detector measures the offset that exists now, not the increment; error bounded at ~half a bin, worst late/early ratio 1.00 | `evidence/subbin_lag.md` | `sec:dyskusjaWynikow` |
| N-12 | Background lock explains the training-target failure | refuted as the explanation (the mechanism is real and measured); `BG_PAN` decorrelates 6.6× and changed the tracker not at all | GAP | `subsec:zrodlaObrazu` |
| N-13 | Re-detection / search-window expansion would help | retired by the protocol — VOT terminates 10 frames after failure, so recovery after a loss scores nothing | `evidence/robustness_proposals.md` | `sec:dalszePrace` |
| N-14 | Relaxing the gate would help | refuted — 88% of vetoes are `NEGATIVE_PEAK`, which `PSR_GATE_MIN` cannot disable, and 95.8% land after the run is already lost | `evidence/robustness_gap.md` | `sec:dyskusjaWynikow` |
| N-15 | Channel pruning is worth doing | retired — moot with ReLU off and `BIAS_SCALE=roi`; the real redundancy is the rank-9 collapse, whose fix is RGB | GAP | `subsec:wyborSieci` |
| N-16 | ReLU on (as in the donor network) | **MECHANISM CONFIRMED, DELIVERABLE MARGINAL, AND IT NOW SHIPS.** Refuted on the SHIPPING 3x3/16 bank (dR relu −0.0332). On a LAYER-1 bank it beats its own linear twin four times offline (+0.0398 vgg 3x3+maxpool, +0.0599 resnet 7x7/2 16ch, +0.0575 at 32ch surviving drop-top-5, +0.0655 at 5x5/1); the ANALYTIC Gabor bank LOSES when rectified, so the property that matters is that the bank is LEARNED, not oriented. **HARDWARE 2026-09-02: A 0.5129 / R 0.4279 / EAO 0.1960, the best on record, 419 trajectories.** Paired per-sequence R +0.0317 mean, **+0.0112 after drop-top-5, P(dR<=0)=0.011**, and accuracy improved too. **But dEAO = +0.0029 against a pre-registered +0.005 bar — NOT ACCEPTED on its own falsifier**; it ships on the paired R result and on the conv-feature requirement (N-15), not on the bar. **The linear twin `ARM=l1lin` has NOT run**, so the gain is attributable to the arm, not yet to the rectifier | `evidence/proposed_build_l1relu.md` sec.10-12, `evidence/layer1_features.md` | `subsec:wyborSieci` |
| N-17 | `SCALE_MAX_STEP=1` | refuted — parks the sim's smooth arm 123 of 200 frames and ends 28.0% wrong | GAP (`make scale_sim`) | `subsec:filtrSkali` |
| N-18 | `SCALE_CONF_MIN` distinguishes a wrong proposal from a big correct one | refuted — it cannot; both match the model poorly for the same reason | GAP | `subsec:filtrSkali` |
| N-19 | `eps_rel` and `MOSSE_SIGMA` need tuning | settled/retired — ε=1e-3 optimal (closed form `R = G·B/(B+ε)`); PSR is monotone in σ and so cannot select it | GAP | `sec:dyskusjaWynikow` |

## M — Methodology claims (chapter 8)

| id | claim | evidence |
|---|---|---|
| M-01 | An offline R can be raised by **degrading** the filter — demonstrated with a deliberately broken init (dR +0.0525, 6.5× the correct arm, while tracking worse by every direct measure) | `evidence/proposed_build_mask.md` |
| M-02 | Accuracy is averaged over tracked frames, so a longer-surviving arm is scored on harder ones — score A on the common survived prefix | `evidence/proposed_build_mask.md` |
| M-03 | Mean IoU and the toolkit's AR order two arms **oppositely on the same runs** | `evidence/evidence_ar.md` |
| M-04 | A byte-identical trajectory is not a bit-identical run — found by a negative control, fixed with a per-frame FNV-1a digest | `evidence/phase3.md` |
| M-05 | Never size a change from one member of an interleaved async/wait group — the single most repeated measurement error in this design (4 occurrences) | GAP |
| M-06 | Verify a feature flag on a build that can exercise it; `strings` on the ELF can report a false absence | `evidence/phase4.md` |
| M-07 | A multi-start CSV collides on frame index — keyed by frame alone it under-reported rails 66× next to a confident wrong verdict | GAP |
| M-08 | Write predictions and falsifiers down before the run; two of three fired on the arm that shipped anyway | `evidence/eta05.md` |
| M-13 | **Bound the prize before attributing the defect.** The scale filter was fully root-caused (frozen ~90% of frames, detector gain −0.003 against the position detector's 0.93, a self-confirming loop) before anyone asked what repairing it was worth. An ORACLE over the board's own trajectories — tracker's centre, ground-truth size, VOT's rule re-applied — puts a PERFECT scale filter at **+0.0023 R** on the shipping arm and **−0.0089** on `sigma4`, lifting mean IoU +0.054 and converting none of it into survival. Attribution says WHY; an oracle says WHETHER to care, and it is the cheaper of the two | `../engineering/scale_filter.md`, `scripts/scale_oracle_bound.py` |
| M-12 | A detector that reports "no change" is not necessarily blind — separate the two with the NULL RATE, not the gain. The scale filter's α = −0.003 against the warranted correction (position: 0.93) reads as blindness; `P(idx==0)` = **88.4% against 3.0% for a noise argmax over 33 bins** shows it is strongly LOCKED to the current size, and `scale_conf` is flat to ±0.04 from correct to 36% wrong. The same estimator reaches α 0.93 in `scale_loop_sim` once the target outruns the lock, so the loop is SELF-CONFIRMING and the code is sound | `../engineering/scale_filter.md` |
| M-11 | An aggregate rate can invert when split by the state it depends on: the mask's `NEGATIVE_PEAK` drop (15.4% → 10.2%) is a RISE on the frames that matter (3.07% → 3.87% on target) | `evidence/spatial_mask.md` |
| M-09 | Benchmark a host-side change on the host — the *ordering* did not transfer, not just the magnitude, because the working set crossed a cache boundary | GAP |
| M-10 | A caveat that is not priced is a hope | `evidence/pooled_features.md` |

## O — Open, at the time of writing

| id | item | state | evidence |
|---|---|---|---|
| O-01 | Spatial mask on the filter (`FILTER_MASK=1`) | **SWEPT 2026-08-31 — see R-10; EAO rose but the arm is NOT separable from a null** (EAO +0.0110). No longer open; the id is kept because `@thesis` tags bind to it. The predicted `PSR_GATE_MIN` re-tune is NOT supported by the hardware data (the gate's bite did not move: `LOW_PSR` 0.10% → 0.09%); what remains open is whether the mask helps AT ALL on 62 sequences (bootstrap CI includes zero) — and the CSR-DCF mechanism is refuted here: it is a PROJECTION not a constraint, `A`/`B` are untouched so it cannot compound, and the unmasked filter self-concentrates (e_box 0.60 → 0.86 unaided) | `evidence/spatial_mask.md`, `evidence/proposed_build_mask.md` |
| O-02 | Channel reliability in Stage B3 | untested; both halves available by Parseval inside the existing loop | `evidence/robustness_proposals.md` |
| O-03 | Two-filter temporal ensemble (short/long eta) | untested, host-only | `evidence/robustness_proposals.md` |
| O-04 | Feature-bank **geometry** — not the weights | **CONFIRMED AND SHIPPED 2026-09-02.** It was the only feature axis left (the weights are a linear lift; one-hot ties the network) and it is the one that paid: 7x7 stride 2, 32 channels, resnet18-PCA bank, ReLU on, 64x64 map → EAO 0.1960. Geometry, not weights, is where feature work belongs | `evidence/feature_bank.md`, `evidence/proposed_build_l1relu.md` |
| O-05 | Software-pipeline the channel loop (~8.7 ms exposed) | untested | `CLAUDE.md` §Next — PERFORMANCE |
