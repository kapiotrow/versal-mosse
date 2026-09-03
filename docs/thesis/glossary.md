# Glossary — code English → thesis Polish

**Status:** current · **Updated:** — · **Scope:** code English -> the Polish the thesis has already committed to

**All documentation in this repository is English.** The thesis is Polish. This file is the
bridge, and it exists because `teoria.tex` and `przeglad.tex` are already written: the
terminology is *fixed*, and `projekt.tex`/`ewaluacja.tex` must reuse it rather than invent a
second translation for the same thing. Two names for one concept, four chapters apart, reads as
two authors.

**How to use it.** Writing a section of `cha:projekt` or `cha:ewaluacja`: look up each English
term you are about to translate. If it is in table A, use that Polish phrase verbatim — it is
already in print. If it is in table B, the Polish is a *proposal* and you should either accept
it or fix this file, so the next section agrees with this one.

**Direction matters.** Nothing here licenses Polish in the code. Identifiers, comments,
`@thesis` tags, evidence notes and CSVs stay English.

---

## A. Fixed by the thesis — use verbatim

Every row is a term already used in `teoria.tex` or `przeglad.tex`, with the section that
introduces it. Where the thesis glosses the English itself (`(ang. …)`), that gloss is the
canonical one; do not re-gloss the same term in a later chapter.

| English (code / literature) | Polish (thesis) | introduced in |
|---|---|---|
| visual object tracking | śledzenie obiektów | `cha:wstep` |
| model-free tracking | śledzenie bezmodelowe | `sec:sledzenieObiektow` |
| bounding box | prostokąt otaczający | `sec:sledzenieObiektow` |
| drift | dryf | `sec:sledzenieObiektow` |
| target loss (the VOT rule) | utrata celu | `sec:sledzenieObiektow` |
| ROI, region of interest | obszar zainteresowania | `subsec:roi` |
| patch (the cropped ROI) | wycinek | `subsec:roi` |
| padding factor (`TARGET_PADDING`) | współczynnik dopełnienia $p$ | `subsec:roi` |
| search margin | margines | `subsec:roi` |
| response map | mapa odpowiedzi | `subsec:mosse` |
| bin (one spectral coefficient) | prążek widma | `subsec:mosse` |
| numerator $A^l$ / denominator $B$ | licznik $A^l$ / mianownik $B$ | `subsec:mosse` |
| regularization factor $\lambda$ | współczynnik regularyzacji | `subsec:mosse` |
| learning rate $\eta$ | współczynnik uczenia | `subsec:aktualizacja` |
| model update | aktualizacja modelu | `subsec:aktualizacja` |
| boundary effects | efekty brzegowe | `subsec:efektyBrzegowe` |
| Hann window | okno Hanna | `subsec:efektyBrzegowe` |
| smoothing window | okno wygładzające | `subsec:efektyBrzegowe` |
| spatial regularization | regularyzacja przestrzenna | `subsec:efektyBrzegowe` |
| PSR, peak to sidelobe ratio | stosunek wartości szczytowej do listków bocznych | `subsec:psr` |
| sidelobes | listki boczne | `subsec:psr` |
| update hold / skipping the update | wstrzymanie aktualizacji | `subsec:psr` |
| scale estimation | estymacja skali | `subsec:skala` |
| PS, processing system | podsystem procesorowy | `subsec:architekturaAcap` |
| PL, programmable logic | logika programowalna | `subsec:architekturaAcap` |
| AIE array | macierz AI~Engine | `subsec:architekturaAcap` |
| compute tile | kafelek obliczeniowy | `subsec:architekturaAcap` |
| memory tile | kafelek pamięciowy | `subsec:architekturaAcap` |
| transpose (in a memory tile) | transpozycja | `subsec:architekturaAcap` |
| stream processing | przetwarzanie strumieniowe | `subsec:architekturaAcap` |
| HOG | histogram zorientowanych gradientów | `subsec:cechyReczne` |
| feature map | mapa cech | `subsec:cechyKonwolucyjne` |
| feature-map channel | kanał mapy cech | `subsec:wymiarowoscBanku` |
| filter bank | bank filtrów | `subsec:wymiarowoscBanku` |
| kernel (conv) | jądro | `subsec:warstwaKonwolucyjna` |
| stride | krok przesunięcia jądra | `subsec:wagiWyuczone` |
| batch normalization (folded) | normalizacja wsadowa | `subsec:warstwaKonwolucyjna` |
| effective dimensionality / participation ratio | wymiarowość efektywna / współczynnik uczestnictwa | `subsec:wymiarowoscBanku` |
| post-training quantization | kwantyzacja potreningowa | `subsec:kwantyzacja` (teoria) |
| fixed-point arithmetic | arytmetyka stałoprzecinkowa | `subsec:kwantyzacja` (teoria) |
| saturation (exceeding the range) | nasycenie | `subsec:kwantyzacja` (teoria) |
| IoU / overlap | pokrycie | `subsec:iou` |
| mean IoU | średnie pokrycie | `subsec:iou` |
| multi-start / anchor-based protocol | protokół wielokrotnego startu | `subsec:vot` |
| anchors | punkty początkowe | `subsec:vot` |
| accuracy $A$ | dokładność $A$ | `subsec:vot` |
| robustness $R$ | odporność $R$ | `subsec:vot` |
| EAO | oczekiwane średnie pokrycie | `subsec:vot` |
| FPS | liczba klatek na sekundę | `subsec:metrykiSystemowe` |
| resource utilisation | zajętość zasobów | `subsec:metrykiSystemowe` |

## B. Implementation vocabulary — PROPOSED, not yet in the thesis

`projekt.tex` and `ewaluacja.tex` are skeletons, so none of this is settled. Accept or amend a
row here **before** using it, and amend it here rather than only in the `.tex`, so the next
section agrees.

| English (code) | Polish (proposed) | note |
|---|---|---|
| shift budget (`FFT_SHIFT`, `H_SHIFT`, …) | budżet przesunięć bitowych | the project's own term; `subsec:arytmetyka` |
| bit shift / downshift | przesunięcie bitowe | |
| rails, railing (a saturated sample) | nasycenie / próbka nasycona | reuse `nasycenie`, already fixed in table A |
| headroom | zapas zakresu | |
| Q1.15 format | format Q1.15 | keep the notation |
| gate (the PSR decision) | bramkowanie odpowiedzi | the *mechanism* is `wstrzymanie aktualizacji`, already fixed |
| veto (a gate's reason) | odrzucenie / powód odrzucenia | `NEGATIVE_PEAK` etc. stay in English as identifiers |
| hold (position frozen) | zamrożenie położenia | |
| coasting (`HOLD_COAST`) | ekstrapolacja ruchu | off by default; see claim N-08 |
| peak detection / argmax | wyszukiwanie maksimum | |
| training target $g$ | odpowiedź wzorcowa | phrase already used in `subsec:mosse` |
| spatial mask / reliability map | maska przestrzenna filtra | claim O-01 |
| channel reliability | niezawodność kanałowa | `przeglad.tex` already uses it for CSR-DCF |
| line buffer | bufor wierszowy | conv2d |
| ping-pong buffer | bufor naprzemienny | |
| GMIO / PLIO port | port GMIO / PLIO | keep the AMD names |
| user-managed CU | jednostka obliczeniowa sterowana z aplikacji | claim P-03 |
| host application | aplikacja hosta | matches `sec:warstwaSterujaca` |
| frame buffer | bufor ramki | |
| arm (of an A/B experiment) | wariant | `results/arms.csv` |
| sweep | seria pomiarowa | `scripts/vot_sweep.sh` |
| run (one anchored trajectory) | przebieg | matches `subsec:vot`'s use |
| bit-identical | identyczny co do bitu | the acceptance criterion; claim B-07 |
| run-state digest | skrót stanu przebiegu | claim M-04 |
| flagstamp | *flagstamp* (leave in English, gloss once) | a file name; see `reproduce.md` |
| golden model / reference model | model odniesienia | `scripts/roi_crop_ref.py` |
| mutation testing | testowanie mutacyjne | claim R-08 |
| offline bench | stanowisko programowe / bench programowy | pick one, then keep it |

## C. Code identifier → concept

For reading `code_map.md` next to the thesis. These are names, never translated.

| identifier | is | Polish phrase to use in prose |
|---|---|---|
| `roi_crop` | the PL kernel doing Stage A | moduł wycinania obszaru zainteresowania |
| `camera_capture` | the frame-buffer stub | moduł akwizycji obrazu (zaślepka) |
| `conv2d_kernel` | 3×3 INT8 conv + Hann | jądro warstwy konwolucyjnej |
| `cmul_accum_kernel` | spectral multiply + channel sum | jądro mnożenia widm i akumulacji |
| `fft2d` / `ifft2d` | the 2-D transform graphs | grafy transformacji Fouriera |
| `mosse_filter` | filter init/update/PSR/scale on the APU | moduł filtra korelacyjnego |
| `mosse_tracker` | the host application | aplikacja hosta |
| `vot_source` | manifest/blob/trajectory bookkeeping | źródło sekwencji VOT |
| Stage A | log, zero-mean, unit-L2, int8, in `roi_crop` | wstępne przetwarzanie wycinka |
| Stage B1 / B2 / B3 | mean removal / 9-bin correction / energy normalisation | usunięcie średniej / korekcja widmowa / normalizacja energii |
| `H_SHIFT` | filter-product shift | przesunięcie iloczynu filtra |
| `PSR_GATE_MIN` | the gate threshold | próg bramkowania |
| `MOSSE_ETA` | $\eta$ | współczynnik uczenia |
| `TARGET_PADDING` | $p$ | współczynnik dopełnienia |
| `track.csv` | one row per frame | zapis przebiegu |

## D. Do not conflate

Four pairs this project has already confused at least once. Each needs distinct Polish, not a
shared word.

1. **Two different statistics are both called PSR.** The aiesim `snr_ratio_pct` is
   $|peak| / \max|sidelobe|$; Bolme's is $(g_{max} - \mu_{sl}) / \sigma_{sl}$. Same exclusion
   window, different statistic, and neither's thresholds transfer. If both appear, name them
   separately — `subsec:psr` defines Bolme's, which is the meaningful one.
2. **`pokrycie` carries three loads**: per-frame IoU (`subsec:iou`), the accuracy $A$ averaged
   over tracked frames, and the overlap averaged into EAO *including zeros after a loss*. They
   are not interchangeable and claim M-03 exists because two of them ordered the same arms
   oppositely. Qualify every use.
3. **`kanał`** is a feature-map channel (16 of them) *and* a colour plane (3 of them). The RGB
   sections talk about both in one sentence. `kanał mapy cech` vs `kanał barwny` — the thesis
   already uses both forms; keep them.
4. **`utrata celu`** is the VOT rule — overlap ≤ 0.1 for ten consecutive frames — not the
   everyday sense of losing the target. A run can be visually lost long before, and hold the
   box near the target long after. `subsec:vot` defines it; use it only in that sense.
