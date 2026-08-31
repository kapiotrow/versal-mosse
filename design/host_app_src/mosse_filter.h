/*
 * mosse_filter.h
 * MOSSE correlation filter: initialisation, online update, and Q1.15 export.
 *
 * Bolme et al., "Visual Object Tracking using Adaptive Correlation Filters"
 * (CVPR 2010), eq. 10-12, generalised to N_CHANNELS feature channels with a
 * SHARED denominator (the Danelljan/DSST form):
 *
 *     B     = eta * SUM_ch F_ch (*) conj(F_ch)  + (1-eta) * B_prev   (real, one map)
 *     A_ch  = eta * conj(G) (*) F_ch            + (1-eta) * A_ch_prev
 *     H_ch  = A_ch / (B + eps)
 *
 * One denominator instead of sixteen: 16k reciprocals per frame instead of 262k,
 * and it is the better-conditioned formulation — a channel whose energy collapses
 * cannot blow up its own filter.
 *
 * -------------------------------------------------------------------------
 * NO FFT IS REQUIRED HERE, and that is not an accident:
 *
 *   - F_ch arrives already transformed, drained from the AIE column FFT via
 *     gmio_fft_col_out. The host never transforms anything.
 *   - G, the target spectrum, has a closed form. The DFT of a circularly-wrapped
 *     Gaussian is another Gaussian (theta-function identity), so
 *     gaussian_target_spectrum() fills it directly from exp() calls.
 *
 * The `filter_update_kissfft` stub this replaces was named after a dependency the
 * design turned out not to need.
 *
 * -------------------------------------------------------------------------
 * CONJUGATION — the one thing that is easy to get wrong and silent when wrong.
 *
 * Bolme writes the filter as H* = G (*) conj(F) / (F (*) conj(F)), because the
 * correlation he forms is F (*) H*. cmul_accum_kernel applies the conjugation
 * itself, so what this module STORES is H, not H*:
 *
 *     F (*) conj(H_stored) = G    =>    H_stored = conj(G) (*) F / (B + eps)
 *
 * Storing Bolme's expression verbatim yields F (*) conj(H) = conj(G)*F/conj(F),
 * whose phase is noise; the response then peaks at an arbitrary bin. The
 * distinction is INVISIBLE whenever the target is centred, because a centred real
 * Gaussian has conj(G) = G — which is why the aiesim scenario s7 deliberately
 * places its target off-centre, and why this header says so twice.
 *
 * -------------------------------------------------------------------------
 * This file and mosse_filter.cpp include NO XRT and NO ADF header, on purpose:
 * `make test_host` compiles them with the system g++ and checks them against a
 * NumPy golden in seconds. The alternative is a ~90 min hw_emu frame.
 *
 * @thesis subsec:aktualizacjaFiltra | A-03,B-06 | The filter contract: Bolme eq. 10-12 with a
 *   SHARED denominator, why no FFT is needed on the host, and the conjugation convention that
 *   is silent when wrong.
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

// The shift cmul_accum applies to the F*H product. Fed from the Makefile's single
// H_SHIFT variable; the default matches the kernel's so a standalone native build
// still works.
//
// NOTE this is NOT the filter's quantization ceiling. H is always normalized to
// the full int16 range (32767) regardless of H_SHIFT — see filter_quantize_q15().
// H_SHIFT only decides where the product lands in the cint16 accumulator.
#ifndef CMUL_H_SHIFT
#  define CMUL_H_SHIFT 10
#endif

// Bolme §3.5's occlusion / tracking-failure threshold, on the PSR statistic
// (g_max - mu_sl)/sigma_sl. Fed from the Makefile's PSR_GATE_MIN; the default
// matches the paper so a standalone native build behaves the same.
//
// SEMANTICS: <= 0 DISABLES the threshold test — structural failures (zero
// response, flat sidelobe, negative peak) are still reported and still veto, but
// a merely weak peak no longer does. That is the report-only / pre-gating
// behaviour, and it makes the A/B lever one make variable instead of an #if.
//
// NOTE this threshold applies to Bolme's PSR ONLY, never to the |peak|/max|side|
// ratio that gen_aiesim_vectors.py calls PSR. They are different statistics and
// differ by several times — see the note on PsrResult below.
#ifndef PSR_GATE_MIN
#  define PSR_GATE_MIN 7.0
#endif

namespace mosse {

using cfloat = std::complex<float>;

// Exponential learning rate. Bolme §3.3 uses 0.125 and reports it works well
// across the whole test set. DSST §6.1 uses 0.025 for both its filters.
//
// Given an #ifndef escape for the same reason PSR_GATE_MIN has one: it is a
// calibration constant, and this project's record is that calibration constants
// which can only be changed by editing a header do not get swept.
#ifndef MOSSE_ETA
#  define MOSSE_ETA 0.125
#endif
constexpr float DEFAULT_ETA = (float)(MOSSE_ETA);

// Target Gaussian width, in PATCH pixels. Bolme §3.1 uses sigma = 2.0 on a 64x64
// window sized to the target.
//
// WHY THIS IS STILL 2.0 AND NOT THE DSST RULE — measured 2026-08-16.
// DSST §6.1 anchors it: "The standard deviation of the desired correlation output
// g is set to 1/16 of the target size in the translation dimensions." At padding
// 2 the target occupies 64 of the 128 patch pixels, so the rule gives sigma = 4.
// The offline sweep does NOT support making that switch, and the reason is that
// the available metric cannot arbitrate:
//
//   sigma    0.75   1.00   1.50   2.00   2.50   3.20   4.00   5.33
//   PSR(B)   80.3   71.8   55.2   45.7   41.2   37.5   30.5   19.3
//
// Bolme PSR is MONOTONE DECREASING in sigma all the way to sub-pixel, so it does
// not select sigma — it rewards a sharp peak, and a delta target would maximise
// it. What sigma actually buys is robustness to appearance change, which a
// single-frame translation-only holdout does not exercise. (The obvious
// mechanistic explanation — Stage B2 nulling the 9 low bins where a wide Gaussian
// keeps its energy — was tested with B2 off and REFUTED: the drop survives.)
//
// So: default unchanged, and the rule is reachable in one make variable via
// SIGMA_FROM_TARGET. Decide it when real video, or a holdout with scale and
// appearance change, can measure what sigma is actually for.
#ifndef MOSSE_SIGMA
#  define MOSSE_SIGMA 2.0
#endif
constexpr float DEFAULT_SIGMA = (float)(MOSSE_SIGMA);

// DSST §6.1's divisor: sigma = target_size_in_patch_px / SIGMA_FACTOR.
#ifndef SIGMA_FACTOR
#  define SIGMA_FACTOR 16.0
#endif
constexpr float DEFAULT_SIGMA_FACTOR = (float)(SIGMA_FACTOR);

// 0 (default) = use DEFAULT_SIGMA literally. 1 = derive it from the target box
// per DSST §6.1. This is the A/B lever; see the note above for why it is off.
#ifndef SIGMA_FROM_TARGET
#  define SIGMA_FROM_TARGET 0
#endif

// ROI padding factor: roi = target * TARGET_PADDING. Both papers use a window
// LARGER than the object so the filter learns target-vs-background (Bolme §3.1,
// Danelljan §3.1); DSST §6.1 sets it to 2, fDSST §6.1 to 3.
//
// The offline sweep (padding 1.5 / 2.0 / 2.5 / 3.0, target 64, held out by
// re-cropping a moved target) settles this at >= 2: 1.5 is worst on both metrics
// and shows a real 0.75 px localisation error, while 2.5 and 3.0 edge ahead but
// trigger aliasing (roi_crop's bilinear has no prefilter, so beyond ~2x
// decimation source rows are skipped outright) and 3.0 clips 3.6% of samples.
//
// 2.0 is the default because at target 64 it gives roi = 128, i.e. EXACTLY the
// geometry shipped today — so adopting the box costs no resample change and any
// later padding move is a clean single-variable step.
#ifndef TARGET_PADDING
#  define TARGET_PADDING 2.0
#endif
constexpr float DEFAULT_PADDING = (float)(TARGET_PADDING);
// Regularization, as a FRACTION of mean(B) rather than an absolute value: B's
// magnitude depends on the shift budget and the feature scale, so an absolute
// epsilon would silently become either a no-op or a dominant term after any
// recalibration.
// @thesis sec:dyskusjaWynikow | N-19 | eps_rel is SETTLED at 1e-3 and is not a
//   tuning knob: the response has a closed form R = G*B/(B+eps), and the sweep peaks here.
//   Bolme Fig. 4's flat curve does not transfer -- his eps is absolute, this one relative.
constexpr float DEFAULT_EPS_REL = 1e-3f;

// Half-width of the excluded window around the peak, Bolme §3.5's 11x11.
constexpr int   PSR_EXCL_HALF   = 5;
// Build-configured gate threshold. Written as a cast rather than a suffixed
// literal so `make ... PSR_GATE_MIN=20` and `=7.5` both compile.
constexpr float DEFAULT_PSR_MIN = (float)(PSR_GATE_MIN);

// -----------------------------------------------------------------------
// Peak-to-sidelobe ratio — Bolme §3.5
// -----------------------------------------------------------------------
// "the correlation output g is split into the peak which is the maximum value and
//  the sidelobe which is the rest of the pixels excluding an 11x11 window around
//  the peak. The PSR is then defined as (g_max - mu_sl) / sigma_sl."
//
// Bolme reports 20.0-60.0 under normal tracking and ~7.0 as the occlusion /
// failure indicator. This is the metric `err=0 px` cannot see: aiesim s7
// localised EXACTLY while its PSR collapsed to 5.2 against a golden 38.
//
// Two things this implementation is careful about:
//
//   * The exclusion window uses CIRCULAR distance. The response map wraps, so a
//     peak near an edge has its mainlobe split across the boundary and a linear
//     window would leave half the mainlobe in the sidelobe statistics.
//   * Without the exclusion the check asserts nothing at all. On a smooth
//     sigma=2 Gaussian the peak's immediate neighbour sits at exp(-1/8) = 0.88 of
//     the peak, so a "largest non-peak element" ratio is 1.13 on any blurry blob.
//
// TWO statistics, because the project has two and they are NOT the same number:
//
//   psr   = (g_peak - mu_sl) / sigma_sl        <- Bolme §3.5, literally.
//           Comparable to his published 20.0-60.0 range and ~7.0 failure mark.
//   ratio = |g_peak| / max|sidelobe|           <- what gen_aiesim_vectors.py calls
//           PSR and asserts as `snr_ratio_pct` (s7's "19.6x", golden "38x").
//
// Both use the same 11x11 circular exclusion, but max|sidelobe| is roughly
// mu + 3..4 sigma for a noise-like sidelobe, so the two differ by several times.
// Do not compare one against the other's thresholds.
//
// `resp` is cint16 with {re,im} interleaved: sample i's real part is resp[2*i].
// The detector scans the REAL part only — Stage B1 makes the response bipolar,
// so a legitimately negative peak is possible and `use_abs` decides whether it
// is found. Reading resp[i] instead of resp[2*i] is a real bug this guards.
struct PsrResult {
    int    dr, dc;      // displacement of the located peak (wrapped to +/- N/2)
    long   peak;        // SIGNED response value there
    double mean, sdev;  // sidelobe statistics
    double psr;         // Bolme §3.5
    double side_max;    // max |sidelobe|
    double ratio;       // |peak| / max|sidelobe|  (the aiesim harness's metric)
    long   n_side;      // sidelobe sample count (should be rows*cols - 121)
};

// g_peak is the SIGNED response value, per Bolme. `use_abs` selects only how the
// peak is LOCATED: true finds the largest MAGNITUDE (the peak the tracker acts
// on), false is the paper-literal signed maximum. Computing both is how a sign
// problem becomes visible.
PsrResult compute_psr(const int16_t *resp, int rows, int cols, bool use_abs);

// Why a frame was or was not allowed to train the filter.
//
// An enum rather than a bool because "HOLD" alone is unactionable: a flat
// sidelobe means the pipeline produced nothing, a negative peak means the filter
// has the wrong sign, and a low PSR means occlusion. Those demand different
// responses from whoever reads the log, and on this project a ~26 min frame must
// never report a verdict without its reason.
enum class GateReason {
    Accept,        // psr >= threshold, peak positive, sidelobe non-degenerate
    Disabled,      // psr_min <= 0: threshold test off, accepted by policy
    ZeroResponse,  // peak == 0: the map is identically zero, no information
    EmptySidelobe, // n_side == 0: geometry smaller than the exclusion window
    FlatSidelobe,  // sdev == 0: constant map; PSR undefined, not infinite
    NegativePeak,  // largest |real| is negative => anti-correlation
    LowPsr,        // Bolme's occlusion / failure indicator
};

struct GateDecision {
    // Default FALSE on purpose: an unset decision must never train the filter.
    bool       accept    = false;
    GateReason reason    = GateReason::ZeroResponse;
    double     psr       = 0.0;
    float      threshold = 0.0f;
};

// Bolme §3.5 update gating. `p` must come from compute_psr(..., use_abs=true),
// i.e. the peak the tracker actually acted on.
//
// The threshold is a PARAMETER rather than being read from DEFAULT_PSR_MIN
// internally, so the unit tests can sweep it without depending on the build's -D.
GateDecision psr_gate(const PsrResult &p, float psr_min);

const char *gate_reason_tag(GateReason r);   // "ACCEPT" / "LOW_PSR" / ...
const char *gate_reason_why(GateReason r);   // one sentence, for the log

// -----------------------------------------------------------------------
// Target box and ROI geometry
// -----------------------------------------------------------------------
// Until 2026-08-16 the tracker's entire state was pos_row/pos_col, two ints. That
// had three consequences: sigma had no defined relation to the object, the ROI
// was pinned 1:1 to the patch so the filter saw no background context, and there
// was no box to score an IoU against — i.e. no way to report the metric both
// papers report.
//
// These live in mosse_filter.h, not in mosse_tracker.cpp, so they are under
// `make test_host`. That matters specifically for the two conversions below.
struct TargetBox {
    double row = 0.0, col = 0.0;   // CENTRE, frame pixels
    double h   = 0.0, w   = 0.0;   // size, frame pixels
};

// The ROI actually handed to roi_crop, plus the patch it is resampled into.
struct RoiGeometry {
    int roi_row = 0, roi_col = 0;      // top-left, frame px (may be NEGATIVE)
    int roi_h   = 0, roi_w   = 0;      // extent, frame px
    int patch_rows = 0, patch_cols = 0;
};

// roi = box * padding, centred on the box. roi_row/roi_col go negative for a
// target near an edge, which roi_crop handles by border replication — and which
// is why the kernel must not left-shift them (UB before C++20).
RoiGeometry roi_for(const TargetBox &box, float padding,
                    int patch_rows, int patch_cols);

// -----------------------------------------------------------------------
// THE CONVERSION THAT IS CURRENTLY INVISIBLE
// -----------------------------------------------------------------------
// The correlation peak is located in PATCH bins. The tracked position lives in
// FRAME pixels. While roi_h == patch_rows those are the same number, so the
// tracker gets away with `pos_row += dr` and with asserting `dr == IMPULSE_DR`.
// The moment padding makes roi_h != patch_rows, both are wrong by the resample
// ratio, and the failure mode is a tracker that localises confidently and drifts
// — the class of bug `err=0 px` has passed through four times in this project.
//
// One bin is roi_h/patch_rows frame pixels. That ratio is also the localisation
// QUANTUM: at 2.0 the tracker cannot resolve better than 2 frame px however good
// its PSR looks, which is a cost of padding no PSR number reveals.
double patch_dr_to_frame(int dr, const RoiGeometry &g);
double patch_dc_to_frame(int dc, const RoiGeometry &g);
// The inverse, for turning an expected frame-pixel displacement into the patch
// bin a correct pipeline must report.
int    frame_dr_to_patch(double dr, const RoiGeometry &g);
int    frame_dc_to_patch(double dc, const RoiGeometry &g);

// Target size measured in PATCH pixels — the units sigma is expressed in.
double target_h_in_patch(const TargetBox &box, const RoiGeometry &g);
double target_w_in_patch(const TargetBox &box, const RoiGeometry &g);

// sigma per DSST §6.1, or DEFAULT_SIGMA, depending on SIGMA_FROM_TARGET.
// Returned per axis because the paper says "in the translation dimensions" and a
// non-square target has a different width in each.
void sigma_for(const TargetBox &box, const RoiGeometry &g,
               float *sigma_r, float *sigma_c);

// Intersection-over-union of two boxes. This is what makes the tracker scoreable
// — both papers report overlap precision (OTB) and neither can be reproduced
// without it.
double box_iou(const TargetBox &a, const TargetBox &b);


struct FilterState {
    int    rows      = 0;
    int    cols      = 0;
    int    channels  = 0;
    bool   initialized = false;

    std::vector<cfloat> A;   // channels * rows * cols — per-channel numerator
    std::vector<float>  B;   // rows * cols           — shared denominator

    void resize(int rows_, int cols_, int channels_);
    size_t elems() const { return (size_t)rows * cols; }
};

// Fill G with the spectrum of a circularly-wrapped Gaussian of width `sigma`
// centred at spatial offset (dr, dc), in the ROW-MAJOR bin layout
// G[u * cols + v], u = row frequency.
//
// Closed form, no FFT: the periodic summation of a Gaussian has a Gaussian DFT.
// At sigma = 2 and N = 128 the theta-function truncation error is far below one
// cint16 LSB.
//
// TWO RUNTIME USES, and conflating them was the tracker's primary defect until
// 2026-08-20:
//   DETECTION scale  dr = dc = 0. A target displaced by (dr,dc) must produce a
//                    peak at (dr,dc), so the reference target carries no offset.
//   TRAINING target  dr,dc = THIS frame's measured displacement. filter_update()
//                    is handed the patch cropped at the PRE-update position, so
//                    the object in it sits at (dr,dc); training that against a
//                    centred G teaches a zero-shift response and the error
//                    compounds at the learning rate. See the filter_update()
//                    call site in mosse_tracker.cpp, and
//                    run_training_target_tests() in test_mosse_filter.cpp.
void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma, int dr, int dc);

// Anisotropic form. DSST §6.1 anchors sigma to "the target size in the
// translation dimensionS" — plural — and a non-square target has a different
// extent per axis. The closed form already separates them (the row and column
// exponents are computed independently), so carrying two sigmas is nearly free
// and collapsing to sqrt(area) would throw away real information.
//
// The scalar overload above forwards here with sigma_r == sigma_c, so every
// existing caller and the NumPy golden are unaffected.
void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma_r, float sigma_c, int dr, int dc);

// First frame: A = conj(G) (*) F_ch, B = SUM |F_ch|^2. Equivalent to
// filter_update() with eta = 1 against a zeroed state.
//
// This is single-patch initialisation (Bolme's N = 1). Bolme Fig. 3 puts N = 1 at
// PSR ~4 versus ~19 at N = 8 affine perturbations, but MOSSE degrades gracefully
// where ASEF and UMACE collapse, and the online update recovers within a few
// frames. Perturbations are a later increment.
//
// F_all is channels * rows * cols, channel-major.
void filter_init(FilterState &st, const cfloat *F_all,
                 const cfloat *G, int channels, int rows, int cols);

// Subsequent frames: exponential smoothing, Bolme eq. 10-12.
void filter_update(FilterState &st, const cfloat *F_all,
                   const cfloat *G, float eta);

// -----------------------------------------------------------------------
// FILTER_MASK — spatial reliability, as a one-shot projection h <- m (.) h
// -----------------------------------------------------------------------
// HOST-ONLY. The mask lands on H before pack_filter(), so the AIE sees an
// already-masked filter and detection and training stay consistent with no
// graph change: an scp, not a card swap. aie.flagstamp must come back unchanged.
//
// AT FILTER_MASK=0 THE ELF IS *NOT* BYTE-IDENTICAL, AND EXPECTING IT TO BE IS A
// MISREADING WORTH WRITING DOWN. Every call site is #if'd out, but the two
// functions below have EXTERNAL LINKAGE and are emitted whether or not anything
// calls them, so the image grows and every later address relocates. `cmp` on the
// ELF — this project's usual inertness check, per PROGRESS_EVERY — therefore
// reports a difference that means nothing.
//
// The check that does mean something, and it was run (2026-08-29): disassemble
// both builds, group by symbol, and compare each function's instruction stream
// with addresses and immediates normalised. Result: 245 functions in both,
// **0 differing**; the FILTER_MASK=0 build adds exactly filter_mask_project,
// filter_box_energy_fraction and one PLT entry (__cxa_thread_atexit, for the
// thread_local scratch) and changes NO existing code path. Control first: the
// same source built twice IS byte-identical, so the instrument is sound.
#ifndef FILTER_MASK
#  define FILTER_MASK 0
#endif

// CSR-DCF's contribution and the literature's highest-priced item on the
// robustness list. At TARGET_PADDING=2 the target is 27% of the ROI area and
// nothing masks the rest, so the filter trains on background every accepted
// frame — measured, a centred 64x64 box (exactly the target box at padding 2)
// holds only 51.6% (car1) / 54.9% (tiger) of SUM|h|^2. That is
// filter_box_energy_fraction() below, and it is this build's mechanism check.
//
// EVIDENCE: docs/thesis/evidence/proposed_build_mask.md, claim O-01, and the rows
// `off_rgb` / `off_mask0_boardform` in docs/thesis/results/arms_offline.csv -- which is
// where these figures live and where they get corrected if the arm is re-swept.
// 62 sequences, shipping eta/gate, vot_ar_offline: dR +0.0601 in the BOARD form of the
// window (3.0x the instrument's measured resolution), surviving a symmetric trim at
// +0.0409, mean IoU 0.1792 -> 0.2016, 20.7% more frames tracked.
// NOTE these are OFFLINE single-start numbers and are not comparable to the hardware
// arms in arms.csv -- see claims.md rule 2.
//
// THE WINDOW IS THE PERIODIC HANN, and it must be the one centred at n/2 —
// i.e. hanning_<N>.h's sin^2(pi i / n), the SAME window conv2d applies to the
// patch. The offline bench centres its axis at (n-1)/2 instead; half a sample,
// max|dm| = 0.0123, and worth mean IoU 0.1715 vs 0.2813 on `tiger`. The board
// form is the one that was swept and is the one to build.
//
// WHY IT NEEDS NO FFT, AND WHY IT NEEDS NO MULTIPLIES EITHER.
// Multiplying by m in the spatial domain is circular convolution by DFT(m) in
// the frequency domain, and a raised cosine of period n has a THREE-BIN
// spectrum: {n/2, -n/4, -n/4} at bins {0, +1, -1}, REAL — the identical rule
// already written down in apply_dc_correction()'s subtract branch. The mask is
// separable, so the 2-D convolution factors into two 1-D ones, and both
// constants cancel against the 1/(rows*cols) of the transform pair:
//
//     h <- m (.) h    ==    H <- D_row(D_col(H)) / 16
//     with  D(X)[i] = 2*X[i] - X[i-1] - X[i+1]     (CIRCULAR on that axis)
//
// Verified against an exact FFT round-trip to 8e-16. So it is 8 complex ADDS
// per bin and one scaling — not the 9 complex MACs a naive reading of "9 bins"
// suggests, and not the 2.4 MMAC/frame the proposal first costed it at.
//
// WHERE IT GOES IS MEASURED, NOT ASSUMED. resp[0,0] is the zero-displacement
// bin, but h's ENERGY is centred at the PATCH CENTRE (peak of SUM|h|^2 at
// (64,64); a corner-wrapped 64x64 box holds 8-12%). The convolution form above
// carries that convention for free — DFT(m) is taken about the same origin as
// H — which is exactly why it is expressed as a convolution and not as an
// index-space window.
//
// AXES SHORTER THAN 3 BINS ARE SKIPPED, NOT WRAPPED. At rows == 1 the circular
// D would read X[-1] == X[+1] == X[0] and return 2X - X - X = ZERO, i.e. it
// would silently delete the filter. FilterState is also the DSST scale
// filter's type at rows == 1, so this is a live trap and not a hypothetical.
//
// In place on `H`, one channel's map. The intermediate is a function-local
// `static thread_local` vector, grown once: TAIL_PARALLEL runs the fused path on
// core 1 and the frame-0 path on core 0, so a plain file-static would be a
// shared buffer across two threads for no gain. 128 KB per thread at 128x128.
void filter_mask_project(cfloat *H, int rows, int cols);

// Fraction of the filter's energy inside a centred box of box_rows x box_cols
// BINS — the mechanism check for FILTER_MASK, and the only way this build can
// answer its own falsifier. Without it "EAO moved" is unattributable to the
// mask. `H_all` is channels * rows * cols, channel-major, UNQUANTISED (the
// h_scratch the fused path already holds), so this costs one pass and no
// device traffic. Returns 0 when the box or the geometry is degenerate.
//
// Centred at the PATCH CENTRE (rows/2, cols/2), matching the mask.
double filter_box_energy_fraction(const cfloat *H_all, int channels,
                                  int rows, int cols,
                                  int box_rows, int box_cols);

// H_ch = A_ch / (B + eps), scaled per channel by 1/sqrt(energy[ch]) (Stage B3),
// then quantised to Q1.15 with ONE global scale so that the largest |H| bin
// across all channels maps to full scale.
//
// A single global scale, not per-channel: cmul_accum sums the channels, so a
// per-channel scale would silently reweight them relative to each other.
//
// `energy` may be null to skip the B3 normalisation. `out` receives
// channels * rows * cols cint16 values as interleaved int16 {re, im}, matching
// filter_bo's layout. out_scale and out_max_abs are diagnostics — the caller
// should log them, because a spiky filter that leaves the response far below the
// rails shows up here and nowhere else.
void filter_quantize_q15(const FilterState &st, const double *energy,
                         float eps_rel, int16_t *out,
                         float *out_scale, float *out_max_abs);

// filter_update() followed by filter_quantize_q15(), fused. BIT-IDENTICAL output
// to calling the two in sequence — asserted element by element in
// run_fusion_tests(), not merely intended.
//
// WHY. Measured on the A72 2026-08-20 (runs/run_0820_1807.log): `filter update`
// 10.18 ms/frame and `publish filter` 5.60 ms, together a quarter of a 62.7 ms
// frame, and -fcx-limited-range bought only 0.48 ms of it because the pair is
// memory-bound rather than arithmetic-bound. Unfused, A is streamed FOUR times
// per frame — written by the update, then read twice by the quantiser, which
// recomputes the same divide on both passes. Fused, H is formed once in the pass
// that writes A and parked in `h_scratch`; the write-out is a scalar multiply.
// The B accumulation is also flipped to channel-major, which turns st.channels
// strided readers of F_all into one sequential one.
//
// `h_scratch` is the caller's, resized on first use and reused thereafter, so a
// 2 MB allocation does not land in the per-frame path. It holds H BEFORE the
// Q1.15 scaling and is otherwise private to this function.
void filter_update_quantize(FilterState &st, const cfloat *F_all,
                            const cfloat *G, float eta,
                            const double *energy, float eps_rel,
                            std::vector<cfloat> &h_scratch,
                            int16_t *out, float *out_scale, float *out_max_abs);

// -----------------------------------------------------------------------
// 1-D scale filter — DSST (docs/1609.06141v1.pdf) §5.1
// -----------------------------------------------------------------------
// "We propose the Discriminative Scale Space Tracking (DSST), which is based on
//  learning a separate 1-dimensional scale correlation filter."
//
// WHY A SEPARATE 1-D FILTER RATHER THAN AN EXHAUSTIVE MULTI-RESOLUTION SEARCH.
// Danelljan's ICCV'15 paper applies the translation filter at several
// resolutions; DSST Table 1 shows the separate filter beats that on BOTH axes
// (OP 67.7 vs 65.2, 25.4 vs 16.9 FPS — THEIR numbers, docs/papers/danelljan2017_fdsst.pdf,
// not measurements of this design). On THIS hardware the gap is wider than
// the paper's, for a reason specific to the fixed-point pipeline: an exhaustive
// search pushes patches resampled by +/-30% through roi_crop -> conv2d -> FFT
// every frame, which moves |F| and therefore the accumulator scale and therefore
// the shift budget — the coupling that has already forced two budget hunts here.
// DSST's translation filter always runs at the CURRENT scale, so the budget that
// was validated stays valid. Costing it against the post-vectorization figures,
// a 3-scale exhaustive search spends exactly the 30 fps headroom that
// vectorizing conv2d and cmul bought back.
//
// THE FILTER IS THE EXISTING FILTER AT rows == 1. DSST §3 says so outright:
// "the same approach can be used to learn 1-dimensional scale estimation
//  filters, 2-dimensional translation estimation filters and 3-dimensional joint
//  scale and translation estimation filters. This is accomplished by only
//  adapting the feature extraction step for each case."
// Confirmed against this code: filter_init/filter_update/FilterState::resize read
// their geometry from the state and touch only rows*cols and channels, and
// gaussian_target_spectrum(G, 1, S, sigma, 0, 0) degenerates cleanly because
// signed_freq(0,1) == 0 makes the row factor identically 1. So the reused surface
// is large and the genuinely new code is the feature extraction and a DFT.
//
// WHAT THIS DELIBERATELY DOES NOT DO YET: fDSST's PCA compression (§5.2.3) and
// sub-grid interpolation (§5.2.1). The compression is PROVABLY LOSSLESS for the
// scale filter — rank(C_scale) <= S, so the d~1000 template compresses to exactly
// S dimensions "without any loss of information" — which means it can be added
// later as a pure optimisation with a bit-exactness test against this path, the
// same pattern CONV_VECTORIZE used. Doing it now would add a QR decomposition and
// an interpolation grid to calibrate before anything has been measured.

// Number of scales. DSST §6.1 uses S = 33. MUST BE ODD so that index (S-1)/2 is
// "no scale change". SCALE_N = 1 disables the filter entirely and reproduces the
// pre-scale-filter behaviour exactly.
#ifndef SCALE_N
#  define SCALE_N 33
#endif
// Scale factor between adjacent levels; DSST §6.1 uses a = 1.02, giving a total
// range of 1.02^+/-16 = +/-38%.
#ifndef SCALE_STEP
#  define SCALE_STEP 1.02
#endif
// Learning rate. DSST §6.1 uses 0.025 for BOTH filters — note the translation
// filter here still uses Bolme's 0.125 (MOSSE_ETA), so these are deliberately
// separate knobs rather than one shared constant.
#ifndef SCALE_ETA
#  define SCALE_ETA 0.025
#endif
// sigma of the desired 1-D output, as a fraction of S. DSST §6.1: "the standard
// deviation in the scale dimension of the desired correlation output g is set to
// 1/16 times the number of scales S."
#ifndef SCALE_SIGMA_FACTOR
#  define SCALE_SIGMA_FACTOR 16.0
#endif
// Cap on the scale template's area, DSST §6.1's 512 px. The feature dimension d
// is the template's pixel count, and the DFT cost is d * S^2, so this is what
// keeps the filter at ~0.5M complex MACs/frame against the translation update's
// ~2M.
#ifndef SCALE_TMPL_AREA
#  define SCALE_TMPL_AREA 512
#endif

constexpr int   DEFAULT_SCALE_N     = (int)(SCALE_N);
constexpr float DEFAULT_SCALE_STEP  = (float)(SCALE_STEP);
constexpr float DEFAULT_SCALE_ETA   = (float)(SCALE_ETA);

// Scale-gate thresholds — see scale_gate() below for the hardware data these
// come from. Casts rather than suffixed literals so `SCALE_CONF_MIN=2` and
// `=2.5` both compile, matching DEFAULT_PSR_MIN.
#ifndef SCALE_CONF_MIN
#  define SCALE_CONF_MIN 2.0
#endif
#ifndef SCALE_MIN_REL
#  define SCALE_MIN_REL 0.5
#endif
#ifndef SCALE_MAX_REL
#  define SCALE_MAX_REL 2.0
#endif
// Largest |idx| a SINGLE frame may move the box. 0 disables the test, matching
// PSR_GATE_MIN=0 and SCALE_CONF_MIN=0 (the structural vetoes still apply).
// See scale_gate()'s MaxStep block for the two datasets behind the default.
#ifndef SCALE_MAX_STEP
#  define SCALE_MAX_STEP 2
#endif
constexpr float  DEFAULT_SCALE_CONF_MIN = (float)(SCALE_CONF_MIN);
constexpr double DEFAULT_SCALE_MIN_REL  = (double)(SCALE_MIN_REL);
constexpr double DEFAULT_SCALE_MAX_REL  = (double)(SCALE_MAX_REL);
constexpr int    DEFAULT_SCALE_MAX_STEP = (int)(SCALE_MAX_STEP);

// Direct O(n^2) DFT. n = 33 is not a power of two, so an FFT would need a
// Bluestein or mixed-radix path; at 33 points a direct transform is ~1089
// complex MACs and about twenty lines. This design deliberately eliminated
// KissFFT (see the header note above) and should not reacquire a dependency for
// a transform this small.
void dft_1d(const cfloat *in, cfloat *out, int n, bool inverse);

struct ScaleFilter {
    int   n_scales = 0;
    int   tmpl_h = 0, tmpl_w = 0;     // scale template, <= SCALE_TMPL_AREA px
    float step  = 1.0f;               // a
    float sigma = 0.0f;               // in scale bins
    bool  initialized = false;

    FilterState         st;           // rows=1, cols=n_scales, channels=tmpl_h*tmpl_w
    std::vector<cfloat> G;            // desired 1-D output, length n_scales

    int dims()  const { return tmpl_h * tmpl_w; }        // d
    int sample_elems() const { return dims() * n_scales; }
    bool enabled() const { return n_scales > 1; }
};

// Size the template from the initial target box and precompute G.
void scale_filter_config(ScaleFilter &sf, int n_scales, float step,
                         double target_h, double target_w, float sigma_factor);

// Build the d x S training/test sample: for each scale level n, crop a
// (step^n * box) region centred at (row,col), resample to the template, apply a
// Hann window, then zero-mean and unit-L2 normalise THAT LEVEL.
//
// Per level, not jointly: it is the direct analogue of Stage A (which zero-means
// and unit-norms each patch) and it makes the filter robust to illumination. It
// discards the absolute-energy cue between levels and keeps the PATTERN cue,
// which is the one that actually identifies the scale — at the correct level the
// target fills the template as it did in training, at a wrong level it does not.
//
// `frame` is the raw uint8 frame. Deliberately a plain pointer and not an
// xrt::bo: this function has to stay compilable by the native test.
void scale_extract(const ScaleFilter &sf, const uint8_t *frame,
                   int frame_rows, int frame_cols,
                   double row, double col, double box_h, double box_w,
                   cfloat *F_out);

struct ScaleResult {
    int    idx    = 0;      // winning level, signed: 0 = no change
    double factor = 1.0;    // step^idx, already clamped
    double peak   = 0.0;    // correlation score at the winner
    double psr    = 0.0;    // (peak - mu)/sigma over the other levels
    bool   valid  = false;  // false when the filter is disabled or untrained
};

// Apply the filter to a sample and pick the winning scale.
ScaleResult scale_detect(const ScaleFilter &sf, const cfloat *Z, float eps_rel);

// Train. First call initialises (eta = 1 against a zeroed state), as for the
// translation filter.
void scale_update(ScaleFilter &sf, const cfloat *F, float eta);

// Train on the sample that was extracted for DETECTION, i.e. at the box BEFORE
// the accepted resize, by shifting the target `idx` levels instead of
// re-extracting at the resized box.
//
// THE IDENTITY THIS RESTS ON IS EXACT, not an approximation. Level n of an
// extraction at box*a^idx crops box*a^idx*a^n = box*a^(idx+n), which is level
// idx+n of the extraction already in hand — same centre, same template, same
// Hann window, same per-level normalisation. So F_new[k] = Z[k+idx], a pure
// shift along the scale axis, and by the shift theorem
//
//     conj(G_0)[m] * F_new[m]  ==  conj(G_idx)[m] * Z[m]
//
// where G_idx = gaussian_target_spectrum(..., dc = idx). Both the numerator and
// |F|^2 in the denominator are therefore unchanged, in exact arithmetic, by
// training on Z against a shifted target instead. This is the same manoeuvre the
// translation filter already makes with psr_abs.dr/dc — one axis down — and the
// SIGN is the same: G is centred at +idx, the level the detector reported.
//
// WHAT IT COSTS. Only the |idx| levels at the far end of the search range differ:
// the re-extraction would have cropped genuinely new levels there, whereas here
// they simply are not trained on. On hardware the detector proposed only -1, 0
// or +1 across 199 frames (run_0820_1807.log), so that is at most one level of
// 33, and at idx = 0 — 174 of those 199 frames — the two are identical.
//
// WHAT IT BUYS. scale_extract() is 4.73 ms/call and ran twice per frame, 15% of
// the frame; this removes one of the two calls.
void scale_update_shifted(ScaleFilter &sf, const cfloat *F, int idx, float eta);

// -----------------------------------------------------------------------
// Scale-update gating — the direct analogue of psr_gate() for the size axis
// -----------------------------------------------------------------------
// WHY THIS EXISTS. Measured on hardware 2026-08-20, ch16, TRAJECTORY=0 with the
// background panning and position tracking EXACT (IoU 1.0000, centre error
// 0.00 px through frame 5): the scale filter jumped to level -12 on frame 6 and
// took the box from 64.0 to 50.5 in one step. That shrank the ROI, which then
// broke position tracking, which fed the scale filter off-target patches, and
// the box thrashed -12/+5/+10/-14/+4/+10/-14/+16 from there.
//
// **THE ORDERING MATTERS AND IT REVERSES WHAT WAS PREVIOUSLY BELIEVED.** The
// collapse was recorded as a SYMPTOM of background lock -> position drift. It is
// not: it fires first, with position perfect and the background moving. Nothing
// upstream of it needs to be wrong.
//
// `conf` (ScaleResult::psr) separates the two populations cleanly, on three
// independent hardware runs:
//
//   healthy   3.30-3.31 (2026-08-20 A) | 2.24-3.30 (2026-08-20 B) | 2.37-3.24
//   collapsed 0.91-1.87                | 1.05-1.63               | 0.72-1.85
//
// 2.0 sits in the gap every time. `ScaleResult::peak` separates too (0.15-0.21 vs
// 0.033-0.051) but is not scale-free, so `conf` is the gate.
//
// THREE VETOES, NOT ONE, and they are reported separately for the same reason
// GateReason is an enum: "size held" alone does not tell you whether the filter
// was uncertain, structurally wrong, or merely clamped.
//
//   AtSearchRail   |idx| == (n_scales-1)/2. An argmax ON the boundary of the
//                  search range cannot be the true optimum — the true one lies
//                  outside it. At the defaults the filter steps 2%/frame against
//                  a size envelope moving under 1%/frame, so the target can never
//                  legitimately be at the rail. Checked BEFORE LowConf because it
//                  is the stronger statement: frame 13 of the run above argmaxed
//                  at exactly +16 of +/-16.
//   MaxStep        |idx| > max_step. A RATE limit on one frame's proposal, where
//                  OutOfRange is a drift bound on the accumulated box. Checked
//                  after AtSearchRail (an argmax on the boundary is the stronger
//                  statement) and before the range and confidence tests, because
//                  a proposal this large is implausible whatever its conf says.
//
//                  THE PRECONDITION THIS DOES NOT REPLACE: the caller must
//                  already have decided the POSITION is trustworthy. The tracker
//                  runs this whole block under `gate.accept`, so a held frame
//                  never reaches the scale filter at all. That slaving is real
//                  and was verified on hardware (run_0825_1314: 577 position
//                  ACCEPTs, 577 scale evaluations, zero on a held frame) — this
//                  veto exists because it is NOT sufficient. On that run the
//                  position gate accepted frame 490 at PSR 7.87 against a 7.00
//                  threshold while the tracker was 227 px off target, and the
//                  scale filter then inflated the box 1.42x in one frame.
//
//                  DEFAULT 2, AND THE OBVIOUS 1 IS WRONG. The hardware evidence
//                  alone argues for 1: on car1's 742 frames every proposal with
//                  |idx| >= 2 (seven of them, up to +9) landed on a frame whose
//                  IoU was 0.000. `make scale_sim` then measured what 1 costs on
//                  the arm that tracks NORMALLY, and it is not subtle:
//
//                    --max-step   moving max|err|   end err   held   step end err
//                        0 (off)          10.4%       1.0%       5         8.3%
//                        1                81.2%      28.0%     123        42.9%
//                        2                10.4%       1.0%       5        42.9%
//                        3                10.4%       1.0%       5        42.9%
//
//                  The sim's detector legitimately USES |idx| = 2 on a smooth
//                  envelope, so a limit of 1 parks the filter for 123 of 200
//                  frames and ends 28% wrong. 2 is the tightest value that
//                  costs the smooth arm exactly nothing, and it still vetoes
//                  three of car1's seven bad proposals — including frame 490's
//                  +9, the 1.42x inflation that motivated this. The four at
//                  |idx| = 2 now pass; that is the price of not breaking the
//                  normal case, and it is stated rather than hidden.
//
//                  NOTE this corrects a claim in CLAUDE.md: "the detector
//                  proposed only -1, 0 or +1 over 199 frames, never +/-2" is
//                  true of the HARDWARE run it was written from and NOT of the
//                  sim, which is the bench that decides this parameter.
//
//                  THE COST IS REAL AND IT IS ON THE `step` ARM: an abrupt scale
//                  change ends 42.9% wrong at ANY limit against 8.3% with none.
//                  That is the same failure SCALE_CONF_MIN already has — the sim
//                  proposes a correct idx=-14 after a jump and the conf gate
//                  vetoes it — and it cannot be tuned away, because "wrong
//                  proposal" and "big correct correction" are the same
//                  measurement. Set SCALE_MAX_STEP=0 for a sequence with abrupt
//                  scale change and accept the runaway risk knowingly.
//   LowConf        conf < conf_min. The occlusion/deformation indicator.
//   OutOfRange     the PROPOSED box leaves [h0*min_rel, h0*max_rel]. The old
//                  0.25/4.0 was so loose it never fired in any run on record;
//                  it is a backstop against accumulated drift, not a per-frame
//                  test, so it must still admit the test sequence's own envelope
//                  (SCALE_TRAJ_AMP=0.30 => 0.70x..1.30x).
//
// A veto holds the box AND skips scale_update(), exactly as psr_gate() holds the
// position and skips filter_update(). Training on a frame whose estimate was
// rejected is what turns one bad frame into a runaway.
enum class ScaleVeto {
    Accept,        // conf >= threshold, argmax interior, proposed box in range
    Disabled,      // conf_min <= 0: threshold test off, accepted by policy
    Invalid,       // !sr.valid — filter disabled or not yet trained
    AtSearchRail,  // argmax on the boundary of the search range
    MaxStep,       // one frame's proposal moves the box by more than max_step
    LowConf,       // conf below the threshold
    OutOfRange,    // proposed box outside the absolute bounds
};

struct ScaleDecision {
    // Default FALSE on purpose: an unset decision must never resize the box.
    bool      accept    = false;
    ScaleVeto reason    = ScaleVeto::Invalid;
    double    conf      = 0.0;
    double    new_h     = 0.0;   // the PROPOSAL, whether or not it was accepted
    double    new_w     = 0.0;
    float     threshold = 0.0f;
};

// `cur_h/cur_w` are the box now; `h0/w0` are the INITIAL box the relative bounds
// are measured against. Thresholds are parameters rather than being read from the
// DEFAULT_* constants internally, so the unit tests can sweep them without
// depending on the build's -D.
ScaleDecision scale_gate(const ScaleResult &sr, int n_scales,
                         double cur_h, double cur_w, double h0, double w0,
                         float conf_min, double min_rel, double max_rel,
                         int max_step);

// ---------------------------------------------------------------------------
// COASTING THROUGH A HOLD
// ---------------------------------------------------------------------------
// On a gate veto the tracker holds position, which assumes the target stays put
// while the filter is frozen. Measured against stb2022 groundtruth
// (scripts/vot_hold_budget.py), the HOLD BUDGET -- frames before the target
// leaves the frozen box*padding window, after which recovery is impossible for
// ANY tracker -- has a median of 6 frames, is <= 4 on 30 of 62 sequences and is
// 0 on four. car1's budget is 4; its longest hold on hardware was 53.
//
// So a held frame moves the window at the last MEASURED velocity, decayed each
// successive held frame. The decay is what makes this safe: total drift over one
// hold run is bounded by |v| * 1/(1-decay), so a long hold fades back to a
// freeze instead of becoming a second way to lose the target. That bound is the
// property worth testing, and it is why this lives here rather than inline in
// the tracker -- mosse_filter has no XRT header, so `make test_host` checks it
// in seconds.
//
// Swept offline over all 62 sequences (mean over sequences):
//
//   policy             P(survive 1 held frame)  P(survive 3)  median budget
//   freeze                       90.3%             69.9%            6
//   coast decay 0.0              94.8%             74.4%            7
//   coast decay 0.5  <-- ship    94.9%             76.2%            8
//   coast decay 1.0              94.9%             74.6%            6
//
// PURE constant velocity (1.0) is NOT the answer: on a near-stationary target
// the measured velocity is mostly detection noise, and it takes `nature` from 83
// frames of budget to 34 and `girl` from 39 to 21. At 0.5, 40 sequences improve,
// 15 are unchanged and 7 are marginally worse.
struct CoastState {
    double vr = 0.0, vc = 0.0;   // last measured per-frame velocity, FRAME px
    double scale = 1.0;          // decay factor for the CURRENT hold run
};

// Called on an ACCEPTED frame with the displacement that frame applied. Never on
// a coasted frame: a coast moves the box BY the velocity, so learning from it
// would make the estimate self-confirming and the decay would never take effect.
void coast_observe(CoastState &c, double dr_frame, double dc_frame);

// Called on a HELD frame. Writes the offset to apply and advances the decay.
// Returns false when there is nothing to coast on (no velocity measured yet, or
// the run has decayed to nothing), in which case the caller freezes as before.
bool coast_step(CoastState &c, double decay, double *dr_frame, double *dc_frame);

// The drift bound above, as a function: |v| * 1/(1-decay), or infinity at
// decay >= 1. Exposed so the test asserts the bound rather than a magic number.
double coast_drift_bound(const CoastState &c, double decay);

const char *scale_veto_tag(ScaleVeto v);   // "ACCEPT" / "AT_SEARCH_RAIL" / ...
const char *scale_veto_why(ScaleVeto v);   // one sentence, for the log

}  // namespace mosse
