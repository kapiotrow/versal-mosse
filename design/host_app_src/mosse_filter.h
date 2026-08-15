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
// across the whole test set.
constexpr float DEFAULT_ETA = 0.125f;
// Target Gaussian width, in pixels. Bolme §3.1 uses sigma = 2.0.
constexpr float DEFAULT_SIGMA = 2.0f;
// Regularization, as a FRACTION of mean(B) rather than an absolute value: B's
// magnitude depends on the shift budget and the feature scale, so an absolute
// epsilon would silently become either a no-op or a dominant term after any
// recalibration.
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
// At runtime dr = dc = 0 (the target sits at the patch centre, so "no motion"
// maps to displacement (0,0)). Non-zero offsets exist for testing.
void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma, int dr, int dc);

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

}  // namespace mosse
