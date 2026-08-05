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
