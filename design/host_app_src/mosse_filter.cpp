/*
 * mosse_filter.cpp — see mosse_filter.h for the derivation and, in particular,
 * for the conjugation convention. No XRT/ADF headers: this file is compiled
 * natively by `make test_host`.
 */

#include "mosse_filter.h"

#include <climits>
#include <cmath>
#include <cstring>

namespace mosse {

namespace {

// Signed frequency index: bins above N/2 represent negative frequencies.
inline int signed_freq(int k, int n)
{
    return (k > n / 2) ? (k - n) : k;
}

constexpr float kTwoPi = 6.283185307179586f;

}  // namespace

// -----------------------------------------------------------------------
// Peak-to-sidelobe ratio and update gating — Bolme §3.5. See mosse_filter.h.
// -----------------------------------------------------------------------
// Lives here rather than in mosse_tracker.cpp so `make test_host` covers it:
// this file includes no XRT header, so the gate policy is checkable natively in
// seconds instead of in a ~26 min hw_emu frame. That matters more for the gate
// than for anything else in this file, because the gate CANNOT fire on the
// current synthetic test data (PSR ~172 against a threshold of 7) — the unit
// tests are the only place its failure paths are exercised at all.
PsrResult compute_psr(const int16_t *resp, int rows, int cols, bool use_abs)
{
    const int EXCL_HALF = PSR_EXCL_HALF;    // 11x11 window, Bolme §3.5

    long best = LONG_MIN;
    int  pr = 0, pc = 0;
    for (int i = 0; i < rows * cols; ++i) {
        const long re = resp[2 * i];
        const long v  = use_abs ? ((re < 0) ? -re : re) : re;
        if (v > best) { best = v; pr = i / cols; pc = i % cols; }
    }

    // Sidelobe membership test, factored out so the two passes cannot disagree.
    auto in_sidelobe = [&](int r, int c) {
        int ddr = r - pr; if (ddr < 0) ddr = -ddr;
        if (rows - ddr < ddr) ddr = rows - ddr;       // circular distance
        int ddc = c - pc; if (ddc < 0) ddc = -ddc;
        if (cols - ddc < ddc) ddc = cols - ddc;
        return !(ddr <= EXCL_HALF && ddc <= EXCL_HALF);
    };

    // TWO passes for the variance, not the single-pass sum2/n - mean^2 form.
    // The sidelobe can sit on a large DC pedestal, i.e. exactly the mean >> sdev
    // regime where single-pass variance loses precision to cancellation. This is
    // the instrument the gate is judged by, so it is computed the stable way;
    // two extra passes over 16K elements once per frame costs nothing measurable.
    double sum = 0.0, smax = 0.0;
    long   n   = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (!in_sidelobe(r, c)) continue;
            const double v = (double)resp[2 * (r * cols + c)];
            sum += v; ++n;
            const double a = (v < 0.0) ? -v : v;
            if (a > smax) smax = a;
        }

    PsrResult out{};
    out.peak     = (long)resp[2 * (pr * cols + pc)];
    out.n_side   = n;
    out.mean     = n ? sum / n : 0.0;

    double ss = 0.0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (!in_sidelobe(r, c)) continue;
            const double d = (double)resp[2 * (r * cols + c)] - out.mean;
            ss += d * d;
        }
    const double var = n ? ss / n : 0.0;
    out.sdev     = var > 0.0 ? std::sqrt(var) : 0.0;
    // sdev/side_max == 0 means a perfectly flat sidelobe — an all-zero response,
    // not an infinitely good peak. Report 0 so it reads as failure, not success.
    out.psr      = (out.sdev > 0.0) ? ((double)out.peak - out.mean) / out.sdev : 0.0;
    out.side_max = smax;
    const double pabs = (out.peak < 0) ? -(double)out.peak : (double)out.peak;
    out.ratio    = (smax > 0.0) ? pabs / smax : 0.0;
    out.dr       = (pr > rows / 2) ? pr - rows : pr;
    out.dc       = (pc > cols / 2) ? pc - cols : pc;
    return out;
}

GateDecision psr_gate(const PsrResult &p, float psr_min)
{
    GateDecision g{};
    g.psr       = p.psr;
    g.threshold = psr_min;

    // Most-specific first; the first hit wins and is the reported reason. The
    // structural cases are separated from LowPsr on purpose: compute_psr reports
    // psr = 0 for a degenerate sidelobe, which would read as "occluded" when in
    // fact the PIPELINE produced nothing — a shift-budget or filter-scale fault,
    // not a scene event. Those need different responses from a reader.
    if (p.n_side == 0)      { g.reason = GateReason::EmptySidelobe; return g; }
    if (p.peak   == 0)      { g.reason = GateReason::ZeroResponse;  return g; }
    if (p.sdev   <= 0.0)    { g.reason = GateReason::FlatSidelobe;  return g; }
    // The target g is a POSITIVE Gaussian by construction, so the strongest
    // structure being negative is anti-correlation, not a detection. Acting on it
    // would train the filter to invert itself.
    if (p.peak    < 0)      { g.reason = GateReason::NegativePeak;  return g; }

    // Placed AFTER the structural checks so those still veto (and still report
    // their own reason) even with the threshold test switched off.
    if (psr_min <= 0.0f) { g.accept = true; g.reason = GateReason::Disabled; return g; }

    // `>=` matches report_psr's classifier boundary; using `>` there and `>=`
    // here would make the two lines contradict each other at exactly 7.00.
    if (p.psr >= (double)psr_min) { g.accept = true; g.reason = GateReason::Accept; }
    else                          {                  g.reason = GateReason::LowPsr; }
    return g;
}

const char *gate_reason_tag(GateReason r)
{
    switch (r) {
        case GateReason::Accept:        return "ACCEPT";
        case GateReason::Disabled:      return "ACCEPT(gate disabled)";
        case GateReason::ZeroResponse:  return "ZERO_RESPONSE";
        case GateReason::EmptySidelobe: return "EMPTY_SIDELOBE";
        case GateReason::FlatSidelobe:  return "FLAT_SIDELOBE";
        case GateReason::NegativePeak:  return "NEGATIVE_PEAK";
        case GateReason::LowPsr:        return "LOW_PSR";
    }
    return "?";
}

const char *gate_reason_why(GateReason r)
{
    switch (r) {
        case GateReason::Accept:
            return "PSR is at or above the threshold — normal tracking.";
        case GateReason::Disabled:
            return "PSR threshold disabled (PSR_GATE_MIN <= 0); report-only mode.";
        case GateReason::ZeroResponse:
            return "the response map is identically zero — the pipeline produced "
                   "nothing. Check the filter scale and the shift budget, not the scene.";
        case GateReason::EmptySidelobe:
            return "the patch is not larger than the 11x11 exclusion window, so "
                   "there is no sidelobe to measure. This is a geometry error.";
        case GateReason::FlatSidelobe:
            return "the sidelobe is perfectly constant, so PSR is undefined (not "
                   "infinite). Usually a railed or saturated response.";
        case GateReason::NegativePeak:
            return "the strongest response is NEGATIVE, i.e. anti-correlation. The "
                   "target output g is a positive Gaussian, so this is not a detection.";
        case GateReason::LowPsr:
            return "Bolme §3.5: PSR at or below ~7 indicates the object is occluded "
                   "or tracking has failed. Holding position and freezing the filter.";
    }
    return "";
}

void FilterState::resize(int rows_, int cols_, int channels_)
{
    rows     = rows_;
    cols     = cols_;
    channels = channels_;
    A.assign((size_t)channels_ * rows_ * cols_, cfloat(0.0f, 0.0f));
    B.assign((size_t)rows_ * cols_, 0.0f);
    initialized = false;
}

void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma, int dr, int dc)
{
    // A Gaussian of width sigma in space is a Gaussian of width 1/sigma in
    // frequency. For the DISCRETE wrapped Gaussian the exact transform is a theta
    // function; the leading term below is accurate to well under one cint16 LSB at
    // sigma = 2, N = 128, and its error decreases as sigma grows.
    //
    //   |G[u,v]| = 2*pi*sigma^2 * exp(-2*pi^2*sigma^2*(u'^2/M^2 + v'^2/N^2))
    //
    // The leading 2*pi*sigma^2 is a constant gain across all bins. It is dropped:
    // correlation is linear in the filter and filter_quantize_q15() renormalises
    // to full scale anyway, so carrying it would only cost dynamic range.
    const float s2 = sigma * sigma;
    const float ku = -2.0f * 9.869604401089358f * s2 / (float)(rows * rows);
    const float kv = -2.0f * 9.869604401089358f * s2 / (float)(cols * cols);

    for (int u = 0; u < rows; ++u) {
        const int   us = signed_freq(u, rows);
        const float mu = std::exp(ku * (float)(us * us));
        for (int v = 0; v < cols; ++v) {
            const int   vs = signed_freq(v, cols);
            const float mag = mu * std::exp(kv * (float)(vs * vs));
            // Shifting the target by (dr, dc) in space is a linear phase ramp in
            // frequency. Exact, no approximation. This is the ONLY thing that
            // makes G complex; with dr = dc = 0 it is real and symmetric, which is
            // why a centred target hides conjugation mistakes.
            const float ph = -kTwoPi * ((float)(us * dr) / (float)rows
                                      + (float)(vs * dc) / (float)cols);
            G[(size_t)u * cols + v] = cfloat(mag * std::cos(ph), mag * std::sin(ph));
        }
    }
}

void filter_init(FilterState &st, const cfloat *F_all,
                 const cfloat *G, int channels, int rows, int cols)
{
    st.resize(rows, cols, channels);
    // eta = 1 against a zeroed state is exactly the closed form for one training
    // image, so there is no separate code path to keep in sync.
    filter_update(st, F_all, G, 1.0f);
}

void filter_update(FilterState &st, const cfloat *F_all,
                   const cfloat *G, float eta)
{
    const size_t n     = st.elems();
    const float  keep  = 1.0f - eta;

    // Denominator first: B is shared, so it must be fully accumulated over all
    // channels before it means anything. Doing it in the same pass as A would
    // leave the last channel's H divided by a partial sum.
    for (size_t i = 0; i < n; ++i) {
        float energy = 0.0f;
        for (int ch = 0; ch < st.channels; ++ch) {
            const cfloat f = F_all[(size_t)ch * n + i];
            energy += f.real() * f.real() + f.imag() * f.imag();
        }
        st.B[i] = eta * energy + keep * st.B[i];
    }

    // Numerator: conj(G) * F, NOT G * conj(F). See the header.
    for (int ch = 0; ch < st.channels; ++ch) {
        cfloat       *a = st.A.data() + (size_t)ch * n;
        const cfloat *f = F_all + (size_t)ch * n;
        for (size_t i = 0; i < n; ++i)
            a[i] = eta * std::conj(G[i]) * f[i] + keep * a[i];
    }

    st.initialized = true;
}

void filter_quantize_q15(const FilterState &st, const double *energy,
                         float eps_rel, int16_t *out,
                         float *out_scale, float *out_max_abs)
{
    const size_t n = st.elems();

    // eps relative to mean(B), not absolute — B's magnitude moves with the shift
    // budget and the feature scale.
    double b_sum = 0.0;
    for (size_t i = 0; i < n; ++i) b_sum += st.B[i];
    const float eps = eps_rel * (float)(b_sum / (double)(n ? n : 1));

    // Stage B3: normalising a channel to unit energy is identical to scaling its
    // filter by 1/sigma_ch, because correlation is linear in the patch. Folding it
    // in here costs nothing — no AIE work, no extra DDR traffic.
    std::vector<float> chscale((size_t)st.channels, 1.0f);
    if (energy) {
        for (int ch = 0; ch < st.channels; ++ch) {
            const double e = energy[ch];
            chscale[(size_t)ch] = (e > 0.0) ? (float)(1.0 / std::sqrt(e)) : 0.0f;
        }
    }

    // Two passes: the Q1.15 scale is global across channels, so the maximum has to
    // be known before anything is written. A per-channel scale would silently
    // reweight the channels relative to one another, and cmul_accum sums them.
    float max_abs = 0.0f;
    for (int ch = 0; ch < st.channels; ++ch) {
        const cfloat *a = st.A.data() + (size_t)ch * n;
        const float   cs = chscale[(size_t)ch];
        for (size_t i = 0; i < n; ++i) {
            const cfloat h = a[i] * (cs / (st.B[i] + eps));
            const float  m = std::abs(h);
            if (m > max_abs) max_abs = m;
        }
    }

    // Always normalize to the FULL int16 range, independent of CMUL_H_SHIFT.
    //
    // These are two separate things and tying them together was a bug (it merely
    // looked right at H_SHIFT=15, where (1<<15)-1 == 32767):
    //   - this ceiling sets the RESOLUTION of H — always use all 15 bits;
    //   - H_SHIFT sets the SCALE of the product F*H, i.e. where the accumulator
    //     lands in cint16.
    // With them decoupled, lowering H_SHIFT buys accumulator headroom at no cost
    // to filter precision. Coupled, it threw away one bit of H per bit of gain,
    // which is the opposite of what the knob is for.
    //
    // Measured on aiesim s7 (real MOSSE filter, 64x64, FFT_SHIFT=3): a MOSSE
    // filter is SPIKY — max|H| sits where |F| is smallest, because that is where
    // the regularized inverse peaks — so normalizing the peak bin to full scale
    // leaves every informative bin far below it. At H_SHIFT=15 the accumulator
    // reached only 15 of 32767 and the response came back at 21 LSB: it still
    // localised exactly, but PSR collapsed to 5.2x against a golden 28.9x.
    constexpr float Q15_FULL_SCALE = 32767.0f;
    const float scale = (max_abs > 0.0f) ? (Q15_FULL_SCALE / max_abs) : 0.0f;

    for (int ch = 0; ch < st.channels; ++ch) {
        const cfloat *a = st.A.data() + (size_t)ch * n;
        const float   cs = chscale[(size_t)ch];
        int16_t      *o = out + (size_t)ch * n * 2;
        for (size_t i = 0; i < n; ++i) {
            const cfloat h = a[i] * (cs / (st.B[i] + eps)) * scale;
            float re = std::nearbyint(h.real());
            float im = std::nearbyint(h.imag());
            // Clamped to -32767, NOT -32768. cmul_accum computes
            //     in.re*flt.re + in.im*flt.im
            // in int32; with all four operands at -32768 that is exactly 2^31,
            // one past INT32_MAX. Excluding -32768 from the filter caps the
            // magnitude at 2*32767*32768 = 2147418112 and makes the overflow
            // unreachable. Costs one LSB of range on the negative side and
            // nothing else — the scale is set by |H|, which is symmetric.
            if (re >  32767.0f) re =  32767.0f;
            if (re < -32767.0f) re = -32767.0f;
            if (im >  32767.0f) im =  32767.0f;
            if (im < -32767.0f) im = -32767.0f;
            o[2 * i]     = (int16_t)re;
            o[2 * i + 1] = (int16_t)im;
        }
    }

    if (out_scale)   *out_scale   = scale;
    if (out_max_abs) *out_max_abs = max_abs;
}

}  // namespace mosse
