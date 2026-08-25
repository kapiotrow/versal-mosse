/*
 * mosse_filter.cpp — see mosse_filter.h for the derivation and, in particular,
 * for the conjugation convention. No XRT/ADF headers: this file is compiled
 * natively by `make test_host`.
 */

#include "mosse_filter.h"

#include <algorithm>   // std::min/std::max in box_iou — explicit, not transitive:
                       // this file is also cross-compiled for aarch64.
#include <climits>
#include <cmath>
#include <limits>
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

// ---------------------------------------------------------------------------
// Target box / ROI geometry
// ---------------------------------------------------------------------------
RoiGeometry roi_for(const TargetBox &box, float padding,
                    int patch_rows, int patch_cols)
{
    RoiGeometry g;
    g.patch_rows = patch_rows;
    g.patch_cols = patch_cols;
    g.roi_h = (int)std::lround(box.h * (double)padding);
    g.roi_w = (int)std::lround(box.w * (double)padding);
    // Centred on the box. Deliberately allowed to go negative: roi_crop clamps by
    // border replication, and a target near a frame edge must still be croppable.
    g.roi_row = (int)std::lround(box.row - g.roi_h / 2.0);
    g.roi_col = (int)std::lround(box.col - g.roi_w / 2.0);
    return g;
}

// One patch bin is roi_h/patch_rows frame pixels. Identity only while the ROI is
// 1:1 with the patch, which is the entire reason this was never needed before.
double patch_dr_to_frame(int dr, const RoiGeometry &g)
{
    return (double)dr * (double)g.roi_h / (double)g.patch_rows;
}

double patch_dc_to_frame(int dc, const RoiGeometry &g)
{
    return (double)dc * (double)g.roi_w / (double)g.patch_cols;
}

int frame_dr_to_patch(double dr, const RoiGeometry &g)
{
    return (int)std::lround(dr * (double)g.patch_rows / (double)g.roi_h);
}

int frame_dc_to_patch(double dc, const RoiGeometry &g)
{
    return (int)std::lround(dc * (double)g.patch_cols / (double)g.roi_w);
}

double target_h_in_patch(const TargetBox &box, const RoiGeometry &g)
{
    return box.h * (double)g.patch_rows / (double)g.roi_h;
}

double target_w_in_patch(const TargetBox &box, const RoiGeometry &g)
{
    return box.w * (double)g.patch_cols / (double)g.roi_w;
}

void sigma_for(const TargetBox &box, const RoiGeometry &g,
               float *sigma_r, float *sigma_c)
{
#if SIGMA_FROM_TARGET
    *sigma_r = (float)(target_h_in_patch(box, g) / (double)DEFAULT_SIGMA_FACTOR);
    *sigma_c = (float)(target_w_in_patch(box, g) / (double)DEFAULT_SIGMA_FACTOR);
#else
    (void)box; (void)g;
    *sigma_r = DEFAULT_SIGMA;
    *sigma_c = DEFAULT_SIGMA;
#endif
}

double box_iou(const TargetBox &a, const TargetBox &b)
{
    const double ar0 = a.row - a.h / 2.0, ar1 = a.row + a.h / 2.0;
    const double ac0 = a.col - a.w / 2.0, ac1 = a.col + a.w / 2.0;
    const double br0 = b.row - b.h / 2.0, br1 = b.row + b.h / 2.0;
    const double bc0 = b.col - b.w / 2.0, bc1 = b.col + b.w / 2.0;

    const double ih = std::max(0.0, std::min(ar1, br1) - std::max(ar0, br0));
    const double iw = std::max(0.0, std::min(ac1, bc1) - std::max(ac0, bc0));
    const double inter = ih * iw;
    const double uni   = a.h * a.w + b.h * b.w - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma, int dr, int dc)
{
    gaussian_target_spectrum(G, rows, cols, sigma, sigma, dr, dc);
}

void gaussian_target_spectrum(cfloat *G, int rows, int cols,
                              float sigma_r, float sigma_c, int dr, int dc)
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
    // Per axis: the row and column exponents were already independent, so an
    // anisotropic target costs nothing but the second constant.
    const float ku = -2.0f * 9.869604401089358f * (sigma_r * sigma_r)
                     / (float)(rows * rows);
    const float kv = -2.0f * 9.869604401089358f * (sigma_c * sigma_c)
                     / (float)(cols * cols);

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

// ---------------------------------------------------------------------------
// 1-D scale filter — DSST §5.1
// ---------------------------------------------------------------------------
namespace {

// Twiddles for an n-point transform: w[j] = exp(sgn * 2*pi*i*j/n), j = 0..n-1.
//
// PRECOMPUTED, and that is a cost issue rather than a tidiness one. The obvious
// direct DFT evaluates sin/cos inside the inner loop, i.e. n^2 transcendental
// calls per transform. scale_extract runs d = ~484 transforms per sample and two
// samples per frame, so at n = 33 that is ~2M sin/cos per frame — tens of
// milliseconds, which would make the scale filter the most expensive thing in a
// design whose entire AIE budget is 6.4 ms. With a table it is n calls per
// transform, and hoisting the table to the caller (below) makes it n per FRAME.
void twiddle_table(std::vector<cfloat> &w, int n, bool inverse)
{
    const double sgn = inverse ? +1.0 : -1.0;
    w.resize((size_t)n);
    // BUILT CONJUGATE-SYMMETRIC BY CONSTRUCTION: w[n-j] = conj(w[j]).
    //
    // Mathematically that is just cos(2*pi*(n-j)/n) = cos(2*pi*j/n), so it looks
    // like a tidiness change. It is not — it is what makes the Hermitian mirror
    // in real_dft_with_table() EXACT rather than merely close. Evaluating
    // std::cos/std::sin at the two arguments independently agrees only to within
    // an ulp, and bin n-k is then not bitwise conj(bin k), so the mirrored half
    // of the spectrum would differ from a directly computed one in the last
    // bit. Mirroring is only free if it is exact.
    //
    // Second-order benefit: the arguments actually evaluated are all <= pi, where
    // the library's argument reduction has nothing to do.
    const int half = n / 2;
    for (int j = 0; j <= half; ++j) {
        const double a = sgn * 2.0 * M_PI * (double)j / (double)n;
        w[(size_t)j] = cfloat((float)std::cos(a), (float)std::sin(a));
        if (j > 0 && n - j > half) w[(size_t)(n - j)] = std::conj(w[(size_t)j]);
    }
}

void dft_with_table(const cfloat *in, cfloat *out, int n,
                    const std::vector<cfloat> &w, bool inverse)
{
    for (int k = 0; k < n; ++k) {
        double re = 0.0, im = 0.0;
        int idx = 0;                       // (k*t) mod n, stepped rather than %
        for (int t = 0; t < n; ++t) {
            const double c = (double)w[(size_t)idx].real();
            const double s = (double)w[(size_t)idx].imag();
            re += (double)in[t].real() * c - (double)in[t].imag() * s;
            im += (double)in[t].real() * s + (double)in[t].imag() * c;
            idx += k;
            if (idx >= n) idx -= n;
        }
        if (inverse) { re /= (double)n; im /= (double)n; }
        out[k] = cfloat((float)re, (float)im);
    }
}

// REAL-INPUT forward transform, exploiting Hermitian symmetry. Same table, same
// summation order, same accumulator type as dft_with_table() — so for a purely
// real input the bins it computes are BITWISE what the complex routine produced.
//
// WHY IT EXISTS. scale_extract() feeds this a sample that is real BY
// CONSTRUCTION (each element is a windowed, normalised pixel; the imaginary part
// was literally written as 0.0f). The complex routine then spent half its
// multiplies on that zero and computed all n bins, of which only n/2+1 carry
// information. Two independent factors:
//
//   real input      4 mul + 4 add per tap  ->  2 mul + 2 add        2.0x
//   Hermitian out   n bins computed        ->  n/2+1 bins           1.9x at n=33
//
// which is the 3.11x that "Settled questions" measured for this exact transform,
// and at 9.44 ms/frame of scale_extract on the A72 it is worth several ms.
//
// The mirror out[n-k] = conj(out[k]) is exact given a conjugate-symmetric
// twiddle table — see twiddle_table(). The `n - k > kmax` guard is for even n,
// where k = n/2 is its own mirror and must not be overwritten.
void real_dft_with_table(const float *in, cfloat *out, int n,
                         const std::vector<cfloat> &w)
{
    const int kmax = n / 2;
    for (int k = 0; k <= kmax; ++k) {
        double re = 0.0, im = 0.0;
        int idx = 0;                       // (k*t) mod n, stepped rather than %
        for (int t = 0; t < n; ++t) {
            const double x = (double)in[t];
            re += x * (double)w[(size_t)idx].real();
            im += x * (double)w[(size_t)idx].imag();
            idx += k;
            if (idx >= n) idx -= n;
        }
        out[k] = cfloat((float)re, (float)im);
        const int km = n - k;
        if (k > 0 && km > kmax) out[km] = cfloat((float)re, (float)(-im));
    }
}

}  // namespace

void dft_1d(const cfloat *in, cfloat *out, int n, bool inverse)
{
    // Direct O(n^2). At n = 33 that is 1089 complex MACs — cheaper to write and
    // to trust than a mixed-radix FFT, and 33 is not a power of two anyway.
    // Unnormalised forward, 1/n on the inverse, matching numpy's convention so
    // the model and this agree.
    std::vector<cfloat> w;
    twiddle_table(w, n, inverse);
    dft_with_table(in, out, n, w, inverse);
}

namespace {

// Periodic Hann, sin^2(pi*i/N) — the same convention as hanning_*.h. The
// symmetric form is deliberately NOT used anywhere in this design.
void hann_into(std::vector<float> &w, int n)
{
    w.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        const double s = std::sin(M_PI * (double)i / (double)n);
        w[(size_t)i] = (float)(s * s);
    }
}

// Q8 corner-aligned bilinear with border replication — the SAME convention as
// roi_crop.cpp, so the design has one resampling rule rather than two. Runs on
// the APU in float; it does not need to be bit-exact with the PL kernel, but
// matching the sampling grid keeps the geometry reasoning transferable.
double sample_bilinear(const uint8_t *frame, int frame_rows, int frame_cols,
                       double y, double x)
{
    const int max_y = frame_rows - 1, max_x = frame_cols - 1;
    const double fy = y - std::floor(y);
    const double fx = x - std::floor(x);
    int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
    int y1 = y0 + 1, x1 = x0 + 1;
    y0 = y0 < 0 ? 0 : (y0 > max_y ? max_y : y0);
    y1 = y1 < 0 ? 0 : (y1 > max_y ? max_y : y1);
    x0 = x0 < 0 ? 0 : (x0 > max_x ? max_x : x0);
    x1 = x1 < 0 ? 0 : (x1 > max_x ? max_x : x1);

    const double p00 = frame[(size_t)y0 * frame_cols + x0];
    const double p01 = frame[(size_t)y0 * frame_cols + x1];
    const double p10 = frame[(size_t)y1 * frame_cols + x0];
    const double p11 = frame[(size_t)y1 * frame_cols + x1];
    const double top = p00 * (1.0 - fx) + p01 * fx;
    const double bot = p10 * (1.0 - fx) + p11 * fx;
    return top * (1.0 - fy) + bot * fy;
}

}  // namespace

void scale_filter_config(ScaleFilter &sf, int n_scales, float step,
                         double target_h, double target_w, float sigma_factor)
{
    // Even S would have no "no change" level, so the tracker could never report
    // scale 1.0 exactly. Round down to odd rather than silently mis-centring.
    if (n_scales > 1 && (n_scales % 2) == 0) --n_scales;
    sf.n_scales = n_scales < 1 ? 1 : n_scales;
    sf.step     = step;

    // Template area capped at SCALE_TMPL_AREA with the aspect preserved
    // (DSST §6.1). d = tmpl_h*tmpl_w is the feature dimension and the DFT cost is
    // d*S^2, so this cap is what keeps the filter cheap.
    const double area = target_h * target_w;
    double f = 1.0;
    if (area > (double)SCALE_TMPL_AREA && area > 0.0)
        f = std::sqrt((double)SCALE_TMPL_AREA / area);
    sf.tmpl_h = (int)std::floor(target_h * f);
    sf.tmpl_w = (int)std::floor(target_w * f);
    if (sf.tmpl_h < 1) sf.tmpl_h = 1;
    if (sf.tmpl_w < 1) sf.tmpl_w = 1;

    sf.sigma = (float)((double)sf.n_scales / (double)sigma_factor);
    sf.G.assign((size_t)sf.n_scales, cfloat(0.0f, 0.0f));
    if (sf.enabled()) {
        // The 1-D scale target, via the SAME closed form the translation filter
        // uses: at rows = 1 the row factor is identically 1 (signed_freq(0,1) is
        // 0), so this degenerates exactly to the 1-D Gaussian spectrum.
        gaussian_target_spectrum(sf.G.data(), 1, sf.n_scales, sf.sigma, 0, 0);
    }
    sf.st = FilterState();
    sf.initialized = false;
}

void scale_extract(const ScaleFilter &sf, const uint8_t *frame,
                   int frame_rows, int frame_cols,
                   double row, double col, double box_h, double box_w,
                   cfloat *F_out)
{
    if (!sf.enabled()) return;

    const int d  = sf.dims();
    const int S  = sf.n_scales;
    const int half = (S - 1) / 2;

    std::vector<float> wr, wc;
    hann_into(wr, sf.tmpl_h);
    hann_into(wc, sf.tmpl_w);

    std::vector<double> buf((size_t)d);
    // The real sample, in the same [l*S + k] layout F_out uses — so the scale
    // axis is contiguous and the transform below reads it as a plain vector.
    // Held separately from F_out only because the transform cannot be done in
    // place; it replaces the per-dimension gather/scatter the complex path
    // needed, so it costs no traffic.
    std::vector<float> re_sample((size_t)d * (size_t)S);

    for (int n = -half; n <= half; ++n) {
        const double a  = std::pow((double)sf.step, (double)n);
        const double ch = box_h * a, cw = box_w * a;
        const double y0 = row - ch / 2.0, x0 = col - cw / 2.0;
        const double sy = ch / (double)sf.tmpl_h;
        const double sx = cw / (double)sf.tmpl_w;

        for (int r = 0; r < sf.tmpl_h; ++r) {
            const double yy = y0 + (double)r * sy;
            for (int c = 0; c < sf.tmpl_w; ++c) {
                const double xx = x0 + (double)c * sx;
                const double v  = sample_bilinear(frame, frame_rows, frame_cols, yy, xx);
                buf[(size_t)r * sf.tmpl_w + c] = v * (double)wr[(size_t)r]
                                                   * (double)wc[(size_t)c];
            }
        }

        // Zero-mean and unit-L2 for THIS level — see the note in the header for
        // why per level rather than jointly.
        double mean = 0.0;
        for (double v : buf) mean += v;
        mean /= (double)d;
        double nrm = 0.0;
        for (double &v : buf) { v -= mean; nrm += v * v; }
        nrm = std::sqrt(nrm);
        const double inv = nrm > 1e-12 ? 1.0 / nrm : 0.0;

        // Channel-major, matching FilterState's [ch][row*cols + col] layout with
        // rows = 1: element (l, n) lives at l*S + (n + half).
        const int k = n + half;
        for (int l = 0; l < d; ++l)
            re_sample[(size_t)l * S + k] = (float)(buf[(size_t)l] * inv);
    }

    // Transform each feature dimension along the SCALE axis. The twiddle table is
    // built ONCE for all d transforms — see the note on twiddle_table: doing it
    // per transform would put ~2M sin/cos calls in the per-frame path.
    //
    // REAL-INPUT transform: the sample is real by construction, so half the
    // multiplies and half the output bins were redundant — see
    // real_dft_with_table(). Bitwise identical to the complex path it replaces
    // on the bins it computes, and exactly conjugate on the mirrored ones.
    std::vector<cfloat> w;
    twiddle_table(w, S, false);
    for (int l = 0; l < d; ++l)
        real_dft_with_table(re_sample.data() + (size_t)l * S,
                            F_out + (size_t)l * S, S, w);
}

ScaleResult scale_detect(const ScaleFilter &sf, const cfloat *Z, float eps_rel)
{
    ScaleResult res;
    if (!sf.enabled() || !sf.initialized) return res;

    const int d = sf.dims(), S = sf.n_scales;

    double bmean = 0.0;
    for (int k = 0; k < S; ++k) bmean += (double)sf.st.B[(size_t)k];
    bmean /= (double)S;
    const double eps = (double)eps_rel * bmean;

    // Y[k] = SUM_l Z_l[k] * conj(H_l[k]), H = A/(B+eps). Same conjugation
    // convention as the translation path (the stored filter is H, not Bolme's
    // H*, and the correlation applies the conjugation) — one convention in the
    // design, because the header already records how silent a mix-up here is.
    std::vector<cfloat> Y((size_t)S), y((size_t)S);
    for (int k = 0; k < S; ++k) {
        const double den = (double)sf.st.B[(size_t)k] + eps;
        double re = 0.0, im = 0.0;
        for (int l = 0; l < d; ++l) {
            const cfloat z = Z[(size_t)l * S + k];
            const cfloat h = sf.st.A[(size_t)l * S + k] / (float)(den > 0.0 ? den : 1.0);
            // z * conj(h)
            re += (double)z.real() * h.real() + (double)z.imag() * h.imag();
            im += (double)z.imag() * h.real() - (double)z.real() * h.imag();
        }
        Y[(size_t)k] = cfloat((float)re, (float)im);
    }
    dft_1d(Y.data(), y.data(), S, true);

    int best = 0;
    double bestv = -1e300;
    for (int k = 0; k < S; ++k) {
        const double v = (double)y[(size_t)k].real();
        if (v > bestv) { bestv = v; best = k; }
    }
    const int half = (S - 1) / 2;
    // The scale axis WRAPS like any correlation output, so index S-1 is level -1.
    res.idx = (best <= half) ? best : best - S;

    // Confidence over the other levels. Not gated on — DSST does not gate on
    // scale, and compute_psr's 11-bin exclusion would remove a third of a 33-bin
    // map and report n_side = 22 as if it were the documented 121.
    double mu = 0.0, var = 0.0;
    int cnt = 0;
    for (int k = 0; k < S; ++k) {
        if (k == best) continue;
        mu += (double)y[(size_t)k].real();
        ++cnt;
    }
    if (cnt > 0) {
        mu /= (double)cnt;
        for (int k = 0; k < S; ++k) {
            if (k == best) continue;
            const double dv = (double)y[(size_t)k].real() - mu;
            var += dv * dv;
        }
        var = std::sqrt(var / (double)cnt);
    }
    res.peak   = bestv;
    res.psr    = var > 0.0 ? (bestv - mu) / var : 0.0;
    res.factor = std::pow((double)sf.step, (double)res.idx);
    res.valid  = true;
    return res;
}

void scale_update(ScaleFilter &sf, const cfloat *F, float eta)
{
    if (!sf.enabled()) return;
    if (!sf.initialized) {
        filter_init(sf.st, F, sf.G.data(), sf.dims(), 1, sf.n_scales);
        sf.initialized = true;
    } else {
        filter_update(sf.st, F, sf.G.data(), eta);
    }
}

void scale_update_shifted(ScaleFilter &sf, const cfloat *F, int idx, float eta)
{
    if (!sf.enabled()) return;

    // idx = 0 is the common case on hardware (174 of 199 frames in
    // runs/run_0820_1807.log) and the shift is then the identity, so take the
    // unshifted path rather than rebuilding an identical G.
    if (idx == 0) { scale_update(sf, F, eta); return; }

    std::vector<cfloat> Gs((size_t)sf.n_scales);
    gaussian_target_spectrum(Gs.data(), 1, sf.n_scales, sf.sigma, 0, idx);

    if (!sf.initialized) {
        filter_init(sf.st, F, Gs.data(), sf.dims(), 1, sf.n_scales);
        sf.initialized = true;
    } else {
        filter_update(sf.st, F, Gs.data(), eta);
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

void filter_update_quantize(FilterState &st, const cfloat *F_all,
                            const cfloat *G, float eta,
                            const double *energy, float eps_rel,
                            std::vector<cfloat> &h_scratch,
                            int16_t *out, float *out_scale, float *out_max_abs)
{
    const size_t n     = st.elems();
    const size_t n_all = n * (size_t)st.channels;
    const float  keep  = 1.0f - eta;

    // ---- B, exactly as filter_update() computes it -------------------------
    //
    // CHANNEL-MAJOR, which filter_update() is not. The published form reads
    // F_all[ch*n + i] with ch innermost, i.e. st.channels concurrent streams
    // 128 KB apart at ch16 — sixteen strided readers against a prefetcher that
    // tracks a handful. Accumulating into a scratch map instead makes F_all a
    // single sequential read and keeps the map (64 KB at 128x128) in cache.
    //
    // BIT-IDENTICAL, not merely equivalent: each element's channel sum is still
    // accumulated in ch order 0..channels-1, which is the only thing float
    // addition is sensitive to here. Nothing is reassociated.
    if (h_scratch.size() < n_all) h_scratch.resize(n_all);
    std::vector<float> esum(n, 0.0f);
    for (int ch = 0; ch < st.channels; ++ch) {
        const cfloat *f = F_all + (size_t)ch * n;
        for (size_t i = 0; i < n; ++i)
            esum[i] += f[i].real() * f[i].real() + f[i].imag() * f[i].imag();
    }
    for (size_t i = 0; i < n; ++i) st.B[i] = eta * esum[i] + keep * st.B[i];

    // ---- eps and the per-channel scale, exactly as filter_quantize_q15() ----
    double b_sum = 0.0;
    for (size_t i = 0; i < n; ++i) b_sum += st.B[i];
    const float eps = eps_rel * (float)(b_sum / (double)(n ? n : 1));

    std::vector<float> chscale((size_t)st.channels, 1.0f);
    if (energy) {
        for (int ch = 0; ch < st.channels; ++ch) {
            const double e = energy[ch];
            chscale[(size_t)ch] = (e > 0.0) ? (float)(1.0 / std::sqrt(e)) : 0.0f;
        }
    }

    // ---- A update AND the max-|H| scan, in ONE pass ------------------------
    //
    // THIS IS THE POINT OF THE FUSION. Unfused, A (2 MB at 128x128 ch16) is
    // streamed four times per frame: filter_update reads it and writes it, then
    // filter_quantize_q15 reads it for the max scan and again for the write-out,
    // recomputing the same divide both times. Here H is formed once, where A is
    // already in a register, and parked in h_scratch; the write-out is then a
    // multiply by a scalar. Half the divides (262144 -> 131072 at ch16) and one
    // fewer 2 MB read, at the price of one 2 MB scratch.
    //
    // The expression `a[i] * (cs / (st.B[i] + eps))` is copied verbatim from
    // filter_quantize_q15 and the final scaling is still `h * scale` applied to
    // that same intermediate — so the int16 output is BITWISE what the unfused
    // pair produces. run_fusion_tests() in test_mosse_filter.cpp asserts exactly
    // that, because "equivalent" here would silently end ten consecutive runs of
    // bit-identical tracking and make the next hardware comparison unreadable.
    float  max_norm = 0.0f;
    cfloat h_max(0.0f, 0.0f);
    for (int ch = 0; ch < st.channels; ++ch) {
        cfloat       *a  = st.A.data() + (size_t)ch * n;
        const cfloat *f  = F_all + (size_t)ch * n;
        cfloat       *hs = h_scratch.data() + (size_t)ch * n;
        const float   cs = chscale[(size_t)ch];

        // TWO LOOPS, NOT ONE, AND THAT IS THE POINT. Interleaving the two
        // statements in a single loop body is what a first cut does, and it is
        // measurably wrong: GCC contracts mul+add into FMA per expression, and
        // -ffp-contract=fast is its DEFAULT on aarch64, so the interleaved body
        // vectorises differently and contracts differently from the standalone
        // loop in filter_update(). run_fusion_tests() built with
        // -ffp-contract=fast catches it — `fused A bitwise identical` FAILS —
        // which is precisely why that check exists and is not run only at -O2.
        //
        // Split per channel instead. Each loop below is then byte-for-byte the
        // loop it replaces, so the codegen and therefore the last bit are the
        // same; and the re-read of `a` costs no DRAM traffic because a channel is
        // n * 8 = 128 KB at 128x128, written and re-read inside the A72's 1 MB
        // L2. The 2 MB whole-of-A round trip that filter_quantize_q15 paid is
        // what actually goes away.
        for (size_t i = 0; i < n; ++i)
            a[i] = eta * std::conj(G[i]) * f[i] + keep * a[i];

        for (size_t i = 0; i < n; ++i) {
            const cfloat h = a[i] * (cs / (st.B[i] + eps));
            hs[i] = h;
            const float m2 = h.real() * h.real() + h.imag() * h.imag();
            if (m2 > max_norm) { max_norm = m2; h_max = h; }
        }
    }
    st.initialized = true;

    const float max_abs = (max_norm > 0.0f) ? std::abs(h_max) : 0.0f;
    constexpr float Q15_FULL_SCALE = 32767.0f;
    const float scale = (max_abs > 0.0f) ? (Q15_FULL_SCALE / max_abs) : 0.0f;

    // ---- write-out: no divide, no second read of A -------------------------
    for (size_t i = 0; i < n_all; ++i) {
        const cfloat h = h_scratch[i] * scale;
        float re = std::nearbyint(h.real());
        float im = std::nearbyint(h.imag());
        // -32767, not -32768 — see filter_quantize_q15() for why cmul_accum
        // cannot be handed the full negative rail.
        if (re >  32767.0f) re =  32767.0f;
        if (re < -32767.0f) re = -32767.0f;
        if (im >  32767.0f) im =  32767.0f;
        if (im < -32767.0f) im = -32767.0f;
        out[2 * i]     = (int16_t)re;
        out[2 * i + 1] = (int16_t)im;
    }

    if (out_scale)   *out_scale   = scale;
    if (out_max_abs) *out_max_abs = max_abs;
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
    // SCAN ON THE SQUARED MAGNITUDE, then take ONE square root.
    //
    // std::abs() on a std::complex is hypot() — a libm call with its own scaling
    // and overflow handling — and this loop ran it channels*n = 262144 times per
    // frame. Measured 2026-08-20: `publish filter` was 12.16 ms/frame on the A72
    // (9.83 with -O3), and a native x86 benchmark put filter_quantize_q15 at
    // 9.63 ms of which -ffast-math removed 7.6 — i.e. essentially all of it was
    // the transcendental.
    //
    // std::norm() is re*re + im*im, monotone in std::abs(), so it selects the SAME
    // element; the hypot is then applied to that one element alone. This is exact
    // rather than an approximation — the only way it can differ is if two elements
    // are within a float ULP of each other, in which case max_abs is the same to
    // within an ULP either way. Deliberately NOT -ffast-math, which would buy the
    // same time by making every float operation in the file unsafe.
    float  max_norm = 0.0f;
    cfloat h_max(0.0f, 0.0f);
    for (int ch = 0; ch < st.channels; ++ch) {
        const cfloat *a = st.A.data() + (size_t)ch * n;
        const float   cs = chscale[(size_t)ch];
        for (size_t i = 0; i < n; ++i) {
            const cfloat h = a[i] * (cs / (st.B[i] + eps));
            const float  m2 = h.real() * h.real() + h.imag() * h.imag();
            if (m2 > max_norm) { max_norm = m2; h_max = h; }
        }
    }
    const float max_abs = (max_norm > 0.0f) ? std::abs(h_max) : 0.0f;

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

// -----------------------------------------------------------------------
// Scale-update gating — see the header for the hardware data behind it
// -----------------------------------------------------------------------
ScaleDecision scale_gate(const ScaleResult &sr, int n_scales,
                         double cur_h, double cur_w, double h0, double w0,
                         float conf_min, double min_rel, double max_rel,
                         int max_step)
{
    ScaleDecision d;
    d.conf      = sr.psr;
    d.threshold = conf_min;
    d.new_h     = cur_h * sr.factor;
    d.new_w     = cur_w * sr.factor;

    // Nothing to gate: the filter is off or has never been trained. Reported
    // separately from a veto because it is not a rejection of anything.
    if (!sr.valid) { d.reason = ScaleVeto::Invalid; return d; }

    // STRUCTURAL FIRST. An argmax on the boundary of the search range is wrong
    // by construction — the maximum it found is the edge of what was searched,
    // not a maximum of the underlying function. Reported ahead of LowConf
    // because it is the more specific finding when both fire, which is the usual
    // case (frame 13 of the 2026-08-20 run: idx +16 of +/-16, conf 1.57).
    //
    // Guarded on n_scales > 2 so a degenerate filter cannot veto every frame:
    // at n_scales <= 2 every index IS the rail.
    const int rail = (n_scales - 1) / 2;
    if (n_scales > 2 && rail > 0 && (sr.idx == rail || sr.idx == -rail)) {
        d.reason = ScaleVeto::AtSearchRail;
        return d;
    }

    // RATE limit on this frame's proposal, as distinct from the drift bound
    // below. See the MaxStep note in the header for the two datasets behind the
    // default and for what it costs when it is wrong. Disabled at max_step <= 0,
    // the same convention conf_min <= 0 uses.
    //
    // Deliberately checked before the confidence test: conf cannot distinguish a
    // wrong proposal from a big correct correction — that is a documented
    // property of the measurement, not a tuning failure — so a large jump has to
    // be judged on its magnitude rather than on how confident it looks. On
    // hardware the frame that inflated the box 1.42x carried conf 2.00 against a
    // 2.00 threshold, i.e. it passed the confidence test on the nose.
    if (max_step > 0 && (sr.idx > max_step || sr.idx < -max_step)) {
        d.reason = ScaleVeto::MaxStep;
        return d;
    }

    // The absolute backstop is checked before the confidence test so that a
    // proposal which is BOTH confident and out of bounds is reported as what it
    // is. It is a drift bound, not a per-frame plausibility test.
    const bool in_range = d.new_h >= h0 * min_rel && d.new_h <= h0 * max_rel &&
                          d.new_w >= w0 * min_rel && d.new_w <= w0 * max_rel;
    if (!in_range) { d.reason = ScaleVeto::OutOfRange; return d; }

    // conf_min <= 0 disables the THRESHOLD test only, exactly as PSR_GATE_MIN=0
    // does for the translation gate; the structural vetoes above still apply.
    if (conf_min <= 0.0f) { d.accept = true; d.reason = ScaleVeto::Disabled; return d; }

    if (sr.psr < (double)conf_min) { d.reason = ScaleVeto::LowConf; return d; }

    d.accept = true;
    d.reason = ScaleVeto::Accept;
    return d;
}

void coast_observe(CoastState &c, double dr_frame, double dc_frame)
{
    c.vr    = dr_frame;
    c.vc    = dc_frame;
    c.scale = 1.0;          // a fresh hold run starts at full velocity
}

bool coast_step(CoastState &c, double decay, double *dr_frame, double *dc_frame)
{
    if (c.scale <= 0.0 || (c.vr == 0.0 && c.vc == 0.0)) return false;
    *dr_frame = c.vr * c.scale;
    *dc_frame = c.vc * c.scale;
    // Decay AFTER applying, so the first held frame gets the full measured
    // velocity. That frame is the one the hold budget is usually spent on --
    // 30 of 62 sequences have a median budget of 4 frames or fewer.
    c.scale *= decay;
    if (c.scale < 1e-6) c.scale = 0.0;   // a coast that has faded is a freeze
    return true;
}

double coast_drift_bound(const CoastState &c, double decay)
{
    const double v = std::sqrt(c.vr * c.vr + c.vc * c.vc);
    if (decay >= 1.0) return std::numeric_limits<double>::infinity();
    if (decay <= 0.0) return v;
    return v / (1.0 - decay);
}

const char *scale_veto_tag(ScaleVeto v)
{
    switch (v) {
        case ScaleVeto::Accept:       return "ACCEPT";
        case ScaleVeto::Disabled:     return "ACCEPT(conf gate disabled)";
        case ScaleVeto::Invalid:      return "INVALID";
        case ScaleVeto::AtSearchRail: return "AT_SEARCH_RAIL";
        case ScaleVeto::MaxStep:      return "MAX_STEP";
        case ScaleVeto::LowConf:      return "LOW_CONF";
        case ScaleVeto::OutOfRange:   return "OUT_OF_RANGE";
    }
    return "?";
}

const char *scale_veto_why(ScaleVeto v)
{
    switch (v) {
        case ScaleVeto::Accept:
            return "scale confidence is at or above the threshold and the argmax "
                   "is interior — normal size tracking.";
        case ScaleVeto::Disabled:
            return "confidence threshold disabled (SCALE_CONF_MIN <= 0); the "
                   "structural vetoes still apply.";
        case ScaleVeto::Invalid:
            return "the scale filter is disabled or not yet trained, so there is "
                   "no estimate to accept or reject.";
        case ScaleVeto::AtSearchRail:
            return "the argmax sits ON the boundary of the search range, so it is "
                   "the edge of what was searched rather than a maximum. The size "
                   "envelope moves far slower than the filter steps, so the true "
                   "level can never legitimately be at the rail.";
        case ScaleVeto::MaxStep:
            return "the proposal moves the box by more levels in ONE frame than "
                   "SCALE_MAX_STEP allows. A rate limit, not a drift bound: a "
                   "rigid target's size cannot change this fast between frames, "
                   "and confidence cannot tell a wrong proposal from a big "
                   "correct one, so the magnitude is judged directly.";
        case ScaleVeto::LowConf:
            return "scale confidence is below the threshold — the size response "
                   "is not peaked enough to act on. This is the occlusion / "
                   "deformation indicator for the size axis.";
        case ScaleVeto::OutOfRange:
            return "the proposed box leaves the absolute bounds relative to the "
                   "initial size. This is a drift backstop, so hitting it means "
                   "earlier frames were already wrong.";
    }
    return "?";
}

}  // namespace mosse
