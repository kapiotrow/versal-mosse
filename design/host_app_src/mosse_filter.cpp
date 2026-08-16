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
    for (int j = 0; j < n; ++j) {
        const double a = sgn * 2.0 * M_PI * (double)j / (double)n;
        w[(size_t)j] = cfloat((float)std::cos(a), (float)std::sin(a));
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
            F_out[(size_t)l * S + k] = cfloat((float)(buf[(size_t)l] * inv), 0.0f);
    }

    // Transform each feature dimension along the SCALE axis. The twiddle table is
    // built ONCE for all d transforms — see the note on twiddle_table: doing it
    // per transform would put ~2M sin/cos calls in the per-frame path.
    std::vector<cfloat> w, tmp((size_t)S), out((size_t)S);
    twiddle_table(w, S, false);
    for (int l = 0; l < d; ++l) {
        for (int k = 0; k < S; ++k) tmp[(size_t)k] = F_out[(size_t)l * S + k];
        dft_with_table(tmp.data(), out.data(), S, w, false);
        for (int k = 0; k < S; ++k) F_out[(size_t)l * S + k] = out[(size_t)k];
    }
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
