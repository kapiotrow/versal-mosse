/*
 * test_mosse_filter.cpp
 * Native unit test for mosse_filter.{h,cpp} against the NumPy golden produced by
 * scripts/gen_filter_golden.py.
 *
 * Run via `make test_host`, which regenerates the golden first so the reference
 * and the implementation cannot drift apart.
 *
 * What this is for: a wrong filter does not crash and does not look wrong. It
 * produces a plausible response map with a peak in the wrong place, which on this
 * project costs a ~90 min hw_emu frame to discover. Conjugation, the shared
 * denominator, the Q1.15 global scale and the Stage B3 folding are all checked
 * here in under a second.
 */

#include "mosse_filter.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using mosse::cfloat;

namespace {

int g_failures = 0;

std::string g_dir;

std::string path(const char *name)
{
    return g_dir + "/" + name;
}

std::vector<uint8_t> read_all(const std::string &p)
{
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot open %s\n", p.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)n);
    if (fread(buf.data(), 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "FATAL: short read on %s\n", p.c_str());
        exit(2);
    }
    fclose(f);
    return buf;
}

std::vector<cfloat> read_cfloat(const char *name, size_t expect)
{
    std::vector<uint8_t> raw = read_all(path(name));
    if (raw.size() != expect * 2 * sizeof(float)) {
        fprintf(stderr, "FATAL: %s has %zu bytes, expected %zu\n",
                name, raw.size(), expect * 2 * sizeof(float));
        exit(2);
    }
    std::vector<cfloat> out(expect);
    const float *p = reinterpret_cast<const float *>(raw.data());
    for (size_t i = 0; i < expect; ++i) out[i] = cfloat(p[2 * i], p[2 * i + 1]);
    return out;
}

template <typename T>
std::vector<T> read_pod(const char *name, size_t expect)
{
    std::vector<uint8_t> raw = read_all(path(name));
    if (raw.size() != expect * sizeof(T)) {
        fprintf(stderr, "FATAL: %s has %zu bytes, expected %zu\n",
                name, raw.size(), expect * sizeof(T));
        exit(2);
    }
    std::vector<T> out(expect);
    memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// Relative error against the golden's own magnitude scale, not element-wise:
// bins whose golden value is near zero would otherwise dominate the metric with
// meaningless ratios.
void check_cfloat(const char *what, const std::vector<cfloat> &got,
                  const std::vector<cfloat> &exp, float tol)
{
    if (got.size() != exp.size()) {
        printf("  %-10s FAIL — size %zu vs %zu\n", what, got.size(), exp.size());
        ++g_failures;
        return;
    }
    double scale = 0.0, worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < exp.size(); ++i) scale = std::max(scale, (double)std::abs(exp[i]));
    if (scale == 0.0) scale = 1.0;
    for (size_t i = 0; i < exp.size(); ++i) {
        const double e = std::abs(got[i] - exp[i]) / scale;
        if (e > worst) { worst = e; worst_i = i; }
    }
    const bool ok = worst <= tol;
    printf("  %-10s %s — max rel err %.3e (tol %.1e) at [%zu]\n",
           what, ok ? "OK  " : "FAIL", worst, (double)tol, worst_i);
    if (!ok) {
        printf("               got {%g,%g}  expected {%g,%g}\n",
               got[worst_i].real(), got[worst_i].imag(),
               exp[worst_i].real(), exp[worst_i].imag());
        ++g_failures;
    }
}

void check_float(const char *what, const std::vector<float> &got,
                 const std::vector<float> &exp, float tol)
{
    if (got.size() != exp.size()) {
        printf("  %-10s FAIL — size %zu vs %zu\n", what, got.size(), exp.size());
        ++g_failures;
        return;
    }
    double scale = 0.0, worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < exp.size(); ++i) scale = std::max(scale, (double)std::fabs(exp[i]));
    if (scale == 0.0) scale = 1.0;
    for (size_t i = 0; i < exp.size(); ++i) {
        const double e = std::fabs((double)got[i] - (double)exp[i]) / scale;
        if (e > worst) { worst = e; worst_i = i; }
    }
    const bool ok = worst <= tol;
    printf("  %-10s %s — max rel err %.3e (tol %.1e) at [%zu]\n",
           what, ok ? "OK  " : "FAIL", worst, (double)tol, worst_i);
    if (!ok) {
        printf("               got %g  expected %g\n", got[worst_i], exp[worst_i]);
        ++g_failures;
    }
}

// The quantized filter is the thing the hardware actually consumes, so it is
// checked in absolute LSB, not relatively. Both sides round the same way; 1 LSB
// covers float32-vs-float64 accumulation order only.
void check_q15(const std::vector<int16_t> &got, const std::vector<int16_t> &exp)
{
    if (got.size() != exp.size()) {
        printf("  %-10s FAIL — size %zu vs %zu\n", "H_q15", got.size(), exp.size());
        ++g_failures;
        return;
    }
    int worst = 0;
    size_t worst_i = 0, n_off = 0;
    for (size_t i = 0; i < exp.size(); ++i) {
        const int d = std::abs((int)got[i] - (int)exp[i]);
        if (d) ++n_off;
        if (d > worst) { worst = d; worst_i = i; }
    }
    const bool ok = worst <= 1;
    printf("  %-10s %s — max %d LSB, %zu/%zu differ (tol 1 LSB) at [%zu]\n",
           "H_q15", ok ? "OK  " : "FAIL", worst, n_off, exp.size(), worst_i);
    if (!ok) {
        printf("               got %d  expected %d\n",
               (int)got[worst_i], (int)exp[worst_i]);
        ++g_failures;
    }
}

// -----------------------------------------------------------------------
// Scalar helpers — the PSR/gate assertions are predicates, not vector diffs,
// so the three helpers above do not fit. Same contract as they have: print one
// line, bump g_failures, never abort.
// -----------------------------------------------------------------------
void check_true(const char *what, bool cond, const char *detail)
{
    printf("  %-26s %s%s%s\n", what, cond ? "OK  " : "FAIL",
           detail && *detail ? " — " : "", detail ? detail : "");
    if (!cond) ++g_failures;
}

void check_int(const char *what, long got, long exp)
{
    const bool ok = (got == exp);
    printf("  %-26s %s — got %ld, expected %ld\n", what, ok ? "OK  " : "FAIL",
           got, exp);
    if (!ok) ++g_failures;
}

void check_double(const char *what, double got, double exp, double tol)
{
    const double scale = std::fabs(exp) > 1e-12 ? std::fabs(exp) : 1.0;
    const double err   = std::fabs(got - exp) / scale;
    const bool   ok    = err <= tol;
    printf("  %-26s %s — got %.6g, expected %.6g (rel err %.2e, tol %.1e)\n",
           what, ok ? "OK  " : "FAIL", got, exp, err, tol);
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------
// PSR / gate test fixtures
// -----------------------------------------------------------------------
// Built ANALYTICALLY in C++ rather than loaded from a NumPy golden, on purpose.
//
// A golden would require gen_filter_golden.py to synthesize a spatial-domain
// cint16 response map and re-implement Bolme's 11x11 CIRCULAR exclusion, the
// two-pass variance and the wrap arithmetic in NumPy — i.e. a second
// implementation of the exact logic under test, in another language, to be kept
// in sync forever. This project's own record says that is the failure mode and
// not the safeguard (the aiesim generator drifting from the shift budget; s7's
// PSR check asserting 0.7x of a golden it did not resemble).
//
// NumPy earns its keep for filter_update / gaussian_target_spectrum because it is
// a genuinely INDEPENDENT derivation of non-obvious maths. PSR is not that: every
// property worth asserting has a closed form, and several are exact (a constant
// map has sigma = 0, exactly). So: no new golden files, no new params.txt keys,
// no changes to gen_filter_golden.py.
constexpr int PR = 32, PC = 32;      // geometry is free here — no golden to match

using Resp = std::vector<int16_t>;   // interleaved cint16 {re,im}

Resp resp_zeros() { return Resp((size_t)PR * PC * 2, 0); }

void resp_set(Resp &m, int r, int c, int v)
{
    r = ((r % PR) + PR) % PR;
    c = ((c % PC) + PC) % PC;
    m[(size_t)2 * (r * PC + c)] = (int16_t)v;
}

int resp_get(const Resp &m, int r, int c)
{
    r = ((r % PR) + PR) % PR;
    c = ((c % PC) + PC) % PC;
    return m[(size_t)2 * (r * PC + c)];
}

// Deterministic LCG, not rand(): rand()'s sequence is libc-dependent, so a
// golden-free test that used it would not be reproducible across machines.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    int next(int amp) {   // uniform-ish in [-amp, amp]
        s = s * 1103515245u + 12345u;
        return (int)((s >> 16) % (uint32_t)(2 * amp + 1)) - amp;
    }
};

// A sigma=2 Gaussian of the given amplitude at (pr,pc), on an LCG noise floor.
// Circular distance, so the peak may sit anywhere including a corner.
Resp resp_gauss(int pr, int pc, int amp, int noise, uint32_t seed = 20260815u)
{
    Resp m = resp_zeros();
    Lcg  g(seed);
    for (int r = 0; r < PR; ++r)
        for (int c = 0; c < PC; ++c) {
            int dr = r - pr; if (dr >  PR / 2) dr -= PR; if (dr < -PR / 2) dr += PR;
            int dc = c - pc; if (dc >  PC / 2) dc -= PC; if (dc < -PC / 2) dc += PC;
            const double d2 = (double)(dr * dr + dc * dc);
            const int    v  = (int)std::lround((double)amp * std::exp(-d2 / 8.0));
            resp_set(m, r, c, v + (noise ? g.next(noise) : 0));
        }
    return m;
}

void run_psr_tests()
{
    using mosse::compute_psr;
    using mosse::psr_gate;
    using mosse::GateReason;

    printf("\n--- PSR / gate (Bolme §3.5) ---\n");
    printf("  build threshold DEFAULT_PSR_MIN = %.2f\n",
           (double)mosse::DEFAULT_PSR_MIN);

    const float THR = 7.0f;   // passed explicitly, so `make test_host
                              // PSR_GATE_MIN=20` still passes: these tests
                              // validate the MECHANISM, not the build's choice.

    // 1. Identically zero: the degenerate case that used to be indistinguishable
    //    from a correct centred answer.
    {
        const Resp m = resp_zeros();
        const auto p = compute_psr(m.data(), PR, PC, true);
        const auto g = psr_gate(p, THR);
        check_int   ("zero: peak",           p.peak, 0);
        check_double("zero: sdev",           p.sdev, 0.0, 1e-12);
        check_true  ("zero: gate holds",     !g.accept, "");
        check_true  ("zero: reason",         g.reason == GateReason::ZeroResponse,
                     "ZeroResponse, not LowPsr — the pipeline produced nothing");
    }

    // 2. Constant map: non-zero peak but zero variance. Exercises the div-by-zero
    //    guard, and must NOT be reported as an occlusion.
    {
        Resp m = resp_zeros();
        for (int r = 0; r < PR; ++r)
            for (int c = 0; c < PC; ++c) resp_set(m, r, c, 100);
        const auto p = compute_psr(m.data(), PR, PC, true);
        const auto g = psr_gate(p, THR);
        check_int   ("flat: peak",           p.peak, 100);
        check_double("flat: mean",           p.mean, 100.0, 1e-12);
        check_double("flat: sdev",           p.sdev, 0.0,   1e-12);
        check_double("flat: psr",            p.psr,  0.0,   1e-12);
        check_true  ("flat: reason",         g.reason == GateReason::FlatSidelobe,
                     "constant map: PSR undefined, not infinite");
    }

    // 3. Clean detection at the tracker's own test offset.
    {
        const Resp m = resp_gauss(10, -7, 4000, 8);
        const auto p = compute_psr(m.data(), PR, PC, true);
        const auto g = psr_gate(p, THR);
        check_int ("clean: dr",              p.dr, 10);
        check_int ("clean: dc",              p.dc, -7);
        check_int ("clean: n_side",          p.n_side, (long)PR * PC - 121);
        check_true("clean: psr > threshold", p.psr > (double)THR, "");
        check_true("clean: gate accepts",    g.accept, "");
        check_true("clean: reason",          g.reason == GateReason::Accept, "");
        printf("    (clean psr = %.2f, ratio = %.2fx)\n", p.psr, p.ratio);
    }

    // 4. CIRCULAR-EXCLUSION INVARIANCE — the highest-value case here.
    //    The same peak in the middle and wrapped into the corner must score
    //    identically. A linear (non-circular) exclusion window leaves half the
    //    mainlobe in the sidelobe statistics and fails this by a wide margin.
    {
        const Resp a = resp_gauss(16, 16, 4000, 0);
        const Resp b = resp_gauss( 0,  0, 4000, 0);
        const auto pa = compute_psr(a.data(), PR, PC, true);
        const auto pb = compute_psr(b.data(), PR, PC, true);
        check_int   ("wrap: centred n_side",  pa.n_side, (long)PR * PC - 121);
        check_int   ("wrap: corner n_side",   pb.n_side, (long)PR * PC - 121);
        check_int   ("wrap: corner dr",       pb.dr, 0);
        check_int   ("wrap: corner dc",       pb.dc, 0);
        check_double("wrap: psr invariant",   pb.psr, pa.psr, 1e-9);
    }

    // 5. Occlusion: noise only, no target. Must score where Bolme says.
    {
        const Resp m = resp_gauss(0, 0, 0, 800);
        const auto p = compute_psr(m.data(), PR, PC, true);
        const auto g = psr_gate(p, THR);
        check_true("noise: psr < threshold", p.psr < (double)THR, "");
        check_true("noise: gate holds",      !g.accept, "");
        check_true("noise: reason",          g.reason == GateReason::LowPsr,
                   "this is the occlusion path the hw_emu test exercises");
        printf("    (noise psr = %.2f)\n", p.psr);
    }

    // 6. Monotonicity: PSR must rise with peak amplitude on a fixed noise field,
    //    and acceptance must flip exactly once. A single-point assertion passes
    //    happily with a sign error or a mean/sigma mix-up; this does not.
    {
        const int amps[] = {50, 100, 200, 400, 800, 1600, 3200};
        double prev = -1e300;
        bool   mono = true, accept_prev = false, one_flip = true;
        int    flips = 0;
        for (int i = 0; i < 7; ++i) {
            const Resp m = resp_gauss(3, -5, amps[i], 300);
            const auto p = compute_psr(m.data(), PR, PC, true);
            const auto g = psr_gate(p, THR);
            if (p.psr <= prev) mono = false;
            prev = p.psr;
            if (i && g.accept != accept_prev) ++flips;
            if (i && !g.accept && accept_prev) one_flip = false;  // never un-accepts
            accept_prev = g.accept;
            // Threshold consistency, at every sample.
            if (g.accept != (p.psr >= (double)THR)) one_flip = false;
        }
        check_true("sweep: psr monotone",     mono, "psr rises with peak amplitude");
        check_true("sweep: one accept flip",  flips <= 1 && one_flip,
                   "acceptance is monotone in psr and matches psr >= threshold");
    }

    // 7. Threshold plumbing: 0 disables the PSR test but NOT the structural ones.
    {
        const Resp m = resp_gauss(0, 0, 0, 800);      // the noise case from 5
        const auto p = compute_psr(m.data(), PR, PC, true);
        check_true("thr=0: accepts noise",   psr_gate(p, 0.0f).accept, "");
        check_true("thr=0: reason Disabled",
                   psr_gate(p, 0.0f).reason == GateReason::Disabled, "");
        check_true("thr=1e9: rejects clean",
                   !psr_gate(compute_psr(resp_gauss(10, -7, 4000, 8).data(),
                                         PR, PC, true), 1e9f).accept, "");
        const Resp z = resp_zeros();
        const auto pz = compute_psr(z.data(), PR, PC, true);
        check_true("thr=0: zero still holds", !psr_gate(pz, 0.0f).accept,
                   "structural failures veto even with the threshold off");
    }

    // 8. Negative peak = anti-correlation. G is a positive Gaussian, so the
    //    strongest structure being negative is not a detection.
    {
        Resp m = resp_gauss(10, -7, 4000, 8);
        for (size_t i = 0; i < (size_t)PR * PC; ++i)
            m[2 * i] = (int16_t)(-m[2 * i]);
        const auto a = compute_psr(m.data(), PR, PC, true);   // by magnitude
        const auto s = compute_psr(m.data(), PR, PC, false);  // signed max
        const auto g = psr_gate(a, THR);
        check_true("neg: located by |re|",   a.dr == 10 && a.dc == -7, "");
        check_true("neg: peak is negative",  a.peak < 0, "");
        check_true("neg: reason",            g.reason == GateReason::NegativePeak,
                   "not LowPsr — the sign is the fault, not the strength");
        check_true("neg: signed max differs",
                   s.dr != a.dr || s.dc != a.dc,
                   "the two peak definitions disagree, which is the point of both");
    }

    // 9. Wrap arithmetic on the reported displacement.
    {
        Resp m = resp_zeros();
        resp_set(m, 31, 1, 9000);
        const auto p = compute_psr(m.data(), PR, PC, true);
        check_int("wrapidx: dr", p.dr, -1);
        check_int("wrapidx: dc", p.dc,  1);
    }

    // 10. Interleaving: a large IMAGINARY value must not move the peak. Guards
    //     the resp[i] vs resp[2*i] bug now that peak_detect_sw's comment is gone.
    {
        Resp m = resp_gauss(10, -7, 4000, 8);
        m[2 * (5 * PC + 5) + 1] = 32000;          // imaginary part only
        const auto p = compute_psr(m.data(), PR, PC, true);
        check_true("interleave: peak unmoved", p.dr == 10 && p.dc == -7,
                   "imag part must not participate");
        (void)resp_get;
    }
}

}  // namespace

int main(int argc, char **argv)
{
    g_dir = (argc > 1) ? argv[1] : "golden";

    // params.txt uses the same `key value` format as the aiesim scenarios.
    int   rows = 0, cols = 0, channels = 0, h_shift = 0, dr = 0, dc = 0;
    float sigma = 0.0f, eta = 0.0f, eps_rel = 0.0f;
    {
        FILE *f = fopen(path("params.txt").c_str(), "r");
        if (!f) {
            fprintf(stderr, "FATAL: cannot open %s — run `make test_host`, which "
                            "regenerates the golden\n", path("params.txt").c_str());
            return 2;
        }
        char key[64];
        double val;
        while (fscanf(f, "%63s %lf", key, &val) == 2) {
            if      (!strcmp(key, "rows"))     rows     = (int)val;
            else if (!strcmp(key, "cols"))     cols     = (int)val;
            else if (!strcmp(key, "channels")) channels = (int)val;
            else if (!strcmp(key, "h_shift"))  h_shift  = (int)val;
            else if (!strcmp(key, "sigma"))    sigma    = (float)val;
            else if (!strcmp(key, "dr"))       dr       = (int)val;
            else if (!strcmp(key, "dc"))       dc       = (int)val;
            else if (!strcmp(key, "eta"))      eta      = (float)val;
            else if (!strcmp(key, "eps_rel"))  eps_rel  = (float)val;
        }
        fclose(f);
    }

    printf("mosse_filter native test — %dx%d x %d ch, target (%d,%d), eta %.3f\n",
           rows, cols, channels, dr, dc, eta);

    // A mismatch here would make every subsequent comparison meaningless while
    // still producing plausible numbers — exactly the failure this project has
    // been bitten by before with the shift budget.
    if (h_shift != CMUL_H_SHIFT) {
        printf("  FATAL: golden built for H_SHIFT=%d but this binary has "
               "CMUL_H_SHIFT=%d\n", h_shift, CMUL_H_SHIFT);
        return 2;
    }

    const size_t n     = (size_t)rows * cols;
    const size_t n_all = n * (size_t)channels;

    std::vector<cfloat> F0 = read_cfloat("F0.bin", n_all);
    std::vector<cfloat> F1 = read_cfloat("F1.bin", n_all);
    std::vector<cfloat> Gg = read_cfloat("G.bin",  n);
    std::vector<double> energy = read_pod<double>("energy.bin", (size_t)channels);

    // --- gaussian_target_spectrum -------------------------------------------
    std::vector<cfloat> G(n);
    mosse::gaussian_target_spectrum(G.data(), rows, cols, sigma, dr, dc);
    check_cfloat("G", G, Gg, 1e-5f);

    // --- filter_init ---------------------------------------------------------
    mosse::FilterState st;
    mosse::filter_init(st, F0.data(), G.data(), channels, rows, cols);
    check_cfloat("A_init", st.A, read_cfloat("A_init.bin", n_all), 1e-5f);
    check_float ("B_init", st.B, read_pod<float>("B_init.bin", n), 1e-5f);

    // --- filter_update -------------------------------------------------------
    mosse::filter_update(st, F1.data(), G.data(), eta);
    check_cfloat("A_upd", st.A, read_cfloat("A_upd.bin", n_all), 1e-5f);
    check_float ("B_upd", st.B, read_pod<float>("B_upd.bin", n), 1e-5f);

    // --- filter_quantize_q15 -------------------------------------------------
    std::vector<int16_t> H(n_all * 2);
    float scale = 0.0f, max_abs = 0.0f;
    mosse::filter_quantize_q15(st, energy.data(), eps_rel, H.data(), &scale, &max_abs);
    check_q15(H, read_pod<int16_t>("H_q15.bin", n_all * 2));
    printf("  %-10s scale %.6g, max|H| %.6g\n", "quantize", (double)scale, (double)max_abs);

    // The quantized filter must actually reach full scale, or the response comes
    // back crushed against the noise floor with no other symptom.
    //
    // The invariant is on the complex MAGNITUDE, not on either component.
    // Normalising by |H| is what guarantees neither component can overflow, but it
    // also means a bin at 45 degrees of phase puts only 32767/sqrt(2) = 23170 in
    // each of re and im. Asserting a component reaches 32767 would fail on any
    // filter whose strongest bin is not purely real — i.e. on essentially every
    // real filter.
    double peak_mag = 0.0;
    int    peak_cmp = 0;
    for (size_t i = 0; i < n_all; ++i) {
        const double re = H[2 * i], im = H[2 * i + 1];
        peak_mag = std::max(peak_mag, std::sqrt(re * re + im * im));
        peak_cmp = std::max(peak_cmp, std::max((int)std::abs(re), (int)std::abs(im)));
    }
    // Full int16 scale, NOT (1<<CMUL_H_SHIFT)-1. H always uses all 15 bits; the
    // shift only scales the product. Deriving this from CMUL_H_SHIFT is the exact
    // bug that was found in filter_quantize_q15() — do not reintroduce it here.
    const double q15_one = 32767.0;
    // 1.5 LSB covers the independent rounding of re and im.
    const bool full_ok = peak_mag >= q15_one - 1.5;
    printf("  %-10s %s — peak |H| %.1f of %.0f (largest component %d)\n",
           "fullscale", full_ok ? "OK  " : "FAIL", peak_mag, q15_one, peak_cmp);
    if (!full_ok) ++g_failures;

    // No component may exceed full scale either — that would clip in cmul_accum.
    if (peak_cmp > (int)q15_one) {
        printf("  %-10s FAIL — component %d exceeds Q1.15 full scale %.0f\n",
               "noclip", peak_cmp, q15_one);
        ++g_failures;
    }

    // Placed LAST so a missing or short golden still fails fast at exit(2) above
    // — these tests need no golden at all and would otherwise mask that.
    run_psr_tests();

    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
