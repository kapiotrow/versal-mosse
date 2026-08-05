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

    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
