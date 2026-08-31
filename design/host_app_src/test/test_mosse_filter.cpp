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
 *
 * @thesis subsec:weryfikacja | A-10,B-07 | The native unit suite, run TWICE -- the second
 *   time with -ffp-contract=fast, because the board's compiler contracts mul+add by default and
 *   that build caught bugs -O2 missed.
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
// maybe_unused: at FILTER_MASK=1 the only caller is #if'd out (the golden is the
// unmasked filter), and this file is built -Wall -Wextra.
[[maybe_unused]] void check_q15(const std::vector<int16_t> &got,
                                const std::vector<int16_t> &exp)
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

// ---------------------------------------------------------------------------
// Bounding box, ROI geometry and the patch<->frame conversions.
//
// Analytic, no golden — same rule as the PSR suite above: every property here has
// a closed form, so a NumPy reference would be a second implementation of trivial
// arithmetic to keep in sync forever.
//
// These exist because the conversions are INVISIBLE today. While roi_h ==
// patch_rows a patch bin IS a frame pixel, so `pos_row += dr` and
// `dr == IMPULSE_DR` are both accidentally correct. Padding breaks both, and the
// symptom is a tracker that localises confidently and drifts — which `err=0 px`
// cannot see, and which this project has shipped four times.
// ---------------------------------------------------------------------------
void run_box_tests()
{
    using namespace mosse;
    printf("\nbounding box / ROI geometry (analytic)\n");

    // 1. The shipped geometry must survive adopting the box. Target 64 at padding
    //    2 gives roi 128 on a 128 patch — a 1:1 resample, i.e. exactly what runs
    //    today. That is what makes the first hardware step single-variable.
    {
        const TargetBox b{540.0, 960.0, 64.0, 64.0};
        const RoiGeometry g = roi_for(b, 2.0f, 128, 128);
        check_int("roi_h at padding 2", g.roi_h, 128);
        check_int("roi_w at padding 2", g.roi_w, 128);
        check_int("roi_row centred", g.roi_row, 540 - 64);
        check_int("roi_col centred", g.roi_col, 960 - 64);
        check_true("padding 2 keeps 1:1", g.roi_h == g.patch_rows,
                   "so the interpolator stays dormant for this step");
    }

    // 2. IDENTITY at 1:1 — the property that hides the bug. If this were the only
    //    test, both conversions could be `return dr;` and pass.
    {
        const TargetBox b{540.0, 960.0, 64.0, 64.0};
        const RoiGeometry g = roi_for(b, 2.0f, 128, 128);
        check_double("1:1 dr identity", patch_dr_to_frame(10, g), 10.0, 1e-12);
        check_double("1:1 dc identity", patch_dc_to_frame(-7, g), -7.0, 1e-12);
    }

    // 3. NON-identity: the case that actually discriminates. Target 64 at padding
    //    3 gives roi 192, so one patch bin is 1.5 frame px.
    {
        const TargetBox b{540.0, 960.0, 64.0, 64.0};
        const RoiGeometry g = roi_for(b, 3.0f, 128, 128);
        check_int("roi_h at padding 3", g.roi_h, 192);
        check_double("px per bin", patch_dr_to_frame(1, g), 1.5, 1e-12);
        check_double("dr 10 bins -> frame", patch_dr_to_frame(10, g), 15.0, 1e-12);
        check_double("dc -7 bins -> frame", patch_dc_to_frame(-7, g), -10.5, 1e-12);
        // And the inverse, which is what the pass/fail assertion needs.
        check_int("frame 15 -> 10 bins", frame_dr_to_patch(15.0, g), 10);
        check_int("frame 10 -> 7 bins", frame_dr_to_patch(10.0, g), 7);
    }

    // 4. Round-trip over a range of paddings, including a non-integer ratio and
    //    an anisotropic box. Frame -> patch -> frame must return the original to
    //    within one bin, which is the quantisation the tracker genuinely has.
    {
        const double pads[] = {1.5, 2.0, 2.5, 3.0};
        bool ok = true;
        double worst = 0.0;
        for (double p : pads) {
            const TargetBox b{540.0, 960.0, 64.0, 48.0};
            const RoiGeometry g = roi_for(b, (float)p, 128, 128);
            for (int d = -20; d <= 20; ++d) {
                const double back = patch_dr_to_frame(frame_dr_to_patch((double)d, g), g);
                const double err  = std::fabs(back - (double)d);
                if (err > worst) worst = err;
                if (err > patch_dr_to_frame(1, g) * 0.5 + 1e-9) ok = false;
            }
        }
        char det[80];
        snprintf(det, sizeof(det), "worst %.4f frame px, <= half a bin", worst);
        check_true("frame->patch->frame round-trip", ok, det);
    }

    // 5. Anisotropic box: rows and columns must convert INDEPENDENTLY. A single
    //    shared ratio would pass every square-target test and fail here.
    {
        const TargetBox b{540.0, 960.0, 64.0, 32.0};
        const RoiGeometry g = roi_for(b, 2.0f, 128, 128);
        check_int("aniso roi_h", g.roi_h, 128);
        check_int("aniso roi_w", g.roi_w, 64);
        check_double("aniso dr ratio", patch_dr_to_frame(4, g), 4.0, 1e-12);
        check_double("aniso dc ratio", patch_dc_to_frame(4, g), 2.0, 1e-12);
        check_true("aniso ratios differ",
                   patch_dr_to_frame(4, g) != patch_dc_to_frame(4, g),
                   "a shared ratio would pass every square test");
    }

    // 6. Target size in patch pixels, the units sigma is expressed in. The target
    //    always occupies patch/padding regardless of its size in frame pixels —
    //    which is why the DSST rule gives sigma = patch/(16*padding).
    {
        for (double p : {1.5, 2.0, 3.0}) {
            const TargetBox b{540.0, 960.0, 64.0, 64.0};
            const RoiGeometry g = roi_for(b, (float)p, 128, 128);
            check_double("target in patch px", target_h_in_patch(b, g),
                         128.0 / p, 1e-9);
        }
    }

    // 7. sigma_for honours SIGMA_FROM_TARGET. Default is 0, i.e. DEFAULT_SIGMA
    //    literally — see the long note in mosse_filter.h for why the sweep does
    //    not support switching to the rule.
    {
        const TargetBox b{540.0, 960.0, 64.0, 64.0};
        const RoiGeometry g = roi_for(b, 2.0f, 128, 128);
        float sr = 0.0f, sc = 0.0f;
        sigma_for(b, g, &sr, &sc);
#if SIGMA_FROM_TARGET
        check_double("sigma from target (rule on)", sr, 4.0, 1e-6);
#else
        check_double("sigma = DEFAULT_SIGMA", sr, (double)DEFAULT_SIGMA, 1e-6);
#endif
        check_double("sigma isotropic for square box", sr, sc, 1e-9);
    }

    // 8. Anisotropic sigma reaches gaussian_target_spectrum. DSST §6.1 says "the
    //    target size in the translation dimensionS" — the plural is the point.
    {
        constexpr int N = 32;
        std::vector<cfloat> g1(N * N), g2(N * N);
        gaussian_target_spectrum(g1.data(), N, N, 2.0f, 4.0f, 0, 0);
        gaussian_target_spectrum(g2.data(), N, N, 4.0f, 2.0f, 0, 0);
        // Swapping the two sigmas must transpose the spectrum, not leave it alone.
        bool transposed = true, differs = false;
        for (int u = 0; u < N && transposed; ++u)
            for (int v = 0; v < N; ++v) {
                if (std::abs(g1[u * N + v] - g2[v * N + u]) > 1e-6f) transposed = false;
                if (std::abs(g1[u * N + v] - g2[u * N + v]) > 1e-6f) differs = true;
            }
        check_true("aniso sigma transposes", transposed, "");
        check_true("aniso sigma is not a no-op", differs, "");
        // And the scalar overload must equal the equal-sigma anisotropic call,
        // which is what keeps the NumPy golden valid.
        std::vector<cfloat> s1(N * N), s2(N * N);
        gaussian_target_spectrum(s1.data(), N, N, 2.5f, 3, -5);
        gaussian_target_spectrum(s2.data(), N, N, 2.5f, 2.5f, 3, -5);
        bool same = true;
        for (int i = 0; i < N * N; ++i)
            if (std::abs(s1[i] - s2[i]) > 0.0f) same = false;
        check_true("scalar overload forwards exactly", same,
                   "the golden depends on this");
    }

    // 9. IoU. Closed-form cases, including the one that catches an
    //    intersection/union swap (which agrees at overlap 1.0 and nowhere else).
    {
        const TargetBox a{100.0, 100.0, 40.0, 40.0};
        check_double("IoU self", box_iou(a, a), 1.0, 1e-12);
        const TargetBox disjoint{300.0, 300.0, 40.0, 40.0};
        check_double("IoU disjoint", box_iou(a, disjoint), 0.0, 1e-12);
        // Half-overlap along one axis: intersection 20x40, union 2*1600-800.
        const TargetBox half{120.0, 100.0, 40.0, 40.0};
        check_double("IoU half overlap", box_iou(a, half), 800.0 / 2400.0, 1e-12);
        // Concentric, one twice the side: inter 1600, union 6400.
        const TargetBox big{100.0, 100.0, 80.0, 80.0};
        check_double("IoU nested", box_iou(a, big), 1600.0 / 6400.0, 1e-12);
        check_true("IoU symmetric",
                   std::fabs(box_iou(a, half) - box_iou(half, a)) < 1e-15, "");
    }
}

// ---------------------------------------------------------------------------
// DSST 1-D scale filter.
//
// Analytic, no golden. The 1-D filter update is the SAME CODE already checked
// against NumPy at 2-D (filter_init/filter_update are dimension-agnostic), so a
// new golden would re-test tested code. What is genuinely new is the DFT, the
// feature extraction and the detect loop — and every property of those worth
// asserting has a closed form.
// ---------------------------------------------------------------------------
void run_scale_tests()
{
    using namespace mosse;
    printf("\nDSST 1-D scale filter (analytic)\n");

    constexpr int S = 33;

    // 1. DFT closed forms. A delta transforms to a constant, a constant to a
    //    delta, and forward-then-inverse is the identity. Between them these pin
    //    the sign convention, the normalisation and the indexing.
    {
        std::vector<cfloat> in(S), out(S), back(S);
        in[0] = cfloat(1.0f, 0.0f);
        dft_1d(in.data(), out.data(), S, false);
        double worst = 0.0;
        for (int k = 0; k < S; ++k)
            worst = std::max(worst, (double)std::abs(out[k] - cfloat(1.0f, 0.0f)));
        check_double("DFT: delta -> constant", worst, 0.0, 1e-5);

        std::fill(in.begin(), in.end(), cfloat(1.0f, 0.0f));
        dft_1d(in.data(), out.data(), S, false);
        worst = (double)std::abs(out[0] - cfloat((float)S, 0.0f));
        for (int k = 1; k < S; ++k) worst = std::max(worst, (double)std::abs(out[k]));
        check_double("DFT: constant -> delta", worst, 0.0, 1e-4);

        Lcg rng(12345);
        for (int k = 0; k < S; ++k)
            in[k] = cfloat((float)rng.next(1000) / 1000.0f,
                           (float)rng.next(1000) / 1000.0f);
        dft_1d(in.data(), out.data(), S, false);
        dft_1d(out.data(), back.data(), S, true);
        worst = 0.0;
        for (int k = 0; k < S; ++k)
            worst = std::max(worst, (double)std::abs(back[k] - in[k]));
        check_double("DFT: round-trip identity", worst, 0.0, 1e-5);
    }

    // 2. The 1-D target degenerates cleanly at rows = 1. This is the property
    //    the whole "reuse the 2-D filter" argument rests on, and DSST §3 asserts
    //    it in prose; here it is asserted in code. Cross-checked against a direct
    //    DFT of the spatial wrapped Gaussian rather than against a second copy of
    //    the closed form.
    {
        const float sig = (float)S / 16.0f;
        std::vector<cfloat> G(S);
        gaussian_target_spectrum(G.data(), 1, S, sig, 0, 0);

        std::vector<cfloat> g(S), Gd(S);
        for (int n = 0; n < S; ++n) {
            const double dn = (n <= S / 2) ? n : n - S;
            g[n] = cfloat((float)std::exp(-0.5 * dn * dn / ((double)sig * sig)), 0.0f);
        }
        dft_1d(g.data(), Gd.data(), S, false);
        // The closed form drops the constant gain, so compare shapes.
        double sc = (double)Gd[0].real() / (double)G[0].real();
        double worst = 0.0;
        for (int k = 0; k < S; ++k)
            worst = std::max(worst, (double)std::abs(Gd[k] - G[k] * (float)sc));
        check_double("1-D target vs direct DFT", worst / std::abs(sc), 0.0, 1e-3);
        check_true("1-D target is real at dr=0",
                   std::abs(G[3].imag()) < 1e-6f, "centred => conj(G) == G");
    }

    // 3. End to end on a synthetic pyramid. Build a frame with a box of known
    //    size, train at that size, then present the SAME frame while claiming a
    //    wrong box size — the filter must report the level that corrects it.
    //    This is the test that would fail on a sign error in the detect loop,
    //    an off-by-one in the wrap, or a transposed sample layout.
    {
        constexpr int FR = 256, FC = 256;
        std::vector<uint8_t> frame((size_t)FR * FC, 60);
        // A bright square of side 64, centred.
        for (int r = 96; r < 160; ++r)
            for (int c = 96; c < 160; ++c)
                frame[(size_t)r * FC + c] = 200;

        ScaleFilter sf;
        scale_filter_config(sf, S, 1.02f, 64.0, 64.0, 16.0f);
        check_int("template area <= cap", sf.dims() <= SCALE_TMPL_AREA ? 1 : 0, 1);
        check_true("S forced odd", (sf.n_scales % 2) == 1, "");

        std::vector<cfloat> F((size_t)sf.sample_elems());
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, F.data());
        scale_update(sf, F.data(), 1.0f);
        check_true("scale filter initialises", sf.initialized, "");

        // Correct size in, no change out.
        std::vector<cfloat> Z((size_t)sf.sample_elems());
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, Z.data());
        ScaleResult r0 = scale_detect(sf, Z.data(), DEFAULT_EPS_REL);
        check_int("correct scale -> level 0", r0.idx, 0);
        check_double("correct scale -> factor 1", r0.factor, 1.0, 1e-9);

        // Claim the box is 10% too SMALL. The true object then looks larger than
        // the template expects, so the filter must select a POSITIVE level.
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 58.0, 58.0, Z.data());
        ScaleResult rs = scale_detect(sf, Z.data(), DEFAULT_EPS_REL);
        check_true("under-sized box -> positive level", rs.idx > 0,
                   ("idx " + std::to_string(rs.idx) + ", factor "
                    + std::to_string(rs.factor)).c_str());
        // Claim it is 10% too LARGE -> negative level.
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 71.0, 71.0, Z.data());
        ScaleResult rl = scale_detect(sf, Z.data(), DEFAULT_EPS_REL);
        check_true("over-sized box -> negative level", rl.idx < 0,
                   ("idx " + std::to_string(rl.idx) + ", factor "
                    + std::to_string(rl.factor)).c_str());
        check_true("levels are opposite in sign", rs.idx * rl.idx < 0,
                   "a filter that always answers the same way would pass one of these");
    }

    // 3b. CONVERGENCE, which is the property that actually decides whether the
    //     filter is useful. A single application UNDER-corrects — the test above
    //     answers +3 (factor 1.061) to a box that is 10% too small — because the
    //     scale response is a correlation peak on a discrete grid smoothed by a
    //     sigma = S/16 target. DSST relies on iterating across frames, so what
    //     must be asserted is that repeated application walks the box TOWARDS the
    //     truth and settles, not that one shot lands on it.
    //
    //     Trained once at the true size (as frame 0 does, where the box IS the
    //     ground truth) and then applied repeatedly without retraining, which is
    //     what a static scene reduces to.
    {
        constexpr int FR = 256, FC = 256;
        std::vector<uint8_t> frame((size_t)FR * FC, 60);
        for (int r = 96; r < 160; ++r)
            for (int c = 96; c < 160; ++c)
                frame[(size_t)r * FC + c] = 200;

        ScaleFilter sf;
        scale_filter_config(sf, S, 1.02f, 64.0, 64.0, 16.0f);
        std::vector<cfloat> F((size_t)sf.sample_elems());
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, F.data());
        scale_update(sf, F.data(), 1.0f);

        double h = 51.2;                       // start 20% too small
        const double err0 = std::fabs(h - 64.0);
        double prev = err0;
        bool monotone = true;
        for (int it = 0; it < 12; ++it) {
            scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, h, h, F.data());
            const ScaleResult r = scale_detect(sf, F.data(), DEFAULT_EPS_REL);
            h *= r.factor;
            const double e = std::fabs(h - 64.0);
            if (e > prev + 1e-9) monotone = false;   // must never move away
            prev = e;
        }
        char det[96];
        snprintf(det, sizeof(det), "51.2 -> %.2f (truth 64), error %.2f -> %.2f px",
                 h, err0, prev);
        check_true("scale converges toward truth", prev < err0 * 0.35, det);
        check_true("scale never diverges", monotone,
                   "error must not increase on any iteration");
    }

    // 4. SCALE_N = 1 must be a complete no-op, so the whole feature can be
    //    switched off with one make variable — the CONV_VECTORIZE=0 pattern.
    {
        ScaleFilter off;
        scale_filter_config(off, 1, 1.02f, 64.0, 64.0, 16.0f);
        check_true("S=1 disables the filter", !off.enabled(), "");
        std::vector<cfloat> Z(1);
        ScaleResult r = scale_detect(off, Z.data(), DEFAULT_EPS_REL);
        check_true("disabled -> invalid, factor 1", !r.valid && r.factor == 1.0, "");
        scale_update(off, Z.data(), 1.0f);
        check_true("disabled -> no training", !off.initialized, "");
    }

    // 5. An untrained filter must never claim a scale. Without this the very
    //    first frame would apply a filter of zeros and resize the box from noise.
    {
        ScaleFilter fresh;
        scale_filter_config(fresh, S, 1.02f, 64.0, 64.0, 16.0f);
        std::vector<cfloat> Z((size_t)fresh.sample_elems());
        ScaleResult r = scale_detect(fresh, Z.data(), DEFAULT_EPS_REL);
        check_true("untrained -> invalid", !r.valid, "");
    }

    // -------------------------------------------------------------------
    // Scale-update gating. The values are the ones hardware actually produced
    // on 2026-08-20, not invented ones — the point of the gate is that it
    // separates THOSE two populations, so the test asserts against them.
    // -------------------------------------------------------------------
    printf("\nScale-update gate\n");
    {
        const double H0 = 64.0, W0 = 64.0;
        const float  CONF = 2.0f;
        const double MINR = 0.5, MAXR = 2.0;
        // The cases below predate the MaxStep rate limit and exercise the OTHER
        // vetoes, several at |idx| far beyond any rate limit (-12, +-16, 15). They
        // pass max_step = 0 so they keep asserting exactly what they asserted
        // before it existed; MaxStep has its own block at the end.
        const int NOSTEP = 0;

        auto mk = [](int idx, double factor, double conf) {
            ScaleResult r;
            r.idx = idx; r.factor = factor; r.psr = conf; r.valid = true;
            return r;
        };

        // The five healthy frames: level +0, conf 3.30-3.31. All must pass.
        {
            ScaleDecision d = scale_gate(mk(0, 1.0, 3.31), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("healthy (idx 0, conf 3.31) accepts", d.accept, "");
            check_true("healthy reason", d.reason == ScaleVeto::Accept, "");
        }
        // Frame 6, the collapse that started the runaway: level -12, conf 1.22.
        // This is the single frame the whole gate exists to reject.
        {
            ScaleDecision d = scale_gate(mk(-12, 0.7885, 1.22), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("frame-6 collapse (idx -12, conf 1.22) HELD", !d.accept, "");
            check_true("frame-6 reason is LOW_CONF",
                       d.reason == ScaleVeto::LowConf, "");
            check_double("frame-6 proposal recorded", d.new_h, 64.0 * 0.7885, 1e-9);
        }
        // Frame 13: argmax at exactly +16 of +/-16. Structural, and it must be
        // reported as the rail even though conf (1.57) is also below threshold —
        // that ordering is the whole reason the reasons are an enum.
        {
            ScaleDecision d = scale_gate(mk(16, 1.3728, 1.57), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("search-rail (idx +16) HELD", !d.accept, "");
            check_true("search-rail reason beats low-conf",
                       d.reason == ScaleVeto::AtSearchRail, "");
        }
        // ...and the same at the negative rail, because a sign error here would
        // veto one direction only and look like a working gate.
        {
            ScaleDecision d = scale_gate(mk(-16, 0.7284, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("negative search-rail HELD, high conf", !d.accept, "");
            check_true("negative rail reason",
                       d.reason == ScaleVeto::AtSearchRail, "");
        }
        // Interior neighbour of the rail must still pass, or the gate silently
        // costs the outer levels of the search range.
        {
            ScaleDecision d = scale_gate(mk(15, 1.3459, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("idx 15 (interior) accepts", d.accept, "");
        }
        // Drift backstop: confident, interior, but the proposal leaves the bounds.
        {
            ScaleDecision d = scale_gate(mk(1, 1.02, 3.0), S, 20.0, 20.0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("below min_rel HELD", !d.accept, "");
            check_true("below min_rel reason",
                       d.reason == ScaleVeto::OutOfRange, "");
        }
        {
            ScaleDecision d = scale_gate(mk(1, 1.02, 3.0), S, 130.0, 130.0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("above max_rel HELD", !d.accept, "");
        }
        // The test sequence's own envelope (0.70x..1.30x) must NOT be clipped, or
        // the gate would fight the ground truth it is being scored against.
        {
            ScaleDecision a = scale_gate(mk(1, 1.0, 3.0), S, 64.0 * 0.70, 64.0 * 0.70,
                                         H0, W0, CONF, MINR, MAXR, NOSTEP);
            ScaleDecision b = scale_gate(mk(1, 1.0, 3.0), S, 64.0 * 1.30, 64.0 * 1.30,
                                         H0, W0, CONF, MINR, MAXR, NOSTEP);
            check_true("SCALE_TRAJ envelope 0.70x admitted", a.accept, "");
            check_true("SCALE_TRAJ envelope 1.30x admitted", b.accept, "");
        }
        // Threshold off disables ONLY the confidence test — the structural vetoes
        // must survive, exactly as they do for PSR_GATE_MIN=0.
        {
            ScaleDecision d = scale_gate(mk(-12, 0.7885, 1.22), S, H0, W0, H0, W0,
                                         0.0f, MINR, MAXR, NOSTEP);
            check_true("conf_min=0 accepts a low-conf interior estimate",
                       d.accept, "");
            check_true("conf_min=0 reason", d.reason == ScaleVeto::Disabled, "");
            ScaleDecision r = scale_gate(mk(16, 1.3728, 1.57), S, H0, W0, H0, W0,
                                         0.0f, MINR, MAXR, NOSTEP);
            check_true("conf_min=0 still vetoes the search rail", !r.accept, "");
        }
        // ---- MaxStep: the per-frame RATE limit -------------------------
        // Driven by the values hardware produced on 2026-08-25 (car1 anchor 0,
        // runs/run_0825_1314.log), not invented ones — every one of these idx
        // values is one the board actually proposed.
        {
            const int STEP1 = 1;
            // Frame 490: conf 2.00 against a 2.00 threshold, so it PASSES the
            // confidence test on the nose, and moves the box 1.42x in one frame
            // while the tracker was 227 px off target. This is the frame the
            // veto exists for, and it must be rejected as MAX_STEP rather than
            // as anything else — the reason is the finding.
            ScaleDecision d = scale_gate(mk(9, 1.4233, 2.00), S, 105.0, 108.0,
                                         H0, W0, CONF, MINR, 2.5, STEP1);
            check_true("f490 (idx +9, conf 2.00) HELD", !d.accept, "");
            check_true("f490 reason is MAX_STEP",
                       d.reason == ScaleVeto::MaxStep, "");
            check_double("f490 proposal still recorded", d.new_h,
                         105.0 * 1.4233, 1e-9);
        }
        {
            const int STEP1 = 1;
            // Both signs, because a one-sided comparison vetoes growth only and
            // still looks like a working gate — the same trap the search-rail
            // case above guards.
            ScaleDecision up = scale_gate(mk(2, 1.0816, 3.0), S, H0, W0, H0, W0,
                                          CONF, MINR, MAXR, STEP1);
            ScaleDecision dn = scale_gate(mk(-2, 0.9246, 3.0), S, H0, W0, H0, W0,
                                          CONF, MINR, MAXR, STEP1);
            check_true("idx +2 HELD at max_step 1", !up.accept, "");
            check_true("idx -2 HELD at max_step 1", !dn.accept, "");
            check_true("idx -2 reason", dn.reason == ScaleVeto::MaxStep, "");
        }
        {
            const int STEP1 = 1;
            // The boundary is INCLUSIVE, and it has to be: |idx| <= 1 is the
            // entire population the synthetic detector ever proposed across 199
            // frames, so an off-by-one here would freeze the scale filter
            // completely while still reporting a reason that looks deliberate.
            ScaleDecision a = scale_gate(mk(1, 1.04, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, STEP1);
            ScaleDecision b = scale_gate(mk(-1, 0.9615, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, STEP1);
            ScaleDecision z = scale_gate(mk(0, 1.0, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, STEP1);
            check_true("idx +1 accepted at max_step 1", a.accept, "");
            check_true("idx -1 accepted at max_step 1", b.accept, "");
            check_true("idx  0 accepted at max_step 1", z.accept, "");
        }
        {
            // max_step = 0 disables the RATE test only, exactly as conf_min = 0
            // disables the confidence test. The structural vetoes must survive,
            // or "disabled" would quietly mean "ungated".
            ScaleDecision d = scale_gate(mk(9, 1.4233, 3.0), S, 105.0, 108.0,
                                         H0, W0, CONF, MINR, 2.5, 0);
            check_true("max_step=0 admits idx +9", d.accept, "");
            ScaleDecision r = scale_gate(mk(16, 1.3728, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, 0);
            check_true("max_step=0 still vetoes the search rail", !r.accept, "");
            check_true("max_step=0 rail reason",
                       r.reason == ScaleVeto::AtSearchRail, "");
        }
        {
            // ORDERING. A proposal that is both at the rail and beyond the rate
            // limit reports AT_SEARCH_RAIL, the stronger statement: the argmax is
            // the edge of what was searched, so the true optimum is not merely
            // large, it is unmeasured.
            ScaleDecision d = scale_gate(mk(16, 1.3728, 3.0), S, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, 1);
            check_true("rail beats max_step", d.reason == ScaleVeto::AtSearchRail, "");
            // ...and a rate-limited proposal that is ALSO low-conf reports
            // MAX_STEP, because conf provably cannot separate a wrong proposal
            // from a big correct one and the magnitude can.
            ScaleDecision e = scale_gate(mk(7, 1.3159, 1.20), S, H0, W0, H0, W0,
                                         CONF, MINR, 2.5, 1);
            check_true("max_step beats low_conf", e.reason == ScaleVeto::MaxStep, "");
        }
        {
            // THE SHIPPING DEFAULT, asserted directly. The block above sweeps
            // max_step as a parameter; this pins what the build actually uses,
            // because the value was chosen by measurement (scale_loop_sim: a
            // limit of 1 parks the smooth arm 123 of 200 frames) and a silent
            // change back to 1 would look like a tightening rather than the
            // regression it is.
            const int D = DEFAULT_SCALE_MAX_STEP;
            check_true("default max_step is 2, not the tempting 1", D == 2, "");
            ScaleDecision two = scale_gate(mk(2, 1.0816, 3.0), S, H0, W0, H0, W0,
                                           CONF, MINR, MAXR, D);
            ScaleDecision nine = scale_gate(mk(9, 1.4233, 2.00), S, 105.0, 108.0,
                                            H0, W0, CONF, MINR, 2.5, D);
            check_true("default admits idx +-2 (the sim's smooth arm needs it)",
                       two.accept, "");
            check_true("default still vetoes f490's idx +9", !nine.accept, "");
            check_true("default f490 reason", nine.reason == ScaleVeto::MaxStep, "");
        }
        {
            // The tag and the sentence must exist for the new reason. A veto that
            // prints "?" in the [scale] summary is a veto nobody will diagnose.
            check_true("MAX_STEP has a tag",
                       std::string(scale_veto_tag(ScaleVeto::MaxStep)) == "MAX_STEP", "");
            check_true("MAX_STEP has a why",
                       std::string(scale_veto_why(ScaleVeto::MaxStep)).size() > 40, "");
        }

        // ---- COASTING THROUGH A HOLD -----------------------------------
        // The property that makes a coast safe is that its total drift is
        // BOUNDED, so a long hold decays back to the freeze this replaces
        // instead of becoming a second way to lose the target. Everything here
        // asserts that bound rather than a transcribed constant.
        {
            CoastState c;
            double dr = 0.0, dc = 0.0;
            // Nothing measured yet: a hold before the first accepted frame must
            // coast by NOTHING, not by an undefined velocity. This is the real
            // frame-0-then-gated case, not a hypothetical.
            check_true("coast with no measurement does not move",
                       !coast_step(c, 0.5, &dr, &dc), "");

            coast_observe(c, 10.0, -4.0);
            check_true("first held frame gets the FULL measured velocity",
                       coast_step(c, 0.5, &dr, &dc) && dr == 10.0 && dc == -4.0, "");
            check_true("second held frame is halved at decay 0.5",
                       coast_step(c, 0.5, &dr, &dc) && dr == 5.0 && dc == -2.0, "");
            check_true("third is quartered",
                       coast_step(c, 0.5, &dr, &dc) && dr == 2.5 && dc == -1.0, "");

            // THE BOUND. Sum the whole run and check it against the closed form,
            // which is what keeps a hold from walking the window away.
            CoastState b;
            coast_observe(b, 10.0, -4.0);
            double sr = 0.0, sc = 0.0;
            for (int i = 0; i < 200; ++i) {
                double a = 0.0, o = 0.0;
                if (!coast_step(b, 0.5, &a, &o)) break;
                sr += a; sc += o;
            }
            const double drift = std::sqrt(sr * sr + sc * sc);
            const double bound = coast_drift_bound(b, 0.5);
            char det[96];
            snprintf(det, sizeof det, "%.4f <= %.4f", drift, bound);
            check_true("total drift stays within v/(1-decay)", drift <= bound + 1e-9, det);
            // ...and it must actually APPROACH the bound, or "bounded" would be
            // satisfied by a coast that does nothing at all.
            check_true("drift reaches most of the bound", drift > 0.9 * bound, det);
            // A faded run must stop rather than dribble forever, so a 200-frame
            // hold does not accumulate denormal noise.
            double a = 0.0, o = 0.0;
            check_true("a faded coast reports nothing to do",
                       !coast_step(b, 0.5, &a, &o), "");
        }
        {
            // decay 0 is the "coast exactly one frame" policy the sweep measured
            // at 94.8% -- it must be reachable, not an approximation of it.
            CoastState c;
            coast_observe(c, 3.0, 4.0);
            double dr = 0.0, dc = 0.0;
            check_true("decay 0: first held frame moves",
                       coast_step(c, 0.0, &dr, &dc) && dr == 3.0 && dc == 4.0, "");
            check_true("decay 0: second does not",
                       !coast_step(c, 0.0, &dr, &dc), "");
            check_double("decay 0 bound is |v|", coast_drift_bound(c, 0.0), 5.0, 1e-12);
        }
        {
            // An accepted frame RESETS the run, so a hold after a long previous
            // hold coasts at full velocity again. Without the reset the tracker
            // would coast less and less the longer it ran, which is a slow
            // failure and exactly the kind this project keeps not noticing.
            CoastState c;
            coast_observe(c, 8.0, 0.0);
            double dr = 0.0, dc = 0.0;
            coast_step(c, 0.5, &dr, &dc);
            coast_step(c, 0.5, &dr, &dc);          // decayed to 0.5
            coast_observe(c, 8.0, 0.0);            // a frame was accepted
            check_true("an accepted frame restores full velocity",
                       coast_step(c, 0.5, &dr, &dc) && dr == 8.0, "");
        }
        {
            // A stationary target measured as stationary must not be coasted --
            // this is the occlusion case the freeze policy exists for, and it is
            // the direction in which a coast can do harm.
            CoastState c;
            coast_observe(c, 0.0, 0.0);
            double dr = 0.0, dc = 0.0;
            check_true("zero measured velocity never coasts",
                       !coast_step(c, 0.5, &dr, &dc), "");
        }

        // An invalid result must never resize the box, and must say so as
        // INVALID rather than masquerading as a rejection.
        {
            ScaleResult bad;                       // valid = false
            ScaleDecision d = scale_gate(bad, S, H0, W0, H0, W0, CONF, MINR, MAXR, NOSTEP);
            check_true("invalid result HELD", !d.accept, "");
            check_true("invalid reason", d.reason == ScaleVeto::Invalid, "");
        }
        // A degenerate filter must not veto everything: at S <= 2 every index IS
        // the rail, and a gate that holds every frame would freeze the size
        // silently.
        {
            ScaleDecision d = scale_gate(mk(0, 1.0, 3.0), 1, H0, W0, H0, W0,
                                         CONF, MINR, MAXR, NOSTEP);
            check_true("S=1 does not trip the rail veto", d.accept, "");
        }
        // Every reason must have a tag and a sentence — a verdict with no
        // explanation is the thing this design specifically does not ship.
        {
            // The loop is bounded by the LAST enumerator, so a veto appended
            // after OutOfRange would silently escape coverage -- the same shape
            // as the layer0.h #error that turned out to be unreachable. This
            // line makes adding one a compile error instead.
            static_assert((int)ScaleVeto::OutOfRange == 6,
                          "ScaleVeto changed: check this coverage loop still "
                          "reaches every enumerator");
            bool all = true;
            for (int i = 0; i <= (int)ScaleVeto::OutOfRange; ++i) {
                const char *t = scale_veto_tag((ScaleVeto)i);
                const char *w = scale_veto_why((ScaleVeto)i);
                all = all && t && w && t[0] != '?' && w[0] != '?';
            }
            check_true("every ScaleVeto has a tag and a reason", all, "");
        }
    }
}


// ---------------------------------------------------------------------------
// Training target — the defect fixed 2026-08-20
// ---------------------------------------------------------------------------
//
// filter_update() is handed the patch cropped at the PRE-update position, so the
// object in it sits at the measured displacement (dr,dc), not at the origin.
// Training that patch against a G centred at (0,0) teaches "object at (dr,dc)
// => peak at (0,0)", and under near-constant motion the error is coherent frame
// to frame, so it compounds at the learning rate until the origin peak wins.
//
// A SINGLE UPDATE CANNOT SEE THIS — gen_filter_golden.py's one-shot check passed
// throughout — so these tests run a small CLOSED LOOP and compare the two arms.
// The hardware-free, full-size companion is scripts/mosse_loop_sim.py.
namespace tt {

constexpr int TR = 32, TC = 32, TCH = 2;
constexpr float TSIGMA = 1.5f;

using cf = mosse::cfloat;

// Separable naive DFT — TR*TC is 1024 bins, so the O(N^3) form costs ~65k
// complex multiplies per transform. No FFT dependency, which is the whole point
// of mosse_filter.{h,cpp} having none.
void dft2(const cf *in, cf *out, int rows, int cols, int sign)
{
    std::vector<cf> tmp((size_t)rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int v = 0; v < cols; ++v) {
            cf acc(0.0f, 0.0f);
            for (int c = 0; c < cols; ++c) {
                const double th = sign * 2.0 * M_PI * v * c / cols;
                acc += in[(size_t)r * cols + c] * cf((float)std::cos(th),
                                                     (float)std::sin(th));
            }
            tmp[(size_t)r * cols + v] = acc;
        }
    for (int v = 0; v < cols; ++v)
        for (int u = 0; u < rows; ++u) {
            cf acc(0.0f, 0.0f);
            for (int r = 0; r < rows; ++r) {
                const double th = sign * 2.0 * M_PI * u * r / rows;
                acc += tmp[(size_t)r * cols + v] * cf((float)std::cos(th),
                                                      (float)std::sin(th));
            }
            out[(size_t)u * cols + v] = acc;
        }
}

// An asymmetric structured patch, wrapped, with the object centred at (or, oc).
// Asymmetric because a symmetric blob hides sign errors — the same reason s7's
// target is off-centre.
void make_patch(cf *p, int or_, int oc_)
{
    for (int r = 0; r < TR; ++r)
        for (int c = 0; c < TC; ++c) {
            int dr = r - or_; if (dr >  TR / 2) dr -= TR; if (dr < -TR / 2) dr += TR;
            int dc = c - oc_; if (dc >  TC / 2) dc -= TC; if (dc < -TC / 2) dc += TC;
            const double v = 100.0 * std::exp(-(dr * dr + dc * dc) / 18.0)
                           + 60.0 * std::exp(-((dr - 3) * (dr - 3)
                                             + (dc + 4) * (dc + 4)) / 6.0);
            p[(size_t)r * TC + c] = cf((float)v, 0.0f);
        }
}

// One closed-loop run. `shifted` selects the fix. Returns resp00/peak per frame.
std::vector<double> loop(bool shifted, int frames, int vr, int vc,
                         std::vector<int> *drs, std::vector<int> *dcs)
{
    mosse::FilterState st;
    std::vector<cf> patch((size_t)TR * TC), F((size_t)TCH * TR * TC);
    std::vector<cf> G0((size_t)TR * TC), G((size_t)TR * TC);
    std::vector<cf> S((size_t)TR * TC), resp((size_t)TR * TC);

    mosse::gaussian_target_spectrum(G0.data(), TR, TC, TSIGMA, 0, 0);

    // Object position, and the tracker's estimate of it. Both in patch bins;
    // the crop offset is (est - true), i.e. where the object lands in the patch.
    int t_r = 0, t_c = 0, e_r = 0, e_c = 0;
    std::vector<double> out;

    for (int f = 0; f < frames; ++f) {
        // Crop at the ESTIMATE: the object appears at (true - est), wrapped.
        make_patch(patch.data(), ((t_r - e_r) % TR + TR) % TR,
                                 ((t_c - e_c) % TC + TC) % TC);
        for (int ch = 0; ch < TCH; ++ch) {
            // Two trivially different "channels": the patch, and its gradient.
            std::vector<cf> in((size_t)TR * TC);
            for (int i = 0; i < TR * TC; ++i)
                in[i] = ch == 0 ? patch[i]
                                : patch[i] - patch[(i + 1) % (TR * TC)];
            dft2(in.data(), F.data() + (size_t)ch * TR * TC, TR, TC, -1);
        }

        if (f == 0) {
            // Bootstrap: the crop IS centred on the object, so centred is right
            // in both arms.
            mosse::filter_init(st, F.data(), G0.data(), TCH, TR, TC);
        } else {
            // Detect: R = IDFT( sum F * conj(H) ), H = A / (B + eps).
            double bsum = 0.0;
            for (int i = 0; i < TR * TC; ++i) bsum += st.B[i];
            const float eps = mosse::DEFAULT_EPS_REL * (float)(bsum / (TR * TC));
            for (int i = 0; i < TR * TC; ++i) {
                cf acc(0.0f, 0.0f);
                for (int ch = 0; ch < TCH; ++ch)
                    acc += F[(size_t)ch * TR * TC + i]
                         * std::conj(st.A[(size_t)ch * TR * TC + i] / (st.B[i] + eps));
                S[i] = acc;
            }
            dft2(S.data(), resp.data(), TR, TC, +1);

            int bi = 0; double best = 0.0;
            for (int i = 0; i < TR * TC; ++i) {
                const double m = std::fabs(resp[i].real());
                if (m > best) { best = m; bi = i; }
            }
            int dr = bi / TC, dc = bi % TC;
            if (dr > TR / 2) dr -= TR;
            if (dc > TC / 2) dc -= TC;
            out.push_back(best > 0.0 ? std::fabs(resp[0].real()) / best : 0.0);
            if (drs) drs->push_back(dr);
            if (dcs) dcs->push_back(dc);

            e_r += dr; e_c += dc;

            mosse::gaussian_target_spectrum(G.data(), TR, TC, TSIGMA,
                                            shifted ? dr : 0, shifted ? dc : 0);
            mosse::filter_update(st, F.data(), G.data(), mosse::DEFAULT_ETA);
        }
        t_r += vr; t_c += vc;
    }
    return out;
}

}  // namespace tt

// -----------------------------------------------------------------------
// filter_update_quantize() — the fused update+publish path
// -----------------------------------------------------------------------
// BIT-IDENTICAL, ASSERTED, NOT INTENDED. The fusion exists to save memory
// traffic and half the divides; if it also perturbed the last bit of H it would
// end ten consecutive hardware runs of bit-identical tracking and make the next
// before/after comparison unreadable. That property is cheap to keep (the fused
// loop is textually the two original loops) and worthless unless it is checked,
// so this compares raw bytes — memcmp, not a tolerance.
// -----------------------------------------------------------------------
// FILTER_MASK — the spatial-reliability projection h <- m (.) h
// -----------------------------------------------------------------------
// THE ONE NEW THING THAT CAN BE SILENTLY WRONG. filter_mask_project() claims a
// 3-tap circular convolution on H is exactly a Hann multiply in the spatial
// domain. If the constant, an axis, or the wrap is wrong the result is still a
// plausible filter — a slightly rescaled or slightly shifted one — and the
// board would report a number nobody could attribute. So the reference here is
// an INDEPENDENT naive DFT written in this file, not mosse_filter.cpp's own
// transform, and the assertion is made in the SPATIAL domain where the claim
// actually lives: ifft(project(H)) must equal ifft(H) * m, elementwise.
//
// These run at every FILTER_MASK setting, because filter_mask_project() is
// compiled unconditionally — only its CALL SITES are #if'd. A projection that
// is only tested in the arm that uses it is tested exactly when it is too late.
namespace maskref {

// Naive O(n^2) per axis. 16x16 makes that 2*16*256 = 8192 complex MACs, i.e.
// instant, and being naive is the point: it shares no code with the thing under
// test.
void idft2(const std::vector<cfloat> &X, std::vector<cfloat> &x, int R, int C)
{
    x.assign((size_t)R * C, cfloat(0.0f, 0.0f));
    const double tau = 6.283185307179586;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            std::complex<double> acc(0.0, 0.0);
            for (int u = 0; u < R; ++u)
                for (int v = 0; v < C; ++v) {
                    const double th = tau * ((double)u * r / R + (double)v * c / C);
                    const std::complex<double> w(std::cos(th), std::sin(th));
                    const cfloat Xv = X[(size_t)u * C + v];
                    acc += w * std::complex<double>(Xv.real(), Xv.imag());
                }
            acc /= (double)(R * C);
            x[(size_t)r * C + c] = cfloat((float)acc.real(), (float)acc.imag());
        }
}

// The window the BOARD applies: the periodic Hann sin^2(pi i / n), centred at
// n/2 — hanning_<N>.h's table, and the same window conv2d puts on the patch.
// NOT the offline bench's (n-1)/2 centring; that half-sample difference is
// worth mean IoU 0.1715 vs 0.2813 on `tiger` (claim O-01,
// docs/thesis/evidence/proposed_build_mask.md) and is why this is spelled out.
double hann(int i, int n)
{
    const double s = std::sin(3.14159265358979323846 * (double)i / (double)n);
    return s * s;
}

std::vector<cfloat> random_H(int R, int C, unsigned seed)
{
    std::vector<cfloat> H((size_t)R * C);
    unsigned s = seed;
    auto nxt = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (float)((double)(s >> 8) / 16777216.0) - 0.5f;
    };
    for (auto &v : H) v = cfloat(nxt(), nxt());
    return H;
}

}  // namespace maskref

void run_filter_mask_tests()
{
    printf("\n--- FILTER_MASK spatial projection ---\n");
    const int R = 16, C = 16;

    // (1) THE CLAIM ITSELF, in the spatial domain.
    {
        std::vector<cfloat> H = maskref::random_H(R, C, 12345u);
        std::vector<cfloat> h_before, h_after;
        maskref::idft2(H, h_before, R, C);

        std::vector<cfloat> P = H;
        mosse::filter_mask_project(P.data(), R, C);
        maskref::idft2(P, h_after, R, C);

        std::vector<cfloat> expect((size_t)R * C);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                const float m = (float)(maskref::hann(r, R) * maskref::hann(c, C));
                expect[(size_t)r * C + c] = h_before[(size_t)r * C + c] * m;
            }
        check_cfloat("h<-m.h", h_after, expect, 2e-5f);
    }

    // (2) THE CONSTANT. A wrong 1/16 rescales H uniformly, which an argmax
    // cannot see and the Q1.15 renormalisation hides completely — so it is
    // invisible everywhere except here. Pinned against a hand-computed bin: a
    // DC-only H is spatially constant, and multiplying a constant by the Hann
    // leaves DC scaled by mean(m) = 0.25 and bins (0,+-1),(+-1,0),(+-1,+-1)
    // at the outer product of {1/2,-1/4,-1/4} with itself.
    {
        std::vector<cfloat> H((size_t)R * C, cfloat(0.0f, 0.0f));
        H[0] = cfloat(1.0f, 0.0f);
        std::vector<cfloat> P = H;
        mosse::filter_mask_project(P.data(), R, C);
        check_double("mask DC gain", (double)P[0].real(), 0.25, 1e-6);
        check_double("mask (0,1) gain",
                     (double)P[1].real(), -0.125, 1e-6);
        check_double("mask (1,1) gain",
                     (double)P[(size_t)C + 1].real(), 0.0625, 1e-6);
        double off = 0.0;
        for (int r = 2; r < R - 1; ++r)
            for (int c = 2; c < C - 1; ++c)
                off = std::max(off, (double)std::abs(P[(size_t)r * C + c]));
        check_true("mask spectrum is 9 bins", off < 1e-7,
                   "everything outside {0,+-1}^2 is zero");
    }

    // (3) THE DEGENERATE-AXIS TRAP, and it is a live one. FilterState is also
    // the DSST scale filter's type at rows == 1, where a circular D would read
    // X[-1] == X[+1] == X[0] and return 2X - X - X = ZERO — deleting the filter
    // instead of masking it. The row axis must be SKIPPED, not wrapped.
    {
        std::vector<cfloat> H = maskref::random_H(1, C, 777u);
        std::vector<cfloat> P = H;
        mosse::filter_mask_project(P.data(), 1, C);
        double mag = 0.0;
        for (const auto &v : P) mag = std::max(mag, (double)std::abs(v));
        check_true("rows==1 not zeroed", mag > 1e-3, "1-D state survives");

        // and it must be exactly the column-only projection, 1/4 not 1/16.
        std::vector<cfloat> ref((size_t)C);
        for (int c = 0; c < C; ++c) {
            const int cm = (c == 0) ? C - 1 : c - 1;
            const int cp = (c == C - 1) ? 0 : c + 1;
            ref[(size_t)c] = (2.0f * H[(size_t)c] - H[(size_t)cm] - H[(size_t)cp])
                           * 0.25f;
        }
        check_cfloat("rows==1 1-D", P, ref, 1e-6f);
    }

    // (4) THE ENERGY INSTRUMENT — the build's mechanism falsifier. If this is
    // wrong the run cannot tell "the mask worked" from "something else did".
    //
    // EVERY CASE BELOW FEEDS A FREQUENCY-DOMAIN H, BUILT BY FORWARD-TRANSFORMING
    // THE SPATIAL PATTERN THE CASE IS ABOUT. That is the whole correction: the
    // first version of these tests handed the function the SPATIAL array
    // directly, which is self-consistent with a function that does no transform
    // and agrees with the caller about nothing. The caller only ever has
    // frequency-domain H, and on hardware that mismatch read 0.0000 on every
    // frame while all five of these were green.
    {
        const int Rb = 8, Cb = 8;
        const size_t N = (size_t)Rb * Cb;

        // Spatial pattern -> H, with the naive forward DFT in this file. Shares
        // no code with the radix-2 transform under test.
        auto to_freq = [&](const std::vector<cfloat> &sp) {
            std::vector<cfloat> H(sp.size());
            const int nch = (int)(sp.size() / N);
            for (int ch = 0; ch < nch; ++ch)
                tt::dft2(sp.data() + (size_t)ch * N, H.data() + (size_t)ch * N,
                     Rb, Cb, -1);
            return H;
        };

        // A delta AT THE PATCH CENTRE must read 1.0: the filter's energy is
        // centred there, not at the response origin, and getting that backwards
        // is the error that reads as "masking hurts".
        std::vector<cfloat> h(N, cfloat(0.0f, 0.0f));
        h[(size_t)(Rb / 2) * Cb + (Cb / 2)] = cfloat(3.0f, 4.0f);
        check_double("energy frac, centre",
                     mosse::filter_box_energy_fraction(to_freq(h).data(), 1,
                                                       Rb, Cb, 4, 4),
                     1.0, 1e-5);

        // The same delta at the ORIGIN must read 0.0 — the discriminator
        // between the two conventions, and the whole reason this is asserted.
        std::vector<cfloat> h0(N, cfloat(0.0f, 0.0f));
        h0[0] = cfloat(3.0f, 4.0f);
        check_double("energy frac, origin",
                     mosse::filter_box_energy_fraction(to_freq(h0).data(), 1,
                                                       Rb, Cb, 4, 4),
                     0.0, 1e-5);

        // Uniform energy over 2 channels: the fraction is the AREA ratio, which
        // catches an off-by-one in either box extent.
        std::vector<cfloat> hu((size_t)2 * N, cfloat(1.0f, 0.0f));
        check_double("energy frac, uniform",
                     mosse::filter_box_energy_fraction(to_freq(hu).data(), 2,
                                                       Rb, Cb, 4, 4),
                     16.0 / 64.0, 1e-5);

        // A full-patch box is the identity; a degenerate box is 0, not a crash.
        check_double("energy frac, full box",
                     mosse::filter_box_energy_fraction(to_freq(hu).data(), 2,
                                                       Rb, Cb, Rb, Cb),
                     1.0, 1e-5);
        check_double("energy frac, zero box",
                     mosse::filter_box_energy_fraction(to_freq(hu).data(), 2,
                                                       Rb, Cb, 0, 4),
                     0.0, 1e-9);

        // THE CASE THE OLD SUITE COULD NOT HAVE: a frequency-domain H whose
        // energy is NOT at the spatial centre reads well below 1.0, and the
        // SAME array read without a transform reads ~0. A function that forgot
        // the transform passes every case above that happens to be symmetric;
        // this one is the direct assertion that the transform is there.
        std::vector<cfloat> hs(N, cfloat(0.0f, 0.0f));
        hs[0] = cfloat(1.0f, 0.0f);              // delta at the spatial ORIGIN
        const std::vector<cfloat> Hs = to_freq(hs);
        const double got = mosse::filter_box_energy_fraction(Hs.data(), 1,
                                                             Rb, Cb, 4, 4);
        check_double("energy frac, transform is applied", got, 0.0, 1e-5);
        // ... and |H| is FLAT for that delta, so a no-transform implementation
        // would return the area ratio 0.25. Assert the two are distinguishable,
        // which is exactly what the hardware run found the hard way.
        check_true("no-transform would differ", std::fabs(got - 0.25) > 0.2,
                   "|H| is flat for a spatial delta, so no transform gives 0.25");

        // The non-power-of-two axis reports NOT MEASURED (negative), never 0.
        check_true("energy frac, npot axis is negative",
                   mosse::filter_box_energy_fraction(Hs.data(), 1, 6, 6, 4, 4) < 0.0,
                   "not measured, never measured as zero");
    }

    // (5) THE INVERSE TRANSFORM ITSELF, against the naive DFT in this file.
    // filter_box_energy_fraction() only exposes a scalar ratio, which can hide a
    // wrong transform that happens to preserve total energy. This asserts the
    // map element by element.
    {
        const int R = 16, C = 16;
        std::vector<cfloat> H((size_t)R * C);
        unsigned s = 12345u;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u;
                           return (float)((int)(s >> 16) % 2000 - 1000) / 1000.0f; };
        for (size_t i = 0; i < H.size(); ++i) H[i] = cfloat(rnd(), rnd());

        std::vector<cfloat> ref;
        maskref::idft2(H, ref, R, C);            // naive, independent, O(n^2)

        // The board's path, reached through the only entry point that uses it:
        // a full-box fraction is 1.0 for ANY transform, so instead compare the
        // ratio on a sub-box against the same ratio computed from `ref`.
        const int br = 6, bc = 6;
        const int r0 = R / 2 - br / 2, c0 = C / 2 - bc / 2;
        double tot = 0.0, in = 0.0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                const cfloat v = ref[(size_t)r * C + c];
                const double m2 = (double)v.real() * v.real()
                                + (double)v.imag() * v.imag();
                tot += m2;
                if (r >= r0 && r < r0 + br && c >= c0 && c < c0 + bc) in += m2;
            }
        check_double("ifft2 matches the naive DFT (box ratio)",
                     mosse::filter_box_energy_fraction(H.data(), 1, R, C, br, bc),
                     in / tot, 1e-5);
    }
}

void run_fusion_tests()
{
    using namespace mosse;
    printf("\n--- filter_update_quantize (fused update + publish) ---\n");

    constexpr int R = 16, C = 16, CH = 4;
    const size_t n = (size_t)R * C, n_all = n * CH;
    const float  eta = 0.125f, eps_rel = 1e-3f;

    Lcg rng(987654);
    std::vector<cfloat> F0(n_all), F1(n_all), G(n);
    auto fill = [&](std::vector<cfloat> &v) {
        for (auto &z : v)
            z = cfloat((float)rng.next(1000) / 1000.0f,
                       (float)rng.next(1000) / 1000.0f);
    };
    fill(F0); fill(F1);
    // A shifted target, so conj(G) != G and a conjugation slip cannot hide.
    gaussian_target_spectrum(G.data(), R, C, 2.0f, 3, -2);

    std::vector<double> energy((size_t)CH);
    for (int ch = 0; ch < CH; ++ch) energy[(size_t)ch] = 0.5 + 0.25 * ch;

    // Two states with the SAME history: init from F0, then one update on F1.
    FilterState ref, fus;
    filter_init(ref, F0.data(), G.data(), CH, R, C);
    filter_init(fus, F0.data(), G.data(), CH, R, C);

    std::vector<int16_t> H_ref(n_all * 2, 0), H_fus(n_all * 2, 0);
    float s_ref = 0.0f, m_ref = 0.0f, s_fus = 0.0f, m_fus = 0.0f;

    filter_update(ref, F1.data(), G.data(), eta);
    filter_quantize_q15(ref, energy.data(), eps_rel, H_ref.data(), &s_ref, &m_ref);

    std::vector<cfloat> h_scratch;
    filter_update_quantize(fus, F1.data(), G.data(), eta, energy.data(), eps_rel,
                           h_scratch, H_fus.data(), &s_fus, &m_fus);

    check_true("fused A bitwise identical",
               memcmp(ref.A.data(), fus.A.data(), n_all * sizeof(cfloat)) == 0,
               "the numerator must not shift by an ulp");
    check_true("fused B bitwise identical",
               memcmp(ref.B.data(), fus.B.data(), n * sizeof(float)) == 0,
               "channel-major accumulation keeps the ch order, so it must");
    check_true("fused H(q15) bitwise identical",
               memcmp(H_ref.data(), H_fus.data(), n_all * 2 * sizeof(int16_t)) == 0,
               "this is the buffer the AIE consumes");
    check_true("fused scale/max|H| identical",
               s_ref == s_fus && m_ref == m_fus, "");
    check_true("fused H reaches full scale", m_fus > 0.0f && s_fus > 0.0f, "");

    // The scratch is the caller's and must be reusable without being cleared —
    // a second call on a shorter state must not read stale tail elements.
    filter_update_quantize(fus, F0.data(), G.data(), eta, energy.data(), eps_rel,
                           h_scratch, H_fus.data(), &s_fus, &m_fus);
    filter_update(ref, F0.data(), G.data(), eta);
    filter_quantize_q15(ref, energy.data(), eps_rel, H_ref.data(), &s_ref, &m_ref);
    check_true("fused identical on reuse",
               memcmp(H_ref.data(), H_fus.data(), n_all * 2 * sizeof(int16_t)) == 0
               && memcmp(ref.A.data(), fus.A.data(), n_all * sizeof(cfloat)) == 0,
               "h_scratch is reused across frames, not reallocated");

    // `energy = nullptr` skips Stage B3 in both paths and must stay equivalent.
    filter_update_quantize(fus, F1.data(), G.data(), eta, nullptr, eps_rel,
                           h_scratch, H_fus.data(), &s_fus, &m_fus);
    filter_update(ref, F1.data(), G.data(), eta);
    filter_quantize_q15(ref, nullptr, eps_rel, H_ref.data(), &s_ref, &m_ref);
    check_true("fused identical without B3",
               memcmp(H_ref.data(), H_fus.data(), n_all * 2 * sizeof(int16_t)) == 0,
               "energy == nullptr path");
}

// -----------------------------------------------------------------------
// scale_update_shifted() — training on the DETECTION sample
// -----------------------------------------------------------------------
// The claim under test: an extraction at box*a^idx is the extraction already in
// hand, shifted idx levels along the scale axis — so training on the sample in
// hand against a target shifted by idx is the same thing, and the second
// scale_extract() of the frame (4.73 ms on the A72) is redundant.
//
// Asserted in the frequency domain, where scale_extract() leaves the sample: a
// shift of idx levels is multiplication by exp(+2*pi*i*m*idx/S). The residual
// that survives is the |idx| levels at the far end of the range, which a real
// re-extraction would have cropped and this cannot — that is the approximation,
// and it is bounded here rather than asserted away.
void run_scale_reuse_tests()
{
    using namespace mosse;
    printf("\n--- scale_update_shifted (reuse the detection sample) ---\n");

    constexpr int S = 33, FR = 256, FC = 256;
    std::vector<uint8_t> frame((size_t)FR * FC, 60);
    for (int r = 96; r < 160; ++r)
        for (int c = 96; c < 160; ++c)
            frame[(size_t)r * FC + c] = 200;

    ScaleFilter sf;
    scale_filter_config(sf, S, 1.02f, 64.0, 64.0, 16.0f);
    const int d = sf.dims();

    std::vector<cfloat> Z((size_t)sf.sample_elems()), Fn((size_t)sf.sample_elems());

    // 1. THE EXTRACTION IDENTITY, which is the whole argument. Extract at `box`
    //    and at `box * a^idx`; the second must be the first multiplied by the
    //    linear phase of an idx-level shift.
    {
        const int idx = 3;
        const double box = 64.0, boxn = box * std::pow((double)sf.step, (double)idx);
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, box,  box,  Z.data());
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, boxn, boxn, Fn.data());

        double scale = 0.0, worst = 0.0;
        for (int l = 0; l < d; ++l)
            for (int m = 0; m < S; ++m) {
                const double ph = 2.0 * M_PI * (double)m * (double)idx / (double)S;
                const cfloat pred = Z[(size_t)l * S + m]
                                  * cfloat((float)std::cos(ph), (float)std::sin(ph));
                const cfloat got  = Fn[(size_t)l * S + m];
                scale = std::max(scale, (double)std::abs(got));
                worst = std::max(worst, (double)std::abs(got - pred));
            }
        // 10% of peak: the residual is the idx levels at the end of the range
        // that a re-extraction sees and a shift cannot. At idx = 3 of 33 that is
        // 9% of the sample, so this bound is the identity holding everywhere it
        // can and failing only where it must.
        check_double("shift identity (idx=+3)", worst / scale, 0.0, 0.10);
    }

    // 2. idx = 0 — 174 of 199 hardware frames — must be EXACT, not close: the
    //    shift is the identity and the code must take the unshifted path.
    {
        ScaleFilter a = sf, b = sf;
        scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, Z.data());
        scale_update(a, Z.data(), 1.0f);
        scale_update_shifted(b, Z.data(), 0, 1.0f);
        check_true("idx=0 bitwise identical",
                   a.st.A.size() == b.st.A.size()
                   && memcmp(a.st.A.data(), b.st.A.data(),
                             a.st.A.size() * sizeof(cfloat)) == 0
                   && memcmp(a.st.B.data(), b.st.B.data(),
                             a.st.B.size() * sizeof(float)) == 0,
                   "the shift is the identity at idx = 0");
    }

    // 3. END TO END: the trained model must be the same whichever way it got
    //    there. Train one filter the old way (re-extract at the resized box) and
    //    one the new way (shift the target), from the same starting state.
    {
        const int idx = 2;
        const double box = 64.0, boxn = box * std::pow((double)sf.step, (double)idx);

        ScaleFilter base;
        scale_filter_config(base, S, 1.02f, 64.0, 64.0, 16.0f);
        scale_extract(base, frame.data(), FR, FC, 128.0, 128.0, box, box, Z.data());
        scale_update(base, Z.data(), 1.0f);

        ScaleFilter old_way = base, new_way = base;
        scale_extract(base, frame.data(), FR, FC, 128.0, 128.0, boxn, boxn, Fn.data());
        scale_update(old_way, Fn.data(), DEFAULT_SCALE_ETA);
        scale_update_shifted(new_way, Z.data(), idx, DEFAULT_SCALE_ETA);

        double sA = 0.0, wA = 0.0;
        for (size_t i = 0; i < old_way.st.A.size(); ++i) {
            sA = std::max(sA, (double)std::abs(old_way.st.A[i]));
            wA = std::max(wA, (double)std::abs(old_way.st.A[i] - new_way.st.A[i]));
        }
        double sB = 0.0, wB = 0.0;
        for (size_t i = 0; i < old_way.st.B.size(); ++i) {
            sB = std::max(sB, (double)std::fabs(old_way.st.B[i]));
            wB = std::max(wB, (double)std::fabs(old_way.st.B[i] - new_way.st.B[i]));
        }
        check_double("trained A matches re-extract", wA / sA, 0.0, 0.05);
        // B sees |F|^2 and the phase ramp is unimodular, so B is affected ONLY by
        // the edge levels — it should agree more tightly than A, and if it does
        // not, the shift is being applied to the magnitude somewhere.
        check_double("trained B matches re-extract", wB / sB, 0.0, 0.02);
    }

    // 4. The detector must still behave: train at 64, present a box 10% too
    //    small, train via the shifted path, and the model must not be corrupted
    //    — the next detection on a correct box still reports level 0.
    {
        ScaleFilter sf2;
        scale_filter_config(sf2, S, 1.02f, 64.0, 64.0, 16.0f);
        scale_extract(sf2, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, Z.data());
        scale_update(sf2, Z.data(), 1.0f);

        scale_extract(sf2, frame.data(), FR, FC, 128.0, 128.0, 58.0, 58.0, Fn.data());
        const ScaleResult rs = scale_detect(sf2, Fn.data(), DEFAULT_EPS_REL);
        check_true("under-sized box still detected", rs.idx > 0, "");
        scale_update_shifted(sf2, Fn.data(), rs.idx, DEFAULT_SCALE_ETA);

        scale_extract(sf2, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, Z.data());
        const ScaleResult r0 = scale_detect(sf2, Z.data(), DEFAULT_EPS_REL);
        check_int("model still reports level 0 after shifted training", r0.idx, 0);
    }
}

// -----------------------------------------------------------------------
// scale_extract()'s real-input DFT
// -----------------------------------------------------------------------
// The sample is real by construction, so its transform along the scale axis must
// be conjugate-symmetric. real_dft_with_table() computes only the first half and
// MIRRORS the rest, which is free only if the mirror is exact — hence a bitwise
// check rather than a tolerance. A tolerance here would pass just as happily on
// an asymmetric twiddle table, which is the thing that can silently break it.
void run_real_dft_tests()
{
    using namespace mosse;
    printf("\n--- scale_extract real-input DFT ---\n");

    constexpr int S = 33, FR = 256, FC = 256;
    std::vector<uint8_t> frame((size_t)FR * FC, 60);
    Lcg rng(4242);
    // Textured, not flat: a flat frame normalises to zero and every bin with it,
    // which would make the symmetry check vacuous.
    for (auto &px : frame) px = (uint8_t)(100 + rng.next(60));   // [40,160]
    for (int r = 96; r < 160; ++r)
        for (int c = 96; c < 160; ++c)
            frame[(size_t)r * FC + c] = (uint8_t)(200 + rng.next(20));

    ScaleFilter sf;
    scale_filter_config(sf, S, 1.02f, 64.0, 64.0, 16.0f);
    const int d = sf.dims();
    std::vector<cfloat> F((size_t)sf.sample_elems());
    scale_extract(sf, frame.data(), FR, FC, 128.0, 128.0, 64.0, 64.0, F.data());

    bool herm = true, nonzero = false;
    for (int l = 0; l < d && herm; ++l)
        for (int k = 1; k <= S / 2; ++k) {
            const cfloat a = F[(size_t)l * S + k];
            const cfloat b = F[(size_t)l * S + (S - k)];
            if (std::abs(a) > 0.0f) nonzero = true;
            if (!(a.real() == b.real() && a.imag() == -b.imag())) { herm = false; break; }
        }
    check_true("spectrum is Hermitian (bitwise)", herm && nonzero,
               "F[S-k] must be exactly conj(F[k])");

    // Bin 0 of a real transform is real, and the DC bin is the sample's sum —
    // which per-level zero-meaning does NOT force to zero, since the mean is
    // removed along the template axis, not the scale axis.
    bool dc_real = true;
    for (int l = 0; l < d; ++l)
        if (F[(size_t)l * S].imag() != 0.0f) { dc_real = false; break; }
    check_true("bin 0 is exactly real", dc_real, "");

    // Round-trip: the inverse transform of a Hermitian spectrum is real. This is
    // what would fail if the mirror were conjugated the wrong way round — a sign
    // error the symmetry check above cannot see, because it only compares the two
    // halves with each other.
    std::vector<cfloat> row((size_t)S), back((size_t)S);
    double worst_im = 0.0, scale = 0.0;
    for (int l = 0; l < d; ++l) {
        for (int k = 0; k < S; ++k) row[(size_t)k] = F[(size_t)l * S + k];
        dft_1d(row.data(), back.data(), S, true);
        for (int k = 0; k < S; ++k) {
            worst_im = std::max(worst_im, (double)std::fabs(back[(size_t)k].imag()));
            scale    = std::max(scale,    (double)std::fabs(back[(size_t)k].real()));
        }
    }
    check_double("inverse transform is real", worst_im / scale, 0.0, 1e-5);
}

void run_training_target_tests()
{
    printf("\n--- training target (filter_update G offset) ---\n");

    // 1. THE SIGN. gaussian_target_spectrum(G,...,dr,dc) must be the spectrum of
    //    a Gaussian peaked at spatial (dr,dc) — the SAME sign as the response's
    //    peak index, not its negation. Everything above rests on this, and a
    //    centred target would hide it (a centred real Gaussian has conj(G) = G).
    {
        using tt::cf;
        std::vector<cf> G((size_t)tt::TR * tt::TC), g((size_t)tt::TR * tt::TC);
        const int DR = 5, DC = -3;
        mosse::gaussian_target_spectrum(G.data(), tt::TR, tt::TC, tt::TSIGMA, DR, DC);
        tt::dft2(G.data(), g.data(), tt::TR, tt::TC, +1);
        int bi = 0; double best = -1.0;
        for (int i = 0; i < tt::TR * tt::TC; ++i)
            if (g[i].real() > best) { best = g[i].real(); bi = i; }
        int pr = bi / tt::TC, pc = bi % tt::TC;
        if (pr > tt::TR / 2) pr -= tt::TR;
        if (pc > tt::TC / 2) pc -= tt::TC;
        check_int("G sign: peak row", pr, DR);
        check_int("G sign: peak col", pc, DC);
    }

    // 2. THE REGRESSION. Same scene, same motion, same eta — only the training
    //    target differs. Centred must let the origin peak grow; shifted must not.
    {
        std::vector<int> dr_s, dc_s;
        const std::vector<double> cen = tt::loop(false, 10, 2, -1, nullptr, nullptr);
        const std::vector<double> shf = tt::loop(true,  10, 2, -1, &dr_s, &dc_s);

        double cen_max = 0.0, shf_max = 0.0;
        for (double v : cen) if (v > cen_max) cen_max = v;
        for (double v : shf) if (v > shf_max) shf_max = v;
        printf("  resp00/peak max: centred %.3f, shifted %.3f\n", cen_max, shf_max);
        printf("  centred:");  for (double v : cen) printf(" %.2f", v); printf("\n");
        printf("  shifted:");  for (double v : shf) printf(" %.2f", v); printf("\n");

        // ASSERT THE SHAPE, NOT AN ABSOLUTE LEVEL. Both arms start at the same
        // value (~0.39 here) — that is this scene's own zero-shift
        // autocorrelation at 32x32 with sigma 1.5, and it is not a defect.
        // CLAUDE.md's 0.3 healthy ceiling is calibrated for the 128x128 ch16
        // hardware geometry and does NOT transfer, the same way s7's PSR
        // threshold does not transfer across geometries. What IS geometry
        // independent is that the defect makes the ratio GROW at the learning
        // rate while the fix leaves it flat or decaying.
        check_true("centred G lets the origin peak grow", cen_max > 1.5 * cen[0],
                   "the defect must still reproduce, or this test proves nothing");
        check_true("shifted G does not let the origin peak grow",
                   shf.back() <= shf[0],
                   "flat or decaying is the property; the absolute level is the "
                   "scene's own baseline");
        check_true("shifted beats centred", shf_max < cen_max, "");

        // The origin peak grows at the LEARNING RATE, which is the signature
        // that separates this from background lock (which BG_PAN would move).
        check_true("centred: resp00/peak is monotone over the first 5 frames",
                   cen.size() >= 5 && cen[4] > cen[0], "");

        // And the fix must actually track: constant velocity (2,-1) per frame.
        bool tracked = !dr_s.empty();
        for (size_t i = 1; i < dr_s.size(); ++i)
            if (dr_s[i] != 2 || dc_s[i] != -1) tracked = false;
        check_true("shifted G tracks constant velocity (2,-1) exactly", tracked,
                   "every frame after the first must report the true step");
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
#if FILTER_MASK
    // The NumPy golden is the UNMASKED filter, so this one comparison cannot
    // hold here — and regenerating it against the mask would make the golden
    // and the implementation share the projection, which is precisely the
    // "self-consistent test on corrupted data" failure this project has already
    // paid for once. The projection is checked instead by
    // run_filter_mask_tests(), against an independent DFT and six mutants.
    //
    // Everything ABOVE this line is upstream of the mask and still checked
    // against the golden: G, A_init, B_init, A_upd, B_upd all hold unchanged.
    (void)read_pod<int16_t>("H_q15.bin", n_all * 2);
    printf("  %-10s SKIP — FILTER_MASK=1: golden is the unmasked filter "
           "(see run_filter_mask_tests)\n", "H_q15");
#else
    check_q15(H, read_pod<int16_t>("H_q15.bin", n_all * 2));
#endif
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
    run_box_tests();
    run_scale_tests();
    run_training_target_tests();
    run_filter_mask_tests();
    run_fusion_tests();
    run_scale_reuse_tests();
    run_real_dft_tests();

    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
