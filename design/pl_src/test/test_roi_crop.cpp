/*
 * test_roi_crop.cpp
 * Native (non-HLS) harness for the roi_crop PL kernel.
 *
 * WHY THIS EXISTS
 * ---------------
 * CLAUDE.md states twice that roi_crop was "verified bit-exact against a NumPy
 * reference in native C simulation (6 cases: 1:1, 2x up, 2x down, both edge-clamp
 * paths, whole frame)". No such harness was ever in the repository: no testbench,
 * no csim target, no reference, and nothing in git history under any roi path.
 *
 * It is worse than untested. Every build to date runs roi_h == patch_rows, which
 * makes step_y exactly 256, hence fy == fx == 0, hence
 *
 *     top = p00 << 8 ;  val = p00 << 16 ;  pix = p00
 *
 * so the ENTIRE bilinear datapath -- ap_uint<18> top, ap_uint<27> val, the >>16 --
 * collapses to a copy. 11 of the 17 cases here execute it for the first time.
 *
 * TOLERANCE IS ZERO, unlike test_host's 1-LSB slack on H_q15. This is an integer
 * datapath end to end; any difference is a bug, not rounding.
 *
 * The assertion contract is copied from design/host_app_src/test/
 * test_mosse_filter.cpp:172-195 -- helpers print one line, bump g_failures, and
 * never abort, so one broken case does not hide the other sixteen.
 *
 * Build/run:  make test_roi_crop
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

#include "roi_crop.h"

namespace {

int         g_failures = 0;
std::string g_dir;

std::string path(const std::string &name) { return g_dir + "/" + name; }

std::vector<uint8_t> read_all(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------
// Assertion helpers — same contract as test_mosse_filter.cpp:172-195.
// ---------------------------------------------------------------------------
void check_true(const char *what, bool cond, const std::string &detail = "")
{
    printf("    %-30s %s%s%s\n", what, cond ? "OK  " : "FAIL",
           detail.empty() ? "" : " — ", detail.c_str());
    if (!cond) ++g_failures;
}

void check_long(const char *what, long got, long exp)
{
    const bool ok = (got == exp);
    printf("    %-30s %s — got %ld, expected %ld\n", what, ok ? "OK  " : "FAIL",
           got, exp);
    if (!ok) ++g_failures;
}

// Exact byte comparison. Reports the first mismatch and a histogram of how far
// off things are, because "the output differs" is not an actionable diagnostic
// for arithmetic that has never run.
void check_i8(const char *what, const int8_t *got, const int8_t *exp, size_t n,
              int patch_cols)
{
    size_t ndiff = 0, first = 0;
    int    maxdiff = 0;
    bool   found = false;
    for (size_t i = 0; i < n; ++i) {
        const int d = (int)got[i] - (int)exp[i];
        if (d) {
            if (!found) { first = i; found = true; }
            ++ndiff;
            if (std::abs(d) > maxdiff) maxdiff = std::abs(d);
        }
    }
    if (!ndiff) {
        printf("    %-30s OK   — %zu/%zu samples identical\n", what, n, n);
        return;
    }
    ++g_failures;
    printf("    %-30s FAIL — %zu/%zu differ (%.2f%%), max |diff| %d\n",
           what, ndiff, n, 100.0 * (double)ndiff / (double)n, maxdiff);
    printf("      first at idx %zu (r=%d, c=%d): got %d, expected %d\n",
           first, (int)(first / (size_t)patch_cols), (int)(first % (size_t)patch_cols),
           (int)got[first], (int)exp[first]);
}

// ---------------------------------------------------------------------------
// Case parameters
// ---------------------------------------------------------------------------
struct Case {
    std::string name, why;
    int frame_rows = 0, frame_cols = 0;
    int roi_row = 0, roi_col = 0, roi_h = 0, roi_w = 0;
    int patch_rows = 0, patch_cols = 0;
    int twice = 0;
    long exp_mean = 0, exp_var = 0, exp_inv_q = 0, exp_beats = 0;
    long step_y = 0, step_x = 0, fy_nonzero = 0, fx_nonzero = 0;
};

Case load_case(const std::string &name)
{
    Case c;
    c.name = name;
    std::ifstream f(path(name + ".txt"));
    if (!f) { fprintf(stderr, "FATAL: cannot open %s.txt\n", name.c_str()); exit(2); }
    std::string key;
    while (f >> key) {
        if (key == "why" || key == "name") { std::getline(f, c.why); continue; }
        long v = 0;
        f >> v;
        if      (key == "frame_rows") c.frame_rows = (int)v;
        else if (key == "frame_cols") c.frame_cols = (int)v;
        else if (key == "roi_row")    c.roi_row    = (int)v;
        else if (key == "roi_col")    c.roi_col    = (int)v;
        else if (key == "roi_h")      c.roi_h      = (int)v;
        else if (key == "roi_w")      c.roi_w      = (int)v;
        else if (key == "patch_rows") c.patch_rows = (int)v;
        else if (key == "patch_cols") c.patch_cols = (int)v;
        else if (key == "twice")      c.twice      = (int)v;
        else if (key == "exp_mean")   c.exp_mean   = v;
        else if (key == "exp_var")    c.exp_var    = v;
        else if (key == "exp_inv_q")  c.exp_inv_q  = v;
        else if (key == "exp_beats")  c.exp_beats  = v;
        else if (key == "step_y")     c.step_y     = v;
        else if (key == "step_x")     c.step_x     = v;
        else if (key == "fy_nonzero") c.fy_nonzero = v;
        else if (key == "fx_nonzero") c.fx_nonzero = v;
        // Unknown keys ignored, matching test_mosse_filter.cpp:449-455, so the
        // generator can add diagnostics without breaking an older harness.
    }
    return c;
}

// ---------------------------------------------------------------------------
// Drain the AXIS stream and unpack. Also asserts the framing contract, which is
// independently restated in roi_crop.h:26-30, phase1_sweep.py's unpack and
// gen_aiesim_vectors.py's write_plio_txt -- three statements of one convention
// with nothing checking they agree.
// ---------------------------------------------------------------------------
void drain(hls::stream<ap_axiu<32, 0, 0, 0>> &s, std::vector<int8_t> &out,
           long expect_beats, const char *tag, bool check_framing)
{
    out.clear();
    long beats = 0, last_count = 0, bad_keep = 0;
    long last_at = -1;
    while (!s.empty()) {
        ap_axiu<32, 0, 0, 0> w = s.read();
        for (int i = 0; i < 4; ++i)
            out.push_back((int8_t)(uint8_t)(w.data.range(8 * i + 7, 8 * i)));
        if (w.last) { ++last_count; last_at = beats; }
        if (w.keep != (ap_uint<4>)-1 || w.strb != (ap_uint<4>)-1) ++bad_keep;
        ++beats;
    }
    if (!check_framing) return;
    char buf[64];
    snprintf(buf, sizeof(buf), "%s beats", tag);
    check_long(buf, beats, expect_beats);
    snprintf(buf, sizeof(buf), "%s last flag", tag);
    check_true(buf, last_count == 1 && last_at == expect_beats - 1,
               "count " + std::to_string(last_count) + ", at beat " +
               std::to_string(last_at));
    snprintf(buf, sizeof(buf), "%s keep/strb", tag);
    check_true(buf, bad_keep == 0, std::to_string(bad_keep) + " bad beats");
}

void run_case(const std::string &name)
{
    const Case c = load_case(name);
    const std::vector<uint8_t> frame = read_all(path(name + "_frame.bin"));
    const std::vector<uint8_t> exp_raw = read_all(path(name + "_patch.bin"));

    const size_t n_frame = (size_t)c.frame_rows * c.frame_cols;
    const size_t n_patch = (size_t)c.patch_rows * c.patch_cols;
    if (frame.size() != n_frame || exp_raw.size() != n_patch) {
        fprintf(stderr, "FATAL: %s size mismatch (frame %zu/%zu, patch %zu/%zu)\n",
                name.c_str(), frame.size(), n_frame, exp_raw.size(), n_patch);
        exit(2);
    }
    const int8_t *exp = reinterpret_cast<const int8_t *>(exp_raw.data());

    printf("  %-18s roi %dx%d @(%d,%d) -> patch %dx%d  step %ld/%ld  fy nz %ld\n",
           c.name.c_str(), c.roi_h, c.roi_w, c.roi_row, c.roi_col,
           c.patch_rows, c.patch_cols, c.step_y, c.step_x, c.fy_nonzero);

    std::vector<ap_uint<8>> fbuf(n_frame);
    for (size_t i = 0; i < n_frame; ++i) fbuf[i] = frame[i];

    hls::stream<ap_axiu<32, 0, 0, 0>> out_s;
    roi_crop(fbuf.data(), out_s, c.frame_rows, c.frame_cols,
             c.roi_row, c.roi_col, c.roi_h, c.roi_w,
             c.patch_rows, c.patch_cols, /*recompute=*/1);

    std::vector<int8_t> got;
    drain(out_s, got, c.exp_beats, "axis", true);

    if (got.size() != n_patch) {
        check_true("patch size", false,
                   std::to_string(got.size()) + " vs " + std::to_string(n_patch));
        return;
    }
    check_i8("patch bit-exact", got.data(), exp, n_patch, c.patch_cols);

    if (c.twice) {
        // recompute=0 must re-stream the static buffer byte-identically. The
        // contract is roi_crop.h:83-98 and nothing has ever checked it.
        hls::stream<ap_axiu<32, 0, 0, 0>> out2;
        roi_crop(fbuf.data(), out2, c.frame_rows, c.frame_cols,
                 c.roi_row, c.roi_col, c.roi_h, c.roi_w,
                 c.patch_rows, c.patch_cols, /*recompute=*/0);
        std::vector<int8_t> got2;
        drain(out2, got2, c.exp_beats, "cached axis", true);
        check_true("cached == recomputed",
                   got2.size() == got.size() &&
                   memcmp(got2.data(), got.data(), got.size()) == 0);
    }
}

}  // namespace

int main(int argc, char **argv)
{
    g_dir = (argc > 1) ? argv[1] : "golden";

    printf("\nroi_crop native harness — golden: %s\n", g_dir.c_str());
    printf("NOTE: 11 of these cases execute the bilinear interpolator, which has\n"
           "      never run in any build (roi_h == patch_rows makes fy == fx == 0).\n\n");

    std::ifstream mf(path("manifest.txt"));
    if (!mf) { fprintf(stderr, "FATAL: no manifest.txt in %s\n", g_dir.c_str()); exit(2); }
    int n = 0;
    mf >> n;
    std::vector<std::string> names;
    std::string s;
    while (mf >> s) names.push_back(s);
    if ((int)names.size() != n) {
        fprintf(stderr, "FATAL: manifest says %d cases, lists %zu\n", n, names.size());
        exit(2);
    }

    for (const std::string &name : names) run_case(name);

    printf("\n  OVERALL: %s (%d failure%s over %d case%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures,
           g_failures == 1 ? "" : "s", n, n == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
