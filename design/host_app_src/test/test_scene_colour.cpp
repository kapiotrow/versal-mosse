/*
 * test_scene_colour.cpp
 * Native harness for scene_colour.{h,cpp} — the luma -> interleaved-RGB pass.
 *
 * WHY THIS EXISTS
 * ---------------
 * The colour pass lives between the scene generator and a DMA. Both of its
 * failure modes are silent: a wrong interleave or gain hands conv2d a plausible
 * feature map, and a missed touch hands it LAST FRAME's colour in one region.
 * Neither shows up as anything but a slightly worse IoU, which is exactly the
 * class of result CLAUDE.md records as unfalsifiable.
 *
 * The pass used to live inside mosse_tracker.cpp, which includes XRT and so
 * cannot be compiled off-board at all. Splitting it out follows the precedent
 * mosse_filter.{h,cpp} set: no XRT header, seconds to build, and a sign error
 * costs a compile instead of a build-flash-run.
 *
 * TOLERANCE IS ZERO. Every value here is an integer.
 *
 * Assertion contract copied from test_roi_crop.cpp: helpers print one line, bump
 * g_failures, and never abort, so one broken case cannot hide the others.
 *
 * Build/run:  make test_scene
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "scene_colour.h"

namespace {

int g_failures = 0;

void check(const char *what, bool cond, const std::string &detail = "")
{
    printf("    %-46s %s%s%s\n", what, cond ? "OK  " : "FAIL",
           detail.empty() ? "" : " — ", detail.c_str());
    if (!cond) ++g_failures;
}

void hdr(const char *title)
{
    printf("\n  %s\n", title);
}

// A luma frame with structure in both axes, so a transposed or plane-swapped
// index cannot accidentally agree. Deliberately spans the saturating end of the
// tint (values near 255 * 307/256 = 306) so the clamp is exercised by ordinary
// cases rather than only by the dedicated one.
std::vector<uint8_t> make_luma(int rows, int cols, unsigned seed)
{
    std::vector<uint8_t> v((size_t)rows * cols);
    unsigned s = seed;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            s = s * 1664525u + 1013904223u;
            v[(size_t)r * cols + c] =
                (uint8_t)((r * 7 + c * 13 + (int)((s >> 16) & 0x3F)) & 0xFF);
        }
    return v;
}

const int ROWS = 24, COLS = 32, P = 3;

// ---------------------------------------------------------------------------

void t_rect_union()
{
    hdr("rect_union");
    scene::Rect a;
    check("starts empty", a.empty());
    scene::rect_union(a, 5, 6, 7, 8);
    // An empty accumulator must TAKE the rect. Unioning against the
    // (0,0,-1,-1) sentinel would drag r0/c0 to 0 and silently widen every
    // first union of a frame to the top-left corner.
    check("empty acc takes the rect, does not union with the sentinel",
          a.r0 == 5 && a.c0 == 6 && a.r1 == 7 && a.c1 == 8,
          std::to_string(a.r0) + "," + std::to_string(a.c0) + "," +
          std::to_string(a.r1) + "," + std::to_string(a.c1));
    scene::rect_union(a, 2, 9, 6, 20);
    check("union extends both ends",
          a.r0 == 2 && a.c0 == 6 && a.r1 == 7 && a.c1 == 20);
}

void t_colour_mul()
{
    hdr("colour_mul — Q8, rounded, saturating");
    check("unity gain is identity", scene::colour_mul(137, 256) == 137);
    check("zero stays zero", scene::colour_mul(0, 307) == 0);
    // (200*230 + 128) >> 8 = 46128 >> 8 = 180. Truncating instead of rounding
    // would give 179, so this value discriminates the two.
    check("rounds half up, not toward zero", scene::colour_mul(200, 230) == 180,
          std::to_string(scene::colour_mul(200, 230)));
    // A second, independently hand-checked point: (77*282 + 128) >> 8
    // = (21714 + 128) >> 8 = 21842 >> 8 = 85 (truncating would also give 85,
    // so it checks the gain and not the rounding).
    check("background blue gain on 77 is 85", scene::colour_mul(77, 282) == 85,
          std::to_string(scene::colour_mul(77, 282)));
    // 255 * 307 >> 8 = 305 -> must clamp, NOT wrap to 49.
    check("saturates at 255, never wraps", scene::colour_mul(255, 307) == 255,
          std::to_string(scene::colour_mul(255, 307)));
    bool mono = true;
    for (int v = 1; v < 256; ++v)
        if (scene::colour_mul((uint8_t)(v - 1), 282) >
            scene::colour_mul((uint8_t)v, 282)) mono = false;
    check("monotone in v", mono);
}

void t_planes1_is_identity()
{
    hdr("planes = 1 — the grayscale path is a straight copy");
    const auto lum = make_luma(ROWS, COLS, 11);
    std::vector<uint8_t> dst((size_t)ROWS * COLS, 0xAA);
    scene::colourise_rect(dst.data(), lum.data(), ROWS, COLS, 1,
                          scene::MODE_TINT, scene::Rect{2, 3, 9, 9},
                          0, 0, ROWS - 1, COLS - 1);
    check("identical to luma, tint ignored at 1 plane",
          std::memcmp(dst.data(), lum.data(), lum.size()) == 0);
}

void t_replicate()
{
    hdr("MODE_REPLICATE — the colour-free control");
    const auto lum = make_luma(ROWS, COLS, 12);
    std::vector<uint8_t> dst((size_t)ROWS * COLS * P, 0);
    scene::colourise_rect(dst.data(), lum.data(), ROWS, COLS, P,
                          scene::MODE_REPLICATE, scene::Rect{2, 3, 9, 9},
                          0, 0, ROWS - 1, COLS - 1);
    bool ok = true, interleaved = true;
    for (int r = 0; r < ROWS && ok; ++r)
        for (int c = 0; c < COLS && ok; ++c) {
            const uint8_t v = lum[(size_t)r * COLS + c];
            const uint8_t *px = &dst[((size_t)r * COLS + c) * P];
            if (px[0] != v || px[1] != v || px[2] != v) ok = false;
        }
    check("all three planes carry luma", ok);
    // The target rect must NOT matter in replicate mode — if it does, the
    // control arm is not colour-free and comparing against it means nothing.
    std::vector<uint8_t> dst2((size_t)ROWS * COLS * P, 0);
    scene::colourise_rect(dst2.data(), lum.data(), ROWS, COLS, P,
                          scene::MODE_REPLICATE, scene::Rect{},
                          0, 0, ROWS - 1, COLS - 1);
    check("target rect is ignored — the control really is colour-free",
          dst == dst2);
    (void)interleaved;
}

void t_tint()
{
    hdr("MODE_TINT — per-plane gains, target hue distinct from background");
    const auto lum = make_luma(ROWS, COLS, 13);
    const scene::Rect tgt{6, 8, 11, 15};
    std::vector<uint8_t> dst((size_t)ROWS * COLS * P, 0);
    scene::colourise_rect(dst.data(), lum.data(), ROWS, COLS, P,
                          scene::MODE_TINT, tgt, 0, 0, ROWS - 1, COLS - 1);

    bool bg_ok = true, tg_ok = true;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            const bool in = r >= tgt.r0 && r <= tgt.r1 && c >= tgt.c0 && c <= tgt.c1;
            const int *g = in ? scene::kTgtGain : scene::kBgGain;
            const uint8_t v = lum[(size_t)r * COLS + c];
            const uint8_t *px = &dst[((size_t)r * COLS + c) * P];
            for (int p = 0; p < P; ++p)
                if (px[p] != scene::colour_mul(v, g[p])) {
                    (in ? tg_ok : bg_ok) = false;
                }
        }
    check("background gains applied outside the target", bg_ok);
    check("target gains applied inside the target", tg_ok);

    // The whole point of the tint is that the planes DIFFER. If they do not,
    // joint normalization has nothing to preserve and an RGB run against this
    // scene would be measuring the control by accident.
    long dRG = 0, dGB = 0;
    for (size_t i = 0; i < (size_t)ROWS * COLS; ++i) {
        dRG += std::abs((int)dst[i * P + 0] - (int)dst[i * P + 1]);
        dGB += std::abs((int)dst[i * P + 1] - (int)dst[i * P + 2]);
    }
    check("planes are genuinely different", dRG > 0 && dGB > 0,
          "sum|R-G| " + std::to_string(dRG) + ", sum|G-B| " + std::to_string(dGB));

    // ...and the target's hue must differ from the background's, or the target
    // is invisible in chroma and only the luma edge carries it.
    const uint8_t v = 200;
    const bool hue_differs =
        scene::colour_mul(v, scene::kTgtGain[0]) != scene::colour_mul(v, scene::kBgGain[0]) &&
        scene::colour_mul(v, scene::kTgtGain[2]) != scene::colour_mul(v, scene::kBgGain[2]);
    check("target hue differs from background hue", hue_differs);
}

void t_clipping()
{
    hdr("rect clipping — no overrun, no wrap onto the next row");
    const auto lum = make_luma(ROWS, COLS, 14);
    const size_t n = (size_t)ROWS * COLS * P;
    // The padding must be LARGER than the worst over-range write this test
    // makes, or a dropped clip runs past the allocation, corrupts the heap and
    // aborts before the canary check can name the problem. Measured: dropping
    // the r1 clip writes 41 rows * COLS * P = 3936 bytes past the end. An abort
    // does catch the bug, but a named FAIL says which check found it.
    const size_t pad = 8192;
    std::vector<uint8_t> buf(n + 2 * pad, 0x5A);
    uint8_t *dst = buf.data() + pad;
    std::memset(dst, 0, n);

    // Every edge out of range at once, plus a rect entirely off-frame.
    scene::colourise_rect(dst, lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                          scene::Rect{}, -50, -70, ROWS + 40, COLS + 90);
    bool canary_ok = true;
    for (size_t i = 0; i < pad; ++i)
        if (buf[i] != 0x5A || buf[pad + n + i] != 0x5A) canary_ok = false;
    check("canaries intact after an over-range rect", canary_ok);

    std::vector<uint8_t> before(dst, dst + n);
    scene::colourise_rect(dst, lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                          scene::Rect{}, ROWS + 5, 0, ROWS + 9, COLS - 1);
    check("fully off-frame rect writes nothing",
          std::memcmp(before.data(), dst, n) == 0);

    // A partial rect must touch its rows ONLY. Row stride is cols*planes; an
    // off-by-one there writes into the next row and looks like a smear.
    std::memset(dst, 0, n);
    scene::colourise_rect(dst, lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                          scene::Rect{}, 4, 2, 6, 5);
    bool outside_untouched = true;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            const bool in = r >= 4 && r <= 6 && c >= 2 && c <= 5;
            const uint8_t *px = &dst[((size_t)r * COLS + c) * P];
            const bool written = px[0] || px[1] || px[2];
            if (!in && written) outside_untouched = false;
        }
    check("a partial rect writes inside its bounds only", outside_untouched);
}

void t_incremental_matches_full()
{
    hdr("incremental == full — the invariant the device depends on");
    auto lum = make_luma(ROWS, COLS, 15);
    const scene::Rect tgt{9, 11, 14, 18};
    std::vector<uint8_t> dst((size_t)ROWS * COLS * P, 0), scratch;

    // Startup: expand everything once.
    scene::Rect touched;
    scene::rect_union(touched, 0, 0, ROWS - 1, COLS - 1);
    scene::colourise(dst.data(), lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                     tgt, touched);
    check("touched is cleared by colourise", touched.empty());

    size_t first = 0;
    check("full-frame expansion verifies",
          scene::verify(dst.data(), lum.data(), ROWS, COLS, P,
                        scene::MODE_TINT, tgt, scratch, &first) == 0);

    // Three frames of edits, each correctly unioned into `touched`.
    for (int f = 0; f < 3; ++f) {
        const int r0 = 3 + f * 2, c0 = 5 + f * 3, r1 = r0 + 4, c1 = c0 + 6;
        for (int r = r0; r <= r1; ++r)
            for (int c = c0; c <= c1; ++c)
                lum[(size_t)r * COLS + c] = (uint8_t)(40 * f + r * 3 + c);
        scene::rect_union(touched, r0, c0, r1, c1);
        scene::colourise(dst.data(), lum.data(), ROWS, COLS, P,
                         scene::MODE_TINT, tgt, touched);
    }
    const size_t bad = scene::verify(dst.data(), lum.data(), ROWS, COLS, P,
                                     scene::MODE_TINT, tgt, scratch, &first);
    check("3 correctly-touched edits still verify", bad == 0,
          std::to_string(bad) + " bytes differ");
}

void t_missed_touch_is_caught()
{
    hdr("THE VERIFIER FIRES — a luma write that skipped scene_touch()");
    auto lum = make_luma(ROWS, COLS, 16);
    const scene::Rect tgt{9, 11, 14, 18};
    std::vector<uint8_t> dst((size_t)ROWS * COLS * P, 0), scratch;

    scene::Rect touched;
    scene::rect_union(touched, 0, 0, ROWS - 1, COLS - 1);
    scene::colourise(dst.data(), lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                     tgt, touched);

    // Write luma and DO NOT union it. This is the real bug, reproduced: the
    // device would keep the previous frame's colour over this rect.
    const int r0 = 17, c0 = 20, r1 = 19, c1 = 25;
    for (int r = r0; r <= r1; ++r)
        for (int c = c0; c <= c1; ++c)
            lum[(size_t)r * COLS + c] = (uint8_t)(200 + ((r + c) & 31));
    scene::colourise(dst.data(), lum.data(), ROWS, COLS, P, scene::MODE_TINT,
                     tgt, touched);   // touched is empty: a no-op, as in the bug

    size_t first = 0;
    const size_t bad = scene::verify(dst.data(), lum.data(), ROWS, COLS, P,
                                     scene::MODE_TINT, tgt, scratch, &first);
    check("verify() reports the missed region", bad > 0,
          std::to_string(bad) + " bytes differ");

    const size_t px = first / P;
    const int fr = (int)(px / COLS), fc = (int)(px % COLS);
    check("first mismatch is inside the missed rect",
          fr >= r0 && fr <= r1 && fc >= c0 && fc <= c1,
          "r=" + std::to_string(fr) + " c=" + std::to_string(fc));

    // And it must be confined to that rect — a verifier that flags the whole
    // frame localises nothing.
    const size_t rect_bytes = (size_t)(r1 - r0 + 1) * (c1 - c0 + 1) * P;
    check("mismatch is confined to the missed rect", bad <= rect_bytes,
          std::to_string(bad) + " <= " + std::to_string(rect_bytes));
}

}  // namespace

int main()
{
    printf("\nscene_colour native harness — zero tolerance, integer datapath\n");
    t_rect_union();
    t_colour_mul();
    t_planes1_is_identity();
    t_replicate();
    t_tint();
    t_clipping();
    t_incremental_matches_full();
    t_missed_touch_is_caught();
    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures,
           g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
