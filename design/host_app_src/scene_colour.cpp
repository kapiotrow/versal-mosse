/*
 * scene_colour.cpp — see scene_colour.h for what this is and why it is its own
 * translation unit.
 *
 * Integer only. No float appears anywhere in the colour path, so there is no
 * FMA-contraction question here and no need for the second -ffp-contract build
 * that make test_host runs for mosse_filter.
 *
 * @thesis subsec:kosztTransferow | P-04 | The luma-to-interleaved-RGB pass, incremental over
 *   the touched rect: colourising the whole 2.07 Mpx frame every frame is what RGB would
 *   otherwise cost.
 */

#include "scene_colour.h"

#include <algorithm>
#include <cstring>

namespace scene {

// 0.90 / 1.00 / 1.10 and 1.20 / 0.95 / 0.70, in Q8.
const int kBgGain[3]  = { 230, 256, 282 };
const int kTgtGain[3] = { 307, 243, 179 };

void rect_union(Rect &acc, int r0, int c0, int r1, int c1)
{
    if (acc.empty()) { acc = Rect{r0, c0, r1, c1}; return; }
    acc.r0 = std::min(acc.r0, r0);  acc.c0 = std::min(acc.c0, c0);
    acc.r1 = std::max(acc.r1, r1);  acc.c1 = std::max(acc.c1, c1);
}

uint8_t colour_mul(uint8_t v, int gain_q8)
{
    const int x = ((int)v * gain_q8 + 128) >> 8;
    return (uint8_t)(x > 255 ? 255 : x);
}

void colourise_rect(uint8_t *dst, const uint8_t *lum,
                    int rows, int cols, int planes, int mode,
                    const Rect &target,
                    int r0, int c0, int r1, int c1)
{
    // Clip, do not trust. The caller's rects come from target geometry and a
    // pan offset and routinely run off the frame; an unclipped rect here is an
    // out-of-bounds write into the buffer that is about to be DMA'd.
    r0 = std::max(0, r0);  r1 = std::min(rows - 1, r1);
    c0 = std::max(0, c0);  c1 = std::min(cols - 1, c1);
    if (r1 < r0 || c1 < c0) return;

    for (int r = r0; r <= r1; ++r) {
        const uint8_t *lrow = lum + (std::size_t)r * cols;
        uint8_t       *drow = dst + (std::size_t)r * cols * planes;

        if (planes == 1) {
            std::memcpy(drow + c0, lrow + c0, (std::size_t)(c1 - c0 + 1));
            continue;
        }

        if (mode == MODE_REPLICATE) {
            for (int c = c0; c <= c1; ++c) {
                const uint8_t v = lrow[c];
                for (int p = 0; p < planes; ++p) drow[planes * c + p] = v;
            }
            continue;
        }

        // Hoisted: the row test does not depend on c, so a target that does not
        // span this row costs one comparison for the whole row.
        const bool row_in_tgt = !target.empty() &&
                                r >= target.r0 && r <= target.r1;
        for (int c = c0; c <= c1; ++c) {
            const bool in_tgt = row_in_tgt && c >= target.c0 && c <= target.c1;
            const int *g = in_tgt ? kTgtGain : kBgGain;
            const uint8_t v = lrow[c];
            for (int p = 0; p < planes; ++p)
                drow[planes * c + p] = colour_mul(v, g[p]);
        }
    }
}

void colourise(uint8_t *dst, const uint8_t *lum,
               int rows, int cols, int planes, int mode,
               const Rect &target, Rect &touched)
{
    if (touched.empty()) return;
    colourise_rect(dst, lum, rows, cols, planes, mode, target,
                   touched.r0, touched.c0, touched.r1, touched.c1);
    touched = Rect{};
}

std::size_t verify(const uint8_t *dst, const uint8_t *lum,
                   int rows, int cols, int planes, int mode,
                   const Rect &target, std::vector<uint8_t> &scratch,
                   std::size_t *first_bad)
{
    const std::size_t n = (std::size_t)rows * cols * planes;
    scratch.assign(n, 0);
    colourise_rect(scratch.data(), lum, rows, cols, planes, mode, target,
                   0, 0, rows - 1, cols - 1);

    if (std::memcmp(scratch.data(), dst, n) == 0) return 0;

    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (scratch[i] != dst[i]) { if (!bad) first = i; ++bad; }
    if (first_bad) *first_bad = first;
    return bad;
}

}  // namespace scene
