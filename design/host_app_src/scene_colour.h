/*
 * scene_colour.h
 * Luma scene -> the interleaved buffer the device reads.
 *
 * NO XRT / ADF HEADER, for the same reason mosse_filter.{h,cpp} has none: it
 * makes the code compilable by system g++ in seconds (`make test_scene`), and
 * the alternative for a colour or indexing error is a full build-flash-run.
 *
 * WHY THIS EXISTS AS A MODULE
 * ---------------------------
 * The scene is generated in LUMA — background, pan/dirty-rect restore, target
 * injection, sensor noise, the occluder, and scale_extract's 33 crops are all
 * single-plane and stay that way. At CONV_IN_CH=3 roi_crop reads a
 * PIXEL-INTERLEAVED RGB frame, so one pass expands the luma scene into that
 * layout on the way to the device.
 *
 * Only the region that changed is expanded. The frame is 2.07 M pixels and a
 * full pass every frame would cost more than the push it feeds; the scene
 * machinery already tracks what changed, so this is the same argument that makes
 * the dirty-rect restore cheap, applied one stage later.
 *
 * THE INVARIANT, AND WHY IT NEEDS AN INSTRUMENT
 * --------------------------------------------
 * The incremental pass is correct only if EVERY luma write is unioned into the
 * touched rect. Miss one and the device reads the PREVIOUS frame's colour there:
 * the tracker still produces a peak, the run still completes, and the result is
 * a slightly worse number rather than a visible failure. That is the shape of
 * bug this design has paid for repeatedly, so verify() re-expands the whole
 * frame and reports the disagreement instead of leaving it to be inferred from
 * an IoU column.
 *
 * verify() and colourise() share colourise_rect(), deliberately: a verifier
 * carrying its own copy of the colour rule proves only that two copies of the
 * rule agree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scene {

// Inclusive rect; empty when r1 < r0. Field names match the tracker's DirtyRect,
// which is an alias for this type — one rect vocabulary, not two.
struct Rect {
    int r0 = 0, c0 = 0, r1 = -1, c1 = -1;
    bool empty() const { return r1 < r0; }
};

// acc |= rect. An empty acc takes the rect outright rather than unioning with
// the (0,0,-1,-1) sentinel, which would drag r0/c0 to 0.
void rect_union(Rect &acc, int r0, int c0, int r1, int c1);

// Colour modes for the SYNTHETIC scene. A real frame source supplies its own
// colour and calls none of this.
enum Mode {
    MODE_REPLICATE = 0,  // all planes carry luma — the colour-free control
    MODE_TINT      = 1,  // per-plane gain, with a distinct hue inside `target`
};

// Q8 per-plane gains. Background cool, target warm. Mild on purpose: a
// saturating tint would clip at 255 and hand Stage A a flat plane, which is the
// one outcome that would make the RGB path look broken for a reason that is not
// the RGB path.
extern const int kBgGain[3];
extern const int kTgtGain[3];

// v * gain_q8, rounded, saturating at 255. Never wraps.
uint8_t colour_mul(uint8_t v, int gain_q8);

// Expand [r0..r1] x [c0..c1] of `lum` into `dst`. The rect is clipped to the
// frame, so an out-of-range rect is a no-op on the out-of-range part rather than
// an overrun. `planes` is 1 (straight copy) or 3 (interleaved R G B).
// `dst` is indexed [r][c][plane]; `lum` is [r][c].
void colourise_rect(uint8_t *dst, const uint8_t *lum,
                    int rows, int cols, int planes, int mode,
                    const Rect &target,
                    int r0, int c0, int r1, int c1);

// Expand `touched`, then clear it. No-op when `touched` is empty.
void colourise(uint8_t *dst, const uint8_t *lum,
               int rows, int cols, int planes, int mode,
               const Rect &target, Rect &touched);

// Re-expand the WHOLE frame into `scratch` and compare against `dst`.
// Returns the number of differing bytes (0 = the incremental result is exact)
// and, when non-zero and `first_bad` is non-null, the index of the first one.
std::size_t verify(const uint8_t *dst, const uint8_t *lum,
                   int rows, int cols, int planes, int mode,
                   const Rect &target, std::vector<uint8_t> &scratch,
                   std::size_t *first_bad);

}  // namespace scene
