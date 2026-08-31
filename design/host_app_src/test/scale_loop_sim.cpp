/*
 * scale_loop_sim.cpp — closed-loop DSST scale simulation, no hardware.
 *
 * WHY THIS EXISTS. Hardware run runs/run_0820_1418.log, 200 frames,
 * TRAJECTORY=1 SCALE_TRAJ=1: the scale estimate tracked the size envelope well
 * for 130 frames (64 -> 82.79 at f45 against a truth of 83, then back down),
 * and then FROZE. est_h = 59.13 exactly, frames 130..199 — 70 consecutive
 * frames, never moving in either direction, while truth went 48 -> 45 (f150)
 * -> 63 (f199), passing back THROUGH 59.13 around f190 with no response.
 *
 * The gate is not the cause: 199 accepted, 0 held, conf 2.15..3.31 against a
 * 2.00 threshold. Nor is it the runaway scale_gate() was built for — it is the
 * opposite failure, a FROZEN, CONFIDENT, WRONG scale. Position drift is
 * downstream: centre error stayed under 1.6 px through f120, while the scale
 * was already 15% off, then grew monotonically 1.26 -> 10.97 px after the
 * freeze. PSR 53..135 and resp00_over_peak <= 0.03 throughout — the tracker was
 * confidently locked to a mis-sized ROI the whole time.
 *
 * HYPOTHESIS UNDER TEST. mosse_filter.h already records that a single detection
 * UNDER-CORRECTS (+3 where +5 is exact, because the response is a discrete peak
 * smoothed by a sigma = S/16 target), and run_scale_tests() asserts only that
 * REPEATED application converges monotonically — against a STATIC target. Under
 * a continuously moving size, per-frame under-correction plus training the model
 * on the current (wrong) box can stall the estimate at a fixed offset. That fits
 * the data. It is not established, which is what this program is for.
 *
 * NOTE ON THE SCALE MODEL UPDATE, checked rather than assumed: the tracker
 * re-extracts the sample at the UPDATED box before scale_update(), so the
 * level-0 training target is correct here. This is NOT the centred-G defect one
 * axis down — that one is already handled by the re-extraction.
 *
 * WHAT MAKES THE COUNTERFACTUALS MEAN ANYTHING: arm `moving` must REPRODUCE the
 * hardware freeze first. If it does not, the scene or the geometry has drifted
 * from the tracker's and nothing below can be trusted. The program says so
 * explicitly rather than printing a verdict either way.
 *
 * Usage:
 *   make scale_sim && build/.../scale_loop_sim              # all arms
 *   build/.../scale_loop_sim --arm moving --frames 200 -v   # per-frame trace
 *   build/.../scale_loop_sim --arm moving --eta 0.1         # sweep the fix
 *
 * @thesis subsec:filtrSkali | A-09,N-17 | The closed-loop scale bench that reproduces the
 *   board's f130 stall and that decided SCALE_STEP and SCALE_MAX_STEP. Trust its ordering, not
 *   its absolute magnitudes.
 */
#include "mosse_filter.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int    g_every    = 10;      // verbose print stride (--every)
float  g_conf_min = mosse::DEFAULT_SCALE_CONF_MIN;   // --conf-min
// --max-step: the per-frame rate limit. Its own switch because this sim's `step`
// arm is the ONLY place a large correction is known to be CORRECT -- the
// detector proposes idx=-14 after an abrupt jump and it is right -- so it is the
// bench that measures what the limit costs, not just that it fires.
int    g_max_step = mosse::DEFAULT_SCALE_MAX_STEP;   // --max-step
float  g_sigma_f  = (float)(SCALE_SIGMA_FACTOR);  // --sigma-factor
// --pos-err: offset the box centre from the target's true centre, in frame px.
// The hardware run's centre error grew 1.26 -> 10.97 px AFTER the scale parked,
// and the claim was that the drift then prevented recovery. The isolated sim
// recovers (it has no position error), so this knob is what tests that claim.
double g_pos_err  = 0.0;
// --bg-pan: scroll the background under the target, px/frame, as BG_PAN does on
// the board (31,47). The scale template spans up to 1.37x the box, so most of it
// is background; on hardware that background is DIFFERENT EVERY FRAME and the
// sim had no equivalent. Refuting --pos-err left this as the last structural
// difference between the sim (recovers) and the board (did not).
int    g_pan_r = 0, g_pan_c = 0;
double g_step_ratio = 0.70;  // --step-ratio: size of the `step` arm's jump
// --reuse / --re-extract: train the model on the DETECTION sample with a shifted
// target (scale_update_shifted) instead of re-extracting at the resized box.
// Defaults to the tracker's behaviour so `make scale_sim` scores what the board
// runs; --re-extract restores the old path, which is what makes this a
// controlled comparison rather than an assertion.
bool   g_reuse    = true;
bool   g_no_gate  = false;   // --no-gate: accept every proposal, to separate
                             // "the detector cannot see it" from "the gate
                             // refused it". Those have opposite fixes.

// Frame geometry. Smaller than the board's 1080x1920 — the scale filter only
// ever reads a few box-widths around the target, and a 512x512 frame keeps the
// background generation instant.
constexpr int FR = 512, FC = 512;

// Defaults mirroring the hardware run being explained.
constexpr double BOX0        = 64.0;
constexpr double TRAJ_AMP    = 0.30;
constexpr double TRAJ_PERIOD = 200.0;

// ---------------------------------------------------------------------------
// Scene — MIRRORED FROM mosse_tracker.cpp, and it has to stay that way.
// ---------------------------------------------------------------------------
// The scale filter keys on the target's APPEARANCE PATTERN (scale_extract
// zero-means and unit-norms each level, discarding absolute energy), so the
// shape matters and a generic blob would not reproduce anything. This is the
// bar-and-spur of inject_target_frame(), same NOMINAL, same values, same
// asymmetry. If that function changes, change this one.
void fill_background(std::vector<uint8_t> &f)
{
    uint32_t s = 20260816u;
    auto next = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (double)(s >> 8) / (double)(1u << 24);
    };
    struct { double fy, fx, ph, amp; } comp[6];
    for (auto &k : comp) {
        k.fy  = std::round(1.0 + 5.0 * next()) / (double)FR;
        k.fx  = std::round(1.0 + 5.0 * next()) / (double)FC;
        k.ph  = 2.0 * M_PI * next();
        k.amp = 0.4 + 0.6 * next();
    }
    double amp_sum = 0.0;
    for (const auto &k : comp) amp_sum += k.amp;
    for (int r = 0; r < FR; ++r)
        for (int c = 0; c < FC; ++c) {
            double v = 0.0;
            for (const auto &k : comp)
                v += k.amp * std::sin(2.0 * M_PI * (k.fy * r + k.fx * c) + k.ph);
            v /= amp_sum;
            const double p = 110.0 + 90.0 * 0.35 * v + 3.0 * (next() - 0.5);
            f[(size_t)r * FC + c] = (uint8_t)(p < 0.0 ? 0.0 : (p > 255.0 ? 255.0 : p));
        }
}

void draw_target(std::vector<uint8_t> &f, const std::vector<uint8_t> &bg,
                 double tr, double tc, double th, double tw, int frame)
{
    constexpr uint8_t BAR = 220, SPUR = 150;
    constexpr double  NOMINAL = 11.0;
    if (g_pan_r || g_pan_c) {
        const int orr = ((long long)g_pan_r * frame) % FR + FR;
        const int occ = ((long long)g_pan_c * frame) % FC + FC;
        for (int r = 0; r < FR; ++r) {
            const int sr = (r + orr) % FR;
            for (int c = 0; c < FC; ++c)
                f[(size_t)r * FC + c] = bg[(size_t)sr * FC + (c + occ) % FC];
        }
    } else {
        f = bg;
    }
    const double sh = th / NOMINAL, sw = tw / NOMINAL;
    const int r0 = (int)std::floor(tr - 5.0 * sh), r1 = (int)std::ceil(tr + 5.0 * sh);
    for (int r = std::max(0, r0); r <= std::min(FR - 1, r1); ++r) {
        const double dr = (double)r - tr;
        if (dr < -5.0 * sh || dr > 5.0 * sh) continue;
        for (int c = 0; c < FC; ++c) {
            const double dc = (double)c - tc;
            uint8_t v = 0;
            if (dc >= -2.0 * sw && dc <= 2.0 * sw) v = BAR;
            else if (dr >= 2.0 * sh && dc >= 3.0 * sw && dc <= 8.0 * sw) v = SPUR;
            if (v) f[(size_t)r * FC + c] = v;
        }
    }
}

// Truth size envelope. `moving` is the hardware config; `static` holds it fixed
// (the discriminator); `step` applies one jump and then holds (measures pure
// convergence, which is the property run_scale_tests() asserts).
double truth_size(const std::string &arm, int frame)
{
    if (arm == "static") return BOX0;
    if (arm == "step")   return frame < 5 ? BOX0 : BOX0 * g_step_ratio;
    return BOX0 * (1.0 + TRAJ_AMP * std::sin(2.0 * M_PI * frame / TRAJ_PERIOD));
}

struct Result {
    // LONGEST run anywhere, not the terminal one. First cut measured only the
    // terminal run and reported "2 frames" for an arm that had parked for 31 —
    // the stall is mid-run whenever the envelope later turns back and frees it.
    int    frozen_from = -1;
    int    frozen_len  = 0;
    double max_rel_err = 0.0;   // max |est/truth - 1| after a warm-up
    double end_rel_err = 0.0;
    double conf_min = 1e9, conf_max = -1e9;
    int    n_held = 0;
};

Result run(const std::string &arm, int frames, float eta, int n_scales,
           float step, int corrections, bool verbose)
{
    std::vector<uint8_t> bg((size_t)FR * FC), frame((size_t)FR * FC);
    fill_background(bg);

    mosse::ScaleFilter sf;
    mosse::scale_filter_config(sf, n_scales, step, BOX0, BOX0, g_sigma_f);
    std::vector<mosse::cfloat> sample((size_t)sf.sample_elems());

    const double tr = FR / 2.0, tc = FC / 2.0;   // position held: this isolates
                                                 // the size axis, which the
                                                 // hardware run could not.
    double box = BOX0;
    Result res;
    std::vector<double> hist;

    if (verbose)
        printf("  %5s %8s %8s %7s %6s %6s  %s\n",
               "frame", "truth_h", "est_h", "rel", "idx", "conf", "veto");

    for (int f = 0; f < frames; ++f) {
        const double th = truth_size(arm, f);
        draw_target(frame, bg, tr, tc, th, th, f);
        // The target is drawn at (tr,tc); the box is centred g_pos_err away.
        const double br = tr + g_pos_err, bc = tc;

        int    idx  = 0;
        double conf = 0.0;
        const char *veto = "-";

        if (sf.initialized) {
            // Detection and resize, in the tracker's exact order. `corrections`
            // applies scale_detect more than once per frame WITHOUT retraining
            // — the direct test of the under-correction hypothesis, since
            // mosse_filter.h records that repeated application converges.
            for (int it = 0; it < corrections; ++it) {
                mosse::scale_extract(sf, frame.data(), FR, FC, br, bc, box, box,
                                     sample.data());
                const mosse::ScaleResult sr =
                    mosse::scale_detect(sf, sample.data(), mosse::DEFAULT_EPS_REL);
                mosse::ScaleDecision d =
                    mosse::scale_gate(sr, n_scales, box, box, BOX0, BOX0,
                                      g_conf_min,
                                      mosse::DEFAULT_SCALE_MIN_REL,
                                      mosse::DEFAULT_SCALE_MAX_REL,
                                      g_max_step);
                if (g_no_gate && sr.valid) { d.accept = true; d.new_h = box * sr.factor;
                                             d.new_w = box * sr.factor; }
                if (it == 0) { idx = sr.idx; conf = d.conf; veto = mosse::scale_veto_tag(d.reason);
                               if (d.conf < res.conf_min) res.conf_min = d.conf;
                               if (d.conf > res.conf_max) res.conf_max = d.conf;
                               if (!d.accept) ++res.n_held; }
                if (!d.accept) break;
                box = d.new_h;
            }
        }

        // Model update. Same ordering as mosse_tracker.cpp, and the same choice:
        // either re-extract at the UPDATED box, or train on the detection sample
        // already in `sample` against a target shifted by the accepted level.
        // The two are the same update in exact arithmetic — see
        // scale_update_shifted() — and this flag is here so that claim is tested
        // in the loop rather than only element by element in test_host.
        if (g_reuse && sf.initialized) {
            mosse::scale_update_shifted(sf, sample.data(), idx, eta);
        } else {
            mosse::scale_extract(sf, frame.data(), FR, FC, br, bc, box, box,
                                 sample.data());
            mosse::scale_update(sf, sample.data(), eta);
        }

        hist.push_back(box);
        const double rel = box / th - 1.0;
        if (f >= 10 && std::fabs(rel) > res.max_rel_err) res.max_rel_err = std::fabs(rel);
        if (verbose && (f % g_every == 0 || f == frames - 1))
            printf("  %5d %8.2f %8.2f %+6.1f%% %6d %6.2f  %s\n",
                   f, th, box, 100.0 * rel, idx, conf, veto);
    }

    res.end_rel_err = hist.back() / truth_size(arm, frames - 1) - 1.0;

    // Longest run of an unchanging box, anywhere in the sequence.
    int len = 1, best = 1, best_end = 0;
    for (size_t i = 1; i < hist.size(); ++i) {
        if (hist[i] == hist[i - 1]) ++len; else len = 1;
        if (len > best) { best = len; best_end = (int)i; }
    }
    res.frozen_len  = best;
    res.frozen_from = best_end - best + 1;
    return res;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string arm = "";
    int   frames = 200, n_scales = mosse::DEFAULT_SCALE_N, corrections = 1;
    // PINNED TO 1.02, THE CONFIGURATION THAT EXHIBITED THE DEFECT, and NOT to
    // DEFAULT_SCALE_STEP. The Makefile default moved to 1.04 once hardware
    // confirmed it; had this tracked the build, the premise arm would have
    // quietly stopped reproducing the stall and the regression would have
    // started asserting nothing while still printing REPRODUCES. The defect's
    // configuration is part of the test, not a build parameter. Use --step to
    // sweep, which is how 1.04 was found in the first place.
    float eta = mosse::DEFAULT_SCALE_ETA, step = 1.02f;
    bool  verbose = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char *d) { return (i + 1 < argc) ? argv[++i] : d; };
        if      (a == "--arm")         arm = val("moving");
        else if (a == "--frames")      frames = atoi(val("200"));
        else if (a == "--eta")         eta = (float)atof(val("0.025"));
        else if (a == "--n-scales")    n_scales = atoi(val("33"));
        else if (a == "--step")        step = (float)atof(val("1.02"));
        else if (a == "--corrections") corrections = atoi(val("1"));
        else if (a == "-v")            verbose = true;
        else if (a == "--every")       g_every = atoi(val("10"));
        else if (a == "--conf-min")    g_conf_min = (float)atof(val("2.0"));
        else if (a == "--max-step")    g_max_step = atoi(val("1"));
        else if (a == "--no-gate")     g_no_gate = true;
        else if (a == "--reuse")       g_reuse = true;
        else if (a == "--re-extract")  g_reuse = false;
        else if (a == "--step-ratio")  g_step_ratio = atof(val("0.70"));
        else if (a == "--sigma-factor") g_sigma_f = (float)atof(val("16.0"));
        else if (a == "--pos-err")     g_pos_err = atof(val("0"));
        else if (a == "--bg-pan")      { g_pan_r = atoi(val("31")); g_pan_c = atoi(val("47")); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    printf("\nDSST scale filter — closed loop, position held, no hardware\n");
    printf("  update trains on the %s\n",
           g_reuse ? "DETECTION sample, target shifted by idx (scale_update_shifted)"
                   : "RE-EXTRACTED sample at the resized box (scale_update)");
    printf("  S=%d step=%.3f eta=%.3f corrections/frame=%d  conf_min=%.2f%s  frames=%d\n",
           n_scales, (double)step, (double)eta, corrections,
           (double)g_conf_min, g_no_gate ? " GATE BYPASSED" : "", frames);

    if (!arm.empty()) {
        const Result r = run(arm, frames, eta, n_scales, step, corrections, true);
        printf("\n  arm=%s  max|rel err| %.1f%%  end %+.1f%%  terminal freeze %d frames "
               "(from %d)  conf %.2f..%.2f  held %d\n",
               arm.c_str(), 100.0 * r.max_rel_err, 100.0 * r.end_rel_err,
               r.frozen_len, r.frozen_from, r.conf_min, r.conf_max, r.n_held);
        return 0;
    }

    printf("\n  %-8s %14s %10s %18s %14s %6s\n",
           "arm", "max|rel err|", "end err", "longest park", "conf range", "held");
    Result moving{};
    for (const char *a : {"moving", "static", "step"}) {
        const Result r = run(a, frames, eta, n_scales, step, corrections, verbose);
        if (!strcmp(a, "moving")) moving = r;
        printf("  %-8s %13.1f%% %9.1f%% %6d frames @f%-4d %7.2f..%-5.2f %6d\n",
               a, 100.0 * r.max_rel_err, 100.0 * r.end_rel_err,
               r.frozen_len, r.frozen_from, r.conf_min, r.conf_max, r.n_held);
    }

    // THE PREMISE CHECK. The counterfactual arms are worthless unless `moving`
    // reproduces what the board did: a large terminal error and a long freeze.
    // Calibrated against what the board actually showed: est_h parked at 59.13
    // for 70 frames at up to +31% error. The isolated filter stalls the same way
    // but recovers when the envelope turns, so the threshold is on the STALL,
    // not on the never-recovering version — see the note in main()'s output.
    const bool reproduced = moving.frozen_len >= 20 && moving.max_rel_err > 0.10;
    printf("\n  PREMISE: arm `moving` %s the hardware stall "
           "(park >= 20 frames AND max err > 10%%).\n",
           reproduced ? "REPRODUCES" : "DOES NOT REPRODUCE");
    if (!reproduced)
        printf("           => the scene or the geometry has drifted from the "
               "tracker's. Fix that before reading anything else here.\n");
    return reproduced ? 0 : 1;
}
