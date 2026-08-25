/*
 * test_vot_source.cpp
 * Native harness for vot_source.{h,cpp} — manifest, blob, run order, trajectory.
 *
 * WHY THIS EXISTS
 * ---------------
 * Nothing in this translation unit computes anything. It is pure bookkeeping,
 * and every one of its failure modes produces a complete, plausible, entirely
 * invalid AR report rather than an error: a backward run emitted in sequence
 * order scores as a tracker that runs backwards; a trajectory one entry short
 * scores as a tracker that stopped early; a blob read one frame short shifts
 * every frame after it. Phase 1 already lost every groundtruth box in 62
 * manifests to exactly this shape of bug (see runs/vot/phase1.md).
 *
 * MUTATION-TESTED, for the reason every RGB suite is: a passing test on a path
 * with no prior coverage is worth nothing until it has been shown to FAIL. Each
 * mutant corrupts one thing and must be caught; `t_mutants_are_caught` is the
 * assertion, not the happy path above it.
 *
 * TOLERANCE IS ZERO — every value here is an integer or an exactly-representable
 * .5, and the trajectory text is compared as text.
 *
 * Assertion contract copied from test_scene_colour.cpp: helpers print one line,
 * bump g_failures, and never abort, so one broken case cannot hide the others.
 *
 * Build/run:  make test_vot_source
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vot_source.h"

namespace {

int g_failures = 0;

void check(const char *what, bool cond, const std::string &detail = "")
{
    printf("    %-52s %s%s%s\n", what, cond ? "OK  " : "FAIL",
           detail.empty() ? "" : "   ", detail.c_str());
    if (!cond) ++g_failures;
}

std::string g_dir;

std::string tmpdir()
{
    if (!g_dir.empty()) return g_dir;
    char t[] = "/tmp/vot_source_test_XXXXXX";
    const char *d = mkdtemp(t);
    if (!d) { printf("    cannot create a temp dir\n"); ++g_failures; return "/tmp"; }
    g_dir = d;
    return g_dir;
}

void write_file(const std::string &path, const std::string &data)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { printf("    cannot write %s\n", path.c_str()); ++g_failures; return; }
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
}

std::string read_file(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string s;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}

// ---------------------------------------------------------------------------
// A fixture the size of the real thing in every dimension that matters, and
// small in the one that does not: 5 frames of 3x4x1, three jobs covering a
// forward anchor at 0, a forward anchor in the interior, and a backward one.
// ---------------------------------------------------------------------------
constexpr int ROWS = 3, COLS = 4, CH = 1, FRAMES = 5;
constexpr size_t FB = (size_t)ROWS * COLS * CH;

std::string good_manifest()
{
    return R"({
  "schema": 1,
  "sequence": "fixture",
  "frames": 5,
  "rows": 3,
  "cols": 4,
  "channels": 1,
  "dtype": "uint8",
  "frame_bytes": 12,
  "layout": "frames back-to-back, row-major, no header; luma plane",
  "blob": "fixture.raw",
  "blob_md5": "0123456789abcdef0123456789abcdef",
  "luma_blob": null,
  "gt_format": "rectangle",
  "gt_convention": "4-value x,y,w,h -> centre",
  "empty_boxes": 0,
  "groundtruth": [[10.5, 20.5, 4.0, 6.0], [11.0, 21.0, 4.0, 6.0],
                  [12.0, 22.0, 4.0, 6.0], [13.0, 23.0, 4.0, 6.0],
                  [14.0, 24.0, 4.0, 6.0]],
  "anchors_source": "dataset",
  "jobs": [
    {"anchor": 0, "direction": "forward",  "init_box": [10.5, 20.5, 4.0, 6.0], "length": 5},
    {"anchor": 2, "direction": "forward",  "init_box": [12.0, 22.0, 4.0, 6.0], "length": 3},
    {"anchor": 3, "direction": "backward", "init_box": [13.0, 23.0, 4.0, 6.0], "length": 4}
  ],
  "tracked_frames": 12,
  "min_box_side": 4.0,
  "roi_exceeds_frame": 0,
  "generator": {"script": "scripts/vot_prepare.py", "mutation": null}
})";
}

// Frame i is filled with the byte i+1, so a frame index error is visible in one
// byte rather than needing a diff.
std::string good_blob(int frames = FRAMES, size_t frame_bytes = FB)
{
    std::string s;
    for (int i = 0; i < frames; ++i) s.append(frame_bytes, (char)(i + 1));
    return s;
}

// ---------------------------------------------------------------------------
void t_manifest_fields()
{
    printf("\n  manifest — every field the board reads\n");
    vot::Manifest m;
    std::string err;
    const std::string txt = good_manifest();
    const bool ok = vot::manifest_parse(txt.data(), txt.size(), m, err);
    check("parses", ok, err);
    if (!ok) return;

    check("sequence", m.sequence == "fixture", m.sequence);
    check("frames", m.frames == FRAMES);
    check("rows/cols/channels", m.rows == ROWS && m.cols == COLS && m.channels == CH);
    check("frame_bytes", m.frame_bytes == FB);
    check("blob name", m.blob == "fixture.raw", m.blob);
    check("luma_blob null -> empty", m.luma_blob.empty());
    check("gt_format", m.gt_format == "rectangle");
    check("anchors_source", m.anchors_source == "dataset");
    check("groundtruth length == frames", (int)m.groundtruth.size() == FRAMES);
    check("groundtruth is (row,col,h,w)",
          m.groundtruth[0].row == 10.5 && m.groundtruth[0].col == 20.5 &&
          m.groundtruth[0].h == 4.0 && m.groundtruth[0].w == 6.0);
    check("3 jobs", m.jobs.size() == 3);
    if (m.jobs.size() != 3) return;
    check("job 0 forward from 0, length 5",
          m.jobs[0].anchor == 0 && m.jobs[0].forward && m.jobs[0].length == 5);
    check("job 2 is BACKWARD from 3, length 4",
          m.jobs[2].anchor == 3 && !m.jobs[2].forward && m.jobs[2].length == 4);
    check("init_box is the anchor's groundtruth",
          m.jobs[2].init_box.row == 13.0 && m.jobs[2].init_box.w == 6.0);
}

void t_blob()
{
    printf("\n  blob — one read, computed offsets, no header\n");
    const std::string dir = tmpdir();
    const std::string path = dir + "/fixture.raw";
    write_file(path, good_blob());

    vot::Manifest m;
    std::string err;
    const std::string txt = good_manifest();
    if (!vot::manifest_parse(txt.data(), txt.size(), m, err)) { check("fixture parses", false, err); return; }

    vot::Blob b;
    const bool ok = b.load(path, m, err);
    check("loads", ok, err);
    if (!ok) return;
    check("byte count", b.bytes() == FB * FRAMES);
    check("frame count", b.frames() == FRAMES);

    bool offsets = true;
    for (int i = 0; i < FRAMES; ++i) {
        const uint8_t *f = b.frame(i);
        if (!f) { offsets = false; break; }
        for (size_t k = 0; k < FB; ++k)
            if (f[k] != (uint8_t)(i + 1)) { offsets = false; break; }
    }
    check("every frame lands at frame_bytes * i", offsets);
    check("frame(-1) and frame(N) are nullptr",
          b.frame(-1) == nullptr && b.frame(FRAMES) == nullptr);
    check("staging time was measured", b.load_seconds() >= 0.0);
}

void t_job_order()
{
    printf("\n  run order — the direction convention, which is silent when wrong\n");
    vot::Job f; f.anchor = 2; f.forward = true;  f.length = 3;
    vot::Job r; r.anchor = 3; r.forward = false; r.length = 4;

    const std::vector<int> of = vot::job_order(f, FRAMES);
    const std::vector<int> ob = vot::job_order(r, FRAMES);

    check("forward covers [anchor .. end]", of == std::vector<int>({2, 3, 4}));
    check("backward covers [anchor .. 0]",  ob == std::vector<int>({3, 2, 1, 0}));
    check("order[0] is the anchor in both", !of.empty() && !ob.empty() &&
          of[0] == f.anchor && ob[0] == r.anchor);
    check("lengths match the manifest's", (int)of.size() == f.length &&
                                          (int)ob.size() == r.length);

    // The bug this is really guarding: a backward run written in SEQUENCE order.
    // It is a valid-looking file, so the only thing that can catch it is that
    // run order and sequence order are asserted to differ.
    std::vector<int> sequence_order = ob;
    for (size_t i = 0; i < sequence_order.size(); ++i) sequence_order[i] = (int)i;
    check("backward run order != sequence order", ob != sequence_order);

    vot::Job bad; bad.anchor = 99; bad.forward = true; bad.length = 1;
    check("out-of-range anchor yields no frames", vot::job_order(bad, FRAMES).empty());
}

void t_trajectory()
{
    printf("\n  trajectory — index 0 is a SPECIAL, boxes are top-left x,y,w,h\n");
    const std::string dir = tmpdir();
    vot::Job j; j.anchor = 3; j.forward = false; j.length = 4;

    vot::Trajectory t;
    t.begin(j);
    t.push_init(1.5);
    vot::Box b;
    b.row = 10.0; b.col = 20.0; b.h = 4.0; b.w = 6.0;
    t.push(b, 2.25);
    b.row = 11.0; b.col = 21.0;
    t.push(b, 2.5);
    b.row = 12.0; b.col = 22.0;
    t.push(b, 2.75);

    std::string err;
    const bool ok = t.write(dir, "fixture", err);
    check("writes", ok, err);
    if (!ok) return;

    const std::string traj = read_file(dir + "/fixture_00000003.txt");
    const std::string tv   = read_file(dir + "/fixture_00000003_time.value");

    check("filename is {sequence}_{anchor:08d}.txt", !traj.empty());
    // x = col - w/2, y = row - h/2. Written as text and compared as text, because
    // the toolkit parses this file with float() and nothing else validates it.
    check("content is exact",
          traj == "1\n"
                  "17.0000,8.0000,6.0000,4.0000\n"
                  "18.0000,9.0000,6.0000,4.0000\n"
                  "19.0000,10.0000,6.0000,4.0000\n",
          traj);
    check("first line is the INITIALIZATION code, not the init box",
          traj.compare(0, 2, "1\n") == 0);
    check("time sidecar has one line per region",
          tv == "1.500\n2.250\n2.500\n2.750\n", tv);
}

// ---------------------------------------------------------------------------
// The assertion: every mutant must be REJECTED. A parser that accepts a corrupt
// manifest is indistinguishable, from the console, from one that works.
// ---------------------------------------------------------------------------
std::string mutate(const std::string &s, const std::string &from, const std::string &to)
{
    const size_t at = s.find(from);
    if (at == std::string::npos) return std::string();   // caught below as "not applied"
    return s.substr(0, at) + to + s.substr(at + from.size());
}

void t_mutants_are_caught()
{
    printf("\n  mutants — each must be REJECTED, with a message\n");
    const std::string good = good_manifest();

    struct M { const char *name; std::string text; };
    const std::vector<M> mutants = {
        {"schema 2 (a format this build cannot read)", mutate(good, "\"schema\": 1", "\"schema\": 2")},
        {"frame_bytes != rows*cols*channels",          mutate(good, "\"frame_bytes\": 12", "\"frame_bytes\": 16")},
        // Both of these must be caught by the LENGTH check, not by the JSON
        // syntax check — a mutant that trips the parser earlier than intended
        // tests the parser twice and the semantics not at all.
        {"groundtruth one frame short",                mutate(good, "[12.0, 22.0, 4.0, 6.0], ", "")},
        {"groundtruth one frame long",                 mutate(good, "[14.0, 24.0, 4.0, 6.0]]",
                                                                    "[14.0, 24.0, 4.0, 6.0], [15.0, 25.0, 4.0, 6.0]]")},
        {"groundtruth box with 3 values",              mutate(good, "[10.5, 20.5, 4.0, 6.0],", "[10.5, 20.5, 4.0],")},
        {"job length disagrees with its anchor",       mutate(good, "\"init_box\": [12.0, 22.0, 4.0, 6.0], \"length\": 3",
                                                                    "\"init_box\": [12.0, 22.0, 4.0, 6.0], \"length\": 4")},
        {"direction is neither forward nor backward",  mutate(good, "\"backward\"", "\"sideways\"")},
        {"backward job relabelled forward",            mutate(good, "{\"anchor\": 3, \"direction\": \"backward\"",
                                                                    "{\"anchor\": 3, \"direction\": \"forward\"")},
        {"anchor past the end of the sequence",        mutate(good, "\"anchor\": 2", "\"anchor\": 9")},
        {"a required key is missing (cols)",           mutate(good, "\"cols\": 4,", "")},
        {"jobs list empty",                            mutate(good, "\"jobs\": [", "\"jobs\": [], \"unused\": [")},
        {"truncated JSON",                             good.substr(0, good.size() / 2)},
        {"not JSON at all",                            std::string("<html>404</html>")},
    };

    for (const M &m : mutants) {
        if (m.text.empty()) { check(m.name, false, "mutation did not apply"); continue; }
        vot::Manifest parsed;
        std::string err;
        const bool accepted = vot::manifest_parse(m.text.data(), m.text.size(), parsed, err);
        check(m.name, !accepted, accepted ? "ACCEPTED" : err);
    }

    // Blob mutants. The manifest is the authority on size, so both directions of
    // disagreement are errors — a byte long means the geometry is wrong, and every
    // frame offset after the first is then a guess.
    const std::string dir = tmpdir();
    vot::Manifest m;
    std::string err;
    if (!vot::manifest_parse(good.data(), good.size(), m, err)) { check("fixture parses", false, err); return; }

    struct B { const char *name; std::string data; };
    const std::vector<B> blobs = {
        {"blob one frame short", good_blob(FRAMES - 1)},
        {"blob one frame long",  good_blob(FRAMES + 1)},
        {"blob one byte short",  good_blob().substr(0, FB * FRAMES - 1)},
        {"blob one byte long",   good_blob() + std::string(1, 'X')},
        {"blob empty",           std::string()},
    };
    for (const B &b : blobs) {
        const std::string path = dir + "/mutant.raw";
        write_file(path, b.data);
        vot::Blob blob;
        std::string berr;
        const bool accepted = blob.load(path, m, berr);
        check(b.name, !accepted, accepted ? "ACCEPTED" : berr);
    }

    // Trajectory mutants — both are "valid file, wrong result" shapes.
    {
        vot::Job j; j.anchor = 3; j.forward = false; j.length = 4;
        vot::Trajectory t;
        t.begin(j);
        t.push_init(1.0);
        vot::Box b; b.row = 1; b.col = 1; b.h = 2; b.w = 2;
        t.push(b, 1.0);
        std::string terr;
        check("trajectory short of the job length is refused",
              !t.write(dir, "mutant", terr), terr);
    }
    {
        vot::Job j; j.anchor = 0; j.forward = true; j.length = 2;
        vot::Trajectory t;
        t.begin(j);
        vot::Box b; b.row = 1; b.col = 1; b.h = 2; b.w = 2;
        t.push(b, 1.0);      // no push_init: index 0 is a rectangle
        t.push(b, 1.0);
        std::string terr;
        check("trajectory without the init special is refused",
              !t.write(dir, "mutant2", terr), terr);
    }
}

// ---------------------------------------------------------------------------
// If $VOT_ROOT/data is present, parse a REAL manifest. Not a substitute for the
// fixture — it asserts nothing about values — but it is the only check that the
// converter's actual output and this parser agree, and it costs nothing.
// ---------------------------------------------------------------------------
void t_real_manifest_if_present()
{
    printf("\n  real manifest (skipped if $VOT_ROOT/data is absent)\n");
    const char *root = getenv("VOT_ROOT");
    if (!root) { printf("    VOT_ROOT unset — skipped\n"); return; }

    int parsed = 0, failed = 0;
    std::string first_err;
    FILE *p = popen((std::string("ls ") + root + "/data/*.json 2>/dev/null").c_str(), "r");
    if (!p) { printf("    cannot list %s/data — skipped\n", root); return; }
    char line[1024];
    while (fgets(line, sizeof line, p)) {
        std::string path(line);
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) path.pop_back();
        vot::Manifest m;
        std::string err;
        if (vot::manifest_load(path, m, err)) ++parsed;
        else { ++failed; if (first_err.empty()) first_err = path + ": " + err; }
    }
    pclose(p);
    if (parsed == 0 && failed == 0) { printf("    no manifests found — skipped\n"); return; }
    check("every manifest in $VOT_ROOT/data parses", failed == 0,
          std::to_string(parsed) + " parsed, " + std::to_string(failed) +
          " failed" + (first_err.empty() ? "" : "  " + first_err));
}

// ---------------------------------------------------------------------------
// EMIT MODE: `test_vot_source <dir>` writes a trajectory with the REAL writer,
// plus the boxes it was given in the tracker's own CENTRE convention. The
// centre -> top-left conversion is then re-derived independently, in Python,
// against what the toolkit's own parser reads back — see
// scripts/vot_check_trajectory.py and `make test_vot_format`.
//
// The expectation file carries the INPUT, not the writer's output, on purpose:
// an expectation produced by the code under test agrees with it no matter what
// it does. That is the trap the corrupted weights_ch0.bin taught this project.
// ---------------------------------------------------------------------------
int emit(const char *dir)
{
    vot::Job j; j.anchor = 42; j.forward = false; j.length = 4;
    vot::Trajectory t;
    t.begin(j);
    t.push_init(1.0);

    const vot::Box boxes[3] = {
        {100.25, 200.50, 30.0, 40.0},
        {101.75, 201.25, 31.5, 41.5},
        {103.00, 202.00, 33.0, 43.0},
    };
    for (int i = 0; i < 3; ++i) t.push(boxes[i], 2.0 + i);

    std::string err;
    if (!t.write(dir, "seq", err)) {
        fprintf(stderr, "emit: %s\n", err.c_str());
        return 1;
    }
    const std::string ep = std::string(dir) + "/expect.txt";
    FILE *f = fopen(ep.c_str(), "w");
    if (!f) { fprintf(stderr, "emit: cannot write %s\n", ep.c_str()); return 1; }
    fprintf(f, "# row,col,h,w  CENTRE convention, as handed to Trajectory::push\n");
    for (int i = 0; i < 3; ++i)
        fprintf(f, "%.6f,%.6f,%.6f,%.6f\n",
                boxes[i].row, boxes[i].col, boxes[i].h, boxes[i].w);
    fclose(f);
    printf("emitted seq_%08d.txt, _time.value and expect.txt into %s\n",
           j.anchor, dir);
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc > 1) return emit(argv[1]);

    printf("\nvot_source native harness — zero tolerance, mutation-tested\n");
    t_manifest_fields();
    t_blob();
    t_job_order();
    t_trajectory();
    t_mutants_are_caught();
    t_real_manifest_if_present();
    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures,
           g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
