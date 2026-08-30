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
 * manifests to exactly this shape of bug (see docs/thesis/evidence/phase1.md).
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
// Defined below, next to the manifest mutants that are its main user.
std::string mutate(const std::string &s, const std::string &from,
                   const std::string &to);

// -----------------------------------------------------------------------
// THE LUMA SIDECAR (channels=3). New path, so it needs a test that is known to
// be able to FAIL before it is worth anything.
//
// The one way this breaks silently is SIZING: the sidecar is one plane,
// rows*cols per frame, while the manifest's frame_bytes is rows*cols*channels.
// Size it with frame_bytes and every frame offset is 3x too large -- which does
// NOT crash, it reads within a correctly-sized-but-wrong window and hands
// scale_extract a stripe of the wrong frame. The scale filter then degrades and
// nothing else in the system notices, because no other consumer reads the
// sidecar. So the assertion is BOTH directions: the right size is accepted, and
// the frame_bytes-sized one is rejected.
void t_luma_sidecar()
{
    printf("\n  luma sidecar (channels=3)\n");
    constexpr int  LROWS = 3, LCOLS = 4, LCH = 3, LFRAMES = 5;
    constexpr size_t LFB   = (size_t)LROWS * LCOLS * LCH;   // 36, interleaved
    constexpr size_t LUMAFB = (size_t)LROWS * LCOLS;        // 12, one plane

    std::string mtext = good_manifest();
    mtext = mutate(mtext, "\"channels\": 1", "\"channels\": 3");
    mtext = mutate(mtext, "\"frame_bytes\": 12", "\"frame_bytes\": 36");
    mtext = mutate(mtext, "\"luma_blob\": null", "\"luma_blob\": \"fixture.luma\"");

    vot::Manifest m;
    std::string err;
    if (!vot::manifest_parse(mtext.data(), mtext.size(), m, err)) {
        check("channels=3 fixture parses", false, err);
        return;
    }
    check("luma_blob parsed", m.luma_blob == "fixture.luma", m.luma_blob);
    check("frame_bytes is the INTERLEAVED size", m.frame_bytes == LFB,
          std::to_string(m.frame_bytes));

    const std::string dir = tmpdir();

    // (1) the correct sidecar is ACCEPTED and indexes one plane per frame.
    {
        const std::string path = dir + "/good.luma";
        write_file(path, good_blob(LFRAMES, LUMAFB));
        vot::Blob luma;
        std::string lerr;
        const bool ok = luma.load_luma(path, m, lerr);
        check("rows*cols sidecar accepted", ok, ok ? "" : lerr);
        if (ok) {
            check("sidecar frame_bytes == rows*cols",
                  luma.frame_bytes() == LUMAFB, std::to_string(luma.frame_bytes()));
            check("sidecar frames == manifest frames",
                  luma.frames() == LFRAMES, std::to_string(luma.frames()));
            // good_blob fills frame i with byte i+1, so a stride error shows up
            // as the wrong frame rather than as a crash.
            const uint8_t *f2 = luma.frame(2);
            check("sidecar frame(2) has frame 2's content", f2 && f2[0] == 3,
                  f2 ? std::to_string((int)f2[0]) : "null");
            check("sidecar frame(LFRAMES) is out of range",
                  luma.frame(LFRAMES) == nullptr, "");
        }
    }

    // (2) THE MUTANT: a sidecar sized rows*cols*channels must be REJECTED.
    // This is what load_luma would happily read if it used m.frame_bytes, so if
    // this check ever passes-by-accepting, the sizing bug is back.
    {
        const std::string path = dir + "/interleaved_sized.luma";
        write_file(path, good_blob(LFRAMES, LFB));
        vot::Blob luma;
        std::string lerr;
        const bool accepted = luma.load_luma(path, m, lerr);
        check("sidecar sized rows*cols*channels REJECTED", !accepted,
              accepted ? "ACCEPTED — load_luma is using frame_bytes" : lerr);
    }

    // (3) the ordinary short/long checks still apply through the shared reader.
    {
        const std::string path = dir + "/short.luma";
        write_file(path, good_blob(LFRAMES, LUMAFB).substr(0, LUMAFB * LFRAMES - 1));
        vot::Blob luma;
        std::string lerr;
        check("sidecar one byte short REJECTED", !luma.load_luma(path, m, lerr), lerr);
    }
}

// ---------------------------------------------------------------------------
// StreamBlob — the reader for sequences that do not fit in heap.
//
// WHAT MAKES THIS TESTABLE AT ALL, and why it is worth more than the usual
// "new code, new test": StreamBlob changes no arithmetic. It returns the SAME
// bytes as Blob in the SAME order, from a file instead of from heap. So the
// assertion is not a property of the streamed output -- it is EQUALITY with the
// resident reader, frame for frame, on the same run order. That is the native
// form of the acceptance test the board will run (identical run-state digests
// in both modes on a sequence that fits), and it costs milliseconds.
//
// The failure modes are the ones a ring buffer has and a heap array does not,
// and none of them crash:
//   * a slot released one frame too early -> the producer overwrites the frame
//     under the caller's memcpy, so the tracker sees a frame from K ahead;
//   * an off-by-one in the slot index -> every frame is the neighbour of the
//     right one, which on real video looks like a tracker that lags by a frame;
//   * a run re-armed without begin_run() -> job N streams job N-1's order,
//     which is the backward-run-in-sequence-order bug wearing a new hat.
// Each is exercised below with a ring DELIBERATELY smaller than the run, since
// a ring larger than the run wraps zero times and proves nothing about any of
// them.
constexpr int STREAM_FRAMES = 17;
constexpr size_t STREAM_FB  = 40;

// Distinct in BOTH indices, unlike good_blob()'s constant-per-frame fill: a
// constant frame cannot tell a short read from a complete one, and a short read
// is exactly what a per-frame reader risks and a single whole-blob read does not.
std::string stream_blob(int frames = STREAM_FRAMES, size_t fb = STREAM_FB)
{
    std::string s;
    s.reserve((size_t)frames * fb);
    for (int i = 0; i < frames; ++i)
        for (size_t j = 0; j < fb; ++j)
            s.push_back((char)(uint8_t)((i * 31 + j * 7 + 5) & 0xFF));
    return s;
}

// Every frame of `order`, streamed, compared against the file's own bytes.
// Returns the run index of the first mismatch, or -1.
int stream_matches(vot::StreamBlob &sb, const std::vector<int> &order,
                   const std::string &truth, size_t fb)
{
    for (size_t k = 0; k < order.size(); ++k) {
        const uint8_t *p = sb.at(k);
        if (!p) return (int)k;
        if (memcmp(p, truth.data() + (size_t)order[k] * fb, fb) != 0) return (int)k;
    }
    return -1;
}

void t_stream_blob()
{
    printf("\n  StreamBlob — streamed frames must equal resident frames\n");
    const std::string dir  = tmpdir();
    const std::string path = dir + "/stream.raw";
    const std::string truth = stream_blob();
    write_file(path, truth);

    std::string err;

    // (1) EQUALITY WITH THE RESIDENT READER, forward order, ring < run length.
    // Ring 3 against 17 frames wraps five times, so a slot-index error cannot
    // hide in a ring that never wraps.
    {
        vot::Blob resident;
        const bool rok = resident.load(path, [&] {
            vot::Manifest m; m.frames = STREAM_FRAMES; m.frame_bytes = STREAM_FB;
            return m; }(), err);
        check("resident reader loads the fixture", rok, rok ? "" : err);

        vot::StreamBlob sb;
        const bool ok = sb.open(path, STREAM_FB, STREAM_FRAMES, 3, err);
        check("stream opens (ring 3, 17 frames)", ok, ok ? "" : err);
        if (ok && rok) {
            std::vector<int> order;
            for (int i = 0; i < STREAM_FRAMES; ++i) order.push_back(i);
            sb.begin_run(order, order.size());
            int bad = -1;
            for (size_t k = 0; k < order.size() && bad < 0; ++k) {
                const uint8_t *sp = sb.at(k);
                const uint8_t *rp = resident.frame(order[k]);
                if (!sp || !rp || memcmp(sp, rp, STREAM_FB) != 0) bad = (int)k;
            }
            check("forward run: every frame == the resident reader's",
                  bad < 0, bad < 0 ? "" : "first mismatch at run index "
                                          + std::to_string(bad) + " " + sb.error());
        }
    }

    // (2) BACKWARD ORDER. Descending offsets defeat sequential read-ahead, so
    // this is the case where a reader that quietly assumed forward-only would
    // still return data -- the wrong data, silently, on half the dataset's runs.
    {
        vot::StreamBlob sb;
        const bool ok = sb.open(path, STREAM_FB, STREAM_FRAMES, 4, err);
        check("stream reopens for the backward run", ok, ok ? "" : err);
        if (ok) {
            std::vector<int> order;
            for (int i = STREAM_FRAMES - 1; i >= 0; --i) order.push_back(i);
            sb.begin_run(order, order.size());
            const int bad = stream_matches(sb, order, truth, STREAM_FB);
            check("backward run: every frame matches the blob", bad < 0,
                  bad < 0 ? "" : "run index " + std::to_string(bad) + " " + sb.error());
        }
    }

    // (3) RE-ARM. One StreamBlob serves every job of a sequence, so begin_run()
    // is the per-anchor reset. A stream that kept the previous job's order would
    // return real frames in the wrong order -- scored without complaint.
    {
        vot::StreamBlob sb;
        const bool ok = sb.open(path, STREAM_FB, STREAM_FRAMES, 2, err);
        check("stream opens for the re-arm case", ok, ok ? "" : err);
        if (ok) {
            const std::vector<int> jobA = {5, 6, 7, 8, 9, 10};
            const std::vector<int> jobB = {12, 11, 10, 9};
            sb.begin_run(jobA, jobA.size());
            const int a1 = stream_matches(sb, jobA, truth, STREAM_FB);
            sb.begin_run(jobB, jobB.size());
            const int b = stream_matches(sb, jobB, truth, STREAM_FB);
            sb.begin_run(jobA, jobA.size());
            const int a2 = stream_matches(sb, jobA, truth, STREAM_FB);
            check("job A, then B, then A again — all three correct",
                  a1 < 0 && b < 0 && a2 < 0,
                  "A1=" + std::to_string(a1) + " B=" + std::to_string(b) +
                  " A2=" + std::to_string(a2) + " " + sb.error());
            check("ring 2 is the minimum that can work (it wraps every frame)",
                  sb.ring() == 2, std::to_string(sb.ring()));
        }
    }

    // (4) TRUNCATION. --vot-max-frames shortens the run; the prefetcher must
    // stop at the truncated length rather than read frames the run never uses.
    // Asserted by asking for the frame PAST it and requiring a refusal, because
    // "it read too much" is otherwise invisible from outside.
    {
        vot::StreamBlob sb;
        const bool ok = sb.open(path, STREAM_FB, STREAM_FRAMES, 3, err);
        check("stream opens for the truncation case", ok, ok ? "" : err);
        if (ok) {
            std::vector<int> order;
            for (int i = 0; i < STREAM_FRAMES; ++i) order.push_back(i);
            sb.begin_run(order, 4);
            const std::vector<int> first4(order.begin(), order.begin() + 4);
            check("truncated run returns its 4 frames",
                  stream_matches(sb, first4, truth, STREAM_FB) < 0, sb.error());
            // ONE call, into a local. at() is STATEFUL -- it advances the
            // stream -- and naming it twice in one check() left the order of
            // the two calls unspecified: with the detail evaluated first, the
            // condition's call was rejected as out-of-order and the check
            // passed for the wrong reason. It survived the "begin_run ignores
            // the truncated length" mutant because of exactly that.
            const uint8_t *past = sb.at(4);
            check("frame 4 of a 4-frame run REFUSED", past == nullptr,
                  past == nullptr ? sb.error() : "returned a pointer");
        }
    }

    // (5) THE LUMA SIDECAR, through the streaming reader. Same assertion as
    // t_luma_sidecar makes of the resident one, and it is made HERE rather than
    // trusted because open_luma() is a second call site of the one-plane rule.
    {
        vot::Manifest m;
        m.rows = 3; m.cols = 4; m.channels = 3;
        m.frames = 5; m.frame_bytes = 36;          // interleaved
        const std::string lpath = dir + "/stream.luma";
        write_file(lpath, stream_blob(5, 12));      // rows*cols
        vot::StreamBlob sb;
        check("open_luma accepts a rows*cols sidecar",
              sb.open_luma(lpath, m, 2, err), err);
        check("streamed sidecar frame_bytes == rows*cols",
              sb.frame_bytes() == 12, std::to_string(sb.frame_bytes()));

        const std::string ipath = dir + "/stream_interleaved.luma";
        write_file(ipath, stream_blob(5, 36));
        vot::StreamBlob bad;
        const bool accepted = bad.open_luma(ipath, m, 2, err);
        check("open_luma REJECTS a rows*cols*channels sidecar", !accepted,
              accepted ? "ACCEPTED — open_luma is using frame_bytes" : err);
    }
}

// The StreamBlob mutants. Separated from the parser mutants only because they
// need a file on disk; the contract is the same one — each must be REJECTED,
// and a reader that accepts them produces plausible frames instead of an error.
void t_stream_mutants()
{
    printf("\n  StreamBlob mutants — each must be REJECTED\n");
    const std::string dir = tmpdir();
    const std::string truth = stream_blob();
    std::string err;

    // A blob one frame SHORT. The resident reader catches this in its single
    // read; a per-frame reader would not notice until the last frame of the run,
    // hours in — or never, on a backward run that ends at frame 0.
    {
        const std::string p = dir + "/short_stream.raw";
        write_file(p, truth.substr(0, truth.size() - STREAM_FB));
        vot::StreamBlob sb;
        check("blob one frame short REJECTED at open",
              !sb.open(p, STREAM_FB, STREAM_FRAMES, 3, err), err);
    }
    // ...and one byte LONG, which means the manifest and the file disagree about
    // geometry, so every frame offset after the first is a guess.
    {
        const std::string p = dir + "/long_stream.raw";
        write_file(p, truth + std::string(1, 'x'));
        vot::StreamBlob sb;
        check("blob one byte long REJECTED at open",
              !sb.open(p, STREAM_FB, STREAM_FRAMES, 3, err), err);
    }
    // ring 1 cannot work: at(k) holds slot k while the producer fills ahead, so
    // one slot is overwritten under the caller. Refused, not clamped to 2 — a
    // silent promotion makes the one broken configuration look configured.
    {
        const std::string p = dir + "/stream.raw";
        vot::StreamBlob sb;
        check("ring 1 REJECTED", !sb.open(p, STREAM_FB, STREAM_FRAMES, 1, err), err);
    }
    // OUT-OF-ORDER ACCESS. The whole design rests on the run order being known
    // in advance, so a caller that skips, repeats or looks ahead must get an
    // error rather than a seek: seeking silently serialises the run against NFS.
    {
        const std::string p = dir + "/stream.raw";
        vot::StreamBlob sb;
        if (sb.open(p, STREAM_FB, STREAM_FRAMES, 4, err)) {
            std::vector<int> order;
            for (int i = 0; i < STREAM_FRAMES; ++i) order.push_back(i);
            sb.begin_run(order, order.size());
            (void)sb.at(0);
            check("skipping a run index REJECTED", sb.at(2) == nullptr, sb.error());
        }
        vot::StreamBlob sb2;
        if (sb2.open(p, STREAM_FB, STREAM_FRAMES, 4, err)) {
            std::vector<int> order;
            for (int i = 0; i < STREAM_FRAMES; ++i) order.push_back(i);
            sb2.begin_run(order, order.size());
            (void)sb2.at(0);
            (void)sb2.at(1);
            check("re-reading a run index REJECTED", sb2.at(1) == nullptr, sb2.error());
        }
    }
    // at() before begin_run(): no order has been declared, so there is nothing
    // to serve. Returning frame 0 would be the plausible-and-wrong answer.
    {
        const std::string p = dir + "/stream.raw";
        vot::StreamBlob sb;
        if (sb.open(p, STREAM_FB, STREAM_FRAMES, 3, err))
            check("at() before begin_run() REJECTED", sb.at(0) == nullptr, sb.error());
    }
    // A missing file is an open failure, not an empty stream.
    {
        vot::StreamBlob sb;
        check("missing blob REJECTED",
              !sb.open(dir + "/does_not_exist.raw", STREAM_FB, STREAM_FRAMES, 3, err),
              err);
    }
}

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
// ---------------------------------------------------------------------------
// THE STREAMED READER ON A REAL BLOB, at the real geometry. Skipped without
// $VOT_ROOT.
//
// The fixtures above use 40-byte frames, which exercise the ring's bookkeeping
// and nothing about a 7.9 MB pread over a real filesystem: a partial read is
// impossible at 40 bytes and routine at 8 MB, and it is the one failure the
// resident reader's single whole-blob read never had to handle. So this walks a
// slice of a real forward job and a real backward job on the LARGEST blob
// present -- which is one of the five sequences this reader exists for -- and
// compares every byte against an independent fopen/fread, a different mechanism
// from the pread the producer uses.
struct RealPick {
    std::string dir, manifest;
    vot::Manifest m;
    size_t bytes = 0;
};

bool pick_largest_blob(RealPick &out)
{
    const char *root = getenv("VOT_ROOT");
    if (!root) return false;
    for (const char *sub : {"/data-rgb", "/data"}) {
        const std::string dir = std::string(root) + sub;
        FILE *p = popen(("ls " + dir + "/*.json 2>/dev/null").c_str(), "r");
        if (!p) continue;
        char line[1024];
        while (fgets(line, sizeof line, p)) {
            std::string path(line);
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
                path.pop_back();
            vot::Manifest m;
            std::string err;
            if (!vot::manifest_load(path, m, err)) continue;
            const size_t b = m.frame_bytes * (size_t)m.frames;
            if (b <= out.bytes) continue;
            FILE *bf = fopen((dir + "/" + m.blob).c_str(), "rb");
            if (!bf) continue;                    // manifest without its blob
            fclose(bf);
            out.dir = dir; out.manifest = path; out.m = m; out.bytes = b;
        }
        pclose(p);
    }
    return out.bytes > 0;
}

// One frame, read by a mechanism the reader under test does not use.
bool read_frame_directly(const std::string &path, size_t fb, int index,
                         std::vector<uint8_t> &out)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.assign(fb, 0);
    const bool ok = (fseeko(f, (off_t)index * (off_t)fb, SEEK_SET) == 0) &&
                    (fread(out.data(), 1, fb, f) == fb);
    fclose(f);
    return ok;
}

void t_stream_on_real_blob()
{
    printf("\n  StreamBlob on a real blob (skipped without $VOT_ROOT)\n");
    RealPick pick;
    if (!pick_largest_blob(pick)) {
        printf("    no real blob found — skipped\n");
        return;
    }
    const vot::Manifest &m = pick.m;
    const std::string bpath = pick.dir + "/" + m.blob;
    printf("    %s  %dx%dx%d  %d frames  %.0f MB  %.2f MB/frame\n",
           m.sequence.c_str(), m.rows, m.cols, m.channels, m.frames,
           pick.bytes / 1048576.0, m.frame_bytes / 1048576.0);

    // A slice, not the run: this is a per-frame reader test, and 12 frames at
    // this geometry already moves ~100 MB. The ring depth is deliberately below
    // the slice length so it wraps.
    constexpr size_t SLICE = 12;
    std::string err;
    vot::StreamBlob sb;
    const bool bopen = sb.open_blob(bpath, m, 4, err);
    check("real blob opens (length check against the manifest)", bopen,
          bopen ? "" : err);
    if (!bopen) return;

    // Forward from the first anchor, and backward from the last job that runs
    // backward -- the two directions the dataset actually contains.
    for (int backward = 0; backward < 2; ++backward) {
        std::vector<int> order;
        if (!backward)
            for (int i = 0; i < m.frames && order.size() < SLICE; ++i)
                order.push_back(i);
        else
            for (int i = m.frames - 1; i >= 0 && order.size() < SLICE; --i)
                order.push_back(i);

        sb.begin_run(order, order.size());
        int bad = -1;
        std::vector<uint8_t> truth;
        for (size_t k = 0; k < order.size() && bad < 0; ++k) {
            const uint8_t *p = sb.at(k);
            if (!p) { bad = (int)k; break; }
            if (!read_frame_directly(bpath, m.frame_bytes, order[k], truth)) {
                bad = (int)k; break;
            }
            if (memcmp(p, truth.data(), m.frame_bytes) != 0) bad = (int)k;
        }
        check(backward ? "real backward slice matches a direct read"
                       : "real forward slice matches a direct read",
              bad < 0,
              bad < 0 ? std::to_string(m.frame_bytes) + " B/frame x "
                        + std::to_string(order.size())
                      : "run index " + std::to_string(bad) + " " + sb.error());
    }

    // The sidecar, if this manifest has one: the board arms both streams
    // together and they must agree about frame size, which is the one place the
    // two differ.
    if (!m.luma_blob.empty()) {
        const std::string lpath = pick.dir + "/" + m.luma_blob;
        vot::StreamBlob sl;
        const bool lopen = sl.open_luma(lpath, m, 4, err);
        check("real luma sidecar opens at rows*cols", lopen, lopen ? "" : err);
        if (lopen) {
            std::vector<int> order;
            for (int i = 0; i < m.frames && order.size() < SLICE; ++i)
                order.push_back(i);
            sl.begin_run(order, order.size());
            int bad = -1;
            std::vector<uint8_t> truth;
            for (size_t k = 0; k < order.size() && bad < 0; ++k) {
                const uint8_t *p = sl.at(k);
                if (!p || !read_frame_directly(lpath, vot::luma_frame_bytes(m),
                                               order[k], truth)) { bad = (int)k; break; }
                if (memcmp(p, truth.data(), vot::luma_frame_bytes(m)) != 0) bad = (int)k;
            }
            check("real sidecar slice matches a direct read", bad < 0,
                  bad < 0 ? "" : "run index " + std::to_string(bad) + " " + sl.error());
        }
    }
}

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
    t_luma_sidecar();
    t_stream_blob();
    t_stream_mutants();
    t_mutants_are_caught();
    t_real_manifest_if_present();
    t_stream_on_real_blob();
    printf("\n  OVERALL: %s (%d failure%s)\n\n",
           g_failures ? "FAIL" : "PASS", g_failures,
           g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
