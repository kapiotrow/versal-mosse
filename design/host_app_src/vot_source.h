// ---------------------------------------------------------------------------
// vot_source — the board's VOT frame source: blob, manifest, trajectory.
//
// NO XRT AND NO ADF HEADER, deliberately, for exactly the reason mosse_filter
// and scene_colour carry the same rule: everything in here is bookkeeping, and
// bookkeeping is what this project keeps getting wrong SILENTLY. A manifest
// field read into the wrong variable, a backward run emitted in sequence order,
// a trajectory one entry short — none of those crash, and all of them produce a
// complete, plausible, entirely invalid AR report. `make test_vot_source`
// compiles this file with system g++ and runs the mutants in seconds; the
// alternative is a multi-hour board run whose output looks fine.
//
// THE BOARD PARSES A MANIFEST AND NOTHING ELSE. The blob is frames back to back
// with no header (scripts/vot_prepare.py), so a frame is a memcpy at a computed
// offset. Anchors, failure detection, reset policy and AR scoring all live on
// the PC — see runs/vot/phase0b.md for what the toolkit does with the result.
//
// Conventions, all matching the manifest and NOT the toolkit's own:
//   Box is (row, col, h, w) with row/col the CENTRE, the same convention
//   mosse::TargetBox uses. The toolkit's x,y,w,h top-left form appears in
//   exactly one place — Trajectory::write() — and nowhere else.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vot {

// Frame-pixel box, CENTRE convention. Same fields and meaning as
// mosse::TargetBox, which this deliberately does not include (that header
// pulls in the filter state; this one is used by the converter tests too).
struct Box {
    double row = 0.0, col = 0.0, h = 0.0, w = 0.0;
    bool empty() const { return !(h > 0.0 && w > 0.0); }
};

// One multi-start run. EACH ANCHOR HAS EXACTLY ONE DIRECTION — the toolkit's
// find_anchors() splits the dataset's per-frame anchor values on their SIGN
// into two disjoint lists. An earlier job list emitted both directions per
// interior anchor and overstated the run's cost by 24%; see phase0b.md.
struct Job {
    int  anchor  = 0;
    bool forward = true;
    Box  init_box;
    int  length  = 0;     // frames this run covers, init frame INCLUDED
};

struct Manifest {
    int         schema = 0;
    std::string sequence, blob, luma_blob, gt_format, anchors_source;
    int         rows = 0, cols = 0, channels = 0, frames = 0;
    size_t      frame_bytes = 0;
    std::string blob_md5;
    int         empty_boxes = 0, roi_exceeds_frame = 0;
    double      min_box_side = 0.0;
    std::vector<Box> groundtruth;   // one per frame, for reporting only
    std::vector<Job> jobs;
};

// Both return false and fill `err` rather than throwing or exiting: this runs on
// the board, where an abort in the middle of a 79-minute batch is expensive and
// a message on the console is not.
bool manifest_parse(const char *text, size_t len, Manifest &m, std::string &err);
bool manifest_load(const std::string &path, Manifest &m, std::string &err);

// The frame blob, resident in heap for the whole sequence.
//
// ONE read() PER SEQUENCE, NEVER mmap. mmap over NFS turns staging into demand
// paging, which moves the I/O inside the frame loop as page faults — it would
// surface as unattributed frame time instead of an honest staging slot, which is
// the one thing this whole strategy exists to prevent (phase0a.md).
class Blob {
  public:
    bool load(const std::string &path, const Manifest &m, std::string &err);
    const uint8_t *frame(int i) const;      // nullptr if out of range
    size_t frame_bytes() const { return frame_bytes_; }
    int    frames()      const { return frames_; }
    size_t bytes()       const { return data_.size(); }
    double load_seconds() const { return secs_; }   // the staging slot, measured
  private:
    std::vector<uint8_t> data_;
    size_t frame_bytes_ = 0;
    int    frames_ = 0;
    double secs_ = 0.0;
};

// Frame indices a job visits, IN RUN ORDER: forward [anchor .. frames-1],
// backward [anchor .. 0]. Run order, not sequence order — a backward trajectory
// written in sequence order is read back without complaint and scored as a
// tracker that runs backwards (phase0b.md).
std::vector<int> job_order(const Job &j, int frames);

// Accumulated in RAM, written once at the end of the run. Nothing touches the
// NFS mount inside a frame loop.
class Trajectory {
  public:
    void begin(const Job &j);
    // The anchor frame. Written as Special(INITIALIZATION), NOT as the init box:
    // trajectory index 0 is a special code in every VOT result file.
    void push_init(double ms);
    void push(const Box &b, double ms);
    size_t size() const { return kinds_.size(); }
    // <dir>/<seq>_<anchor:08d>.txt  plus  ..._time.value
    // Text, not the binary default: the toolkit reads `.txt` regardless of
    // config.results_binary, and a text writer on the board is a printf instead
    // of a struct layout the board would have to keep in step with phase0b.md.
    bool write(const std::string &dir, const std::string &sequence,
               std::string &err) const;
    // EXACTLY the bytes write() emits, for the multi-start determinism test
    // (run job A, job B, job A; A's two trajectories must be byte-identical).
    // Shared with write() rather than reimplemented: a comparison that carries
    // its own copy of the format proves only that two copies agree.
    std::string as_text() const;
  private:
    std::vector<Box>    boxes_;
    std::vector<double> ms_;
    std::vector<char>   kinds_;      // 'i' = init special, 'r' = rectangle
    int anchor_ = 0, length_ = 0;
};

// INITIALIZATION, the toolkit's Special code for trajectory index 0.
constexpr int SPECIAL_INITIALIZATION = 1;

}  // namespace vot
