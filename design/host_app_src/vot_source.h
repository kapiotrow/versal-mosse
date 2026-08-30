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
// the PC — see docs/thesis/evidence/phase0b.md for what the toolkit does with the result.
//
// Conventions, all matching the manifest and NOT the toolkit's own:
//   Box is (row, col, h, w) with row/col the CENTRE, the same convention
//   mosse::TargetBox uses. The toolkit's x,y,w,h top-left form appears in
//   exactly one place — Trajectory::write() — and nowhere else.
// ---------------------------------------------------------------------------
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
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

// THE SIDECAR IS ONE PLANE, whatever the blob's channel count is. This is the
// ONE site that knows it, and both readers below derive their frame size from
// here. It used to live inside Blob::load_luma alone; StreamBlob would have been
// a second copy, and the failure that copy reintroduces -- reading
// rows*cols*channels per luma frame -- does not crash. It reads a stripe of the
// wrong frame inside a correctly-sized window and hands it to scale_extract,
// which degrades the scale filter and nothing else in the system notices.
// t_luma_sidecar asserts BOTH directions for BOTH readers for that reason.
inline size_t luma_frame_bytes(const Manifest &m)
{
    return (size_t)m.rows * (size_t)m.cols;
}

// The frame blob, resident in heap for the whole sequence.
//
// ONE read() PER SEQUENCE, NEVER mmap. mmap over NFS turns staging into demand
// paging, which moves the I/O inside the frame loop as page faults — it would
// surface as unattributed frame time instead of an honest staging slot, which is
// the one thing this whole strategy exists to prevent (phase0a.md).
class Blob {
  public:
    bool load(const std::string &path, const Manifest &m, std::string &err);
    // THE LUMA SIDECAR, at channels=3 only. Same container and the same one
    // read(); the ONLY difference is the frame size, rows*cols instead of
    // rows*cols*channels.
    //
    // It exists because scale_extract() is an INTENSITY template matcher and a
    // VOT frame arrives pixel-interleaved. Deriving luma on the board would put
    // a rows*cols BT.601 pass inside the frame loop; shipping the plane costs
    // 33% of the blob and nothing per frame. The convention is the manifest's
    // `luma_convention` and it is NOT PIL's `convert('L')` -- see vot_prepare.py.
    bool load_luma(const std::string &path, const Manifest &m, std::string &err);
    const uint8_t *frame(int i) const;      // nullptr if out of range
    size_t frame_bytes() const { return frame_bytes_; }
    int    frames()      const { return frames_; }
    size_t bytes()       const { return data_.size(); }
    double load_seconds() const { return secs_; }   // the staging slot, measured
  private:
    bool load_sized(const std::string &path, size_t frame_bytes, int frames,
                    std::string &err);
    std::vector<uint8_t> data_;
    size_t frame_bytes_ = 0;
    int    frames_ = 0;
    double secs_ = 0.0;
};

// ---------------------------------------------------------------------------
// StreamBlob — the same frames as Blob, for a sequence that does not fit in heap.
//
// WHY IT EXISTS. The board maps 2 GB of the VEK280's 12 GB and 512 MB of that is
// CMA, so usable heap is ~0.9-1.2 GB (docs/thesis/evidence/TODO_board_memory.md). Five RGB
// sequences exceed it -- flamingo1 3631 MB, zebrafish1 2373, nature 1482,
// frisbee 1471, girl 1318 -- and the 2026-08-26 full-62 RGB sweep lost exactly
// those five to std::bad_alloc and the OOM killer while the other 57 completed.
//
// WHAT IT IS NOT. It is not mmap: over NFS that turns staging into demand paging
// and moves the I/O into the frame loop as page faults, reported as unattributed
// frame time (phase0a.md). It is not a synchronous refill every K frames either
// -- that amortises to the same bytes on the same critical path and merely makes
// the cost lumpy. It is a ring of K frames filled by a prefetch thread, so the
// wait is (a) usually zero and (b) always MEASURED, in its own slot.
//
// WHAT MAKES IT SIMPLE ENOUGH TO TRUST. The entire access pattern is known
// before the run starts: job_order() is the exact list of dataset indices, in
// the exact order at() will ask for them. So the producer walks a known list and
// the consumer's index is checked against it. OUT-OF-ORDER ACCESS IS AN ERROR,
// NOT A SEEK: silently seeking would make a future caller that reads a frame
// twice, or looks one ahead, quietly serialise the whole run against NFS -- a
// 2.4x frame-time regression that reads as "streaming is slow" instead of as the
// misuse it is.
//
// THE ARITHMETIC IS UNCHANGED, so the acceptance test is free and exact: a
// sequence that fits in heap must produce an IDENTICAL run-state digest in both
// modes. `--vot-stream always` exists to make that comparison runnable on car1.
class StreamBlob {
  public:
    StreamBlob() = default;
    ~StreamBlob();
    StreamBlob(const StreamBlob &) = delete;
    StreamBlob &operator=(const StreamBlob &) = delete;

    // Open and size-check ONLY -- no frame is read until begin_run(). The length
    // check is the same one Blob makes (a blob a byte short or a byte long means
    // the manifest and the file disagree about geometry, and every frame offset
    // after the first is then a guess), and it is made here because open() is the
    // last moment it costs nothing.
    bool open(const std::string &path, size_t frame_bytes, int frames,
              int ring, std::string &err);
    // The two named entry points mirror Blob::load / Blob::load_luma so the
    // per-frame size is derived at the same two sites in both readers.
    bool open_blob(const std::string &path, const Manifest &m, int ring,
                   std::string &err);
    bool open_luma(const std::string &path, const Manifest &m, int ring,
                   std::string &err);

    // Declare the run. `order` is the dataset index per run index, `length` how
    // many of them will actually be visited (--vot-max-frames truncates it, and
    // prefetching past the end would read frames the run never uses). Restarts
    // the prefetch thread, so it is also the per-job re-arm: run_reset() calls
    // it once per anchor, and a job that inherited the previous job's order
    // would stream the right frames in the wrong order -- silent, and scored.
    void begin_run(const std::vector<int> &order, size_t length);

    // Frame for RUN index k. Blocks only if the prefetcher has not reached k.
    // Returns nullptr on an I/O error or on out-of-order access; error() then
    // says which. THE CALLER MUST CHECK -- the resident path can hand a nullptr
    // only for an out-of-range index it never generates, so the existing call
    // sites never had to.
    const uint8_t *at(size_t k);

    const std::string &error() const { return err_; }
    // Seconds blocked in at() SINCE begin_run(), i.e. for the current job. The
    // honest staging slot: a reader that quietly inflates frame time is worse
    // than one that fails.
    double wait_seconds() const { return wait_s_; }
    size_t frame_bytes() const { return frame_bytes_; }
    int    frames()      const { return frames_; }
    int    ring()        const { return ring_; }
    size_t ring_bytes()  const { return frame_bytes_ * (size_t)ring_; }
    bool   is_open()     const { return fd_ >= 0; }
    void   close();

  private:
    void stop_thread();
    void producer();

    int    fd_ = -1;
    size_t frame_bytes_ = 0;
    int    frames_ = 0;
    int    ring_ = 0;
    double wait_s_ = 0.0;
    std::string path_, err_;

    std::vector<uint8_t> buf_;        // ring_ frames, contiguous
    std::vector<int>     order_;
    size_t len_ = 0;                  // frames this run will visit

    std::thread             th_;
    std::mutex              mu_;
    std::condition_variable cv_prod_, cv_cons_;
    size_t produced_ = 0;             // frames fully written into the ring
    size_t consumed_ = 0;             // lowest run index still needed
    // The next run index at() will accept. SEPARATE from consumed_ on purpose:
    // consumed_ is what the producer may overwrite past, served_ is what the
    // caller must ask for next, and collapsing the two into one counter makes
    // at(1) look out of order while the ring is still correct.
    size_t served_ = 0;
    bool   stop_ = false;
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
