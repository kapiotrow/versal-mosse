/*
 * mosse_graph.cpp
 * Instantiation and aiesim entry point for the MOSSE AIE graph.
 *
 * In the new architecture there is one FFT2D instance and one IFFT2D instance
 * (serial per-channel processing). All inter-stage data lives in DDR and is
 * exchanged via GMIO; the APU (or the aiesim host below) orchestrates transfers.
 *
 * === Data-driven aiesim scenario testing ===
 *
 * Test data is loaded at runtime from the scenario directory specified by
 * the AIESIM_SCENARIO_DIR environment variable (default: aiesim_data/s0).
 *
 * Each scenario directory contains:
 *   patch_in.txt       — PLIO input (int8, plio_128_bits)
 *   cmul_filter.bin    — H* as flat int16 LE pairs
 *   cmul_accum.bin     — accum_prev as flat int16 LE pairs
 *   expected.txt       — pass/fail bounds (key-value pairs)
 *
 * Generate all scenarios with:  make gen_vectors
 * Run a scenario with:  make aiesim SCENARIO=s0  (or s1..s4)
 *
 * Scenarios (see scripts/gen_aiesim_vectors.py):
 *   s0  impulse@(0,0),  H*={1,0},    acc={1,0}    — baseline accumulation
 *   s1  impulse@(17,42),H*={1,0},    acc={0,0}    — spatial localisation
 *   s2  constant patch, H*={1,0},    acc={0,0}    — DC/large-value path
 *   s3  impulse@(0,0),  H*={0,1},    acc={0,0}    — imaginary cross-term sign
 *   s4  impulse@(0,0),  H*=Gaussian, acc={0,0}    — per-element flt_local
 */

#include "mosse_graph.h"

MOSSE_graph mosse_graph;

#ifdef __AIESIM__

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// load_cint16_bin / load_raw_bin / dump_raw_bin. Shared with the kernel-only
// bit-exactness harness (kernel_only_graph.cpp) so the two harnesses cannot
// drift apart in how they read scenario data.
#include "aiesim_scenario_io.h"

// ---------------------------------------------------------------------------
// Data-driven scenario support
// ---------------------------------------------------------------------------

struct ScenarioExpected {
    int peak_idx;
    int peak_re_lo, peak_re_hi;
    int peak_im_lo, peak_im_hi;
    int max_noise;
    // Allowed peak displacement in PIXELS (Chebyshev radius). 0 = exact argmax.
    // Smooth responses (e.g. s6's) have near-flat peaks: for a sigma=8 Gaussian the
    // true maximum exceeded its neighbour by 0.78%, i.e. under 1 LSB at low
    // amplitude, so the argmax was decided by rounding. Exact equality there tests
    // luck, not correctness; +/-1 px is what a tracker actually needs.
    int peak_tol;
    int skip_snr;
    // RELATIVE peak-to-sidelobe assertion, in percent. The peak must be at least
    // this fraction of the largest element OUTSIDE an 11x11 window around it
    // (Bolme §3.5). Scale-invariant, so unlike max_noise it survives a change to
    // the shift budget — which is why s7 uses it and max_noise is left at 0 there.
    // 0 disables.
    int snr_ratio_pct;
    // Minimum normalized correlation (percent) between the drained
    // gmio_fft_col_out tap and the golden fft_col_out.bin. 0 disables.
    int fcol_corr_pct;
    int check_accum0;
    int accum0_re, accum0_im;
};

static bool load_expected(const char *path, ScenarioExpected *e)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[aiesim] ERROR: cannot open %s\n", path);
        return false;
    }
    // Set defaults
    e->peak_idx   = 0;
    e->peak_re_lo = 1; e->peak_re_hi = 100;
    e->peak_im_lo = -10; e->peak_im_hi = 10;
    e->max_noise  = 4;
    e->peak_tol   = 0;   // default: exact argmax, preserving s0-s4 behaviour
    e->skip_snr   = 0;
    e->snr_ratio_pct = 0;   // default off: s0-s4 assert an absolute noise floor
    e->fcol_corr_pct = 0;   // default off: only the conv2d scenarios have a golden
    e->check_accum0 = 0;
    e->accum0_re  = 0; e->accum0_im = 0;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char key[64]; int val;
        if (sscanf(line, "%63s %d", key, &val) != 2) continue;
        if (!strcmp(key, "peak_idx"))     e->peak_idx     = val;
        if (!strcmp(key, "peak_re_lo"))   e->peak_re_lo   = val;
        if (!strcmp(key, "peak_re_hi"))   e->peak_re_hi   = val;
        if (!strcmp(key, "peak_im_lo"))   e->peak_im_lo   = val;
        if (!strcmp(key, "peak_im_hi"))   e->peak_im_hi   = val;
        if (!strcmp(key, "max_noise"))    e->max_noise     = val;
        if (!strcmp(key, "peak_tol"))     e->peak_tol      = val;
        if (!strcmp(key, "skip_snr"))     e->skip_snr      = val;
        if (!strcmp(key, "snr_ratio_pct")) e->snr_ratio_pct = val;
        if (!strcmp(key, "fcol_corr_pct")) e->fcol_corr_pct = val;
        if (!strcmp(key, "check_accum0")) e->check_accum0  = val;
        if (!strcmp(key, "accum0_re"))    e->accum0_re     = val;
        if (!strcmp(key, "accum0_im"))    e->accum0_im     = val;
    }
    fclose(f);
    return true;
}

// In-place 2D transpose of a PATCH_ROWS × PATCH_COLS cint16 matrix stored
// as int16_t pairs (re, im) in row-major order.
static void transpose_inplace(int16_t *buf, int rows, int cols)
{
    int16_t *tmp = new int16_t[rows * cols * 2];
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            tmp[(c * rows + r) * 2]     = buf[(r * cols + c) * 2];
            tmp[(c * rows + r) * 2 + 1] = buf[(r * cols + c) * 2 + 1];
        }
    memcpy(buf, tmp, rows * cols * 4);
    delete[] tmp;
}

int main(int argc, char **argv)
{
    // ----------------------------------------------------------------
    // Scenario directory — set via AIESIM_SCENARIO_DIR env var.
    // Falls back to "aiesim_data/s0" (relative to BUILD_DIR).
    // ----------------------------------------------------------------
    const char *scenario_dir = getenv("AIESIM_SCENARIO_DIR");
    if (!scenario_dir || scenario_dir[0] == '\0')
        scenario_dir = "aiesim_data/s0";
    printf("[aiesim] scenario: %s\n", scenario_dir);
    // Report the shift budget the BINARY was built with, not what the Makefile
    // currently says. A flag-only change that silently reused a stale libadf.a has
    // produced convincing false results on this project before; a log that names
    // its own build parameters is the cheapest defence.
    printf("[aiesim] shifts: FFT=%d  IFFT_ROW=%d  IFFT_COL=%d  H=%d\n",
           FFT_2D_TP_SHIFT, FFT_2D_TP_IFFT_ROW_SHIFT, FFT_2D_TP_IFFT_COL_SHIFT,
           CMUL_H_SHIFT);
    fflush(stdout);

    // Make every log self-describing about WHICH conv2d it was built with.
    // Twice now, a run was interpreted against the wrong build: MODE=2 emits a
    // synthetic ramp and ignores patch_in entirely, so scenario expectations
    // cannot possibly match and the "failure" is meaningless. Printing the mode
    // (and what it implies) makes that obvious from the log alone.
#ifndef CONV2D_ECHO_TEST
#  define CONV2D_ECHO_TEST (-1)
#endif
#ifndef N_CHANNELS
#  define N_CHANNELS 1
#endif
    printf("[aiesim] conv2d build mode: CONV2D_ECHO_TEST=%d  (%s)\n",
           (int)CONV2D_ECHO_TEST,
           (CONV2D_ECHO_TEST == 0) ? "real 3x3 convolution" :
           (CONV2D_ECHO_TEST == 1) ? "echo patch_in (scenario values apply)" :
           (CONV2D_ECHO_TEST == 2) ? "SYNTHETIC RAMP — ignores patch_in, "
                                     "scenario expectations DO NOT APPLY" :
                                     "unknown");
    if (CONV2D_ECHO_TEST == 2)
        printf("[aiesim] WARNING: MODE=2 is a dataflow bisect build. Any PASS/FAIL "
               "against scenario data below is meaningless.\n");
    fflush(stdout);

    // ----------------------------------------------------------------
    // GMIO::malloc buffers — aiesim requires DMA buffers to be allocated
    // via GMIO::malloc so the GMIO model can track their addresses.
    // Regular static/heap memory is rejected by gm2aie_nb / aie2gm_nb.
    // ----------------------------------------------------------------
    constexpr int PATCH_ELEMS = PATCH_ROWS * PATCH_COLS;
    constexpr int PATCH_BYTES = PATCH_ELEMS * 4;   // cint16 = 4 B/sample

    // Combined cmul input: [filter_chunk_c | accum_chunk_c] per kernel invocation.
    // Layout interleaved by chunk so each 2*CHUNK-element kernel window sees both halves.
    // See cmul_accum_kernel.h for rationale (single GMIO buffer avoids ISS deadlock).
    constexpr int CHUNK_ELEMS   = PATCH_COLS * FFT_COL_WS;  // 256 cint16 per chunk
    constexpr int CHUNK_BYTES   = CHUNK_ELEMS * 4;           // 1024 B per chunk
    constexpr int N_CHUNKS      = PATCH_ROWS / FFT_COL_WS;  // 64 chunks
    constexpr int CMUL_IN_BYTES = PATCH_BYTES * 2;           // 131072 B total

    int8_t  *weights_buf   = (int8_t*)  GMIO::malloc(64);
    int16_t *filter_buf    = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *accum_prev_buf= (int16_t*) GMIO::malloc(PATCH_BYTES);  // loaded from file
    int16_t *cmul_in_buf   = (int16_t*) GMIO::malloc(CMUL_IN_BYTES);
    int16_t *fft_scratch   = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *accum_buf     = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *resp_buf      = (int16_t*) GMIO::malloc(PATCH_BYTES);
    // Drain target for the gmio_fft_col_out broadcast tap (F_ch).
    int16_t *fcol_out_buf  = (int16_t*) GMIO::malloc(PATCH_BYTES);

    // Load channel-0 conv2d weights from the scenario directory.
    // Generated by gen_aiesim_vectors.py from layer0_weights.bin (make weights).
    // Falls back to all-zeros (no-op conv) if the file is absent.
    memset(weights_buf, 0, 64);
    {
        char wpath[512];
        snprintf(wpath, sizeof(wpath), "%s/weights_ch0.bin", scenario_dir);
        FILE *wf = fopen(wpath, "rb");
        if (wf) {
            size_t nr = fread(weights_buf, 1, 64, wf);
            fclose(wf);
            printf("[aiesim] Loaded conv2d weights from %s (%zu bytes)\n", wpath, nr);
        } else {
            printf("[aiesim] WARNING: %s not found — using zero weights\n", wpath);
            printf("[aiesim]          Run 'make weights && make gen_vectors' to generate.\n");
        }
        fflush(stdout);
    }

    // ----------------------------------------------------------------
    // Load per-scenario filter and accum_prev from binary files.
    // ----------------------------------------------------------------
    char filter_path[512], accum_path[512], expected_path[512];
    snprintf(filter_path,   sizeof(filter_path),   "%s/cmul_filter.bin", scenario_dir);
    snprintf(accum_path,    sizeof(accum_path),     "%s/cmul_accum.bin",  scenario_dir);
    snprintf(expected_path, sizeof(expected_path),  "%s/expected.txt",    scenario_dir);

    if (!load_cint16_bin(filter_path,   filter_buf,     PATCH_ELEMS)) { _exit(3); }
    if (!load_cint16_bin(accum_path,    accum_prev_buf, PATCH_ELEMS)) { _exit(3); }

    ScenarioExpected exp;
    if (!load_expected(expected_path, &exp)) { _exit(3); }

    printf("[aiesim] expected: peak_idx=%d  re=[%d,%d]  im=[%d,%d]  max_noise=%d%s\n",
           exp.peak_idx, exp.peak_re_lo, exp.peak_re_hi,
           exp.peak_im_lo, exp.peak_im_hi, exp.max_noise,
           exp.skip_snr ? "  (SNR check skipped)" : ""); fflush(stdout);

    // cmul_in ([filter | accum_prev] interleaved by chunk) is packed inside the
    // channel loop in step 4/5, because accum_prev changes on every channel.
    memset(cmul_in_buf, 0, CMUL_IN_BYTES);

    mosse_graph.init();

    // ----------------------------------------------------------------
    // PRE-LOAD weights BEFORE run(-1); arm output GMIOs AFTER run(-1).
    //
    // RC-1 (weights deadlock, fixed): the PLIO model starts streaming
    // patch_in.txt once PL-Interface is configured (inside init()).  After
    // run(-1) enables the cores the PLIO consumes ALL cycle credits in
    // cycle-approximate mode, starving the weights GMIO DMA.  Fix: load
    // weights between init() and run(-1) while the PLIO model is still idle
    // (cores not yet enabled) so the 64-byte DMA completes uncontested.
    //
    // RC-2 (output GMIO before run, now fixed): calling aie2gm_nb on an
    // output GMIO before run(-1) sets up a DMA descriptor before the AIE
    // output channel is active; in Vitis 2025.2 cycle-approximate mode that
    // DMA never captures post-run output.  Output GMIOs must be armed after
    // run(-1).  In cycle-approximate mode zero simulation cycles advance
    // between run(-1) and the next API call, so there is no back-pressure
    // risk from the FFT's output FIFO filling before the DMA is armed.
    // ----------------------------------------------------------------
    // NOTE: this pre-load satisfies conv2d's FIRST firing only. The `weights`
    // input_buffer is acquired once per invocation, so the remaining
    // CONV_INVOCATIONS-1 buffers must be fed as the kernel runs — see the
    // per-invocation gm2aie_nb in the fft_row_out drain loop below.
    mosse_graph.gmio_weights.gm2aie_nb(weights_buf, 64);
    mosse_graph.gmio_weights.wait();   // completes while PLIO is idle (cores not yet enabled)

    mosse_graph.run(-1);   // enables cores + PLIO; conv2d starts immediately (weights ready)

    // PatchIn PLIO reads from aiesim_data/patch_in.txt (impulse at (0,0),
    // value=1) followed by padding zeros — see gen_aiesim_vectors.py.
    // The padding keeps the PLIO model "active" (not stalled) for the full
    // ~50 000-cycle test so cycle-credit starvation cannot occur.
    // conv2d casts int8 → cint16, feeds fft2d.fft_rows.

    // ----------------------------------------------------------------
    // Step 2/3: Populate fft_scratch with the (transposed) row-FFT output.
    //
    // Cycle-approximate aiesim has known limitations with the PLIO → stream →
    // window adapter: for non-trivial patches the sample ordering delivered to
    // the row-FFT kernel is wrong, so the row-FFT output is incorrect for any
    // impulse that is not at position 0.
    //
    // Workaround: gen_aiesim_vectors.py pre-computes the expected transposed
    // row-FFT output (= the col-FFT input) in floating-point and writes it to
    // fft_col_in.bin.  When this file is present we load it directly, bypassing
    // the PLIO → conv2d → row_FFT → transpose chain entirely.  The gmio_fft_row_out
    // GMIO still runs (to keep the graph alive) but its output is discarded.
    // ----------------------------------------------------------------
#if MEMTILE_TRANSPOSE
    // ----------------------------------------------------------------
    // MEMTILE PATH: steps 2 and 3 DO NOT EXIST.
    //
    // The row->col transpose closes inside the graph through memTileFwd, so
    // there is no fft_row_out to drain, nothing to transpose on the host, and
    // no fft_col_in to feed. The pre-computed fft_col_in.bin bypass is also
    // unreachable by construction — which is a feature: this harness can only
    // run the REAL PatchIn -> conv2d -> row FFT -> transpose -> col FFT chain,
    // so a PASS here is a statement about the whole path.
    //
    // What still has to happen is the weights feed. conv2d's `weights` is an
    // input_buffer and ADF acquires one per FIRING, so the harness must supply
    // PATCH_ROWS/FFT_ROW_WS of them per channel or conv2d blocks on the acquire
    // and the row FFT starves — the root cause of every historical "PLIO hang".
    // Previously they were fed one per drained invocation, which paced them for
    // free. With the drain gone they are queued up front; gm2aie_nb does not
    // block, and ADF consumes them in order.
    {
        constexpr int N_INV = PATCH_ROWS / FFT_ROW_WS;
        printf("[aiesim] steps 2/3: SKIPPED — memTileFwd does the transpose "
               "in-graph. Queueing %d weights buffers for conv2d.\n", N_INV);
        fflush(stdout);
        for (int inv = 0; inv < N_INV; ++inv)
            mosse_graph.gmio_weights.gm2aie_nb(weights_buf, 64);
    }
#else
    char fci_path[512];
    snprintf(fci_path, sizeof(fci_path), "%s/fft_col_in.bin", scenario_dir);
    bool use_precomputed = load_cint16_bin(fci_path, fft_scratch, PATCH_ELEMS);

    if (use_precomputed) {
        printf("[aiesim] step 2/3: loaded pre-computed fft_col_in from %s\n", fci_path);
        // OBSOLETE BYPASS — kept only so existing scenarios still run unchanged.
        //
        // This branch used to be justified by "the PLIO→stream→window adapter
        // never sets the FFT row tile's input lock, so the drain hangs forever".
        // That diagnosis was WRONG, and was disproved on 2026-07-30:
        //   * the real cause of the hang was weights-buffer starvation — conv2d's
        //     `weights` input_buffer is acquired once per FIRING, and the harness
        //     supplied one buffer per PATCH (see the drain loop below);
        //   * with weights fed per invocation, the drain completes normally;
        //   * a row-FFT scan then put the s1 impulse's energy in row 17, exactly
        //     where it belongs — so PatchIn delivers correctly AND in the right
        //     order, including for off-centre impulses.
        // There is also no stream-to-window adapter here any more: mosse_graph.h
        // wires conv2d's output_buffer straight into fft_row_in.
        //
        // Consequence: taking this branch SKIPS the real PatchIn→conv2d→row-FFT
        // path, so plain `make aiesim` proves nothing about it — which is exactly
        // why aiesim "passed" for months while hw_emu hung. Use `make aiesim_plio`
        // (deletes fft_col_in.bin) to exercise the real path.
        printf("[aiesim] step 2/3: BYPASS — skipping the real PatchIn->conv2d->row-FFT path\n");
        printf("[aiesim]           col-FFT is fed from pre-computed fft_col_in.bin.\n");
        printf("[aiesim]           This run does NOT validate PatchIn; use 'make aiesim_plio'.\n");
        fflush(stdout);
    } else {
        // The REAL path: capture row-FFT output and transpose. This is what
        // `make aiesim_plio` exercises, and it works for off-centre impulses too
        // (verified with s1: impulse@(17,42) lands in row 17). The old claim that
        // it "fails for off-centre impulses" was a symptom of weights starvation,
        // not of PLIO sample ordering.
        constexpr int FFT_INV_BYTES = PATCH_COLS * FFT_ROW_WS * 4;
        constexpr int N_INV         = PATCH_ROWS / FFT_ROW_WS;
        printf("[aiesim] step 2: waiting for fft_row_out (%d × %d B) [fallback]...\n",
               N_INV, FFT_INV_BYTES); fflush(stdout);
        for (int inv = 0; inv < N_INV; ++inv) {
            // conv2d's `weights` is an input_buffer, and ADF acquires every input
            // buffer before each firing — so conv2d consumes ONE 64-byte weights
            // buffer per invocation, not one per patch. The single pre-load above
            // only satisfies firing 0; conv2d then blocks forever on the next
            // acquire and the row-FFT starves, which is what made this drain hang
            // (and hangs identically with CONV2D_MODE=2, where the kernel never
            // even touches patch_in — proving it is the weights port, not the PLIO).
            //
            // Feeding one buffer per drained invocation keeps supply matched to
            // consumption. No wait() on an input GMIO (see note below): waiting
            // here would block until conv2d fires, which is what we are unblocking.
            mosse_graph.gmio_weights.gm2aie_nb(weights_buf, 64);

            mosse_graph.gmio_fft_row_out.aie2gm_nb(
                (int8_t*)fft_scratch + inv * FFT_INV_BYTES, FFT_INV_BYTES);
            mosse_graph.gmio_fft_row_out.wait();
        }
        printf("[aiesim] step 2: fft_row_out done\n");
        printf("[aiesim] fft_row_out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
               fft_scratch[0], fft_scratch[1], fft_scratch[2], fft_scratch[3],
               fft_scratch[4], fft_scratch[5], fft_scratch[6], fft_scratch[7]);

        // ------------------------------------------------------------
        // Did the patch actually arrive over PatchIn, and at what offset?
        //
        // conv2d completing does NOT mean it got the right data — if the ISS
        // PLIO model starts streaming patch_in.txt at init() (before run(-1)
        // enables the cores), the leading words are consumed while nothing is
        // reading, and conv2d ends up in the trailing zero padding. That looks
        // identical to "delivered correctly" from the drain loop's point of view:
        // no hang, just zeros.
        //
        // Scan the RAW row-FFT output (pre-transpose). For scenario s1 the
        // impulse sits in row 17, so a correct delivery puts energy in row 17's
        // 64 bins and nowhere else. The row we actually find the energy in gives
        // the delivery offset in rows; all-zero means nothing arrived at all.
        {
            int nz = 0, max_abs = 0, max_idx = -1;
            for (int i = 0; i < (int)PATCH_ELEMS; ++i) {
                int re = fft_scratch[2*i], im = fft_scratch[2*i + 1];
                int a  = (re < 0 ? -re : re) + (im < 0 ? -im : im);
                if (a != 0) ++nz;
                if (a > max_abs) { max_abs = a; max_idx = i; }
            }
            if (max_idx < 0) {
                printf("[aiesim] row-FFT scan: ALL ZERO — nothing arrived over PatchIn\n");
            } else {
                // A single-impulse patch should light up exactly ONE row (its own)
                // = PATCH_COLS non-zero bins. Far more than that means the input
                // was not the expected impulse: e.g. 4096/4096 non-zero is the
                // MODE=2 ramp, not any scenario patch.
                printf("[aiesim] row-FFT scan: %d/%d non-zero, max|.|=%d at idx=%d "
                       "(row=%d, bin=%d)  [impulse scenario expects %d non-zero, "
                       "all in the impulse's row]\n",
                       nz, (int)PATCH_ELEMS, max_abs, max_idx,
                       max_idx / PATCH_COLS, max_idx % PATCH_COLS,
                       (int)PATCH_COLS);
            }
            fflush(stdout);
        }

        transpose_inplace(fft_scratch, PATCH_ROWS, PATCH_COLS);
        fflush(stdout);
    }
    printf("[aiesim] fft_col_in[col=0,r=0]:  {%d,%d}\n", fft_scratch[0], fft_scratch[1]);
    printf("[aiesim] fft_col_in[col=0,r=17]: {%d,%d}\n",
           fft_scratch[17*2], fft_scratch[17*2+1]);
    fflush(stdout);
#endif  // !MEMTILE_TRANSPOSE

    // Step 4: fft_cols + cmul → accum_out
    // Arm output GMIO first, then fire all input GMIOs.
    // Do NOT call wait() on input GMIOs (gm2aie_nb) — only output GMIOs need wait().
    // Calling input wait() blocks until the kernel consumes the buffer, which causes
    // a deadlock if the kernel cannot run while we're holding the thread here.
    // col FFT + cmul: one invocation at a time (same fix as fft_row_out above).
    // CMUL kernel invocation size matches col-FFT window size (PATCH_COLS*FFT_COL_WS*4).
    constexpr int CMUL_INV_BYTES  = PATCH_COLS * FFT_COL_WS * 4;       // 1024 B per invocation
    constexpr int CMUL_IN_INV_BYTES = CMUL_INV_BYTES * 2;               // 2048 B (filter+accum)
    constexpr int N_CMUL_INV      = PATCH_ROWS / FFT_COL_WS;            // 64
    // ----------------------------------------------------------------
    // Channel loop.
    //
    // The real host processes channels serially, re-running cmul_accum per channel
    // and carrying the accumulator through DDR. This harness had NO channel loop at
    // all, so N_CHANNELS was never exercised and the cint16 DDR accumulator's
    // headroom was pure speculation.
    //
    // SCOPE — this measures ACCUMULATOR HEADROOM, not 16 distinct channels:
    // the same col-FFT spectrum and the same filter are reused for every channel,
    // because producing 16 genuinely different spectra would need 16 patches through
    // conv2d (16x the PLIO stimulus and weights). That is deliberate and it makes
    // the test STRONGER for this purpose: with identical channels the answer is known
    // in closed form — after channel k the accumulator must be exactly (k+1)x the
    // single-channel value — so any deviation is overflow or arithmetic error, not
    // an artifact of channel-to-channel variation.
    //
    // Real filters differ per channel and partially cancel, so this is the COHERENT
    // WORST CASE for headroom. If it fits here, it fits in practice.
    printf("[aiesim] step 4/5: %d channel(s) x %d invocations x %d B\n",
           (int)N_CHANNELS, N_CMUL_INV, CMUL_INV_BYTES); fflush(stdout);
    if (N_CHANNELS > 1)
        printf("[aiesim]   NOTE: scenario expectations are calibrated for ONE channel;\n"
               "[aiesim]         with N_CHANNELS>1 read the per-channel table, not PASS/FAIL.\n");

    int  first_sat_ch = -1;
    for (int ch = 0; ch < (int)N_CHANNELS; ++ch) {

        // Pack [filter | accum_prev] per chunk. Channel 0 uses the scenario's
        // accum_prev from file; later channels carry forward the previous result.
        const int16_t *acc_prev = (ch == 0) ? accum_prev_buf : accum_buf;
        for (int c = 0; c < N_CHUNKS; ++c) {
            memcpy((int8_t*)cmul_in_buf + (size_t)c * 2 * CHUNK_BYTES,
                   (const int8_t*)filter_buf + (size_t)c * CHUNK_BYTES, CHUNK_BYTES);
            memcpy((int8_t*)cmul_in_buf + (size_t)c * 2 * CHUNK_BYTES + CHUNK_BYTES,
                   (const int8_t*)acc_prev + (size_t)c * CHUNK_BYTES, CHUNK_BYTES);
        }

        for (int inv = 0; inv < N_CMUL_INV; ++inv) {
            mosse_graph.gmio_accum_out.aie2gm_nb(
                (int8_t*)accum_buf    + inv * CMUL_INV_BYTES,   CMUL_INV_BYTES);
            // The F_ch tap is 1:1 with the accumulator output by construction —
            // both are driven by the same col-FFT invocation — so it arms in the
            // same iteration and needs no loop of its own. Armed BEFORE the inputs
            // fire, like accum_out, so a full output window is never what blocks.
            mosse_graph.gmio_fft_col_out.aie2gm_nb(
                (int8_t*)fcol_out_buf + inv * CMUL_INV_BYTES,   CMUL_INV_BYTES);
            mosse_graph.gmio_cmul_in.gm2aie_nb(
                (int8_t*)cmul_in_buf  + inv * CMUL_IN_INV_BYTES, CMUL_IN_INV_BYTES);
#if !MEMTILE_TRANSPOSE
            mosse_graph.gmio_fft_col_in.gm2aie_nb(
                (int8_t*)fft_scratch  + inv * CMUL_INV_BYTES,   CMUL_INV_BYTES);
#endif
            mosse_graph.gmio_fft_col_out.wait();
            mosse_graph.gmio_accum_out.wait();
        }

        // Per-channel saturation, not just at the end: a later channel can pull a
        // clipped element back below the rail, so a final-state-only check would
        // miss that information was already lost.
        int ch_sat = 0, ch_max = 0;
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int re = accum_buf[i*2], im = accum_buf[i*2 + 1];
            if (re == 32767 || re == -32768 || im == 32767 || im == -32768) ++ch_sat;
            int a = (re < 0 ? -re : re) + (im < 0 ? -im : im);
            if (a > ch_max) ch_max = a;
        }
        if (ch_sat && first_sat_ch < 0) first_sat_ch = ch;

        // With identical channels accum[0] must be exactly (ch+1)x the ch-0 value.
        printf("[aiesim]   ch %2d: accum[0]={%d,%d}  max|.|=%d  rails=%d%s\n",
               ch, accum_buf[0], accum_buf[1], ch_max, ch_sat,
               ch_sat ? "   <-- SATURATED" : "");
        fflush(stdout);
    }
    if (first_sat_ch >= 0)
        printf("[aiesim] HEADROOM EXCEEDED: first saturation at channel %d of %d — the "
               "cint16 DDR accumulator cannot hold this many channels\n",
               first_sat_ch, (int)N_CHANNELS);
    printf("[aiesim] step 4/5: accum_out done\n"); fflush(stdout);

    printf("[aiesim] accum_out[0..3]:   {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           accum_buf[0], accum_buf[1],
           accum_buf[2], accum_buf[3],
           accum_buf[4], accum_buf[5],
           accum_buf[6], accum_buf[7]);
    // Scan accum_out for first non-zero element and maximum magnitude.
    {
        int max_mag = 0, max_idx = -1;
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int mag = (int)accum_buf[i*2] * accum_buf[i*2] +
                      (int)accum_buf[i*2+1] * accum_buf[i*2+1];
            if (mag > max_mag) { max_mag = mag; max_idx = i; }
        }
        if (max_idx >= 0)
            printf("[aiesim] accum_out max: {%d,%d} at idx=%d (r=%d,c=%d)  mag^2=%d\n",
                   accum_buf[max_idx*2], accum_buf[max_idx*2+1], max_idx,
                   max_idx/PATCH_COLS, max_idx%PATCH_COLS, max_mag);
        else
            printf("[aiesim] accum_out: ALL ZERO\n");
        fflush(stdout);
    }

    // ----------------------------------------------------------------
    // Saturation check.
    //
    // cmul_accum now clamps instead of wrapping (it used to sign-flip the whole
    // spectrum on overflow). Clamping is the right failure mode, but it is still
    // a failure: a clipped channel contributes less than it should, and the
    // accumulated spectrum is wrong in a way that looks merely "disappointing"
    // rather than broken.
    //
    // The kernel cannot report that it saturated, so detect it here: values
    // sitting exactly at the int16 rails are the signature. This matters most for
    // N_CHANNELS=16, where the DDR accumulator is cint16 and 16 x per-channel
    // magnitude must fit 32767.
    //
    // Note a legitimate value can land on a rail by chance, so a tiny count is
    // not proof of overflow — but any nonzero count on a scenario whose expected
    // magnitudes are far from the rails is worth investigating.
    bool sat_pass = true;   // folded into the OVERALL verdict at the end
    {
        int sat_re = 0, sat_im = 0, first_idx = -1;
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int re = accum_buf[i*2], im = accum_buf[i*2 + 1];
            bool r = (re ==  32767 || re == -32768);
            bool m = (im ==  32767 || im == -32768);
            if (r) ++sat_re;
            if (m) ++sat_im;
            if ((r || m) && first_idx < 0) first_idx = i;
        }
        if (sat_re || sat_im) {
            // Fail the run. None of the current scenarios legitimately reach the
            // rails, and a saturation that only prints is exactly the kind of
            // signal that gets scrolled past in a long log.
            sat_pass = false;
            printf("[aiesim] accum_out SATURATED: %d re + %d im elements at the int16 "
                   "rails (first idx=%d, r=%d, c=%d) — accumulator headroom exceeded\n",
                   sat_re, sat_im, first_idx,
                   first_idx / PATCH_COLS, first_idx % PATCH_COLS);
        } else
            printf("[aiesim] accum_out saturation: none (headroom OK)\n");
        fflush(stdout);
    }

    // ----------------------------------------------------------------
    // IFFT pass
    // ----------------------------------------------------------------

    // Step 6: row IFFT — same per-invocation loop fix as fft_row_out above.
    constexpr int IFFT_INV_BYTES = PATCH_COLS * FFT_ROW_WS * 4;   // 1024 B
    constexpr int N_IFFT_INV     = PATCH_ROWS / FFT_ROW_WS;        // 64
    printf("[aiesim] step 6: waiting for ifft_row_out (%d × %d B)...\n",
           N_IFFT_INV, IFFT_INV_BYTES); fflush(stdout);
    for (int inv = 0; inv < N_IFFT_INV; ++inv) {
#if MEMTILE_TRANSPOSE
        // No ifft_row_out to drain — memTileInv takes it straight to the column
        // IFFT. Only the accumulated spectrum goes in.
        mosse_graph.gmio_ifft_row_in.gm2aie_nb(
            (int8_t*)accum_buf   + inv * IFFT_INV_BYTES, IFFT_INV_BYTES);
#else
        mosse_graph.gmio_ifft_row_out.aie2gm_nb(
            (int8_t*)fft_scratch + inv * IFFT_INV_BYTES, IFFT_INV_BYTES);
        mosse_graph.gmio_ifft_row_in.gm2aie_nb(
            (int8_t*)accum_buf   + inv * IFFT_INV_BYTES, IFFT_INV_BYTES);
        mosse_graph.gmio_ifft_row_out.wait();
#endif
    }
    printf("[aiesim] step 6: ifft_row done\n"); fflush(stdout);
#if MEMTILE_TRANSPOSE
    // The prints and the saturation scan below read fft_scratch, which on the
    // memtile path is NEVER WRITTEN — ifft_row_out does not exist, so nothing
    // drains into it. Printing it anyway produced plausible-looking numbers in
    // the first memtile aiesim run ({16992,26880}...) that mean nothing at all.
    // Say so rather than print them: this project has lost days to output that
    // looks current and is not (see the "H(q15): unchanged" guard in the host).
    printf("[aiesim] ifft_row_out: NOT OBSERVABLE — memTileInv carries the row "
           "IFFT straight to the column IFFT, so there is no host-visible "
           "intermediate. The row-IFFT saturation scan is skipped with it; "
           "rails would show in the response instead.\n");
    fflush(stdout);
    // Declared in BOTH branches: the verdict line and the OVERALL conjunction
    // below read it unconditionally. `true` is the honest value here — the stage
    // is not skipped, it is not OBSERVABLE, so it cannot contribute a FAIL. The
    // saturation it would have caught still shows up in the response check.
    const bool ifft_row_sat_pass = true;
#else

    printf("[aiesim] ifft_row_out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           fft_scratch[0], fft_scratch[1],
           fft_scratch[2], fft_scratch[3],
           fft_scratch[4], fft_scratch[5],
           fft_scratch[6], fft_scratch[7]);

    // ----------------------------------------------------------------
    // Row-IFFT saturation — THE BLIND SPOT.
    //
    // accum_out and response are both checked, but this stage sits BETWEEN them
    // and was unchecked, so a failure here was invisible from either end. That is
    // exactly what happened at N_CHANNELS=16 with budget 3/0/6: the accumulator
    // was clean (30864, rails=0) and the response showed no rails, yet the row
    // IFFT was running at ~101000 against a 32767 ceiling. The only visible
    // symptom was the response scaling 8.8x instead of 16x and the peak drifting
    // 6 px — neither of which names the guilty stage.
    //
    // The row IFFT is the natural place to overflow: its input is the ACCUMULATED
    // spectrum (N_CHANNELS x one channel) and IFFT_ROW_SHIFT defaults to 0, so it
    // takes the full summation growth of the pass with no attenuation.
    bool ifft_row_sat_pass = true;
    {
        int sat = 0, mx = 0;
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int re = fft_scratch[i*2], im = fft_scratch[i*2 + 1];
            if (re == 32767 || re == -32768 || im == 32767 || im == -32768) ++sat;
            int a = (re < 0 ? -re : re) + (im < 0 ? -im : im);
            if (a > mx) mx = a;
        }
        printf("[aiesim] ifft_row_out range: max|.|=%d  rails=%d%s\n", mx, sat,
               sat ? "   <-- SATURATED" : "");
        if (sat) {
            ifft_row_sat_pass = false;
            printf("[aiesim] ROW-IFFT SATURATED: %d elements clipped — raise "
                   "IFFT_ROW_SHIFT (or FFT_SHIFT) and lower IFFT_COL_SHIFT to keep "
                   "2*FFT_SHIFT + IFFT_ROW + IFFT_COL = 12\n", sat);
        }
        fflush(stdout);
    }

#endif  // MEMTILE_TRANSPOSE — end of the ifft_row_out observability block

    // Step 7: APU transpose for IFFT
#if !MEMTILE_TRANSPOSE
    transpose_inplace(fft_scratch, PATCH_ROWS, PATCH_COLS);
#endif

    // Step 8: col IFFT — same per-invocation loop fix.
    constexpr int RESP_INV_BYTES = PATCH_COLS * FFT_COL_WS * 4;   // 1024 B
    constexpr int N_RESP_INV     = PATCH_ROWS / FFT_COL_WS;        // 64
    printf("[aiesim] step 8: waiting for response (%d × %d B)...\n",
           N_RESP_INV, RESP_INV_BYTES); fflush(stdout);
    for (int inv = 0; inv < N_RESP_INV; ++inv) {
        mosse_graph.gmio_response.aie2gm_nb(
            (int8_t*)resp_buf   + inv * RESP_INV_BYTES, RESP_INV_BYTES);
#if !MEMTILE_TRANSPOSE
        mosse_graph.gmio_ifft_col_in.gm2aie_nb(
            (int8_t*)fft_scratch + inv * RESP_INV_BYTES, RESP_INV_BYTES);
#endif
        mosse_graph.gmio_response.wait();
    }
    printf("[aiesim] step 8: response done\n"); fflush(stdout);

    printf("[aiesim] response[0..3]:     {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           resp_buf[0], resp_buf[1],
           resp_buf[2], resp_buf[3],
           resp_buf[4], resp_buf[5],
           resp_buf[6], resp_buf[7]);

    // ----------------------------------------------------------------
    // Response saturation + dynamic-range report.
    //
    // The col IFFT shift (FFT_2D_TP_IFFT_COL_SHIFT) trades two failure modes off
    // against each other, and BOTH are silent without this:
    //   shift too HIGH -> a concentrated spectrum is divided into nothing; the
    //                     response goes all-zero and the argmax is meaningless.
    //                     (This is what shift=12 does to scenario s2.)
    //   shift too LOW  -> the response clips at the int16 rails; the peak flattens
    //                     into a plateau and the argmax moves to whichever clipped
    //                     element happens to come first.
    // Peak location can look "OK" in the second case purely by tie-breaking, so the
    // rail count is the real signal when sweeping the shift.
    bool resp_sat_pass = true;
    {
        int sat = 0, nz = 0, peak_abs = 0;
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int re = resp_buf[i*2], im = resp_buf[i*2 + 1];
            if (re ==  32767 || re == -32768 || im == 32767 || im == -32768) ++sat;
            if (re || im) ++nz;
            int a = (re < 0 ? -re : re) + (im < 0 ? -im : im);
            if (a > peak_abs) peak_abs = a;
        }
        printf("[aiesim] response range: %d/%d non-zero, max|.|=%d\n",
               nz, (int)PATCH_ELEMS, peak_abs);
        if (sat) {
            resp_sat_pass = false;
            printf("[aiesim] response SATURATED: %d elements at the int16 rails — "
                   "IFFT col shift too LOW for this input\n", sat);
        }
        if (nz == 0)
            printf("[aiesim] response ALL ZERO — IFFT col shift too HIGH for this "
                   "spectrum (concentrated spectra get no summation gain)\n");
        fflush(stdout);
    }

    // ----------------------------------------------------------------
    // Optional: verify accum_out[0] against expected
    // ----------------------------------------------------------------
    // Tolerance, not equality. This is a cint16 pipeline: the value crosses two
    // FFT passes and a complex multiply, each rounding. Measured for the s1
    // amplitude-100 impulse: 100 (input sum) -> 97 (row-FFT DC) -> 95 (after
    // col-FFT + cmul), i.e. ~5% cumulative loss. That is the arithmetic working
    // as designed, so an `==` assertion can never pass and only masks the checks
    // below it.
    //
    // ACCUM0_TOL_PCT is a floor of 2 as well, so scenarios still calibrated at
    // amplitude 1 (s0/s2/s3/s4) are not held to a sub-LSB tolerance.
    bool acc0_pass = true;   // folded into the OVERALL verdict at the end
    if (exp.check_accum0) {
        const int ACCUM0_TOL_PCT = 10;
        int got_re = accum_buf[0], got_im = accum_buf[1];

        // Scale the single-channel expectation by the channel count.
        //
        // Valid because this harness reuses ONE spectrum and ONE filter for every
        // channel (see the channel-loop note), so the accumulator is exactly N x the
        // single-channel result. That is not an assumption — it was measured:
        // accum[0] ran 95, 190, 285, 380 over 4 channels, and max|.| ran
        // 143, 286, 429, 572. Perfectly linear, no drift.
        //
        // Without this, every multi-channel run reports a spurious accum0 FAIL and
        // the check becomes noise that gets ignored — which is exactly how real
        // failures slip past.
        int exp_re = exp.accum0_re * (int)N_CHANNELS;
        int exp_im = exp.accum0_im * (int)N_CHANNELS;

        int tol_re = exp_re * ACCUM0_TOL_PCT / 100;
        int tol_im = exp_im * ACCUM0_TOL_PCT / 100;
        if (tol_re < 0) tol_re = -tol_re;
        if (tol_im < 0) tol_im = -tol_im;
        if (tol_re < 2) tol_re = 2;
        if (tol_im < 2) tol_im = 2;

        int d_re = got_re - exp_re; if (d_re < 0) d_re = -d_re;
        int d_im = got_im - exp_im; if (d_im < 0) d_im = -d_im;

        bool acc0_ok = (d_re <= tol_re) && (d_im <= tol_im);
        printf("[aiesim] accum_out[0]: got={%d,%d}  expected={%d,%d} +/-{%d,%d}"
               "  (%d ch x {%d,%d})  %s\n",
               got_re, got_im, exp_re, exp_im, tol_re, tol_im,
               (int)N_CHANNELS, exp.accum0_re, exp.accum0_im,
               acc0_ok ? "OK" : "FAIL"); fflush(stdout);
        acc0_pass = acc0_ok;
        if (!acc0_ok) {
            // Do NOT _exit() here: the response/peak checks below are the real
            // end-to-end validation, and bailing out early means they never run.
            // The result is still folded into OVERALL, so this cannot hide a failure.
            printf("[aiesim] accum_out[0] outside tolerance — continuing to the "
                   "peak check anyway\n"); fflush(stdout);
        }
    }

    // ----------------------------------------------------------------
    // Verification
    //
    // Peak finding: use L1 magnitude (|re|+|im|) so purely imaginary
    // responses (e.g. S3 with imaginary filter) are located correctly.
    // ----------------------------------------------------------------

    // Find dominant element by L1 magnitude
    int dom_mag = 0, dom_idx = 0;
    for (int i = 0; i < PATCH_ELEMS; ++i) {
        int re = resp_buf[i*2]   > 0 ? resp_buf[i*2]   : -resp_buf[i*2];
        int im = resp_buf[i*2+1] > 0 ? resp_buf[i*2+1] : -resp_buf[i*2+1];
        int mag = re + im;
        if (mag > dom_mag) { dom_mag = mag; dom_idx = i; }
    }

    // Max L1 magnitude of all non-dominant elements
    int max_noise = 0;
    for (int i = 0; i < PATCH_ELEMS; ++i) {
        if (i == dom_idx) continue;
        int re = resp_buf[i*2]   > 0 ? resp_buf[i*2]   : -resp_buf[i*2];
        int im = resp_buf[i*2+1] > 0 ? resp_buf[i*2+1] : -resp_buf[i*2+1];
        int v  = re + im;
        if (v > max_noise) max_noise = v;
    }

    int resp_peak_re = resp_buf[dom_idx * 2];
    int resp_peak_im = resp_buf[dom_idx * 2 + 1];

    // Load per-scenario checks
    // Location within peak_tol pixels (Chebyshev distance in the 2-D map, NOT a
    // flat-index difference — a 1-row error is 64 in flat index but 1 pixel).
    int dom_r = dom_idx / PATCH_COLS,       dom_c = dom_idx % PATCH_COLS;
    int exp_r = exp.peak_idx / PATCH_COLS,  exp_c = exp.peak_idx % PATCH_COLS;
    int dr = dom_r - exp_r; if (dr < 0) dr = -dr;
    int dc = dom_c - exp_c; if (dc < 0) dc = -dc;
    int loc_err  = (dr > dc) ? dr : dc;
    bool loc_ok  = (loc_err <= exp.peak_tol);
    // Response magnitude scales with the channel count for the same reason accum0
    // does — identical channels accumulate linearly (measured: peak 98 at 1 ch,
    // 394 at 4 ch). Scale the bounds so a multi-channel run is a real test rather
    // than a guaranteed FAIL. Clamped to what cint16 can represent.
    auto sc = [](int v) {
        long s = (long)v * (long)N_CHANNELS;
        if (s >  32767) return  32767;
        if (s < -32768) return -32768;
        return (int)s;
    };
    int re_lo = sc(exp.peak_re_lo), re_hi = sc(exp.peak_re_hi);
    int im_lo = sc(exp.peak_im_lo), im_hi = sc(exp.peak_im_hi);
    int noise_lim = sc(exp.max_noise);

    bool norm_ok = (resp_peak_re >= re_lo) && (resp_peak_re <= re_hi);
    bool imag_ok = (resp_peak_im >= im_lo) && (resp_peak_im <= im_hi);
    bool snr_ok  = exp.skip_snr || (max_noise <= noise_lim);

    // ----------------------------------------------------------------
    // Peak-to-sidelobe ratio (Bolme §3.5), when the scenario asks for it.
    //
    // The sidelobe region EXCLUDES an 11x11 window around the peak. Without that
    // exclusion the "largest non-peak element" of a smooth response is simply its
    // neighbour — for a sigma=2 Gaussian that is exp(-1/8) = 0.88 of the peak, so
    // the ratio is 1.13 and the assertion passes on any blurry blob. Excluding the
    // mainlobe is the whole point of PSR.
    //
    // Relative, so it does not need rescaling for N_CHANNELS or for the shift
    // budget — both scale peak and sidelobe together.
    int  psr_sidelobe = 0;
    bool psr_ok       = true;
    if (exp.snr_ratio_pct > 0) {
        const int PSR_EXCL = 5;   // half-width of the excluded window
        for (int i = 0; i < PATCH_ELEMS; ++i) {
            int r = i / PATCH_COLS, c = i % PATCH_COLS;
            int ddr = r - dom_r; if (ddr < 0) ddr = -ddr;
            int ddc = c - dom_c; if (ddc < 0) ddc = -ddc;
            // Circular distance: the response map wraps, so a peak near an edge
            // has its mainlobe split across the boundary.
            if (ddr > PATCH_ROWS / 2) ddr = PATCH_ROWS - ddr;
            if (ddc > PATCH_COLS / 2) ddc = PATCH_COLS - ddc;
            if (ddr <= PSR_EXCL && ddc <= PSR_EXCL) continue;
            int re = resp_buf[i*2]   > 0 ? resp_buf[i*2]   : -resp_buf[i*2];
            int im = resp_buf[i*2+1] > 0 ? resp_buf[i*2+1] : -resp_buf[i*2+1];
            int v  = re + im;
            if (v > psr_sidelobe) psr_sidelobe = v;
        }
        psr_ok = ((long)dom_mag * 100L
                  >= (long)exp.snr_ratio_pct * (long)(psr_sidelobe ? psr_sidelobe : 1));
    }

    // ----------------------------------------------------------------
    // gmio_fft_col_out tap: normalized correlation against the golden.
    //
    // NOT an equality check. cint16 rounding at every FFT stage and DSPLib's
    // additive DC loss make the tap differ from an ideal float FFT by a few LSB
    // per bin, and the bypass path feeds an unscaled spectrum so even the overall
    // gain differs. Correlation is invariant to both, while still going to ~0 the
    // moment the tap is empty, shuffled, or transposed — which is what the tap
    // needs to prove.
    bool   fcol_ok   = true;
    double fcol_corr = 0.0;
    if (exp.fcol_corr_pct > 0) {
        char fcol_path[512];
        snprintf(fcol_path, sizeof(fcol_path), "%s/fft_col_out.bin", scenario_dir);
        int16_t *golden = (int16_t*) malloc((size_t)PATCH_BYTES);
        if (golden && load_cint16_bin(fcol_path, golden, PATCH_ELEMS)) {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int i = 0; i < PATCH_ELEMS * 2; ++i) {
                const double a = fcol_out_buf[i], b = golden[i];
                dot += a * b; na += a * a; nb += b * b;
            }
            fcol_corr = (na > 0.0 && nb > 0.0) ? (dot / (sqrt(na) * sqrt(nb))) : 0.0;
            fcol_ok = (fcol_corr * 100.0 >= (double)exp.fcol_corr_pct);
        } else {
            printf("[aiesim] WARNING: cannot load %s — fft_col_out tap NOT checked\n",
                   fcol_path);
            // Absent golden must not silently pass a check the scenario asked for.
            fcol_ok = false;
        }
        free(golden);
    }

    // ------------------------------------------------------------------
    // HERMITIAN SYMMETRY OF F_ch — the premise behind halving the host filter.
    //
    // conv2d emits cint16 with imag = 0, so the 2-D spectrum of a REAL input
    // must satisfy F(u,v) = conj(F(-u,-v)) and A/B in the host filter are
    // half-redundant. In EXACT arithmetic that is a theorem; in cint16 it is a
    // question, because DSPLib's DIT butterflies do not compute a conjugate pair
    // by symmetric operations and every stage rounds. Measured here rather than
    // assumed — CLAUDE.md records two occasions where an unverified premise moved
    // a calibrated constant.
    //
    // Layout: the tap is col-FFT order, element [v*PATCH_ROWS + u] is bin (u,v).
    // conj => the real parts match and the imaginary parts are opposite.
    {
        double max_res = 0.0, max_mag = 0.0, sum_res = 0.0;
        int    n_bad = 0, n_pairs = 0;
        int    worst_u = 0, worst_v = 0;
        for (int u = 0; u < PATCH_ROWS; ++u)
            for (int v = 0; v < PATCH_COLS; ++v) {
                const int uu = (PATCH_ROWS - u) % PATCH_ROWS;
                const int vv = (PATCH_COLS - v) % PATCH_COLS;
                const int i  = v  * PATCH_ROWS + u;
                const int j  = vv * PATCH_ROWS + uu;
                const double re1 = fcol_out_buf[2*i],   im1 = fcol_out_buf[2*i+1];
                const double re2 = fcol_out_buf[2*j],   im2 = fcol_out_buf[2*j+1];
                const double res = fabs(re1 - re2) + fabs(im1 + im2);
                const double mag = sqrt(re1*re1 + im1*im1);
                if (mag > max_mag) max_mag = mag;
                if (res > max_res) { max_res = res; worst_u = u; worst_v = v; }
                if (res > 0.0) ++n_bad;
                sum_res += res;
                ++n_pairs;
            }
        printf("[aiesim] F_ch Hermitian check: max|residual| = %.0f LSB at bin "
               "(%d,%d), mean %.3f, %d/%d bins asymmetric, max|F| = %.0f\n",
               max_res, worst_u, worst_v, sum_res / (double)n_pairs,
               n_bad, n_pairs, max_mag);
        printf("[aiesim]   -> relative asymmetry %.3f%% of max|F|. Halving the "
               "host filter is safe if this is well under the ~0.05%% that one "
               "int16 LSB of H represents.\n",
               max_mag > 0.0 ? 100.0 * max_res / max_mag : 0.0);
        fflush(stdout);
    }

    printf("\n=== scenario result ===\n");
    printf("  Peak:  {%d,%d} at flat index %d (r=%d, c=%d)  expected idx=%d (r=%d, c=%d)"
           "  err=%d px (tol=%d)  %s\n",
           resp_peak_re, resp_peak_im, dom_idx, dom_r, dom_c,
           exp.peak_idx, exp_r, exp_c,
           loc_err, exp.peak_tol, loc_ok ? "OK" : "FAIL");
    printf("  re in [%d,%d]: %s\n", re_lo, re_hi, norm_ok?"OK":"FAIL");
    printf("  im in [%d,%d]: %s\n", im_lo, im_hi, imag_ok?"OK":"FAIL");
    if (!exp.skip_snr)
        printf("  Noise: max |non-peak| = %d (threshold = %d)  %s\n",
               max_noise, noise_lim, snr_ok?"OK":"FAIL");
    else
        printf("  Noise: check skipped (uniform/broad response expected)\n");
    if (exp.snr_ratio_pct > 0)
        printf("  PSR:   peak %d vs sidelobe %d (11x11 excluded) = %.1fx, "
               "required %.2fx  %s\n",
               dom_mag, psr_sidelobe,
               (double)dom_mag / (double)(psr_sidelobe ? psr_sidelobe : 1),
               exp.snr_ratio_pct / 100.0, psr_ok ? "OK" : "FAIL");
    if (exp.fcol_corr_pct > 0)
        printf("  F_ch tap: correlation with golden = %.4f, required %.2f  %s\n",
               fcol_corr, exp.fcol_corr_pct / 100.0, fcol_ok ? "OK" : "FAIL");
    printf("  location=%s  normalization=%s  imag=%s  SNR=%s  PSR=%s  fcol=%s  accum0=%s  "
           "accum_sat=%s  resp_sat=%s  ifftrow_sat=%s\n",
           loc_ok?"OK":"FAIL", norm_ok?"OK":"FAIL",
           imag_ok?"OK":"FAIL", snr_ok?"OK":"FAIL",
           exp.snr_ratio_pct ? (psr_ok?"OK":"FAIL") : "n/a",
           exp.fcol_corr_pct ? (fcol_ok?"OK":"FAIL") : "n/a",
           acc0_pass?"OK":"FAIL",
           sat_pass?"OK":"FAIL", resp_sat_pass?"OK":"FAIL",
           // n/a, not OK, on the memtile path: the row-IFFT output is not
           // host-visible there, so this check did not run. Printing OK for a
           // check that never executed is the same class of lie as printing the
           // stale buffer it used to read — see the NOT OBSERVABLE note above.
           MEMTILE_TRANSPOSE ? "n/a" : (ifft_row_sat_pass?"OK":"FAIL"));

    bool pass = loc_ok && norm_ok && imag_ok && snr_ok && psr_ok && fcol_ok && acc0_pass
                && sat_pass && resp_sat_pass && ifft_row_sat_pass;
    printf("  OVERALL: %s\n\n", pass ? "PASS" : "FAIL");

    if (!norm_ok && resp_peak_re > 100)
        printf("  HINT: col IFFT shift may be wrong (resp_peak_re=%d).\n"
               "        Check FFT_2D_TP_IFFT_COL_SHIFT in ifft_graph.h.\n\n",
               resp_peak_re);

    // Do NOT call end() — with run(-1), end()'s post-disable cleanup competes for
    // cycle credits with --simulation-cycle-timeout; if the timeout fires first,
    // neither can proceed and the simulation deadlocks permanently.
    // _exit() kills our process immediately; conv2d stalls on exhausted PLIO stream,
    // the event loop drains the remaining cycle budget uncontested then exits.
    // Makefile `timeout 1200` is a safety net. GMIO::free must NOT be called.
    fflush(stdout);
    _exit(pass ? 0 : 1);
}

#endif  // __AIESIM__
