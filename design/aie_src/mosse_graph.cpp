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
#include <algorithm>

// ---------------------------------------------------------------------------
// Data-driven scenario support
// ---------------------------------------------------------------------------

// Load PATCH_ELEMS cint16 from a flat int16 LE binary file into buf (int16_t*).
static bool load_cint16_bin(const char *path, int16_t *buf, int n_elems)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[aiesim] ERROR: cannot open %s\n", path);
        return false;
    }
    size_t got = fread(buf, sizeof(int16_t), (size_t)n_elems * 2, f);
    fclose(f);
    if ((int)got != n_elems * 2) {
        fprintf(stderr, "[aiesim] ERROR: %s: read %zu int16, expected %d\n",
                path, got, n_elems * 2);
        return false;
    }
    return true;
}

struct ScenarioExpected {
    int peak_idx;
    int peak_re_lo, peak_re_hi;
    int peak_im_lo, peak_im_hi;
    int max_noise;
    int skip_snr;
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
    e->skip_snr   = 0;
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
        if (!strcmp(key, "skip_snr"))     e->skip_snr      = val;
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
    printf("[aiesim] scenario: %s\n", scenario_dir); fflush(stdout);

    // Make every log self-describing about WHICH conv2d it was built with.
    // Twice now, a run was interpreted against the wrong build: MODE=2 emits a
    // synthetic ramp and ignores patch_in entirely, so scenario expectations
    // cannot possibly match and the "failure" is meaningless. Printing the mode
    // (and what it implies) makes that obvious from the log alone.
#ifndef CONV2D_ECHO_TEST
#  define CONV2D_ECHO_TEST (-1)
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

    // Build combined cmul input: interleave filter and accum_prev by chunk.
    memset(cmul_in_buf, 0, CMUL_IN_BYTES);
    for (int c = 0; c < N_CHUNKS; ++c) {
        memcpy((int8_t*)cmul_in_buf + (size_t)c * 2 * CHUNK_BYTES,
               (int8_t*)filter_buf  + (size_t)c * CHUNK_BYTES,
               CHUNK_BYTES);
        memcpy((int8_t*)cmul_in_buf + (size_t)c * 2 * CHUNK_BYTES + CHUNK_BYTES,
               (int8_t*)accum_prev_buf + (size_t)c * CHUNK_BYTES,
               CHUNK_BYTES);
    }

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
    printf("[aiesim] step 4/5: waiting for accum_out (%d × %d B)...\n",
           N_CMUL_INV, CMUL_INV_BYTES); fflush(stdout);
    for (int inv = 0; inv < N_CMUL_INV; ++inv) {
        mosse_graph.gmio_accum_out.aie2gm_nb(
            (int8_t*)accum_buf    + inv * CMUL_INV_BYTES,   CMUL_INV_BYTES);
        mosse_graph.gmio_cmul_in.gm2aie_nb(
            (int8_t*)cmul_in_buf  + inv * CMUL_IN_INV_BYTES, CMUL_IN_INV_BYTES);
        mosse_graph.gmio_fft_col_in.gm2aie_nb(
            (int8_t*)fft_scratch  + inv * CMUL_INV_BYTES,   CMUL_INV_BYTES);
        mosse_graph.gmio_accum_out.wait();
    }
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
        mosse_graph.gmio_ifft_row_out.aie2gm_nb(
            (int8_t*)fft_scratch + inv * IFFT_INV_BYTES, IFFT_INV_BYTES);
        mosse_graph.gmio_ifft_row_in.gm2aie_nb(
            (int8_t*)accum_buf   + inv * IFFT_INV_BYTES, IFFT_INV_BYTES);
        mosse_graph.gmio_ifft_row_out.wait();
    }
    printf("[aiesim] step 6: ifft_row done\n"); fflush(stdout);

    printf("[aiesim] ifft_row_out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           fft_scratch[0], fft_scratch[1],
           fft_scratch[2], fft_scratch[3],
           fft_scratch[4], fft_scratch[5],
           fft_scratch[6], fft_scratch[7]);

    // Step 7: APU transpose for IFFT
    transpose_inplace(fft_scratch, PATCH_ROWS, PATCH_COLS);

    // Step 8: col IFFT — same per-invocation loop fix.
    constexpr int RESP_INV_BYTES = PATCH_COLS * FFT_COL_WS * 4;   // 1024 B
    constexpr int N_RESP_INV     = PATCH_ROWS / FFT_COL_WS;        // 64
    printf("[aiesim] step 8: waiting for response (%d × %d B)...\n",
           N_RESP_INV, RESP_INV_BYTES); fflush(stdout);
    for (int inv = 0; inv < N_RESP_INV; ++inv) {
        mosse_graph.gmio_response.aie2gm_nb(
            (int8_t*)resp_buf   + inv * RESP_INV_BYTES, RESP_INV_BYTES);
        mosse_graph.gmio_ifft_col_in.gm2aie_nb(
            (int8_t*)fft_scratch + inv * RESP_INV_BYTES, RESP_INV_BYTES);
        mosse_graph.gmio_response.wait();
    }
    printf("[aiesim] step 8: response done\n"); fflush(stdout);

    printf("[aiesim] response[0..3]:     {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           resp_buf[0], resp_buf[1],
           resp_buf[2], resp_buf[3],
           resp_buf[4], resp_buf[5],
           resp_buf[6], resp_buf[7]);

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

        int tol_re = exp.accum0_re * ACCUM0_TOL_PCT / 100;
        int tol_im = exp.accum0_im * ACCUM0_TOL_PCT / 100;
        if (tol_re < 0) tol_re = -tol_re;
        if (tol_im < 0) tol_im = -tol_im;
        if (tol_re < 2) tol_re = 2;
        if (tol_im < 2) tol_im = 2;

        int d_re = got_re - exp.accum0_re; if (d_re < 0) d_re = -d_re;
        int d_im = got_im - exp.accum0_im; if (d_im < 0) d_im = -d_im;

        bool acc0_ok = (d_re <= tol_re) && (d_im <= tol_im);
        printf("[aiesim] accum_out[0]: got={%d,%d}  expected={%d,%d} +/-{%d,%d}  %s\n",
               got_re, got_im, exp.accum0_re, exp.accum0_im, tol_re, tol_im,
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
    bool loc_ok  = (dom_idx == exp.peak_idx);
    bool norm_ok = (resp_peak_re >= exp.peak_re_lo) && (resp_peak_re <= exp.peak_re_hi);
    bool imag_ok = (resp_peak_im >= exp.peak_im_lo) && (resp_peak_im <= exp.peak_im_hi);
    bool snr_ok  = exp.skip_snr || (max_noise <= exp.max_noise);

    printf("\n=== scenario result ===\n");
    printf("  Peak:  {%d,%d} at flat index %d (r=%d, c=%d)  expected idx=%d  %s\n",
           resp_peak_re, resp_peak_im, dom_idx,
           dom_idx / PATCH_COLS, dom_idx % PATCH_COLS,
           exp.peak_idx, loc_ok ? "OK" : "FAIL");
    printf("  re in [%d,%d]: %s\n", exp.peak_re_lo, exp.peak_re_hi, norm_ok?"OK":"FAIL");
    printf("  im in [%d,%d]: %s\n", exp.peak_im_lo, exp.peak_im_hi, imag_ok?"OK":"FAIL");
    if (!exp.skip_snr)
        printf("  Noise: max |non-peak| = %d (threshold = %d)  %s\n",
               max_noise, exp.max_noise, snr_ok?"OK":"FAIL");
    else
        printf("  Noise: check skipped (uniform/broad response expected)\n");
    printf("  location=%s  normalization=%s  imag=%s  SNR=%s  accum0=%s  saturation=%s\n",
           loc_ok?"OK":"FAIL", norm_ok?"OK":"FAIL",
           imag_ok?"OK":"FAIL", snr_ok?"OK":"FAIL", acc0_pass?"OK":"FAIL",
           sat_pass?"OK":"FAIL");

    bool pass = loc_ok && norm_ok && imag_ok && snr_ok && acc0_pass && sat_pass;
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
