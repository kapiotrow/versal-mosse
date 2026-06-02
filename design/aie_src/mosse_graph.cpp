/*
 * mosse_graph.cpp
 * Instantiation and aiesim entry point for the MOSSE AIE graph.
 *
 * In the new architecture there is one FFT2D instance and one IFFT2D instance
 * (serial per-channel processing). All inter-stage data lives in DDR and is
 * exchanged via GMIO; the APU (or the aiesim host below) orchestrates transfers.
 *
 * === aiesim round-trip FFT/IFFT test ===
 *
 * Test signal:
 *   patch_in.txt = unit impulse at spatial position (r=0, c=0), int8 value 1.
 *   (All other samples = 0.)
 *
 * Expected output after the full pipeline (FFT → identity filter + accum → IFFT):
 *   2D FFT of δ[r=0,c=0]  = constant spectrum {1,0} for all (k1,k2).
 *   cmul_accum: H*={1,0}, accum_prev={1,0} (simulates ch≥1, not ch=0).
 *     element-wise: re = F_re*H_re + F_im*H_im = 1; im = 0
 *     output = accum_prev + product = {1,0} + {1,0} = {2,0} everywhere.
 *   2D IFFT (row shift=0, col shift=12) of all-{2,0} spectrum:
 *     row IFFT → each output row has ≈{86,0} at n=0, {0,0} elsewhere (2×43)
 *     after transpose → row 0 = all {86,0}, rows 1..127 = zero
 *     col IFFT raw acc ≈ 10240 (2×5120); 10240>>12 = 2
 *   Final: resp[0,0] ≈ {2,0}, all others = {0,0}.
 *
 * This tests the accumulation code path: if accum_prev reads are broken the
 * kernel falls back to {1,0} (product only) → resp0_re=1 → norm_ok FAIL.
 *
 * Tolerance: ±2 LSB for elements expected to be 0 (cint16 twiddle quantization).
 *
 * PASS criteria (see verification block at end of main()):
 *   1. Dominant peak is at index 0 (correct location after round-trip).
 *   2. Peak real part is in [2, 6] — not 0/1 (accumulation broken/deadlock) and
 *      not large (e.g. 16384 would indicate the col IFFT shift is 0, not 12).
 *   3. Peak imag part in [-2, 2] (real input → real output).
 *   4. All non-peak elements have magnitude ≤ 2.
 */

#include "mosse_graph.h"

MOSSE_graph mosse_graph;

#ifdef __AIESIM__

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

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

    int8_t  *weights_buf = (int8_t*)  GMIO::malloc(64);
    int16_t *filter_buf  = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *cmul_in_buf = (int16_t*) GMIO::malloc(CMUL_IN_BYTES);
    int16_t *fft_scratch = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *accum_buf   = (int16_t*) GMIO::malloc(PATCH_BYTES);
    int16_t *resp_buf    = (int16_t*) GMIO::malloc(PATCH_BYTES);

    memset(weights_buf, 0, 64);

    // Identity filter: H[i] = {re=1, im=0} — round-trip test expects IFFT(FFT(x)) = x
    for (int i = 0; i < PATCH_ELEMS; ++i) {
        filter_buf[i * 2]     = 1;
        filter_buf[i * 2 + 1] = 0;
    }

    // Build combined cmul input: interleave filter and accum_prev by chunk.
    // accum_prev = {1,0} everywhere — simulates ch≥1 (adding to existing accumulator).
    // Expected kernel output: F*H* + accum_prev = {1,0} + {1,0} = {2,0} everywhere.
    // If accumulation is broken (accum reads ignored) the output stays {1,0} → FAIL.
    memset(cmul_in_buf, 0, CMUL_IN_BYTES);
    for (int c = 0; c < N_CHUNKS; ++c) {
        // Filter half
        memcpy((int8_t*)cmul_in_buf + (size_t)c * 2 * CHUNK_BYTES,
               (int8_t*)filter_buf  + (size_t)c * CHUNK_BYTES,
               CHUNK_BYTES);
        // Accum-prev half: {1,0} everywhere
        int16_t* acc_ptr = (int16_t*)((int8_t*)cmul_in_buf
                           + (size_t)c * 2 * CHUNK_BYTES + CHUNK_BYTES);
        for (int i = 0; i < CHUNK_ELEMS; ++i) {
            acc_ptr[i * 2]     = 1;
            acc_ptr[i * 2 + 1] = 0;
        }
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
    mosse_graph.gmio_weights.gm2aie_nb(weights_buf, 64);
    mosse_graph.gmio_weights.wait();   // completes while PLIO is idle (cores not yet enabled)

    mosse_graph.run(-1);   // enables cores + PLIO; conv2d starts immediately (weights ready)

    // PatchIn PLIO reads from aiesim_data/patch_in.txt (impulse at (0,0),
    // value=1) followed by padding zeros — see gen_aiesim_vectors.py.
    // The padding keeps the PLIO model "active" (not stalled) for the full
    // ~50 000-cycle test so cycle-credit starvation cannot occur.
    // conv2d casts int8 → cint16, feeds fft2d.fft_rows.

    // Step 2: collect row-FFT output (PATCH_ROWS rows of PATCH_COLS-pt FFT)
    mosse_graph.gmio_fft_row_out.aie2gm_nb(fft_scratch, PATCH_BYTES);  // arm AFTER run()
    printf("[aiesim] step 2: waiting for fft_row_out...\n"); fflush(stdout);
    mosse_graph.gmio_fft_row_out.wait();
    printf("[aiesim] step 2: fft_row_out done\n"); fflush(stdout);

    printf("[aiesim] fft_row_out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           fft_scratch[0], fft_scratch[1],
           fft_scratch[2], fft_scratch[3],
           fft_scratch[4], fft_scratch[5],
           fft_scratch[6], fft_scratch[7]);

    // Step 3: APU transpose — convert row-FFT output from row-major to column-major
    //         so that each "row" fed to the col-FFT is one column of the 2D spectrum.
    transpose_inplace(fft_scratch, PATCH_ROWS, PATCH_COLS);

    // Step 4: fft_cols + cmul → accum_out
    // Arm output GMIO first, then fire all input GMIOs.
    // Do NOT call wait() on input GMIOs (gm2aie_nb) — only output GMIOs need wait().
    // Calling input wait() blocks until the kernel consumes the buffer, which causes
    // a deadlock if the kernel cannot run while we're holding the thread here.
    mosse_graph.gmio_accum_out.aie2gm_nb(accum_buf, PATCH_BYTES);      // arm output first
    mosse_graph.gmio_cmul_in.gm2aie_nb(cmul_in_buf, CMUL_IN_BYTES);   // [filter | zero_accum]
    mosse_graph.gmio_fft_col_in.gm2aie_nb(fft_scratch, PATCH_BYTES);

    // Step 5: collect cmul_accum output (= col-FFT pass-through in stub)
    printf("[aiesim] step 4/5: waiting for accum_out...\n"); fflush(stdout);
    mosse_graph.gmio_accum_out.wait();
    printf("[aiesim] step 4/5: accum_out done\n"); fflush(stdout);

    printf("[aiesim] accum_out[0..3]:   {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           accum_buf[0], accum_buf[1],
           accum_buf[2], accum_buf[3],
           accum_buf[4], accum_buf[5],
           accum_buf[6], accum_buf[7]);

    // ----------------------------------------------------------------
    // IFFT pass
    // ----------------------------------------------------------------

    // Step 6: row IFFT (128-pt, shift=0)
    // Arm output first, then fire input. No wait() on input GMIO.
    mosse_graph.gmio_ifft_row_out.aie2gm_nb(fft_scratch, PATCH_BYTES);
    mosse_graph.gmio_ifft_row_in.gm2aie_nb(accum_buf, PATCH_BYTES);
    printf("[aiesim] step 6: waiting for ifft_row_out...\n"); fflush(stdout);
    mosse_graph.gmio_ifft_row_out.wait();
    printf("[aiesim] step 6: ifft_row done\n"); fflush(stdout);

    printf("[aiesim] ifft_row_out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           fft_scratch[0], fft_scratch[1],
           fft_scratch[2], fft_scratch[3],
           fft_scratch[4], fft_scratch[5],
           fft_scratch[6], fft_scratch[7]);

    // Step 7: APU transpose for IFFT
    transpose_inplace(fft_scratch, PATCH_ROWS, PATCH_COLS);

    // Step 8: col IFFT (128-pt, shift=14) + collect response
    // Arm output first, then fire input. No wait() on input GMIO.
    mosse_graph.gmio_response.aie2gm_nb(resp_buf, PATCH_BYTES);
    mosse_graph.gmio_ifft_col_in.gm2aie_nb(fft_scratch, PATCH_BYTES);
    printf("[aiesim] step 8: waiting for response...\n"); fflush(stdout);
    mosse_graph.gmio_response.wait();
    printf("[aiesim] step 8: response done\n"); fflush(stdout);

    printf("[aiesim] response[0..3]:     {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           resp_buf[0], resp_buf[1],
           resp_buf[2], resp_buf[3],
           resp_buf[4], resp_buf[5],
           resp_buf[6], resp_buf[7]);

    // ----------------------------------------------------------------
    // Verification
    //
    // Expected (impulse at (0,0) → FFT → identity filter → IFFT with shift=14):
    //   resp_buf[0] = {1,0}     (the recovered impulse)
    //   resp_buf[i] = {0,0}     for i > 0 (all other positions)
    //
    // Tolerance: ±2 LSB for cint16 twiddle quantisation noise.
    // ----------------------------------------------------------------

    // Find dominant real-part peak
    int dom_re = 0, dom_idx = 0;
    for (int i = 0; i < PATCH_ELEMS; ++i) {
        int v = resp_buf[i * 2] > 0 ? resp_buf[i * 2] : -resp_buf[i * 2];
        if (v > dom_re) { dom_re = v; dom_idx = i; }
    }

    // Max magnitude of all non-(0,0) elements
    int max_noise = 0;
    for (int i = 1; i < PATCH_ELEMS; ++i) {
        int re = resp_buf[i*2]   > 0 ? resp_buf[i*2]   : -resp_buf[i*2];
        int im = resp_buf[i*2+1] > 0 ? resp_buf[i*2+1] : -resp_buf[i*2+1];
        int v  = re > im ? re : im;
        if (v > max_noise) max_noise = v;
    }

    int resp0_re = resp_buf[0];
    int resp0_im = resp_buf[1];

    // Checks (see header comment for rationale):
    bool loc_ok  = (dom_idx == 0);          // peak at spatial origin
    bool norm_ok = (resp0_re >= 2) && (resp0_re <= 6);   // 2× accum: {1,0}+{1,0}={2,0}; not 1 (no accum) or 0 (deadlock)
    bool imag_ok = (resp0_im >= -2) && (resp0_im <= 2);  // near-zero imag
    bool snr_ok  = (max_noise <= 2);         // quantisation noise floor

    printf("\n=== FFT/IFFT round-trip + accumulation test ===\n");
    printf("  Input:    unit impulse at (r=0, c=0), int8 = 1\n");
    printf("  Filter:   H*={1,0}, accum_prev={1,0} (ch>=1 simulation)\n");
    printf("  Expected: resp[0,0]~={2,0}, all others={0,0}\n");
    printf("  Peak:     {%d,%d} at flat index %d (r=%d, c=%d)\n",
           resp0_re, resp0_im, dom_idx,
           dom_idx / PATCH_COLS, dom_idx % PATCH_COLS);
    printf("  Dominant: {%d,%d} at index %d%s\n",
           dom_re, resp_buf[dom_idx*2+1], dom_idx,
           dom_idx == 0 ? "" : "  <-- WRONG LOCATION");
    printf("  Noise:    max |non-peak| = %d (threshold = 2)\n", max_noise);
    printf("  location=%s  normalization=%s  imag=%s  SNR=%s\n",
           loc_ok?"OK":"FAIL", norm_ok?"OK":"FAIL",
           imag_ok?"OK":"FAIL", snr_ok?"OK":"FAIL");

    bool pass = loc_ok && norm_ok && imag_ok && snr_ok;
    printf("  OVERALL: %s\n\n", pass ? "PASS" : "FAIL");

    if (!norm_ok && resp0_re > 100)
        printf("  HINT: col IFFT shift may be wrong (resp0_re=%d; expected ~2).\n"
               "        Check FFT_2D_TP_IFFT_COL_SHIFT in ifft_graph.h.\n\n",
               resp0_re);
    if (!norm_ok && resp0_re == 1)
        printf("  HINT: accumulation may be broken — accum_prev={1,0} was supplied\n"
               "        but resp0_re=1 suggests only the product was used (no add).\n\n");

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
