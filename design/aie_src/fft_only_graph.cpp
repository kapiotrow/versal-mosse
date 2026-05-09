/*
 * fft_only_graph.cpp
 * Graph instantiation and aiesim entry point for the FFT-only smoke test.
 *
 * Test signal:
 *   2 × FFT_ROW_WS rows, each with a unit impulse at sample 0.
 *   Covers both double-buffer BDs (ping + pong) so the kernel runs twice
 *   without stalling on an empty pong BD.
 *
 * Expected output:
 *   FFT(δ[0]) = constant spectrum {1, 0} for all 128 bins.
 *   Tolerance: ±2 LSB for cint16 twiddle quantisation noise.
 *
 * PASS criteria:
 *   Every output cint16 sample satisfies -1 ≤ re ≤ 3 and -2 ≤ im ≤ 2.
 */

#include "fft_only_graph.h"

FFTOnly_graph fft_only_graph;

#ifdef __AIESIM__

#include <cstring>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
    // Fill both double-buffer BDs to avoid a stall on the second invocation.
    constexpr int NBYTES   = 2 * FFT_ROW_WINDOW_BUFF_SIZE; // 2 × 1024 = 2048 bytes
    constexpr int NSAMPLES = NBYTES / FFT_SAMPLE_BYTES;    // 512 cint16

    int16_t *in_buf  = (int16_t*)GMIO::malloc(NBYTES);
    int16_t *out_buf = (int16_t*)GMIO::malloc(NBYTES);

    // Unit impulse at sample 0 of every row; all other samples = {0,0}.
    memset(in_buf, 0, NBYTES);
    for (int row = 0; row < 2 * FFT_ROW_WS; ++row)       // 4 rows total (2 BDs × 2 rows/BD)
        in_buf[row * FFT_ROW_TP_POINT_SIZE * 2] = 1;     // re=1, im=0 at sample 0

    fft_only_graph.init();
    // run(-1) is required for GMIO output to work in cycle-approximate aiesim.
    // With run(N), after N kernel invocations complete the ADF runtime begins
    // simulation teardown.  During teardown the shim MM2S DMA (GMIO output)
    // can no longer advance — no simulation cycles are driven for it — so
    // gmio_fft_out.wait() blocks forever.  With run(-1) the kernel loops
    // forever, the simulation never tears down, DMA cycles keep advancing,
    // and gmio_fft_out.wait() returns correctly.
    // Termination: end(5000) force-stops the kernel; _exit() bypasses the
    // ADF graph destructor which would hang trying to communicate with the
    // now-stopped simulation event loop.
    fft_only_graph.run(-1);

    // Arm output receiver BEFORE sending input (back-pressure deadlock prevention).
    fft_only_graph.gmio_fft_out.aie2gm_nb(out_buf, NBYTES);
    fft_only_graph.gmio_fft_in.gm2aie_nb(in_buf, NBYTES);

    // gmio_fft_out.wait() is the correct sync point for GMIO output: it returns
    // only after the shim MM2S DMA has written all NBYTES into out_buf.
    // graph::wait() only waits for kernel iterations — the MM2S DMA may still
    // be in flight — so it gives wrong data if used as the sole sync point.
    fft_only_graph.gmio_fft_out.wait();

    // ----------------------------------------------------------------
    // Verification
    // ----------------------------------------------------------------
    printf("[fft_only] out[0..3]: {%d,%d} {%d,%d} {%d,%d} {%d,%d}\n",
           out_buf[0], out_buf[1],
           out_buf[2], out_buf[3],
           out_buf[4], out_buf[5],
           out_buf[6], out_buf[7]);

    bool pass = true;
    for (int i = 0; i < NSAMPLES; ++i) {
        int ore = out_buf[i * 2], oim = out_buf[i * 2 + 1];
        if (ore < -1 || ore > 3 || oim < -2 || oim > 2) {
            printf("[fft_only] FAIL at [%d]: {%d,%d} expected ~{1,0}\n", i, ore, oim);
            pass = false;
            if (i > 4) { printf("  ...\n"); break; }
        }
    }

    printf("=== FFT-only test: %s ===\n\n", pass ? "PASS" : "FAIL");

    // Do NOT call end() before _exit(): end() and the global
    // --simulation-cycle-timeout deadlock each other (end() waits for cycle
    // credits that the timeout has already consumed, neither can proceed).
    //
    // _exit() kills our process immediately; the aiesimulator event loop then
    // runs out the remaining --simulation-cycle-timeout budget (~75K cycles,
    // ~75 s at 100K limit) uncontested and exits on its own.  The Makefile
    // wraps the run in `timeout 120` as a safety net in case that path stalls.
    //
    // GMIO::free must NOT be called — after _exit() the event loop is still
    // live; free()'s unmap notification would race with the cycle drain.
    fflush(stdout);
    _exit(pass ? 0 : 1);
}

#endif // __AIESIM__
