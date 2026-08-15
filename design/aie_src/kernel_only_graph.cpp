/*
 * kernel_only_graph.cpp
 * Harness entry point for the single-kernel bit-exactness test.
 *
 * Drives ONE kernel with scenario data and dumps its raw output to
 * kernel_out.bin, which scripts/check_kernel_bitexact.py diffs against the
 * Python model in gen_aiesim_vectors.py. See kernel_only_graph.h for why.
 *
 * Run with:
 *   make x86sim_check KUT=conv2d SCENARIO=s6 CONV2D_MODE=0
 *   make x86sim_check KUT=cmul   SCENARIO=s7
 *
 * The DRIVE ORDER below deliberately mirrors mosse_tracker.cpp rather than being
 * simplified. The interleave of the weights feed with the output drain, and
 * arming a drain before the input it depends on, are both load-bearing there —
 * see the aie2gm_nb entry in CLAUDE.md, where each rule was learned from a
 * deadlock. A harness that drives the kernel differently from production is
 * testing a configuration nobody ships.
 */

#include "kernel_only_graph.h"

KernelOnly_graph kernel_only_graph;

#if defined(__X86SIM__) || defined(__AIESIM__)

#include "aiesim_scenario_io.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static constexpr int PATCH_ELEMS = PATCH_ROWS * PATCH_COLS;
static constexpr int PATCH_BYTES = PATCH_ELEMS * 4;          // cint16

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    const char *scen = getenv("AIESIM_SCENARIO_DIR");
    if (!scen || scen[0] == '\0') scen = "aiesim_data/s6";

    const char *outdir = getenv("KERNEL_OUT_DIR");
    if (!outdir || outdir[0] == '\0') outdir = ".";

    char path[512];

#if KERNEL_UNDER_TEST == KUT_CONV2D
    printf("[kut] kernel=conv2d  scenario=%s  CONV2D_ECHO_TEST=%d\n",
           scen, CONV2D_ECHO_TEST);
    // A bit-exactness run in echo mode compares the passthrough, not the
    // convolution. Say so loudly — this is the same footgun that made the
    // 2026-08-13 ch16 baseline meaningless.
    if (CONV2D_ECHO_TEST != 0)
        printf("[kut] WARNING: conv2d is NOT in real-convolution mode. "
               "Build with CONV2D_MODE=0 to test the MAC path.\n");

    constexpr int CONV_INVOCATIONS = PATCH_ELEMS / CONV_OUT_CHUNK;
    constexpr int CHUNK_BYTES      = CONV_OUT_CHUNK * 4;      // cint16 out

    int8_t  *weights = (int8_t*) GMIO::malloc(CONV_WEIGHT_BYTES_PAD);
    int16_t *out_buf = (int16_t*)GMIO::malloc(PATCH_BYTES);

    // Which channel's weights to drive. NOT always 0: on the s6 patch ReLU never
    // fires for ch0 (nor for 12 of 16 channels — bias_acc is oversized), so a
    // ch0-only harness cannot distinguish CONV_RELU=1 from CONV_RELU=0. ch11 is
    // the one that clamps some but not all pixels.
    const char *chs = getenv("KUT_CH");
    const int ch = (chs && chs[0]) ? atoi(chs) : 0;

    memset(weights, 0, CONV_WEIGHT_BYTES_PAD);
    snprintf(path, sizeof(path), "%s/weights_ch%d.bin", scen, ch);
    if (!load_raw_bin(path, weights, CONV_WEIGHT_BYTES_PAD)) return 1;
    printf("[kut] weights channel %d  (CONV_RELU=%d)\n", ch, CONV_RELU);

    memset(out_buf, 0, PATCH_BYTES);

    kernel_only_graph.init();
    kernel_only_graph.run(-1);

    // One weight buffer per FIRING, interleaved with one output drain per
    // firing. conv2d acquires its `weights` input_buffer before EVERY
    // invocation (CLAUDE.md: "consumed per FIRING, not per patch"), and only
    // ~2 fit in flight, so feeding them all up front deadlocks.
    for (int k = 0; k < CONV_INVOCATIONS; ++k) {
        kernel_only_graph.gmio_kernel_out.aie2gm_nb(
            (char*)out_buf + (size_t)k * CHUNK_BYTES, CHUNK_BYTES);
        kernel_only_graph.gmio_weights.gm2aie_nb(weights, CONV_WEIGHT_BYTES_PAD);
        kernel_only_graph.gmio_weights.wait();
        kernel_only_graph.gmio_kernel_out.wait();
    }
    printf("[kut] conv2d drained %d x %d B\n", CONV_INVOCATIONS, CHUNK_BYTES);

#else  // KUT_CMUL
    printf("[kut] kernel=cmul  scenario=%s  CMUL_H_SHIFT=%d\n",
           scen, CMUL_H_SHIFT);

    constexpr int CMUL_N       = PATCH_COLS * FFT_COL_WS;
    constexpr int COL_CHUNKS   = PATCH_ELEMS / CMUL_N;
    constexpr int CHUNK_BYTES  = CMUL_N * 4;
    constexpr int CMUL_IN_ELEMS = 2 * PATCH_ELEMS;

    int16_t *fcol_in = (int16_t*)GMIO::malloc(PATCH_BYTES);
    int16_t *cmul_in = (int16_t*)GMIO::malloc(CMUL_IN_ELEMS * 4);
    int16_t *out_buf = (int16_t*)GMIO::malloc(PATCH_BYTES);
    int16_t *flt     = (int16_t*)malloc(PATCH_BYTES);
    int16_t *acc     = (int16_t*)malloc(PATCH_BYTES);

    snprintf(path, sizeof(path), "%s/fft_col_in.bin", scen);
    if (!load_cint16_bin(path, fcol_in, PATCH_ELEMS)) return 1;
    snprintf(path, sizeof(path), "%s/cmul_filter.bin", scen);
    if (!load_cint16_bin(path, flt, PATCH_ELEMS)) return 1;
    snprintf(path, sizeof(path), "%s/cmul_accum.bin", scen);
    if (!load_cint16_bin(path, acc, PATCH_ELEMS)) return 1;

    // Pack [filter_chunk | accum_chunk] per invocation — the layout documented
    // in cmul_accum_kernel.h and produced by mosse_tracker.cpp.
    for (int c = 0; c < COL_CHUNKS; ++c) {
        memcpy(cmul_in + (size_t)c * 2 * CMUL_N * 2,
               flt + (size_t)c * CMUL_N * 2, (size_t)CMUL_N * 4);
        memcpy(cmul_in + (size_t)c * 2 * CMUL_N * 2 + CMUL_N * 2,
               acc + (size_t)c * CMUL_N * 2, (size_t)CMUL_N * 4);
    }

    memset(out_buf, 0, PATCH_BYTES);

    kernel_only_graph.init();
    kernel_only_graph.run(-1);

    kernel_only_graph.gmio_fft_col_in.gm2aie_nb(fcol_in, PATCH_BYTES);
    kernel_only_graph.gmio_cmul_in.gm2aie_nb(cmul_in, CMUL_IN_ELEMS * 4);

    // Drain before waiting on the inputs: a full output window stalls the
    // kernel, which stalls the very input DMAs we would be waiting on.
    for (int k = 0; k < COL_CHUNKS; ++k) {
        kernel_only_graph.gmio_kernel_out.aie2gm_nb(
            (char*)out_buf + (size_t)k * CHUNK_BYTES, CHUNK_BYTES);
        kernel_only_graph.gmio_kernel_out.wait();
    }
    kernel_only_graph.gmio_fft_col_in.wait();
    kernel_only_graph.gmio_cmul_in.wait();
    printf("[kut] cmul drained %d x %d B\n", COL_CHUNKS, CHUNK_BYTES);
#endif

    // Cheap sanity line so an all-zero dump is obvious in the log rather than
    // showing up as a mystery diff.
    long nz = 0, maxabs = 0;
    for (int i = 0; i < PATCH_ELEMS * 2; ++i) {
        long v = out_buf[i] < 0 ? -(long)out_buf[i] : out_buf[i];
        if (v) ++nz;
        if (v > maxabs) maxabs = v;
    }
    printf("[kut] output: %ld/%d non-zero int16, max|.|=%ld\n",
           nz, PATCH_ELEMS * 2, maxabs);
    if (nz == 0)
        printf("[kut] WARNING: output is identically zero — the kernel probably "
               "never ran or never received data.\n");

    snprintf(path, sizeof(path), "%s/kernel_out.bin", outdir);
    bool ok = dump_raw_bin(path, out_buf, PATCH_BYTES);

    // Same _exit() rationale as fft_only_graph.cpp: end() and the simulator's
    // cycle timeout deadlock each other, and the ADF destructor hangs talking to
    // a stopped event loop. The Makefile wraps the run in `timeout` as a net.
    fflush(stdout);
    _exit(ok ? 0 : 1);
}

#endif // __X86SIM__ || __AIESIM__
