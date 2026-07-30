/*
 * plio_smoke_host.cpp
 * Host app for the minimal PLIO smoke test.
 *
 * Flow:
 *   1. load xclbin, start the AIE graph
 *   2. arm the GMIO output DMA (AIE -> DDR, SMOKE_N int32)
 *   3. run the stream_src PL kernel -> it pushes 0,1,2,... into the SmokeIn PLIO
 *   4. wait for the GMIO -> if it completes, the PLIO delivered
 *   5. verify out[i] == i
 *
 * Every stage prints + flushes, so wherever it stops is exactly what failed.
 * The interesting line is "gmio_smoke_out received": if it never appears, the
 * PLIO did not deliver a single word and PLIO is unusable in this setup.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_graph.h"
#include "xrt/xrt_aie.h"

#ifndef SMOKE_N
#  define SMOKE_N  256
#endif

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr size_t OUT_BYTES = (size_t)SMOKE_N * sizeof(int32_t);

    printf("[smoke] loading xclbin\n"); fflush(stdout);
    xrt::device device(0);
    xrt::uuid   uuid = device.load_xclbin(argv[1]);

    printf("[smoke] starting graph\n"); fflush(stdout);
    xrt::graph gr(device, uuid, "smoke_graph");
    gr.run();

    xrt::aie::buffer gm_out(device, uuid, "gmio_smoke_out");
    auto out_bo = xrt::bo(device, OUT_BYTES, xrt::bo::flags::normal, 0);
    auto src    = xrt::kernel(device, uuid, "stream_src:{stream_src_0}");

    // Arm the AIE->DDR DMA before the PL starts pushing.
    printf("[smoke] arming gmio_smoke_out (%zu bytes)\n", OUT_BYTES); fflush(stdout);
    gm_out.async(out_bo, XCL_BO_SYNC_BO_AIE_TO_GMIO, OUT_BYTES, 0);

    printf("[smoke] launching stream_src (%d words)\n", SMOKE_N); fflush(stdout);
    auto run = src(SMOKE_N);
    run.wait();
    printf("[smoke] stream_src done\n"); fflush(stdout);

    // THE test: does the AIE kernel ever receive the PLIO data and emit it?
    gm_out.wait();
    printf("[smoke] gmio_smoke_out received  <-- PLIO DELIVERED\n"); fflush(stdout);

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const int32_t *p = out_bo.map<int32_t *>();

    int errors = 0;
    for (int i = 0; i < SMOKE_N; ++i) {
        if (p[i] != i) {
            if (errors < 8)
                printf("[smoke] MISMATCH out[%d] = %d (expected %d)\n", i, p[i], i);
            ++errors;
        }
    }

    if (errors == 0) {
        printf("[smoke] PASS — all %d words correct\n", SMOKE_N);
    } else {
        printf("[smoke] FAIL — %d/%d words wrong (data arrived but corrupted)\n",
               errors, SMOKE_N);
    }
    fflush(stdout);

    gr.end(0);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
