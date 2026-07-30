#include "smoke_passthrough.h"

void smoke_passthrough(
    input_stream<int32>  *sin,
    output_buffer<int32> &sout)
{
    int32_t *o = sout.data();

#if SMOKE_SKIP_STREAM
    // BISECT BUILD: produce the expected pattern WITHOUT touching the PLIO stream.
    //
    // The PLIO variant hangs with S00_AXIS throughput = 0.00 MBps, which has two
    // possible causes that look identical from the host:
    //   (a) the PL->AIE stream never delivers, or
    //   (b) the AIE core never runs, so nothing ever consumes the stream.
    // This build removes the stream read entirely. If gmio_smoke_out now arrives,
    // the core runs and the fault is specifically the PLIO stream (a). If it still
    // hangs, the core is not running (b) and the PLIO is a red herring.
    (void)sin;
    for (int i = 0; i < SMOKE_N; i++)
    chess_prepare_for_pipelining
    chess_loop_range(SMOKE_N, SMOKE_N)
    {
        o[i] = i;
    }
#else
    for (int i = 0; i < SMOKE_N; i++)
    chess_prepare_for_pipelining
    chess_loop_range(SMOKE_N, SMOKE_N)
    {
        o[i] = readincr(sin);
    }
#endif
}
