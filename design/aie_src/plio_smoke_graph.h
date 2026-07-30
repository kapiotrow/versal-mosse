/*
 * plio_smoke_graph.h
 * Smallest possible PLIO test graph.
 *
 *   stream_src (PL) --AXIS--> SmokeIn (PLIO, 32-bit) --> smoke_passthrough
 *                                                            |
 *                                                     gmio_smoke_out --> DDR
 *
 * Purpose: answer ONE question — can a PLIO deliver data from PL into an AIE
 * kernel at all, in this project's toolchain + platform setup?
 *
 * The full MOSSE design hangs because conv2d never receives a single word from
 * its PatchIn PLIO (it blocks forever on readincr), even though the shim
 * placement, port allocation, and widths all check out. Everything else has
 * been eliminated. This graph strips away conv2d's weights buffer, the FFT, the
 * stream->window adapter, roi_crop, and DDR, leaving only the PLIO.
 *
 *   - PLIO delivers  -> the mechanism works; diff this graph against MOSSE_graph.
 *   - PLIO hangs     -> PLIO is unusable here; feed the AIE via GMIO instead.
 *
 * NOTE: SmokeIn is intentionally left UNCONSTRAINED (no .aiecst shim pin), so
 * the compiler places it. The main design pins PatchIn to column 15/channel 0;
 * leaving it free here also tests whether that hand-pin was the problem.
 */

#pragma once

#include "adf.h"
#include "smoke_passthrough.h"

using namespace adf;

class PLIO_smoke_graph : public graph
{
public:
    input_plio  smoke_in;        // "SmokeIn"  <- stream_src PL kernel
    output_gmio gmio_smoke_out;  // -> DDR (host verifies 0,1,2,...)
    kernel      pass;

    PLIO_smoke_graph()
    {
        smoke_in       = input_plio::create("SmokeIn", plio_32_bits, "smoke_in.txt");
        gmio_smoke_out = output_gmio::create("gmio_smoke_out", 64, 1000);

        pass = kernel::create(smoke_passthrough);
        source(pass) = "smoke_passthrough.cpp";
        runtime<ratio>(pass) = 0.9;

        // PLIO stream -> kernel stream input (the construct under test)
        adf::connect<stream>(smoke_in.out[0], pass.in[0]);

        // kernel window output -> GMIO (proven-working pattern)
        adf::connect<>(pass.out[0], gmio_smoke_out.in[0]);
        adf::dimensions(pass.out[0]) = {SMOKE_N};
    }
};
