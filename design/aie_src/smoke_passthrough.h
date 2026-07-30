/*
 * smoke_passthrough.h
 * Minimal AIE kernel for the PLIO smoke test.
 *
 * Reads SMOKE_N int32 words from the SmokeIn PLIO stream and copies them into
 * an output buffer (window) that is wired straight to a GMIO.
 *
 * Deliberate design choices, so that a hang can ONLY mean "the PLIO stream did
 * not deliver":
 *   - input  = input_stream<int32>  : the exact construct under test
 *   - output = output_buffer<int32> : GMIO-from-a-window is already proven to
 *                                     work in the main design (all its GMIO
 *                                     outputs are fed from windows)
 *   - no weights buffer, no compute, no FFT, no adapter.
 */

#pragma once

#include <adf.h>
using namespace adf;

#ifndef SMOKE_N
#  define SMOKE_N  256   // 32-bit words to move (1 KB) — tiny, fast in cosim
#endif

void smoke_passthrough(
    input_stream<int32>  *sin,   // ← SmokeIn PLIO (32-bit)
    output_buffer<int32> &sout   // → gmio_smoke_out (SMOKE_N words)
);
