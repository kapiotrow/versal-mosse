/*
 * stream_src.h
 * Minimal PL kernel for the PLIO smoke test.
 *
 * Streams `n` 32-bit words (a simple counter: 0, 1, 2, ...) to an AXIS port
 * wired to the AIE SmokeIn PLIO. No DDR, no m_axi — the only thing under test
 * is "can a PL kernel push data into the AIE over a PLIO stream at all".
 */

#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

void stream_src(
    hls::stream<ap_axiu<32,0,0,0>> &out,   // AXIS → AIE SmokeIn PLIO
    int n                                   // number of 32-bit words to send
);
