/*
 * mosse_graph.cpp
 * Instantiation and aiesim entry point for the MOSSE AIE graph.
 */

#include "mosse_graph.h"

uint8_t fftRows_grInsts  = 0;
uint8_t fftCols_grInsts  = 0;
uint8_t ifftRows_grInsts = 0;
uint8_t ifftCols_grInsts = 0;

MOSSE_graph mosse_graph;

#ifdef __AIESIM__

// For aiesim each FFT2D instance processes PATCH_ROWS rows per iteration,
// and the single IFFT2D instance processes PATCH_ROWS rows per iteration.
// Total graph iterations = ITER_CNT * PATCH_ROWS (one row window at a time).
int main(int argc, char **argv)
{
    mosse_graph.init();
    // TODO: replace with correct iteration count once aiesim data is prepared
    mosse_graph.run(ITER_CNT * PATCH_ROWS);
    mosse_graph.end();
    return 0;
}

#endif
