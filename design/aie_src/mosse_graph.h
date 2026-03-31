/*
 * mosse_graph.h
 * Top-level AIE graph for the MOSSE correlation filter tracker.
 *
 * Contains:
 *   - N_CHANNELS instances of FFT2D_graph  (one per conv feature channel)
 *   - 1 instance  of IFFT2D_graph (response computation after PL accumulation)
 *
 * Processing flow (serial channel-by-channel):
 *
 *   For ch = 0 .. N_CHANNELS-1:
 *     PL fmt_adapter  --[FFTRowIn{ch}]-->  AIE fft2d[ch].fft_rows
 *     AIE fft2d[ch].fft_rows  --[FFTRowOut{ch}]--> PL transpose_buf (DDR)
 *     PL transpose_buf (col-major) --[FFTColIn{ch}]--> AIE fft2d[ch].fft_cols
 *     AIE fft2d[ch].fft_cols --[FFTColOut{ch}]--> PL cmul_accum  (accumulates Σ H_c* ⊙ F_c)
 *
 *   After all channels:
 *     PL cmul_accum --[IFFTRowIn0]--> AIE ifft2d.ifft_rows
 *     AIE ifft2d.ifft_rows --[IFFTRowOut0]--> PL transpose_buf (DDR)
 *     PL transpose_buf (col-major) --[IFFTColIn0]--> AIE ifft2d.ifft_cols
 *     AIE ifft2d.ifft_cols --[IFFTColOut0]--> PL peak_detect
 *
 * PLIO summary (4 per FFT channel + 4 for IFFT = 4*N_CHANNELS + 4 total):
 *   FFTRowIn{0..N_CHANNELS-1}   FFTRowOut{0..N_CHANNELS-1}
 *   FFTColIn{0..N_CHANNELS-1}   FFTColOut{0..N_CHANNELS-1}
 *   IFFTRowIn0  IFFTRowOut0  IFFTColIn0  IFFTColOut0
 */

#pragma once

#include "fft_graph.h"
#include "ifft_graph.h"

class MOSSE_graph : public graph
{
public:
    FFT2D_graph  fft2d[N_CHANNELS];
    IFFT2D_graph ifft2d;
};
