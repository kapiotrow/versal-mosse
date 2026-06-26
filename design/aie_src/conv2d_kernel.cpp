/*
 * conv2d_kernel.cpp
 * AIE-ML kernel: 3×3 INT8 convolution + separable Hanning window.
 *
 * One invocation processes one full PATCH_ROWS×PATCH_COLS grayscale patch:
 *   - Reads  PATCH_ROWS × PATCH_COLS  int8  samples from patch_in stream.
 *   - Writes PATCH_ROWS × PATCH_COLS  cint16 samples to feature_out stream.
 *
 * Algorithm
 * ---------
 *   Maintains a 3-row circular line buffer (with 1-sample zero-padding on
 *   each side) to form 3×3 sliding windows.
 *
 *   For each output pixel (r, c):
 *     acc  = bias_acc + Σ_{kr,kc} w[kr][kc] * buf[r+kr-1][c+kc-1]
 *     out  = saturate_int16(acc >> out_shift)
 *     wnd  = out * HANNING_128[r] / 32768 * HANNING_128[c] / 32768
 *     emit cint16{wnd, 0}
 *
 *   Border rows/cols are zero-padded (padding=1 matches the original model's
 *   same-size convolution).  The Hanning window zeroes rows 0 and PATCH_ROWS-1
 *   and cols 0 and PATCH_COLS-1, so the MAC is still performed but the output
 *   is guaranteed zero for those samples.
 *
 * Memory footprint on AIE tile
 * ----------------------------
 *   line buffer:    3 × (PATCH_COLS+2)  =  390 bytes
 *   weight scalars: 9 int8 + 1 int8 shift + 4 int32 bias  =  ~20 bytes
 *   Hanning table:  128 × 2 bytes  =  256 bytes  (in hanning_128.h, const)
 *   Total ≪ 64 KB tile data memory.
 */

#include "conv2d_kernel.h"
#include "hanning_128.h"
#include <cstring>

void conv2d_kernel(
    input_stream<int8>    *patch_in,
    output_stream<cint16> *feature_out,
    input_buffer<int8_t>  &weights)
{
    // ----------------------------------------------------------------
    // Load per-channel kernel parameters from the 64-byte weight buffer.
    // Layout (see conv2d_kernel.h):
    //   [0:9]   int8  w[KSIZE][KSIZE]   9 bytes, row-major
    //   [9]     int8  out_shift
    //   [10:14] int32 bias_acc (LE)
    // ----------------------------------------------------------------
    const int8_t *wb = weights.data();

    const int8_t w00 = wb[0], w01 = wb[1], w02 = wb[2];
    const int8_t w10 = wb[3], w11 = wb[4], w12 = wb[5];
    const int8_t w20 = wb[6], w21 = wb[7], w22 = wb[8];

    const int out_shift = (int)(uint8_t)wb[9];

    int32_t bias;
    // Byte-by-byte copy avoids alignment UB when wb is not 4-byte aligned.
    bias  = (int32_t)(uint8_t)wb[10];
    bias |= (int32_t)(uint8_t)wb[11] << 8;
    bias |= (int32_t)(uint8_t)wb[12] << 16;
    bias |= (int32_t)(uint8_t)wb[13] << 24;

    // ----------------------------------------------------------------
    // 3-row circular line buffer.
    // Each row stores [0-pad | PATCH_COLS pixels | 0-pad] → PATCH_COLS+2 bytes.
    // Initialised to 0: the zero-pad bytes remain 0 throughout, providing
    // implicit zero-padding at the left and right borders.
    // ----------------------------------------------------------------
    int8_t buf[3][PATCH_COLS + 2];
    memset(buf, 0, sizeof(buf));

    // ----------------------------------------------------------------
    // Phase 1: read first row — no output yet (need the row above row 0
    // to form a 3-row window; it is implicitly all-zeros from memset).
    // ----------------------------------------------------------------
    for (int c = 0; c < PATCH_COLS; c++)
    chess_prepare_for_pipelining
    chess_loop_range(PATCH_COLS, PATCH_COLS)
    {
        buf[0][c + 1] = readincr(patch_in);
    }

    // ----------------------------------------------------------------
    // Phase 2: main loop — read row r, output row r-1.
    //
    // At iteration r (r = 1 .. PATCH_ROWS-1):
    //   cur  = r % 3          — slot we write into (row r)
    //   mid  = (r + 2) % 3   — slot holding row r-1  (output row)
    //   top  = (r + 1) % 3   — slot holding row r-2
    //   bot  = cur            — slot we just wrote (row r = "row below" output row)
    // ----------------------------------------------------------------
    for (int r = 1; r < PATCH_ROWS; r++) {

        const int cur = r % 3;
        const int top = (r + 1) % 3;   // row r-2
        const int mid = (r + 2) % 3;   // row r-1  (output row index = r-1)
        // bot = cur (row r)

        // -- Read row r --
        buf[cur][0]             = 0;    // already 0, written for clarity
        for (int c = 0; c < PATCH_COLS; c++)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS, PATCH_COLS)
        {
            buf[cur][c + 1] = readincr(patch_in);
        }
        buf[cur][PATCH_COLS + 1] = 0;

        // -- Output row r-1 --
        const int out_r = r - 1;
        const int16_t h_r = HANNING_128[out_r];

        const int8_t *row_top = buf[top];
        const int8_t *row_mid = buf[mid];
        const int8_t *row_bot = buf[cur];

        for (int c = 0; c < PATCH_COLS; c++)
        chess_prepare_for_pipelining
        chess_loop_range(PATCH_COLS, PATCH_COLS)
        {
            const int c1 = c + 1;   // column index in padded buffer

            // 3×3 MAC: 9 int8 × int8 → accumulated into int32
            int32_t acc = bias;
            acc += (int32_t)w00 * row_top[c1 - 1];
            acc += (int32_t)w01 * row_top[c1];
            acc += (int32_t)w02 * row_top[c1 + 1];
            acc += (int32_t)w10 * row_mid[c1 - 1];
            acc += (int32_t)w11 * row_mid[c1];
            acc += (int32_t)w12 * row_mid[c1 + 1];
            acc += (int32_t)w20 * row_bot[c1 - 1];
            acc += (int32_t)w21 * row_bot[c1];
            acc += (int32_t)w22 * row_bot[c1 + 1];

            // Scale int32 acc → int16
            int32_t shifted = acc >> out_shift;
            int16_t out16;
            if      (shifted >  32767) out16 =  32767;
            else if (shifted <=     0) out16 =  0;        // ReLU
            else                       out16 = (int16_t)shifted;

            // Separable Hanning window (Q1.15 × Q1.15 → int16)
            // Two right-shifts of 15 apply both row and column weights.
            int16_t h_c   = HANNING_128[c];
            int32_t wnd   = ((int32_t)out16 * h_r) >> 15;
            wnd           = (wnd * h_c) >> 15;
            int16_t wnd16;
            if      (wnd >  32767) wnd16 =  32767;
            else if (wnd < -32768) wnd16 = -32768;
            else                   wnd16 = (int16_t)wnd;

            cint16_t result;
            result.real = wnd16;
            result.imag = 0;
            writeincr(feature_out, result);
        }
    }

    // ----------------------------------------------------------------
    // Phase 3: output the last row (PATCH_ROWS - 1).
    //
    // The "row below" (row PATCH_ROWS) doesn't exist; we need zeros there.
    // Conveniently, buf[PATCH_ROWS % 3] was last written at row r where
    // r % 3 == PATCH_ROWS % 3.  For PATCH_ROWS=128: that slot was last
    // used at r = 125.  Zero it out so the bottom border is correct.
    //
    // HANNING_128[PATCH_ROWS-1] = HANNING_128[127] = 0, so the entire
    // last row output is zero.  We still write PATCH_COLS zero samples
    // to keep the downstream FFT's sample count correct.
    // ----------------------------------------------------------------
    const int last_bot = PATCH_ROWS % 3;
    memset(buf[last_bot], 0, PATCH_COLS + 2);

    const int last_r   = PATCH_ROWS - 1;
    const int last_top = (PATCH_ROWS + 1) % 3;   // row PATCH_ROWS-2
    const int last_mid = (PATCH_ROWS + 2) % 3;   // row PATCH_ROWS-1

    const int16_t h_last = HANNING_128[last_r];   // = 0

    const int8_t *row_top_last = buf[last_top];
    const int8_t *row_mid_last = buf[last_mid];
    const int8_t *row_bot_last = buf[last_bot];   // zeroed above

    for (int c = 0; c < PATCH_COLS; c++)
    chess_prepare_for_pipelining
    chess_loop_range(PATCH_COLS, PATCH_COLS)
    {
        const int c1 = c + 1;

        int32_t acc = bias;
        acc += (int32_t)w00 * row_top_last[c1 - 1];
        acc += (int32_t)w01 * row_top_last[c1];
        acc += (int32_t)w02 * row_top_last[c1 + 1];
        acc += (int32_t)w10 * row_mid_last[c1 - 1];
        acc += (int32_t)w11 * row_mid_last[c1];
        acc += (int32_t)w12 * row_mid_last[c1 + 1];
        acc += (int32_t)w20 * row_bot_last[c1 - 1];
        acc += (int32_t)w21 * row_bot_last[c1];
        acc += (int32_t)w22 * row_bot_last[c1 + 1];

        int32_t shifted = acc >> out_shift;
        int16_t out16;
        if      (shifted >  32767) out16 =  32767;
        else if (shifted <=     0) out16 =  0;        // ReLU
        else                       out16 = (int16_t)shifted;

        int16_t h_c   = HANNING_128[c];
        int32_t wnd   = ((int32_t)out16 * h_last) >> 15;   // h_last = 0 → wnd = 0
        wnd           = (wnd * h_c) >> 15;
        int16_t wnd16;
        if      (wnd >  32767) wnd16 =  32767;
        else if (wnd < -32768) wnd16 = -32768;
        else                   wnd16 = (int16_t)wnd;

        cint16_t result;
        result.real = wnd16;
        result.imag = 0;
        writeincr(feature_out, result);
    }
}
