/*
 * camera_capture.h
 * PL stub kernel: captures a video frame and writes it to DDR.
 *
 * In production: replace with MIPI CSI-2 RX Subsystem + ISP pipeline.
 * The stub zero-fills the frame buffer so the rest of the pipeline
 * can be tested without real camera hardware.
 *
 * Output layout: frame_buf[row * frame_cols + col] = {R, G, B} packed
 * as three consecutive uint8 bytes (interleaved RGB, row-major).
 */

#pragma once

#include "ap_int.h"

void camera_capture(
    ap_uint<8>  *frame_buf,    // DDR output: [frame_rows * frame_cols * 3] uint8
    int          frame_rows,
    int          frame_cols
);
