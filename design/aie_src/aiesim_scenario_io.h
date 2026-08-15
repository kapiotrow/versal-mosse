/*
 * aiesim_scenario_io.h
 * Scenario file I/O shared by the simulation harnesses.
 *
 * Lifted out of mosse_graph.cpp so the kernel-only bit-exactness harness
 * (kernel_only_graph.cpp) reads scenario data with exactly the same code as the
 * full-graph harness. A second copy would be free to drift, and this project has
 * already paid for duplicated contracts more than once — see the
 * preprocessing-coupling note in CLAUDE.md.
 *
 * Header-only (`static inline`) because both users are single translation units
 * compiled by aiecompiler; there is no link step to share an object with.
 *
 * Compiled under both __AIESIM__ and __X86SIM__.
 */

#pragma once

#include <cstdio>
#include <cstdint>

// Load n_elems cint16 (i.e. n_elems*2 int16 LE) from a flat binary into buf.
static inline bool load_cint16_bin(const char *path, int16_t *buf, int n_elems)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[sim] ERROR: cannot open %s\n", path);
        return false;
    }
    size_t got = fread(buf, sizeof(int16_t), (size_t)n_elems * 2, f);
    fclose(f);
    if ((int)got != n_elems * 2) {
        fprintf(stderr, "[sim] ERROR: %s: read %zu int16, expected %d\n",
                path, got, n_elems * 2);
        return false;
    }
    return true;
}

// Load exactly `bytes` raw bytes (used for the 64-byte conv2d weight buffer).
static inline bool load_raw_bin(const char *path, void *buf, int bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[sim] ERROR: cannot open %s\n", path);
        return false;
    }
    size_t got = fread(buf, 1, (size_t)bytes, f);
    fclose(f);
    if ((int)got != bytes) {
        fprintf(stderr, "[sim] ERROR: %s: read %zu bytes, expected %d\n",
                path, got, bytes);
        return false;
    }
    return true;
}

// Write raw bytes out. This is the harness's product: the kernel's exact output,
// for check_kernel_bitexact.py to diff against the Python golden.
static inline bool dump_raw_bin(const char *path, const void *buf, int bytes)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[sim] ERROR: cannot write %s\n", path);
        return false;
    }
    size_t put = fwrite(buf, 1, (size_t)bytes, f);
    fclose(f);
    if ((int)put != bytes) {
        fprintf(stderr, "[sim] ERROR: %s: wrote %zu bytes, expected %d\n",
                path, put, bytes);
        return false;
    }
    printf("[sim] wrote %s (%d bytes)\n", path, bytes);
    return true;
}
