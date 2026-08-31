/*
 * ebox_crosscheck.cpp — emit an H and this build's filter_box_energy_fraction()
 * of it, so scripts/check_ebox_crosscheck.py can score the SAME H with the
 * OFFLINE implementation (rgb_vs_gray_loop.py's box_energy_fraction).
 *
 * WHY A CROSS-CHECK AND NOT ANOTHER UNIT TEST. The unit tests in
 * test_mosse_filter.cpp assert this function against a naive DFT written in the
 * same file — good for the transform, blind to the CONTRACT. The board and the
 * offline bench are supposed to compute the same statistic, and the first cut
 * did not: the C++ half squared H directly while the Python half transformed
 * first. Both were self-consistent, both were green, and the disagreement only
 * appeared on hardware as 0.0000 on every frame. The only instrument that can
 * see that class of defect is one that runs BOTH implementations on the SAME
 * input and compares the numbers.
 *
 * Deterministic: the PRNG is a fixed LCG, so the Python side regenerates
 * nothing — it reads the exact bytes this program scored.
 *
 * Usage:  ebox_crosscheck <out.bin> [channels] [rows] [cols] [box_r] [box_c] [seed]
 *         prints one line: "<value>"
 */

#include "mosse_filter.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out.bin> [ch] [rows] [cols] [br] [bc] [seed]\n",
                     argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int ch   = (argc > 2) ? std::atoi(argv[2]) : 3;
    const int rows = (argc > 3) ? std::atoi(argv[3]) : 32;
    const int cols = (argc > 4) ? std::atoi(argv[4]) : 32;
    const int br   = (argc > 5) ? std::atoi(argv[5]) : 16;
    const int bc   = (argc > 6) ? std::atoi(argv[6]) : 16;
    unsigned    s  = (argc > 7) ? (unsigned)std::strtoul(argv[7], 0, 10) : 2026u;

    // A FLAT-SPECTRUM H would make the two conventions agree by accident (a
    // spatial delta transforms to constant |H|, and both a transformed and an
    // untransformed reading of a constant are the area ratio). So the spectrum
    // is deliberately shaped: an exponential taper in frequency, which puts the
    // spatial energy somewhere non-degenerate and makes the two readings differ
    // by a lot when the transform is missing.
    const size_t n = (size_t)ch * rows * cols;
    std::vector<mosse::cfloat> H(n);
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return (float)((int)(s >> 16) % 2000 - 1000) / 1000.0f;
    };
    for (int k = 0; k < ch; ++k)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const int dr = (r <= rows / 2) ? r : rows - r;   // circular freq
                const int dc = (c <= cols / 2) ? c : cols - c;
                const float taper = 1.0f / (1.0f + 0.35f * (float)(dr + dc));
                H[((size_t)k * rows + r) * cols + c] =
                    mosse::cfloat(rnd() * taper, rnd() * taper);
            }

    std::FILE *f = std::fopen(path, "wb");
    if (!f) { std::perror("fopen"); return 1; }
    // Interleaved float32 {re, im}, channel-major — the layout the header
    // documents for H_all, so the Python side needs no convention of its own.
    if (std::fwrite(H.data(), sizeof(mosse::cfloat), n, f) != n) {
        std::fprintf(stderr, "short write\n"); std::fclose(f); return 1;
    }
    std::fclose(f);

    std::printf("%.12f\n",
                mosse::filter_box_energy_fraction(H.data(), ch, rows, cols, br, bc));
    return 0;
}
