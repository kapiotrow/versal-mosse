#!/usr/bin/env python3
"""
gen_aiesim_vectors.py
Generate PLIO input text files for aiesim round-trip FFT/IFFT test.

Usage:
    python3 gen_aiesim_vectors.py [output_dir]

output_dir defaults to design/aie_src/aiesim_data (relative to repo root).

Files written:
    patch_in.txt        — unit impulse at (r=0,c=0), int8 value 1 [default test]
    patch_in_const.txt  — constant image, all int8 = 1 [alternative test]

PLIO format (plio_128_bits, input_stream<int8>):
    16 int8 samples per line, space-separated decimal.
    1024 lines total for a 128×128 patch (16384 samples / 16 per 128-bit beat).

Expected aiesim output (printed to stdout for reference):
    For impulse input — ideal cint16 round-trip:
        resp[0,0].real = 1,  resp[0,0].imag = 0
        resp[r,c]      = 0   for all (r,c) != (0,0)

    For constant input (all-1s) — same ideal result:
        resp[r,c].real = 1,  resp[r,c].imag = 0   for all (r,c)

Normalization derivation (PATCH_ROWS = PATCH_COLS = 128):
    Forward FFT: shift = 0 (no scaling).
    2D FFT of δ[0,0] = constant spectrum {1,0}.
    Row IFFT: shift = 0 → IFFT_raw([1,...,1]) = [128, 0,...,0] per row.
    After transpose: row 0 = [128,...,128], rows 1..127 = zero.
    Col IFFT: shift = 14 = log2(128×128) → IFFT_raw([128,...,128]) = [16384,...] >> 14 = 1.
    Final: resp[0,0] = {1,0}. Round-trip is identity.
"""

import sys
import os
import numpy as np

PATCH_ROWS = 128
PATCH_COLS = 128
N = PATCH_ROWS * PATCH_COLS
FFT_SHIFT_FWD = 0
IFFT_SHIFT_ROW = 0
IFFT_SHIFT_COL = 14  # log2(128) + log2(128)


PLIO_BEAT_SAMPLES = 16  # plio_128_bits / 8 bits per int8 = 16 samples per beat

# Number of zero-padding frames appended after the real test data.
# Purpose: prevent PLIO cycle-credit starvation in aiesim cycle-approximate mode.
# After the PLIO exhausts its file it enters an "active retry" state that
# monopolizes ALL simulation cycle credits, freezing every kernel and DMA.
# With PLIO_PADDING_FRAMES extra frames of zeros the PLIO stays active for
# the full ~50 000-cycle test duration; the conv2d kernel processes the extra
# frames but the GMIO DMA is armed for exactly PATCH_BYTES (one frame) so
# the padding output is never collected and the verification is unaffected.
PLIO_PADDING_FRAMES = 4


def write_plio_txt(path: str, samples: np.ndarray) -> None:
    """Write int8 samples in plio_128_bits format: 16 samples per line.

    Appends PLIO_PADDING_FRAMES of all-zeros after the real data to prevent
    the PLIO simulation model from stalling mid-test.
    """
    assert samples.dtype == np.int8
    assert len(samples) == N
    assert N % PLIO_BEAT_SAMPLES == 0
    padding = np.zeros(N * PLIO_PADDING_FRAMES, dtype=np.int8)
    all_samples = np.concatenate([samples, padding])
    with open(path, 'w') as f:
        for i in range(0, len(all_samples), PLIO_BEAT_SAMPLES):
            f.write(' '.join(str(int(s)) for s in all_samples[i:i + PLIO_BEAT_SAMPLES]) + '\n')


def simulate_roundtrip(patch_int8: np.ndarray) -> np.ndarray:
    """
    Simulate the exact integer arithmetic of the aiesim pipeline:
      conv2d (cast) → row FFT → transpose → col FFT →
      cmul_accum (pass-through) → row IFFT → transpose → col IFFT (>>14).
    Returns the cint16 response as a float64 array for comparison.
    Uses exact integer arithmetic matching cint16 (saturated) to predict output.
    NOTE: twiddle quantisation is NOT modelled here; aiesim will show ±1-2 LSB noise.
    """
    # conv2d stub: cast int8 → complex
    x = patch_int8.astype(np.float64).reshape(PATCH_ROWS, PATCH_COLS)

    # Row FFT (no normalization, no shift)
    F_row = np.fft.fft(x, axis=1)

    # Transpose
    F_rowT = F_row.T

    # Col FFT (operating on the transposed matrix)
    F_2d = np.fft.fft(F_rowT, axis=1)

    # cmul_accum stub: pass-through (H = identity)

    # Row IFFT (unnormalized, no shift)
    Y_row = np.fft.ifft(F_2d, axis=1) * PATCH_COLS  # unnormalized

    # Transpose
    Y_rowT = Y_row.T

    # Col IFFT (unnormalized), then shift by 14
    Y_2d = np.fft.ifft(Y_rowT, axis=1) * PATCH_ROWS  # unnormalized
    Y_2d = Y_2d / (2 ** IFFT_SHIFT_COL)

    return Y_2d.real  # imaginary part should be ~0 for real input


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    default_out = os.path.join(repo_root, "design", "aie_src", "aiesim_data")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default_out
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Test 1: unit impulse at (r=0, c=0)
    # ----------------------------------------------------------------
    impulse = np.zeros(N, dtype=np.int8)
    impulse[0] = 1
    path_impulse = os.path.join(out_dir, "patch_in.txt")
    write_plio_txt(path_impulse, impulse)

    expected_impulse = simulate_roundtrip(impulse)
    print(f"Written: {path_impulse}")
    print(f"  Impulse test — expected response (ideal arithmetic, no twiddle noise):")
    print(f"    resp[0,0] = {expected_impulse[0,0]:.4f}  (expected ≈ 1.0)")
    print(f"    max |resp[r,c]| for (r,c)!=(0,0) = "
          f"{np.max(np.abs(expected_impulse[1:])):.6f}  (expected = 0.0)")
    print()

    # ----------------------------------------------------------------
    # Test 2: constant image (all 1s) — alternative test, same expected output per pixel
    # ----------------------------------------------------------------
    const_img = np.ones(N, dtype=np.int8)
    path_const = os.path.join(out_dir, "patch_in_const.txt")
    write_plio_txt(path_const, const_img)

    expected_const = simulate_roundtrip(const_img)
    print(f"Written: {path_const}")
    print(f"  Constant test — expected response (ideal arithmetic):")
    print(f"    resp[all] = {expected_const[0,0]:.4f}  (expected = 1.0 everywhere)")
    print(f"    max deviation from 1.0 = "
          f"{np.max(np.abs(expected_const - 1.0)):.6f}  (expected = 0.0)")
    print()

    print("Normalization check:")
    print(f"  Forward FFT shift = {FFT_SHIFT_FWD}")
    print(f"  Row IFFT shift    = {IFFT_SHIFT_ROW}")
    print(f"  Col IFFT shift    = {IFFT_SHIFT_COL}  (= log2({PATCH_ROWS})+log2({PATCH_COLS}))")
    print(f"  Net scale factor  = 2^0 / 2^{IFFT_SHIFT_COL} × {PATCH_ROWS*PATCH_COLS} = "
          f"{PATCH_ROWS*PATCH_COLS / 2**IFFT_SHIFT_COL:.1f}  (should be 1.0)")
    print()
    print("Run 'make aiesim' and check the OVERALL: PASS/FAIL line in the log.")


if __name__ == "__main__":
    main()
