#!/usr/bin/env python3
"""
scripts/conv_weight_layout.py

Python mirror of design/aie_src/conv_weight_layout.h — the conv2d weight-buffer
layout, derived from the input-plane count rather than hardcoded.

Read the header for the layout itself and for why it is derived. This module
exists because Python cannot include a C header; the two carry the SAME
formula, so a divergence requires editing both in the same wrong way, which is
a much narrower failure than four independent sets of hardcoded offsets.

The tag byte is the runtime backstop for exactly that risk: unpack() reads
byte 63 and refuses a buffer whose layout is not the one the caller asked for,
so a gray reader pointed at an RGB export fails loudly instead of reporting
sixteen healthy channels assembled out of R-plane taps.

Every consumer of layer0_weights.bin should go through here:
  export_weights.py, check_collapse.py, gen_aiesim_vectors.py, phase1_sweep.py
"""

import struct

KSIZE     = 3
BUF_BYTES = 64      # CONV_WEIGHT_BYTES_PAD


class Layout:
    """Byte offsets for one output channel's 64-byte buffer."""

    def __init__(self, n_in: int):
        if n_in not in (1, 3):
            raise ValueError(f"n_in must be 1 (luminance) or 3 (RGB), got {n_in}")
        self.n_in    = n_in
        self.raw     = n_in * KSIZE * KSIZE          # 9 or 27
        self.taps    = 0
        self.shift   = self.raw
        self.bias    = self.shift + 1
        self.dequant = self.bias + 4
        self.mean    = self.dequant + 4
        self.end     = self.mean + 4
        self.tag     = BUF_BYTES - 1
        assert self.end <= self.tag, "fields overrun the 64-byte buffer"

    def __repr__(self):
        return (f"Layout(n_in={self.n_in}, taps=[0:{self.raw}), shift={self.shift}, "
                f"bias={self.bias}, dequant={self.dequant}, mean={self.mean}, "
                f"tag={self.tag})")

    def plane(self, ic: int) -> int:
        """Offset of input plane ic's 3x3 block."""
        return ic * KSIZE * KSIZE


def pack(lay: Layout, taps_int8, shift: int, bias_acc: int,
         dequant_scale: float, mean_prev: int = 0) -> bytes:
    """Pack one output channel. `taps_int8` is flattened [n_in, K, K], row-major.

    mean_prev defaults to 0 and is normally left there: the HOST seeds it from
    bias_acc >> out_shift before frame 0 and rewrites it every frame after. An
    unseeded mean_prev on frame 0 rails the ch16 response — see the mean_prev
    entry in CLAUDE.md — but that is the host's job, not the exporter's.
    """
    taps = bytes(bytearray((int(t) & 0xFF) for t in taps_int8))
    if len(taps) != lay.raw:
        raise ValueError(f"expected {lay.raw} taps, got {len(taps)}")

    buf = bytearray(BUF_BYTES)
    buf[lay.taps:lay.raw] = taps
    buf[lay.shift] = int(shift) & 0xFF
    struct.pack_into('<i', buf, lay.bias,    int(bias_acc))
    struct.pack_into('<f', buf, lay.dequant, float(dequant_scale))
    struct.pack_into('<i', buf, lay.mean,    int(mean_prev))
    buf[lay.tag] = lay.n_in
    return bytes(buf)


def detect(buf: bytes) -> int:
    """Input-plane count of a 64-byte channel buffer, from the tag byte.

    Returns 1 for files exported before the tag existed (byte 63 == 0), which
    were all grayscale.
    """
    if len(buf) < BUF_BYTES:
        raise ValueError(f"buffer is {len(buf)} bytes, need {BUF_BYTES}")
    tag = buf[BUF_BYTES - 1]
    if tag == 0:
        return 1            # legacy export, pre-tag: grayscale by construction
    if tag not in (1, 3):
        raise ValueError(f"layout tag byte is {tag}, expected 0, 1 or 3 — "
                         "this file was not written by export_weights.py")
    return tag


def unpack(buf: bytes, expect_n_in: int = None):
    """(taps_int8_list, out_shift, bias_acc, dequant_scale, mean_prev, layout).

    Taps come back signed. Raises if expect_n_in is given and disagrees with the
    tag — that mismatch is the whole point of the tag.
    """
    n_in = detect(buf)
    if expect_n_in is not None and n_in != expect_n_in:
        raise ValueError(f"weight buffer is CONV_IN_CH={n_in}, caller expected "
                         f"{expect_n_in}. Re-run `make weights CONV_IN_CH={expect_n_in}` "
                         f"or fix the caller.")
    lay  = Layout(n_in)
    taps = [b - 256 if b > 127 else b for b in buf[lay.taps:lay.raw]]
    return (taps,
            buf[lay.shift],
            struct.unpack_from('<i', buf, lay.bias)[0],
            struct.unpack_from('<f', buf, lay.dequant)[0],
            struct.unpack_from('<i', buf, lay.mean)[0],
            lay)


def load_bin(path, expect_n_in: int = None):
    """All channels of a layer0_weights.bin as a list of unpack() tuples."""
    raw = open(path, 'rb').read()
    if len(raw) % BUF_BYTES:
        raise ValueError(f"{path}: {len(raw)} bytes is not a multiple of {BUF_BYTES}")
    return [unpack(raw[i:i + BUF_BYTES], expect_n_in)
            for i in range(0, len(raw), BUF_BYTES)]


if __name__ == "__main__":
    for n in (1, 3):
        print(Layout(n))
