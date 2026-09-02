#!/usr/bin/env python3
"""
scripts/conv_weight_layout.py

Python mirror of design/aie_src/conv_weight_layout.h — the conv2d weight-buffer
layout, derived from the input-plane count rather than hardcoded.

Read the header for the layout itself and for why it is derived. This module
exists because Python cannot include a C header; the two carry the SAME
formula, so a divergence requires editing both in the same wrong way, which is
a much narrower failure than four independent sets of hardcoded offsets.

The tag bytes are the runtime backstop for exactly that risk: unpack() reads
the LAST TWO bytes of a channel buffer -- CONV_KSIZE then CONV_IN_CH -- and
refuses a buffer whose layout is not the one the caller asked for, so a gray
reader pointed at an RGB export fails loudly instead of reporting sixteen
healthy channels assembled out of R-plane taps.

The buffer is no longer a fixed 64 bytes. A 7x7 RGB bank is 147 taps and does
not fit; buf_bytes() rounds taps + fields + tags up to the 64-byte GMIO
granularity, which reproduces 64 for every 3x3 bank and gives 192 for 7x7 RGB.
Because the size is no longer a constant, a reader that does not already know
(n_in, ksize) from its build must call sniff() rather than slice a fixed 64.

Every consumer of layer0_weights.bin should go through here:
  export_weights.py, check_collapse.py, gen_aiesim_vectors.py, phase1_sweep.py
"""

import struct

KSIZE     = 3       # the DEFAULT kernel size, i.e. every bank shipped before 7x7
BUF_BYTES = 64      # CONV_WEIGHT_BYTES_PAD at KSIZE=3, for callers that only
                    # ever handle 3x3 banks. Anything kernel-size-generic must
                    # use Layout.buf / buf_bytes() instead.

# Field bytes after the taps (out_shift, bias_acc, dequant_scale, mean_prev)
# plus the two tag bytes. Mirrors CONV_WEIGHT_BYTES_MIN in the header.
FIELD_BYTES = 13
TAG_BYTES   = 2
GRAIN       = 64    # GMIO alignment granularity


def buf_bytes(n_in: int, ksize: int = KSIZE) -> int:
    """CONV_WEIGHT_BYTES_PAD -- the same formula as the header."""
    raw = n_in * ksize * ksize
    return ((raw + FIELD_BYTES + TAG_BYTES + GRAIN - 1) // GRAIN) * GRAIN


class Layout:
    """Byte offsets for one output channel's 64-byte buffer."""

    def __init__(self, n_in: int, ksize: int = KSIZE):
        if n_in not in (1, 3):
            raise ValueError(f"n_in must be 1 (luminance) or 3 (RGB), got {n_in}")
        if ksize not in (3, 5, 7):
            raise ValueError(f"ksize must be 3, 5 or 7, got {ksize}")
        self.n_in    = n_in
        self.ksize   = ksize
        self.raw     = n_in * ksize * ksize          # 9, 27 or 147
        self.buf     = buf_bytes(n_in, ksize)
        self.taps    = 0
        self.shift   = self.raw
        self.bias    = self.shift + 1
        self.dequant = self.bias + 4
        self.mean    = self.dequant + 4
        self.end     = self.mean + 4
        self.tag     = self.buf - 1                  # CONV_IN_CH
        self.tag_k   = self.buf - 2                  # CONV_KSIZE
        assert self.end <= self.tag_k, "fields overrun the buffer"

    def __repr__(self):
        return (f"Layout(n_in={self.n_in}, k={self.ksize}, buf={self.buf}, "
                f"taps=[0:{self.raw}), shift={self.shift}, "
                f"bias={self.bias}, dequant={self.dequant}, mean={self.mean}, "
                f"tag_k={self.tag_k}, tag={self.tag})")

    def plane(self, ic: int) -> int:
        """Offset of input plane ic's KxK block."""
        return ic * self.ksize * self.ksize


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

    buf = bytearray(lay.buf)
    buf[lay.taps:lay.raw] = taps
    buf[lay.shift] = int(shift) & 0xFF
    struct.pack_into('<i', buf, lay.bias,    int(bias_acc))
    struct.pack_into('<f', buf, lay.dequant, float(dequant_scale))
    struct.pack_into('<i', buf, lay.mean,    int(mean_prev))
    buf[lay.tag_k] = lay.ksize
    buf[lay.tag]   = lay.n_in
    return bytes(buf)


def detect(buf: bytes) -> int:
    """Input-plane count of ONE channel buffer, from its last byte.

    `buf` must be exactly one channel buffer (or start at one and be sliced to
    its length) -- the tag lives at the END, so a wrong length reads a tap.

    Returns 1 for files exported before the tags existed (last byte 0), which
    were all grayscale 3x3.
    """
    if len(buf) < GRAIN:
        raise ValueError(f"buffer is {len(buf)} bytes, need at least {GRAIN}")
    tag = buf[len(buf) - 1]
    if tag == 0:
        return 1            # legacy export, pre-tag: grayscale by construction
    if tag not in (1, 3):
        raise ValueError(f"layout tag byte is {tag}, expected 0, 1 or 3 — "
                         "this file was not written by export_weights.py")
    return tag


def detect_ksize(buf: bytes) -> int:
    """Kernel size of ONE channel buffer, from its second-to-last byte.

    Returns 3 for legacy exports (byte 0), which were all 3x3.
    """
    if len(buf) < GRAIN:
        raise ValueError(f"buffer is {len(buf)} bytes, need at least {GRAIN}")
    tag = buf[len(buf) - 2]
    if tag == 0:
        return KSIZE        # legacy export, pre-tag: 3x3 by construction
    if tag not in (3, 5, 7):
        raise ValueError(f"ksize tag byte is {tag}, expected 0, 3, 5 or 7 — "
                         "this file was not written by export_weights.py")
    return tag


def sniff(path):
    """(n_out, n_in, ksize, buf_bytes) of a layer0_weights.bin, from its tags.

    For a reader that does NOT already know the build's (CONV_IN_CH,
    CONV_KSIZE). Candidate buffer sizes are tried LARGEST FIRST and a candidate
    wins only if every channel carries the same NON-ZERO tag pair and the size
    is the one that pair implies. Largest-first plus the non-zero requirement is
    what disambiguates: a 32x192 B file is also a whole number of 64 B blocks,
    but read that way most blocks end in 0x00 tags, which only a legacy file may
    do -- and a legacy file is always 64 B, so it can never be the loser of this
    ordering.
    """
    raw = open(path, 'rb').read()
    for cand in sorted({buf_bytes(n, k) for n in (1, 3) for k in (3, 5, 7)},
                       reverse=True):
        if not raw or len(raw) % cand:
            continue
        pairs = {(raw[i + cand - 2], raw[i + cand - 1])
                 for i in range(0, len(raw), cand)}
        if len(pairs) != 1:
            continue
        k, n_in = pairs.pop()
        # Legacy zeros mean "the tag did not exist yet", and every such file was
        # 3x3 -- the ksize tag is newer than the CONV_IN_CH one, so a shipped RGB
        # export carries (0, 3) and must still be readable.
        if k == 0:
            k = KSIZE
        if n_in == 0:
            n_in = 1
        if k not in (3, 5, 7) or n_in not in (1, 3):
            continue
        if buf_bytes(n_in, k) != cand:
            continue
        return len(raw) // cand, n_in, k, cand
    # No tagged interpretation: legacy pre-tag file, 64 B grayscale 3x3.
    if len(raw) % GRAIN:
        raise ValueError(f"{path}: {len(raw)} bytes is not a whole number of "
                         f"channel buffers under any known layout")
    return len(raw) // GRAIN, 1, KSIZE, GRAIN


def unpack(buf: bytes, expect_n_in: int = None, expect_ksize: int = None):
    """(taps_int8_list, out_shift, bias_acc, dequant_scale, mean_prev, layout).

    `buf` must be exactly one channel buffer: the tags are at its end.

    Taps come back signed. Raises if expect_n_in / expect_ksize are given and
    disagree with the tags — that mismatch is the whole point of the tags.
    """
    n_in  = detect(buf)
    ksize = detect_ksize(buf)
    if expect_n_in is not None and n_in != expect_n_in:
        raise ValueError(f"weight buffer is CONV_IN_CH={n_in}, caller expected "
                         f"{expect_n_in}. Re-run `make weights CONV_IN_CH={expect_n_in}` "
                         f"or fix the caller.")
    if expect_ksize is not None and ksize != expect_ksize:
        raise ValueError(f"weight buffer is CONV_KSIZE={ksize}, caller expected "
                         f"{expect_ksize}. Re-run `make weights CONV_KSIZE={expect_ksize}` "
                         f"or fix the caller.")
    lay  = Layout(n_in, ksize)
    taps = [b - 256 if b > 127 else b for b in buf[lay.taps:lay.raw]]
    return (taps,
            buf[lay.shift],
            struct.unpack_from('<i', buf, lay.bias)[0],
            struct.unpack_from('<f', buf, lay.dequant)[0],
            struct.unpack_from('<i', buf, lay.mean)[0],
            lay)


def load_bin(path, expect_n_in: int = None, expect_ksize: int = None):
    """All channels of a layer0_weights.bin as a list of unpack() tuples.

    The buffer size comes from sniff() when the caller does not pin it, so this
    reads a 7x7 file and a 3x3 file with the same call.
    """
    n_out, n_in, ksize, cand = sniff(path)
    if expect_n_in is not None or expect_ksize is not None:
        # Pin the size to what the CALLER expects, so a mismatch is reported by
        # unpack() against the tags rather than silently re-blocked by sniff().
        cand = buf_bytes(expect_n_in if expect_n_in is not None else n_in,
                         expect_ksize if expect_ksize is not None else ksize)
    raw = open(path, 'rb').read()
    if len(raw) % cand:
        raise ValueError(f"{path}: {len(raw)} bytes is not a multiple of {cand}")
    return [unpack(raw[i:i + cand], expect_n_in, expect_ksize)
            for i in range(0, len(raw), cand)]


if __name__ == "__main__":
    for k in (3, 5, 7):
        for n in (1, 3):
            print(Layout(n, k))
