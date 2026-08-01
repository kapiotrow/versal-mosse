#!/usr/bin/env python3
"""
analyze_plio_vcd.py — summarise the PL->AIE AXIS handshake from plio_probe.vcd.

Answers, for the PLIO smoke test:
  * did the producer ever assert TVALID?
  * did the AIE side ever assert TREADY?
  * how many beats actually completed (TVALID & TREADY on a clock edge)?
  * where did it stall, and in which state?

Usage: analyze_plio_vcd.py <plio_probe.vcd> [expected_beats]

The VCD is written incrementally by xsim, so a truncated file (emulation killed
mid-hang) is expected and handled.
"""

import re
import sys
from collections import OrderedDict


def parse_vcd(path):
    """Return (id->[names], timescale, [(time, id, value_str)]) from a VCD.

    An id maps to a LIST of names, not one name: xsim gives aliased nets the same
    VCD identifier. Here `stream_src_0/out_r_TVALID` and `VitisRegion/out_r_tvalid`
    share an id precisely because they are the same net with nothing in between —
    which is itself useful evidence, so it must not be flattened away.
    """
    names = OrderedDict()
    changes = []
    time = 0
    timescale = "1ps"

    var_re = re.compile(r"^\$var\s+\S+\s+(\d+)\s+(\S+)\s+(.+?)\s*\$end")

    with open(path, "r", errors="replace") as fh:
        in_defs = True
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if in_defs:
                m = var_re.match(line)
                if m:
                    _width, ident, name = m.groups()
                    names.setdefault(ident, []).append(name.strip())
                    continue
                if line.startswith("$timescale"):
                    parts = line.split()
                    if len(parts) > 1:
                        timescale = parts[1]
                    continue
                if line.startswith("$enddefinitions"):
                    in_defs = False
                    continue
                continue

            if line.startswith("#"):
                try:
                    time = int(line[1:])
                except ValueError:
                    pass
                continue

            # scalar change: e.g. "1!" / "0!" / "x!"
            if line[0] in "01xzXZ" and len(line) > 1:
                changes.append((time, line[1:], line[0]))
            # vector change: e.g. "b1010 !"
            elif line[0] in "bB":
                parts = line.split()
                if len(parts) == 2:
                    changes.append((time, parts[1], parts[0][1:]))

    return names, timescale, changes


def find_id(names, suffix):
    """Locate the VCD id whose declared name ends with `suffix` (case-insensitive)."""
    suffix = suffix.lower()
    for ident, name in names.items():
        base = name.split("[")[0].lower()
        if base.endswith(suffix):
            return ident
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    names, timescale, changes = parse_vcd(path)
    if not names:
        print(f"ERROR: no $var declarations in {path} — probe never armed?")
        return 1

    print(f"timescale : {timescale}")
    print(f"unique ids: {len(names)}")
    print(f"changes   : {len(changes)}")
    print()
    print("declared signals (ids shared by >1 name are ALIASED = the same net):")
    for ident, nlist in names.items():
        tag = "  <-- ALIASED" if len(nlist) > 1 else ""
        print(f"  {ident:>4}  {', '.join(nlist)}{tag}")
    print()

    # The HLS kernel port is upper-case (out_r_TVALID); the VitisRegion boundary
    # port that feeds the AIE TLM adapter is lower-case (out_r_tvalid). Every name
    # on every id is checked, so an id carrying both ends populates both keys.
    ids = {}
    for ident, nlist in names.items():
        for name in nlist:
            base = name.split("[")[0].strip()
            if base.endswith("TVALID"):
                ids["src_tvalid"] = ident
            elif base.endswith("TREADY"):
                ids["src_tready"] = ident
            elif base.endswith("tvalid"):
                ids["aie_tvalid"] = ident
            elif base.endswith("tready"):
                ids["aie_tready"] = ident
            elif base.endswith("ap_clk"):
                ids["clk"] = ident
            elif base.endswith("ap_rst_n"):
                ids["rstn"] = ident

    aliased = (ids.get("src_tvalid") is not None
               and ids.get("src_tvalid") == ids.get("aie_tvalid"))
    if aliased:
        print("NOTE: the producer port and the AIE-facing boundary port share VCD ids,")
        print("      i.e. they are ONE net — there is no interconnect or FIFO between")
        print("      the PL kernel and the shim that could buffer or hide beats.")
        print()

    # Replay the changes, tracking state and counting completed beats.
    state = {}
    stats = {k: {"asserted": 0, "first_high": None, "last_high": None}
             for k in ("src_tvalid", "src_tready", "aie_tvalid", "aie_tready")}
    beats_src = 0
    beats_aie = 0
    prev_clk = None

    for t, ident, val in changes:
        state[ident] = val

        for key in stats:
            if ids.get(key) == ident and val == "1":
                stats[key]["asserted"] += 1
                if stats[key]["first_high"] is None:
                    stats[key]["first_high"] = t
                stats[key]["last_high"] = t

        # Count beats on the rising edge of ap_clk.
        if ids.get("clk") == ident and val == "1" and prev_clk != "1":
            if state.get(ids.get("src_tvalid")) == "1" and \
               state.get(ids.get("src_tready")) == "1":
                beats_src += 1
            if state.get(ids.get("aie_tvalid")) == "1" and \
               state.get(ids.get("aie_tready")) == "1":
                beats_aie += 1
        if ids.get("clk") == ident:
            prev_clk = val

    print("handshake activity:")
    for key in ("src_tvalid", "src_tready", "aie_tvalid", "aie_tready"):
        s = stats[key]
        if ids.get(key) is None:
            print(f"  {key:<11} NOT CAPTURED")
            continue
        if s["asserted"] == 0:
            print(f"  {key:<11} NEVER ASSERTED")
        else:
            print(f"  {key:<11} rose {s['asserted']:>6} times, "
                  f"first @ {s['first_high']}, last @ {s['last_high']}")
    print()
    print(f"beats completed (producer port) : {beats_src} / {expected}")
    if aliased:
        print("beats completed (AIE boundary)  : same net as above (aliased)")
        beats_aie = beats_src
    else:
        print(f"beats completed (AIE boundary)  : {beats_aie} / {expected}")
    print()

    # Verdict. When the two ends are aliased, the producer's TREADY *is* the AIE's
    # TREADY, so fall back to it rather than reporting "never asserted" — that
    # mis-scored a fully passing 256/256 run as a permanent-backpressure failure.
    sv = stats["src_tvalid"]["asserted"]
    ar = stats["aie_tready"]["asserted"] or (stats["src_tready"]["asserted"] if aliased else 0)
    # ---- Diagnostic signals ------------------------------------------------
    # Everything not part of the AXIS handshake: the kernel's ap_ctrl state, the
    # HLS stall-cause flags, the m_axi read channel, and any pipelined sub-loops.
    # `last` is what the signal was left sitting at, which for a hang is the
    # interesting part; `edges` says whether it ever moved at all.
    classified = {ids.get(k) for k in ids}
    extra = [(i, n) for i, n in names.items() if i not in classified]
    if extra:
        print("diagnostic signals (last value / transitions):")
        for ident, nlist in extra:
            vals = [v for (_t, i, v) in changes if i == ident]
            label = nlist[0].split("[")[0].strip()
            last = vals[-1] if vals else "?"
            print(f"  {label:<46} last={last:<6} edges={len(vals)}")
        print()
        print("  interpreting the HLS stall flags (0 = stalled on that resource):")
        print("    ap_ext_blocking_n -> external memory (m_axi / DDR read)")
        print("    ap_str_blocking_n -> a stream (the AXIS write)")
        print("    ap_int_blocking_n -> intra-kernel dataflow")
        print("  m_axi_gmem0_RVALID with many edges = DDR reads ARE flowing;")
        print("  frozen with the kernel non-idle = stalled waiting on memory.")
        print()

    print("VERDICT:")
    if sv == 0:
        print("  PL never asserted TVALID -> the kernel did not produce.")
        print("  Check ap_start/ap_idle above: if ap_start=1 and ap_done=0 it is")
        print("  running but stuck — use the ap_*_blocking_n flags and the")
        print("  sub-loop ap_done signals to see which stage it is sitting in.")
        print("  This is NOT an AIE-side fault.")
    elif ar == 0:
        print("  TVALID asserted but TREADY never did -> the AIE side never accepted")
        print("  a single word. Backpressure is permanent: the shim/TLM adapter is")
        print("  not draining. Suspect PLIO<->shim config (port width, channel,")
        print("  placement) or the AIE core not being started before the PL pushes.")
    elif beats_aie >= expected:
        print(f"  All {expected} beats completed on the AIE boundary — the PLIO link")
        print("  itself is fine. If the host still hangs, the fault is downstream")
        print("  (kernel loop count, GMIO drain, or the host's wait).")
    else:
        print(f"  Partial transfer: {beats_aie}/{expected} beats, then stalled.")
        print("  Mid-stream backpressure — check the AIE kernel's consumption rate")
        print("  and whether it exited its loop early.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
