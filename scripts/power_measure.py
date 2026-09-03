#!/usr/bin/env python3
"""
scripts/power_measure.py -- energy per frame, by difference.

@thesis subsec:metrykiSystemowe | P-12 | Drives the three-phase power protocol and does the
  arithmetic: the board's marginal power is a DIFFERENCE against an idle baseline, never a
  wattmeter reading quoted as the design's cost.

WHAT THIS PAYS
--------------
`subsec:metrykiSystemowe` promises energy per frame as the metric that makes embedded
comparisons fair, `sec:wydajnoscZasoby` lists `pobor mocy`, and `docs/thesis/results/`
measures neither. claims.md calls this out as the one debt with no claim behind it.

WHY A DIFFERENCE AND NOT A READING
----------------------------------
The design uses 2% of the AIE array (6 of 304 cores, 1 of 76 memory tiles, 3.4% DSP) and
the frame is 84% CPU-bound. A VEK280 is a development board: fans, PHYs, the System
Controller itself and regulator loss dominate anything this design draws. A board-total
wattage would therefore be a true number that answers the wrong question, and would rank
two arms by how warm the room was.

So the protocol is three states and the answer is the difference between them:

  static  device idle, xclbin NOT loaded        -> board floor
  graph   xclbin loaded, graph free-running,    -> STATIC cost of the design: the
          no frames being processed                bitstream and the AIE array powered
  run     frames processing                     -> + the DYNAMIC cost of the workload

  P_design = P_graph - P_static      what the design costs to be resident
  P_work   = P_run   - P_graph       what processing a frame costs
  J/frame  = P_work * frame_time

`graph` is the tracker's `--power-pause <s>`: it holds with the graph up and no frame in
flight, TWICE -- once before frame 0 and once after the last frame. The second window is a
CONTROL, not a duplicate: two equal readings mean the board was thermally settled and the run
delta is the workload; a higher second one means the board was still warming and the delta is
contaminated. Without --power-pause the run reports `graph` as NOT MEASURED and falls back to
P_run - P_static, which conflates resident cost with per-frame work. NOT MEASURED is an empty
cell, never a zero -- FILTER_MASK_STAT's `-1` convention, for the same reason.

PHASE BOUNDARIES COME FROM THE TRACKER'S OWN MARKERS, NOT FROM A DURATION
------------------------------------------------------------------------
The tracker prints `[power] PHASE <name> BEGIN|END` and this script stamps those lines as
they ARRIVE. It must, because staging a VOT sequence reads up to 1.27 GB over NFS before
frame 0: a window timed from the start of the run command would price that transfer as the
design's resident power. The run log is streamed for the same reason.

TWO INSTRUMENTS, NOT ONE INSTRUMENT TWICE
-----------------------------------------
Frame time is taken two independent ways and both are printed: the tracker's own
`[apu] CUMULATIVE ... mean frame body`, and the run phase's wall duration divided by the
frames it reported. They measure different things (frame body excludes some per-run
overhead) so they are not expected to agree exactly -- but a large gap means the phase
boundaries do not bracket the work, which is the failure this cross-check is here to catch.

WHAT THIS CANNOT DO
-------------------
Every available sensor is an I2C/ADC read at a few Hz, averaging over ~100 frames at
24 ms/frame. This measures energy per frame over a sustained run. It CANNOT attribute
power to a pipeline stage, and no sampling rate available here would let it.

Transport is a confound exactly as it is for FPS: the ethernet PHY and ssh's CPU share are
inside the board total, and CLAUDE.md already restricts quotable FPS to serial-console
runs for this reason. Record which transport the run used and compare like with like.

--power-pause IS HOST-ONLY. It reaches no flagstamp, so a power build is an scp and not a
card swap. The minimum is 5 s, enforced in the tracker: the sampler runs at a few Hz, and a
mean over two samples prints exactly like a mean over sixty.

Usage
-----
  # Full protocol against the System Controller, wrapping one board run:
  scripts/power_measure.py --sampler root@<sc-host> --backend sc_app \\
      --run-cmd './board_run.sh 900 a.xclbin --vot-seq car1 --power-pause 60' \\
      --arm l1relu --out runs/power/0903_l1relu

  # No instrument yet: characterise the die-temperature channel on the board itself.
  scripts/power_measure.py --sampler root@192.168.10.2 --backend sysmon \\
      --run-cmd '...' --arm l1relu --out runs/power/probe
"""

import argparse
import base64
import csv
import datetime
import hashlib
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE = ROOT / "scripts" / "power_probe.sh"

# The tracker's own summary line, which is also board_run.sh's completion marker.
RE_CUM = re.compile(r"\[apu\] CUMULATIVE over (\d+) frame\(s\), mean frame body ([\d.]+) ms")

PHASES = ("static", "graph", "run", "graph_post", "tail")

# The tracker's own phase markers (mosse_tracker.cpp, power_pause_hold()).
RE_PHASE = re.compile(r"\[power\] PHASE (\w+) (BEGIN|END)")


class Sampler:
    """Runs power_probe.sh on the sampling host and collects timestamped rows.

    Rows are stamped with the LOCAL arrival time, not the remote clock. The board's
    clock was unset when this was written (it read 2025-05-29 against the PC's
    2026-09-03) and the SC's is a separate clock again, so aligning phases on a remote
    timestamp would silently mis-bin every sample. Arrival jitter over ssh is tens of
    milliseconds against phases of tens of seconds. The remote stamp is kept anyway and
    its drift reported, so a stalled sampler is visible rather than inferred.
    """

    def __init__(self, target, backend, period_ms):
        self.target, self.backend, self.period_ms = target, backend, period_ms
        self.rows = []              # (t_local, channel, unit, value_or_nan)
        self.meta = []              # header/comment lines from the probe
        self.remote_skew = []       # (t_local, t_remote) for drift reporting
        self.proc = None
        self.serial = None
        self._stop_serial = False
        self.err = []
        self._lock = threading.Lock()

    def start(self):
        # "serial:/dev/ttyUSB3" reaches a console; anything else is an ssh target.
        # The System Controller -- the ONLY source of watts on this board -- has no
        # network presence, so its transport is a UART and the probe has to be carried
        # there over the console rather than piped in over ssh.
        if self.target.startswith("serial:"):
            return self._start_serial(self.target.split(":", 1)[1])
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=10", self.target,
               "sh -s -- --backend %s --period-ms %d" % (self.backend, self.period_ms)]
        self.proc = subprocess.Popen(
            cmd, stdin=PROBE.open("rb"), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read(self, lines=None):
        t0 = time.monotonic()
        for line in (lines if lines is not None else self.proc.stdout):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                with self._lock:
                    self.meta.append(line)
                continue
            if line.startswith("t_remote,"):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            t_rem, chan, unit, val = parts
            try:
                v = float(val)
            except ValueError:
                v = math.nan
            now = time.monotonic() - t0
            with self._lock:
                self.rows.append((now, chan, unit, v))
                try:
                    self.remote_skew.append((now, float(t_rem)))
                except ValueError:
                    pass

    def _start_serial(self, dev):
        """Log into a serial console, carry power_probe.sh across, and run it.

        THE PROBE IS UPLOADED, NOT REIMPLEMENTED. Driving `sc_app` command-by-command
        from here would be a second copy of the sampling loop, and the two would drift.

        TWO TRAPS, BOTH PAID FOR ON 2026-09-03, BOTH SPECIFIC TO CONSOLES:

        1. A CONSOLE ECHOES WHAT YOU TYPE, so a sentinel you grep for appears in the
           echo of the command that tests for it. `command -v base64 && echo B64_OK`
           "passed" on an SC that has no base64 at all, because the echoed command line
           contains B64_OK. Every sentinel below is therefore SPLIT (`"B64" "_OK"`), so
           the literal exists in the reply and never in the echo.
        2. THE SC HAS NO base64. The transfer is a quoted heredoc instead -- which is
           the natural encoding anyway, a shell script being typed at a shell -- paced in
           chunks because there is no flow control, and verified by md5 afterwards.

        The credential is read from SC_USER/SC_PASS in the environment and is never
        written to disk or into the run directory -- a console password in a thesis
        repo would outlive the board.
        """
        import serial                                   # pyserial, PC-side only
        user = os.environ.get("SC_USER", "root")
        pw = os.environ.get("SC_PASS")
        if pw is None:
            raise SystemExit(
                "power_measure: set SC_PASS (and SC_USER if not 'root') for the serial\n"
                "  console at %s. Keep it out of the shell history:\n"
                "     read -rs SC_PASS && export SC_PASS SC_USER=<user>" % dev)

        ser = serial.Serial(dev, 115200, timeout=1)
        self.serial = ser

        def rd(seconds):
            end, buf = time.monotonic() + seconds, ""
            while time.monotonic() < end:
                n = ser.in_waiting
                if n:
                    buf += ser.read(n).decode("utf-8", "replace")
                else:
                    time.sleep(0.03)
            return buf.replace("\r", "")

        def snd(line, wait=1.5):
            ser.reset_input_buffer()
            ser.write((line + "\n").encode())
            ser.flush()
            return rd(wait)

        # Clear anything half-typed, then log in only if the console asks us to.
        ser.write(b"\x03\n")
        time.sleep(0.8)
        ser.reset_input_buffer()
        out = snd("", 2.0)
        if "login:" in out:
            out = snd(user, 2.0)
            if "assword" in out:
                out = snd(pw, 3.0)
            if "incorrect" in out.lower() or "login:" in out:
                ser.close()
                raise SystemExit("power_measure: serial login refused on %s as '%s' -- "
                                 "check SC_USER/SC_PASS." % (dev, user))

        # Split sentinel: the reply contains SHELL_OK, the echoed command never does.
        if "SHELL_OK" not in snd('echo SHELL"_OK"', 2.0):
            ser.close()
            raise SystemExit("power_measure: no shell prompt on %s after login." % dev)

        # ---- transfer, as a quoted heredoc -------------------------------------
        text = PROBE.read_text()
        if "PROBE_EOF" in text:
            ser.close()
            raise SystemExit("power_measure: probe contains the heredoc delimiter.")
        # SENT AS EXACT BYTES, not re-joined lines. Chunking by line and re-adding "\n"
        # per chunk appended one byte to a file that already ended in a newline, and the
        # md5 guard caught it -- which is the whole reason the guard is here. A heredoc
        # passes the payload through verbatim, so the only safe thing to do is not touch it.
        payload = text if text.endswith("\n") else text + "\n"
        snd("rm -f /tmp/power_probe.sh", 1.0)
        ser.write(b"cat > /tmp/power_probe.sh <<'PROBE_EOF'\n")
        ser.flush()
        time.sleep(0.3)
        data = payload.encode()
        for i in range(0, len(data), 512):
            ser.write(data[i:i + 512])
            ser.flush()
            time.sleep(0.15)          # no flow control on this link; let sh keep up
        ser.write(b"PROBE_EOF\n")
        ser.flush()
        rd(2.0)

        # md5 BOTH ENDS, over the bytes actually SENT. A half-landed transfer would
        # otherwise present as "zero channels discovered", which reads like a board
        # problem rather than a truncated file. Split sentinel, as above.
        want = hashlib.md5(payload.encode()).hexdigest()
        got = snd("md5sum /tmp/power_probe.sh 2>/dev/null || echo NO"+'"MD5"', 4.0)
        if want not in got:
            ser.close()
            raise SystemExit(
                "power_measure: probe upload to %s is INCOMPLETE or CORRUPT.\n"
                "  expected md5 %s\n  console said: %s\n"
                "  Refusing to sample from a script that is not the one in the repo."
                % (dev, want, got.strip()[-200:]))

        ser.write(("sh /tmp/power_probe.sh --backend %s --period-ms %d\n"
                   % (self.backend, self.period_ms)).encode())
        ser.flush()

        def lines_iter():
            buf = ""
            while not self._stop_serial:
                try:
                    n = ser.in_waiting
                    chunk = ser.read(n).decode("utf-8", "replace") if n else ""
                except Exception:
                    break
                if not chunk:
                    time.sleep(0.05)
                    continue
                buf += chunk.replace("\r", "")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line

        threading.Thread(target=self._read, args=(lines_iter(),), daemon=True).start()

    def _read_err(self):
        for line in self.proc.stderr:
            with self._lock:
                self.err.append(line.rstrip())

    def wait_for_first_sample(self, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.rows:
                    return True
                # A dead ssh transport is a definite failure; the serial one has no
                # process to poll, so it can only time out.
                if self.proc is not None and self.proc.poll() is not None:
                    return False
            time.sleep(0.2)
        return False

    def stop(self):
        if getattr(self, "serial", None):
            self._stop_serial = True
            time.sleep(0.3)
            try:
                self.serial.write(b"\x03")      # Ctrl-C: stop the sampling loop
                self.serial.flush()
                time.sleep(0.3)
                self.serial.close()
            except Exception:
                pass
            return
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def snapshot(self):
        with self._lock:
            return list(self.rows)


def _lsb(vals):
    """Smallest positive gap between distinct readings -- the effective ADC step."""
    u = sorted(set(vals))
    steps = [round(b - a, 9) for a, b in zip(u, u[1:]) if b > a]
    return min(steps) if steps else 0.0


# Below this many distinct readings in a window, the sample sd no longer describes the
# instrument and only a quantization bound can be honestly quoted.
MIN_LEVELS = 5


def neff(x):
    """Effective sample size under autocorrelation (Bartlett), summed to the first
    non-positive lag.

    THIS IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN A RESULT AND A NULL. These
    are time series from a slow physical process: consecutive samples half a second
    apart are strongly correlated, so `n` is not the number of independent
    observations. Measured on the 2026-09-03 thermal run, a 383-sample run phase had
    n_eff = 31 and a 112-sample hold had n_eff = 6.

    Getting this wrong is available in both directions. Judging a difference of means
    against the single-sample sd (this script's first version) called a 3.6-sigma
    effect NOT RESOLVED; judging it against the naive sem would have called the same
    effect 7.7 sigma. Both are wrong, and they are wrong in opposite directions.
    """
    n = len(x)
    if n < 8:
        return float(n)
    m = statistics.fmean(x)
    var = sum((v - m) ** 2 for v in x) / n
    if var <= 0:
        return float(n)
    tot = 0.0
    for k in range(1, n // 4):
        r = sum((x[i] - m) * (x[i + k] - m) for i in range(n - k)) / ((n - k) * var)
        if r <= 0:
            break
        tot += r
    return n / (1.0 + 2.0 * tot)


def resolved(a, b):
    """(delta, standard error, is-it-resolved) between two phase stats dicts.

    The criterion is 2 standard errors of the DIFFERENCE OF MEANS, with each phase's
    n_eff standing in for its n -- not two single-sample standard deviations.
    """
    if not a or not b or not a["n"] or not b["n"]:
        return None, None, False
    se = ((a["sd"] ** 2) / max(a["neff"], 1.0) + (b["sd"] ** 2) / max(b["neff"], 1.0)) ** 0.5
    # Floor at the quantization noise of the mean (uniform step q has sd q/sqrt(12)).
    # Averaging can legitimately beat one LSB when the signal DITHERS across levels --
    # VCCINT spans 13 levels and its 0.122 W step supports a 0.017 W standard error --
    # but it cannot when the rail sits on three.
    qfloor = max((x["q"] / (12.0 * max(x["neff"], 1.0)) ** 0.5) for x in (a, b))
    se = max(se, qfloor)
    d = a["mean"] - b["mean"]
    if min(a["levels"], b["levels"]) < MIN_LEVELS:
        # Quantization-limited: the only defensible statement is one step.
        q = max(a["q"], b["q"])
        return d, max(se, q), abs(d) > 2 * max(se, q)
    return d, se, (abs(d) > 2 * se if se > 0 else True)


def summarise(rows, window, settle_s):
    """Per-channel stats over [t0 + settle, t1] of one phase window."""
    t0, t1 = window
    lo = t0 + settle_s
    out = {}
    for t, chan, unit, v in rows:
        if lo <= t <= t1:
            out.setdefault(chan, {"unit": unit, "vals": [], "nan": 0})
            if math.isnan(v):
                out[chan]["nan"] += 1
            else:
                out[chan]["vals"].append(v)
    stats = {}
    for chan, d in out.items():
        vals = d["vals"]
        stats[chan] = {
            "unit": d["unit"],
            "n": len(vals),
            "nan": d["nan"],
            "mean": statistics.fmean(vals) if vals else math.nan,
            # Population sd of the phase: this is the instrument's noise floor, and it
            # is what decides whether a delta is resolved at all.
            "sd": statistics.pstdev(vals) if len(vals) > 1 else math.nan,
            # Autocorrelation-corrected: what the error bars must be built from.
            "neff": neff(vals) if len(vals) > 1 else float(len(vals)),
            # QUANTIZATION. These rails are read through very different ADC steps --
            # measured 2026-09-03: VCC_PSFP resolves 149 distinct values while VCC_PMC
            # resolves THREE. On a 3-level rail the sample sd is near zero because
            # almost every sample lands on one level, the standard error collapses with
            # it, and any difference at all then clears "2 s.e." That is how a null rail
            # reported CONTROL FAILED at +-0.000 W. Carry the step and the level count
            # so the gate can refuse to over-resolve.
            "levels": len(set(vals)),
            "q": _lsb(vals),
        }
    return stats


def report(rows, windows, outdir, args, frames, frame_ms_log, settle=None):
    """Statistics, the verdict, and summary.csv. Shared by a live run and --reanalyse."""
    settle_s = settle_s if settle is None else settle
    # --- per-phase statistics ----------------------------------------------
    stats = {p: summarise(rows, w_, settle_s) for p, w_ in windows.items()}
    channels = sorted({c for s in stats.values() for c in s})
    watt_ch = [c for c in channels
               if stats.get("static", {}).get(c, {}).get("unit") == "W"]

    print("\n=== per-phase means (settle %.0f s discarded from each) ===" % settle_s)
    hdr = "%-22s %-5s" % ("channel", "unit") + "".join("%14s" % p for p in PHASES if p in stats)
    print(hdr)
    for c in channels:
        unit = next((stats[p][c]["unit"] for p in stats if c in stats[p]), "")
        line = "%-22s %-5s" % (c, unit)
        for p in PHASES:
            if p not in stats:
                continue
            s = stats[p].get(c)
            line += "%14s" % ("--" if not s or not s["n"] else "%.6g" % s["mean"])
        print(line)

    print("\n=== sample counts / nan (a channel that stops parsing shows here) ===")
    for p in PHASES:
        if p not in stats:
            continue
        ns = [stats[p][c]["n"] for c in stats[p]]
        nn = sum(stats[p][c]["nan"] for c in stats[p])
        print("  %-8s window %6.1f-%6.1f s   n/channel %s   nan %d"
              % (p, windows[p][0], windows[p][1],
                 ("%d" % ns[0]) if ns and len(set(ns)) == 1 else str(ns), nn))

    # --- the answer ---------------------------------------------------------
    print("\n=== energy ===")
    if not watt_ch:
        print("  The '%s' backend reports no channel in watts, so NO POWER NUMBER IS\n"
              "  PRODUCED. This run characterises the instrument and the thermal\n"
              "  channel only. Watts need the sc_app backend on the System Controller."
              % args.backend)
    frame_s = None
    if args.frame_ms:
        frame_s = args.frame_ms / 1000.0
        print("  frame time %.2f ms (given on the command line)" % args.frame_ms)
    elif frame_ms_log:
        frame_s = frame_ms_log / 1000.0
        # Two independent instruments, printed side by side on purpose.
        wall = windows["run"][1] - windows["run"][0]
        print("  frame time %.2f ms (tracker's mean frame body over %d frames)"
              % (frame_ms_log, frames))
        print("  frame time %.2f ms (run-phase wall %.1f s / %d frames) -- these measure "
              "different spans; a large gap means the phase does not bracket the work"
              % (1000.0 * wall / frames, wall, frames))

    summary = []
    for c in watt_ch:
        st = stats.get("static", {}).get(c)
        gr = stats.get("graph", {}).get(c)
        rn = stats.get("run", {}).get(c)
        tl = stats.get("tail", {}).get(c)
        if not st or not rn or not st["n"] or not rn["n"]:
            continue
        noise = max(st["sd"] or 0.0, rn["sd"] or 0.0)
        # RESOLUTION GATE, against 2 standard errors of the DIFFERENCE OF MEANS with
        # autocorrelation-corrected n (see neff()). A delta inside that is not a small
        # delta, it is no reading, and printing "0.03 W" for it would be the
        # confident-wrong-number failure this repo has paid for elsewhere.
        p_design, se_design, res_design = resolved(gr, st) if gr else (None, None, False)
        base = gr if (gr and gr["n"]) else st
        p_work, se_work, res_work = resolved(rn, base)
        is_res = res_work
        row = {
            "channel": c, "arm": args.arm,
            "p_static_w": st["mean"], "p_graph_w": gr["mean"] if gr and gr["n"] else "",
            "p_run_w": rn["mean"],
            "p_graph_post_w": (stats.get("graph_post", {}).get(c, {}) or {}).get("mean", ""),
            "p_tail_w": tl["mean"] if tl and tl["n"] else "",
            "noise_w": noise,
            "p_design_w": p_design if p_design is not None else "",
            "se_design_w": se_design if se_design is not None else "",
            "p_work_w": p_work, "se_work_w": se_work, "resolved": int(is_res),
            "frame_ms": (frame_s * 1000.0) if frame_s else "",
            "j_per_frame": (p_work * frame_s) if (frame_s and is_res) else "",
        }
        summary.append(row)

        print("  %-18s static %7.3f W   run %7.3f W   delta %+7.3f +/- %.3f W (1 s.e., "
              "n_eff %.0f/%.0f)"
              % (c, st["mean"], rn["mean"], p_work, se_work or 0.0,
                 base["neff"], rn["neff"]))
        if p_design is not None:
            print("      design resident %+7.3f +/- %.3f W (graph - static)   %s"
                  % (p_design, se_design or 0.0,
                     "RESOLVED" if res_design else "not resolved"))
        if not is_res:
            print("      NOT RESOLVED: |delta| is within 2 s.e. This is a BOUND, not a "
                  "measurement -- report it as < %.3f W." % (2 * (se_work or 0.0)))
        elif frame_s:
            print("      %.3f mJ/frame at %.2f ms/frame"
                  % (1000.0 * p_work * frame_s, frame_s * 1000.0))
        # CONTROL 1: the two graph windows are the same board state. A gap between
        # them is the board still warming, and it contaminates p_work by that much.
        gp = stats.get("graph_post", {}).get(c)
        for other, ref, name, what in ((gp, gr, "graph_post vs graph", "die"),
                                       (tl, st, "tail vs static", "board")):
            if not other or not ref or not other["n"] or not ref["n"]:
                continue
            d, se, bad = resolved(other, ref)
            if bad:
                print("      CONTROL FAILED (%s): %+.3f +/- %.3f W. Same %s state, "
                      "different power -- it was still warming, so p_work carries that "
                      "drift as well as the workload." % (name, d, se, what))
            else:
                # AN UNDERPOWERED CONTROL IS NOT A PASSED CONTROL. A 60 s hold sampled at
                # 2 Hz had n_eff = 6 on the first thermal run, and could not have resolved
                # a drift a third the size of the signal. Print the bound it actually
                # achieved so a wide one is visible instead of reassuring.
                print("      control OK (%s): %+.3f +/- %.3f W, i.e. any drift is under "
                      "%.3f W. n_eff %.0f/%.0f -- widen the holds if that bound is not "
                      "small against p_work." % (name, d, se, 2 * se,
                                                 ref["neff"], other["neff"]))

    if summary:
        with (outdir / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for r in summary:
                w.writerow(r)

    (outdir / "meta.txt").write_text(
        "date=%s\narm=%s\nsampler=%s\nbackend=%s\nboard=%s\nperiod_ms=%d\n"
        "static_s=%s\npause_s=%s\ntail_s=%s\nsettle_s=%s\nrun_cmd=%s\n"
        "transport=ssh\ngraph_phase=%s\n"
        % (datetime.date.today(), args.arm, args.sampler, args.backend, args.board,
           args.period_ms, args.static_s, args.pause_s, args.tail_s, settle_s,
           args.run_cmd or "", "measured" if "graph" in windows else "NOT MEASURED"))

    if args.run_cmd and "graph" not in windows:
        print("\nCAVEAT: the `graph` phase was NOT MEASURED, so the delta above is "
              "run - static and\n  conflates the design's resident cost with the "
              "per-frame work. Put --power-pause N\n  on the run command to separate them.")
    written = ["samples.csv", "probe_meta.txt", "meta.txt"]
    if summary:
        written.append("summary.csv")
    if args.run_cmd:
        written.append("run.log")
    print("\nwrote %s/{%s}" % (outdir, ",".join(sorted(written))))

    # Provenance: results/power.csv rows must carry the run's flagstamps, exactly as
    # arms.csv rows do. Copying config/ is vot_sweep.sh's job; say so rather than
    # letting an unstamped number look quotable.
    print("Before appending to docs/thesis/results/power.csv, copy the run's config/ "
          "(flagstamps) beside\n  this directory -- results/README.md's provenance rule.")
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampler", required=True,
                    help="ssh target running power_probe.sh (the System Controller for "
                         "the sc_app backend; the board for sysmon)")
    ap.add_argument("--backend", required=True, choices=("sc_app", "sysmon"))
    ap.add_argument("--run-cmd",
                    help="command executed on --board for the run phase. Omit to "
                         "characterise the idle baseline and the instrument alone.")
    ap.add_argument("--board", default="root@192.168.10.2",
                    help="ssh target the run command executes on (default %(default)s)")
    ap.add_argument("--arm", default="", help="arm name recorded with the result")
    ap.add_argument("--out", required=True, help="output directory")
    # A 22-minute board run must never be repeated to fix arithmetic. This replays a
    # finished run's samples.csv through the current statistics.
    ap.add_argument("--reanalyse", metavar="DIR",
                    help="recompute the report from an existing run directory and exit")
    ap.add_argument("--period-ms", type=int, default=500)
    ap.add_argument("--static-s", type=float, default=60.0,
                    help="idle baseline duration (default %(default)s)")
    # NOT an alignment input: the graph windows are found from the tracker's markers.
    # This only records what was asked for, so meta.txt says whether a missing `graph`
    # phase was intended or a failure.
    ap.add_argument("--pause-s", type=float, default=0.0,
                    help="the --power-pause value on the run command, recorded for "
                         "provenance. Boundaries come from the tracker's markers.")
    ap.add_argument("--tail-s", type=float, default=30.0,
                    help="idle period after the run: a return to baseline is the control "
                         "that the delta was the workload and not a warming board")
    ap.add_argument("--settle-s", type=float, default=5.0,
                    help="discarded from the head of every phase (regulator and sensor "
                         "averaging transients)")
    ap.add_argument("--frame-ms", type=float,
                    help="override frame time instead of parsing it from the run log")
    args = ap.parse_args()

    if args.reanalyse:
        d = pathlib.Path(args.reanalyse)
        rows, windows = [], {}
        with (d / "samples.csv").open() as fh:
            for r in csv.DictReader(fh):
                if not r["value"]:
                    continue
                t = float(r["t_s"])
                rows.append((t, r["channel"], r["unit"], float(r["value"])))
                if r["phase"] != "between":
                    a, b = windows.get(r["phase"], (t, t))
                    windows[r["phase"]] = (min(a, t), max(b, t))
        frames = frame_ms_log = None
        log = d / "run.log"
        if log.exists():
            m = RE_CUM.search(log.read_text())
            if m:
                frames, frame_ms_log = int(m.group(1)), float(m.group(2))
        # settle already applied when samples.csv was binned; do not discard twice
        report(rows, windows, d, args, frames, frame_ms_log, settle=0.0)
        return 0

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("power_measure: sampler=%s backend=%s period=%d ms" %
          (args.sampler, args.backend, args.period_ms))
    sampler = Sampler(args.sampler, args.backend, args.period_ms)
    sampler.start()
    if not sampler.wait_for_first_sample():
        sampler.stop()
        print("ERROR: no samples from the probe. Its stderr:", file=sys.stderr)
        for line in sampler.err:
            print("  " + line, file=sys.stderr)
        print("  Try:  ssh %s sh -s -- --backend %s --list < %s"
              % (args.sampler, args.backend, PROBE), file=sys.stderr)
        return 3

    windows = {}
    t_start = time.monotonic()

    def mark(phase, t0, t1):
        windows[phase] = (t0 - t_start, t1 - t_start)

    # --- static -------------------------------------------------------------
    print("phase static: %.0f s" % args.static_s)
    t0 = time.monotonic()
    time.sleep(args.static_s)
    mark("static", t0, time.monotonic())

    # --- graph + run --------------------------------------------------------
    # STREAMED, not captured at the end: the phase boundaries are the arrival times of
    # the tracker's own `[power] PHASE` markers, and a run that is only read after it
    # exits has no arrival times at all.
    frames = frame_ms_log = None
    runlog = ""
    if args.run_cmd:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               args.board, args.run_cmd]
        print("phase run: %s" % args.run_cmd)
        t_run0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines, marks = [], []
        for line in proc.stdout:
            lines.append(line)
            m = RE_PHASE.search(line)
            if m:
                marks.append((m.group(1), m.group(2), time.monotonic()))
                print("  marker: %s %s" % (m.group(1), m.group(2)))
        proc.wait()
        t_run1 = time.monotonic()
        runlog = "".join(lines)
        (outdir / "run.log").write_text(runlog)

        def between(name, edge):
            for n, e, t in marks:
                if n == name and e == edge:
                    return t
            return None

        g0, g1 = between("graph", "BEGIN"), between("graph", "END")
        p0, p1 = between("graph_post", "BEGIN"), between("graph_post", "END")

        if g0 and g1:
            mark("graph", g0, g1)
            # The workload is everything between the two holds. If the post hold is
            # missing (an older ELF, or a run that died), the run window closes at
            # process exit instead -- which is correct but includes teardown.
            mark("run", g1, p0 if p0 else t_run1)
        else:
            mark("run", t_run0, t_run1)
        if p0 and p1:
            mark("graph_post", p0, p1)

        if not marks:
            # An ELF without the knob prints nothing here, and so does one that has it
            # while --power-pause was left off the command line. Both look identical
            # from here, so say what to check rather than guessing which it was.
            print("NOTE: no '[power] PHASE' markers in the run log. The `graph` phase is "
                  "NOT MEASURED -- is --power-pause on the run command, and does the "
                  "board's ELF have the knob?")

        m = RE_CUM.search(runlog)
        if m:
            frames, frame_ms_log = int(m.group(1)), float(m.group(2))
        else:
            # board_run.sh treats this same marker as the completion signal, so its
            # absence means the run did not finish -- not merely that a regex missed.
            print("WARNING: no '[apu] CUMULATIVE' line in the run log. The run did not "
                  "complete; the run phase does not bracket a full workload.")
        if proc.returncode != 0:
            print("WARNING: run command exited %d" % proc.returncode)

        # --- tail -----------------------------------------------------------
        if args.tail_s > 0:
            print("phase tail: %.0f s" % args.tail_s)
            t0 = time.monotonic()
            time.sleep(args.tail_s)
            mark("tail", t0, time.monotonic())

    sampler.stop()
    rows = sampler.snapshot()

    # --- raw sample CSV -----------------------------------------------------
    def phase_of(t):
        for p, (a, b) in windows.items():
            if a <= t <= b:
                return p
        return "between"

    with (outdir / "samples.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "phase", "channel", "unit", "value"])
        for t, chan, unit, v in rows:
            w.writerow(["%.3f" % t, phase_of(t), chan, unit,
                        "" if math.isnan(v) else "%g" % v])
    (outdir / "probe_meta.txt").write_text("\n".join(sampler.meta) + "\n")

    report(rows, windows, outdir, args, frames, frame_ms_log)

    if sampler.remote_skew:
        first, last = sampler.remote_skew[0], sampler.remote_skew[-1]
        local_el = last[0] - first[0]
        remote_el = last[1] - first[1]
        print("  clock: local elapsed %.1f s, remote elapsed %.1f s (drift %+.1f s) -- "
              "phases are binned on LOCAL arrival" % (local_el, remote_el,
                                                      remote_el - local_el))

    return 0


if __name__ == "__main__":
    sys.exit(main())
