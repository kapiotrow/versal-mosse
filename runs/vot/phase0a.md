# VOT Phase 0a — NFS staging link, board ↔ PC

**2026-08-24. CLOSED. The TCP-preload fallback (S3) is retired.**

Feasibility spike for the VOT-STb2022 run: is there an NFS client in the packaged
rootfs, and what does a per-sequence blob read actually cost? The throughput figure
here is the number Phase 5 budgets its staging slot against.

## Result

| | measured |
|---|---|
| Kernel NFS client | `nfs`, `nfs4` in `/proc/filesystems` (also `nfsd`, unused) |
| `mount.nfs` | present, `-r-s--x--x root root 199360`, dated 2018 — one binary at both `/sbin` and `/usr/sbin` |
| Interface | `end0`, MAC `00:0a:35:1d:63:f1` |
| Link | 1000 Mb/s, full duplex, carrier up |
| **Board read throughput** | **117.2 MB/s** |
| PC disk (for contrast) | 452 MB/s cold, 547 MB/s warm |

`end0`, not `eth0` — do not hardcode `eth0` in any staging script.

## Physical layer

Direct point-to-point, no switch, no DHCP. Board on the **Versal PS GEM** RJ45, not the
System Controller port; confirmed by the Xilinx OUI `00:0a:35` on the interface the
Versal's own kernel enumerates, so it needs no trust in the silkscreen.

```
PC   192.168.10.1/24   enp3s0
board 192.168.10.2/24  end0
```

Console stays on the USB-UART. **Do not move it onto this link** — the frame-time
instrument is `picocom … | ts`, and sharing a wire with bulk staging would leak
staging cost into frame time as unattributed jitter.

`ufw` inactive on the PC, so no firewall rules were needed. Ping 0.238 ms avg, 0% loss.

## Commands that produced the numbers

PC:

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /srv/vot/data /srv/vot/results
sudo chown nobody:nogroup /srv/vot/results     # root_squash: board runs as root
# /etc/exports
#   /srv/vot/data     192.168.10.2(ro,sync,no_subtree_check)
#   /srv/vot/results  192.168.10.2(rw,sync,no_subtree_check)
sudo exportfs -ra
dd if=/dev/urandom of=/srv/vot/data/probe.bin bs=1M count=500
```

Board:

```bash
ip addr add 192.168.10.2/24 dev end0
mkdir -p /mnt/vot
mount -t nfs -o vers=3,nolock,ro,rsize=1048576,proto=tcp \
      192.168.10.1:/srv/vot/data /mnt/vot
sync; echo 3 > /proc/sys/vm/drop_caches
time dd if=/mnt/vot/probe.bin of=/dev/null bs=1M
#   500+0 records in / 500+0 records out
#   real 0m4.475s   ->  524288000 B / 4.475 s = 117.2 MB/s
```

`nolock` is load-bearing: a trimmed rootfs has no `rpc.statd`, and without it the mount
hangs rather than failing.

## Why 117.2 MB/s is believed

It sits in a band nothing else can produce: **10× too fast for a 100 Mb link, 5× too slow
for DRAM, and 94% of GbE's theoretical 125 MB/s** — normal efficiency after Ethernet/IP/TCP
framing. The timing alone discriminates all three hypotheses, so the planned `tx_bytes`
counter cross-check was not needed.

**The first attempt returned 547 / 452 MB/s and was discarded — those were the PC reading
its own file, i.e. PC page cache and PC disk, never the link.** Both exceed the wire's
physical ceiling, which is the only reason the error was caught immediately. Had a cached
read landed at a plausible 105 MB/s it would have gone into the Phase 5 budget unquestioned.
Two lessons, both already in `CLAUDE.md` and both re-earned here:

- **Sanity-check against a physical bound before recording a number.** >125 MB/s on GbE is
  not a surprising result, it is a wrong question.
- **Read the elapsed time, not the rate.** 500 MB in 0.96 s is obviously impossible;
  "547 MB/s" merely looks surprising. Busybox `dd` on this rootfs prints only record
  counts, so use `time` — or read the deltas off the already-timestamped console.

Board page cache, not PC page cache, was the culprit: 500 MB is nothing against 12 GB of
LPDDR4, and dropping the PC's cache does nothing when the request never reaches the PC.

## What this means for Phase 5

| | |
|---|---|
| `girl`, worst case, 461 MB | **3.9 s**, amortised over ~30 runs of that sequence |
| All 60 sequences, ~6.5 GB | **~55 s total staging** |
| Against tracking | 25–50 min ⇒ staging is **2–4%** of the run |

Wide margin, but still report the staging slot separately from the frame body the way the
`AP_*` slots already are — measure it, don't assume it amortises.

**Jumbo frames: retired, do not reopen.** At 94% efficiency there is ~6% of headroom, not
worth an MTU mismatch that would present as mysterious stalls rather than a clean failure.

**Do not `mmap` the blob over NFS.** It converts staging into demand paging, moving the I/O
inside the frame loop as page faults — which would surface as unattributed frame time
instead of an honest staging slot.

## Still open

- The **results** export (`rw`) has not been exercised. `chown nobody:nogroup` is in place
  against root-squash, but the first actual trajectory write is the test.
- Throughput was measured on a `/dev/urandom` probe, not a real converted blob. No reason
  to expect a difference — it is the same sequential read — but Phase 5's staging slot is
  the measurement of record.
