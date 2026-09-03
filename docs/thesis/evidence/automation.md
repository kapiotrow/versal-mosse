# Board automation — ssh control, pushed ELF, one-command sweeps

**Status:** closed · **Updated:** 2026-08-25 · **Scope:** board automation: ssh control, a pushed ELF, one-command sweeps

**2026-08-25.** Every board session used to open with four typed lines and one
hand-typed ELF invocation per sequence, and every arm change was a card swap.
Two of the coast A/B's three defects came from that flow rather than from the
tracker. This is what replaced it, and what was verified rather than assumed.

## The rootfs was already 90% of the way there

Read out of `build/rootfs/rootfs_compat.ext4` with `debugfs`, not assumed:

| | |
|---|---|
| `/usr/sbin/sshd`, `/usr/bin/{ssh,scp,ssh-keygen}` | present — **openssh**, not dropbear |
| `sockets.target.wants/sshd.socket` | **enabled** — sshd listens from boot |
| `/etc/shadow` → `root:*` | password login **locked** |
| `sshd_config` | no `PermitRootLogin` ⇒ default `prohibit-password` ⇒ **keys work** |
| `/etc/systemd/network` | **empty** — nothing gives `end0` an address |

So sshd was already running and the board was unreachable for exactly two
missing files. `scripts/board_provision.sh` writes them into the image with
`debugfs` — no root, no loop device, the same offset trick `fix_sd_rootfs.sh`
uses — and it takes either a bare `rootfs.ext4` (before packaging) or an
existing `sd_card.img` (partition 2, geometry read from the image's own MBR).

`make rootfs` now calls it. `BOARD_KEY=none` opts out **explicitly**; there is no
silent skip when the key is missing, because a rootfs quietly built without it
boots unreachable and that is indistinguishable from a cable fault at the moment
it costs the most.

**The trap this cost:** `debugfs`'s `mkdir` **allocates the inode before**
discovering the directory already exists, so running the script twice left an
unconnected inode and an fs `e2fsck` called dirty. The read-back check passed
both times — only the closing `e2fsck -fn` caught it. Test-then-create now, and
the fsck stays.

## `scripts/vot_sweep.sh` — one command per sweep

```bash
scripts/vot_sweep.sh --arm coast0 --seqs car1,tiger,nature --ingest
scripts/vot_sweep.sh --arm subbin1 --seqs @runs/vot/seqs8.txt --resume
scripts/vot_sweep.sh --arm x --seqs car1 --dry-run     # prints, runs nothing
```

Mounts (idempotently), pushes, guards, runs each sequence with its own log,
collects the `track_<seq>.csv` files, and optionally ingests. Three of its checks
exist because the manual flow already failed at that exact point:

- **It refuses to start when the arm's results directory is non-empty.** Arm B
  overwriting arm A would be silent — same filenames, a successful write message,
  nothing in either log. `--resume` continues an interrupted sweep and skips
  sequences that already have trajectories.
- **It compares the board's `a.xclbin` against the PC's packaged one and refuses
  on a mismatch.** The ELF is *meant* to differ from the card's; the bitstream is
  not.
- **It records the flagstamps, ELF/xclbin/weights md5s and the git SHA beside the
  results.** `runs/.last_cfg` once recorded a configuration the run did not
  execute.

It also mounts the results export at `/mnt/vot-results` and puts the arm in a
directory *inside* it. Mounting the export **at** `.../coast0` is what put 54
trajectories in the export root.

## The ELF is pushed, not flashed

163 KB over `scp`. Every host-only knob — `HOLD_COAST`, `PSR_GATE_MIN`,
`PROGRESS_EVERY`, sub-bin interpolation — leaves the xclbin untouched, so an arm
change is a copy, not a card swap and a reboot. The run happens in `/tmp/mosse`
with `layer0_weights.bin` and `xrt.ini` pushed alongside and `a.xclbin`
symlinked from the card:

- **weights** because they carry a layout tag the host checks at runtime, and an
  RGB file under a grayscale ELF must fail loudly (it already did once);
- **`xrt.ini`** because XRT reads it from the process's **CWD** — a run from a
  directory without it silently loses `Runtime.rw_shared`, and an ini key that is
  not read looks exactly like an ini key that had no effect.

## SSH CHANGES THE MEASUREMENT — read this before quoting a frame time

Launching over ssh moves the ELF's stdout off the 115200 serial console. That
console is itself a distortion — 15% of the frame at `VERBOSITY=0` and **58% on
`animal`**, where `correlation(gated%, unattributed) = 0.963` — so ssh gives
*more* honest frame times, but:

- they are **not comparable** to any run recorded before today, including
  `run_0821_1725` (26.29 ms), and
- `ts` on the PC side of a TCP stream is good to about a second. It locates a
  stall; it is **not** the frame-time instrument `picocom … | ts` was.

Frame time comes from the run's own `AP_*` slots and `track.csv`, which have been
the authority since 2026-08-24 anyway. If a quotable FPS number is needed, take
it from one serial-console run under the old conditions and say so.

One incidental gain: `ssh` without a pty emits clean `\n`. picocom's bare `\r` is
what made `readlines()` and `grep` disagree about line numbers during the coast
A/B analysis.

## Status

`board_provision.sh` is tested on both image types, including idempotency and the
fsck. `vot_sweep.sh` has been exercised end to end with `--dry-run` and every
guard confirmed to fire. **Neither has run against the board yet** — the first
real use is the sub-bin interpolation A/B, and the first thing to check is that
the card was flashed from a provisioned image (`ssh root@192.168.10.2 uname -a`).
