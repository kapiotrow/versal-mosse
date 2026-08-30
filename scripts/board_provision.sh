#!/bin/bash
# board_provision.sh — make the packaged rootfs reachable over ssh at boot.
#
# Why this exists
# ---------------
# Every board session used to start with the same four lines typed into picocom:
# `ip addr add`, two `mount -t nfs`, and then one ELF invocation per sequence.
# None of that is interesting and all of it is a place to mistype a flag — the
# coast A/B was nearly lost to an arm that ran with the wrong build flags, and an
# 8-sequence sweep is 8 chances to make the same class of mistake.
#
# The packaged rootfs already contains everything needed to drive the board from
# the PC instead. Verified by reading the image, not assumed:
#
#     /usr/sbin/sshd, /usr/bin/{ssh,scp,ssh-keygen}     openssh, present
#     sockets.target.wants/sshd.socket                  ENABLED at boot
#     /etc/shadow  root:*                               password login LOCKED
#     sshd_config  no PermitRootLogin                   default prohibit-password
#     /etc/systemd/network                              EMPTY
#
# So sshd already listens; what is missing is an address on `end0` and a key for
# root. Both are files. This script writes them into the image with `debugfs`,
# which needs no root and no loop device — the same offset-based approach
# fix_sd_rootfs.sh uses.
#
# Root's password is `*`, i.e. locked. That is not a limitation to work around:
# key-only auth on a point-to-point cable is the right posture, and it is why
# this script refuses to run without a public key.
#
# Usage
# -----
#   scripts/board_provision.sh build/rootfs/rootfs_compat.ext4     # before packaging
#   scripts/board_provision.sh build/hw/.../package/sd_card.img    # an existing card
#
#   --key FILE     public key to install       (default ~/.ssh/id_ed25519.pub)
#   --ip  CIDR     static address for the NIC   (default 192.168.10.2/24)
#   --iface NAME   interface                    (default end0)
#
# `end0`, not `eth0`: the Versal PS GEM enumerates as end0 (phase0a.md), and a
# .network file matching nothing is a silent no-op — the board would boot with no
# address and the failure would look like a cable problem.

set -euo pipefail

IMG=""
KEY="$HOME/.ssh/id_ed25519.pub"
ADDR="192.168.10.2/24"
IFACE="end0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)   KEY="$2";   shift 2 ;;
        --ip)    ADDR="$2";  shift 2 ;;
        --iface) IFACE="$2"; shift 2 ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *)       IMG="$1";   shift ;;
    esac
done

[[ -n "$IMG" && -f "$IMG" ]] || { echo "ERROR: usage: $0 <rootfs.ext4|sd_card.img> [--key F] [--ip CIDR] [--iface N]" >&2; exit 1; }
[[ -f "$KEY" ]] || { echo "ERROR: public key not found: $KEY  (ssh-keygen -t ed25519)" >&2; exit 1; }
grep -q '^ssh-' "$KEY" || { echo "ERROR: $KEY does not look like a PUBLIC key (no ssh-* prefix)" >&2; exit 1; }
command -v debugfs >/dev/null || { echo "ERROR: debugfs not found (install e2fsprogs)" >&2; exit 1; }

# --- locate the filesystem -------------------------------------------------
# An sd_card.img carries the rootfs in partition 2; a bare .ext4 is the
# filesystem itself. Read the geometry from the image's own MBR rather than
# assuming an offset.
FS="$IMG"
if partx -g -o START,SECTORS -n 2 "$IMG" >/dev/null 2>&1; then
    read -r START _ < <(partx -g -o START,SECTORS -n 2 "$IMG" | awk '{print $1, $2}')
    if [[ -n "${START:-}" ]]; then
        FS="$IMG?offset=$((START * 512))"
        echo "[provision] $IMG is partitioned; using partition 2 at byte $((START * 512))"
    fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$KEY" "$TMP/authorized_keys"
cat > "$TMP/10-board.network" <<NET
# Written by scripts/board_provision.sh — static, point-to-point, no DHCP.
# The PC side is the other half of this link; see docs/thesis/evidence/phase0a.md.
[Match]
Name=$IFACE

[Network]
Address=$ADDR
LinkLocalAddressing=no
IPv6AcceptRA=no
NET

d() { debugfs -w -R "$*" "$FS" 2>&1 | grep -viE '^debugfs [0-9]|^$' || true; }

echo "[provision] installing $(basename "$KEY") for root, $ADDR on $IFACE"

# Idempotent — but NOT by letting mkdir fail. `debugfs`'s mkdir ALLOCATES the
# inode before it discovers the directory already exists, so a second run leaves
# an unconnected inode and an fs that e2fsck calls dirty. Test first, then create.
# (Found by running this script twice; the read-back check passed both times and
# only the closing fsck caught it — which is the argument for having it.)
if ! debugfs -R "stat /root/.ssh" "$FS" 2>/dev/null | grep -q "Inode:"; then
    d "mkdir /root/.ssh"
fi
d "rm /root/.ssh/authorized_keys"          | grep -v 'File not found' || true
d "write $TMP/authorized_keys /root/.ssh/authorized_keys" >/dev/null
d "sif /root/.ssh mode 040700"
d "sif /root/.ssh uid 0"; d "sif /root/.ssh gid 0"
d "sif /root/.ssh/authorized_keys mode 0100600"
d "sif /root/.ssh/authorized_keys uid 0"; d "sif /root/.ssh/authorized_keys gid 0"

d "rm /etc/systemd/network/10-board.network" | grep -v 'File not found' || true
d "write $TMP/10-board.network /etc/systemd/network/10-board.network" >/dev/null
d "sif /etc/systemd/network/10-board.network mode 0100644"

# --- verify by reading it back ---------------------------------------------
# A write that silently did nothing looks exactly like a write that worked, and
# the symptom would be a board that boots unreachable — indistinguishable from a
# cable fault at the point where it costs the most to diagnose.
fail=0
got_key="$(debugfs -R "cat /root/.ssh/authorized_keys" "$FS" 2>/dev/null || true)"
if [[ "$got_key" != "$(cat "$KEY")" ]]; then
    echo "  FAIL: authorized_keys does not read back as $KEY"; fail=1
fi
if ! debugfs -R "cat /etc/systemd/network/10-board.network" "$FS" 2>/dev/null | grep -q "Address=$ADDR"; then
    echo "  FAIL: 10-board.network does not carry Address=$ADDR"; fail=1
fi
modes="$(debugfs -R "ls -l /root/.ssh" "$FS" 2>/dev/null | tr -s ' ')"
grep -q '100600 .* authorized_keys' <<<"$modes" || { echo "  FAIL: authorized_keys is not mode 600 — sshd will refuse it"; fail=1; }
grep -qE '^ +[0-9]+ +40700 ' <<<"$modes"        || { echo "  FAIL: /root/.ssh is not mode 700 — sshd will refuse it"; fail=1; }

if [[ $fail -ne 0 ]]; then
    echo "[provision] FAILED — image left as-is, do not flash it" >&2
    exit 1
fi

echo "[provision] verified: key mode 600 in a 700 dir, $ADDR on $IFACE"
echo "[provision] filesystem check:"
e2fsck -fn "$FS" >/dev/null 2>&1 && echo "  clean" || { echo "  DIRTY — run scripts/fix_sd_rootfs.sh before flashing" >&2; exit 1; }
echo
echo "Next: flash the image, boot, then from the PC:"
echo "  ssh -o StrictHostKeyChecking=accept-new root@${ADDR%%/*} uname -a"
