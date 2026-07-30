#!/bin/bash
# fix_sd_rootfs.sh — repair the ext4 root partition inside a v++-generated sd_card.img
#
# Why this exists
# ---------------
# `v++ --package --package.image_format=ext4` copies $COMMON_IMAGE_VERSAL/rootfs.ext4
# into partition 2 of sd_card.img and injects the boot files.  The tool it uses to do
# that injection does not understand the newer ext4 features present in the 2025.2
# rootfs (orphan_file / metadata_csum_seed), so it writes the partition without
# maintaining the journal.  The result passes a superblock sanity check but the target
# kernel refuses it at boot:
#
#     EXT4-fs (mmcblk0p2): Could not load journal inode
#     Kernel panic - not syncing: VFS: Unable to mount root fs on "/dev/mmcblk0p2"
#
# The pristine rootfs.ext4 is clean (verify with `e2fsck -fn`), so the damage is
# introduced by packaging and must be repaired after every `v++ --package` run.
#
# What it does
# ------------
# Extracts partition 2, runs e2fsck -fy (which regenerates the journal), and writes
# it back in place.  Requires no root: everything is done with file offsets, not
# loop devices.
#
# Usage:  scripts/fix_sd_rootfs.sh <path/to/sd_card.img>

set -euo pipefail

IMG="${1:-}"
if [[ -z "$IMG" || ! -f "$IMG" ]]; then
    echo "ERROR: usage: $0 <sd_card.img>" >&2
    exit 1
fi

command -v e2fsck >/dev/null || { echo "ERROR: e2fsck not found (install e2fsprogs)" >&2; exit 1; }

# Partition 2 geometry, in 512-byte sectors, straight from the image's own MBR.
read -r START SECTORS < <(
    partx -g -o START,SECTORS -n 2 "$IMG" 2>/dev/null | awk '{print $1, $2}'
)
if [[ -z "${START:-}" || -z "${SECTORS:-}" ]]; then
    echo "ERROR: could not read partition 2 geometry from $IMG" >&2
    exit 1
fi

OFF=$((START * 512))
LEN=$((SECTORS * 512))
TMP="$(mktemp --tmpdir sd_rootfs_XXXXXX.ext4)"
trap 'rm -f "$TMP"' EXIT

echo "[fix_sd_rootfs] $IMG partition 2: offset=$OFF bytes, size=$LEN bytes"

# Fast path: if the filesystem already passes, leave the image untouched.
if e2fsck -fn "$IMG?offset=$OFF" >/dev/null 2>&1; then
    echo "[fix_sd_rootfs] rootfs already clean — nothing to do"
    exit 0
fi

echo "[fix_sd_rootfs] rootfs is damaged; extracting and repairing..."
dd if="$IMG" of="$TMP" bs=1M skip=$((OFF / 1048576)) count=$((LEN / 1048576)) status=none

# e2fsck returns 0 = clean, 1 = errors corrected, 2 = corrected + reboot advised.
# Anything >= 4 means it could not fix the filesystem.
set +e
e2fsck -fy "$TMP"
RC=$?
set -e
if (( RC >= 4 )); then
    echo "[fix_sd_rootfs] ERROR: e2fsck could not repair the rootfs (rc=$RC)" >&2
    exit 1
fi

# Confirm the repair actually took before touching the image.
if ! e2fsck -fn "$TMP" >/dev/null 2>&1; then
    echo "[fix_sd_rootfs] ERROR: rootfs still reports errors after repair" >&2
    exit 1
fi

dd if="$TMP" of="$IMG" bs=1M seek=$((OFF / 1048576)) conv=notrunc status=none
sync
echo "[fix_sd_rootfs] rootfs repaired and written back — image is bootable"
