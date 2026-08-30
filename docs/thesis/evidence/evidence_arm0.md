# Multi-anchor evidence run — arm A (`HOLD_COAST=0`), 8 sequences

**2026-08-25, `runs/run_0825_1604.log`. 54 runs, 23,297 frames, 54 trajectories
written, 0 failures.** This is the first multi-sequence tracking result the
project has, and the baseline arm B is measured against.

```
sequence   runs  frames  meanIoU   op%    cerr   gated%  longest  budget
nature       14   10604   0.1535  10.8   328.7     0.17       11      60
car1         15    8434   0.5105  63.6   178.0    15.70      281       4
tiger         7    2011   0.1293   6.6    82.1     9.05       29       4
book          4     576   0.0431   0.8   161.3    55.56      174        3
crabs1        4     531   0.0959  10.3   328.2    26.37       29        3
ball3         4     564   0.0182   0.6   119.0    69.15      170        0
soccer2       3     326   0.0107   1.0   222.8    29.14        5        0
animal        3     251   0.0139   0.7   205.1    76.10       50        0

frame-weighted mean IoU over all 54 runs: 0.2699
```

## `car1` is an outlier, not a representative

Every result quoted in this project until today came from `car1` or from a
synthetic scene. `car1` scores 0.5105 here; **the other seven sequences score
0.011 to 0.154.** The synthetic scene's 0.9188 and `car1`'s 0.5 are not
"the tracker's accuracy" — they are the two easiest points in the set.

## The two failure modes split cleanly, and `gated%` is the discriminator

| regime | sequences | gated% | what fails |
|---|---|---|---|
| gate-dominated | `ball3` 69, `animal` 76, `book` 56, `soccer2` 29, `crabs1` 26 | high | the target outruns the frozen window; longest holds 50-174 frames |
| gate-silent | `nature` 0.17 | ~zero | sub-bin lag; the gate never fires and PSR RISES as IoU falls ([subbin_lag.md](subbin_lag.md)) |
| mixed | `car1` 15.7, `tiger` 9.1 | middling | both |

The hold-budget analysis in [hold_policy.md](hold_policy.md) predicted exactly
which sequences would be gate-dominated, and it is right: the four sequences with
budget 0-3 are the four with the highest hold rates. It simply does not describe
`nature`, whose budget of 60 is real and whose failure has another cause
entirely. **Both analyses are needed; neither is general.**

That also means arm B can only act on part of the set. Predict: `nature`'s
digests IDENTICAL between arms (no holds, so the coast cannot fire), and the
measurable effect concentrated on `ball3` / `animal` / `book`.

## Frame time: compute is flat, the console is not

| | ms/frame | unattributed | gated% |
|---|---|---|---|
| `nature` | 26.87 | 2.11 | 0.17 |
| `car1` | 28.48 | 4.98 | 15.70 |
| `tiger` | 28.18 | 2.58 | 9.05 |
| `crabs1` | 28.46 | 5.45 | 26.37 |
| `soccer2` | 34.80 | 12.07 | 29.14 |
| `ball3` | 38.99 | 18.92 | 69.15 |
| `book` | 41.22 | 20.46 | 55.56 |
| `animal` | 47.19 | 27.55 | 76.10 |

**correlation(gated%, unattributed) = 0.963.** The compute cost is ~27-28 ms on
every sequence; the 35-47 ms readings are the two HOLD lines that `gate_track()`
prints at EVERY verbosity, ~160 characters at 115200 baud on the majority of
frames. That is the design working as intended — "the tracker stopped updating"
is never noise — but it means **frame times from gate-heavy sequences are not
comparable and must not be quoted as FPS.** It also raises the priority of
Phase 4: on `animal`, 58% of the frame is console.

## The trajectories went to the wrong directory

The board reported 54 successful writes to `/mnt/vot-results/coast0/...` and all
54 files are on the PC — in **`/srv/vot/results/`**, not `/srv/vot/results/coast0/`,
which is empty. The board's NFS mount is at `/mnt/vot-results/coast0`, so
`--vot-results /mnt/vot-results/coast0` resolved to the export root. Nothing was
lost, and the arm separation did not happen.

**Arm B will silently overwrite arm A** unless the files are moved first:

```bash
sudo mkdir -p /srv/vot/results/coast0
sudo mv /srv/vot/results/*.txt /srv/vot/results/*.value /srv/vot/results/coast0/
```

Then either remount on the board at `/mnt/vot-results` and pass
`--vot-results /mnt/vot-results/coast1`, or leave the mount alone and move arm
B's files to `coast1/` afterwards the same way. **The overwrite would have been
silent** — same filenames, a successful write message, and an arm-A result
replaced by an arm-B one with nothing in either log to say so.

## Caveats

- `car1`'s 15 anchors are 36% of the frames and `nature`'s 14 are 46%, so the
  frame-weighted 0.2699 is mostly those two. The unweighted mean of the eight
  per-sequence means is 0.12.
- `tiger` scores IoU 0.129 with a centre error of only 82 px, about one box
  width. That is not obviously either failure mode and has not been diagnosed.
- These are not AR numbers. Ingesting the 54 trajectories into a toolkit
  workspace would produce the project's first real AR figures on a subset, at no
  further board time.
