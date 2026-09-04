"""VOT toolkit tracker integration for Danilowicz & Kryjak's deepDCF.

THE KNOWN-ANSWER CHECK for `offline_multistart.py`'s `deepdcf:` backend: run
their tracker under THEIR published protocol (VOT2015 supervised) and check it
lands near THEIR published table before any STb2022 number from that backend is
quoted. `evidence/deepdcf_reproduction.md` carries the pre-registered band and
the falsifiers; the short version is that a port which is subtly wrong yields a
plausible trajectory and an uninterpretable score, and `opencv-kcf` is this
project's standing example of exactly that (R 0.280 against a published 0.532).

THIS FILE IS OURS, NOT THEIRS, and that is the point. Their own
`vot_integration.py` does the same job but hardcodes
`/home/vision/danilowi/CF_tracking/MOSSE_fpga/configs/config.json`, an absolute
path on the author's machine. Writing the glue here keeps `external/deep_mosse`
byte-identical to the published commit -- see `offline_multistart.py`'s DeepDCF
block for why an unmodified checkout is the whole value of the comparison.

`import vot` below MUST resolve to THEIR `vot.py`, the TraX stub. That is the
opposite of what `offline_multistart.py` needs, where the same file shadowing
the installed toolkit package is a bug -- but this runs in the toolkit's own
tracker subprocess, which has no use for the toolkit package.

Driven by trackers.ini; `DEEPDCF_PRESET` and `DEEPDCF_ROOT` come from its env_*.

@thesis sec:porownanieReferencyjne | R-15 | their published tracker under their
published protocol, as the known-answer check for running it under ours
"""
import json
import os
import sys

import cv2

import vot                                    # THEIR TraX stub -- see above
from deep_mosse import DeepMosse
from offline_multistart import DEEP_PRESETS   # ONE definition of the presets

preset = os.environ.get("DEEPDCF_PRESET", "best")
root = os.environ.get("DEEPDCF_ROOT")
if root is None:
    raise SystemExit("set DEEPDCF_ROOT to the pinned deepDCF checkout")
if preset not in DEEP_PRESETS:
    raise SystemExit(f"unknown preset '{preset}'; expected {sorted(DEEP_PRESETS)}")

with open(os.path.join(root, "configs", "config.json")) as fh:
    config = json.load(fh)
config.update(DEEP_PRESETS[preset])
# The 4-bit checkpoint is not in their repo, so every arm here is the FLOAT
# software model. Their Table 1 rows are all 4-bit: this is a BAND check
# against them, never a reproduction. evidence/deepdcf_reproduction.md.
config["deep"], config["quantized"] = True, False

# SINGLE OBJECT, which is what VOT2015 is and what this tracker does.
#
# THIS NEEDS A PATCHED TOOLKIT, and the patch is to the TOOLKIT, never to their
# tracker: vot-toolkit's legacy TraX path is broken at both ends for a
# single-object experiment. `_get_initialization` hands the runtime a bare
# ObjectStatus -- a (region, properties) 2-tuple -- while the runtime guards
# with `if len(new) != 1: raise`, so 2 != 1 and initialization always fails
# with "Tracker does not support multiple objects". Declaring multi-object
# routes around that guard and then dies at the other end, in the supervised
# experiment, with "'list' object has no attribute 'properties'". Identical in
# 0.8.1 and 0.9.0 -- the legacy path was left behind by the multi-object
# refactor, and VOT2015 stacks are legacy.
#
# scripts/patches/vot-toolkit-0.9.0-single-object-trax.patch fixes it, applied
# ONLY to the isolated ~/vot/venv-vot09. The main ./.venv keeps vot-toolkit
# 0.8.1 untouched because every row of results/arms.csv was scored with it.
# The patch touches the tracker RUNTIME, never the analysis.
handle = vot.VOT("rectangle")
selection = handle.region()
selection = [selection.x, selection.y, selection.width, selection.height]

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# cv2.imread gives BGR, which is what their tracker expects and what their own
# integration feeds it. NO FLIP HERE -- offline_multistart's backend flips
# because the toolkit hands IT RGB. `opencv-kcf-rgb` is the standing control
# proving this is not free to get wrong.
tracker = DeepMosse(cv2.imread(imagefile), selection, config)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    region = tracker.track(cv2.imread(imagefile))
    handle.report(vot.Rectangle(*[float(v) for v in region]))
