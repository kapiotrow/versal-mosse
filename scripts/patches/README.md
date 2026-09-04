# Patches against pinned third-party checkouts

**Status:** current · **Updated:** 2026-09-04 · **Scope:** the policy for modifying vendored-by-reference third-party code, and the patches that exist

**THIS DIRECTORY IS EMPTY OF PATCHES, AND THAT IS THE RESULT.**

`external/deep_mosse` is a pinned clone of Danilowicz & Kryjak's published
implementation (`github.com/mdanilow/MOSSE_fpga` @ `ee0f93ab`, branch
`deep_features`, MIT), used as a reference tracker on this project's own
benchmark — claim `R-15`, `docs/thesis/evidence/deepdcf_reproduction.md`. The
comparison is worth something precisely because the algorithm is *theirs*, so
the checkout runs unmodified and every adaptation lives on our side of the line:

- the harness backend is `scripts/offline_multistart.py` (`deepdcf:<preset>`),
- the VOT toolkit glue is `scripts/deepdcf_vot2015.py`, written here rather than
  reusing their `vot_integration.py`, which hardcodes an absolute config path
  from the author's machine.

Three things that looked like they would need a patch and did not:
`torchvision` still accepts the removed `pretrained=True` through `**kwargs`;
`brevitas` installs without moving torch, so their module-scope FINN imports
resolve; and their hardcoded `cuda:0` works as-is on this machine.

**If a patch ever becomes necessary**, it belongs here as a reviewable `.patch`
file applied by the fetch step — never as an edit in place. A reader must be
able to see, in one diff, that nothing touching the algorithm moved. State the
patch in the evidence note too: an unstated modification turns a reference
result into an unattributable one.
