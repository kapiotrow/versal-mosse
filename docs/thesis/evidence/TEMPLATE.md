# <Claim in one line, stating the VERDICT, not the topic>

**Status:** current · **Updated:** — · **Scope:** the skeleton every new evidence note starts from

**<date>.** `<run dir or offline script>`, <N sequences / M frames>. <What differs from the
comparison arm, and how that was verified — a flagstamp diff, an ELF `cmp`, an xclbin md5.>
Claim id: `<X-NN>` in `docs/thesis/claims.md`.

## The prediction, written down first

<What was expected, and — the load-bearing part — the falsifier: what result would mean the
mechanism is not what you think. Written BEFORE the run. If it was not written before the run,
say so here; that is a weaker note and the reader should know.>

## The result

| arm | A | R | EAO | frames |
|---|---|---|---|---|
| | | | | |

<One paragraph. Lead with the number that decides it.>

## Did the mechanism hold?

<Check each falsifier by name and say FIRED or HELD. An arm can ship while its mechanism story
fails — say so plainly and do not build on the explanation.>

## Controls

<What makes this a measurement and not a coincidence: the known-good comparator, the negative
control, the mutant that proves the test can fail. A suite with no failing mutant is worth
nothing until one has been shown.>

## What not to re-derive

<The corrections, the dead ends, the arithmetic that was wrong the first time. This section is
why the note exists a year later.>
