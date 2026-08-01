# Shift budget sweep

- scenario: `s6`  geometry: 64x64  N_CHANNELS: 1  CONV2D_MODE: 0
- total budget held at 12: `IFFT_COL = 12 - 2*FFT_SHIFT`, IFFT_ROW = 0
- generated 2026-07-31T18:09:25+02:00 by `scripts/sweep_shift.sh`

| FFT_SHIFT | IFFT_COL | first_sat_ch | accum max {re,im} | proj @ch16 | resp nz | resp max | peak err | accum_sat | resp_sat | OVERALL |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | none | `{-631,7079}` | 113264 RAILS | 3927/4096 | 418 | 0 px | OK | OK | PASS |
| 3 | 6 | none | `{-160,-1769}` | 28304 (86%) | 3925/4096 | 417 | 0 px | OK | OK | PASS |
| 4 | 4 | none | `{-41,-442}` | 7072 (21%) | 4029/4096 | 418 | 1 px | OK | OK | PASS |
| 5 | 2 | none | `{-11,-111}` | 1776 (5%) | 4090/4096 | 1148 | 36 px | OK | OK | FAIL |

## Decision rule

1. `first_sat_ch` must be `none` — zero rails at EVERY channel, not just the
   final state. `HEADROOM EXCEEDED` and `accum_sat` are different checks: a
   channel can rail and be pulled back off the rail by a later one, which the
   final-state scan misses.
2. `resp_sat` OK and the response not crushed (`resp nz` high, `resp max` >> 1).
   Rails mean the col shift is too low; `nz = 0` means it is too high.
3. `peak err` = 0 px.
4. Among survivors take the SMALLEST FFT_SHIFT — every extra bit of forward
   shift is spectrum precision thrown away permanently.

`proj @ch16` scales the measured worst component to 16 channels. Exact for this
harness (identical channels ⇒ linear accumulator), and the coherent worst case.
Confirm the winner with a real ch16 run before shipping it.
