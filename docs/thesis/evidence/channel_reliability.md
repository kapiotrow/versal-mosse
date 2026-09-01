# Stage B3 channel reliability — THE MECHANISM IS REAL AND THE GAIN IS NOT SEPARABLE FROM A NULL

**2026-09-01.** `runs/vot/0901_offline-chrel/chrel62.json`, `scripts/rgb_vs_gray_loop.py` arm
suffix `-chrel<N>`, 62 sequences / 19,903 frames, shipping eta 0.05 / gate 5.0, scored with
`vot_ar_offline.py`. Offline only — no hardware. **Control: the `rgb` arm reproduces the stored
board-form baseline 0.5394 / 0.2910 / 5792 exactly**, so the arms are readable.

CSR-DCF's third contribution and the one `baselines.md` ranked "untested, host-only, and now the
cheapest live candidate" (their ablation: −12%). Stage B3 weights each channel by
`1/sqrt(sum|F|^2)` — by how LOUD it is, never by how well it discriminates.

## The statistic, and why it would have been free

Per channel, with `P = F .* conj(H)`:

```
r_c(0)    = (1/N) SUM_k P_c(k)              response AT the target
||r_c||^2 = (1/N) SUM_k |P_c(k)|^2          total response energy      (Parseval)
rho_c     = |SUM P|^2 / (N * SUM |P|^2)     fraction of energy on target
```

Both are single accumulations over a spectrum the host loop already walks — one complex sum and
one magnitude-squared sum per bin, **no inverse FFT** (the direct measurement would cost the
30.1 ms/channel inverse the `FILTER_MASK_STAT` note prices). `rho` is SCALE-INVARIANT in `H`, so
weighting by `rho^gamma` does not change `rho` — no circularity. Weights are normalised to mean
1 so the arm does not also move the global response scale and drag the gate with it.
`gamma = 0` is exactly today's behaviour, i.e. **the null is built in**.

## The prediction, written down first

Board time only on **dR >= +0.02 with trim-3 surviving, AND the anti-reliability mutant losing.**
`-chrelneg` (`gamma = -1`) weights the UNRELIABLE channels up; if it does not lose, the statistic
is inert and any gain is "perturbing the channel weights", not reliability.

## The result

```
arm                 A        R    tracked   pooled dR   per-seq dR   trim-3    trim-5   b/w/tied
rgb            0.5394   0.2910       5792                                                (control)
chrel05  g=0.5 0.4940   0.3120       6209     +0.0210     +0.0257   +0.0023   -0.0060   15/ 8/39
chrel10  g=1.0 0.5127   0.2903       5777     -0.0007     -0.0115   -0.0228   -0.0264   14/10/38
chrel20  g=2.0 0.5331   0.2331       4639     -0.0579     -0.0492   -0.0685   -0.0743   15/18/29
chrelneg g=-1  0.5018   0.2514       5004     -0.0396     -0.0118   -0.0398   -0.0503   19/17/26
```

**A mild weighting is worth +0.0210 pooled and nothing after a symmetric trim (+0.0023), with 39
of 62 sequences EXACTLY TIED. REJECTED against the pre-registered bar.**

## Did the mechanism hold?

**HELD — and this is the part worth keeping.** The anti-reliability mutant LOSES, and decisively:
−0.0396 pooled, −0.0398 after trim-3. **The reliability statistic therefore carries real
information** — inverting it costs 0.04 R — which is more than the spatial mask's mechanism check
could show (`baselines.md`: the mask's `NEGATIVE_PEAK` story INVERTED when split by on-target).

What the sweep says is that the information is there and the *lever* is short: `gamma = 0.5`
gains a little, `gamma = 1` is a null, `gamma = 2` is a disaster (−0.058, worse than
anti-weighting), and every arm loses accuracy on the common survived prefix (−0.0046 to −0.0143).
**You can destroy this tracker by over-concentrating the channel weights and you cannot gain much
by mildly concentrating them.** The 16-channel bank's usable width is 1.43 in activation space
(`feature_bank.md`), so there is little diversity to reweight in the first place — which is the
most likely reason CSR-DCF's −12% does not reproduce here.

## What to do

**No board time.** The proxy reads +0.021 pooled / +0.002 trimmed; nothing that small has ever
transferred, and its two over-predictions (`pad30` +0.088 -> +0.0077, the gate +0.056 -> +0.0063)
were both on knobs acting through the filter/veto path rather than the response shape.

**One legitimate re-open, and it must be labelled as such rather than sold as a rescue:** this
was screened against the sigma-2 baseline, and `MOSSE_SIGMA=4.0` is now the best arm on record
(EAO 0.1931) with a materially different response shape — median accepted PSR 30.9 -> 18.6. The
per-channel `rho` distribution is a function of that shape, so the screen does not transfer by
assumption. Re-screening `-chrel05` on top of the sigma-4 operating point is one offline run.
