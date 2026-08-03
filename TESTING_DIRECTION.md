# Testing direction — what actually improves accuracy

Branch `testing-direction`, opened 2026-08-04. The goal of every experiment here is
**accuracy on the target datasets**, not a better description of the problem.

---

## Where we actually stand

Four label-free results have ever produced a significant gain over CARPRT:

| what | gain |
|---|---|
| truncate CARPRT to its own top-3, uniform weights | eurosat **+4.27** ✱✱✱ |
| count-power score, α=0.5 | eurosat **+4.15** ✱✱✱, pets +0.76 ✱, imagenet +0.19 ✱✱ |
| **use less unlabeled data** | eurosat **+1.64**, pets +0.76 |
| pseudo-label W, top-10 uniform | flowers **+1.10** ✱✱ |

Three independent mechanisms — hard truncation, count amplification, and starving the
estimator of data so cells go empty — and all three are the same lever: **concentration**.
Nothing else in six method attempts moved accuracy at all. Concentration is the only
thing that has ever worked.

Its *sign* is governed by how good the ranking already is:

| dataset | CARPRT top-3 overlap with oracle | gain from concentrating to top-3 |
|---|---|---|
| eurosat | 17% | **+4.27** |
| oxford_pets | 10% | −0.24 |
| oxford_flowers | 8% | −0.65 |
| dtd | 6% | −2.61 |
| ucf101 | 3% | −2.75 |

Above ~15% overlap concentration wins; below it loses. Right now we place that bet blind,
on every class of every dataset.

## The apparent contradiction, and its resolution

Two measured facts that look incompatible:

- Uniform weights over the **oracle's** top-3 give **+19.2** on eurosat.
- The `pseudo` selector raised overlap with the oracle's top-10 from 15% → **47%** on
  caltech101 and moved accuracy by **−0.16**.

They are compatible if the payoff is **convex in overlap**. Uniform weights over *k*
prompts put all the mass on *k* choices; a wrong choice is no longer diluted by the other
246, so a partially-right set pays the concentration cost without earning the
concentration reward. Under convexity there is no partial credit: nothing is worth
anything until the set is nearly right, and then everything arrives at once.

If that is the shape, "selection" is still the mechanism — the success criterion is just
"almost exactly right" rather than "closer than before". If the shape is linear instead,
the `pseudo` result means something else is wrong and the framing needs revisiting.

**This is the first thing to measure. Everything downstream depends on the answer.**

---

## Experiment 1 — the swap curve  ← RUNNING FIRST

`promptop/swap.py`, command `swapcurve`.

Build a path from CARPRT's top-*k* set to the oracle's top-*k* set, one prompt at a time:

```
S_c(j) = { oracle's top-j prompts for class c }
       ∪ { best CARPRT prompts not already chosen, until |S_c(j)| = k }
```

`j=0` is exactly CARPRT's top-*k*; `j=k` is exactly the oracle's top-*k*. Score every point
with uniform weights. A **random path** (j random prompts instead of j oracle prompts,
same fill rule) runs alongside to separate "swapped in the *oracle's* prompts" from
"perturbed the set at all".

Labels enter only through `w_oracle`, as a ruler. Nothing on this path is a method.

**Numbers the run reports**

| statistic | meaning |
|---|---|
| `acc[j]` | accuracy at each point on the path |
| `gain frac` | `(acc[j] − acc[0]) / (acc[k] − acc[0])` — linear ⇒ `j/k` |
| `half-gain point` | smallest `j/k` reaching 50% of the gain. Linear ⇒ 0.50, convex ⇒ > 0.50 |
| **`j_breakeven`** | **smallest `j` where the set beats full 247-prompt CARPRT** |
| `overlap[j]` | realized overlap with the oracle's set, since the CARPRT fill can pick oracle prompts by luck |

`j_breakeven` is the actionable one: **how many of k prompts a selector must get exactly
right just to break even with the incumbent.**

**Confirms convexity if:** half-gain point ≥ 0.7 on a majority of datasets, and
`j_breakeven ≥ k/2`. Then the `select` null result is fully explained, the bar for a
selector is quantified, and Experiment 3 is the right next move.

**Refutes it if:** the curve is roughly linear (half-gain point ≈ 0.5) and `j_breakeven`
is small. Then improving overlap *should* have paid, `pseudo` failed for some other
reason, and the selection framing needs re-examining before anything is built on it.

**Also possible:** the curve is flat or non-monotone. That would mean overlap does not
cause accuracy at all and the top-k analysis is measuring an artifact of the full-fit
oracle. Worst case, and worth knowing in one run.

**Cross-check (Experiment 1b, `--with-pseudo`).** Read `pseudo`'s measured overlap off
the curve and compare the predicted accuracy to what `pseudo` actually scored. If they
agree, convexity explains the anomaly completely. If `pseudo` lands *below* the curve, its
non-oracle picks are actively harmful — a different and worse problem.

### RESULT — oxford_pets, 2026-08-04 (1 of 8 datasets; laptop has no others)

```
dataset           k   CARPRT   top-k  oracle-k    span  half-gain  gain@half  breakeven
oxford_pets k=3   3    89.45   89.21     93.81   +4.61       0.67       0.82        1/3
oxford_pets k=6   6    89.45   88.44     93.38   +4.93       0.50       0.69        1/6
oxford_pets k=10 10    89.45   88.44     92.72   +4.28       0.40       0.63       2/10
```

**Convexity is refuted. The curve is linear and monotone** (mean half-gain point 0.52;
linear = 0.50). Overlap *does* convert to accuracy, proportionally.

**The break-even bar is low.** Getting **1 of 3** prompts exactly right — with the other two
being CARPRT's own best — already beats full 247-prompt CARPRT. Mean break-even is 0.23 of
the set. This is far cheaper than expected and it means a selector does not have to be
nearly right to be worth using.

**So why did `pseudo` fail?** The overlay answers it. Placed on the curve at its *own*
measured overlap:

| k | pseudo overlap | pseudo acc | curve says | **residual** |
|---|---|---|---|---|
| 3 | 23% | 88.14 | 89.61 | **−1.47** |
| 6 | 32% | 89.29 | 90.13 | **−0.84** |
| 10 | 40% | 88.96 | 89.88 | **−0.92** |

`pseudo` lands ~1 point **below** the curve at matched overlap, at every k. It is not that
overlap fails to pay — it is that **`pseudo`'s non-oracle picks are worse than the CARPRT
prompts the curve falls back on.** It buys overlap and pays more than that back on its
misses.

**The asymmetry that makes this happen.** From the random path at k=3: swapping in one
*oracle* prompt gains **+0.92**; swapping in one *random* prompt costs **−4.20**. A wrong
pick costs ~4.6× what a right pick gains, because at k=3 a single prompt carries a third of
the mass. The penalty shrinks with k (at k=10 the ratio is ~1.2:1) — concentration risk is
`1/k`.

**Design rules this hands us, all measured:**

1. **Never replace a CARPRT prompt without strong evidence.** Its ranking is a good
   *fallback* even where it is a bad *ranking* — the curve's fillers are CARPRT's, and they
   are what make the low break-even possible. `pseudo` discarded them wholesale.
2. **Prefer larger k.** At k=10 a wrong pick is bounded, and the set still delivers +3.27.
   Small k maximises both upside and ruin.
3. The target is a selector that **overrides selectively**, not one that re-ranks
   everything. That is Experiment 3.

**Caveat.** One dataset, and the one with the *smallest* real headroom (+3.1 held-out
vs the +4.61 full-fit span used here) — so the top of this curve is partly memorisation.
Needs the other seven before any of the above is load-bearing.

---

## Experiment 2 — the native-prompt control

CLIP ships a per-dataset prompt list (24 satellite templates for EuroSAT, 6 for DTD, 45
action templates for UCF101). The 247-template pool was built by **pooling those lists**,
so ~90% of it is off-domain for any given dataset.

Score each dataset with (a) its own native CLIP prompt list, uniform, and (b) the full 247
pool, uniform (= MPE), against CARPRT.

Zero labels, essentially free, and never run in this project. If native prompts alone put
EuroSAT near 60, then the **+4.27** is rediscovering domain matching and not reweighting at
all — which changes what the paper can claim. Must be known before building on top of it.

---

## Experiment 3 — adaptive concentration (the method candidate)

Conditional on Experiment 1. Concentration is the only lever that works and its sign
depends on ranking quality, which varies **per class**, not just per dataset. Every method
so far concentrates uniformly everywhere: CARPRT spreads over all 247 for every class,
top-k concentrates for every class.

The proposal is a per-class decision of *how much* to concentrate, driven by a label-free
estimate of how trustworthy that class's ranking is. This is not magnitude estimation —
the magnitudes stay uniform inside the chosen set. It is a choice of set *size*.

Design waits on the shape of the swap curve, which determines what a per-class confidence
signal has to achieve to be worth anything.

---

## Ground rules

- Report the **unweighted mean across all datasets at one fixed configuration.** The
  incumbent's failure mode is per-dataset wins masking a negative mean: the best label-free
  selector so far is `+0.49` when method and `k` are chosen per dataset and **−1.02** when
  they are fixed globally.
- No method may use labels. Oracles are rulers only.
- McNemar against CARPRT, with the discordant count printed.
- Every claim in `FINDINGS.md` gets the measurement that supports it, or gets deleted.
  ("Existing methods select at chance" is already wrong — chance overlap for top-3 of 247
  is 1.2%, and CARPRT scores 2.2–13.7× that.)
