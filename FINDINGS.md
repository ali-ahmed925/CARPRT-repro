# Findings

Record of the `prompt-operator`, `bayes-reweight` and `testing-direction` branches.
Every number was measured. Written 2026-08-01, restructured 2026-08-04 after twelve
method attempts.

**Setup.** CLIP ViT-B/16 · 247 templates (Allingham et al. pool) · τ = 1.0 (1.5 for ImageNet,
per App. C.3) · micro-average top-1 · paired McNemar for all comparisons.

---

## STATE OF PLAY

**Twelve methods tried. None beats CARPRT on average.** Best label-free mean over eight
datasets is **−0.37** (`spread`), positive on 2/8. The oracle reaches **+8.25**, positive on
8/8.

What *is* established, and is not measured anywhere else in this literature:

- A per-class set of 3–10 prompts, uniformly weighted, reaches most of the oracle (§1).
- Accuracy is **linear** in overlap with the oracle's set, and the break-even bar is low (§2).
- That target is real and it **transfers** across disjoint image samples (§3).
- A selector needs **~40% precision**; the best label-free signal reaches **26%** (§4).
- Every pseudo-label-derived signal degrades **exactly where the headroom is** (§5).

---

## 1. The selection ceiling

Take the oracle's top-$k$ prompts per class, **discard its weights**, weight uniformly:

| dataset | CARPRT | oracle | top-k set + UNIFORM | k | recovered |
|---|---|---|---|---|---|
| eurosat | 55.01 | 75.95 | 74.23 | 3 | 92% |
| food101 | 85.85 | 89.01 | 88.57 | 6 | 86% |
| caltech101 | 94.52 | 98.86 | 98.09 | 3 | 82% |
| dtd | 49.00 | 68.44 | 64.42 | 6 | 79% |
| ucf101 | 69.97 | 86.47 | 82.98 | 3 | 79% |
| oxford_pets | 89.45 | 95.07 | 93.81 | 3 | 78% |
| oxford_flowers | 71.38 | 84.41 | 80.88 | 3 | 73% |
| fgvc | 24.60 | 40.14 | 35.10 | 10 | 68% |
| | | | | **mean** | **≈ 80%** |

**Two caveats that must travel with this number.** The sets come from the **full-fit** oracle,
scored on the images it was fitted to. And `k` is the post-hoc best **per dataset**, chosen
with labels — an oracle over `k` on top of an oracle over `W`. The 80% is a maximum, not an
operating point.

### Overlap between CARPRT's top-3 and the oracle's

| eurosat | pets | flowers | food101 | dtd | fgvc | ucf101 | caltech |
|---|---|---|---|---|---|---|---|
| 0.50 (17%) | 0.30 (10%) | 0.24 (8%) | 0.23 (8%) | 0.17 (6%) | 0.16 (5%) | 0.09 (3%) | 0.08 (3%) |

**Correction to an earlier version of this file:** these are *not* chance. Expected overlap
between two random size-3 subsets of 247 is 3²/247 = 0.036 prompts, **1.2% of k**. CARPRT
scores **2.2×–13.7× chance**. Its selection is weak, not random.

CARPRT's per-class Spearman against the oracle is 0.774, but that correlation is carried by
the *bottom* of the ranking: it knows which prompts are bad and has little idea which are
best. Forcing it onto its own top-$k$ makes accuracy worse on 6 of 8 datasets.

## 2. Accuracy is LINEAR in overlap (`swapcurve`, 8 datasets × k=3,6,10)

Path from CARPRT's top-$k$ to the oracle's top-$k$, one prompt at a time; non-oracle slots
filled with CARPRT's best remaining prompts.

- **Mean half-gain point 0.51** (linear = 0.50), ≥0.70 on **1 of 24** configs. Partial credit
  exists — a selector does not have to be nearly right. **Convexity was hypothesised and
  refuted.**
- **Mean break-even 0.18 of k.** At k=10, one forced oracle prompt beats full CARPRT on 6/8.
- **Concentration risk scales as 1/k.** At k=3 a right pick gains +0.92 and a random one
  costs −4.20 (4.6:1); at k=10 the ratio is ~1.2:1. **k=10 is the operating point.**
- **eurosat is non-monotone** at k=6 and k=10 (gain@half 1.76 and 1.66): 3 oracle prompts +
  3 CARPRT fillers ≈ 67.4 beats all 6 oracle prompts at 61.64. Its weights are so peaked
  that its #4–6 prompts are harmful at equal weight. eurosat k=3 is the one convex config
  in 24, and the only case where CARPRT's own top-k beats full CARPRT (+4.27).

**CARPRT's ranking is a good fallback even where it is a bad ranking.** The break-even is
only that cheap because the unfilled slots are CARPRT's. `pseudo` discarded them and landed
~1 point *below* the curve at matched overlap.

## 3. The target is real and it transfers (`stability`, oxford_pets only)

Two oracles fitted on disjoint halves of the images:

```
top-1 identical           18.9%   (chance 0.4%)      -> 47x chance
A's pick in B's ranking   88.1th percentile           median rank 15 of 247
top-10 overlap            41.6%   (chance 4.0%)      -> 10x chance
```

The second line matters most: even when the halves disagree on the exact prompt, A's pick
sits near the top of B's list. Disagreements are between near-equals.

Forcing A's picks into CARPRT's top-10 and scoring on B, where CARPRT is re-estimated from
B alone:

| | k=10 |
|---|---|
| CARPRT on B | 89.54 |
| **transfer (A→B)** | **91.34 (+1.80)** |
| in-sample (B→B) | 93.24 (+3.71) |

Generalisation costs about half. **Transfer is an upper bound on any label-free selector** —
it holds real labels, just not on the images it is scored on.

**But the out-of-sample bar is higher than §2 suggested:** transfer breaks even at **j=3**,
versus 2/10 in-sample. And at k=3 forcing in a single pick is **−1.20** before recovering.

*Only run on oxford_pets — the smallest-headroom dataset of the eight.*

## 4. The three label-free selectors, and the precision bar

| selector | signal | mean over 8 | positive | precision |
|---|---|---|---|---|
| `spread` | distribution shape; **no labels, no pseudo-labels** | **−0.37** | 2/8 | 5–26% |
| `crossfit` | consensus from half the prompt pool, scoring the other half | −0.57 | 1/8 | 5–13% |
| `margin` | margin against the class actually competing | −0.72 | 2/8 | 6–15% |
| ORACLE | — | +8.25 | 8/8 | 100% |

Per-dataset, best `j`, at k=10:

| dataset | CARPRT | crossfit | margin | **spread** | ORACLE |
|---|---|---|---|---|---|
| eurosat | 55.01 | −0.10 | −1.72 | **+1.91** | +11.16 |
| ucf101 | 69.97 | −1.69 | −1.72 | −0.48 | +12.32 |
| dtd | 49.00 | −0.77 | −0.47 | −1.00 | +14.42 |
| oxford_flowers | 71.38 | −0.04 | +0.45 | −1.18 | +8.61 |
| fgvc | 24.60 | −0.51 | −0.66 | −1.11 | +10.50 |
| caltech101 | 94.52 | −0.53 | −0.57 | −0.16 | +3.29 |
| food101 | 85.85 | +0.17 | +0.23 | +0.10 | +2.39 |
| oxford_pets | 89.45 | −1.09 | −1.31 | −1.06 | +3.30 |

**The break-even calculation.** From the random-swap path at k=10: a correct pick is worth
≈ **+1.2**, a wrong one costs ≈ **−0.8**. So

```
1.2p > 0.8(1 − p)   ->   p > 40%
```

**Required ~40%. Best available 26%.** That gap is why every selector loses despite carrying
real information. Precision against the oracle is also the *wrong* bar — a pick must beat the
CARPRT prompt it **displaces**, which is far stronger than random.

**Abstention (oxford_pets).** Overriding only the most confident classes, precision at the
top 10 / 25 / 50 / 100% of classes:

```
spread     50%   22%   17%   11%     <- monotone; top decile clears 40%
margin      0%    0%    6%   14%     <- ANTI-calibrated
crossfit    0%    0%    0%    8%     <- ANTI-calibrated
```

Only `spread` has a usable confidence signal. (10% of 37 classes is 4 classes, so 50% is
2/4 — pets cannot settle this.)

## 5. Why the whole family fails: the signal is weakest where the headroom is

| dataset | CARPRT | held-out headroom |
|---|---|---|
| eurosat | 55.01 | **+22.2** |
| ucf101 | 69.97 | **+13.2** |
| dtd | 49.00 | **+11.6** |
| oxford_flowers | 71.38 | **+10.4** |
| fgvc | 24.60 | +7.2 |
| oxford_pets | 89.45 | +3.1 |
| food101 | 85.85 | +2.2 |
| caltech101 | 94.52 | +1.4 |

Every label-free scoring rule in this literature — ZPE, CARPRT Eq. 10, `pseudo`, `margin`,
`crossfit` — derives its signal from CLIP's own predictions. **Those predictions are worst
exactly where the headroom is largest.** Mean real headroom ≈ +9; the overfit gap tracks the
*error rate* (fgvc 8.4, dtd 7.8), not the parameter-to-image ratio — flowers has the most
oracle parameters per image (10.2) and one of the smallest gaps (2.6).

`spread` is the only score that escapes the coupling, and its precision does go the right
way: 26% on dtd, 23% ucf101, 20% eurosat versus 10–11% on pets and food101. It still loses
on average.

---

## Reproduction fidelity

| dataset | C | test N | our CARPRT | paper | Δ |
|---|---|---|---|---|---|
| imagenet | 1000 | 50,000 | 68.86 | 68.59 | +0.27 |
| oxford_pets | 37 | 3,669 | 89.45 | 89.13 | +0.32 |
| dtd | 47 | 1,692 | 48.88 | 48.90 | −0.02 |
| eurosat | 10 | 8,100 | 55.12 | 55.56 | −0.44 |
| oxford_flowers | 102 | 2,463 | 71.30 | 71.36 | −0.06 |
| caltech101 | 100 | 2,465 | 94.60 | 94.16 | +0.44 |
| ucf101 | 101 | 3,783 | 69.97 | 70.41 | −0.44 |
| fgvc | 100 | 3,333 | 24.69 | 24.49 | +0.20 |
| food101 | 101 | 30,300 | 85.85 | 86.31 | −0.46 |

Nine benchmarks within ±0.5; ImageNet MPE lands at 67.57 against the paper's 67.59.

---

## The twelve attempts

### Phase 1 — six estimated magnitudes

**Prompts behave like operators, and it doesn't help.** $z_{i,c} \approx T_i m_c$ reaches
0.9166 held-out centred cosine (additive null 0.8553, identity null −0.0121); the `{}`
template recovers $T=I$ to $\|W-I\|_F = 0.005$; the 247 operators lie on a manifold of
effective dimension 16.3. Synthesized embeddings score **83.84 vs 89.45**. A nested affine
model shows the matrix is worth **+0.22 for 262,144 parameters** over a 512-parameter
additive shift — 422× less efficient.

**Reconstruction anti-correlates with accuracy.** Across the low-rank ladder, centred cosine
rises 0.8554 → 0.9104 while accuracy falls 83.57 → 83.21, **Spearman ρ ≈ −0.89**. Variance
and discriminability are anti-aligned in prompt space.

**Transferable ≠ useful.** 94% of a prompt's effect by energy is class-independent, but that
component delivers only 22% of the headroom. The interaction term cannot drive weight
estimation either: monotone decline at every α, in two separate designs.

**Learned W: right form, wrong content.** Optimising W against CARPRT's pseudo-labels matched
the oracle's structure almost exactly (6.6 effective prompts vs the oracle's 6.2, cls-std
3.45 vs 3.335, top-10 mass 0.986 vs 0.972) and gained **+0.16**. In hindsight that was the
selection result shouting.

**Shrinkage cannot concentrate.** $B \le 1$, so shrinking toward the class mean only flattens
— effective prompts rose 95.6 → 134 as λ grew. Concentration needs an amplifier.

**The count factor: one large gain, no predictor.** $(\mu-\bar\mu)n^\alpha$ at frozen α=0.5:
eurosat +4.15✱✱✱, pets +0.76✱, imagenet +0.19✱✱, flowers +0.08, caltech −0.20, fgvc −0.39,
ucf101 −1.37✱, dtd −4.96✱✱✱. **Mean −0.22.** Both candidate explanations refuted by
controlled experiment: count density (eurosat still +3.71 at median count 21, where dtd loses
4.67 at 29) and class count (CIFAR-10 at C=10 gives −0.11 while eurosat at C=10 gives +4.12;
ImageNet at C=1000 is *positive*).

Supporting measurements from the same phase:

- **Pseudo-labels are not the bottleneck for weighting.** Eq. 10 with *true labels* gains
  **+0.14** over pseudo-labels on Pets. The estimator's functional form, not its inputs.
- **The gap is class-specific.** Class-agnostic correction recovers 5%; class-specific
  recovers 70%. Rules out anything WPE-shaped.
- **Optimal weights are sparse.** Oracle: 6.2 effective prompts, 97.2% of mass in the top
  ten, cls-std 3.335. CARPRT: 95.6, 34.0%, 0.781.

### Phase 2 — six tried to select

**Truncation.** CARPRT's own top-k, uniform: eurosat **+4.27**✱✱✱, food101 +0.31, negative
elsewhere. Helps only where CARPRT's ranking is already good.

**Bootstrap stability.** Flat on every dataset. It removes *noise* from CARPRT's ranking;
CARPRT's problem is *bias*.

**Pseudo-label selection.** Raised top-k overlap with the oracle 2–3× (caltech 15% → 47%)
and moved accuracy **−0.16**. It is the oracle's own algorithm with CARPRT's predictions
substituted for ground truth, so it inherits the pseudo-label tax in full.

**Self-training gains nothing, quantified.** Fitting W to labels degraded to 89.3% accuracy
*by random corruption* gives **+3.00**; fitting to CARPRT's own 89.45%-accurate pseudo-labels
gives **+0.22**. A **14× shortfall** — random errors cancel, systematic ones compound.

**`margin` / `crossfit` / `spread`** — §4.

### A property of the baseline worth reporting

**CARPRT improves with *less* unlabeled data.** Pets 89.48 → 90.24 as the estimation set
shrinks 3,669 → 459; eurosat 55.12 → 56.76 at 253 images. Mechanism: fewer images → more
empty cells → $w'=0$ → implicitly sparser weights.

---

## Known defect in the Phase-2 evaluation

Every `selectors` number uses **uniform-over-top-10**, so `j=0` is CARPRT *truncated*, not
CARPRT. On Pets that is **88.44 vs 89.45** — a ~1-point handicap applied before any selector
acts. Truncation only pays where CARPRT's top-k already beats it (eurosat k=3, food101).
An adjustment-based evaluation — add mass to selected prompts inside CARPRT's existing soft
distribution, so the baseline is CARPRT exactly — has **not** been run.

## Caveats

- Nine datasets, one backbone (ViT-B/16), one seed except the dose-response runs (2–5 seeds).
- Oracle weights come from optimising against labels; labels are used **only as a ruler**.
- §1's top-k analysis uses full-fit oracle sets at per-dataset best `k`. §3 is oxford_pets
  only. §4's abstention table is oxford_pets only.
- All Phase-2 means are **best-j per dataset** — an oracle over `j`. A method must also win
  at one *fixed* j across every dataset; none does.

## Implementation caveats (worth not re-discovering)

- **Two scoring implementations.** `--impl v1` sends singleton cells to the class prior and
  floors empty cells; its α=0 is only approximately CARPRT. `--impl v2` is **bit-exact at
  α=0** (0 discordant predictions across four sparsity regimes). Report v2.
- **fp16 ties**: `topk` and `argmax` return identical *values* but different *indices*.
  `topk` silently routes images into different cells than CARPRT uses. Always `argmax`.
- **Temperature confound**: changing a signal's magnitude silently changes softmax sharpness.
  Rescale to the baseline's per-class spread.
- **Empty ≠ missing.** $n_{i,c}=0$ means the prompt never once chose that class — evidence
  against it. Substituting the class mean cost 3.19 points.
- **$C < d$** for operator fitting: needs ≥512 class observations; Pets alone gives 37.
- **Metric**: `test.py` reports mean-of-batch-means, ~0.4 points of shuffle variance. Use
  micro-average.
- **Paired testing**: use McNemar and read the discordant count — one control scored p=0.039
  on 9 discordant pairs. The exact binomial needs log-space (`lgamma` + log-sum-exp);
  `comb(n,k)/2**n` overflows past n ≈ 1023.
- **Loader ids**: ImageNet and variants dispatch on `I`/`A`/`V`/`R`/`S`. `LOADER_ALIAS` in
  `run_operator.py` accepts both.
- **`stability.run` must not be wrapped in `torch.no_grad()`** — it calls `oracle_optimal_w`,
  which optimises by Adam.
