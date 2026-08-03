# Findings — prompt reweighting is a *selection* problem

Record of the `prompt-operator` and `bayes-reweight` branches. Every number was measured.
Written 2026-08-01, restructured 2026-08-03 around the selection result.

**Setup.** CLIP ViT-B/16 · 247 templates (Allingham et al. pool) · τ = 1.0 (1.5 for ImageNet,
per App. C.3) · micro-average top-1 · paired McNemar for all comparisons.

---

## HEADLINE — the gain is in choosing prompts, not weighting them

Take the oracle's top-$k$ prompts per class, **discard its weights entirely**, and give every
selected prompt equal weight:

| dataset | CARPRT | oracle | **top-k set + UNIFORM** | headroom recovered |
|---|---|---|---|---|
| eurosat | 55.01 | 75.95 | **74.23** (k=3) | **92%** |
| food101 | 85.85 | 89.01 | 88.57 (k=6) | 86% |
| caltech101 | 94.52 | 98.86 | 98.09 (k=3) | 82% |
| dtd | 49.00 | 68.44 | 64.42 (k=6) | 79% |
| ucf101 | 69.97 | 86.47 | 82.98 (k=3) | 79% |
| oxford_pets | 89.45 | 95.07 | 93.81 (k=3) | 78% |
| oxford_flowers | 71.38 | 84.41 | 80.88 (k=3) | 73% |
| fgvc | 24.60 | 40.14 | 35.10 (k=10) | 68% |
| | | | **mean** | **≈ 80%** |

**Knowing merely *which* 3–6 prompts matter, with no magnitude information at all, recovers
~80% of the entire achievable gain on every one of eight benchmarks.** The precise weights —
the quantity every method in this literature is built to estimate — are worth the other 20%.

### …and existing methods select at chance

Overlap between CARPRT's top-3 prompts per class and the oracle's top-3, out of 3:

| eurosat | pets | flowers | food101 | dtd | fgvc | ucf101 | caltech |
|---|---|---|---|---|---|---|---|
| 0.50 (17%) | 0.30 (10%) | 0.24 (8%) | 0.23 (8%) | 0.17 (6%) | 0.16 (5%) | 0.09 (3%) | 0.08 (3%) |

CARPRT's per-class Spearman against the oracle is 0.774, which sounds adequate — but that
correlation is carried almost entirely by the *bottom* of the ranking. It knows which prompts
are bad and has essentially no idea which are best. Forcing it onto its own top-$k$ makes
accuracy **worse** on 6 of 8 datasets, because it concentrates on the wrong set.

> **Prompt reweighting is a selection problem, not a weighting problem. Uniform weights over
> the right ~3 prompts per class recover ~80% of the achievable gain across eight benchmarks,
> while current methods identify those prompts at near-chance rates.**

## The headroom is real, and it is large

Oracle weights fitted on 50% of images and scored on the held-out 50%, so this is what a
perfect *generalising* estimator could reach — not a memorisation artifact.

| dataset | CARPRT | oracle (full fit) | **oracle (held-out)** | **real headroom** | overfit gap |
|---|---|---|---|---|---|
| eurosat | 55.01 | 75.95 | 77.23 | **+22.2** | −1.3 |
| ucf101 | 69.97 | 86.47 | 83.19 | **+13.2** | 3.3 |
| dtd | 49.00 | 68.44 | 60.64 | **+11.6** | 7.8 |
| oxford_flowers | 71.38 | 84.41 | 81.82 | **+10.4** | 2.6 |
| fgvc | 24.60 | 40.14 | 31.79 | +7.2 | 8.4 |
| oxford_pets | 89.45 | 95.07 | 92.59 | +3.1 | 2.5 |
| food101 | 85.85 | 89.01 | 88.01 | +2.2 | 1.0 |
| caltech101 | 94.52 | 98.86 | 95.94 | +1.4 | 2.9 |

**Mean real headroom ≈ +9 points.** The overfit gap tracks the *error rate* (FGVC and DTD,
the two hardest datasets, lose the most), not the parameter-to-image ratio — Flowers has the
most oracle parameters per image (10.2) yet one of the smallest gaps.

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

## Why six method attempts failed: all six estimated magnitudes

Every one operated on the 20% and never touched the 80%.

**Pseudo-labels are not the bottleneck.** Eq. 10 with *true labels* gains **+0.14** over
pseudo-labels on Pets. The estimator's functional form, not its inputs, is the limit.

**The gap is class-specific.** Applying only the class-agnostic part of the oracle's
correction recovers 5%; only the class-specific part recovers 70%. Rules out anything
WPE-shaped.

**Optimal weights are sparse.** Oracle: 6.2 effective prompts, 97.2% of mass in the top ten,
cls-std 3.335. CARPRT: 95.6, 34.0%, 0.781. But *reproducing that sparsity gains nothing* —
the learned-W experiment matched the oracle's structure (6.6 effective prompts, cls-std 3.45,
top-10 mass 0.986) and gained **0.16**. Right form, wrong content. In hindsight that was the
selection result shouting.

**Prompts behave like operators, and it doesn't help.** $z_{i,c} \approx T_i m_c$ reaches
0.9166 held-out centred cosine (additive null 0.8553, identity null −0.0121); the `{}`
template recovers $T=I$ to $\|W-I\|_F = 0.005$; the 247 operators lie on a manifold of
effective dimension 16.3. Synthesized embeddings then score **83.84 vs 89.45**. A nested
affine model shows the matrix is worth **+0.22 for 262,144 parameters** over a 512-parameter
additive shift — 422× less efficient.

**Reconstruction anti-correlates with accuracy.** Across the low-rank ladder, centred cosine
rises 0.8554 → 0.9104 while accuracy falls 83.57 → 83.21, **Spearman ρ ≈ −0.89**. Variance
and discriminability are anti-aligned in prompt space.

**Transferable ≠ useful.** 94% of a prompt's effect by energy is class-independent, but that
component delivers only 22% of the headroom. Yet the interaction term cannot drive weight
estimation either: monotone decline at every α, in two separate designs.

**Self-training gains nothing, quantified.** Fitting W to labels degraded to 89.3% accuracy
*by random corruption* gives +3.00; fitting to CARPRT's own 89.45%-accurate pseudo-labels
gives **+0.22**. A 14× shortfall — random errors cancel, systematic ones compound.

**The count factor: one large gain, no predictor.** $\text{score} = (\mu-\bar\mu)n^\alpha$
across nine datasets at frozen α=0.5: eurosat +4.15✱✱✱, pets +0.76✱, imagenet +0.19✱✱,
flowers +0.08, caltech −0.20, fgvc −0.39, ucf101 −1.37✱, dtd −4.96✱✱✱. **Mean −0.22.** Both
candidate explanations were refuted by controlled experiment: count density (EuroSAT still
+3.71 at median count 21, where DTD loses 4.67 at 29) and class count (CIFAR-10 at C=10 gives
−0.11 while EuroSAT at C=10 gives +4.12; ImageNet at C=1000 is *positive*).

## A property of the baseline worth reporting

**CARPRT improves with *less* unlabeled data.** Pets 89.48 → 90.24 as the estimation set
shrinks 3,669 → 459; EuroSAT 55.12 → 56.76 at 253 images. Mechanism: fewer images → more
empty cells → $w'=0$ → implicitly sparser weights, which §Headline shows is the right
direction.

---

## What to build next

The measurement hands the field a target: **a label-free selector**, not a better weighter.
Two properties make it tractable:

- You need only ~3 prompts per class, not a calibrated 247-vector.
- Uniform weights over the right set are enough, so the output is a *set*, not a distribution.

Candidate signals, in the order I'd test them:

1. **Pseudo-label selection.** §"Pseudo-labels are not the bottleneck" shows they are worth
   +0.14 of ground truth *for Eq. 10*. Nobody has asked whether they are good enough for
   *selection*: take the W learned from pseudo-labels, keep its top-k, weight uniformly. The
   learned-W experiment produced that W already and we only ever evaluated its full,
   magnitude-laden form.
2. **Stability selection.** Bootstrap the unlabeled set; keep prompts that rank top-k
   consistently. Directly targets the ranking noise that made concentration fail.
3. **Greedy forward selection** against an unsupervised objective (prediction entropy,
   consistency across augmentations, cluster separation) — the natural algorithm for a
   combinatorial selection problem, and untouched here.
4. **Image-manifold structure** (Idea 4 in the plan). CARPRT collapses each image to an
   `argmax` and discards the rest.

## Caveats

- Nine datasets, one backbone (ViT-B/16), one seed per configuration except the dose-response
  runs (2–5 seeds).
- Oracle weights are obtained by optimising against labels; labels are used **only as a
  ruler**, never in any proposed method.
- The top-k analysis uses the *full-fit* oracle's sets. Held-out oracle sets would be the
  stricter test and have not been computed.

## Implementation caveats (worth not re-discovering)

- **Two scoring implementations.** `--impl v1` sends singleton cells to the class prior and
  floors empty cells; its α=0 is only approximately CARPRT. `--impl v2` is **bit-exact at
  α=0** (0 discordant predictions across four sparsity regimes). Report v2. The spread
  between them tracks each dataset's empty-cell fraction.
- **fp16 ties**: `topk` and `argmax` return identical *values* but different *indices*.
  `topk` silently routes images into different cells than CARPRT uses. Always `argmax`.
- **Temperature confound**: changing a signal's magnitude silently changes softmax sharpness.
  At one point entropy hit 0.9999 of maximum — weights flat, CARPRT degenerated to MPE —
  which reads as signal failure but is not. Rescale to the baseline's per-class spread.
- **Shrinkage cannot concentrate.** $B \le 1$, so shrinking toward the class mean only
  flattens (effective prompts rose 95.6 → 134 as λ grew). Concentration needs an amplifier.
- **Empty ≠ missing.** $n_{i,c}=0$ means the prompt never once chose that class — evidence
  against it. Substituting the class mean cost 3.19 points.
- **$C < d$** for operator fitting: needs ≥512 class observations; Pets alone gives 37.
- **Metric**: `test.py` reports mean-of-batch-means, ~0.4 points of shuffle variance. Use
  micro-average.
- **Paired testing**: use McNemar, and read the discordant count — one control scored p=0.039
  on 9 discordant pairs. The exact binomial needs log-space (`lgamma` + log-sum-exp);
  `comb(n,k)/2**n` overflows past n ≈ 1023.
- **Loader ids**: ImageNet and variants dispatch on `I`/`A`/`V`/`R`/`S`, not the spelled-out
  registry names. `LOADER_ALIAS` in `run_operator.py` accepts both.
