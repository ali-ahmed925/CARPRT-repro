# Findings — the ceiling of prompt reweighting, and six ways of not reaching it

Record of the `prompt-operator` and `bayes-reweight` branches. Every number was measured.
Written 2026-08-01, updated 2026-08-02 with six datasets.

**Setup.** CLIP ViT-B/16 · 247 templates (Allingham et al. pool) · τ = 1.0 · micro-average
top-1 throughout (not the paper's mean-of-batch-means, which carries ±0.4 of shuffle
variance). Operator fits use ImageNet class **names** only — no ImageNet images.

---

## 0. Reproduction fidelity

| dataset | C | test N | our CARPRT | paper | Δ |
|---|---|---|---|---|---|
| oxford_pets | 37 | 3,669 | 89.45 | 89.13 | +0.32 |
| dtd | 47 | 1,692 | 48.88 | 48.90 | −0.02 |
| eurosat | 10 | 8,100 | 55.12 | 55.56 | −0.44 |
| oxford_flowers | 102 | 2,463 | 71.30 | 71.36 | −0.06 |
| caltech101 | 100 | 2,465 | 94.60 | 94.16 | +0.44 |
| cifar10 | 10 | 10,000 | 90.05 | 90.82 (Tab. 12) | −0.77 |

All within ±0.8. The baseline is trustworthy.

## 1. The ceiling is large, and it generalises

Oracle = best possible $(P,C)$ weight matrix fitted **with ground-truth labels**. Used purely
as a ruler; the setting forbids labels. Full-fit numbers are inflated (9,139 parameters), so
they upper-bound the reachable headroom.

| dataset | CARPRT | oracle (full fit) | headroom |
|---|---|---|---|
| eurosat | 55.12 | 75.95 | **+20.8** |
| dtd | 48.88 | 68.44 | **+19.6** |
| oxford_flowers | 71.30 | 84.41 | **+13.1** |
| oxford_pets | 89.45 | 95.04 | +5.6 |
| caltech101 | 94.60 | 98.86 | +4.3 |
| cifar10 | 90.05 | 93.68 | +3.6 |

On Pets, where the held-out oracle was also computed, the *generalisable* headroom is
**+3.57** (93.02). Even discounted heavily, DTD and EuroSAT have far more room than anyone
in this literature has been extracting.

## 2. Pseudo-labels are not the bottleneck; Eq. 10's form is

| gap (Pets) | size |
|---|---|
| CARPRT → Eq. 10 with **true labels** | **+0.14** |
| Eq. 10 with true labels → best possible W | **+3.43** |

Hand CARPRT ground truth and it gains 5 images out of 3,669. The estimator's *functional
form*, not its inputs, is the limit. This retires the entire "improve the pseudo-labels"
direction.

## 3. The missing accuracy is class-specific

| correction applied to CARPRT's weights (Pets) | accuracy | recovered |
|---|---|---|
| class-agnostic (per-prompt) part only | 89.72 | **5%** |
| class-specific part only | 93.35 | **70%** |
| both (= oracle) | 95.04 | 100% |

Class-agnostic share of the correction's energy: 2.7%. Validates the paper's thesis while
showing CARPRT captures a fraction of it — and rules out anything WPE-shaped.

## 4. Optimal weights are sparse; CARPRT's are not

| (Pets) | eff. prompts | top-10 mass | cls-std |
|---|---|---|---|
| CARPRT | 95.6 | 34.0% | 0.781 |
| oracle | **6.2** | **97.2%** | **3.335** |

Pearson 0.199 but per-class **Spearman 0.774**: the ordering is roughly right, the magnitudes
badly too flat. Sharpening τ does not fix it (paper Tab. 10: Pets 88.69 at τ=0.5 vs 89.13 at
τ=1.0) — a 0.77-accurate ranking cannot support that concentration.

## 5. Six method attempts, and what each measured

### 5a. Prompts as operators — confirmed as a hypothesis, useless as a method

$z_{i,c} \approx T_i m_c$ by ridge with shrinkage toward identity, fitted on ImageNet names,
evaluated on held-out Pets classes: **0.9166** centred cosine vs **0.8553** for an additive
null and **−0.0121** for the class-name-only null. Two free confirmations: the `{}` template
recovered $T=I$ to $\|W-I\|_F = 0.005$ against a mean of 12.58; the semantic ordering of
"generic" prompts is correct. All 247 operators lie on a manifold of **effective dimension
16.3** (54 PCs for 90% variance).

Then: synthesized embeddings score **83.84 vs 89.45** (−5.61) using 37 encodings instead of
9,139. Manifold-sampled "novel prompts" land at the no-information floor (81.96).

### 5b. The matrix does not earn its parameters

| model | params/prompt | accuracy | pts per 1k params |
|---|---|---|---|
| additive shift | 512 | 83.48 | **3.25** |
| low-rank r=1 | 1,023 | 83.57 | 1.71 |
| affine ($Tm + a$, nested) | 262,656 | 83.70 | 0.007 |
| full ridge | 262,144 | 83.84 | 0.008 |

The nested affine model reads the matrix's incremental value *given the shift* directly:
**+0.22 points for 262,144 extra parameters** — the shift is **422× more parameter-efficient**.

### 5c. Reconstruction quality anti-correlates with accuracy

Low-rank ladder r = 1…64: centred cosine rises 0.8554 → 0.9104 while accuracy falls
83.57 → 83.21. **Spearman ρ ≈ −0.89.** Rank truncation adds directions by variance, and those
directions carry no discriminative value: **variance and discriminability are anti-aligned in
prompt space.**

### 5d. Transferable ≠ useful

94% of a prompt's effect (by energy) is a class-independent transformation, but that component
delivers only **22%** of the accuracy headroom. Class-aware reweighting's own value collapses
from +8.34 to +3.11 on synthesized embeddings. The 6%-by-energy interaction term carries most
of the value.

### 5e. …but the interaction term cannot drive weight estimation

Two designs, both anchored so α=0 reproduces CARPRT exactly. Monotone decline, no peak at any
α, in either. Signal mode fails because the residual carries no class identity, destroying the
`argmax` that produces pseudo-labels. Value mode keeps pseudo-labels intact and still fails —
it *did* make weights far more class-specific (cls-std 0.78 → 2.75), and those were **worse**.
Sanity check: at α=2, $z - 2Tm \approx -z$, the estimator picks the *worst* prompts, 58.5%.

### 5f. Self-training gains nothing — quantified

| labels used to fit W (Pets) | achieved | gain |
|---|---|---|
| 100% accurate | 94.99 | +5.53 |
| 89.3%, **randomly** corrupted | 92.45 | **+3.00** |
| 89.45%, **real pseudo-labels** | 89.67 | **+0.22** |

Identical label accuracy, **14× shortfall**. Random errors cancel; CARPRT's real errors are
its own systematic confusions, so fitting W to them reinforces exactly what the signal cannot
correct. And the learned weights reproduced the oracle's **structure** (6.6 effective prompts
vs 6.2, cls-std 3.45 vs 3.34) while gaining 0.16 — **right form, wrong content.**

## 6. The count factor: one large gain, no predictor

$n_{i,c}$ — how often prompt $i$ selects class $c$ — is information Eq. 10 discards entirely.
Scoring $\text{score} = (\mu - \bar\mu)\cdot n^{\alpha}$ (α=0 is CARPRT) across six datasets:

| dataset | C | med. count | `dev·√n` | `dev/sd` | `dev·√n/sd` (t-stat) |
|---|---|---|---|---|---|
| **eurosat** | 10 | 648 | **+4.12** | −5.17 | −3.68 |
| oxford_pets | 37 | 100 | **+0.76** | −1.96 | −0.08 |
| oxford_flowers | 102 | 20 | +0.32 | −3.65 | −1.14 |
| caltech101 | 100 | 18 | −0.20 | −0.89 | −0.20 |
| dtd | 47 | 29 | **−4.67** | −4.73 | −2.42 |
| cifar10 | 10 | 1002 | −0.11 | **+0.89** | **+0.77** |
| **mean** | | | **+0.04** | −2.59 | −1.13 |
| positive on | | | 3/6 | 1/6 | 1/6 |

**No rule improves on average.** `dev·√n` nets +0.04 only because EuroSAT's +4.12 cancels
DTD's −4.67. The dispersion factor hurts on five datasets and helps on the sixth — noise in
both directions, not a rule.

EuroSAT alone is striking: peak **+6.30 at α=0.375 (p = 3.9e−53)**, i.e. **61.42** against the
paper's best published EuroSAT figure of 55.56, and robust to 32× subsampling and to three
separate implementations.

### 6a. Two explanations proposed, both refuted by controlled experiment

**Count density — refuted.** Subsampling EuroSAT's *images* while holding C=10, the prompt
pool and the evaluation set fixed:

| EuroSAT med. count | 648 | 330 | 165 | 82 | 41 | 21 |
|---|---|---|---|---|---|---|
| gain (α=0.375) | +6.30 | +6.35 | +6.04 | +6.04 | +5.05 | **+3.71** |

Still +3.71 at median count 21 — where DTD (29) loses 4.67. An 8-point swing at matched
density. (On Pets the same experiment *did* decay: +0.76 → −0.44 as count fell 100 → 12.5, so
even the sensitivity to count is dataset-specific.)

**Number of classes — refuted.** Class-subsampling DTD holds evidence per cell roughly
constant ($n_{\text{img}} \propto C'$, so median count $\approx N/C$ is preserved) and shows
the count factor moving monotonically from −4.49 at C=47 to −0.89 at C=15. That looked like C
was the driver — until CIFAR-10:

| | C | med. count | `dev·√n` |
|---|---|---|---|
| eurosat | 10 | 648 | **+4.12** |
| **cifar10** | **10** | **1002** | **−0.11** |

Same class count, *more* evidence per cell, opposite sign. **Neither count density nor C
predicts where the count factor helps.** EuroSAT is an outlier we cannot identify from any
observable quantity tested.

## 7. CARPRT improves with *less* unlabeled data

Estimating weights from a subsample while always evaluating on the full test set:

| Pets: images used | 3,669 | 1,834 | 917 | 459 |
|---|---|---|---|---|
| CARPRT | 89.48 | 89.75 | 89.90 | **90.24** |

Replicated on EuroSAT (55.12 → 56.76 at 253 images). Mechanism: fewer images → more empty
cells → empty cells get $w'=0$ → weights become **implicitly sparser**, and §4 showed optimal
weights are 15× sparser than CARPRT's. Small samples accidentally sparsify in the right
direction. This is a pointed, unreported property of the baseline.

---

## Conclusion

The headroom above CARPRT is real and large (+3.6 to +20.8 by full-fit oracle; +3.57
generalisable on Pets) and it is **class-specific**. Six principled attacks — operator
synthesis, residual signals as weight-estimation input (two variants), discriminative
accumulation statistics, learned weights from pseudo-labels, and the count factor — all fail
to close it, each for a diagnosed reason.

> **Everything tried reads the same $(N, P, C)$ prompt–image similarity tensor, and CARPRT
> already extracts what it contains.** The one exception is a large unexplained gain on
> EuroSAT that no observable property of the dataset predicts.

Closing the gap plausibly requires a signal none of these methods use: image-manifold
structure (CARPRT collapses each image to an `argmax` and discards the rest), external
knowledge (LLM class descriptions), or cross-prompt agreement structure.

## Caveats

- Six datasets, one backbone (ViT-B/16), one seed per configuration except the dose-response
  runs (2–5 seeds).
- Full-fit oracle ceilings are inflated. Only Pets has a held-out oracle (93.02 vs 95.04);
  the others should be discounted similarly before being quoted.
- The EuroSAT gain is a single dataset. CIFAR-10 was a pre-registered attempt to replicate it
  at matched C and it failed.

## Implementation caveats (worth not re-discovering)

- **Two scoring implementations exist.** `--impl v1` (default) sends singleton cells (n=1) to
  the class prior and floors empty cells; its α=0 is *approximately* CARPRT. `--impl v2` makes
  **α=0 bit-exact against CARPRT** (verified 0 discordant predictions across four sparsity
  regimes). Report v2: under v1 the "gain" mixes the count factor with the singleton/empty-cell
  handling. The spread between them tracks each dataset's empty-cell fraction — Caltech (0.6%)
  and EuroSAT (1.8%) stable to 0.1, Flowers (16.9%) swings 0.5.
- **fp16 ties**: `topk` and `argmax` return identical *values* but different *indices* when
  similarities tie, which happens often in fp16. `topk` silently routes images into different
  (prompt, class) cells than CARPRT uses. Always `argmax`.
- **Temperature confound**: changing the magnitude of a weight-estimation signal silently
  changes softmax sharpness. At one point the entropy hit 0.9999 of maximum — weights flat,
  CARPRT degenerated to MPE — which reads as signal failure but is not. Rescale to the
  baseline's per-class spread before comparing.
- **Shrinkage cannot concentrate.** $B \le 1$ always, so shrinking toward the class mean only
  ever *flattens* the softmax (effective prompts rose 95.6 → 134 as λ increased). To
  concentrate you need an operator that can amplify, e.g. $1/\mathrm{se}$.
- **Empty ≠ missing.** $n_{i,c}=0$ means the prompt never once chose that class across
  thousands of images — evidence against it, not absent data. Substituting the class mean cost
  3.19 points.
- **$C < d$**: fitting a $512\times512$ operator needs ≥512 class observations. Pets alone
  gives 37 and would fit perfectly while meaning nothing.
- **Metric**: `test.py` reports mean-of-batch-means, ~0.4 points of shuffle variance. Use
  micro-average.
- **Paired testing**: comparing two weighting schemes on identical images is paired, so the
  single-proportion SE is the wrong yardstick. Use McNemar. Watch the discordant count — one
  control scored p=0.039 on **9 discordant pairs**, i.e. 7 images.
- **Overflow**: the exact binomial needs log-space (`lgamma` + log-sum-exp); `comb(n,k)/2**n`
  overflows a float past n ≈ 1023, which any dataset with a few thousand images clears.
