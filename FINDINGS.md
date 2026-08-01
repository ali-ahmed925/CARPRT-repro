# Findings — prompt operators and the ceiling of prompt reweighting

Record of the `prompt-operator` branch. Every number below was measured, not estimated.
Written 2026-08-01.

**Setup.** CLIP ViT-B/16 · 247 templates (Allingham et al. pool) · target Oxford Pets
(37 classes, 3669 images) · operator fit corpus = 982 ImageNet class **names** (16 names
overlapping with Pets dropped to prevent leakage; no ImageNet images used) · τ = 1.0 ·
micro-average top-1 throughout.

CARPRT reproduction verified at **89.45** (paper reports 89.13; the 0.32 sits inside the
±0.4 shuffle variance of the paper's mean-of-batch-means metric).

## Reference points

| | accuracy |
|---|---|
| MPE (uniform weights) | 81.17 |
| identity null (bare class name, zero prompt information) | 81.82 |
| **CARPRT** | **89.45** |
| oracle: Eq. 10 with true labels | 89.59 |
| oracle: best possible W, held-out 50% | 93.02 |
| oracle: best possible W, full fit | 95.04 |

---

## 1. Prompts really do behave like operators

Fitting $z_{i,c} \approx T_i m_c$ by ridge regression with shrinkage toward the identity,
on ImageNet class names, evaluated on held-out Pets classes:

| model | centred cosine |
|---|---|
| operator | **0.9166** |
| additive null ($z \approx m + a_i$) | 0.8553 |
| identity null (class name only) | −0.0121 |

Centred cosine (per-class mean removed) is the headline; raw cosine is inflated to ~0.95
for everything because prompted embeddings already sit near their class name.

Two independent confirmations:

- The `{}` template — which *is* in the pool and must map to $T = I$ by construction —
  came back at $\|W - I\|_F = 0.005$ against a mean of 12.58 across prompts.
- The semantic ordering is right: most generic are `a {}.`, `{} thing`, `{}, an animal`;
  least generic are the action/travel imports, `a photo of the person during {}.`,
  `a photo i took while visiting {}.`

All 247 operators lie on a low-dimensional manifold: **54 PCs for 90% of variance,
effective dimension 16.3**.

## 2. As a replacement for the text encoder, it fails

| text features | encodings | accuracy |
|---|---|---|
| true CLIP | 9,139 | 89.45 |
| synthesized $T m$ | **37** | **83.84** |
| novel pool sampled from the operator manifold | 37 | 81.96 |

247× fewer encodings, −5.61 accuracy. Not a trade anyone wants, since encoding is amortised
once per dataset. Manifold-sampled "novel prompts" land at the no-information floor.

## 3. The matrix does not earn its parameters

| model | params/prompt | accuracy | pts per 1k params |
|---|---|---|---|
| additive shift | 512 | 83.48 | **3.25** |
| low-rank r=1 | 1,023 | 83.57 | 1.71 |
| affine ($Tm + a$, nested) | 262,656 | 83.70 | 0.007 |
| full ridge | 262,144 | 83.84 | 0.008 |

The **affine** model nests both, so the matrix's incremental value *given the shift* is
readable directly: **+0.22 points for 262,144 extra parameters.** A plain additive shift
captures 82% of the operator's total gain over the floor with 0.2% of the parameters —
**422× more parameter-efficient**. The whole 83.13–83.84 spread is ~26 images, inside one
standard error.

## 4. Reconstruction quality anti-correlates with accuracy

Across the low-rank ladder r = 1, 2, 4, 8, 16, 64:

| | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| centred cosine | 0.8554 | 0.8638 | 0.8858 | 0.9015 | 0.9052 | 0.9104 |
| accuracy | 83.57 | 83.43 | 83.29 | 83.40 | 83.13 | 83.21 |

Reconstruction rises monotonically, accuracy falls. **Spearman ρ ≈ −0.89.** Rank truncation
adds directions by variance, and those directions carry no discriminative value: **variance
and discriminability are anti-aligned in prompt space.**

## 5. The transferable part of a prompt is not the useful part

| | share |
|---|---|
| prompt effect explained by a class-independent transformation | **94%** by energy |
| accuracy headroom that component delivers | **22%** (+2.02 of +7.63 over the floor) |

Class-aware reweighting's own value collapses from **+8.34** (CARPRT − MPE on true
embeddings) to **+3.11** on synthesized ones. The interaction term $R_{i,c}$ is 6% of the
energy and carries most of the value.

## 6. …but the interaction term cannot be used for weight estimation

Two attempts, both anchored so that α=0 reproduces CARPRT exactly:

| α | signal mode | value mode |
|---|---|---|
| 0 | 89.48 | 89.48 |
| 0.5 | 88.72 | 89.13 |
| 1.0 | 81.52 | 80.57 |

Monotone decline, no peak at any α, in either mode.

- **Signal mode** (replace the whole weight-estimation tensor) fails because the residual
  carries no class identity, so the `argmax` that produces pseudo-labels is destroyed.
- **Value mode** (pseudo-labels from the full embedding, magnitude from the residual) keeps
  pseudo-labels intact and still fails. It *did* make weights far more class-specific
  (cls-std 0.78 → 2.75) — and those more class-specific weights were **worse**.
- Sanity check: at α = 2, $z - 2Tm \approx -z$, so the estimator selects the prompts that fit
  each class *worst*. Accuracy 58.5%, exactly as it should.

## 7. Pseudo-labels are not the bottleneck; Eq. 10's form is

| gap | size |
|---|---|
| CARPRT → Eq. 10 with **true labels** | **+0.14** |
| Eq. 10 with true labels → best possible W | **+3.43** |
| CARPRT → best possible W (held-out) | +3.57 |

Give CARPRT ground truth and it gains 5 images. The limitation is the mean-of-max-similarity
estimator itself, not its inputs.

## 8. The missing accuracy is class-specific

| correction applied to CARPRT's weights | accuracy | recovered |
|---|---|---|
| class-agnostic (per-prompt) part only | 89.72 | **5%** |
| class-specific part only | 93.35 | **70%** |
| both (= oracle) | 95.04 | 100% |

Class-agnostic share of the correction's energy: **2.7%**. This validates the paper's thesis
— class-awareness is where the value is — while showing CARPRT captures a fraction of it.
It also rules out anything WPE-shaped: improving the global prompt ranking is worth 5%.

## 9. Optimal weights are sparse; CARPRT's are not

| | eff. prompts | top-10 mass | cls-std |
|---|---|---|---|
| CARPRT | 95.6 | 34.0% | 0.781 |
| oracle | **6.2** | **97.2%** | **3.335** |

Pearson 0.199 but **per-class Spearman 0.774**: CARPRT gets the ordering roughly right and
the magnitudes badly wrong — far too flat. Sharpening τ does not fix it (the paper's Table 10
has Pets at 88.69 for τ=0.5 vs 89.13 at τ=1.0), because a 0.77-accurate ranking cannot
support that concentration.

## 10. Hand-designed accumulation rules do not help

| rule | Spearman vs oracle |
|---|---|
| **raw similarity (CARPRT's own)** | **0.774** |
| mean margin | 0.584 |
| top-1 margin | 0.508 |
| log-softmax | 0.336 |

Spearman is scale-invariant (softmax is monotone within a class), so this comparison is
immune to the temperature confound that affects the raw accuracy column. CARPRT's rule
already ranks prompts better than every discriminative alternative tried.

## 11. Self-training gains nothing — and here is the number

| labels used to fit W | achieved | gain |
|---|---|---|
| 100% accurate | 94.99 | +5.53 |
| 89.3% accurate, **randomly** corrupted | 92.45 | **+3.00** |
| 89.45% accurate, **real pseudo-labels** | 89.67 | **+0.22** |

Identical label accuracy, **14× shortfall**. Random errors point in uncorrelated directions
and cancel; CARPRT's real errors are its own systematic confusions, so fitting W to them
reinforces exactly the mistakes the signal cannot correct.

And the decisive detail — the learned weights reproduced the oracle's **structure**:

| | eff. prompts | cls-std | top-10 mass | accuracy |
|---|---|---|---|---|
| learned (KL=0) | 6.6 | 3.452 | 0.986 | 89.62 |
| oracle | 6.2 | 3.335 | 0.972 | **95.04** |

Right form, wrong content. It concentrates on ~6 prompts per class exactly as it should, and
picks the wrong six. **Sparsity and class-specificity were never the missing ingredient.**

---

## Conclusion

The headroom above CARPRT is real (**+3.57** generalisable, +5.59 full-fit) and it is
class-specific. But five independent attacks — better embeddings, better residual signals,
better accumulation statistics, better pseudo-labels, learned weights — all land inside one
standard error.

> **The information needed to close the gap is not present in the $(N, P, C)$ prompt–image
> similarity tensor over unlabeled data.** Every method in this family reads that same tensor,
> and CARPRT already extracts what it contains.

Closing it requires a signal none of these methods use:

1. **Image-manifold structure** — cluster geometry, density, neighbourhood consistency.
   CARPRT collapses each image to an `argmax` and discards the rest. (Idea 4 in the plan;
   the paper's own Table 6 shows InMaP + CARPRT helps.)
2. **External knowledge** — LLM class descriptions know a priori that "a type of pet" suits a
   breed and "satellite photo of" does not (paper App. G.5).
3. **Cross-prompt agreement structure**, as distinct from each prompt's own magnitude.

## Caveats

- **Single dataset.** Pets has CARPRT's largest margin in the paper (+9.67 over MPE) and may
  be atypical. Re-running the `oracle` and `characterize` commands on DTD (+2.02) and
  Caltech101 (+1.66) is the first thing to do before generalising any of this.
- Single backbone (ViT-B/16), single seed.
- The full-fit best-W ceiling (95.04) is inflated — 9,139 free parameters on 3,669 images.
  The held-out 93.02 is the number to quote.

## Methodological notes (bugs caught, worth not re-discovering)

- **fp16 ties**: `topk` and `argmax` return identical *values* but different *indices* when
  similarities tie, which they do often in fp16. Using `topk` silently routed images into
  different (prompt, class) cells than CARPRT uses. Always `argmax`.
- **Temperature confound**: changing the magnitude of the weight-estimation signal silently
  changes softmax sharpness. At α=1 the entropy went to 0.9999 of maximum — weights flat,
  CARPRT degenerated to MPE — which reads as signal failure but is not. Rescale w' to the
  baseline's per-class spread before comparing.
- **$C < d$**: fitting a $512\times512$ operator needs ≥512 class observations. Pets alone
  gives 37 and would fit perfectly while meaning nothing.
- **Metric**: `test.py` reports mean-of-batch-means, which varies ~0.4 points with the loader
  shuffle. Use micro-average.
- **Degenerate prompts**: the `{}` template has zero prompt effect by construction, so any
  per-prompt statistic normalised by that effect explodes. Exclude it.
