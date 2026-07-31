# Implementation Plan — Beyond Similarity-Based Prompt Weighting

Research directions that move past scalar reweighting of prompt–class similarities.
Written 2026-08-01. Self-contained: assumes no memory of the conversation that produced it.

---

## 0. Where things stand

**Reproduction is verified.** CARPRT reproduces on Oxford Pets with CLIP ViT-B/16:

| | Value |
|---|---|
| CARPRT, micro-average top-1 | **89.45** (paper: 89.13) |
| MPE control | 81.17 (no renorm) / 82.94 (renormalized) |
| Weight diagnostics | entropy 4.475 / 5.509 nats, ~96 effective prompts of 247, across-class std 0.78× uniform |

Detail lives in [CLAUDE.md](CLAUDE.md) (paper↔code map, reproduction gaps) and [repro_check.py](repro_check.py).

**Environment.** Reuse the existing conda env, do *not* create a new one and do *not* run
`pip install -r requirements.txt` (it downgrades all nine pinned packages including a ~2.5 GB torch reinstall):

```bash
conda activate avatar-gen          # torch 2.5.1+cu121, tv 0.20.1, numpy 1.26.4 — all deps present
cd "/home/owais/prompt weight/CARPRT"
```

Data: `~/datasets/oxford_pets/{images/, split_zhou_OxfordPets.json}` — verified, 3669 images, 0 missing.

**Key artifact everything below depends on.** One call produces the whole object of study:

```python
text_feature = utils.clip_classifier(classnames, template, clip_model)   # (P=247, C, D=512)
```

fp16, L2-normalized then scaled by `logit_scale.exp() ≈ 100`. **Divide by 100 and re-normalize before
doing any geometry** — the scale is a scoring convenience and will distort covariances and regressions.

Note the modality gap: image and text embeddings occupy different cones. Centre each modality by its own
mean before any quadratic form or distance computation, or the geometry is dominated by a constant offset.

---

## Idea 1 — Second-Order Class Geometry (whiten by the prompt-induced covariance)

### Core intuition

Wrong assumption: **a class is a point.** Under $n$ prompts a class is a point *cloud*
$\{z_{i,c}\}_{i=1}^n$; any weighting collapses it to a convex combination — a rank-1 summary of a
rank-$\min(n,d)$ object. The cloud's shape (principal axes, anisotropy) is discarded before scoring.

Second wrong assumption: **nuisance is attributable to prompts.** Spurious semantics live along
*directions* in $\mathbb{R}^d$ that many prompts contribute to jointly. A scalar per prompt cannot
delete a direction without deleting every prompt touching it — and those carry class signal too.

### Mechanism

With $Z_c = [z_{1,c};\dots;z_{n,c}] \in \mathbb{R}^{n\times d}$, $\mu_c = \frac1n\sum_i z_{i,c}$:

$$\Sigma_c = \tfrac1n \sum_i (z_{i,c}-\mu_c)(z_{i,c}-\mu_c)^\top, \qquad \Sigma_{\mathcal P} = \tfrac1C\sum_c \Sigma_c$$

$\Sigma_{\mathcal P}$ is the **prompt-variation metric**: directions along which rephrasing moves an
embedding, irrespective of class. Score in the whitened geometry:

$$s_c(x) = -(\tilde z_I - \tilde\mu_c)^\top(\Sigma_{\mathcal P} + \lambda I)^{-1}(\tilde z_I - \tilde\mu_c)$$

$\Sigma_{\mathcal P}$ has rank $\le n-1 = 246$ in $d = 512$, so shrinkage $\lambda$ is load-bearing —
set it from the spectrum (e.g. Ledoit–Wolf), do not tune it on test accuracy.

Class-aware variant: residual $\Delta_c = \Sigma_c - \Sigma_{\mathcal P}$, score with
$(\Sigma_{\mathcal P} + \alpha\Delta_c + \lambda I)^{-1}$.

Cheap variant: $U_c$ = top-$k$ right singular vectors of $Z_c$; score $= \|U_c^\top \tilde z_I\|^2$.

### Why it is not incremental

Weighting searches $\mathbb{R}^{n\times C}$ (scalars indexed by prompt). This searches the PSD cone on
$\mathbb{R}^d$ — a space indexed by *directions*. Not nested. No weight vector can suppress a nuisance
direction spread across 40 prompts while retaining those prompts' class signal; a metric does it in one
operation. Current practice is **first-order class-aware**; this is **second-order class-aware**.

### Structural limitation bypassed

The prompt→nuisance attribution bottleneck. Weighting acts only on nuisance *separable by prompt index*.
Nuisance distributed across the pool is not hard to estimate — it is **unrepresentable**.

### Testable hypothesis

**H:** top eigenvectors of $\Sigma_c$ are largely shared across classes, and the shared subspace carries
little class-discriminative information.

Measure principal angles between per-class leading subspaces. Mean alignment $>0.7$ at $k=10$ ⟹ shared
nuisance subspace exists ⟹ projecting it out must raise class separability. Verify as a Fisher ratio
between class means *before* running any accuracy experiment. **Falsifier:** near-orthogonal per-class
subspaces ⟹ $\Sigma_{\mathcal P}$ is meaningless, idea dies cheaply.

---

## Idea 2 — Prompts as Operators, not Strings

### Core intuition

Wrong assumption: **a prompt is a string yielding a vector.** A prompt behaves like a *map* on semantic
space — "a satellite photo of {}" applies approximately the same transformation to *airport* as to
*forest*. If so, the object of interest is not $z_{i,c}$ but the operator $T_i$ with $z_{i,c}\approx T_i m_c$.
The pool of 247 is then a sparse, arbitrary sample of a continuous operator manifold.

### Mechanism

$M = [m_1;\dots;m_C] \in \mathbb{R}^{C\times d}$ = class-name embeddings, no template.
$Z_i = [z_{i,1};\dots;z_{i,C}]$. Closed form, no gradients:

$$T_i = \arg\min_T \|Z_i - MT^\top\|_F^2 + \lambda\|T-I\|_F^2 \;\Longrightarrow\; T_i = (M^\top M + \lambda I)^{-1}(M^\top Z_i + \lambda I)$$

Optionally constrain $T_i \in O(d)$ via orthogonal Procrustes so prompts act as rotations on the sphere.

Three consequences:

- **Residual** $R_{i,c} = z_{i,c} - T_i m_c$ isolates the genuinely *interactive* part of a prompt–class
  pair from the transferable part.
- **Prompt manifold**: PCA on $\{\mathrm{vec}(T_i)\}$ gives a continuous basis; interpolate to synthesize
  prompts corresponding to no written template. The pool becomes a chart, not a vocabulary.
- **Zero-shot prompt transfer**: unseen class $c^*$ gets all prompted variants as $\{T_i m_{c^*}\}$ from a
  *single* text encoding.

### Why it is not incremental

Changes a prompt's type signature from $\mathbb{R}^d$ to $\mathbb{R}^{d\times d}$ — from a point to a
group-like element that composes, inverts, interpolates. Unavailable to any scheme mixing fixed vectors.
Also inverts the cost model: $O(C)$ text encodes instead of $O(nC)$ — for ImageNet, 1,000 instead of 247,000.

### Structural limitation bypassed

The **closed-pool** constraint. Weighting is confined to the convex hull of whatever templates someone
wrote down and can never express a prompt outside the pool. Operators make prompt space continuous and
generative; the hull stops being a boundary.

### Testable hypothesis

**H:** prompt operators are class-transferable.

Fit $\{T_i\}$ on 80% of classes, predict $z_{i,c}$ for the held-out 20%, report cosine against the true
encoder output. Median cosine $>0.9$ ⟹ prompts are operator-like and the pool compresses. Then the
decisive test: classify using *synthesized* embeddings only and check accuracy matches true encodings.
That single number decides the paper.

---

## Idea 3 — Prompt-Invariant Difference Geometry + Tournament Decision

### Core intuition

Wrong assumption A: **prompt variation should be aggregated.** Aggregation is the worst response to a
nuisance variable you can *cancel*. Wrong assumption B: **classification should be prototype-wise.**
Class identity is only ever used comparatively, and comparisons have better invariance than prototypes.

Model $z_{i,c} = m_c + a_i + E_{i,c}$ ($a_i$ = prompt style, $E$ = interaction). Then

$$d_i^{cc'} = z_{i,c} - z_{i,c'} = (m_c - m_{c'}) + (E_{i,c} - E_{i,c'})$$

**The prompt-style term cancels exactly** — not shrunk, not down-weighted. Algebraically removed at zero
estimation cost.

### Mechanism

Treat each prompt as an *environment*. Per class pair, stack $D^{cc'} = [d_1^{cc'};\dots;d_n^{cc'}]
\in \mathbb{R}^{n\times d}$; take leading right singular vector $v^{cc'}$ (prompt-invariant discriminative
direction) with stability $\rho^{cc'} = \sigma_1^2 / \sum_k \sigma_k^2$.

Decide pairwise in difference geometry:

$$g^{cc'}(x) = \mathrm{sign}(v^{cc'\top}\tilde z_I - b^{cc'}), \qquad b^{cc'} = \tfrac12 v^{cc'\top}(\mu_c + \mu_{c'})$$

Resolve $\binom{C}{2}$ comparisons by **Copeland score or Bradley–Terry**, not summation. Pairs with
$\rho^{cc'}$ below a spectral threshold **abstain** — the distinction is prompt-dependent, so no vote.

### Why it is not incremental

Aggregate-then-decide becomes decide-in-the-invariant-subspace. Nuisance is eliminated by construction,
not estimated and suppressed, so it carries no estimator variance — which matters most in low-sample and
long-tailed regimes. And $\rho^{cc'}$ makes **prompt ambiguity a measurable quantity per class pair**,
enabling principled abstention. Score aggregation has nowhere to put that information.

### Structural limitation bypassed

The single-global-ranking bottleneck. One score per class forces every comparison through a common
scalarization, so a prompt sharpening $(c_1,c_2)$ while confusing $(c_3,c_4)$ must get one weight.
Pairwise geometry gives each comparison its own subspace: $\binom{C}{2}$ decision rules where scoring
allows $C$.

### Testable hypothesis

**H:** the additive-style model holds well enough that differencing removes most cross-class variance.

Two-way ANOVA on the $(247, C, 512)$ tensor: what share of variance does a class-independent $a_i$
explain? Large share ⟹ differencing is near-exact. Signature prediction: **accuracy gain correlates
positively with per-class prompt-weight entropy** — classes where the pool is most confused are exactly
where cancellation helps most.

---

## Idea 4 — Structure-to-Structure Alignment (no cross-modal similarity at all)

### Core intuition

Deepest assumption: **the cross-modal inner product $\langle z_I, z_T\rangle$ is a meaningful primitive.**
It presumes both modalities are registered in a shared frame — which the modality gap says they are not.
Absolute cross-modal positions are unreliable; *relational* structure is far more stable. Classes close in
text space should map to image clusters close in image space, and that holds under transformations that
scramble absolute cross-modal geometry.

### Mechanism

Never compare an image to a text embedding.

1. **Text relational matrix** $A_{cc'} = \|m_c - m_{c'}\|$ over class-name embeddings (prompts optional,
   used only to stabilize $m_c$).
2. **Image relational matrix** — cluster the unlabeled test set into $K \ge C$ clusters (spherical
   $k$-means or diffusion on the $k$-NN graph); $B_{kk'} = \|\bar z_k - \bar z_{k'}\|$ over centroids.
3. **Gromov–Wasserstein alignment**, entropically regularized:

$$\pi^* = \arg\min_{\pi\in\Pi(p,q)} \sum_{c,c',k,k'} (A_{cc'} - B_{kk'})^2 \pi_{ck}\pi_{c'k'} - \epsilon H(\pi)$$

4. **Label** each image by the class matched to its cluster through $\pi^*$.

GW compares distances *within* each space, never *across*. Invariant to isometries of either space
independently.

### Why it is not incremental

Removes similarity scoring from the decision path entirely, and with it the modality gap — not by
correcting the gap but by choosing an objective the gap cannot affect. Classification becomes
**graph matching**. Inductive bias shifts from "images near their class text" to "semantic relations among
classes are mirrored in the visual data" — strictly weaker and more transferable. Fully transductive,
label-free by construction.

### Structural limitation bypassed

Cross-modal frame dependence. Similarity-based methods inherit whatever misregistration the encoders were
trained with, and reweighting is applied *after* the corrupted comparison, so it cannot repair a rotated
frame. GW never performs the comparison.

### Testable hypothesis

**H (decisive):** apply a random orthogonal $Q \in O(512)$ to *image* embeddings only.

Every similarity-based method — MPE, WPE, class-aware weighting — collapses to chance. GW should be
**exactly unaffected**, since $B$ is invariant to $Q$. Qualitative, all-or-nothing separation; no amount of
prompt reweighting survives it. Secondary: domain shift (ImageNet-R/Sketch), where the modality gap widens
and relational structure should degrade more slowly than absolute alignment.

**Honest risks:** GW is non-convex and init-sensitive; needs $K \approx C$ and roughly balanced clusters;
will likely trail similarity methods on clean in-domain data. **Frame the paper as robustness and
invariance, not raw accuracy.** Chasing raw accuracy here is a mistake.

---

## Phase 0 — Premise diagnostics (do this first)

Every premise above is pure linear algebra on the $(247, C, 512)$ tensor. No GPU time beyond one encode.
Write `probe_geometry.py`, reusing the feature-caching pattern in [repro_check.py](repro_check.py):

| Probe | For | Computes | Kill criterion |
|---|---|---|---|
| Subspace alignment | Idea 1 | principal angles between per-class top-$k$ subspaces of $Z_c$ | near-orthogonal ⟹ drop Idea 1 |
| Spectrum of $\Sigma_{\mathcal P}$ | Idea 1 | eigenvalue decay, effective rank | flat spectrum ⟹ whitening is a no-op |
| Operator regression | Idea 2 | fit $T_i$ on 80% classes, cosine on held-out 20% | median cosine $<0.7$ ⟹ drop Idea 2 |
| Two-way ANOVA | Idea 3 | variance share of class-independent $a_i$ | small share ⟹ differencing buys little |
| Pair stability $\rho^{cc'}$ | Idea 3 | distribution over class pairs | uniformly high ⟹ no abstention signal |
| Relational correlation | Idea 4 | Spearman between $A$ (text) and $B$ (image clusters) | near zero ⟹ GW has nothing to match |

All six run on Pets in minutes and each has a clean falsifier. **Run these before committing to a
direction** — three of the four ideas can be killed or confirmed without a single accuracy number.

---

## Recommended order

1. **Idea 2 first** — best effort-to-payoff. Closed-form fit on a tensor you can produce in one call,
   falsifiable in an afternoon, and it carries a standalone compute result (247× fewer text encodings)
   that holds even if the accuracy story is modest.
2. **Ideas 1 + 3 compose** — whiten by $\Sigma_{\mathcal P}$, *then* run the pairwise tournament in the
   whitened space. They attack first- and second-order nuisance respectively and share all machinery.
3. **Idea 4 is the highest ceiling and the highest risk.** The rotation experiment is the kind of
   qualitative separation reviewers remember, but it will lose on clean benchmarks. Pursue only with the
   robustness framing locked in.

## Constraints these were designed under

No scalar reweighting variants, no normalization/temperature tricks, no confidence or entropy reweighting,
nothing confined to the output/logit level, no labels or supervised fine-tuning. All mechanisms are
training-free at inference; the only fitting is closed-form least squares on existing embeddings.

## Open questions

- Modality gap handling is assumed to be "centre each modality" throughout. Whether that is sufficient for
  the quadratic forms in Idea 1 is untested and could sink it independently of the subspace-alignment result.
- Idea 2's operator model is stated for a shared $T_i$ across all classes. If residuals $R_{i,c}$ turn out
  large and structured, the right model may be a *mixture* of operators indexed by semantic domain
  (the 247-pool is a union of per-dataset template sets — satellite, video/action, texture, histopathology,
  digits — so a domain-indexed mixture is the natural next hypothesis).
- Idea 4 needs a cluster-count strategy for $K \ne C$ and a story for long-tailed data, where balanced
  marginals $\Pi(p,q)$ are wrong.
