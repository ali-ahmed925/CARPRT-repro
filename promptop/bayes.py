"""Prompt reweighting as posterior inference.

CARPRT derives a Bayesian framework over Pr(W | P, D) and then discards it for a
point estimate: Eq. 10 averages the winning similarity per (prompt, class) cell and
softmaxes. Their App. H notes the prior "does not play a role in the methodology"
and leaves alternative priors to future work. This module fills that gap.

Three components, each independently ablatable:

  1. EMPTY-CELL CORRECTION. 8% of (prompt, class) cells receive no image under
     CARPRT's pseudo-labelling. Eq. 10 assigns them w' = 0, i.e. the *lowest*
     weight in the class. But an unestimated cell carries no evidence either way;
     the correct fallback is the class's prior mean, not zero.

  2. VARIANCE-AWARE SHRINKAGE. Every method in this line reports a point estimate
     of w' and ignores its uncertainty. Each cell is a sample mean over n_{i,c}
     images -- median ~99 on Pets, often far fewer -- so the standard error is
     large and highly uneven. Empirical-Bayes shrinkage toward the class mean
     trusts a cell in proportion to its precision. This is what lets weights
     concentrate: the oracle uses ~6 effective prompts per class where CARPRT uses
     ~96, and naive sharpening (lower tau) fails because it amplifies noisy cells
     along with informative ones. Shrinkage sharpens only where warranted.

  3. TEXT-GEOMETRIC PRIOR. A prompt that maps every class to nearly the same point
     cannot discriminate, however strongly it responds. On Pets, CARPRT's single
     most over-weighted prompt is "a type of pet {}" -- and all 37 classes ARE
     pets, so it displaces every class embedding identically. Measurable with no
     images and no labels, from the geometry of {z_{i,c}}_c alone. No existing
     method uses this: MPE, WPE and CARPRT all measure response magnitude through
     image-text similarity.

Anchors: lam=0 with empty="zero" and beta=0 reproduces CARPRT exactly;
lam -> infinity drives every cell to the class mean, i.e. uniform weights (MPE).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# sufficient statistics
# --------------------------------------------------------------------------

@torch.no_grad()
def weight_moments(
    image_features: torch.Tensor,
    text_feature: torch.Tensor,
    chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First and second moments of Eq. 10's accumuland, per (prompt, class).

    Returns (sum, sumsq, count), each (P, C). One pass; the second moment is what
    turns a point estimate into an estimate with a standard error, at no extra cost.

    Uses torch.max (not topk) so the pseudo-labels match test.get_matrix exactly --
    fp16 similarities tie often enough that the two disagree on the winning class.
    """
    p, c, _ = text_feature.shape
    dev = text_feature.device
    s1 = torch.zeros((p, c), dtype=torch.float32, device=dev)
    s2 = torch.zeros((p, c), dtype=torch.float32, device=dev)
    n = torch.zeros((p, c), dtype=torch.long, device=dev)

    for i in range(0, image_features.shape[0], chunk):
        logits = torch.einsum("pcd,nd -> pcn", text_feature,
                              image_features[i:i + chunk])
        val, idx = torch.max(logits, dim=1)                 # (P, n)
        val = val.float()
        s1.scatter_add_(1, idx, val)
        s2.scatter_add_(1, idx, val * val)
        n.scatter_add_(1, idx, torch.ones_like(idx))
    return s1, s2, n


def cell_statistics(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-cell mean, squared standard error, and an 'unestimated' mask.

    A cell with n <= 1 has no usable estimate of the mean's variance, so it is
    marked unestimated and will be shrunk fully to the prior regardless of lambda.
    """
    safe = torch.where(n == 0, torch.ones_like(n), n)
    mu = s1 / safe
    var = (s2 / safe - mu * mu).clamp_min(0.0)
    se2 = var / safe.float()
    unestimated = n <= 1
    return mu, se2, unestimated


# --------------------------------------------------------------------------
# component 1 + 2: empty cells and variance-aware shrinkage
# --------------------------------------------------------------------------

def shrink(
    mu: torch.Tensor,
    se2: torch.Tensor,
    unestimated: torch.Tensor,
    n: torch.Tensor,
    lam: float = 1.0,
    empty: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Empirical-Bayes shrinkage of w' toward the per-class mean across prompts.

        B_{i,c} = tau_c^2 / (tau_c^2 + lam * se_{i,c}^2)
        mu~     = mu_bar_c + B * (mu - mu_bar_c)

    tau_c^2 is the spread of w' across prompts within class c -- the signal the
    shrinkage is trying to preserve. A precise cell (se << tau) keeps its estimate;
    a noisy one collapses toward the class mean and stops distorting the softmax.

    lam = 0 leaves every estimated cell untouched, so with empty="zero" this
    returns CARPRT's own w'. Large lam sends everything to the class mean, which
    is uniform weights after the softmax.
    """
    est = ~unestimated
    cnt = est.sum(dim=0, keepdim=True).clamp_min(1).float()
    mu_bar = torch.where(est, mu, torch.zeros_like(mu)).sum(dim=0, keepdim=True) / cnt
    dev = torch.where(est, mu - mu_bar, torch.zeros_like(mu))
    tau2 = (dev * dev).sum(dim=0, keepdim=True) / cnt

    if lam == 0.0:
        # Exact identity, so the anchor holds bit-for-bit. Computing 0 * inf for
        # the unestimated cells would give NaN, so short-circuit instead.
        b = torch.ones_like(mu)
    else:
        se2_safe = torch.where(unestimated, torch.ones_like(se2), se2)
        b = tau2 / (tau2 + lam * se2_safe).clamp_min(1e-12)
        b = torch.where(unestimated, torch.zeros_like(b), b)

    mu_t = mu_bar + b * (mu - mu_bar)

    # n == 0 cells hold no value at all -- mu = 0 there is a placeholder, not
    # data. This substitution must apply at EVERY lambda including 0, otherwise
    # the empty-cell correction is only reachable when shrinkage is also active
    # and the ablation row is vacuous. A cell with n == 1 does have a usable
    # value (CARPRT uses it), so only n == 0 is touched here.
    if empty == "mean":
        mu_t = torch.where(n == 0, mu_bar.expand_as(mu_t), mu_t)
    else:                                    # "zero" = CARPRT's own behaviour
        mu_t = torch.where(n == 0, torch.zeros_like(mu_t), mu_t)

    return {"mu_tilde": mu_t, "shrinkage": b, "mu_bar": mu_bar, "tau2": tau2}


def tstat_scores(
    mu: torch.Tensor,
    se2: torch.Tensor,
    unestimated: torch.Tensor,
    mu_bar: torch.Tensor,
    rescale_to: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Precision-weighted deviation: (mu - mu_bar) / se.

    Shrinkage cannot concentrate weights -- B <= 1 always, so it only ever
    reduces the spread across prompts and flattens the softmax toward MPE. To
    concentrate while still respecting uncertainty the operator has to be able to
    AMPLIFY, and 1/se is unbounded above. A cell is promoted when its deviation
    from the class mean is large relative to its own noise, not merely large.

    Cells with no variance estimate get 0, i.e. placed at the class mean: no
    evidence, no deviation.
    """
    t = (mu - mu_bar) / se2.clamp_min(1e-12).sqrt()
    t = torch.where(unestimated, torch.zeros_like(t), t)
    if rescale_to is not None:
        # Match the reference per-class spread so a fixed tau keeps meaning the
        # same thing; isolates "is this a better ranking" from "did the
        # temperature change".
        t = t / t.std(dim=0, keepdim=True).clamp_min(1e-8) \
            * rescale_to.std(dim=0, keepdim=True)
    return t


# --------------------------------------------------------------------------
# component 3: text-geometric prior
# --------------------------------------------------------------------------

@torch.no_grad()
def text_separability(
    text_feature: torch.Tensor,
    mode: str = "nearest",
    chunk_prompts: int = 32,
) -> torch.Tensor:
    """delta_{i,c}: room prompt i leaves class c against the other classes.

        nearest:  1 - max_{c' != c} cos(z_{i,c}, z_{i,c'})     (margin-relevant)
        mean:     1 - mean_{c' != c} cos(z_{i,c}, z_{i,c'})

    Image-free and label-free -- computed entirely from the text embeddings. A
    prompt whose class embeddings collapse together cannot separate them, no matter
    how strongly it responds to the images.
    """
    z = text_feature.float()
    z = z / z.norm(dim=-1, keepdim=True)
    p, c, _ = z.shape
    out = torch.zeros((p, c), device=z.device)
    eye = torch.eye(c, device=z.device, dtype=torch.bool).unsqueeze(0)

    for i in range(0, p, chunk_prompts):
        blk = z[i:i + chunk_prompts]
        cos = torch.einsum("bcd,bed->bce", blk, blk)
        m = eye.expand(cos.shape[0], -1, -1)
        if mode == "nearest":
            out[i:i + chunk_prompts] = 1.0 - cos.masked_fill(
                m, -float("inf")).max(dim=2).values
        else:
            s = cos.masked_fill(m, 0.0).sum(dim=2) / max(c - 1, 1)
            out[i:i + chunk_prompts] = 1.0 - s
    return out


def apply_prior(
    mu_tilde: torch.Tensor,
    delta: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Add the log-prior, standardised per class and rescaled to the evidence.

    The prior is z-scored within each class and multiplied by that class's spread
    of mu_tilde, so beta is expressed in standard deviations of the evidence term
    rather than in whatever units the cosine happens to have. beta = 0 is exactly
    the shrinkage-only estimator.
    """
    if beta == 0.0:
        return mu_tilde
    lp = delta.clamp_min(1e-6).log()
    lp = (lp - lp.mean(dim=0, keepdim=True)) / lp.std(dim=0, keepdim=True).clamp_min(1e-8)
    return mu_tilde + beta * mu_tilde.std(dim=0, keepdim=True) * lp


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def posterior_weights(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
    delta: Optional[torch.Tensor] = None,
    lam: float = 1.0,
    beta: float = 0.0,
    temp: float = 1.0,
    empty: str = "mean",
    score_mode: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Full estimator: moments -> (shrinkage | t-stat) -> prior -> softmax."""
    mu, se2, unest = cell_statistics(s1, s2, n)
    sh = shrink(mu, se2, unest, n, lam, empty)

    if score_mode == "tstat":
        base = tstat_scores(mu, se2, unest, sh["mu_bar"], rescale_to=mu)
    else:
        base = sh["mu_tilde"]

    scores = base if delta is None else apply_prior(base, delta, beta)
    w = F.softmax(scores / temp, dim=0)
    return {"weights": w, "scores": scores, "shrinkage": sh["shrinkage"],
            "unestimated_frac": float(unest.float().mean())}


def _deviation(s1: torch.Tensor, s2: torch.Tensor, n: torch.Tensor):
    """Shared pieces: per-cell mean, class mean, deviation, dispersion, mask."""
    mu, _, _ = cell_statistics(s1, s2, n)
    est = n > 1
    cnt = est.sum(dim=0, keepdim=True).clamp_min(1).float()
    mu_bar = torch.where(est, mu, torch.zeros_like(mu)).sum(dim=0, keepdim=True) / cnt
    dev = mu - mu_bar
    safe = torch.where(n == 0, torch.ones_like(n), n)
    sd = (s2 / safe - mu * mu).clamp_min(1e-12).sqrt()
    return mu, mu_bar, dev, sd, est


def _rescale_and_floor(
    v: torch.Tensor,
    dev: torch.Tensor,
    est: torch.Tensor,
    n: torch.Tensor,
    empty: str = "floor",
) -> torch.Tensor:
    """Put a score on the plain deviation's per-class scale, then floor empties.

    Without this a rule can win purely by being sharper, which is the confound
    that made the earlier t-statistic row read +1.12 instead of its true -0.08:
    it rescaled against a std inflated by the zeros in the empty cells.
    """
    ref_sd = torch.where(est, dev, torch.zeros_like(dev)).std(
        dim=0, keepdim=True).clamp_min(1e-8)
    v = torch.where(est, v, torch.zeros_like(v))
    v = v - v.mean(dim=0, keepdim=True)
    v = v / v.std(dim=0, keepdim=True).clamp_min(1e-8) * ref_sd
    if empty == "floor":
        floor = v.min(dim=0, keepdim=True).values - 10.0 * ref_sd
        v = torch.where(n == 0, floor.expand_as(v), v)
    return v


def count_power_scores(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
    alpha: float = 0.5,
    empty: str = "carprt",
    min_count: int = 0,
    impl: str = "v1",
) -> torch.Tensor:
    """Dispatch between the two implementations. See _count_power_v1 / _v2.

    impl="v1" is the original and the default: it reproduces every number
    generated before 2026-08-02. impl="v2" makes alpha=0 bit-exact against CARPRT.
    """
    if impl == "v1":
        _, _, dev, _, est = _deviation(s1, s2, n)
        v = dev * n.float().clamp_min(1.0).pow(alpha)
        return _rescale_and_floor(v, dev, est, n, "floor")
    return _count_power_v2(s1, s2, n, alpha, empty, min_count)


def _count_power_v2(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
    alpha: float = 0.5,
    empty: str = "carprt",
    min_count: int = 0,
) -> torch.Tensor:
    """score = mu_bar + (mu - mu_bar) * n^alpha, renormalised so alpha=0 is EXACT.

    A one-parameter family whose alpha=0 member reproduces CARPRT's weights
    bit-for-bit. n_{i,c} counts how often prompt i selects class c across the
    unlabeled set -- evidence of affinity that Eq. 10 discards, since it averages
    the winning similarities and never asks how many there were. The dispersion
    factor 1/sd is deliberately absent: the decomposition showed it hurts on every
    dataset tested.

    Getting the alpha=0 anchor exact requires care on two cell types, and the
    earlier "floor" convention got both wrong:

      n == 0 : no value at all. CARPRT leaves mu = 0, which sits ~mu_bar below the
               class mean and yields negligible weight. Reproduced exactly by
               writing 0 into the score and excluding these cells from the
               normaliser -- flooring them at min - 10*ref_sd instead drifted the
               alpha=0 row up to +1.89 over CARPRT on sparse class subsets.
      n == 1 : no variance estimate, but a perfectly good value, which CARPRT
               uses. This family needs only the count, so such cells are ordinary
               members; forcing them to the class mean was gratuitous.

    The deviation is rescaled to its own alpha=0 spread over cells with n >= 1, so
    a fixed tau means the same thing at every alpha and no alpha wins by sharpness.
    At alpha=0 that factor is exactly 1.
    """
    mu, _, _ = cell_statistics(s1, s2, n)
    # min_count is the real second component. A cell backed by n images estimates
    # its mean with error ~sd/sqrt(n), so a SINGLETON cell (n=1) is close to
    # worthless; min_count=1 shrinks every such cell to the class prior instead of
    # trusting one observation. Empirically worth ~+0.5 on DTD and Flowers, and it
    # is the variance-aware idea applied where it actually bites. min_count=0
    # trusts every visited cell, which is what makes alpha=0 exactly CARPRT.
    est = n > min_count
    cnt = est.sum(dim=0, keepdim=True).clamp_min(1).float()
    mu_bar = torch.where(est, mu, torch.zeros_like(mu)).sum(dim=0, keepdim=True) / cnt
    dev = torch.where(est, mu - mu_bar, torch.zeros_like(mu))

    v = dev * n.float().clamp_min(1.0).pow(alpha)
    v = torch.where(est, v, torch.zeros_like(v))

    ref = dev.std(dim=0, keepdim=True).clamp_min(1e-8)
    v = v / v.std(dim=0, keepdim=True).clamp_min(1e-8) * ref      # == 1 at alpha=0
    scores = mu_bar + v

    if empty == "carprt":
        # Exactly what Eq. 10 produces for an unvisited cell, so alpha=0 is
        # bit-identical to CARPRT. Use this to isolate the count factor.
        scores = torch.where(~est, torch.zeros_like(scores), scores)
    elif empty == "floor":
        # A DIFFERENT and deliberate treatment of unvisited cells: place them a
        # fixed distance below every visited cell rather than at CARPRT's implicit
        # -mu_bar, whose depth varies with the dataset's similarity scale. This is
        # a real component of the method (worth ~+0.2 on average, +0.5 on DTD and
        # Flowers), NOT an approximation of CARPRT -- so alpha=0 under this setting
        # is CARPRT + flooring, and any gain must be attributed between the two.
        floor = scores.min(dim=0, keepdim=True).values - 10.0 * ref
        scores = torch.where(n == 0, floor.expand_as(scores), scores)
    else:
        raise ValueError(f"unknown empty mode {empty!r}")
    return scores


def attribute_components(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
    alpha: float,
    temp: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """The four weight matrices needed to split a gain into its two components.

        carprt      alpha=0, empty=carprt   -- bit-identical to Eq. 10
        floor_only  alpha=0, empty=floor    -- flooring alone
        count_only  alpha>0, empty=carprt   -- count factor alone
        both        alpha>0, empty=floor    -- the full method

    Reporting only 'both' against CARPRT conflates the two changes, which is what
    made the class dose-response look like evidence for C.
    """
    def w(a, mc):
        return F.softmax(count_power_scores(s1, s2, n, a, "carprt", mc) / temp, dim=0)
    return {
        "carprt": w(0.0, 0),          # bit-identical to Eq. 10
        "singleton_only": w(0.0, 1),  # shrink n<=1 cells to the class prior
        "count_only": w(alpha, 0),    # count factor alone
        "both": w(alpha, 1),          # full method
    }


def score_variants(
    s1: torch.Tensor,
    s2: torch.Tensor,
    n: torch.Tensor,
    empty: str = "floor",
) -> Dict[str, torch.Tensor]:
    """Decompose the t-statistic into its factors, on a common per-class scale.

        t = (mu - mu_bar) * sqrt(n) / sd

    so the deviation can be multiplied by the count factor, divided by the
    dispersion factor, or both. Isolating them says WHICH one carries any gain.
    "count alone" drops the deviation entirely and scores purely by how often
    prompt i selects class c -- information Eq. 10 discards completely, since it
    averages the values and never asks how many there were.

    Every variant is centred and rescaled to the per-class spread of the plain
    deviation, so a fixed tau means the same thing for all of them and no rule
    wins merely by being sharper. The rescaled plain deviation is returned too,
    as a control: if it already departs from CARPRT, the rescaling itself is the
    confound rather than any variant.

    empty="floor" puts n == 0 cells far below every estimated cell, matching
    CARPRT's effective treatment (w' = 0 against values of ~30). Note that is
    NOT missing data -- a count of zero means the prompt never once chose that
    class, which is evidence against it.
    """
    _, _, dev, sd, est = _deviation(s1, s2, n)
    nf = n.float().clamp_min(1.0)

    raw = {
        "dev (control)": dev,
        "dev * sqrt(n)": dev * nf.sqrt(),
        "dev / sd": dev / sd,
        "dev * sqrt(n) / sd": dev * nf.sqrt() / sd,
        "count alone: log n": nf.log(),
    }
    return {k: _rescale_and_floor(v, dev, est, n, empty) for k, v in raw.items()}


def weight_summary(w: torch.Tensor) -> Dict[str, float]:
    p = w.shape[0]
    ent = -(w * w.clamp_min(1e-12).log()).sum(dim=0)
    return {
        "entropy_frac": float(ent.mean() / torch.log(torch.tensor(float(p)))),
        "effective_prompts": float(ent.exp().mean()),
        "cls_std": float(w.std(dim=1).mean() * p),
        "top10_mass": float(w.sort(dim=0, descending=True).values[:10]
                            .sum(dim=0).mean()),
    }
