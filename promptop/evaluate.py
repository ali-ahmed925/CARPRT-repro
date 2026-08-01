"""Metrics for the operator hypothesis.

The headline reconstruction number is deliberately NOT raw cosine. Every prompted
embedding z_{i,c} already sits close to its class-name embedding m_c, so the
identity null scores a high raw cosine while carrying zero prompt information.
The quantity that decides whether prompts behave like operators is the cosine on
the **class-centred** embeddings, i.e. on the prompt-induced deviation

    z~_{i,c} = z_{i,c} - (1/P) sum_j z_{j,c}

which is the only part any prompt-weighting scheme can act on. Raw cosine is
reported alongside it purely as a sanity figure.
"""

from typing import Dict, Optional, Sequence

import torch


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return num / den.clamp_min(eps)


def _centre(z: torch.Tensor) -> torch.Tensor:
    """Remove the per-class mean over prompts: isolates the prompt effect."""
    return z - z.mean(dim=0, keepdim=True)


def reconstruction_report(
    z_true: torch.Tensor,
    z_pred: torch.Tensor,
    name: str = "model",
) -> Dict[str, float]:
    """Compare predicted vs true prompted embeddings, shapes (P, C, D)."""
    raw = _cos(z_pred, z_true)
    cen = _cos(_centre(z_pred), _centre(z_true))
    rel = (z_pred - z_true).norm(dim=-1) / z_true.norm(dim=-1).clamp_min(1e-8)

    return {
        "name": name,
        "cos_raw_median": float(raw.median()),
        "cos_centred_median": float(cen.median()),
        "cos_centred_p10": float(cen.flatten().quantile(0.10)),
        "cos_centred_frac_above_0.5": float((cen > 0.5).float().mean()),
        "rel_l2_median": float(rel.median()),
    }


def print_reconstruction_table(rows: Sequence[Dict[str, float]]) -> None:
    hdr = (f"{'model':<26}{'cos(centred)':>14}{'p10':>9}"
           f"{'frac>0.5':>10}{'cos(raw)':>10}{'rel-L2':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['cos_centred_median']:>14.4f}"
              f"{r['cos_centred_p10']:>9.4f}{r['cos_centred_frac_above_0.5']:>10.3f}"
              f"{r['cos_raw_median']:>10.4f}{r['rel_l2_median']:>9.4f}")


def structure_report(z_true: torch.Tensor, z_pred: torch.Tensor) -> Dict[str, float]:
    """Does the predicted class-class geometry match the true one, per prompt?

    Ranking-relevant: two embedding sets can differ pointwise yet induce the same
    ordering over classes, which is all that survives into a decision.
    """
    p = z_true.shape[0]
    sims = []
    for i in range(p):
        a = z_true[i] @ z_true[i].T
        b = z_pred[i] @ z_pred[i].T
        iu = torch.triu_indices(a.shape[0], a.shape[0], offset=1)
        va, vb = a[iu[0], iu[1]], b[iu[0], iu[1]]
        va = (va - va.mean()) / va.std().clamp_min(1e-8)
        vb = (vb - vb.mean()) / vb.std().clamp_min(1e-8)
        sims.append(float((va * vb).mean()))
    t = torch.tensor(sims)
    return {"class_geometry_corr_mean": float(t.mean()),
            "class_geometry_corr_min": float(t.min())}


def _spherical_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0):
    """Tiny spherical k-means for carving pseudo-domains out of class names.

    Needed because a single-source fit corpus (ImageNet alone) has one group, so
    provenance cannot supply the domain structure the residual test requires.
    ImageNet's own breadth -- animals, vehicles, instruments, food -- does.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    cent = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        assign = (x @ cent.T).argmax(dim=1)
        for j in range(k):
            sel = x[assign == j]
            if sel.numel():
                c = sel.mean(0)
                cent[j] = c / c.norm().clamp_min(1e-8)
    return assign


def residual_report(
    m: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    templates: Optional[Sequence[str]] = None,
    n_clusters: int = 12,
    seed: int = 0,
) -> Dict[str, object]:
    """Is what the operator fails to explain noise, or is it domain-structured?

    This is the empirical answer to the strongest objection against a universal
    operator: "sketch of a cat behaves unlike sketch of a building". If residuals
    R_{i,c} = z_{i,c} - T_i m_c are unstructured, one operator per prompt is the
    right model. If they cluster by semantic domain, prompt effects genuinely are
    class-dependent and fit_group_refinement is required rather than optional.

    Note a class-independent bias term could not fix that case either -- a shared
    offset cannot express a class-dependent effect.
    """
    pred = torch.einsum("cd,pde->pce", m, w)            # unnormalized, as fitted
    resid = z - pred
    effect = z - m.unsqueeze(0)                          # total prompt effect

    explained = 1.0 - (resid.pow(2).sum() / effect.pow(2).sum().clamp_min(1e-12))

    # A template identical to base_template (e.g. "{}") has zero prompt effect by
    # construction, so its explained-fraction denominator is numerical noise and
    # the ratio explodes. Such prompts are degenerate for this statistic, not
    # badly modelled -- exclude them rather than let one entry destroy the mean.
    eff_energy = effect.pow(2).sum(dim=(1, 2))
    valid = eff_energy > 1e-4 * eff_energy.median()
    per_prompt = 1.0 - (resid.pow(2).sum(dim=(1, 2)) / eff_energy.clamp_min(1e-12))
    pp_valid = per_prompt[valid]

    assign = _spherical_kmeans(m, n_clusters, seed=seed)
    rn = resid / resid.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    within, between = [], []
    for j in range(n_clusters):
        idx = (assign == j).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        oth = (assign != j).nonzero(as_tuple=True)[0]
        if len(oth) < 2:
            continue
        take = idx[:64]
        oth_take = oth[torch.randperm(len(oth))[:64]]
        a = rn[:, take, :]
        within.append(float(torch.einsum("pid,pjd->pij", a, a).mean()))
        b = rn[:, oth_take, :]
        between.append(float(torch.einsum("pid,pjd->pij", a, b).mean()))

    w_mean = float(torch.tensor(within).mean()) if within else float("nan")
    b_mean = float(torch.tensor(between).mean()) if between else float("nan")
    gap = w_mean - b_mean

    valid_idx = valid.nonzero(as_tuple=True)[0]
    order = valid_idx[torch.argsort(per_prompt[valid_idx])]
    worst = [(templates[i] if templates else f"prompt[{i}]", float(per_prompt[i]))
             for i in order[:5]]

    return {
        "explained_fraction": float(explained),
        "per_prompt_explained_mean": float(pp_valid.mean()),
        "per_prompt_explained_min": float(pp_valid.min()),
        "n_degenerate_prompts": int((~valid).sum()),
        "residual_within_domain_cos": w_mean,
        "residual_between_domain_cos": b_mean,
        "domain_structure_gap": gap,
        "domain_structured": bool(gap > 0.05),
        "worst_explained_prompts": worst,
        "n_clusters": n_clusters,
    }


def print_residual_report(rep: Dict[str, object]) -> None:
    deg = rep.get("n_degenerate_prompts", 0)
    note = f", {deg} zero-effect prompt(s) excluded" if deg else ""
    print(f"  operator explains {100 * rep['explained_fraction']:.1f}% of the "
          f"prompt effect (per-prompt mean "
          f"{100 * rep['per_prompt_explained_mean']:.1f}%, "
          f"min {100 * rep['per_prompt_explained_min']:.1f}%{note})")
    print(f"  residual cosine: within-domain {rep['residual_within_domain_cos']:.4f}"
          f"  between-domain {rep['residual_between_domain_cos']:.4f}"
          f"  gap {rep['domain_structure_gap']:+.4f}")
    if rep["domain_structured"]:
        print("  >>> RESIDUALS ARE DOMAIN-STRUCTURED: prompt effects are "
              "class-dependent.\n      A universal operator is insufficient; use "
              "--group-refine (mixture).")
    else:
        print("  >>> residuals are not domain-structured: a universal operator "
              "per prompt\n      is the right model; refinement is unnecessary.")
    print("  prompts the operator explains worst:")
    for t, v in rep["worst_explained_prompts"]:
        print(f"     {100 * v:6.1f}%  {t}")


def operator_manifold_report(
    w: torch.Tensor,
    templates: Sequence[str],
    top_k: int = 5,
) -> Dict[str, object]:
    """Genericness ranking and the spectrum of the operator manifold.

    ||W_i - I||_F measures how far a prompt moves semantic space: near-identity
    prompts are generic, large-norm prompts are domain-specific. PCA over
    {vec(W_i - I)} is computed through the P x P Gram matrix, since D^2 = 262144
    makes the covariance route pointless.
    """
    p, d, _ = w.shape
    dev = (w - torch.eye(d, device=w.device, dtype=w.dtype).unsqueeze(0)).reshape(p, -1)

    norms = dev.norm(dim=1)
    order = torch.argsort(norms)

    gram = dev @ dev.T
    evals = torch.linalg.eigvalsh(gram.double()).flip(0).clamp_min(0)
    total = evals.sum().clamp_min(1e-12)
    ratio = evals / total
    cum = torch.cumsum(ratio, dim=0)
    eff_dim = float(torch.exp(-(ratio * ratio.clamp_min(1e-12).log()).sum()))

    return {
        "most_generic": [(templates[i], float(norms[i])) for i in order[:top_k]],
        "least_generic": [(templates[i], float(norms[i])) for i in order.flip(0)[:top_k]],
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "pc_var_ratio_top10": [float(x) for x in ratio[:10]],
        "n_pcs_for_90pct": int((cum < 0.90).sum().item() + 1),
        "effective_dim": eff_dim,
        "n_prompts": p,
    }


def print_manifold_report(rep: Dict[str, object]) -> None:
    print(f"  ||W - I||_F : mean {rep['norm_mean']:.3f} +- {rep['norm_std']:.3f}")
    print(f"  operator manifold: {rep['n_pcs_for_90pct']} PCs for 90% variance "
          f"of {rep['n_prompts']} prompts (effective dim {rep['effective_dim']:.1f})")
    print("  most generic prompts (W closest to identity):")
    for t, n in rep["most_generic"]:
        print(f"     {n:8.3f}  {t}")
    print("  least generic prompts (largest transformation):")
    for t, n in rep["least_generic"]:
        print(f"     {n:8.3f}  {t}")
