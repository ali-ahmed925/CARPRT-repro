"""The operator manifold: PCA over {vec(T_i - I)}, and generation of novel prompts.

This is the part of the operator view that no weighting scheme can express. Once a
prompt is an element of R^{DxD} rather than a string, the 247 written templates are
a *sample* of a continuous manifold, not the vocabulary itself. You can then
interpolate between prompts, extrapolate along principal axes, and synthesize
operators that correspond to no English sentence anyone wrote -- and use them as
prompts.

Computation goes through the P x P Gram matrix throughout: D^2 = 262,144 makes the
covariance route pointless when P = 247.
"""

from typing import Dict

import torch


def _deviation(w: torch.Tensor) -> torch.Tensor:
    """vec(W_i - I) as (P, D*D)."""
    p, d, _ = w.shape
    eye = torch.eye(d, device=w.device, dtype=w.dtype).unsqueeze(0)
    return (w - eye).reshape(p, -1)


def operator_pca(w: torch.Tensor, n_components: int = 20) -> Dict[str, torch.Tensor]:
    """PCA of the operator deviations, via the Gram matrix.

    Returns components (K, D, D), per-prompt coefficients (P, K), eigenvalues (K,)
    and the mean deviation (D, D). Coefficients are in units of the component, so
    the empirical spread of `coeffs` is what a sampler should match.
    """
    p, d, _ = w.shape
    dev = _deviation(w)                                        # (P, D*D)
    mean = dev.mean(dim=0, keepdim=True)                       # (1, D*D)
    centred = dev - mean

    gram = centred @ centred.T                                 # (P, P)
    evals, evecs = torch.linalg.eigh(gram.double())
    evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)

    k = min(n_components, int((evals > 1e-10).sum().item()))
    scale = evals[:k].sqrt().clamp_min(1e-12)
    comps = (centred.T.double() @ evecs[:, :k]) / scale        # (D*D, K), unit-norm
    coeffs = centred.double() @ comps                          # (P, K)

    return {
        "mean": mean.reshape(d, d),
        "components": comps.T.float().reshape(k, d, d),
        "coeffs": coeffs.float(),
        "eigenvalues": evals[:k].float(),
        "var_ratio": (evals[:k] / evals.sum().clamp_min(1e-12)).float(),
        "dim": d,
    }


def synthesize_operators(
    pca: Dict[str, torch.Tensor],
    n_new: int,
    mode: str = "gaussian",
    seed: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Generate n_new operators from the fitted manifold. Returns (n_new, D, D).

    modes
      gaussian  sample coefficients from a diagonal Gaussian matched to the
                empirical per-component std of the real prompts
      grid      walk the top-2 principal axes on a grid (deterministic, good for
                visualising what the axes mean semantically)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    comps, coeffs = pca["components"], pca["coeffs"]
    k, d, _ = comps.shape
    eye = torch.eye(d, device=comps.device, dtype=comps.dtype)

    if mode == "gaussian":
        std = coeffs.std(dim=0, unbiased=False).cpu()
        c = torch.randn(n_new, k, generator=g) * std.unsqueeze(0) * scale
    elif mode == "grid":
        side = max(2, int(n_new ** 0.5))
        s0, s1 = coeffs[:, 0].std().item(), coeffs[:, 1].std().item()
        a = torch.linspace(-2 * s0, 2 * s0, side)
        b = torch.linspace(-2 * s1, 2 * s1, side)
        gx, gy = torch.meshgrid(a, b, indexing="ij")
        c = torch.zeros(side * side, k)
        c[:, 0], c[:, 1] = gx.reshape(-1), gy.reshape(-1)
        c = c[:n_new] * scale
    else:
        raise ValueError(f"unknown mode {mode!r}")

    c = c.to(comps.device)
    dev = pca["mean"].unsqueeze(0) + torch.einsum("nk,kde->nde", c, comps)
    return eye.unsqueeze(0) + dev


def interpolate_operators(
    w: torch.Tensor,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    t: float = 0.5,
) -> torch.Tensor:
    """Convex interpolation in deviation space: I + (1-t)(W_a - I) + t(W_b - I).

    Interpolating the *deviation* rather than the operator keeps the identity as
    the origin, so t=0 and t=1 recover the endpoints and the path never drifts
    away from "small transformation of the class name".
    """
    d = w.shape[1]
    eye = torch.eye(d, device=w.device, dtype=w.dtype).unsqueeze(0)
    da, db = w[idx_a] - eye, w[idx_b] - eye
    return eye + (1.0 - t) * da + t * db


def print_pca_report(pca: Dict[str, torch.Tensor], top: int = 8) -> None:
    vr = pca["var_ratio"]
    cum = torch.cumsum(vr, 0)
    head = "  ".join(f"{float(x):.3f}" for x in vr[:top])
    print(f"  operator-PCA var ratio (top {top}): {head}")
    if float(cum[-1]) >= 0.90:
        n90 = int((cum < 0.90).sum().item()) + 1
        print(f"  components for 90% variance: {n90} of {len(vr)}")
    else:
        print(f"  retained {len(vr)} components covering "
              f"{100 * float(cum[-1]):.1f}% of variance "
              f"(raise --pca-components to reach 90%)")
