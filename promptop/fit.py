"""Closed-form operator estimators. No gradients, no labels, no learned MLPs.

Convention throughout: W_i is the row-acting operator, i.e.

    Z[i] ~= M @ W[i]          M: (C, D)   Z: (P, C, D)   W: (P, D, D)

so W[i] = T_i^T for the column-acting T_i of the write-up. Everything is float32
unit-norm input; the ridge solve is done in float64 for conditioning.

Deliberately NOT implemented: a per-prompt MLP residual. It would forfeit the
closed form, the training-free property and the encoding-cost result, and -- worse
-- make the hypothesis unfalsifiable, since sufficient capacity fits anything and
a good fit would then say nothing about whether prompts behave like operators.
The principled response to class-dependent prompt effects is fit_group_refinement,
not extra capacity.
"""

from typing import Dict, Optional, Sequence

import torch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _gram(m: torch.Tensor, lam: float) -> torch.Tensor:
    d = m.shape[1]
    g = (m.T @ m).double()
    return g + lam * torch.eye(d, dtype=torch.float64, device=m.device)


def auto_lambda(m: torch.Tensor, scale: float = 1e-2) -> float:
    """Scale-aware ridge: a fraction of the mean eigenvalue of M^T M.

    Keeps lambda meaningful whether M holds 37 or 1000 unit-norm rows.
    """
    d = m.shape[1]
    return float(scale * torch.trace(m.T @ m).item() / d)


def _solve(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve gram @ X = rhs for batched rhs (P, D, D)."""
    chol = torch.linalg.cholesky(gram)
    inv = torch.cholesky_inverse(chol).float()
    return torch.matmul(inv, rhs)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def fit_ridge_identity(
    m: torch.Tensor,
    z: torch.Tensor,
    lam: Optional[float] = None,
) -> torch.Tensor:
    """min_W ||Z_i - M W||_F^2 + lam ||W - I||_F^2, for every prompt i.

    Shrinking toward I (not toward 0) encodes "a prompt is a small perturbation of
    the bare class name" and is what keeps the operators interpretable.

    Returns W of shape (P, D, D).
    """
    p, c, d = z.shape
    lam = auto_lambda(m) if lam is None else lam

    gram = _gram(m, lam)
    eye = torch.eye(d, device=m.device, dtype=m.dtype)
    rhs = torch.einsum("cd,pce->pde", m, z) + lam * eye        # (P, D, D)
    return _solve(gram, rhs)


def fit_procrustes(m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """min_W ||Z_i - M W||_F s.t. W orthogonal -- prompts act as rotations.

    Sphere-preserving by construction, which matters because CLIP embeddings are
    L2-normalized and a general linear map does not keep them on the sphere.
    """
    cross = torch.einsum("cd,pce->pde", m, z)                  # (P, D, D)
    u, _, vh = torch.linalg.svd(cross.double(), full_matrices=False)
    return (u @ vh).float()


def fit_lowrank(
    m: torch.Tensor,
    z: torch.Tensor,
    rank: int = 64,
    lam: Optional[float] = None,
) -> torch.Tensor:
    """W = I + B_r with rank(B_r) <= rank, via reduced-rank regression.

    Fits the *deviation from identity*, so the parameter count is r*(2d - r)
    rather than d^2 and the operator degrades gracefully toward the identity
    null as rank -> 0.
    """
    p, c, d = z.shape
    lam = auto_lambda(m) if lam is None else lam

    gram = _gram(m, lam)
    resid = z - m.unsqueeze(0)                                  # (P, C, D)
    b_full = _solve(gram, torch.einsum("cd,pce->pde", m, resid))

    # Right singular vectors of the fitted values, obtained from the D x D Gram
    # rather than a batched SVD of (P, C, D): with C=1000 that route allocates
    # >1GB for U alone and OOMs a 6GB card.
    fitted = torch.einsum("cd,pde->pce", m, b_full)             # (P, C, D)
    gram_f = torch.einsum("pce,pcf->pef", fitted, fitted)        # (P, D, D)
    _, evec = torch.linalg.eigh(gram_f.double())                 # ascending
    vr = evec[:, :, -rank:].float()                              # (P, D, r)
    proj = torch.matmul(vr, vr.transpose(1, 2))                  # (P, D, D)

    b_r = torch.matmul(b_full, proj)
    return torch.eye(d, device=m.device, dtype=m.dtype).unsqueeze(0) + b_r


def fit_additive(m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The additive-style null: z_{i,c} ~= m_c + a_i. Returns a of shape (P, D).

    This is the null that matters. It is the model implied by treating prompt
    style as a class-independent shift, it costs D parameters per prompt against
    the operator's D^2, and it is exactly the structure that cancels under class
    differencing. An operator that cannot beat this is not earning its parameters.
    """
    return (z - m.unsqueeze(0)).mean(dim=1)


def fit_group_refinement(
    m: torch.Tensor,
    z: torch.Tensor,
    w_global: torch.Tensor,
    groups: Sequence[str],
    lam_group: Optional[float] = None,
    lam_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Hierarchical per-group correction: W_i^(g) = W_global_i + Delta_i^(g).

    The principled answer to "sketch of a cat behaves unlike sketch of a building":
    let the operator vary by semantic domain, but shrink the group-specific part
    hard, because C_g is typically far below D. Adding a class-independent bias
    b_i instead would not help at all -- a shared offset cannot express a
    class-dependent effect.

    Note this only applies to groups seen at fit time; a genuinely unseen domain
    falls back to w_global, which is what leave-one-dataset-out measures.
    """
    out: Dict[str, torch.Tensor] = {}

    for g in sorted(set(groups)):
        idx = torch.tensor([i for i, gg in enumerate(groups) if gg == g],
                           device=m.device)
        m_g, z_g = m[idx], z[:, idx, :]
        lam_g = (auto_lambda(m_g) * lam_scale) if lam_group is None else lam_group

        resid = z_g - torch.einsum("cd,pde->pce", m_g, w_global)
        delta = _solve(_gram(m_g, lam_g), torch.einsum("cd,pce->pde", m_g, resid))
        out[g] = w_global + delta

    return out


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

def predict(m_target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Synthesize (P, C, D) unit-norm prompted embeddings from class names."""
    z = torch.einsum("cd,pde->pce", m_target, w)
    return z / z.norm(dim=-1, keepdim=True)


def predict_identity(m_target: torch.Tensor, n_prompts: int) -> torch.Tensor:
    """Null: ignore the prompt entirely, use the bare class name."""
    z = m_target.unsqueeze(0).expand(n_prompts, -1, -1).contiguous()
    return z / z.norm(dim=-1, keepdim=True)


def predict_additive(m_target: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Null: class-independent additive prompt shift."""
    z = m_target.unsqueeze(0) + a.unsqueeze(1)
    return z / z.norm(dim=-1, keepdim=True)


ESTIMATORS = {
    "ridge": fit_ridge_identity,
    "procrustes": fit_procrustes,
    "lowrank": fit_lowrank,
}
