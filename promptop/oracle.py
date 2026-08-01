"""Ceilings for prompt reweighting: how much accuracy is reachable at all?

Labels are used here ONLY as a measuring instrument. Nothing in this module is a
method -- the problem setting (Problem 1 in the paper) forbids labels, and any
number produced here is an upper bound to report, never something to ship. The
CARPRT authors use the same device in Fig. 1, applying WPE per class with
ground-truth labels to motivate class-specific weighting.

What the ceilings mean:

  oracle_eq10       keeps Eq. 10's functional form (mean of the similarity over
                    images belonging to class c) but replaces the pseudo-label
                    with the true label. Isolates how much of the gap is
                    pseudo-label noise.
  oracle_optimal_w  optimises W directly against the labels. This is the ceiling
                    of the WEIGHT FAMILY itself -- one scalar per (prompt, class),
                    shared across all images of that class. No label-free
                    estimator of this form can beat it.

The gap between them says where the loss lives: if oracle_eq10 ~= CARPRT but
oracle_optimal_w is far above, pseudo-labels are fine and Eq. 10's mean-of-max
functional form is the limitation.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def similarity_tensor(
    image_features: torch.Tensor,
    text_feature: torch.Tensor,
    chunk: int = 512,
) -> torch.Tensor:
    """Full (N, P, C) similarity tensor s_{j,i,c}.

    Memory is N*P*C*4 bytes: fine for fine-grained sets (Pets ~134 MB), hopeless
    for ImageNet (~49 GB). Callers should check before invoking.
    """
    out = []
    for i in range(0, image_features.shape[0], chunk):
        imgs = image_features[i:i + chunk]
        out.append(torch.einsum("pcd,nd -> npc", text_feature, imgs).float())
    return torch.cat(out, dim=0)


def estimate_bytes(n: int, p: int, c: int) -> float:
    return n * p * c * 4 / 1e9


def _accuracy(sim: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> float:
    logits = torch.einsum("npc,pc->nc", sim, w)
    return 100.0 * (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def oracle_eq10(
    sim: torch.Tensor,
    targets: torch.Tensor,
    temp: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 10 with true labels in place of pseudo-labels. Returns (W, w_raw).

    With true labels the assignment no longer depends on the prompt, so every
    prompt aggregates over the same image set -- exactly the quantity CARPRT is
    trying to approximate when it pseudo-labels.
    """
    n, p, c = sim.shape
    vals = sim.gather(2, targets.view(-1, 1, 1).expand(n, p, 1)).squeeze(2)   # (N,P)

    w_sum = torch.zeros((p, c), dtype=torch.float32, device=sim.device)
    w_cnt = torch.zeros((p, c), dtype=torch.long, device=sim.device)
    idx = targets.unsqueeze(0).expand(p, n)                                   # (P,N)
    w_sum.scatter_add_(1, idx, vals.t().contiguous())
    w_cnt.scatter_add_(1, idx, torch.ones_like(idx))

    w_raw = w_sum / torch.where(w_cnt == 0, 1, w_cnt)
    return F.softmax(w_raw / temp, dim=0), w_raw


def oracle_optimal_w(
    sim: torch.Tensor,
    targets: torch.Tensor,
    init_raw: torch.Tensor,
    temp: float = 1.0,
    steps: int = 400,
    lr: float = 0.05,
    constrained: bool = True,
    eval_sim: Optional[torch.Tensor] = None,
    eval_targets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Optimise W against the labels. Returns (W, fit_acc, eval_acc).

    Initialised at the CARPRT solution so the optimum can only be >= CARPRT,
    which is what an upper bound must satisfy.

    constrained=True keeps W = softmax over prompts, i.e. exactly the simplex
    family CARPRT searches (Eq. 11), so the result is that family's ceiling.
    constrained=False drops the constraint for a looser bound.

    Pass eval_sim/eval_targets to fit on one split and score on another: the
    unsplit number is an absolute ceiling that may be overfit, while the split
    number is what a perfect estimator could actually generalise to.
    """
    # Starts at the CARPRT solution, so the bound can only improve on it.
    theta = init_raw.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)

    # The objective is cross-entropy but the reported quantity is accuracy, and
    # the two do not move together step for step. Keep the best iterate BY FIT
    # ACCURACY -- legitimate for an upper bound, and it stops an under-converged
    # or overshooting run from understating the ceiling. Selection never touches
    # the held-out split.
    def _w(t):
        return F.softmax(t / temp, dim=0) if constrained else t

    best_acc, best_w = -1.0, None
    for step in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(
            torch.einsum("npc,pc->nc", sim, _w(theta)), targets)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                w = _w(theta)
                acc = _accuracy(sim, w, targets)
                if acc > best_acc:
                    best_acc, best_w = acc, w.clone()

    with torch.no_grad():
        eval_acc = (_accuracy(eval_sim, best_w, eval_targets)
                    if eval_sim is not None else float("nan"))
    return best_w, best_acc, eval_acc


def split_indices(n: int, seed: int = 0, frac: float = 0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = int(n * frac)
    return perm[:k], perm[k:]


def print_ceiling_table(rows) -> None:
    hdr = f"{'method':<44}{'labels?':>9}{'accuracy':>11}{'vs CARPRT':>12}"
    print(hdr)
    print("-" * len(hdr))
    base = next((r["acc"] for r in rows if r["key"] == "carprt"), None)
    for r in rows:
        delta = "" if base is None or r["key"] == "carprt" else f"{r['acc'] - base:+.2f}"
        print(f"{r['name']:<44}{r['labels']:>9}{r['acc']:>11.2f}{delta:>12}")
