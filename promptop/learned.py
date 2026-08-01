"""Learn the weight matrix instead of computing it from a formula.

The characterization showed Eq. 10 gets the per-class prompt ORDERING roughly
right (spearman 0.77 vs the oracle) but is far too flat: the oracle concentrates
~97% of each class's mass on ~6 prompts, CARPRT spreads it over ~96. Sharpening
tau does not fix that, because a 0.77-accurate ranking cannot support that much
concentration -- errors get amplified faster than the good prompts get exploited.

So learn the ranking rather than assume it. Optimise W against CARPRT's own
pseudo-labels, which the oracle measurement showed are worth within +0.14 of
ground truth when used inside Eq. 10.

This is a METHOD, not a diagnostic: it consumes no labels, so full-set
optimisation is the legitimate transductive protocol and the resulting accuracy
is directly comparable to CARPRT's. The risk is not test leakage but
self-confirmation -- W being fitted to predictions that W produced. Guards: the
pseudo-labels are frozen from the initial solution (no iterating), the
initialisation is CARPRT's own solution, and a KL penalty toward it bounds how
far the result may drift. kl_weight -> infinity recovers CARPRT exactly.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def pseudo_labels(sim: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One label per image from a weight matrix: argmax_c sum_i w_{i,c} s_{j,i,c}."""
    return torch.einsum("npc,pc->nc", sim, w).argmax(dim=1)


@torch.no_grad()
def corrupt_labels(
    targets: torch.Tensor,
    accuracy: float,
    n_classes: int,
    seed: int = 0,
) -> torch.Tensor:
    """Degrade labels to a target accuracy by uniform random reassignment.

    Optimistic by construction: real pseudo-label errors concentrate on confusable
    classes rather than spreading uniformly, so a sweep over this gives an UPPER
    bound on what a given pseudo-label accuracy can support.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = targets.shape[0]
    flip = torch.rand(n, generator=g).to(targets.device) > accuracy
    rnd = torch.randint(0, n_classes - 1, (n,), generator=g).to(targets.device)
    # shift past the true class so a "corrupted" label is never accidentally right
    rnd = rnd + (rnd >= targets).long()
    return torch.where(flip, rnd, targets)


def learn_weights(
    sim: torch.Tensor,
    labels: torch.Tensor,
    init_raw: torch.Tensor,
    temp: float = 1.0,
    steps: int = 400,
    lr: float = 0.05,
    kl_weight: float = 0.0,
    eval_targets: Optional[torch.Tensor] = None,
    eval_sim: Optional[torch.Tensor] = None,
    log_every: int = 50,
) -> Dict[str, object]:
    """Minimise CE(scores, labels) over W = softmax(theta/temp), from CARPRT's init.

    eval_targets is used ONLY to record a diagnostic trajectory. Nothing is
    selected on it -- the returned weights are always the final iterate, so the
    reported accuracy is what a label-free run would actually produce.
    """
    theta = init_raw.clone().detach().requires_grad_(True)
    w_ref = F.softmax(init_raw / temp, dim=0).detach()
    opt = torch.optim.Adam([theta], lr=lr)

    traj: List[Dict[str, float]] = []
    for step in range(steps + 1):
        w = F.softmax(theta / temp, dim=0)

        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                rec = {"step": step}
                if eval_targets is not None:
                    es = eval_sim if eval_sim is not None else sim
                    logits = torch.einsum("npc,pc->nc", es, w)
                    rec["acc"] = 100.0 * (logits.argmax(1) == eval_targets
                                          ).float().mean().item()
                traj.append(rec)

        if step == steps:
            break

        opt.zero_grad()
        loss = F.cross_entropy(torch.einsum("npc,pc->nc", sim, w), labels)
        if kl_weight > 0:
            loss = loss + kl_weight * (
                w * (w.clamp_min(1e-12).log() - w_ref.clamp_min(1e-12).log())
            ).sum(dim=0).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w_final = F.softmax(theta / temp, dim=0)
        ent = -(w_final * w_final.clamp_min(1e-12).log()).sum(dim=0)
        p = w_final.shape[0]
        stats = {
            "entropy_frac": float(ent.mean() / torch.log(torch.tensor(float(p)))),
            "effective_prompts": float(ent.exp().mean()),
            "across_class_std_over_uniform": float(w_final.std(dim=1).mean() * p),
            "top10_mass": float(w_final.sort(dim=0, descending=True)
                                .values[:10].sum(dim=0).mean()),
        }
    return {"weights": w_final.detach(), "trajectory": traj, "stats": stats}


def print_trajectory(traj: Sequence[Dict[str, float]], label: str) -> None:
    pts = " ".join(f"{t['step']}:{t.get('acc', float('nan')):.2f}" for t in traj)
    print(f"    {label:<22} {pts}")
