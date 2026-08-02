"""Where exactly does prompt reweighting fall short: selection, or magnitudes?

Two questions the six method attempts never answered, both cheap.

1. HOW MUCH HEADROOM IS REAL. The oracle fits P*C parameters against N images.
   EuroSAT gets 0.30 parameters per image; Flowers and Caltech get 10. At those
   ratios the oracle is largely memorising the test set, so a full-fit "headroom"
   of +19.6 may be almost entirely inflation. Fitting on half the images and
   scoring on the other half converts the number into something a real estimator
   could conceivably reach.

2. SELECTION OR MAGNITUDES. The oracle concentrates ~97% of each class's mass on
   ~6 prompts. Scoring the oracle's top-k SET with UNIFORM weights separates
   "knowing which prompts" from "knowing their exact weights". If uniform weights
   over the right set recovers most of the oracle, the whole problem is
   identification and every magnitude-estimation idea was misdirected.

Labels are used purely as a ruler here, as in oracle.py -- nothing in this module
is a method.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

from .oracle import _accuracy, oracle_optimal_w, split_indices


def held_out_headroom(
    sim: torch.Tensor,
    targets: torch.Tensor,
    theta_init: torch.Tensor,
    carprt_acc: float,
    temp: float = 1.0,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Full-fit vs held-out oracle, plus the overfitting gap between them."""
    n, p, c = sim.shape
    _, full, _ = oracle_optimal_w(sim, targets, theta_init, temp, steps, lr)

    fit_i, ev_i = split_indices(n, seed, 0.5)
    _, fit_half, held = oracle_optimal_w(
        sim[fit_i], targets[fit_i], theta_init, temp, steps, lr,
        eval_sim=sim[ev_i], eval_targets=targets[ev_i])

    return {
        "params": p * c, "images": n, "params_per_image": p * c / max(n, 1),
        "carprt": carprt_acc,
        "oracle_full": full, "headroom_full": full - carprt_acc,
        "oracle_heldout": held, "headroom_heldout": held - carprt_acc,
        "overfit_gap": full - held, "fit_half": fit_half,
    }


@torch.no_grad()
def topk_set_scores(
    sim: torch.Tensor,
    targets: torch.Tensor,
    w_oracle: torch.Tensor,
    w_carprt: torch.Tensor,
    ks: List[int] = (3, 6, 10, 25),
) -> List[Dict[str, object]]:
    """Separate the SET of prompts from the WEIGHTS assigned to them.

    For each k, take the top-k prompts per class under the oracle and under
    CARPRT, and score each set twice -- once with uniform weight inside the set,
    once with the oracle's own weights restricted to it.

        oracle set + uniform    how far knowing only WHICH prompts gets you
        carprt set + uniform    can CARPRT identify the right ones?
        oracle set + oracle w   the ceiling, restricted to k prompts

    Also reports top-k overlap: if CARPRT's top-6 shares only 1-2 prompts with the
    oracle's, its ranking is wrong exactly where it matters, and concentrating on
    it cannot help however well the concentration is done.
    """
    p, c = w_oracle.shape
    out = []
    for k in ks:
        k = min(int(k), p)
        o_idx = w_oracle.topk(k, dim=0).indices          # (k, C)
        c_idx = w_carprt.topk(k, dim=0).indices

        def mask_of(idx):
            m = torch.zeros_like(w_oracle)
            m.scatter_(0, idx, 1.0)
            return m

        o_m, c_m = mask_of(o_idx), mask_of(c_idx)

        # uniform inside the set
        o_uni = o_m / o_m.sum(dim=0, keepdim=True).clamp_min(1e-12)
        c_uni = c_m / c_m.sum(dim=0, keepdim=True).clamp_min(1e-12)
        # oracle weights restricted to the oracle set
        o_res = w_oracle * o_m
        o_res = o_res / o_res.sum(dim=0, keepdim=True).clamp_min(1e-12)

        overlap = (o_m * c_m).sum(dim=0)                  # per class, out of k
        out.append({
            "k": k,
            "oracle_set_uniform": _accuracy(sim, o_uni, targets),
            "carprt_set_uniform": _accuracy(sim, c_uni, targets),
            "oracle_set_oracle_w": _accuracy(sim, o_res, targets),
            "overlap_mean": float(overlap.mean()),
            "overlap_frac": float(overlap.mean() / k),
        })
    return out


def print_headroom(r: Dict[str, float], name: str) -> None:
    print(f"{name:<16}{r['params']:>9,}{r['images']:>8,}"
          f"{r['params_per_image']:>9.2f}{r['carprt']:>9.2f}"
          f"{r['oracle_full']:>9.2f}{r['headroom_full']:>+9.2f}"
          f"{r['oracle_heldout']:>10.2f}{r['headroom_heldout']:>+10.2f}"
          f"{r['overfit_gap']:>9.2f}")


def print_topk(rows: List[Dict[str, object]], carprt: float, oracle: float) -> None:
    hdr = (f"{'k':>4}{'oracle set':>12}{'oracle set':>12}{'CARPRT set':>12}"
           f"{'top-k overlap':>16}")
    print(hdr)
    print(f"{'':>4}{'+ oracle w':>12}{'+ UNIFORM':>12}{'+ uniform':>12}"
          f"{'(of k)':>16}")
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['k']:>4}{r['oracle_set_oracle_w']:>12.2f}"
              f"{r['oracle_set_uniform']:>12.2f}{r['carprt_set_uniform']:>12.2f}"
              f"{r['overlap_mean']:>10.2f} ({100 * r['overlap_frac']:.0f}%)")
    print(f"\n  reference: CARPRT {carprt:.2f}   unrestricted oracle {oracle:.2f}")
