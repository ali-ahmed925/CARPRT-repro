"""Label-free prompt SELECTION.

The diagnosis: uniform weights over the oracle's top-3 prompts per class recover
~80% of the achievable gain on all eight benchmarks, while CARPRT's own top-3
overlaps the oracle's by 3-17%. So the problem is choosing a small SET per class,
not calibrating a 247-vector -- and every method tried so far estimated magnitudes.

This module implements selectors that output a set, scored with uniform weights.
Labels appear only in the oracle selector, which is the ceiling, never a method.

  carprt     top-k of CARPRT's own weights. The incumbent, and near chance.
  pseudo     top-k of a W optimised against CARPRT's PSEUDO-labels. Eq. 10 with
             true labels beats pseudo-labels by only +0.14, so pseudo-labels carry
             almost all the usable signal -- but that was measured for WEIGHTING.
             Whether they suffice for RANKING is untested, and the learned W was
             only ever evaluated in its full magnitude-laden form.
  stability  prompts that stay in the top-k across bootstrap resamples of the
             unlabeled set. A prompt that ranks high by luck will not survive
             resampling; this targets the ranking noise directly rather than
             trusting a single point estimate.
  oracle     top-k of the label-fitted W. Upper bound.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

from .infer import carprt_weights
from .oracle import _accuracy


def uniform_over_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Weight matrix placing equal mass on each class's top-k prompts."""
    idx = scores.topk(min(k, scores.shape[0]), dim=0).indices
    w = torch.zeros_like(scores)
    w.scatter_(0, idx, 1.0)
    return w / w.sum(dim=0, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def stability_scores(
    image_features: torch.Tensor,
    text_feature: torch.Tensor,
    k: int = 3,
    n_boot: int = 20,
    temp: float = 1.0,
    chunk: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    """How often each (prompt, class) survives in the top-k under resampling.

    Bootstrap the unlabeled images, recompute CARPRT's weights on each draw, and
    count top-k membership. Selection frequency is a different statistic from the
    weight itself: a cell can be large once and unstable, which is exactly the
    failure mode that made concentrating on a point estimate lose accuracy.
    """
    p, c, _ = text_feature.shape
    n = image_features.shape[0]
    votes = torch.zeros((p, c), dtype=torch.float32, device=text_feature.device)
    g = torch.Generator(device="cpu").manual_seed(seed)

    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g).to(image_features.device)
        w = carprt_weights(image_features[idx], text_feature, temp, chunk)
        top = w.topk(min(k, p), dim=0).indices
        votes.scatter_add_(0, top, torch.ones_like(top, dtype=torch.float32))
    return votes / n_boot


def pseudo_label_scores(
    sim: torch.Tensor,
    w_carprt: torch.Tensor,
    theta_init: torch.Tensor,
    temp: float = 1.0,
    steps: int = 300,
    lr: float = 0.05,
) -> torch.Tensor:
    """W optimised against CARPRT's own pseudo-labels. No ground truth used."""
    from .learned import learn_weights, pseudo_labels

    pl = pseudo_labels(sim, w_carprt)
    out = learn_weights(sim, pl, theta_init, temp, steps, lr, kl_weight=0.0)
    return out["weights"]


def evaluate_selectors(
    sim: torch.Tensor,
    targets: torch.Tensor,
    selectors: Dict[str, torch.Tensor],
    oracle_scores: torch.Tensor,
    ks: List[int] = (3, 6, 10),
) -> List[Dict[str, object]]:
    """Score every selector's top-k with uniform weights, plus overlap vs oracle."""
    rows = []
    for k in ks:
        o_idx = oracle_scores.topk(min(k, oracle_scores.shape[0]), dim=0).indices
        o_mask = torch.zeros_like(oracle_scores).scatter_(0, o_idx, 1.0)
        for name, sc in selectors.items():
            w = uniform_over_topk(sc, k)
            s_idx = sc.topk(min(k, sc.shape[0]), dim=0).indices
            s_mask = torch.zeros_like(sc).scatter_(0, s_idx, 1.0)
            overlap = float((o_mask * s_mask).sum(dim=0).mean())
            rows.append({
                "k": k, "selector": name,
                "acc": _accuracy(sim, w, targets),
                "overlap": overlap, "overlap_frac": overlap / k,
            })
    return rows


def print_selectors(rows, carprt: float, oracle: float, ks: List[int]) -> None:
    names = []
    for r in rows:
        if r["selector"] not in names:
            names.append(r["selector"])
    hdr = f"{'selector':<12}" + "".join(f"{'k=' + str(k):>20}" for k in ks)
    print(hdr)
    print(f"{'':<12}" + "".join(f"{'acc   overlap':>20}" for _ in ks))
    print("-" * len(hdr))
    for nm in names:
        line = f"{nm:<12}"
        for k in ks:
            r = next(x for x in rows if x["selector"] == nm and x["k"] == k)
            line += f"{r['acc']:>13.2f}{100 * r['overlap_frac']:>6.0f}%"
        print(line)
    print(f"\n  CARPRT (all 247, own weights) {carprt:.2f}   "
          f"unrestricted oracle {oracle:.2f}")
