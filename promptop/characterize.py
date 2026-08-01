"""What does the optimal weight matrix look like, and what is Eq. 10 missing?

Three questions, in increasing order of usefulness:

1. How do the oracle and CARPRT weight matrices differ descriptively?
2. Is the missing accuracy CLASS-AGNOSTIC or CLASS-SPECIFIC? Split the gap into a
   per-prompt correction shared by all classes and a per-(prompt, class) one, then
   apply each separately. If the class-agnostic half recovers most of the gap,
   CARPRT's class-awareness is fine and its per-prompt quality estimate is what's
   broken -- a much easier fix, and a different paper.
3. Which candidate statistic best predicts the oracle? Eq. 10 accumulates the raw
   similarity; the hypothesis is that it should accumulate something
   DISCRIMINATIVE. Ranking candidates by how well they align with the oracle
   predicts which estimator will work before any of them is implemented.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# descriptive
# --------------------------------------------------------------------------

def weight_stats(w: torch.Tensor, name: str) -> Dict[str, object]:
    p, c = w.shape
    ent = -(w * w.clamp_min(1e-12).log()).sum(dim=0)
    order = w.sort(dim=0, descending=True).values
    return {
        "name": name,
        "entropy_frac": float(ent.mean() / torch.log(torch.tensor(float(p)))),
        "effective_prompts": float(ent.exp().mean()),
        "across_class_std_over_uniform": float(w.std(dim=1).mean() * p),
        "top1_mass": float(order[0].mean()),
        "top10_mass": float(order[:10].sum(dim=0).mean()),
        "max_over_uniform": float(w.max() * p),
    }


def _rank(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x.argsort(dim=dim).argsort(dim=dim).float()


def compare_pair(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Agreement between two (P, C) weight matrices."""
    fa, fb = (a - a.mean()).flatten(), (b - b.mean()).flatten()
    pearson = float((fa * fb).sum() / (fa.norm() * fb.norm()).clamp_min(1e-12))

    ra, rb = _rank(a), _rank(b)                       # rank within each class
    ra = ra - ra.mean(dim=0, keepdim=True)
    rb = rb - rb.mean(dim=0, keepdim=True)
    spear = float(((ra * rb).sum(dim=0)
                   / (ra.norm(dim=0) * rb.norm(dim=0)).clamp_min(1e-12)).mean())
    return {"pearson": pearson, "spearman_per_class": spear}


def top_movers(
    w_oracle: torch.Tensor,
    w_carprt: torch.Tensor,
    templates: Sequence[str],
    k: int = 6,
) -> Tuple[List, List]:
    """Prompts the oracle up- or down-weights most, averaged over classes."""
    diff = (w_oracle - w_carprt).mean(dim=1)
    order = diff.argsort(descending=True)
    up = [(templates[i], float(diff[i]), float(w_carprt[i].mean()),
           float(w_oracle[i].mean())) for i in order[:k]]
    down = [(templates[i], float(diff[i]), float(w_carprt[i].mean()),
             float(w_oracle[i].mean())) for i in order.flip(0)[:k]]
    return up, down


# --------------------------------------------------------------------------
# where does the missing accuracy live?
# --------------------------------------------------------------------------

def decompose_gap(
    sim: torch.Tensor,
    targets: torch.Tensor,
    theta_carprt: torch.Tensor,
    w_oracle: torch.Tensor,
    temp: float = 1.0,
) -> List[Dict[str, object]]:
    """Split the oracle's correction into class-agnostic and class-specific parts.

    Working in pre-softmax space, the correction is

        delta = temp*log(W_oracle) - theta_carprt      (up to a per-class constant,
                                                        which softmax ignores)

    Its row mean over classes is the part every class shares -- a pure per-prompt
    quality adjustment, the kind a CLASS-AGNOSTIC method like WPE could express.
    The remainder is genuinely class-specific.
    """
    from .oracle import _accuracy

    theta_oracle = temp * w_oracle.clamp_min(1e-12).log()
    delta = theta_oracle - theta_carprt
    d_prompt = delta.mean(dim=1, keepdim=True)          # (P, 1) class-agnostic
    d_class = delta - d_prompt                          # (P, C) class-specific

    variants = [
        ("CARPRT (no correction)", theta_carprt),
        ("+ class-agnostic part only", theta_carprt + d_prompt),
        ("+ class-specific part only", theta_carprt + d_class),
        ("+ both (= oracle)", theta_carprt + delta),
    ]
    out = []
    for name, th in variants:
        w = F.softmax(th / temp, dim=0)
        out.append({"name": name, "acc": _accuracy(sim, w, targets)})

    frac_prompt = delta.mean(dim=1).pow(2).sum() / delta.pow(2).sum().clamp_min(1e-12)
    out.append({"name": "__meta__",
                "class_agnostic_energy_frac": float(frac_prompt)})
    return out


# --------------------------------------------------------------------------
# which statistic should Eq. 10 accumulate?
# --------------------------------------------------------------------------

@torch.no_grad()
def candidate_statistics(
    sim: torch.Tensor,
    chunk: int = 512,
) -> Dict[str, torch.Tensor]:
    """Per-(prompt, class) w' under several accumulation rules, all label-free.

    Pseudo-labels are CARPRT's own (argmax over classes per prompt), which the
    oracle measurement showed are already as informative as ground truth. Only the
    accumulated quantity varies.
    """
    n, p, c = sim.shape
    device = sim.device
    acc = {k: torch.zeros((p, c), dtype=torch.float32, device=device)
           for k in ("similarity", "log_softmax", "margin_top1", "margin_mean")}
    cnt = torch.zeros((p, c), dtype=torch.long, device=device)

    for i in range(0, n, chunk):
        s = sim[i:i + chunk]                                    # (n, P, C)
        # argmax, NOT topk. fp16 similarities tie exactly often enough that the
        # two ops disagree on which class wins; the values match but the indices
        # do not, and an image then lands in a different (prompt, class) cell than
        # CARPRT puts it in. argmax reproduces test.get_matrix's tie-breaking.
        idx = s.argmax(dim=2)                                   # (n, P)
        smax = s.gather(2, idx.unsqueeze(2)).squeeze(2)
        second = s.scatter(2, idx.unsqueeze(2),
                           float("-inf")).max(dim=2).values
        lse = torch.logsumexp(s, dim=2)
        mean_rest = (s.sum(dim=2) - smax) / max(c - 1, 1)

        vals = {
            "similarity": smax,
            "log_softmax": smax - lse,
            "margin_top1": smax - second,
            "margin_mean": smax - mean_rest,
        }
        idx_t = idx.t().contiguous()                            # (P, n)
        for k, v in vals.items():
            acc[k].scatter_add_(1, idx_t, v.t().contiguous().float())
        cnt.scatter_add_(1, idx_t, torch.ones_like(idx_t))

    safe = torch.where(cnt == 0, 1, cnt)
    return {k: v / safe for k, v in acc.items()}


def statistic_alignment(
    stats: Dict[str, torch.Tensor],
    w_oracle: torch.Tensor,
    sim: torch.Tensor,
    targets: torch.Tensor,
    temp: float = 1.0,
) -> List[Dict[str, object]]:
    """Rank candidate statistics by agreement with the oracle, and score each.

    The alignment column predicts which estimator is worth building; the accuracy
    column is what that estimator would actually deliver as a drop-in replacement
    for Eq. 10's accumulation.
    """
    from .oracle import _accuracy

    rows = []
    for name, raw in stats.items():
        w = F.softmax(raw / temp, dim=0)
        cmp = compare_pair(w, w_oracle)
        rows.append({
            "name": name,
            "spearman_vs_oracle": cmp["spearman_per_class"],
            "pearson_vs_oracle": cmp["pearson"],
            "acc": _accuracy(sim, w, targets),
        })
    return sorted(rows, key=lambda r: -r["acc"])


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------

def print_stats_table(rows: Sequence[Dict[str, object]]) -> None:
    hdr = (f"{'matrix':<26}{'H/Hmax':>9}{'eff.prompts':>13}{'cls-std':>10}"
           f"{'top1 mass':>11}{'top10 mass':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['entropy_frac']:>9.3f}"
              f"{r['effective_prompts']:>13.1f}"
              f"{r['across_class_std_over_uniform']:>10.3f}"
              f"{r['top1_mass']:>11.4f}{r['top10_mass']:>12.4f}")


def print_movers(up: List, down: List) -> None:
    def show(title, rows):
        print(f"  {title}")
        for t, d, a, b in rows:
            print(f"     {d:+.5f}  ({a:.5f} -> {b:.5f})  {t}")
    show("oracle UP-weights (CARPRT under-values these):", up)
    show("oracle DOWN-weights (CARPRT over-values these):", down)
