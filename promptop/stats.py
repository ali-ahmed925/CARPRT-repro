"""Paired significance testing for classifier comparisons.

Comparing two weighting schemes on the same images with the same backbone is a
PAIRED comparison: the predictions are highly correlated, so the standard error
of a single proportion (sqrt(p(1-p)/n)) is the wrong yardstick and is far too
conservative. McNemar's test conditions on the discordant pairs, which is what
actually carries the evidence.
"""

import math
from typing import Dict

import torch


def mcnemar(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    targets: torch.Tensor,
    name_a: str = "A",
    name_b: str = "B",
) -> Dict[str, object]:
    """McNemar's test with continuity correction, plus the exact binomial p.

    b = A right, B wrong.  c = A wrong, B right.  Under H0 the two are
    exchangeable, so c ~ Binomial(b + c, 1/2); the chi-square form is the normal
    approximation to that. Both are reported since the exact test is reliable
    when b + c is small.
    """
    ok_a = pred_a == targets
    ok_b = pred_b == targets
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    nd = b + c

    if nd == 0:
        chi2, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / nd
        # P(chi2_1 > x) = erfc(sqrt(x/2))
        p_chi2 = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
        # Log space: math.comb(nd, i) / 2**nd overflows a float past nd ~ 1023,
        # and any dataset with a few thousand images clears that easily. The
        # log-sum-exp form is exact and has no such limit.
        k = max(b, c)
        ln2 = math.log(2.0)
        logs = [math.lgamma(nd + 1) - math.lgamma(i + 1)
                - math.lgamma(nd - i + 1) - nd * ln2
                for i in range(k, nd + 1)]
        mx = max(logs)
        tail = math.exp(mx + math.log(sum(math.exp(v - mx) for v in logs)))
        p_exact = min(1.0, 2.0 * tail)

    # Keys are identical in every branch: callers index this dict directly, and a
    # branch-dependent schema is a latent KeyError whenever two schemes happen to
    # agree on every image.
    return {
        "name_a": name_a, "name_b": name_b, "b": b, "c": c, "discordant": nd,
        "chi2": chi2, "p_chi2": p_chi2, "p_exact": p_exact,
        "identical": nd == 0,
        "acc_a": float(ok_a.float().mean()) * 100,
        "acc_b": float(ok_b.float().mean()) * 100,
    }


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def print_mcnemar(r: Dict[str, object]) -> None:
    if r.get("identical"):
        print(f"  {r['name_b']:<34} identical predictions to {r['name_a']}")
        return
    print(f"  {r['name_b']:<34}{r['acc_b']:>8.2f}"
          f"{r['acc_b'] - r['acc_a']:>+8.2f}"
          f"{r['b']:>7}{r['c']:>7}{r['discordant']:>7}"
          f"{r['p_exact']:>11.2e}  {stars(r['p_exact'])}")
