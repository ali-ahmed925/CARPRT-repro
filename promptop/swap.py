"""Does moving toward the oracle's prompt set actually buy accuracy?

The selection result says uniform weights over the oracle's top-k recover ~80% of
its gain, while CARPRT's top-k overlaps that set by 3-17%. The obvious reading --
get closer to the oracle's set and accuracy follows -- is contradicted by the
`select` run, where a pseudo-label selector raised overlap 15% -> 47% on
caltech101 and moved accuracy by -0.16.

Both can hold if the payoff is CONVEX in overlap. Uniform weights over k prompts
put all the mass on k choices; a wrong choice is no longer diluted by the other
246, so a partially-right set pays the concentration cost without earning the
concentration reward. Under convexity there is no partial credit, which is a
statement about how hard selection is -- not a refutation of it.

This module measures the shape directly. Build a path from CARPRT's top-k to the
oracle's top-k, one prompt at a time:

    S_c(j) = { oracle's top-j prompts for class c }
           U { best CARPRT prompts not already chosen, until |S_c(j)| = k }

j=0 is exactly CARPRT's top-k; j=k is exactly the oracle's top-k. A LINEAR curve
means overlap converts to accuracy proportionally and any selector that improves
overlap helps. A CONVEX curve quantifies how nearly-right a set has to be before
it is worth anything, which is the bar a selector must clear to be worth building.

A random path (j uniformly random prompts instead of j oracle prompts, same fill
rule) runs alongside to separate "swapped in the ORACLE's prompts" from
"perturbed the set at all".

Labels enter only through w_oracle, as a ruler. Nothing here is a method.
"""

from typing import Dict, List, Optional

import torch

from .oracle import _accuracy


def _fill_to_k(head_idx: Optional[torch.Tensor], order: torch.Tensor,
               k: int, p: int, c: int) -> torch.Tensor:
    """(P,C) 0/1 mask: the head prompts, topped up from `order` to exactly k.

    `order` is a (P,C) ranking of prompt indices per class, best first -- the
    prompts we fall back on for the slots the head does not fill. Prompts already
    in the head are skipped rather than double-counted, so a head that overlaps
    the fallback ranking still yields exactly k distinct prompts.
    """
    m = torch.zeros((p, c), device=order.device, dtype=torch.float32)
    if head_idx is not None and head_idx.numel():
        m.scatter_(0, head_idx, 1.0)

    free = 1.0 - m.gather(0, order)          # 1 where that prompt is still available
    rank = free.cumsum(dim=0)                # its position among the available ones
    need = float(k) - m.sum(dim=0, keepdim=True)
    m.scatter_add_(0, order, ((free > 0) & (rank <= need)).float())
    return m


def _uniform(mask: torch.Tensor) -> torch.Tensor:
    return mask / mask.sum(dim=0, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def swap_curve(
    sim: torch.Tensor,
    targets: torch.Tensor,
    w_oracle: torch.Tensor,
    w_carprt: torch.Tensor,
    k: int = 3,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Accuracy at every point on the CARPRT-set -> oracle-set path."""
    p, c = w_oracle.shape
    k = min(int(k), p)

    carprt_order = w_carprt.argsort(dim=0, descending=True)
    oracle_order = w_oracle.argsort(dim=0, descending=True)
    oracle_top = oracle_order[:k]                       # (k, C)
    o_mask = torch.zeros_like(w_oracle).scatter_(0, oracle_top, 1.0)

    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_order = torch.argsort(torch.rand((p, c), generator=g), dim=0).to(w_oracle.device)

    rows = []
    for j in range(k + 1):
        head = oracle_top[:j] if j else None
        m = _fill_to_k(head, carprt_order, k, p, c)

        r_head = rand_order[:j] if j else None
        r_m = _fill_to_k(r_head, carprt_order, k, p, c)

        rows.append({
            "j": j,
            "acc": _accuracy(sim, _uniform(m), targets),
            "acc_random": _accuracy(sim, _uniform(r_m), targets),
            "overlap": float((m * o_mask).sum(dim=0).mean()) / k,
            "overlap_random": float((r_m * o_mask).sum(dim=0).mean()) / k,
        })
    return rows


def curve_summary(rows: List[Dict[str, float]], carprt_full: float) -> Dict[str, float]:
    """Shape statistics: is the payoff linear in overlap, or back-loaded?

    half_gain_point  smallest j/k reaching 50% of the total gain. Linear -> 0.50,
                     convex -> above it. The headline shape number.
    gain_at_half     share of the gain realised half way along. Linear -> 0.50.
    j_breakeven      smallest j whose set beats full 247-prompt CARPRT. This is
                     the practical bar: how many of k prompts a selector has to
                     get exactly right before it is worth using at all.
    """
    k = rows[-1]["j"]
    a0, ak = rows[0]["acc"], rows[-1]["acc"]
    span = ak - a0

    fracs = [(r["acc"] - a0) / span if abs(span) > 1e-9 else 0.0 for r in rows]

    half = next((r["j"] / k for r, f in zip(rows, fracs) if f >= 0.5), 1.0)
    mid = rows[(k + 1) // 2]
    even = next((r["j"] for r in rows if r["acc"] >= carprt_full), None)

    return {
        "k": k,
        "acc_carprt_topk": a0,
        "acc_oracle_topk": ak,
        "span": span,
        "half_gain_point": half,
        "gain_at_half": fracs[(k + 1) // 2],
        "j_half": mid["j"],
        "j_breakeven": even,
        "gain_frac": fracs,
        "monotone": all(b >= a - 0.05 for a, b in zip(fracs, fracs[1:])),
    }


@torch.no_grad()
def overlay_point(sim: torch.Tensor, targets: torch.Tensor, scores: torch.Tensor,
                  w_oracle: torch.Tensor, k: int,
                  rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Where a real selector lands relative to the curve at the SAME overlap.

    The swap path fills its non-oracle slots with CARPRT's best remaining
    prompts, so the curve is "this much overlap, with good fillers". A real
    selector chooses its own fillers. Interpolating the curve at the selector's
    measured overlap therefore predicts what it would have scored had its misses
    been as good as CARPRT's; the residual isolates filler quality from overlap.

    residual < 0  the selector's non-oracle picks are WORSE than CARPRT's, and
                  its overlap gain is being spent paying for them.
    residual ~ 0  overlap is the whole story; the curve is the design target.
    """
    p, c = w_oracle.shape
    k = min(int(k), p)
    idx = scores.topk(k, dim=0).indices
    m = torch.zeros_like(w_oracle).scatter_(0, idx, 1.0)
    o_m = torch.zeros_like(w_oracle).scatter_(
        0, w_oracle.topk(k, dim=0).indices, 1.0)

    acc = _accuracy(sim, _uniform(m), targets)
    ov = float((m * o_m).sum(dim=0).mean()) / k

    xs = [r["overlap"] for r in rows]
    ys = [r["acc"] for r in rows]
    if ov <= xs[0]:
        pred = ys[0]
    elif ov >= xs[-1]:
        pred = ys[-1]
    else:
        i = max(j for j in range(len(xs)) if xs[j] <= ov)
        i = min(i, len(xs) - 2)
        span = xs[i + 1] - xs[i]
        t = 0.0 if span < 1e-9 else (ov - xs[i]) / span
        pred = ys[i] + t * (ys[i + 1] - ys[i])
    return {"k": k, "acc": acc, "overlap": ov, "pred": pred,
            "residual": acc - pred}


def print_curve(rows, summ: Dict[str, float], carprt_full: float,
                oracle_full: float) -> None:
    k = summ["k"]
    print(f"\n  swap curve, k={k}   (j = how many of the {k} prompts are the "
          f"oracle's; the rest are CARPRT's best)")
    hdr = (f"{'j':>4}{'acc':>9}{'vs CARPRT':>11}{'gain frac':>11}"
           f"{'overlap':>9}  |{'random j':>10}{'vs CARPRT':>11}")
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r, f in zip(rows, summ["gain_frac"]):
        star = " <- breaks even" if r["j"] == summ["j_breakeven"] else ""
        print(f"  {r['j']:>4}{r['acc']:>9.2f}{r['acc'] - carprt_full:>+11.2f}"
              f"{f:>11.2f}{100 * r['overlap']:>8.0f}%  |"
              f"{r['acc_random']:>10.2f}{r['acc_random'] - carprt_full:>+11.2f}{star}")
    be = summ["j_breakeven"]
    print(f"\n  full 247-prompt CARPRT {carprt_full:.2f}   unrestricted oracle "
          f"{oracle_full:.2f}")
    print(f"  half-gain point {summ['half_gain_point']:.2f} of k "
          f"(linear = 0.50, convex > 0.50)   gain at j={summ['j_half']}: "
          f"{summ['gain_at_half']:.2f}")
    print(f"  break-even: {'j=' + str(be) if be is not None else 'never'}"
          + (f" -- a selector must place {be}/{k} prompts exactly right just to "
             f"match CARPRT" if be else " -- no partial set beats CARPRT"))
    if not summ["monotone"]:
        print("  [!] curve is not monotone in j")


def print_overlay(pts: Dict[str, Dict[str, float]], carprt_full: float) -> None:
    """Real selectors placed against the curve at their own overlap."""
    hdr = (f"{'selector':<12}{'overlap':>9}{'acc':>9}{'vs CARPRT':>11}"
           f"{'curve says':>12}{'residual':>10}")
    print("\n  " + hdr)
    print("  " + "-" * len(hdr))
    for nm, r in pts.items():
        print(f"  {nm:<12}{100 * r['overlap']:>8.0f}%{r['acc']:>9.2f}"
              f"{r['acc'] - carprt_full:>+11.2f}{r['pred']:>12.2f}"
              f"{r['residual']:>+10.2f}")
    print("  residual = how much the selector loses purely because its NON-oracle "
          "picks\n  are worse than the CARPRT prompts the curve falls back on.")


def print_summary(all_summ: Dict[str, Dict[str, float]]) -> None:
    hdr = (f"{'dataset':<16}{'k':>3}{'CARPRT':>9}{'top-k':>8}{'oracle-k':>10}"
           f"{'span':>8}{'half-gain':>11}{'gain@half':>11}{'breakeven':>11}")
    print(hdr); print("-" * len(hdr))
    halves, evens = [], []
    for name, s in all_summ.items():
        be = s["j_breakeven"]
        halves.append(s["half_gain_point"])
        evens.append(be / s["k"] if be is not None else 1.0)
        print(f"{name:<16}{s['k']:>3}{s['carprt_full']:>9.2f}"
              f"{s['acc_carprt_topk']:>8.2f}{s['acc_oracle_topk']:>10.2f}"
              f"{s['span']:>+8.2f}{s['half_gain_point']:>11.2f}"
              f"{s['gain_at_half']:>11.2f}"
              f"{(str(be) + '/' + str(s['k'])) if be is not None else 'never':>11}")
    if not halves:
        return
    mh = sum(halves) / len(halves)
    me = sum(evens) / len(evens)
    convex = sum(1 for h in halves if h >= 0.7)
    print(f"\n  mean half-gain point {mh:.2f} of k   "
          f"(linear = 0.50; >= 0.70 on {convex}/{len(halves)} datasets)")
    print(f"  mean break-even {me:.2f} of k -- the share of a k-prompt set a "
          f"selector must get exactly right to match CARPRT")
    if mh >= 0.7:
        print("\n  CONVEX. Partial selection accuracy is worth nearly nothing; the "
              "payoff arrives\n  only when the set is almost exactly right. This "
              "explains the `select` null result\n  and quantifies the bar any "
              "selector has to clear.")
    elif mh <= 0.55:
        print("\n  LINEAR. Improving overlap should convert to accuracy "
              "proportionally, so `pseudo`\n  raising overlap 15% -> 47% with no "
              "gain is NOT explained by set geometry.\n  Something else is wrong "
              "-- most likely the full-fit oracle's sets do not generalise.")
    else:
        print("\n  Intermediate. Neither reading is clean; check the per-dataset "
              "curves before\n  committing to a selector.")
