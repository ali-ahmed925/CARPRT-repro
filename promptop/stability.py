"""Is there a stable target to find at all?

Everything proposed downstream assumes the oracle's best prompt for a class is a
real property of the data that a label-free signal could detect. If instead it is
an artifact of the particular images the oracle was fitted to, there is nothing
to detect and no selector can work. That is cheap to settle and it settles the
whole direction, so it runs before any method is written.

Split the images in half. Fit the oracle on A, fit it again on B, and ask:

  IDENTITY   does argmax_i W_A[i,c] equal argmax_i W_B[i,c]?  Chance is 1/247.
  VALUE      where does A's pick land in B's ranking?  Chance is the 50th
             percentile. A pick can be functionally right while not identical --
             if A's choice sits in B's top 2%, the disagreement is between near
             equals and does not matter.

Then the measurement that actually decides it:

  TRANSFER   force A's oracle picks into CARPRT's top-k and score on B, where
             CARPRT's own weights are estimated from B. A saw labels; B saw none
             of A. This is a selector with PERFECT information from an
             independent labeled sample -- a strict upper bound on any label-free
             selector. If it cannot beat CARPRT on B, nothing can, and the low
             break-even measured by `swapcurve` was an in-sample illusion.

The in-sample curve (B's own oracle, fitted on B, scored on B) runs alongside as
the ceiling, so the gap between the two is exactly the price of generalisation.
"""

from typing import Dict, List

import torch

from .infer import carprt_weights
from .oracle import _accuracy, oracle_optimal_w, split_indices
from .swap import _fill_to_k, _uniform


@torch.no_grad()
def agreement(w_a: torch.Tensor, w_b: torch.Tensor,
              ks: List[int] = (1, 3, 10)) -> Dict[str, float]:
    """How much do two independently fitted oracles agree per class?"""
    p, c = w_a.shape
    out = {}

    top1_a = w_a.argmax(dim=0)
    out["top1_identity"] = float((top1_a == w_b.argmax(dim=0)).float().mean())
    out["top1_chance"] = 1.0 / p

    # where A's pick lands in B's ranking: 1.0 = B's best, 0.0 = B's worst
    order_b = w_b.argsort(dim=0, descending=True)
    rank_of = torch.empty_like(order_b)
    rank_of.scatter_(0, order_b, torch.arange(p, device=w_a.device)
                     .unsqueeze(1).expand(p, c))
    r = rank_of.gather(0, top1_a.unsqueeze(0)).squeeze(0).float()
    out["value_percentile"] = float(1.0 - (r / max(p - 1, 1)).mean())
    out["value_median_rank"] = float(r.median()) + 1.0

    for k in ks:
        if k == 1:
            continue
        ma = torch.zeros_like(w_a).scatter_(0, w_a.topk(k, dim=0).indices, 1.0)
        mb = torch.zeros_like(w_b).scatter_(0, w_b.topk(k, dim=0).indices, 1.0)
        out[f"top{k}_overlap"] = float((ma * mb).sum(dim=0).mean()) / k
        out[f"top{k}_chance"] = k / p
    return out


@torch.no_grad()
def transfer_curve(
    sim_b: torch.Tensor,
    targets_b: torch.Tensor,
    w_head: torch.Tensor,
    w_carprt_b: torch.Tensor,
    k: int,
) -> List[Dict[str, float]]:
    """Force w_head's top-j into CARPRT's top-k, scored on B.

    Identical construction to swap.swap_curve, but the head may come from an
    oracle that never saw B. That is the whole point: it turns an in-sample
    ceiling into an out-of-sample one.
    """
    p, c = w_head.shape
    k = min(int(k), p)
    order = w_carprt_b.argsort(dim=0, descending=True)
    head = w_head.topk(k, dim=0).indices

    rows = []
    for j in range(k + 1):
        m = _fill_to_k(head[:j] if j else None, order, k, p, c)
        rows.append({"j": j, "acc": _accuracy(sim_b, _uniform(m), targets_b)})
    return rows


def run(                       # no no_grad: oracle_optimal_w optimises W by Adam
    sim: torch.Tensor,
    targets: torch.Tensor,
    img_f: torch.Tensor,
    tf: torch.Tensor,
    theta0: torch.Tensor,
    ks: List[int] = (3, 10),
    temp: float = 1.0,
    steps: int = 400,
    lr: float = 0.05,
    chunk: int = 512,
    seed: int = 0,
) -> Dict[str, object]:
    n = sim.shape[0]
    a_i, b_i = split_indices(n, seed, 0.5)

    w_a, _, _ = oracle_optimal_w(sim[a_i], targets[a_i], theta0, temp, steps, lr)
    w_b, _, _ = oracle_optimal_w(sim[b_i], targets[b_i], theta0, temp, steps, lr)

    # CARPRT re-estimated from B alone: label-free, so this is legitimate, and it
    # keeps the baseline on exactly the images the transfer is scored on.
    w_c_b = carprt_weights(img_f[b_i], tf, temp, chunk)
    base_b = _accuracy(sim[b_i], w_c_b, targets[b_i])

    out = {
        "n_a": len(a_i), "n_b": len(b_i),
        "carprt_b": base_b,
        "agree": agreement(w_a, w_b),
        "curves": {},
    }
    for k in ks:
        out["curves"][k] = {
            "transfer": transfer_curve(sim[b_i], targets[b_i], w_a, w_c_b, k),
            "insample": transfer_curve(sim[b_i], targets[b_i], w_b, w_c_b, k),
        }
    return out


def print_report(r: Dict[str, object]) -> None:
    a = r["agree"]
    print(f"\n  two oracles fitted on disjoint halves "
          f"(A n={r['n_a']}, B n={r['n_b']})\n")
    print(f"    top-1 identical         {100 * a['top1_identity']:>6.1f}%   "
          f"(chance {100 * a['top1_chance']:.1f}%)")
    print(f"    A's pick in B's ranking {100 * a['value_percentile']:>6.1f}th pct"
          f"  (chance 50.0th, median rank {a['value_median_rank']:.0f} of 247)")
    for k in (3, 10):
        if f"top{k}_overlap" in a:
            print(f"    top-{k} overlap{'':<10}{100 * a[f'top{k}_overlap']:>6.1f}%   "
                  f"(chance {100 * a[f'top{k}_chance']:.1f}%)")

    base = r["carprt_b"]
    for k, blk in r["curves"].items():
        tr, ins = blk["transfer"], blk["insample"]
        print(f"\n  k={k}: forcing the oracle's top-j into CARPRT's top-{k}, "
              f"scored on B")
        hdr = (f"{'j':>4}{'transfer (A->B)':>18}{'vs CARPRT':>11}"
               f"{'in-sample (B->B)':>19}{'vs CARPRT':>11}{'price':>8}")
        print("  " + hdr); print("  " + "-" * len(hdr))
        for t, i_ in zip(tr, ins):
            print(f"  {t['j']:>4}{t['acc']:>18.2f}{t['acc'] - base:>+11.2f}"
                  f"{i_['acc']:>19.2f}{i_['acc'] - base:>+11.2f}"
                  f"{i_['acc'] - t['acc']:>8.2f}")
        be = next((t["j"] for t in tr if t["acc"] >= base), None)
        print(f"\n    CARPRT on B (all 247): {base:.2f}")
        print(f"    transfer break-even: "
              f"{('j=' + str(be)) if be is not None else 'NEVER'}")


def verdict(all_r: Dict[str, Dict[str, object]], k: int = 10) -> None:
    hdr = (f"{'dataset':<16}{'top-1 id':>10}{'A in B pct':>12}{'top10 ovl':>11}"
           f"{'CARPRT(B)':>11}{'transfer j=1':>14}{'best transfer':>15}"
           f"{'break-even':>12}")
    print(hdr); print("-" * len(hdr))

    n_ok, gains = 0, []
    for name, r in all_r.items():
        a, base = r["agree"], r["carprt_b"]
        blk = r["curves"].get(k) or list(r["curves"].values())[-1]
        tr = blk["transfer"]
        best = max(t["acc"] for t in tr)
        be = next((t["j"] for t in tr if t["acc"] >= base), None)
        ok = be is not None
        n_ok += ok
        gains.append(best - base)
        print(f"{name:<16}{100 * a['top1_identity']:>9.1f}%"
              f"{100 * a['value_percentile']:>11.1f}%"
              f"{100 * a.get('top10_overlap', float('nan')):>10.1f}%"
              f"{base:>11.2f}{tr[1]['acc'] - base:>+14.2f}{best - base:>+15.2f}"
              f"{('j=' + str(be)) if ok else 'never':>12}")

    if not gains:
        return
    mg = sum(gains) / len(gains)
    print(f"\n  transfer beats CARPRT on {n_ok}/{len(gains)} datasets, "
          f"mean best gain {mg:+.2f}")
    print("  'transfer' had ground-truth labels on a DISJOINT half. It is an upper "
          "bound on\n  any label-free selector, not a method.")

    if n_ok >= 0.75 * len(gains) and mg >= 1.0:
        print("\n  >>> THE TARGET IS REAL AND IT TRANSFERS. Prompt choices fitted on "
              "one sample\n      carry to unseen images and beat CARPRT there. A "
              "label-free selector has\n      something to aim at; the remaining "
              "question is only how close a signal\n      can get to it.")
    elif n_ok <= 0.25 * len(gains) or mg <= 0.0:
        print("\n  >>> THE TARGET DOES NOT TRANSFER. Even a selector holding real "
              "labels on an\n      independent half cannot beat CARPRT on unseen "
              "images. The break-even\n      measured by `swapcurve` was in-sample "
              "only, and no label-free selector\n      can do better. Stop here -- "
              "selection is not the direction.")
    else:
        print("\n  >>> MIXED. Transfer works on some datasets and not others; read "
              "the per-dataset\n      rows before committing. Check whether the "
              "successes are the datasets with\n      real headroom (eurosat, "
              "ucf101, dtd, oxford_flowers).")
