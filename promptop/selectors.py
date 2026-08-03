"""Label-free per-class prompt selection.

The measurements that shaped this, all from `swapcurve` and `stability`:

  * Accuracy is LINEAR in overlap with the oracle's set (mean half-gain point
    0.51 over 24 configs), so partial credit exists -- a selector does not have
    to be nearly right.
  * The target is real and it TRANSFERS: two oracles fitted on disjoint halves
    pick the same top prompt for 18.9% of classes against 0.4% chance, and picks
    made on one half beat CARPRT on the other by +1.80.
  * Concentration risk scales as 1/k. At k=3 a wrong pick costs -4.20 against
    +0.92 for a right one; at k=10 the ratio is about 1.2:1. So k=10.
  * CARPRT's ranking is a good FALLBACK even where it is a bad ranking. The swap
    curves only break even so cheaply because their non-oracle slots are filled
    by CARPRT. `pseudo` discarded those fillers and landed ~1 point below the
    curve at matched overlap.

So the shape of a method here is fixed: keep CARPRT's top-k, override only the
first j slots, and abstain per class where the signal has no evidence. What is
left to choose is the score, and every one below avoids CARPRT's two structural
defects -- it ranks by CONFIDENCE (mean winning similarity), and it scores each
prompt on the images that prompt itself selected.

  margin    Rival margin. CARPRT asks "how confident is prompt i when it picks
            class c"; a prompt with uniformly high similarity to everything
            scores well and separates nothing. Accuracy only changes when the
            top-1/top-2 order flips, so score instead the margin prompt i puts
            between c and the class actually competing for that image. Cells come
            from the ENSEMBLE's argmax, not the prompt's own, which removes the
            self-selection. Centred per prompt across classes to strip the
            class-agnostic part, measured to carry 94% of the energy but deliver
            only 22% of the headroom.

  spread    Separability signature. Uses no labels AND no pseudo-labels: a
            discriminative prompt makes a subgroup of images score high and the
            rest low, a useless one scores everything alike. Since all 247 text
            embeddings are L2-normalised to the same length before scaling, their
            similarity spreads are directly comparable. Score = mean of the top
            N/C similarities minus the overall mean.

  crossfit  Cross-fit consensus. CARPRT judges prompt i on images prompt i itself
            argmaxed to c -- textbook self-selection. Here the cell labels come
            from a random half of the prompt pool and only the OTHER half is
            scored against them, averaged over many splits, so no prompt ever
            confirms its own decisions. Reduces to CARPRT's shape when the two
            halves coincide.

Nothing here reads `targets`. Labels appear only in the oracle row of the
evaluation table, as a ceiling.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .oracle import _accuracy
from .swap import _fill_to_k, _uniform


@torch.no_grad()
def margin_scores(sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rival margin per (prompt, class). Returns (scores, valid-class mask)."""
    n, p, c = sim.shape
    ens = sim.mean(dim=1)                                   # (N,C) uniform ensemble
    top2 = ens.topk(2, dim=1).indices
    a, r = top2[:, 0], top2[:, 1]

    s_a = sim.gather(2, a.view(n, 1, 1).expand(n, p, 1)).squeeze(2)
    s_r = sim.gather(2, r.view(n, 1, 1).expand(n, p, 1)).squeeze(2)
    d = (s_a - s_r).t().contiguous()                        # (P,N)

    g = torch.zeros((p, c), device=sim.device)
    g.index_add_(1, a, d)
    cnt = torch.zeros(c, device=sim.device).index_add_(
        0, a, torch.ones(n, device=sim.device))
    g = g / cnt.clamp_min(1.0).unsqueeze(0)

    return g - g.mean(dim=1, keepdim=True), cnt > 0


@torch.no_grad()
def spread_scores(sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """How sharply prompt i singles out a subgroup for class c. No labels at all."""
    n, p, c = sim.shape
    q = max(1, min(n, n // max(c, 1)))                      # expected class size
    top = sim.topk(q, dim=0).values.mean(dim=0)             # (P,C)
    s = top - sim.mean(dim=0)
    return s - s.mean(dim=1, keepdim=True), torch.ones(c, dtype=torch.bool,
                                                       device=sim.device)


@torch.no_grad()
def crossfit_scores(sim: torch.Tensor, n_splits: int = 10,
                    seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score each prompt against a consensus built from prompts it is not in."""
    n, p, c = sim.shape
    acc = torch.zeros((p, c), device=sim.device)
    cnt = torch.zeros((p, c), device=sim.device)
    ones_n = torch.ones(n, device=sim.device)
    g = torch.Generator(device="cpu").manual_seed(seed)

    for _ in range(n_splits):
        held = torch.randperm(p, generator=g)[: p // 2].to(sim.device)
        lbl = sim[:, held, :].mean(dim=1).argmax(dim=1)      # consensus from held

        s_l = sim.gather(2, lbl.view(n, 1, 1).expand(n, p, 1)).squeeze(2)
        d = (s_l - sim.mean(dim=2)).t().contiguous()         # (P,N) margin vs class mean

        contrib = torch.zeros((p, c), device=sim.device).index_add_(1, lbl, d)
        counts = torch.zeros(c, device=sim.device).index_add_(0, lbl, ones_n)

        scored = torch.ones(p, device=sim.device)
        scored[held] = 0.0                                   # only the complement
        acc += contrib * scored.unsqueeze(1)
        cnt += counts.unsqueeze(0) * scored.unsqueeze(1)

    s = acc / cnt.clamp_min(1.0)
    return s - s.mean(dim=1, keepdim=True), (cnt.sum(dim=0) > 0)


def build_scores(sim: torch.Tensor, n_splits: int = 10,
                 seed: int = 0) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    return {
        "margin": margin_scores(sim),
        "spread": spread_scores(sim),
        "crossfit": crossfit_scores(sim, n_splits, seed),
    }


@torch.no_grad()
def override_curve(
    sim: torch.Tensor,
    targets: torch.Tensor,
    scores: torch.Tensor,
    valid: Optional[torch.Tensor],
    w_carprt: torch.Tensor,
    w_oracle: torch.Tensor,
    k: int = 10,
) -> List[Dict[str, float]]:
    """Override CARPRT's first j slots with the selector's top-j; fill the rest.

    j=0 is CARPRT's own top-k for every selector, so the whole table shares an
    origin and the columns are directly comparable. Classes the selector has no
    evidence for keep CARPRT's picks -- abstention, not a guess.
    """
    p, c = w_carprt.shape
    k = min(int(k), p)
    order = w_carprt.argsort(dim=0, descending=True)
    head = scores.topk(k, dim=0).indices                     # (k,C)

    if valid is not None and not bool(valid.all()):
        fallback = order[:k]
        head = torch.where(valid.unsqueeze(0), head, fallback)

    o_mask = torch.zeros_like(w_oracle).scatter_(
        0, w_oracle.topk(k, dim=0).indices, 1.0)

    rows = []
    for j in range(k + 1):
        m = _fill_to_k(head[:j] if j else None, order, k, p, c)
        prec = (float(o_mask.gather(0, head[:j]).sum(dim=0).mean()) / j
                if j else float("nan"))
        rows.append({"j": j, "acc": _accuracy(sim, _uniform(m), targets),
                     "precision": prec})
    return rows


@torch.no_grad()
def confidence(scores: torch.Tensor) -> torch.Tensor:
    """Per class, how decisively the signal picks its favourite prompt.

    z-score of the best prompt within its own class column, so it is comparable
    across classes whose scores live on different scales.
    """
    mu, sd = scores.mean(dim=0), scores.std(dim=0).clamp_min(1e-9)
    return (scores.max(dim=0).values - mu) / sd


@torch.no_grad()
def abstain_curve(
    sim: torch.Tensor,
    targets: torch.Tensor,
    scores: torch.Tensor,
    valid: Optional[torch.Tensor],
    w_carprt: torch.Tensor,
    w_oracle: torch.Tensor,
    k: int = 10,
    j: int = 1,
    fracs: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
) -> List[Dict[str, float]]:
    """Override only the most confident fraction of classes; keep CARPRT elsewhere.

    At 26% precision a selector loses, because a correct pick is worth about +1.2
    and a wrong one costs about -0.8, putting break-even near 40%. Abstention is
    the only lever that raises precision without a better score: if confidence
    predicts correctness, the confident classes clear 40% even though the average
    does not. If precision is FLAT in confidence, no abstention rule can work and
    the signal is finished.
    """
    p, c = w_carprt.shape
    k, j = min(int(k), p), min(int(j), k)
    order = w_carprt.argsort(dim=0, descending=True)
    head = scores.topk(k, dim=0).indices
    if valid is not None and not bool(valid.all()):
        head = torch.where(valid.unsqueeze(0), head, order[:k])

    o_mask = torch.zeros_like(w_oracle).scatter_(
        0, w_oracle.topk(k, dim=0).indices, 1.0)
    conf = confidence(scores)
    if valid is not None:
        conf = torch.where(valid, conf, torch.full_like(conf, -float("inf")))
    rank = conf.argsort(descending=True)

    rows = []
    for f in fracs:
        n_on = int(round(f * c))
        on = torch.zeros(c, dtype=torch.bool, device=scores.device)
        if n_on:
            on[rank[:n_on]] = True
        h = torch.where(on.unsqueeze(0), head, order[:k])
        m = _fill_to_k(h[:j] if j else None, order, k, p, c)
        prec = (float(o_mask.gather(0, head[:j])[:, on].mean())
                if n_on else float("nan"))
        rows.append({"frac": f, "n_classes": n_on,
                     "acc": _accuracy(sim, _uniform(m), targets),
                     "precision": prec})
    return rows


def print_abstain(res: Dict[str, List[Dict[str, float]]], carprt_full: float,
                  k: int, j: int, p: int = 247) -> None:
    fr = [r["frac"] for r in next(iter(res.values()))]
    hdr = f"{'selector':<11}" + "".join(f"{'top ' + str(int(100 * f)) + '%':>16}"
                                        for f in fr)
    print(f"\n  overriding only the most confident classes (j={j}, k={k})")
    print("  " + hdr)
    print(f"  {'':<11}" + "".join(f"{'acc   prec':>16}" for _ in fr))
    print("  " + "-" * len(hdr))
    for name, rows in res.items():
        line = f"{name:<11}"
        for r in rows:
            pr = "  -- " if r["n_classes"] == 0 else f"{100 * r['precision']:>4.0f}%"
            line += f"{r['acc']:>11.2f}{pr}"
        print("  " + line)
    print(f"\n  CARPRT {carprt_full:.2f}   'top 0%' abstains everywhere and must "
          f"equal CARPRT's top-{k}")
    print(f"  break-even precision is ~40% (a right pick ~+1.2, a wrong one "
          f"~-0.8); chance is {100 * k / p:.0f}%")


@torch.no_grad()
def evaluate(
    sim: torch.Tensor,
    targets: torch.Tensor,
    w_carprt: torch.Tensor,
    w_oracle: torch.Tensor,
    k: int = 10,
    n_splits: int = 10,
    seed: int = 0,
) -> Dict[str, List[Dict[str, float]]]:
    out = {}
    for name, (sc, valid) in build_scores(sim, n_splits, seed).items():
        out[name] = override_curve(sim, targets, sc, valid, w_carprt, w_oracle, k)
    out["ORACLE"] = override_curve(sim, targets, w_oracle, None,
                                   w_carprt, w_oracle, k)
    return out


def print_table(res: Dict[str, List[Dict[str, float]]], carprt_full: float,
                k: int, p: int = 247) -> Dict[str, Dict[str, float]]:
    js = [0, 1, 2, 3, 5, k]
    js = sorted({j for j in js if j <= k})

    hdr = f"{'selector':<11}" + "".join(f"{'j=' + str(j):>16}" for j in js)
    print("  " + hdr)
    print(f"  {'':<11}" + "".join(f"{'acc   prec':>16}" for _ in js))
    print("  " + "-" * len(hdr))

    best = {}
    for name, rows in res.items():
        line = f"{name:<11}"
        for j in js:
            r = rows[j]
            pr = "  -- " if j == 0 else f"{100 * r['precision']:>4.0f}%"
            line += f"{r['acc']:>11.2f}{pr}"
        print("  " + line)
        free = [r for r in rows if r["j"] > 0]
        b = max(free, key=lambda r: r["acc"])
        best[name] = {"j": b["j"], "acc": b["acc"],
                      "delta": b["acc"] - carprt_full,
                      "precision": b["precision"]}

    print(f"\n  CARPRT (all {p} prompts) {carprt_full:.2f}   "
          f"j=0 is CARPRT's own top-{k}   chance precision {100 * k / p:.0f}%")
    return best


def print_summary(all_best: Dict[str, Dict[str, Dict[str, float]]],
                  bases: Dict[str, float]) -> None:
    names = sorted({n for b in all_best.values() for n in b})
    names = [n for n in names if n != "ORACLE"] + (
        ["ORACLE"] if any("ORACLE" in b for b in all_best.values()) else [])

    hdr = f"{'dataset':<16}{'CARPRT':>9}" + "".join(f"{n:>18}" for n in names)
    print(hdr)
    print(f"{'':<16}{'':>9}" + "".join(f"{'delta  j  prec':>18}" for _ in names))
    print("-" * len(hdr))

    tot = {n: [] for n in names}
    for ds, best in all_best.items():
        line = f"{ds:<16}{bases[ds]:>9.2f}"
        for n in names:
            b = best.get(n)
            if b is None:
                line += f"{'--':>18}"
                continue
            tot[n].append(b["delta"])
            line += f"{b['delta']:>+9.2f}{b['j']:>3}{100 * b['precision']:>5.0f}%"
        print(line)

    print()
    for n in names:
        v = tot[n]
        if not v:
            continue
        wins = sum(1 for d in v if d > 0)
        print(f"  {n:<10} mean {sum(v) / len(v):+.2f} over {len(v)} datasets, "
              f"positive on {wins}/{len(v)}")
    print("\n  Per-dataset best j is an ORACLE over j -- read the mean as an upper "
          "bound.\n  A method must also win at ONE fixed j across every dataset.")
