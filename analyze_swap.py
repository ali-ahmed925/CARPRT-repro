"""What does each additional ORACLE prompt actually buy?

`swapcurve` reports the shape of the path; this reads its json back and reports the
thing that sets the design: how much accuracy the FIRST correctly-identified prompt
delivers, and how fast the returns fall off after it.

Break-even at j=1 says one right pick beats CARPRT. If that pick is worth a point or
more, the method target is per-class top-1 identification -- precision@1 against the
oracle's best prompt -- rather than selecting a set. If it is worth 0.1, the low
break-even is an accounting artifact of a set that starts below CARPRT, and the
target stays a set.

    python analyze_swap.py swap_all.json [--k 10]
"""

import argparse
import json


def curve(summ):
    """acc at every j, reconstructed from the stored gain fractions."""
    a0, span = summ["acc_carprt_topk"], summ["span"]
    return [a0 + f * span for f in summ["gain_frac"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    blocks = json.load(open(a.path))["swapcurve"]
    rows = []
    for key, s in blocks.items():
        if int(s["k"]) != a.k:
            continue
        acc = curve(s)
        base = s["carprt_full"]
        rows.append((key.split(" k=")[0], base, acc, s["oracle_full"],
                     s["j_breakeven"]))

    hdr = (f"{'dataset':<16}{'CARPRT':>8}{'top-k':>8}{'+1st':>8}{'+2nd':>8}"
           f"{'+3rd':>8}{'all k':>8}{'|':>3}{'d1':>7}{'d2':>7}{'d3':>7}{'rest':>7}")
    print(f"\nk={a.k}: accuracy as the oracle's best prompts are forced in\n")
    print(hdr); print("-" * len(hdr))

    firsts, wins = [], 0
    for name, base, acc, orc, be in rows:
        d = [acc[j] - base for j in range(len(acc))]
        marg = [acc[j] - acc[j - 1] for j in range(1, len(acc))]
        firsts.append(marg[0])
        wins += acc[1] >= base
        print(f"{name:<16}{base:>8.2f}{acc[0]:>8.2f}{acc[1]:>8.2f}{acc[2]:>8.2f}"
              f"{acc[3]:>8.2f}{acc[-1]:>8.2f}{'|':>3}"
              f"{marg[0]:>+7.2f}{marg[1]:>+7.2f}{marg[2]:>+7.2f}"
              f"{sum(marg[3:]):>+7.2f}")

    if not firsts:
        print(f"  no blocks with k={a.k}")
        return
    mf = sum(firsts) / len(firsts)
    print(f"\n  mean value of the 1st oracle prompt: {mf:+.2f}   "
          f"(beats full CARPRT on {wins}/{len(rows)})")
    print("  d1/d2/d3 = marginal gain of the 1st/2nd/3rd forced prompt; "
          "'rest' = all remaining together.")
    if mf >= 1.0:
        print("\n  >>> Per-class TOP-1 identification is the target. One right pick is "
              "worth\n      a point or more, and the fillers can stay CARPRT's.")
    else:
        print("\n  >>> The 1st prompt is worth little on its own; the low break-even "
              "comes from\n      CARPRT's top-k starting BELOW full CARPRT. Target "
              "stays a set, not a top-1.")


if __name__ == "__main__":
    main()
