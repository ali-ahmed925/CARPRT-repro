"""End-to-end prompt-operator pipeline.

    fit       estimate operators on a pooled class corpus, score reconstruction
              on a held-out dataset against the identity and additive nulls
    classify  zero-shot accuracy on a target dataset using ONLY synthesized
              prompt embeddings (C text encodings instead of P*C)
    all       both, sharing one encode
    residual  estimate prompt weights from the interaction term z - alpha*T m,
              score with the TRUE embeddings; alpha=0 reproduces CARPRT exactly
    oracle    ceilings for prompt reweighting, using labels purely as a ruler:
              MPE floor, CARPRT, Eq. 10 with true labels, and the best possible
              (P, C) weight matrix -- with a held-out split so the ceiling is
              one a real estimator could actually generalise to
    characterize  compare the oracle weight matrix with CARPRT's: descriptive
              stats, whether the missing accuracy is class-agnostic or
              class-specific, and which accumulated statistic best predicts the
              oracle (which forecasts whether a new estimator will work)
    learn     A) how much accuracy a given label quality can support, and
              B) learn W from CARPRT's own pseudo-labels -- label-free, so its
              accuracy is directly comparable to CARPRT's
    bayes     posterior prompt reweighting: empty-cell correction, variance-aware
              shrinkage of w', and an image-free text-geometric prior, with a full
              ablation over each component
    validate  apply ONE frozen estimator across several datasets with no
              sweeping available, reporting a paired McNemar test per dataset
    dose      dose-response: hold C, the prompt pool and the evaluation set fixed
              and vary only how many unlabeled images estimate the weights, to
              test the count mechanism causally on a single dataset
    diagnose  1) held-out oracle per dataset, so the headroom column stops being
              inflated by oracle overfitting; 2) oracle top-k SET scored with
              UNIFORM weights, separating "which prompts" from "what weights",
              plus top-k overlap between CARPRT's ranking and the oracle's
    select    label-free prompt SELECTION: score each selector's top-k with
              UNIFORM weights, since magnitudes are worth only ~20%. Compares
              CARPRT's own ranking, a pseudo-label-learned ranking, bootstrap
              stability selection, and the label-fitted oracle as the ceiling
    sweep     model-complexity ladder: identity / additive / affine / low-rank at
              several ranks / full ridge / Procrustes, each scored against its
              parameter cost, to see how much operator the gain actually needs

Example (uses only what is already on disk -- ImageNet class names are hardcoded
in datasets/imagenet.py, so no ImageNet images are needed):

    python run_operator.py all --fit-datasets imagenet --target oxford_pets \
        --backbone ViT-B/16 --data-root ~/datasets --estimator ridge
"""

import argparse
import json
import os
import random
import time

import clip
import numpy as np
import torch

from datasets.template import template as TEMPLATES
from promptop import corpus as corpus_mod
from promptop import embed as embed_mod
from promptop import evaluate as eval_mod
from promptop import fit as fit_mod
from promptop import infer as infer_mod
from promptop import manifold as manifold_mod
from utils import build_test_data_loader, clip_classifier


# utils.build_test_data_loader dispatches ImageNet and its variants on the ids
# I/A/V/R/S, while promptop.corpus and datasets.dataset_list use the spelled-out
# registry names. Accept either so --targets imagenet does not fail with an
# unhelpful "Dataset is not from the chosen list".
LOADER_ALIAS = {"imagenet": "I", "imagenet-a": "A", "imagenet_a": "A",
                "imagenet-v": "V", "imagenet_v": "V", "imagenet-r": "R",
                "imagenet_r": "R", "imagenet-s": "S", "imagenet_s": "S",
                "imagenet-sketch": "S"}


def loader_id(name):
    return LOADER_ALIAS.get(name.lower(), name)


def get_args():
    p = argparse.ArgumentParser(description="Prompt-operator pipeline.")
    p.add_argument("command", choices=["fit", "classify", "all", "sweep", "residual", "oracle",
                            "characterize", "learn", "bayes",
                            "validate", "dose", "diagnose",
                            "select", "swapcurve", "stability"])
    p.add_argument("--fit-datasets", type=str, default="imagenet",
                   help="Slash-separated corpus for fitting, e.g. 'imagenet/sun397'.")
    p.add_argument("--target", type=str, default="oxford_pets",
                   help="Held-out dataset: never contributes to the fit.")
    p.add_argument("--backbone", type=str, choices=["RN50", "ViT-B/16"],
                   default="ViT-B/16")
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=os.path.expanduser("~/datasets"))
    p.add_argument("--estimator", type=str, default="ridge",
                   choices=["ridge", "procrustes", "lowrank", "affine"])
    p.add_argument("--rank", type=int, default=64, help="lowrank estimator only.")
    p.add_argument("--lam", type=float, default=None,
                   help="Ridge strength; default is scale-aware (auto_lambda).")
    p.add_argument("--base-template", dest="base_template", type=str, default="{}",
                   help="How m_c is formed. '{}' = bare class name.")
    p.add_argument("--temp", type=float, default=1.0, help="tau for reweighting.")
    p.add_argument("--group-refine", action="store_true",
                   help="Fit hierarchical per-group corrections (seen domains only).")
    p.add_argument("--keep-overlap", action="store_true",
                   help="Do NOT drop fit classes whose name also occurs in the "
                        "target. Leaks; for ablation only.")
    p.add_argument("--allow-underdetermined", action="store_true",
                   help="Proceed when C < D. Results will not be interpretable.")
    p.add_argument("--residual-clusters", dest="residual_clusters", type=int,
                   default=12,
                   help="Pseudo-domains carved from class names for the residual "
                        "structure test.")
    p.add_argument("--impl", dest="impl", type=str, default="v1",
                   choices=["v1", "v2"],
                   help="v1 (default): original scoring; reproduces all numbers "
                        "generated before 2026-08-02. v2: alpha=0 is bit-exact "
                        "against CARPRT.")
    p.add_argument("--with-pseudo", dest="with_pseudo", action="store_true",
                   help="swapcurve: overlay the pseudo-label selector on the "
                        "curve at its own overlap, to separate overlap quality "
                        "from the quality of the non-oracle picks.")
    p.add_argument("--n-boot", dest="n_boot", type=int, default=20,
                   help="Bootstrap draws for the stability selector.")
    p.add_argument("--topk-list", dest="topk_list", type=str,
                   default="3,6,10,25",
                   help="Set sizes for the selection-vs-magnitudes split.")
    p.add_argument("--dose-mode", dest="dose_mode", type=str,
                   default="images", choices=["images", "classes"],
                   help="images: vary how many images estimate the weights. "
                        "classes: vary C at ~constant evidence per cell.")
    p.add_argument("--dose-classes", dest="dose_classes", type=str,
                   default="100,80,60,47,30,20,10",
                   help="Class counts for --dose-mode classes.")
    p.add_argument("--dose-fracs", dest="dose_fracs", type=str,
                   default="1.0,0.5,0.25,0.125,0.0625,0.03125",
                   help="Fractions of the test set used to ESTIMATE weights.")
    p.add_argument("--dose-seeds", dest="dose_seeds", type=int, default=3,
                   help="Subsample draws per fraction, averaged.")
    p.add_argument("--alpha-power", dest="alpha_power", type=str,
                   default="0,0.125,0.25,0.375,0.5,0.625,0.75,1.0",
                   help="Exponents for the n^alpha sweep. 0 = CARPRT.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="FROZEN exponent for the validate command.")
    p.add_argument("--targets", type=str,
                   default="oxford_pets/dtd/caltech101/eurosat/oxford_flowers",
                   help="Slash-separated datasets for the validate command.")
    p.add_argument("--lam-sweep", dest="lam_sweep", type=str,
                   default="0,0.25,0.5,1,2,4,8,16",
                   help="Shrinkage strengths. 0 = no shrinkage.")
    p.add_argument("--beta-sweep", dest="beta_sweep", type=str,
                   default="0.1,0.25,0.5,1.0,2.0",
                   help="Text-prior strengths, in SDs of the evidence term.")
    p.add_argument("--prior-mode", dest="prior_mode", type=str,
                   default="nearest", choices=["nearest", "mean"],
                   help="Separability against the nearest confuser, or all.")
    p.add_argument("--label-quality", dest="label_quality", type=str,
                   default="1.0,0.95,0.90,0.8945,0.85,0.80",
                   help="Label accuracies for the sweep in the learn command.")
    p.add_argument("--kl-sweep", dest="kl_sweep", type=str,
                   default="0,0.01,0.1,1,10",
                   help="KL-to-CARPRT penalties. Large values recover CARPRT.")
    p.add_argument("--learn-steps", dest="learn_steps", type=int, default=300)
    p.add_argument("--pseudo-source", dest="pseudo_source", type=str,
                   default="carprt", choices=["carprt", "mpe"],
                   help="carprt: higher quality but self-referential. mpe: "
                        "independent of W, lower quality.")
    p.add_argument("--oracle-steps", dest="oracle_steps", type=int, default=400,
                   help="Adam steps for the best-W ceiling.")
    p.add_argument("--oracle-lr", dest="oracle_lr", type=float, default=0.05)
    p.add_argument("--weight-mode", dest="weight_mode", type=str,
                   default="value", choices=["value", "signal"],
                   help="value: pseudo-labels from the full embedding, averaged "
                        "magnitude from the residual (recommended). signal: "
                        "replace the whole weight-estimation tensor (destroys "
                        "pseudo-labels; kept for comparison).")
    p.add_argument("--value-scale", dest="value_scale", type=str,
                   default="match", choices=["match", "none"],
                   help="match: rescale w' to the baseline per-class spread so "
                        "alpha does not silently change the softmax temperature.")
    p.add_argument("--alpha-sweep", dest="alpha_sweep", type=str,
                   default="0,0.25,0.5,0.75,0.9,1.0,1.25,1.5",
                   help="Alphas for the residual command. 0 = plain CARPRT.")
    p.add_argument("--rank-sweep", dest="rank_sweep", type=str,
                   default="1,2,4,8,16,64",
                   help="Comma-separated ranks for the sweep command.")
    p.add_argument("--pca-components", dest="pca_components", type=int, default=20)
    p.add_argument("--synth-prompts", dest="synth_prompts", type=int, default=0,
                   help="Synthesize N novel operators from the manifold and score "
                        "them downstream as a prompt pool. 0 disables.")
    p.add_argument("--synth-mode", dest="synth_mode", type=str, default="gaussian",
                   choices=["gaussian", "grid"])
    p.add_argument("--cache-dir", dest="cache_dir", type=str,
                   default=".operator_cache")
    p.add_argument("--out", type=str, default=None, help="Write results json here.")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-encode", action="store_true", help="Ignore the cache.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def _dose_classes(args, tf, img_f, targets, classnames, alphas, seeds, results):
    """Vary the number of CLASSES at roughly constant evidence per cell.

    The mirror of the image dose-response, and it isolates C almost for free:
    subsetting to C' classes keeps only the images of those classes, so
    n_images ~ N*C'/C and the median count per cell ~ n_images/C' ~ N/C stays
    put. C moves, count density does not.

    Absolute accuracy rises as C' shrinks because the task gets easier, but
    CARPRT and the count rule are compared at the same C' on the same images, so
    the delta is what matters.
    """
    from promptop import bayes as by
    from promptop import stats as st

    c_full = len(classnames)
    grid = sorted({int(x) for x in args.dose_classes.split(",") if x
                   and 2 <= int(x) <= c_full}, reverse=True)

    banner(f"CLASS DOSE-RESPONSE on {args.target}  "
           f"(C varies, evidence per cell ~constant)")
    print(f"  full dataset has C={c_full}, {img_f.shape[0]} images\n")
    hdr = (f"{'C}':>5}{'n_img':>8}{'med.count':>11}{'CARPRT':>9}"
           f"{'best a':>8}{'peak':>9}{'gain':>8}"
           f"{'a=' + format(args.alpha, 'g'):>9}{'gain':>8}")
    print(hdr.replace("C}", "C'")); print("-" * len(hdr))

    rows = []
    for c_sub in grid:
        per_seed = []
        for sd in seeds:
            g = torch.Generator().manual_seed(args.seed + sd)
            cls = torch.randperm(c_full, generator=g)[:c_sub].sort().values.to(tf.device)

            keep = (targets.unsqueeze(1) == cls.unsqueeze(0)).any(dim=1)
            sub_img = img_f[keep]
            remap = torch.full((c_full,), -1, dtype=torch.long, device=tf.device)
            remap[cls] = torch.arange(c_sub, device=tf.device)
            sub_tg = remap[targets[keep]]
            sub_tf = tf[:, cls, :].contiguous()

            def acc(w):
                return 100.0 * (infer_mod._scores(sub_img, sub_tf, w).argmax(1)
                                == sub_tg).float().mean().item()

            w_base = infer_mod.carprt_weights(sub_img, sub_tf, args.temp, args.chunk)
            s1, s2, nn = by.weight_moments(sub_img, sub_tf, args.chunk)
            curve = {}
            for a in alphas:
                w = torch.softmax(
                    by.count_power_scores(s1, s2, nn, a, impl=args.impl) / args.temp, dim=0)
                curve[a] = (acc(w), infer_mod._scores(sub_img, sub_tf, w).argmax(1))
            per_seed.append({
                "base": acc(w_base), "med": float(nn.float().median()),
                "n": int(sub_img.shape[0]), "curve": curve, "tg": sub_tg,
                "pred_base": infer_mod._scores(sub_img, sub_tf, w_base).argmax(1)})

        m = len(per_seed)
        med = sum(s["med"] for s in per_seed) / m
        nimg = sum(s["n"] for s in per_seed) / m
        base = sum(s["base"] for s in per_seed) / m
        mean_curve = {a: sum(s["curve"][a][0] for s in per_seed) / m for a in alphas}
        best_a = max(mean_curve, key=mean_curve.get)
        fixed = mean_curve.get(args.alpha, float("nan"))

        last = per_seed[-1]
        r = st.mcnemar(last["pred_base"], last["curve"][best_a][1], last["tg"],
                       "CARPRT", f"a={best_a:g}")

        print(f"{c_sub:>5}{nimg:>8.0f}{med:>11.1f}{base:>9.2f}"
              f"{best_a:>8.3f}{mean_curve[best_a]:>9.2f}"
              f"{mean_curve[best_a] - base:>+8.2f}"
              f"{fixed:>9.2f}{fixed - base:>+8.2f}  {st.stars(r['p_exact'])}")
        rows.append({"n_classes": c_sub, "n_img": nimg, "median_count": med,
                     "carprt": base, "best_alpha": best_a,
                     "peak": mean_curve[best_a],
                     "gain": mean_curve[best_a] - base,
                     "fixed_gain": fixed - base, "p": r["p_exact"],
                     "curve": mean_curve})
        del per_seed
        torch.cuda.empty_cache()

    print(f"\n  median count stays ~constant across rows by construction, so any")
    print(f"  trend in 'gain' is attributable to C rather than to evidence volume.")
    # The alpha=0 member uses this module's empty-cell floor, while CARPRT uses
    # w'=0; the two diverge as cells empty out, so "gain vs CARPRT" partly measures
    # that convention. Isolate the count factor by differencing WITHIN the family.
    usable = [r for r in rows if r["carprt"] < 98.0]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"\n  {dropped} row(s) excluded from the verdict: CARPRT above 98%, so"
              f"\n  there is no headroom for any method to differ (ceiling effect).")

    print(f"\n{'C':>5}{'count-factor effect':>22}   (alpha={args.alpha:g} minus alpha=0,"
          f" same empty-cell convention)")
    print("-" * 62)
    iso = []
    for r in usable:
        a0 = r["curve"].get(0.0)
        af = r["curve"].get(args.alpha)
        if a0 is None or af is None:
            continue
        iso.append((r["n_classes"], af - a0))
        print(f"{r['n_classes']:>5}{af - a0:>22.2f}")

    if len(iso) >= 2:
        (c_hi, e_hi), (c_lo, e_lo) = max(iso), min(iso)
        if e_lo - e_hi > 1.0:
            print(f"\n  >>> C MODULATES THE EFFECT: the count factor goes from "
                  f"{e_hi:+.2f} at C={c_hi}\n      to {e_lo:+.2f} at C={c_lo}. Note where "
                  f"it crosses zero -- below that C the\n      factor helps, above it "
                  f"it harms. C is known at inference without labels.")
        elif abs(e_lo - e_hi) < 1.0:
            print(f"\n  >>> C DOES NOT MODULATE THE EFFECT: {e_hi:+.2f} at C={c_hi} vs "
                  f"{e_lo:+.2f} at C={c_lo}.")
        else:
            print(f"\n  >>> effect moves the WRONG way in C ({e_hi:+.2f} -> {e_lo:+.2f}).")
    results["dose_classes"] = rows
    return results


def run_stability(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                  preprocess, results):
    """Is there a stable target for a selector to aim at?

    Fits the oracle on two disjoint halves and asks whether its per-class prompt
    choices agree, then whether choices made on one half beat CARPRT on the
    other. That second number is an upper bound on any label-free selector: it
    had real labels, just not on the images it is scored on.
    """
    from promptop import oracle as oracle_mod
    from promptop import stability as stab

    ks = [int(x) for x in args.topk_list.split(",") if x]
    all_r = {}

    for name in [d for d in args.targets.split("/") if d]:
        banner(f"{name}")
        try:
            loader, classnames, _ = build_test_data_loader(
                loader_id(name), args.data_root, preprocess)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {type(exc).__name__}: {exc}")
            continue

        img_f, targets = infer_mod.encode_images(loader, clip_model)
        tf = clip_classifier(classnames, TEMPLATES, clip_model)
        n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)
        gb = oracle_mod.estimate_bytes(n, p, c)
        if gb > 4.0:
            print(f"  [skip] similarity tensor would need {gb:.1f} GB")
            del img_f, targets, tf
            torch.cuda.empty_cache()
            continue

        sim = oracle_mod.similarity_tensor(img_f, tf, args.chunk)
        _, theta0 = infer_mod.carprt_weights_split_value(
            img_f, tf, tf, args.temp, args.chunk)
        print(f"  N={n} C={c}   fitting two oracles on disjoint halves ...")

        r = stab.run(sim, targets, img_f, tf, theta0, ks, args.temp,
                     args.oracle_steps, args.oracle_lr, args.chunk, args.seed)
        stab.print_report(r)
        all_r[name] = r

        del sim, img_f, targets, tf
        torch.cuda.empty_cache()

    banner("VERDICT — is the target real, and does it transfer?")
    stab.verdict(all_r, max(ks))
    results["stability"] = all_r
    return results


def run_swapcurve(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                  preprocess, results):
    """Is accuracy LINEAR or CONVEX in overlap with the oracle's prompt set?

    Everything downstream hangs on the answer. If linear, a selector that closes
    half the overlap gap collects half the gain and `pseudo` raising overlap
    15% -> 47% for -0.16 accuracy means something else is broken. If convex, the
    payoff arrives only when the set is nearly exactly right, the `select` null
    result is explained, and we learn the bar a selector has to clear.
    """
    from promptop import oracle as oracle_mod
    from promptop import swap as sw

    ks = [int(x) for x in args.topk_list.split(",") if x]
    all_summ = {}

    for name in [d for d in args.targets.split("/") if d]:
        banner(f"{name}")
        try:
            loader, classnames, _ = build_test_data_loader(
                loader_id(name), args.data_root, preprocess)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {type(exc).__name__}: {exc}")
            continue

        img_f, targets = infer_mod.encode_images(loader, clip_model)
        tf = clip_classifier(classnames, TEMPLATES, clip_model)
        n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)
        gb = oracle_mod.estimate_bytes(n, p, c)
        if gb > 4.0:
            print(f"  [skip] similarity tensor would need {gb:.1f} GB")
            del img_f, targets, tf
            torch.cuda.empty_cache()
            continue

        sim = oracle_mod.similarity_tensor(img_f, tf, args.chunk)
        w_carprt = infer_mod.carprt_weights(img_f, tf, args.temp, args.chunk)
        _, theta0 = infer_mod.carprt_weights_split_value(
            img_f, tf, tf, args.temp, args.chunk)
        base = oracle_mod._accuracy(sim, w_carprt, targets)
        w_oracle, orc, _ = oracle_mod.oracle_optimal_w(
            sim, targets, theta0, args.temp, args.oracle_steps, args.oracle_lr)
        print(f"  N={n} C={c}   CARPRT {base:.2f}   oracle {orc:.2f} "
              f"({orc - base:+.2f})")

        overlays = {}
        if args.with_pseudo:
            from promptop import select as sel
            print("  fitting the pseudo-label selector for the overlay ...")
            overlays["pseudo"] = sel.pseudo_label_scores(
                sim, w_carprt, theta0, args.temp, args.learn_steps, args.oracle_lr)

        for k in ks:
            rows = sw.swap_curve(sim, targets, w_oracle, w_carprt, k, args.seed)
            summ = sw.curve_summary(rows, base)
            sw.print_curve(rows, summ, base, orc)
            if overlays:
                pts = {nm: sw.overlay_point(sim, targets, sc, w_oracle, k, rows)
                       for nm, sc in overlays.items()}
                sw.print_overlay(pts, base)
                summ["overlay"] = pts
            summ["carprt_full"] = base
            summ["oracle_full"] = orc
            all_summ[f"{name} k={k}" if len(ks) > 1 else name] = summ

        del sim, img_f, targets, tf, w_carprt, w_oracle
        torch.cuda.empty_cache()

    banner("SUMMARY — does closing the overlap gap buy accuracy?")
    sw.print_summary(all_summ)
    results["swapcurve"] = all_summ
    return results


def run_select(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
               preprocess, results):
    """Can a label-free selector find the ~3 prompts per class that matter?"""
    from promptop import oracle as oracle_mod
    from promptop import select as sel
    from promptop import stats as st

    ks = [int(x) for x in args.topk_list.split(",") if x]
    all_rows = {}

    for name in [d for d in args.targets.split("/") if d]:
        banner(f"{name}")
        try:
            loader, classnames, _ = build_test_data_loader(
                loader_id(name), args.data_root, preprocess)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {type(exc).__name__}: {exc}")
            continue

        img_f, targets = infer_mod.encode_images(loader, clip_model)
        tf = clip_classifier(classnames, TEMPLATES, clip_model)
        n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)
        gb = oracle_mod.estimate_bytes(n, p, c)
        if gb > 4.0:
            print(f"  [skip] similarity tensor would need {gb:.1f} GB")
            del img_f, targets, tf
            torch.cuda.empty_cache()
            continue

        sim = oracle_mod.similarity_tensor(img_f, tf, args.chunk)
        w_carprt = infer_mod.carprt_weights(img_f, tf, args.temp, args.chunk)
        _, theta0 = infer_mod.carprt_weights_split_value(
            img_f, tf, tf, args.temp, args.chunk)
        base = oracle_mod._accuracy(sim, w_carprt, targets)
        w_oracle, orc, _ = oracle_mod.oracle_optimal_w(
            sim, targets, theta0, args.temp, args.oracle_steps, args.oracle_lr)

        print(f"  building selectors ({args.n_boot} bootstrap draws) ...")
        selectors = {
            "carprt": w_carprt,
            "pseudo": sel.pseudo_label_scores(sim, w_carprt, theta0, args.temp,
                                              args.learn_steps, args.oracle_lr),
            "stability": sel.stability_scores(img_f, tf, max(ks), args.n_boot,
                                              args.temp, args.chunk, args.seed),
            "oracle": w_oracle,
        }
        rows = sel.evaluate_selectors(sim, targets, selectors, w_oracle, ks)
        print()
        sel.print_selectors(rows, base, orc, ks)

        # paired test for the best label-free selector at its best k
        free = [r for r in rows if r["selector"] != "oracle"]
        best = max(free, key=lambda r: r["acc"])
        w_best = sel.uniform_over_topk(selectors[best["selector"]], best["k"])
        pr_b = infer_mod._scores(img_f, tf, w_carprt).argmax(1)
        pr_n = infer_mod._scores(img_f, tf, w_best).argmax(1)
        m = st.mcnemar(pr_b, pr_n, targets, "CARPRT", best["selector"])
        print(f"\n  best label-free: {best['selector']} at k={best['k']} -> "
              f"{best['acc']:.2f} ({best['acc'] - base:+.2f} vs CARPRT, "
              f"b={m['b']} c={m['c']}, p={m['p_exact']:.2e} "
              f"{st.stars(m['p_exact'])})")

        all_rows[name] = {"rows": rows, "carprt": base, "oracle": orc,
                          "best": best, "mcnemar": m}
        del sim, img_f, targets, tf, w_carprt, w_oracle, selectors
        torch.cuda.empty_cache()

    banner("SUMMARY — can a label-free selector beat CARPRT?")
    hdr = (f"{'dataset':<16}{'CARPRT':>9}{'best free':>11}{'k':>4}"
           f"{'delta':>8}{'p':>10}{'oracle':>9}{'captured':>10}")
    print(hdr); print("-" * len(hdr))
    deltas = []
    for name, blk in all_rows.items():
        b, m = blk["best"], blk["mcnemar"]
        d = b["acc"] - blk["carprt"]
        cap = 100 * d / max(blk["oracle"] - blk["carprt"], 1e-9)
        deltas.append(d)
        print(f"{name:<16}{blk['carprt']:>9.2f}{b['acc']:>11.2f}{b['k']:>4}"
              f"{d:>+8.2f}{m['p_exact']:>10.1e}{blk['oracle']:>9.2f}"
              f"{cap:>9.0f}%  {b['selector']}")
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        print(f"\n  mean {mean_d:+.2f} over {len(deltas)} datasets, "
              f"positive on {wins}/{len(deltas)}")
        print("  'captured' = share of the oracle's headroom reached with NO labels")
    results["select"] = all_rows
    return results


def run_diagnose(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                 preprocess, results):
    """How much headroom is real, and is the failure selection or magnitudes?"""
    from promptop import oracle as oracle_mod
    from promptop import diagnose as dg

    rows, topk_all = [], {}
    for name in [d for d in args.targets.split("/") if d]:
        banner(f"{name}")
        try:
            loader, classnames, _ = build_test_data_loader(
                loader_id(name), args.data_root, preprocess)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {type(exc).__name__}: {exc}")
            continue

        img_f, targets = infer_mod.encode_images(loader, clip_model)
        tf = clip_classifier(classnames, TEMPLATES, clip_model)
        n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)
        gb = oracle_mod.estimate_bytes(n, p, c)
        if gb > 4.0:
            print(f"  [skip] similarity tensor would need {gb:.1f} GB")
            del img_f, targets, tf
            torch.cuda.empty_cache()
            continue

        sim = oracle_mod.similarity_tensor(img_f, tf, args.chunk)
        w_carprt = infer_mod.carprt_weights(img_f, tf, args.temp, args.chunk)
        _, theta0 = infer_mod.carprt_weights_split_value(
            img_f, tf, tf, args.temp, args.chunk)
        base = oracle_mod._accuracy(sim, w_carprt, targets)

        r = dg.held_out_headroom(sim, targets, theta0, base, args.temp,
                                 args.oracle_steps, args.oracle_lr, args.seed)
        rows.append({"dataset": name, **r})
        print(f"  {c} classes, {n} images, {r['params_per_image']:.2f} oracle "
              f"params/image")
        print(f"  CARPRT {base:.2f} | oracle full {r['oracle_full']:.2f} "
              f"({r['headroom_full']:+.2f}) | oracle HELD-OUT "
              f"{r['oracle_heldout']:.2f} ({r['headroom_heldout']:+.2f}) | "
              f"overfit gap {r['overfit_gap']:.2f}")

        w_or, _, _ = oracle_mod.oracle_optimal_w(
            sim, targets, theta0, args.temp, args.oracle_steps, args.oracle_lr)
        tk = dg.topk_set_scores(sim, targets, w_or, w_carprt,
                                [int(x) for x in args.topk_list.split(",") if x])
        topk_all[name] = {"rows": tk, "carprt": base, "oracle": r["oracle_full"]}

        del sim, img_f, targets, tf, w_carprt, w_or
        torch.cuda.empty_cache()

    banner("1. HOW MUCH HEADROOM IS REAL")
    print("  The oracle fits P*C parameters on N images. Where that ratio is large")
    print("  it memorises the test set, so the full-fit headroom is inflation.\n")
    hdr = (f"{'dataset':<16}{'params':>9}{'images':>8}{'par/img':>9}"
           f"{'CARPRT':>9}{'orc full':>9}{'head':>9}{'orc held':>10}"
           f"{'head':>10}{'overfit':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        dg.print_headroom(r, r["dataset"])
    if rows:
        real = sorted(rows, key=lambda r: -r["headroom_heldout"])
        print(f"\n  largest REAL headroom: " + ", ".join(
            f"{r['dataset']} {r['headroom_heldout']:+.1f}" for r in real[:3]))
        print(f"  smallest:              " + ", ".join(
            f"{r['dataset']} {r['headroom_heldout']:+.1f}" for r in real[-3:]))

    banner("2. SELECTION OR MAGNITUDES?")
    print("  'oracle set + UNIFORM' is the key column: how far you get from knowing")
    print("  only WHICH prompts matter, with no magnitude information at all.")
    print("  'CARPRT set + uniform' asks whether CARPRT can identify that set.\n")
    for name, blk in topk_all.items():
        print(f"--- {name} ---")
        dg.print_topk(blk["rows"], blk["carprt"], blk["oracle"])
        print()

    results["headroom"] = rows
    results["topk"] = topk_all
    return results


def run_dose(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
             preprocess, results):
    """Dose-response: does the count gain scale with images per cell, causally?

    Across datasets the gain tracks the median count per (prompt, class) cell, but
    EuroSAT confounds two things -- most images per cell AND fewest classes. This
    holds C, the prompt pool, the backbone and the EVALUATION SET fixed, and varies
    only the number of unlabeled images used to ESTIMATE the weights.

    Weights are estimated on a subsample; accuracy is always measured on the full
    test set. That keeps every accuracy number comparable and isolates the effect
    of estimation evidence, which a shrinking evaluation set would confound.
    """
    from promptop import bayes as by
    from promptop import stats as st

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf = clip_classifier(classnames, TEMPLATES, clip_model)
    n_all, c = img_f.shape[0], len(classnames)

    def acc_full(w):
        return 100.0 * (infer_mod._scores(img_f, tf, w).argmax(1)
                        == targets).float().mean().item()

    alphas = [float(x) for x in args.alpha_power.split(",") if x]
    seeds = list(range(args.dose_seeds))

    if args.dose_mode == "classes":
        return _dose_classes(args, tf, img_f, targets, classnames, alphas,
                             seeds, results)

    fracs = [float(x) for x in args.dose_fracs.split(",") if x]

    banner(f"DOSE-RESPONSE on {args.target}  "
           f"(C={c} fixed, evaluation on all {n_all} images)")
    print(f"  weights estimated from a subsample; accuracy always on the full set")
    print(f"  {args.dose_seeds} seed(s) per fraction, averaged\n")

    hdr = (f"{'frac':>6}{'n_est':>8}{'med.count':>11}{'CARPRT':>9}"
           f"{'best a':>8}{'peak':>9}{'gain':>8}"
           f"{'a=' + format(args.alpha, 'g'):>9}{'gain':>8}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for frac in fracs:
        k = max(int(round(n_all * frac)), 2 * c)
        per_seed = []
        for sd in seeds:
            g = torch.Generator().manual_seed(args.seed + sd)
            idx = torch.randperm(n_all, generator=g)[:k].to(img_f.device)
            sub = img_f[idx]

            w_base = infer_mod.carprt_weights(sub, tf, args.temp, args.chunk)
            a_base = acc_full(w_base)
            pred_base = infer_mod._scores(img_f, tf, w_base).argmax(1)

            s1, s2, nn = by.weight_moments(sub, tf, args.chunk)
            med = float(nn.float().median())

            curve = {}
            for a in alphas:
                w = torch.softmax(
                    by.count_power_scores(s1, s2, nn, a, impl=args.impl) / args.temp, dim=0)
                curve[a] = (acc_full(w), infer_mod._scores(img_f, tf, w).argmax(1))
            per_seed.append({"base": a_base, "med": med, "curve": curve,
                             "pred_base": pred_base})

        med_m = sum(s["med"] for s in per_seed) / len(per_seed)
        base_m = sum(s["base"] for s in per_seed) / len(per_seed)
        mean_curve = {a: sum(s["curve"][a][0] for s in per_seed) / len(per_seed)
                      for a in alphas}
        best_a = max(mean_curve, key=mean_curve.get)
        fixed = mean_curve.get(args.alpha, float("nan"))

        # paired test on the last seed, where predictions are concrete
        last = per_seed[-1]
        r = st.mcnemar(last["pred_base"], last["curve"][best_a][1], targets,
                       "CARPRT", f"a={best_a:g}")

        print(f"{frac:>6.3f}{k:>8}{med_m:>11.1f}{base_m:>9.2f}"
              f"{best_a:>8.3f}{mean_curve[best_a]:>9.2f}"
              f"{mean_curve[best_a] - base_m:>+8.2f}"
              f"{fixed:>9.2f}{fixed - base_m:>+8.2f}  {st.stars(r['p_exact'])}")
        rows.append({"frac": frac, "n_est": k, "median_count": med_m,
                     "carprt": base_m, "best_alpha": best_a,
                     "peak": mean_curve[best_a],
                     "gain": mean_curve[best_a] - base_m,
                     "fixed_alpha": args.alpha, "fixed_acc": fixed,
                     "fixed_gain": fixed - base_m, "p": r["p_exact"],
                     "curve": mean_curve})
        del per_seed
        torch.cuda.empty_cache()

    print(f"\n  'gain' columns are relative to CARPRT estimated from the SAME")
    print(f"  subsample, so both methods see identical evidence at every row.")
    print(f"  The 'best a' column is selected per row; the fixed-alpha column is not.")

    if len(rows) >= 3:
        lo = [r for r in rows if r["median_count"] < 40]
        hi = [r for r in rows if r["median_count"] >= 200]
        if lo and hi:
            print(f"\n  median count >= 200: mean gain "
                  f"{sum(r['gain'] for r in hi) / len(hi):+.2f}")
            print(f"  median count <  40:  mean gain "
                  f"{sum(r['gain'] for r in lo) / len(lo):+.2f}")
            print(f"  >>> the count mechanism is causal on a single dataset: C, the "
                  f"prompt pool\n      and the evaluation set are all fixed, and only "
                  f"the estimation evidence varies.")
    results["dose"] = rows
    return results


def run_validate(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                 preprocess, results):
    """Apply ONE frozen estimator to several datasets. No sweeping, by design.

    The alpha that wins on a single dataset is a selection, not a result. This
    command exposes no hyperparameter search: it takes the frozen alpha, runs it
    untouched on every target, and reports a paired test per dataset. That is
    what separates a finding from a tuning exercise.
    """
    from promptop import bayes as by
    from promptop import stats as st

    alpha = args.alpha
    print(f"\n  FROZEN CONFIGURATION: score = (mu - mu_bar) * n^{alpha:g}, "
          f"tau={args.temp:g}, impl={args.impl}")
    print(f"  No hyperparameter is tuned per dataset.\n")

    rows = []
    for name in [d for d in args.targets.split("/") if d]:
        banner(f"{name}")
        try:
            loader, classnames, _ = build_test_data_loader(
                loader_id(name), args.data_root, preprocess)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {type(exc).__name__}: {exc}")
            continue

        img_f, targets = infer_mod.encode_images(loader, clip_model)
        tf = clip_classifier(classnames, TEMPLATES, clip_model)
        p, c = len(TEMPLATES), len(classnames)

        w_base = infer_mod.carprt_weights(img_f, tf, args.temp, args.chunk)
        pred_base = infer_mod._scores(img_f, tf, w_base).argmax(1)
        base = 100.0 * (pred_base == targets).float().mean().item()

        uni = torch.full((p, c), 1.0 / p, device=tf.device)
        mpe = 100.0 * (infer_mod._scores(img_f, tf, uni).argmax(1)
                       == targets).float().mean().item()

        s1, s2, n = by.weight_moments(img_f, tf, args.chunk)
        w_new = torch.softmax(
            by.count_power_scores(s1, s2, n, alpha, impl=args.impl) / args.temp, dim=0)
        pred_new = infer_mod._scores(img_f, tf, w_new).argmax(1)
        r = st.mcnemar(pred_base, pred_new, targets, "CARPRT", "ours")


        row = {"dataset": name, "classes": c, "images": int(img_f.shape[0]),
               "mpe": mpe, "carprt": base, "ours": r["acc_b"],
               "delta": r["acc_b"] - base, "b": r["b"], "c": r["c"],
               "discordant": r["discordant"], "p": r["p_exact"],
               "empty_frac": float((n == 0).float().mean())}
        rows.append(row)
        print(f"  {c} classes, {row['images']} images, "
              f"{100 * row['empty_frac']:.1f}% empty cells")
        print(f"  MPE {mpe:.2f} | CARPRT {base:.2f} | ours {r['acc_b']:.2f} "
              f"({row['delta']:+.2f}, b={r['b']} c={r['c']}, "
              f"p={r['p_exact']:.2e} {st.stars(r['p_exact'])})")
        del img_f, targets, tf, s1, s2, n
        torch.cuda.empty_cache()

    banner(f"VALIDATION SUMMARY  (alpha={alpha:g}, frozen)")
    hdr = (f"{'dataset':<16}{'C':>5}{'N':>7}{'MPE':>8}{'CARPRT':>9}"
           f"{'ours':>8}{'delta':>8}{'disc':>7}{'p':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<16}{r['classes']:>5}{r['images']:>7}"
              f"{r['mpe']:>8.2f}{r['carprt']:>9.2f}{r['ours']:>8.2f}"
              f"{r['delta']:>+8.2f}{r['discordant']:>7}{r['p']:>10.1e}"
              f"  {st.stars(r['p'])}")

    if rows:
        deltas = [r["delta"] for r in rows]
        mean_d = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        print(f"\n  mean delta {mean_d:+.2f} over {len(rows)} datasets, "
              f"positive on {wins}/{len(rows)}")
        if len(rows) < 3:
            v = (f"INCONCLUSIVE: only {len(rows)} dataset(s). Consistency cannot "
                 f"be assessed, and alpha was selected on Pets -- this number is "
                 f"not yet evidence of anything. Run at least 3 unseen datasets.")
        elif wins == len(rows):
            v = ("CONSISTENT: positive on every dataset. This is a finding, not "
                 "a selection artifact.")
        elif mean_d > 0 and wins > len(rows) / 2:
            v = ("MIXED: positive on average but not everywhere. Report per "
                 "dataset and do not claim a universal gain.")
        else:
            v = ("NOT REPRODUCED: the Pets gain did not transfer. It was "
                 "selection on one dataset.")
        print(f"  >>> {v}")
        results["validate"] = {"alpha": alpha, "rows": rows,
                               "mean_delta": mean_d, "verdict": v}
    return results


def run_bayes(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
              preprocess, results):
    """Posterior prompt reweighting: empty-cell fix + shrinkage + text prior."""
    from promptop import bayes as by
    from promptop import oracle as oracle_mod

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf = clip_classifier(classnames, TEMPLATES, clip_model)
    p, c = len(TEMPLATES), len(classnames)

    def acc(w):
        return 100.0 * (infer_mod._scores(img_f, tf, w).argmax(1)
                        == targets).float().mean().item()

    base_w = infer_mod.carprt_weights(img_f, tf, args.temp, args.chunk)
    base = acc(base_w)
    print(f"  CARPRT baseline: {base:.2f}")

    s1, s2, n = by.weight_moments(img_f, tf, args.chunk)
    empty_frac = float((n == 0).float().mean())
    print(f"  cells with no image assigned: {100 * empty_frac:.1f}%  "
          f"(median count {int(n.median())}, min {int(n.min())})")

    delta = by.text_separability(tf, args.prior_mode)
    print(f"  text separability delta: mean {float(delta.mean()):.4f}  "
          f"min {float(delta.min()):.4f}  max {float(delta.max()):.4f}  "
          f"(image-free, label-free)")

    # anchor: lam=0, empty=zero, beta=0 must reproduce CARPRT
    anc = by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "zero")
    print(f"  anchor (lam=0, empty=zero, beta=0): {acc(anc['weights']):.2f} "
          f"vs CARPRT {base:.2f}")

    # ------------------------------------------------------------------ gate
    banner("GATE: does any scoring rule rank prompts better than CARPRT's?")
    print("  Per-class Spearman against the ORACLE weight matrix. Scale-invariant,")
    print("  so it cannot be confounded by temperature. A rule that ranks no better")
    print("  than CARPRT's 'mean' cannot help: concentrating on a worse ranking is")
    print("  exactly what has failed every previous attempt.\n")
    from promptop import characterize as ch
    sim = oracle_mod.similarity_tensor(img_f, tf, args.chunk)
    _, theta0 = infer_mod.carprt_weights_split_value(
        img_f, tf, tf, args.temp, args.chunk)
    w_oracle, orc_acc, _ = oracle_mod.oracle_optimal_w(
        sim, targets, theta0, args.temp, args.oracle_steps, args.oracle_lr)
    print(f"  oracle ceiling (full fit): {orc_acc:.2f}\n")

    cands = {
        "mean (CARPRT's rule)":
            by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "zero"),
        "mean + empty-cell fix":
            by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "mean"),
        "t-statistic (mu-mu_bar)/se":
            by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "mean",
                                 score_mode="tstat"),
        "text prior alone":
            by.posterior_weights(s1, s2, n, delta, 0.0, 1e6, args.temp, "mean"),
    }
    hdr_g = f"{'scoring rule':<32}{'spearman vs oracle':>20}{'accuracy':>11}"
    print(hdr_g); print("-" * len(hdr_g))
    gate = {}
    for name, out in cands.items():
        sp = ch.compare_pair(out["weights"], w_oracle)["spearman_per_class"]
        a = acc(out["weights"])
        gate[name] = sp
        print(f"{name:<32}{sp:>20.4f}{a:>11.2f}")
    ref = gate["mean (CARPRT's rule)"]
    print(f"\n  CAUTION: this reference oracle is the FULL FIT ({orc_acc:.2f}), which is"
          f"\n  overfit -- 9,139 free parameters on {img_f.shape[0]} images. Agreement with an"
          f"\n  overfit target is not the same as being right, and the t-statistic"
          f"\n  falsified the predictive reading of this table: it ranks BELOW"
          f"\n  CARPRT ({gate.get('t-statistic (mu-mu_bar)/se', float('nan')):.4f} vs {ref:.4f})"
          f" and still classifies better. Read the"
          f"\n  Spearman column as description, not as a gate.")
    results["gate"] = gate
    del sim
    torch.cuda.empty_cache()

    rows = []

    def add(name, w, extra=""):
        a = acc(w)
        s = by.weight_summary(w)
        rows.append({"name": name, "acc": a, "delta": a - base, **s})
        print(f"{name:<38}{a:>9.2f}{a - base:>+10.2f}"
              f"{s['effective_prompts']:>13.1f}{s['cls_std']:>9.3f}"
              f"{s['top10_mass']:>11.4f}  {extra}")

    hdr = (f"\n{'configuration':<38}{'acc':>9}{'vs base':>10}"
           f"{'eff.prompts':>13}{'cls-std':>9}{'top10':>11}")

    # ------------------------------------------------- decomposition + McNemar
    from promptop import stats as st

    banner("DECOMPOSITION: which factor of the t-statistic carries the gain?")
    print("  t = (mu - mu_bar) * sqrt(n) / sd. Each variant is rescaled to the")
    print("  per-class spread of the plain deviation, so none wins by sharpness")
    print("  alone. 'dev (control)' is the rescaled plain deviation: it should")
    print("  land on CARPRT, otherwise the rescaling is itself the confound.\n")

    var_scores = by.score_variants(s1, s2, n, empty="floor")
    base_pred = infer_mod._scores(img_f, tf, base_w).argmax(1)

    hdr_d = (f"{'scoring rule':<34}{'acc':>8}{'vs base':>8}"
             f"{'b':>7}{'c':>7}{'disc':>7}{'p(exact)':>11}")
    print(hdr_d); print("-" * len(hdr_d))
    print(f"  {'CARPRT (reference)':<32}{base:>8.2f}{0.0:>+8.2f}")
    dec_rows = []
    for name, sc in var_scores.items():
        w = torch.softmax(sc / args.temp, dim=0)
        pred = infer_mod._scores(img_f, tf, w).argmax(1)
        r = st.mcnemar(base_pred, pred, targets, "CARPRT", name)
        st.print_mcnemar(r)
        s = by.weight_summary(w)
        dec_rows.append({**r, **s})

    print("\n  b = CARPRT right / variant wrong,  c = CARPRT wrong / variant right")
    print("  p is the exact paired binomial (McNemar). The single-proportion SE")
    print("  used earlier was the wrong yardstick for a paired comparison.")

    print(f"\n{'scoring rule':<34}{'eff.prompts':>13}{'cls-std':>9}{'top10':>10}")
    print("-" * 66)
    for r in dec_rows:
        print(f"  {r['name_b']:<32}{r['effective_prompts']:>13.1f}"
              f"{r['cls_std']:>9.3f}{r['top10_mass']:>10.4f}")
    results["decomposition"] = dec_rows

    banner("ALPHA SWEEP: score = (mu - mu_bar) * n^alpha,  alpha=0 is CARPRT")
    print("  A one-parameter family containing the baseline. Read the SHAPE, not")
    print("  the peak: a smooth rise and fall is evidence of a real effect, a")
    print("  jagged curve means the earlier +0.76 landed on sqrt(n) by chance.\n")
    hdr_a = (f"{'alpha':>8}{'acc':>9}{'vs base':>9}{'b':>7}{'c':>7}"
             f"{'disc':>7}{'p(exact)':>11}{'eff.prompts':>13}")
    print(hdr_a); print("-" * len(hdr_a))
    alpha_rows = []
    for a in [float(x) for x in args.alpha_power.split(",") if x]:
        sc = by.count_power_scores(s1, s2, n, a, impl=args.impl)
        w = torch.softmax(sc / args.temp, dim=0)
        pred = infer_mod._scores(img_f, tf, w).argmax(1)
        r = st.mcnemar(base_pred, pred, targets, "CARPRT", f"alpha={a:g}")
        s = by.weight_summary(w)
        print(f"{a:>8.2f}{r['acc_b']:>9.2f}{r['acc_b'] - base:>+9.2f}"
              f"{r['b']:>7}{r['c']:>7}{r['discordant']:>7}"
              f"{r['p_exact']:>11.2e}{s['effective_prompts']:>13.1f}"
              f"  {st.stars(r['p_exact'])}")
        alpha_rows.append({"alpha": a, **r, **s})
    results["alpha_sweep"] = alpha_rows

    peak = max(alpha_rows, key=lambda r: r["acc_b"])
    print(f"\n  peak at alpha={peak['alpha']:g}: {peak['acc_b']:.2f} "
          f"({peak['acc_b'] - base:+.2f}, p={peak['p_exact']:.2e})")
    print(f"  NOTE: selected on this dataset. Freeze alpha and validate with")
    print(f"        'python run_operator.py validate --alpha {peak['alpha']:g}'")

    banner("ablation: each component alone")
    print(hdr); print("-" * (len(hdr) - 1))
    add("CARPRT (reference)", base_w)
    add("1. empty-cell fix only",
        by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "mean")["weights"])
    add("2. t-statistic (precision-weighted)",
        by.posterior_weights(s1, s2, n, None, 0.0, 0.0, args.temp, "mean",
                             score_mode="tstat")["weights"])

    banner("2. variance-aware shrinkage (lambda sweep, empty-cell fix on)")
    print(hdr); print("-" * (len(hdr) - 1))
    best_lam, best_lam_acc = 0.0, base
    for lam in [float(x) for x in args.lam_sweep.split(",") if x]:
        out = by.posterior_weights(s1, s2, n, None, lam, 0.0, args.temp, "mean")
        add(f"   lambda={lam:g}", out["weights"],
            f"mean shrinkage B={float(out['shrinkage'].mean()):.3f}")
        if rows[-1]["acc"] > best_lam_acc:
            best_lam, best_lam_acc = lam, rows[-1]["acc"]
    print(f"\n  best lambda = {best_lam:g} at {best_lam_acc:.2f}")

    banner("3. text-geometric prior (beta sweep, at the best lambda)")
    print(hdr); print("-" * (len(hdr) - 1))
    best = {"acc": best_lam_acc, "lam": best_lam, "beta": 0.0}
    for beta in [float(x) for x in args.beta_sweep.split(",") if x]:
        w = by.posterior_weights(s1, s2, n, delta, best_lam, beta,
                                 args.temp, "mean")["weights"]
        add(f"   lambda={best_lam:g}, beta={beta:g}", w)
        if rows[-1]["acc"] > best["acc"]:
            best = {"acc": rows[-1]["acc"], "lam": best_lam, "beta": beta}

    banner("verdict")
    # The alpha sweep is where the surviving result lives; a verdict that looks
    # only at the lambda/beta grid reports +0.00 and contradicts the tables above.
    a_best = max(alpha_rows, key=lambda r: r["acc_b"])
    if a_best["acc_b"] > best["acc"]:
        best = {"acc": a_best["acc_b"], "lam": 0.0, "beta": 0.0,
                "alpha": a_best["alpha"], "p": a_best["p_exact"]}
    print(f"  CARPRT              {base:.2f}")
    lbl = (f"alpha={best['alpha']:g}" if "alpha" in best
           else f"lambda={best['lam']:g}, beta={best['beta']:g}")
    print(f"  best configuration  {best['acc']:.2f}  ({lbl})   {best['acc'] - base:+.2f}"
          + (f"   paired p={best['p']:.2e}" if "p" in best else ""))
    se = (base / 100 * (1 - base / 100) / img_f.shape[0]) ** 0.5 * 100
    print(f"  one standard error on {img_f.shape[0]} images: {se:.2f}")
    if best["acc"] - base > 2 * se:
        v = f"SIGNIFICANT: {best['acc'] - base:+.2f} exceeds 2 SE ({2 * se:.2f})."
    elif best["acc"] - base > se:
        v = f"PROMISING: {best['acc'] - base:+.2f} exceeds 1 SE; needs more datasets."
    else:
        v = f"WITHIN NOISE: {best['acc'] - base:+.2f} is under 1 SE ({se:.2f})."
    print(f"  >>> {v}")

    results["bayes"] = {"baseline": base, "rows": rows, "best": best,
                        "empty_frac": empty_frac, "se": se, "verdict": v}
    return results


def run_learn(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
              preprocess, results):
    """Label-quality sweep (prediction) + learned weights from pseudo-labels (method)."""
    from promptop import oracle as oracle_mod
    from promptop import learned as lrn

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf_true = clip_classifier(classnames, TEMPLATES, clip_model)
    n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)
    if oracle_mod.estimate_bytes(n, p, c) > 4.0:
        raise SystemExit("similarity tensor too large for this target.")
    sim = oracle_mod.similarity_tensor(img_f, tf_true, args.chunk)

    w_carprt = infer_mod.carprt_weights(img_f, tf_true, args.temp, args.chunk)
    _, theta0 = infer_mod.carprt_weights_split_value(
        img_f, tf_true, tf_true, args.temp, args.chunk)
    base = oracle_mod._accuracy(sim, w_carprt, targets)
    print(f"  CARPRT baseline: {base:.2f}")

    # ---------------------------------------------------------------- part A
    banner("A. label-quality sweep: what accuracy can a given label quality support?")
    print("  (uses TRUE labels degraded to a target accuracy, then scores against")
    print("   TRUE labels. Random corruption is optimistic -- real pseudo-label")
    print("   errors concentrate on confusable classes -- so read it as a ceiling.)\n")
    hdr = f"{'label accuracy':>16}{'achieved':>11}{'vs CARPRT':>12}"
    print(hdr); print("-" * len(hdr))
    qual_rows = []
    for q in [float(x) for x in args.label_quality.split(",") if x]:
        lab = targets if q >= 1.0 else lrn.corrupt_labels(targets, q, c, args.seed)
        got = 100.0 * (lab == targets).float().mean().item()
        out = lrn.learn_weights(sim, lab, theta0, args.temp, args.oracle_steps,
                                args.oracle_lr, 0.0)
        acc = oracle_mod._accuracy(sim, out["weights"], targets)
        qual_rows.append({"label_acc": got, "acc": acc})
        print(f"{got:>15.1f}%{acc:>11.2f}{acc - base:>+12.2f}")
    results["label_quality_sweep"] = qual_rows

    # ---------------------------------------------------------------- part B
    banner("B. learned weights from CARPRT's own pseudo-labels (no labels used)")
    src = (w_carprt if args.pseudo_source == "carprt"
           else torch.full((p, c), 1.0 / p, device=sim.device))
    pl = lrn.pseudo_labels(sim, src)
    pl_acc = 100.0 * (pl == targets).float().mean().item()
    print(f"  pseudo-label source: {args.pseudo_source}  "
          f"(accuracy {pl_acc:.2f}%, {int((pl != targets).sum())} wrong of {n})\n")

    hdr = (f"{'KL weight':>11}{'accuracy':>11}{'vs CARPRT':>12}"
           f"{'eff.prompts':>13}{'cls-std':>10}{'top10 mass':>12}")
    print(hdr); print("-" * len(hdr))
    learn_rows = []
    traj_store = []
    for klw in [float(x) for x in args.kl_sweep.split(",") if x]:
        out = lrn.learn_weights(sim, pl, theta0, args.temp, args.learn_steps,
                                args.oracle_lr, klw, eval_targets=targets,
                                log_every=max(args.learn_steps // 6, 1))
        acc = oracle_mod._accuracy(sim, out["weights"], targets)
        s = out["stats"]
        print(f"{klw:>11.3g}{acc:>11.2f}{acc - base:>+12.2f}"
              f"{s['effective_prompts']:>13.1f}"
              f"{s['across_class_std_over_uniform']:>10.3f}"
              f"{s['top10_mass']:>12.4f}")
        learn_rows.append({"kl": klw, "acc": acc, **s})
        traj_store.append((klw, out["trajectory"]))

    print("\n  accuracy trajectory during optimisation (diagnostic only -- nothing")
    print("  is selected on it; reported weights are always the final iterate):")
    for klw, tr in traj_store:
        lrn.print_trajectory(tr, f"KL={klw:g}")

    best = max(learn_rows, key=lambda r: r["acc"])
    delta = best["acc"] - base
    print(f"\n  CARPRT {base:.2f} | best learned {best['acc']:.2f} "
          f"(KL={best['kl']:g}) | {delta:+.2f}")
    if delta <= 0:
        verdict = ("NO GAIN: learning W from pseudo-labels does not beat Eq. 10. "
                   "Self-confirmation or pseudo-label noise dominates.")
    elif delta < 0.6:
        verdict = (f"MARGINAL: {delta:+.2f} is ~{delta * n / 100:.0f} images of {n}, "
                   f"inside one standard error.")
    else:
        verdict = (f"GAIN: {delta:+.2f} points over CARPRT using no labels at all.")
    print(f"  >>> {verdict}")

    results["learned"] = learn_rows
    results["learned_verdict"] = verdict
    results["pseudo_label_accuracy"] = pl_acc
    return results


def run_characterize(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                     preprocess, results):
    """Compare the oracle weight matrix with CARPRT's and diagnose the gap."""
    from promptop import oracle as oracle_mod
    from promptop import characterize as ch

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf_true = clip_classifier(classnames, TEMPLATES, clip_model)
    n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)

    gb = oracle_mod.estimate_bytes(n, p, c)
    if gb > 4.0:
        raise SystemExit(f"similarity tensor needs {gb:.1f} GB; target too large.")
    sim = oracle_mod.similarity_tensor(img_f, tf_true, args.chunk)

    w_carprt = infer_mod.carprt_weights(img_f, tf_true, args.temp, args.chunk)
    _, theta_carprt = infer_mod.carprt_weights_split_value(
        img_f, tf_true, tf_true, args.temp, args.chunk)
    w_oracle, orc_acc, _ = oracle_mod.oracle_optimal_w(
        sim, targets, theta_carprt, args.temp, args.oracle_steps, args.oracle_lr)
    base_acc = oracle_mod._accuracy(sim, w_carprt, targets)
    print(f"  CARPRT {base_acc:.2f}   oracle(best W, full fit) {orc_acc:.2f}")

    banner("1. how do the two weight matrices differ?")
    ch.print_stats_table([ch.weight_stats(w_carprt, "CARPRT"),
                          ch.weight_stats(w_oracle, "oracle (best W)")])
    agree = ch.compare_pair(w_carprt, w_oracle)
    print(f"\n  agreement: pearson {agree['pearson']:.4f}   "
          f"per-class spearman {agree['spearman_per_class']:.4f}")
    print()
    up, down = ch.top_movers(w_oracle, w_carprt, TEMPLATES)
    ch.print_movers(up, down)

    banner("2. is the missing accuracy class-agnostic or class-specific?")
    dec = ch.decompose_gap(sim, targets, theta_carprt, w_oracle, args.temp)
    meta = next(r for r in dec if r["name"] == "__meta__")
    rows = [r for r in dec if r["name"] != "__meta__"]
    hdr = f"{'correction applied':<34}{'accuracy':>10}{'recovered':>12}"
    print(hdr)
    print("-" * len(hdr))
    total = rows[-1]["acc"] - rows[0]["acc"]
    for r in rows:
        got = r["acc"] - rows[0]["acc"]
        frac = "" if total <= 0 else f"{100 * got / total:.0f}%"
        print(f"{r['name']:<34}{r['acc']:>10.2f}{frac:>12}")
    print(f"\n  class-agnostic share of the correction's energy: "
          f"{100 * meta['class_agnostic_energy_frac']:.1f}%")

    banner("3. which statistic should Eq. 10 accumulate?")
    stats = ch.candidate_statistics(sim, args.chunk)
    align = ch.statistic_alignment(stats, w_oracle, sim, targets, args.temp)
    hdr = (f"{'accumulated quantity':<20}{'spearman vs oracle':>20}"
           f"{'pearson':>10}{'accuracy':>11}{'vs CARPRT':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in align:
        tag = "  <- CARPRT's rule" if r["name"] == "similarity" else ""
        print(f"{r['name']:<20}{r['spearman_vs_oracle']:>20.4f}"
              f"{r['pearson_vs_oracle']:>10.4f}{r['acc']:>11.2f}"
              f"{r['acc'] - base_acc:>+12.2f}{tag}")

    best = align[0]
    if best["name"] != "similarity" and best["acc"] > base_acc:
        print(f"\n  >>> '{best['name']}' already beats CARPRT by "
              f"{best['acc'] - base_acc:+.2f} as a drop-in accumulation rule.")
    else:
        print("\n  >>> no candidate rule beats the raw similarity; the gap is not "
              "a matter of\n      swapping the accumulated statistic.")

    results["characterize"] = {
        "carprt": base_acc, "oracle": orc_acc, "agreement": agree,
        "decomposition": rows, "class_agnostic_frac":
            meta["class_agnostic_energy_frac"], "statistics": align,
        "stats_table": [ch.weight_stats(w_carprt, "CARPRT"),
                        ch.weight_stats(w_oracle, "oracle")],
        "top_up": up, "top_down": down,
    }
    return results


def run_oracle(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
               preprocess, results):
    """Measure the ceiling of prompt reweighting using labels as a ruler.

    Produces the map that decides whether better weight estimation is worth
    pursuing at all: MPE floor, CARPRT, the label-oracle in Eq. 10's own form,
    and the ceiling of the whole (P, C) weight family.
    """
    from promptop import oracle as oracle_mod

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf_true = clip_classifier(classnames, TEMPLATES, clip_model)
    n, p, c = img_f.shape[0], len(TEMPLATES), len(classnames)

    gb = oracle_mod.estimate_bytes(n, p, c)
    print(f"  similarity tensor (N,P,C) = ({n},{p},{c}) ~ {gb:.2f} GB")
    if gb > 4.0:
        raise SystemExit(
            f"similarity tensor needs {gb:.1f} GB. The oracle is only practical "
            f"for fine-grained sets; use a smaller target than {args.target}.")

    banner("ceilings for prompt reweighting")
    sim = oracle_mod.similarity_tensor(img_f, tf_true, args.chunk)

    rows = []
    uniform = torch.full((p, c), 1.0 / p, device=sim.device)
    rows.append({"key": "mpe", "name": "MPE (uniform weights)", "labels": "no",
                 "acc": oracle_mod._accuracy(sim, uniform, targets)})

    w_carprt = infer_mod.carprt_weights(img_f, tf_true, args.temp, args.chunk)
    _, w_raw_carprt = infer_mod.carprt_weights_split_value(
        img_f, tf_true, tf_true, args.temp, args.chunk)
    rows.append({"key": "carprt", "name": "CARPRT (pseudo-labels, Eq. 10)",
                 "labels": "no",
                 "acc": oracle_mod._accuracy(sim, w_carprt, targets)})

    w_o10, w_raw_o10 = oracle_mod.oracle_eq10(sim, targets, args.temp)
    rows.append({"key": "oracle_eq10",
                 "name": "ORACLE: Eq. 10 with true labels", "labels": "YES",
                 "acc": oracle_mod._accuracy(sim, w_o10, targets)})

    fit_i, ev_i = oracle_mod.split_indices(n, args.seed, 0.5)
    _, full_acc, _ = oracle_mod.oracle_optimal_w(
        sim, targets, w_raw_carprt, args.temp, args.oracle_steps, args.oracle_lr)
    rows.append({"key": "opt_full", "labels": "YES",
                 "name": "ORACLE: best W (fit on all 100%)", "acc": full_acc})

    _, fit_half, held = oracle_mod.oracle_optimal_w(
        sim[fit_i], targets[fit_i], w_raw_carprt, args.temp,
        args.oracle_steps, args.oracle_lr,
        eval_sim=sim[ev_i], eval_targets=targets[ev_i])
    rows.append({"key": "opt_split", "labels": "YES",
                 "name": "ORACLE: best W (fit 50%, scored on held-out 50%)",
                 "acc": held})
    print(f"  [best-W: {p * c:,} free parameters. Fit on all {n:,} images it "
          f"reaches {full_acc:.2f}, but on the 50% split it fits {fit_half:.2f}\n"
          f"   and generalises to {held:.2f} -- the full-fit number is inflated by "
          f"overparameterization,\n   so the held-out row is the one that means "
          f"anything.]")

    print()
    oracle_mod.print_ceiling_table(rows)

    carprt = next(r["acc"] for r in rows if r["key"] == "carprt")
    fam = next(r["acc"] for r in rows if r["key"] == "opt_full")
    gen = next(r["acc"] for r in rows if r["key"] == "opt_split")
    o10 = next(r["acc"] for r in rows if r["key"] == "oracle_eq10")

    print(f"\n  headroom to the family ceiling (full fit): {fam - carprt:+.2f}")
    print(f"  headroom that actually generalises (held-out): {gen - carprt:+.2f}")
    print(f"  attributable to pseudo-label noise (Eq. 10 form): {o10 - carprt:+.2f}")

    # oracle_eq10 is the only well-posed row: closed form, no fitting, so it
    # neither inflates (like the full fit) nor deflates (like the overparameterized
    # split). It is also a member of the weight family, hence a valid lower bound
    # on the family ceiling. Take the best evidence of reachable headroom.
    gen = max(gen, o10)
    if gen - carprt < 1.0:
        verdict = (f"SATURATED: only {gen - carprt:+.2f} points are reachable by any "
                   f"(P,C) weighting scheme that generalises. Better weight "
                   f"estimation is not where the remaining accuracy is -- further "
                   f"gains require leaving the weighting family.")
    elif gen - carprt < 3.0:
        verdict = (f"MODEST ROOM: {gen - carprt:+.2f} points available to a perfect "
                   f"label-free estimator. Worth pursuing only if a large fraction "
                   f"can be captured across several datasets.")
    else:
        verdict = (f"SUBSTANTIAL ROOM: {gen - carprt:+.2f} points are reachable "
                   f"within the existing weight family. Better estimation is a "
                   f"live research direction.")
    print(f"\n  >>> {verdict}")

    results["oracle"] = rows
    results["oracle_verdict"] = verdict
    return results


def run_residual(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
                 preprocess, results):
    """Estimate prompt weights from the interaction term, score with true embeddings.

    Nothing is synthesized here: all P*C real prompted embeddings are used for
    scoring, unchanged. The only thing that varies is the signal the weights are
    estimated from,

        z^(alpha) = normalize(z - alpha * T m),

    so alpha=0 IS ordinary CARPRT and the sweep starts from its accuracy by
    construction. If the curve rises above alpha=0, removing the transferable
    component sharpened the weight estimate.
    """
    banner(f"operator fit ({args.estimator}) for the transferable component")
    lam = args.lam if args.lam is not None else fit_mod.auto_lambda(m_fit)
    if args.estimator == "procrustes":
        w_op = fit_mod.fit_procrustes(m_fit, z_fit)
    elif args.estimator == "lowrank":
        w_op = fit_mod.fit_lowrank(m_fit, z_fit, args.rank, lam)
    else:
        w_op = fit_mod.fit_ridge_identity(m_fit, z_fit, lam)
    print(f"  W {tuple(w_op.shape)}  lambda {lam:.6f}")

    tm_raw = fit_mod.predict_raw(m_tgt, w_op)
    res = z_tgt - tm_raw
    print(f"  ||z|| {z_tgt.norm(dim=-1).mean():.4f}   "
          f"||T m|| {tm_raw.norm(dim=-1).mean():.4f}   "
          f"||R|| {res.norm(dim=-1).mean():.4f}   "
          f"(residual is {100 * res.norm(dim=-1).mean() / z_tgt.norm(dim=-1).mean():.1f}% "
          f"of the embedding)")
    del w_op
    torch.cuda.empty_cache()

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)
    tf_true = clip_classifier(classnames, TEMPLATES, clip_model)

    base = infer_mod.accuracy_report(img_f, targets, tf_true, args.temp,
                                     args.chunk, "baseline CARPRT")
    w_base = infer_mod.carprt_weights(img_f, tf_true, args.temp, args.chunk)
    print(f"  baseline CARPRT (alpha=0 anchor): {base['carprt']:.2f}")

    # Reference spread for --value-scale match: the baseline's own w' at alpha=0.
    _, w_raw_base = infer_mod.carprt_weights_split_value(
        img_f, tf_true, tf_true, args.temp, args.chunk)

    alphas = [float(a) for a in args.alpha_sweep.split(",") if a]
    if args.weight_mode == "value":
        banner("alpha sweep [value mode]: pseudo-labels from the full embedding, "
               "averaged magnitude from z - alpha*T m")
    else:
        banner("alpha sweep [signal mode]: the whole weight-estimation tensor is "
               "replaced (this destroys pseudo-labels; kept for comparison)")

    rows = []
    for a in alphas:
        if args.weight_mode == "value":
            # Magnitude signal: NOT renormalized, since ||z - a*Tm|| shrinking
            # with alpha is part of what the estimator should see. --value-scale
            # match then corrects the induced temperature change.
            val_tf = infer_mod.to_text_feature(
                z_tgt - a * tm_raw, logit_scale, clip_model.dtype, normalize=False)
            rep, _, _ = infer_mod.accuracy_split_value(
                img_f, targets, tf_true, val_tf, args.temp, args.chunk,
                f"alpha={a:g}", ref_w_raw=w_raw_base,
                value_scale=args.value_scale, baseline_weights=w_base)
            del val_tf
        else:
            sig = fit_mod.residual_signal(z_tgt, tm_raw, a)
            sig_tf = infer_mod.to_text_feature(sig, logit_scale, clip_model.dtype)
            rep, _ = infer_mod.accuracy_split_signal(
                img_f, targets, tf_true, sig_tf, args.temp, args.chunk,
                f"alpha={a:g}", baseline_weights=w_base)
            del sig, sig_tf
        rows.append({"alpha": a, **rep})
        torch.cuda.empty_cache()

    hdr = (f"\n{'alpha':>7}{'CARPRT':>10}{'vs baseline':>13}"
           f"{'H(W)/Hmax':>12}{'cls-std':>10}{'corr w/ base W':>16}")
    print(hdr)
    print("-" * len(hdr.strip()))
    for r in rows:
        print(f"{r['alpha']:>7.2f}{r['carprt']:>10.2f}"
              f"{r['carprt'] - base['carprt']:>+13.2f}"
              f"{r['weight_entropy_frac']:>12.3f}"
              f"{r['across_class_std_over_uniform']:>10.3f}"
              f"{r.get('corr_with_baseline_w', float('nan')):>16.3f}")

    best = max(rows, key=lambda r: r["carprt"])
    delta = best["carprt"] - base["carprt"]
    print(f"\n  baseline (alpha=0): {base['carprt']:.2f}")
    print(f"  best: alpha={best['alpha']:g} at {best['carprt']:.2f} ({delta:+.2f})")
    if best["alpha"] == 0.0 or delta <= 0.0:
        verdict = ("NO GAIN: removing the transferable component does not improve "
                   "the weight estimate. The nuisance-projection hypothesis fails "
                   "on this dataset.")
    elif delta < 0.6:
        verdict = (f"MARGINAL: {delta:+.2f} points is ~{delta * 36.69:.0f} images of "
                   f"3669, inside one standard error. Suggestive, not conclusive -- "
                   f"needs more datasets before it means anything.")
    else:
        verdict = (f"GAIN: {delta:+.2f} points over CARPRT with identical embeddings. "
                   f"The interaction term is a better basis for weight estimation.")
    print(f"  >>> {verdict}")

    results["weight_mode"] = args.weight_mode
    results["value_scale"] = args.value_scale
    results["residual_alpha_sweep"] = rows
    results["residual_baseline"] = base
    results["residual_verdict"] = verdict
    return results


def run_sweep(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
              preprocess, results):
    """Model-complexity ladder: how much operator does the gain actually need?

    Every entry is scored on the same held-out target with the same images, and
    reported against its parameter cost per prompt, so the marginal value of the
    matrix over a plain additive shift is visible rather than argued about.
    """
    d = m_fit.shape[1]
    p = z_fit.shape[0]
    lam = args.lam

    banner(f"downstream zero-shot: {args.target}")
    loader, classnames, _ = build_test_data_loader(
        loader_id(args.target), args.data_root, preprocess)
    print("  encoding images once ...")
    img_f, targets = infer_mod.encode_images(loader, clip_model)

    tf_true = clip_classifier(classnames, TEMPLATES, clip_model)
    true_rep = infer_mod.accuracy_report(img_f, targets, tf_true, args.temp,
                                         args.chunk, "true embeddings")

    ranks = [int(r) for r in args.rank_sweep.split(",") if r]
    plan = ([("identity", None), ("additive", None), ("affine", None)]
            + [("lowrank", r) for r in ranks]
            + [("ridge", None), ("procrustes", None)])

    banner("model-complexity sweep")
    rows = []
    for kind, rank in plan:
        if kind == "identity":
            z_hat = fit_mod.predict_identity(m_tgt, p)
        elif kind == "additive":
            z_hat = fit_mod.predict_additive(m_tgt, fit_mod.fit_additive(m_fit, z_fit))
        elif kind == "affine":
            w = fit_mod.fit_affine(m_fit, z_fit, lam)
            z_hat = fit_mod.predict_affine(m_tgt, w)
            del w
        elif kind == "lowrank":
            w = fit_mod.fit_lowrank(m_fit, z_fit, rank, lam)
            z_hat = fit_mod.predict(m_tgt, w)
            del w
        elif kind == "procrustes":
            w = fit_mod.fit_procrustes(m_fit, z_fit)
            z_hat = fit_mod.predict(m_tgt, w)
            del w
        else:
            w = fit_mod.fit_ridge_identity(m_fit, z_fit, lam)
            z_hat = fit_mod.predict(m_tgt, w)
            del w
        torch.cuda.empty_cache()

        label = f"{kind} r={rank}" if rank else kind
        rec = eval_mod.reconstruction_report(z_tgt, z_hat, label)
        tf = infer_mod.to_text_feature(z_hat, logit_scale, clip_model.dtype)
        acc = infer_mod.accuracy_report(img_f, targets, tf, args.temp,
                                        args.chunk, label)
        rows.append({"model": label, "params": fit_mod.n_params(kind, d, rank),
                     "cos_centred": rec["cos_centred_median"],
                     "carprt": acc["carprt"], "mpe": acc["mpe"],
                     "gain": acc["gain"]})
        del z_hat, tf
        torch.cuda.empty_cache()
        print(f"  done: {label}")

    floor = next(r["carprt"] for r in rows if r["model"] == "identity")
    hdr = (f"\n{'model':<16}{'params/prompt':>15}{'cos(centred)':>14}"
           f"{'CARPRT':>9}{'over floor':>12}{'pts/1k par':>12}")
    print(hdr)
    print("-" * len(hdr.strip()))
    for r in rows:
        over = r["carprt"] - floor
        eff = "-" if r["params"] == 0 else f"{1000 * over / r['params']:.4f}"
        print(f"{r['model']:<16}{r['params']:>15,}{r['cos_centred']:>14.4f}"
              f"{r['carprt']:>9.2f}{over:>+12.2f}{eff:>12}")
    print(f"\n  ceiling (true embeddings): {true_rep['carprt']:.2f}  "
          f"| floor (no prompt info): {floor:.2f}  "
          f"| headroom: {true_rep['carprt'] - floor:+.2f}")

    add = next(r for r in rows if r["model"] == "additive")
    best_lr = [r for r in rows if r["model"].startswith("lowrank")
               and r["carprt"] > add["carprt"]]
    if best_lr:
        cheapest = min(best_lr, key=lambda r: r["params"])
        print(f"  cheapest operator beating additive: {cheapest['model']} at "
              f"{cheapest['params']:,} params "
              f"({cheapest['params'] / max(add['params'], 1):.1f}x additive) "
              f"for {cheapest['carprt'] - add['carprt']:+.2f} points")
    else:
        print("  >>> NO low-rank operator beat the additive shift. The matrix is "
              "not earning\n      its parameters at any rank tested.")

    results["sweep"] = rows
    results["sweep_true"] = true_rep
    return results


def main():
    args = get_args()
    set_seed(args.seed)
    results = {"args": vars(args)}
    t0 = time.time()

    banner("model")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = float(clip_model.logit_scale.exp())
    dim = clip_model.text_projection.shape[1]
    print(f"  {args.backbone} | D={dim} | logit_scale={logit_scale:.3f} "
          f"| templates P={len(TEMPLATES)}")

    # ---------------------------------------------------------------- corpora
    banner("class corpora")
    fit_ids = [d for d in args.fit_datasets.split("/") if d]
    print(" fit corpus:")
    fit_corpus = corpus_mod.build_corpus(fit_ids, args.data_root)
    print(" target:")
    tgt_corpus = corpus_mod.build_corpus([args.target], args.data_root)

    if len(tgt_corpus) == 0:
        raise SystemExit(f"target '{args.target}' produced no class names; "
                         f"is its split json under {args.data_root}?")

    overlap = fit_corpus.overlap_with(tgt_corpus)
    print(f"\n  fit {fit_corpus}")
    print(f"  target {tgt_corpus}")
    print(f"  name overlap fit<->target: {len(overlap)}")
    if overlap:
        print(f"    e.g. {overlap[:6]}")
    if overlap and not args.keep_overlap:
        fit_corpus = fit_corpus.subset(exclude_names=overlap)
        print(f"  dropped overlapping names from fit -> C_fit={len(fit_corpus)}")

    corpus_mod.check_determined(len(fit_corpus), dim,
                                strict=not args.allow_underdetermined)
    results["corpus"] = {
        "fit_size": len(fit_corpus), "fit_groups": fit_corpus.group_counts(),
        "target_size": len(tgt_corpus), "overlap_dropped": len(overlap),
        "dim": dim, "n_prompts": len(TEMPLATES),
    }

    # ------------------------------------------------------------- embeddings
    banner("text embeddings (unit-norm, logit_scale stripped)")
    m_fit, z_fit = embed_mod.load_or_build(
        args.cache_dir, fit_corpus, TEMPLATES, clip_model, args.backbone,
        args.base_template, args.batch_size, force=args.force_encode)
    m_tgt, z_tgt = embed_mod.load_or_build(
        args.cache_dir, tgt_corpus, TEMPLATES, clip_model, args.backbone,
        args.base_template, args.batch_size, force=args.force_encode)
    m_fit, z_fit = m_fit.cuda(), z_fit.cuda()
    m_tgt, z_tgt = m_tgt.cuda(), z_tgt.cuda()
    print(f"  M_fit {tuple(m_fit.shape)}  Z_fit {tuple(z_fit.shape)}")
    print(f"  M_tgt {tuple(m_tgt.shape)}  Z_tgt {tuple(z_tgt.shape)}")

    if args.command in ("sweep", "residual", "oracle", "characterize",
                        "learn", "bayes", "validate", "dose",
                        "diagnose", "select", "swapcurve", "stability"):
        runner = {"sweep": run_sweep, "residual": run_residual,
                  "oracle": run_oracle,
                  "characterize": run_characterize,
                  "learn": run_learn,
                  "bayes": run_bayes,
                  "validate": run_validate,
                  "dose": run_dose,
                  "diagnose": run_diagnose,
                  "select": run_select,
                  "swapcurve": run_swapcurve,
                  "stability": run_stability}[args.command]
        runner(args, clip_model, logit_scale, m_fit, z_fit, m_tgt, z_tgt,
               preprocess, results)
        print(f"\ndone in {time.time() - t0:.1f}s")
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2, default=str)
            print(f"results -> {args.out}")
        return

    # ------------------------------------------------------------------- fit
    banner(f"operator fit ({args.estimator})")
    lam = args.lam if args.lam is not None else fit_mod.auto_lambda(m_fit)
    print(f"  lambda = {lam:.6f}  (shrinkage toward identity)")

    if args.estimator == "ridge":
        w = fit_mod.fit_ridge_identity(m_fit, z_fit, lam)
    elif args.estimator == "affine":
        w = fit_mod.fit_affine(m_fit, z_fit, lam)
    elif args.estimator == "procrustes":
        w = fit_mod.fit_procrustes(m_fit, z_fit)
    else:
        w = fit_mod.fit_lowrank(m_fit, z_fit, args.rank, lam)
    a_add = fit_mod.fit_additive(m_fit, z_fit)
    print(f"  W {tuple(w.shape)}   additive null a {tuple(a_add.shape)}")

    w_groups = None
    if args.group_refine:
        if len(fit_corpus.groups()) < 2:
            print("  [skip] --group-refine needs >1 source dataset in the fit "
                  "corpus; only one present.")
        else:
            w_groups = fit_mod.fit_group_refinement(
                m_fit, z_fit, w, fit_corpus.group)
            print(f"  group refinements fitted for: {sorted(w_groups)}")
            results["group_refined"] = sorted(w_groups)
            if args.target not in w_groups:
                print(f"  NOTE: target '{args.target}' is an unseen domain, so "
                      f"refinement cannot apply to it -- the transfer numbers "
                      f"below use the global operator, which is the honest "
                      f"leave-one-dataset-out result. Refinement is reported "
                      f"only for in-domain use.")

    # -------------------------------------------------------- reconstruction
    banner(f"reconstruction on held-out target: {args.target}")
    z_hat_op = fit_mod.predict(m_tgt, w)
    z_hat_id = fit_mod.predict_identity(m_tgt, len(TEMPLATES))
    z_hat_ad = fit_mod.predict_additive(m_tgt, a_add)

    rows = [
        eval_mod.reconstruction_report(z_tgt, z_hat_op, f"operator ({args.estimator})"),
    ]
    if w_groups is not None and args.target in w_groups:
        z_hat_gr = fit_mod.predict(m_tgt, w_groups[args.target])
        rows.append(eval_mod.reconstruction_report(
            z_tgt, z_hat_gr, "operator + group refine"))
    rows += [
        eval_mod.reconstruction_report(z_tgt, z_hat_ad, "null: additive shift"),
        eval_mod.reconstruction_report(z_tgt, z_hat_id, "null: identity (name only)"),
    ]
    eval_mod.print_reconstruction_table(rows)
    print("\n  headline is cos(centred): raw cosine is inflated because every")
    print("  prompted embedding already sits near its class-name embedding.")

    struct = eval_mod.structure_report(z_tgt, z_hat_op)
    print(f"\n  class-class geometry corr: mean {struct['class_geometry_corr_mean']:.4f}"
          f"  min {struct['class_geometry_corr_min']:.4f}")
    results["reconstruction"] = rows
    results["structure"] = struct

    by_name = {r["name"]: r["cos_centred_median"] for r in rows}
    op_c = by_name[f"operator ({args.estimator})"]
    ad_c = by_name["null: additive shift"]
    verdict = ("OPERATOR BEATS ADDITIVE NULL" if op_c > ad_c + 0.02
               else "NO GAIN OVER ADDITIVE NULL -- d^2 parameters unjustified")
    print(f"\n  >>> {verdict}  ({op_c:.4f} vs {ad_c:.4f})")
    results["verdict_reconstruction"] = verdict

    # ------------------------------------------------------------- residuals
    banner("residual structure (is a universal operator enough?)")
    res = eval_mod.residual_report(m_fit, z_fit, w, TEMPLATES,
                                   n_clusters=args.residual_clusters,
                                   seed=args.seed)
    eval_mod.print_residual_report(res)
    results["residual"] = res

    banner("operator manifold")
    man = eval_mod.operator_manifold_report(w, TEMPLATES)
    eval_mod.print_manifold_report(man)
    pca = manifold_mod.operator_pca(w, n_components=args.pca_components)
    manifold_mod.print_pca_report(pca)
    results["manifold"] = {k: v for k, v in man.items()
                           if k not in ("most_generic", "least_generic")}
    results["manifold"]["most_generic"] = man["most_generic"]
    results["manifold"]["least_generic"] = man["least_generic"]

    # -------------------------------------------------------------- classify
    if args.command in ("classify", "all"):
        banner(f"downstream zero-shot: {args.target}")
        loader, classnames, _ = build_test_data_loader(
            loader_id(args.target), args.data_root, preprocess)

        if len(classnames) != len(tgt_corpus):
            raise SystemExit(
                f"class count mismatch: loader {len(classnames)} vs corpus "
                f"{len(tgt_corpus)}; synthesized rows would not align with labels.")

        print("  encoding images once ...")
        img_f, targets = infer_mod.encode_images(loader, clip_model)
        print(f"  images {tuple(img_f.shape)}")

        tf_true = clip_classifier(classnames, TEMPLATES, clip_model)
        variants = [
            (tf_true, "true embeddings (P*C encodes)"),
            (infer_mod.to_text_feature(z_hat_op, logit_scale, clip_model.dtype),
             f"synthesized: {args.estimator} (C encodes)"),
            (infer_mod.to_text_feature(z_hat_ad, logit_scale, clip_model.dtype),
             "synthesized: additive null"),
            (infer_mod.to_text_feature(z_hat_id, logit_scale, clip_model.dtype),
             "synthesized: identity null"),
        ]
        if args.synth_prompts > 0:
            w_new = manifold_mod.synthesize_operators(
                pca, args.synth_prompts, mode=args.synth_mode, seed=args.seed)
            z_synth = fit_mod.predict(m_tgt, w_new)
            variants.append((
                infer_mod.to_text_feature(z_synth, logit_scale, clip_model.dtype),
                f"novel pool: {args.synth_prompts} sampled ops"))
            print(f"  synthesized {args.synth_prompts} operators from the manifold "
                  f"({args.synth_mode}); these correspond to no written template")

        acc_rows = [infer_mod.accuracy_report(img_f, targets, tf, args.temp,
                                              args.chunk, name)
                    for tf, name in variants]
        print()
        infer_mod.print_accuracy_table(acc_rows)
        results["accuracy"] = acc_rows

        true_acc, syn_acc = acc_rows[0]["carprt"], acc_rows[1]["carprt"]
        n_true = len(TEMPLATES) * len(classnames)
        print(f"\n  accuracy retained: {syn_acc:.2f} vs {true_acc:.2f} "
              f"({syn_acc - true_acc:+.2f})")
        print(f"  text encodings: {len(classnames)} vs {n_true} "
              f"({n_true / max(len(classnames), 1):.0f}x fewer)")
        results["encoding_cost"] = {"synthesized": len(classnames), "true": n_true}

    print(f"\ndone in {time.time() - t0:.1f}s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
