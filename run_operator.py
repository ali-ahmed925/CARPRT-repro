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


def get_args():
    p = argparse.ArgumentParser(description="Prompt-operator pipeline.")
    p.add_argument("command", choices=["fit", "classify", "all", "sweep", "residual", "oracle"])
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
        args.target, args.data_root, preprocess)
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
        args.target, args.data_root, preprocess)
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
        args.target, args.data_root, preprocess)
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

    if args.command in ("sweep", "residual", "oracle"):
        runner = {"sweep": run_sweep, "residual": run_residual,
                  "oracle": run_oracle}[args.command]
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
            args.target, args.data_root, preprocess)

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
