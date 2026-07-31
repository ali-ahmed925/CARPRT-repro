"""End-to-end prompt-operator pipeline.

    fit       estimate operators on a pooled class corpus, score reconstruction
              on a held-out dataset against the identity and additive nulls
    classify  zero-shot accuracy on a target dataset using ONLY synthesized
              prompt embeddings (C text encodings instead of P*C)
    all       both, sharing one encode

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
    p.add_argument("command", choices=["fit", "classify", "all"])
    p.add_argument("--fit-datasets", type=str, default="imagenet",
                   help="Slash-separated corpus for fitting, e.g. 'imagenet/sun397'.")
    p.add_argument("--target", type=str, default="oxford_pets",
                   help="Held-out dataset: never contributes to the fit.")
    p.add_argument("--backbone", type=str, choices=["RN50", "ViT-B/16"],
                   default="ViT-B/16")
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=os.path.expanduser("~/datasets"))
    p.add_argument("--estimator", type=str, default="ridge",
                   choices=["ridge", "procrustes", "lowrank"])
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

    # ------------------------------------------------------------------- fit
    banner(f"operator fit ({args.estimator})")
    lam = args.lam if args.lam is not None else fit_mod.auto_lambda(m_fit)
    print(f"  lambda = {lam:.6f}  (shrinkage toward identity)")

    if args.estimator == "ridge":
        w = fit_mod.fit_ridge_identity(m_fit, z_fit, lam)
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
