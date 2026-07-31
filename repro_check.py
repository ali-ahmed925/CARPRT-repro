"""Single-dataset reproduction harness for CARPRT.

Additive: does not modify test.py. Encodes each test image exactly once, then
derives CARPRT weights and reports several accuracy definitions side by side
plus weight-health diagnostics.

Pass 1 of test.py accumulates over the whole loader, so caching image features
yields bit-comparable weights while removing the shuffle dependence.

    python repro_check.py --dataset oxford_pets --backbone ViT-B/16 \
        --data-root /path/to/datasets --temp 1.0
"""

import argparse
import random

import clip
import numpy as np
import torch
import torch.nn.functional as F

from test import get_matrix
from utils import build_test_data_loader, clip_classifier


def get_args():
    p = argparse.ArgumentParser(description="CARPRT reproduction check (one dataset).")
    p.add_argument('--dataset', type=str, required=True, help="Single dataset id, e.g. oxford_pets")
    p.add_argument('--backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True)
    p.add_argument('--data-root', dest='data_root', type=str, required=True)
    p.add_argument('--temp', type=float, default=1.0, help="tau in Eq. 11")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk', type=int, default=512, help="Images per scoring chunk (memory only).")
    p.add_argument('--batch-mean-size', dest='batch_mean_size', type=int, default=512,
                   help="Batch size used to emulate test.py's mean-of-batch-means metric.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_all_images(loader, clip_model):
    feats, targets = [], []
    for images, target in loader:
        images = images.cuda()
        f = clip_model.encode_image(images)
        f /= f.norm(dim=-1, keepdim=True)
        feats.append(f)
        targets.append(target.cuda())
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


@torch.no_grad()
def carprt_weights(image_features, text_feature, temp, chunk):
    num_prompt, num_class, _ = text_feature.shape
    device = text_feature.device
    w_sum = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=device)
    w_cnt = torch.zeros((num_prompt, num_class), dtype=torch.long, device=device)
    max_logit_vals = []

    for i in range(0, image_features.shape[0], chunk):
        feats = image_features[i:i + chunk]
        logits = torch.einsum('pcd,nd -> pcn', text_feature, feats)
        s, c = get_matrix(logits, num_class, num_prompt)
        w_sum += s
        w_cnt += c
        max_logit_vals.append(torch.max(logits, dim=1)[0].float().flatten())

    safe_cnt = torch.where(w_cnt == 0, 1, w_cnt)
    w_raw = w_sum / safe_cnt
    weights = F.softmax(w_raw / temp, dim=0)
    return weights, w_raw, w_cnt, torch.cat(max_logit_vals)


@torch.no_grad()
def scores_from_weights(image_features, text_feature, weights):
    """Eq. 12: collapse prompts with W, then dot with image features."""
    tf = torch.einsum('pc,pcd -> cd', weights.type(text_feature.dtype), text_feature)
    return image_features @ tf.t()


def micro_acc(logits, targets):
    return 100.0 * (logits.argmax(dim=1) == targets).float().mean().item()


def batch_mean_acc(logits, targets, bs):
    accs = []
    for i in range(0, logits.shape[0], bs):
        chunk_logits, chunk_t = logits[i:i + bs], targets[i:i + bs]
        accs.append(100.0 * (chunk_logits.argmax(dim=1) == chunk_t).float().mean().item())
    return sum(accs) / len(accs)


def main():
    args = get_args()
    set_seed(args.seed)

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    loader, classnames, template = build_test_data_loader(args.dataset, args.data_root, preprocess)
    print(f"\nclassnames: {len(classnames)}   templates: {len(template)}")
    if len(template) != 247:
        print(f"  !! WARNING: paper uses 247 templates, found {len(template)}")

    text_feature = clip_classifier(classnames, template, clip_model)
    P, C, D = text_feature.shape
    print(f"text_feature shape (P, C, D): {tuple(text_feature.shape)}")

    scale = clip_model.logit_scale.exp().item()
    tf_norm = text_feature[0, 0].float().norm().item()
    print(f"logit_scale.exp(): {scale:.3f}   ||text_feature[0,0]||: {tf_norm:.3f}  (should equal logit_scale)")

    image_features, targets = encode_all_images(loader, clip_model)
    N = image_features.shape[0]
    print(f"encoded images: {N}")

    weights, w_raw, w_cnt, max_logits = carprt_weights(
        image_features, text_feature, args.temp, args.chunk)

    # ---- diagnostics -------------------------------------------------------
    uniform = 1.0 / P
    ent = -(weights * weights.clamp_min(1e-12).log()).sum(dim=0)   # nats, per class
    max_ent = np.log(P)
    eff_prompts = ent.exp()

    print("\n--- weight diagnostics ---")
    print(f"raw w' (pre-softmax): min {w_raw.min():.3f}  mean {w_raw.mean():.3f}  max {w_raw.max():.3f}")
    print(f"max logit (100x cosine): mean {max_logits.mean():.3f}  p99 {max_logits.quantile(0.99):.3f}")
    print(f"W: min {weights.min():.3e}  max {weights.max():.3e}  uniform would be {uniform:.3e}")
    print(f"W max/uniform ratio: {(weights.max() / uniform):.2f}x   (~1.0 means weighting is inert)")
    print(f"per-class entropy: mean {ent.mean():.3f} / max {max_ent:.3f} nats "
          f"({100 * ent.mean() / max_ent:.1f}% of uniform)")
    print(f"effective prompts per class: mean {eff_prompts.mean():.1f} of {P}")
    print(f"empty (prompt, class) cells: {(w_cnt == 0).float().mean() * 100:.1f}%")

    col_std = weights.std(dim=1).mean().item()
    print(f"mean across-class std of a prompt's weight: {col_std:.3e} "
          f"({col_std / uniform:.2f}x uniform)  <- 0 means class-awareness is absent")

    print("\ntop-3 prompts for first 3 classes:")
    for c in range(min(3, C)):
        top = torch.topk(weights[:, c], 3)
        print(f"  [{classnames[c]}]")
        for w, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"     {w:.5f}  {template[idx]}")

    # ---- accuracies --------------------------------------------------------
    carprt_logits = scores_from_weights(image_features, text_feature, weights)

    mpe_w = torch.full((P, C), 1.0 / P, device=text_feature.device)
    mpe_logits = scores_from_weights(image_features, text_feature, mpe_w)

    tf_mean = text_feature.float().mean(dim=0)
    tf_mean = tf_mean / tf_mean.norm(dim=-1, keepdim=True)
    mpe_norm_logits = image_features.float() @ tf_mean.t()

    print("\n--- accuracy ---")
    print(f"CARPRT   micro-average (true top-1)      : {micro_acc(carprt_logits, targets):.2f}")
    print(f"CARPRT   mean-of-batch-means (bs={args.batch_mean_size:<4d})   : "
          f"{batch_mean_acc(carprt_logits, targets, args.batch_mean_size):.2f}   <- metric test.py reports")
    print(f"MPE      micro-average, no renorm         : {micro_acc(mpe_logits, targets):.2f}")
    print(f"MPE      micro-average, renormalized      : {micro_acc(mpe_norm_logits, targets):.2f}")
    print(f"\nCARPRT - MPE(no renorm) = "
          f"{micro_acc(carprt_logits, targets) - micro_acc(mpe_logits, targets):+.2f} points")


if __name__ == "__main__":
    main()
