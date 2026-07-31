"""Downstream zero-shot classification driven by synthesized prompt embeddings.

This is the payoff stage: encode only the C bare class names, synthesize all P
prompted embeddings with the fitted operators, and run the ordinary class-aware
reweighting pipeline on top. If accuracy holds, the P*C text encodings collapse to
C (247,000 -> 1,000 on ImageNet).

Images are encoded once and reused across every text-feature variant, so the
comparison between true and synthesized embeddings is exact rather than confounded
by loader shuffling.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def _get_matrix():
    """Import test.get_matrix lazily.

    `test` is also a stdlib package name, so a module-level import here resolves
    correctly only when the repo root leads sys.path. Deferring it keeps
    `import promptop` safe from any working directory.
    """
    from test import get_matrix
    return get_matrix


def to_text_feature(
    z: torch.Tensor,
    logit_scale: float,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """(P, C, D) unit-norm float32  ->  drop-in replacement for clip_classifier.

    Re-applies the logit_scale that promptop strips for the geometry, so the result
    is bit-compatible in convention with utils.clip_classifier's output.
    """
    z = z / z.norm(dim=-1, keepdim=True)
    return (z * logit_scale).to(dtype).cuda()


@torch.no_grad()
def encode_images(loader, clip_model, device: str = "cuda"):
    """Encode a whole test loader once. Returns (features (N, D), targets (N,))."""
    feats, targets = [], []
    for images, target in loader:
        images = images.to(device)
        f = clip_model.encode_image(images)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
        targets.append(target.to(device))
    return torch.cat(feats, 0), torch.cat(targets, 0)


@torch.no_grad()
def carprt_weights(
    image_features: torch.Tensor,
    text_feature: torch.Tensor,
    temp: float = 1.0,
    chunk: int = 512,
) -> torch.Tensor:
    """Class-aware prompt weights W* of shape (P, C), softmax over prompts."""
    get_matrix = _get_matrix()
    p, c, _ = text_feature.shape
    device = text_feature.device
    w_sum = torch.zeros((p, c), dtype=torch.float32, device=device)
    w_cnt = torch.zeros((p, c), dtype=torch.long, device=device)

    for i in range(0, image_features.shape[0], chunk):
        logits = torch.einsum("pcd,nd -> pcn", text_feature,
                              image_features[i:i + chunk])
        s, n = get_matrix(logits, c, p)
        w_sum += s
        w_cnt += n

    w_raw = w_sum / torch.where(w_cnt == 0, 1, w_cnt)
    return F.softmax(w_raw / temp, dim=0)


@torch.no_grad()
def _scores(image_features: torch.Tensor,
            text_feature: torch.Tensor,
            weights: torch.Tensor) -> torch.Tensor:
    tf = torch.einsum("pc,pcd -> cd", weights.type(text_feature.dtype), text_feature)
    return image_features @ tf.t()


def _micro(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return 100.0 * (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def accuracy_report(
    image_features: torch.Tensor,
    targets: torch.Tensor,
    text_feature: torch.Tensor,
    temp: float = 1.0,
    chunk: int = 512,
    name: str = "text_feature",
) -> Dict[str, object]:
    """CARPRT and MPE accuracy for one set of text features.

    Micro-average top-1 only. The mean-of-batch-means figure that test.py prints is
    order-dependent and varies ~0.4 points with the loader shuffle, so it is not a
    usable basis for comparing two text-feature variants.
    """
    p, c, _ = text_feature.shape
    w = carprt_weights(image_features, text_feature, temp, chunk)
    carprt = _micro(_scores(image_features, text_feature, w), targets)

    uniform = torch.full((p, c), 1.0 / p, device=text_feature.device)
    mpe = _micro(_scores(image_features, text_feature, uniform), targets)

    ent = -(w * w.clamp_min(1e-12).log()).sum(dim=0)
    return {
        "name": name,
        "carprt": carprt,
        "mpe": mpe,
        "gain": carprt - mpe,
        "weight_entropy_mean": float(ent.mean()),
        "weight_entropy_frac": float(ent.mean() / torch.log(torch.tensor(float(p)))),
        "across_class_std_over_uniform": float(w.std(dim=1).mean() * p),
    }


def print_accuracy_table(rows) -> None:
    hdr = (f"{'text features':<30}{'CARPRT':>9}{'MPE':>9}{'gain':>8}"
           f"{'H(W)/Hmax':>11}{'cls-std':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<30}{r['carprt']:>9.2f}{r['mpe']:>9.2f}{r['gain']:>+8.2f}"
              f"{r['weight_entropy_frac']:>11.3f}"
              f"{r['across_class_std_over_uniform']:>9.3f}")
