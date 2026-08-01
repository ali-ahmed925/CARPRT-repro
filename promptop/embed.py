"""Text encoding and caching for operator fitting.

Everything here returns **unit-norm float32** embeddings. utils.clip_classifier
multiplies text features by logit_scale.exp() (~100) because that scale is baked
into the scoring path; leaving it in would inflate every Gram matrix and
regression target by 1e4 and wreck the conditioning. The scale is re-applied only
at the very end, in promptop.infer.to_text_feature.
"""

import hashlib
import os
from typing import List, Optional, Sequence, Tuple

import torch

from .corpus import ClassCorpus


def _fingerprint(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    clip_model,
    batch_size: int = 256,
    device: str = "cuda",
    progress_every: int = 50,
    label: str = "",
) -> torch.Tensor:
    """Encode strings to unit-norm float32 embeddings of shape (N, D)."""
    import clip

    out = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(texts), batch_size)):
        chunk = list(texts[start:start + batch_size])
        tokens = clip.tokenize(chunk, truncate=True).to(device)
        feats = clip_model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.half().cpu())
        if progress_every and (bi % progress_every == 0 or bi == n_batches - 1):
            print(f"    {label} batch {bi + 1}/{n_batches}", flush=True)
    return torch.cat(out, dim=0).float()


def build_class_embeddings(
    corpus: ClassCorpus,
    clip_model,
    base_template: str = "{}",
    batch_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """m_c for every class in the corpus. Shape (C, D), unit-norm float32.

    base_template="{}" embeds the bare class name, which is the cleanest reading
    of "class content with no prompt applied". Passing a neutral template such as
    "a photo of a {}." is better conditioned and makes T_i the identity for that
    template; it is a legitimate alternative, not a bug.
    """
    texts = [base_template.format(name) for name in corpus.names]
    return encode_texts(texts, clip_model, batch_size, device, label="class-names")


def build_prompted_tensor(
    corpus: ClassCorpus,
    templates: Sequence[str],
    clip_model,
    batch_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """z_{i,c} for every (prompt, class). Shape (P, C, D), unit-norm float32.

    Cost is P * C text encodings -- 247,000 for ImageNet. This is exactly the cost
    the operator model is meant to eliminate downstream.
    """
    texts: List[str] = []
    for name in corpus.names:                     # class-major, reshaped below
        for tmpl in templates:
            texts.append(tmpl.format(name))

    flat = encode_texts(texts, clip_model, batch_size, device, label="prompted")
    c, p, d = len(corpus), len(templates), flat.shape[-1]
    return flat.view(c, p, d).permute(1, 0, 2).contiguous()


def load_or_build(
    cache_dir: str,
    corpus: ClassCorpus,
    templates: Sequence[str],
    clip_model,
    backbone: str,
    base_template: str = "{}",
    batch_size: int = 256,
    device: str = "cuda",
    force: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (M, Z) = ((C, D), (P, C, D)), caching to disk.

    Cache key covers backbone, base template, the class-name list and the template
    list, so changing any of them invalidates cleanly.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _fingerprint(
        backbone, base_template,
        "|".join(corpus.names), "|".join(corpus.group), "|".join(templates),
    )
    path = os.path.join(cache_dir, f"embeddings_{key}.pt")

    if os.path.exists(path) and not force:
        # Explicit: the cache stores python lists of class names alongside the
        # tensors, so weights_only=True cannot load it. torch 2.6 flips this
        # default, which would break the cache silently.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  loaded cached embeddings: {path}")
        return blob["M"].float(), blob["Z"].float()

    print(f"  encoding {len(corpus)} class names + "
          f"{len(templates) * len(corpus)} prompted texts ...")
    m = build_class_embeddings(corpus, clip_model, base_template, batch_size, device)
    z = build_prompted_tensor(corpus, templates, clip_model, batch_size, device)

    torch.save(
        {"M": m.half(), "Z": z.half(), "names": corpus.names,
         "group": corpus.group, "backbone": backbone,
         "base_template": base_template},
        path,
    )
    print(f"  cached -> {path}")
    return m, z


def load_cached_names(cache_path: str) -> Optional[ClassCorpus]:
    if not os.path.exists(cache_path):
        return None
    blob = torch.load(cache_path, map_location="cpu")
    return ClassCorpus(names=blob["names"], group=blob["group"])
