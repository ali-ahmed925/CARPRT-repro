"""Class-name corpora, pooled across datasets, with provenance.

The operator regression solves for T_i in R^{d x d} from C observations, so it is
underdetermined whenever C < d (= 512 for ViT-B/16 / RN50). A single fine-grained
benchmark is nowhere near enough: Oxford Pets has C = 37. Pooling class names
across datasets is therefore not an enhancement, it is a precondition.

ImageNet's 1000 class names are hardcoded in datasets/imagenet.py, so they are
available with no data on disk at all. That alone clears C >= d.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

# Registry ids accepted by --datasets. "imagenet" needs no files on disk.
NEEDS_NO_DATA = {"imagenet"}

# Everything else is resolved through datasets.build_dataset, which needs at
# minimum that dataset's split json (images are not touched to read classnames).
BUILDABLE = [
    "oxford_pets", "caltech101", "dtd", "eurosat", "fgvc", "food101",
    "oxford_flowers", "stanford_cars", "sun397", "ucf101",
]

ALL_SOURCES = ["imagenet"] + BUILDABLE


def normalize_name(name: str) -> str:
    """Match the normalization applied in utils.clip_classifier."""
    return name.replace("_", " ").strip()


@dataclass
class ClassCorpus:
    """Pooled class names with the dataset each came from."""

    names: List[str] = field(default_factory=list)
    group: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        counts = self.group_counts()
        inner = ", ".join(f"{g}={n}" for g, n in sorted(counts.items()))
        return f"ClassCorpus(C={len(self)}, {inner})"

    def groups(self) -> Set[str]:
        return set(self.group)

    def group_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for g in self.group:
            out[g] = out.get(g, 0) + 1
        return out

    def indices_for(self, groups: Sequence[str]) -> List[int]:
        wanted = set(groups)
        return [i for i, g in enumerate(self.group) if g in wanted]

    def subset(
        self,
        groups: Optional[Sequence[str]] = None,
        exclude_groups: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
    ) -> "ClassCorpus":
        keep_g = set(groups) if groups is not None else self.groups()
        drop_g = set(exclude_groups or ())
        drop_n = {normalize_name(n).lower() for n in (exclude_names or ())}

        out = ClassCorpus()
        for name, g in zip(self.names, self.group):
            if g not in keep_g or g in drop_g:
                continue
            if name.lower() in drop_n:
                continue
            out.names.append(name)
            out.group.append(g)
        return out

    def overlap_with(self, other: "ClassCorpus") -> List[str]:
        """Class names present in both corpora (case-insensitive).

        Matters for leave-one-dataset-out: ImageNet contains many dog breeds that
        also appear in Oxford Pets, so an unfiltered transfer test leaks.
        """
        mine = {n.lower() for n in self.names}
        return sorted({n for n in other.names if n.lower() in mine})


def _classnames_from_dataset(dataset_id: str, data_root: str) -> List[str]:
    from datasets import build_dataset

    ds = build_dataset(dataset_id, data_root)
    return list(ds.classnames)


def _imagenet_classnames() -> List[str]:
    from datasets.imagenet import imagenet_classes

    return list(imagenet_classes)


def build_corpus(
    dataset_ids: Sequence[str],
    data_root: str,
    verbose: bool = True,
) -> ClassCorpus:
    """Pool class names from the requested datasets, skipping unavailable ones.

    Sources that raise (missing split json, missing directory) are skipped with a
    warning rather than aborting -- the usual case is that only some benchmarks
    have been downloaded.
    """
    corpus = ClassCorpus()
    seen: Set[str] = set()

    for did in dataset_ids:
        try:
            if did in NEEDS_NO_DATA:
                names = _imagenet_classnames()
            else:
                names = _classnames_from_dataset(did, data_root)
        except Exception as exc:  # noqa: BLE001 - want the reason, not a traceback
            if verbose:
                print(f"  [skip] {did}: {type(exc).__name__}: {exc}")
            continue

        added = 0
        for raw in names:
            name = normalize_name(raw)
            key = f"{did}::{name.lower()}"
            if key in seen:
                continue
            seen.add(key)
            corpus.names.append(name)
            corpus.group.append(did)
            added += 1

        if verbose:
            print(f"  [ok]   {did}: {added} class names")

    return corpus


def check_determined(n_classes: int, dim: int, strict: bool = True) -> None:
    """Guard against fitting a d x d operator from fewer than d observations."""
    if n_classes >= dim:
        return
    msg = (
        f"UNDERDETERMINED: fitting a {dim}x{dim} operator from only {n_classes} "
        f"class observations. The in-sample fit will be near-perfect and will not "
        f"generalize to held-out classes; a poor transfer score would tell you "
        f"nothing about the operator hypothesis. Pool more datasets (ImageNet "
        f"alone supplies 1000 names with no data on disk) or constrain the "
        f"operator (--estimator procrustes / --estimator lowrank --rank r)."
    )
    if strict:
        raise ValueError(msg)
    print(f"\n!! WARNING: {msg}\n")
