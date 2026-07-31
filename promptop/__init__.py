"""Prompt-operator modelling for CLIP-like VLMs.

Treats a prompt template not as a string producing a vector, but as an operator
acting on class-name embeddings:

    z_{i,c}  ~=  T_i m_c

where m_c is the embedding of the bare class name and T_i is shared across
classes. Fitting is closed-form ridge regression (no gradients, no labels).

Package is named `promptop`, not `operator`, to avoid shadowing the stdlib
`operator` module that torch imports.
"""

from .corpus import ClassCorpus, build_corpus
from .embed import encode_texts, build_class_embeddings, build_prompted_tensor
from .fit import (
    fit_ridge_identity,
    fit_procrustes,
    fit_lowrank,
    fit_additive,
    fit_group_refinement,
    predict,
    predict_identity,
    predict_additive,
)
from .evaluate import (
    reconstruction_report,
    operator_manifold_report,
    residual_report,
    structure_report,
)
from .manifold import (
    operator_pca,
    synthesize_operators,
    interpolate_operators,
)
from .infer import to_text_feature, encode_images, carprt_weights, accuracy_report

__all__ = [
    "ClassCorpus", "build_corpus",
    "encode_texts", "build_class_embeddings", "build_prompted_tensor",
    "fit_ridge_identity", "fit_procrustes", "fit_lowrank", "fit_additive",
    "fit_group_refinement", "predict", "predict_identity", "predict_additive",
    "reconstruction_report", "operator_manifold_report", "residual_report",
    "structure_report",
    "operator_pca", "synthesize_operators", "interpolate_operators",
    "to_text_feature", "encode_images", "carprt_weights", "accuracy_report",
]
