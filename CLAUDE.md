# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CARPRT (ICLR 2026, Dong et al., [OpenReview](https://openreview.net/pdf?id=AScQDQqVXY)) — a **training-free, black-box** method that estimates *class-specific* weights over a fixed prompt ensemble for CLIP zero-shot classification. There is no training loop, no checkpointing, no optimizer. The entire codebase is one evaluation pass plus data plumbing.

The one-line idea: prior work (WPE/ZPE, Allingham et al. 2023) gives each prompt a single weight shared across all classes; CARPRT makes the weight matrix `W ∈ R^{n×C}` genuinely two-dimensional, so "an aerial view of {}" can be up-weighted for *airport* and suppressed for *apple*.

## Environment & commands

Requires a **CUDA** PyTorch build — `.cuda()` is hardcoded in `utils.py` and `clip/clip.py` defaults to CUDA; there is no CPU fallback.

```bash
conda create -y -n carprt python=3.8 && conda activate carprt
pip install -r requirements.txt
```

Run evaluation (the only entry point):

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --datasets caltech101/dtd/eurosat \
  --backbone ViT-B/16 \
  --data-root /path/to/datasets \
  --temp 1.0
```

- `--datasets` is **slash-separated**; each id is evaluated sequentially in one process. Valid ids are enumerated in `utils.build_test_data_loader` (`I`/`A`/`V`/`R`/`S` for ImageNet variants, plus the named fine-grained sets and `cifar10`/`imcifar10`/`cifar100`/`imcifar100`).
- `--data-root` defaults to `/projects/datasets` — almost always needs overriding.
- `--config` is parsed and deliberately discarded (`test.py:93`); it exists only so old command lines don't break.
- `run.sh` is a convenience wrapper, but its `CUDA_VISIBLE_DEVICES=0` line is a bare assignment on its own line, so it does *not* actually reach `python`. Prefer the inline form above.

There are no tests, linters, or CI in this repo. Despite the name, `test.py` is the evaluation script.

## Architecture

### The core tensor contract

Everything hinges on a three-axis text-feature tensor. `utils.clip_classifier` returns `text_feature` of shape **`(P, C, D)`** = (num_prompts, num_classes, embed_dim), where:

- `P` is the length of the shared template list (247 entries) — see below.
- Embeddings are L2-normalized **and then multiplied by `clip_model.logit_scale.exp()`** inside `clip_classifier`. The scale is baked into the text side, so downstream code never re-applies it. Anything that rebuilds text features must preserve this.

Two logit paths consume it:
- `get_clip_logits` → `einsum('pcd,nd->pcn')`, i.e. per-prompt logits kept **separate** (used for weight estimation).
- `get_res_logits` → first collapses prompts with the learned weights, `einsum('pc,pcd->cd')`, then does a normal `image @ text.T` (used for final prediction).

### The two-pass algorithm (`test.py`)

`run_test_carprt` iterates the **same** `test_loader` twice:

1. **Pass 1 (unlabeled, no grad)** — for each batch, `get_matrix` takes the per-prompt argmax over classes, then scatter-adds the winning similarity value into a `(P, C)` sum matrix and a parallel `(P, C)` count matrix. Labels are explicitly discarded here; this is what makes the method black-box/label-free.
2. **Weight estimation** — mean the sums by counts (zero counts are clamped to 1 to avoid div-by-zero), then `softmax(weights / temp, dim=0)` — normalization is **over prompts**, per class. That column-wise softmax is the "class-aware" part of the method.
3. **Pass 2 (labeled)** — re-iterate the loader with the frozen weights and accumulate top-1 accuracy per batch.

Accuracy is a plain mean over batch accuracies, so the last (short) batch is slightly over-weighted. The loader is built with `shuffle=True`, so the two passes see different orderings — fine for this algorithm, but relevant if you ever try to cache pass-1 features for pass 2.

### Prompt templates

`datasets/template.py` is a single flat list of **247** `"... {} ..."` strings — the fixed pool from Allingham et al. (2023), used unchanged for every experiment in the paper so baselines stay comparable. **Every dataset imports the identical list**; there are no per-dataset templates. Its length directly sets the `P` axis and the compute cost of `clip_classifier` (one `encode_text` call per class over all 247 prompts). The list is intentionally heterogeneous (satellite, video/action, histopathology, texture, digit phrasings mixed together) — it is pooled from 16 benchmarks, so most entries are irrelevant to any single dataset *by design*. Suppressing those per class is the method. Do not prune it to "relevant" templates: that turns the experiment into the paper's "Human Selection" baseline, which CARPRT beats.

### Datasets layer

`datasets/__init__.py` holds `dataset_list`, a name→class registry consumed by `build_dataset`. Note the **two-level naming**: user-facing ids in `utils.build_test_data_loader` (`A`, `V`, `R`, `S`) are translated to registry keys (`imagenet-a`, …). Adding a dataset means touching both.

All loaders subclass `DatasetBase` (`datasets/utils.py`), which derives `num_classes` and `classnames` from the `test` split alone via `Datum(impath, label, domain, classname)` records. `classnames` ordering comes from sorted label ids and must line up with the `C` axis.

Two loading conventions coexist:
- **CoOp/Zhou-style splits** — most loaders (`caltech101`, `dtd`, `eurosat`, `sun397`, `ucf101`, `oxford_flowers`, `stanford_cars`, …) call `OxfordPets.read_split(split_json, image_dir)`. `OxfordPets` is imported as a static-method utility, not as a base class. Some datasets remap raw folder names to natural-language classnames (see `NEW_CLASSNAMES` in `eurosat.py`) — that remapping matters for prompt quality.
- **In-memory torchvision** — the CIFAR loaders materialize every PIL image into `datum.image` and set `impath='in_memory_image'`. `DatasetWrapper.__getitem__` branches on that exact sentinel string to decide between `read_image(path)` and the in-memory attribute.

`DatasetWrapper.__getitem__` builds a dict internally but returns only `(img, label)`. `build_data_loader` hardcodes `num_workers=0` (a deliberate workaround, per an inline comment) — data loading is single-threaded and is usually the bottleneck. Batch size is hardcoded to 512 in `build_test_data_loader`, not exposed as a flag.

`AugMixAugmenter` / `augmix` / `datasets/augmix_ops.py` are carried over from test-time-augmentation baselines (TPT-style) and are **not used** by the CARPRT path.

### CLIP

`clip/` is a vendored copy of OpenAI CLIP (`clip.py`, `model.py`, `simple_tokenizer.py` + BPE vocab). Weights auto-download to `~/.cache/clip`. `_MODELS` lists all OpenAI backbones, but `--backbone` restricts choices to `RN50` and `ViT-B/16` — widening the `choices` list is all that's needed to try others. `clip_model.dtype` is fp16 for these checkpoints, which is why `get_res_logits` casts the weights with `weights.type(data_type)` before the einsum; new tensors that meet text features need the same cast.

## Paper ↔ code

Notation map: paper `n` = code `num_prompt` (247), `C` = `num_class`, `m` = test-set size, `τ` = `--temp`, `W*` = `carprt_weight`, `s_{j,i,c}` = entries of `get_clip_logits`' `(P, C, N)` output.

| Paper | Code |
|---|---|
| Stage 1, Eq. 9 (score tensor) | `get_clip_logits` — `einsum('pcd,nd->pcn')` |
| Pseudo-labels `ŷ_{j,i} = argmax_c s_{j,i,c}` | `torch.max(logits, dim=1)` in `get_matrix` (per prompt, per image) |
| Eq. 10 (`w'_{i,c}`, mean score over images pseudo-labeled `c` under prompt `i`) | scatter-add sum ÷ count in `run_test_carprt` |
| Eq. 11 (softmax over prompts, per class, temperature τ) | `F.softmax(carprt_weight / temp, dim=0)` |
| Eq. 12 (`s_c(x) = Σ_i w*_{i,c} s_{*,i,c}`) | `get_res_logits` — collapses prompts first (`einsum('pc,pcd->cd')`), algebraically the same thing |

**Effective temperature.** Eq. 11 is stated over raw cosine similarities, but `clip_classifier` bakes `logit_scale.exp()` (≈100 for released OpenAI checkpoints) into the text features, so `w'` is on a ~100× scale before the softmax. `--temp 1.0` therefore corresponds to a temperature of ~0.01 on cosine similarity. If you ever remove the `logit_scale` multiply, the softmax collapses to near-uniform and the method silently degrades to MPE — τ must be rescaled to compensate.

**App. D.2 divergence.** The appendix describes `s_{j,i,c}` as a softmax over classes with a normalization scale `λ`; the code has no `λ` and aggregates raw scaled cosines. Likewise the frequency-bias correction of Eq. 16 (App. G.6) is not implemented — the code is the `none` scheme, which is what the main tables (1/15) report, so this matches the headline numbers.

### Reproduction gaps to know before trusting a run

- **ImageNet needs `--temp 1.5`.** App. C.3 uses τ=1.0 for fine-grained datasets but **τ=1.5 for ImageNet and its variants**. The code default is 1.0 for everything and `run.sh` does not override it.
- **The distribution-shift protocol (Table 2) is not implemented.** The paper estimates weights *once* on in-distribution ImageNet and transfers them to `-A/-R/-Sketch/-V2`. `run_test_carprt` re-estimates weights from each dataset's own test loader, so `--datasets I/A/V/R/S` produces per-dataset weights, not the transferred setting. Reproducing Table 2 requires hoisting weight estimation out of the per-dataset loop.
- **Not present at all:** iCARPRT (Alg. 2, App. E), the CARPRT-Uniform ablation (§5.3), pseudo-label filtering (App. G.1), DeCLIP backbones, Tiny-ImageNet, and the LLM-generated template pools (App. G.5).
- **Imbalance factor is hardcoded.** `IMBALANCECIFAR10.__init__` takes `imbalance_ratio=100`, but `build_dataset` only forwards `root`, so β is pinned at 100; Table 12's β=10/50 runs need the default changed. Note it applies the exponential-decay imbalance to the **test** split, which is correct here — CARPRT is transductive over test data.

### Expected numbers (CLIP ViT-B/16, Table 15)

Useful as a regression check after touching the weighting path:

| caltech101 | dtd | eurosat | fgvc | food101 | oxford_flowers | oxford_pets | stanford_cars | sun397 | ucf101 | I |
|---|---|---|---|---|---|---|---|---|---|---|
| 94.16 | 48.90 | 55.56 | 24.49 | 86.31 | 71.36 | 89.13 | 66.14 | 66.93 | 70.41 | 68.59 |

Mean prompt ensembling (uniform `W`) is the floor to compare against — e.g. 92.50 on Caltech101, 79.46 on Pets. If a change makes CARPRT land near those, prompt weighting has stopped doing anything.

## Adding a dataset

1. New module in `datasets/` subclassing `DatasetBase`, setting `self.template = template` and passing a `test` list of `Datum`.
2. Register it in `datasets/__init__.py` (`dataset_list`).
3. Add the user-facing id to the dispatch chain in `utils.build_test_data_loader`.
4. Document the expected directory layout under `--data-root`; `DatasetBase.download_data` uses `gdown` for Google-Drive archives.
