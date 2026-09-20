# Task 1 -- Inductive Biases and Feature Representations (STL-10)

Frozen ResNet-50 (IMAGENET1K_V2), ViT-B/16 (IMAGENET1K_V1) and OpenCLIP ViT-B-32 (`openai`),
each with a trained linear head, plus zero-shot CLIP ("a photo of a {class}."). Four
"predictors" are evaluated everywhere: `resnet50`, `vit_b_16`, `clip_head`, `clip_zeroshot`.

## Status
- [x] Clean baseline, color bias (grayscale + 180-degree hue rotation), translation curve,
      patch shuffle, cosine representation stability, t-SNE grid -- `scripts/run_task1.py`.
- [x] Shape vs. texture cue conflicts (Step 3): `data/adain.py` (public pytorch-AdaIN wrapper),
      `data/make_cue_conflicts.py` (generation + pre-registered rejection rule; loads no
      classifier), `scripts/run_cue_conflicts.py` (evaluation, shape bias / coverage, examples).
      Settings to choose and freeze *before* generating: `alpha`, class pairs, rule thresholds.
- [ ] Hypotheses written in `hypotheses.md` before each result is interpreted (yours to write).

## Layout
- `configs/task1.yaml` -- every hyperparameter (seed 6304, head training, interventions, t-SNE).
- `data/` -- `make_subset.py` (class-balanced 500-image test subset, ids saved to
  `eval_subset_seed6304.json`), `stl10.py` (stratified 80/20 train/val split), `transforms.py`
  (grayscale, hue rotation, reflection-pad translation, patch shuffle, all on a common
  un-normalized 224x224 tensor).
- `models/backbones.py` -- frozen backbones; per-model normalization is applied only inside
  `FrozenBackbone.features`, so every model receives identical pixels. `models/heads.py` --
  linear head (AdamW lr 1e-3, wd 1e-4, <=50 epochs, early stopping patience 5).
- `analysis/evaluator.py` -- one forward pass per backbone yields the four predictors' logits and
  the frozen features; `evaluate_bias.py` (shape bias / coverage), `feature_similarity.py`
  (cosine stability I_T), `representation.py` (t-SNE).

## Commands

```bash
python -m task1.scripts.run_task1 --data-root /path/to/data     # Steps 1, 2, 4, 5, 6

# Step 3 -- in this order (see the make_cue_conflicts docstring for the rejection rule):
python -m task1.data.make_cue_conflicts --data-root /path/to/data --preview   # calibrate alpha/thresholds by eye
#   ... set cue_conflicts.{alpha,pairs,rejection_rule} in configs/task1.yaml and freeze them ...
python -m task1.data.make_cue_conflicts --data-root /path/to/data             # generate the accepted set
python -m task1.scripts.run_cue_conflicts --data-root /path/to/data           # evaluate all predictors
```

Outputs: `results/task1_results.json` (every number in the report), `results/compact_comparison.csv`,
`report/figures/task1_translation_curve.png`, `report/figures/task1_tsne.png`; for Step 3
`results/cue_conflicts_meta.json` (rule, per-cell accepted/rejected counts, every kept and rejected item),
`results/cue_conflicts.json` + `cue_conflicts_summary.csv` (decision counts, shape bias, coverage), and
the accepted/rejected/examples/t-SNE figures.

## Notes for interpreting results
- The linear head is trained on raw frozen features (no standardization, as specified). CLIP's
  unit-norm embeddings give small logits at lr 1e-3, so its head's confidence is low even when
  accuracy is high -- relevant when comparing it with zero-shot CLIP.
- OpenCLIP is built with `force_quick_gelu=True`: the `openai` weights expect QuickGELU, and open_clip 3.x
  otherwise builds plain GELU for `ViT-B-32` + `openai` (it prints a mismatch warning).
- Translation at delta = 0 is the identity, so consistency is exactly 1 there.
- t-SNE coordinates are only comparable *within* one panel (one joint fit per backbone x condition).
