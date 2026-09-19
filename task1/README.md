# Task 1 -- Inductive Biases and Feature Representations (STL-10)

## Status
- [x] `data/make_subset.py` -- class-balanced 500-image STL-10 test subset, seed 6304.
- [x] `data/transforms.py` -- grayscale, hue rotation, translation (reflection pad + shifted
      crop, 4 cardinal directions), patch shuffle (4x4 grid, seed 6304).
- [x] `models/backbones.py` -- frozen ResNet-50 / ViT-B/16 / CLIP ViT-B/32 wrappers, shared
      "common 224x224 tensor -> per-model normalize -> features" interface.
- [x] `analysis/feature_similarity.py` (cosine stability I_T), `analysis/representation.py`
      (t-SNE/UMAP + the required clean-vs-transformed plot), `analysis/evaluate_bias.py`
      (shape-bias/coverage formulas).
- [ ] `data/make_cue_conflicts.py` -- needs a public AdaIN implementation wired in (Day 2);
      rejection rule is pre-registered in the module docstring.
- [ ] `scripts/run_task1.py` -- the actual end-to-end driver tying everything above together.

## Experimental-design choices to state before interpreting results (per the assignment)
- Dataset: STL-10 (recommended, lower compute).
- Cue-conflict class pairs + style strength: TBD when `make_cue_conflicts.py` is implemented.
- Additional color intervention: hue rotation (`transforms.hue_rotate`), chosen because it
  changes chromatic identity while preserving luminance/geometry -- contrasts with grayscale,
  which removes color entirely.
- Representation-visualization method: t-SNE (`analysis/representation.fit_tsne`) unless UMAP
  is substituted; settings (perplexity, seed) are logged alongside the figure.

## Commands

```bash
python -m task1.data.make_subset --data-root /path/to/stl10
python -m task1.scripts.run_task1   # once implemented
```
