from __future__ import annotations

"""Orchestrates Task 1 end to end. Each step is runnable independently once its inputs exist,
since the assignment's steps are sequential but the compute is cheap enough to iterate on one
step at a time during development.

Steps (see task1/README.md for status):
1. Clean baseline: extract features for all 3 backbones + zero-shot CLIP on the eval subset,
   train linear heads (`task1/configs/linear_head.yaml`), report accuracy/macro-F1/mean-max-confidence.
2. Color bias: grayscale + hue_rotate (`task1/data/transforms.py`) on the eval subset.
3. Shape vs. texture: `task1/data/make_cue_conflicts.py` + `task1/analysis/evaluate_bias.py`.
4. Translation: `translate_all_directions` at deltas {0, 8, 16, 32}, avg over directions.
5. Patch structure: `patch_shuffle` with a seed-6304 generator, same shuffled images for all models.
6. Representation analysis: `task1/analysis/feature_similarity.py` + `representation.py`.

TODO (Task 1, Day 1-2): implement this as the actual driver once make_subset.py has been run
and backbones.py / transforms.py have been smoke-tested on a handful of images.
"""

if __name__ == "__main__":
    raise NotImplementedError("Fill in per the module docstring -- see task1/README.md for the day-by-day plan.")
