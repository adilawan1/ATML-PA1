# Task 4 -- Open-Set Recognition (CIFAR-10 known, CIFAR-100 near/far unknown)

## Status
- [x] `models/resnet_cifar.py` -- CIFAR ResNet-18 (3x3/stride-1 stem, no max-pool), plus a
      layer2/layer3 split forward pass for PROSER's manifold mixup.
- [x] `data/cifar10.py` (90/10 stratified split, seed 6304), `data/cifar100_unknowns.py`
      (fixed near/far groups, 800 images each, CIFAR-100 TEST only).
- [x] `scores/{msp,mls,energy,mahalanobis}.py` -- fully implemented, all read the same cached
      logits/features.
- [x] `evaluation/thresholds.py` -- AUROC + 95th-percentile-on-known-val rejection protocol.
- [x] `methods/vanilla.py`, `methods/gcsc.py` -- complete training loops.
- [x] `methods/manifold_mixup.py` -- complete (cross-class pairing + Beta(2,2) mix).
- [x] `methods/proser.py` -- PROSER (Zhou et al., 2021): 5 dummy heads (max-combined), classifier placeholders
      (Eq. 5, beta = 1) on the first half of each batch, manifold-mixup data placeholders (Eq. 7, layer2,
      Beta(2,2), cross-class pairs, gamma = 0.1) on the second half; checkpoint by known-logit val accuracy;
      placeholder detection score from the authors' reference code (T = 1024). Tested on real CIFAR-10.
- [ ] `methods/rpl.py` -- optional, deferred.
- [x] `extract_outputs.py` -- caches penultimate features + logits (unaugmented) for
      CIFAR-10 train/val/test and both CIFAR-100 unknown groups, per checkpoint.
- [x] `evaluate_osr.py` -- builds both required tables (post-hoc scores on Vanilla;
      Vanilla/GCSC[/PROSER] via MLS), the 3-panel score-distribution figure, and the
      incorrectly-accepted near/far failure cases. Table 2 picks up PROSER automatically once a
      `proser_*.pt` cache exists, with two PROSER rows: MLS on the known logits and the placeholder score.

## Commands

```bash
python -m task4.data.make_splits --data-root /path/to/cifar_root

python -m task4.train --config task4/configs/vanilla.yaml --data-root /path/to/cifar_root
python -m task4.train --config task4/configs/gcsc.yaml    --data-root /path/to/cifar_root
# PROSER (initialized from the Vanilla checkpoint; see notebooks/colab_setup.ipynb for the cell):
#   from task4.methods.proser import train_proser; train_proser(data_root, vanilla_checkpoint=..., ...)

python -m task4.extract_outputs --checkpoint task4/results/vanilla/checkpoint.pt --data-root /path/to/cifar_root --method-name vanilla
python -m task4.extract_outputs --checkpoint task4/results/gcsc/checkpoint.pt    --data-root /path/to/cifar_root --method-name gcsc

python -m task4.extract_outputs --checkpoint <proser ckpt> --data-root /path/to/cifar_root --method-name proser
python -m task4.evaluate_osr --data-root /path/to/cifar_root
```

Note: `train_vanilla` now seeds itself (seed 6304). Vanilla/GCSC were first trained before that change, i.e.
unseeded -- retrain them (delete their checkpoints and rerun) for exact reproducibility.

## Hard constraint
CIFAR-100 (both near and far groups) is evaluation-only: no CIFAR-100 image may influence
training, checkpoint selection, score definitions, or threshold calibration. Thresholds are
calibrated on CIFAR-10 validation scores only (`task4.evaluation.thresholds.evaluate_score`).
