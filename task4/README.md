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
- [ ] `methods/proser.py` -- architecture pieces in place; classifier/data-placeholder losses
      and the placeholder detection score are TODO (read Zhou et al. 2021 closely first).
- [ ] `methods/rpl.py` -- optional, deferred.
- [x] `extract_outputs.py` -- caches penultimate features + logits (unaugmented) for
      CIFAR-10 train/val/test and both CIFAR-100 unknown groups, per checkpoint.
- [x] `evaluate_osr.py` -- builds both required tables (post-hoc scores on Vanilla;
      Vanilla/GCSC[/PROSER] via MLS), the 3-panel score-distribution figure, and the
      incorrectly-accepted near/far failure cases. Table 2 and the figure/analysis
      auto-extend once a `proser_*.pt` cache exists; a second PROSER row using its
      placeholder score still needs writing once PROSER itself is implemented.

## Commands

```bash
python -m task4.data.make_splits --data-root /path/to/cifar_root

python -m task4.train --config task4/configs/vanilla.yaml --data-root /path/to/cifar_root
python -m task4.train --config task4/configs/gcsc.yaml    --data-root /path/to/cifar_root
# PROSER once implemented:
python -m task4.train --config task4/configs/proser.yaml  --data-root /path/to/cifar_root

python -m task4.extract_outputs --checkpoint task4/results/vanilla/checkpoint.pt --data-root /path/to/cifar_root --method-name vanilla
python -m task4.extract_outputs --checkpoint task4/results/gcsc/checkpoint.pt    --data-root /path/to/cifar_root --method-name gcsc

python -m task4.evaluate_osr --data-root /path/to/cifar_root
```

## Hard constraint
CIFAR-100 (both near and far groups) is evaluation-only: no CIFAR-100 image may influence
training, checkpoint selection, score definitions, or threshold calibration. Thresholds are
calibrated on CIFAR-10 validation scores only (`task4.evaluation.thresholds.evaluate_score`).
