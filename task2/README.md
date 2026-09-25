# Task 2 -- Unsupervised Domain Adaptation (PACS; source = Photo/Art/Cartoon, target = Sketch)

ResNet-18 (IMAGENET1K_V1) fine-tuned end to end with a 7-way head; BatchNorm running statistics frozen at
ImageNet values (`freeze_batchnorm_stats`, applied after every `model.train()`); AdamW lr 1e-4 / wd 1e-4;
<= 30 epochs; early stopping after 5 epochs without improvement in mean source-validation macro-F1; batches of
8 images per source domain (+ 24 unlabeled Sketch images for UDA methods); seed 6304 throughout.

## Status
- [x] `shared/trainer.py` (one loop for every method), `methods/{source_only,dan,dann,cdan}.py`.
- [x] `evaluate_final.py` (source-val per domain, target acc/F1, change vs. Source-only, domain separability,
      per-class target accuracy + dominant confusions), curves via `shared/curves.py`.
- [x] `run_experiments.py`: everything in one resumable command incl. your chosen controlled study.
- [ ] `hypotheses.md` -- your expected effect of stronger alignment, committed BEFORE launching.

## Commands
```bash
python -m shared.prepare_pacs --out /content/pacs --parquet <cache>.parquet   # once per session (see repo README)
python -m shared.pacs_protocol --root /content/pacs                            # once; commit the split json
python -m task2.run_experiments --pacs-root /content/pacs --ckpt-root <drive>/checkpoints --study dan   # or dann
python -m task2.train --run dann --pacs-root ... --ckpt-root ...               # a single run
```
Outputs (`results/`): `<run>/metrics.jsonl` (per-epoch losses, alignment term, domain accuracy, per-domain val),
`<run>/run_summary.json`, `summary.json` / `summary.csv` (the main comparison table), `class_analysis.json`;
figures `report/figures/task2_curves.png`, `task2_study_<dan|dann>_curves.png`.

## Design notes
- **Leakage**: the trainer's `Batch` has no target-label field; `PACSData.labels()` refuses the Sketch domain;
  Sketch labels are read only via `target_labels_for_final_eval()` inside `evaluate_final.py`.
- **Speed**: images are decoded once and resized to 256x256 (the spec's Resize), held as uint8, then randomly
  cropped/flipped per batch -- the same preprocessing without per-step JPEG decoding.
- **Same across methods**: initialization (seeded), the source batch sequence (its own generator), augmentation,
  optimizer, budget and stopping rule -- only `Method.step` differs.
- **MMD** uses `exp(-d^2 / gamma)` with gamma = {0.5, 1, 2} x the median pairwise squared distance of the combined
  batch (DAN's convention), 24 source vs. 24 target features per step.
- **Domain separability**: frozen backbone, equal numbers of source-val and (randomly drawn) target features,
  70/30 split (seed 6304), balanced logistic regression C = 1; 50% = chance. Lower is not automatically better.
- **DANN/CDAN training instability**: both showed a numerical blowup (source accuracy collapsing well below
  Source-only, huge/erratic losses) traced to unbounded feature scale feeding the discriminator through GRL.
  Gradient clipping (`max_norm=1.0`) + L2-normalizing the discriminator's input (`dann.py`/`cdan.py`) stabilized
  **CDAN** (adopted as the reported result) but did **not** stabilize **DANN**, which remains unstable under
  identical fixes -- `task2/results/dann/` predates this code change and reflects the original, unfixed run.
