# Task 3 -- Domain Generalization (PACS; unseen target = Sketch)

Same protocol as Task 2 (`base.yaml`): identical splits, ResNet-18 initialization, preprocessing, domain-balanced
batches (8 per source), optimizer, budget, early stopping (mean source-val macro-F1) and seed 6304.

## Methods
- **ERM** = the Task 2 Source-only checkpoint, loaded unchanged (never retrained here).
- **DAN-DG** (`methods/dan_dg.py`): ERM loss + (lambda_DG / 3) x sum of MMD^2 over the three source-domain pairs,
  same MMD/kernel as Task 2's DAN; lambda_DG = 1.
- **SAM** (`methods/sam.py`): non-adaptive SAM, rho = 0.05, AdamW base optimizer, two passes per step, BatchNorm
  statistics frozen in both.

## Hard constraint: no Sketch before the final evaluation
`run_experiments.py` builds `PACSData(include_target=False)`: no Sketch file is opened, decoded or cached (its
cache holds only source images). Training, checkpoint selection and the source-side diagnostics all run there.
`evaluate_sketch.py` is the only place Sketch is opened. The test suite runs the training script with the `sketch/`
folder physically hidden to verify this.

## Commands
```bash
python -m task3.run_experiments --pacs-root /content/pacs --ckpt-root <drive>/checkpoints --study sam   # or dan_dg
python -m task3.evaluate_sketch --pacs-root /content/pacs --ckpt-root <drive>/checkpoints
```
Outputs (`results/`): `<run>/metrics.jsonl`, `source_diagnostics.json` (per-source metrics, source-domain
separability, sharpness proxy), `sketch_results.json`, `summary.csv`, `class_analysis.json`,
`task2_vs_task3.csv`, `study.csv`; figure `report/figures/task3_curves.png`.

## Diagnostics
- **Source-domain separability**: 3-way (Photo/Art/Cartoon) multinomial logistic regression, C = 1, on balanced
  frozen source-val features (equal images per domain), 70/30 split, seed 6304; chance = 33.3%.
- **Sharpness proxy**: `L(theta + eps) - L(theta)`, `eps = 0.05 * g / ||g||` on ONE fixed validation batch (32
  images per source, seed 6304), eval mode -- identical for every model; a local diagnostic, not a global claim.
- `hypotheses.md`: your expected effects, committed before launching; Task 2's Sketch results must not influence
  any Task 3 setting.
