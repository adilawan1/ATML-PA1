# Task 3 -- Domain Generalization (PACS, unseen target = Sketch)

## Status
- [x] Reuses Task 2's backbone/BatchNorm policy (`task3/models/backbone.py`) and evaluation
      helpers unchanged.
- [x] `erm.py` -- loads the Task 2 Source-only checkpoint (not retrained).
- [x] `sam.py` -- `SAMOptimizer` (two-step ascent/descent) fully implemented; `train_sam`
      training-loop wiring left as TODO.
- [ ] `dan_dg.py` -- stub wired to `shared/mmd.py`; loop needs filling in.
- [x] `evaluation/source_domain_separability.py` (3-way Photo/Art/Cartoon classifier) and
      `evaluation/sharpness.py` (local sharpness proxy) fully implemented.
- [ ] `evaluate_sketch.py` -- the only script allowed to touch Sketch labels.

## Hard constraint
No Sketch image may be loaded by training, selection, or diagnostics -- only by
`evaluate_sketch.py`, and only after every Task 3 setting is frozen. `train.py` deliberately
never reads `protocol["sketch"]` to make this structurally hard to violate by accident.

## Commands

```bash
python -m task3.train --config task3/configs/erm.yaml       # just loads/verifies the Task 2 checkpoint
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml

python -m task3.evaluate_sketch
```
