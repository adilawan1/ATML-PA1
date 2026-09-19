# Task 2 -- Unsupervised Domain Adaptation (PACS, target = Sketch)

## Status
- [x] Shared PACS protocol (`shared/pacs_protocol.py`), backbone, discriminator, BatchNorm
      freezing policy.
- [x] `source_only.py` fully implemented (also the Task 3 ERM baseline -- do not retrain it there).
- [ ] `dan.py`, `dann.py`, `cdan.py` -- stubs with the exact loss/architecture wired to shared
      pieces (`shared/mmd.py`, `task2/models/domain_discriminator.py`); loop needs filling in.
- [ ] `evaluate_final.py` -- common evaluation + alignment diagnostic across all four methods.

## Commands

```bash
# one-time, shared with Task 3
python -m shared.pacs_protocol --root /path/to/pacs

python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml

python -m task2.evaluate_final
```

## Notes
- BatchNorm running stats are frozen at ImageNet values for every method (`freeze_batchnorm_stats`,
  called after every `model.train()`); gamma/beta stay trainable.
- Target class labels must never be read before `evaluate_final.py`. Target domain identity
  (i.e. that an image is unlabeled Sketch) is fine to use; target labels are not.
- The controlled design study (Step 6) reuses whichever of `dan.yaml`/`dann.yaml` you pick --
  see the `controlled_study` block in each config -- and must not feed back into the main
  lambda_mmd=1 / max_alpha=1 comparison.
