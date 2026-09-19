from __future__ import annotations

"""Task 3 final evaluation -- the ONLY script allowed to load Sketch labels, and only after
every Task 3 model/config/checkpoint decision has been frozen. Reports, for ERM, DAN-DG, SAM:
- per-source-domain and mean/worst-source accuracy & macro-F1 (already available pre-Sketch)
- Sketch accuracy & macro-F1, and the change vs. ERM
- source-domain separability (task3.evaluation.source_domain_separability)
- the local sharpness proxy (task3.evaluation.sharpness) on the fixed seed-6304 validation batch
- per-class Sketch accuracy shifts vs. ERM (task3.evaluation.domain_metrics)

TODO (Task 3, Day 4, after ERM/DAN-DG/SAM checkpoints all exist):
1. Load each checkpoint.
2. Evaluate on the full target split from `shared/pacs_protocol.load_pacs_protocol(...)["sketch"]["all"]`.
3. Assemble one table into `task3/results/summary.json` for the report.
"""

if __name__ == "__main__":
    raise NotImplementedError("Fill in per the module docstring once all Task 3 checkpoints exist.")
