# ATML PA1 -- Beyond IID, Closed-Set Learning

EE-5102/CS-6304 Programming Assignment 1: inductive biases and representations (Task 1),
unsupervised domain adaptation (Task 2), domain generalization (Task 3), and open-set
recognition (Task 4).

All experiments use seed **6304** wherever the assignment specifies a seed.

## Repository layout

```
common/       shared utilities: seeding, logging/metric logging, evaluation metrics, plotting
shared/       PACS dataset loader + the single Task 2/3 train/val/target split protocol, MMD kernel
task1/        inductive biases & representations (STL-10 / Pets, ResNet-50, ViT-B/16, CLIP)
task2/        unsupervised domain adaptation on PACS (Source-only, DAN, DANN, CDAN)
task3/        domain generalization on PACS (ERM, DAN-DG, SAM)
task4/        open-set recognition on CIFAR-10 vs. CIFAR-100 unknowns (Vanilla, GCSC, PROSER[, RPL])
report/       figures pulled into the PDF report (report text/PDF itself is not committed here)
notebooks/    Colab bootstrap notebook
```

Each `taskN/` mirrors the structure suggested in the assignment: `configs/` (one YAML per
method, hyperparameters copied verbatim from the spec), `models/`, `methods/` (one file per
method), `evaluation/`, `train.py` / `evaluate_*.py` entry points, and `results/` (small
JSON/CSV/plot outputs only -- no checkpoints, no raw data).

## Setup

```bash
pip install -r requirements.txt
```

Datasets (not committed; see `.gitignore`):
- **STL-10** / **CIFAR-10** / **CIFAR-100**: fetched automatically via `torchvision.datasets`
  into whatever `--data-root` you pass.
- **PACS**: not distributed by torchvision. `python -m shared.prepare_pacs --out <pacs_root> --parquet <cache.parquet>`
  downloads the Hugging Face copy of the standard release (`flwrlabs/pacs`, 9,991 images) and writes the
  layout `<pacs_root>/<domain>/<class>/<image>` (domains `photo`, `art_painting`, `cartoon`, `sketch`; 7 classes);
  `python -m shared.verify_pacs --root <pacs_root>` checks the per-domain counts.

## Reproducing each task

Everything is driven by `notebooks/colab_setup.ipynb` (one section per task) or, equivalently, these commands.
Seed 6304 everywhere the assignment specifies a seed; small results (JSON/CSV/figures) are committed, checkpoints
and datasets are not.

```bash
# PACS (Tasks 2 and 3): Hugging Face copy -> folders, verify counts, build + commit the split (once)
python -m shared.prepare_pacs --out /content/pacs --parquet /path/to/cache/pacs_flwrlabs.parquet
python -m shared.verify_pacs  --root /content/pacs
python -m shared.pacs_protocol --root /content/pacs

# Task 1 (STL-10): interventions, then cue conflicts in the order documented in task1/README.md
python -m task1.scripts.run_task1 --data-root /path/to/data
python -m task1.data.make_cue_conflicts --data-root /path/to/data --preview   # calibrate, then freeze the rule
python -m task1.data.make_cue_conflicts --data-root /path/to/data
python -m task1.scripts.run_cue_conflicts --data-root /path/to/data

# Task 2, then Task 3 (needs Task 2's Source-only checkpoint); pick each task's controlled study
python -m task2.run_experiments --pacs-root /content/pacs --ckpt-root /path/to/checkpoints --study dan
python -m task3.run_experiments --pacs-root /content/pacs --ckpt-root /path/to/checkpoints --study sam
python -m task3.evaluate_sketch --pacs-root /content/pacs --ckpt-root /path/to/checkpoints

# Task 4 (CIFAR-10 known / CIFAR-100 unknown): see task4/README.md (Vanilla, GCSC, PROSER, evaluation)
python -m task4.data.make_splits --data-root /path/to/cifar_root
```

Per-task details, design notes and the exact outputs live in each `taskN/README.md`. Every reported number
should trace back to a file under the corresponding `taskN/results/`.

## Before submitting

```bash
python tools/check_submission.py                                   # PASS / MISSING / WARN for every Required-Evidence item + repo state
python tools/check_submission.py --write-map report/evidence_map.md  # which committed file backs each item
```

The audit checks the committed result tables/figures/logs of all four tasks, the split-index files (PACS, STL-10,
CIFAR-10), that each task's `hypotheses.md` is filled in and was committed, the recorded environment (`env/`), and that the
repository is clean, pushed, public, and free of checkpoints, datasets and large files.

## Running on Colab

See `notebooks/colab_setup.ipynb`. In short: mount Drive for datasets/checkpoints (git-ignored,
large, must survive a session disconnect), clone this repo into `/content/` for code (git is
the source of truth for the directory structure), and run everything as
`!python -m task2.train --config task2/configs/dann.yaml` style commands rather than inlining
logic into notebook cells, so the same scripts work locally.

## Attribution

External code and models used (none is vendored into this repo; third-party code is cloned at
runtime into the git-ignored `third_party/`):

- **AdaIN style transfer** (Task 1 cue conflicts): [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
  (MIT), pinned to commit `47950d0`, implementing Huang & Belongie, "Arbitrary Style Transfer in
  Real-time with Adaptive Instance Normalization" (ICCV 2017). We use its `net.decoder`/`net.vgg`
  definitions, its `function.adaptive_instance_normalization`, and its released `decoder.pth` /
  `vgg_normalised.pth` weights unmodified; our wrapper (`task1/data/adain.py`) only re-states the
  short `style_transfer` routine from its `test.py` so it can run on batches.
- **Pretrained backbones**: torchvision ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`),
  ResNet-18 (`IMAGENET1K_V1`); OpenCLIP ViT-B-32 (`pretrained='openai'`).
- **PACS**: the Hugging Face copy [`flwrlabs/pacs`](https://huggingface.co/datasets/flwrlabs/pacs) of the standard
  release (per-domain and per-class counts verified against the published statistics: 1670 / 2048 / 2344 / 3929
  images); original data: Li et al., "Deeper, Broader and Artier Domain Generalization", ICCV 2017.
  (The Google Drive link used by DomainBed's download script was refused by `gdown` with a quota error.)
- The Task 3 `SAMOptimizer` follows the standard public two-step SAM pattern (Foret et al., 2021;
  cf. the widely used davda54/sam implementation); Task 4 PROSER follows Zhou et al. (2021).
