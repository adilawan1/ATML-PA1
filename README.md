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

Build the shared PACS split protocol once (used by both Task 2 and Task 3):

```bash
python -m shared.pacs_protocol --root /path/to/pacs
```

Build the CIFAR-10 90/10 split used by Task 4 (CIFAR-100 unknown groups are fixed, see
`task4/data/cifar100_unknowns.py`, and need no split file):

```bash
python -m task4.data.make_splits --data-root /path/to/cifar_root
```

Per-task instructions and exact commands live in each `taskN/README.md`. Every reported
number in the report should trace back to a file under the corresponding `taskN/results/`.

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
