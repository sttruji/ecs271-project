# ECS 271 Project

This repository is a starter scaffold for a dropout-robust transcriptomics representation learning project inspired by the course proposal and the DRVI paper:

- Proposal direction: bulk-first, dropout-aware RNA-seq encodings for comparing healthy bulk RNA-seq with single-cell-derived pseudobulk data.
- Paper inspiration: Disentangled Representation Variational Inference (DRVI), a variational model for interpretable single-cell omics factors.

The Python files below are intentionally empty placeholders. Implementation details should be added as the project pipeline becomes concrete.

## Project Scripts

- `scripts/load_data.py`
  - TODO: Download or import the raw datasets used by the project.
  - TODO: Start with healthy whole-blood bulk RNA-seq from GTEx.
  - TODO: Optionally add healthy bulk RNA-seq cohorts from NCBI GEO if more donor or tissue coverage is needed.
  - TODO: Pull healthy single-cell RNA-seq datasets from HCA or comparable annotated atlases for pseudobulk comparison.
  - TODO: Keep any donor-matched or tissue-matched bulk/single-cell benchmark data reserved for evaluation only.
  - TODO: Write raw or minimally processed files into `data/raw/`.

- `scripts/preprocess_data.py`
  - TODO: Read raw inputs from `data/raw/`.
  - TODO: Normalize, filter, and align gene identifiers across bulk and single-cell datasets.
  - TODO: Split donors into train, validation, and test sets while avoiding donor leakage.
  - TODO: Build single-cell pseudobulk summaries and any cell-type-resolved comparison tables.
  - TODO: Save cleaned matrices, metadata, masks, and split files into `data/processed/`.

- `models/drvi_model.py`
  - TODO: Implement the DRVI-inspired variational model architecture.
  - TODO: Include an encoder, structured latent representation, additive or factorized decoder design, and masking-aware reconstruction objective.
  - TODO: Adapt the model for the project goal: bulk-first training with single-cell/pseudobulk comparison rather than direct single-cell-only training.
  - TODO: Expose reusable model classes or factory functions for the training and evaluation scripts.

- `scripts/train.py`
  - TODO: Load processed bulk training data from `data/processed/`.
  - TODO: Train the DRVI-inspired model with dropout or masking-aware objectives.
  - TODO: Track validation reconstruction, biological-signal preservation, and robustness metrics.
  - TODO: Save trained checkpoints, configs, and training logs into `outputs/`.

- `scripts/evaluate.py`
  - TODO: Load trained checkpoints from `outputs/` and held-out test data from `data/processed/`.
  - TODO: Evaluate reconstruction quality on unmasked and artificially masked inputs.
  - TODO: Compare bulk embeddings against single-cell-derived pseudobulk representations.
  - TODO: Report gene-wise correlation, rank concordance, clustering or classification quality, and downstream donor/tissue prediction metrics.
  - TODO: Save metrics, figures, and result tables into `outputs/`.

## Suggested Directory Layout

```text
data/
  raw/            # Downloaded or imported source datasets
  processed/      # Cleaned matrices, metadata, masks, and splits
models/
  drvi_model.py   # Empty placeholder for the DRVI-inspired model
notebooks/        # Exploration and analysis notebooks
outputs/          # Checkpoints, logs, metrics, figures, and tables
scripts/
  load_data.py
  preprocess_data.py
  train.py
  evaluate.py
```
