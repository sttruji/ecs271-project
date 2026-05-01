# ECS 271 Project

This repository is a starter scaffold for a dropout-robust transcriptomics representation learning project inspired by the course proposal and the DRVI paper:

- Proposal direction: bulk-first, dropout-aware RNA-seq encodings for comparing healthy bulk RNA-seq with single-cell-derived pseudobulk data.
- Paper inspiration: Disentangled Representation Variational Inference (DRVI), a variational model for interpretable single-cell omics factors.

The Python files below are intentionally empty placeholders. Implementation details should be added as the project pipeline becomes concrete.

## Project Scripts

- `scripts/load_data.py`
  - TODO: Optionally add healthy bulk RNA-seq cohorts from NCBI GEO if more donor or tissue coverage is needed.
  - TODO: Keep any donor-matched or tissue-matched bulk/single-cell benchmark data reserved for evaluation only.
  - Current CLI:
    - `python3 scripts/load_data.py --list`
    - `python3 scripts/load_data.py --whole_blood`
    - `python3 scripts/load_data.py --hca_blood_scrna`
    - Dataset flags also accept hyphens, e.g. `--whole-blood` and `--hca-blood-scrna`.
    - Add `--force` to re-download an existing file.
    - Add `--data-dir <path>` to write somewhere other than `data/`.
  - Current datasets:
    - `whole_blood`: GTEx v11 bulk RNA-seq whole-blood gene read counts.
      - Output: `data/raw/gene_reads_v11_whole_blood.gct.gz`
      - Download size: about 34 MB compressed, about 150 MB uncompressed.
    - `hca_blood_scrna`: Human Cell Atlas peripheral blood single-cell RNA-seq gene expression matrix.
      - Output: `data/raw/BL_standard_design.h5ad`
      - Download size: about 2.1 GB.

- `scripts/preprocess_data.py`
  - TODO: Split donors into train, validation, and test sets while avoiding donor leakage.
  - TODO: Save split files and any artificial dropout/masking files into `data/processed/`.
  - Current CLI:
    - `python3 scripts/preprocess_data.py`
    - Add `--raw-dir <path>` to read from somewhere other than `data/raw`.
    - Add `--output-dir <path>` to write somewhere other than `data/processed`.
    - Add `--top-genes <n>` to change the number of selected shared variable genes. Default: `2000`.
    - Add `--chunk-size <n>` to tune HCA sparse-matrix processing memory use. Default: `5000`.
  - Current behavior:
    - Reads GTEx bulk counts from `data/raw/gene_reads_v11_whole_blood.gct.gz`.
    - Reads HCA single-cell raw counts from `data/raw/BL_standard_design.h5ad`.
    - Aligns both datasets by Ensembl gene ID after removing version suffixes.
    - Found 24,461 overlapping genes in the current raw inputs.
    - Normalizes counts to counts per million, applies `log1p`, ranks genes by combined bulk/single-cell variance, and keeps the top 2,000 genes by default.
    - Builds HCA pseudobulk count matrices by donor and by donor-cell-type.
  - Current outputs in `data/processed/`:
    - `bulk_counts.npy`: GTEx raw counts, shape `(803, 2000)`.
    - `bulk_log_cpm.npy`: GTEx log-CPM matrix, shape `(803, 2000)`.
    - `bulk_sample_ids.txt`: GTEx sample IDs.
    - `hca_counts_csr.npz`: HCA selected raw counts as CSR sparse arrays, shape `(323269, 2000)`.
    - `hca_log_cpm_csr.npz`: HCA selected log-CPM values as CSR sparse arrays, shape `(323269, 2000)`.
    - `hca_cell_metadata.tsv`: HCA cell barcode, donor, channel, cell type, total counts, detected genes, and mitochondrial percentage.
    - `hca_pseudobulk_counts_by_donor.npy`: HCA raw counts aggregated by donor, shape `(8, 2000)`.
    - `hca_pseudobulk_counts_by_donor_celltype.npy`: HCA raw counts aggregated by donor-cell-type, shape `(120, 2000)`.
    - `hca_pseudobulk_donor_celltype_metadata.tsv`: row metadata for donor-cell-type pseudobulk matrix.
    - `hca_donors.txt` and `hca_cell_types.txt`: category labels.
    - `gene_metadata.tsv`: selected gene IDs, symbols, and variance scores.

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

## Data Inspection

### GTEx Whole-Blood Bulk RNA-seq

- File: `data/raw/gene_reads_v11_whole_blood.gct.gz`
- Format: GCT v1.2 raw gene read-count matrix.
- Shape: 74,628 genes/features by 803 samples.
- Donors: 803 unique GTEx donor/sample prefixes, appearing as one whole-blood sample per donor in this file.
- Sparsity: about 61.02% zeros.
- All-zero genes/features: 3,807.
- Median total reads per sample: 42,081,277.

Each row is one gene or feature. The first two columns are the Ensembl gene ID and gene symbol, then each remaining column is one GTEx whole-blood sample.

```text
Name                Description   GTEX-1117F-0005-SM-HL9SH   GTEX-111CU-0005-SM-GJ3PH
ENSG00000310526.1   WASH7P        90                          279
```

For modeling, one natural instance is one bulk sample/donor represented as a vector of 74,628 raw gene counts.

### HCA Peripheral Blood Single-Cell RNA-seq

- File: `data/raw/BL_standard_design.h5ad`
- Format: AnnData `.h5ad`.
- Shape: 323,269 cells by 25,825 genes.
- Donors: 8.
- Channels/sample batches: 64.
- Cell annotations: 15.
- Sparse nonzero values: 311,884,272.
- Sparsity: about 96.26% zeros.
- Highly variable features marked in metadata: 2,000.

Each row is one cell. Cell metadata includes barcode, donor, channel, cell type annotation, total counts, detected genes, and mitochondrial percentage. The `.h5ad` stores transformed expression in `X` and raw UMI counts in `raw/X`.

Example cell:

```text
barcode: BL1_1-AAACCTGAGAGGGATA
donor: 1
channel: BL1_1
annotation: Cytotoxic T cells
raw counts: 3440
detected genes: 1113
```

For modeling or pseudobulk construction, one natural instance is one cell represented as a sparse vector over 25,825 genes. For comparison with bulk RNA-seq, cells can later be aggregated by donor, channel, or cell type.

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
