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
  - TODO: Train the DRVI-inspired model with dropout or masking-aware objectives.
  - TODO: Track biological-signal preservation and robustness metrics beyond reconstruction.
  - Current CLI:
    - `python3 scripts/train.py --model ae`
    - Add `--data-dir <path>` to read processed data from somewhere other than `data/processed`.
    - Add `--output-dir <path>` to write model outputs somewhere other than `outputs/ae`.
    - Add `--latent-dim <n>`, `--hidden-dim <n>`, `--max-iter <n>`, `--batch-size <n>`, or `--learning-rate <value>` to tune the baseline autoencoder.
  - Current behavior for `--model ae`:
    - Trains a simple dense scikit-learn autoencoder on `data/processed/bulk_log_cpm.npy`.
    - Uses GTEx bulk log-CPM vectors as both input and target for reconstruction.
    - Saves a checkpoint and train/validation reconstruction metrics.
  - Current outputs:
    - `outputs/ae/ae_model.pkl`: saved autoencoder checkpoint.
    - `outputs/ae/ae_train_metrics.json`: train/validation MSE and MAE.
    - `outputs/ae/ae_train_indices.txt` and `outputs/ae/ae_val_indices.txt`: split indices.

- `scripts/evaluate.py`
  - TODO: Evaluate reconstruction quality on unmasked and artificially masked inputs.
  - TODO: Report gene-wise correlation, rank concordance, clustering or classification quality, and downstream donor/tissue prediction metrics.
  - Current CLI:
    - `python3 scripts/evaluate.py --model ae`
    - Add `--checkpoint <path>` to evaluate a checkpoint other than `outputs/ae/ae_model.pkl`.
    - Add `--data-dir <path>` to read processed data from somewhere other than `data/processed`.
    - Add `--output-dir <path>` to write evaluation outputs somewhere other than `outputs/ae`.
  - Current behavior for `--model ae`:
    - Loads the trained autoencoder.
    - Loads HCA donor pseudobulk raw counts from `data/processed/hca_pseudobulk_counts_by_donor.npy`.
    - Normalizes HCA pseudobulk with donor total counts from all genes, applies `log1p`, reconstructs with the AE, and reports reconstruction error.
  - Current outputs:
    - `outputs/ae/ae_hca_pseudobulk_eval_metrics.json`: aggregate HCA pseudobulk MSE and MAE.
    - `outputs/ae/ae_hca_pseudobulk_eval_by_donor.tsv`: donor-level MSE and MAE.
    - `outputs/ae/ae_hca_pseudobulk_reconstruction.npy`: reconstructed HCA donor pseudobulk matrix.

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

## Baseline Results

The current baseline is a simple dense autoencoder trained on GTEx whole-blood bulk log-CPM data and evaluated on HCA donor pseudobulk log-CPM data.

Latest run:

```text
GTEx bulk validation MSE: 0.100
GTEx bulk validation MAE: 0.234
HCA donor pseudobulk MSE: 1.236
HCA donor pseudobulk MAE: 0.898
```

The HCA pseudobulk error is higher than the held-out GTEx validation error, which is expected because the model is trained on GTEx bulk samples and evaluated on single-cell-derived pseudobulk. This gap is a useful baseline signal for bulk-to-pseudobulk domain shift.
