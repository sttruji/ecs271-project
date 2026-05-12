# Single-cell foundation models — what we used and didn't

This document records which scFM design choices we adopted, which we explicitly skipped, and why — in the context of our small-data (16-donor) cross-modality VAE on GTEx + Eraslan paired bulk/sn data.

## Models surveyed

| Model | Paper | Core design |
|---|---|---|
| **scGPT** | [Cui et al. *Nat Methods* 2024](https://www.nature.com/articles/s41592-024-02201-0) | Transformer over gene tokens; value-binning of expression; condition tokens for batch/perturbation/modality |
| **scFoundation** | [Hao et al. *Nat Methods* 2024](https://www.nature.com/articles/s41587-024-02486-8) | xTrimoGene 100M-param transformer; read-depth recovery pretraining task; asymmetric encoder-decoder |
| **CellPLM** | [Wen et al. *ICLR* 2024](https://openreview.net/pdf?id=BKXvPDekud) | Flowformer (cell-cell attention); learnable batch lookup table; downstream MLP heads |
| **scVI / scANVI** | [Lopez et al. *Nat Methods* 2018](https://www.nature.com/articles/s41592-018-0229-2) | VAE with NB/ZINB likelihood; one-hot batch covariate fed to encoder + decoder; scANVI adds semi-supervised classification head on z |
| **Geneformer** | [Theodoris et al. *Nature* 2023](https://www.nature.com/articles/s41586-023-06139-9) | Rank-based gene tokens; pretrained on Genecorpus-30M; technology-agnostic by construction |
| **Cell2Sentence** | [Levine et al. *ICML* 2024](https://arxiv.org/abs/2402.03540) | Text-based; bypasses gene-vector framing entirely |
| **SAMS-VAE** | [Bereket & Karaletsos *NeurIPS* 2023](https://arxiv.org/abs/2311.02794) | Sparse additive mechanism-shift VAE; the paper our proposal anchors on |
| **DRVI** | [Kanwar et al. *bioRxiv* 2024](https://www.biorxiv.org/content/10.1101/2024.10.16.618579) | Disentangled representation VAE for cells |

## What we used (and from where)

### 1. Condition tokens for technology — **scGPT-style**
Added a categorical `tech_idx` (bulk_illumina vs 10x_chromium) as a dedicated z_meta dimension with a learnable embedding lookup. Embeds get concatenated into the decoder input. In Run 27 (`models/disentangled_vae.py` + `analysis/q45_run27_tech_aware.py`).

We took the **lookup-table encoding** part (also in CellPLM) but DID NOT take scGPT's full transformer + value-binning + masked-language-modeling framework — that needs ≥100k cells and we have 261 sn pseudobulks.

### 2. Read-depth recovery — **scFoundation-style**
Added a continuous `log_depth` z_meta dimension with an MSE head (scFoundation's "T" token). The flip operation re-targets depth when flipping modality, since 10x sc and bulk-illumina have wildly different read depths. In Run 27.

What we kept: depth as an explicit, supervised metadata axis. What we skipped: scFoundation's 100M-param asymmetric transformer — overkill at our scale.

### 3. One-hot batch covariate at encoder + decoder — **scVI-style**
Our `z_meta` is fed to BOTH the encoder (supervised heads enforce it) AND the decoder (concat with z_bio). This is the standard scVI pattern. The difference: we use Gaussian likelihood on standardized log-CPM (since we work on continuous bulk-style data), not NB/ZINB on counts.

### 4. Cycle consistency — **CycleGAN / MUNIT-style** (image translation)
Our cycle losses (Run 5 onwards) are direct adaptations of CycleGAN's cycle consistency. Not scFM but the right tool for cross-modality translation.

### 5. Linear-probe protocol for biological validation — **scVI / scANVI / scIB convention**
For metadata prediction (Q12 in past work, Q47 now) we use a **frozen-encoder linear probe** rather than fine-tuning. This isolates what the embedding inherently contains. scIB uses similar metrics (NMI, ARI, isolated label F1) to evaluate scVI/scANVI embeddings.

### 6. Mechanism-shift decomposition — **SAMS-VAE-inspired**
Our latent split z = (z_meta, z_bio) is directly inspired by SAMS-VAE's "mechanism shift" — instead of one entangled latent, we have a structured slice for known covariates and a free slice for biology. We dropped SAMS-VAE's sparse additive parameterization because our setting is bulk/sc translation, not perturbation prediction.

### 7. F-test cross-tissue gene selection (NOT from scFM)
Inspired by classical eQTL-discovery and donor-discriminative gene sets (e.g., HLA, mitochondrial heteroplasmy). Picks genes with high donor-vs-tissue F-statistic. Used in Q35 and the winning Q42 ensemble.

## What we explicitly did NOT use (and why)

### 1. Transformer architecture
- **scGPT**: 51M params, transformer over gene tokens. Needs ≥100k cells. We have 261 sn pseudobulks.
- **scFoundation**: 100M params. Same data scale issue.
- **CellPLM**: 80M params. Same.

At our data scale (16 donors), a 25M-param MLP VAE is already over-parameterized. Going bigger just memorizes faster.

### 2. Value-binning of counts (scGPT)
scGPT bins expression into 51 discrete tokens (0-50) then uses the bin index as a token type. This handles cross-batch normalization implicitly. We use continuous log2(CPM+1) standardized values, which is the convention for bulk RNA-seq (GTEx, BulkFormer). Binning loses information in low-cell-count pseudobulks.

### 3. Masked-language-modeling pretraining (scGPT, scFoundation)
MLM works when you have huge cell counts to mask random subsets and predict from context. With 261 sn pseudobulks, MLM doesn't learn useful structure — every pseudobulk is unique.

### 4. Cell-cell attention (CellPLM Flowformer)
CellPLM uses Flowformer to model relations BETWEEN cells in a batch. Our pseudobulks are aggregates; there's no within-sample cell graph to attend over.

### 5. Rank-based gene tokens (Geneformer)
Geneformer ranks genes within each cell, uses the rank list as tokens. Robust to depth but loses absolute-expression information. We DID try rank-based features (Q35 spearman) as a baseline; it was comparable to raw cosine.

### 6. Semi-supervised cell-type classifier head (scANVI)
scANVI adds a classifier on z for KNOWN cell types, semi-supervised over unlabeled cells. We don't have cell-type labels for our bulk samples (bulks are mixtures), so this doesn't apply.

### 7. NB/ZINB likelihood (scVI)
NB/ZINB models count data with overdispersion. Useful for raw 10x counts. We work on log2(CPM+1) standardized values — Gaussian likelihood is the right match.

### 8. Text-based tokens (Cell2Sentence)
Tokenizes expression into text (e.g., "this cell expresses GAPDH, ACTB, ..."). Bypasses gene-vector framing. Cool but adds a tokenization layer that obscures the metadata-flip operation we care about.

### 9. Hierarchical priors / Bayesian nonparametrics (DRVI, scPhere)
Add inductive biases for pseudotime / continuous trajectories. Our data is donor-paired snapshots, not trajectories.

## Direct package use (could plug in later)

We could DIRECTLY plug into these toolkits without re-implementing:

| Package | Pip | Use case | Compatibility with our model |
|---|---|---|---|
| `scvi-tools` | `scvi-tools` | scVI/scANVI for comparison | Yes — submit bulk + sn as 2-batch dataset, train scVI, compare to our z_bio |
| `harmony-pytorch` | `harmonypy` | Batch correction in PCA space | Yes — replace our HSIC penalty with harmony post-hoc |
| `scIB` | `scib` | 14 cross-modality integration metrics | Yes — evaluate our z_bio against scIB benchmark suite |
| `Open Problems` | via the bench framework | Modality matching / prediction | Yes — submit our model as a candidate |
| `BulkFormer` | not yet packaged | Bulk pretraining model (Kang 2025) | Future: pretrain on bulk, use our flip to augment with sn |

## Gap we identified — and a proposed contribution back to scFMs

While trying to *plug in* a pretrained "technology" token from these models into our small VAE (Q52 research, 2026-05-11), we found that **none of the public scFM checkpoints actually expose a learned, transferable assay/technology token**, even though several papers describe the architecture as having "condition" or "batch" tokens.

| Model | What the paper / arch implies | What the public checkpoint actually contains |
|---|---|---|
| scGPT | "Condition tokens for batch/perturbation/modality" | `BatchLabelEncoder` is added at fine-tune time and sized from the user's `adata.obs['batch_id']`. Pretrained vocab is gene symbols + `<pad>/<cls>/<mask>` only. |
| STATE (Arc) | "Assay-aware dataset embedding" | `dataset_embedder` keyed to ~70 dataset-IDs from pretraining (Tahoe, Replogle subsets) — not to assay categories. |
| scFoundation | "T/S read-depth tokens" | These are continuous scalars, not categorical assay rows. Confirms no assay token. |
| CellPLM | "Learnable batch lookup table" | Per-sample/FOV-ID rows tied to their pretrain corpus, not per-technology. Thousands of dataset-specific rows. |
| Geneformer | "Technology-agnostic by rank tokens" | Confirmed: no batch/assay embedding by design. |

**Why this matters.** Users like us with a clean cross-technology setting (bulk_illumina ↔ 10x_chromium, or any two assays) cannot transfer a "this is what `10x_chromium` means" prior from a foundation model. We have to learn that 2-row lookup from scratch on tiny data (in our case, 353 pseudobulks from 16 donors).

### Two concrete contributions an scFM could make

1. **Add a generic assay vocabulary to the pretrained checkpoint.** Define a fixed set of technology categories (`10x_chromium_v2`, `10x_chromium_v3`, `smart_seq2`, `dropseq`, `bulk_illumina`, `bulk_pacbio`, …) and learn a `d`-dim embedding for each during pretraining by including a per-cell assay label in the pretraining stream. Ship those rows in the checkpoint. Users fine-tuning on a new dataset would then have a real technology prior, not a randomly initialized lookup.

2. **Consume a small VAE's learned assay embedding at fine-tune time.** The bulk_illumina ↔ 10x_chromium offset our Run 27 VAE learns on 353 pseudobulks is exactly the type of 2-row prior an scFM would want to inject. An scFM fine-tune API could accept a user-supplied `init_batch_embedding: np.ndarray` (shape `n_batches × d`) so that our pseudobulk-aligned VAE's `tech_embed.weight` is used as initialization rather than `N(0, 0.02)`. Even a single epoch of fine-tuning would propagate that prior through the gene encoder.

**Self-test of (2): does our pseudobulk embedding actually help a foundation model?**

The natural empirical check is to (a) embed our 353 paired samples with a foundation model alone, (b) re-embed with our `tech_embed` rows injected as the batch-embedding initialization (or concatenated as side features), and compare on the donor-NN LOO and metadata-probe harness from Q47/Q50/Q51. If the augmented embedding wins, that's direct evidence that small task-specific VAEs and large pretrained scFMs are complementary, not redundant.

**Smaller-model path for the self-test.** Full scGPT/scFoundation are 50–100M params and require gene-name alignment + a multi-hour fine-tune. The most accessible "their embedding + ours" comparison uses **scVI as the foundation-model stand-in** (1–2M params, same VAE family, native batch covariate, pip-installable, n=200-epoch fine-tune in ~2 min on CPU) — implemented in Q51. For a real-scFM ablation, **Geneformer (HuggingFace `ctheodoris/Geneformer`)** is the most install-friendly true-foundation option — Q52 (TODO) should run that path next.

## The bottom line for our project

The scFM literature gave us **two architectural ingredients** that we adopted: scGPT-style condition tokens and scFoundation-style read-depth. Everything else (transformer scale, MLM, value binning) is wrong for our 16-donor data regime.

The rest of our architecture (SAMS-VAE-inspired mechanism shift, CycleGAN-inspired cycle consistency, F-test gene selection, ensemble fusion) comes from causal representation learning, image-to-image translation, and classical statistics — NOT from the foundation-model wave.

Our headline result — **Q42 ensemble top-1 = 0.708 on donor-faithful cross-modality NN** — was achieved by combining one small VAE component with two small linear methods (CCA-5, PCA + F-test). The right tool at 16 donors is "ensemble of small things", not "one big foundation model."
