# ECS 271 — Cross-Modality VAE · Complete Project Report

**Authors:** Robin Sayar & Steven Trujillo  
**Repo:** `sttruji/ecs271-project` (branch `main`)  
**Last updated:** 2026-05-26  
**Goal:** Dropout-robust cross-modality encodings for bulk and single-cell RNA-seq (Trujillo + Sayar proposal)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Phase 1 — VAE Diagnosis & Baselines (Q1–Q15)](#3-phase-1--vae-diagnosis--baselines-q1q15)
4. [Phase 2A — Disentangled Metadata-Flip VAE (Q16–Q51)](#4-phase-2a--disentangled-metadata-flip-vae-q16q51)
5. [Phase 2B — FiLM MetaInjection VAE (Q53–Q59)](#5-phase-2b--film-metainjection-vae-q53q59)
6. [Phase 2C — PULSAR Foundation Model Experiments (Q62)](#6-phase-2c--pulsar-foundation-model-experiments-q62)
7. [RQ1 — COMBAT Bulk↔sc Donor Retrieval](#7-rq1--combat-bulksc-donor-retrieval)
8. [RQ1 — Pretraining Experiments (v5, v6)](#8-rq1--pretraining-experiments-v5-v6)
9. [RQ1 — Diagnosis & Ceiling](#9-rq1--diagnosis--ceiling)
10. [Next Direction — SAMS-VAE Adaptation](#10-next-direction--sams-vae-adaptation)
11. [Research Questions Status](#11-research-questions-status)
12. [Key Concepts Glossary](#12-key-concepts-glossary)
13. [Reproducibility](#13-reproducibility)

---

## 1. Project Overview

**Core problem:** Bulk RNA-seq and single-cell RNA-seq measure the same biology through fundamentally different lenses. Bulk averages across millions of cells; sc captures each cell but introduces dropout noise and dissociation stress. Building a shared embedding where the same donor's bulk and sc samples are close — while donors are well-separated — requires solving both a modality-alignment and a donor-discrimination problem simultaneously.

**Three research questions:**

| RQ | Question | Status |
|---|---|---|
| **RQ1** | Can we retrieve the matching sc donor from bulk (and vice versa) better than raw cosine similarity? | ~65% — COMBAT: argmax=0.248, Hungarian=0.587 (vs. random=0.009) |
| **RQ2** | Can we separate biological variation from technical confounds (ischemia, modality, prep time)? | ~80% — ischemia R²=−0.002 in z_bio after FiLM VAE |
| **RQ3** | Does the joint encoding improve downstream prediction over single-modality baselines? | ~40% — N=4 cluster tokens +0.008 F1; VAE token alone adds nothing |

---

## 2. Repository Layout

```
/Users/rls/ecs271/
├── data/                                   # shared dataset root
│   ├── paths.py                            # canonical paths (importable)
│   ├── bulk/
│   │   ├── gtex_v11_whole_blood.gct.gz     # GTEx v11 803 whole-blood donors
│   │   ├── GSE223613/                      # time-of-day circadian cohort (10 donors × 8 timepoints)
│   │   ├── GSE279480/                      # Smithmyer 2025 living-donor cohort (255 Null samples)
│   │   └── pbmc_bulk/
│   │       └── GSE162632_pbmc_NI_combat_genes.npz  # 89 healthy PBMC bulk donors
│   ├── sc/
│   │   ├── hca_blood_pseudobulk.npz        # HCA blood pseudobulk
│   │   ├── blood_h5/                       # per-donor h5 single-cell files
│   │   └── combat/
│   │       ├── combat_paired.npz           # 109 COMBAT matched bulk+sc donors (4193 shared genes)
│   │       └── CBD-KEY-CLINVAR/            # COVID severity metadata (Hospitalstay, SARSCoV2PCR)
│   ├── models/cross_modality_vae.pt        # original trained checkpoint (collapsed)
│   └── annotations/
│       ├── GTEx_v10_Annotations_SubjectPhenotypesDS.txt
│       └── GTEx_v10_Annotations_SampleAttributesDS.txt
│
└── vae_health/                             # main repo (sttruji/ecs271-project)
    ├── pipeline/                           # reusable evaluation suite
    │   ├── protocols.py                    # LatentModel protocol
    │   ├── data.py                         # GTEx loader + metadata join
    │   ├── reconstruction.py               # Q1 metrics
    │   ├── latent.py                       # latent probes + extraction
    │   ├── enrichment.py                   # Enrichr REST wrapper
    │   ├── adapters.py                     # wrap VAE / PCA / AE
    │   └── runner.py                       # EvalConfig + run_evaluation
    ├── models/
    │   ├── disentangled_vae.py             # metadata-flip VAE (Q16–Q51)
    │   ├── meta_injection_vae.py           # FiLM VAE (Q53–Q59)
    │   ├── sparse_global_vae.py            # SAMS-VAE-inspired sparse additive VAE
    │   └── drvi_model.py                   # DRVI wrapper
    ├── analysis/
    │   ├── q{1..15}_*.py                   # Phase 1 explorations
    │   ├── q{20..51}_*.py                  # Phase 2A disentanglement
    │   ├── q{52..62}_*.py                  # Phase 2B FiLM + PULSAR + additive
    │   ├── rq1_combat_infonce.py           # COMBAT baseline: raw InfoNCE
    │   ├── rq1_combat_v2.py                # z-score + larger encoder → 0.211
    │   ├── rq1_combat_v4.py                # MoCo + DCL + severity negatives
    │   ├── rq1_combat_v4b_ablation.py      # γ sweep → Hungarian 0.587 best
    │   ├── rq1_combat_v5_pretrain.py       # SSL PBMC pretrain → hurts (0.506)
    │   ├── rq1_combat_v6_supervised_pretrain.py  # supervised pretrain → no gain
    │   ├── rq1_gamma_sweep.py              # γ sweep: 3.0, 5.0, 8.0, 15.0
    │   ├── rq1_supervised_infonce.py       # Eraslan supervised InfoNCE (in-sample=1.000)
    │   ├── rq1_lodo_infonce.py             # Eraslan LODO: 0.171 (15.5× random)
    │   └── docs/
    │       ├── rq1_literature_survey.md
    │       └── pulsar_vae_colab*.ipynb
    ├── analysis/results/
    │   ├── q47_disentangled/               # Run 3 JSON + log
    │   ├── q60_interpretable_basis/        # NMF/ICA results + figures
    │   ├── q61_pulsar_nmf/                 # PULSAR NMF augmentation
    │   ├── q62_pbmc_vae/                   # PBMC VAE metadata + embeddings
    │   └── rq1_combat_v{4b,5,6}.json       # COMBAT retrieval results
    ├── scripts/
    │   └── build_shared_gene_matrix.py     # builds data/processed_11k/
    ├── REPORT.md                           # this file
    ├── LEARNING.md                         # concept glossary
    └── .gitignore                          # *.npy, *.pt, outreach/ excluded
```

**Python environments:**
- `bulk-project/venv` (Python 3.12, torch 2.2.2, MPS) — all analysis except DRVI
- `.venv-drvi` — DRVI only

---

## 3. Phase 1 — VAE Diagnosis & Baselines (Q1–Q15)

### TL;DR

The original `cross_modality_vae.pt` had **posterior-collapsed** (0/64 active dims, R²=−3.27). A vanilla 3-layer MLP autoencoder reaches R²=0.81. PCA-128 beats every VAE on reconstruction and metadata probes.

### Q1 — Reconstruction quality on held-out GTEx blood

| Method | R² | Per-sample r̄ |
|---|---:|---:|
| Mean (zeros) | 0.000 | 0.000 |
| PCA-50 | 0.859 | 0.901 |
| AE-3 (this work) | 0.813 | 0.874 |
| Sane VAE (β=10⁻³) | 0.790 | 0.854 |
| **Cross-modality VAE (trained)** | **−3.269** | **−0.010** |

The trained VAE decoder output is independent of the donor.

### Q2 — What is the trained VAE encoding?

The encoder learned the right *directions* (max |corr(z_k, bulk PC1)| = 0.949) but compressed them to ~10⁻⁵ variance. The KL prior pulled q(z|x) back to N(0, I). State-B classification AUC from z: 0.564 (vs. 0.936 from bulk PCs).

**Root cause:** β=1.0 + λ_adversarial=0.1. The adversarial term also failed (discriminator accuracy stalled at 78%). Fix: β≤10⁻³ + free bits, no adversarial term.

### Q3 — 3-layer MLP autoencoder

Architecture: Linear(11374→1024)→BN→LeakyReLU→Linear(1024→512)→...→Linear(64). R²=0.813 with all 64 dims active. The trained VAE's failure is entirely loss design, not architecture.

### Q4 — PCs vs biology & metadata

**Pathway enrichment (top-5 PCs):**

| PC | Var % | + direction | − direction |
|---|---:|---|---|
| PC1 | 31.1 | Ribosome biogenesis | Regulation of phagocytosis (myeloid) |
| **PC2** | **20.2** | **ER→Golgi vesicle transport** | **Response to unfolded protein (adj-p=2.3×10⁻⁸)** |
| PC3 | 9.5 | CREM ChIP-Seq | Cellular respiration (adj-p=9.4×10⁻⁷) |
| PC4 | 5.7 | Translation (adj-p=1.7×10⁻²⁵) | TCF7 ChIP-Seq |
| PC5 | 3.0 | Glycolytic process (adj-p=5.1×10⁻⁷) | Heme metabolism |

**GTEx metadata correlations:**

| Metadata | Best PC | |ρ| |
|---|---|---:|
| SMTSISCH (post-mortem ischemia, min) | PC2 | **0.652** |
| DTHHRDY (Hardy death class) | PC2 | **0.664** |
| SMRIN (RNA quality) | PC2 | 0.460 |
| AGE_mid | PC2 | 0.357 |
| SEX | PC50 | (η²=0.215) |

PC2 is the *agonal-stress / ex-vivo handling* axis — validated independently by pathway enrichment and metadata correlation. This is **the dominant confound** in GTEx blood.

### Q7+Q8 — Noise floor: only ~30 PCs are real signal

| Method | # PCs above noise |
|---|---:|
| Marchenko–Pastur analytical | 32 |
| Horn parallel analysis | 33 |
| Bootstrap stability (median |cos| > 0.5) | 32 |

Of K₉₅ = 251 PCs, only ~30 carry stable biology. PCs 1–6 are rock-solid (bootstrap |cos| > 0.97). Trust PCs 1–30, treat 31+ as decoration.

### Q9 — Cross-cohort replication (GSE279480, 255 living donors)

| GTEx PC | What it encodes | |cos| with best GSE PC |
|---|---|---:|
| **PC1** | MYC / myeloid-lymphoid composition | **0.75 — replicates** |
| PC2 | PU.1 myeloid vs ex-vivo ischemia | 0.22 — **fails** |

PC2 fails because GSE279480 donors are *living* — the ischemia axis literally cannot exist in a living-donor cohort. This is the expected and correct asymmetry: PC1 = biology (universal), PC2 = cohort-specific artefact.

### Q12 — Linear-probe metadata recovery from unsupervised latents

| Target | PCA-50 | Rand-64 | CM-VAE | Sane-VAE |
|---|---:|---:|---:|---:|
| SMTSISCH R² | **0.696** | 0.636 | 0.590 | 0.660 |
| DTHHRDY R² | **0.528** | 0.500 | 0.505 | 0.515 |
| SEX AUC | **0.996** | 0.784 | 0.656 | 0.677 |
| AGE_mid R² | **0.219** | 0.142 | 0.119 | 0.169 |

PCA-50 dominates every probe. Sane-VAE ≈ random projection on most targets — reconstruction loss does not concentrate metadata-relevant axes beyond what random subspaces already have.

### Q13 — Time-of-day probe (GSE223613, 10 donors × 8 timepoints)

**LODO (honest cross-individual):**

| Latent | LODO Hour MAE | LODO Age R² |
|---|---:|---:|
| PCA-50 | 4.41 h | 0.344 |
| Random-64 | 5.27 h | 0.078 |
| CM-VAE | **4.39 h** | **0.424** |

Random circular guess ≈ 6.0 h. Time-of-day does **not** generalize cross-donor at n=10. Age weakly recoverable (R²≈0.34).

### Q14 — Pathway-aware structured-decoder VAE

Latent dims anchor to MSigDB Hallmark pathways via a learnable (51×50) matrix with L1 penalty. Interpretability works — dims map to OXPHOS, TNF-α, HEME, IFN-α axes — but Hallmark covers only 28% of shared genes. Held-out R²=0.221 vs Sane-VAE 0.794.

### Q15 — Latent-dimension sweep

| N | PCA R² | VAE R² | VAE active dims | VAE SEX AUC |
|---|---:|---:|---:|---:|
| 16 | 0.793 | **0.789** | **15/16 (94%)** | 0.680 |
| 64 | 0.867 | 0.794 | 16/64 (25%) | 0.677 |
| 256 | 0.890 | 0.790 | 131/256 (51%) | 0.632 |

VAE reconstruction plateaus at N=16. The MLP hidden layer (512), not the latent dim, is the bottleneck. PCA keeps gaining with N; VAE collapses extra capacity.

**Recommendation: use N=16 for any Sane-VAE.**

---

## 4. Phase 2A — Disentangled Metadata-Flip VAE (Q16–Q51)

### Architecture (`models/disentangled_vae.py`)

- **Encoder:** x → [512,256] → z_bio (never sees metadata)
- **z_meta:** 8 dims (2 per field × 4 fields: modality, ischemia, sex, DTHHRDY)
- **Prediction heads:** Linear(d,32) → GELU → Linear(32,1) per z_meta field
- **HSIC penalty:** pushes z_bio away from modality label (λ=0.3)
- **Cycle losses:** z_bio round-trip + modality re-encoding

### What works — self-consistency (Run 10, Q47)

| Metric | Value |
|---|---|
| Modality flip GTEx → sc (NN accuracy) | **1.00** |
| Modality flip HCA → bulk (NN accuracy) | **0.98** |
| Round-trip Pearson | **0.78** |
| z_bio cycle cosine through flip | **0.95** |
| Active z_bio dims | 50/50 |
| Ischemia R² in z_bio | 0.77 |

### The central negative result — donor-faithful translation

**Test 2:** does flipping a held-out sc sample's modality produce output whose NN among real bulk samples is the *same donor*?

| Experiment | Setup | Top-1 same-donor |
|---|---|---:|
| Q21 (Run 10 zero-shot) | 7 melanoma PBMC pairs | 0.286 |
| Q22 LOO fine-tune | 6 paired pairs/fold | 0.143 |
| Q23 bootstrap LOO | 60 pairs/fold | 0.000 |
| Q24 (Eraslan multi-tissue) | 245 pairs/fold, 16 donors × 8 tissues | 0.292 |
| **Raw cosine (no model)** | — | **0.500** |
| **PCA-50 (no model)** | — | **0.510** |

Across **22 architectural variants** (Q25–Q34), no learned approach beat raw cosine or PCA-50. **Diagnosis: the cycle losses teach cluster-level self-consistency but cannot render per-donor identity in an unseen modality at this data scale (16 paired donors).**

### Ensemble breakthrough (Q36–Q42)

| Method | Top-1 |
|---|---:|
| Raw cosine | 0.500 |
| PCA-50 | 0.510 |
| CCA-5 | 0.604 |
| Q39 ensemble (top-3) | 0.615 |
| **Q42 best ensemble** | **0.708** |

Winning combination: `VAE latent NN + CCA-5 + raw cosine` with equal weighting. The VAE latent is below chance alone but adds complementary signal in ensemble.

### Auxiliary probes (Q44–Q51)

- **Q47:** z_full (16-dim VAE latent) beats raw 11,374-gene expression on Age and Autolysis recovery
- **Q48–Q49:** Masking augmentation +0.03 F1 on age prediction; count-weighted MSE essential for highly expressed genes
- **Q50:** Whitening z_bio before probing lifts ischemia R² from 0.04 → 0.31 (nonlinear encoding)
- **Q51:** z_bio sits between scVI (better cluster separation) and Harmony (better metadata removal)

### Q47 — Disentangled VAE Run 3 (final, from REPORT.md)

Best configuration: 200 epochs, β_bio=10⁻³, λ_cycle=0.5, free_bits=0.5, gradient clip max_norm=5.0, batch=128, AdamW lr=3×10⁻⁴.

| Metadata field | Head metric | Value |
|---|---|---|
| Modality (bulk vs. sc) | balanced accuracy | **0.99** |
| Ischemia time (SMTSISCH) | R² (val GTEx) | **0.77** |
| Sex | balanced accuracy | **0.99** |
| DTHHRDY (Hardy scale) | R² (val GTEx) | **0.52** |

Flip tests: Test A (bulk→sc) = 0.9999, Test B (HCA→bulk) = 0.930, Round-trip Pearson = 0.721.

---

## 5. Phase 2B — FiLM MetaInjection VAE (Q53–Q59)

### Architecture (`models/meta_injection_vae.py`)

- **Encoder:** `x_residualized → [512,256] → z_bio` — input is ischemia-residualized gene expression
- **Meta embedder:** `meta(4) → [128,128] → meta_emb`
- **Decoder:** z_bio + FiLM-modulated meta_emb at every hidden layer
- **Best checkpoint:** `q54b_count_weighted/` — K=16, (512,512) decoder, 600 epochs, count-weighted MSE

### Results

| Metric | Value |
|---|---|
| Genes passing FC threshold | **99.99%** |
| High-expressed genes (>1000 CPM) | **98.98%** |
| HBB (hemoglobin beta) | Fails — saturating nonlinear ischemia response; PCA-50 also fails → **proven ceiling** |
| Median per-sample R² (round-trip) | **0.9975** |
| Within-sample ischemia flip FC match | **93.0%**, Pearson r=0.982 |
| SMTSISCH R² in z_bio | **−0.002** ← essentially zero |

**Ischemia is fully in the meta pathway. z_bio is clean.**

### Biological axes (Q59 sub-cell-type panel)

| z dim | Cell type | Per-dim R² |
|---|---|---|
| z15 | CD4 memory T-cells | **0.58** |
| z4 | Classical monocytes | 0.35 |
| z16/z1 | Erythroid / mitochondrial | 0.28–0.36 |
| z2, z5, z7 | Monocyte sub-populations | 0.18–0.20 |

### Deconvolution comparison (Q57–Q58)

| Feature space | Dims | Mean R² vs cell-type proxies |
|---|---|---|
| All genes (Ridge) | 11,374 | 0.97 |
| PCA-50 | 50 | **0.93** |
| PCA-16 | 16 | 0.78 |
| **z_bio (K=16)** | 16 | **0.61** |

z_bio loses to PCA-16 by 0.17 R² — because reconstruction + disentanglement objectives sacrifice variance preservation. PCA keeps the maximum-variance (cell-composition) directions; z_bio smooths them out.

**Bottom line for Thread B:** the FiLM MetaInjection VAE is the best single model we have. Correct-by-construction for ischemia removal; reconstructs to 99.99% FC; produces interpretable cell-type latent axes. Not better than PCA for deconvolution, but is the right tool for downstream tasks requiring ischemia-clean biology.

---

## 6. Phase 2C — PULSAR Foundation Model Experiments (Q62)

**Model:** PULSAR-pbmc (KuanP/PULSAR-pbmc, 87.4M params) — Stanford MCT transformer ingesting 256 cells × 1280-d UCE embeddings → 512-d donor CLS embedding.

**Task:** Lupus vs. healthy on 261 donors (162 lupus, 99 normal), 5-fold CV.

### VAE token experiments

| Strategy | F1-macro |
|---|---|
| A. PULSAR CLS frozen (baseline) | **0.948 ± 0.010** |
| D. PULSAR frozen + VAE token | 0.948 ± 0.010 |
| E. Partial unfreeze (top-4 layers) + VAE token | 0.947 ± 0.012 |
| F. Full fine-tune + VAE token | 0.947 ± 0.010 |
| G. Ablation: frozen, z_bio = 0 | 0.935 ± 0.020 |

**The VAE token adds nothing.** PULSAR already saturates the lupus signal; 261 donors is insufficient data to adapt 87.4M parameters.

### N-cluster UCE tokens

K-means on all 66,816 cells (261 × 256). Per-donor cluster means (1280-d). Shared bottleneck projector 1280 → 32 → 768 (63K params). Frozen PULSAR backbone.

| Method | F1-macro |
|---|---|
| A. PULSAR CLS frozen | 0.948 ± 0.010 |
| **N4. PULSAR + N=4 cluster tokens** | **0.956 ± 0.023** |
| N8. PULSAR + N=8 cluster tokens | 0.947 ± 0.027 |

+0.008 F1 with N=4. Mechanism: PULSAR's cross-attention selectively attends to the diagnostically relevant cluster token (e.g. monocyte cluster for IFN-response donors). A flat LogReg/MLP can't do this — concatenating 5120 features just adds noise.

**Key insight: mean UCE → LogReg (0.951) already matches PULSAR frozen (0.948). UCE embeddings are saturated for this binary task without any transformer.**

---

## 7. RQ1 — COMBAT Bulk↔sc Donor Retrieval

### Dataset

**COMBAT (Cell 2022, Zenodo 10.5281/zenodo.6120249, CC-BY):** 109 matched bulk blood RNA-seq + sc PBMC donors from the Cambridge COVID-19 Biobank. 4193 shared genes. Bulk = whole-blood log-CPM, sc = PBMC pseudobulk log-RPM.

**Biological context:** COMBAT bulk is whole-blood — dominated by HBB/HBA1 (hemoglobin, from red blood cells, which are absent in PBMC). The sc pseudobulk is PBMC-only. This is an intrinsic biological mismatch, not just technical noise.

**Severity:** COVID hospitalization duration (Hospitalstay) used as a continuous severity proxy, normalized 0–1.

### Evaluation metrics

- **Argmax top-1:** each sc query picks its nearest bulk donor by cosine similarity. Fraction where it picks the correct donor out of all 109.
- **Hungarian top-1:** `scipy.optimize.linear_sum_assignment(-sim[:, test_idx])` — globally optimal one-to-one matching among test donors. Eliminates severity crowding (many sc converging on same severity cluster bulk).
- **Random baseline:** 1/109 = 0.009.

### Model architecture (shared across all versions)

```python
class Enc(nn.Module):
    def __init__(self):
        self.body = nn.Sequential(
            nn.Linear(G, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.proj = nn.Linear(256, 128)  # Z_DIM = 128
```

One shared encoder for both bulk and sc (weight sharing forces same embedding space).

### Preprocessing

```python
top_idx = np.argsort(bulk_raw.var(0))[-2000:]   # top 2000 HVGs by bulk variance
bh = zscore(bulk_raw[:, top_idx])                # z-score per sample
sh = zscore(sc_raw[:, top_idx])
```

Z-score per sample removes per-donor expression-level differences.

### Loss: DCL + Severity-weighted negatives

```python
def dcl_severity(z1, z2, t, sev_w):
    # sev_w[i,j] = 1 + γ · (1 - |sev_i - sev_j|)
    # Donors with similar severity are harder negatives → upweight them
    s12 = (z1n @ z2n.T) / t
    s12 += log(sev_w)    # add weight to off-diagonal (negatives)
    pos = s12.diagonal()
    # DCL: positive excluded from denominator (unlike standard InfoNCE)
    l1 = (-pos + logsumexp([s12_off, s11_off], dim=1)).mean()
    l2 = (-pos + logsumexp([s12T_off, s22_off], dim=1)).mean()
    return (l1 + l2) / 2
```

**Why DCL?** Standard InfoNCE includes the positive pair in the denominator ("negative-positive coupling"). DCL removes it — far more stable at small batch sizes (87 training donors). From Yeh et al., ECCV 2022.

**Why severity weighting?** COVID severity dominates blood expression variation. Donors with similar severity look similar in gene space → hard negatives. `w_ij = 1 + γ·(1 − |sev_i − sev_j|)` upweights these pairs, forcing the model to separate same-severity donors.

### Results progression

| Version | Key change | Argmax | Hungarian | Notes |
|---|---|---:|---:|---|
| Baseline InfoNCE (v1) | τ=0.2, raw features | 0.120 | — | 13× above random |
| v2 | z-score + big encoder + τ annealing 0.3→0.05 | 0.211 | — | |
| v3 (DANN+InfoNCE) | added gradient reversal alignment | 0.138 | — | **DANN hurts** — fights InfoNCE |
| v4 (MoCo+DCL+sev) | MoCo queue + DCL + γ=1.0 | 0.201 | 0.458 | First Hungarian result |
| **v4b γ=5 (best)** | **γ sweep found γ=5 optimal** | **0.220** | **0.587** | **65× random** |
| v5 (PBMC pretrain) | SSL pretrain on 89 PBMC donors | 0.194 | 0.506 | **Hurts** — wrong invariances |
| v6 (supervised pretrain) | Full-batch donor-ID pretrain on PBMC | 0.220 | 0.587 | No gain |

### γ sweep (v4b ablation)

| γ | Argmax | Hungarian |
|---|---:|---:|
| 0.0 (no weighting) | 0.203 | 0.458 |
| 0.5 | 0.207 | 0.497 |
| 1.0 | 0.210 | 0.530 |
| 2.0 | 0.214 | 0.578 |
| **5.0** | **0.220** | **0.587** |
| 8.0 | 0.209 | 0.551 |
| 15.0 | 0.190 | 0.490 |

γ=5 is optimal. γ≥8 collapses — the negatives become too hard and training destabilizes.

### Why Hungarian gives +0.255 boost over argmax

At γ=5, argmax=0.220 and Hungarian=0.587 — a 0.367 gap. This is because severity crowding makes many sc donors' nearest bulk neighbor the same "average severity" bulk embedding. Argmax allows all sc to compete for the same bulk donors; Hungarian enforces one-to-one assignment, which eliminates crowding at zero cost at inference.

---

## 8. RQ1 — Pretraining Experiments (v5, v6)

### Motivation

The core hypothesis: COMBAT bulk is whole-blood (HBB-dominated), sc is PBMC. If we pretrain on 89 GSE162632 healthy PBMC bulk donors, the encoder learns PBMC-relevant co-expression rather than HBB/HBA1 patterns irrelevant for sc retrieval.

**Gene alignment:** GSE162632 has 4165 genes vs COMBAT's 4193. 28 missing → zero-padded. 99.3% overlap.

### v5 — SSL pretraining (fails)

**Stage 1:** Within-batch InfoNCE on PBMC donors. Two augmented views of the same donor (30% masking) = positive pair. Batch=32.

**Why it fails:** At batch=32 from N=89 donors, many negatives are easy. But more importantly, the augmented-view SSL objective teaches "be invariant to masking noise." Donor variation *looks like* masking noise to this loss — so the encoder learns to **ignore** donor identity. This is exactly the wrong invariance for our retrieval task.

**Result:** Hungarian=0.506 < 0.587 baseline. Pretraining hurts.

### v6 — Supervised full-batch pretraining (no gain)

**Fix:** Use full-batch InfoNCE where every donor contrasts against all 88 other donors simultaneously — "strong donor discrimination signal."

```python
def nt_xent_full(z1, z2, t):
    """Full-batch InfoNCE: z1[i] paired with z2[i], all others negative."""
    N = z1.size(0)
    z = torch.cat([normalize(z1), normalize(z2)])
    s = (z @ z.T) / t; s.fill_diagonal_(-1e9)
    lab = torch.cat([arange(N, 2*N), arange(N)])
    return cross_entropy(s, lab)
```

**PBMC self-top1 = 0.000 throughout 300 epochs.** The 89 PBMC donors are too similar to each other — disease severity doesn't vary (all healthy), so no discriminative features exist. The model can't learn "donors should be distinct" because they aren't distinct enough in this healthy cohort.

**Result:** Hungarian=0.587 = v4b. No gain from pretraining.

### Conclusion

Pretraining cannot help because the core bottleneck is *biological*, not *data scale*:
1. Whole-blood bulk (HBB-dominated) vs PBMC sc — fundamentally different tissues, not just different measurement noise
2. COVID disease severity dominates the expression landscape; even with 87 donors of DCL training, many donors are indistinguishable at the cellular level
3. Any more training data would need to be **matched PBMC bulk + sc**, not unmatched PBMC bulk alone

---

## 9. RQ1 — Diagnosis & Ceiling

**Current ceiling:** argmax=0.248, Hungarian=0.587 on 109 COMBAT donors.

**Root causes (in order of impact):**

1. **Biological mismatch (largest):** COMBAT bulk captures whole blood including erythrocytes (HBB/HBA1 make up >50% of expression). sc pseudobulk is PBMC-only. The "right" match would require PBMC bulk + PBMC sc per donor. We have neither.

2. **Severity crowding:** COVID severity is the dominant axis of variation in both bulk and sc. Donors with similar severity scores cluster together in embedding space. Hungarian assignment partially fixes this by enforcing one-to-one matching, but it doesn't remove the underlying clustering.

3. **Scale:** 87 training pairs is extremely limited for learning a cross-modal alignment with 109-way retrieval. Adversarial alignment (DANN) was tried but hurts — the gradient reversal removes exactly the modality signal that InfoNCE needs for orientation.

**What would push further:**
- Truly matched PBMC bulk + PBMC sc dataset (e.g. 10x multiome on blood)
- SAMS-VAE-style additive decomposition: `z_donor = z_bio + Δz_modality` — learn to explicitly model and remove the whole-blood vs. PBMC shift
- BulkRNABert (InstaDeepAI/BulkRNABert) frozen encoder as backbone — pretrained on ~500k bulk RNA-seq profiles, much richer initialization than our from-scratch 512→512→256 MLP

---

## 10. Next Direction — SAMS-VAE Adaptation

### How SAMS-VAE works

**Paper:** Bereket & Karaletsos, NeurIPS 2023. "Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift VAE."

**Core generative model:**
```
z_cell = z_basal + θ_k ⊙ w_k
```
- `z_basal` — intrinsic cell state; amortized per cell via encoder
- `θ_k ∈ {0,1}^D` — binary gate: which latent dims perturbation k touches (sparse!)
- `w_k ∈ R^D` — magnitude of shift in those dims
- Sparsity prior: `p(θ_d=1) = π` (e.g. 0.1)
- Implemented as relaxed Bernoulli (concrete distribution) during training

**E-ELBO:** Two tiers of variables:
- Amortized (encoder): `q(z_basal | x)` — one inference pass per cell
- Non-amortized (learned directly): `q(θ_k, w_k)` — one set of parameters per perturbation, shared across all cells under that perturbation

**Combinatorial generalization:** Effects are additive, so combo A+B ≈ `z_basal + Δz_A + Δz_B`. Zero-shot prediction of untested perturbation combinations.

### Adaptation to our problem

| SAMS-VAE (perturbation bio) | Our project |
|---|---|
| Perturbation = drug / CRISPR KO | "Perturbation" = measurement modality (bulk vs sc) |
| Hundreds of discrete perturbation labels | 2 modalities + continuous confounds (severity, ischemia) |
| Same cell type, different treatment | Same donor, different measurement: whole-blood bulk vs PBMC sc |
| Perturbation effect is small vs cell-type variation | **Modality effect is HUGE** (HBB dominates bulk, absent in sc) |
| Unperturbed control exists | No "unperturbed" anchor — both modalities are valid |

**Direct adaptation:** `z_donor = z_bio + Δz_modality`

Strip the modality effect to get a modality-invariant `z_bio` that matches across bulk and sc.

### `models/sparse_global_vae.py`

Initial implementation of the SAMS-VAE-inspired model in this repo:
- Relaxed Bernoulli gate on latent dims
- Modality-specific shift vectors `w_bulk`, `w_sc`
- Sparsity prior on the gate
- Per-sample `z_bio` encoded by the shared encoder

### Questions for the SAMS-VAE author

**Architecture:**
1. How would you extend to continuous metadata (ischemia time, COVID severity) — parameterize `w_k` as `MLP(metadata)`?
2. Does the additive assumption break when the modality effect is large (whole-blood vs. PBMC is nearly a tissue-type change)?
3. How to handle non-additive interactions? (Modality shift interacts with severity — COVID donors' bulk-to-PBMC delta is larger than healthy donors')

**Training:**
4. What prevents posterior collapse in SAMS-VAE? Is it the structured prior, non-amortized perturbation params, or training schedule?
5. Identifiability without a "control group"? In our setup every donor has both bulk and sc — no unperturbed baseline.

**Evaluation:**
6. What metric do you trust most for confirming `z_basal` is perturbation-free? HSIC? Linear probing R²? Or the metadata-flip test we've been using?
7. Is there tension between reconstruction quality and retrieval quality in the latent space?

---

## 11. Research Questions Status

### RQ1 — Bulk↔sc donor reconciliation

**Status: ~65% achieved.**

✅ COMBAT 5-fold CV pipeline (honest, no leakage)  
✅ DCL + severity weighting + Hungarian assignment  
✅ Best result: argmax=0.248, Hungarian=0.587 (65× random)  
✅ Ceiling diagnosis: whole-blood bulk vs PBMC sc biological mismatch  
✅ Pretraining experiments ruled out (v5, v6)  
❌ No matched PBMC bulk+sc dataset for fair comparison  
❌ SAMS-VAE adaptation not yet trained end-to-end  
❌ BulkRNABert backbone not tested  

### RQ2 — Drivers of variation separation

**Status: ~80% achieved.**

✅ Ischemia fully disentangled from z_bio (R² = −0.002, FiLM VAE Q54b)  
✅ Round-trip reconstruction R² = 0.9975  
✅ Cell-type-named latent axes (Q59)  
✅ Sex/modality prediction accuracy: 0.99  
✅ HBB proven ceiling (PCA-50 also fails → not a model limitation)  
❌ Age signal weak (R² ≈ 0.20) — lifestyle metadata (BMI, sleep, ancestry) not integrated  
❌ Formal Pareto curve (recon vs biology vs masking robustness) not plotted  

### RQ3 — Joint encoding improves downstream prediction

**Status: ~40% achieved.**

✅ N=4 cluster tokens: +0.008 F1 over PULSAR baseline  
✅ Q42 ensemble (VAE+CCA+cosine): top-1 = 0.708 vs cosine = 0.500  
❌ VAE token alone adds 0 over saturated PULSAR  
❌ Deconvolution: z_bio R²=0.61 vs PCA-50 R²=0.93  
❌ SAMS-VAE direction untested for retrieval improvement  

---

## 12. Key Concepts Glossary

See `LEARNING.md` for full entries. Summary:

| Concept | One-liner |
|---|---|
| **InfoNCE / NT-Xent** | Contrastive loss: pull matched pairs together, push all others apart. Temperature τ controls hardness. |
| **DCL** | Decoupled Contrastive Learning — removes positive from denominator; stable at small batch (Yeh et al. ECCV 2022). |
| **Severity-weighted negatives** | `w_ij = 1 + γ·(1−|sev_i−sev_j|)` — upweight donors with similar disease severity as hard negatives. |
| **Hungarian assignment** | `scipy.optimize.linear_sum_assignment(-sim)` — globally optimal one-to-one matching; eliminates crowding at inference. |
| **LODO** | Leave-One-Donor-Out CV — held-out one donor entirely; stricter than k-fold when donors have multiple samples. |
| **Pseudobulk** | Aggregate sc counts across all cells from one donor → single expression profile comparable to bulk. |
| **DANN** | Domain-Adversarial Training — gradient reversal makes encoder modality-invariant. HURTS for retrieval because it removes the modality signal InfoNCE needs. |
| **VAE posterior collapse** | KL overwhelms reconstruction → encoder outputs prior N(0,I) for every input; fixed with β≤10⁻³ + free bits. |
| **FiLM** | Feature-wise Linear Modulation — conditions decoder on metadata via per-layer γ,β scale/shift vectors. |
| **HSIC** | Hilbert-Schmidt Independence Criterion — penalizes dependence between z_bio and metadata label. |
| **SAMS-VAE** | Sparse Additive Mechanism Shift VAE — z = z_basal + θ_k ⊙ w_k; sparse, interpretable perturbation effects. |
| **MoCo** | Momentum Contrast — maintains queue of past-batch embeddings as negatives; effective with small N. |
| **BulkRNABert** | BERT pretrained on ~500k bulk RNA-seq profiles (InstaDeepAI/BulkRNABert); 768-d CLS token per sample. |
| **HVG** | Highly Variable Genes — top genes by variance across samples; used to reduce dimensionality. |
| **PULSAR** | Stanford MCT transformer; ingests 256 cells × 1280-d UCE embeddings → 512-d donor CLS. |

---

## 13. Reproducibility

### Running the Phase 1 evaluation pipeline

```bash
cd /Users/rls/ecs271/vae_health
PYTHON=/Users/rls/Desktop/programming-projects/single-cell/bulk-project/venv/bin/python

$PYTHON analysis/q1_reconstruction.py
$PYTHON analysis/q2_latent_meaning.py
$PYTHON analysis/q3_mlp_autoencoder.py
$PYTHON analysis/q4_pcs_vs_biology.py

# Or via the pipeline runner:
$PYTHON -m pipeline --model vae --out-dir eval_output/vae --skip-enrichment
$PYTHON -m pipeline --model pca --out-dir eval_output/pca_64
$PYTHON -m pipeline --model ae  --out-dir eval_output/ae3 --epochs 200
```

### Running the COMBAT RQ1 retrieval experiments

```bash
# Best result (v4b, DCL + severity γ=5 + Hungarian)
$PYTHON analysis/rq1_combat_v4b_ablation.py
# → saves analysis/results/rq1_combat_v4b.json

# v5 PBMC pretrain experiment
$PYTHON analysis/rq1_combat_v5_pretrain.py

# v6 supervised pretrain experiment
$PYTHON analysis/rq1_combat_v6_supervised_pretrain.py
```

### Running the Q47 Disentangled VAE (Run 3)

```bash
$PYTHON analysis/q47_disentangled_train.py \
    --epochs 200 \
    --tag run3 \
    --lam-cycle 0.5 \
    --lam-cap-dthhrdy 0.3
# Requires: data/processed_11k/ (built by scripts/build_shared_gene_matrix.py)
```

### Key file sizes / what is NOT in git

- `analysis/results/q62_pbmc_vae/uce_batches_256.npy` — 326 MB (UCE batch embeddings); regenerate from Q62 script
- `*.pt` model checkpoints — regenerate from training scripts
- `outreach/` — excluded
- `*.npy` binary arrays — excluded

All scripts are self-contained and regenerate their results from the raw data in `data/`.

---

*End of report. Questions: rsayar728@gmail.com*
