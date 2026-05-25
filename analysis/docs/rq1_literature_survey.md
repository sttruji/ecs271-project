# Literature Survey: Cross-Modal Bulk↔SC Donor Retrieval

> Deep research report — May 2026. Problem: 109 matched COMBAT donors, top-1 NN retrieval of bulk given sc pseudobulk. Current best: 0.211 (5-fold CV).

## Key Finding: This Problem is Novel

No published paper does individual-level donor-identity retrieval across bulk↔sc modalities. 0.211 is a baseline for an open problem. Closest analogues: CNV↔RNA contrastive retrieval (cell-level, 87.3% top-1) and MARIO (protein modalities).

---

## Ranked Recommendations

### Tier 1 — High impact, low cost

| Method | What | Expected gain | Cost |
|--------|------|---------------|------|
| **MoCo-style queue** | Use ALL 87 training donors as negatives every step (not just in-batch) | +5–15% | ~20 lines |
| **DCL** | Remove positive from InfoNCE denominator — stabilises small-batch training | +2–5% | 1 line change |
| **Severity hard negatives** | Weight InfoNCE negatives by COVID severity similarity: `w_j = 1 + γ·cos_sim(sev_i, sev_j)` | +5–15% (novel) | ~10 lines |
| **Hungarian at inference** | Replace argmax with `linear_sum_assignment(-sim_matrix)` at test time — no retraining | small but free | 1 scipy call |

### Tier 2 — Medium impact

| Method | What | Notes |
|--------|------|-------|
| **BulkRNABert** | Frozen pre-trained bulk encoder (HuggingFace: InstaDeepAI/BulkRNABert) | TCGA domain vs COVID blood — uncertain transfer |
| **Temperature annealing** | τ: 0.3 → 0.05 over training | Prevents early collapse at small N |
| **VICReg auxiliary** | Variance + covariance penalty on embedding dims | Prevents collapse, not top-1 gain |

### Tier 3 — Investigate if time allows

- scFoundation/scGPT as frozen sc encoder (fine-tune with LoRA)
- Pathway-correlated gene masking (biology-informed augmentation)
- GROOVE/GroupCLIP (reduces to standard InfoNCE at N=109 without multi-view augmentation)

---

## Confirmed Red Flags

- **DANN** — already confirmed experimentally (0.138 < 0.211). Adversarial modality removal fights paired contrastive.
- **Zero-shot foundation models** — encode cell-type/tissue, not donor identity. Confirmed by Genome Biology 2025 benchmark.
- **Pre-processing severity regression** — causes confound leakage (arXiv 2022). Condition during training instead.
- **Random masking augmentation** — "Less is more" (Bioinformatics 2025): augmentation-free outperforms augmented for dense expression data.
- **OT methods (SCOT, GW-OT)** — designed for unpaired alignment. Discards your paired supervision signal.
- **Deep CCA** — D=4193, N=87 → rank-deficient. Only linear CCA after PCA-50 is safe.
- **Mixup** — creates virtual donors with no paired counterpart.

---

## Key Papers

| Paper | Key result |
|-------|-----------|
| DCL (ECCV 2022, arxiv:2110.06848) | Small-batch-robust InfoNCE |
| BulkRNABert (bioRxiv 2024) | First bulk RNA BERT — HF: InstaDeepAI/BulkRNABert |
| MARIO (Nat Commun 2023, PMC9911356) | CCA + Hungarian for cross-modal cell matching |
| CNV↔RNA contrastive (bioRxiv 2026) | Hard negatives by confound → 87.3% top-1 cross-modal |
| Less is more (Bioinformatics 2025, PMC12417077) | No augmentation beats augmentation for sc dense data |
| Zero-shot sc FM eval (Genome Biology 2025) | Foundation models fail zero-shot donor-level retrieval |
| Confound leakage (arXiv 2022) | Linear deconfounding before ML introduces leakage |
| GROOVE (bioRxiv 2026, arxiv:2602.04021) | Group contrastive for weakly-paired multimodal |

---

## The Core Diagnosis

Disease severity dominates inter-donor blood expression variation in COMBAT. InfoNCE learns to separate donors but mixes up same-severity donors. The **severity-informed hard negative weighting** is the most targeted fix and is genuinely novel — no paper has done this for RNA-seq.
