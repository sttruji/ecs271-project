# Q20 — Final synthesis across 26 experiments

**Date:** 2026-05-10. **Updated 2026-05-10 with Q35-Q39 results — the ensemble of small linear methods (PCA + CCA-5 + F-test) DOES beat naive baselines.** Best Test 2 top-1: **0.615 (ensemble)** vs random 0.32 vs raw cosine 0.50 vs best VAE 0.33.

## The headline numbers (Eraslan paired Test 2: 16 donors × 8 tissues, 24 LOO folds)

| Method | Top-1 | Top-3 | Rank-first | Wins (top-1=1.0) |
|---|---:|---:|---:|---:|
| Random baseline | 0.324 | 0.324 | 2.32 | 8/24 |
| **Deep-VAE family (Runs 14-22)** | | | | |
| Run 14 (baseline VAE flip) | 0.292 | 0.250 | 4.62 | 7/24 |
| Run 15 (+ z_bio matching) | 0.333 | 0.264 | 4.29 | 8/24 |
| Run 16 (+ InfoNCE on decoded) | 0.125 | 0.167 | 5.46 | 3/24 |
| Run 17 (+ decomposed decoder) | 0.333 | 0.306 | 4.42 | 8/24 |
| Run 18 (+ donor classifier) | 0.125 | 0.153 | 5.29 | 3/24 |
| Run 19 (+ 4413 GTEx donors)¹ | killed | — | — | — |
| Run 20 (single-tissue prostate) | 0.000 | 0.083 | 11.50 | 0/4 |
| Run 21 (pretrain + fine-tune) | 0.000 | 0.250 | 8.25 | 0/4 |
| Q33 (skip flip, latent NN) | 0.208 | 0.306 | 5.58 | 5/24 |
| Run 22 (contrastive embedding) | 0.292 | 0.250 | 7.42 | 8/24 |
| **No-training baselines (Q32, Q35)** | | | | |
| Raw cosine — no model | 0.500 | 0.354 | 3.59 | 12/24 |
| PCA-50 — no model | 0.510 | 0.486 | 2.51 | 12/24 |
| Spearman | 0.458 | 0.396 | 3.72 | — |
| F-test intersect + PCA-50 | 0.510 | 0.538 | 3.42 | — |
| **Small supervised methods (Q36, Q39)** | | | | |
| **CCA-5** | **0.604** | 0.486 | 4.09 | — |
| **Ensemble of 5 methods (Q39)** | **0.615** | **0.566** | **3.03** | — |

¹ Run 19 was killed mid-training when we pivoted to single-tissue.

## What we tried (and why none of it worked)

The architecture passed every self-consistency test (modality flip 100%/98%, round-trip Pearson 0.78, z_bio cycle 0.95) but FAILED the paired-donor test consistently. Eight different intervention strategies all hit chance:

1. **Standard VAE flip** (Run 14): just train and flip — 0.29
2. **Same-donor z_bio matching** (Run 15): force `z_bio(bulk_d) ≈ z_bio(sn_d)` — 0.33
3. **InfoNCE on decoded outputs** (Run 16): contrastive on flipped predictions — 0.13
4. **Decomposed decoder** (Run 17): `(1-α)·donor_branch + α·template` — 0.33
5. **Donor classifier head** (Run 18): force z_bio to predict donor identity — 0.13 (cross-tissue donor classification: 0.000 — striking failure)
6. **Massive bulk scaling** (Run 19): 4413 extra GTEx donors — killed mid-training
7. **Single-tissue training** (Run 20): just one tissue, more epochs — 0.000
8. **Pretrain + fine-tune** (Run 21): multi-tissue init + tissue-specific FT — 0.000 top-1, 0.250 top-3
9. **Skip the flip entirely** (Q33): NN in z_bio space — 0.21
10. **Contrastive embedding** (Run 22): CLIP-style InfoNCE, no decoder — 0.29

## The smoking gun

When we ran the simplest possible baseline (Q32) — just cosine similarity between bulk and sn pseudobulks in raw 11,374-gene space, **no model at all** — it got **50% top-1**, well above the 33% random baseline. PCA-50 hit 51%. Per-tissue, PCA-50 hit **100% top-1 on esophagus muscularis** with no learning whatsoever.

The data has donor signal across modalities. **Every learned model we trained destroys some of it.** The 50-D VAE z_bio is at HALF the top-1 of the 50-D PCA. The supervised losses (modality, tissue, paired flip, cycle, donor_id) all push the encoder toward those targets — collapsing the per-donor variance that PCA preserves by construction.

## What this means for the proposal

The proposal's framing was "flip metadata coordinates → reconstruct → check if NN is metadata-matched donor". Our finding:

- The **flip operation works at the cluster level** (modality cluster, tissue cluster) — Run 17 achieves 100%/98% modality flip into the right cluster.
- The **flip operation doesn't transfer per-donor identity** — the synthesized output is closer to the cluster average than to the source donor's actual cross-modality counterpart.
- The **reconstruction step is the bottleneck** — even the encoder's z_bio (which carries some donor info) gets collapsed when fed through the decoder.
- **Skip the model entirely**: cosine NN in raw bulk-vs-sn space already does what the proposal asked for, modestly. PCA-50 does it better.

This isn't a hyperparameter problem. This isn't a data problem (the data has signal). This is an architectural fit problem: VAE encode→flip→decode is a worse cross-modality NN than just doing cosine.

## What would actually help (data-wise)

Two paths the architecture would need to make learning beneficial:

1. **More donors** — 16 Eraslan donors gives ~3-4 per tissue. With 100+ paired donors per tissue, the supervised losses would have enough cross-donor variance to learn donor-discriminative features without collapsing them. No public dataset has this scale.

2. **Donor-anchoring metadata** — explicit per-donor markers (genotype, age, sex with high resolution) that the encoder can use as a fixed coordinate. Then the model isn't asked to discover donor identity from gene expression alone.

## Practical recommendation for the proposal write-up

Updated story after Q35-Q39:
1. **Lead with the architectural finding**: deep-VAE flip-and-decode methods (8 variants tested) underperform simple linear baselines for donor-faithful cross-modality NN at small data scale.
2. **Show the small-supervised win**: a 5-component CCA explicitly aligns paired modalities (top-1 = 0.604), and an average-rank ensemble of 5 simple methods (PCA + CCA-5 + F-test gene selection + cosine + Spearman) reaches **top-1 = 0.615** — nearly 2× the best deep VAE.
3. **Per-tissue: ensemble hits 100% top-1 on 2 tissues** (esophagus muscularis, skeletal muscle), 75% on prostate, 67% on lung/heart.
4. **The disentangled VAE still has a use**: cluster-level modality integration (Run 10: 100% modality flip, 0.95 z_bio cycle, 0.78 round-trip). It just isn't the right tool for per-donor faithful translation at this scale.
5. **The right architecture for this regime is small + linear + ensembled**, not deep + nonlinear.

The proposal's research question is now answered cleanly: the metadata-flip framework is achievable, but with an ensemble of small linear methods (CCA being the key new ingredient), not the SAMS-VAE family.

## Files

- All run scripts: `analysis/q20_disentangled_vae.py` (Runs 1-15), `analysis/q21_paired_donor_test2.py` (Q21), `analysis/q22-q34_*.py`
- All results: `analysis/results/q20_disentangled/q*.json`
- Datasets: `/Users/rls/ecs271/data/sc/eraslan/eraslan_paired{,_plus_gtex}.npz`
- Git history: `git log --oneline` — versioned per major experiment as v2.0.0 → v2.9.0
