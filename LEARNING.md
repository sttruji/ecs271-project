# Things to Learn

Concepts encountered during ECS 271 project — add new ones as they come up.

---

## ML / Statistics

- [ ] **LODO (Leave-One-Donor-Out)** — a cross-validation strategy where each fold holds out one donor entirely. The model trains on all other donors and is tested on the held-out donor. Measures genuine generalisation to unseen individuals, not just unseen samples from known individuals. Stricter than k-fold CV when donors have multiple samples.

- [ ] **5-fold CV vs LODO** — in 5-fold CV you hold out 20% of donors per fold and train fresh from scratch each time. This is the honest eval; "warm-start LODO" where the base model already saw all donors is leaky.

- [ ] **InfoNCE / NT-Xent** — contrastive losses that pull matched pairs (e.g. bulk donor X, sc donor X) together and push all other pairs apart in embedding space. NT-Xent uses a temperature τ: lower τ makes the distribution sharper and negatives harder. In-batch negatives = all other samples in the batch are negatives.

- [ ] **Convex Hull** — the smallest convex shape that encloses a set of points. Used here as a "territory" metric: hull fraction = fraction of sc embeddings that fall inside the convex hull of bulk embeddings. Hull fraction 0 = modalities completely separated; 1 = sc points are inside bulk territory. Does NOT mean paired samples are close to each other.

- [ ] **DANN (Domain Adversarial Neural Network)** — adds a gradient reversal layer before a domain classifier. The encoder tries to fool the classifier (is this bulk or sc?) while the classifier tries to succeed. At convergence, the encoder produces modality-invariant representations.

- [ ] **VAE posterior collapse** — when the KL term dominates, the encoder learns to output the prior (μ=0, σ=1) for every input. All latent dims become inactive. The decoder ignores z entirely and just learns the data mean.

- [ ] **Free bits** — a trick to prevent posterior collapse: clamp KL per dimension to a minimum value (e.g. 0.2 nats). Forces the encoder to keep each dim at least slightly informative.

- [ ] **FiLM (Feature-wise Linear Modulation)** — conditions a neural network on a side-input (e.g. metadata) by learning per-layer scale (γ) and shift (β) vectors from the side-input. Applied to each hidden layer: `h' = γ ⊙ h + β`. Gives the conditioning signal direct access to every layer.

---

## Genomics / Biology

- [ ] **Pseudobulk** — aggregating single-cell RNA-seq counts across all cells from one donor to get a single expression profile per donor. Makes sc data comparable in shape to bulk RNA-seq. Can be done per-cell-type (cell-type pseudobulk) or across all cells.

- [ ] **Bulk vs single-cell RNA-seq** — bulk measures average expression across millions of cells in a tissue sample (fast, cheap, stable). scRNA-seq measures each cell individually (expensive, sparse due to dropout, but reveals cell-type heterogeneity). Same donor's bulk and sc pseudobulk are correlated (~0.94 raw cosine) but models can introduce artificial gaps.

- [ ] **Dissociation stress** — when tissue is enzymatically dissociated to get single cells, some genes are stress-induced by the process itself (heat shock proteins, etc.). Introduces technical signal in scRNA-seq that doesn't exist in bulk.

- [ ] **HVG (Highly Variable Genes)** — genes with the highest variance across samples. Used to reduce dimensionality for model input, keeping the most informative genes.

- [ ] **eQTL (expression Quantitative Trait Locus)** — genomic variants that correlate with gene expression levels. Used in datasets like eQTLGen to study genetic regulation of expression.

---

## Datasets encountered

- [ ] **Eraslan 2022 (Science)** — 16 donors, 8 tissues (snRNA-seq + bulk). Multi-tissue confounds donor matching; stashed as wrong dataset for RQ1.

- [ ] **COMBAT (Cell 2022)** — 109 matched bulk+sc blood donors. CC-BY on Zenodo. Used for main RQ1 experiment.

- [ ] **GTEx** — large bulk RNA-seq atlas across many tissues and donors. Used for RQ2 age/metadata encoding.

- [ ] **HCA (Human Cell Atlas)** — large single-cell reference. Used for blood pseudobulk in Hetzner experiments.
