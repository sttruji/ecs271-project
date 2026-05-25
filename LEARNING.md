# Things to Learn

Concepts encountered during ECS 271 project — add new ones as they come up.

---

## ML / Statistics

- [ ] **LODO (Leave-One-Donor-Out)** — a cross-validation strategy where each fold holds out one donor entirely. The model trains on all other donors and is tested on the held-out donor. Measures genuine generalisation to unseen individuals, not just unseen samples from known individuals. Stricter than k-fold CV when donors have multiple samples.

- [ ] **5-fold CV vs LODO** — in 5-fold CV you hold out 20% of donors per fold and train fresh from scratch each time. This is the honest eval; "warm-start LODO" where the base model already saw all donors is leaky.

- [ ] **InfoNCE / NT-Xent** — contrastive losses that pull matched pairs (e.g. bulk donor X, sc donor X) together and push all other pairs apart in embedding space. NT-Xent uses a temperature τ: lower τ makes the distribution sharper and negatives harder. In-batch negatives = all other samples in the batch are negatives.

- [ ] **Convex Hull** — the smallest convex shape that encloses a set of points. Used here as a "territory" metric: hull fraction = fraction of sc embeddings that fall inside the convex hull of bulk embeddings. Hull fraction 0 = modalities completely separated; 1 = sc points are inside bulk territory. Does NOT mean paired samples are close to each other.

- [ ] **DANN (Domain-Adversarial Training of Neural Networks)** — Ganin et al. 2016. Adds a gradient reversal layer (GRL) between the encoder and a domain classifier. Forward pass: normal. Backward pass: gradients from the domain classifier are *negated* before reaching the encoder — so the encoder is pushed to produce representations the classifier *can't* distinguish. At convergence: encoder is modality-invariant (bulk and sc land in the same region), while still minimising the task loss. Key insight: no special training loop needed; GRL makes it a single end-to-end model. Paper: https://arxiv.org/abs/1505.07818

- [ ] **VAE posterior collapse** — when the KL term dominates, the encoder learns to output the prior (μ=0, σ=1) for every input. All latent dims become inactive. The decoder ignores z entirely and just learns the data mean.

- [ ] **Free bits** — a trick to prevent posterior collapse: clamp KL per dimension to a minimum value (e.g. 0.2 nats). Forces the encoder to keep each dim at least slightly informative.

- [ ] **FiLM (Feature-wise Linear Modulation)** — conditions a neural network on a side-input (e.g. metadata) by learning per-layer scale (γ) and shift (β) vectors from the side-input. Applied to each hidden layer: `h' = γ ⊙ h + β`. Gives the conditioning signal direct access to every layer.

- [ ] **DCL (Decoupled Contrastive Learning)** — Yeh et al., ECCV 2022. Standard InfoNCE includes the positive pair in the denominator, which creates "negative-positive coupling" — as the positive becomes closer, the denominator shrinks and destabilises the loss. DCL removes the positive from the denominator, separating alignment (pull positive close) from uniformity (push negatives away). Far more stable at small batch sizes. arxiv:2110.06848

- [ ] **MoCo (Momentum Contrast)** — He et al., CVPR 2020. Maintains a large queue of negative embeddings from previous batches. Each step: update the queue with the current batch's embeddings, use the full queue as negatives in InfoNCE. Gives N_queue negatives without needing them all in-batch. Especially powerful when N is small (e.g. 87 donors per fold). The key trick: a momentum encoder (slow-moving copy of the main encoder) encodes the queue entries for consistency.

- [ ] **Hard negative mining** — selecting negatives that are "hard" (close to the anchor but from a different class) rather than random negatives. In biology: a donor with similar disease severity is a hard negative because their expression is similar to the query. Methods: online semi-hard mining (Schroff 2015, FaceNet), adaptive hard weighting (scHSC 2025). Key insight: the confound (disease severity) defines which negatives are hard — use domain knowledge to find them.

- [ ] **Severity-informed contrastive weighting** — novel approach (not yet published): multiply each negative pair's InfoNCE contribution by `1 + γ · similarity(sev_i, sev_j)`. Donors with similar disease severity are harder negatives and contribute more to the loss. Targets the core confound in COMBAT (COVID severity dominates blood expression variation).

- [ ] **Hungarian assignment (linear_sum_assignment)** — instead of greedy argmax retrieval (each query picks its nearest neighbour independently), Hungarian assignment enforces one-to-one matching across all queries simultaneously. Solves the optimal bipartite matching problem. Available in `scipy.optimize.linear_sum_assignment`. Zero cost at inference — just replace argmax with this call.

- [ ] **VICReg** — Bardes et al., ICLR 2022. Variance-Invariance-Covariance Regularization. Prevents embedding collapse by penalising: (1) dimensions with near-zero variance (variance term), (2) high off-diagonal covariance between dims (covariance term). Batch-size robust. Used as auxiliary loss on top of InfoNCE to prevent the encoder from collapsing all embeddings to a point or subspace at small N.

- [ ] **BulkRNABert** — InstaDeepAI, bioRxiv 2024. First BERT-style transformer pretrained on bulk RNA-seq (TCGA + GTEx + ENCODE, ~500k profiles). Treats each gene as a token. Weights on HuggingFace: `InstaDeepAI/BulkRNABert`. Can be used as a frozen encoder for bulk RNA-seq to get rich 768-dim embeddings without training from scratch on small N.

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
