# Variational Autoencoders

*An algorithm explainer for computational biologists.*

---

## The Problem Autoencoders Solve

You have 803 blood donors, each described by 11,374 gene expression values. Most of that 11,374-dimensional variation is structured — donors cluster by age, ischemia time, cell composition — not random. You want a compact representation that keeps the biology and throws away the noise.

A standard **autoencoder** does this with two networks:

```
x ──[ Encoder ]──> z ──[ Decoder ]──> x̂
(11374-dim)        (16-dim)            (11374-dim)
```

Train on reconstruction loss `‖x - x̂‖²`. The bottleneck forces `z` to retain only what's needed to rebuild `x`. This works, but the learned `z` has two problems:

1. **No density model.** The encoder outputs a single point per sample, so the latent space has holes — you can't sample new plausible gene expression profiles.
2. **Entangled dimensions.** Nothing stops `z₁` from encoding a mix of age, ischemia, and cell type simultaneously. Interpolating between two donors in `z` gives biological nonsense.

---

## From Autoencoder to VAE

The key idea: instead of encoding `x` to a point `z`, encode it to a **distribution** over `z`.

The encoder outputs two vectors for each input:
- `μ(x)` — the mean of the posterior
- `σ(x)` — the (log-)standard deviation

Then `z` is **sampled** from this distribution: `z ~ N(μ(x), σ(x)²)`.

The decoder reconstructs from the sample: `x̂ = Dec(z)`.

### The ELBO (Evidence Lower BOund)

Training minimizes the **negative ELBO**:

```
L = E[‖x - x̂‖²]  +  β · KL[q(z|x) ‖ p(z)]
      reconstruction       regularisation
```

- **Reconstruction term** — how well does the decoder rebuild `x`? Lower is better.
- **KL term** — how close is the learned posterior `q(z|x) = N(μ, σ²)` to the prior `p(z) = N(0, I)`? This pushes `z` toward a unit Gaussian, preventing any dimension from collapsing to a delta function or exploding to infinity.

For a diagonal Gaussian encoder, the KL has a closed form:

```
KL[q ‖ p] = Σₖ [ -½ (1 + log σₖ² - μₖ² - σₖ²) ]
```

### The Reparameterisation Trick

Sampling `z ~ N(μ, σ²)` is non-differentiable — gradients can't flow through a stochastic node. The fix: rewrite the sample as a deterministic transformation of a noise variable:

```
z = μ + σ · ε,   ε ~ N(0, I)
```

Now gradients flow through `μ` and `σ` normally; only the `ε` draw is random and lies outside the computation graph.

### Free Bits (Anti-Collapse)

With a very small `β`, the KL loss has little pull and some dimensions collapse — `σ → 0`, `μ → 0`, the dimension encodes nothing. The **free bits** trick clips per-dimension KL before summing:

```
KL_total = Σₖ max(KL_k, λ)
```

Setting `λ = 0.1` means each dimension is "free" up to 0.1 nats of KL — dimensions don't get penalised for using a small amount of capacity, so they stay active.

---

## β-VAE and Disentanglement

Standard VAE has `β = 1`. **β-VAE** uses `β > 1` to add extra pressure on the KL term. This forces the model to use the latent space more efficiently, often resulting in each `z` dimension capturing one independent factor of variation.

```
L_βVAE = reconstruction + β · KL,   β > 1
```

The intuition: the stronger the KL penalty, the more the posterior is pushed toward `N(0, I)`. Each dimension must either encode something critical for reconstruction or get zeroed out. Correlated dimensions are wasteful — the model learns to separate them.

**β-TCVAE** (Total Correlation VAE) makes the disentanglement objective explicit. It decomposes the KL into three parts:

```
KL = MI(x; z)  +  TC(z)  +  Σₖ KL[q(zₖ) ‖ p(zₖ)]
     mutual info   total corr   dimension-wise KL
```

where **Total Correlation** measures statistical dependence across all pairs of dimensions simultaneously:

```
TC(z) = KL[ q(z) ‖ ∏ₖ q(zₖ) ]
```

β-TCVAE penalises TC directly:

```
L = reconstruction + MI + β·TC + dim-KL
```

This explicitly encourages each `zₖ` to be marginally independent of every `zⱼ` for `k ≠ j`.

### Minibatch TC Estimate

TC requires the *aggregate* posterior `q(z)` which involves all training points. We approximate it using a minibatch: for a batch of `N` samples,

```
log q(zᵢ) ≈ logsumexp_j [log q(zᵢ|xⱼ)] - log N
```

TC = mean over `i` of `[log q(zᵢ) - Σₖ log q(zᵢₖ)]` — the gap between the joint and the product of marginals.

---

## Metadata-Injection: Separating Biology from Confounders

In GTEx blood, a large fraction of gene expression variance comes from **ischemia time** (SMTSISCH) — the time from death to tissue preservation. This is a technical confounder, not biology.

Standard β-VAE will encode it in `z` because it helps reconstruction. We want `z` to contain only biology.

### Architecture (MetaInjection VAE)

The design constraint: *metadata enters the decoder only, never the encoder*.

```
Encoder:  x ──────────────────────────> z_bio
Decoder:  concat(z_bio, meta) ─────────> x̂
```

At training time, every sample has different metadata, so the decoder is forced to use it. The encoder has no access to metadata — whatever metadata effects are in `x` must either be captured by `z_bio` (bad) or learned by the decoder from `meta` (good).

### FiLM Conditioning

Simple concatenation gives metadata one input slot in the first decoder layer. **Feature-wise Linear Modulation (FiLM)** gives metadata control over *every* layer:

```
h_l = Linear(h_{l-1})
h_l = γ_l ⊙ h_l + β_l,     where γ_l, β_l = f_l(meta_emb)
h_l = GELU(h_l)
```

`meta_emb` is a 128-dim embedding of the raw metadata. `γ_l` and `β_l` are per-neuron scale and shift parameters, produced by a linear layer on `meta_emb`. The decoder can now gate any subset of neurons based on metadata, allowing arbitrarily expressive metadata conditioning.

FiLM was originally developed for visual question answering (film a question about an image to the image encoder) and works well whenever one input needs to *modulate* the processing of another.

### Ischemia-Residual Encoding

Even with FiLM, the encoder learns some ischemia signal because ischemia-correlated expression is in `x`. We break this by preprocessing:

```
X_resid[i] = X_scaled[i] - β_isch · SMTSISCH_z[i]
```

where `β_isch` (shape `G`) is the per-gene OLS slope of expression on ischemia time, fit on training data.

The encoder now sees `X_resid` — expression *after* removing the linear ischemia effect. The decoder target is still the full `X_scaled`. The decoder is therefore forced to *re-add* the ischemia effect from `meta.SMTSISCH`, and the encoder's job is purely biological.

Result: SMTSISCH R² in `z_bio` drops from 0.446 → 0.021.

---

## Evaluating the Latent Space

### Reconstruction (Roundtrip)

Encode `x` to `z_bio`, decode with original metadata, compare to `x`. Good roundtrip = the bottleneck is not losing information critical for reconstruction. We measure per-sample R² and per-gene Pearson correlation across samples.

### FC Resolution (Marioni Criterion)

A count-based precision benchmark. For a gene with average count `μ`, the 95% Poisson confidence interval on its log₂ fold-change between two groups is:

```
tolerance = 1.96 · √(2/μ) / ln(2)  ≈  4.0 / √μ  (log₂ units)
```

A model passes if `|pred_FC - true_FC| ≤ tolerance`. This is analogous to AlphaFold's pLDDT — a data-driven, count-calibrated confidence threshold.

For biological replicates, Poisson understates variance. A **negative-binomial** model with dispersion `φ` gives:

```
tolerance_NB = 1.96 · √(2/μ + 2φ) / ln(2)
```

At `φ = 0.10` (human outbred), tolerance ≈ 1.265 log₂ for any high-count gene — much looser than the Poisson 0.015 at `μ = 70,754`.

### Flip Test (Within-Sample)

For a disentangled model, the flip test measures: *does the decoder's meta pathway encode the correct biological effect?*

**Wrong design (group FC flip):** Take lo-ischemia `z_bio`, decode with hi-ischemia meta, compare group mean to true group FC. Fails for disentangled models — with ischemia removed from `z_bio`, decoding both groups with the same meta gives FC ≈ 0, which is wrong.

**Correct design (within-sample):** For every sample `i`, compute:

```
δᵢ = decode(z_bio_i, meta_hi) - decode(z_bio_i, meta_lo)
```

Average `δ` across all `i` and compare to the true group FC. This isolates the decoder's learned metadata effect from individual biology.

### Disentanglement (Linear Probes + MIG)

For each metadata variable, fit a cross-validated Ridge probe on `z_bio`. R² measures how much metadata is linearly recoverable from the embedding.

Per-dimension: fit a probe on each `zₖ` independently. **MIG gap** = (best dim R²) − (2nd best dim R²). A high gap means one dimension "owns" that factor.

---

## Key Implementation Choices

| Choice | Why |
|--------|-----|
| β = 5×10⁻⁴ with warmup | Start at β=0 to prevent early collapse; raise slowly |
| free bits λ = 0.1 | Keep all K dimensions active through training |
| FiLM γ init = 1, β init = 0 | Model starts as standard VAE; FiLM gradually takes over |
| Count-weighted MSE | Aligns training objective with FC resolution criterion |
| Cosine LR schedule | Smooth convergence without manual LR stepping |
| Gradient clipping at 5.0 | Stability when β is small and KL gradients spike |

---

## Connections to Other Methods

| Method | How it relates |
|--------|----------------|
| PCA | Linear VAE with β→∞; special case where encoder and decoder are transposed matrices |
| scVI (Lopez et al. 2018) | VAE for scRNA-seq; uses NB likelihood instead of MSE; no metadata injection |
| DRVI (Lotfollahi et al. 2024) | Additive decoder (per-dim MLPs); VAE for disentangled gene programs |
| CVAE (Sohn et al. 2015) | Conditional VAE; meta goes into BOTH encoder and decoder; violates our design constraint |
| SAMS-VAE (Bereket & Karaletsos 2023) | Additive structure; meta injection into decoder; closest to our architecture |
| FiLM (Perez et al. 2018) | Feature-wise Linear Modulation; originally for visual QA |

---

## Glossary

| Term | Meaning |
|------|---------|
| ELBO | Evidence Lower BOund — the objective maximised during training |
| KL divergence | Kullback-Leibler divergence; measures how one distribution differs from another |
| Posterior collapse | When a latent dimension collapses to the prior; encodes nothing |
| Aggregate posterior | `q(z) = E_{p(x)}[q(z|x)]` — marginal latent distribution over all data |
| Disentanglement | Each latent dimension captures one generative factor independently |
| FiLM | Feature-wise Linear Modulation — scale/shift a layer's activations conditioned on another input |
| TC | Total Correlation — measures statistical dependence across all dimensions jointly |
| Reparameterisation | Rewrite a random sample as `μ + σε` to enable gradient flow |
| Free bits | Per-dim KL clipping to prevent posterior collapse |
