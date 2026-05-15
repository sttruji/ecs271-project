# Bulk RNA-seq Fold-Change Resolution Benchmark

**Derived from Marioni 2008 (Poisson technical noise) and Gierliński 2015 (biological dispersion)**

---

## Provenance and Verification

This benchmark is built from first principles, not from numbers in secondary sources.

### What Marioni 2008 directly establishes

> *"the variation across technical replicates can be captured using a Poisson model, with only a small proportion (∼0.5%) of genes showing clear deviations"*
>
> *"the gene counts are highly correlated across lanes (average Spearman correlation = 0.96)"*

- Design: 7 technical replicates (lanes) per sample, liver + kidney
- Median count per gene: 46 (liver), 101 (kidney)
- **Conclusion: technical (lane-to-lane) variance is Poisson — variance = mean.**

Cite as: Marioni, J.C., Mason, C.E., Mane, S.M., Stephens, M. & Gilad, Y. *Genome Res* 18:1509–1517 (2008).

### What Gierliński 2015 establishes (48-replicate yeast)

- Measured biological dispersion: **φ̃_WT = 2.35 × 10⁻²**, **φ̃_Δsnf2 = 1.35 × 10⁻²**
- Confirms inter-lane (technical) data follow Poisson exactly.

Cite as: Gierliński, M. et al. *Bioinformatics* 31:3625–3630 (2015).

---

## Derivation

Under pure Poisson (technical replicates), for counts X₁, X₂ ~ Poisson(μ):

```
Var(log X) ≈ 1/μ                       [delta method]
Var(log(X₁/X₂)) ≈ 2/μ
SD(log₂ FC) ≈ √(2/μ) / ln(2) = 2.04 / √μ
95% CI half-width = 1.96 × 2.04/√μ = 4.0/√μ  [log₂ scale]
Resolvable fold-change at 95% = 2^(4.0/√μ)
```

**There is no plateau.** Under pure Poisson, resolution improves indefinitely with √count.

### Count-dependent resolution table (Poisson, technical floor)

| Count μ | SD(log₂ FC) | 95% CI half-width (log₂) | Fold-change resolution |
|--------:|------------:|-------------------------:|----------------------:|
| 10      | 0.645       | 1.265                    | 2.40×                 |
| 50      | 0.288       | 0.566                    | 1.48×                 |
| 100     | 0.204       | 0.400                    | 1.32×                 |
| 500     | 0.091       | 0.179                    | 1.13×                 |
| 1 000   | 0.065       | 0.127                    | 1.09×                 |
| 10 000  | 0.020       | 0.040                    | 1.03×                 |
| 100 000 | 0.0065      | 0.013                    | 1.009×                |

---

## When a Floor Appears (Biological Replicates)

A plateau requires extra-Poisson (library prep or biological) variance. For a negative binomial model with dispersion φ:

```
SD(log₂ FC) ≈ √(2/μ + 2φ) / ln(2)
Asymptotic floor (μ → ∞): SD = √(2φ) / ln(2)
```

| φ       | Asymptotic fold-change resolution (95%) | Context                                     |
|--------:|----------------------------------------:|:--------------------------------------------|
| 0       | None — keeps improving                  | Same library, different lanes (Marioni)     |
| 0.005   | ~1.22×                                  | Very tight library-to-library (hypothetical)|
| 0.014   | ~1.45×                                  | Yeast biological replicates (Gierliński)    |
| 0.024   | ~1.65×                                  | Yeast biological replicates (Gierliński)    |
| 0.05    | ~1.93×                                  | Inbred mouse biological replicates          |
| 0.10    | ~2.43×                                  | Outbred human biological replicates         |

---

## Benchmark Specification

### Evaluation criterion

A model prediction of log₂ fold-change is **within measurement resolution** if:

```
|predicted_log2_fc - measured_log2_fc| ≤ 4.0 / √(observed_count)
```

This is the analog of AlphaFold's pLDDT confidence threshold: a count-dependent tolerance that reflects what the measurement itself can actually resolve.

### Pass/fail at common count ranges

| Count range | Tolerance (log₂) | Tolerance (fold-change) | Rule of thumb          |
|------------:|------------------:|------------------------:|:-----------------------|
| < 10        | > 1.26            | > 2.4×                  | Skip or flag low-count |
| 10–100      | 0.40–1.26         | 1.32–2.40×              | Coarse resolution      |
| 100–1 000   | 0.13–0.40         | 1.09–1.32×              | Moderate resolution    |
| > 1 000     | < 0.13            | < 1.09×                 | High resolution        |

### Recommended benchmark procedure

1. For each gene *i*, compute observed count μᵢ (average across replicates).
2. Compute tolerance: `tol_i = 4.0 / sqrt(mu_i)` (log₂ units).
3. Score: `pass_i = |pred_log2fc_i - true_log2fc_i| <= tol_i`.
4. Report:
   - **Within-resolution accuracy**: fraction of genes where `pass_i = True`
   - Stratified by count bin (< 10, 10–100, 100–1000, > 1000)
   - Mean absolute error in log₂ FC units (unweighted and count-weighted)
5. Baseline: a null predictor that always predicts 0 log₂ FC scores `pass_i = True` iff `|true_log2fc_i| ≤ tol_i` — i.e., only for genes not actually differentially expressed.

### What counts as a "good" model

- **Within-resolution accuracy ≥ 90%** on high-count genes (μ > 1000): model predictions are as precise as the measurement allows.
- **Within-resolution accuracy ≥ 70%** on moderate-count genes (100–1000): acceptable given larger Poisson noise.
- **Mean |log₂ FC error| < 0.5** overall: practically meaningful accuracy.

---

## Notes on What Was Retracted

An earlier version of this analysis cited a "1.23-fold floor" for bulk RNA-seq technical replicates. That number cannot be found in any primary source and appears to have been generated from an uncited back-of-envelope calculation. **It is retracted.** The correct claim is:

- Under Marioni 2008's Poisson finding, **there is no floor** — resolution improves with √count.
- A floor only appears if you include extra-Poisson variance (φ > 0 in the NB model).
- The numbers in the table above are **derived**, not directly cited. Marioni 2008 is the justification for the Poisson model; the table follows from Poisson statistics.
