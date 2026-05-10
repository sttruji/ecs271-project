#!/usr/bin/env python3
"""Q19b — Is the chromosome distribution of expressed genes informative?

Tests the null hypothesis that the 11,374 VAE 'shared genes' are distributed
across chromosomes in proportion to the total number of annotated human
genes on each chromosome (i.e. the chromosome view just recapitulates
genomic gene density and tells us nothing biology-specific).

Inputs:
  - results/q19_gene_chromosomes.csv     # expressed → chromosome (q19)
  - results/q19_chrom_total_genes.json   # all-Ensembl genes per chromosome

Outputs:
  - results/q19b_chrom_uninformative.csv     per-chrom obs / exp / o-e ratio
  - results/q19b_chrom_uninformative.json    chi^2 stats + interpretation
  - figures/q19b_chrom_uninformative.png     two-panel comparison

Run:  python analysis/q19b_chrom_uninformative.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)
RESULTS.mkdir(exist_ok=True, parents=True)

CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


def main() -> None:
    expr = pd.read_csv(RESULTS / "q19_gene_chromosomes.csv", dtype={"chr": str})
    expr = expr[expr["chr"].isin(CHROM_ORDER)]
    obs = expr.groupby("chr").size().reindex(CHROM_ORDER, fill_value=0)

    totals_raw = json.loads((RESULTS / "q19_chrom_total_genes.json").read_text())
    tot = pd.Series(
        {c: int(totals_raw[c]["total"]) for c in CHROM_ORDER},
        name="total_genome",
    )

    # Expected count per chromosome under H0:
    #   expressed_chr / total_expressed  ==  total_chr / total_genome
    n_obs = int(obs.sum())
    n_tot = int(tot.sum())
    expected = tot * (n_obs / n_tot)

    # Drop MT for the chi-square (its tiny size + biotype mix is noise here)
    # and renormalise expected so totals match exactly (scipy requirement).
    keep = [c for c in CHROM_ORDER if c != "MT"]
    obs_keep = obs.loc[keep].values.astype(float)
    exp_keep_unnorm = tot.loc[keep].values.astype(float)
    exp_keep = exp_keep_unnorm * (obs_keep.sum() / exp_keep_unnorm.sum())
    chi2, p = stats.chisquare(obs_keep, exp_keep)
    df = len(keep) - 1
    cramer_v = float(np.sqrt(chi2 / obs_keep.sum()))  # k=1 ⇒ φ = √(χ²/N)

    # Per-chrom observed/expected ratio + standardised residual.
    oe = obs / expected
    resid = (obs - expected) / np.sqrt(expected)
    summary = pd.DataFrame({
        "expressed": obs.astype(int),
        "annotated": tot.astype(int),
        "expressed_frac": (obs / tot).round(4),
        "expected_under_H0": expected.round(1),
        "obs_over_exp": oe.round(3),
        "std_residual": resid.round(2),
    })
    summary.index.name = "chr"
    summary.to_csv(RESULTS / "q19b_chrom_uninformative.csv")

    median_oe = float(np.median(oe.loc[keep]))
    iqr_oe = float(np.percentile(oe.loc[keep], 75) - np.percentile(oe.loc[keep], 25))
    max_dev = float(np.max(np.abs(oe.loc[keep] - 1.0)))

    payload = {
        "n_expressed_mapped": n_obs,
        "n_genome_annotated": n_tot,
        "chi2": float(chi2),
        "df": int(df),
        "p_value": float(p),
        "cramers_v": cramer_v,
        "obs_over_exp_median": median_oe,
        "obs_over_exp_iqr": iqr_oe,
        "obs_over_exp_max_abs_deviation": max_dev,
        "interpretation": (
            "Even when chi-square rejects H0 (n is large), the per-chrom "
            "observed/expected ratio lives close to 1.0 — the chromosome "
            "view recapitulates genomic gene density, not blood biology."
        ),
    }
    (RESULTS / "q19b_chrom_uninformative.json").write_text(json.dumps(payload, indent=2))

    print(f"expressed = {n_obs:,}   genome-annotated = {n_tot:,}")
    print(f"chi2 = {chi2:.1f}  df = {df}  p = {p:.2e}   "
          f"Cramér's V = {cramer_v:.3f}")
    print(f"obs/exp:  median = {median_oe:.2f}, IQR = {iqr_oe:.2f}, "
          f"max |o/e − 1| = {max_dev:.2f}")
    print("\n", summary.to_string(), sep="")

    # ── figure ───────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 7),
                                   gridspec_kw={"height_ratios": [1.1, 1]})

    # A) side-by-side counts (expressed vs total) on a log y to keep MT visible
    x = np.arange(len(CHROM_ORDER))
    w = 0.4
    axA.bar(x - w / 2, tot.values, width=w, color="#cfcfcf",
            label=f"all annotated genes (Ensembl, n={n_tot:,})")
    axA.bar(x + w / 2, obs.values, width=w, color="#3b6db5",
            label=f"expressed (VAE shared, n={n_obs:,})")
    axA.set_yscale("log")
    axA.set_ylabel("genes (count, log)")
    axA.set_xticks(x)
    axA.set_xticklabels([f"chr{c}" for c in CHROM_ORDER], rotation=45,
                        ha="right", fontsize=8)
    axA.set_title("Expressed-gene counts ≈ scaled copy of all-annotated counts")
    axA.legend(loc="upper right", fontsize=9, frameon=False)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # B) observed/expected ratio per chromosome
    colors = ["#999999"] * len(CHROM_ORDER)
    for i, c in enumerate(CHROM_ORDER):
        if abs(oe[c] - 1.0) > 0.2 or c == "MT":
            colors[i] = "#d2691e"
    axB.bar(x, oe.values, color=colors, edgecolor="white")
    axB.axhline(1.0, color="black", lw=0.8)
    axB.axhspan(0.8, 1.2, color="black", alpha=0.05, lw=0)
    for i, c in enumerate(CHROM_ORDER):
        if c == "MT":
            axB.annotate(f"{oe[c]:.1f}", (i, min(oe[c], 2.6)), ha="center",
                         va="bottom", fontsize=7, color="#d2691e")
    axB.set_ylim(0, 2.7)
    axB.set_ylabel("observed / expected")
    axB.set_xticks(x)
    axB.set_xticklabels([f"chr{c}" for c in CHROM_ORDER], rotation=45,
                        ha="right", fontsize=8)
    axB.set_title(
        f"Per-chromosome ratio (median={median_oe:.2f}, "
        f"max |o/e − 1|={max_dev:.2f}; shaded band = ±20 %).  "
        f"χ²(no MT)={chi2:.0f}, V={cramer_v:.3f}"
    )
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "q19b_chrom_uninformative.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG / 'q19b_chrom_uninformative.png'}")
    print(f"wrote {RESULTS / 'q19b_chrom_uninformative.csv'}")
    print(f"wrote {RESULTS / 'q19b_chrom_uninformative.json'}")


if __name__ == "__main__":
    main()
