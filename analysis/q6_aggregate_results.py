#!/usr/bin/env python3
"""Aggregate Q6 per-PC enrichment results into consumable tables + a markdown digest.

Reads:
  results/q6_extended_pc_biology/eigenvalues.csv
  results/q6_extended_pc_biology/pc_metadata_spearman.csv
  results/q6_extended_pc_biology/pc_metadata_eta2.csv
  results/q6_extended_pc_biology/PC{k}_{pos|neg}/{LIBRARY}.csv

Writes:
  results/q6_aggregated/per_pc_summary.csv            — one row per PC
  results/q6_aggregated/all_top_terms.csv             — all top hits (long form)
  results/q6_aggregated/q6_pc_biology.md              — readable digest
  figures/q6_pc_biology_strip.png                     — stacked summary strip

Safe to re-run any time — picks up whatever PCs have completed so far.

Run:  python q6_aggregate_results.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES_Q6 = ROOT / "results" / "q6_extended_pc_biology"
OUT_DIR = ROOT / "results" / "q6_aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"


def list_completed_pcs():
    pcs = set()
    for d in RES_Q6.glob("PC*_*"):
        m = re.match(r"PC(\d+)_(pos|neg)", d.name)
        if m:
            pcs.add(int(m.group(1)))
    return sorted(pcs)


def load_pc_terms(pc: int, direction: str) -> pd.DataFrame:
    d = RES_Q6 / f"PC{pc}_{direction}"
    rows = []
    for csv in d.glob("*.csv"):
        if csv.name.startswith("_"):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty or "term" not in df.columns:
            continue
        # Keep only top-1 per library for the digest, full set saved in long form
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    completed = list_completed_pcs()
    if not completed:
        print("No completed PCs found yet — run q6_extended_pc_biology.py first.")
        return 1
    print(f"Found {len(completed)} completed PCs: PC{completed[0]} .. PC{completed[-1]}")

    eig = pd.read_csv(RES_Q6 / "eigenvalues.csv")
    rho = pd.read_csv(RES_Q6 / "pc_metadata_spearman.csv", index_col=0)
    eta = pd.read_csv(RES_Q6 / "pc_metadata_eta2.csv", index_col=0)

    long_rows = []
    summary_rows = []

    for k in completed:
        eig_row = eig[eig["pc"] == k]
        if eig_row.empty:
            continue
        var_pct = float(eig_row["var_explained"].iloc[0])
        cum = float(eig_row["cum_var"].iloc[0])

        per_dir = {}
        for direction in ("pos", "neg"):
            df = load_pc_terms(k, direction)
            if df.empty:
                per_dir[direction] = (None, None, None, None)
                continue
            df["direction"] = direction
            df["pc"] = k
            df["var_explained"] = var_pct
            df["cum_var"] = cum
            long_rows.append(df)

            # best across libraries by adj_p
            best = df.sort_values("adj_p").iloc[0]
            per_dir[direction] = (
                str(best["term"]),
                str(best["library"]),
                float(best["adj_p"]),
                float(best.get("combined_score", float("nan"))),
            )

        # best metadata
        col = f"PC{k}"
        meta_cont_best, meta_cont_val = None, np.nan
        if col in rho.columns:
            cont = rho[col].abs().dropna()
            if not cont.empty:
                meta_cont_best = str(cont.idxmax())
                meta_cont_val = float(cont.max())
        meta_cat_best, meta_cat_val = None, np.nan
        if col in eta.columns:
            cat = eta[col].dropna()
            if not cat.empty:
                meta_cat_best = str(cat.idxmax())
                meta_cat_val = float(cat.max())

        summary_rows.append({
            "pc": k,
            "var_explained_pct": var_pct * 100,
            "cum_var_pct": cum * 100,
            "pos_term": per_dir["pos"][0],
            "pos_library": per_dir["pos"][1],
            "pos_adj_p": per_dir["pos"][2],
            "neg_term": per_dir["neg"][0],
            "neg_library": per_dir["neg"][1],
            "neg_adj_p": per_dir["neg"][2],
            "best_continuous_var": meta_cont_best,
            "best_continuous_rho": meta_cont_val,
            "best_categorical_var": meta_cat_best,
            "best_categorical_eta2": meta_cat_val,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "per_pc_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'per_pc_summary.csv'}  ({len(summary)} rows)")

    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        long_df.to_csv(OUT_DIR / "all_top_terms.csv", index=False)
        print(f"Wrote {OUT_DIR / 'all_top_terms.csv'}  ({len(long_df)} rows)")

    # ── Markdown digest ───────────────────────────────────────────────────
    lines = []
    lines.append(f"# Q6 — Per-PC biology digest (top {len(completed)} PCs)\n")
    K_95 = int(eig[eig["cum_var"] >= 0.95].iloc[0]["pc"]) if (eig["cum_var"] >= 0.95).any() else None
    if K_95 is not None:
        lines.append(f"K_95 (PCs needed for >=95% cumulative variance): **{K_95}**\n")
    if completed:
        lines.append(f"Completed enrichment for **PC1..PC{completed[-1]}**.\n")

    # Headline list — show every PC up to the first that fails the "interesting" threshold (adj_p < 0.05)
    lines.append("\n## Per-PC biology summary\n")
    lines.append("| PC | var % | cum % | Positive direction (top term) | Negative direction (top term) | Best metadata |")
    lines.append("|---:|---:|---:|---|---|---|")
    for row in summary_rows:
        pos = row["pos_term"] if row["pos_term"] else "—"
        neg = row["neg_term"] if row["neg_term"] else "—"
        pos_p = f"  *(adj-p={row['pos_adj_p']:.1e})*" if row["pos_term"] else ""
        neg_p = f"  *(adj-p={row['neg_adj_p']:.1e})*" if row["neg_term"] else ""
        meta = ""
        if row["best_continuous_var"]:
            meta = f"{row['best_continuous_var']} \\|ρ\\|={row['best_continuous_rho']:.2f}"
        if row["best_categorical_var"] and (np.isfinite(row["best_categorical_eta2"]) and row["best_categorical_eta2"] > 0.05):
            meta += (f"; {row['best_categorical_var']} η²={row['best_categorical_eta2']:.2f}"
                     if meta else f"{row['best_categorical_var']} η²={row['best_categorical_eta2']:.2f}")
        lines.append(
            f"| PC{row['pc']} | {row['var_explained_pct']:.2f} | {row['cum_var_pct']:.1f} | "
            f"{pos}{pos_p} | {neg}{neg_p} | {meta} |"
        )

    # ── Per-PC top-3 across libraries ─────────────────────────────────────
    lines.append("\n\n## Top 3 enriched terms per PC (across all libraries)\n")
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    for k in completed[:50]:  # cap text dump at 50
        for direction in ("pos", "neg"):
            sub = long_df[(long_df["pc"] == k) & (long_df["direction"] == direction)]
            if sub.empty:
                continue
            top3 = sub.sort_values("adj_p").head(3)
            head = f"\n### PC{k} ({direction}) — var = {float(eig[eig['pc']==k]['var_explained'].iloc[0]) * 100:.2f}%"
            lines.append(head)
            for _, r in top3.iterrows():
                lines.append(f"- *{r['library']}* — {r['term']}  (adj-p = {float(r['adj_p']):.1e}, "
                             f"combined = {float(r.get('combined_score', float('nan'))):.1f})")

    (OUT_DIR / "q6_pc_biology.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR / 'q6_pc_biology.md'}")

    # ── Strip figure: |ρ| of best metadata + var explained, by PC ─────────
    if not summary.empty:
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 1, figsize=(15, 5),
                                 gridspec_kw={"height_ratios": [1.2, 1]})
        sub = summary.sort_values("pc")
        axes[0].bar(sub["pc"], sub["var_explained_pct"], color="#3fb950")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("var %  (log)")
        axes[0].set_title(f"Q6 aggregated — {len(sub)} PCs evaluated")
        axes[0].axhline(0.1, color="#7d8590", ls=":", lw=0.7, label="0.1% threshold")
        axes[0].legend()

        axes[1].bar(sub["pc"], sub["best_continuous_rho"].abs().fillna(0),
                    color="#58a6ff", label="best |ρ| continuous")
        axes[1].bar(sub["pc"], -sub["best_categorical_eta2"].fillna(0),
                    color="#f78166", label="best η² categorical (flipped)")
        axes[1].set_ylabel("|ρ| (top) / η² (bottom)")
        axes[1].axhline(0, color="#7d8590", lw=0.5)
        axes[1].set_xlabel("PC index")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q6_pc_biology_strip.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'q6_pc_biology_strip.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
