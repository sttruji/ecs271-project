#!/usr/bin/env python3
"""Q6 — Extended PC biology: every PC up to >=95% cumulative variance.

For each PC k in [1, K_95]:
  - Top 200 +/- loading genes submitted to Enrichr against several libraries:
        GO_Biological_Process_2023, GO_Molecular_Function_2023,
        GO_Cellular_Component_2023, KEGG_2021_Human, Reactome_2022,
        WikiPathway_2023_Human, MSigDB_Hallmark_2020,
        ChEA_2022 (TF binding), TRRUST_Transcription_Factors_2019,
        Human_Phenotype_Ontology
  - Spearman correlation of PC scores with all available continuous +
    categorical GTEx metadata.
  - Top-3 hits per (PC, direction, library) saved to per-PC CSVs; a
    consolidated `q6_pc_biology_master.csv` is the consumable summary.

Resumes from cache: if a per-PC enrichment file exists for a (k, dir, lib)
triple, that call is skipped on rerun.

Run:  python q6_extended_pc_biology.py [--max-pcs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
RESULTS = ROOT / "results" / "q6_extended_pc_biology"
RESULTS.mkdir(exist_ok=True, parents=True)
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True)

ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
    "TRRUST_Transcription_Factors_2019",
]
# Wider library set used selectively for the top-30 highest-variance PCs
LIBRARIES_RICH = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathway_2023_Human",
    "MSigDB_Hallmark_2020",
    "ChEA_2022",
    "TRRUST_Transcription_Factors_2019",
    "Human_Phenotype_Ontology",
]
RICH_PC_LIMIT = 30
TOP_GENES_PER_DIRECTION = 200
CUM_VAR_TARGET = 0.95
TOP_K_PER_LIB = 3   # how many top hits to keep per (PC, direction, library)


def sample_to_subject(s: str) -> str:
    return "-".join(s.split("-")[:2])


def age_to_midpoint(b):
    if not isinstance(b, str) or "-" not in b:
        return np.nan
    a, c = b.split("-")
    try:
        return (int(a) + int(c)) / 2
    except ValueError:
        return np.nan


def enrichr_submit(genes, description):
    import requests
    payload = {"list": (None, "\n".join(genes)), "description": (None, description)}
    r = requests.post(f"{ENRICHR}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def enrichr_query(uid, library, top_k=10):
    import requests
    r = requests.get(
        f"{ENRICHR}/enrich",
        params={"userListId": uid, "backgroundType": library},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json().get(library, [])
    cols = ["rank", "term", "p", "z", "combined_score", "overlap_genes",
            "adj_p", "old_p", "old_adj_p"]
    out = pd.DataFrame(rows, columns=cols)
    out["overlap_genes"] = out["overlap_genes"].apply(
        lambda g: ";".join(g) if isinstance(g, list) else g)
    return out.sort_values("p").head(top_k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pcs", type=int, default=None,
                    help="cap K_95 at this many (debug; default: full K_95)")
    ap.add_argument("--max-libs", type=int, default=None,
                    help="cap number of libraries per call (debug)")
    args = ap.parse_args()

    libraries = LIBRARIES[: args.max_libs] if args.max_libs else LIBRARIES

    print("Loading shared genes / scaler ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")

    print("Loading GTEx ...")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)

    # Sample IDs
    import gzip
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        sample_ids = np.asarray(fh.readline().rstrip("\n").split("\t")[2:])

    # Full PCA (no train/test split — biology characterization, not generalization)
    n = expr_scaled.shape[0]
    n_components = min(n - 1, expr_scaled.shape[1])
    print(f"Fitting PCA up to {n_components} components on full 803 × 11,374 matrix ...")
    pca = PCA(n_components=n_components).fit(expr_scaled)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    eigs = pca.explained_variance_  # the actual eigenvalues (variances along each PC)

    K_95 = int(np.searchsorted(cum_var, CUM_VAR_TARGET) + 1)
    print(f"\n→ K_95 (cumvar >= {CUM_VAR_TARGET}): PC1..PC{K_95}  (cumulative variance {cum_var[K_95 - 1]:.3f})")
    K_run = min(K_95, args.max_pcs) if args.max_pcs else K_95
    print(f"→ Will run enrichment for PC1..PC{K_run}  (limited by --max-pcs={args.max_pcs})")

    # Save eigenvalue table
    eig_df = pd.DataFrame({
        "pc": np.arange(1, n_components + 1),
        "eigenvalue": eigs,
        "var_explained": var_ratio,
        "cum_var": cum_var,
    })
    eig_df.to_csv(RESULTS / "eigenvalues.csv", index=False)

    # Eigenvalue scree + cumulative-variance plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(np.arange(1, len(eigs) + 1), eigs, color="#3fb950", lw=1.2)
    ax[0].set_xlabel("PC index"); ax[0].set_ylabel("Eigenvalue (variance)")
    ax[0].set_yscale("log"); ax[0].set_xscale("log")
    ax[0].set_title("Q6: Eigenvalue scree (full GTEx blood, 803 donors)")
    ax[0].axvline(K_95, color="#f78166", ls="--", lw=1, label=f"K_95 = {K_95}")
    ax[0].legend()

    ax[1].plot(np.arange(1, len(cum_var) + 1), cum_var, color="#58a6ff", lw=1.5)
    ax[1].axhline(CUM_VAR_TARGET, color="#f78166", ls="--", lw=1, label=f"{CUM_VAR_TARGET*100:.0f}% target")
    ax[1].axvline(K_95, color="#f78166", ls="--", lw=1, label=f"K_95 = {K_95}")
    ax[1].set_xlabel("PC index"); ax[1].set_ylabel("Cumulative explained variance")
    ax[1].set_title(f"Cumulative variance — first {K_95} PCs reach 95%")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "q6_eigenvalues.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q6_eigenvalues.png'}")

    # ── Metadata join + per-PC Spearman ───────────────────────────────────
    print("\nLoading metadata ...")
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
                       sep="\t", low_memory=False)
    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(samp[["SAMPID", "SMRIN", "SMTSISCH", "SMCENTER", "SMNABTCH",
                        "SMGEBTCH", "SMRDLGTH", "SMTSPAX"]],
                  on="SAMPID", how="left")
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)

    pc_scores = pca.transform(expr_scaled)
    cont_cols = ["AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"]
    cat_cols = ["SEX", "SMCENTER", "SMNABTCH", "SMGEBTCH"]

    rho_table = pd.DataFrame(index=cont_cols, columns=[f"PC{k+1}" for k in range(K_run)], dtype=float)
    for col in cont_cols:
        v = pd.to_numeric(df[col], errors="coerce").values
        for k in range(K_run):
            x = pc_scores[:, k]
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 30:
                continue
            r, _ = spearmanr(v[mask], x[mask])
            rho_table.iloc[cont_cols.index(col), k] = r
    rho_table.to_csv(RESULTS / "pc_metadata_spearman.csv")

    eta_table = pd.DataFrame(index=cat_cols, columns=[f"PC{k+1}" for k in range(K_run)], dtype=float)
    for col in cat_cols:
        groups = df[col].fillna("NA").astype(str).values
        if len(set(groups)) > 30:
            top = pd.Series(groups).value_counts().head(30).index.tolist()
            mk = np.isin(groups, top); groups = groups[mk]
        else:
            mk = np.ones_like(groups, dtype=bool)
        for k in range(K_run):
            x = pc_scores[mk, k]
            buckets = [x[groups == g] for g in sorted(set(groups)) if (groups == g).sum() > 1]
            if len(buckets) < 2:
                continue
            try:
                fstat, _ = f_oneway(*buckets)
                kg = len(buckets); nobs = sum(b.size for b in buckets)
                eta_table.iloc[cat_cols.index(col), k] = (fstat * (kg - 1)) / (fstat * (kg - 1) + (nobs - kg))
            except Exception:
                pass
    eta_table.to_csv(RESULTS / "pc_metadata_eta2.csv")

    # ── Enrichment per PC, with caching ───────────────────────────────────
    components = pca.components_  # (n_components, n_genes)
    print(f"\nRunning enrichment on {K_run} PCs × 2 directions × {len(libraries)} libraries ...")
    print(f"Estimated calls: {K_run * 2 * (len(libraries) + 1)}  (~{K_run * 2 * (len(libraries) + 1) * 0.6 / 60:.1f} min)")

    master_rows: list[dict] = []
    for k in range(K_run):
        loadings = components[k]
        order = np.argsort(loadings)
        bot = order[:TOP_GENES_PER_DIRECTION]
        top = order[-TOP_GENES_PER_DIRECTION:]
        for direction, idx in [("pos", top), ("neg", bot)]:
            tag = f"PC{k+1}_{direction}"
            tag_dir = RESULTS / tag
            tag_dir.mkdir(exist_ok=True)
            genes = [str(g) for g in shared_genes[idx]]

            # Submit once per (PC, direction)
            list_id_path = tag_dir / "_list_id.txt"
            if list_id_path.exists():
                lid = list_id_path.read_text().strip()
            else:
                try:
                    lid = enrichr_submit(genes, tag)["userListId"]
                    list_id_path.write_text(str(lid))
                except Exception as exc:
                    print(f"  !! submit {tag}: {exc}")
                    continue

            # First-30 PCs get the rich library set; the long tail uses LIBRARIES.
            libs_for_this_pc = LIBRARIES_RICH if (k < RICH_PC_LIMIT and not args.max_libs) else libraries
            for lib in libs_for_this_pc:
                cache = tag_dir / f"{lib}.csv"
                if cache.exists():
                    tbl = pd.read_csv(cache)
                else:
                    try:
                        tbl = enrichr_query(lid, lib, top_k=TOP_K_PER_LIB)
                        tbl["pc"] = k + 1
                        tbl["direction"] = direction
                        tbl["library"] = lib
                        tbl.to_csv(cache, index=False)
                        time.sleep(0.35)
                    except Exception as exc:
                        print(f"  !! {tag} {lib}: {exc}")
                        continue
                if not tbl.empty and "term" in tbl.columns:
                    row = tbl.iloc[0]
                    master_rows.append({
                        "pc": k + 1,
                        "var_explained": float(var_ratio[k]),
                        "cum_var": float(cum_var[k]),
                        "direction": direction,
                        "library": lib,
                        "top_term": str(row["term"]),
                        "top_adj_p": float(row.get("adj_p", float("nan"))),
                        "combined_score": float(row.get("combined_score", float("nan"))),
                    })
        if (k + 1) % 5 == 0 or k == 0:
            print(f"  ... done PC{k+1}/{K_run}  (cum var {cum_var[k]:.3f})")

    master = pd.DataFrame(master_rows)
    master.to_csv(RESULTS / "q6_pc_biology_master.csv", index=False)
    print(f"\nWrote {RESULTS / 'q6_pc_biology_master.csv'}  ({len(master)} rows)")

    # ── Build a per-PC summary across all libraries ───────────────────────
    rows = []
    for k in range(K_run):
        for direction in ("pos", "neg"):
            sub_m = master[(master["pc"] == k + 1) & (master["direction"] == direction)]
            if sub_m.empty:
                continue
            best = sub_m.sort_values("top_adj_p").iloc[0]
            rows.append({
                "pc": k + 1,
                "var_explained": float(var_ratio[k]),
                "cum_var": float(cum_var[k]),
                "direction": direction,
                "best_term_overall": best["top_term"],
                "best_library_overall": best["library"],
                "best_adj_p_overall": best["top_adj_p"],
                "best_combined_overall": best["combined_score"],
            })
    pd.DataFrame(rows).to_csv(RESULTS / "q6_pc_top_term_per_direction.csv", index=False)

    # also: best metadata per PC
    meta_rows = []
    for k in range(K_run):
        col = f"PC{k+1}"
        cont = rho_table[col].abs()
        cat = eta_table[col]
        meta_rows.append({
            "pc": k + 1,
            "var_explained": float(var_ratio[k]),
            "cum_var": float(cum_var[k]),
            "best_continuous_var": cont.idxmax() if cont.notna().any() else None,
            "best_continuous_rho": float(cont.max()) if cont.notna().any() else float("nan"),
            "best_categorical_var": cat.idxmax() if cat.notna().any() else None,
            "best_categorical_eta2": float(cat.max()) if cat.notna().any() else float("nan"),
        })
    pd.DataFrame(meta_rows).to_csv(RESULTS / "q6_pc_top_metadata.csv", index=False)

    # ── Headline figure: PC heat-strip ────────────────────────────────────
    print("\nBuilding summary heatmap ...")
    show_pcs = min(K_run, 80)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5),
                             gridspec_kw={"height_ratios": [1, 1]})

    # Variance-explained bars colored by best library
    var_colors = []
    for k in range(show_pcs):
        sub_m = master[(master["pc"] == k + 1)].sort_values("top_adj_p")
        if sub_m.empty:
            var_colors.append("#7d8590")
        else:
            best_lib = sub_m.iloc[0]["library"]
            cmap = {
                "GO_Biological_Process_2023": "#3fb950",
                "GO_Molecular_Function_2023": "#56d364",
                "GO_Cellular_Component_2023": "#7ee787",
                "KEGG_2021_Human": "#58a6ff",
                "Reactome_2022": "#79c0ff",
                "WikiPathway_2023_Human": "#a5d6ff",
                "MSigDB_Hallmark_2020": "#d2a8ff",
                "ChEA_2022": "#f78166",
                "TRRUST_Transcription_Factors_2019": "#fa8067",
                "Human_Phenotype_Ontology": "#f0883e",
            }
            var_colors.append(cmap.get(best_lib, "#7d8590"))
    axes[0].bar(range(1, show_pcs + 1), var_ratio[:show_pcs] * 100, color=var_colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Var explained (%, log)")
    axes[0].set_title(f"Q6: PCs 1..{show_pcs} of {K_run} (K_95 = {K_95}); bar color = best-enrichment library")
    axes[0].axvline(K_95 + 0.5, color="#f78166", ls="--", lw=1)

    # Metadata heatmap
    meta_disp = rho_table.iloc[:, :show_pcs].astype(float).abs()
    eta_disp = eta_table.iloc[:, :show_pcs].astype(float)
    combined_meta = pd.concat([meta_disp, eta_disp])
    im = axes[1].imshow(combined_meta.values, aspect="auto", cmap="viridis", vmin=0, vmax=0.7)
    axes[1].set_yticks(range(combined_meta.shape[0]))
    axes[1].set_yticklabels(list(rho_table.index) + list(eta_table.index))
    axes[1].set_xlabel("PC index")
    axes[1].set_xticks(np.arange(0, show_pcs, 5))
    axes[1].set_xticklabels(np.arange(1, show_pcs + 1, 5))
    axes[1].set_title("Metadata correlation per PC: |Spearman ρ| (top) + η² (bottom)")
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig(OUT / "q6_pc_biology_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT / 'q6_pc_biology_heatmap.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
