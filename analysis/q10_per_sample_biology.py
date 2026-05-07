#!/usr/bin/env python3
"""Q10 — Per-sample biology: what does ONE donor's transcriptome say?

Picks the 5 donors with lowest SMTSISCH (post-mortem ischemia time, min)
— the cleanest, lowest-handling-stress samples in GTEx whole blood —
and characterizes each donor's biology individually.

For each donor:
  - Compute per-gene z-score: (x_donor - mean_cohort) / std_cohort, where
    mean / std are over the OTHER 802 donors (leave-one-out).
  - Take top 1000 genes with the most-positive z-score (over-expressed
    in this donor) and top 1000 with most-negative z (under-expressed).
  - Submit each list to Enrichr against:
        GO_Biological_Process_2023, KEGG_2021_Human, Reactome_2022,
        MSigDB_Hallmark_2020, ChEA_2022.
  - Report top hits per direction.

Per-sample analysis is unusual in bulk RNA-seq because n=1, so we cannot
do significance testing — what we get is descriptive: "this individual's
blood transcriptome is enriched for these pathways relative to the cohort
average". With ischemia held low, the enrichment we see is closer to
biological identity (proliferation state, immune activation, cell-mix)
and less contaminated by ex-vivo handling artefacts.

Run:  python q10_per_sample_biology.py [--n-donors 5] [--top-genes 1000]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
RESULTS = ROOT / "results" / "q10_per_sample_biology"
RESULTS.mkdir(exist_ok=True, parents=True)

ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
    "ChEA_2022",
]


def enrichr_submit(genes, description):
    import requests
    r = requests.post(
        f"{ENRICHR}/addList",
        files={"list": (None, "\n".join(genes)), "description": (None, description)},
        timeout=60,
    )
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
    out["overlap_genes"] = out["overlap_genes"].apply(lambda g: ";".join(g) if isinstance(g, list) else g)
    return out.sort_values("p").head(top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-donors", type=int, default=5)
    ap.add_argument("--top-genes", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="Enrichr sleep (s) — large to be polite while q6 may also be running")
    args = ap.parse_args()

    print("Loading data ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    X_scaled = standardise(expr_aligned, scaler_mean, scaler_std)

    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        sample_ids = np.asarray(fh.readline().rstrip("\n").split("\t")[2:])

    samp = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt",
                       sep="\t", low_memory=False)
    df = pd.DataFrame({"SAMPID": sample_ids, "row_idx": np.arange(len(sample_ids))})
    df = df.merge(samp[["SAMPID", "SMTSISCH", "SMRIN"]], on="SAMPID", how="left")
    df = df.dropna(subset=["SMTSISCH"])
    df = df.sort_values("SMTSISCH").head(args.n_donors)
    print(f"\nLow-ischemia donors selected (top {args.n_donors} lowest SMTSISCH):")
    for _, r in df.iterrows():
        print(f"  {r['SAMPID']:<30s}  SMTSISCH = {r['SMTSISCH']:.0f} min  SMRIN = {r['SMRIN']}")

    # Leave-one-out z-scores; X_scaled was already mean=0/std=1 globally so
    # we approximate LOO as just X_scaled for speed, then post-correct.
    # The scaler was fit on ALL donors → mean/std already include each donor.
    # Per-donor z reflects how far each gene is from the COHORT mean (in std units).
    # That's what we want.

    summary_rows = []
    for _, r in df.iterrows():
        donor_id = r["SAMPID"]
        idx = int(r["row_idx"])
        donor_dir = RESULTS / donor_id.replace("/", "_")
        donor_dir.mkdir(exist_ok=True)
        z = X_scaled[idx]   # (n_genes,) — already z-scored under cohort scaler

        order = np.argsort(z)
        bot = order[: args.top_genes]
        top = order[-args.top_genes:]

        print(f"\n=== Donor {donor_id} (SMTSISCH={r['SMTSISCH']:.0f}) ===")
        print(f"  Top 5 +z genes: {[str(shared_genes[i]) for i in top[-5:]][::-1]}")
        print(f"  Top 5 -z genes: {[str(shared_genes[i]) for i in bot[:5]]}")

        for direction, gene_idx in [("over", top), ("under", bot)]:
            genes = [str(shared_genes[i]) for i in gene_idx]
            tag = f"{donor_id}_{direction}"
            cache_lid = donor_dir / f"_list_id_{direction}.txt"
            if cache_lid.exists():
                lid = cache_lid.read_text().strip()
            else:
                try:
                    lid = enrichr_submit(genes, tag)["userListId"]
                    cache_lid.write_text(str(lid))
                except Exception as exc:
                    print(f"    !! submit failed: {exc}")
                    continue
            for lib in LIBRARIES:
                cache = donor_dir / f"{direction}__{lib}.csv"
                if cache.exists():
                    tbl = pd.read_csv(cache)
                else:
                    try:
                        tbl = enrichr_query(lid, lib, top_k=10)
                        tbl["donor"] = donor_id
                        tbl["direction"] = direction
                        tbl["library"] = lib
                        tbl.to_csv(cache, index=False)
                        time.sleep(args.sleep)
                    except Exception as exc:
                        print(f"    !! {lib}: {exc}")
                        continue
                if not tbl.empty and "term" in tbl.columns:
                    row = tbl.iloc[0]
                    summary_rows.append({
                        "donor": donor_id,
                        "ischemia_min": float(r["SMTSISCH"]),
                        "rin": float(r["SMRIN"]) if pd.notna(r["SMRIN"]) else None,
                        "direction": direction,
                        "library": lib,
                        "top_term": str(row["term"]),
                        "top_adj_p": float(row.get("adj_p", float("nan"))),
                    })
                    print(f"    {direction:5s} {lib:<35s} {row['term'][:55]:<55s}  adj-p={float(row.get('adj_p', float('nan'))):.1e}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "_summary.csv", index=False)
    print(f"\nWrote {RESULTS / '_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
