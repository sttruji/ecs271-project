#!/usr/bin/env python3
"""Q5 — Does an AE-3 latent dim recover the same biology that PCs do?

Trains the same 3-layer MLP autoencoder used in q3, then for each of the
*64 latent dims*:

  • computes Spearman ρ with the same 5 metadata variables as q4
    (AGE_mid, DTHHRDY, SMRIN, SMTSISCH, SMRDLGTH)
  • for the top-3 most-active dims (highest variance), runs Enrichr on
    the genes that load most positively / negatively under the *linear
    decoder approximation*: the gradient ∂x̂/∂z_k evaluated at z=0,
    which equals the row of the decoder Jacobian at the latent origin —
    a faithful linear approximation of the dim's effect direction.

Compares to Q4 PC results: the AE-3 should recover analogous biology
(UPR / ribosome / glycolysis / OXPHOS / handling stress) but along
non-orthogonal nonlinear axes.

Run:  python q5_ae3_latent_biology.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib_data import align_to_shared, load_gtex_blood, standardise  # noqa: E402
from lib_model import load_trained  # noqa: E402
from q3_mlp_autoencoder import MLPAutoencoder  # noqa: E402

CHECKPOINT = "/Users/rls/ecs271/data/models/cross_modality_vae.pt"
ANNOT = Path("/Users/rls/ecs271/data/annotations")
OUT = ROOT / "figures"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
TOP_GENES = 200


def sample_to_subject(sample_id: str) -> str:
    return "-".join(sample_id.split("-")[:2])


def age_to_midpoint(age_band):
    if not isinstance(age_band, str) or "-" not in age_band:
        return np.nan
    a, b = age_band.split("-")
    try:
        return (int(a) + int(b)) / 2
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
    cols = ["rank", "term", "p", "z", "combined_score", "overlap_genes", "adj_p", "old_p", "old_adj_p"]
    out = pd.DataFrame(rows, columns=cols)
    out["overlap_genes"] = out["overlap_genes"].apply(lambda g: ";".join(g) if isinstance(g, list) else g)
    return out.sort_values("p").head(top_k)


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    print("Loading shared-genes / scaler ...")
    _, shared_genes, scaler_mean, scaler_std = load_trained(CHECKPOINT, device="cpu")

    print("Loading GTEx ...")
    expr_log, gene_names = load_gtex_blood()
    expr_aligned, _, _ = align_to_shared(expr_log, gene_names, shared_genes)
    expr_scaled = standardise(expr_aligned, scaler_mean, scaler_std)
    n_genes = expr_scaled.shape[1]

    rng = np.random.default_rng(0)
    perm = rng.permutation(expr_scaled.shape[0])
    n_test = int(round(0.2 * expr_scaled.shape[0]))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_t = torch.from_numpy(expr_scaled[train_idx])
    full_t = torch.from_numpy(expr_scaled)

    print("\nTraining AE-3 ...")
    ae = MLPAutoencoder(n_genes=n_genes, latent_dim=64).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(train_t.to(device)), batch_size=64, shuffle=True, drop_last=True)
    for ep in range(1, 201):
        ae.train()
        for (xb,) in loader:
            opt.zero_grad()
            xh, _ = ae(xb)
            loss = F.mse_loss(xh, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            opt.step()
        if ep % 50 == 0:
            ae.eval()
            with torch.no_grad():
                tr = float(F.mse_loss(ae(train_t.to(device))[0], train_t.to(device)))
            print(f"  ep {ep}/200  train MSE={tr:.4f}")

    ae.eval()
    with torch.no_grad():
        _, z_all = ae(full_t.to(device))
    z = z_all.cpu().numpy()  # (803, 64)
    var_per_dim = z.var(0)
    print(f"\nLatent var range: [{var_per_dim.min():.3g}, {var_per_dim.max():.3g}]  median={np.median(var_per_dim):.3g}")

    # ── Decoder-Jacobian linear approximation around z=0 ──────────────────
    # gradient ∂x̂/∂z_k at z=0 is the k-th column of J, equivalent to a one-hot
    # forward-diff. We compute the full Jacobian (n_genes × 64) by passing
    # one-hot z's (with very small perturbation) through the decoder.
    print("Computing decoder Jacobian at z=0 ...")
    eps = 1e-2
    with torch.no_grad():
        z0 = torch.zeros(1, 64, device=device)
        x0 = ae.decoder(z0).cpu().numpy()[0]  # (n_genes,)
        jac = np.zeros((n_genes, 64), dtype=np.float32)
        for k in range(64):
            zk = torch.zeros(1, 64, device=device)
            zk[0, k] = eps
            xk = ae.decoder(zk).cpu().numpy()[0]
            jac[:, k] = (xk - x0) / eps

    # ── Metadata join ─────────────────────────────────────────────────────
    print("Loading GTEx annotations ...")
    sub = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samp = pd.read_csv(ANNOT / "GTEx_v10_Annotations_SampleAttributesDS.txt", sep="\t", low_memory=False)

    import gzip
    with gzip.open("/Users/rls/ecs271/data/bulk/gtex_v11_whole_blood.gct.gz", "rt") as fh:
        for _ in range(2):
            fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    sample_ids = header[2:]

    df = pd.DataFrame({"SAMPID": sample_ids})
    df["SUBJID"] = df["SAMPID"].apply(sample_to_subject)
    df = df.merge(sub, on="SUBJID", how="left")
    df = df.merge(samp[["SAMPID", "SMRIN", "SMTSISCH", "SMRDLGTH"]], on="SAMPID", how="left")
    df["AGE_mid"] = df["AGE"].apply(age_to_midpoint)

    cont_cols = ["AGE_mid", "DTHHRDY", "SMRIN", "SMTSISCH", "SMRDLGTH"]
    rho_table = pd.DataFrame(index=cont_cols, columns=[f"z{k+1}" for k in range(64)], dtype=float)
    for col in cont_cols:
        v = pd.to_numeric(df[col], errors="coerce").values
        for k in range(64):
            x = z[:, k]
            mask = np.isfinite(v) & np.isfinite(x)
            if mask.sum() < 30:
                rho_table.iloc[cont_cols.index(col), k] = np.nan
                continue
            rho, _ = spearmanr(v[mask], x[mask])
            rho_table.iloc[cont_cols.index(col), k] = rho

    print("\nTop |ρ| of any AE-3 latent dim for each metadata column:")
    for col in cont_cols:
        vals = rho_table.loc[col].abs()
        if vals.notna().any():
            best_k = vals.idxmax()
            print(f"  {col:<10}: max |ρ| = {vals.max():.3f} at {best_k}  "
                  f"(corresponding latent variance = {var_per_dim[int(best_k.replace('z','')) - 1]:.4f})")

    rho_table.to_csv(RESULTS / "q5_ae3_latent_metadata_spearman.csv")

    # ── Pick top-3 active dims and run Enrichr on Jacobian columns ────────
    sort_order = np.argsort(var_per_dim)[::-1]
    top_dims = sort_order[:3]
    print(f"\nTop-3 active AE-3 latent dims (by variance): {[int(d)+1 for d in top_dims]} "
          f"(var = {var_per_dim[top_dims]})")

    summary = []
    for dim_idx in top_dims:
        loadings = jac[:, dim_idx]
        order = np.argsort(loadings)
        for direction, idx in [("pos", order[-TOP_GENES:]), ("neg", order[:TOP_GENES])]:
            genes = [str(g) for g in shared_genes[idx]]
            tag = f"AE3_z{dim_idx+1}_{direction}"
            try:
                sub_resp = enrichr_submit(genes, tag)
                lid = sub_resp["userListId"]
                for lib in LIBRARIES:
                    try:
                        tbl = enrichr_query(lid, lib, top_k=10)
                        tbl["dim"] = dim_idx + 1
                        tbl["direction"] = direction
                        tbl["library"] = lib
                        tbl.to_csv(RESULTS / f"q5_enrich_{tag}_{lib}.csv", index=False)
                        if not tbl.empty:
                            row = tbl.iloc[0]
                            summary.append({
                                "dim": int(dim_idx) + 1,
                                "direction": direction,
                                "library": lib,
                                "top_term": row["term"],
                                "top_adj_p": float(row["adj_p"]),
                            })
                            print(f"  {tag} {lib}: {row['term']}  adj-p={float(row['adj_p']):.2e}")
                        time.sleep(0.6)
                    except Exception as exc:
                        print(f"  !! {tag} {lib} failed: {exc}")
            except Exception as exc:
                print(f"  !! submit {tag} failed: {exc}")

    pd.DataFrame(summary).to_csv(RESULTS / "q5_enrichment_summary.csv", index=False)
    print(f"\nWrote {RESULTS / 'q5_enrichment_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
