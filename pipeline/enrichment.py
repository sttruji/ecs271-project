"""Enrichr REST wrapper for pathway / GO / KEGG enrichment of dim loadings."""
from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

DEFAULT_LIBRARIES = (
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
)


def enrichr_submit(genes: Sequence[str], description: str = "vae_health") -> dict:
    import requests
    payload = {"list": (None, "\n".join(genes)), "description": (None, description)}
    r = requests.post(f"{ENRICHR_URL}/addList", files=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def enrichr_query(user_list_id: str, library: str, top_k: int = 15) -> pd.DataFrame:
    import requests
    r = requests.get(
        f"{ENRICHR_URL}/enrich",
        params={"userListId": user_list_id, "backgroundType": library},
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


def enrich_dim_loadings(
    loadings: np.ndarray,
    gene_names: np.ndarray,
    *,
    top_n: int = 200,
    libraries: Iterable[str] = DEFAULT_LIBRARIES,
    description: str = "loadings",
    pause_s: float = 0.6,
) -> dict[str, dict[str, pd.DataFrame]]:
    """For a single (n_genes,) loading vector, run Enrichr on the top-n
    positive- and top-n negative-loading genes against each library.

    Returns:
        {"pos": {library: DataFrame}, "neg": {library: DataFrame}}
    """
    order = np.argsort(loadings)
    bot_idx = order[:top_n]
    top_idx = order[-top_n:]
    out: dict[str, dict[str, pd.DataFrame]] = {"pos": {}, "neg": {}}
    for direction, idx in (("pos", top_idx), ("neg", bot_idx)):
        genes = [str(g) for g in gene_names[idx]]
        try:
            sub = enrichr_submit(genes, f"{description}_{direction}")
            uid = sub["userListId"]
        except Exception as exc:
            out[direction]["_error"] = str(exc)
            continue
        for lib in libraries:
            try:
                tbl = enrichr_query(uid, lib, top_k=15)
                tbl["library"] = lib
                tbl["direction"] = direction
                out[direction][lib] = tbl
                time.sleep(pause_s)
            except Exception as exc:
                out[direction][lib] = pd.DataFrame({"_error": [str(exc)]})
    return out
