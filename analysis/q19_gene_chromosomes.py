#!/usr/bin/env python3
"""Q19 — Map the 11,374 expressed (shared) genes to chromosome regions.

Pulls genomic coordinates for every gene symbol used by the trained
cross-modality VAE from MyGene.info (cached on disk after the first run),
then renders a chromosome ideogram with gene-density tracks plus a
per-chromosome count bar chart.

Run:  python analysis/q19_gene_chromosomes.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from pipeline.data import load_shared_genes  # noqa: E402

OUT = ROOT / "figures"
RESULTS = ROOT / "results"
OUT.mkdir(exist_ok=True, parents=True)
RESULTS.mkdir(exist_ok=True, parents=True)
CACHE = RESULTS / "q19_gene_chromosomes.csv"

# GRCh38 chromosome sizes (bp), Ensembl release used by mygene.
CHROM_SIZES = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
    "MT": 16569,
}
CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


def _resolve(mg, queries: list[str], scopes: str) -> list[dict]:
    """Helper: query MyGene with a given scope and return one row per query."""
    out: list[dict] = []
    BATCH = 800
    for i in range(0, len(queries), BATCH):
        chunk = queries[i:i + BATCH]
        res = mg.querymany(
            chunk,
            scopes=scopes,
            fields="symbol,genomic_pos.chr,genomic_pos.start,genomic_pos.end,genomic_pos.strand",
            species="human",
            returnall=False,
        )
        # collapse multi-hit rows: prefer one with a canonical chromosome
        by_q: dict[str, dict] = {}
        for r in res:
            q = r.get("query")
            gp = r.get("genomic_pos")
            if isinstance(gp, list):
                gp = next((x for x in gp if str(x.get("chr", "")) in CHROM_SIZES), gp[0])
            chrom = str(gp.get("chr")) if isinstance(gp, dict) else None
            row = {
                "query": q,
                "symbol": r.get("symbol"),
                "chr": chrom,
                "start": (gp or {}).get("start") if isinstance(gp, dict) else None,
                "end":   (gp or {}).get("end")   if isinstance(gp, dict) else None,
                "strand": (gp or {}).get("strand") if isinstance(gp, dict) else None,
                "notfound": bool(r.get("notfound", False)),
            }
            keep = by_q.get(q)
            if keep is None or (
                not (keep["chr"] in CHROM_SIZES) and (chrom in CHROM_SIZES)
            ):
                by_q[q] = row
        out.extend(by_q.values())
        print(f"  {i + len(chunk)}/{len(queries)}  ({scopes})")
        time.sleep(0.2)
    return out


def fetch_gene_positions(symbols: np.ndarray) -> pd.DataFrame:
    """Resolve gene symbols → (chr, start, end, strand). Cached to CSV.

    Tries the canonical 'symbol' scope first, then retries unresolved
    symbols against alias/prev-symbol/Ensembl-id to recover renames.
    Final cache contains exactly one row per input symbol.
    """
    if CACHE.exists():
        cached = pd.read_csv(CACHE, dtype={"chr": str})
        missing = sorted(set(symbols) - set(cached["query"].astype(str)))
        if not missing:
            return cached

    import mygene
    mg = mygene.MyGeneInfo()

    cached_rows: list[dict] = []
    seen: set[str] = set()
    if CACHE.exists():
        cached = pd.read_csv(CACHE, dtype={"chr": str})
        cached_rows = cached.to_dict("records")
        seen = {str(r["query"]) for r in cached_rows}

    todo = [s for s in symbols if s not in seen]
    print(f"resolving {len(todo)} symbols via MyGene.info ({len(seen)} cached)")
    rows = _resolve(mg, todo, scopes="symbol")

    # Retry symbols that didn't map to a canonical chromosome.
    unresolved = [r["query"] for r in rows if r["chr"] not in CHROM_SIZES]
    if unresolved:
        print(f"retrying {len(unresolved)} unresolved symbols via aliases")
        retry = _resolve(mg, unresolved, scopes="alias,symbol,ensembl.gene,entrezgene")
        retry_by_q = {r["query"]: r for r in retry if r["chr"] in CHROM_SIZES}
        for r in rows:
            if r["query"] in retry_by_q:
                r.update(retry_by_q[r["query"]])

    df = pd.DataFrame(cached_rows + rows).drop_duplicates("query", keep="first")
    df.to_csv(CACHE, index=False)
    return df


def summarise(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_mapped = df["chr"].isin(CHROM_SIZES).sum()
    counts = (
        df[df["chr"].isin(CHROM_SIZES)]
        .groupby("chr").size().reindex(CHROM_ORDER, fill_value=0)
    )
    densities = {
        c: float(counts[c] / (CHROM_SIZES[c] / 1e6))  # genes per Mb
        for c in CHROM_ORDER
    }
    return {
        "n_total_symbols": int(n_total),
        "n_mapped": int(n_mapped),
        "n_unmapped": int(n_total - n_mapped),
        "counts_per_chrom": counts.to_dict(),
        "genes_per_mb": densities,
    }


def plot_ideogram_and_counts(df: pd.DataFrame, summary: dict, out_png: Path) -> None:
    """Two-panel figure:
       (top)    chromosome ideogram with each expressed gene as a tick,
                shaded by local gene density.
       (bottom) gene count + genes-per-Mb bar chart per chromosome.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.35)
    ax_ideo = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    n = len(CHROM_ORDER)
    track_h = 0.55
    bin_mb = 5.0  # 5 Mb bins for density colour
    ymax = n + 0.5

    # density colour normalisation: pool all per-bin densities to choose vmax
    all_bin_counts: list[int] = []
    chrom_bins: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in CHROM_ORDER:
        size_bp = CHROM_SIZES[c]
        edges = np.arange(0, size_bp + bin_mb * 1e6, bin_mb * 1e6)
        sub = df[(df["chr"] == c) & df["start"].notna()]
        starts = sub["start"].astype(float).values
        if starts.size:
            cnts, _ = np.histogram(starts, bins=edges)
        else:
            cnts = np.zeros(len(edges) - 1, dtype=int)
        chrom_bins[c] = (edges, cnts)
        all_bin_counts.extend(cnts.tolist())
    vmax = max(1, int(np.percentile(all_bin_counts, 99))) if all_bin_counts else 1
    cmap = plt.get_cmap("magma")

    for i, c in enumerate(CHROM_ORDER):
        y = n - i  # chr1 at top
        size_mb = CHROM_SIZES[c] / 1e6
        # backbone (rounded rect via FancyBboxPatch)
        bg = mpatches.FancyBboxPatch(
            (0, y - track_h / 2), size_mb, track_h,
            boxstyle="round,pad=0,rounding_size=0.18",
            ec="black", fc="#f4f4f4", lw=0.7,
        )
        ax_ideo.add_patch(bg)
        # density-coloured bin overlays
        edges, cnts = chrom_bins[c]
        for j, k in enumerate(cnts):
            if k == 0:
                continue
            x0 = edges[j] / 1e6
            x1 = edges[j + 1] / 1e6
            shade = cmap(min(k / vmax, 1.0))
            ax_ideo.add_patch(mpatches.Rectangle(
                (x0, y - track_h / 2 + 0.03),
                x1 - x0, track_h - 0.06,
                ec="none", fc=shade, alpha=0.95,
            ))
        # individual gene ticks
        sub = df[(df["chr"] == c) & df["start"].notna()]
        if len(sub):
            xs = sub["start"].astype(float).values / 1e6
            ax_ideo.vlines(xs, y - track_h / 2 + 0.03, y + track_h / 2 - 0.03,
                           color="black", linewidth=0.12, alpha=0.45)
        ax_ideo.text(-3, y, f"chr{c}", ha="right", va="center",
                     fontsize=9, family="monospace")
        ax_ideo.text(size_mb + 1.2, y, f"{summary['counts_per_chrom'][c]:>4d}",
                     ha="left", va="center", fontsize=8, family="monospace",
                     color="#444")

    ax_ideo.set_xlim(-12, max(CHROM_SIZES.values()) / 1e6 + 14)
    ax_ideo.set_ylim(0.4, ymax + 0.4)
    ax_ideo.set_yticks([])
    ax_ideo.set_xlabel("position along chromosome (Mb)")
    ax_ideo.set_title(
        f"Genomic locations of the {summary['n_mapped']:,} expressed "
        f"(VAE shared) genes — coloured by 5 Mb density",
        fontsize=12,
    )
    for sp in ("top", "right", "left"):
        ax_ideo.spines[sp].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_ideo, fraction=0.018, pad=0.01)
    cb.set_label(f"genes / 5 Mb (capped @ {vmax})", fontsize=8)

    # bottom: counts bar + density (genes/Mb) line
    counts = [summary["counts_per_chrom"][c] for c in CHROM_ORDER]
    densities = [summary["genes_per_mb"][c] for c in CHROM_ORDER]
    x = np.arange(len(CHROM_ORDER))
    ax_bar.bar(x, counts, color="#3b6db5", edgecolor="white", label="genes (count)")
    ax_bar.set_ylabel("expressed genes (count)", color="#3b6db5")
    ax_bar.tick_params(axis="y", labelcolor="#3b6db5")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"chr{c}" for c in CHROM_ORDER], rotation=45, ha="right",
                           fontsize=8)

    ax_d = ax_bar.twinx()
    ax_d.plot(x, densities, "o-", color="#d2691e", lw=1.2, ms=4,
              label="density (genes / Mb)")
    ax_d.set_yscale("log")
    ax_d.set_ylabel("density (genes / Mb, log)", color="#d2691e")
    ax_d.tick_params(axis="y", labelcolor="#d2691e")
    # annotate the MT outlier so its log placement is unambiguous
    if "MT" in CHROM_ORDER:
        idx = CHROM_ORDER.index("MT")
        ax_d.annotate(f"{densities[idx]:.0f}",
                      xy=(idx, densities[idx]),
                      xytext=(idx, densities[idx] * 0.55),
                      ha="center", color="#d2691e", fontsize=7)

    for sp in ("top",):
        ax_bar.spines[sp].set_visible(False)
        ax_d.spines[sp].set_visible(False)
    ax_bar.set_title("Per-chromosome expressed-gene count and density")

    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    shared, _, _ = load_shared_genes()
    print(f"shared genes from VAE checkpoint: {shared.size}")

    df = fetch_gene_positions(shared)
    df_mapped = df[df["chr"].isin(CHROM_SIZES)].copy()
    print(f"mapped {len(df_mapped):,}/{len(df):,} symbols to chromosomes")

    summary = summarise(df)
    with open(RESULTS / "q19_gene_chromosomes.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {RESULTS / 'q19_gene_chromosomes.json'}")

    print("\nTop 5 by count:")
    for c, n in sorted(summary["counts_per_chrom"].items(),
                       key=lambda kv: -kv[1])[:5]:
        print(f"  chr{c:>2}: {n:>5} genes  ({summary['genes_per_mb'][c]:.2f} / Mb)")
    print("\nTop 5 by density (genes / Mb):")
    for c, d in sorted(summary["genes_per_mb"].items(),
                       key=lambda kv: -kv[1])[:5]:
        print(f"  chr{c:>2}: {d:.2f} / Mb  ({summary['counts_per_chrom'][c]} genes)")

    plot_ideogram_and_counts(
        df_mapped, summary, OUT / "q19_gene_chromosomes.png",
    )


if __name__ == "__main__":
    main()
