#!/usr/bin/env python3
"""Rebuild the shared GTEx × HCA gene matrix at 11,374 genes.

The preprocess pipeline by default selects the top 2,000 genes by combined
variance rank.  This script re-runs it with a limit high enough to capture
ALL shared genes (report shows ~11,374 overlap between GTEx v11 whole-blood
and HCA BL_standard_design.h5ad), outputting to data/processed_11k/.

Why 11,374?  Only ~30 bulk PCs carry stable signal (report §Q7/Q8), but the
full gene set is needed so the decoder can reconstruct the transcriptome
faithfully and the enrichment analyses cover the whole genome.

Usage:
    cd /path/to/ecs271-project
    python scripts/build_shared_gene_matrix.py

    # To customise the gene count:
    python scripts/build_shared_gene_matrix.py --top-genes 11374
    python scripts/build_shared_gene_matrix.py --top-genes 20000  # all shared
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow both top-level and scripts/ invocation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_data import (  # noqa: E402
    read_hca_genes,
    read_gtex_overlap,
    log_cpm,
    compute_hca_log_variance,
    select_genes,
    write_gene_metadata,
    write_hca_outputs,
    save_text_lines,
)
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild shared gene matrix at 11,374 genes → data/processed_11k/."
    )
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed_11k",
        help="Where to write the new processed files (default: data/processed_11k/).",
    )
    p.add_argument(
        "--top-genes",
        type=int,
        default=11374,
        help="Number of genes to keep (by combined variance rank).  "
             "Set to 20000 to capture all shared genes.  Default: 11374.",
    )
    p.add_argument("--scale-factor", type=float, default=1_000_000.0)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--gtex-file", default="gene_reads_v11_whole_blood.gct.gz")
    p.add_argument("--hca-file", default="BL_standard_design.h5ad")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gtex_path = args.raw_dir / args.gtex_file
    hca_path = args.raw_dir / args.hca_file
    if not gtex_path.exists():
        print(
            f"ERROR: GTEx file not found: {gtex_path}\n"
            "Download with: python scripts/load_data.py --whole-blood",
            file=sys.stderr,
        )
        return 1
    if not hca_path.exists():
        print(
            f"ERROR: HCA file not found: {hca_path}\n"
            "Download with: python scripts/load_data.py --hca-blood-scrna",
            file=sys.stderr,
        )
        return 1

    print(f"Reading HCA genes from {hca_path.name}")
    hca_gene_ids, hca_gene_symbols = read_hca_genes(hca_path)
    hca_gene_bases = np.array(
        [g.split(".", 1)[0] for g in hca_gene_ids], dtype=object
    )

    print(f"Reading GTEx genes overlapping with HCA ({len(hca_gene_bases):,} HCA genes)")
    sample_ids, overlap_ids, overlap_symbols, bulk_counts, bulk_lib = read_gtex_overlap(
        gtex_path, set(hca_gene_bases)
    )
    print(f"  found {len(overlap_ids):,} overlapping genes across {len(sample_ids)} GTEx samples")

    bulk_log = log_cpm(bulk_counts, bulk_lib, args.scale_factor)

    print("Computing per-gene variance for joint gene selection")
    hca_variance = compute_hca_log_variance(
        hca_path, overlap_ids, hca_gene_bases, args.scale_factor, args.chunk_size
    )

    n_select = min(args.top_genes, len(overlap_ids))
    print(f"Selecting top {n_select:,} genes by combined variance rank")
    selected_idx = select_genes(bulk_log, hca_variance, n_select)

    selected_ids = np.array(overlap_ids, dtype=object)[selected_idx]
    selected_symbols = np.array(overlap_symbols, dtype=object)[selected_idx]
    selected_bulk_log = bulk_log[selected_idx].T  # (samples, genes)
    selected_bulk_counts = bulk_counts[selected_idx].T
    selected_hca_cols = np.array(
        [np.where(hca_gene_bases == g)[0][0] for g in selected_ids],
        dtype=np.int32,
    )

    print(f"Saving {len(selected_ids):,} genes to {args.output_dir}")

    np.save(args.output_dir / "bulk_log_cpm.npy", selected_bulk_log.astype(np.float32))
    np.save(args.output_dir / "bulk_counts.npy", selected_bulk_counts.astype(np.float32))
    save_text_lines(args.output_dir / "bulk_sample_ids.txt", sample_ids)
    write_gene_metadata(
        args.output_dir / "gene_metadata.tsv",
        selected_ids,
        selected_symbols,
        np.var(bulk_log[selected_idx], axis=1),
        hca_variance[selected_idx],
    )

    print("Writing HCA pseudobulk outputs")
    write_hca_outputs(
        hca_path,
        args.output_dir,
        selected_hca_cols,
        len(selected_ids),
        args.scale_factor,
        args.chunk_size,
    )

    # Cross-check
    final_bulk = np.load(args.output_dir / "bulk_log_cpm.npy")
    final_pb = np.load(args.output_dir / "hca_pseudobulk_counts_by_donor_celltype.npy")
    print(
        f"\nDone.\n"
        f"  bulk_log_cpm:  {final_bulk.shape}  (samples × genes)\n"
        f"  hca_pseudobulk_celltype: {final_pb.shape}  (pseudobulks × genes)\n"
        f"  output: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
