#!/usr/bin/env python3
"""Preprocess raw bulk and single-cell RNA-seq datasets."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import h5py
import numpy as np


def strip_gene_version(gene_id: str) -> str:
    return gene_id.split(".", 1)[0]


def decode_array(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.array(decoded, dtype=object)


def read_hca_genes(hca_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hca_path, "r") as h5:
        gene_names = decode_array(h5["var/featurekey"][:])
        gene_ids = decode_array(h5["var/featureid"][:])
    return gene_ids, gene_names


def read_gtex_overlap(
    gtex_path: Path,
    hca_gene_bases: set[str],
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    kept_ids: list[str] = []
    kept_symbols: list[str] = []
    kept_counts: list[np.ndarray] = []

    with gzip.open(gtex_path, "rt") as handle:
        version = handle.readline().strip()
        if version != "#1.2":
            raise ValueError(f"Expected GCT v1.2 file, got {version}")

        handle.readline()
        header = handle.readline().rstrip("\n").split("\t")
        sample_ids = header[2:]
        library_sizes = np.zeros(len(sample_ids), dtype=np.float64)

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene_id = parts[0]
            gene_base = strip_gene_version(gene_id)
            counts = np.array(parts[2:], dtype=np.float32)
            library_sizes += counts

            if gene_base in hca_gene_bases:
                kept_ids.append(gene_base)
                kept_symbols.append(parts[1])
                kept_counts.append(counts)

    if not kept_counts:
        raise ValueError("No overlapping genes found between GTEx and HCA data.")

    counts_by_gene = np.vstack(kept_counts)
    return sample_ids, kept_ids, kept_symbols, counts_by_gene, library_sizes


def log_cpm(counts: np.ndarray, library_sizes: np.ndarray, scale_factor: float) -> np.ndarray:
    safe_library_sizes = np.maximum(library_sizes, 1.0)
    return np.log1p((counts / safe_library_sizes) * scale_factor)


def categorical_values(h5: h5py.File, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = h5[f"obs/{column}"]
    codes = dataset[:]
    categories_ref = dataset.attrs["categories"]
    categories = decode_array(h5[categories_ref][:])
    values = np.array([categories[code] if code >= 0 else "" for code in codes], dtype=object)
    return values, categories, codes


def compute_hca_log_variance(
    hca_path: Path,
    overlap_gene_bases: list[str],
    hca_gene_bases: np.ndarray,
    scale_factor: float,
    chunk_size: int,
) -> np.ndarray:
    overlap_index = {gene_id: i for i, gene_id in enumerate(overlap_gene_bases)}
    hca_to_overlap = np.full(len(hca_gene_bases), -1, dtype=np.int32)
    for hca_idx, gene_id in enumerate(hca_gene_bases):
        if gene_id in overlap_index:
            hca_to_overlap[hca_idx] = overlap_index[gene_id]

    sums = np.zeros(len(overlap_gene_bases), dtype=np.float64)
    sumsq = np.zeros(len(overlap_gene_bases), dtype=np.float64)

    with h5py.File(hca_path, "r") as h5:
        n_cells = int(h5["raw/X"].attrs["shape"][0])
        indptr = h5["raw/X/indptr"]
        indices = h5["raw/X/indices"]
        data = h5["raw/X/data"]
        library_sizes = np.maximum(h5["obs/n_counts"][:].astype(np.float64), 1.0)

        for row_start in range(0, n_cells, chunk_size):
            row_stop = min(row_start + chunk_size, n_cells)
            data_start = int(indptr[row_start])
            data_stop = int(indptr[row_stop])

            row_counts = np.diff(indptr[row_start : row_stop + 1])
            rows = np.repeat(np.arange(row_start, row_stop, dtype=np.int32), row_counts)
            raw_indices = indices[data_start:data_stop]
            raw_values = data[data_start:data_stop].astype(np.float64)

            overlap_cols = hca_to_overlap[raw_indices]
            mask = overlap_cols >= 0
            if not np.any(mask):
                continue

            normalized = np.log1p((raw_values[mask] / library_sizes[rows[mask]]) * scale_factor)
            cols = overlap_cols[mask]
            sums += np.bincount(cols, weights=normalized, minlength=len(overlap_gene_bases))
            sumsq += np.bincount(cols, weights=normalized * normalized, minlength=len(overlap_gene_bases))

    means = sums / n_cells
    return (sumsq / n_cells) - (means * means)


def select_genes(
    bulk_log: np.ndarray,
    hca_variance: np.ndarray,
    top_genes: int,
) -> np.ndarray:
    bulk_variance = np.var(bulk_log, axis=1)
    bulk_rank = np.argsort(np.argsort(-bulk_variance))
    hca_rank = np.argsort(np.argsort(-hca_variance))
    combined_rank = bulk_rank + hca_rank
    selected_count = min(top_genes, len(combined_rank))
    return np.argsort(combined_rank)[:selected_count]


def save_text_lines(path: Path, values: list[str] | np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_gene_metadata(
    path: Path,
    selected_gene_ids: np.ndarray,
    selected_gene_symbols: np.ndarray,
    bulk_variance: np.ndarray,
    hca_variance: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("gene_id\tgene_symbol\tbulk_log_cpm_variance\thca_log_cpm_variance\n")
        for gene_id, symbol, bulk_var, hca_var in zip(
            selected_gene_ids,
            selected_gene_symbols,
            bulk_variance,
            hca_variance,
        ):
            handle.write(f"{gene_id}\t{symbol}\t{bulk_var:.8g}\t{hca_var:.8g}\n")


def write_hca_outputs(
    hca_path: Path,
    output_dir: Path,
    selected_hca_cols: np.ndarray,
    selected_gene_count: int,
    scale_factor: float,
    chunk_size: int,
) -> None:
    hca_to_selected = np.full(int(selected_hca_cols.max()) + 1, -1, dtype=np.int32)
    hca_to_selected[selected_hca_cols] = np.arange(len(selected_hca_cols), dtype=np.int32)

    data_chunks: list[np.ndarray] = []
    indices_chunks: list[np.ndarray] = []
    raw_data_chunks: list[np.ndarray] = []
    indptr_out = [0]

    with h5py.File(hca_path, "r") as h5:
        n_cells = int(h5["raw/X"].attrs["shape"][0])
        indptr = h5["raw/X/indptr"]
        indices = h5["raw/X/indices"]
        data = h5["raw/X/data"]
        library_sizes = np.maximum(h5["obs/n_counts"][:].astype(np.float64), 1.0)

        donors, donor_categories, donor_codes = categorical_values(h5, "Donor")
        channels, _, _ = categorical_values(h5, "Channel")
        annotations, annotation_categories, annotation_codes = categorical_values(h5, "anno")
        barcodes = decode_array(h5["obs/barcodekey"][:])
        n_counts = h5["obs/n_counts"][:]
        n_genes = h5["obs/n_genes"][:]
        percent_mito = h5["obs/percent_mito"][:]

        pseudobulk_by_donor = np.zeros((len(donor_categories), selected_gene_count), dtype=np.float64)
        pair_count = len(donor_categories) * len(annotation_categories)
        pseudobulk_by_donor_celltype = np.zeros((pair_count, selected_gene_count), dtype=np.float64)

        with (output_dir / "hca_cell_metadata.tsv").open("w", encoding="utf-8") as handle:
            handle.write("barcode\tdonor\tchannel\tcell_type\tn_counts\tn_genes\tpercent_mito\n")
            for values in zip(barcodes, donors, channels, annotations, n_counts, n_genes, percent_mito):
                handle.write(
                    f"{values[0]}\t{values[1]}\t{values[2]}\t{values[3]}\t"
                    f"{int(values[4])}\t{int(values[5])}\t{float(values[6]):.6g}\n"
                )

        for row_start in range(0, n_cells, chunk_size):
            row_stop = min(row_start + chunk_size, n_cells)
            data_start = int(indptr[row_start])
            data_stop = int(indptr[row_stop])

            row_counts = np.diff(indptr[row_start : row_stop + 1])
            rows = np.repeat(np.arange(row_start, row_stop, dtype=np.int32), row_counts)
            raw_indices = indices[data_start:data_stop]
            raw_values = data[data_start:data_stop].astype(np.float32)

            selected_cols = np.full(len(raw_indices), -1, dtype=np.int32)
            valid_hca_col = raw_indices < len(hca_to_selected)
            selected_cols[valid_hca_col] = hca_to_selected[raw_indices[valid_hca_col]]
            mask = selected_cols >= 0

            filtered_rows = rows[mask]
            filtered_cols = selected_cols[mask]
            filtered_raw = raw_values[mask]
            filtered_log = np.log1p(
                (filtered_raw.astype(np.float64) / library_sizes[filtered_rows]) * scale_factor
            ).astype(np.float32)

            row_nnz = np.bincount(
                filtered_rows - row_start,
                minlength=row_stop - row_start,
            )
            indptr_out.extend((indptr_out[-1] + np.cumsum(row_nnz)).tolist())

            data_chunks.append(filtered_log)
            indices_chunks.append(filtered_cols.astype(np.int32))
            raw_data_chunks.append(filtered_raw)

            donor_chunk = donor_codes[filtered_rows]
            annotation_chunk = annotation_codes[filtered_rows]
            np.add.at(pseudobulk_by_donor, (donor_chunk, filtered_cols), filtered_raw)
            pair_ids = donor_chunk * len(annotation_categories) + annotation_chunk
            np.add.at(pseudobulk_by_donor_celltype, (pair_ids, filtered_cols), filtered_raw)

    csr_data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)
    csr_indices = np.concatenate(indices_chunks) if indices_chunks else np.array([], dtype=np.int32)
    csr_raw_data = np.concatenate(raw_data_chunks) if raw_data_chunks else np.array([], dtype=np.float32)
    csr_indptr = np.array(indptr_out, dtype=np.int64)

    np.savez_compressed(
        output_dir / "hca_log_cpm_csr.npz",
        data=csr_data,
        indices=csr_indices,
        indptr=csr_indptr,
        shape=np.array([len(csr_indptr) - 1, selected_gene_count], dtype=np.int64),
    )
    np.savez_compressed(
        output_dir / "hca_counts_csr.npz",
        data=csr_raw_data,
        indices=csr_indices,
        indptr=csr_indptr,
        shape=np.array([len(csr_indptr) - 1, selected_gene_count], dtype=np.int64),
    )

    save_text_lines(output_dir / "hca_donors.txt", donor_categories)
    save_text_lines(output_dir / "hca_cell_types.txt", annotation_categories)
    np.save(output_dir / "hca_pseudobulk_counts_by_donor.npy", pseudobulk_by_donor)

    nonzero_pairs = []
    rows = []
    for donor_idx, donor in enumerate(donor_categories):
        for cell_type_idx, cell_type in enumerate(annotation_categories):
            pair_idx = donor_idx * len(annotation_categories) + cell_type_idx
            if pseudobulk_by_donor_celltype[pair_idx].sum() > 0:
                nonzero_pairs.append((donor, cell_type))
                rows.append(pseudobulk_by_donor_celltype[pair_idx])

    np.save(
        output_dir / "hca_pseudobulk_counts_by_donor_celltype.npy",
        np.vstack(rows) if rows else np.zeros((0, selected_gene_count), dtype=np.float64),
    )
    with (output_dir / "hca_pseudobulk_donor_celltype_metadata.tsv").open("w", encoding="utf-8") as handle:
        handle.write("donor\tcell_type\n")
        for donor, cell_type in nonzero_pairs:
            handle.write(f"{donor}\t{cell_type}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GTEx bulk and HCA single-cell RNA-seq data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--scale-factor", type=float, default=1_000_000.0)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--gtex-file", default="gene_reads_v11_whole_blood.gct.gz")
    parser.add_argument("--hca-file", default="BL_standard_design.h5ad")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gtex_path = args.raw_dir / args.gtex_file
    hca_path = args.raw_dir / args.hca_file
    if not gtex_path.exists():
        raise FileNotFoundError(f"Missing GTEx file: {gtex_path}")
    if not hca_path.exists():
        raise FileNotFoundError(f"Missing HCA file: {hca_path}")

    print("Reading HCA genes")
    hca_gene_ids, hca_gene_symbols = read_hca_genes(hca_path)
    hca_gene_bases = np.array([strip_gene_version(gene_id) for gene_id in hca_gene_ids], dtype=object)

    print("Reading GTEx counts and overlapping genes")
    sample_ids, overlap_gene_ids, overlap_symbols, bulk_counts_by_gene, bulk_library_sizes = read_gtex_overlap(
        gtex_path,
        set(hca_gene_bases),
    )
    print(f"Found {len(overlap_gene_ids)} overlapping genes")

    bulk_log_by_gene = log_cpm(bulk_counts_by_gene, bulk_library_sizes, args.scale_factor)
    hca_variance = compute_hca_log_variance(
        hca_path,
        overlap_gene_ids,
        hca_gene_bases,
        args.scale_factor,
        args.chunk_size,
    )
    selected_overlap_indices = select_genes(bulk_log_by_gene, hca_variance, args.top_genes)

    selected_gene_ids = np.array(overlap_gene_ids, dtype=object)[selected_overlap_indices]
    selected_symbols = np.array(overlap_symbols, dtype=object)[selected_overlap_indices]
    selected_bulk_counts = bulk_counts_by_gene[selected_overlap_indices].T
    selected_bulk_log = bulk_log_by_gene[selected_overlap_indices].T
    selected_hca_cols = np.array(
        [np.where(hca_gene_bases == gene_id)[0][0] for gene_id in selected_gene_ids],
        dtype=np.int32,
    )

    print(f"Saving {len(selected_gene_ids)} selected genes")
    np.save(args.output_dir / "bulk_counts.npy", selected_bulk_counts.astype(np.float32))
    np.save(args.output_dir / "bulk_log_cpm.npy", selected_bulk_log.astype(np.float32))
    save_text_lines(args.output_dir / "bulk_sample_ids.txt", sample_ids)
    write_gene_metadata(
        args.output_dir / "gene_metadata.tsv",
        selected_gene_ids,
        selected_symbols,
        np.var(bulk_log_by_gene[selected_overlap_indices], axis=1),
        hca_variance[selected_overlap_indices],
    )

    print("Writing HCA selected sparse matrices and pseudobulk counts")
    write_hca_outputs(
        hca_path,
        args.output_dir,
        selected_hca_cols,
        len(selected_gene_ids),
        args.scale_factor,
        args.chunk_size,
    )

    print(f"Done. Processed data written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
