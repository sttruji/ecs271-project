#!/usr/bin/env python3
"""Evaluate trained project models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ae_model import SklearnAutoEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate project models.")
    parser.add_argument("--model", required=True, choices=["ae"])
    parser.add_argument("--checkpoint", default="outputs/ae/ae_model.pkl")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/ae")
    parser.add_argument("--scale-factor", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.model == "ae":
        metrics = evaluate_autoencoder(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(json.dumps(metrics, indent=2))
    return 0


def evaluate_autoencoder(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SklearnAutoEncoder.load(Path(args.checkpoint))

    pseudobulk_counts = np.load(data_dir / "hca_pseudobulk_counts_by_donor.npy").astype(np.float32)
    donors = read_lines(data_dir / "hca_donors.txt")
    library_sizes = hca_pseudobulk_library_sizes(data_dir / "hca_cell_metadata.tsv", donors)
    pseudobulk_log_cpm = log_cpm(pseudobulk_counts, library_sizes, args.scale_factor)
    reconstruction = model.reconstruct(pseudobulk_log_cpm)

    per_sample_mse = np.mean((pseudobulk_log_cpm - reconstruction) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(pseudobulk_log_cpm - reconstruction), axis=1)

    metrics = {
        "model": "ae",
        "checkpoint": str(args.checkpoint),
        "evaluation_data": "hca_pseudobulk_counts_by_donor",
        "n_samples": int(pseudobulk_log_cpm.shape[0]),
        "n_genes": int(pseudobulk_log_cpm.shape[1]),
        "normalization": "log1p(selected_gene_counts / donor_total_counts_all_genes * scale_factor)",
        "mse": float(np.mean(per_sample_mse)),
        "mae": float(np.mean(per_sample_mae)),
        "per_sample_mse_min": float(np.min(per_sample_mse)),
        "per_sample_mse_max": float(np.max(per_sample_mse)),
    }

    with (output_dir / "ae_hca_pseudobulk_eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    np.save(output_dir / "ae_hca_pseudobulk_reconstruction.npy", reconstruction.astype(np.float32))
    with (output_dir / "ae_hca_pseudobulk_eval_by_donor.tsv").open("w", encoding="utf-8") as handle:
        handle.write("donor\tmse\tmae\n")
        for donor, mse, mae in zip(donors, per_sample_mse, per_sample_mae):
            handle.write(f"{donor}\t{float(mse):.8g}\t{float(mae):.8g}\n")

    return metrics


def log_cpm(counts: np.ndarray, library_sizes: np.ndarray, scale_factor: float) -> np.ndarray:
    library_sizes = np.maximum(library_sizes.reshape(-1, 1), 1.0)
    return np.log1p((counts / library_sizes) * scale_factor).astype(np.float32)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hca_pseudobulk_library_sizes(metadata_path: Path, donors: list[str]) -> np.ndarray:
    donor_to_index = {donor: index for index, donor in enumerate(donors)}
    library_sizes = np.zeros(len(donors), dtype=np.float64)

    with metadata_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        donor_idx = header.index("donor")
        counts_idx = header.index("n_counts")

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            donor = parts[donor_idx]
            if donor in donor_to_index:
                library_sizes[donor_to_index[donor]] += float(parts[counts_idx])

    if np.any(library_sizes <= 0):
        missing = [donor for donor, value in zip(donors, library_sizes) if value <= 0]
        raise ValueError(f"Missing library sizes for HCA donors: {missing}")

    return library_sizes


if __name__ == "__main__":
    raise SystemExit(main())
