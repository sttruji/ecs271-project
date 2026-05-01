#!/usr/bin/env python3
"""Download raw datasets for the project."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


DATASETS = {
    "whole_blood": {
        "url": "https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/counts-by-tissue/gene_reads_v11_whole_blood.gct.gz",
        "filename": "gene_reads_v11_whole_blood.gct.gz",
        "description": "GTEx v11 bulk RNA-seq whole-blood gene read counts.",
    },
    "hca_blood_scrna": {
        "url": "https://service.azul.data.humancellatlas.org/repository/files/f272debc-d53d-4ac1-b578-176558c0005c?catalog=dcp59&version=2022-11-28T10%3A43%3A37.354000Z&fileName=BL_standard_design.h5ad",
        "filename": "BL_standard_design.h5ad",
        "description": "Human Cell Atlas single-cell RNA-seq peripheral blood gene expression matrix.",
    },
}


def download_dataset(dataset_name: str, data_dir: Path, force: bool) -> Path:
    dataset = DATASETS[dataset_name]
    output_dir = data_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / dataset["filename"]
    if output_path.exists() and not force:
        print(f"Already exists: {output_path}")
        return output_path

    print(f"Downloading {dataset_name} to {output_path}")
    urllib.request.urlretrieve(dataset["url"], output_path)
    print(f"Done: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Directory where raw datasets are stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download a dataset even if it already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )

    for dataset_name in DATASETS:
        flag = dataset_name.replace("_", "-")
        parser.add_argument(
            f"--{flag}",
            f"--{dataset_name}",
            dest="datasets",
            action="append_const",
            const=dataset_name,
            help=f"Download {dataset_name}.",
        )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for dataset_name, dataset in DATASETS.items():
            print(f"{dataset_name}: {dataset['description']}")
        return 0

    if not args.datasets:
        available = ", ".join(f"--{name.replace('_', '-')}" for name in DATASETS)
        print(f"No dataset selected. Available datasets: {available}", file=sys.stderr)
        return 1

    for dataset_name in args.datasets:
        download_dataset(dataset_name, args.data_dir, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
