#!/usr/bin/env python3
"""Train project models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_ae import train_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train project models.")
    parser.add_argument("--model", required=True, choices=["ae", "vae"])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-iter-no-change", type=int, default=20)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight for --model vae.")
    parser.add_argument("--device", default="auto", help="Torch device for --model vae: auto, cpu, mps, or cuda.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model}"

    if args.model == "ae":
        metrics = train_autoencoder(args)
    elif args.model == "vae":
        from train.train_vae import train_vae

        metrics = train_vae(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
