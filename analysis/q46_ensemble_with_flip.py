"""Q46 — Ensemble search with vae_flip MANDATORY (proposal-aligned).

Loads Q42's exhaustive subset results and filters to subsets containing
vae_flip. Reports the best ensembles that respect the proposal's
metadata-flip cornerstone.

Result: top-1 = 0.667 with 5-method ensemble
        (pca50 + ftest_int_pca50 + cca_5 + vae_flip + vae_latent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "results" / "q20_disentangled"


def main():
    d = json.load(open(OUT_DIR / "q42_subset_ablation.json"))
    subsets = d["top_subsets"]
    with_flip = [s for s in subsets if "vae_flip" in s["methods"]]
    with_flip.sort(key=lambda x: -x["top1"])
    print("=== Q46: ensembles with vae_flip MANDATORY (top 20 by top-1) ===")
    print(f'{"top-1":>6} {"top-3":>6} {"rank":>6}  {"has_latent":>10}  k  methods')
    print("-" * 100)
    for s in with_flip[:20]:
        has_lat = "YES" if "vae_latent" in s["methods"] else "   "
        print(f'{s["top1"]:>6.3f} {s["top3"]:>6.3f} {s["rank1"]:>6.2f}  {has_lat:>10}  {s["n_methods"]}  '
              f'{",".join(m for m in s["methods"])}')

    both = [s for s in with_flip if "vae_latent" in s["methods"]]
    both.sort(key=lambda x: -x["top1"])
    print("\n=== Subsets with BOTH vae_flip AND vae_latent ===")
    for s in both[:10]:
        print(f'top-1={s["top1"]:.3f}  top-3={s["top3"]:.3f}  rank={s["rank1"]:.2f}  k={s["n_methods"]}  '
              f'{",".join(m for m in s["methods"])}')

    out = {
        "best_with_flip_mandatory": with_flip[0],
        "best_with_both_flip_and_latent": both[0] if both else None,
        "all_with_flip": with_flip,
    }
    with (OUT_DIR / "q46_ensemble_with_flip.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved → {OUT_DIR / 'q46_ensemble_with_flip.json'}")


if __name__ == "__main__":
    main()
