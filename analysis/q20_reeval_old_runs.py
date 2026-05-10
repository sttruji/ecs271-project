"""Re-evaluate the saved Q20 checkpoints with the round-trip cycle metric."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.disentangled_vae import DisentangledVAE, DisentangledConfig
from analysis.q20_disentangled_vae import build_data, flip_test, DEVICE, OUT, SEED  # noqa: E402

CHECKPOINTS = [
    "q20_run1.pt",
    "q20_run2_balanced_leak1.pt",
    "q20_run3_balanced_leak03.pt",
    "q20_run4_modality_only_hsic.pt",
    "q20_run5_cycle1.pt",
    "q20_run6_cycle_bio_meta.pt",
    "q20_run7_cycle05.pt",
    "q20_run8_strong_bio_cycle.pt",
    "q20_run9_bio07_meta03.pt",
    "q20_run10_celltype_pb.pt",
    "q20_run11_donor_plus_ct.pt",
]

print(f"loading data, device={DEVICE}")
data = build_data()
n_gtex = data["x_gtex"].shape[0]
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_gtex)
n_val = int(0.2 * n_gtex)
val_idx = perm[:n_val]
print(f"  n_val_gtex={len(val_idx)}, n_hca={data['x_hca'].shape[0]}")

results = []
for name in CHECKPOINTS:
    p = OUT / name
    if not p.exists():
        print(f"  SKIP missing: {name}")
        continue
    ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
    cfg = DisentangledConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DisentangledVAE(cfg).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"\n=== {name} ===")
    res = flip_test(model, data, val_idx)
    res["name"] = name
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif k == "name":
            pass
        else:
            print(f"  {k}: {v}")
    results.append(res)

with (OUT / "q20_reeval_summary.json").open("w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved → {OUT/'q20_reeval_summary.json'}")
