"""
Section 10 — N-cluster UCE pseudobulk token injection into PULSAR MCT
=====================================================================

Instead of ONE global pseudobulk VAE token per donor, inject N tokens —
one per cell-type cluster.  Clusters are assigned globally by K-means on
the 1280-d UCE embeddings.  Each cluster token is the mean UCE of cells in
that cluster for that donor, projected to 768-d via a shared low-rank MLP.

Architecture:
  [CLS] [clust_0] [clust_1] ... [clust_{N-1}] [cell_0 ... cell_255]
  length = 1 + N + 256

This script is exec()-ed inside the running Colab kernel so it has access
to all variables already defined (pulsar, uce_batches, z_bio, y, results, …).

Usage (in a Colab cell):
    exec(open('/content/ncluster_exp.py').read())
"""

print("=" * 65)
print("SECTION 10 — N-cluster UCE pseudobulk token injection")
print("=" * 65)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ── 10a. K-means clustering on UCE embeddings ────────────────────────────────
# uce_batches: (261, 256, 1280) already in scope from section 2

def compute_cluster_means(uce_b, n_clusters, seed=42):
    """
    Global K-means on all 261×256 cells, then compute per-donor cluster means.

    Returns
    -------
    cluster_means : (n_donors, n_clusters, 1280) float32
    cluster_mask  : (n_donors, n_clusters) bool  — True = donor has 0 cells in that cluster
    km_labels     : (n_donors, 256) int  — per-cell cluster assignment
    """
    n_donors, n_cells, uce_dim = uce_b.shape
    uce_flat = uce_b.reshape(-1, uce_dim).astype(np.float32)           # (D*C, 1280)
    norms = np.linalg.norm(uce_flat, axis=1, keepdims=True).clip(1e-8)
    uce_norm = uce_flat / norms                                         # L2-normalise

    print(f"  K-means N={n_clusters} on {uce_flat.shape[0]:,} cells × {uce_dim}-d UCE …")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=5, batch_size=4096, max_iter=200)
    km.fit(uce_norm)
    labels = km.labels_.reshape(n_donors, n_cells)                     # (D, C)

    cluster_means = np.zeros((n_donors, n_clusters, uce_dim), dtype=np.float32)
    cluster_mask  = np.zeros((n_donors, n_clusters), dtype=bool)

    for d in range(n_donors):
        for c in range(n_clusters):
            in_c = (labels[d] == c)
            if in_c.sum() == 0:
                cluster_mask[d, c] = True
            else:
                cluster_means[d, c] = uce_b[d][in_c].mean(0)

    missing = cluster_mask.sum()
    print(f"  Done. cluster_means {cluster_means.shape}  "
          f"missing donor×cluster pairs: {missing}/{n_donors*n_clusters} "
          f"({100*missing/(n_donors*n_clusters):.1f}%)")
    return cluster_means, cluster_mask, labels


# Compute for N=4 and N=8 (reuse K-means; N=4 is a subset experiment)
print("\n── Clustering ──────────────────────────────────────────────────")
clust8_means, clust8_mask, clust8_labels = compute_cluster_means(uce_batches, 8)
clust4_means, clust4_mask, clust4_labels = compute_cluster_means(uce_batches, 4)


# ── 10b. Architecture ─────────────────────────────────────────────────────────

class PULSARWithUCEClusters(nn.Module):
    """
    PULSAR MCT + N per-cluster tokens derived from mean UCE embeddings.

    Shared low-rank projector: UCE (1280) → bottleneck (bn_dim) → hidden (768).
    Each cluster also gets a learnable type embedding so the encoder can
    distinguish cluster 0 (e.g. T cells) from cluster 3 (e.g. monocytes).
    Missing clusters (0 cells in donor) are replaced by a learned null token.
    """
    def __init__(self, pulsar, n_clusters=8, uce_dim=1280, bn_dim=32):
        super().__init__()
        self.pulsar     = pulsar
        self.n_clusters = n_clusters
        h = pulsar.config.hidden_size

        # Shared bottleneck projector  1280 → 32 → 768
        self.cluster_proj = nn.Sequential(
            nn.Linear(uce_dim, bn_dim),
            nn.LayerNorm(bn_dim),
            nn.GELU(),
            nn.Linear(bn_dim, h),
            nn.LayerNorm(h),
        )
        # Cluster identity — so CLS can distinguish cell types
        self.cluster_type_emb = nn.Embedding(n_clusters, h)
        # Learned null token for missing clusters
        self.missing_token = nn.Parameter(torch.zeros(h))

        # Init cluster_type_emb small so it doesn't dominate at start
        nn.init.normal_(self.cluster_type_emb.weight, 0, 0.01)

    def forward(self, cell_emb, cluster_means, cluster_mask=None):
        """
        cell_emb      : (B, 256, 1280)
        cluster_means : (B, N, 1280)  — mean UCE per cluster per donor
        cluster_mask  : (B, N) bool   — True = cluster absent for this donor
        """
        B, N, _ = cluster_means.shape
        cell_h  = self.pulsar.in_proj(cell_emb)                            # (B,256,h)
        cls_tok = self.pulsar.cls_embedding.unsqueeze(0).expand(B,1,-1).clone()

        # Project all cluster means at once
        cm_flat  = cluster_means.reshape(B * N, -1)                        # (B*N,1280)
        tok_flat = self.cluster_proj(cm_flat)                              # (B*N,h)
        clust_h  = tok_flat.reshape(B, N, -1)                             # (B,N,h)

        # Add cluster-identity offset
        ids     = torch.arange(N, device=cell_emb.device)
        clust_h = clust_h + self.cluster_type_emb(ids).unsqueeze(0)

        # Replace absent clusters with the null token
        if cluster_mask is not None:
            mask    = cluster_mask.unsqueeze(-1).float()                   # (B,N,1)
            clust_h = clust_h * (1 - mask) + self.missing_token * mask

        seq = torch.cat([cls_tok, clust_h, cell_h], dim=1)                # (B,1+N+256,h)
        return self.pulsar.encoder(hidden_states=seq)[0][:, 0, :]          # (B,CLS_DIM)


class PULSARUCEClusterClassifier(nn.Module):
    def __init__(self, pulsar, n_clusters=8, cls_dim=512, num_labels=2):
        super().__init__()
        self.model      = PULSARWithUCEClusters(pulsar, n_clusters)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, cls_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cls_dim * 2, num_labels),
        )

    def forward(self, cell_emb, cluster_means, cluster_mask=None, labels=None):
        cls    = self.dropout(self.model(cell_emb, cluster_means, cluster_mask))
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            w    = torch.tensor([162/99, 1.0], device=cls.device)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return loss, logits, cls


print("\n── Architecture defined ✓  (PULSARWithUCEClusters)")
n_single  = sum(p.numel() for p in PULSARUCEClusterClassifier(pulsar, 1).model.cluster_proj.parameters())
n_type8   = sum(p.numel() for p in PULSARUCEClusterClassifier(pulsar, 8).model.cluster_type_emb.parameters())
print(f"   Shared projector params  : {n_single:,}  (same for N=1..16)")
print(f"   Type embeddings N=8      : {n_type8:,}  (trivial)")
print(f"   Sequence length  N=4/N=8 : {1+4+256} / {1+8+256}")


# ── 10c. Training utilities ───────────────────────────────────────────────────

def make_ncluster_dataloaders(uce_b, z_clus, mask, y, fold_tr, fold_te, batch_size=8):
    def ds(idx):
        return TensorDataset(
            torch.from_numpy(uce_b[idx].astype(np.float32)),
            torch.from_numpy(z_clus[idx].astype(np.float32)),
            torch.from_numpy(mask[idx].astype(np.float32)),   # float for masking
            torch.from_numpy(y[idx]),
        )
    return (DataLoader(ds(fold_tr), batch_size, shuffle=True),
            DataLoader(ds(fold_te), batch_size))


def train_one_fold_ncluster(pulsar_base, uce_b, z_clus, mask, y, tr, te,
                             n_clusters, strategy='frozen', epochs=20, lr=2e-4):
    pc    = copy.deepcopy(pulsar_base.cpu()).to(DEVICE)
    pulsar_base.to(DEVICE)
    model = PULSARUCEClusterClassifier(pc, n_clusters=n_clusters).to(DEVICE)

    if strategy == 'frozen':
        for p in pc.parameters(): p.requires_grad = False
        trainable = (list(model.model.cluster_proj.parameters()) +
                     list(model.model.cluster_type_emb.parameters()) +
                     [model.model.missing_token] +
                     list(model.classifier.parameters()))
        opt = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"strategy={strategy!r} not implemented yet")

    sched    = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_dl, te_dl = make_ncluster_dataloaders(uce_b, z_clus, mask, y, tr, te)

    for ep in range(epochs):
        model.train()
        for cells, zc, mk, lab in tr_dl:
            cells = cells.to(DEVICE)
            zc    = zc.to(DEVICE)
            mk    = mk.bool().to(DEVICE)
            lab   = lab.to(DEVICE)
            opt.zero_grad()
            loss, _, _ = model(cells, zc, mk, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval(); preds, labs_all = [], []
    with torch.no_grad():
        for cells, zc, mk, lab in te_dl:
            cells = cells.to(DEVICE)
            zc    = zc.to(DEVICE)
            mk    = mk.bool().to(DEVICE)
            _, logits, _ = model(cells, zc, mk)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs_all.extend(lab.tolist())
    return accuracy_score(labs_all, preds), f1_score(labs_all, preds, average='macro')


def cv_ncluster(pulsar, uce_b, z_clus, mask, y, n_clusters, label,
                strategy='frozen', epochs=20, lr=2e-4, n_splits=5):
    kf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for k, (tr, te) in enumerate(kf.split(uce_b, y)):
        acc, f1 = train_one_fold_ncluster(pulsar, uce_b, z_clus, mask, y,
                                           tr, te, n_clusters, strategy, epochs, lr)
        accs.append(acc); f1s.append(f1)
        print(f"    fold {k+1}: acc={acc:.3f}  f1={f1:.3f}")
    a, sa = np.mean(accs), np.std(accs)
    f, sf = np.mean(f1s), np.std(f1s)
    print(f"  {label:<47}  acc={a:.3f}±{sa:.3f}  f1={f:.3f}±{sf:.3f}")
    return {"label": label, "acc": a, "f1": f, "acc_std": sa, "f1_std": sf}


# ── 10d. Experiments ──────────────────────────────────────────────────────────

ncluster_results = []

print("\n── Experiments ─────────────────────────────────────────────────")
print("Baseline for comparison:  F. PULSAR+VAE full FT (20 ep)  f1=0.947±0.010")
print()

print("=== N=4 cluster UCE tokens (frozen PULSAR, 20 ep) ===")
ncluster_results.append(cv_ncluster(
    pulsar, uce_batches, clust4_means, clust4_mask, y,
    n_clusters=4, label="N4. PULSAR+UCE-clust N=4 frozen",
    strategy='frozen', epochs=20, lr=2e-4,
))

print("\n=== N=8 cluster UCE tokens (frozen PULSAR, 20 ep) ===")
ncluster_results.append(cv_ncluster(
    pulsar, uce_batches, clust8_means, clust8_mask, y,
    n_clusters=8, label="N8. PULSAR+UCE-clust N=8 frozen",
    strategy='frozen', epochs=20, lr=2e-4,
))

# ── 10e. Summary ──────────────────────────────────────────────────────────────

baseline_f1 = 0.948   # A. PULSAR CLS frozen
single_ft   = 0.947   # F. PULSAR+VAE full FT

print("\n" + "=" * 65)
print("N-CLUSTER RESULTS — 5-fold CV (261 donors, T4 GPU)")
print("=" * 65)
print(f"{'Method':<49}  {'F1-macro':>10}")
print("-" * 65)
print(f"  {'A. PULSAR CLS frozen (baseline)':<47}  0.948±0.010")
print(f"  {'F. PULSAR+VAE global token full FT':<47}  0.947±0.010")
for r in ncluster_results:
    print(f"  {r['label']:<47}  {r['f1']:.3f}±{r['f1_std']:.3f}")

best_n = max(ncluster_results, key=lambda r: r['f1'])
print(f"\n  Δ F1 best N-cluster vs PULSAR baseline : {best_n['f1'] - baseline_f1:+.3f}")
print(f"  Δ F1 best N-cluster vs global-token FT : {best_n['f1'] - single_ft:+.3f}")
print("=" * 65)
