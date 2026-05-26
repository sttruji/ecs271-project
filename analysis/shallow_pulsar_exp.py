"""
Section 11 — Shallow-PULSAR × N-cluster UCE tokens (P2 ablation)
=================================================================

Hypothesis (P2): the +0.008 F1 gain from N=4 UCE-cluster tokens is
modest because PULSAR is already near-saturated (baseline F1 ≈ 0.948).
If the base model is weaker, the tokens should contribute more.

Method: truncate PULSAR to K transformer layers (K ∈ {2,4,6,8,12}).
For each K run:
  (a) baseline  — frozen shallow-K CLS + linear probe
  (b) +N4 clust — same shallow-K backbone + 4 cluster tokens (frozen)

Key output: Δ F1 = (b) − (a)  as a function of K (and thus baseline F1).

Requires in scope: pulsar, uce_batches, y, clust4_means, clust4_mask,
                   DEVICE  (all set by prior cells / ncluster_exp.py)

exec()-ed inside the running Colab kernel.
"""

print("=" * 65)
print("SECTION 11 — Shallow-PULSAR × N=4 cluster tokens")
print("=" * 65)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# ── 11a. Re-compute N=4 cluster means if not already in scope ────────────────
try:
    _ = clust4_means
    print("  clust4_means already in scope — reusing")
except NameError:
    from sklearn.cluster import MiniBatchKMeans
    print("  Computing N=4 K-means (clust4_means not in scope)…")
    n_donors, n_cells, uce_dim = uce_batches.shape
    uce_flat = uce_batches.reshape(-1, uce_dim).astype(np.float32)
    norms = np.linalg.norm(uce_flat, axis=1, keepdims=True).clip(1e-8)
    km = MiniBatchKMeans(n_clusters=4, random_state=42,
                         n_init=5, batch_size=4096, max_iter=200)
    km.fit(uce_flat / norms)
    labels = km.labels_.reshape(n_donors, n_cells)
    clust4_means = np.zeros((n_donors, 4, uce_dim), dtype=np.float32)
    clust4_mask  = np.zeros((n_donors, 4), dtype=bool)
    for d in range(n_donors):
        for c in range(4):
            in_c = (labels[d] == c)
            if in_c.sum() == 0:
                clust4_mask[d, c] = True
            else:
                clust4_means[d, c] = uce_batches[d][in_c].mean(0)
    print(f"  Done. cluster_means {clust4_means.shape}")


# ── 11b. Architecture ─────────────────────────────────────────────────────────

class ShallowPULSAR(nn.Module):
    """
    Run only the first `n_layers` transformer blocks, then apply the
    existing 768→512 down-projection.  Sequence: [CLS][cell_0...cell_255].
    """
    def __init__(self, pulsar, n_layers):
        super().__init__()
        self.pulsar   = pulsar
        self.n_layers = n_layers
        self.total    = len(pulsar.encoder.encoder.layers)

    def forward(self, cell_emb):
        B = cell_emb.size(0)
        # Project cells to hidden dim
        cell_h  = self.pulsar.in_proj(cell_emb)                        # (B,256,768)
        cls_tok = self.pulsar.cls_embedding.unsqueeze(0).expand(B,1,-1).clone()
        seq     = torch.cat([cls_tok, cell_h], dim=1)                  # (B,257,768)

        # Run only first n_layers
        hidden = seq
        for layer in self.pulsar.encoder.encoder.layers[:self.n_layers]:
            hidden = layer(hidden_states=hidden)[0]

        cls_768 = hidden[:, 0, :]                                      # (B,768)
        return self.pulsar.encoder.down(cls_768)                       # (B,512)


class ShallowPULSARWithClusters(nn.Module):
    """
    Shallow PULSAR (first n_layers) + N cluster tokens prepended.
    Sequence: [CLS][clust_0...clust_{N-1}][cell_0...cell_255].
    """
    def __init__(self, pulsar, n_layers, n_clusters=4,
                 uce_dim=1280, bn_dim=32):
        super().__init__()
        self.pulsar     = pulsar
        self.n_layers   = n_layers
        self.n_clusters = n_clusters
        h = pulsar.config.hidden_size  # 768

        self.cluster_proj = nn.Sequential(
            nn.Linear(uce_dim, bn_dim),
            nn.LayerNorm(bn_dim),
            nn.GELU(),
            nn.Linear(bn_dim, h),
            nn.LayerNorm(h),
        )
        self.cluster_type_emb = nn.Embedding(n_clusters, h)
        self.missing_token    = nn.Parameter(torch.zeros(h))
        nn.init.normal_(self.cluster_type_emb.weight, 0, 0.01)

    def forward(self, cell_emb, cluster_means, cluster_mask=None):
        B, N, _ = cluster_means.shape
        cell_h  = self.pulsar.in_proj(cell_emb)                        # (B,256,768)
        cls_tok = self.pulsar.cls_embedding.unsqueeze(0).expand(B,1,-1).clone()

        # Project cluster means
        cm_flat  = cluster_means.reshape(B*N, -1)
        tok_flat = self.cluster_proj(cm_flat)
        clust_h  = tok_flat.reshape(B, N, -1)
        ids      = torch.arange(N, device=cell_emb.device)
        clust_h  = clust_h + self.cluster_type_emb(ids).unsqueeze(0)
        if cluster_mask is not None:
            mask    = cluster_mask.unsqueeze(-1).float()
            clust_h = clust_h * (1 - mask) + self.missing_token * mask

        seq = torch.cat([cls_tok, clust_h, cell_h], dim=1)            # (B,1+N+256,768)

        hidden = seq
        for layer in self.pulsar.encoder.encoder.layers[:self.n_layers]:
            hidden = layer(hidden_states=hidden)[0]

        cls_768 = hidden[:, 0, :]
        return self.pulsar.encoder.down(cls_768)                       # (B,512)


class ShallowClassifier(nn.Module):
    def __init__(self, backbone, cls_dim=512, num_labels=2):
        super().__init__()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, cls_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cls_dim * 2, num_labels),
        )

    def forward_base(self, *args, **kwargs):
        return self.dropout(self.backbone(*args, **kwargs))

    def classify(self, cls):
        return self.classifier(cls)


# ── 11c. Training helpers ─────────────────────────────────────────────────────

def train_shallow_fold(pulsar_base, uce_b, y, tr, te,
                       mode, n_layers, n_clusters=4,
                       epochs=20, lr=2e-4, batch_size=8):
    """
    mode: 'baseline'  — ShallowPULSAR, no cluster tokens
          'cluster'   — ShallowPULSARWithClusters (N=4)
    """
    pc = copy.deepcopy(pulsar_base.cpu()).to(DEVICE)
    pulsar_base.to(DEVICE)

    if mode == 'baseline':
        backbone = ShallowPULSAR(pc, n_layers)
    else:
        backbone = ShallowPULSARWithClusters(pc, n_layers, n_clusters)

    # Freeze PULSAR backbone; only train new adapter params
    for p in pc.parameters():
        p.requires_grad = False

    if mode == 'cluster':
        trainable = (list(backbone.cluster_proj.parameters()) +
                     list(backbone.cluster_type_emb.parameters()) +
                     [backbone.missing_token])
    else:
        trainable = []  # baseline: head only

    cls_dim = 512
    head = nn.Sequential(
        nn.Linear(cls_dim, cls_dim * 2), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(cls_dim * 2, 2),
    ).to(DEVICE)
    trainable += list(head.parameters())

    backbone = backbone.to(DEVICE)
    opt   = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w     = torch.tensor([162/99, 1.0], device=DEVICE)

    if mode == 'baseline':
        tr_ds = TensorDataset(
            torch.from_numpy(uce_b[tr].astype(np.float32)),
            torch.from_numpy(y[tr]),
        )
        te_ds = TensorDataset(
            torch.from_numpy(uce_b[te].astype(np.float32)),
            torch.from_numpy(y[te]),
        )
    else:
        tr_ds = TensorDataset(
            torch.from_numpy(uce_b[tr].astype(np.float32)),
            torch.from_numpy(clust4_means[tr].astype(np.float32)),
            torch.from_numpy(clust4_mask[tr].astype(np.float32)),
            torch.from_numpy(y[tr]),
        )
        te_ds = TensorDataset(
            torch.from_numpy(uce_b[te].astype(np.float32)),
            torch.from_numpy(clust4_means[te].astype(np.float32)),
            torch.from_numpy(clust4_mask[te].astype(np.float32)),
            torch.from_numpy(y[te]),
        )

    tr_dl = DataLoader(tr_ds, batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size)

    for ep in range(epochs):
        backbone.train(); head.train()
        for batch in tr_dl:
            opt.zero_grad()
            if mode == 'baseline':
                cells, lab = [b.to(DEVICE) for b in batch]
                cls = head(backbone(cells))
            else:
                cells, zc, mk, lab = [b.to(DEVICE) for b in batch]
                cls = head(backbone(cells, zc, mk.bool()))
            loss = nn.CrossEntropyLoss(weight=w)(cls, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        sched.step()

    backbone.eval(); head.eval()
    preds, labs_all = [], []
    with torch.no_grad():
        for batch in te_dl:
            if mode == 'baseline':
                cells, lab = [b.to(DEVICE) for b in batch]
                logits = head(backbone(cells))
            else:
                cells, zc, mk, lab = [b.to(DEVICE) for b in batch]
                logits = head(backbone(cells, zc, mk.bool()))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs_all.extend(lab.tolist())

    return accuracy_score(labs_all, preds), f1_score(labs_all, preds, average='macro')


def cv_shallow(pulsar, uce_b, y, mode, n_layers,
               epochs=20, lr=2e-4, n_splits=5, label=""):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for k, (tr, te) in enumerate(kf.split(uce_b, y)):
        acc, f1 = train_shallow_fold(pulsar, uce_b, y, tr, te,
                                     mode=mode, n_layers=n_layers,
                                     epochs=epochs, lr=lr)
        accs.append(acc); f1s.append(f1)
    a, sa = np.mean(accs), np.std(accs)
    f, sf = np.mean(f1s),  np.std(f1s)
    tag = label or f"K={n_layers} {mode}"
    print(f"  {tag:<52}  acc={a:.3f}±{sa:.3f}  f1={f:.3f}±{sf:.3f}")
    return {"label": tag, "n_layers": n_layers, "mode": mode,
            "acc": a, "f1": f, "acc_std": sa, "f1_std": sf}


# ── 11d. Sweep K ∈ {2, 4, 6, 8, 12} × {baseline, +N4} ─────────────────────────

LAYER_DEPTHS = [2, 4, 6, 8, 12]

print("\n── Sweep: K layers × baseline vs +N=4 cluster tokens ──────────────")
print("  (frozen PULSAR, 20 epochs, 5-fold CV)\n")

results_shallow = []
total_runs = len(LAYER_DEPTHS) * 2
run = 0
for K in LAYER_DEPTHS:
    run += 1
    r_base = cv_shallow(pulsar, uce_batches, y,
                        mode='baseline', n_layers=K,
                        label=f"K={K:2d}  baseline           ",
                        epochs=20)
    results_shallow.append(r_base)

    run += 1
    r_clust = cv_shallow(pulsar, uce_batches, y,
                         mode='cluster', n_layers=K,
                         label=f"K={K:2d}  +N=4 cluster tokens",
                         epochs=20)
    results_shallow.append(r_clust)
    print()

# ── 11e. Summary table ────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("SHALLOW-PULSAR × N=4 CLUSTER TOKENS — F1 summary")
print("=" * 72)
print(f"{'K':>4}  {'Mode':<22}  {'F1-macro':>12}  {'Δ vs baseline':>14}")
print("-" * 72)

base_full = next((r for r in results_shallow
                  if r['n_layers'] == 12 and r['mode'] == 'baseline'), None)

for K in LAYER_DEPTHS:
    rb = next(r for r in results_shallow
              if r['n_layers'] == K and r['mode'] == 'baseline')
    rc = next(r for r in results_shallow
              if r['n_layers'] == K and r['mode'] == 'cluster')

    delta = rc['f1'] - rb['f1']
    sign  = '+' if delta >= 0 else ''
    print(f"{K:>4}  {'baseline':<22}  {rb['f1']:.3f}±{rb['f1_std']:.3f}       {'—':>8}")
    print(f"{K:>4}  {'+N=4 clusters':<22}  {rc['f1']:.3f}±{rc['f1_std']:.3f}  {sign}{delta:+.3f}")
    print()

print("=" * 72)

# Key insight table
print("\n── Key metric: Δ F1 (cluster − baseline) vs base model quality ──")
print(f"  {'K layers':>9}  {'Baseline F1':>12}  {'Δ F1':>8}  Interpretation")
print("  " + "-" * 58)
for K in LAYER_DEPTHS:
    rb = next(r for r in results_shallow if r['n_layers'] == K and r['mode'] == 'baseline')
    rc = next(r for r in results_shallow if r['n_layers'] == K and r['mode'] == 'cluster')
    delta = rc['f1'] - rb['f1']
    sign  = '+' if delta >= 0 else ''
    note  = "↑ cluster tokens help more" if delta > 0.010 else (
            "↔ marginal"                  if abs(delta) <= 0.010 else
            "↓ cluster tokens hurt")
    print(f"  {K:>9}  {rb['f1']:.3f}±{rb['f1_std']:.3f}  {sign}{delta:+.3f}     {note}")

print()
best_delta_r = max(
    [(rc, next(r for r in results_shallow if r['n_layers'] == rc['n_layers'] and r['mode'] == 'baseline'))
     for rc in results_shallow if rc['mode'] == 'cluster'],
    key=lambda x: x[0]['f1'] - x[1]['f1']
)
rc_best, rb_best = best_delta_r
delta_best = rc_best['f1'] - rb_best['f1']
print(f"  Max Δ F1 = {delta_best:+.3f}  at K={rc_best['n_layers']} layers "
      f"(baseline F1={rb_best['f1']:.3f})")
print(f"  Full model (K=12) Δ F1 = "
      f"{next(r for r in results_shallow if r['n_layers']==12 and r['mode']=='cluster')['f1'] - next(r for r in results_shallow if r['n_layers']==12 and r['mode']=='baseline')['f1']:+.3f}")
