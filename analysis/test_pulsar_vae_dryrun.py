"""
Dry-run / shape test for PULSAR+VAE token injection — all three strategies.

MockPULSAR replicates the exact interface:
  pulsar.in_proj                        : Linear(1280 -> 768)
  pulsar.cls_embedding                  : Parameter(768,)
  pulsar.encoder(hidden_states=...)     : returns tuple ([0] = (B, seq, 512))
  pulsar.encoder.encoder.layers         : ModuleList of 12 transformer blocks
  pulsar.config.hidden_size = 768
"""
import copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

DEVICE  = "cpu"
ROOT    = Path(__file__).resolve().parents[1]
Q62     = ROOT / "analysis" / "results" / "q62_pbmc_vae"
HIDDEN  = 768
CLS_OUT = 512
UCE_DIM = 1280
N_CELLS = 256

# ── Mock PULSAR (with nested encoder.encoder.layers) ─────────────────────────

class MockPULSARConfig:
    hidden_size = HIDDEN

class MockBertLayer(nn.Module):
    """Minimal stand-in for a single PULSAR transformer layer."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
    def forward(self, x):
        return x + self.fc(x) * 0.01   # residual-ish, stays in range

class MockBertEncoder(nn.Module):
    """Mimics pulsar.encoder.encoder  (the inner BertEncoder)."""
    def __init__(self, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([MockBertLayer() for _ in range(n_layers)])

class MockEncoder(nn.Module):
    """
    Mimics pulsar.encoder:
      input  : hidden_states (B, seq, 768)
      output : tuple whose [0] is (B, seq, 512)
    Also exposes .encoder.layers so the partial-unfreeze strategy can work.
    """
    def __init__(self, n_layers=12):
        super().__init__()
        self.encoder = MockBertEncoder(n_layers)
        self.down    = nn.Linear(HIDDEN, CLS_OUT)

    def forward(self, hidden_states):
        h = hidden_states
        for layer in self.encoder.layers:
            h = layer(h)
        return (self.down(h),)   # tuple: ([0] = (B, seq, 512))

class MockPULSAR(nn.Module):
    def __init__(self, n_layers=12):
        super().__init__()
        self.config        = MockPULSARConfig()
        self.in_proj       = nn.Linear(UCE_DIM, HIDDEN)
        self.cls_embedding = nn.Parameter(torch.randn(HIDDEN))
        self.encoder       = MockEncoder(n_layers)

    def encode(self, x):
        B   = x.size(0)
        h   = self.in_proj(x)
        cls = self.cls_embedding.unsqueeze(0).expand(B, 1, -1).clone()
        seq = torch.cat([cls, h], dim=1)
        return self.encoder(hidden_states=seq)


# ── Architecture under test ──────────────────────────────────────────────────

CLS_DIM = CLS_OUT  # 512

class PULSARWithVAEToken(nn.Module):
    def __init__(self, pulsar, vae_dim: int = 16):
        super().__init__()
        self.pulsar = pulsar
        self.vae_projector = nn.Sequential(
            nn.Linear(vae_dim, pulsar.config.hidden_size),
            nn.LayerNorm(pulsar.config.hidden_size),
            nn.GELU(),
            nn.Linear(pulsar.config.hidden_size, pulsar.config.hidden_size),
            nn.LayerNorm(pulsar.config.hidden_size),
        )

    def forward(self, cell_emb, z_bio):
        B       = cell_emb.size(0)
        cell_h  = self.pulsar.in_proj(cell_emb)
        cls_tok = self.pulsar.cls_embedding.unsqueeze(0).expand(B, 1, -1).clone()
        vae_tok = self.vae_projector(z_bio).unsqueeze(1)
        seq     = torch.cat([cls_tok, vae_tok, cell_h], dim=1)
        return self.pulsar.encoder(hidden_states=seq)[0][:, 0, :]


class PULSARVAEClassifier(nn.Module):
    def __init__(self, pulsar_vae, num_labels=2, cls_dim=CLS_DIM):
        super().__init__()
        self.model      = pulsar_vae
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, cls_dim*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cls_dim*2, num_labels)
        )

    def forward(self, cell_emb, z_bio, labels=None):
        cls    = self.dropout(self.model(cell_emb, z_bio))
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            w    = torch.tensor([162/99, 1.0], device=cls.device)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return loss, logits, cls


def make_dataloaders(uce_b, z, y, fold_tr, fold_te, batch_size=8):
    def ds(idx): return TensorDataset(
        torch.from_numpy(uce_b[idx].astype(np.float32)),
        torch.from_numpy(z[idx].astype(np.float32)),
        torch.from_numpy(y[idx]))
    return (DataLoader(ds(fold_tr), batch_size, shuffle=True),
            DataLoader(ds(fold_te), batch_size))


def train_one_fold(pulsar_base, uce_b, z, y, tr, te,
                   strategy='frozen', epochs=2, lr=2e-4):
    pulsar_copy = copy.deepcopy(pulsar_base)
    model = PULSARVAEClassifier(PULSARWithVAEToken(pulsar_copy)).to(DEVICE)

    if strategy == 'frozen':
        for p in pulsar_copy.parameters(): p.requires_grad = False
        trainable = (list(model.model.vae_projector.parameters()) +
                     list(model.classifier.parameters()))
        opt = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    elif strategy == 'partial':
        for p in pulsar_copy.parameters(): p.requires_grad = False
        n = len(pulsar_copy.encoder.encoder.layers)
        for layer in pulsar_copy.encoder.encoder.layers[n-4:]:
            for p in layer.parameters(): p.requires_grad = True
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=lr, weight_decay=1e-4)

    elif strategy == 'full':
        opt = optim.AdamW([
            {'params': pulsar_copy.parameters(),               'lr': lr/10},
            {'params': model.model.vae_projector.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(),          'lr': lr},
        ], weight_decay=1e-4)

    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_dl, te_dl = make_dataloaders(uce_b, z, y, tr, te)

    for _ in range(epochs):
        model.train()
        for cells, zb, lab in tr_dl:
            cells, zb, lab = cells.to(DEVICE), zb.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            loss, _, _ = model(cells, zb, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval(); preds, labs_all = [], []
    with torch.no_grad():
        for cells, zb, lab in te_dl:
            _, logits, _ = model(cells.to(DEVICE), zb.to(DEVICE))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs_all.extend(lab.tolist())
    return accuracy_score(labs_all, preds), f1_score(labs_all, preds, average='macro')


# ── Tests ────────────────────────────────────────────────────────────────────

def load_real_data(synthetic_uce=True):
    z_bio = np.load(Q62 / "z_bio.npy").astype(np.float32)
    meta  = pd.read_csv(Q62 / "donor_meta.csv")
    y     = meta["label"].values.astype(np.int64)
    if synthetic_uce:
        rng   = np.random.default_rng(42)
        uce_b = rng.standard_normal((len(y), N_CELLS, UCE_DIM)).astype(np.float32)
    else:
        uce_b = np.load(Q62 / "uce_batches_256.npy")
    return uce_b, z_bio, y


def test_unfreeze_strategies():
    print("=" * 60)
    print("TEST: all three unfreeze strategies (1 fold, 2 epochs, CPU)")
    uce_b, z_bio, y = load_real_data()
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    tr, te = next(iter(kf.split(uce_b, y)))

    results = {}
    for strategy, lr in [('frozen', 2e-4), ('partial', 1e-4), ('full', 5e-5)]:
        pulsar = MockPULSAR()

        # Count trainable params before training
        model = PULSARVAEClassifier(PULSARWithVAEToken(copy.deepcopy(pulsar)))
        if strategy == 'frozen':
            for p in model.model.pulsar.parameters(): p.requires_grad = False
        elif strategy == 'partial':
            for p in model.model.pulsar.parameters(): p.requires_grad = False
            n = len(model.model.pulsar.encoder.encoder.layers)
            for layer in model.model.pulsar.encoder.encoder.layers[n-4:]:
                for p in layer.parameters(): p.requires_grad = True

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())

        acc, f1 = train_one_fold(pulsar, uce_b, z_bio, y, tr, te,
                                  strategy=strategy, epochs=2, lr=lr)
        results[strategy] = (acc, f1, n_trainable, n_total)
        print(f"  {strategy:<8}  trainable={n_trainable:>8,}/{n_total:>8,} "
              f"({100*n_trainable/n_total:.1f}%)   acc={acc:.3f}  f1={f1:.3f}  ✓")

    print()
    frozen_f1  = results['frozen'][1]
    partial_f1 = results['partial'][1]
    full_f1    = results['full'][1]
    print(f"  Δ(partial - frozen) = {partial_f1 - frozen_f1:+.3f}")
    print(f"  Δ(full    - frozen) = {full_f1    - frozen_f1:+.3f}")
    print("  (noise at 1 fold / 2 epochs — real signal only from Colab run)")
    print("PASS\n")


# ── N-cluster pseudobulk architecture ────────────────────────────────────────

class PULSARWithNClusters(nn.Module):
    """
    PULSAR MCT + N per-cluster pseudobulk tokens.

    Instead of one global pseudobulk token, inject one token per cell-type
    cluster.  PULSAR's self-attention can then weight each cluster's bulk
    signal against individual cells and against each other.

    Sequence layout:
      [CLS] [clust_0] [clust_1] ... [clust_{N-1}] [cell_0 ... cell_255]
      total length = 1 + N + 256

    Args
    ----
    pulsar      : PULSAR model
    n_clusters  : number of cell-type clusters (N)
    vae_dim     : z_bio dimensionality (16 by default)
    """
    def __init__(self, pulsar, n_clusters: int = 8, vae_dim: int = 16):
        super().__init__()
        self.pulsar     = pulsar
        self.n_clusters = n_clusters
        h = pulsar.config.hidden_size

        # Shared projector: maps any cluster's z_bio → hidden space
        self.vae_projector = nn.Sequential(
            nn.Linear(vae_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
        )
        # Learnable type embedding so PULSAR knows cluster 0 ≠ cluster 3
        self.cluster_type_emb = nn.Embedding(n_clusters, h)

        # Missing-cluster mask: learned "empty" token used when a donor has
        # 0 cells in a cluster (e.g. a rare cell type absent in that sample)
        self.missing_token = nn.Parameter(torch.zeros(h))

    def forward(self, cell_emb, z_clusters, cluster_mask=None):
        """
        cell_emb     : (B, 256, 1280)   — per-cell UCE embeddings
        z_clusters   : (B, N, 16)       — per-cluster pseudobulk VAE embeddings
        cluster_mask : (B, N) bool       — True where cluster is MISSING for donor
                       (None = all clusters present)

        Returns: (B, CLS_DIM) — same shape as single-token variant
        """
        B, N, _ = z_clusters.shape
        assert N == self.n_clusters, \
            f"z_clusters has {N} clusters but model expects {self.n_clusters}"

        cell_h  = self.pulsar.in_proj(cell_emb)                           # (B,256,h)
        cls_tok = self.pulsar.cls_embedding.unsqueeze(0).expand(B,1,-1).clone()

        # Project all clusters at once (batch over B*N)
        z_flat   = z_clusters.reshape(B * N, -1)                           # (B*N, 16)
        tok_flat = self.vae_projector(z_flat)                              # (B*N, h)
        clust_h  = tok_flat.reshape(B, N, -1)                             # (B, N, h)

        # Add cluster-identity embeddings
        ids      = torch.arange(N, device=cell_emb.device)
        clust_h  = clust_h + self.cluster_type_emb(ids).unsqueeze(0)      # (B, N, h)

        # Replace missing clusters with the learned missing_token
        if cluster_mask is not None:                                        # (B, N)
            mask = cluster_mask.unsqueeze(-1).float()                      # (B, N, 1)
            clust_h = clust_h * (1 - mask) + self.missing_token * mask

        seq = torch.cat([cls_tok, clust_h, cell_h], dim=1)                # (B,1+N+256,h)
        return self.pulsar.encoder(hidden_states=seq)[0][:, 0, :]          # (B, 512)


class PULSARNClusterClassifier(nn.Module):
    """Thin wrapper: N-cluster backbone + MLP head."""
    def __init__(self, pulsar, n_clusters=8, vae_dim=16, cls_dim=CLS_DIM, num_labels=2):
        super().__init__()
        self.model      = PULSARWithNClusters(pulsar, n_clusters, vae_dim)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, cls_dim*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cls_dim*2, num_labels)
        )

    def forward(self, cell_emb, z_clusters, cluster_mask=None, labels=None):
        cls    = self.dropout(self.model(cell_emb, z_clusters, cluster_mask))
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            w    = torch.tensor([162/99, 1.0], device=cls.device)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return loss, logits, cls


def test_ncluster_shapes():
    print("=" * 60)
    print("TEST: N-cluster pseudobulk architecture shape checks")
    pulsar = MockPULSAR()

    for N in [4, 8]:
        B      = 4
        cells  = torch.randn(B, N_CELLS, UCE_DIM)
        z_clus = torch.randn(B, N, 16)

        model  = PULSARNClusterClassifier(pulsar, n_clusters=N)
        loss, logits, _ = model(cells, z_clus, labels=torch.zeros(B, dtype=torch.long))

        assert logits.shape == (B, 2), f"N={N}: expected (4,2) got {logits.shape}"
        seq_len = 1 + N + N_CELLS
        print(f"  N={N}  seq_len={seq_len}  logits={logits.shape}  loss={loss.item():.4f}  ✓")

    # Test with missing-cluster mask
    N      = 8; B = 3
    cells  = torch.randn(B, N_CELLS, UCE_DIM)
    z_clus = torch.randn(B, N, 16)
    mask   = torch.zeros(B, N, dtype=torch.bool)
    mask[0, 7] = True   # donor 0 has no cells in cluster 7
    mask[2, 2] = True   # donor 2 has no cells in cluster 2
    model  = PULSARNClusterClassifier(pulsar, n_clusters=N)
    loss, logits, _ = model(cells, z_clus, mask, labels=torch.zeros(B, dtype=torch.long))
    assert logits.shape == (B, 2)
    print(f"  N=8 with missing-cluster mask  logits={logits.shape}  ✓")
    print("PASS\n")


def test_ncluster_train():
    print("=" * 60)
    print("TEST: N-cluster forward + backward (1 fold, 2 epochs)")
    z_bio   = np.load(Q62 / "z_bio.npy").astype(np.float32)  # (261, 16)
    meta    = pd.read_csv(Q62 / "donor_meta.csv")
    y       = meta["label"].values.astype(np.int64)
    rng     = np.random.default_rng(0)
    uce_b   = rng.standard_normal((len(y), N_CELLS, UCE_DIM)).astype(np.float32)

    # Simulate 8 cluster z_bios: for now, tile and add noise (placeholder for real cluster embeds)
    N_CLUST = 8
    z_clus  = np.stack([z_bio + rng.standard_normal(z_bio.shape).astype(np.float32) * 0.1
                        for _ in range(N_CLUST)], axis=1)   # (261, 8, 16)

    kf = StratifiedKFold(2, shuffle=True, random_state=42)
    tr, te = next(iter(kf.split(uce_b, y)))

    pulsar = MockPULSAR()
    model  = PULSARNClusterClassifier(pulsar, n_clusters=N_CLUST).to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    tr_ds  = TensorDataset(
        torch.from_numpy(uce_b[tr]), torch.from_numpy(z_clus[tr]), torch.from_numpy(y[tr]))
    te_ds  = TensorDataset(
        torch.from_numpy(uce_b[te]), torch.from_numpy(z_clus[te]), torch.from_numpy(y[te]))

    for ep in range(2):
        model.train()
        for cells, zc, lab in DataLoader(tr_ds, 8, shuffle=True):
            opt.zero_grad()
            loss, _, _ = model(cells, zc, labels=lab)
            loss.backward()
            opt.step()

    model.eval(); preds, labs_all = [], []
    with torch.no_grad():
        for cells, zc, lab in DataLoader(te_ds, 8):
            _, logits, _ = model(cells, zc)
            preds.extend(logits.argmax(-1).tolist())
            labs_all.extend(lab.tolist())
    acc = accuracy_score(labs_all, preds)
    f1  = f1_score(labs_all, preds, average='macro')
    print(f"  N={N_CLUST} clusters  acc={acc:.3f}  f1={f1:.3f}  ✓")
    print("PASS\n")


def print_architecture_summary():
    print("=" * 60)
    print("ARCHITECTURE COMPARISON")
    print()
    pulsar = MockPULSAR()

    models = {
        "Single pseudobulk (N=1)": PULSARVAEClassifier(PULSARWithVAEToken(pulsar)),
        "N=4 clusters":  PULSARNClusterClassifier(pulsar, n_clusters=4),
        "N=8 clusters":  PULSARNClusterClassifier(pulsar, n_clusters=8),
        "N=16 clusters": PULSARNClusterClassifier(pulsar, n_clusters=16),
    }

    for name, model in models.items():
        total    = sum(p.numel() for p in model.parameters())
        backbone = sum(p.numel() for p in model.model.pulsar.parameters())
        added    = total - backbone
        # seq len estimate
        if hasattr(model.model, 'n_clusters'):
            seq = 1 + model.model.n_clusters + N_CELLS
        else:
            seq = 1 + 1 + N_CELLS
        print(f"  {name:<28}  seq={seq:<4}  added_params={added:>8,}  total={total:,}")
    print()


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} | device={DEVICE}\n")

    test_unfreeze_strategies()
    test_ncluster_shapes()
    test_ncluster_train()
    print_architecture_summary()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
