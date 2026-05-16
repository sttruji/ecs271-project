"""Interactive VAE Health Explorer — play with the trained model.

Run with:
    source bulk-project/venv/bin/activate
    cd vae_health
    python -m analysis.docs.vae_explorer

Then open the URL printed in the terminal (usually http://127.0.0.1:7860).

Three tabs:
  1. Forward       — set z_bio + metadata sliders, see predicted gene expression
  2. Reverse       — pick a real donor, see their encoded z_bio + reconstruction
  3. Counterfactual — pick a donor, change one metadata variable, see the gene-by-gene effect
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.data import load_gtex_blood, load_metadata, load_shared_genes
from models.meta_injection_vae import FiLMMetaInjectionVAE, FiLMMetaInjectionConfig

CKPT_DATA = "/Users/rls/Desktop/programming-projects/single-cell/bulk-project/analysis/14_cross_modality_vae/cross_modality_vae.pt"
MODEL_CKPT = ROOT / "analysis" / "results" / "q54b_count_weighted" / "film_resid_vae_weighted.pt"
DEVICE = "cpu"   # forced CPU for the demo — Gradio loop is cleaner on CPU
TOP_N_GENES = 30


# ──────────────────────────────────────────────────────────────────────────
# Load model + data ONCE at startup
# ──────────────────────────────────────────────────────────────────────────

print("Loading GTEx blood ...")
GTEX        = load_gtex_blood(checkpoint_path=CKPT_DATA)
META        = load_metadata(GTEX.sample_ids)
_, SCALER_MEAN, SCALER_STD = load_shared_genes(CKPT_DATA)
GENE_NAMES  = list(GTEX.shared_genes)
N_GENES     = len(GENE_NAMES)

print(f"  Genes: {N_GENES}   Donors: {len(GTEX.sample_ids)}")

print("Loading Q54b model ...")
_ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
_cfg  = _ckpt["cfg"]
_mc   = FiLMMetaInjectionConfig(
    input_dim      = _cfg["input_dim"],
    meta_dim       = _cfg["meta_dim"],
    z_bio_dim      = _cfg["z_bio_dim"],
    decoder_hidden = tuple(_cfg["decoder_hidden"]),
    meta_embed_dim = _cfg["meta_embed_dim"],
    beta           = _cfg["beta"],
    free_bits      = _cfg["free_bits"],
    lambda_tc      = _cfg["lambda_tc"],
)
MODEL = FiLMMetaInjectionVAE(_mc).to(DEVICE).eval()
MODEL.load_state_dict(_ckpt["state_dict"])

BETA_ISCH = np.array(_ckpt["residualizer"]["beta_isch"], dtype=np.float32)
S_MEAN    = float(_ckpt["residualizer"]["s_mean"])
K         = MODEL.n_latent
print(f"  Model: K={K}  decoder_hidden={_mc.decoder_hidden}")

# Pre-compute z_bio for every donor (used in Reverse / Counterfactual tabs)
print("Pre-computing z_bio for all donors ...")
def _build_meta_matrix(df):
    def _z(s):
        v = pd.to_numeric(s, errors="coerce").values.astype(float)
        return np.where(np.isfinite(v), (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8), 0.0).astype(np.float32)
    isch = _z(df["SMTSISCH"]); dthh = _z(df["DTHHRDY"]); age = _z(df["AGE_mid"])
    sex  = pd.to_numeric(df["SEX"], errors="coerce").values
    sex  = np.where(np.isfinite(sex), (sex - 1).astype(np.float32), 0.5)
    return np.column_stack([isch, dthh, age, sex]).astype(np.float32)

M_ALL = _build_meta_matrix(META)
X_RESID_ALL = (GTEX.expr_scaled - np.outer(M_ALL[:, 0] - S_MEAN, BETA_ISCH)).astype(np.float32)
with torch.no_grad():
    _xt = torch.from_numpy(X_RESID_ALL).to(DEVICE)
    Z_ALL = MODEL.encode(_xt)[0].cpu().numpy()
print(f"  z_bio shape: {Z_ALL.shape}")

# Top-variable genes across all reconstructions (for default gene set)
_TOP_VAR_GENE_IDX = np.argsort(GTEX.expr_aligned.var(0))[::-1][:200]
TOP_GENES_DEFAULT = [GENE_NAMES[i] for i in _TOP_VAR_GENE_IDX[:TOP_N_GENES]]

# Per-dim gene loadings (Jacobian at z=0 with mean meta)
META_REF = M_ALL.mean(0).astype(np.float32)
print("Computing gene loadings (Jacobian at z=0) ...")
LOADINGS = MODEL.gene_loadings(META_REF)   # (K, n_genes), in standardised space
print(f"  Loadings shape: {LOADINGS.shape}")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _decode(z_vec: np.ndarray, meta_vec: np.ndarray) -> np.ndarray:
    """Decode (z, meta) → log-CPM expression."""
    with torch.no_grad():
        z_t = torch.from_numpy(z_vec.astype(np.float32).reshape(1, -1)).to(DEVICE)
        m_t = torch.from_numpy(meta_vec.astype(np.float32).reshape(1, -1)).to(DEVICE)
        xh_sc = MODEL.decode(z_t, m_t).cpu().numpy().flatten()
    return xh_sc * SCALER_STD + SCALER_MEAN


def _encode(x_lc: np.ndarray, isch_z: float) -> np.ndarray:
    """Standardise + residualise + encode log-CPM expression → z_bio."""
    x_sc = (x_lc - SCALER_MEAN) / SCALER_STD
    x_resid = x_sc - (isch_z - S_MEAN) * BETA_ISCH
    with torch.no_grad():
        t = torch.from_numpy(x_resid.astype(np.float32).reshape(1, -1)).to(DEVICE)
        mu = MODEL.encode(t)[0].cpu().numpy().flatten()
    return mu


def _top_movers(x_baseline: np.ndarray, x_changed: np.ndarray,
                top_n: int = 30) -> pd.DataFrame:
    """Return top N genes by |Δexpression| with their baseline/changed/diff."""
    delta = x_changed - x_baseline
    idx = np.argsort(np.abs(delta))[::-1][:top_n]
    return pd.DataFrame({
        "gene":      [GENE_NAMES[i] for i in idx],
        "baseline":  x_baseline[idx].round(3),
        "changed":   x_changed[idx].round(3),
        "Δ log₂CPM": delta[idx].round(3),
        "fold":      (2 ** delta[idx]).round(3),
    })


def _bar_plot_expression(x_logcpm: np.ndarray, gene_names_subset: list,
                         title: str = "Predicted expression") -> go.Figure:
    """Plot a bar chart of expression for a subset of genes."""
    idx = [GENE_NAMES.index(g) for g in gene_names_subset if g in GENE_NAMES]
    vals = x_logcpm[idx]
    labels = [GENE_NAMES[i] for i in idx]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color="steelblue"))
    fig.update_layout(
        title=title,
        xaxis_title="gene", yaxis_title="log₂(CPM+1)",
        xaxis_tickangle=-45, height=400,
        margin=dict(l=40, r=40, t=60, b=120),
    )
    return fig


def _bar_plot_difference(delta_logcpm: np.ndarray, gene_names_subset: list,
                         title: str = "Expression change") -> go.Figure:
    """Plot Δ-expression bar chart (positive = up, negative = down)."""
    idx = [GENE_NAMES.index(g) for g in gene_names_subset if g in GENE_NAMES]
    vals = delta_logcpm[idx]
    labels = [GENE_NAMES[i] for i in idx]
    colours = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colours))
    fig.update_layout(
        title=title,
        xaxis_title="gene", yaxis_title="Δ log₂(CPM+1)",
        xaxis_tickangle=-45, height=400,
        margin=dict(l=40, r=40, t=60, b=120),
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Tab 1 — Forward (z + meta → expression)
# ──────────────────────────────────────────────────────────────────────────

def forward_predict(*args):
    """args = (z_1..z_16, meta_isch, meta_hardy, meta_age, meta_sex, gene_text)"""
    z = np.array(args[:K], dtype=np.float32)
    meta = np.array(args[K:K+4], dtype=np.float32)
    gene_text = args[K+4]

    x_lc = _decode(z, meta)

    if gene_text.strip():
        requested = [g.strip().upper() for g in gene_text.split(",") if g.strip()]
        # Match against gene names case-insensitively
        upper_map = {g.upper(): g for g in GENE_NAMES}
        genes = [upper_map[g] for g in requested if g in upper_map]
        not_found = [g for g in requested if g not in upper_map]
        info = f"Showing {len(genes)} requested genes" + (f" ({len(not_found)} not found: {not_found})" if not_found else "")
    else:
        # Default: top-N most variable
        genes = TOP_GENES_DEFAULT
        info = f"Showing top {TOP_N_GENES} most-variable genes"

    fig = _bar_plot_expression(x_lc, genes, "Predicted expression for current settings")

    # Per-gene table
    idx = [GENE_NAMES.index(g) for g in genes]
    table = pd.DataFrame({
        "gene": genes,
        "log₂(CPM+1)": x_lc[idx].round(3),
        "approx CPM":  (2 ** x_lc[idx] - 1).round(0).astype(int),
    })

    return fig, table, info


# ──────────────────────────────────────────────────────────────────────────
# Tab 2 — Reverse (donor → z_bio)
# ──────────────────────────────────────────────────────────────────────────

DONOR_IDS = [str(s) for s in GTEX.sample_ids]


def reverse_inspect(donor_id, gene_text):
    """Show the encoded z_bio for a chosen donor + reconstruction overlay."""
    idx = DONOR_IDS.index(donor_id)
    meta_row = META.iloc[idx]
    z_actual = Z_ALL[idx]
    meta_actual = M_ALL[idx]

    # Decode with original metadata → reconstruction
    x_recon = _decode(z_actual, meta_actual)
    x_true_lc = GTEX.expr_aligned[idx]

    # z_bio bar plot
    fig_z = go.Figure(go.Bar(
        x=[f"z{i+1}" for i in range(K)],
        y=z_actual,
        marker_color=["steelblue" if v >= 0 else "indianred" for v in z_actual],
    ))
    fig_z.update_layout(
        title=f"z_bio for donor {donor_id}",
        xaxis_title="latent dim", yaxis_title="value",
        height=350, margin=dict(l=40, r=40, t=60, b=60),
    )
    fig_z.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

    # Reconstruction vs true expression scatter
    if gene_text.strip():
        upper_map = {g.upper(): g for g in GENE_NAMES}
        genes = [upper_map[g.strip().upper()] for g in gene_text.split(",")
                 if g.strip().upper() in upper_map]
    else:
        genes = TOP_GENES_DEFAULT[:TOP_N_GENES]

    gidx = [GENE_NAMES.index(g) for g in genes]
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(name="true",  x=genes, y=x_true_lc[gidx],
                           marker_color="steelblue", opacity=0.7))
    fig_r.add_trace(go.Bar(name="recon", x=genes, y=x_recon[gidx],
                           marker_color="orange", opacity=0.7))
    fig_r.update_layout(
        barmode="group",
        title="True vs reconstructed expression",
        xaxis_title="gene", yaxis_title="log₂(CPM+1)",
        xaxis_tickangle=-45, height=400, margin=dict(l=40, r=40, t=60, b=120),
    )

    # Metadata summary
    meta_summary = pd.DataFrame({
        "field": ["SAMPID", "SUBJID", "SEX (1=M,2=F)", "AGE", "AGE_mid",
                  "DTHHRDY", "SMRIN", "SMTSISCH (min)", "SMCENTER", "SMRDLGTH"],
        "value": [str(meta_row.get(c, "?")) for c in
                  ["SAMPID","SUBJID","SEX","AGE","AGE_mid","DTHHRDY","SMRIN",
                   "SMTSISCH","SMCENTER","SMRDLGTH"]],
    })

    # Z table (for copying)
    z_table = pd.DataFrame({
        "dim":   [f"z{i+1}" for i in range(K)],
        "value": z_actual.round(4),
    })

    return fig_z, fig_r, meta_summary, z_table


# ──────────────────────────────────────────────────────────────────────────
# Tab 3 — Counterfactual (flip a metadata field for one donor)
# ──────────────────────────────────────────────────────────────────────────

def counterfactual_predict(donor_id, meta_isch, meta_hardy, meta_age, meta_sex,
                           gene_text):
    """Take donor's z_bio, decode with original meta vs user-set meta, show Δ."""
    idx = DONOR_IDS.index(donor_id)
    z_actual = Z_ALL[idx]
    meta_actual = M_ALL[idx]
    meta_new = np.array([meta_isch, meta_hardy, meta_age, meta_sex], dtype=np.float32)

    x_baseline = _decode(z_actual, meta_actual)
    x_changed  = _decode(z_actual, meta_new)
    delta      = x_changed - x_baseline

    if gene_text.strip():
        upper_map = {g.upper(): g for g in GENE_NAMES}
        genes = [upper_map[g.strip().upper()] for g in gene_text.split(",")
                 if g.strip().upper() in upper_map]
        if not genes:
            genes = TOP_GENES_DEFAULT[:TOP_N_GENES]
    else:
        # Default: top movers
        idx_movers = np.argsort(np.abs(delta))[::-1][:TOP_N_GENES]
        genes = [GENE_NAMES[i] for i in idx_movers]

    fig_delta = _bar_plot_difference(delta, genes,
        f"Δ expression for donor {donor_id} | meta change → custom")

    table = _top_movers(x_baseline, x_changed, top_n=20)

    # Show what changed in metadata
    meta_change = pd.DataFrame({
        "field":        ["SMTSISCH_z", "DTHHRDY_z", "AGE_mid_z", "SEX_bin"],
        "original":     meta_actual.round(3),
        "new":          meta_new.round(3),
        "Δ":            (meta_new - meta_actual).round(3),
    })

    return fig_delta, table, meta_change


# ──────────────────────────────────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="VAE Health Explorer") as demo:
    gr.Markdown("# VAE Health Explorer")
    gr.Markdown(
        f"Trained model: **Q54b** (FiLM MetaInjection VAE, K={K}, ischemia-residual encoder). "
        f"Loaded {len(DONOR_IDS)} donors, {N_GENES} genes.  "
        f"See `analysis/docs/vae_health_report.md` for what each piece means."
    )

    # ── Tab 1: Forward ───────────────────────────────────────────────────
    with gr.Tab("Forward (z + meta → expression)"):
        gr.Markdown(
            "Set the 16 latent dimensions and 4 metadata values, see what the decoder predicts. "
            "Each slider is in standard-deviation units (z-scored).  "
            "z=0 → average value; z=±1 → 1σ away; z=±3 → 3σ (rare)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### z_bio sliders (16)")
                z_sliders = [gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label=f"z{i+1}")
                             for i in range(K)]
            with gr.Column(scale=1):
                gr.Markdown("### Metadata sliders (4)")
                m_isch  = gr.Slider(-3.0, 3.0, value=0.0, step=0.1,
                    label="SMTSISCH_z (ischemia time, z-scored)")
                m_hardy = gr.Slider(-3.0, 3.0, value=0.0, step=0.1,
                    label="DTHHRDY_z (death hardiness, z-scored)")
                m_age   = gr.Slider(-3.0, 3.0, value=0.0, step=0.1,
                    label="AGE_mid_z (age, z-scored)")
                m_sex   = gr.Slider(0.0, 1.0, value=0.5, step=0.1,
                    label="SEX_bin (0=male, 1=female)")
                gr.Markdown("---")
                gene_input = gr.Textbox(
                    label="Genes to show (comma-separated, blank = top 30 most variable)",
                    placeholder="e.g. HBB, ACTB, FTL, GAPDH",
                )
                btn_fwd = gr.Button("Predict expression", variant="primary")

        info_box = gr.Markdown()
        plot_fwd = gr.Plot()
        table_fwd = gr.Dataframe(label="Per-gene predictions")

        btn_fwd.click(
            forward_predict,
            inputs=z_sliders + [m_isch, m_hardy, m_age, m_sex, gene_input],
            outputs=[plot_fwd, table_fwd, info_box],
        )

    # ── Tab 2: Reverse ────────────────────────────────────────────────────
    with gr.Tab("Reverse (donor → z_bio)"):
        gr.Markdown(
            "Pick a real GTEx donor and see (a) their encoded `z_bio`, (b) the model's "
            "reconstruction overlaid on the true expression."
        )

        with gr.Row():
            donor_dropdown = gr.Dropdown(
                choices=DONOR_IDS, value=DONOR_IDS[0],
                label="Choose donor", filterable=True,
            )
            gene_input2 = gr.Textbox(
                label="Genes (blank = top 30 most variable)",
                placeholder="e.g. HBB, ACTB",
            )
            btn_rev = gr.Button("Inspect donor", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                meta_table_rev = gr.Dataframe(label="Donor metadata")
            with gr.Column(scale=1):
                z_table_rev = gr.Dataframe(label="Encoded z_bio")

        plot_z_rev = gr.Plot()
        plot_recon = gr.Plot()

        btn_rev.click(
            reverse_inspect,
            inputs=[donor_dropdown, gene_input2],
            outputs=[plot_z_rev, plot_recon, meta_table_rev, z_table_rev],
        )

    # ── Tab 3: Counterfactual ─────────────────────────────────────────────
    with gr.Tab("Counterfactual (change metadata, see Δexpression)"):
        gr.Markdown(
            "Pick a donor. Their `z_bio` (biology) is held fixed. "
            "Change the metadata sliders to see how the decoder's prediction changes — "
            "i.e., 'what would this person's blood look like if their ischemia / age / sex / death were different?'"
        )

        with gr.Row():
            donor_dropdown_c = gr.Dropdown(
                choices=DONOR_IDS, value=DONOR_IDS[0],
                label="Choose donor", filterable=True,
            )
            gene_input3 = gr.Textbox(
                label="Genes (blank = top movers)",
                placeholder="e.g. HBB, FTL",
            )
            btn_cf = gr.Button("Compute counterfactual", variant="primary")

        gr.Markdown("### Counterfactual metadata (this is what the decoder will use):")
        with gr.Row():
            m_isch_c  = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label="SMTSISCH_z")
            m_hardy_c = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label="DTHHRDY_z")
            m_age_c   = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label="AGE_mid_z")
            m_sex_c   = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="SEX_bin")

        meta_change_table = gr.Dataframe(label="Metadata before / after")
        plot_cf = gr.Plot()
        table_cf = gr.Dataframe(label="Top 20 changed genes")

        btn_cf.click(
            counterfactual_predict,
            inputs=[donor_dropdown_c, m_isch_c, m_hardy_c, m_age_c, m_sex_c, gene_input3],
            outputs=[plot_cf, table_cf, meta_change_table],
        )

    # ── Footer ────────────────────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "**Quick tips:**\n"
        "- z values typically lie in [-3, +3]; far outside this range = extrapolation\n"
        "- SMTSISCH_z = +1.5 means ischemia ~1.5σ above mean (roughly +800 min)\n"
        "- The model was trained with ischemia *residualised* — its effect lives in the decoder via metadata.\n"
        "  Try setting all `z` to 0 and only sweeping SMTSISCH_z to see the pure ischemia-effect signature.\n"
        "- The Reverse tab takes the actual sample as input, applies the residualiser internally, and encodes."
    )


if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch(share=False, inbrowser=False)
