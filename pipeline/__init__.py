"""Reusable evaluation pipeline for bulk-RNA-seq representation models.

Public API (everything you need to evaluate a new model):

  from pipeline import EvalConfig, run_evaluation
  from pipeline import GTExBlood, evaluate_reconstruction, evaluate_latent_metadata
  from pipeline import enrich_dim_loadings

The pipeline is built around a small protocol — any model with .encode(x) and
.decode(z) (or that returns the reconstruction directly via __call__) plugs in.
See pipeline.protocols.LatentModel for the contract.
"""
from .protocols import LatentModel  # noqa: F401
from .data import GTExBlood, load_shared_genes, load_metadata  # noqa: F401
from .reconstruction import evaluate_reconstruction, ReconstructionResult  # noqa: F401
from .latent import evaluate_latent_meta, evaluate_latent_activity  # noqa: F401
from .enrichment import enrich_dim_loadings, enrichr_submit, enrichr_query  # noqa: F401
from .runner import EvalConfig, run_evaluation  # noqa: F401
