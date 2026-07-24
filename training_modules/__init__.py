"""Reusable training and reference-sampling components.

The notebooks remain experiment entry points. Stable configuration, physics,
relaxation, best-state, and MCMC functionality lives in this package so batch
scripts and notebooks use the same implementation.
"""

__all__ = [
    "config",
    "mcmc_phi4",
    "pipeline",
    "physics",
    "rg_relaxation",
    "training_best_state",
    "validation",
]
