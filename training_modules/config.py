"""Configuration and runtime resource helpers for lattice training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return the default CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Phi4Config:
    """Complete serializable configuration for the layerwise experiments."""

    # Target action parameters.
    m2: float = -0.4658
    lam: float = 0.3007

    # Lattice hierarchy.
    L0: int = 4
    n_lift_layers: int = 1
    partition_mode: str = "full_kadanoff"

    # Conditional information and transport backbone.
    embedding_type: Optional[str] = None
    transportation_backbone: str = "mlp"
    cnn_channels: int = 64
    cnn_layers: int = 4
    cnn_kernel_size: int = 3
    inherit_previous_lift_cnn_weights: bool = False

    # Legacy fields retained so older checkpoints can still be upgraded.
    use_context_embedding: bool = False
    embedding_backbone: str = "mlp"
    context_dim: int = 64
    context_hidden_dim: int = 128
    embedding_cnn_channels: int = 32
    embedding_cnn_layers: int = 3
    fourier_context_modes: int = 4
    condition_mode: str = "raw_plus_context"

    # Optional full-lattice RG-guided relaxation after each lift.
    rg_relaxation_enabled: bool = False
    rg_relaxation_train_steps: int = 1000
    rg_relaxation_multi_gpu: bool = False
    rg_relaxation_gradient_accumulation_steps: int = 1
    rg_relaxation_cnf_steps: int = 8
    rg_relaxation_channels: int = 32
    rg_relaxation_layers: int = 4
    rg_relaxation_kernel_size: int = 3
    rg_relaxation_uv_scale: float = 1.0
    rg_relaxation_ir_scale: float = 0.1
    rg_relaxation_trust_weight: float = 0.0
    rg_relaxation_trust_anneal_fraction: float = 0.5

    # Base and residual initialization.
    residual_sigma: float = 0.7
    coarse_sigma: float = 0.7
    device: str = str(get_device())
    init_type: str = "gaussian"
    init_free_m2: float = 1.0

    # Common training settings.
    batch_size: int = 64
    lr: float = 2e-4
    n_train_steps: int = 500
    grad_clip: float = 1.0

    # GPU memory/throughput policy.
    gpu_resources_abundant: bool = False
    gpu_parent_pool_size: int = 32768
    gpu_parent_generation_batch_size: int = 256

    # Stage-specific overrides. Layer keys are coarse sizes L for L -> 2L.
    base_train_overrides: Dict[str, Any] = field(default_factory=dict)
    layer_train_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # CNF/ODE settings.
    cnf_steps: int = 8
    hidden_dim: int = 128
    cnf_divergence_mode: str = "exact"
    hutchinson_samples: int = 1

    # EWFM settings.
    ewfm_max_weight: float = 20.0
    ewfm_outer_loops: int = 8
    ewfm_buffer_size: int = 4096

    # RFM posterior settings.
    rfm_mc_samples: int = 8
    rfm_t_min: float = 0.02
    rfm_use_stein: bool = True
    rfm_stein_ridge: float = 1e-4

    # iDEM settings.
    idem_outer_loops: int = 8
    idem_buffer_size: int = 4096
    idem_exploration_noise: float = 1.0

    # NETS and common SDE settings.
    nets_eps: float = 1.0
    sde_steps: int = 32
    sde_step_size: Optional[float] = None


def configure_gpu_resource_policy(cfg: Phi4Config) -> None:
    """Configure CUDA runtime choices for the selected resource policy."""
    abundant = bool(cfg.gpu_resources_abundant) and torch.cuda.is_available()
    if abundant:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    print({"gpu_resource_policy": "abundant" if abundant else "oom_conservative"})


def make_adamw(module: nn.Module, cfg: Phi4Config) -> torch.optim.Optimizer:
    """Use fused AdamW in abundant CUDA mode and fall back safely."""
    use_fused = bool(cfg.gpu_resources_abundant) and torch.cuda.is_available()
    if use_fused:
        try:
            return torch.optim.AdamW(module.parameters(), lr=cfg.lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(module.parameters(), lr=cfg.lr)
