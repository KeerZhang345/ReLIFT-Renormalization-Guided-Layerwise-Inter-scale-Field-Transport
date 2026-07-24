"""Target action and source-distribution helpers for the phi-four model."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .config import Phi4Config


def phi4_action(phi: torch.Tensor, cfg: Phi4Config) -> torch.Tensor:
    """Square-lattice phi-four action with periodic boundary conditions."""
    if phi.ndim == 3:
        phi = phi[:, None, :, :]
    dx = phi - torch.roll(phi, shifts=-1, dims=-2)
    dy = phi - torch.roll(phi, shifts=-1, dims=-1)
    kinetic = 0.5 * (dx.square() + dy.square())
    potential = 0.5 * cfg.m2 * phi.square() + (cfg.lam / 4.0) * phi.pow(4)
    return (kinetic + potential).sum(dim=(-1, -2, -3))


def log_gaussian_field(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalized log density of independent zero-mean Gaussian entries."""
    dimension = x[0].numel()
    quadratic = (x.square() / sigma**2).flatten(1).sum(dim=1)
    return -0.5 * quadratic - dimension * math.log(
        sigma * math.sqrt(2.0 * math.pi)
    )


def free_theory_energy(x: torch.Tensor, m2: float) -> torch.Tensor:
    """Quadratic lambda=0 lattice energy with periodic boundaries."""
    if x.ndim == 3:
        x = x[:, None, :, :]
    dx = x - torch.roll(x, shifts=-1, dims=-2)
    dy = x - torch.roll(x, shifts=-1, dims=-1)
    density = 0.5 * (dx.square() + dy.square()) + 0.5 * m2 * x.square()
    return density.sum(dim=(-1, -2, -3))


def log_free_theory_field(x: torch.Tensor, m2: float) -> torch.Tensor:
    """Normalized log density of the periodic Gaussian free field."""
    if m2 <= 0:
        raise ValueError("Normalized free-theory density requires m2 > 0.")
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4 or x.shape[-2] != x.shape[-1]:
        raise ValueError("Expected free-theory fields with shape [B, C, L, L].")

    channels, lattice_size = x.shape[1], x.shape[-1]
    momenta = 2.0 * math.pi * torch.fft.fftfreq(
        lattice_size, device=x.device, dtype=x.dtype
    )
    eigenvalues = (
        m2
        + 4.0
        - 2.0 * torch.cos(momenta)[:, None]
        - 2.0 * torch.cos(momenta)[None, :]
    )
    log_determinant = channels * torch.log(eigenvalues).sum()
    dimension = x[0].numel()
    log_normalizer = (
        0.5 * log_determinant
        - 0.5 * dimension * math.log(2.0 * math.pi)
    )
    return -free_theory_energy(x, m2) + log_normalizer


def sample_free_theory_field(
    batch_size: int,
    channels: int,
    L: int,
    m2: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Sample the Gaussian free field through its Fourier diagonalization."""
    if m2 <= 0:
        raise ValueError("free_theory initialization requires init_free_m2 > 0.")
    white = torch.randn(
        batch_size,
        channels,
        L,
        L,
        device=device,
        dtype=dtype,
    )
    momenta = 2.0 * math.pi * torch.fft.fftfreq(
        L, device=device, dtype=dtype
    )
    eigenvalues = (
        m2
        + 4.0
        - 2.0 * torch.cos(momenta)[:, None]
        - 2.0 * torch.cos(momenta)[None, :]
    )
    white_fourier = torch.fft.fft2(white, norm="ortho")
    scaled = white_fourier / torch.sqrt(eigenvalues)[None, None, :, :]
    return torch.fft.ifft2(scaled, norm="ortho").real


def _init_sigma_for_role(cfg: Phi4Config, role: str) -> float:
    if role == "base":
        return cfg.coarse_sigma
    if role == "residual":
        return cfg.residual_sigma
    raise ValueError(f"Unknown initialization role: {role}")


def sample_initial_field(
    cfg: Phi4Config,
    batch_size: int,
    channels: int,
    L: int,
    device=None,
    dtype=torch.float32,
    role: str = "residual",
) -> torch.Tensor:
    """Sample the configured source law for base or residual variables."""
    device = torch.device(cfg.device) if device is None else device
    init_type = getattr(cfg, "init_type", "gaussian")
    if init_type == "gaussian":
        sigma = _init_sigma_for_role(cfg, role)
        shape = (batch_size, channels, L, L)
        return sigma * torch.randn(*shape, device=device, dtype=dtype)
    if init_type == "free_theory":
        return sample_free_theory_field(
            batch_size,
            channels,
            L,
            cfg.init_free_m2,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown init_type: {init_type}")


def initial_base_energy(
    x: torch.Tensor, cfg: Phi4Config, role: str = "residual"
) -> torch.Tensor:
    """Unnormalized source energy for the configured initialization law."""
    init_type = getattr(cfg, "init_type", "gaussian")
    if init_type == "gaussian":
        sigma = _init_sigma_for_role(cfg, role)
        return 0.5 * (x.square() / sigma**2).flatten(1).sum(dim=1)
    if init_type == "free_theory":
        return free_theory_energy(x, cfg.init_free_m2)
    raise ValueError(f"Unknown init_type: {init_type}")


def initial_log_prob(
    x: torch.Tensor, cfg: Phi4Config, role: str = "residual"
) -> torch.Tensor:
    """Normalized source log density, including additive constants."""
    init_type = getattr(cfg, "init_type", "gaussian")
    if init_type == "gaussian":
        return log_gaussian_field(x, _init_sigma_for_role(cfg, role))
    if init_type == "free_theory":
        return log_free_theory_field(x, cfg.init_free_m2)
    raise ValueError(f"Unknown init_type: {init_type}")


def resolve_sde_grid(
    cfg: Phi4Config, n_steps: Optional[int] = None
) -> Tuple[int, float]:
    """Resolve a uniform SDE integration grid over the interval [0, 1]."""
    if n_steps is not None:
        steps = int(n_steps)
    else:
        step_size = getattr(cfg, "sde_step_size", None)
        if step_size is None:
            steps = int(cfg.sde_steps)
        else:
            if step_size <= 0:
                raise ValueError("sde_step_size must be positive when set.")
            steps = int(math.ceil(1.0 / float(step_size)))
    steps = max(1, steps)
    return steps, 1.0 / steps
