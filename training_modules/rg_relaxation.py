"""RG-guided full-lattice CNF relaxation used after a layer-wise residual lift."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .training_best_state import (
        BestStateTracker,
        rg_relaxation_record_is_stable,
        safe_optimizer_step,
    )
except ImportError:
    # Direct-file fallback for environments that execute this module outside
    # its package. Normal notebook/script usage should use the relative import.
    from training_best_state import (
        BestStateTracker,
        rg_relaxation_record_is_stable,
        safe_optimizer_step,
    )


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Return the checkpoint-compatible module behind DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _prepare_relaxation_training_model(
    model: "RGRelaxationCNF", cfg: Any, global_microbatch_size: int
) -> Tuple[nn.Module, List[int]]:
    """Optionally split each relaxation microbatch over all visible CUDA GPUs."""
    requested = bool(getattr(cfg, "rg_relaxation_multi_gpu", False))
    if not requested:
        return model, []
    device = torch.device(cfg.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "rg_relaxation_multi_gpu=True requires CUDA, but cfg.device is "
            f"{cfg.device!r}."
        )
    n_devices = int(torch.cuda.device_count())
    if n_devices < 2:
        raise RuntimeError(
            "rg_relaxation_multi_gpu=True requires at least two visible GPUs. "
            "Request multiple GPUs from the scheduler or set it to False."
        )
    primary = device.index
    if primary is None:
        primary = int(torch.cuda.current_device())
    device_ids = [primary] + [
        index for index in range(n_devices) if index != primary
    ]
    if int(global_microbatch_size) < len(device_ids):
        raise ValueError(
            "The global relaxation batch_size must be at least the number of "
            f"visible GPUs ({len(device_ids)}), got {global_microbatch_size}."
        )
    model.to(torch.device("cuda", primary))
    return (
        nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=primary,
        ),
        device_ids,
    )


def rg_block_coarsen(x: torch.Tensor) -> torch.Tensor:
    """Average each 2x2 block to an RG coarse field."""
    if x.ndim != 4 or x.shape[-1] != x.shape[-2] or x.shape[-1] % 2:
        raise ValueError(f"RG coarsening expects [B,C,2L,2L], got {tuple(x.shape)}.")
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def rg_block_prolong(coarse: torch.Tensor) -> torch.Tensor:
    """Piecewise-constant prolongation satisfying R(P(c)) = c."""
    return coarse.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def rg_ir_project(x: torch.Tensor) -> torch.Tensor:
    """Block-constant infrared component P R x."""
    return rg_block_prolong(rg_block_coarsen(x))


def rg_uv_project(x: torch.Tensor) -> torch.Tensor:
    """Within-block ultraviolet component (I - P R) x."""
    return x - rg_ir_project(x)


def rg_old_site_mask(fine_L: int, device=None, dtype=None) -> torch.Tensor:
    """Mask of inherited even-even sites in a completed fine lattice."""
    mask = torch.zeros(1, 1, int(fine_L), int(fine_L), device=device, dtype=dtype)
    mask[..., 0::2, 0::2] = 1
    return mask


def _time_channel(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    batch = ref.shape[0]
    if t.ndim == 0:
        t = t.expand(batch)
    elif t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    return t.to(device=ref.device, dtype=ref.dtype)[:, None, None, None].expand(
        batch, 1, ref.shape[-2], ref.shape[-1]
    )


def _make_periodic_cnn(in_channels: int, out_channels: int, cfg: Any) -> nn.Sequential:
    kernel = int(cfg.rg_relaxation_kernel_size)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("rg_relaxation_kernel_size must be a positive odd integer.")
    padding = kernel // 2
    width = int(cfg.rg_relaxation_channels)
    depth = max(1, int(cfg.rg_relaxation_layers))
    layers: List[nn.Module] = []
    channels = int(in_channels)
    for _ in range(depth):
        layers.append(
            nn.Conv2d(
                channels,
                width,
                kernel_size=kernel,
                padding=padding,
                padding_mode="circular",
            )
        )
        layers.append(nn.SiLU())
        channels = width
    layers.append(
        nn.Conv2d(
            channels,
            int(out_channels),
            kernel_size=kernel,
            padding=padding,
            padding_mode="circular",
        )
    )
    return nn.Sequential(*layers)


class RGRelaxationVelocityCNN(nn.Module):
    """Separate old/new candidate heads followed by RG UV/IR projection."""

    def __init__(self, fine_L: int, cfg: Any):
        super().__init__()
        self.fine_L = int(fine_L)
        self.uv_scale = float(cfg.rg_relaxation_uv_scale)
        self.ir_scale = float(cfg.rg_relaxation_ir_scale)
        # Inputs: current field, prolonged current coarse field, old mask, time.
        # Outputs: old candidate, new candidate, independent IR candidate.
        self.net = _make_periodic_cnn(4, 3, cfg)

        # Near-identity initialization. The first optimization step trains the
        # final heads while leaving the accepted generator output unchanged.
        final_conv = [m for m in self.net.modules() if isinstance(m, nn.Conv2d)][-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def components(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.fine_L or x.shape[-2] != self.fine_L:
            raise ValueError(
                f"Relaxation expected {self.fine_L}x{self.fine_L}, got {tuple(x.shape)}."
            )
        batch = x.shape[0]
        mask = rg_old_site_mask(
            self.fine_L, device=x.device, dtype=x.dtype
        ).expand(batch, -1, -1, -1)
        current_coarse = rg_block_coarsen(x)
        inputs = torch.cat(
            [x, rg_block_prolong(current_coarse), mask, _time_channel(t, x)], dim=1
        )
        raw = self.net(inputs)
        candidate = mask * raw[:, 0:1] + (1.0 - mask) * raw[:, 1:2]
        uv = rg_uv_project(candidate)
        ir = rg_ir_project(raw[:, 2:3])
        velocity = self.uv_scale * uv + self.ir_scale * ir
        return velocity, uv, ir

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.components(x, t)[0]


def _exact_divergence(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    velocity = velocity_fn(x, t)
    flat_velocity = velocity.flatten(1)
    divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for index in range(flat_velocity.shape[1]):
        grad = torch.autograd.grad(
            flat_velocity[:, index].sum(),
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        divergence = divergence + grad.flatten(1)[:, index]
    return velocity, divergence


def _hutchinson_divergence(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    samples: int,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    velocity = velocity_fn(x, t)
    estimate = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(max(1, int(samples))):
        probe = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
        jtv = torch.autograd.grad(
            (velocity * probe).sum(),
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        estimate = estimate + (jtv * probe).flatten(1).sum(dim=1)
    return velocity, estimate / max(1, int(samples))


def _divergence(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    mode: str,
    hutchinson_samples: int,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "exact":
        return _exact_divergence(velocity_fn, x, t, create_graph)
    if mode == "hutchinson":
        return _hutchinson_divergence(
            velocity_fn, x, t, hutchinson_samples, create_graph
        )
    raise ValueError(f"Unknown divergence mode: {mode}")


class RGRelaxationCNF(nn.Module):
    """Near-identity CNF acting on the complete fine lattice."""

    def __init__(
        self,
        fine_L: int,
        cfg: Any,
        n_steps: Optional[int] = None,
        divergence_mode: Optional[str] = None,
        hutchinson_samples: Optional[int] = None,
    ):
        super().__init__()
        self.fine_L = int(fine_L)
        self.n_steps = int(n_steps or cfg.rg_relaxation_cnf_steps)
        self.divergence_mode = divergence_mode or cfg.cnf_divergence_mode
        self.hutchinson_samples = int(
            hutchinson_samples
            if hutchinson_samples is not None
            else cfg.hutchinson_samples
        )
        self.velocity = RGRelaxationVelocityCNN(self.fine_L, cfg)
        self.cfg = cfg

    def forward(
        self,
        x0: torch.Tensor,
        compute_logq: bool = True,
        create_graph: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x0
        logq_delta = (
            torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            if compute_logq
            else None
        )
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            t = torch.full(
                (x.shape[0],),
                (step + 0.5) * dt,
                device=x.device,
                dtype=x.dtype,
            )
            if compute_logq:
                x = x.requires_grad_(True)
                velocity, divergence = _divergence(
                    self.velocity,
                    x,
                    t,
                    mode=self.divergence_mode,
                    hutchinson_samples=self.hutchinson_samples,
                    create_graph=create_graph,
                )
                logq_delta = logq_delta - dt * divergence
                x = x + dt * velocity
                if not create_graph:
                    x = x.detach()
                    logq_delta = logq_delta.detach()
            else:
                x = x + dt * self.velocity(x, t)
        return x, logq_delta


def normalized_rg_coarse_drift(
    relaxed: torch.Tensor,
    initial: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dimensionless endpoint drift of the block-coarse field."""
    coarse_initial = rg_block_coarsen(initial)
    coarse_relaxed = rg_block_coarsen(relaxed)
    numerator = (
        (coarse_relaxed - coarse_initial).flatten(1).square().mean(dim=1)
    )
    if scale is None:
        scale = coarse_initial.detach().flatten().var(unbiased=False)
    else:
        scale = torch.as_tensor(
            scale, device=coarse_initial.device, dtype=coarse_initial.dtype
        ).detach()
    scale = scale.clamp_min(1e-6)
    return (numerator / scale).mean()


def rg_relaxation_loss(
    model: nn.Module,
    initial: torch.Tensor,
    cfg: Any,
    action_fn: Callable[[torch.Tensor, Any], torch.Tensor],
    trust_weight: float = 0.0,
    coarse_scale: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Full-target RKL loss; the frozen pre-relaxation logq is constant."""
    core_model = _unwrap_data_parallel(model)
    relaxed, logq_delta = model(initial, compute_logq=True, create_graph=True)
    action_before = action_fn(initial, cfg)
    action_after = action_fn(relaxed, cfg)
    n_sites = int(core_model.fine_L**2)
    rkl = ((action_after + logq_delta) / n_sites).mean()
    coarse_drift = normalized_rg_coarse_drift(
        relaxed, initial, scale=coarse_scale
    )
    loss = rkl + float(trust_weight) * coarse_drift

    displacement = (relaxed - initial).square()
    old_mask = rg_old_site_mask(
        core_model.fine_L, device=relaxed.device, dtype=relaxed.dtype
    )
    new_mask = 1.0 - old_mask
    old_mse = (displacement * old_mask).sum() / (
        initial.shape[0] * old_mask.sum()
    ).clamp_min(1.0)
    new_mse = (displacement * new_mask).sum() / (
        initial.shape[0] * new_mask.sum()
    ).clamp_min(1.0)
    return {
        "loss": loss,
        "rkl": rkl.detach(),
        "action_before": action_before.detach(),
        "action_after": action_after.detach(),
        "logq_delta": logq_delta.detach(),
        "coarse_drift": coarse_drift.detach(),
        "old_mse": old_mse.detach(),
        "new_mse": new_mse.detach(),
        "samples": relaxed.detach(),
    }


def _make_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    use_fused = bool(cfg.gpu_resources_abundant) and torch.cuda.is_available()
    if use_fused:
        try:
            return torch.optim.AdamW(model.parameters(), lr=cfg.lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr)


def _finish_accumulated_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    *,
    stage: str,
    step: int,
) -> bool:
    """Clip and apply gradients that were accumulated over several microbatches."""
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=grad_clip
    )
    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
        optimizer.zero_grad(set_to_none=True)
        print(
            {
                "training_stopped": "nonfinite_gradient",
                "stage": stage,
                "step": int(step),
            }
        )
        return False

    optimizer.step()
    if not all(
        bool(torch.isfinite(parameter.detach()).all())
        for parameter in model.parameters()
    ):
        print(
            {
                "training_stopped": "nonfinite_parameter",
                "stage": stage,
                "step": int(step),
            }
        )
        return False
    return True


def train_rg_relaxation_cnf(
    cfg: Any,
    pre_relaxation_sampler: Callable,
    fine_L: int,
    action_fn: Callable[[torch.Tensor, Any], torch.Tensor],
    n_steps: Optional[int] = None,
) -> RGRelaxationCNF:
    """Train the full-lattice corrector while keeping its proposal frozen."""
    if not bool(cfg.rg_relaxation_enabled):
        raise ValueError("Set rg_relaxation_enabled=True before training RG relaxation.")
    train_steps = int(n_steps or cfg.rg_relaxation_train_steps)
    accumulation_steps = int(
        getattr(cfg, "rg_relaxation_gradient_accumulation_steps", 1)
    )
    if accumulation_steps < 1:
        raise ValueError(
            "rg_relaxation_gradient_accumulation_steps must be at least 1."
        )
    microbatch_size = int(cfg.batch_size)
    if microbatch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    effective_batch_size = microbatch_size * accumulation_steps
    stage_name = f"rg_relaxation_L{int(fine_L)}"
    core_model = RGRelaxationCNF(fine_L=fine_L, cfg=cfg).to(
        torch.device(cfg.device)
    )
    model, multi_gpu_device_ids = _prepare_relaxation_training_model(
        core_model,
        cfg,
        global_microbatch_size=microbatch_size,
    )
    multi_gpu_active = bool(multi_gpu_device_ids)
    print(
        {
            "stage": "rg_relaxation",
            "fine_L": int(fine_L),
            "batch_size": microbatch_size,
            "microbatch_size": microbatch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "batch_size_semantics": "global_across_visible_gpus",
            "multi_gpu_requested": bool(
                getattr(cfg, "rg_relaxation_multi_gpu", False)
            ),
            "multi_gpu_active": multi_gpu_active,
            "multi_gpu_device_ids": multi_gpu_device_ids,
            "per_gpu_microbatch_size_approx": (
                (microbatch_size + len(multi_gpu_device_ids) - 1)
                // len(multi_gpu_device_ids)
                if multi_gpu_active
                else microbatch_size
            ),
            "n_train_steps": train_steps,
            "cnf_steps": core_model.n_steps,
            "divergence_mode": core_model.divergence_mode,
            "hutchinson_samples": core_model.hutchinson_samples,
            "uv_scale": float(cfg.rg_relaxation_uv_scale),
            "ir_scale": float(cfg.rg_relaxation_ir_scale),
            "trust_weight": float(cfg.rg_relaxation_trust_weight),
        }
    )
    optimizer = _make_optimizer(model, cfg)
    history: List[Dict[str, Any]] = []
    best = BestStateTracker(
        model,
        stage=stage_name,
        score_key="rkl_per_site",
        validator=rg_relaxation_record_is_stable,
    )
    anneal_fraction = float(cfg.rg_relaxation_trust_anneal_fraction)

    for step in range(1, train_steps + 1):
        with torch.no_grad():
            initial_microbatches = [
                pre_relaxation_sampler(batch_size=microbatch_size).detach()
                for _ in range(accumulation_steps)
            ]
            if accumulation_steps > 1:
                coarse_scale = (
                    torch.cat(
                        [
                            rg_block_coarsen(initial)
                            for initial in initial_microbatches
                        ],
                        dim=0,
                    )
                    .flatten()
                    .var(unbiased=False)
                    .clamp_min(1e-6)
                )
            else:
                coarse_scale = None
        progress = (step - 1) / max(1, train_steps - 1)
        if anneal_fraction > 0:
            trust_weight = float(cfg.rg_relaxation_trust_weight) * max(
                0.0, 1.0 - progress / anneal_fraction
            )
        else:
            trust_weight = float(cfg.rg_relaxation_trust_weight)

        summary = {
            "loss": 0.0,
            "rkl_per_site": 0.0,
            "action_before_per_site": 0.0,
            "action_after_per_site": 0.0,
            "logq_delta_per_site": 0.0,
            "coarse_drift": 0.0,
            "old_site_mse": 0.0,
            "new_site_mse": 0.0,
        }
        n_sites = int(fine_L**2)

        if accumulation_steps == 1:
            output = rg_relaxation_loss(
                model,
                initial_microbatches[0],
                cfg,
                action_fn,
                trust_weight=trust_weight,
            )
            update_ok = safe_optimizer_step(
                output["loss"],
                model,
                optimizer,
                cfg.grad_clip,
                stage=stage_name,
                step=step,
            )
            if update_ok:
                summary = {
                    "loss": float(output["loss"].detach().cpu()),
                    "rkl_per_site": float(output["rkl"].cpu()),
                    "action_before_per_site": float(
                        (output["action_before"] / n_sites).mean().cpu()
                    ),
                    "action_after_per_site": float(
                        (output["action_after"] / n_sites).mean().cpu()
                    ),
                    "logq_delta_per_site": float(
                        (output["logq_delta"] / n_sites).mean().cpu()
                    ),
                    "coarse_drift": float(output["coarse_drift"].cpu()),
                    "old_site_mse": float(output["old_mse"].cpu()),
                    "new_site_mse": float(output["new_mse"].cpu()),
                }
        else:
            optimizer.zero_grad(set_to_none=True)
            update_ok = True
            for micro_step, initial in enumerate(initial_microbatches, start=1):
                output = rg_relaxation_loss(
                    model,
                    initial,
                    cfg,
                    action_fn,
                    trust_weight=trust_weight,
                    coarse_scale=coarse_scale,
                )
                if not bool(torch.isfinite(output["loss"].detach()).all()):
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        {
                            "training_stopped": "nonfinite_loss",
                            "stage": stage_name,
                            "step": int(step),
                            "micro_step": int(micro_step),
                        }
                    )
                    update_ok = False
                    break
                (output["loss"] / accumulation_steps).backward()
                weight = 1.0 / accumulation_steps
                summary["loss"] += (
                    float(output["loss"].detach().cpu()) * weight
                )
                summary["rkl_per_site"] += float(output["rkl"].cpu()) * weight
                summary["action_before_per_site"] += (
                    float((output["action_before"] / n_sites).mean().cpu())
                    * weight
                )
                summary["action_after_per_site"] += (
                    float((output["action_after"] / n_sites).mean().cpu())
                    * weight
                )
                summary["logq_delta_per_site"] += (
                    float((output["logq_delta"] / n_sites).mean().cpu())
                    * weight
                )
                summary["coarse_drift"] += (
                    float(output["coarse_drift"].cpu()) * weight
                )
                summary["old_site_mse"] += (
                    float(output["old_mse"].cpu()) * weight
                )
                summary["new_site_mse"] += (
                    float(output["new_mse"].cpu()) * weight
                )
                del output
            if update_ok:
                update_ok = _finish_accumulated_optimizer_step(
                    model,
                    optimizer,
                    cfg.grad_clip,
                    stage=stage_name,
                    step=step,
                )
        if not update_ok:
            break

        if step % max(1, train_steps // 20) == 0 or step == 1:
            with torch.no_grad():
                uv_mean_square = 0.0
                ir_mean_square = 0.0
                for initial in initial_microbatches:
                    t_mid = torch.full(
                        (initial.shape[0],),
                        0.5,
                        device=initial.device,
                        dtype=initial.dtype,
                    )
                    _, uv, ir = core_model.velocity.components(initial, t_mid)
                    uv_mean_square += (
                        float(uv.square().mean().cpu()) / accumulation_steps
                    )
                    ir_mean_square += (
                        float(ir.square().mean().cpu()) / accumulation_steps
                    )
            record = {
                "step": step,
                **summary,
                "uv_velocity_rms": float(uv_mean_square**0.5),
                "ir_velocity_rms": float(ir_mean_square**0.5),
                "trust_weight": float(trust_weight),
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "multi_gpu_active": multi_gpu_active,
                "multi_gpu_n_devices": len(multi_gpu_device_ids),
            }
            history.append(record)
            best.consider(model, record)
            print(record)

    model.training_history = history
    best.restore(model)
    core_model.training_history = history
    core_model.training_selection = dict(model.training_selection)
    core_model.multi_gpu_training = {
        "requested": bool(getattr(cfg, "rg_relaxation_multi_gpu", False)),
        "active": multi_gpu_active,
        "device_ids": list(multi_gpu_device_ids),
        "global_microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
    }
    return core_model


@torch.no_grad()
def sample_rg_relaxation_cnf(
    model: RGRelaxationCNF, x: torch.Tensor
) -> torch.Tensor:
    relaxed, _ = model(x, compute_logq=False)
    return relaxed.detach()


def inverse_rg_relaxation_cnf(
    model: RGRelaxationCNF, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-Euler inverse and the relaxation contribution to final logq."""
    initial = x
    integral_divergence = torch.zeros(
        x.shape[0], device=x.device, dtype=x.dtype
    )
    dt = 1.0 / model.n_steps
    for step in reversed(range(model.n_steps)):
        t = torch.full(
            (initial.shape[0],),
            (step + 0.5) * dt,
            device=initial.device,
            dtype=initial.dtype,
        )
        initial = initial.detach().requires_grad_(True)
        velocity, divergence = _divergence(
            model.velocity,
            initial,
            t,
            mode=model.divergence_mode,
            hutchinson_samples=model.hutchinson_samples,
            create_graph=False,
        )
        integral_divergence = integral_divergence + dt * divergence
        initial = initial - dt * velocity
    return initial.detach(), -integral_divergence.detach()
