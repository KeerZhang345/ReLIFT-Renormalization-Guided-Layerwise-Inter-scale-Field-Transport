"""Shared safeguards and best-state restoration for the training package."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn


RecordValidator = Callable[[Mapping[str, Any]], bool]


def finite_record(record: Mapping[str, Any]) -> bool:
    """Return True when every numeric scalar in a log record is finite."""
    for value in record.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def rkl_density_record_is_stable(record: Mapping[str, Any]) -> bool:
    """Reject numerically attractive but clearly collapsed RKL-CNF states.

    The reverse-KL estimate can become spuriously very negative when the CNF
    density/divergence calculation becomes unstable.  These deliberately loose
    per-site limits retain normal training fluctuations while excluding that
    failure mode.
    """
    if not finite_record(record):
        return False
    logq = record.get("logq_per_site", record.get("logq_per_new_site"))
    energy = record.get("action_per_site", record.get("energy_per_new_site"))
    loss = record.get("loss")
    return (
        logq is not None
        and energy is not None
        and loss is not None
        and abs(float(logq)) <= 2.0
        and -1.0 <= float(energy) <= 5.0
        and abs(float(loss)) <= 2.0
    )


def rg_relaxation_record_is_stable(record: Mapping[str, Any]) -> bool:
    """Gate out relaxation-flow runaway before comparing its RKL objective."""
    if not finite_record(record):
        return False
    limits = {
        "coarse_drift": 0.25,
        "old_site_mse": 0.25,
        "new_site_mse": 0.25,
        "uv_velocity_rms": 1.0,
        "ir_velocity_rms": 5.0,
        "logq_delta_per_site": 1.0,
    }
    return all(
        key in record and abs(float(record[key])) <= limit
        for key, limit in limits.items()
    )


def safe_optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    *,
    stage: str,
    step: int,
) -> bool:
    """Apply one update only when the loss, gradients, and result are finite."""
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss.detach()).all()):
        print({"training_stopped": "nonfinite_loss", "stage": stage, "step": int(step)})
        return False

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
        optimizer.zero_grad(set_to_none=True)
        print({"training_stopped": "nonfinite_gradient", "stage": stage, "step": int(step)})
        return False

    optimizer.step()
    if not all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters()):
        print({"training_stopped": "nonfinite_parameter", "stage": stage, "step": int(step)})
        return False
    return True


class BestStateTracker:
    """Keep and restore the best eligible moving-average training checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        *,
        stage: str,
        score_key: str = "loss",
        minimize: bool = True,
        smoothing_window: int = 3,
        validator: RecordValidator = finite_record,
    ) -> None:
        self.stage = str(stage)
        self.score_key = str(score_key)
        self.minimize = bool(minimize)
        self.validator = validator
        self.recent_scores: deque[float] = deque(maxlen=max(1, int(smoothing_window)))
        self.best_score = math.inf if self.minimize else -math.inf
        self.best_step: Optional[int] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.n_eligible = 0
        self.n_rejected = 0

    def consider(self, model: nn.Module, record: Mapping[str, Any]) -> bool:
        if not self.validator(record):
            self.n_rejected += 1
            return False
        raw_score = record.get(self.score_key)
        if raw_score is None or not math.isfinite(float(raw_score)):
            self.n_rejected += 1
            return False

        self.n_eligible += 1
        self.recent_scores.append(float(raw_score))
        smoothed_score = sum(self.recent_scores) / len(self.recent_scores)
        improved = (
            smoothed_score < self.best_score
            if self.minimize
            else smoothed_score > self.best_score
        )
        if improved:
            self.best_score = smoothed_score
            self.best_step = int(record.get("step", self.n_eligible))
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        return improved

    def restore(self, model: nn.Module) -> Dict[str, Any]:
        if self.best_state is None or self.best_step is None:
            raise RuntimeError(
                f"No finite, stable training state was recorded for stage {self.stage!r}; "
                "refusing to return/save the final weights."
            )
        model.load_state_dict(self.best_state)
        selection = {
            "policy": "best_stable_smoothed_metric",
            "stage": self.stage,
            "score_key": self.score_key,
            "smoothing_window": int(self.recent_scores.maxlen or 1),
            "best_step": int(self.best_step),
            "best_score": float(self.best_score),
            "n_eligible_records": int(self.n_eligible),
            "n_rejected_records": int(self.n_rejected),
            "restored": True,
        }
        model.training_selection = selection
        print({"restored_best_training_state": selection})
        return selection
