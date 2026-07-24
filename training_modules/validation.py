"""Full multiscale validation shared by notebooks and batch runners.

Definitions are kept checkpoint-compatible with the validation section of
``residual_sr_layerwise_pipeline.ipynb``; no experiment runs at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import Phi4Config
from .physics import initial_log_prob, phi4_action, sample_initial_field
from .pipeline import (
    BaseFieldCNF,
    BaseFieldVelocityMLP,
    ResidualCNF,
    ResidualVelocityMLP,
    _restore_module_requires_grad,
    _set_module_requires_grad,
    compute_divergence,
    get_partition,
    insert_residual,
    integrate_base_velocity_with_logq,
    integrate_residual_velocity_with_logq,
    inverse_rg_relaxation_cnf,
    residual_channels_for_L,
    sample_idem_base_model,
    sample_idem_residual_layer,
    sample_nets_langevin_base_model,
    sample_nets_langevin_residual_layer,
    set_cnf_stack_divergence,
)


# --- Source notebook validation cell 51 ---

TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class ParadigmOutput:
    """
    Container for one trained sampler's full-rollout outputs.

    samples:
        Generated full lattice samples, shape [N, ...].

    logq_samples:
        Fully normalized log q_theta(x) for generated samples.
        Needed for RKL and density-based ESS.
        Available for CNF / likelihood-based EWFM.

    logq_mcmc:
        Fully normalized log q_theta(x_mcmc) evaluated on MCMC samples.
        Needed for NLL / forward-KL proxy.

    path_logw:
        Total target-corrected full-path log weights.
        Native for NETS / AIS / Jarzynski-style methods and aligned with
        the same generated particle ancestries.

    training_metrics:
        Optional dictionary, e.g. natural loss values from training.
    """
    samples: TensorLike
    logq_samples: Optional[TensorLike] = None
    logq_mcmc: Optional[TensorLike] = None
    path_logw: Optional[TensorLike] = None
    training_metrics: Optional[Dict[str, Any]] = None


class FullLatticeSamplerEvaluator:
    """
    Evaluate full-lattice samples against MCMC reference samples.

    Main philosophy:
        - Do not evaluate conditional per-layer quality unless explicitly needed.
        - Compare final generated L128 samples against final MCMC L128 samples.
        - Use likelihood-based metrics only when log q is actually available.
        - Use path ESS only when valid path weights are available, e.g. NETS.

    Parameters
    ----------
    mcmc_samples:
        Full MCMC lattice samples, shape [N, ...].

    energy_fn:
        Callable mapping samples [N, ...] -> action/energy S(x), shape [N].
        This should be the full target action at the final lattice size.

    observable_fn:
        Optional callable mapping samples [N, ...] -> dict[str, TensorLike].
        Example observables: energy, magnetization, correlation, structure factor bins.

    feature_fn:
        Optional callable mapping samples [N, ...] -> features [N, D] for C2ST.
        If None, flattened raw samples are used.

    normalize_by_dim:
        If True, RKL proxy values are divided by number of lattice variables.
        NLL and unnormalized RNLL are always reported per site so that
        different lattice sizes can be compared.

    device:
        Torch device.
    """

    def __init__(
        self,
        mcmc_samples: TensorLike,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        observable_fn: Optional[Callable[[torch.Tensor], Dict[str, TensorLike]]] = None,
        feature_fn: Optional[Callable[[torch.Tensor], TensorLike]] = None,
        normalize_by_dim: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 4096,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Keep the full reference ensemble on CPU; move only working batches.
        self.mcmc_samples = self._to_tensor(mcmc_samples).cpu()
        self.energy_fn = energy_fn
        self.observable_fn = observable_fn
        self.feature_fn = feature_fn
        self.normalize_by_dim = normalize_by_dim
        self.batch_size = batch_size

        if self.mcmc_samples.ndim < 2:
            raise ValueError("mcmc_samples should have shape [N, ...].")

        self.sample_shape = tuple(self.mcmc_samples.shape[1:])
        self.dim = int(np.prod(self.sample_shape))

        self._mcmc_energy_cache = None
        self._mcmc_observable_cache = None

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x: TensorLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach()
        return torch.as_tensor(x, dtype=dtype)

    def _scale(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalize_by_dim:
            return value / self.dim
        return value

    def _batch_apply(
        self,
        fn: Callable[[torch.Tensor], TensorLike],
        x: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                xb = x[start:start + self.batch_size].to(self.device)
                out = fn(xb)
                out = self._to_tensor(out).to(self.device)
                outs.append(out)
        return torch.cat(outs, dim=0)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        e = self._batch_apply(self.energy_fn, x)
        return e.reshape(-1)

    def _mcmc_energy(self) -> torch.Tensor:
        if self._mcmc_energy_cache is None:
            self._mcmc_energy_cache = self._energy(self.mcmc_samples)
        return self._mcmc_energy_cache

    # ------------------------------------------------------------------
    # Unnormalized reverse NLL (target action)
    # ------------------------------------------------------------------

    def rnll_unnormalized(self, generated_samples: TensorLike) -> Dict[str, float]:
        """
        Unnormalized reverse NLL per site:
            RNLL_unnorm / V = E_q[S(x)] / V.

        For p(x) = exp(-S(x)) / Z, the normalized reverse cross-entropy
        additionally contains log Z / V. We do not estimate or report Z.
        Consequently this metric is comparable only for the same target
        action and is interpreted together with NLL and diversity metrics.

        This is available for all methods.
        """
        xg = self._to_tensor(generated_samples).to(self.device)
        eg = self._energy(xg)
        em = self._mcmc_energy()

        gen_mean = eg.mean() / self.dim
        mcmc_mean = em.mean() / self.dim
        gap = gen_mean - mcmc_mean

        return {
            "rnll_unnormalized_per_site": float(gen_mean.cpu()),
            "mcmc_action_per_site": float(mcmc_mean.cpu()),
            "action_gap_per_site_gen_minus_mcmc": float(gap.cpu()),
            "generated_action_std_per_site": float((eg.std(unbiased=False) / self.dim).cpu()),
            "mcmc_action_std_per_site": float((em.std(unbiased=False) / self.dim).cpu()),
        }

    # ------------------------------------------------------------------
    # Normalized model NLL
    # ------------------------------------------------------------------

    def nll(self, logq_mcmc: TensorLike) -> Dict[str, float]:
        """
        Site-average NLL on MCMC reference samples:
            NLL / V = -E_{p_mcmc}[log q_theta(x)] / V.

        This is a proxy for forward KL:
            KL(p || q) = E_p[log p - log q].

        Requires the fully normalized model log density log q_theta,
        including the base-density normalization constants. Target Z is
        not involved.
        """
        logq = self._to_tensor(logq_mcmc).to(self.device).reshape(-1)

        if logq.shape[0] != self.mcmc_samples.shape[0]:
            raise ValueError(
                f"logq_mcmc length {logq.shape[0]} does not match "
                f"MCMC sample count {self.mcmc_samples.shape[0]}."
            )

        nll_value = (-logq).mean() / self.dim

        return {
            "nll_per_site": float(nll_value.cpu()),
        }

    # ------------------------------------------------------------------
    # RKL
    # ------------------------------------------------------------------

    def rkl(
        self,
        generated_samples: TensorLike,
        logq_samples: TensorLike,
    ) -> Dict[str, float]:
        """
        Reverse KL up to unknown log Z:
            KL(q || p) = E_q[log q(x) + S(x)] + log Z.

        Requires log q_theta for generated samples.
        """
        xg = self._to_tensor(generated_samples).to(self.device)
        logq = self._to_tensor(logq_samples).to(self.device).reshape(-1)

        if logq.shape[0] != xg.shape[0]:
            raise ValueError(
                f"logq_samples length {logq.shape[0]} does not match "
                f"generated sample count {xg.shape[0]}."
            )

        eg = self._energy(xg)
        rkl_value = self._scale((logq + eg).mean())

        return {
            "rkl_proxy": float(rkl_value.cpu()),
            "mean_logq_generated": float(self._scale(logq.mean()).cpu()),
        }

    # ------------------------------------------------------------------
    # ESS
    # ------------------------------------------------------------------

    @staticmethod
    def _ess_from_log_weights(logw: torch.Tensor) -> Dict[str, float]:
        """
        Compute absolute ESS and relative ESS from log weights.

        rESS = ESS / N = 1 / (N * sum_i wbar_i^2).
        """
        logw = logw.reshape(-1)
        n = logw.numel()

        logw_norm = logw - torch.logsumexp(logw, dim=0)
        wbar = torch.exp(logw_norm)

        ess = 1.0 / torch.sum(wbar ** 2)
        ress = ess / n
        if ess > n + 1e-4:
            raise RuntimeError(f"Invalid ESS {float(ess.cpu())} exceeds number of weights {n}.")

        return {
            "n_weights": int(n),
            "ess_over_n": float(ress.cpu()),
            "ess_abs": float(ess.cpu()),
            "logw_mean": float(logw.mean().cpu()),
            "logw_std": float(logw.std(unbiased=False).cpu()),
            "logw_min": float(logw.min().cpu()),
            "logw_max": float(logw.max().cpu()),
        }

    def density_ess(
        self,
        generated_samples: TensorLike,
        logq_samples: TensorLike,
    ) -> Dict[str, float]:
        """
        Final-sample IS ESS using:
            log w = -S(x) - log q_theta(x).

        Available for likelihood-based samplers, e.g. CNF or likelihood-based EWFM.
        """
        xg = self._to_tensor(generated_samples).to(self.device)
        logq = self._to_tensor(logq_samples).to(self.device).reshape(-1)

        if logq.shape[0] != xg.shape[0]:
            raise ValueError(
                f"logq_samples length {logq.shape[0]} does not match "
                f"generated sample count {xg.shape[0]}."
            )

        eg = self._energy(xg)
        logw = -eg - logq

        out = self._ess_from_log_weights(logw)
        out = {f"density_{k}": v for k, v in out.items()}

        return out

    def _normalized_density_weights(
        self,
        generated_samples: TensorLike,
        logq_samples: TensorLike,
    ) -> torch.Tensor:
        """Return stable self-normalized weights for generated samples."""
        xg = self._to_tensor(generated_samples).to(self.device)
        logq = self._to_tensor(logq_samples).to(self.device).reshape(-1)
        if logq.shape[0] != xg.shape[0]:
            raise ValueError(
                f"logq_samples length {logq.shape[0]} does not match "
                f"generated sample count {xg.shape[0]}."
            )
        logw = (-self._energy(xg) - logq).double()
        return torch.softmax(logw, dim=0)

    def path_ess(self, path_logw: TensorLike) -> Dict[str, float]:
        """
        ESS from target-corrected path weights, e.g. NETS / AIS / Jarzynski.

        path_logw must be the accumulated ancestry weight after any
        hierarchical-to-target action correction, not a normalized local
        training weight or a per-layer ESS.
        """
        logw = self._to_tensor(path_logw).to(self.device).reshape(-1)
        out = self._ess_from_log_weights(logw)
        return {f"path_{k}": v for k, v in out.items()}

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _observables(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.observable_fn is None:
            return {}

        raw = self.observable_fn(x)
        out = {}
        for key, value in raw.items():
            value = self._to_tensor(value).to(self.device)
            if value.ndim == 0:
                value = value.reshape(1)
            out[key] = value
        return out

    def _mcmc_observables(self) -> Dict[str, torch.Tensor]:
        if self.observable_fn is None:
            return {}
        if self._mcmc_observable_cache is None:
            self._mcmc_observable_cache = self._observables(self.mcmc_samples)
        return self._mcmc_observable_cache

    def observable_gaps(
        self,
        generated_samples: TensorLike,
        logq_samples: Optional[TensorLike] = None,
    ) -> Dict[str, float]:
        """
        Compare raw and, when log q is available, self-normalized IS
        observables between generated and MCMC samples.

        The observable_fn may return:
            {"magnetization": [N],
             "corr_r1": [N],
             "structure_factor": [N, K],
             ...}

        For vector-valued observables, reports mean absolute gap and RMSE.
        """
        if self.observable_fn is None:
            return {}

        xg = self._to_tensor(generated_samples).to(self.device)
        obs_g = self._observables(xg)
        obs_m = self._mcmc_observables()
        weights = (
            self._normalized_density_weights(xg, logq_samples)
            if logq_samples is not None
            else None
        )

        metrics = {}

        for key in obs_m.keys():
            if key not in obs_g:
                continue

            gm = obs_g[key].float()
            mm = obs_m[key].float()

            # Average over samples, preserve observable dimension if any.
            gm_mean = gm.mean(dim=0)
            mm_mean = mm.mean(dim=0)
            gap = gm_mean - mm_mean

            # Std over MCMC samples, used for normalized error.
            mm_std = mm.std(dim=0, unbiased=False)
            z_gap = gap / (mm_std + 1e-8)

            metrics[f"obs_{key}_gen_mean"] = float(gm_mean.mean().cpu())
            # Explicit raw aliases make the comparison with IS estimates clear;
            # the historical gen_mean names are retained for compatibility.
            metrics[f"obs_{key}_raw_mean"] = float(gm_mean.mean().cpu())
            metrics[f"obs_{key}_mcmc_mean"] = float(mm_mean.mean().cpu())
            metrics[f"obs_{key}_mean_abs_gap"] = float(gap.abs().mean().cpu())
            metrics[f"obs_{key}_rmse_gap"] = float(torch.sqrt((gap ** 2).mean()).cpu())
            metrics[f"obs_{key}_mean_abs_z_gap"] = float(z_gap.abs().mean().cpu())
            metrics[f"obs_{key}_raw_mean_abs_gap"] = float(gap.abs().mean().cpu())

            if weights is not None:
                wshape = (weights.shape[0],) + (1,) * (gm.ndim - 1)
                is_mean = (gm.double() * weights.reshape(wshape)).sum(dim=0)
                is_gap = is_mean - mm_mean.double()
                is_z_gap = is_gap / (mm_std.double() + 1e-8)
                metrics[f"obs_{key}_is_mean"] = float(is_mean.mean().cpu())
                metrics[f"obs_{key}_is_mean_abs_gap"] = float(is_gap.abs().mean().cpu())
                metrics[f"obs_{key}_is_rmse_gap"] = float(
                    torch.sqrt((is_gap ** 2).mean()).cpu()
                )
                metrics[f"obs_{key}_is_mean_abs_z_gap"] = float(
                    is_z_gap.abs().mean().cpu()
                )

        return metrics

    def critical_observable_metrics(
        self,
        generated_samples: TensorLike,
        logq_samples: Optional[TensorLike] = None,
    ) -> Dict[str, float]:
        """Compute raw and self-normalized-IS susceptibility and Binder cumulant.

        Susceptibility uses the Z2-symmetric finite-volume convention
            chi_abs = V * (<m^2> - <|m|>^2),
        while the Binder cumulant is
            U4 = 1 - <m^4> / (3 * <m^2>^2).
        These are ensemble-level metrics and must not be averaged per sample.
        """
        xg = self._to_tensor(generated_samples).to(self.device)
        xm = self.mcmc_samples
        reduce_g = tuple(range(1, xg.ndim))
        reduce_m = tuple(range(1, xm.ndim))
        mg = xg.double().mean(dim=reduce_g).reshape(-1)
        mm = xm.double().mean(dim=reduce_m).reshape(-1)
        volume = float(self.dim)

        def summarize(
            m: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if weights is None:
                m2 = m.square().mean()
                m4 = m.pow(4).mean()
                mean_abs_m = m.abs().mean()
            else:
                weights = weights.to(device=m.device, dtype=m.dtype)
                m2 = (weights * m.square()).sum()
                m4 = (weights * m.pow(4)).sum()
                mean_abs_m = (weights * m.abs()).sum()
            chi_abs = volume * (m2 - mean_abs_m.square())
            binder = 1.0 - m4 / (3.0 * m2.square().clamp_min(1e-16))
            return chi_abs, binder

        chi_g, binder_g = summarize(mg)
        chi_m, binder_m = summarize(mm)
        chi_m = chi_m.to(chi_g.device)
        binder_m = binder_m.to(binder_g.device)
        metrics = {
            "obs_susceptibility_gen": float(chi_g.cpu()),
            "obs_susceptibility_raw": float(chi_g.cpu()),
            "obs_susceptibility_mcmc": float(chi_m.cpu()),
            "obs_susceptibility_abs_gap": float((chi_g - chi_m).abs().cpu()),
            "obs_susceptibility_raw_abs_gap": float((chi_g - chi_m).abs().cpu()),
            "obs_binder_cumulant_gen": float(binder_g.cpu()),
            "obs_binder_cumulant_raw": float(binder_g.cpu()),
            "obs_binder_cumulant_mcmc": float(binder_m.cpu()),
            "obs_binder_cumulant_abs_gap": float((binder_g - binder_m).abs().cpu()),
            "obs_binder_cumulant_raw_abs_gap": float((binder_g - binder_m).abs().cpu()),
        }

        if logq_samples is not None:
            weights = self._normalized_density_weights(xg, logq_samples)
            chi_is, binder_is = summarize(mg, weights)
            metrics.update({
                "obs_susceptibility_is": float(chi_is.cpu()),
                "obs_susceptibility_is_abs_gap": float((chi_is - chi_m).abs().cpu()),
                "obs_binder_cumulant_is": float(binder_is.cpu()),
                "obs_binder_cumulant_is_abs_gap": float(
                    (binder_is - binder_m).abs().cpu()
                ),
            })

        return metrics

    # ------------------------------------------------------------------
    # C2ST
    # ------------------------------------------------------------------

    def _features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract features for C2ST.

        If feature_fn is provided, use it.
        Otherwise flatten raw samples.
        """
        with torch.no_grad():
            if self.feature_fn is not None:
                feat = self.feature_fn(x)
                feat = self._to_tensor(feat).to(self.device)
            else:
                feat = x.reshape(x.shape[0], -1).float()

        if feat.ndim != 2:
            feat = feat.reshape(feat.shape[0], -1)

        return feat.detach().cpu().numpy()

    def c2st(
        self,
        generated_samples: TensorLike,
        max_samples_per_class: int = 5000,
        test_size: float = 0.3,
        n_repeats: int = 5,
        classifier: str = "logreg",
        random_state: int = 0,
    ) -> Dict[str, float]:
        """
        Classifier two-sample test.

        Label:
            MCMC      -> 1
            Generated -> 0

        Accuracy close to 0.5 means generated samples are hard to distinguish
        from MCMC samples.

        classifier:
            "logreg" or "rf".
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError as e:
            return {
                "c2st_error": f"scikit-learn is required for C2ST: {str(e)}"
            }

        xg = self._to_tensor(generated_samples).to(self.device)
        xm = self.mcmc_samples

        n = min(xg.shape[0], xm.shape[0], max_samples_per_class)
        if n < 10:
            return {"c2st_error": "Not enough samples for C2ST."}

        rng = np.random.default_rng(random_state)

        accs: List[float] = []
        aucs: List[float] = []

        for r in range(n_repeats):
            idx_g = rng.choice(xg.shape[0], size=n, replace=False)
            idx_m = rng.choice(xm.shape[0], size=n, replace=False)

            fg = self._features(xg[idx_g])
            fm = self._features(xm[idx_m])

            X = np.concatenate([fm, fg], axis=0)
            y = np.concatenate([np.ones(n), np.zeros(n)], axis=0)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state + r,
                stratify=y,
            )

            if classifier == "logreg":
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                )
            elif classifier == "rf":
                clf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state + r,
                    n_jobs=-1,
                )
            else:
                raise ValueError("classifier should be 'logreg' or 'rf'.")

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            accs.append(acc)

            try:
                prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, prob)
                aucs.append(auc)
            except Exception:
                pass

        out = {
            "c2st_acc_mean": float(np.mean(accs)),
            "c2st_acc_std": float(np.std(accs)),
            "c2st_acc_excess_over_0p5": float(np.mean(accs) - 0.5),
        }

        if len(aucs) > 0:
            out["c2st_auc_mean"] = float(np.mean(aucs))
            out["c2st_auc_std"] = float(np.std(aucs))

        return out

    # ------------------------------------------------------------------
    # Main evaluation call
    # ------------------------------------------------------------------

    def evaluate_one(
        self,
        output: Union[ParadigmOutput, Dict[str, Any], TensorLike],
        paradigm_name: str = "unknown",
        compute_c2st: bool = True,
        c2st_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate one sampler.

        output can be:
            - ParadigmOutput
            - dict with keys:
                samples, logq_samples, logq_mcmc, path_logw, training_metrics
            - raw generated samples tensor/array
        """
        if isinstance(output, ParadigmOutput):
            pack = output
        elif isinstance(output, dict):
            pack = ParadigmOutput(
                samples=output["samples"],
                logq_samples=output.get("logq_samples", None),
                logq_mcmc=output.get("logq_mcmc", None),
                path_logw=output.get("path_logw", None),
                training_metrics=output.get("training_metrics", None),
            )
        else:
            pack = ParadigmOutput(samples=output)

        metrics: Dict[str, Any] = {
            "paradigm": paradigm_name,
            "n_generated": int(self._to_tensor(pack.samples).shape[0]),
            "n_mcmc": int(self.mcmc_samples.shape[0]),
            "sample_dim": int(self.dim),
        }

        # Common metrics available for all.
        metrics.update(self.rnll_unnormalized(pack.samples))
        metrics.update(self.observable_gaps(pack.samples, pack.logq_samples))
        metrics.update(self.critical_observable_metrics(pack.samples, pack.logq_samples))

        # Site-average NLL if normalized log q on MCMC samples is available.
        if pack.logq_mcmc is not None:
            metrics.update(self.nll(pack.logq_mcmc))
        else:
            metrics["nll_per_site"] = None

        # RKL and density ESS if log q on generated samples is available.
        if pack.logq_samples is not None:
            metrics.update(self.rkl(pack.samples, pack.logq_samples))
            metrics.update(self.density_ess(pack.samples, pack.logq_samples))
        else:
            metrics["rkl_proxy"] = None
            metrics["density_ess_over_n"] = None
            metrics["density_ess_abs"] = None

        # Path ESS if path weights are available, e.g. NETS.
        if pack.path_logw is not None:
            metrics.update(self.path_ess(pack.path_logw))
        else:
            metrics["path_ess_over_n"] = None
            metrics["path_ess_abs"] = None

        # C2ST is common and likelihood-free.
        if compute_c2st:
            kwargs = c2st_kwargs or {}
            metrics.update(self.c2st(pack.samples, **kwargs))

        # Attach training metrics if provided.
        if pack.training_metrics is not None:
            for k, v in pack.training_metrics.items():
                metrics[f"train_{k}"] = v

        return metrics

    def evaluate(
        self,
        paradigm_outputs: Dict[str, Union[ParadigmOutput, Dict[str, Any], TensorLike]],
        compute_c2st: bool = True,
        c2st_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple paradigms.

        Example keys:
            "rkl_cnf", "ewfm", "nets", "idem".
        """
        results = {}

        for name, output in paradigm_outputs.items():
            results[name] = self.evaluate_one(
                output=output,
                paradigm_name=name,
                compute_c2st=compute_c2st,
                c2st_kwargs=c2st_kwargs,
            )

        return results



# --- Source notebook validation cell 52 ---

def make_observable_fn(cfg: Phi4Config):
    def observable_fn(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "magnetization": x.mean(dim=(-2, -1)),
            "energy": phi4_action(x, cfg),
            # "corr_r1": nearest_neighbor_corr(x),
            # "structure_factor": structure_factor_bins(x),
        }
    return observable_fn


def make_feature_fn(cfg: Phi4Config):
    def feature_fn(x: torch.Tensor) -> torch.Tensor:
        energy = phi4_action(x, cfg).reshape(-1, 1)
        mag = x.mean(dim=(-2, -1), keepdim=False).reshape(-1, 1)
        abs_mag = mag.abs()
        return torch.cat([energy, mag, abs_mag], dim=1)
    return feature_fn


def notebook_dir() -> Path:
    """Return the lattice_scaling_final directory containing MCMC_samples."""
    return Path(__file__).resolve().parents[1]


def find_mcmc_file_for_L(L: int, mcmc_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find the MCMC .npy file matching lattice size L."""
    mcmc_dir = Path(mcmc_dir) if mcmc_dir is not None else notebook_dir() / "MCMC_samples"

    preferred = mcmc_dir / f"phi4_L{L}_Singha2026_better.npy"
    if preferred.exists():
        return preferred
    default = mcmc_dir / f"phi4_L{L}_Singha2026.npy"
    if default.exists():
        return default
    matches = sorted(mcmc_dir.glob(f"phi4_L{L}_*.npy"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No MCMC sample file found for L={L} under {mcmc_dir}")


def load_mcmc_samples_for_L(
    L: int,
    mcmc_dir: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    convert_to_pipeline_units: bool = True,
) -> torch.Tensor:
    """Load MCMC samples as [N, 1, L, L] float tensor.

    Samples written by mcmc_phi4.py use the field convention
        S(y) = sum[(dy)^2 + m2*y^2 + lam*y^4].
    The generative pipeline uses
        S(x) = sum[0.5*(dx)^2 + 0.5*m2*x^2 + 0.25*lam*x^4].
    They describe the same Singha2026 target with x = sqrt(2)*y.
    Conversion is enabled by default so all validation is performed in
    the pipeline convention. Set convert_to_pipeline_units=False only
    when loading samples that are already stored in pipeline units.
    """
    path = find_mcmc_file_for_L(L, mcmc_dir=mcmc_dir)
    arr = np.load(path, mmap_mode="r")
    if max_samples is not None:
        arr = arr[:max_samples]
    # Scaling requires an owned array; the copy also avoids PyTorch's
    # non-writable warning for memory-mapped NumPy arrays.
    x = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
    field_scale = float(np.sqrt(2.0)) if convert_to_pipeline_units else 1.0
    if convert_to_pipeline_units:
        x.mul_(field_scale)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.shape[-2:] != (L, L):
        raise ValueError(f"Loaded MCMC samples from {path} have shape {tuple(x.shape)}, expected final lattice {L}x{L}.")
    if device is not None:
        x = x.to(torch.device(device))
    print({
        "loaded_mcmc": str(path),
        "shape": tuple(x.shape),
        "field_conversion": "mcmc_phi4 -> pipeline" if convert_to_pipeline_units else "none",
        "field_scale": field_scale,
    })
    return x


def make_default_validation_evaluator(
    mcmc_samples: TensorLike,
    cfg: Phi4Config,
    batch_size: int = 2048,
    normalize_by_dim: bool = True,
) -> FullLatticeSamplerEvaluator:
    """Create the default full-lattice evaluator.

    C2ST uses raw per-site values by default: feature_fn=None makes the
    evaluator flatten each [1, L, L] configuration to L^2 site features.
    This is a joint full-lattice test, not L^2 independent sitewise tests.
    Pass make_feature_fn(cfg) explicitly for the optional three-feature
    [energy, m, |m|] diagnostic.
    """
    return FullLatticeSamplerEvaluator(
        mcmc_samples=mcmc_samples,
        energy_fn=lambda x: phi4_action(x, cfg),
        observable_fn=make_observable_fn(cfg),
        # Default C2ST: retain every raw site as a separate classifier feature.
        feature_fn=None,
        normalize_by_dim=normalize_by_dim,
        batch_size=batch_size,
    )


def validation_scales_from_training_result(training_result: Dict[str, object]) -> List[int]:
    """Return all lattice sizes represented by a trained layerwise stack."""
    scales = [int(training_result["base_L"])]
    scales.extend(int(rec["L_fine"]) for rec in training_result["layers"])
    return scales


def make_validation_evaluators_by_scale(
    training_result: Dict[str, object],
    cfg: Phi4Config,
    mcmc_dir: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 2048,
    convert_mcmc_to_pipeline_units: bool = True,
) -> Tuple[Dict[int, FullLatticeSamplerEvaluator], Dict[int, torch.Tensor]]:
    """Load matching MCMC references and create one evaluator for each rollout scale."""
    evaluators: Dict[int, FullLatticeSamplerEvaluator] = {}
    mcmc_by_scale: Dict[int, torch.Tensor] = {}
    scales = validation_scales_from_training_result(training_result)
    print({"validation_scales": scales})
    for L in scales:
        # Keep full reference lattices on CPU; likelihood evaluation streams GPU batches.
        mcmc = load_mcmc_samples_for_L(
            L,
            mcmc_dir=mcmc_dir,
            max_samples=max_samples,
            device="cpu",
            convert_to_pipeline_units=convert_mcmc_to_pipeline_units,
        )
        mcmc_by_scale[L] = mcmc
        evaluators[L] = make_default_validation_evaluator(mcmc, cfg, batch_size=batch_size)
    return evaluators, mcmc_by_scale




# --- Source notebook validation cell 53 ---

def extract_layer_variable(fine: torch.Tensor, coarse: torch.Tensor, mode: str = "residual", cfg: Optional[Phi4Config] = None) -> torch.Tensor:
    """Extract the trained layer variable from a fine lattice and its even-even coarse ancestor."""
    return get_partition(cfg).extract(fine, coarse, mode=mode)


def decompose_lattice_stack(x: torch.Tensor, base_L: int, n_layers: int, mode: str = "residual", cfg: Optional[Phi4Config] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Decompose an external lattice into base field and per-layer (coarse, variable)."""
    if x.ndim == 3:
        x = x[:, None, :, :]
    expected_L = base_L * (2 ** n_layers)
    if x.shape[-1] != expected_L or x.shape[-2] != expected_L:
        raise ValueError(f"Expected {expected_L}x{expected_L} samples, got {tuple(x.shape)}.")

    fine_by_level: List[torch.Tensor] = [x]
    cur = x
    for _ in range(n_layers):
        cur = get_partition(cfg).coarsen_completed_lattice(cur)
        fine_by_level.append(cur)
    fine_by_level = list(reversed(fine_by_level))
    base = fine_by_level[0]

    layer_vars: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(n_layers):
        coarse = fine_by_level[layer_idx]
        fine = fine_by_level[layer_idx + 1]
        var = extract_layer_variable(fine, coarse, mode=mode, cfg=cfg)
        layer_vars.append((coarse, var))
    return base, layer_vars


def inverse_base_cnf_logq(model: BaseFieldCNF, x: torch.Tensor) -> torch.Tensor:
    """Approximate log q(x) by Euler-integrating a base CNF backward to its source."""
    z = x
    integral_div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    dt = 1.0 / model.n_steps
    for k in reversed(range(model.n_steps)):
        t = torch.full((z.shape[0],), (k + 0.5) * dt, device=z.device, dtype=z.dtype)
        z = z.detach().requires_grad_(True)
        v, div = compute_divergence(
            lambda x_, t_, _unused: model.velocity(x_, t_),
            z,
            t,
            None,
            mode=model.divergence_mode,
            hutchinson_samples=model.hutchinson_samples,
            create_graph=False,
        )
        integral_div = integral_div + dt * div
        z = z - dt * v
    return model.base_log_prob(z.detach()) - integral_div.detach()


def inverse_residual_cnf_logq(model: ResidualCNF, r: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    """Approximate log q(r|coarse) by Euler-integrating a residual CNF backward."""
    z = r
    integral_div = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
    dt = 1.0 / model.n_steps
    for k in reversed(range(model.n_steps)):
        t = torch.full((z.shape[0],), (k + 0.5) * dt, device=z.device, dtype=z.dtype)
        z = z.detach().requires_grad_(True)
        v, div = compute_divergence(
            model.velocity,
            z,
            t,
            coarse,
            mode=model.divergence_mode,
            hutchinson_samples=model.hutchinson_samples,
            create_graph=False,
        )
        integral_div = integral_div + dt * div
        z = z - dt * v
    return model.base_log_prob(z.detach()) - integral_div.detach()


def inverse_base_velocity_logq(velocity: BaseFieldVelocityMLP, x: torch.Tensor, cfg: Phi4Config, n_steps: Optional[int] = None) -> torch.Tensor:
    """Approximate log q(x) for a velocity-trained base ODE by backward Euler integration."""
    z = x
    n_steps = n_steps or cfg.cnf_steps
    dt = 1.0 / n_steps
    integral_div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    old_flags = _set_module_requires_grad(velocity, False)
    was_training = velocity.training
    velocity.eval()
    try:
        for k in reversed(range(n_steps)):
            t = torch.full((z.shape[0],), (k + 0.5) * dt, device=z.device, dtype=z.dtype)
            z = z.detach().requires_grad_(True)
            v, div = compute_divergence(
                lambda x_, t_, _unused: velocity(x_, t_),
                z,
                t,
                None,
                mode=cfg.cnf_divergence_mode,
                hutchinson_samples=cfg.hutchinson_samples,
                create_graph=False,
            )
            integral_div = integral_div + dt * div
            z = z - dt * v
    finally:
        velocity.train(was_training)
        _restore_module_requires_grad(velocity, old_flags)
    return initial_log_prob(z.detach(), cfg, role="base") - integral_div.detach()


def inverse_residual_velocity_logq(velocity: ResidualVelocityMLP, r: torch.Tensor, coarse: torch.Tensor, cfg: Phi4Config, n_steps: Optional[int] = None) -> torch.Tensor:
    """Approximate log q(r|coarse) for a velocity-trained residual ODE by backward Euler integration."""
    z = r
    n_steps = n_steps or cfg.cnf_steps
    dt = 1.0 / n_steps
    integral_div = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
    old_flags = _set_module_requires_grad(velocity, False)
    was_training = velocity.training
    velocity.eval()
    try:
        for k in reversed(range(n_steps)):
            t = torch.full((z.shape[0],), (k + 0.5) * dt, device=z.device, dtype=z.dtype)
            z = z.detach().requires_grad_(True)
            v, div = compute_divergence(
                velocity,
                z,
                t,
                coarse,
                mode=cfg.cnf_divergence_mode,
                hutchinson_samples=cfg.hutchinson_samples,
                create_graph=False,
            )
            integral_div = integral_div + dt * div
            z = z - dt * v
    finally:
        velocity.train(was_training)
        _restore_module_requires_grad(velocity, old_flags)
    return initial_log_prob(z.detach(), cfg, role="residual") - integral_div.detach()


def inverse_rkl_cnf_stack_logq(
    training_result: Dict[str, object],
    cfg: Phi4Config,
    x: torch.Tensor,
    mode: str = "residual",
) -> torch.Tensor:
    """Invert relaxation before residual decomposition at every hierarchy level."""
    current = x
    relaxation_logq = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    reversed_layers: List[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]] = []

    for rec in reversed(training_result["layers"]):
        relaxation = rec.get("relaxation")
        if relaxation is not None:
            current, logq_delta = inverse_rg_relaxation_cnf(relaxation, current)
            relaxation_logq = relaxation_logq + logq_delta
        coarse = get_partition(cfg).coarsen_completed_lattice(current)
        residual = extract_layer_variable(current, coarse, mode=mode, cfg=cfg)
        reversed_layers.append((rec, coarse, residual))
        current = coarse

    logq = inverse_base_cnf_logq(training_result["base_model"], current)
    for rec, coarse, residual in reversed(reversed_layers):
        logq = logq + inverse_residual_cnf_logq(rec["model"], residual, coarse)
    return logq + relaxation_logq


def ode_logq_on_external_lattice(training_result: Dict[str, object], cfg: Phi4Config, x: torch.Tensor, mode: str = "residual") -> torch.Tensor:
    """Evaluate approximate log q_theta(x) for external samples under an ODE-trained stack."""
    paradigm = training_result["paradigm"]
    if paradigm not in {"rkl_cnf", "ewfm", "rfm"}:
        raise ValueError("External logq is only available for ODE paradigms: rkl_cnf, ewfm, rfm.")
    x = x.to(torch.device(cfg.device))
    base_L = int(training_result["base_L"])
    layers = training_result["layers"]
    if paradigm == "rkl_cnf" and any(rec.get("relaxation") is not None for rec in layers):
        return inverse_rkl_cnf_stack_logq(training_result, cfg, x, mode=mode).detach().cpu()
    base_x, layer_vars = decompose_lattice_stack(x, base_L, len(layers), mode=mode, cfg=cfg)

    if paradigm == "rkl_cnf":
        logq = inverse_base_cnf_logq(training_result["base_model"], base_x)
        for rec, (coarse, r) in zip(layers, layer_vars):
            logq = logq + inverse_residual_cnf_logq(rec["model"], r, coarse)
        return logq.detach().cpu()

    logq = inverse_base_velocity_logq(training_result["base_model"], base_x, cfg)
    for rec, (coarse, r) in zip(layers, layer_vars):
        logq = logq + inverse_residual_velocity_logq(rec["model"], r, coarse, cfg)
    return logq.detach().cpu()


def external_logq_by_scale_for_ode(training_result: Dict[str, object], cfg: Phi4Config, mcmc_by_scale: Optional[Dict[int, torch.Tensor]], mode: str = "residual", likelihood_batch_size: int = 8) -> Dict[int, torch.Tensor]:
    """Compute external log q in CPU-backed batches at every rollout scale.

    Only one batch is moved to the GPU at a time. Each result is detached
    and returned to CPU before processing the next batch.
    """
    if mcmc_by_scale is None or training_result["paradigm"] not in {"rkl_cnf", "ewfm", "rfm"}:
        return {}
    if likelihood_batch_size <= 0:
        raise ValueError("likelihood_batch_size must be positive.")
    out: Dict[int, torch.Tensor] = {}
    full_layers = training_result["layers"]
    for L, x_mcmc in mcmc_by_scale.items():
        n_layers = int(round(math.log2(int(L) / int(training_result["base_L"]))))
        if n_layers < 0 or int(training_result["base_L"]) * (2 ** n_layers) != int(L):
            continue
        partial = dict(training_result)
        partial["layers"] = full_layers[:n_layers]
        partial["final_L"] = int(L)
        logq_batches: List[torch.Tensor] = []
        for start in range(0, x_mcmc.shape[0], likelihood_batch_size):
            x_batch_cpu = x_mcmc[start:start + likelihood_batch_size].detach().cpu()
            with torch.enable_grad():
                logq_batch_cpu = ode_logq_on_external_lattice(partial, cfg, x_batch_cpu, mode=mode)
            logq_batches.append(logq_batch_cpu.detach().cpu())
        out[int(L)] = torch.cat(logq_batches, dim=0)
    return out


def selected_training_record(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Return the history row corresponding to the weights restored before saving."""
    history = getattr(model, "training_history", None)
    if not history:
        return None
    selection = getattr(model, "training_selection", None) or {}
    best_step = selection.get("best_step")
    if best_step is not None:
        for record in reversed(history):
            if int(record.get("step", -1)) == int(best_step):
                return record
    return history[-1]


def collect_training_metrics(training_result: Dict[str, object], upto_layer_idx: Optional[int] = None) -> Dict[str, Any]:
    """Collect metrics for the selected/restored weights through a layer index."""
    metrics: Dict[str, Any] = {}
    base_model = training_result["base_model"]
    base_record = selected_training_record(base_model)
    if base_record:
        for k, v in base_record.items():
            metrics[f"base_{k}"] = v
        for k, v in (getattr(base_model, "training_selection", None) or {}).items():
            metrics[f"base_selection_{k}"] = v
    for k, v in (getattr(base_model, "training_resources", None) or {}).items():
        metrics[f"base_resource_{k}"] = v

    for rec in training_result["layers"]:
        if upto_layer_idx is not None and rec["idx"] > upto_layer_idx:
            continue
        layer_model = rec["model"]
        layer_record = selected_training_record(layer_model)
        if layer_record:
            for k, v in layer_record.items():
                metrics[f"layer{rec['idx']}_{k}"] = v
            for k, v in (getattr(layer_model, "training_selection", None) or {}).items():
                metrics[f"layer{rec['idx']}_selection_{k}"] = v
        for k, v in (getattr(layer_model, "training_resources", None) or {}).items():
            metrics[f"layer{rec['idx']}_resource_{k}"] = v
        relaxation_model = rec.get("relaxation")
        relaxation_record = selected_training_record(relaxation_model) if relaxation_model is not None else None
        if relaxation_record:
            for k, v in relaxation_record.items():
                metrics[f"relax{rec['idx']}_{k}"] = v
            for k, v in (getattr(relaxation_model, "training_selection", None) or {}).items():
                metrics[f"relax{rec['idx']}_selection_{k}"] = v
        if relaxation_model is not None:
            for k, v in (getattr(relaxation_model, "training_resources", None) or {}).items():
                metrics[f"relax{rec['idx']}_resource_{k}"] = v
    return metrics


def output_key(paradigm: str, L: int) -> str:
    return f"{paradigm}_L{int(L)}"


def rollout_rkl_cnf_multiscale_for_validation(training_result: Dict[str, object], cfg: Phi4Config, batch_size: Optional[int] = None, mode: str = "residual", logq_mcmc_by_scale: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, ParadigmOutput]:
    """Roll out RKL-CNF and record samples plus cumulative log q at every scale."""
    base_model = training_result["base_model"]
    batch_size = batch_size or cfg.batch_size
    device = torch.device(cfg.device)
    outputs: Dict[int, ParadigmOutput] = {}

    with torch.enable_grad():
        z = base_model.sample_base(batch_size, device=device)
        x, logq_total = base_model(z, compute_logq=True, create_graph=False)
        base_L = int(training_result["base_L"])
        outputs[base_L] = ParadigmOutput(
            samples=x.detach(),
            logq_samples=logq_total.detach(),
            logq_mcmc=None if logq_mcmc_by_scale is None else logq_mcmc_by_scale.get(base_L),
            training_metrics=collect_training_metrics(training_result, upto_layer_idx=-1),
        )

        for rec in training_result["layers"]:
            layer = rec["model"]
            z = layer.sample_base(batch_size, device=x.device)
            r, logq_layer = layer(z, x, compute_logq=True, create_graph=False)
            x = insert_residual(x, r, mode=mode)
            logq_total = logq_total + logq_layer
            relaxation = rec.get("relaxation")
            if relaxation is not None:
                x, logq_relax = relaxation(x, compute_logq=True, create_graph=False)
                logq_total = logq_total + logq_relax
            outputs[int(rec["L_fine"])] = ParadigmOutput(
                samples=x.detach(),
                logq_samples=logq_total.detach(),
                logq_mcmc=None if logq_mcmc_by_scale is None else logq_mcmc_by_scale.get(int(rec["L_fine"])),
                training_metrics=collect_training_metrics(training_result, upto_layer_idx=rec["idx"]),
            )
    return outputs


def rollout_velocity_ode_multiscale_for_validation(training_result: Dict[str, object], cfg: Phi4Config, batch_size: Optional[int] = None, mode: str = "residual", logq_mcmc_by_scale: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, ParadigmOutput]:
    """Roll out an EWFM/RFM velocity ODE and record cumulative log q at every scale."""
    base_velocity = training_result["base_model"]
    batch_size = batch_size or cfg.batch_size
    device = torch.device(cfg.device)
    outputs: Dict[int, ParadigmOutput] = {}

    with torch.enable_grad():
        z = sample_initial_field(cfg, batch_size, 1, base_velocity.L, device=device, role="base")
        x, logq_total = integrate_base_velocity_with_logq(base_velocity, z, cfg, create_graph=False)
        base_L = int(training_result["base_L"])
        outputs[base_L] = ParadigmOutput(
            samples=x.detach(),
            logq_samples=logq_total.detach(),
            logq_mcmc=None if logq_mcmc_by_scale is None else logq_mcmc_by_scale.get(base_L),
            training_metrics=collect_training_metrics(training_result, upto_layer_idx=-1),
        )

        for rec in training_result["layers"]:
            velocity = rec["model"]
            _, _, L, _ = x.shape
            z = sample_initial_field(cfg, batch_size, residual_channels_for_L(L, cfg), L, device=x.device, dtype=x.dtype)
            r, logq_layer = integrate_residual_velocity_with_logq(velocity, z, x, cfg, create_graph=False)
            x = insert_residual(x, r, mode=mode)
            logq_total = logq_total + logq_layer
            outputs[int(rec["L_fine"])] = ParadigmOutput(
                samples=x.detach(),
                logq_samples=logq_total.detach(),
                logq_mcmc=None if logq_mcmc_by_scale is None else logq_mcmc_by_scale.get(int(rec["L_fine"])),
                training_metrics=collect_training_metrics(training_result, upto_layer_idx=rec["idx"]),
            )
    return outputs


rollout_ewfm_multiscale_for_validation = rollout_velocity_ode_multiscale_for_validation


@torch.no_grad()
def rollout_idem_multiscale_for_validation(training_result: Dict[str, object], cfg: Phi4Config, batch_size: Optional[int] = None, mode: str = "residual") -> Dict[int, ParadigmOutput]:
    """Roll out iDEM and record samples at every scale. No global log q/path IS is produced."""
    batch_size = batch_size or cfg.batch_size
    outputs: Dict[int, ParadigmOutput] = {}
    base_out = sample_idem_base_model(training_result["base_model"], cfg, batch_size=batch_size)
    x = base_out["samples"]
    base_L = int(training_result["base_L"])
    outputs[base_L] = ParadigmOutput(
        samples=x.detach(),
        training_metrics=collect_training_metrics(training_result, upto_layer_idx=-1),
    )

    for rec in training_result["layers"]:
        x = sample_idem_residual_layer(rec["model"], x, cfg, mode=mode)["fine"]
        outputs[int(rec["L_fine"])] = ParadigmOutput(
            samples=x.detach(),
            training_metrics=collect_training_metrics(training_result, upto_layer_idx=rec["idx"]),
        )
    return outputs


@torch.no_grad()
def rollout_nets_multiscale_for_validation(training_result: Dict[str, object], cfg: Phi4Config, batch_size: Optional[int] = None, mode: str = "residual") -> Dict[int, ParadigmOutput]:
    """Roll out one complete NETS ancestry and return target-corrected path weights.

    Raw local Jarzynski log weights multiply across conditional layers, so
    they are accumulated by addition along each particle's ancestry. The
    local endpoint energies define a hierarchical surrogate action
        S_hier = S_base + sum_l E_l.
    At every completed scale, path ESS must instead target the ordinary
    phi4 action S_L, hence
        logw_target = logw_path_total + S_hier - S_L.
    The normalized weights used inside NETS training are never accumulated.
    """
    batch_size = batch_size or cfg.batch_size
    outputs: Dict[int, ParadigmOutput] = {}
    base_out = sample_nets_langevin_base_model(training_result["base_model"], cfg, batch_size=batch_size)
    x = base_out["samples"]
    path_logw_total = base_out["path_logw"]
    hierarchical_action = base_out["action"].detach()
    target_action = phi4_action(x, cfg).detach()
    corrected_logw = path_logw_total + hierarchical_action - target_action
    base_L = int(training_result["base_L"])
    metrics = collect_training_metrics(training_result, upto_layer_idx=-1)
    metrics["path_logw_var"] = float(torch.var(corrected_logw.detach(), unbiased=False).cpu())
    outputs[base_L] = ParadigmOutput(samples=x.detach(), path_logw=corrected_logw.detach(), training_metrics=metrics)

    for rec in training_result["layers"]:
        out = sample_nets_langevin_residual_layer(rec["model"], x, cfg, mode=mode)
        x = out["fine"]
        path_logw_total = path_logw_total + out["path_logw"]
        hierarchical_action = hierarchical_action + out["energy"].detach()
        target_action = phi4_action(x, cfg).detach()
        corrected_logw = path_logw_total + hierarchical_action - target_action
        metrics = collect_training_metrics(training_result, upto_layer_idx=rec["idx"])
        metrics["path_logw_var"] = float(torch.var(corrected_logw.detach(), unbiased=False).cpu())
        outputs[int(rec["L_fine"])] = ParadigmOutput(samples=x.detach(), path_logw=corrected_logw.detach(), training_metrics=metrics)
    return outputs


def build_multiscale_paradigm_outputs_from_training_result(training_result: Dict[str, object], cfg: Phi4Config, batch_size: Optional[int] = None, mode: str = "residual", mcmc_by_scale: Optional[Dict[int, torch.Tensor]] = None, likelihood_batch_size: int = 8, generation_batch_size: Optional[int] = None) -> Dict[str, ParadigmOutput]:
    """Convert a trained stack into evaluator inputs, generating paths in GPU-safe chunks."""
    paradigm = training_result["paradigm"]
    total_batch_size = int(batch_size or cfg.batch_size)
    generation_batch_size = int(generation_batch_size or total_batch_size)
    if total_batch_size <= 0 or generation_batch_size <= 0:
        raise ValueError("batch_size and generation_batch_size must be positive.")
    generation_batch_size = min(generation_batch_size, total_batch_size)
    logq_mcmc_by_scale = external_logq_by_scale_for_ode(
        training_result, cfg, mcmc_by_scale, mode=mode, likelihood_batch_size=likelihood_batch_size
    )

    def rollout_once(chunk_size: int, external_logq: Optional[Dict[int, torch.Tensor]]) -> Dict[int, ParadigmOutput]:
        if paradigm == "rkl_cnf":
            return rollout_rkl_cnf_multiscale_for_validation(training_result, cfg, batch_size=chunk_size, mode=mode, logq_mcmc_by_scale=external_logq)
        if paradigm in {"ewfm", "rfm"}:
            return rollout_velocity_ode_multiscale_for_validation(training_result, cfg, batch_size=chunk_size, mode=mode, logq_mcmc_by_scale=external_logq)
        if paradigm == "idem":
            return rollout_idem_multiscale_for_validation(training_result, cfg, batch_size=chunk_size, mode=mode)
        if paradigm == "nets":
            return rollout_nets_multiscale_for_validation(training_result, cfg, batch_size=chunk_size, mode=mode)
        raise ValueError(f"Unknown trained paradigm: {paradigm}")

    if generation_batch_size == total_batch_size:
        by_scale = rollout_once(total_batch_size, logq_mcmc_by_scale)
        return {output_key(paradigm, L): out for L, out in by_scale.items()}

    chunks_by_scale: Dict[int, List[ParadigmOutput]] = {}
    completed = 0
    while completed < total_batch_size:
        chunk_size = min(generation_batch_size, total_batch_size - completed)
        # External-MCMC log q is computed once above, not once per generated chunk.
        chunk_outputs = rollout_once(chunk_size, {})
        for L, out in chunk_outputs.items():
            chunks_by_scale.setdefault(int(L), []).append(ParadigmOutput(
                samples=torch.as_tensor(out.samples).detach().cpu(),
                logq_samples=None if out.logq_samples is None else torch.as_tensor(out.logq_samples).detach().cpu(),
                path_logw=None if out.path_logw is None else torch.as_tensor(out.path_logw).detach().cpu(),
                training_metrics=out.training_metrics,
            ))
        del out, chunk_outputs  # Release this chunk's GPU tensors before starting the next one.
        completed += chunk_size
        if completed == total_batch_size or completed % max(generation_batch_size, total_batch_size // 10) == 0:
            print({"validation_generated": completed, "validation_total": total_batch_size})

    by_scale: Dict[int, ParadigmOutput] = {}
    for L, chunks in chunks_by_scale.items():
        def concatenate_optional(field: str) -> Optional[torch.Tensor]:
            values = [getattr(chunk, field) for chunk in chunks]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise RuntimeError(f"Inconsistent {field} availability across validation chunks.")
            return torch.cat(values, dim=0)

        by_scale[L] = ParadigmOutput(
            samples=torch.cat([torch.as_tensor(chunk.samples) for chunk in chunks], dim=0),
            logq_samples=concatenate_optional("logq_samples"),
            logq_mcmc=logq_mcmc_by_scale.get(L),
            path_logw=concatenate_optional("path_logw"),
            training_metrics=chunks[0].training_metrics,
        )
    return {output_key(paradigm, L): out for L, out in by_scale.items()}




# --- Source notebook validation cell 54 ---

def scale_from_output_key(name: str) -> int:
    """Parse lattice scale from keys like 'rkl_cnf_L32'."""
    return int(name.rsplit("_L", 1)[1])




# --- Source notebook validation cell 55 ---

def validation_metric_names(results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return scalar metric names present in validation results."""
    names = set()
    for metrics in results.values():
        for key, value in metrics.items():
            if key == "paradigm" or value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                names.add(key)
    return sorted(names)


def validation_results_to_series(results: Dict[str, Dict[str, Any]], metric: str) -> Tuple[List[int], List[float], List[str]]:
    """Extract one metric as y(L), ordered by lattice size."""
    rows = []
    for name, metrics in results.items():
        value = metrics.get(metric, None)
        if value is None:
            continue
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        rows.append((scale_from_output_key(name), float(value), name))
    rows.sort(key=lambda row: row[0])
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def plot_validation_metric(
    results: Dict[str, Dict[str, Any]],
    metric: str,
    ax=None,
    logy: bool = False,
    marker: str = "o",
):
    """Plot one validation metric versus lattice size L."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    Ls, ys, _ = validation_results_to_series(results, metric)
    if not Ls:
        ax.set_axis_off()
        ax.set_title(f"{metric} unavailable")
        return ax
    ax.plot(Ls, ys, marker=marker)
    ax.set_xlabel("lattice size L")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.set_xticks(Ls)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    return ax


def plot_validation_diagnostics(
    results: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    log_metrics: Optional[set] = None,
    ncols: int = 2,
):
    """Plot validation metrics across all evaluated lattice sizes."""
    if results is None:
        raise ValueError("results is None; run validation evaluation first.")
    available = set(validation_metric_names(results))
    if metrics is None:
        preferred = [
            "nll_per_site",
            "rnll_unnormalized_per_site",
            "action_gap_per_site_gen_minus_mcmc",
            "obs_energy_mean_abs_z_gap",
            "obs_energy_is_mean_abs_z_gap",
            "density_ess_over_n",
            "path_ess_over_n",
            "c2st_acc_mean",
            "c2st_auc_mean",
            "obs_magnetization_mean_abs_gap",
            "obs_magnetization_is_mean_abs_gap",
            "obs_susceptibility_abs_gap",
            "obs_susceptibility_is_abs_gap",
            "obs_binder_cumulant_abs_gap",
            "obs_binder_cumulant_is_abs_gap",
            "density_logw_std",
            "path_logw_std",
        ]
        metrics = [m for m in preferred if m in available]
    else:
        metrics = [m for m in metrics if m in available]
    if not metrics:
        raise ValueError(f"No requested validation metrics available. Available: {sorted(available)}")

    log_metrics = log_metrics or set()
    ncols = max(1, ncols)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        plot_validation_metric(results, metric, ax=ax, logy=(metric in log_metrics))
    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def run_full_validation(
    training_result: Dict[str, object],
    cfg: Phi4Config,
    *,
    mcmc_dir: Optional[Union[str, Path]] = None,
    n_generated: int = 10000,
    generation_batch_size: int = 8,
    likelihood_batch_size: int = 4,
    evaluator_batch_size: int = 2048,
    mcmc_max_samples: Optional[int] = None,
    convert_mcmc_to_pipeline_units: bool = True,
    sampling_seed: int = 482014,
    compute_mcmc_logq: bool = False,
    cnf_divergence_mode: Optional[str] = "exact",
    hutchinson_samples: Optional[int] = None,
    compute_c2st: bool = True,
    c2st_kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "residual",
) -> Dict[str, Any]:
    """Run the notebook's complete multiscale validation as one reusable call.

    This loads an MCMC reference for every represented scale, generates complete
    paths, evaluates raw and importance-weighted physical observables, density
    or path ESS when available, and C2ST. ``compute_mcmc_logq=False`` matches
    the current notebook protocol, where NLL on external MCMC configurations is
    omitted to avoid a second expensive exact-divergence pass.
    """
    n_generated = int(n_generated)
    generation_batch_size = int(generation_batch_size)
    likelihood_batch_size = int(likelihood_batch_size)
    if n_generated < 1:
        raise ValueError("n_generated must be positive.")
    if generation_batch_size < 1 or likelihood_batch_size < 1:
        raise ValueError("Validation batch sizes must be positive.")

    if (
        training_result["paradigm"] == "rkl_cnf"
        and cnf_divergence_mode is not None
    ):
        set_cnf_stack_divergence(
            training_result,
            mode=cnf_divergence_mode,
            hutchinson_samples=hutchinson_samples,
        )

    torch.manual_seed(int(sampling_seed))
    np.random.seed(int(sampling_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))

    evaluators_by_scale, mcmc_by_scale = make_validation_evaluators_by_scale(
        training_result,
        cfg,
        mcmc_dir=mcmc_dir,
        max_samples=mcmc_max_samples,
        batch_size=int(evaluator_batch_size),
        convert_mcmc_to_pipeline_units=convert_mcmc_to_pipeline_units,
    )
    paradigm_outputs = build_multiscale_paradigm_outputs_from_training_result(
        training_result,
        cfg,
        batch_size=n_generated,
        mode=mode,
        mcmc_by_scale=mcmc_by_scale if compute_mcmc_logq else None,
        likelihood_batch_size=likelihood_batch_size,
        generation_batch_size=generation_batch_size,
    )

    classifier_options = c2st_kwargs or {
        "max_samples_per_class": 3000,
        "n_repeats": 5,
        "classifier": "logreg",
    }
    results: Dict[str, Dict[str, Any]] = {}
    for name, output in paradigm_outputs.items():
        L = scale_from_output_key(name)
        print(
            {
                "evaluating": name,
                "L": L,
                "n_generated": int(torch.as_tensor(output.samples).shape[0]),
            }
        )
        results[name] = evaluators_by_scale[L].evaluate_one(
            output=output,
            paradigm_name=name,
            compute_c2st=compute_c2st,
            c2st_kwargs=classifier_options,
        )

    print("validation results:")
    for name, metrics in results.items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return {
        "results": results,
        "paradigm_outputs": paradigm_outputs,
        "evaluators_by_scale": evaluators_by_scale,
        "mcmc_by_scale": mcmc_by_scale,
        "sampling_seed": int(sampling_seed),
        "n_generated": n_generated,
        "generation_batch_size": generation_batch_size,
        "likelihood_batch_size": likelihood_batch_size,
        "compute_mcmc_logq": bool(compute_mcmc_logq),
        "cnf_divergence_mode": cnf_divergence_mode,
    }
