"""
mcmc_phi4.py — Metropolis MCMC reference sampler for 1-component phi^4 theory.

Generates equilibrated field configurations that serve as the ground-truth
reference distribution p(x) ∝ exp(-S[phi]) for:
  - Comparing model samples (P(M), G(r), moments, Binder cumulant)
  - Computing R-NLL: evaluating log q(x) at x ~ p
  - Estimating physical observables (susceptibility, correlation length)

Algorithm: Hybrid Monte Carlo (HMC) by default.
  HMC proposes global updates using molecular dynamics, which is far more
  efficient than local Metropolis for correlated lattice fields. The
  acceptance rate stays high (~70-90%) even for large lattices.
  Local Metropolis (Metropolis-adjusted Langevin, single-site) is also
  provided as a fallback and for comparison.

Hardware: CPU is standard and sufficient for reference generation.
  See bottom of file for a detailed GPU/CPU discussion.

Usage from another script or notebook:
    from lattice_scaling_final.training_modules.mcmc_phi4 import (
        MCMCSampler,
        run_hmc,
        run_local_metropolis,
    )
"""

import time
import numpy as np
import torch
import os


# ---------------------------------------------------------------------------
# phi^4 action
# ---------------------------------------------------------------------------

class ScalarPhi4Action:
    """
    S[phi] = sum_x [ M^2 phi^2 + lam phi^4 + sum_mu (phi_x - phi_{x+mu})^2 ]
           = sum_x [ M^2 phi^2 + lam phi^4 + 2*phi^2 - phi*roll(phi,-1,mu) - phi*roll(phi,1,mu) ]

    The kinetic term (nearest-neighbour coupling) enforces spatial correlations.
    Returned shape: (batch,) — one action value per configuration.
    """
    def __init__(self, M2: float, lam: float):
        self.M2  = M2
        self.lam = lam

    def __call__(self, cfgs: torch.Tensor) -> torch.Tensor:
        """cfgs: (batch, L, L) or (L, L). Returns (batch,) or scalar."""
        squeeze = cfgs.ndim == 2
        if squeeze:
            cfgs = cfgs.unsqueeze(0)

        rho = self.M2 * cfgs**2 + self.lam * cfgs**4
        # Kinetic term: sum over both spatial directions
        for mu in (1, 2):
            rho = rho + 2 * cfgs**2
            rho = rho - cfgs * torch.roll(cfgs, -1, mu)
            rho = rho - cfgs * torch.roll(cfgs,  1, mu)

        S = rho.sum(dim=(1, 2))
        return S.squeeze(0) if squeeze else S

    def grad(self, phi: torch.Tensor) -> torch.Tensor:
        """
        dS/dphi_x — gradient used in HMC leapfrog.
        phi: (L, L) or (batch, L, L). Returns same shape.
        """
        squeeze = phi.ndim == 2
        if squeeze:
            phi = phi.unsqueeze(0)

        g  = 2 * self.M2 * phi + 4 * self.lam * phi**3
        for mu in (1, 2):
            g = g + 4 * phi
            g = g - torch.roll(phi, -1, mu) - torch.roll(phi,  1, mu)
            g = g - torch.roll(phi, -1, mu) - torch.roll(phi,  1, mu)
        # Correct: dS/dphi_x from kinetic = 4*phi_x - roll(-1) - roll(+1) per dim
        # Redo cleanly:
        g2 = 2 * self.M2 * phi + 4 * self.lam * phi**3
        for mu in (1, 2):
            g2 = g2 + 4 * phi
            g2 = g2 - torch.roll(phi, -1, mu)
            g2 = g2 - torch.roll(phi,  1, mu)

        result = g2.squeeze(0) if squeeze else g2
        return result


# Correct, cleaner gradient implementation
def _phi4_action_and_grad(phi: torch.Tensor, M2: float, lam: float):
    """
    Compute S[phi] and dS/dphi simultaneously via autograd.
    phi: (L, L), requires_grad should be set by caller.
    Returns (S scalar, grad (L,L)).
    """
    rho = M2 * phi**2 + lam * phi**4
    for mu in (0, 1):
        rho = rho + 2 * phi**2 - phi * torch.roll(phi, -1, mu) - phi * torch.roll(phi, 1, mu)
    S = rho.sum()
    S.backward()
    return S.detach(), phi.grad.detach().clone()


# ---------------------------------------------------------------------------
# HMC sampler
# ---------------------------------------------------------------------------

def hmc_step(phi: torch.Tensor, M2: float, lam: float,
             n_leapfrog: int = 10, step_size: float = 0.1,
             device: str = 'cpu') -> tuple[torch.Tensor, bool]:
    """
    One HMC proposal-accept step for a single configuration.

    Algorithm:
      1. Draw momenta pi ~ N(0, I)
      2. Leapfrog integrate Hamilton's equations for n_leapfrog steps
      3. Metropolis accept/reject using ΔH = H_proposed - H_current

    The Hamiltonian is H[phi, pi] = S[phi] + 0.5 * ||pi||^2.
    HMC leaves p(phi) ∝ exp(-S[phi]) invariant by construction.

    Args:
        phi:         current configuration (L, L)
        M2, lam:     phi^4 parameters
        n_leapfrog:  number of leapfrog steps
        step_size:   leapfrog step size (tune so acceptance ~0.7-0.9)
        device:      'cpu' or 'cuda'
    Returns:
        phi_new:  new configuration (L, L), either proposed or current
        accepted: bool
    """
    phi = phi.to(device)

    # Draw momenta
    pi = torch.randn_like(phi)

    # Current Hamiltonian
    phi_curr = phi.clone().requires_grad_(True)
    with torch.enable_grad():
        rho = M2 * phi_curr**2 + lam * phi_curr**4
        for mu in (0, 1):
            rho = rho + 2 * phi_curr**2
            rho = rho - phi_curr * torch.roll(phi_curr, -1, mu)
            rho = rho - phi_curr * torch.roll(phi_curr,  1, mu)
        S_curr = rho.sum()
        S_curr.backward()
        grad_curr = phi_curr.grad.detach().clone()

    H_curr = S_curr.detach() + 0.5 * (pi**2).sum()

    # Leapfrog integration
    phi_prop = phi.clone()
    pi_prop  = pi.clone()
    grad     = grad_curr.clone()

    # Half step for momenta
    pi_prop = pi_prop - 0.5 * step_size * grad

    for _ in range(n_leapfrog):
        # Full step for position
        phi_prop = phi_prop + step_size * pi_prop

        # Compute gradient at new position
        phi_prop_g = phi_prop.clone().requires_grad_(True)
        with torch.enable_grad():
            rho = M2 * phi_prop_g**2 + lam * phi_prop_g**4
            for mu in (0, 1):
                rho = rho + 2 * phi_prop_g**2
                rho = rho - phi_prop_g * torch.roll(phi_prop_g, -1, mu)
                rho = rho - phi_prop_g * torch.roll(phi_prop_g,  1, mu)
            S_prop = rho.sum()
            S_prop.backward()
        grad = phi_prop_g.grad.detach().clone()

        # Full step for momenta (except last)
        pi_prop = pi_prop - step_size * grad

    # Final half step for momenta
    pi_prop = pi_prop + 0.5 * step_size * grad  # undo the last full step's extra half

    # Proposed Hamiltonian
    phi_p2 = phi_prop.clone().requires_grad_(False)
    with torch.no_grad():
        rho = M2 * phi_p2**2 + lam * phi_p2**4
        for mu in (0, 1):
            rho = rho + 2 * phi_p2**2
            rho = rho - phi_p2 * torch.roll(phi_p2, -1, mu)
            rho = rho - phi_p2 * torch.roll(phi_p2,  1, mu)
        S_prop_val = rho.sum()

    H_prop = S_prop_val + 0.5 * (pi_prop**2).sum()

    # Metropolis accept/reject
    delta_H = (H_prop - H_curr).item()
    accept   = (delta_H < 0) or (torch.rand(1).item() < np.exp(-delta_H))

    return (phi_prop if accept else phi.clone()), accept


# ---------------------------------------------------------------------------
# HMC step at arbitrary inverse temperature (used by parallel tempering)
# ---------------------------------------------------------------------------

def _hmc_step_beta(phi: torch.Tensor, M2: float, lam: float,
                   beta: float, n_leapfrog: int, step_size: float,
                   device: str = 'cpu') -> tuple[torch.Tensor, bool, float]:
    """
    One HMC step at inverse temperature beta.

    Samples from p_beta(phi) ∝ exp(-beta * S[phi]).
    At beta=1 this is identical to hmc_step(); at beta<1 the landscape
    is smoother and the chain can tunnel between modes.

    Returns:
        phi_new:    new configuration (L, L)
        accepted:   bool
        S_unscaled: unscaled S[phi_new] — needed for replica swap acceptance
    """
    phi = phi.to(device)
    pi  = torch.randn_like(phi)

    # Gradient at current position, scaled by beta
    phi_curr = phi.clone().requires_grad_(True)
    with torch.enable_grad():
        rho = M2 * phi_curr**2 + lam * phi_curr**4
        for mu in (0, 1):
            rho = rho + 2 * phi_curr**2
            rho = rho - phi_curr * torch.roll(phi_curr, -1, mu)
            rho = rho - phi_curr * torch.roll(phi_curr,  1, mu)
        S_curr = rho.sum()
        S_curr.backward()
        grad = phi_curr.grad.detach().clone() * beta

    H_curr = beta * S_curr.detach() + 0.5 * (pi**2).sum()

    # Leapfrog integration with beta-scaled gradient
    phi_p = phi.clone()
    pi_p  = pi.clone()
    pi_p  = pi_p - 0.5 * step_size * grad           # initial half step

    for _ in range(n_leapfrog):
        phi_p = phi_p + step_size * pi_p             # full position step

        phi_g = phi_p.clone().requires_grad_(True)
        with torch.enable_grad():
            rho = M2 * phi_g**2 + lam * phi_g**4
            for mu in (0, 1):
                rho = rho + 2 * phi_g**2
                rho = rho - phi_g * torch.roll(phi_g, -1, mu)
                rho = rho - phi_g * torch.roll(phi_g,  1, mu)
            S_p = rho.sum()
            S_p.backward()
        grad = phi_g.grad.detach().clone() * beta

        pi_p = pi_p - step_size * grad               # full momentum step

    pi_p = pi_p + 0.5 * step_size * grad             # final half step (correct leapfrog)

    # Proposed Hamiltonian
    with torch.no_grad():
        rho = M2 * phi_p**2 + lam * phi_p**4
        for mu in (0, 1):
            rho = rho + 2 * phi_p**2
            rho = rho - phi_p * torch.roll(phi_p, -1, mu)
            rho = rho - phi_p * torch.roll(phi_p,  1, mu)
        S_prop = rho.sum()

    H_prop  = beta * S_prop + 0.5 * (pi_p**2).sum()
    delta_H = (H_prop - H_curr).item()
    accept  = (delta_H < 0) or (torch.rand(1).item() < np.exp(-delta_H))

    phi_out = phi_p if accept else phi.clone()
    S_out   = float(S_prop) if accept else float(S_curr.detach())
    return phi_out, accept, S_out


# ---------------------------------------------------------------------------
# Local Metropolis (single-site updates)
# ---------------------------------------------------------------------------

def local_metropolis_sweep(phi: torch.Tensor, M2: float, lam: float,
                            delta: float = 0.5) -> tuple[torch.Tensor, float]:
    """
    One full sweep of single-site Metropolis updates (visits every site once).

    Proposes phi_x -> phi_x + U(-delta, delta) for each site x.
    Uses the LOCAL action change ΔS to avoid recomputing the full action.

    ΔS_x when changing phi_x -> phi_x':
      ΔS = (M^2 + 2*Nd)(phi'^2 - phi^2) + lam(phi'^4 - phi^4)
           - 2 * (phi' - phi) * sum_{mu} (phi_{x+mu} + phi_{x-mu})

    This is O(L^2) per sweep — identical cost to HMC per unit of
    configuration-space displacement, but with much higher autocorrelation
    for correlated (near-critical) systems. Use HMC unless debugging.

    Args:
        phi:   current configuration (L, L)
        M2:    mass squared
        lam:   quartic coupling
        delta: proposal width
    Returns:
        phi_new:       updated configuration
        accept_rate:   fraction of sites accepted this sweep
    """
    L = phi.shape[0]
    Nd = 2  # 2D lattice
    phi = phi.clone()
    n_accept = 0

    for i in range(L):
        for j in range(L):
            phi_x  = phi[i, j].item()
            phi_xp = phi_x + (torch.rand(1).item() * 2 - 1) * delta

            # Nearest neighbours (periodic BC)
            nb_sum = (phi[(i+1) % L, j] + phi[(i-1) % L, j] +
                      phi[i, (j+1) % L] + phi[i, (j-1) % L]).item()

            # Local action change
            dS = ((M2 + 2 * Nd) * (phi_xp ** 2 - phi_x ** 2)
                  + lam * (phi_xp ** 4 - phi_x ** 4)
                  - 2.0 * (phi_xp - phi_x) * nb_sum)

            if dS < 0 or torch.rand(1).item() < np.exp(-dS):
                phi[i, j] = phi_xp
                n_accept  += 1

    return phi, n_accept / (L * L)


# ---------------------------------------------------------------------------
# Main sampler class
# ---------------------------------------------------------------------------

class MCMCSampler:
    """
    MCMC reference sampler for phi^4 theory on an L×L periodic lattice.

    Supports HMC (recommended) and local Metropolis (fallback / comparison).
    Stores the full chain history so you can inspect autocorrelations and
    choose the thinning interval manually.

    Example:
        sampler = MCMCSampler(L=8, M2=-4.0, lam=16.0, device='cpu')
        sampler.thermalize(n_therm=500)
        cfgs = sampler.sample(n_samples=2000, thin=5)
        np.save('mcmc_reference.npy', cfgs)
    """

    def __init__(self, L: int, M2: float, lam: float,
                 algorithm: str = 'hmc',
                 hmc_step_size: float = 0.1,
                 hmc_n_leapfrog: int = 10,
                 local_delta: float = 0.5,
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Args:
            L:               lattice side length (total DOF = L*L)
            M2:              mass squared (negative = symmetry-breaking phase)
            lam:             quartic coupling (positive for stability)
            algorithm:       'hmc' (recommended) or 'local' (Metropolis)
            hmc_step_size:   leapfrog step size; tune so acceptance ~0.7-0.9
            hmc_n_leapfrog:  leapfrog steps per proposal; tune so trajectory
                             length ~ 1.0 (step_size * n_leapfrog ≈ 1)
            local_delta:     proposal width for local Metropolis
            seed:            RNG seed for reproducibility
            device:          'cpu' recommended; see GPU/CPU note at bottom
        """
        self.L           = L
        self.M2          = M2
        self.lam         = lam
        self.algorithm   = algorithm
        self.step_size   = hmc_step_size
        self.n_leapfrog  = hmc_n_leapfrog
        self.local_delta = local_delta
        self.device      = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialise field: small random values (near the disordered phase)
        self.phi = torch.randn(L, L, device=device) * 0.1

        # History
        self.n_accepted   = 0
        self.n_proposed   = 0
        self.energy_trace = []   # S[phi] at each saved step
        self.mag_trace    = []   # M = mean(phi) at each saved step

    def _action(self, phi: torch.Tensor) -> float:
        with torch.no_grad():
            rho = self.M2 * phi**2 + self.lam * phi**4
            for mu in (0, 1):
                rho = rho + 2 * phi**2
                rho = rho - phi * torch.roll(phi, -1, mu)
                rho = rho - phi * torch.roll(phi,  1, mu)
            return rho.sum().item()

    def step(self) -> bool:
        """Advance chain by one step (one HMC proposal or one full sweep)."""
        if self.algorithm == 'hmc':
            self.phi, accepted = hmc_step(
                self.phi, self.M2, self.lam,
                n_leapfrog=self.n_leapfrog,
                step_size=self.step_size,
                device=self.device,
            )
            self.n_proposed += 1
            self.n_accepted += int(accepted)
            return accepted
        else:
            self.phi, rate = local_metropolis_sweep(
                self.phi, self.M2, self.lam, delta=self.local_delta
            )
            # For local: count each site attempt
            self.n_proposed += self.L * self.L
            self.n_accepted += int(rate * self.L * self.L)
            return rate > 0

    @property
    def acceptance_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def thermalize(self, n_therm: int = 500, verbose: bool = True):
        """
        Run n_therm steps to bring the chain to equilibrium.
        The initial configuration is random (disordered); thermalisation
        lets the chain forget this and reach the typical set of p(phi).

        Rule of thumb:
          - n_therm >= 200  for L=8 with HMC
          - n_therm >= 500  for L=16 with HMC
          - n_therm >= 2000 for local Metropolis (high autocorrelation)
        """
        if verbose:
            print(f'Thermalising ({self.algorithm}, {n_therm} steps)...')
        t0 = time.perf_counter()
        for k in range(n_therm):
            self.step()
            if verbose and (k + 1) % (n_therm // 5) == 0:
                E = self._action(self.phi)
                M = self.phi.mean().item()
                print(f'  step {k+1:5d}/{n_therm}  S={E:.1f}  M={M:.3f}'
                      f'  acc={self.acceptance_rate:.3f}')
        if verbose:
            print(f'Thermalisation done in {time.perf_counter()-t0:.1f}s\n')

        # Reset acceptance counters after thermalisation
        self.n_accepted = 0
        self.n_proposed = 0

    def sample(self, n_samples: int = 1000, thin: int = 5,
               verbose: bool = True) -> np.ndarray:
        """
        Collect n_samples configurations, keeping every thin-th step.

        Thinning reduces autocorrelation between saved samples. The integrated
        autocorrelation time (IAT) tells you the minimum useful thin value:
          - HMC with well-tuned step size: IAT ~ 1-5, thin=1-5 is fine
          - Local Metropolis near Tc:      IAT can reach 100+, thin=50-200

        Use compute_autocorrelation() after sampling to check.

        Args:
            n_samples: number of configurations to collect
            thin:      steps between saved configurations
        Returns:
            cfgs: (n_samples, L, L) numpy array, dtype float32
        """
        if verbose:
            print(f'Sampling {n_samples} configs (thin={thin}, '
                  f'total steps={n_samples*thin})...')

        cfgs = np.empty((n_samples, self.L, self.L), dtype=np.float32)
        t0   = time.perf_counter()

        for k in range(n_samples):
            for _ in range(thin):
                self.step()

            cfg_np = self.phi.cpu().numpy().astype(np.float32)
            cfgs[k] = cfg_np

            E = self._action(self.phi)
            M = self.phi.mean().item()
            self.energy_trace.append(E)
            self.mag_trace.append(M)

            if verbose and (k + 1) % max(1, n_samples // 10) == 0:
                elapsed = time.perf_counter() - t0
                rate    = (k + 1) / elapsed
                print(f'  {k+1:5d}/{n_samples}  S={E:.1f}  M={M:.4f}'
                      f'  acc={self.acceptance_rate:.3f}'
                      f'  {rate:.1f} cfg/s')

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f'\nDone: {n_samples} configs in {elapsed:.1f}s '
                  f'({n_samples/elapsed:.1f} cfg/s)')
            print(f'Acceptance rate: {self.acceptance_rate:.3f}')
            self._print_summary(cfgs)

        return cfgs

    def _print_summary(self, cfgs: np.ndarray):
        """Print basic observables after sampling."""
        print('\n--- Observables ---')
        print(f"<phi>   = {cfgs.mean():.4f}   (should be ~0 for Z2-symmetric phase)")
        print(f"<phi^2> = {(cfgs**2).mean():.4f}")
        print(f"<phi^4> = {(cfgs**4).mean():.4f}")
        mag  = cfgs.mean(axis=(1, 2))
        binder = 1 - (cfgs**4).mean() / (3 * (cfgs**2).mean()**2)
        print(f"<|M|>   = {np.abs(mag).mean():.4f}")
        print(f"U4      = {binder:.4f}")
        E_arr = np.array(self.energy_trace)
        print(f"<S>     = {E_arr.mean():.2f} ± {E_arr.std():.2f}")


# ---------------------------------------------------------------------------
# Parallel Tempering (Replica Exchange Monte Carlo)
# ---------------------------------------------------------------------------

class ParallelTemperingSampler:
    """
    Parallel tempering sampler for phi^4 theory on an L×L periodic lattice.

    Motivation
    ----------
    A single HMC chain cannot tunnel between the two Z2 modes in the deeply
    broken phase because the collective barrier (ΔV ∝ L² × m⁴/λ) is too
    large for any single Gaussian momentum draw to overcome coherently.
    Parallel tempering solves this by running N_replicas chains at a ladder
    of inverse temperatures β_0 < β_1 < ... < β_{K-1} = 1 (geometric spacing).

    At high temperature (low β), the energy barrier is β × ΔV ≈ 0, so the
    chain tunnels freely between modes. Neighboring replicas periodically
    swap configurations: if replica k (cold) is stuck in one well, a hot
    replica may have already diffused to the other well and can pass it along
    through a sequence of accepted swaps.

    After thermalisation, only configurations from the β=1 (physical)
    replica are collected as samples.

    Swap acceptance criterion (detailed balance between replica k and k+1):
        delta = (beta_{k+1} - beta_k) * (S[phi_{k+1}] - S[phi_k])
        p_swap = min(1, exp(delta))

    Target swap acceptance rates: 20–50% per neighboring pair.
    Tune n_replicas and beta_min so that adjacent pairs overlap well.

    Rule of thumb for beta_min:
        beta_min ≈ 1 / (barrier_per_site) = λ / m⁴
        For M2=-4, lam=1: beta_min ≈ 1/4 = 0.25  (start conservative)
        For M2=-4, lam=4: beta_min ≈ 4/256 ≈ 0.1

    Usage:
        pt = ParallelTemperingSampler(L=8, M2=-4.0, lam=1.0,
                                      n_replicas=8, beta_min=0.1)
        pt.thermalize(n_therm=500)
        cfgs = pt.sample(n_samples=2000, thin=5)
    """

    def __init__(self, L: int, M2: float, lam: float,
                 n_replicas: int = 8,
                 beta_min: float = 0.1,
                 beta_max: float = 1.0,
                 hmc_step_size: float = 0.15,
                 hmc_n_leapfrog: int = 10,
                 seed: int = 42,
                 device: str = 'cpu'):
        self.L          = L
        self.M2         = M2
        self.lam        = lam
        self.n_replicas = n_replicas
        self.step_size  = hmc_step_size
        self.n_leapfrog = hmc_n_leapfrog
        self.device     = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Geometric ladder: beta_0 = beta_min, beta_{K-1} = beta_max = 1
        self.betas = list(np.geomspace(beta_min, beta_max, n_replicas))

        # Initialise each replica from independent small-noise configurations
        self.phis    = [torch.randn(L, L, device=device) * 0.1
                        for _ in range(n_replicas)]
        # Cache unscaled action values — updated by HMC and swap steps
        self.actions = [self._action(phi) for phi in self.phis]

        # Per-replica HMC acceptance counters
        self.n_hmc_accepted = [0] * n_replicas
        self.n_hmc_proposed = [0] * n_replicas
        # Per-pair swap counters (pair k ↔ k+1)
        self.n_swap_accepted = [0] * (n_replicas - 1)
        self.n_swap_proposed = [0] * (n_replicas - 1)

        # Traces from the physical (β=1) replica
        self.mag_trace    = []
        self.energy_trace = []

    # -----------------------------------------------------------------------

    def _action(self, phi: torch.Tensor) -> float:
        """Unscaled S[phi] (no beta factor)."""
        with torch.no_grad():
            rho = self.M2 * phi**2 + self.lam * phi**4
            for mu in (0, 1):
                rho = rho + 2 * phi**2
                rho = rho - phi * torch.roll(phi, -1, mu)
                rho = rho - phi * torch.roll(phi,  1, mu)
            return rho.sum().item()

    def _hmc_all(self):
        """Advance every replica by one HMC step at its own temperature."""
        for k in range(self.n_replicas):
            phi_new, accepted, S_new = _hmc_step_beta(
                self.phis[k], self.M2, self.lam, self.betas[k],
                self.n_leapfrog, self.step_size, self.device,
            )
            self.phis[k]    = phi_new
            self.actions[k] = S_new
            self.n_hmc_proposed[k] += 1
            self.n_hmc_accepted[k] += int(accepted)

    def _swap_all(self):
        """
        Full alternating sweep of neighboring swap proposals.
        Even pairs first, then odd pairs (avoids conflicts on the same replica).
        """
        for parity in (0, 1):
            for k in range(parity, self.n_replicas - 1, 2):
                # Metropolis acceptance for swapping replicas k and k+1
                delta = ((self.betas[k + 1] - self.betas[k])
                         * (self.actions[k + 1] - self.actions[k]))
                self.n_swap_proposed[k] += 1
                if delta >= 0 or np.random.rand() < np.exp(delta):
                    self.phis[k],    self.phis[k + 1]    = self.phis[k + 1],    self.phis[k]
                    self.actions[k], self.actions[k + 1] = self.actions[k + 1], self.actions[k]
                    self.n_swap_accepted[k] += 1

    def step(self):
        """One full PT step: HMC for all replicas, then a swap sweep."""
        self._hmc_all()
        self._swap_all()

    # -----------------------------------------------------------------------

    def thermalize(self, n_therm: int = 500, verbose: bool = True):
        """
        Run n_therm PT steps (HMC + swaps) to reach equilibrium.
        Thermalisation counters are reset afterwards.
        """
        if verbose:
            print(f'Parallel tempering thermalisation '
                  f'({n_therm} steps, {self.n_replicas} replicas, '
                  f'β ∈ [{self.betas[0]:.3f}, {self.betas[-1]:.3f}])...')
        t0 = time.perf_counter()
        for k in range(n_therm):
            self.step()
            if verbose and (k + 1) % max(1, n_therm // 5) == 0:
                S_phys = self.actions[-1]
                M_phys = self.phis[-1].mean().item()
                acc_hmc  = self.n_hmc_accepted[-1] / max(1, self.n_hmc_proposed[-1])
                acc_swap = np.mean([
                    self.n_swap_accepted[i] / max(1, self.n_swap_proposed[i])
                    for i in range(self.n_replicas - 1)
                ])
                print(f'  step {k+1:5d}/{n_therm}  '
                      f'S(β=1)={S_phys:.1f}  M(β=1)={M_phys:.3f}  '
                      f'hmc_acc={acc_hmc:.3f}  swap_acc(mean)={acc_swap:.3f}')
        if verbose:
            print(f'Thermalisation done in {time.perf_counter()-t0:.1f}s\n')
        # Reset counters so sampling-phase diagnostics are clean
        self.n_hmc_accepted  = [0] * self.n_replicas
        self.n_hmc_proposed  = [0] * self.n_replicas
        self.n_swap_accepted = [0] * (self.n_replicas - 1)
        self.n_swap_proposed = [0] * (self.n_replicas - 1)

    def sample(self, n_samples: int = 1000, thin: int = 5,
               verbose: bool = True) -> np.ndarray:
        """
        Collect n_samples configurations from the physical (β=1) replica.

        Args:
            n_samples: number of configurations to save
            thin:      PT steps between saved configurations
        Returns:
            cfgs: (n_samples, L, L) float32 array
        """
        if verbose:
            print(f'Sampling {n_samples} configs from β=1 replica '
                  f'(thin={thin}, total PT steps={n_samples * thin})...')
        cfgs = np.empty((n_samples, self.L, self.L), dtype=np.float32)
        t0   = time.perf_counter()

        for k in range(n_samples):
            for _ in range(thin):
                self.step()

            cfg_np = self.phis[-1].cpu().numpy().astype(np.float32)
            cfgs[k] = cfg_np
            self.energy_trace.append(self.actions[-1])
            self.mag_trace.append(self.phis[-1].mean().item())

            if verbose and (k + 1) % max(1, n_samples // 10) == 0:
                elapsed  = time.perf_counter() - t0
                acc_hmc  = self.n_hmc_accepted[-1] / max(1, self.n_hmc_proposed[-1])
                acc_swap = np.mean([
                    self.n_swap_accepted[i] / max(1, self.n_swap_proposed[i])
                    for i in range(self.n_replicas - 1)
                ])
                print(f'  {k+1:5d}/{n_samples}  '
                      f'S={self.actions[-1]:.1f}  M={self.mag_trace[-1]:.4f}  '
                      f'hmc_acc={acc_hmc:.3f}  swap_acc={acc_swap:.3f}  '
                      f'{(k+1)/elapsed:.1f} cfg/s')

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f'\nDone: {n_samples} configs in {elapsed:.1f}s '
                  f'({n_samples/elapsed:.1f} cfg/s)')
            self._print_summary(cfgs)
        return cfgs

    def _print_summary(self, cfgs: np.ndarray):
        print('\n--- Observables (β=1 replica) ---')
        print(f"<phi>   = {cfgs.mean():.4f}   (target ~0: both modes visited)")
        print(f"<phi^2> = {(cfgs**2).mean():.4f}")
        print(f"<phi^4> = {(cfgs**4).mean():.4f}")
        mag    = cfgs.mean(axis=(1, 2))
        binder = 1 - (cfgs**4).mean() / (3 * (cfgs**2).mean()**2 + 1e-12)
        print(f"<|M|>   = {np.abs(mag).mean():.4f}")
        print(f"U4      = {binder:.4f}")
        E_arr  = np.array(self.energy_trace)
        print(f"<S>     = {E_arr.mean():.2f} ± {E_arr.std():.2f}")

        print('\n--- Swap acceptance per pair ---')
        for k in range(self.n_replicas - 1):
            acc = self.n_swap_accepted[k] / max(1, self.n_swap_proposed[k])
            print(f'  β={self.betas[k]:.3f} ↔ β={self.betas[k+1]:.3f} :  '
                  f'acc={acc:.3f}')
        print('  (target: 0.20–0.50 per pair; tune n_replicas / beta_min if outside range)')

        print('\n--- HMC acceptance per replica ---')
        for k in range(self.n_replicas):
            acc = self.n_hmc_accepted[k] / max(1, self.n_hmc_proposed[k])
            print(f'  β={self.betas[k]:.3f} :  hmc_acc={acc:.3f}')


# ---------------------------------------------------------------------------
# Autocorrelation diagnostics
# ---------------------------------------------------------------------------

def compute_autocorrelation(obs: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Normalised autocorrelation function of a 1D observable trace.
    C(t) = <O(s) O(s+t)>_s / <O^2>_s

    Use this on the mag_trace or energy_trace to decide the thinning interval.
    The integrated autocorrelation time IAT = 0.5 + sum_{t=1}^{T_cut} C(t)
    gives the effective number of independent samples: N_eff = N / (2 * IAT).
    """
    obs   = obs - obs.mean()
    var   = np.dot(obs, obs) / len(obs)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    C = np.array([
        np.dot(obs[:len(obs)-t], obs[t:]) / (var * (len(obs) - t))
        for t in range(max_lag + 1)
    ])
    return C


def integrated_autocorrelation_time(obs: np.ndarray,
                                     max_lag: int = 100,
                                     c_stop: float = 6.0) -> float:
    """
    Estimate the integrated autocorrelation time (IAT) using the Madras-Sokal
    windowing method: stop summing when the window length > c_stop * running IAT.

    Returns IAT in units of MCMC steps.
    """
    C   = compute_autocorrelation(obs, max_lag)
    iat = 0.5
    for t in range(1, len(C)):
        iat += C[t]
        if t >= c_stop * iat:
            break
    return iat


def recommend_sampler(L: int, M2: float, lam: float) -> str:
    """
    Sampler recommendation for reference generation.

    Negative M2 with small lambda can be ordered/broken and have two
    Z2-related modes. Plain HMC can look healthy while staying in one mode
    once the lattice is large enough. For small lattices, tunnelling is often
    visible directly in the magnetization diagnostics, so HMC is acceptable
    unless those diagnostics fail.
    """
    if M2 < 0 and L >= 8 and lam <= 8.0:
        return 'parallel_tempering'
    return 'hmc'


def validate_reference_samples(
    cfgs: np.ndarray,
    L: int,
    M2: float,
    lam: float,
    energy_trace: np.ndarray | None = None,
    mag_trace: np.ndarray | None = None,
    algorithm: str = 'unknown',
    thin: int | None = None,
    swap_acceptance: list[float] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Print and return diagnostics for deciding whether samples are usable as
    reference/ground truth.

    These checks do not prove convergence, but they catch the most common
    reference failures: wrong shape, non-finite values, bad Z2 mode balance,
    too few effective samples, and poor PT swap overlap.
    """
    cfgs = np.asarray(cfgs)
    mag = cfgs.mean(axis=(1, 2))
    abs_mag = np.abs(mag)
    signs = np.sign(mag)
    nonzero = signs != 0
    signs_nz = signs[nonzero]
    sign_flips = int(np.sum(signs_nz[1:] != signs_nz[:-1])) if len(signs_nz) > 1 else 0
    pos_frac = float(np.mean(mag > 0))

    if mag_trace is None:
        mag_trace = mag
    else:
        mag_trace = np.asarray(mag_trace)
    iat_M = integrated_autocorrelation_time(mag_trace)
    iat_absM = integrated_autocorrelation_time(abs_mag)
    n_eff = len(mag_trace) / (2 * max(iat_M, iat_absM, 0.5))

    m2 = float(np.mean(mag**2))
    m4 = float(np.mean(mag**4))
    binder_M = float(1.0 - m4 / (3.0 * m2**2 + 1e-30))

    rec = recommend_sampler(L, M2, lam)
    issues = []
    if cfgs.shape != (cfgs.shape[0], L, L):
        issues.append(f'unexpected shape {cfgs.shape}; expected (N,{L},{L})')
    if not np.isfinite(cfgs).all():
        issues.append('non-finite field values')
    if cfgs.shape[0] < 1000:
        issues.append('fewer than 1000 saved configurations')
    if rec == 'parallel_tempering':
        if not (0.35 <= pos_frac <= 0.65):
            issues.append(f'Z2 mode imbalance: positive magnetization fraction {pos_frac:.3f}')
        if sign_flips < 20:
            issues.append(f'few sign flips in saved chain: {sign_flips}')
    if n_eff < 200:
        issues.append(f'low effective sample count from saved trace: {n_eff:.0f}')
    if max(iat_M, iat_absM) > 2:
        issues.append(
            f'saved samples are still correlated: '
            f'max saved-sample IAT = {max(iat_M, iat_absM):.1f}'
        )
    if swap_acceptance is not None:
        low = [x for x in swap_acceptance if x < 0.10]
        if low:
            issues.append('one or more PT swap rates below 0.10')
        very_high = [x for x in swap_acceptance if x > 0.90]
        if len(very_high) == len(swap_acceptance):
            issues.append(
                'all PT swap rates exceed 0.90; beta ladder may be too dense '
                'or swap diagnostics should be checked'
            )

    status = 'PASS' if not issues else 'CHECK'
    result = {
        'status': status,
        'recommended_sampler': rec,
        'algorithm': algorithm,
        'n_samples': int(cfgs.shape[0]),
        'field_mean': float(cfgs.mean()),
        'field_std': float(cfgs.std()),
        'mean_abs_magnetization': float(abs_mag.mean()),
        'positive_magnetization_fraction': pos_frac,
        'sign_flips': sign_flips,
        'binder_magnetization': binder_M,
        'iat_magnetization': float(iat_M),
        'iat_abs_magnetization': float(iat_absM),
        'n_eff_estimate': float(n_eff),
        'issues': issues,
    }

    if verbose:
        print('\n--- Reference-quality checks ---')
        print(f'Recommended sampler       : {rec}')
        print(f'Used sampler              : {algorithm}')
        print(f'Status                    : {status}')
        print(f'N samples                 : {cfgs.shape[0]}')
        print(f'<phi>, std(phi)           : {result["field_mean"]:.5f}, {result["field_std"]:.5f}')
        print(f'<|M|>                     : {result["mean_abs_magnetization"]:.5f}')
        print(f'positive M fraction       : {pos_frac:.3f}')
        print(f'sign flips                : {sign_flips}')
        print(f'Binder U4(M)              : {binder_M:.4f}')
        print(f'IAT(M), IAT(|M|)          : {iat_M:.2f}, {iat_absM:.2f}')
        print(f'N_eff estimate            : {n_eff:.0f} / {len(mag_trace)}')
        if swap_acceptance is not None:
            print('PT swap acceptance        : ' +
                  ', '.join(f'{x:.3f}' for x in swap_acceptance))
        if issues:
            print('Issues:')
            for issue in issues:
                print(f'  - {issue}')
        else:
            print('No obvious reference-quality issues detected.')

    return result


# ---------------------------------------------------------------------------
# Convenience runner functions
# ---------------------------------------------------------------------------

def run_hmc(L: int = 8, M2: float = -4.0, lam: float = 16.0,
            n_therm: int = 300, n_samples: int = 2000,
            thin: int = 5, step_size: float = 0.15,
            n_leapfrog: int = 10, seed: int = 42,
            save_path: str | None = None,
            device: str = 'cpu') -> np.ndarray:
    """
    One-call HMC run. Returns (n_samples, L, L) float32 array.

    Reports IAT(M), IAT(|M|), and IAT(S). For comparison with Singha2026,
    use IAT(|M|), because their HMC benchmark reports tau_int(|m|).
    """
    sampler = MCMCSampler(
        L=L, M2=M2, lam=lam,
        algorithm='hmc',
        hmc_step_size=step_size,
        hmc_n_leapfrog=n_leapfrog,
        seed=seed,
        device=device,
    )

    sampler.thermalize(n_therm=n_therm)
    cfgs = sampler.sample(n_samples=n_samples, thin=thin)

    # Autocorrelation diagnostics.
    # These traces are recorded only after thinning, so IAT is in saved-sample units.
    mag_arr = np.array(sampler.mag_trace)
    abs_mag_arr = np.abs(mag_arr)
    energy_arr = np.array(sampler.energy_trace)

    max_lag = min(len(mag_arr) // 2, 5000)

    iat_M = integrated_autocorrelation_time(mag_arr, max_lag=max_lag)
    iat_absM = integrated_autocorrelation_time(abs_mag_arr, max_lag=max_lag)
    iat_E = integrated_autocorrelation_time(energy_arr, max_lag=max_lag)

    print(f'\nAutocorrelation time:')
    print(f'  IAT(M)              = {iat_M:.2f} saved samples')
    print(f'  IAT(|M|)            = {iat_absM:.2f} saved samples')
    print(f'  IAT(S)              = {iat_E:.2f} saved samples')
    print(f'  IAT(|M|) raw ≈      = {iat_absM * thin:.2f} HMC steps')

    # Observable-wise effective sample sizes.
    n_eff_M = n_samples / (2 * max(iat_M, 0.5))
    n_eff_absM = n_samples / (2 * max(iat_absM, 0.5))
    n_eff_E = n_samples / (2 * max(iat_E, 0.5))
    n_eff_conservative = min(n_eff_M, n_eff_absM, n_eff_E)

    print(f'\nEffective sample size:')
    print(f'  N_eff(M)            ~ {n_eff_M:.0f} / {n_samples}')
    print(f'  N_eff(|M|)          ~ {n_eff_absM:.0f} / {n_samples}   <-- compare with Singha2026')
    print(f'  N_eff(S)            ~ {n_eff_E:.0f} / {n_samples}')
    print(f'  N_eff conservative  ~ {n_eff_conservative:.0f} / {n_samples}')

    if iat_absM > 2:
        print(f'  NOTE: Saved samples are still correlated in |M| '
              f'(IAT(|M|)={iat_absM:.2f}). Consider increasing thin or chain length.')
    else:
        print(f'  Saved-sample thinning looks OK for |M|: IAT(|M|) ~ {iat_absM:.2f}.')

    validate_reference_samples(
        cfgs, L, M2, lam,
        energy_trace=np.array(sampler.energy_trace),
        mag_trace=np.array(sampler.mag_trace),
        algorithm='hmc',
        thin=thin,
    )

    if save_path is not None:
        np.save(save_path, cfgs)
        print(f'\nSaved {cfgs.shape} array to {save_path}')

    return cfgs


def run_local_metropolis(L: int = 8, M2: float = -4.0, lam: float = 16.0,
                          n_therm: int = 1000, n_samples: int = 2000,
                          thin: int = 20, delta: float = 0.5,
                          seed: int = 42,
                          save_path: str | None = None) -> np.ndarray:
    """
    One-call local Metropolis run. Slower than HMC due to high autocorrelation.
    Useful as a simple baseline or sanity check.
    """
    sampler = MCMCSampler(
        L=L, M2=M2, lam=lam,
        algorithm='local',
        local_delta=delta,
        seed=seed,
        device='cpu',  # local Metropolis is inherently sequential — GPU gives no benefit
    )
    sampler.thermalize(n_therm=n_therm, verbose=True)
    cfgs = sampler.sample(n_samples=n_samples, thin=thin)

    if save_path is not None:
        np.save(save_path, cfgs)
        print(f'\nSaved {cfgs.shape} array to {save_path}')

    return cfgs


def run_parallel_tempering(L: int = 8, M2: float = -4.0, lam: float = 1.0,
                            n_replicas: int = 8,
                            beta_min: float = 0.1,
                            n_therm: int = 500, n_samples: int = 2000,
                            thin: int = 5,
                            step_size: float = 0.15,
                            n_leapfrog: int = 10,
                            seed: int = 42,
                            save_path: str | None = None,
                            device: str = 'cpu') -> np.ndarray:
    """
    One-call parallel tempering run.

    Use this instead of run_hmc when the target has strong two-mode structure
    and a single HMC chain does not tunnel between Z2 modes.
    """
    sampler = ParallelTemperingSampler(
        L=L, M2=M2, lam=lam,
        n_replicas=n_replicas,
        beta_min=beta_min,
        hmc_step_size=step_size,
        hmc_n_leapfrog=n_leapfrog,
        seed=seed,
        device=device,
    )

    sampler.thermalize(n_therm=n_therm)
    cfgs = sampler.sample(n_samples=n_samples, thin=thin)

    # Autocorrelation diagnostics.
    # IAT is in saved-sample units because mag_trace is recorded after thinning.
    mag_arr = np.array(sampler.mag_trace)
    abs_mag_arr = np.abs(mag_arr)
    energy_arr = np.array(sampler.energy_trace)

    max_lag = min(len(mag_arr) // 2, 5000)

    iat_M = integrated_autocorrelation_time(mag_arr, max_lag=max_lag)
    iat_absM = integrated_autocorrelation_time(abs_mag_arr, max_lag=max_lag)
    iat_E = integrated_autocorrelation_time(energy_arr, max_lag=max_lag)

    print(f'\nAutocorrelation time:')
    print(f'  IAT(M)              = {iat_M:.2f} saved samples')
    print(f'  IAT(|M|)            = {iat_absM:.2f} saved samples')
    print(f'  IAT(S)              = {iat_E:.2f} saved samples')
    print(f'  IAT(|M|) raw ≈      = {iat_absM * thin:.2f} PT steps')

    n_eff_M = n_samples / (2 * max(iat_M, 0.5))
    n_eff_absM = n_samples / (2 * max(iat_absM, 0.5))
    n_eff_E = n_samples / (2 * max(iat_E, 0.5))
    n_eff_conservative = min(n_eff_M, n_eff_absM, n_eff_E)

    print(f'\nEffective sample size:')
    print(f'  N_eff(M)            ~ {n_eff_M:.0f} / {n_samples}')
    print(f'  N_eff(|M|)          ~ {n_eff_absM:.0f} / {n_samples}')
    print(f'  N_eff(S)            ~ {n_eff_E:.0f} / {n_samples}')
    print(f'  N_eff conservative  ~ {n_eff_conservative:.0f} / {n_samples}')

    if iat_absM > 2:
        print(f'  NOTE: Saved samples are still correlated in |M| '
              f'(IAT(|M|)={iat_absM:.2f}). Consider increasing thin or chain length.')
    else:
        print(f'  Saved-sample thinning looks OK for |M|: IAT(|M|) ~ {iat_absM:.2f}.')

    swap_acc = [
        sampler.n_swap_accepted[k] / max(1, sampler.n_swap_proposed[k])
        for k in range(sampler.n_replicas - 1)
    ]

    validate_reference_samples(
        cfgs, L, M2, lam,
        energy_trace=np.array(sampler.energy_trace),
        mag_trace=np.array(sampler.mag_trace),
        algorithm='parallel_tempering',
        thin=thin,
        swap_acceptance=swap_acc,
    )

    if save_path is not None:
        np.save(save_path, cfgs)
        print(f'\nSaved {cfgs.shape} array to {save_path}')

    return cfgs


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_diagnostics(sampler: MCMCSampler, cfgs: np.ndarray | None = None):
    """
    Plot MCMC diagnostics:
      1. Energy trace
      2. Magnetization trace
      3. Autocorrelation of M and |M|
      4. P(M) histogram, if cfgs provided
    """
    import matplotlib.pyplot as plt

    E_arr = np.array(sampler.energy_trace)
    M_arr = np.array(sampler.mag_trace)
    absM_arr = np.abs(M_arr)

    max_lag = min(100, len(M_arr) // 2)

    C_M = compute_autocorrelation(M_arr, max_lag=max_lag)
    C_absM = compute_autocorrelation(absM_arr, max_lag=max_lag)

    iat_M = integrated_autocorrelation_time(M_arr, max_lag=max_lag)
    iat_absM = integrated_autocorrelation_time(absM_arr, max_lag=max_lag)

    ncols = 4 if cfgs is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(E_arr, lw=0.8, alpha=0.8)
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Action S[phi]')
    axes[0].set_title('Energy trace')

    axes[1].plot(M_arr, lw=0.8, alpha=0.8)
    axes[1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Magnetization M')
    axes[1].set_title('Magnetization trace')

    axes[2].plot(C_M, 'o-', ms=3, label='M')
    axes[2].plot(C_absM, 'o-', ms=3, label='|M|')
    axes[2].axhline(0, color='k', ls='--', alpha=0.3)
    axes[2].set_xlabel('Lag (saved samples)')
    axes[2].set_ylabel('C(lag)')
    axes[2].set_title(f'Autocorrelation\nIAT(M)={iat_M:.1f}, IAT(|M|)={iat_absM:.1f}')
    axes[2].legend()

    if cfgs is not None:
        mag = cfgs.mean(axis=(1, 2))
        axes[3].hist(mag, bins=60, density=True, alpha=0.8)
        axes[3].axvline(0, color='k', ls='--', alpha=0.4)
        axes[3].set_xlabel('M = <phi>')
        axes[3].set_ylabel('Density')
        axes[3].set_title('P(M) histogram')

    plt.tight_layout()
    plt.show()
