# ReLIFT

**Renormalization-Guided Layerwise Inter-scale Field Transport**

ReLIFT is a multiscale generative-sampling framework for lattice field
theories. It grows configurations through successive lattice refinements,
learns the conditional transport at each refinement, and can apply an
RG-guided relaxation flow after lifting.

The current implementation provides:

- layerwise scratch, lift, relaxation, and validation workflows;
- RKL-CNF, EWFM, RFM, iDEM, and NETS transportation backends;
- checkpoint-compatible extension from a trained coarse lattice;
- density ESS for tractable deterministic flows and path ESS for NETS;
- raw and importance-weighted physical observables;
- MCMC/HMC reference-sample utilities; and
- best-stable-state restoration and GPU resource diagnostics.

The reusable implementation is under [`training_modules/`](training_modules/).
The main module-driven experiment entry point is layerwise_experiment_runner.ipynb.
[`layerwise_experiment_runner.ipynb`](layerwise_experiment_runner.ipynb).
residual_sr_layerwise_pipeline.ipynb is the experimental draft that needs further cleanup.

## Current limitation

Layerwise transport is implemented for all five backends. The explicit
RG-guided full-lattice relaxation block is currently implemented for the
RKL-CNF backend.

