# Training modules

This directory contains reusable code shared by interactive notebooks and
non-interactive batch jobs:

- `config.py`: experiment configuration and GPU runtime policy.
- `physics.py`: target action and source-distribution functions.
- `pipeline.py`: model definitions, training dispatch, and checkpoint workflows.
- `rg_relaxation.py`: RG-guided full-lattice relaxation CNF.
- `training_best_state.py`: stable best-state selection and restoration.
- `validation.py`: full multiscale MCMC, ESS, observable, and C2ST evaluation.
- `mcmc_phi4.py`: independent HMC/Metropolis reference sampler.

`pipeline.py` exposes the checkpoint-compatible model classes and the
scratch/lift/relaxation workflows without executing an experiment at import
time. `validation.py` exposes the complete validation protocol without
executing it at import time. `../layerwise_experiment_runner.ipynb` is the
small module-based entry point for all four workflows.

From the repository root, import modules through the package:

```python
from lattice_scaling_final.training_modules.config import Phi4Config
from lattice_scaling_final.training_modules.pipeline import load_training_checkpoint
from lattice_scaling_final.training_modules.physics import phi4_action
from lattice_scaling_final.training_modules.rg_relaxation import RGRelaxationCNF
```

When a notebook is launched with `lattice_scaling_final` as its working
directory, the fallback package prefix is `training_modules`.

Use the reference sampler from a notebook or batch entry point with:

```python
from lattice_scaling_final.training_modules.mcmc_phi4 import run_hmc
```
