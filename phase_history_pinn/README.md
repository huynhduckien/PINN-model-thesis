# Phase-history thermal PINN

This directory is an isolated research branch for a temperature-only,
physics-only PINN for a single LPBF scan near a 45-degree downward wedge.
Everything created for this branch (code, preregistration, tests, run metadata,
and figures) stays below this directory.

## Scientific scope

The model combines:

- the Moser et al. (2019) powder correlations for effective absorptivity,
  optical extinction, and effective thermal conductivity;
- a Beer-Lambert volumetric Gaussian laser source;
- apparent heat capacity for latent heat;
- an effective liquid conductivity multiplier of 2.8;
- an irreversible powder-to-consolidated conductivity history updated by an
  outer fixed-point loop;
- a temperature-only PINN trained without field data.

Finite-difference or finite-element data are not used for training or
assessment in this branch. Results are internal outputs of the stated model;
they are not claimed to be verified PDE solutions or validated LPBF
predictions.

## Layout

```text
phase_history_pinn/
  prereg/       locked scope and acceptance rules
  ph_pinn/      implementation
  scripts/      entry points
  tests/        internal implementation checks
  runs/         checkpoints and run records (large files ignored)
  figures/      generated figures
```

## Commands

Run internal checks:

```powershell
conda run -n base python -m unittest discover -s phase_history_pinn/tests -v
```

Run a small end-to-end smoke case:

```powershell
conda run -n base python -m phase_history_pinn.scripts.run_case --preset smoke --geometry normal --mode history
```

Run all four preregistered smoke cases (C1--C4):

```powershell
conda run -n base python -m phase_history_pinn.scripts.run_matrix --preset smoke
```

The smoke preset checks execution only. It is not a scientific training run.
The latest implementation-check record is in `IMPLEMENTATION_STATUS.md`.

Corrective optimization attempts and their acceptance status are recorded in
`diagnostics/CORRECTIVE_PILOTS.md`. A `corrective` preset exists for reproducing
the second C1 pilot, but it has not passed the internal thermal gate.

## Provenance

The equations were reimplemented in this directory rather than importing the
legacy project. Formula provenance and deviations are recorded in
`prereg/00_MODEL_SCOPE.md`. The previous project and its FD records remain
unchanged outside this directory.

