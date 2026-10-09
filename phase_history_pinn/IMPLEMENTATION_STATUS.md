# Implementation status

Date: 2026-10-09

## Completed

- Isolated, temperature-only PyTorch PINN package.
- Moser et al. (2019) powder absorptivity, extinction, and conductivity
  correlations with frozen IN718/argon parameters.
- Beer--Lambert volumetric Gaussian source.
- Apparent heat capacity and effective liquid conductivity multiplier 2.8.
- Irreversible phase history with an outer fixed-point loop.
- Normal and 45-degree wedge geometries.
- Four-case matrix: C1 normal/static, C2 wedge/static, C3 normal/history,
  and C4 wedge/history.
- Run manifests, checkpoints, histories, progress logs, and model-internal
  indicators.

## Checks performed

- `python -m compileall -q phase_history_pinn`: passed.
- Fifteen unit tests covering the powder correlations, laser-source integral,
  constitutive states and continuation override, hard constraints, moving
  source/top sampling, and history interpolation/update:
  passed.
- Volumetric-source deterministic integral relative error:
  `1.25968e-4`, below the preregistered 1% tolerance.
- C1--C4 smoke matrix: completed; index
  `runs/matrix_smoke_20261009_142642.json`.
- Post-correction wedge/history smoke case: history remained monotone and the
  outer update met the smoke convergence checks.
- C1--C4 default matrix: completed; index
  `runs/matrix_default_20261009_142741.json`.

## First default-matrix outcome

| Case | Runtime (s) | Peak cycle T (K) | Snapshot peak T (K) | Melt detected |
|---|---:|---:|---:|---|
| C1 normal/static | 145.6 | 487.6 | 472.6 | no |
| C2 wedge/static | 147.5 | 653.6 | 607.8 | no |
| C3 normal/history | 282.2 | 558.3 | 534.8 | no |
| C4 wedge/history | 303.2 | 733.0 | 679.9 | no |

The history cases stopped after two outer iterations because no point reached
the solidus, so the history field remained zero. The final sampled PDE losses
were approximately 2.0--2.5 and did not show convincing convergence. Therefore
this run is an execution baseline, not a usable scientific comparison of phase
history. Optimizer/loss scaling and convergence diagnostics require revision
before interpreting C1--C4 physically.

A checkpoint-level diagnosis is recorded in
`diagnostics/FACT_CHECK_LOW_TEMPERATURE.md`. It confirms that the trained local
thermal operator remained below 0.2% of the laser-source RMS in the
strong-source region; the low temperature is therefore an optimization/loss
localization failure rather than a weak implemented laser source.

Two corrective C1-only pilots are recorded in
`diagnostics/CORRECTIVE_PILOTS.md`. Importance sampling, split source losses,
laser-following coordinates, increased network width, and learning-rate decay
reduced the source residual ratio from approximately 1.0 to 0.5596 and the
PDE/zero-rise ratio to 0.3168. This is a material improvement but still fails
the preregistered 0.10 thermal gate. C2--C4 were not rerun, and the corrective
pilot temperatures are not accepted as scientific results.

The subsequent laser-following top-boundary and five-stage physics curriculum
is recorded in `diagnostics/CURRICULUM_TOP_C1.md`. It removed the 12,111 K
surface runaway and nearly passed the simplified A/B1 stages, but the final
Moser low-conductivity stage failed with source residual ratio 0.7879. C2--C4
remain blocked.

The preregistered gated conductivity continuation is recorded in
`diagnostics/CONDUCTIVITY_CONTINUATION_C1.md`. Refinement at K1 improved the
source ratio from 0.2071 to 0.1614; hotspot, PDE/zero, and top-core ratios all
passed 0.10, but the source ratio did not. The automatic stop therefore
prevented K03 and lower conductivity stages from running. C2--C4 remain
blocked.

## Interpretation boundary

These checks test implementation consistency and execution only. They do not
compare against FD/FEM data, prove that the learned field solves the PDE, or
validate the model against experiments. The smoke optimizer uses only two
steps per outer iteration, so its temperature and melt indicators are not
scientific results.

## Not yet performed

- A converged full-resolution GPU training of the four-case matrix.
- A K1 checkpoint that passes the held-out local thermal gate before lowering
  powder conductivity.
- Training-repeat or seed-sensitivity study.
- Experimental validation.
