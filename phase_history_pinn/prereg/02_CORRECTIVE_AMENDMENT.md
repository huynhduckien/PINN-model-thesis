# Preregistration 02 - corrective amendment after failed source learning

Date: 2026-10-09

This amendment was written after the first default C1--C4 matrix and the
checkpoint audit in `diagnostics/FACT_CHECK_LOW_TEMPERATURE.md`.  It does not
change the material model, Moser correlations, phase-history definition, laser
power, or Beer--Lambert source equation.

## Reason for amendment

In the first matrix, the learned thermal operator was below 0.2% of the source
RMS in the strong-source region.  The original source-biased sampler placed
only about 1.2% of its samples above 10% of peak source, and a single averaged
PDE loss allowed these points to be ignored.

## Corrective changes

1. Draw source-core coordinates from distributions matched to the transverse
   Gaussian and Beer--Lambert depth profile.
2. Keep separate uniform, source-core, hotspot, and near/wake PDE losses.
3. Delay and ramp the global energy penalty rather than applying its full
   weight from the first optimizer step.
4. Add laser-following input coordinates based on `x - x_laser(t)`.
5. Evaluate a fixed-seed thermal gate after every inner training stage.

The network depth and width remain unchanged for the first corrective pilot so
that the effect of sampling, loss construction, and coordinates can be
identified separately from model-capacity changes.

## Thermal gate

Before a phase-history update is allowed:

- source-core residual RMS / source RMS <= 0.10;
- hotspot residual RMS / hotspot source RMS <= 0.10;
- total PDE MSE / zero-temperature-rise PDE MSE <= 0.10.

If this gate fails, the history update is skipped and the failure is recorded.
Passing is an internal optimization gate, not verification of the PDE solution.

## Pilot decision rule

Run C1 normal/static first.  C2--C4 are not rerun until C1 demonstrates a
material improvement in the fixed-seed local source-balance metrics.  Reaching
the solidus alone is not sufficient; the thermal gate controls the decision.
