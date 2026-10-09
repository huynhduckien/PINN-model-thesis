# Preregistration 01 - implementation and run gates

These gates detect implementation and training failures. Passing them does not
verify the PDE solution.

## Implementation gates

1. Moser Eqs. (10)-(12) match the frozen numerical values for the default
   configuration to relative tolerance 1e-12.
2. The integrated volumetric source agrees with absorbed laser power within
   1% in the deterministic quadrature check.
3. The hard transform returns ambient temperature at t=0, at the bottom, and
   at all four side faces to absolute dimensionless tolerance 1e-6.
4. Conductivity state checks recover r in cold unmolten powder, 1 in cold
   consolidated material, and 2.8 in the high-temperature consolidated limit.
5. History is bounded in [0,1], nondecreasing in time, and never decreases
   between outer iterations.

## Outer-loop convergence gates

For two consecutive history fields:

- maximum absolute history change <= 0.02;
- changed-volume fraction for |delta M| > 0.05 <= 0.05;
- change in consolidated volume fraction <= 0.02.

Failure is reported; thresholds are not relaxed after seeing a result.

## Training records

Every run stores configuration, random seed, device, loss history, checkpoint,
history field, outer-loop diagnostics, elapsed time, and git commit when
available. Smoke runs are labeled and cannot be used as scientific results.

## Interpretation

No internal gate is an accuracy or validation criterion. Quantitative outputs
are model-internal comparisons only.

