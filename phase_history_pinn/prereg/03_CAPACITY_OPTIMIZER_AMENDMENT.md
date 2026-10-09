# Preregistration 03 - second corrective C1 pilot

Date: 2026-10-09

The first corrective C1 pilot used the original 6 x 24 network for 2,000 Adam
steps.  It improved the fixed-seed source residual ratio from approximately
1.0 to `0.7254` and PDE/zero-baseline ratio to `0.5293`, but failed the 0.10
thermal gate.  Its high unconverged snapshot peak is not interpreted.

Before running a second pilot, the following changes are fixed:

- network: 6 hidden layers x 48 neurons;
- collocation per step: 2,048 uniform, 2,048 source core, 1,024 near source,
  1,024 wake, and 1,024 top-boundary points;
- optimizer: 5,000 Adam steps;
- learning rate: cosine decay from `3e-4` to `5e-5`;
- top-boundary weight: 5;
- energy weight: ramp from 0 after step 1,500 to 1 at step 3,500;
- all thermal-gate thresholds remain unchanged.

Only C1 normal/static is run.  C2--C4 remain blocked until the fixed-seed C1
thermal gate passes.  This amendment changes optimization capacity only; it
does not change the physical equations or source power.
