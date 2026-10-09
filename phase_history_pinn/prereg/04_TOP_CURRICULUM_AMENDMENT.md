# Preregistration 04 - laser-following top boundary and physics curriculum

Date: 2026-10-09

The comparison audit found that the original successful optimization used both
a moving top-boundary point cloud and a staged warm start.  The new branch used
neither.  This amendment restores those optimization devices without changing
the final physical model.

## Top-boundary sampling

Every training batch contains separate top-surface groups:

- 1,024 uniform points;
- 2,048 points in a radius-2-rb disc moving with the laser;
- 1,024 points in the wake, up to 16 rb behind the laser.

Their losses are not averaged together.  Weights are 5 for uniform, 10 for the
moving core, and 2 for the wake.  The final gate additionally requires the
normalized moving-core top residual to be at most 0.10.

## C1 physics curriculum

One network is warm-started through the following fixed stages:

| Stage | Powder k | Latent heat | Evaporation | Liquid k x2.8 | Adam steps |
|---|---|---|---|---|---:|
| A | solid-like | off | off | off | 3,000 |
| B1 | solid-like | on | off | off | 2,000 |
| B2 | solid-like | on | on | off | 2,000 |
| B3 | solid-like | on | on | on | 2,000 |
| B4 | Moser low k | on | on | on | 5,000 |

Stage A uses learning rate `3e-4 -> 1e-4`.  Stages B1--B3 use
`1e-4 -> 5e-5`; B4 uses `1e-4 -> 2e-5`.  Cosine decay is used within each
stage.  The final network remains 6 x 48.

Only C1 normal/static is run.  No phase-history update and no C2--C4 run is
allowed unless the final B4 thermal gate passes.  Temperature exceeding the
solidus does not override a failed gate.
