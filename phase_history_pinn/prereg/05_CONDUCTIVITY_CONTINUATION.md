# Preregistration 05 - gated powder-conductivity continuation

Date: 2026-10-09

The top/curriculum C1 run isolated the direct conductivity jump from 1 to the
Moser ratio 0.023458 as the largest remaining optimization shock.  This run
starts from the saved B3 checkpoint before that jump:
`runs/curriculum_top_C1/model_B3.pt`.

## Fixed sequence

| Stage | Powder k / solid k | Adam steps | Learning rate |
|---|---:|---:|---:|
| K1 | 1.0 | 3,000 | 5e-5 -> 1e-5 |
| K03 | 0.3 | 3,000 | 5e-5 -> 1e-5 |
| K01 | 0.1 | 3,000 | 3e-5 -> 5e-6 |
| K005 | 0.05 | 3,000 | 2e-5 -> 5e-6 |
| KM | 0.02345804558 | 5,000 | 2e-5 -> 2e-6 |

Latent heat, evaporation, and the 2.8 liquid-conductivity multiplier remain on
at every stage.  Network, source, top sampling, and loss weights remain those
of corrective pilot 2 plus the laser-following top-boundary amendment.

## Sequential stop rule

After each stage, evaluate the same fixed-seed thermal gate:

- source residual ratio <= 0.10;
- hotspot residual ratio <= 0.10;
- PDE/zero-rise ratio <= 0.10;
- top-core residual ratio <= 0.10.

Proceed to the next conductivity only if all four conditions pass.  On the
first failure, stop and retain that checkpoint.  C2--C4 remain blocked unless
the final Moser stage passes.  Temperature alone cannot override this rule.
