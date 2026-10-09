# C1 gated conductivity-continuation result

Date: 2026-10-09

Run: `runs/conductivity_continuation_C1`

Starting checkpoint: `runs/curriculum_top_C1/model_B3.pt`

## Result

The preregistered sequence was `1.0 -> 0.3 -> 0.1 -> 0.05 -> Moser`.
The run stopped after K1 because its held-out thermal gate failed.

| Stage | Powder/solid k ratio | Source ratio | Hotspot ratio | PDE/zero ratio | Top-core ratio | Gate |
|---|---:|---:|---:|---:|---:|---|
| K1 | 1.0 | 0.1614 | 0.0905 | 0.0297 | 0.0336 | fail |

The tolerance was 0.10 for every reported ratio. Only the source-region ratio
failed. K03, K01, K005, and KM were therefore not run.

Internal indicators at K1 were a snapshot peak of 5,661.2 K, cycle peak of
3,143.5 K, apparent depth of 100 um, and apparent penetration of 40 um. They
are not accepted as scientific outputs because the gate failed.

Total runtime was 252.1 s. The checkpoint `model_K1.pt` was retained for
diagnosis and possible refinement.

## Interpretation

Relative to the incoming B3 checkpoint, the source residual ratio improved
from 0.2071 to 0.1614. The other three gate components passed. However, K1
still uses solid-like powder conductivity, so this failure occurred before
introducing any conductivity contrast. It therefore does not test, confirm,
or reject the lower conductivity levels.

The result narrows the immediate bottleneck to convergence of the volumetric
source region at K1. Conductivity continuation remains appropriate only after
K1 passes the same held-out gate. The next controlled attempt should retain
K1, refine `model_K1.pt` with a larger fixed source-region collocation set and
a deterministic L-BFGS finishing stage, then evaluate the unchanged gate.

## Boundary on claims

No FD/FEM comparison was used. This run does not establish a verified PDE
solution, and C2--C4 remain blocked.
