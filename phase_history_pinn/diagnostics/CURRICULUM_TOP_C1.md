# C1 top-boundary and physics-curriculum result

Date: 2026-10-09

Run: `runs/curriculum_top_C1`

## Result

| Stage | Source residual ratio | Hotspot ratio | PDE/zero ratio | Top-core ratio | Gate |
|---|---:|---:|---:|---:|---|
| A | 0.1117 | 0.1013 | 0.0137 | 0.0324 | fail |
| B1 | 0.1026 | 0.0940 | 0.0119 | 0.0321 | fail |
| B2 | 0.7823 | 0.7678 | 0.6241 | 0.6135 | fail |
| B3 | 0.2071 | 0.1418 | 0.0476 | 0.0628 | fail |
| B4 | 0.7879 | 0.7857 | 0.6229 | 0.2190 | fail |

Final internal indicators were:

- snapshot peak: 4,563.6 K;
- cycle peak at the powder/substrate probe: 3,108.4 K;
- apparent melt depth: 110 um;
- total runtime: 1,150.6 s.

These temperature indicators are not accepted because the final thermal gate
failed.

## What the run established

The moving top-boundary cloud controlled the previous surface runaway: the
peak fell from 12,111 K in corrective pilot 2 to 4,564 K.  For the simplified
A and B1 stages, the source and top residuals came close to the fixed gate.

Turning on evaporation in B2 caused a discontinuous optimization shock.  The
first B2 step had top-core loss about `1.43e5`; after 2,000 steps the source and
top ratios remained 0.782 and 0.614.  Adding liquid conductivity in B3 and
continuing from that checkpoint recovered the ratios to 0.207 and 0.063.

The direct B4 jump from solid-like powder conductivity to the Moser ratio
`0.023458` broke the local PDE balance again.  After 5,000 steps, the source
ratio was still 0.788.  Therefore the principal remaining difficulty is the
abrupt coefficient contrast, with evaporation continuation a secondary issue.

## Comparison boundary

The original history branch did not apply low conductivity everywhere to a
zero-history powder field.  It first obtained a hot B3 solution, constructed a
melt envelope, and then kept the consolidated core near solid conductivity
while lowering conductivity only in surrounding unmolten powder.  The B4 pilot
was a static-zero-history case and is therefore more abrupt.

## Next candidate

For a static low-k qualification, introduce conductivity continuation, for
example `1.0 -> 0.3 -> 0.1 -> 0.05 -> 0.023458`, with a gate at every level.
For the intended history model, alternatively build the first history field
from the converged hot B3 stage before activating low powder conductivity.
Either change requires a new preregistered run; neither was performed here.
