# Fact check: why the original training appeared to pass while the new branch failed

Date: 2026-10-09

## Finding

The runs are not equivalent experiments.  The original model reached a melt
pool after a long, staged warm-start procedure on a very large fixed Sobol
collocation set.  The first new branch run started the full nonlinear,
low-powder-conductivity problem from random weights using a much smaller
resampled batch and Adam only.  The original and new pass/fail labels also use
different acceptance gates.

The corrective pilots did not fail to reach the solidus.  Their cycle peaks
were 1,776 K and 1,892 K.  They failed the newly introduced local PDE-balance
gate, and their unconverged snapshot peaks were unphysically high.

## Direct comparison

| Item | Original project | First new default run | Corrective pilot 2 |
|---|---|---|---|
| Initialization | staged warm starts | random | random |
| Interior uniform points | 65,536 fixed Sobol | 4,096 resampled/step | 2,048 resampled/step |
| Source-following interior points | 262,144 fixed Sobol | 8,192 broad resampled/step | 2,048 source-core plus 2,048 near/wake |
| Source-following top points | 4,096 | none | none |
| Uniform top points | 32,768 | 2,048 | 1,024 |
| Optimizer history | staged Adam and L-BFGS, then warm-started refinements | 2,000 Adam | 5,000 Adam |
| Network start for low powder k/history | already-hot B3 checkpoint | random weights | random weights |
| Main acceptance idea | global energy/positivity gates | no local gate at run time | local source residual gate |

The original fixed anchor set contains about 364,544 points.  The first new run
used 14,336 points per stochastic step, and corrective pilot 2 used 7,168.
Point count alone is not a quality measure, but here it combines with fixed
Sobol coverage, source-following top points, warm starts, and L-BFGS.

## Original curriculum

The archived chain did not solve the final model from random weights:

1. `A1s`: simplified thermal problem;
2. `B1`: add latent heat;
3. `B2`: add evaporation;
4. `B3`: add the 2.8 liquid-conductivity multiplier;
5. `B3s`: introduce the low-conductivity powder/history mask from a hot B3
   checkpoint;
6. later refinement, including F08 C3 for another 20,000 Adam steps.

The legacy A1s/B1/B2/B3 metadata record roughly 20,000--25,000 optimizer
iterations per stage, including L-BFGS after Adam.  The new branch instead
activated latent heat, evaporation, liquid conductivity, low powder
conductivity, and the history-ready constitutive law from the first random
checkpoint.

This is the strongest explanation for why the original optimization found a
hot branch while the first new run found a diffuse cold branch.

## Sampling difference

The original source sampler was not the same as the first new sampler.  It used
a fixed source-following ellipsoidal Sobol cloud and explicitly added a moving
disc of top-boundary points around the laser.  The first new implementation
used a broad rectangular source distribution and a uniformly sampled top
boundary.  Its checkpoint audit showed that the learned thermal operator was
below 0.2% of source RMS in the strong-source region.

The corrective source-core sampler repaired much of the interior problem, but
the top boundary remains uniformly sampled.  This is consistent with the
extreme unconverged surface peaks in the corrective pilots.

## Physics-path difference

The core 300 W Beer--Lambert source, material constants, latent-heat width,
evaporation law, hard initial/side/bottom conditions, and 6 x 24 base network
are substantially the same.

However, the original B3 warm-start stage used powder volumetric heat capacity
but solid-like conductivity (`powder_k=False`).  Low powder conductivity was
introduced later through the history-mask branch.  The new branch used the
Moser conductivity ratio, approximately 0.02346, from the beginning.  This
creates a much stiffer coefficient contrast for a randomly initialized PINN.

The history laws are also not identical.  The original branch used a traveling
front multiplied by a precomputed maximum-temperature envelope.  The new
branch uses a four-dimensional irreversible history field updated by an outer
loop.  Therefore the original result is useful as an optimization reference,
but it is not a same-model baseline for the new phase-history formulation.

## Acceptance-gate difference

The original `gate_full` checked integrated residual energy, an independent
energy identity, temperature undershoot, and total stored-plus-evaporated
energy.  It did not require the local source residual RMS to be below 10%.

For the accepted original F08 C3 checkpoint:

- reported peak temperature: 5,582 K;
- melt depth: 90 um;
- training PDE loss: 0.00545;
- original full gate: pass;
- previously recorded relative residual near the laser: median 14.1%,
  90th percentile 43.1%.

Those residual statistics are not identical to the new RMS gate, so an exact
pass/fail translation is not claimed.  They do show that “passed the original
gate” did not mean “local residual everywhere below 10%.”

## Interpretation

There are two separate observations:

1. The first new default run genuinely failed to learn the laser source and
   therefore stayed cold.
2. The corrective runs learned enough source response to exceed the solidus,
   but failed a stricter local residual gate and developed uncontrolled surface
   peaks.

Thus the new proposal did not simply turn a successful identical run into a
failed one.  It changed the model path, initialization, point set, optimizer
budget, top-boundary coverage, history representation, and acceptance rule at
the same time.
