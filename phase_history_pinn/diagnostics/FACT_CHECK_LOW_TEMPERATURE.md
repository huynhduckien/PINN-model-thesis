# Fact check: why the first default run stayed below the solidus

Date: 2026-10-09

Run index: `runs/matrix_default_20261009_142741.json`

## Conclusion

The low peak temperatures were caused by failed PINN optimization of the
localized PDE source, not by insufficient laser power in the implemented
model.  The trained networks remained very close to the zero-temperature-rise
PDE baseline in the laser region.  The global energy penalty nevertheless
created a broad, low-amplitude temperature rise, which explains why the energy
residual could be small while the local peak remained below the solidus.

## Evidence

The implemented Moser/Beer--Lambert source has:

- effective absorptivity: `0.422207125`;
- absorbed power at 300 W: `126.662 W`;
- absorbed scan energy: `0.205826 J`;
- optical penetration depth: `26.728 um`;
- dimensionless peak source: `45.711`;
- deterministic source-integral relative error: `1.25968e-4`.

Thus, no missing-power or source-unit error was found.  The powder conductivity
(`0.516 W/(m K)`) and powder volumetric heat-capacity ratio (`0.55`) would tend
to increase, rather than suppress, the local peak temperature.

On a fixed audit batch, only about 115--116 of 12,288 interior points had a
source greater than 10% of its peak.  The trained PDE MSE was nearly identical
to the MSE obtained by setting the thermal response to zero:

| Case | Trained PDE MSE | Zero-rise PDE MSE | learned operator/source RMS in strong-source region |
|---|---:|---:|---:|
| C1 | 2.7018 | 2.7035 | 0.00045 |
| C2 | 2.6916 | 2.6951 | 0.00187 |
| C3 | 2.7004 | 2.7035 | 0.00162 |
| C4 | 2.6892 | 2.6951 | 0.00124 |

Here, the learned operator is
`capacity*T_t - div(k*grad(T))`.  In an accurate local balance it should match
the source.  Instead, it was only 0.05--0.19% of the source RMS, so the PDE
residual in the strong-source region remained essentially `-Q`.

The training logs agree with this audit.  The PDE loss started near 1.86 and
ended around 1.98--2.49, with no convincing downward trend.  Continuing the
normal case from 2,000 to 4,000 Adam steps (C1 versus C3 before any phase
change) did not resolve the localized source.

## Mechanism

The nominally source-biased sampler draws:

- longitudinal offset uniformly from `-16 rb` to `+2 rb`;
- transverse offset uniformly from `-4 rb` to `+4 rb`;
- depth over `0--3 rb`.

For this distribution, the median sampled source is effectively zero, only
about 3.37% of source-biased samples exceed 1% of peak, and about 1.21% exceed
10% of peak.  PDE MSE then gives the few high-source points very little weight.

The separate energy loss constrains only the space-time mean residual.  It can
be reduced by a smooth, distributed temperature rise without satisfying the
local laser balance.  This is the pattern present in the checkpoints.

The small sine network (six layers of width 24), lack of moving-coordinate or
Fourier features, stochastic resampling, and Adam-only optimization make the
narrow moving source still harder to learn.  These are contributing factors;
the direct failure demonstrated above is the unresolved local source balance.

## Consequence for phase history

Because the inner PINN never reached the solidus, the history field remained
zero.  The outer-loop convergence test then stopped after two iterations.  It
correctly detected an unchanged history field but did not establish that the
inner thermal PINN had converged.  Therefore the reported history convergence
must not be interpreted as physical convergence.

## Reproduction

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
conda run -n base python -m phase_history_pinn.diagnostics.fact_check_temperature
```

The environment variable works around a duplicate OpenMP runtime issue in the
current local Conda environment; it does not alter the saved checkpoints.
