# Corrective C1 pilot results

Date: 2026-10-09

Neither pilot is accepted as a converged thermal solution.  The temperatures
below are recorded only to show why temperature alone is not an adequate gate.

| Metric | Original C1 | Corrective pilot 1 | Corrective pilot 2 |
|---|---:|---:|---:|
| Network | 6 x 24 | 6 x 24 | 6 x 48 |
| Adam steps | 2,000 | 2,000 | 5,000 |
| Source residual ratio | approximately 1.000 | 0.7254 | 0.5596 |
| Hotspot residual ratio | approximately 1.000 | 0.7200 | 0.5464 |
| PDE/zero-rise ratio | approximately 1.000 | 0.5293 | 0.3168 |
| Final energy error (J) | near zero on last random batch | +0.1341 | -0.0144 |
| Snapshot peak (K) | 472.6 | 8,388.5 | 12,111.4 |
| Thermal gate | fail | fail | fail |

The new sampling, split losses, and laser-following coordinates materially
improved local source learning.  Increasing capacity and applying learning-rate
decay improved it further.  However, the fixed tolerance is 0.10, and the
second pilot remains more than five times above it.  Its extreme snapshot peak
also shows unresolved local/upper-boundary behavior.  C2--C4 were therefore not
rerun.

Further progress now requires a method-level change rather than more iterations
of the same Adam setup.  The next candidate should be preregistered separately:
time-window/causal training in the laser-following frame, followed by a fixed
collocation L-BFGS stage or residual-adaptive refinement.  No such change is
included in these pilots.
