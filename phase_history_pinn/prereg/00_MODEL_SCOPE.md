# Preregistration 00 - model scope

Status: locked before implementation and scientific runs.

## Objective

Develop a temperature-only, physics-only PINN with irreversible phase history
to study the model-internal effect of a 45-degree downward wedge relative to a
normal single track under one process condition.

The research questions are:

1. Can an irreversible powder-to-consolidated conductivity history be coupled
   to a temperature PINN through a convergent outer fixed-point loop?
2. Within this model, how does that history change the wedge-minus-normal
   thermal indicators?

## Fixed process and material assumptions

- IN718 effective constants: rho = 8000 kg/m3, Cp = 600 J/(kg K),
  k_s = 22 W/(m K), T_amb = 298 K, T_sol = 1533 K, T_liq = 1609 K,
  latent heat = 272 kJ/kg.
- Laser: 300 W, 0.8 m/s, beam radius 40 um, 10 us linear power ramp.
- Powder: D50 = 30 um, porosity = 0.45, solid absorptivity = 0.35,
  argon conductivity = 0.042 W/(m K), layer thickness = 60 um.
- Domain: 1500 x 380 x 240 um (normal) and 1500 x 480 x 240 um
  (wedge). The 45-degree wedge tip is on the scan centerline.

## Powder correlations

The following are Eqs. (10)-(12) reported by Moser et al., Additive
Manufacturing 29 (2019) 100770:

    alpha_eff = 0.053 + 1.37 alpha_s - 1.04 alpha_s^2 + 0.399 alpha_s^3
    beta R = 0.325 + 1.03 alpha_s - 1.22 alpha_s^2 + 0.587 alpha_s^3
    k_eff/k_s = 0.001 + 3.57 theta + 11.7 kappa
                + 15.2 theta kappa - 2.44 theta^2 - 25.2 kappa^2

where R = D50/2, theta = sigma T_ref^3 D50/k_s,
kappa = k_g/k_s, and T_ref = (T_amb + T_sol)/2.

This is explicitly an extrapolation from a metal-powder correlation to
IN718/argon. The reported +/-11% correlation uncertainty does not cover all
material-transfer, gas, particle-shape, or size-distribution uncertainty.

## Governing model

The network predicts only temperature. There is no velocity or pressure
network and no advective energy term. Marangoni transport is represented only
by an effective liquid-conductivity multiplier K_L(T), increasing smoothly
from 1 to 2.8.

The dimensionless conductivity is

    kappa = [1 + (r - 1) s(y,z) (1 - M(x,y,z,t))] K_L(T),

where r = k_eff/k_s, s is the smooth powder indicator, and M is irreversible
phase history. This gives:

- unmolten powder: k = k_eff;
- liquid: k approaches 2.8 k_s;
- resolidified material: k returns to k_s;
- original solid: k_s (or its liquid multiplier at high temperature).

The volumetric heat capacity remains

    rho Cp = [1 - porosity s(y,z)] (rho Cp)_solid

after melting. This preserves the initial powder mass in a fixed geometry.
Changing it to the full-density solid value would require a shrinkage,
porosity, or volume-fraction model and is outside scope.

Latent heat uses the existing Gaussian apparent-heat-capacity formulation.
Evaporative cooling at the top surface and the Beer-Lambert volumetric source
are retained.

## Irreversible phase history

The instantaneous liquid fraction is a continuous smoothstep from zero at
T_sol to one at T_liq. For outer iteration n,

    M_new(x,t) = max(M_old(x,t), max_{tau <= t} f_L(T_n(x,tau))).

M is frozen during each PINN training stage and updated only between stages.
It must be nondecreasing in physical time and between outer iterations.

## Four cases

| Case | Geometry | Phase history |
|---|---|---|
| C1 | normal | static powder |
| C2 | wedge | static powder |
| C3 | normal | irreversible history |
| C4 | wedge | irreversible history |

Geometry effects are C2-C1 and C4-C3. History effects are C3-C1 and C4-C2.

## Explicit exclusions

- No FD, FEA, experimental temperature, or other field data in the loss.
- No FD or FEA comparison in this branch.
- No claim that the PINN is the correct PDE solution.
- No melt flow, recoil pressure, free surface, keyhole, dross, distortion, or
  support-free printability prediction.
- No experimental calibration or claim of real-world error.

Results must be described as internal outputs of this specified effective
thermal model.

