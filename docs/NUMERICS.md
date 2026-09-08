# Numerical method and scope

We solve the **unforced**, three-dimensional incompressible Navier-Stokes equations
on the periodic box `[0, 2*pi)^3`. Viscosity is a constant nonnegative `nu`.
This code uses the ordinary Laplacian; it does not vary the dissipation exponent.

For nonzero Fourier mode k the projection is
`P_k u = u - k (k dot u)/|k|^2`. The mean mode is unchanged. We retain only
`|k_i| < N/3` and truncate initial fields and every Runge-Kutta stage/nonlinear
term consistently. The nonlinear term is `P(u cross curl(u))`; projected onto
divergence-free fields this equals the usual negative convective term.

Time integration uses classical RK4 on both convection and viscosity. Each step
is capped by the requested `dt_max`, an advective estimate
`0.25 / (k_max * sum_i max_x |u_i|)`, and a diffusive estimate
`0.5 / (nu * max_retained |k|^2)`. These are conservative timestep controls, not
proofs of stability. No global warning suppression is used.

Energy and enstrophy are **volume means**:

- `E = 0.5 * mean(|u|^2)`.
- `Z = 0.5 * mean(|curl(u)|^2)`.
- For unforced smooth periodic flow, `E(t) + 2*nu*integral_0^t Z(s) ds = E(0)`.

FFT sums are normalized by `N^6`. Enstrophy is computed from the curl, rather
than substituting a gradient norm which would conceal divergence contamination.
Viscous dissipation is integrated with the RK4 stage weights.

## Numerical screening gates

Every recorded state must be finite, essentially real and divergence-free. A run
is rejected if any of these gates fail:

| Diagnostic | Default threshold |
| --- | --- |
| Relative divergence, compared with the velocity-gradient norm | `1e-10` |
| Imaginary-to-total physical velocity RMS | `1e-10` |
| Fraction of energy outside retained modes | `1e-20` |
| Relative integrated energy-budget error | `1e-5` |
| Energy in modes with any component above 75% of retained component cutoff | `1e-3` |

The spectral-tail gate is a heuristic warning for lost spatial resolution. Small
tail energy by itself cannot certify resolved vorticity or gradients. The search
also compares full final velocity and vorticity fields, including modes missing
on the coarser grid, at N=24,32,48. Both adjacent-grid relative differences must
be below 1% for both fields. N=32 with `dt_max=0.005` and `0.0025` must differ by
less than 0.2% for both fields. Reports retain
the actual largest step taken so that the effect of timestep caps is visible.
These tolerances support a small exploratory experiment, not a singularity study.

## Corrected initial conditions and objective

Anti-parallel tubes are defined as opposite axial **vorticity** distributions
through a periodic Gaussian Fourier series, radius 0.65, then converted to velocity
using periodic Biot-Savart inversion. A fixed transverse velocity perturbation
of amplitude 0.05 supplies z dependence. This is a new family; the old code put
Gaussian profiles directly into axial velocity and did not define the claimed
vorticity tubes.

The filter `1 + (A-1)*exp(-0.5*((|k|-center)/width)^2)` is evaluated at the actual
wave numbers on each grid. It is never upscaled with `np.kron`. All candidates
are projected, truncated, then normalized to the same initial energy `E(0)=0.5`.
The objective is the finite ratio `Z(T)/Z(0)` at the same fixed T. Rejected or
unfinished runs receive no admissible score. A ratio below 1 means enstrophy
decreased over that interval; the initial value is not inserted as an artificial
lower bound on the objective.

The small seeded differential-evolution budget identifies the **best sampled**
case only. It does not establish an optimum, universal separation, singularity,
novel physical mechanism or regularity theorem. The previous value 1.26 is used
as a historical baseline, with no special physical status.
