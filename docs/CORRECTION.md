# Correction to the original NavDiscover claims

The current revision withdraws the repository's claims of detected finite-time
Navier-Stokes singularities, a validated critical separation near 1.26, and
demonstrated self-similar collapse. These claims are not supported by the previous
implementation. The historical PDF on Zenodo is an earlier manuscript; editing
this repository does not amend that external record.

## Confirmed implementation errors

1. In `reproduce_singularity.py` and its copies, the pressure formula
   `p_hat = 1j*div_u_star/(dt*k2)` combined with the velocity correction maps
   Fourier divergence `D` to `(1+1j)*D`, instead of zero. Its magnitude grows by
   sqrt(2) per projection. With the same update convention the pressure must be
   `-div_u_star/(dt*k2)`.
2. The separate Kida implementation maps divergence to `2*D` because of a wrong
   sign. Thus neither implementation enforces the incompressibility it claims.
3. The main enstrophy diagnostic substitutes the velocity-gradient norm for the
   curl norm. They agree for incompressible periodic fields; the erroneous
   projection lets divergence contaminate that diagnostic.
4. The so-called blowup-time estimator extrapolates to `log(coefficient)=25`.
   On the globally finite function `exp(2*t)`, it returns 12.5. The fitted
   exponential-of-polynomial model has no finite-time pole.
5. The profile script labels a fitted trend `decay rate abs(m)` even for positive
   growth. Figure 8 in the manuscript shows profile differences increasing,
   contradicting the written convergence claim. It also chooses the final saved
   snapshot time as T*, without independently estimating a singularity time.
6. Block-repeating Fourier filters with `np.kron` changes the filter in physical
   wavenumber coordinates and can break conjugate symmetry. Cropping a spatial
   subcube for low-resolution discovery also changes the represented domain.

## Controlled audit, before this rewrite

These paired checks used the original routines and a copy with only the pressure
formula corrected. They used no optimized filter and did not rerun the 512^3
experiments. Times below are in the code's simulation units.

| Initial field | Grid | dt | Original | Only pressure formula corrected |
| --- | --- | --- | --- | --- |
| Exact 2D Taylor-Green embedded in 3D | 16^3 | 0.0005 | Threshold at t=0.026; energy 0.25 -> 3.19e19 | Reached t=0.1; relative velocity error 2.50e-9 |
| Legacy s=1.26 axial velocity field | 32^3 | 0.0005 | Threshold at t=0.0245 | Reached t=0.4; energy 0.44915 -> 0.44060 |
| Kida | 64^3 | 0.0001 | Threshold at t=0.0025; energy 0.375 -> 2.18e12 | Reached t=0.01; energy 0.375 -> 0.37459 |

For the exact smooth flow, the original threshold time changed from 0.048 to
0.026 to 0.014 as dt decreased from 0.001 to 0.0005 to 0.00025. It did not
converge to a physical event. This confirms false numerical blowups; it is not
evidence for or against global regularity of the continuum equations.

To repeat the exact-flow pressure comparison and the false time estimate, run
`python -m experiments.audit_legacy`. Last-step magnitudes can differ with roundoff
because the old dynamics amplify numerical errors. The exact solution provides
the independent reference; exploding legacy output is not a target to match.

## What is preserved and what is new

Historical scripts, images and the preprint package are preserved in
`archive/legacy/`, and the full original remains in Git at
`bff8d412d373338b655f4d31e0e3217eb0feb812`.

The active code is a shared Fourier solver with regression tests and explicit
numerical rejection gates. The new search uses actual periodic vorticity tubes,
fixed initial energy, physical-wavenumber filters, and grid/time comparisons.
Its results are new exploratory calculations. None of the old plots are
relabelled as corrected evidence. No novelty or singularity claim is made.
