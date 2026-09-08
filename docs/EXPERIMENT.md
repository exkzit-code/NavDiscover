# Corrected exploratory experiment, 2026-09-08

The recorded search completed 37 evaluations (one baseline and 36 optimizer
calls) at N=24, T=0.2, viscosity 0.005 and fixed initial energy 0.5. All evaluated
candidates passed the per-run screening gates. The best sampled case also passed
the stated velocity and vorticity refinement checks.

**Enstrophy decreased. This experiment found no net enstrophy amplification over
the tested interval and provides no evidence of a singularity.**

## Search outcome

The objective is Z(T)/Z(0), not absolute final enstrophy. The best sampled
parameters were separation 1.364151519, filter center 2.035616127, width
1.232686073, amplitude 1.139781363. This is a sampled parameter choice, not a
critical separation. The optimizer exhausted its small two-generation budget;
its success flag is false, and no optimization convergence or global optimality
is claimed.

At N=48, the selected ratio was 0.992736031048, a 0.7264% decline. The historical
baseline separation 1.26, with amplitude-one filter, gave 0.992492618949, a
0.7507% decline. The selected case retained only about 0.0243 percentage points
more of its initial enstrophy. This is a modest difference in this particular
objective, not a new physical mechanism. Initial enstrophy is allowed to vary;
initial kinetic energy is fixed for every candidate.

## Refinement evidence

| Grid | dt limit | Largest actual dt | Z(T)/Z(0) | Largest relative energy-budget error |
| --- | --- | --- | --- | --- |
| 24^3 | 0.0050 | 0.005000 | 0.992736032128 | 1.08e-12 |
| 32^3 | 0.0050 | 0.005000 | 0.992736031053 | 1.08e-12 |
| 48^3 | 0.0050 | 0.003637 | 0.992736031048 | 3.17e-13 |
| 32^3 | 0.0025 | 0.002500 | 0.992736031047 | 7.55e-14 |

The N=48 timestep was automatically reduced by the advective cap. The dedicated
N=32 timestep comparison actually halved the largest step from 0.005 to 0.0025.

| Comparison | Relative velocity L2 difference | Relative vorticity L2 difference |
| --- | --- | --- |
| 24^3 vs 32^3 | 6.993e-05 | 3.864e-04 |
| 32^3 vs 48^3 | 1.786e-06 | 1.387e-05 |
| dt 0.005 vs 0.0025 at 32^3 | 2.970e-10 | 1.289e-09 |

All comparisons include Fourier modes missing from the coarser grid. Spatial
thresholds were 0.01 and temporal thresholds 0.002 for both fields. These are
finite numerical checks, not rigorous error bounds or regularity criteria.

## Reproduce and inspect

```bash
python -m unittest discover -s tests -v
python -m experiments.search --seed 20260908 --iterations 2 --population 3 --time 0.2 --output results/search.json
python -m experiments.audit_legacy
```

The checked run used Python 3.12.13, NumPy 2.3.5, SciPy
1.17.0. The package versions are in `requirements-reproduce.txt`.
The [JSON report](../results/corrected_search_2026-09-08.json) retains every
candidate, the refinement histories, rejection status, optimizer termination,
parameters, environment and SHA-256 hashes of the numerical source files. Its
parent-commit field identifies the checkout used before committing the repair;
the recorded worktree-modified flag and source hashes identify the new code.

The old s=1.26 plots, viscosity sweep, Kida claim and 512^3 study have not been
revalidated by this experiment. The initial tube family is different and is
specified in [NUMERICS.md](NUMERICS.md). Longer times or sharper structures may
require much greater spatial resolution and an independent numerical review.
