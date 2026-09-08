# NavDiscover

A small research sandbox for searching for extreme, finite-time behaviour in
periodic incompressible fluid simulations.

**Correction:** the earlier singularity, critical-separation and self-similarity
claims are unsupported. The original solvers contained pressure-projection
errors that generated false numerical blowups. This revision withdraws those
repository claims. See the [correction and audit evidence](docs/CORRECTION.md).
The original scripts and manuscript package remain in [the historical archive](archive/legacy/).

## What the corrected version does

- Solves unforced periodic 3D incompressible Navier-Stokes using one shared solver.
- Checks divergence, energy balance, reality and spectral resolution during a run.
- Searches at fixed initial kinetic energy for the largest sampled `Z(T)/Z(0)`,
  where Z is enstrophy computed from vorticity.
- Rejects invalid runs and checks the selected candidate on finer grids and a
  smaller timestep.
- Records finite outcomes without labelling thresholds or solver failures as
  singularities.

This is exploratory numerical software. Passing its checks is not a proof of
regularity or singularity, and the search does not establish a global optimum.
The historical separation value 1.26 is a baseline only.

## Run

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.search
```

The default search uses seed 20260908, N=24, viscosity 0.005, T=0.2 and initial
energy 0.5. It uses a deliberately small differential-evolution budget. The best
sampled case is compared at N=24,32,48 and at two timestep limits. Its structured
report is written to `results/search.json`. Exit code 2 means no candidate passed
the full refinement checks; inspect the report for the reason.

A recorded run is in [results/corrected_search_2026-09-08.json](results/corrected_search_2026-09-08.json).
The [experiment note](docs/EXPERIMENT.md) explains its actual outcome and limits.
To reproduce that run without overwriting the tracked result:

```bash
python -m experiments.search --seed 20260908 --iterations 2 --population 3 --time 0.2 --output results/search.json
```

## Method and interpretation

See [NUMERICS.md](docs/NUMERICS.md) for the equation, normalization, timestep
controls, fixed-energy objective, initial-condition construction and rejection
thresholds. The corrected tubes are initialized through vorticity and periodic
Biot-Savart inversion, unlike the old axial velocity fields. This is a fresh
experiment, not a validation of the historical result.

| Path | Purpose |
| --- | --- |
| `src/solver.py` | Shared solver, diagnostics and field comparisons |
| `src/initial_conditions.py` | Exact benchmark, Kida flow and periodic vortex tubes |
| `experiments/search.py` | Seeded finite-time search with refinement checks |
| `tests/` | Analytic benchmarks and numerical failure regression tests |
| `docs/` | Numerical specification, correction and observed experiment result |
| `archive/legacy/` | Unchanged historical code with known errors |

The [2025 Zenodo manuscript](https://doi.org/10.5281/zenodo.15769062) documents the
historical work and its now-unsupported conclusions. It is not evidence validating
the corrected solver. The external manuscript has not been amended by this change.

MIT license. See [LICENSE.md](LICENSE.md).
