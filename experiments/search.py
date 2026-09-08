"""Reproducible search for finite-time enstrophy amplification, not blowup."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import scipy
from scipy.optimize import differential_evolution
from src.solver import NavierStokesSolver, relative_field_difference
from src.initial_conditions import antiparallel_tubes, filtered_fixed_energy

BASELINE = [1.26, 3.0, 1.5, 1.0]
BOUNDS = [(0.9, 1.4), (2.0, 4.0), (1.0, 2.0), (1.0, 1.5)]


def simulate(parameters, *, n, nu, final_time, dt_max):
    separation, center, width, amplitude = parameters
    solver = NavierStokesSolver(n, nu)
    state = filtered_fixed_energy(solver, antiparallel_tubes(solver, separation=separation),
                                  center=center, width=width, amplitude=amplitude, energy=0.5)
    return solver.run(state, final_time=final_time, dt_max=dt_max)


def objective_score(result):
    """Rejected or unfinished runs cannot compete with valid candidates."""
    if not result.completed:
        return np.inf
    ratio = result.summary()["enstrophy_ratio"]
    return -ratio if ratio is not None and np.isfinite(ratio) else np.inf


def compare_fields(a, b, *, vorticity=False):
    if not (a.completed and b.completed):
        return None
    first, second = a.final_state, b.final_state
    if vorticity:
        first = NavierStokesSolver(first.shape[1]).curl(first)
        second = NavierStokesSolver(second.shape[1]).curl(second)
    return relative_field_difference(first, second)


def run_search(*, seed=20260908, iterations=2, population=3, final_time=0.2):
    settings = dict(n=24, nu=0.005, final_time=final_time, dt_max=0.005)
    evaluations = []

    def objective(parameters):
        result = simulate(parameters, **settings)
        score = objective_score(result)
        evaluations.append({"parameters": [float(x) for x in parameters], "summary": result.summary(),
                            "objective": float(score) if np.isfinite(score) else None})
        if len(evaluations) % 12 == 0:
            print(f"Evaluated {len(evaluations)} candidates; latest status={result.status}", flush=True)
        return score

    objective(BASELINE)
    optimization = differential_evolution(objective, BOUNDS, seed=seed, maxiter=iterations,
                                         popsize=population, polish=False, workers=1,
                                         updating="immediate", tol=1e-8, atol=0)
    finite = [x for x in evaluations if x["objective"] is not None]
    report = {
        "purpose": "Finite-time enstrophy ratio search; no singularity inference",
        "parameter_order": ["separation", "center_wavenumber", "filter_width", "filter_amplitude"],
        "bounds": BOUNDS, "baseline_parameters": BASELINE,
        "settings": {**settings, "seed": seed, "iterations": iterations, "population_multiplier": population,
                     "initial_energy": 0.5, "tube_radius": 0.65, "perturbation": 0.05,
                     "energy_budget_tolerance": 1e-5, "spectral_tail_tolerance": 1e-3,
                     "spatial_field_tolerance": 0.01, "temporal_field_tolerance": 0.002},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "optimizer": {"success": bool(optimization.success), "message": str(optimization.message),
                      "evaluations": int(optimization.nfev), "global_optimum_established": False},
        "evaluations": evaluations,
    }
    root = Path(__file__).resolve().parents[1]
    report["source_files_sha256"] = {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ["src/solver.py", "src/initial_conditions.py", "experiments/search.py"]
    }
    try:
        report["source_parent_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["source_worktree_modified"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report["source_parent_commit"] = None
    if not finite:
        report.update(status="no_admissible_candidate", refinement_passed=False)
        return report

    best = min(finite, key=lambda x: x["objective"])
    parameters = best["parameters"]
    print(f"Refining best sampled parameters: {parameters}", flush=True)
    results, refinement = {}, []
    for n, dt in [(24,.005), (32,.005), (48,.005), (32,.0025)]:
        result = simulate(parameters, n=n, nu=settings["nu"], final_time=final_time, dt_max=dt)
        results[(n,dt)] = result
        refinement.append({"n": n, "dt_max": dt, "summary": result.summary(), "history": result.history})
    spatial_24_32 = compare_fields(results[(24,.005)], results[(32,.005)])
    spatial_32_48 = compare_fields(results[(32,.005)], results[(48,.005)])
    temporal_32 = compare_fields(results[(32,.005)], results[(32,.0025)])
    vort_24_32 = compare_fields(results[(24,.005)], results[(32,.005)], vorticity=True)
    vort_32_48 = compare_fields(results[(32,.005)], results[(48,.005)], vorticity=True)
    vort_time = compare_fields(results[(32,.005)], results[(32,.0025)], vorticity=True)
    refinement_passed = (all(x.completed for x in results.values())
                         and spatial_24_32 is not None and spatial_24_32 < .01
                         and spatial_32_48 is not None and spatial_32_48 < .01
                         and temporal_32 is not None and temporal_32 < .002
                         and vort_24_32 is not None and vort_24_32 < .01
                         and vort_32_48 is not None and vort_32_48 < .01
                         and vort_time is not None and vort_time < .002)
    baseline_fine = simulate(BASELINE, n=48, nu=settings["nu"], final_time=final_time, dt_max=.005)
    fine_best = results[(48,.005)]
    ratio_best, ratio_baseline = fine_best.summary()["enstrophy_ratio"], baseline_fine.summary()["enstrophy_ratio"]
    report.update(
        status="refinement_passed" if refinement_passed else "refinement_failed",
        best_sampled=best, refinement=refinement, refinement_passed=bool(refinement_passed),
        relative_field_differences={"n24_vs_n32": spatial_24_32, "n32_vs_n48": spatial_32_48,
                                    "dt005_vs_dt0025_at_n32": temporal_32},
        relative_vorticity_differences={"n24_vs_n32": vort_24_32, "n32_vs_n48": vort_32_48,
                                        "dt005_vs_dt0025_at_n32": vort_time},
        baseline_fine=baseline_fine.summary(),
        fine_objective_improvement=(ratio_best-ratio_baseline) if (fine_best.completed and baseline_fine.completed) else None,
        interpretation="These numerical gates assess this finite calculation only. They neither prove regularity nor establish singularity formation."
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/search.json"))
    parser.add_argument("--iterations",type=int,default=2)
    parser.add_argument("--population",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20260908)
    parser.add_argument("--time",type=float,default=.2)
    args=parser.parse_args()
    if args.iterations < 0 or args.population < 1 or not np.isfinite(args.time) or args.time<=0:
        parser.error("iterations >= 0, population >= 1 and time > 0 are required")
    report=run_search(seed=args.seed,iterations=args.iterations,population=args.population,final_time=args.time)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"evaluations":len(report["evaluations"]),
                      "refinement_passed":report["refinement_passed"],"output":str(args.output)},indent=2))
    if not report["refinement_passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
