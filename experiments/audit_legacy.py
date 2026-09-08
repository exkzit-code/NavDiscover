"""Reproduce the old solver's false blowup on an exact smooth flow.

Run from the repository root: python -m experiments.audit_legacy
Only reviewed class definitions are loaded; no archived entry point is run.
"""
import ast
import json
from pathlib import Path
import numpy as np
from scipy.fft import fftn, ifftn
from src.solver import NavierStokesSolver, relative_field_difference
from src.initial_conditions import taylor_green_2d


def legacy_namespace(correct_pressure=False):
    path=Path(__file__).resolve().parents[1]/"archive/legacy/reproduce_singularity.py"
    source=path.read_text()
    if correct_pressure:
        old="p_hat = (1j/(dt*self.k2_nozero))*div_u_star"
        assert source.count(old)==1
        source=source.replace(old,"p_hat = -div_u_star/(dt*self.k2_nozero)")
    tree=ast.parse(source,str(path))
    names={"HRFNavierStokesPOC_3D","SymbolicRegressionAnalyzer"}
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in names]
    namespace={"np":np,"fftn":fftn,"ifftn":ifftn}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def main():
    n,nu,T=16,.005,.1
    reference_solver=NavierStokesSolver(n,nu)
    state=taylor_green_2d(reference_solver)
    report={"initial_field":"Exact 2D Taylor-Green embedded in 3D, no filter",
            "n":n,"nu":nu,"target_time":T,"legacy_runs":[]}
    for corrected,dt in [(False,.001),(False,.0005),(False,.00025),(True,.0005)]:
        cls=legacy_namespace(corrected)["HRFNavierStokesPOC_3D"]
        solver=cls(N=n,nu=nu)
        current=state.copy()
        for step in range(round(T/dt)+1):
            max_coefficient=np.max(np.abs(current))
            if not np.isfinite(max_coefficient) or max_coefficient>1e12 or step==round(T/dt):
                break
            current=np.array(solver._time_step(*current,dt))
        time=step*dt
        record={"pressure_corrected_only":corrected,"dt":dt,"time":time,
                "threshold_reached":bool(not np.isfinite(max_coefficient) or max_coefficient>1e12),
                "energy":reference_solver.energy(current),
                "relative_velocity_error":relative_field_difference(current,state*np.exp(-2*nu*time))}
        report["legacy_runs"].append(record)
    result=reference_solver.run(state,final_time=T,dt_max=.0005)
    report["new_solver"]={**result.summary(),"relative_velocity_error":
                          relative_field_difference(result.final_state,state*np.exp(-2*nu*T))}
    times=np.linspace(0,1,11)
    estimator=legacy_namespace()["SymbolicRegressionAnalyzer"]()
    report["legacy_estimator_on_exp_2t"]={"estimated_time":float(estimator._estimate_blow_up_time(times,np.exp(2*times))),
                                          "actual_finite_time_singularity":False}
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
