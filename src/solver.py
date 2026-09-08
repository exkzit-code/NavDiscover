"""Unforced incompressible Navier-Stokes on the periodic box [0, 2*pi)^3.

Fourier Galerkin truncation with strict 2/3 dealiasing, Leray projection,
rotational-form nonlinearity and explicit RK4. A completed run is a finite
numerical calculation, never a singularity detection or a regularity proof.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import fftn, ifftn

AXES = (1, 2, 3)


@dataclass
class RunResult:
    status: str
    reason: str | None
    history: list
    final_state: np.ndarray

    @property
    def completed(self):
        return self.status == "completed"

    def summary(self):
        first, last = self.history[0], self.history[-1]
        return {
            "status": self.status,
            "reason": self.reason,
            "final_time": last["time"],
            "steps": len(self.history) - 1,
            "max_actual_dt": max((b["time"]-a["time"] for a,b in zip(self.history,self.history[1:])), default=0.0),
            "initial_energy": first["energy"],
            "final_energy": last["energy"],
            "initial_enstrophy": first["enstrophy"],
            "final_enstrophy": last["enstrophy"],
            "enstrophy_ratio": last["enstrophy"] / first["enstrophy"] if first["enstrophy"] > 0 else None,
            "max_energy_budget_error": max(x["energy_budget_error"] for x in self.history),
            "max_relative_divergence": max(x["relative_divergence"] for x in self.history),
            "max_reality_error": max(x["reality_error"] for x in self.history),
            "max_spectral_tail_fraction": max(x["spectral_tail_fraction"] for x in self.history),
        }


class NavierStokesSolver:
    def __init__(self, n=24, nu=0.005):
        if isinstance(n, bool) or int(n) != n or n < 8 or n % 2:
            raise ValueError("n must be an even integer >= 8")
        if not np.isfinite(nu) or nu < 0:
            raise ValueError("nu must be finite and nonnegative")
        self.n, self.nu = int(n), float(nu)
        modes = np.fft.fftfreq(self.n) * self.n
        self.k = np.array(np.meshgrid(modes, modes, modes, indexing="ij"))
        self.k2 = np.sum(self.k**2, axis=0)
        self.inv_k2 = np.divide(1.0, self.k2, out=np.zeros_like(self.k2), where=self.k2 != 0)
        # Retaining only |k_i| < n/3 avoids quadratic aliasing on the n^3 grid.
        self.mask = np.all(np.abs(self.k) < self.n / 3, axis=0)
        self.max_mode = float(np.max(np.abs(self.k[:, self.mask])))
        self.max_k2 = float(np.max(self.k2[self.mask]))
        self.tail_mask = self.mask & np.any(np.abs(self.k) > 0.75 * self.max_mode, axis=0)
        self.normalization = float(self.n**6)

    def project(self, state):
        """P_k u = u - k (k dot u)/|k|^2, including the unchanged mean mode."""
        state = np.asarray(state, dtype=np.complex128)
        return state - self.k * (np.sum(self.k * state, axis=0) * self.inv_k2)

    def prepare(self, velocity):
        """Explicitly truncate and project a real physical initial velocity."""
        velocity = np.asarray(velocity)
        if velocity.shape != (3, self.n, self.n, self.n):
            raise ValueError("velocity must have shape (3, n, n, n)")
        if np.iscomplexobj(velocity) or not np.all(np.isfinite(velocity)):
            raise ValueError("initial physical velocity must be real and finite")
        return self.project(fftn(velocity, axes=AXES) * self.mask)

    def curl(self, state):
        kx, ky, kz = self.k
        u, v, w = state
        return 1j * np.array([ky*w-kz*v, kz*u-kx*w, kx*v-ky*u])

    def energy(self, state):
        return float(0.5 * np.sum(np.abs(state)**2) / self.normalization)

    def rhs(self, state):
        # P(u cross omega) equals -P((u dot grad)u) for divergence-free u.
        velocity = ifftn(state, axes=AXES).real
        omega = ifftn(self.curl(state), axes=AXES).real
        u, v, w = velocity
        ox, oy, oz = omega
        nonlinear = np.array([v*oz-w*oy, w*ox-u*oz, u*oy-v*ox])
        return self.project(fftn(nonlinear, axes=AXES) * self.mask) - self.nu*self.k2*state

    def dissipation(self, state):
        return float(self.nu * np.sum(np.abs(self.curl(state))**2) / self.normalization)

    def step(self, state, dt):
        """Return RK4 state and its consistently integrated viscous dissipation."""
        a = state
        k1 = self.rhs(a)
        b = state + 0.5*dt*k1
        k2 = self.rhs(b)
        c = state + 0.5*dt*k2
        k3 = self.rhs(c)
        d = state + dt*k3
        k4 = self.rhs(d)
        updated = self.project((state + dt/6*(k1+2*k2+2*k3+k4)) * self.mask)
        loss = dt/6*(self.dissipation(a)+2*self.dissipation(b)+2*self.dissipation(c)+self.dissipation(d))
        return updated, loss

    def diagnostics(self, state):
        norm2 = np.sum(np.abs(state)**2)
        gradient2 = np.sum(self.k2*np.abs(state)**2)
        divergence = np.sum(self.k*state, axis=0)
        physical = ifftn(state, axes=AXES)
        return {
            "energy": self.energy(state),
            "enstrophy": float(0.5*np.sum(np.abs(self.curl(state))**2)/self.normalization),
            "relative_divergence": float(np.sqrt(np.sum(np.abs(divergence)**2)/max(gradient2, 1e-300))),
            "reality_error": float(np.sqrt(np.sum(physical.imag**2)/max(np.sum(np.abs(physical)**2), 1e-300))),
            "spectral_tail_fraction": float(np.sum(np.abs(state[:, self.tail_mask])**2)/max(norm2, 1e-300)),
            "out_of_band_fraction": float(np.sum(np.abs(state[:, ~self.mask])**2)/max(norm2, 1e-300)),
        }

    def run(self, state, *, final_time=0.2, dt_max=0.005,
            energy_tolerance=1e-5, tail_tolerance=1e-3, max_steps=100000):
        """Stop and mark a run rejected if a numerical gate fails.

        Tail energy is a conservative screening diagnostic, not an error bound.
        Spatial and temporal refinement are still required for interpretation.
        """
        for value in (final_time, dt_max, energy_tolerance, tail_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("times and tolerances must be finite and positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        state = np.array(state, dtype=np.complex128, copy=True)
        if state.shape != (3, self.n, self.n, self.n) or not np.all(np.isfinite(state)):
            raise ValueError("spectral state must be finite with shape (3, n, n, n)")
        initial_energy = self.energy(state)
        scale = max(initial_energy, 1e-30)
        time, integrated_loss = 0.0, 0.0
        history = []
        while True:
            if not np.all(np.isfinite(state)) or not np.isfinite(integrated_loss):
                return RunResult("rejected", "nonfinite_state", history, state)
            metrics = self.diagnostics(state)
            metrics.update(time=float(time), energy_budget_error=float(abs(metrics["energy"]+integrated_loss-initial_energy)/scale))
            history.append(metrics)
            checks = (
                ("relative_divergence", 1e-10),
                ("reality_error", 1e-10),
                ("out_of_band_fraction", 1e-20),
                ("energy_budget_error", energy_tolerance),
                ("spectral_tail_fraction", tail_tolerance),
            )
            for key, tolerance in checks:
                if not np.isfinite(metrics[key]) or metrics[key] > tolerance:
                    return RunResult("rejected", key, history, state)
            if time >= final_time:
                return RunResult("completed", None, history, state)
            if len(history) - 1 >= max_steps:
                return RunResult("rejected", "step_limit", history, state)
            velocity = ifftn(state, axes=AXES).real
            advective_rate = self.max_mode * np.sum(np.max(np.abs(velocity), axis=(1, 2, 3)))
            dt_advective = 0.25 / max(advective_rate, 1e-30)
            dt_diffusive = 0.5 / max(self.nu*self.max_k2, 1e-30)
            dt = min(dt_max, dt_advective, dt_diffusive, final_time-time)
            if time + dt == time:
                return RunResult("rejected", "timestep_underflow", history, state)
            # Nonfinite stages are detected at the next iteration, never scored.
            with np.errstate(over="ignore", invalid="ignore"):
                state, loss = self.step(state, dt)
            integrated_loss += loss
            time = min(final_time, time + dt)


def relative_field_difference(coarse, fine):
    """Relative L2 difference on the fine Fourier grid, including unresolved modes."""
    n, m = coarse.shape[1], fine.shape[1]
    if n > m:
        return relative_field_difference(fine, coarse)
    modes = np.rint(np.fft.fftfreq(n)*n).astype(int)
    indices = modes % m
    lifted = np.zeros_like(fine)
    for component in range(3):
        lifted[component][np.ix_(indices, indices, indices)] = coarse[component] / n**3
    reference = fine / m**3
    return float(np.linalg.norm(lifted-reference)/max(np.linalg.norm(reference), 1e-30))
