"""Periodic, divergence-free initial fields and explicit fixed-energy filtering."""
import numpy as np
from .solver import AXES
from scipy.fft import fftn


def coordinates(n):
    x = np.arange(n)*2*np.pi/n
    return np.meshgrid(x, x, x, indexing="ij")


def taylor_green_2d(solver):
    """Exact smooth 2D Taylor-Green flow embedded in 3D; decay exp(-2*nu*t)."""
    x, y, z = coordinates(solver.n)
    return solver.prepare(np.array([np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y), np.zeros_like(z)]))


def kida(solver):
    x, y, z = coordinates(solver.n)
    velocity = np.array([
        np.sin(x)*(np.cos(3*y)*np.cos(z)-np.cos(y)*np.cos(3*z)),
        np.sin(y)*(np.cos(3*z)*np.cos(x)-np.cos(z)*np.cos(3*x)),
        np.sin(z)*(np.cos(3*x)*np.cos(y)-np.cos(x)*np.cos(3*y)),
    ])
    return solver.prepare(velocity)


def antiparallel_tubes(solver, separation=1.26, radius=0.65, perturbation=0.05):
    """Opposite axial vorticity tubes using a periodic Gaussian Fourier series.

    Velocity is recovered from vorticity by Biot-Savart in the periodic box.
    This replaces the legacy axial velocity jets; it is a new initial family.
    Separation and radius are box-coordinate lengths, not universal constants.
    """
    if not 0 < separation < 2*np.pi or not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid separation or radius")
    if not np.isfinite(perturbation):
        raise ValueError("perturbation must be finite")
    kx, ky, kz = solver.k
    x1, x2, y0 = np.pi-separation/2, np.pi+separation/2, np.pi
    omega_z = (solver.n**3 * np.exp(-0.5*radius**2*(kx*kx+ky*ky))
               * (np.exp(-1j*kx*x1)-np.exp(-1j*kx*x2))*np.exp(-1j*ky*y0)
               * (kz == 0) * solver.mask)
    state = np.array([1j*ky*omega_z*solver.inv_k2, -1j*kx*omega_z*solver.inv_k2, np.zeros_like(omega_z)])
    # A fixed divergence-free transverse perturbation makes the flow depend on z.
    _, _, z = coordinates(solver.n)
    seed = np.array([perturbation*np.sin(2*z), perturbation*np.cos(2*z), np.zeros_like(z)])
    state += fftn(seed, axes=AXES) * solver.mask
    return solver.project(state)


def filtered_fixed_energy(solver, state, *, center=3.0, width=1.5, amplitude=1.0, energy=0.5):
    """Evaluate the same filter at actual wavenumbers on every resolution."""
    if not all(np.isfinite(x) for x in (center, width, amplitude, energy)) or width <= 0 or amplitude <= 0 or energy <= 0:
        raise ValueError("filter parameters must be finite; width, amplitude and energy positive")
    profile = 1+(amplitude-1)*np.exp(-0.5*((np.sqrt(solver.k2)-center)/width)**2)
    filtered = solver.project(state * profile * solver.mask)
    current = solver.energy(filtered)
    if current <= 0 or not np.isfinite(current):
        raise ValueError("cannot normalize an empty or invalid field")
    return filtered*np.sqrt(energy/current)
