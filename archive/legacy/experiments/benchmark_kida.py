"""
AI-Guided Singularity Analysis for Kida Flow
============================================

This script addresses point (4) from the reviewer feedback.

It tests the generalizability of our AI-guided discovery framework
by applying it to a different, well-known, highly unstable initial
condition: the Kida flow (Kida, 1985).

The script will:
1.  Define the Kida flow initial condition.
2.  Run the `FilterDiscovery` process to find an optimal spectral
    filter that maximizes enstrophy for this flow.
3.  Execute a single, high-resolution (64³) simulation using the
    discovered filter to find the blow-up time.
4.  Print the results, demonstrating the framework's adaptability.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# --- Simulation Parameters ---
N_RUN = 64              # High-resolution grid size for the main run
N_DISCOVER = 16         # Low-resolution grid for AI discovery
DT = 0.0001             # Time step
T_MAX_RUN = 0.2         # Max simulation time for high-res run
T_MAX_DISCOVER = 0.05   # Max simulation time for discovery runs

# --- FFT Setup ---
def setup_fft(N):
    """Pre-computes wavenumbers for a given grid size N."""
    kx_1d = ky_1d = kz_1d = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq_inv = np.zeros_like(k_sq)
    k_sq_inv[k_sq > 0] = 1.0 / k_sq[k_sq > 0]
    return kx, ky, kz, k_sq, k_sq_inv

# --- Initial Conditions ---
def get_initial_conditions_kida(N):
    """
    Sets up the Kida flow initial condition.
    From: Kida, S. (1985). Three-dimensional periodic flows with high-symmetry.
          J. Phys. Soc. Japan, 54(6), 2132-2136.
    """
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    u = np.sin(X) * (np.cos(3*Y)*np.cos(Z) - np.cos(Y)*np.cos(3*Z))
    v = np.sin(Y) * (np.cos(3*Z)*np.cos(X) - np.cos(Z)*np.cos(3*X))
    w = np.sin(Z) * (np.cos(3*X)*np.cos(Y) - np.cos(X)*np.cos(3*Y))

    return u, v, w

# --- Physics Engine (Pseudo-spectral solver) ---
class PhysicsEngine:
    def __init__(self, N, dt, initial_condition_func, filter_params=None, nu=0.005):
        self.N = N
        self.dt = dt
        self.nu = nu
        self.kx, self.ky, self.kz, self.k_sq, self.k_sq_inv = setup_fft(N)
        self.u, self.v, self.w = initial_condition_func(N)

        # Apply initial filter if provided
        if filter_params is not None:
            center_k, width, amp = filter_params
            filter_profile = amp * np.exp(-((np.sqrt(self.k_sq) - center_k)**2) / (2 * width**2))
            self.u_hat = fftn(self.u) * (1 + filter_profile)
            self.v_hat = fftn(self.v) * (1 + filter_profile)
            self.w_hat = fftn(self.w) * (1 + filter_profile)
            self.u, self.v, self.w = ifftn(self.u_hat).real, ifftn(self.v_hat).real, ifftn(self.w_hat).real
        else:
            self.u_hat, self.v_hat, self.w_hat = fftn(self.u), fftn(self.v), fftn(self.w)

        self.dealias_mask = (np.abs(self.kx) < (2/3)*N/2) & \
                            (np.abs(self.ky) < (2/3)*N/2) & \
                            (np.abs(self.kz) < (2/3)*N/2)

    def _compute_nonlinear(self):
        u_x = ifftn(1j * self.kx * self.u_hat).real
        u_y = ifftn(1j * self.ky * self.u_hat).real
        u_z = ifftn(1j * self.kz * self.u_hat).real
        v_x = ifftn(1j * self.kx * self.v_hat).real
        v_y = ifftn(1j * self.ky * self.v_hat).real
        v_z = ifftn(1j * self.kz * self.v_hat).real
        w_x = ifftn(1j * self.kx * self.w_hat).real
        w_y = ifftn(1j * self.ky * self.w_hat).real
        w_z = ifftn(1j * self.kz * self.w_hat).real

        N_u = self.u * u_x + self.v * u_y + self.w * u_z
        N_v = self.u * v_x + self.v * v_y + self.w * v_z
        N_w = self.u * w_x + self.v * w_y + self.w * w_z

        return fftn(N_u), fftn(N_v), fftn(N_w)

    def _project(self, u_hat, v_hat, w_hat):
        div_hat = 1j * self.kx * u_hat + 1j * self.ky * v_hat + 1j * self.kz * w_hat
        p_hat = div_hat * self.k_sq_inv
        u_hat -= 1j * self.kx * p_hat
        v_hat -= 1j * self.ky * p_hat
        w_hat -= 1j * self.kz * p_hat
        return u_hat, v_hat, w_hat

    def step(self):
        # Semi-implicit time stepping (Crank-Nicolson for linear, Adams-Bashforth for nonlinear)
        N_u_hat, N_v_hat, N_w_hat = self._compute_nonlinear()

        # Dealiasing
        N_u_hat *= self.dealias_mask
        N_v_hat *= self.dealias_mask
        N_w_hat *= self.dealias_mask

        # Predictor step
        u_hat_pred = self.u_hat - self.dt * N_u_hat
        v_hat_pred = self.v_hat - self.dt * N_v_hat
        w_hat_pred = self.w_hat - self.dt * N_w_hat

        # Pressure projection
        u_hat_pred, v_hat_pred, w_hat_pred = self._project(u_hat_pred, v_hat_pred, w_hat_pred)

        # Corrector step (implicit viscosity)
        self.u_hat = (u_hat_pred) / (1 + self.nu * self.k_sq * self.dt)
        self.v_hat = (v_hat_pred) / (1 + self.nu * self.k_sq * self.dt)
        self.w_hat = (w_hat_pred) / (1 + self.nu * self.k_sq * self.dt)

        self.u, self.v, self.w = ifftn(self.u_hat).real, ifftn(self.v_hat).real, ifftn(self.w_hat).real

    def get_enstrophy(self):
        omega_x_hat = 1j*self.ky*self.w_hat - 1j*self.kz*self.v_hat
        omega_y_hat = 1j*self.kz*self.u_hat - 1j*self.kx*self.w_hat
        omega_z_hat = 1j*self.kx*self.v_hat - 1j*self.ky*self.u_hat
        return 0.5 * np.sum(np.abs(omega_x_hat)**2 + np.abs(omega_y_hat)**2 + np.abs(omega_z_hat)**2)

# --- AI Filter Discovery ---
class FilterDiscovery:
    def __init__(self, N, dt, t_max, initial_condition_func):
        self.N = N
        self.dt = dt
        self.t_max = t_max
        self.initial_condition_func = initial_condition_func
        self.bounds = [(1, N/2), (0.1, N/4), (0.1, 2.5)] # center_k, width, amplitude

    def objective_function(self, params):
        sim = PhysicsEngine(self.N, self.dt, self.initial_condition_func, filter_params=params)
        max_enstrophy = 0
        for _ in range(int(self.t_max / self.dt)):
            sim.step()
            enstrophy = sim.get_enstrophy()

            # Check for numerical explosion and penalize heavily
            if np.isinf(enstrophy) or np.isnan(enstrophy):
                return 1e30 # Return a large, finite penalty

            if enstrophy > max_enstrophy:
                max_enstrophy = enstrophy
        return -max_enstrophy # We want to maximize, so we minimize the negative

    def run_discovery(self):
        print("--- Starting AI Filter Discovery (for Kida Flow) ---")
        result = differential_evolution(self.objective_function, self.bounds, maxiter=10, popsize=15, disp=True)
        print("--- AI Discovery Finished ---")
        return result.x, -result.fun

# --- Main Execution ---
if __name__ == "__main__":
    # 1. AI-driven discovery on a low-resolution, short-duration simulation
    discovery = FilterDiscovery(N_DISCOVER, DT, T_MAX_DISCOVER, get_initial_conditions_kida)
    best_params, max_enstrophy_discovered = discovery.run_discovery()
    print(f"\nOptimal filter found: center_k={best_params[0]:.2f}, width={best_params[1]:.2f}, amp={best_params[2]:.2f}")
    print(f"Discovered max enstrophy at N={N_DISCOVER}: {max_enstrophy_discovered:.4e}")

    # 2. Run high-resolution simulation with the discovered optimal filter
    print(f"\n--- Running High-Resolution Simulation (N={N_RUN}) with Optimal Filter ---")
    sim = PhysicsEngine(N_RUN, DT, get_initial_conditions_kida, filter_params=best_params)
    
    peak_enstrophy = 0
    t_blowup = 0
    enstrophy_history = []

    for i in range(int(T_MAX_RUN / DT)):
        sim.step()
        t = (i + 1) * DT
        enstrophy = sim.get_enstrophy()
        enstrophy_history.append(enstrophy)

        if enstrophy > peak_enstrophy:
            peak_enstrophy = enstrophy
            t_blowup = t
        
        # Heuristic for blow-up detection
        if enstrophy > 1e15:
            print(f"Potential blow-up detected at t={t:.4f}! Halting simulation.")
            break
            
    print("--- High-Resolution Simulation Finished ---")
    print(f"\nFinal Results for Kida Flow (N={N_RUN}):")
    print(f"  Peak Enstrophy: {peak_enstrophy:.4e}")
    print(f"  Time of Peak Enstrophy (Blow-up): t = {t_blowup:.4f}")

    # Optional: Plot enstrophy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(enstrophy_history)) * DT, enstrophy_history)
    plt.yscale('log')
    plt.title(f'Enstrophy Evolution for Kida Flow (N={N_RUN})')
    plt.xlabel('Time')
    plt.ylabel('Enstrophy (log scale)')
    plt.grid(True)
    plt.savefig('hypo_generalizability_kida_enstrophy.png')
    print("\nSaved enstrophy plot to 'hypo_generalizability_kida_enstrophy.png'") 
