"""
HRF Navier-Stokes Singularity Discovery POC - 3D Extension
=========================================================

This script extends the 2D proof-of-concept to 3D to explore
singularities in the full Navier-Stokes equations, which is the
subject of the Clay Millennium Prize.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# For symbolic regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import sympy as sp

class HRFNavierStokesPOC_3D:
    """
    3D Navier-Stokes solver using a pseudo-spectral method with
    singularity detection capabilities.
    """
    
    def __init__(self, N=32, nu=0.01):
        self.N = N  # Grid size (N x N x N)
        self.nu = nu  # Viscosity
        self.discovered_filter = None # Add placeholder for the filter
        
        # 3D Wavenumbers for FFT-based derivatives
        kx_1d = np.fft.fftfreq(N) * N
        ky_1d = np.fft.fftfreq(N) * N
        kz_1d = np.fft.fftfreq(N) * N
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        # Avoid division by zero for the k=0 mode in pressure solve
        self.k2_nozero = self.k2.copy()
        self.k2_nozero[0, 0, 0] = 1e-10

        # Dealiasing mask (2/3 rule)
        kmax = self.N * 2/3 / 2
        self.dealias_mask = (np.abs(self.kx) < kmax) & \
                            (np.abs(self.ky) < kmax) & \
                            (np.abs(self.kz) < kmax)

        # Storage for analysis
        self.coefficient_history = []
        self.time_history = []
        self.energy_history = []
        self.enstrophy_history = []
        self.singularity_indicators = []

    def simulate(self, u0, v0, w0, T=1.0, dt=0.001, verbose=True):
        """
        Simulate 3D Navier-Stokes
        """
        # Project initial conditions to spectral space using FFT
        u_hat = fftn(u0)
        v_hat = fftn(v0)
        w_hat = fftn(w0)
        
        # Apply the discovered filter at the start of the simulation
        if self.discovered_filter is not None:
            if verbose: print("Applying discovered spectral filter...")
            u_hat *= self.discovered_filter
            v_hat *= self.discovered_filter
            w_hat *= self.discovered_filter
        
        t = 0
        step = 0
        
        if verbose: print("Starting 3D simulation...")
        
        while t < T:
            # Store for analysis
            self.coefficient_history.append({
                'u_hat': u_hat.copy(),
                'v_hat': v_hat.copy(),
                'w_hat': w_hat.copy()
            })
            self.time_history.append(t)
            
            # Calculate spectral indicators
            self._calculate_indicators(u_hat, v_hat, w_hat)
            
            # Check for blow-up
            max_coeff = max(np.max(np.abs(u_hat)), np.max(np.abs(v_hat)), np.max(np.abs(w_hat)))
            if max_coeff > 1e12 or np.isnan(max_coeff):
                if verbose:
                    print(f"Potential blow-up detected at t={t:.4f}")
                    print(f"Max coefficient magnitude: {max_coeff:.2e}")
                break
            
            # Semi-implicit time stepping with pressure correction
            u_hat, v_hat, w_hat = self._time_step(u_hat, v_hat, w_hat, dt)
            
            t += dt
            step += 1
            
            if verbose and step % 20 == 0:
                print(f"t = {t:.3f}, max|u_hat| = {np.max(np.abs(u_hat)):.3e}")
        
        return self._analyze_results()
    
    def _time_step(self, u_hat, v_hat, w_hat, dt):
        """
        Perform one time step using pseudo-spectral method with pressure projection.
        """
        # --- Calculate nonlinear terms ---
        u = ifftn(u_hat).real
        v = ifftn(v_hat).real
        w = ifftn(w_hat).real

        ux, uy, uz = ifftn(1j*self.kx*u_hat).real, ifftn(1j*self.ky*u_hat).real, ifftn(1j*self.kz*u_hat).real
        vx, vy, vz = ifftn(1j*self.kx*v_hat).real, ifftn(1j*self.ky*v_hat).real, ifftn(1j*self.kz*v_hat).real
        wx, wy, wz = ifftn(1j*self.kx*w_hat).real, ifftn(1j*self.ky*w_hat).real, ifftn(1j*self.kz*w_hat).real

        Nu = -(u*ux + v*uy + w*uz)
        Nv = -(u*vx + v*vy + w*vz)
        Nw = -(u*wx + v*wy + w*wz)

        Nu_hat, Nv_hat, Nw_hat = fftn(Nu), fftn(Nv), fftn(Nw)

        # Dealiasing
        Nu_hat[~self.dealias_mask] = 0
        Nv_hat[~self.dealias_mask] = 0
        Nw_hat[~self.dealias_mask] = 0

        # --- Solve for provisional velocity (explicit convection, implicit diffusion) ---
        diff_term = 1 / (1 + self.nu * dt * self.k2)
        u_star_hat = (u_hat + dt * Nu_hat) * diff_term
        v_star_hat = (v_hat + dt * Nv_hat) * diff_term
        w_star_hat = (w_hat + dt * Nw_hat) * diff_term

        # --- Pressure projection step to enforce incompressibility ---
        # div(u_star) in spectral space
        div_u_star_hat = 1j*self.kx*u_star_hat + 1j*self.ky*v_star_hat + 1j*self.kz*w_star_hat
        
        # Solve Poisson eq for pressure: k^2 * p_hat = (i/dt) * div(u_star_hat)
        pressure_hat = (1j / (dt * self.k2_nozero)) * div_u_star_hat
        
        # Correct velocity field
        u_hat_new = u_star_hat - dt * (1j * self.kx * pressure_hat)
        v_hat_new = v_star_hat - dt * (1j * self.ky * pressure_hat)
        w_hat_new = w_star_hat - dt * (1j * self.kz * pressure_hat)
        
        return u_hat_new, v_hat_new, w_hat_new

    def _calculate_indicators(self, u_hat, v_hat, w_hat):
        """Calculate various indicators of potential singularity in 3D"""
        # Energy spectrum (with FFT normalization)
        energy_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        energy = energy_density / (self.N**6)
        self.energy_history.append(np.sum(energy))
        
        # Enstrophy (integral of vorticity squared)
        enstrophy = np.sum(self.k2 * energy)
        self.enstrophy_history.append(enstrophy)
        
        # High-frequency energy concentration (singularity indicator)
        total_energy = np.sum(energy)
        high_k_mask = self.k2 > (self.N/4)**2
        high_freq_energy = np.sum(energy[high_k_mask])
        
        concentration = high_freq_energy / (total_energy + 1e-12)
        self.singularity_indicators.append(concentration)
    
    def _analyze_results(self):
        """Analyze simulation results for singularity patterns"""
        # Get the final state for visualization
        if not self.coefficient_history:
            return { 'time_history': [] } # Return empty if no steps were taken

        final_coeffs = self.coefficient_history[-1]
        final_state = {
            'u_hat': final_coeffs['u_hat'],
            'v_hat': final_coeffs['v_hat'],
            'w_hat': final_coeffs['w_hat']
        }

        return {
            'coefficient_history': self.coefficient_history,
            'time_history': self.time_history,
            'energy_history': self.energy_history,
            'enstrophy_history': self.enstrophy_history,
            'singularity_indicators': self.singularity_indicators,
            'final_state': final_state
        }


class SymbolicRegressionAnalyzer:
    """
    Discover mathematical laws governing coefficient evolution. Unchanged for 3D.
    """
    
    def __init__(self):
        self.discovered_laws = []
        
    def discover_blow_up_law(self, times, max_coeffs):
        print("\nDiscovering blow-up laws...")
        X = times.reshape(-1, 1)
        y = np.log(max_coeffs + 1e-12)
        
        best_score, best_model, best_formula = -np.inf, None, None
        
        for degree in range(1, 5):
            model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.01))
            model.fit(X, y)
            score = model.score(X, y)
            
            if score > best_score:
                best_score, best_model = score, model
                ridge_coef = model.named_steps['ridge'].coef_
                intercept = model.named_steps['ridge'].intercept_
                t = sp.Symbol('t')
                formula = intercept
                feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(['t'])
                for i, name in enumerate(feature_names[1:]):
                    power = int(name.split('^')[1]) if '^' in name else 1
                    formula += ridge_coef[i+1] * t**power
                best_formula = sp.exp(formula)
        
        blow_up_time = self._estimate_blow_up_time(times, max_coeffs)
        result = {
            'formula': best_formula,
            'score': best_score,
            'blow_up_time': blow_up_time,
            'type': self._classify_singularity(times, max_coeffs)
        }
        self.discovered_laws.append(result)
        return result
    
    def _estimate_blow_up_time(self, times, values):
        if len(times) < 5: return None
        log_values = np.log(values + 1e-12)
        slope = (log_values[-1] - log_values[-5]) / (times[-1] - times[-5])
        if slope > 0:
            return times[-1] + (25 - log_values[-1]) / slope
        return None
    
    def _classify_singularity(self, times, values):
        if len(values) < 10: return "unknown"
        growth_rates = np.diff(np.log(values + 1e-12)) / np.diff(times)
        if np.mean(np.diff(growth_rates)) > 0.01: return "super-exponential"
        elif np.std(growth_rates) < 0.1 * np.mean(growth_rates): return "exponential"
        else: return "irregular"


class FilterDiscovery:
    """
    Use an evolutionary algorithm to discover an optimal 3D spectral filter
    that is sensitive to singularity formation.
    """
    def __init__(self, N=16, nu=0.01):
        self.N = N
        self.nu = nu
        self.discovered_filters = []
        
        # Pre-calculate wavenumbers for the discovery simulations
        kx_1d = np.fft.fftfreq(N) * N
        ky_1d = np.fft.fftfreq(N) * N
        kz_1d = np.fft.fftfreq(N) * N
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        self.k_mag = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)

    def discover_filter(self, u0, v0, w0, objective_type='peak_enstrophy'):
        """
        Run the evolutionary optimization to find the best filter parameters.
        """
        print(f"\n-- Starting discovery of optimal spectral filter (Objective: {objective_type}) --")

        def objective(params):
            # Create a filter from the evolutionary parameters
            filter_3d = self._params_to_filter(params)
            
            # Evaluate how this filter performs in a short simulation
            score = self._evaluate_filter(filter_3d, u0, v0, w0, objective_type)
            
            # We want to maximize the score, but differential_evolution minimizes
            return -score

        # Parameter bounds for the filter shape: [center_k, width, amplitude]
        bounds = [(self.N/8, self.N/2), (self.N/16, self.N/4), (1.0, 1.5)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=10, # Keep low for POC
            popsize=5,  # Keep low for POC
            disp=True
        )
        
        optimal_filter = self._params_to_filter(result.x)
        self.discovered_filters.append(optimal_filter)
        print("-- Filter discovery complete --")
        return optimal_filter

    def _params_to_filter(self, params):
        """
        Convert a parameter vector from the optimizer into a 3D filter.
        This defines the shape of the filter we are searching for.
        Example: A radial Gaussian bump filter.
        """
        center_k, width, amplitude = params
        
        # Create a filter that amplifies a specific band of wavenumbers
        filter_3d = 1.0 + (amplitude - 1.0) * np.exp(-((self.k_mag - center_k)**2) / (2 * width**2))
        return filter_3d

    def _evaluate_filter(self, filter_3d, u0, v0, w0, objective_type):
        """
        Run a quick simulation with the given filter and return a score
        based on the specified objective.
        """
        # Run a very short, low-res simulation
        temp_solver = HRFNavierStokesPOC_3D(N=self.N, nu=self.nu)
        
        # Must override the solver's internal filter with the one we're testing
        temp_solver.discovered_filter = filter_3d

        # Simulate for a very short time
        T_eval = 0.1
        results = temp_solver.simulate(u0, v0, w0, T=T_eval, dt=0.002, verbose=False)
        
        if not results['enstrophy_history']:
            return 0

        if objective_type == 'peak_enstrophy':
            # The score is the maximum enstrophy reached
            return max(results['enstrophy_history'])
        elif objective_type == 'growth_rate':
            # The score is the average rate of enstrophy growth
            enstrophy = np.array(results['enstrophy_history'])
            if len(enstrophy) < 2:
                return 0
            return (enstrophy[-1] - enstrophy[0]) / T_eval
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")


def create_taylor_green_vortex(N):
    """
    Create the Taylor-Green vortex initial condition, a classic
    test case for 3D turbulence and singularity studies.
    """
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
    v0 = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w0 = np.zeros_like(X) # Starts as a 2.5D flow
    
    # Add small-scale perturbation to trigger 3D instability
    noise_level = 0.05
    u0 += noise_level * np.random.randn(N, N, N)
    v0 += noise_level * np.random.randn(N, N, N)
    w0 += noise_level * np.random.randn(N, N, N)
    
    return u0, v0, w0


def create_antiparallel_vortices(N, strength=5.0):
    """
    Creates two anti-parallel vortex tubes, another classic scenario
    for studying turbulent breakdown and singularities.
    """
    print("Creating anti-parallel vortex initial conditions...")
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    u0, v0, w0 = np.zeros((N,N,N)), np.zeros((N,N,N)), np.zeros((N,N,N))
    
    # Define vortex parameters
    radius = np.pi / 4
    x1, y1 = np.pi - np.pi/2, np.pi/2
    x2, y2 = np.pi + np.pi/2, np.pi/2

    # First vortex
    r1_sq = ((X - x1)**2 + (Y - y1)**2) / radius**2
    w0 += strength * np.exp(-r1_sq)
    
    # Second vortex (anti-parallel)
    r2_sq = ((X - x2)**2 + (Y - y2)**2) / radius**2
    w0 -= strength * np.exp(-r2_sq)
    
    # Add a small perturbation in the third dimension to trigger instability
    noise_level = 0.1
    u0 += noise_level * np.sin(2*Z)
    v0 += noise_level * np.cos(2*Z)

    return u0, v0, w0


def run_complete_analysis_3d(initial_condition='taylor_green', objective_type='peak_enstrophy'):
    print("="*60)
    print("HRF 3D NAVIER-STOKES SINGULARITY DISCOVERY")
    print("="*60)
    
    N = 32  # Grid size (increased for higher resolution)
    nu = 0.01 # Viscosity
    T = 0.5   # Shorter time, as blow-up happens fast
    dt = 0.001 # Smaller dt for stability with finer grid
    
    if initial_condition == 'taylor_green':
        print("\n1. Creating 3D Taylor-Green vortex initial conditions...")
        u0, v0, w0 = create_taylor_green_vortex(N)
    elif initial_condition == 'antiparallel_vortices':
        print("\n1. Creating 3D Anti-Parallel vortices initial conditions...")
        u0, v0, w0 = create_antiparallel_vortices(N)
    else:
        raise ValueError(f"Invalid initial condition specified: {initial_condition}")

    # --- Step 2: Discover an optimal filter ---
    # Use smaller versions of the initial conditions for faster discovery
    N_discover = 16
    u0_small = u0[:N_discover, :N_discover, :N_discover]
    v0_small = v0[:N_discover, :N_discover, :N_discover]
    w0_small = w0[:N_discover, :N_discover, :N_discover]
    
    discoverer = FilterDiscovery(N=N_discover, nu=nu)
    discovered_filter = discoverer.discover_filter(u0_small, v0_small, w0_small, objective_type=objective_type)
    
    print("\n3. Running 3D HRF Navier-Stokes simulation with discovered filter...")
    solver = HRFNavierStokesPOC_3D(N=N, nu=nu)
    # Upsample the filter for the high-resolution simulation
    # This is a simple upsampling, more sophisticated methods could be used
    upsampled_filter = np.kron(discovered_filter, np.ones((2,2,2)))
    solver.discovered_filter = upsampled_filter # Inject the filter
    
    results = solver.simulate(u0, v0, w0, T=T, dt=dt)
    
    print("\n4. Analyzing simulation results...")
    times = np.array(results['time_history'])
    if len(times) < 5:
        print("Simulation too short for analysis.")
        return
        
    max_coeffs = np.array([max(np.max(np.abs(c['u_hat'])), np.max(np.abs(c['v_hat'])), np.max(np.abs(c['w_hat']))) for c in results['coefficient_history']])
    
    print("\n5. Discovering mathematical blow-up laws...")
    sr_analyzer = SymbolicRegressionAnalyzer()
    blow_up_law = sr_analyzer.discover_blow_up_law(times, max_coeffs)
    
    print("\n" + "="*60 + "\nDISCOVERIES (3D):\n" + "="*60)
    print(f"\nDiscovered Blow-up Law: {blow_up_law['formula']}")
    print(f"Singularity Type: {blow_up_law['type']}")
    if blow_up_law['blow_up_time']:
        print(f"Estimated Blow-up Time: {blow_up_law['blow_up_time']:.4f}")
    
    print("\nSpectral Concentration (singularity indicator):")
    print(f"Initial: {results['singularity_indicators'][0]:.4f}")
    print(f"Final: {results['singularity_indicators'][-1]:.4f}")
    
    plot_results(times, max_coeffs, results['energy_history'], 
                 results['enstrophy_history'], results['singularity_indicators'],
                 blow_up_law, results.get('final_state'), solver, initial_condition)
    
    # Also plot the vorticity structure
    if results.get('final_state'):
        plot_vorticity_slices(results['final_state'], solver, f'navier_stokes_vorticity_3d_{initial_condition}.png')


def plot_results(times, max_coeffs, energy, enstrophy, singularity_ind, blow_up_law, final_state=None, solver=None, initial_condition=''):
    """
    Visualize the singularity discovery results (plots are the same as 2D).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = '3D Navier-Stokes Singularity Analysis'
    if initial_condition:
        title += f' ({initial_condition.replace("_", " ").title()})'
    fig.suptitle(title, fontsize=16)
    
    # Coefficient growth
    ax = axes[0, 0]
    ax.semilogy(times, max_coeffs, 'b-', linewidth=2, label='Simulation')
    if blow_up_law['formula']:
        t_sym = sp.Symbol('t')
        try:
            formula_func = sp.lambdify(t_sym, blow_up_law['formula'], 'numpy')
            t_fine = np.linspace(times[0], times[-1], 200)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'overflow encountered in exp')
                y_pred = formula_func(t_fine)
            ax.semilogy(t_fine, y_pred, 'r--', linewidth=2, label='Discovered Law')
        except Exception as e:
            print(f"Could not plot discovered law: {e}")
    ax.set_title('Coefficient Blow-up'); ax.set_xlabel('Time'); ax.set_ylabel('Max Coefficient')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Energy, Enstrophy, Singularity Indicator
    axes[0, 1].plot(times, energy, 'g-'); axes[0, 1].set_title('Energy Evolution')
    axes[1, 0].semilogy(times, enstrophy, 'm-'); axes[1, 0].set_title('Enstrophy Growth')
    axes[1, 1].plot(times, singularity_ind, 'r-'); axes[1, 1].set_title('Singularity Indicator')
    for ax in axes.flat[1:]: ax.grid(True, alpha=0.3); ax.set_xlabel('Time')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'navier_stokes_singularity_discovery_3d_{initial_condition}.png', dpi=150)
    plt.show()


def plot_vorticity_slices(final_state, solver, filename):
    """
    Calculates and plots 2D slices of the 3D vorticity magnitude.
    """
    print("\nVisualizing 3D vorticity structure...")
    if not final_state:
        print("Final state not available for plotting.")
        return

    u_hat = final_state['u_hat']
    v_hat = final_state['v_hat']
    w_hat = final_state['w_hat']
    
    # Calculate vorticity components in spectral space
    # omega_x = d(w)/dy - d(v)/dz
    omega_x_hat = 1j * solver.ky * w_hat - 1j * solver.kz * v_hat
    # omega_y = d(u)/dz - d(w)/dx
    omega_y_hat = 1j * solver.kz * u_hat - 1j * solver.kx * w_hat
    # omega_z = d(v)/dx - d(u)/dy
    omega_z_hat = 1j * solver.kx * v_hat - 1j * solver.ky * u_hat
    
    # Transform to physical space
    omega_x = ifftn(omega_x_hat).real
    omega_y = ifftn(omega_y_hat).real
    omega_z = ifftn(omega_z_hat).real
    
    # Vorticity magnitude
    vort_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    # Create plots of 2D slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Vorticity Magnitude Slices at Final Time Step', fontsize=16)
    
    N = solver.N
    slice_idx = N // 2
    
    # XY Slice at z = L/2
    im1 = axes[0].imshow(vort_mag[:, :, slice_idx], cmap='inferno')
    axes[0].set_title(f'XY Plane at z={slice_idx}')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    fig.colorbar(im1, ax=axes[0])
    
    # XZ Slice at y = L/2
    im2 = axes[1].imshow(vort_mag[:, slice_idx, :], cmap='inferno')
    axes[1].set_title(f'XZ Plane at y={slice_idx}')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    fig.colorbar(im2, ax=axes[1])
    
    # YZ Slice at x = L/2
    im3 = axes[2].imshow(vort_mag[slice_idx, :, :], cmap='inferno')
    axes[2].set_title(f'YZ Plane at x={slice_idx}')
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('Y')
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    # --- Run the analysis for Taylor-Green vortex ---
    # run_complete_analysis_3d(initial_condition='taylor_green')
    
    # --- Run the analysis for Anti-Parallel vortices ---
    # run_complete_analysis_3d(initial_condition='antiparallel_vortices', objective_type='peak_enstrophy')

    # --- Run again with the new objective function ---
    run_complete_analysis_3d(initial_condition='antiparallel_vortices', objective_type='growth_rate')
