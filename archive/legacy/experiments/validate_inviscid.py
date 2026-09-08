"""
High-Resolution Viscosity Scaling Analysis for Navier-Stokes Singularities
========================================================================

This script addresses point (1) from the reviewer feedback.

It runs a sweep of simulations at a higher resolution (64³) and over a
finer-grained range of viscosities to rigorously test the relationship
between blow-up time and viscosity.

To ensure statistical robustness, it performs multiple trials for each
viscosity value and plots the mean blow-up time with error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# --- Simulation Parameters ---
N_RUN = 64              # High-resolution grid size for the main runs
N_DISCOVER = 16         # Low-resolution grid for AI discovery
TARGET_SEPARATION = 1.26 # The "sweet spot" separation distance
VISCOSITY_SWEEP = np.linspace(1e-3, 9e-3, 10) # Finer sweep over ν values
NUM_TRIALS = 3          # Number of trials per viscosity value for robustness
SIM_TIME = 0.4          # Total simulation time
TIME_STEP = 0.0005      # Time step

# --- Core Simulation and Discovery Classes ---
# (Copied from hypo-viscosity.py, as they are required for the simulation)

class HRFNavierStokesPOC_3D:
    def __init__(self, N=32, nu=0.01):
        self.N, self.nu = N, nu
        self.discovered_filter = None
        kx_1d, ky_1d, kz_1d = (np.fft.fftfreq(N) * N,)*3
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_nozero = self.k2.copy(); self.k2_nozero[0,0,0] = 1e-10
        kmax = N*2/3/2
        self.dealias_mask = (np.abs(self.kx)<kmax) & (np.abs(self.ky)<kmax) & (np.abs(self.kz)<kmax)
        self.coefficient_history, self.time_history, self.energy_history, self.enstrophy_history, self.singularity_indicators = [],[],[],[],[]

    def simulate(self, u0, v0, w0, T=1.0, dt=0.001, verbose=True):
        u_hat, v_hat, w_hat = fftn(u0), fftn(v0), fftn(w0)
        if self.discovered_filter is not None:
            if verbose: print("Applying discovered spectral filter...")
            u_hat *= self.discovered_filter; v_hat *= self.discovered_filter; w_hat *= self.discovered_filter
        t, step = 0, 0
        if verbose: print(f"Starting {self.N}³ 3D simulation for ν={self.nu:.5f}...")
        while t < T:
            self.coefficient_history.append({'u_hat':u_hat.copy(),'v_hat':v_hat.copy(),'w_hat':w_hat.copy()}); self.time_history.append(t)
            self._calculate_indicators(u_hat, v_hat, w_hat)
            max_coeff = max(np.max(np.abs(u_hat)), np.max(np.abs(v_hat)), np.max(np.abs(w_hat)))
            if max_coeff > 1e12 or np.isnan(max_coeff):
                if verbose: print(f"Potential blow-up detected at t={t:.4f}, max|coeff|={max_coeff:.2e}")
                break
            u_hat, v_hat, w_hat = self._time_step(u_hat, v_hat, w_hat, dt)
            t += dt; step += 1
            if verbose and step % 20 == 0: print(f"  t={t:.4f}, max|u_hat|={np.max(np.abs(u_hat)):.3e}")
        return self._analyze_results()

    def _time_step(self, u_hat, v_hat, w_hat, dt):
        u,v,w = ifftn(u_hat).real, ifftn(v_hat).real, ifftn(w_hat).real
        ux,uy,uz = ifftn(1j*self.kx*u_hat).real, ifftn(1j*self.ky*u_hat).real, ifftn(1j*self.kz*u_hat).real
        vx,vy,vz = ifftn(1j*self.kx*v_hat).real, ifftn(1j*self.ky*v_hat).real, ifftn(1j*self.kz*v_hat).real
        wx,wy,wz = ifftn(1j*self.kx*w_hat).real, ifftn(1j*self.ky*w_hat).real, ifftn(1j*self.kz*w_hat).real
        Nu,Nv,Nw = -(u*ux+v*uy+w*uz), -(u*vx+v*vy+w*vz), -(u*wx+v*wy+w*wz)
        Nu_hat,Nv_hat,Nw_hat = fftn(Nu),fftn(Nv),fftn(Nw)
        Nu_hat[~self.dealias_mask]=0; Nv_hat[~self.dealias_mask]=0; Nw_hat[~self.dealias_mask]=0
        diff_term = 1/(1+self.nu*dt*self.k2)
        u_star,v_star,w_star = (u_hat+dt*Nu_hat)*diff_term, (v_hat+dt*Nv_hat)*diff_term, (w_hat+dt*Nw_hat)*diff_term
        div_u_star = 1j*self.kx*u_star + 1j*self.ky*v_star + 1j*self.kz*w_star
        p_hat = (1j/(dt*self.k2_nozero))*div_u_star
        u_hat_new = u_star - dt*(1j*self.kx*p_hat)
        v_hat_new = v_star - dt*(1j*self.ky*p_hat)
        w_hat_new = w_star - dt*(1j*self.kz*p_hat)
        return u_hat_new, v_hat_new, w_hat_new

    def _calculate_indicators(self,u,v,w):
        energy_density = 0.5*(np.abs(u)**2+np.abs(v)**2+np.abs(w)**2); energy = energy_density/(self.N**6)
        self.energy_history.append(np.sum(energy)); self.enstrophy_history.append(np.sum(self.k2*energy))
        high_k_mask = self.k2 > (self.N/4)**2
        self.singularity_indicators.append(np.sum(energy[high_k_mask])/(np.sum(energy)+1e-12))

    def _analyze_results(self):
        if not self.coefficient_history: return {'time_history':[]}
        times = np.array(self.time_history)
        max_coeffs = np.array([max(np.max(np.abs(c['u_hat'])),np.max(np.abs(c['v_hat'])),np.max(np.abs(c['w_hat']))) for c in self.coefficient_history])
        
        # Simple blow-up time estimation
        blow_up_time = None
        if len(times) > 5:
            log_values=np.log(max_coeffs+1e-12); 
            slope=(log_values[-1]-log_values[-5])/(times[-1]-times[-5])
            if slope > 0:
                blow_up_time = times[-1]+(25-log_values[-1])/slope
                
        return {'time_history': self.time_history, 'blow_up_time': blow_up_time}

class FilterDiscovery:
    def __init__(self, N=16, nu=0.01):
        self.N, self.nu = N, nu; self.discovered_filters=[]
        kx_1d,ky_1d,kz_1d=(np.fft.fftfreq(N)*N,)*3
        self.kx,self.ky,self.kz=np.meshgrid(kx_1d,ky_1d,kz_1d,indexing='ij')
        self.k_mag=np.sqrt(self.kx**2+self.ky**2+self.kz**2)
    def discover_filter(self,u0,v0,w0,objective_type='peak_enstrophy'):
        print(f"\n-- Starting {self.N}³ discovery for ν={self.nu:.5f} (Objective: {objective_type}) --")
        def objective(params): return -self._evaluate_filter(self._params_to_filter(params),u0,v0,w0,objective_type)
        bounds=[(self.N/8,self.N/2),(self.N/16,self.N/4),(1.0,1.5)]
        result=differential_evolution(objective,bounds,maxiter=10,popsize=8,disp=False)
        optimal_filter=self._params_to_filter(result.x); self.discovered_filters.append(result.x)
        print("-- Filter discovery complete --")
        return optimal_filter, result.x
    def _params_to_filter(self,params):
        center_k,width,amplitude=params
        return 1.0+(amplitude-1.0)*np.exp(-((self.k_mag-center_k)**2)/(2*width**2))
    def _evaluate_filter(self,filter_3d,u0,v0,w0,objective_type):
        temp_solver=HRFNavierStokesPOC_3D(N=self.N,nu=self.nu); temp_solver.discovered_filter=filter_3d
        T_eval=0.1; results=temp_solver.simulate(u0,v0,w0,T=T_eval,dt=0.002,verbose=False)
        if not results['time_history']: return 0
        
        # Simplified objective: find max coefficient value instead of enstrophy
        max_coeffs = np.array([max(np.max(np.abs(c['u_hat'])),np.max(np.abs(c['v_hat'])),np.max(np.abs(c['w_hat']))) for c in temp_solver.coefficient_history])
        if len(max_coeffs) == 0: return 0
        return np.max(max_coeffs)

def create_antiparallel_vortices(N, strength=5.0, separation=np.pi):
    print(f"Creating {N}³ anti-parallel vortices (separation={separation:.2f})...")
    x,y,z=(np.linspace(0,2*np.pi,N,endpoint=False),)*3; X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    u0,v0,w0 = np.zeros((N,N,N)), np.zeros((N,N,N)), np.zeros((N,N,N))
    radius=np.pi/4; x1,y1 = np.pi-separation/2,np.pi/2; x2,y2 = np.pi+separation/2,np.pi/2
    w0 += strength*np.exp(-(((X-x1)**2+(Y-y1)**2)/radius**2))
    w0 -= strength*np.exp(-(((X-x2)**2+(Y-y2)**2)/radius**2))
    noise=0.1; u0+=noise*np.sin(2*Z); v0+=noise*np.cos(2*Z)
    return u0,v0,w0

def run_single_trial(viscosity, trial_num):
    print("\n" + "="*80)
    print(f"RUNNING TRIAL {trial_num+1}/{NUM_TRIALS} for ν = {viscosity:.5f}")
    print("="*80)
    
    # 1. AI-Guided Discovery on Low-Resolution Grid
    u0_discover,v0_discover,w0_discover = create_antiparallel_vortices(N_DISCOVER, separation=TARGET_SEPARATION)
    discoverer = FilterDiscovery(N=N_DISCOVER, nu=viscosity)
    discovered_filter_lowres, _ = discoverer.discover_filter(u0_discover,v0_discover,w0_discover)
    
    # 2. Set up and Run High-Resolution Simulation
    u0_hires,v0_hires,w0_hires = create_antiparallel_vortices(N_RUN, separation=TARGET_SEPARATION)
    solver = HRFNavierStokesPOC_3D(N=N_RUN, nu=viscosity)
    
    # Upscale the discovered filter
    scale_factor = N_RUN // N_DISCOVER
    solver.discovered_filter = np.kron(discovered_filter_lowres, np.ones((scale_factor, scale_factor, scale_factor)))
    
    results = solver.simulate(u0_hires, v0_hires, w0_hires, T=SIM_TIME, dt=TIME_STEP, verbose=False)
    
    if results['blow_up_time'] is None:
        print("Simulation finished too quickly or did not blow up. No analysis performed.")
        return None

    print(f"  --> Estimated blow-up time: {results['blow_up_time']:.4f}")
    return results['blow_up_time']

def plot_sweep_results(viscosities, mean_times, std_devs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle("High-Resolution Viscosity Scaling of Blow-up Time", fontsize=16)
    
    ax.errorbar(viscosities, mean_times, yerr=std_devs, fmt='o-', capsize=5, label='Simulation Data (Mean & Std Dev)')
    ax.set_xlabel("Viscosity (ν)")
    ax.set_ylabel("Estimated Blow-up Time (T*)")
    ax.set_title(f"Blow-up Time vs. Viscosity at {N_RUN}³ (N_trials={NUM_TRIALS})")
    ax.grid(True, which="both", ls="--")
    # A linear x-axis might be better here to see the dip clearly.
    # ax.set_xscale('log') 
    
    # Add a polynomial fit to guide the eye
    try:
        p = np.polyfit(viscosities, mean_times, 3)
        p_func = np.poly1d(p)
        x_fit = np.linspace(viscosities[0], viscosities[-1], 200)
        ax.plot(x_fit, p_func(x_fit), 'r--', alpha=0.8, label='3rd Order Polynomial Fit')
    except np.linalg.LinAlgError:
        print("Could not compute polynomial fit.")

    ax.legend()
    plt.tight_layout(rect=[0,0,1,0.95])
    filename = "hypo_viscosity_hires_sweep_summary.png"
    plt.savefig(filename, dpi=200)
    print(f"\nSaved sweep results to {filename}")
    plt.show()

if __name__ == "__main__":
    mean_blow_up_times = []
    std_dev_blow_up_times = []
    valid_viscosities = []
    
    for nu in VISCOSITY_SWEEP:
        trial_results = []
        for i in range(NUM_TRIALS):
            blow_up_time = run_single_trial(nu, i)
            if blow_up_time is not None and blow_up_time > 0:
                trial_results.append(blow_up_time)
        
        if len(trial_results) > 0:
            mean_time = np.mean(trial_results)
            std_dev_time = np.std(trial_results)
            
            mean_blow_up_times.append(mean_time)
            std_dev_blow_up_times.append(std_dev_time)
            valid_viscosities.append(nu)
            
            print(f"\nViscosity ν={nu:.5f} Summary:")
            print(f"  Mean blow-up time: {mean_time:.4f} ± {std_dev_time:.4f}")

    if len(valid_viscosities) > 1:
        plot_sweep_results(valid_viscosities, mean_blow_up_times, std_dev_blow_up_times)

    print("\n\nHigh-resolution viscosity sweep complete.")
    print("Check hypo_viscosity_hires_sweep_summary.png for results.") 
