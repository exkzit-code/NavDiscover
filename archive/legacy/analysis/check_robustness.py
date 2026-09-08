"""
AI Discovery Robustness Analysis
================================

This script addresses point (3) from the reviewer feedback.

It tests the robustness of the `scipy.optimize.differential_evolution`
process for finding optimal spectral filters. It runs the discovery
process many times from different random initializations for a fixed
physical scenario.

The script then plots histograms of the discovered filter parameters
(center_k, width, amplitude) and the resulting objective scores.
Tightly peaked distributions would indicate that the optimizer is
robustly finding a global optimum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# --- Parameters ---
N_DISCOVER = 16          # Low-resolution grid for AI discovery
TARGET_SEPARATION = 1.26 # The "sweet spot" separation distance
VISCOSITY = 0.005        # A representative viscosity
NUM_TRIALS = 15          # Number of times to run the discovery process

# --- Core Simulation and Discovery Classes (Simplified for this test) ---

class HRFNavierStokesPOC_3D:
    # A minimal version of the solver needed for the discovery objective function
    def __init__(self, N=16, nu=0.01):
        self.N, self.nu = N, nu
        self.discovered_filter = None
        kx_1d, ky_1d, kz_1d = (np.fft.fftfreq(N) * N,)*3
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_nozero = self.k2.copy(); self.k2_nozero[0,0,0] = 1e-10
        kmax = N*2/3/2
        self.dealias_mask = (np.abs(self.kx)<kmax) & (np.abs(self.ky)<kmax) & (np.abs(self.kz)<kmax)
        self.coefficient_history = []

    def simulate(self, u0, v0, w0, T=1.0, dt=0.001):
        u_hat, v_hat, w_hat = fftn(u0), fftn(v0), fftn(w0)
        if self.discovered_filter is not None:
            u_hat *= self.discovered_filter; v_hat *= self.discovered_filter; w_hat *= self.discovered_filter
        t, step = 0, 0
        while t < T:
            self.coefficient_history.append({'u_hat':u_hat,'v_hat':v_hat,'w_hat':w_hat})
            max_coeff = max(np.max(np.abs(u_hat)), np.max(np.abs(v_hat)), np.max(np.abs(w_hat)))
            if max_coeff > 1e8 or np.isnan(max_coeff): break
            u_hat, v_hat, w_hat = self._time_step(u_hat, v_hat, w_hat, dt)
            t += dt; step += 1
        return self._analyze_results()

    def _time_step(self, u_hat, v_hat, w_hat, dt):
        u,v,w=ifftn(u_hat).real,ifftn(v_hat).real,ifftn(w_hat).real
        ux,uy,uz=ifftn(1j*self.kx*u_hat).real,ifftn(1j*self.ky*u_hat).real,ifftn(1j*self.kz*u_hat).real
        vx,vy,vz=ifftn(1j*self.kx*v_hat).real,ifftn(1j*self.ky*v_hat).real,ifftn(1j*self.kz*v_hat).real
        wx,wy,wz=ifftn(1j*self.kx*w_hat).real,ifftn(1j*self.ky*w_hat).real,ifftn(1j*self.kz*w_hat).real
        Nu,Nv,Nw=-(u*ux+v*uy+w*uz),-(u*vx+v*vy+w*vz),-(u*wx+v*wy+w*wz)
        Nu_hat,Nv_hat,Nw_hat=fftn(Nu),fftn(Nv),fftn(Nw)
        Nu_hat[~self.dealias_mask]=0;Nv_hat[~self.dealias_mask]=0;Nw_hat[~self.dealias_mask]=0
        diff_term=1/(1+self.nu*dt*self.k2);u_star=(u_hat+dt*Nu_hat)*diff_term;v_star=(v_hat+dt*Nv_hat)*diff_term;w_star=(w_hat+dt*Nw_hat)*diff_term
        div_u_star=1j*self.kx*u_star+1j*self.ky*v_star+1j*self.kz*w_star;p_hat=(1j/(dt*self.k2_nozero))*div_u_star
        return u_star-dt*(1j*self.kx*p_hat),v_star-dt*(1j*self.ky*p_hat),w_star-dt*(1j*self.kz*p_hat)

    def _analyze_results(self):
        if not self.coefficient_history: return 0
        max_coeffs = np.array([max(np.max(np.abs(c['u_hat'])),np.max(np.abs(c['v_hat'])),np.max(np.abs(c['w_hat']))) for c in self.coefficient_history])
        return np.max(max_coeffs) if len(max_coeffs) > 0 else 0

class FilterDiscovery:
    def __init__(self, N, nu, u0, v0, w0):
        self.N, self.nu = N, nu
        self.u0, self.v0, self.w0 = u0, v0, w0
        kx_1d,ky_1d,kz_1d=(np.fft.fftfreq(N)*N,)*3
        self.kx,self.ky,self.kz=np.meshgrid(kx_1d,ky_1d,kz_1d,indexing='ij')
        self.k_mag=np.sqrt(self.kx**2+self.ky**2+self.kz**2)

    def discover_filter(self):
        def objective(params):
            return -self._evaluate_filter(self._params_to_filter(params))
        bounds=[(self.N/8, self.N/2), (self.N/16, self.N/4), (1.0, 1.5)]
        result=differential_evolution(objective, bounds, maxiter=10, popsize=8, disp=False)
        return result.x, -result.fun # result.x are params, result.fun is the score

    def _params_to_filter(self,params):
        center_k,width,amplitude=params
        return 1.0+(amplitude-1.0)*np.exp(-((self.k_mag-center_k)**2)/(2*width**2))

    def _evaluate_filter(self, filter_3d):
        solver = HRFNavierStokesPOC_3D(N=self.N, nu=self.nu)
        solver.discovered_filter = filter_3d
        return solver.simulate(self.u0, self.v0, self.w0, T=0.1, dt=0.002)

def create_antiparallel_vortices(N, strength=5.0, separation=np.pi):
    x,y,z=(np.linspace(0,2*np.pi,N,endpoint=False),)*3; X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    u0,v0,w0=np.zeros((N,N,N)),np.zeros((N,N,N)),np.zeros((N,N,N))
    radius=np.pi/4;x1,y1=np.pi-separation/2,np.pi/2;x2,y2=np.pi+separation/2,np.pi/2
    w0+=strength*np.exp(-(((X-x1)**2+(Y-y1)**2)/radius**2))
    w0-=strength*np.exp(-(((X-x2)**2+(Y-y2)**2)/radius**2))
    noise=0.1;u0+=noise*np.sin(2*Z);v0+=noise*np.cos(2*Z)
    return u0,v0,w0

def plot_robustness_results(results):
    params = np.array([res[0] for res in results])
    scores = np.array([res[1] for res in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AI Discovery Robustness Analysis', fontsize=16)

    # Histogram for Center K
    axes[0, 0].hist(params[:, 0], bins=15, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Discovered "center_k"')
    axes[0, 0].set_xlabel('center_k value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Histogram for Width
    axes[0, 1].hist(params[:, 1], bins=15, color='salmon', edgecolor='black')
    axes[0, 1].set_title('Distribution of Discovered "width"')
    axes[0, 1].set_xlabel('width value')
    
    # Histogram for Amplitude
    axes[1, 0].hist(params[:, 2], bins=15, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Discovered "amplitude"')
    axes[1, 0].set_xlabel('amplitude value')
    axes[1, 0].set_ylabel('Frequency')

    # Histogram for Objective Score
    axes[1, 1].hist(scores, bins=15, color='gold', edgecolor='black')
    axes[1, 1].set_title('Distribution of Final Objective Scores')
    axes[1, 1].set_xlabel('Score (Max Coefficient)')

    for ax in axes.flatten():
        ax.grid(axis='y', alpha=0.7, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = "hypo_robustness_test_summary.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Robustness analysis plot saved to {filename}")

if __name__ == "__main__":
    print(f"Starting robustness test: {NUM_TRIALS} trials...")
    u0, v0, w0 = create_antiparallel_vortices(N_DISCOVER, separation=TARGET_SEPARATION)
    discoverer = FilterDiscovery(N=N_DISCOVER, nu=VISCOSITY, u0=u0, v0=v0, w0=w0)
    
    results = []
    for i in range(NUM_TRIALS):
        print(f"  Running trial {i+1}/{NUM_TRIALS}...")
        discovered_params, score = discoverer.discover_filter()
        results.append((discovered_params, score))
        print(f"    -> Discovered params: k={discovered_params[0]:.2f}, w={discovered_params[1]:.2f}, A={discovered_params[2]:.2f} | Score={score:.3e}")

    plot_robustness_results(results)
    print("\nRobustness test complete.") 
