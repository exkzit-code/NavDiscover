"""
Theory Validation Script for Navier-Stokes Singularities
=========================================================

This script is designed to test specific mathematical theories
about the nature of the observed blow-up phenomena.

Test 1: Helicity Conservation vs. Enstrophy Blow-up
Hypothesis: The singularity is driven by an inviscid mechanism.
Therefore, enstrophy should blow up while helicity, an inviscid
invariant, should remain approximately conserved.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
import warnings
warnings.filterwarnings('ignore')

# --- Simulation Parameters ---
N_RES = 128  # Resolution for the test simulation
TARGET_SEPARATION = 1.26 # The "sweet spot" separation distance
VISCOSITY = 0.005
SIM_TIME = 0.04    # Shorter time, we only need to see the blow-up
TIME_STEP = 0.0005

class TheoryTestNavierStokes:
    def __init__(self, N=128, nu=0.01):
        self.N, self.nu = N, nu
        kx_1d, ky_1d, kz_1d = (np.fft.fftfreq(N) * N,)*3
        self.kx, self.ky, self.kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_nozero = self.k2.copy(); self.k2_nozero[0,0,0] = 1e-10
        kmax = N*2/3/2
        self.dealias_mask = (np.abs(self.kx)<kmax) & (np.abs(self.ky)<kmax) & (np.abs(self.kz)<kmax)
        self.time_history, self.enstrophy_history, self.helicity_history = [],[],[]

    def simulate(self, u0, v0, w0, T=1.0, dt=0.001):
        u_hat, v_hat, w_hat = fftn(u0), fftn(v0), fftn(w0)
        t, step = 0, 0
        print(f"Starting {self.N}³ theory validation simulation...")
        while t < T:
            self.time_history.append(t)
            self._calculate_indicators(u_hat, v_hat, w_hat)

            max_coeff = max(np.max(np.abs(u_hat)), np.max(np.abs(v_hat)), np.max(np.abs(w_hat)))
            if max_coeff > 1e12 or np.isnan(max_coeff):
                print(f"Potential blow-up detected at t={t:.4f}, max|coeff|={max_coeff:.2e}")
                break

            u_hat, v_hat, w_hat = self._time_step(u_hat, v_hat, w_hat, dt)
            t += dt; step += 1
            if step % 10 == 0:
                print(f"t={t:.4f}, Enstrophy={self.enstrophy_history[-1]:.3e}, Helicity={self.helicity_history[-1]:.3e}")

        return {'time_history':self.time_history, 'enstrophy_history':self.enstrophy_history, 'helicity_history': self.helicity_history}

    def _time_step(self, u_hat, v_hat, w_hat, dt):
        # This is the standard pseudo-spectral step
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

    def _calculate_indicators(self, u_hat, v_hat, w_hat):
        # Enstrophy
        energy_density = 0.5*(np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        enstrophy = np.sum(self.k2 * energy_density) / (self.N**6)
        self.enstrophy_history.append(enstrophy)

        # Helicity
        u, v, w = ifftn(u_hat).real, ifftn(v_hat).real, ifftn(w_hat).real
        omega_x_hat = 1j*self.ky*w_hat - 1j*self.kz*v_hat
        omega_y_hat = 1j*self.kz*u_hat - 1j*self.kx*w_hat
        omega_z_hat = 1j*self.kx*v_hat - 1j*self.ky*u_hat
        omega_x, omega_y, omega_z = ifftn(omega_x_hat).real, ifftn(omega_y_hat).real, ifftn(omega_z_hat).real
        helicity_density = u*omega_x + v*omega_y + w*omega_z
        total_helicity = np.sum(helicity_density)
        self.helicity_history.append(total_helicity)

def create_antiparallel_vortices(N, strength=5.0, separation=np.pi):
    print(f"Creating {N}³ anti-parallel vortices (separation={separation:.2f})...")
    x,y,z=(np.linspace(0,2*np.pi,N,endpoint=False),)*3; X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    u0,v0,w0 = np.zeros((N,N,N)), np.zeros((N,N,N)), np.zeros((N,N,N))
    radius=np.pi/4; x1,y1 = np.pi-separation/2,np.pi/2; x2,y2 = np.pi+separation/2,np.pi/2
    w0 += strength*np.exp(-(((X-x1)**2+(Y-y1)**2)/radius**2))
    w0 -= strength*np.exp(-(((X-x2)**2+(Y-y2)**2)/radius**2))
    # Using the same perturbation as before to ensure blow-up
    noise=0.1; u0+=noise*np.sin(2*Z); v0+=noise*np.cos(2*Z)
    return u0,v0,w0

def plot_theory_results(results, filename):
    print(f"Plotting theory test results to {filename}...")
    times = np.array(results['time_history'])
    enstrophy = np.array(results['enstrophy_history'])
    helicity = np.array(results['helicity_history'])

    # Normalize for comparison
    enstrophy_norm = enstrophy / np.max(enstrophy)
    # Handle case where helicity is close to zero
    if np.max(np.abs(helicity)) > 1e-9:
        helicity_norm = helicity / np.max(np.abs(helicity))
    else:
        helicity_norm = helicity # Already zero or near-zero

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle('Theory Validation: Enstrophy vs. Helicity', fontsize=16)

    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Enstrophy (Normalized)', color=color)
    ax1.semilogy(times, enstrophy_norm, color=color, label='Enstrophy (Normalized)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Helicity (Normalized)', color=color)
    ax2.plot(times, helicity_norm, color=color, linestyle='--', label='Helicity (Normalized)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-1.1, 1.1) # Keep helicity axis stable

    fig.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(filename, dpi=150)
    plt.show()


def run_theory_test_1():
    print("\n" + "="*80)
    print("RUNNING THEORY VALIDATION: TEST 1 - INVISCID INVARIANT")
    print("="*80)

    u0, v0, w0 = create_antiparallel_vortices(N_RES, separation=TARGET_SEPARATION)
    solver = TheoryTestNavierStokes(N=N_RES, nu=VISCOSITY)
    results = solver.simulate(u0, v0, w0, T=SIM_TIME, dt=TIME_STEP)

    if len(results['time_history']) < 5:
        print("Simulation finished too quickly. No analysis performed.")
        return

    plot_filename = f"hypo_theory_test_1_inviscid_invariant_{N_RES}cubed.png"
    plot_theory_results(results, plot_filename)


if __name__ == "__main__":
    run_theory_test_1()
    print("\n\nTheory validation Test 1 complete.")
    print("Check the generated image file for results.") 
