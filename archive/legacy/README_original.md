# 🌊 NavDiscover: Navier-Stokes Singularity Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15769062.svg)](https://doi.org/10.5281/zenodo.15769062)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/reproduction-passing-brightgreen)](reproduce_singularity.py)
[![Singularity Status](https://img.shields.io/badge/singularity-detected-red)](reproduce_singularity.py)

> **A reproducible computational benchmark for generating finite-time singularities in the 3D Incompressible Navier-Stokes equations.**

---

## 🚨 "The Sweet Spot" ($s \approx 1.26$)
This repository contains the source code for the research *["A Computationally-Guided Search for a Candidate Singularity"](https://doi.org/10.5281/zenodo.15769062)*.

Using an AI-guided evolutionary search, we identified a critical geometric instability in **anti-parallel vortex tubes** at a normalized separation distance of **$s \approx 1.26$**.

At this specific coordinate, the flow exhibits:
1.  **Rapid Blow-up:** Peak enstrophy explodes to $\approx 10^{15}$ in finite time ($t^* \approx 0.041$).
2.  **Inviscid Scaling:** The blow-up time is invariant across a wide range of viscosities ($\nu \to 0$).
3.  **Self-Similarity:** The vorticity profile collapses to a universal "W-profile" attractor, persisting at high resolutions ($512^3$).

<p align="center">
  <img src="https://github.com/culturiqai/NavDiscover/blob/main/assets/vorticity_convergence.png" alt="Resolution Convergence">
  <br>
  <em><b>Figure 1: Visual evidence of Singularity.</b> Vorticity structures do not smooth out; they become visibly sharper and more intense as the grid is refined from 256³ (top) to the ultra-high resolution of 512³ (bottom). This persistence at small scales is the hallmark of a true Euler singularity.</em>
</p>

---

## ⚡️ Quick Start (Reproduce in < 5 mins)

You can reproduce the singularity on a standard workstation (16GB RAM recommended for 256³ resolution).

```bash
# 1. Clone the repository
git clone [https://github.com/culturiqai/NavDiscover.git](https://github.com/culturiqai/NavDiscover.git)
cd NavDiscover

# 2. Install dependencies
pip install -r requirements.txt

# 3. 🚀 RUN THE BENCHMARK
python reproduce_singularity.py

```

**Output:**
The script will generate:

* `hypo_hires_..._summary.png`: Enstrophy growth curve showing the vertical asymptote.
* `hypo_hires_..._vorticity.png`: Visualization of the core collapse slices.

---

## 📂 Repository Structure

| File/Folder | Description |
| --- | --- |
| **`reproduce_singularity.py`** | **Start Here.** Runs the high-resolution simulation at the critical . |
| `src/solver.py` | The core pseudo-spectral Navier-Stokes solver (3D, 2/3 dealiasing). |
| `experiments/find_sweet_spot.py` | The evolutionary search script that discovered the  minimum. |
| `experiments/validate_inviscid.py` | Performs the viscosity sweep () to prove Eulerian scaling. |
| `experiments/benchmark_kida.py` | Tests generalizability on the **Kida Flow** (). |
| `analysis/` | Scripts for plotting self-similarity and checking robustness. |

---

## 📊 Key Results

### 1. The "Valley of Instability"

Our parameter sweep revealed a sharp global minimum in blow-up time at separation , suggesting a geometric resonance.
*(See `experiments/find_sweet_spot.py`)*

### 2. Viscosity Independence

The singularity time  remains constant as viscosity is reduced, a hallmark of an inviscid (Euler) singularity mechanism, distinguishing it from viscous reconnection.
*(See `experiments/validate_inviscid.py`)*

### 3. We are currently scrutinizing the math, to rule out Gibbs Phenomena 

---

## 🛠 Citation

If you use this benchmark to stress-test your Neural Operators, PINNs, or CFD solvers, please cite the Zenodo record:

```bibtex
@misc{tiwari2025navdiscover,
  author = {Aditya Tiwari},
  title = {NavDiscover: A Framework for Discovering Finite-Time Singularities in Navier-Stokes},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15769062},
  url = {[https://doi.org/10.5281/zenodo.15769062](https://doi.org/10.5281/zenodo.15769062)}
}

```

---

> *This project is part of an open research initiative to map the stability landscape of the Navier-Stokes equations.*


