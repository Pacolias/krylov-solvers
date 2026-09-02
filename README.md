# krylov-solvers

> Python implementations of Krylov subspace methods for solving large sparse linear systems.

## Overview

Iterative methods based on Krylov subspaces are essential for solving large-scale, sparse linear systems $A\mathbf{x} = \mathbf{b}$ where direct methods are computationally prohibitive. 

This project provides custom, educational, yet highly functional and matrix-free implementations of five fundamental Krylov solvers:
- **CG** (Conjugate Gradient) - For Symmetric Positive Definite (SPD) matrices. Includes support for preconditioning.
- **GMRES** (Generalized Minimal Residual) - For general asymmetric matrices (based on the Arnoldi iteration).
- **BiCG** (Biconjugate Gradient) - For general asymmetric matrices (based on the Lanczos biorthogonalization).
- **CGS** (Conjugate Gradient Squared) - A transpose-free variant of BiCG.
- **BiCGSTAB** (Biconjugate Gradient Stabilized) - A transpose-free variant that smooths the irregular convergence of CGS.

## Repository Structure

```text
krylov-solvers/
├── data/                   # SuiteSparse Matrix Collection files
│   ├── bcsstk14.mtx        # SPD matrix (Structural engineering)
│   ├── nos6.mtx            # SPD matrix (Finite differences)
│   ├── pores_2.mtx         # Asymmetric matrix (Computational Fluid Dynamics)
│   └── sherman1.mtx        # Asymmetric matrix (Oil reservoir simulation)
├── src/                    # Core algorithms
│   ├── __init__.py         # Package exports
│   ├── utils.py            # Common auxiliary functions
│   ├── cg.py               # Conjugate Gradient (CG & PCG) script
│   ├── gmres.py            # GMRES script
│   ├── bicg.py             # BiCG script
│   ├── cgs.py              # CGS script
│   └── bicgstab.py         # BiCGSTAB script
├── scripts/                
│   ├── exp_spd.py          # Experiments for SPD matrices (CG vs PCG with Jacobi)
│   └── exp_asym.py         # Experiments for asymmetric matrices (GMRES vs BiCG vs CGS vs BiCGSTAB)
├── plots/                  # Generated convergence history plots (.png / .jpg)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file (excludes __pycache__, etc.)
└── README.md
```

## Installation & Requirements
This project requires **Python 3.8+**. The main dependencies are `numpy`, `scipy` (for reading Matrix Market files and generating ILU preconditioners) and `matplotlib` (for plotting convergence histories).

1. Clone the repository:

```bash
git clone https://github.com/Pacolias/krylov-solvers.git
cd krylov-solvers
```

2. (Optional but recommended) Create a virtual enviroment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage & Experiments
To reproduce the experiments, you will need to download sparse matrices in Matrix Market format (`.mtx`) from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) and place them in the `data/` folder.

Recommended matrices for testing:
* **SPD Matrix**: `bcsstk14.mtx` (Structural engineering)
* **Asymmetric Matrix**: `sherman1.mtx` (Oil reservoir simulation)

Once the matrices are in the `data/` folder, you can run the experimental scripts:

```bash
# Run the comparison for Symmetric Positive Definite matrices
python scripts/exp_spd.py

# Run the comparison for general asymmetric matrices
python scripts/exp_asymmetric.py
```

The scripts will output the number of iterations and the execution time for each method and will display a semi-logarithmic plot comparing their convergence histories ($\|r_k\|_2/\|r_0\|_2$ vs. iterations).

## Theoretical Background
The mathematical derivation and convergence analysis of these algorithms are thoroughly detailed in the accompanying thesis document, heavily based on the theoretical frameworks provided by:

* Dolean, V., Jolivet, P., & Nataf, F. (2015). An Introduction to Domain Decomposition Methods.
* Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.

## License

This project is open-source software licensed under the [MIT License](https://opensource.org/license/mit).
