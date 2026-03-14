# krylov-solvers

> Python implementations of Krylov subspace methods for solving large sparse linear systems.

## 📌 Overview

Iterative methods based on Krylov subspaces are essential for solving large-scale, sparse linear systems $Ax = b$ where direct methods (like LU factorization) are computationally prohibitive. 

This project provides custom, educational, yet highly functional implementations of three fundamental Krylov solvers:
- **CG** (Conjugate Gradient) - For Symmetric Positive Definite (SPD) matrices.
- **GMRES** (Generalized Minimal Residual) - For general asymmetric matrices (based on the Arnoldi iteration).
- **BiCGSTAB** (Biconjugate Gradient Stabilized) - A transpose-free variant for asymmetric matrices with short recurrences.

All implementations support **left-preconditioning** to accelerate convergence on ill-conditioned real-world matrices.

## 📂 Repository Structure

```text
krylov-solvers/
├── data/                   # Directory to store .mtx (Matrix Market) files
├── src/                    
│   └── krylov_solvers.py   # Core algorithms (cg, gmres, bicgstab)
├── scripts/                
│   ├── exp_spd.py          # Experiments for SPD matrices (CG vs GMRES vs BiCGSTAB)
│   └── exp_asymmetric.py   # Experiments for asymmetric matrices (GMRES vs BiCGSTAB)
├── plots/                  # Generated convergence history plots
├── requirements.txt        # Python dependencies
└── README.md

## ⚙️ Installation & Requirements
This project requires **Python 3.8+**. The main dependencies are `numpy`, `scipy` (for reading Matrix Market files and generating ILU preconditioners) and `matplotlib` (for plotting convergence histories).

1. Clone the repository:

```bash
git clone [https://github.com/YOUR_USERNAME/krylov-solvers.git](https://github.com/YOUR_USERNAME/krylov-solvers.git)
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

## 🚀 Usage & Experiments
To reproduce the experiments, you will need to download sparse matrices in Matrix Market format (`.mtx`) from [Matrix Market](https://math.nist.gov/MatrixMarket/) and place them in the `data/` folder.

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

The scripts will output the number of iterations and the execution time for each method and will display a semi-logarithmic plot comparing their convergence histories ($\|r_k\|_2$ vs. iterations).

## 📖 Theoretical Background
The mathematical derivation and convergence analysis of these algorithms are thoroughly detailed in the accompanying thesis document, heavily based on the theoretical frameworks provided by:

* Dolean, V., Jolivet, P., & Nataf, F. (2015). An Introduction to Domain Decomposition Methods.
* Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.
