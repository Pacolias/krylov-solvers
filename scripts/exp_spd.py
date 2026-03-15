import numpy as np 
import scipy.io as sio
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt 
import os
import sys

# Adds project root to the path for importing src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import cg

def solve_spd(matrix_path, name):
    print(f"\n{'='*40}")
    print(f"Solving SPD matrix: {name}")
    print(f"{'='*40}")

    # Loads matrix
    A = sio.mmread(matrix_path).tocsr()
    N = A.shape[0]
    print(f"Size: {N}x{N}, Non-zero elements: {A.nnz}")

    # Configures the model problem
    x_exact = np.ones(N)
    b = A @ x_exact

    # Defines Jacobi preconditioner (diagonal)
    diagonal_A = A.diagonal()
    # M^{-1} * x consists in dividing vector x by diagonal_A
    def apply_jacobi(v):
        return v / diagonal_A

    M_jacobi = spla.LinearOperator((N, N), matvec=apply_jacobi)

    # Executes CG
    print("--> Executing CG...")
    x_cg, hist_cg = cg(A, b)

    # Executes PCG (with Jacobi preconditioner)
    print("--> Executing PCG...")
    x_pcg, hist_pcg = cg(A, b, M=M_jacobi)

    # Calculates relative residue ||r^n|| / ||r^0||
    rel_hist_cg = np.array(hist_cg) / hist_cg[0]
    rel_hist_pcg = np.array(hist_pcg) / hist_pcg[0]

    # Graphs (logarithmic scale in Y axis)
    plt.figure(figsize=(10, 6))
    plt.semilogy(rel_hist_cg, label=f'CG (Iter: {len(rel_hist_cg)-1})', linewidth=2)
    plt.semilogy(rel_hist_pcg, label=f'PCG Jacobi (Iter: {len(rel_hist_pcg)-1})', 
                 linewidth=2, linestyle='--')

    plt.axhline(y=1e-6, color='r', linestyle=':', label='Tolerancia ($10^{-6}$)')
    plt.title(f'Historial de convergencia - Matriz SPD: {name}')
    plt.xlabel('Número de iteraciones')
    plt.ylabel('Residuo relativo $||r^n||_2 / ||r^0||_2$')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    # Creates plots folder if doesnt exist and save graph
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/convergencia_spd_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # The paths assume that the script is run from the scripts/ folder
    # and matrices are stored in data/
    solve_spd('data/nos6.mtx', 'nos6')
    solve_spd('data/bcsstk14.mtx', 'bcsstk14')