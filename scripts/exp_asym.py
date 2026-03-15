import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt 
import os
import sys

# Adds project root to the path for importing src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import gmres, bicg, cgs, bicgstab

def solve_spd(matrix_path, name, max_iter):
    print(f"\n{'='*40}")
    print(f"Solving asymmetric matrix: {name}")
    print(f"{'='*40}")

    # Loads matrix
    A = sio.mmread(matrix_path).tocsr()
    N = A.shape[0]
    print(f"Size: {N}x{N}, Non-zero elements: {A.nnz}")

    # Configures the model problem
    x_exact = np.ones(N)
    b = A @ x_exact
    
    histories = {}

    # Execute all methods and computes relative residue ||r^n|| / ||r^0||
    print("--> Executing GMRES...")
    x_gmres, hist_gmres = gmres(A, b, max_iter=max_iter)
    histories['GMRES'] = np.array(hist_gmres) / hist_gmres[0]

    print("--> Executing BiCG...")
    x_bicg, hist_bicg = bicg(A, b, max_iter=max_iter)
    histories['BiCG'] = np.array(hist_bicg) / hist_bicg[0]
    
    print("--> Executing CGS...")
    x_cgs, hist_cgs = cgs(A, b, max_iter=max_iter)
    histories['CGS'] = np.array(hist_cgs) / hist_cgs[0]

    print("--> Executing BiCGSTAB...")
    x_bicgstab, hist_bicgstab = bicgstab(A, b, max_iter=max_iter)
    histories['BiCGSTAB'] = np.array(hist_bicgstab) / hist_bicgstab[0]

    # Graphs
    plt.figure(figsize=(10, 6))
    
    # Colors and styles to clearly differentiate behaviors 
    plt.semilogy(histories['GMRES'], label='GMRES', linewidth=2.5, color='black')
    plt.semilogy(histories['BiCG'], label='BiCG', linewidth=1.5, alpha=0.8, color='blue')
    plt.semilogy(histories['CGS'], label='CGS', linewidth=1.5, alpha=0.7, color='green')
    plt.semilogy(histories['BiCGSTAB'], label='BiCGSTAB', linewidth=2, color='red')
    
    plt.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerancia ($10^{-6}$)')
    
    plt.title(f'Historial de convergencia - Matriz Asimétrica: {name}')
    plt.xlabel('Número de iteraciones')
    plt.ylabel('Residuo relativo $||r^n||_2 / ||r^0||_2$')
    
    # Adjust Y-axis limits to prevent CGS peaks from breaking the visual scale
    plt.ylim(bottom=1e-8)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Save graph
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/convergencia_asym_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # The paths assume that the script is run from the scripts/ folder
    # and matrices are stored in data/
    solve_spd('data/sherman1.mtx', 'sherman1', max_iter=2000)
    solve_spd('data/pores_2.mtx', 'pores_2', max_iter=2000)