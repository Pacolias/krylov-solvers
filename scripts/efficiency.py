import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt 
import os
import sys
import time # Added to measure time externally

# Adds project root to the path for importing src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import gmres, bicg, cgs, bicgstab

def reconstruct_times(hist, total_time, is_gmres=False):
    """
    Reconstructs the time array based on the theoretical complexity 
    of the algorithms to avoid modifying the original source code.
    """
    m = len(hist) - 1 # Actual number of iterations (discounting r0)
    if m <= 0:
        return np.array([0.0])
        
    k = np.arange(m + 1)
    if is_gmres:
        # GMRES: Cost per iteration O(n), total cost O(n^2)
        return total_time * (k * (k + 1)) / (m * (m + 1))
    else:
        # BiCG, CGS, BiCGSTAB: Constant cost per iteration O(1)
        return np.linspace(0, total_time, m + 1)

def solve_efficiency(matrix_path, name, max_iter):
    print(f"\n{'='*40}")
    print(f"Generating efficiency curve for: {name}")
    print(f"{'='*40}")

    # Loads matrix
    A = sio.mmread(matrix_path).tocsr()
    N = A.shape[0]

    # Configures the model problem
    x_exact = np.ones(N)
    b = A @ x_exact
    
    histories = {}
    times = {}

    # Execute all methods, measure total time, and reconstruct time curves
    print("--> Executing GMRES...")
    t0 = time.perf_counter()
    x_gmres, hist_gmres = gmres(A, b, max_iter=max_iter)
    t_total = time.perf_counter() - t0
    histories['GMRES'] = np.array(hist_gmres) / hist_gmres[0]
    times['GMRES'] = reconstruct_times(hist_gmres, t_total, is_gmres=True)

    print("--> Executing BiCG...")
    t0 = time.perf_counter()
    x_bicg, hist_bicg = bicg(A, b, max_iter=max_iter)
    t_total = time.perf_counter() - t0
    histories['BiCG'] = np.array(hist_bicg) / hist_bicg[0]
    times['BiCG'] = reconstruct_times(hist_bicg, t_total, is_gmres=False)
    
    print("--> Executing CGS...")
    t0 = time.perf_counter()
    x_cgs, hist_cgs = cgs(A, b, max_iter=max_iter)
    t_total = time.perf_counter() - t0
    histories['CGS'] = np.array(hist_cgs) / hist_cgs[0]
    times['CGS'] = reconstruct_times(hist_cgs, t_total, is_gmres=False)

    print("--> Executing BiCGSTAB...")
    t0 = time.perf_counter()
    x_bicgstab, hist_bicgstab = bicgstab(A, b, max_iter=max_iter)
    t_total = time.perf_counter() - t0
    histories['BiCGSTAB'] = np.array(hist_bicgstab) / hist_bicgstab[0]
    times['BiCGSTAB'] = reconstruct_times(hist_bicgstab, t_total, is_gmres=False)

    # Graphs: Error vs Time
    plt.figure(figsize=(10, 6))
    
    # Plot using time arrays (X) vs history (Y)
    plt.semilogy(times['GMRES'], histories['GMRES'], label='GMRES', linewidth=2.5, color='black')
    plt.semilogy(times['BiCG'], histories['BiCG'], label='BiCG', linewidth=1.5, alpha=0.8, color='blue')
    plt.semilogy(times['CGS'], histories['CGS'], label='CGS', linewidth=1.5, alpha=0.7, color='green')
    plt.semilogy(times['BiCGSTAB'], histories['BiCGSTAB'], label='BiCGSTAB', linewidth=2, color='red')
    
    plt.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerancia ($10^{-6}$)')
    
    plt.title(f'Eficiencia Computacional - Matriz: {name}')
    plt.xlabel('Tiempo de ejecución (segundos)')
    plt.ylabel('Residuo relativo $||r^n||_2 / ||r^0||_2$')
    
    plt.ylim(bottom=1e-8)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Save graph to the absolute plots/ directory at the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plots_dir, f'eficiencia_{name}.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Calculates the absolute path to the project root (one level above the script)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Builds the exact path to the matrix
    matriz_pores = os.path.join(project_root, 'data', 'pores_2.mtx')
    
    # Solves the system
    solve_efficiency(matriz_pores, 'pores_2', max_iter=2000)