import numpy as np
from .utils import apply_M_inv

def bicgstab(A, b, x0=None, M=None, tol=1e-6, max_iter=1000):
    """
    Preconditioned Biconjugate Gradient method.
    Suitable for general (asymmetric) matrices.
    """

    N = len(b)
    if x0 is None:
        x = np.zeros(N)
    else:
        x = x0.copy()

    r = b - A @ x
    r_star = r.copy() # Initial dual residue (r_0^0)
    z = apply_M_inv(M, r)
    p = z.copy()

    res_history = [np.linalg.norm(r)]
    tol_abs = tol * res_history[0] if res_history[0] != 0 else tol 

    for i in range(max_iter):
        if res_history[-1] < tol_abs
            break

    # TODO: Complete
        
    return x, res_history