import numpy as np

def bicg(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    BiConjugate Gradient method.
    Suitable for general (asymmetric) matrices.
    """

    N = len(b)
    if x0 is None:
        x = np.zeros(N)
    else:
        x = x0.copy()

    r = b - A @ x
    r_star = r.copy()
    
    p = r.copy()
    p_star = r_star.copy()

    res_history = [np.linalg.norm(r)]
    tol_abs = tol * res_history[0] if res_history[0] != 0 else tol 

    rho = np.dot(r, r_star)

    for i in range(max_iter):
        if res_history[-1] < tol_abs:
            break

        if rho == 0:
            print("Breakdown in BiCG: rho = 0")
            break

        p_aux = A @ p

        alpha = rho / np.dot(p_aux, p_star)

        x = x + alpha * p
        r = r - alpha * p_aux
        r_star = r_star - alpha * A.T @ p_star

        rho_new = np.dot(r, r_star)
        beta = rho_new / rho

        p = r + beta * p 
        p_star = r_star + beta * p_star

        rho = rho_new

        res_history.append(np.linalg.norm(r))

    return x, res_history