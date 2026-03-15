import numpy as np

def bicgstab(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    BiConjugate Gradient Stabilized method.
    Suitable for general (asymmetric) matrices. Transpose-free.
    """

    N = len(b)
    if x0 is None:
        x = np.zeros(N)
    else:
        x = x0.copy()

    r = b - A @ x
    r0_star = r.copy()
    
    p = r.copy()

    res_history = [np.linalg.norm(r)]
    tol_abs = tol * res_history[0] if res_history[0] != 0 else tol 

    rho_old = np.dot(r, r0_star)

    for i in range(max_iter):
        if res_history[-1] < tol_abs:
            break

        if rho_old == 0:
            print("Breakdown in BiCGSTAB: rho = 0")
            break

        p_aux = A @ p

        den_alpha = np.dot(p_aux, r0_star)
        if den_alpha == 0:
            print("Breakdown in BiCGSTAB: den_alpha = 0")
            break

        alpha = rho_old / den_alpha

        s = r - alpha * p_aux 
        s_aux = A @ s

        omega = np.dot(s_aux, s) / np.dot(s_aux, s_aux)
        if omega == 0:
            print("Breakdown in BiCGSTAB: omega = 0")
            break

        x = x + alpha * p + omega * s 
        r = s - omega * s_aux 

        rho_new = np.dot(r, r0_star)

        beta = (rho_new / rho_old) * (alpha / omega) 

        p = r + beta * (p - omega * p_aux)

        rho_old = rho_new

        res_history.append(np.linalg.norm(r))
    
    return x, res_history