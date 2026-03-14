import numpy as np 

def apply_M_inv(M, x):
    """
    Applies the preconditioner M^{-1} to a vector x.
    If M is None, it is equivalent to not preconditioning (M = I).
    """
    if M is None:
        return x.copy()
    # if M is a sparse matrix or a linear operator, we use hit dot method
    return M.dot(x)