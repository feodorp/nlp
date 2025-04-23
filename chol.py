import numpy as np
from scipy.linalg import ldl

def make_positive_definite_iterative(A, epsilon=1e-6, max_iter=100):
    """
    Iteratively modify a symmetric matrix A by adding a*I such that A + a*I is positive definite,
    using LDL decomposition to check for positive definiteness.

    Parameters:
    - A: Symmetric matrix (numpy array)
    - epsilon: Small positive value to ensure strict positive definiteness
    - max_iter: Maximum number of iterations to prevent infinite loops

    Returns:
    - A_prime: Modified positive definite matrix
    - a: Scalar shift added to the diagonal
    """
    # Validate input
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")

    a = 0.0  # Initial shift
    I = np.eye(A.shape[0])

    for k in range(max_iter):
        # Compute LDL decomposition of A + a*I
        L, D, perm = ldl(A + a * I)

        # Check the diagonal elements of D
        d_diag = np.diag(D)
        if np.all(d_diag > 0):
            return A + a * I, a

        # Find the smallest diagonal element
        d_min = np.min(d_diag)

        # Update a to make d_min positive
        a -= d_min + epsilon  # Since d_min <= 0, this increases a

    raise RuntimeError("Failed to make matrix positive definite within max iterations")

def modified_cholesky(A):
    """
    Modify a symmetric matrix A to be positive definite using the Gerschgorin lower bound,
    then return its Cholesky factorization with pivoting.

    Parameters:
    - A: Symmetric matrix (numpy array)
    - epsilon: Small positive value to ensure strict positive definiteness

    Returns:
    - L: Lower triangular matrix from Cholesky factorization
    """
    # Validate input
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")

    Amod, a = make_positive_definite_iterative(A)
    # Modify the matrix

    L = np.linalg.cholesky( Amod)

    return L
