from chol import modified_cholesky, modified_ldlt
import numpy as np
import scipy

def backtracking(x_k, p_k, f, grad_f, c=1e-2, rho=0.5):
    """
    Perform backtracking line search to satisfy Armijo condition.

    Parameters:
        x_k: Current point (numpy array)
        p_k: Search direction (numpy array)
        f: Objective function
        grad_f: Gradient function
        c: Armijo condition parameter (0 < c < 1)
        rho: Backtracking reduction factor (0 < rho < 1)

    Returns:
        alpha: Step size satisfying Armijo condition
    """
    alpha = 1.0
    f_k = f(x_k)
    grad_k = grad_f(x_k)
    while f(x_k + alpha * p_k) > f_k + c * alpha * np.dot(grad_k, p_k):
        alpha *= rho
    return alpha

def line_search_backtracking(x0, f, grad_f, method='steepest', hess_f=None, return_path=False, tol=1e-6, max_iter=1000):
    """
    Line search optimization with specified direction method.

    Parameters:
        x0: Initial point (numpy array)
        f: Objective function
        grad_f: Gradient function
        hess_f: Hessian function (required for 'hessian' method)
        method: Direction method ('steepest', 'bfgs', 'hessian')
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        x: Approximate minimizer (numpy array)
    """
    x_k = np.array(x0)
    k = 0
    if return_path:
        path = [x_k.copy()]

    # BFGS-specific initialization
    if method == 'bfgs':
        n = len(x0)
        H_k = np.eye(n)  # Initial inverse Hessian approximation

    while k < max_iter:
        grad_k = grad_f(x_k)
        if np.linalg.norm(grad_k) < tol:
            break

        # Compute direction based on method
        if method == 'steepest':
            p_k = -grad_k

        elif method == 'bfgs':
            # Compute direction using BFGS inverse Hessian approximation
            p_k = -H_k @ grad_k

        elif method == 'hessian':
            if hess_f is None:
                raise ValueError("Hessian function required for 'hessian' method")
            H_k = hess_f(x_k)
            # Use modified Cholesky to ensure positive definiteness
            # L = modified_cholesky(H_k)
            # y = np.linalg.solve(L, -grad_k)
            # p_k = np.linalg.solve(L.T, y)
            L, D, p, D0 = modified_ldlt(H_k)
            pb = -grad_k[p]
            y = scipy.linalg.solve_triangular(L[p,], pb, lower=True, unit_diagonal=True)
            z =  scipy.linalg.solve(D, y)
            p_k = np.zeros_like(x_k)
            p_k[p] = scipy.linalg.solve_triangular(L[p,:].T, z, lower=False, unit_diagonal=True)

        else:
            raise ValueError("Unknown method. Use 'steepest', 'bfgs', or 'hessian'.")

        # Perform backtracking line search
        alpha_k = backtracking(x_k, p_k, f, grad_f)


        # Update position
        x_k_new = x_k + alpha_k * p_k

        # BFGS update for inverse Hessian approximation
        if method == 'bfgs':
            s_k = x_k_new - x_k
            y_k = grad_f(x_k_new) - grad_k
            if np.dot(y_k, s_k) > 0:  # Ensure positive curvature
                rho_k = 1 / np.dot(y_k, s_k)
                H_k = (np.eye(n) - rho_k * np.outer(s_k, y_k)) @ H_k @ (np.eye(n) - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)

        x_k = x_k_new
        k += 1
        if return_path:
            path.append(x_k.copy())

    if return_path:
        return np.array(path)
    else:
        return x_k
