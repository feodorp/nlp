# linesearch.py
#
# Robust Armijo and Wolfe line-search with Steepest-Descent,
# BFGS, SR1 and (modified-)Newton directions.
#

import numpy as np
import scipy.linalg
from chol import modified_cholesky, modified_ldlt

# ---------------------------------------------------------------------
# 1.  Armijo back-tracking line search
# ---------------------------------------------------------------------
#
# simple backtracking:
def backtracking_simple(phi, phi0, dphi0, c1=1e-4, rho=0.5, alpha0=1.0, amin=0, max_iter=20):
    """
    Classical Armijo backtracking line search.

    Parameters
    ----------
    phi : callable
        Function of step size alpha, phi(alpha) = f(x + alpha * p).
    phi0 : float
        Function value at alpha = 0, i.e., f(x).
    dphi0 : float
        Directional derivative at alpha = 0, i.e., grad_f(x)^T p.
    c1 : float, optional
        Armijo condition parameter, typically small (default: 1e-4).
    rho : float, optional
        Step size reduction factor, 0 < rho < 1 (default: 0.5).
    alpha0 : float, optional
        Initial step size (default: 1.0).
    amin : float, optional
        Minimum allowable step size (default: 0).
    max_iter : int, optional
        Maximum number of iterations for cubic interpolation (default: 20).

    Returns
    -------
    alpha : float or None
        Step size satisfying Armijo condition, or None if search fails.
    phi_alpha : float
        Function value phi(alpha) at the returned step size.
    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * dphi0:
        return alpha0, phi_a0

    alpha = alpha0
    phi_a = phi_a0

    for _ in range(max_iter):
        if alpha <= amin:
            break
        alpha = rho * alpha
        phi_a = phi(alpha)
        if phi_a <= phi0 + c1 * alpha * dphi0:
            return alpha, phi_a

    # Failed to find a suitable step length
    return None, phi_a

#
# backtracking using interpolation
def backtracking_interp(phi, phi0, dphi0, c1=1.e-4, rho=0.5, alpha0=1.0, amin=1.e-6, max_iter = 20):
    """
    Backtracking line search using interpolation (Nocedal and Wright, Section 3.5).

    Parameters
    ----------
    phi : callable
        Function of step size alpha, phi(alpha) = f(x + alpha * p).
    phi0 : float
        Function value at alpha = 0, i.e., f(x).
    dphi0 : float
        Directional derivative at alpha = 0, i.e., grad_f(x)^T p.
    c1 : float, optional
        Armijo condition parameter (default: 1e-4).
    rho : float, optional
        Fallback reduction factor, 0 < rho < 1 (default: 0.5).
    alpha0 : float, optional
        Initial step size (default: 1.0).
    amin : float, optional
        Minimum allowable step size (default: 0).
    max_iter : int, optional
        Maximum number of iterations for cubic interpolation (default: 20).

    Returns
    -------
    alpha : float or None
        Step size satisfying Armijo condition, or None if search fails.
    phi_alpha : float
        Function value phi(alpha) at the returned step size.
    """


    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * dphi0:
        return alpha0, phi_a0

    # Quadratic interpolation (Eq. 3.58 in Nocedal and Wright)
    denom = phi_a0 - phi0 - dphi0 * alpha0
    if denom > 1.e-8:
        alpha1 = -(dphi0) * alpha0**2 / (2.0 * denom)
        alpha1 = max(0.1 * alpha0, min(alpha1, rho * alpha0))  # Constrain within [0.1*alpha0, rho*alpha0]
    else:
        alpha1 = rho * alpha0

    phi_a1 = phi(alpha1)
    if phi_a1 <= phi0 + c1 * alpha1 * dphi0:
        return alpha1, phi_a1

    alpha_prev, alpha_curr = alpha0, alpha1
    phi_prev, phi_curr = phi_a0, phi_a1
    for _ in range(max_iter):
        if alpha_curr <= amin:
            break

        # Cubic interpolation with safeguards
        d1 = phi_prev - phi0 - dphi0 * alpha_prev
        d2 = phi_curr - phi0 - dphi0 * alpha_curr
        denum = alpha_prev**2 * alpha_curr**2 * (alpha_curr - alpha_prev)
        if abs(denum) < 1.e-8:
            alpha_next = rho * alpha_curr
        else:
            a = (alpha_prev**2 * d2 - alpha_curr**2 * d1) / denum
            b = (-alpha_prev**3 * d2 + alpha_curr**3 * d1) / denum
            discriminant = b**2 - 3 * a * dphi0
            if a == 0 or discriminant < 0:
                alpha_next = rho * alpha_curr
            else:
                alpha_next = (-b + np.sqrt(discriminant)) / (3 * a)
                alpha_next = max(0.1 * alpha_curr, min(alpha_next, rho * alpha_curr))  # Constrain step

        phi_next = phi(alpha_next)
        if phi_next <= phi0 + c1 * alpha_next * dphi0:
            return alpha_next, phi_next

        alpha_prev, alpha_curr = alpha_curr, alpha_next
        phi_prev, phi_curr = phi_curr, phi_next

    # Failed to find a suitable step length
    return alpha_curr, phi_curr
    # return None, phi_curr

def cubic(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo):
    """
    Cubic minimizer within interval (alpha_lo, alpha_hi).

    Parameters
    ----------
    alpha_lo : float
        Lower bound of the interval.
    alpha_hi : float
        Upper bound of the interval.
    phi_lo : float
        Function value at alpha_lo.
    phi_hi : float
        Function value at alpha_hi.
    dphi_lo : float
        Derivative at alpha_lo.

    Returns
    -------
    alpha_j : float
        Estimated minimizer within (alpha_lo, alpha_hi), or midpoint if invalid.
    """
    # Check if the interval is too small to avoid division by zero
    if alpha_hi - alpha_lo < 1.e-8:
        return 0.5 * (alpha_lo + alpha_hi)

    # Compute the discriminant for the square root
    discriminant = dphi_lo**2 - (phi_hi - phi_lo) * 3.0 * 2.0 / (alpha_hi - alpha_lo)
    # If discriminant is negative, return midpoint to avoid complex numbers
    if discriminant < 0:
        return 0.5 * (alpha_lo + alpha_hi)

    # Compute d1, ensuring the square root argument is positive
    d1 = dphi_lo + np.sqrt(max(1e-16, discriminant))
    # If d1 is zero, return midpoint to avoid division by zero
    if d1 == 0:
        return 0.5 * (alpha_lo + alpha_hi)

    # Compute the estimated minimizer
    alpha_j = alpha_lo - dphi_lo * (alpha_hi - alpha_lo)**2 / (2.0 * d1)
    # Ensure alpha_j is within the interval, otherwise return midpoint
    if not (alpha_lo < alpha_j < alpha_hi):
        alpha_j = 0.5 * (alpha_lo + alpha_hi)

    return alpha_j

# ---------------------------------------------------------------------------
# 1.  ZOOM ROUTINE
# ---------------------------------------------------------------------------
def zoom(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo,
         f, grad_f, x, p, phi0, dphi0, c1, c2, tol = 1.0e-6):
    """
    Refinement phase to find step size satisfying strong Wolfe conditions within [alpha_lo, alpha_hi].

    Parameters
    ----------
    alpha_lo : float
        Lower bound of the interval.
    alpha_hi : float
        Upper bound of the interval.
    phi_lo : float
        Function value at alpha_lo.
    phi_hi : float
        Function value at alpha_hi.
    dphi_lo : float
        Directional derivative at alpha_lo.
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x : ndarray
        Current point.
    p : ndarray
        Search direction.
    phi0 : float
        Function value at alpha = 0, i.e., f(x).
    dphi0 : float
        Directional derivative at alpha = 0, i.e., grad_f(x)^T p.
    c1 : float
        Armijo condition parameter (default: 1e-4).
    c2 : float
        Curvature condition parameter (default: 0.4).
    tol : float, optional
        Tolerance for interval size (default: 1e-6).

    Returns
    -------
    alpha : float
        Step size satisfying strong Wolfe conditions.
    """
    for _ in range(40):
        denom = 2.0 * (phi_hi - phi_lo - dphi_lo * (alpha_hi - alpha_lo))
        alpha_j = cubic(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo)
        if abs(denom) < 1e-10:
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        else:
            alpha_j = alpha_lo - dphi_lo * (alpha_hi - alpha_lo)**2 / denom
            if not (alpha_lo + 0.1 * (alpha_hi - alpha_lo) < alpha_j < alpha_hi - 0.1 * (alpha_hi - alpha_lo)):
                alpha_j = 0.5 * (alpha_lo + alpha_hi)

        phi_j = f(x + alpha_j * p)

        # Check Armijo condition
        if (phi_j > phi0 + c1 * alpha_j * dphi0) or (phi_j >= phi_lo):
            alpha_hi, phi_hi = alpha_j, phi_j
        else:
            dphi_j = np.dot(grad_f(x + alpha_j * p), p)
            # Check curvature condition
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j
            # Adjust interval based on derivative sign
            if dphi_j * (alpha_hi - alpha_lo) >= 0.0:
                alpha_hi, phi_hi = alpha_lo, phi_lo
            alpha_lo, phi_lo, dphi_lo = alpha_j, phi_j, dphi_j

        if abs(alpha_hi - alpha_lo) <= tol * alpha_lo:
            break

    # Fallback to midpoint
    return 0.5 * (alpha_lo + alpha_hi)

# ---------------------------------------------------------------------------
# 3.  STRONG-WOLFE LINE SEARCH
# ---------------------------------------------------------------------------
def line_search_wolfe(f, grad_f, x, p,
                      phi0=None, dphi0=None,
                      c1 = 1.0e-4, c2 = 0.6,
                      amax = 5.0, maxiter = 50):
    """
    Strong Wolfe line search to find step size along direction p from point x.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x : ndarray
        Current point.
    p : ndarray
        Search direction.
    phi0 : float, optional
        Function value at alpha = 0, i.e., f(x). Computed if None (default: None).
    dphi0 : float, optional
        Directional derivative at alpha = 0, i.e., grad_f(x)^T p. Computed if None (default: None).
    c1 : float, optional
        Armijo condition parameter (default: 1e-4).
    c2 : float, optional
        Curvature condition parameter (default: 0.4).
    amax : float, optional
        Maximum allowable step size (default: 5.0).
    maxiter : int, optional
        Maximum number of iterations (default: 50).

    Returns
    -------
    alpha : float
        Step size satisfying strong Wolfe conditions.
    """
    if phi0 is None:
        phi0 = f(x)
    if dphi0 is None:
        dphi0 = np.dot(grad_f(x), p)

    if dphi0 >= 0.0:
        raise ValueError("Search direction is not a descent direction.")

    alpha0, alpha1 = 0.0, 1.0
    phi_a0, phi_a1 = phi0, f(x + alpha1 * p)

    for k in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or (k > 0 and phi_a1 >= phi_a0):
            return zoom(alpha0, alpha1, phi_a0, phi_a1, dphi0, f, grad_f, x, p, phi0, dphi0, c1, c2)

        dphi_a1 = np.dot(grad_f(x + alpha1 * p), p)
        if abs(dphi_a1) <= -c2 * dphi0:
            return alpha1
        if dphi_a1 >= 0.0:
            return zoom(alpha1, alpha0, phi_a1, phi_a0, dphi_a1, f, grad_f, x, p, phi0, dphi0, c1, c2)

        alpha0, phi_a0 = alpha1, phi_a1
        alpha1 = min(2.0 * alpha1, amax)
        phi_a1 = f(x + alpha1 * p)

    return alpha1  # Fallback


# ---------------------------------------------------------------------
# 4.  High-level optimiser using Armijo
# ---------------------------------------------------------------------
def line_search_backtracking(x0, f, grad_f,
                             line_search_type = 'simple', # ['simple', 'interp']
                             method = 'steepest',
                             hess_f = None,
                             max_iter = 1000,
                             tol_abs = 1.0e-6,
                             tol_rel = 1.0e-6,
                             eps = 1.0e-4):
    """
    Optimization using Armijo line search with various descent methods.

    Parameters
    ----------
    x0 : array-like
        Initial guess for the parameters.
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    line_search_type : str, optional
        Type of line search: 'simple' or 'interp' (default: 'simple').
    method : str, optional
        Optimization method: 'steepest', 'BFGS', 'SR1', or 'hessian' (default: 'steepest').
    hess_f : callable, optional
        Hessian function hess_f(x), required if method='hessian' (default: None).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol_abs : float, optional
        Absolute tolerance for gradient norm (default: 1e-6).
    tol_rel : float, optional
        Relative tolerance for gradient norm (default: 1e-6).
    eps : float, optional
        Minimum tolerance value (default: 1e-6).

    Returns
    -------
    x : ndarray
        Optimized parameters.
    info : list of lists
        Iteration log with entries [iter, ||p||, alpha, ||g||, x] for each iteration.
        - iter : int, iteration number
        - ||p|| : float, norm of search direction
        - alpha : float, step size
        - ||g|| : float, infinity norm of gradient at current point
        - x : ndarray or None, current point (included if dimension <= 2, else None)
    """
    if line_search_type == 'simple':
        scalar_line_search = backtracking_simple
    elif line_search_type == 'interp':
        scalar_line_search = backtracking_interp
    else:
        raise ValueError("line_search_type must be 'simple' or 'interp'")

    if method not in ['steepest', 'BFGS', 'SR1', 'hessian']:
        raise ValueError(f"Unknown method {method}")

    x_k = np.asarray(x0).flatten()
    n = len(x_k)

    g_k = grad_f(x0)
    gnorm0 = np.linalg.norm(g_k, ord=np.inf)
    tolerance = min(tol_rel * gnorm0 + tol_abs, eps)

    # Initialize log
    info = [['iter', '||p||', 'alpha', '||g||', 'x']]
    info.append([0, 'n/a', 'n/a', gnorm0, x_k.copy() if n <= 2 else None])

    # BFGS/SR1 initial state
    if method in ['BFGS', 'SR1']:
        H_k = np.eye(n)

    for k in range(max_iter):
        # Compute search direction
        if method == 'steepest':
            p_k = -g_k
        elif method in ['BFGS', 'SR1']:
            p_k = -H_k @ g_k
        elif method == 'hessian':
            if hess_f is None:
                raise ValueError("Hessian function required for 'hessian' method")
            H = hess_f(x_k)
            L, D, p, _ = modified_ldlt(H, 0.1)
            rhs = -g_k[p]
            y = scipy.linalg.solve_triangular(L[p, :], rhs, lower=True, unit_diagonal=True)
            z = scipy.linalg.solve(D, y)
            p_k = np.zeros_like(x_k)
            p_k[p] = scipy.linalg.solve_triangular(L[p, :].T, z, lower=False, unit_diagonal=True)

        def phi(alpha):
            return f(x_k + alpha * p_k)

        dphi = np.dot(g_k, p_k)
        # ensure descent
        if dphi >= 0:
            p_k = -g_k
            dphi = np.dot(g_k, p_k)

        # Line search (handle None case)
        alpha_k, _ = scalar_line_search(phi, phi(0.0), dphi)
        if alpha_k is None:
            print("The line search could not find a suitable step.")
            break

        x_new = x_k + alpha_k * p_k
        g_new = grad_f(x_new)

        # Update H_k for BFGS or SR1
        if method in ['BFGS', 'SR1']:
            s = x_new - x_k
            y = g_new - g_k
            if method == 'BFGS':
                sy = np.dot(s, y)
                if sy > 1e-12:
                    rho = 1.0 / sy
                    I = np.eye(n)
                    H_k = (I - rho * np.outer(s, y)) @ H_k @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            elif method == 'SR1':
                v = s - H_k @ y
                denominator = np.dot(v, y)
                if abs(denominator) > 1e-12:
                    H_k += np.outer(v, v) / denominator

        # Update and log
        x_k = x_new
        g_k = g_new
        gnorm = np.linalg.norm(g_k, ord=np.inf)
        pnorm = np.linalg.norm(p_k, ord=np.inf)
        info.append([k + 1, pnorm, alpha_k, gnorm, x_k.copy() if n <= 2 else None])

        if gnorm <= tolerance:
            break

    return x_k, info


# ---------------------------------------------------------------------
# 5.  High-level optimiser using Wolfe / zoom line search
# ---------------------------------------------------------------------
def line_search_zoom(x0, f, grad_f,
                     method='steepest',
                     hess_f=None,
                     max_iter=1000,
                     tol_abs=1.0e-6,
                     tol_rel=1.0e-6,
                     eps=1.0e-4):
    """
    High-level optimizer using strong Wolfe line search with zoom subroutine, including SR1 method.

    Parameters
    ----------
    x0 : array-like
        Initial guess for the parameters.
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    method : str, optional
        Optimization method: 'steepest', 'BFGS', 'SR1', or 'hessian' (default: 'steepest').
    hess_f : callable, optional
        Hessian function hess_f(x), required if method='hessian' (default: None).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol_abs : float, optional
        Absolute tolerance for gradient norm (default: 1e-6).
    tol_rel : float, optional
        Relative tolerance for gradient norm (default: 1e-6).
    eps : float, optional
        Minimum tolerance value (default: 1e-6).

    Returns
    -------
    x : ndarray
        Optimized parameters.
    info : list of lists
        Iteration log with entries [iter, ||p||, alpha, ||g||, x] for each iteration.
    """
    if method not in ['steepest', 'BFGS', 'SR1', 'hessian']:
        raise ValueError(f"Unknown method {method}")

    x_k = np.asarray(x0).flatten()
    n = len(x_k)

    g0 = grad_f(x0)
    gnorm0 = np.linalg.norm(g0, ord=np.inf)
    tolerance = min(tol_rel * gnorm0 + tol_abs, eps)

    info = [['iter', '||p||', 'alpha', '||g||', 'x']]
    info.append([0, 'n/a', 'n/a', gnorm0, x_k.copy() if n <= 2 else None])

    if method in ['BFGS', 'SR1']:
        H_k = np.eye(n)

    for k in range(max_iter):
        g_k = grad_f(x_k)

        if method == 'steepest':
            p_k = -g_k
        elif method in ['BFGS', 'SR1']:
            p_k = -H_k @ g_k
        elif method == 'hessian':
            if hess_f is None:
                raise ValueError("Hessian function required for 'hessian' method")
            H = hess_f(x_k)
            L, D, p, _ = modified_ldlt(H, 0.1)
            rhs = -g_k[p]
            y = scipy.linalg.solve_triangular(L[p, :], rhs, lower=True, unit_diagonal=True)
            z = scipy.linalg.solve(D, y)
            p_k = np.zeros_like(x_k)
            p_k[p] = scipy.linalg.solve_triangular(L[p, :].T, z, lower=False, unit_diagonal=True)

        if np.dot(g_k, p_k) >= 0:
            p_k = -g_k

        alpha_k = line_search_wolfe(f, grad_f, x_k, p_k)
        x_new = x_k + alpha_k * p_k

        g_new = grad_f(x_new)
        if method == 'BFGS':
            s = x_new - x_k
            y = g_new - g_k
            sy = np.dot(s, y)
            if sy > 1e-12:
                rho = 1.0 / sy
                I = np.eye(n)
                H_k = (I - rho * np.outer(s, y)) @ H_k @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        elif method == 'SR1':
            s = x_new - x_k
            y = g_new - g_k
            v = s - H_k @ y
            denominator = np.dot(v, y)
            if abs(denominator) > 1e-12:
                H_k += np.outer(v, v) / denominator

        x_k = x_new
        gnorm = np.linalg.norm(g_k, ord=np.inf)
        pnorm = np.linalg.norm(p_k, ord=np.inf)
        info.append([k + 1, pnorm, alpha_k, gnorm, x_k.copy() if n <= 2 else None])

        if gnorm <= tolerance:
            break

    return x_k, info
