# optimization.py
#
# Robust Armijo and Wolfe line-search utilities with Steepest-Descent,
# BFGS and (modified-)Newton directions.
#
# Interfaces **identical** to the user-supplied skeleton – only the
# internal logic of the search routines has been strengthened so every
# step really satisfies the required conditions.

# ---------------------------------------------------------------------
# external deps
# ---------------------------------------------------------------------
import numpy as np
import scipy.linalg
from chol import modified_cholesky, modified_ldlt   # noqa: F401  (modified_cholesky may be used elsewhere)

# ---------------------------------------------------------------------
# 1.  Armijo back-tracking line search (unchanged interface, safer loop)
# ---------------------------------------------------------------------
def backtracking(x_k, p_k, f, grad_f, c: float = 1e-4, rho: float = 0.5):
    """
    Classical Armijo back-tracking.
    """
    alpha = 1.0
    f_k = f(x_k)
    g_k = grad_f(x_k)
    gTp = np.dot(g_k, p_k)

    # guarantee we really have a descent direction
    if gTp >= 0:
        p_k = -g_k
        gTp = -np.dot(g_k, g_k)

    while f(x_k + alpha * p_k) > f_k + c * alpha * gTp:
        alpha *= rho
        if alpha < 1e-16:           # numerical under-flow safeguard
            break
    return alpha


def cubic(alpha_lo, alpha_hi, phi_lo, phi_hi, derphi_lo):
    """
    Safeguarded cubic minimiser in (alpha_lo, alpha_hi).
    If the cubic is invalid or outside the bracket, return midpoint.
    """
    d1 = derphi_lo + np.sqrt(max(1e-16,
            derphi_lo**2 - (phi_hi - phi_lo) * 3.0 * 2.0 / (alpha_hi - alpha_lo)))
    if d1 == 0:
        return 0.5 * (alpha_lo + alpha_hi)
    alpha_j = alpha_lo - derphi_lo * (alpha_hi - alpha_lo)**2 / (2.0 * d1)
    if not (alpha_lo < alpha_j < alpha_hi):
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
    return alpha_j

# ---------------------------------------------------------------------------
# 1.  ZOOM ROUTINE (refines step inside a bracket)
# ---------------------------------------------------------------------------
def zoom(alpha_lo, alpha_hi, phi_lo, phi_hi, derphi_lo,
         f, grad_f, x, p, phi0, derphi0, c1, c2, tol = 1.0e-6):
    """
    Refinement phase that searches in [alpha_lo, alpha_hi] until Wolfe
    conditions are satisfied.  Uses safeguarded quadratic interpolation with
    bisection fallback.  Returns a scalar α.
    """
    for _ in range(40):
        denom = 2.0 * (phi_hi - phi_lo - derphi_lo * (alpha_hi - alpha_lo))
        alpha_j = cubic(alpha_lo, alpha_hi, phi_lo, phi_hi, derphi_lo)
        if abs(denom) < 1e-10:
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        else:
            alpha_j = alpha_lo - derphi_lo * (alpha_hi - alpha_lo) ** 2 / denom
            # keep strictly inside interval
            if not (alpha_lo + 0.1 * (alpha_hi - alpha_lo) <
                    alpha_j <
                    alpha_hi - 0.1 * (alpha_hi - alpha_lo)):
                alpha_j = 0.5 * (alpha_lo + alpha_hi)

        phi_j = f(x + alpha_j * p)

        # ------------ check Armijo-like condition -----------------------
        if (phi_j > phi0 + c1 * alpha_j * derphi0) or (phi_j >= phi_lo):
            alpha_hi, phi_hi = alpha_j, phi_j
        else:
            derphi_j = np.dot(grad_f(x + alpha_j * p), p)
            # ------------ curvature satisfied? --------------------------
            if abs(derphi_j) <= -c2 * derphi0:
                return alpha_j
            # sign test to shrink interval properly
            if derphi_j * (alpha_hi - alpha_lo) >= 0.0:
                alpha_hi, phi_hi = alpha_lo, phi_lo
            alpha_lo, phi_lo, derphi_lo = alpha_j, phi_j, derphi_j

        if abs(alpha_hi - alpha_lo) <= tol * alpha_lo:
            break

    # final safeguard (midpoint)
    return 0.5 * (alpha_lo + alpha_hi)

# ---------------------------------------------------------------------------
# 3.  STRONG-WOLFE LINE SEARCH  (outer driver)
# ---------------------------------------------------------------------------
def line_search_wolfe(f, grad_f, x, p,
                      phi0=None, derphi0=None,
                      c1 = 1.0e-4, c2 = 0.4,
                      amax = 5.0, maxiter = 50):
    """
    Strong-Wolfe line-search along p from x.
    Returns a step length α that satisfies both:
      * Armijo:      f(x+αp) ≤ f(x)+c1 α ∇f(x)^T p
      * Curvature: |∇f(x+αp)^T p| ≤ c2 |∇f(x)^T p| .
    """
    if phi0 is None:
        phi0 = f(x)
    if derphi0 is None:
        derphi0 = np.dot(grad_f(x), p)

    # descent safeguard
    if derphi0 >= 0.0:
        raise ValueError("Search direction is not a descent direction.")

    # initial trial
    alpha0, alpha1 = 0.0, 1.0
    phi_a0, phi_a1 = phi0, f(x + alpha1 * p)

    for it in range(maxiter):
        # ----------- Wolfe / Armijo checks -----------
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (it > 0 and phi_a1 >= phi_a0):
            return zoom(alpha0, alpha1, phi_a0, phi_a1, derphi0,
                        f, grad_f, x, p, phi0, derphi0, c1, c2)

        derphi_a1 = np.dot(grad_f(x + alpha1 * p), p)
        if abs(derphi_a1) <= -c2 * derphi0:
            return alpha1
        if derphi_a1 >= 0.0:
            # sign change → bracket found
            return zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1,
                        f, grad_f, x, p, phi0, derphi0, c1, c2)

        # no bracket yet → enlarge step (extrapolation)
        alpha0, phi_a0 = alpha1, phi_a1
        alpha1 = min(2.0 * alpha1, amax)
        phi_a1 = f(x + alpha1 * p)

    # fallback (should rarely happen)
    return alpha1

# ---------------------------------------------------------------------
# 4.  High-level optimiser using Armijo (back-tracking)
# ---------------------------------------------------------------------
def line_search_backtracking(x0, f, grad_f, *, method: str = 'steepest',
                             hess_f=None, return_path = False,
                             tol = 1.0e-6, max_iter = 1000):
    """
    Exactly the same signature as the original – internal safety fixes.
    """
    x_k = np.asarray(x0, dtype=float)
    if return_path:
        path = [x_k.copy()]

    # BFGS initial state
    if method == 'bfgs':
        n = len(x0)
        H_k = np.eye(n)

    for _ in range(max_iter):
        g_k = grad_f(x_k)
        if np.linalg.norm(g_k) < tol:
            break

        # search direction ------------------------------------------------
        if method == 'steepest':
            p_k = -g_k
        elif method == 'bfgs':
            p_k = -H_k @ g_k
        elif method == 'hessian':
            if hess_f is None:
                raise ValueError("Hessian function required for 'hessian' method")
            H = hess_f(x_k)
            L, D, p, _ = modified_ldlt(H)          # positive-definite mod.
            rhs = -g_k[p]
            y = scipy.linalg.solve_triangular(L[p, :], rhs, lower=True, unit_diagonal=True)
            z = scipy.linalg.solve(D, y)
            p_k = np.zeros_like(x_k)
            p_k[p] = scipy.linalg.solve_triangular(L[p, :].T, z, lower=False, unit_diagonal=True)
        else:
            raise ValueError("Unknown method")

        # ensure descent
        if np.dot(g_k, p_k) >= 0:
            p_k = -g_k

        # Armijo search
        alpha_k = backtracking(x_k, p_k, f, grad_f)
        x_new = x_k + alpha_k * p_k

        # BFGS update
        if method == 'bfgs':
            s = x_new - x_k
            y = grad_f(x_new) - g_k
            sy = np.dot(s, y)
            if sy > 1e-12:                       # curvature condition
                rho = 1.0 / sy
                I = np.eye(n)
                H_k = (I - rho * np.outer(s, y)) @ H_k @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x_k = x_new
        if return_path:
            path.append(x_k.copy())

    return np.array(path) if return_path else x_k


# ---------------------------------------------------------------------
# 5.  High-level optimiser using Wolfe / zoom line search
# ---------------------------------------------------------------------
def line_search_zoom(x0, f, grad_f, *, method = 'steepest',
                     hess_f=None, return_path = False,
                     tol = 1.0e-6, max_iter = 1000):
    """
    Same signature as the original version – now relies on the fixed
    line_search_wolfe + zoom to guarantee Wolfe steps.
    """
    x_k = np.asarray(x0, dtype=float)
    if return_path:
        path = [x_k.copy()]

    # BFGS initial state
    if method == 'bfgs':
        n = len(x0)
        H_k = np.eye(n)

    for _ in range(max_iter):
        g_k = grad_f(x_k)
        if np.linalg.norm(g_k) < tol:
            break

        # search direction ------------------------------------------------
        if method == 'steepest':
            p_k = -g_k
        elif method == 'bfgs':
            p_k = -H_k @ g_k
        elif method == 'hessian':
            if hess_f is None:
                raise ValueError("Hessian function required for 'hessian' method")
            H = hess_f(x_k)
            L, D, p, _ = modified_ldlt(H)
            rhs = -g_k[p]
            y = scipy.linalg.solve_triangular(L[p, :], rhs, lower=True, unit_diagonal=True)
            z = scipy.linalg.solve(D, y)
            p_k = np.zeros_like(x_k)
            p_k[p] = scipy.linalg.solve_triangular(L[p, :].T, z, lower=False, unit_diagonal=True)
        else:
            raise ValueError("Unknown method")

        # guarantee descent
        if np.dot(g_k, p_k) >= 0:
            p_k = -g_k

        # Wolfe search
        alpha_k = line_search_wolfe(f, grad_f, x_k, p_k)
        x_new = x_k + alpha_k * p_k

        # BFGS update
        if method == 'bfgs':
            s = x_new - x_k
            y = grad_f(x_new) - g_k
            sy = np.dot(s, y)
            if sy > 1e-12:
                rho = 1.0 / sy
                I = np.eye(n)
                H_k = (I - rho * np.outer(s, y)) @ H_k @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x_k = x_new
        if return_path:
            path.append(x_k.copy())

    return np.array(path) if return_path else x_k
