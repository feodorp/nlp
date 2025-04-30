"""
lm.py
================
Non-linear least–squares solver based on the Levenberg–Marquardt method
with a continuous damping-parameter update (Nielsen rule).

Notation
--------
m ........ number of residuals           (rows of J)
n ........ number of parameters          (columns of J)

r(x) ..... residual vector        ℝⁿ → ℝᵐ
J(x) ..... Jacobian of r          ℝⁿ → ℝᵐˣⁿ
f(x) ..... ½‖r(x)‖²
g(x) ..... ∇f(x) = Jᵀr            (called *g* in code)
λ   ...... damping coefficient
ν   ...... geometric growth factor (only used when a step is rejected)
ρ   ...... reduction ratio         (actual / predicted)

Algorithm outline
-----------------
1.  Solve (JᵀJ + λ I) p = −g  for the step p.
2.  Compute ρ.  Accept the step if ρ > 0.
3.  Update λ with the continuous Nielsen rule

        if ρ > 0:
            λ ← λ · max{⅓, 1 − (2ρ − 1)³}      # softer damping
            ν ← 2
        else:
            λ ← λ · ν                          # stronger damping
            ν ← 2ν

4.  Repeat until ‖g‖ is below the mixed absolute / relative tolerance.
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la


# -------------------------------------------------------------------------
#  Helper: Givens rotation and QR factorisation for an (m+n)×n matrix
# -------------------------------------------------------------------------
def _givens_rotation(A: np.ndarray, i: int, j: int):
    """Return Q such that Q @ A zeroes A[i, j] (rows j & i)."""
    n, _ = A.shape
    r = np.hypot(A[j, j], A[i, j])
    if r == 0.0:
        return np.eye(n)
    c, s = A[j, j] / r, A[i, j] / r
    Q = np.eye(n)
    Q[[j, j, i, i], [j, i, j, i]] = c, s, -s, c
    return Q


def _givens_qr(R_I: np.ndarray, n: int, m: int):
    """
    Upper-triangularise the (m+n)×n matrix R_I = [ R ; √λ D ] using
    n(n+1)/2 Givens rotations.

    Returns
    -------
    R_I  : rotated upper-triangular matrix  (in-place)
    Q_T  : transpose of the accumulated rotation matrix  (shape (m+n)×(m+n))
    """
    rows = list(range(m + n, m, -1))  # bottom to top
    Q_tot = np.eye(m + n)
    l = 1                              # rotations per row

    for k in rows:
        for i in range(min(l, n), 0, -1):
            Q = _givens_rotation(R_I, k - 1, n - i)
            R_I[...] = Q @ R_I
            Q_tot = Q @ Q_tot
        l += 1
    return R_I, Q_tot.T


# -------------------------------------------------------------------------
#  Levenberg–Marquardt driver
# -------------------------------------------------------------------------
def levenberg_marquardt(
    r, J, x,
    lambda0: float = 1e-4,    # initial damping coefficient
    nu0: float = 2.0,         # initial growth factor ν > 1
    nmax: int = 500,          # iteration limit
    tol_abs: float = 1e-6,    # absolute gradient tolerance
    tol_rel: float = 1e-6,    # relative gradient tolerance
    eps: float = 1e-4,        # safeguard tolerance
    Scaling: bool = True     # activate diagonal scaling
):
    """
    Parameters
    ----------
    r, J : callables
        Residual vector and Jacobian.
    x : ndarray
        Initial parameter vector (modified in place).
    lambda0 : float
        Starting value for the damping coefficient λ.
    nu0 : float
        Starting value for ν (should be > 1).
    nmax : int
        Maximum number of iterations.
    tol_abs, tol_rel, eps : float
        Stopping criterion ‖g‖ ≤ min{ tol_rel·‖g₀‖ + tol_abs , eps }.
    Scaling : bool
        If True, uses the diagonal scaling matrix D = diag(max‖J[:,i]‖).

    Returns
    -------
    x : ndarray
        Optimised parameters.
    info : list[list]
        Iteration log  [k, ‖p‖, λ, ‖g‖, ρ].
    """

    # convenience lambdas
    f = lambda z: 0.5 * norm(r(z)) ** 2
    gradf = lambda z: J(z).T @ r(z)

    # initial function/gradient
    fx = f(x)
    g0 = gradf(x)
    gnorm0 = norm(g0,np.inf)

    # stopping tolerance
    tolerance = min(tol_rel * gnorm0 + tol_abs, eps)

    # problem dimensions
    m, n = J(x).shape

    # diagonal scaling matrices
    D = np.eye(n)
    D_inv = np.eye(n)

    # damping and growth parameters
    lam = max(lambda0, 1e-15)
    nu = max(nu0, 1.1)

    # information log
    info = [['iter', '‖p‖', 'λ', '‖g‖', 'ρ', 'x']]
    info.append([0, 'n/a', lam, gnorm0, 'n/a', x.copy() if n <= 2 else None])

    # ------------------------------------------------------------------ loop
    for k in range(1, nmax + 1):
        Jx = J(x)

        # optional diagonal scaling (updates D and D_inv)
        if Scaling:
            for i in range(n):
                D[i, i] = max(D[i, i], norm(Jx[:, i]))
                D_inv[i, i] = 1.0 / D[i, i]
        D2 = D @ D        # used for φ′ in original radius-based algorithm

        g = Jx.T @ r(x)

        # QR decomposition with column pivoting of J(x)
        Q, R, piv = la.qr(Jx, pivoting=True)
        P = np.eye(n)[:, piv]
        rank = np.linalg.matrix_rank(Jx)

        # ---------------------------------------------------- step p
        if lam == 0.0:              # Gauss-Newton step (unique if rank = n)
            if rank == n:
                y = la.solve_triangular(R[:n, :], Q[:, :n].T @ (-r(x)))
                p = P @ y
            else:                   # rank-deficient GN step
                y = np.zeros(n)
                y[:rank] = la.solve_triangular(
                    R[:rank, :rank], Q[:, :rank].T @ (-r(x)))
                p = P @ y
        else:                       # λ > 0  → augmented QR   [R ; √λ D]
            D_lambda = P.T @ (D @ P)                 # permuted scaling
            R_I = np.vstack((R, np.sqrt(lam) * D_lambda))
            R_lambda, Q_lambdaT = _givens_qr(R_I.copy(), n, m)

            Q_lambda = (np.block([[Q, np.zeros((m, n))],
                                  [np.zeros((n, m)), P]])
                         @ Q_lambdaT)

            r_aug = np.append(r(x), np.zeros(n))
            z = la.solve_triangular(R_lambda[:n, :],
                                    Q_lambda[:, :n].T @ (-r_aug))
            p = P @ z

        # ---------------------------------------------------- ratio ρ
        denom = 0.5 * p.T @ (lam * p - g)
        denom = denom if denom > 0 else 1e-20       # safeguard
        fx_trial = f(x + p)
        rho = (fx - fx_trial) / denom

        # ---------------------------------------------------- λ update
        if rho > 0.0:                                # accept step
            x += p
            fx = fx_trial
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:                                        # reject step
            lam *= nu
            nu *= 2.0

        # gradient norm for stopping test & log
        gnorm = norm(gradf(x),np.inf)
        info.append([k, norm(p,np.inf), lam, gnorm, rho, x.copy() if n <= 2 else None])

        if gnorm <= tolerance:
            break

    return x, info
