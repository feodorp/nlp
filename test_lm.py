import numpy as np
import matplotlib.pyplot as plt
from lm import levenberg_marquardt          # the file you just saved

# -------------------------------------------------------------------------
#  Build the problem in residual form
#     f(x, y) = 100 (y – x²)² + (1 – x)²
#  Write   f = ½‖r‖²   with two residuals
#         r₁(x,y) = √100 · (y – x²)
#         r₂(x,y) = (1 – x)
# -------------------------------------------------------------------------
def residual(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([10.0 * (y - x**2), 1.0 - x])   #  √100 = 10

def jacobian(v: np.ndarray) -> np.ndarray:
    x, _ = v
    return np.array([[-20.0 * x, 10.0],          # ∂r₁/∂x , ∂r₁/∂y
                     [-1.0,        0.0]])        # ∂r₂/∂x , ∂r₂/∂y

# Function to plot the optimization path
def plot_path(path, method):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'ro-', label=f'Optimization path ({len(path)} steps)')
    plt.title(f'Optimization Path for {method}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'optimization_path_{method.lower().replace(" ", "_")}.pdf')

# Function to print iteration log
def print_log(info, method):
    print(f"\n=== Optimization Steps for {method} ===")
    print("{:<6} {:<12} {:<12} {:<12} {:<12} {:<25}".format("Iter", "||p||", "lambda", "||g||", "rho", "Position (x)"))
    print("-" * 67)
    for row in info[1:]:
        iter_num, p_norm, lam, g_norm, rho, x = row
        # Format p_norm: string if 'n/a', scientific notation if float
        p_norm_str = f"{p_norm:<12}" if isinstance(p_norm, str) else f"{p_norm:<12.4e}"
        # Format alpha: string if 'n/a', scientific notation if float
        lambda_str = f"{lam:<12}" if isinstance(lam, str) else f"{lam:<12.4e}"
        # g_norm is always a float, so format in scientific notation
        g_norm_str = f"{g_norm:<12.4e}"
        # Format rho: string if 'n/a', scientific notation if float
        rho_str = f"{rho:<12}" if isinstance(rho, str) else f"{rho:<12.4e}"
        # Format position based on whether x is None or an array
        if x is not None:
            pos_str = f"[{x[0]:.4e}, {x[1]:.4e}]"
            print(f"{iter_num:<6d} {p_norm_str} {lambda_str} {g_norm_str} {rho_str} {pos_str:<25}")
        else:
            print(f"{iter_num:<6d} {p_norm_str} {lambda_str} {g_norm_str} {rho_str} (n > 2)")
    print("=" * 67)

# Initial point
x0 = np.array([-0.5, 1.5])

x_opt, info = levenberg_marquardt(residual, jacobian, x0,
                                 lambda0=1e-3,  # start damping
                                 nu0=2.0,       # Nielsen factor
                                 Scaling=True)  # activate column-scaling

path_hess = [row[-1] for row in info[1:] if row[-1] is not None]
print_log(info, "LM")
plot_path(path_hess, "LM")
