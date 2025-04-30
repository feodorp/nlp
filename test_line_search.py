import numpy as np
import matplotlib.pyplot as plt
from linesearch import line_search_backtracking, line_search_zoom

# Define the Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradient of the Rosenbrock function
def grad_rosenbrock(x):
    dfdx = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])

# Hessian of the Rosenbrock function
def hess_rosenbrock(x):
    d2fdx2 = 1200 * x[0]**2 - 400 * x[1] + 2
    d2fdxdy = -400 * x[0]
    d2fdy2 = 200
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

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
    print("{:<6} {:<12} {:<12} {:<12} {:<25}".format("Iter", "||p||", "alpha", "||g||", "Position (x)"))
    print("-" * 67)
    for row in info[1:]:
        iter_num, p_norm, alpha, g_norm, x = row
        # Format p_norm: string if 'n/a', scientific notation if float
        p_norm_str = f"{p_norm:<12}" if isinstance(p_norm, str) else f"{p_norm:<12.4e}"
        # Format alpha: string if 'n/a', scientific notation if float
        alpha_str = f"{alpha:<12}" if isinstance(alpha, str) else f"{alpha:<12.4e}"
        # g_norm is always a float, so format in scientific notation
        g_norm_str = f"{g_norm:<12.4e}"
        # Format position based on whether x is None or an array
        if x is not None:
            pos_str = f"[{x[0]:.4e}, {x[1]:.4e}]"
            print(f"{iter_num:<6d} {p_norm_str} {alpha_str} {g_norm_str} {pos_str:<25}")
        else:
            print(f"{iter_num:<6d} {p_norm_str} {alpha_str} {g_norm_str} (n > 2)")
    print("=" * 67)

# Initial point
x0 = np.array([-0.5, 1.5])

# Test backtracking with Steepest Descent
x_opt, info = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, line_search_type='interp', method='steepest', hess_f=None)
path_sd = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Backtracking Steepest Descent")
plot_path(path_sd, "Backtracking Steepest Descent")

# Test backtracking with SR1
x_opt, info = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, line_search_type='interp', method='SR1', hess_f=None)
path_bfgs = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Backtracking SR1")
plot_path(path_bfgs, "Backtracking SR1")

# Test backtracking with BFGS
x_opt, info = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, line_search_type='interp', method='BFGS', hess_f=None)
path_bfgs = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Backtracking BFGS")
plot_path(path_bfgs, "Backtracking BFGS")

# Test backtracking with Hessian
x_opt, info = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, line_search_type='interp', method='hessian', hess_f=hess_rosenbrock)
path_hess = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Backtracking Hessian")
plot_path(path_hess, "Backtracking Hessian")

# Test zoom with Steepest Descent
x_opt, info = line_search_zoom(x0, rosenbrock, grad_rosenbrock, method='steepest', hess_f=None)
path_sd = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Zoom Steepest Descent")
plot_path(path_sd, "Zoom Steepest Descent")

# Test zoom with SR1
x_opt, info = line_search_zoom(x0, rosenbrock, grad_rosenbrock, method='BFGS', hess_f=None)
path_bfgs = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Zoom SR1")
plot_path(path_bfgs, "Zoom SR1")

# Test zoom with BFGS
x_opt, info = line_search_zoom(x0, rosenbrock, grad_rosenbrock, method='BFGS', hess_f=None)
path_bfgs = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Zoom BFGS")
plot_path(path_bfgs, "Zoom BFGS")

# Test zoom with Hessian
x_opt, info = line_search_zoom(x0, rosenbrock, grad_rosenbrock, method='hessian', hess_f=hess_rosenbrock)
path_hess = [row[4] for row in info[1:] if row[4] is not None]
print_log(info, "Zoom Hessian")
