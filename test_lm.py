import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from lm import levenberg_marquardt          # the file you just saved

# Function 1: Rosenbrock
def rosenbrock_residual(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([10.0 * (y - x**2), 1.0 - x])

def rosenbrock_jacobian(v: np.ndarray) -> np.ndarray:
    x, _ = v
    return np.array([[-20.0 * x, 10.0], [-1.0, 0.0]])

# Function 2: F(x,y) = (1 - x^4)^2 + 100 (y - x^2)^2 + 100 (sin(y^2) - x)^2
def func2_residual(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([1 - x**4, 10 * (y - x**2), 10 * (np.sin(y**2) - x)])

def func2_jacobian(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([[-4 * x**3, 0], [-20 * x, 10], [-10, 20 * y * np.cos(y**2)]])

# Function 3: F(x,y) = (2y^2 - x)^2 + 100 (y^2 - x^2)^2
def func3_residual(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([2 * y**2 - x, 10 * (y**2 - x**2)])

def func3_jacobian(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([[-1, 4 * y], [-20 * x, 20 * y]])

# Function 4: F(x,y) = (1 - sin(x))^2 + 100 (cos(y) - x^2)^2
def func4_residual(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([1 - np.sin(x), 10 * (np.cos(y) - x**2)])

def func4_jacobian(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([[-np.cos(x), 0], [-20 * x, -10 * np.sin(y)]])

# Function to plot the optimization path
def plot_path(path, method, func_name, func, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    plt.figure()
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'ro-', label=f'Optimization path ({len(path)} steps)')
    plt.title(f'Optimization Path for {method} on {func_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'optimization_path_{method.lower().replace(" ", "_")}_{func_name}.pdf')
    plt.close()

# Function to print iteration log
def print_log(info, method, func_name):
    def print_row(row):
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


    print(f"\n=== Optimization Steps for {method} on {func_name} ===")
    print("{:<6} {:<12} {:<12} {:<12} {:<12} {:<25}".format("Iter", "||p||", "lambda", "||g||", "rho", "Position (x)"))
    print("-" * 67)
    nsteps = len(info[1:])
    if nsteps < 50:
        for row in info[1:]:
            print_row(row)
        print("=" * 67)
    else:
        for row in info[1:5]:
            print_row(row)
        print("." * 20)
        for row in info[-20:]:
            print_row(row)
        print("=" * 67)

# Initial point
x0 = np.array([-0.5, -3])

# Test all functions with Levenberg-Marquardt
functions = [
    ('rosenbrock', rosenbrock_residual, rosenbrock_jacobian, (-2, 2), (-4, 4)),
    ('func2', func2_residual, func2_jacobian, (-2, 2), (-4, 4)),
    ('func3', func3_residual, func3_jacobian, (-2, 2), (-4, 4)),
    ('func4', func4_residual, func4_jacobian, (-2, 2), (-4, 4))
]


for func_name, residual, jacobian, x_range, y_range in functions:
    x_opt, info = levenberg_marquardt(residual, jacobian, x0,
                                      lambda0=1e-3,  # start damping
                                      nu0=2.0,       # Nielsen factor
                                      Scaling=True)  # activate column-scaling
    path = [row[-1] for row in info[1:] if row[-1] is not None]
    func = lambda x: 0.5 * norm(residual(x)) ** 2
    print_log(info, "LM", func_name)
    plot_path(path, "LM", func_name, func, x_range, y_range)
