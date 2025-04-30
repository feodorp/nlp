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


# Function 2: F(x,y) = (1 - x^4)^2 + 100 (y - x^2)^2 + 100 (sin(y^2) - x)^2
def func2(x):
    return (1 - x[0]**4)**2 + 100 * (x[1] - x[0]**2)**2 + 100 * (np.sin(x[1]**2) - x[0])**2

def grad_func2(x):
    dfdx = -4 * x[0]**3 * 2 * (1 - x[0]**4) - 400 * x[0] * (x[1] - x[0]**2) - 200 * (np.sin(x[1]**2) - x[0])
    dfdy = 200 * (x[1] - x[0]**2) + 400 * x[1] * np.cos(x[1]**2) * (np.sin(x[1]**2) - x[0])
    return np.array([dfdx, dfdy])

def hess_func2(x):
    d2fdx2 = 56 * x[0]**6 + 1176 * x[0]**2 - 400 * x[1] + 200
    d2fdxdy = -400 * x[0] - 400 * x[1] * np.cos(x[1]**2)
    d2fdy2 = 200 + 400 * ( (np.cos(x[1]**2) - 2 * x[1]**2 * np.sin(x[1]**2)) * (np.sin(x[1]**2) - x[0]) + 2 * x[1]**2 * np.cos(x[1]**2)**2 )
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

# Function 3: F(x,y) = (2y^2 - x)^2 + 100 (y^2 - x^2)^2
def func3(x):
    return (2 * x[1]**2 - x[0])**2 + 100 * (x[1]**2 - x[0]**2)**2

def grad_func3(x):
    dfdx = -2 * (2 * x[1]**2 - x[0]) - 400 * x[0] * (x[1]**2 - x[0]**2)
    dfdy = 8 * x[1] * (2 * x[1]**2 - x[0]) + 400 * x[1] * (x[1]**2 - x[0]**2)
    return np.array([dfdx, dfdy])

def hess_func3(x):
    d2fdx2 = 2 + 1200 * x[0]**2 - 400 * x[1]**2
    d2fdxdy = -8 * x[1] * (1 + 100 * x[0])
    d2fdy2 = 1248 * x[1]**2 - 400 * x[0]**2 - 8 * x[0]
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

# Function 4: F(x,y) = (1 - sin(x))^2 + 100 (cos(y) - x^2)^2
def func4(x):
    return (1 - np.sin(x[0]))**2 + 100 * (np.cos(x[1]) - x[0]**2)**2

def grad_func4(x):
    dfdx = -2 * np.cos(x[0]) * (1 - np.sin(x[0])) - 400 * x[0] * (np.cos(x[1]) - x[0]**2)
    dfdy = -200 * np.sin(x[1]) * (np.cos(x[1]) - x[0]**2)
    return np.array([dfdx, dfdy])

def hess_func4(x):
    d2fdx2 = 2 * np.cos(x[0])**2 + 2 * np.sin(x[0]) * (1 - np.sin(x[0])) + 800 * x[0]**2 - 400 * (np.cos(x[1]) - x[0]**2)
    d2fdxdy = 400 * x[0] * np.sin(x[1])
    d2fdy2 = -200 * np.cos(x[1]) * (np.cos(x[1]) - x[0]**2) - 200 * np.sin(x[1])**2
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

# Function to plot the optimization path
def plot_path(path, method, func_name, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = globals()[func_name](np.array([X[i, j], Y[i, j]]))

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


    print(f"\n=== Optimization Steps for {method} on {func_name} ===")
    print("{:<6} {:<12} {:<12} {:<12} {:<25}".format("Iter", "||p||", "alpha", "||g||", "Position (x)"))
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

# Test all functions with both line search methods
functions = [
    ('rosenbrock', rosenbrock, grad_rosenbrock, hess_rosenbrock, (-2, 2), (-4, 4)),
    ('func2', func2, grad_func2, hess_func2, (-2, 2), (-4, 4)),
    ('func3', func3, grad_func3, hess_func3, (-2, 2), (-4, 4)),
    ('func4', func4, grad_func4, hess_func4, (-2, 2), (-4, 4))
]

methods = [
    ('Backtracking Steepest Descent', 'steepest', None),
    ('Backtracking SR1', 'SR1', None),
    ('Backtracking BFGS', 'BFGS', None),
    ('Zoom Steepest Descent', 'steepest', None),
    ('Zoom SR1', 'SR1', None),
    ('Zoom BFGS', 'BFGS', None),
]

# Initial point
x0 = np.array([-0.5, -3.0])


for func_name, f, grad_f, hess_f, x_range, y_range in functions:
    for method_name, method, hess_provider in methods:
        hess = hess_provider(functions[functions.index((func_name, f, grad_f, hess_f, x_range, y_range))]) if hess_provider else None
        if method_name.startswith('Backtracking'):
            x_opt, info = line_search_backtracking(x0, f, grad_f, line_search_type='interp', method=method, hess_f=hess)
        else:
            x_opt, info = line_search_zoom(x0, f, grad_f, method=method, hess_f=hess)
        path = [row[4] for row in info[1:] if row[4] is not None]
        print_log(info, method_name, func_name)
        plot_path(path, method_name, func_name, x_range, y_range)
