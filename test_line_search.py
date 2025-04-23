import numpy as np
import matplotlib.pyplot as plt
from optimization import line_search_backtracking

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
    plt.plot(path[:, 0], path[:, 1], 'ro-', label=f'Optimization path {path.size} steps')
    plt.title(f'Optimization Path for {method}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'optimization_path_{method.lower().replace(" ", "_")}.pdf')

# Initial point
x0 = np.array([-0.5, 2.5])

# Test with Steepest Descent
path_sd = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, method='steepest', hess_f=None, return_path=True)
plot_path(path_sd, 'Steepest Descent')

# Test with BFGS
path_bfgs = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, method='bfgs', hess_f=None, return_path=True)
plot_path(path_bfgs, 'BFGS')

# Test with Hessian
path_hess = line_search_backtracking(x0, rosenbrock, grad_rosenbrock, method='hessian', hess_f=hess_rosenbrock, return_path=True)
plot_path(path_hess, 'Hessian')
