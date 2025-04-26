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
    plt.plot(path[:, 0], path[:, 1], 'ro-', label=f'Optimization path {len(path)} steps')
    plt.title(f'Optimization Path for {method}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'optimization_path_{method.lower().replace(" ", "_")}.pdf')

# -------------------------------------------------------------------------
#  Initial point and solver call
# -------------------------------------------------------------------------
x0 = np.array([-0.75, 2.5])          # same start as in many references

x_opt, log, path = levenberg_marquardt(residual, jacobian, x0,
                                 lambda0=1e-3,  # start damping
                                 nu0=2.0,       # Nielsen factor
                                 Scaling=True)  # activate column-scaling

print(len(path))
plot_path(path,"LM")
print("Optimised point :", x_opt)
print("Rosenbrock value:", 100*(x_opt[1]-x_opt[0]**2)**2 + (1-x_opt[0])**2)
print("Iterations run  :", len(log)-1)          # first row in log is header
