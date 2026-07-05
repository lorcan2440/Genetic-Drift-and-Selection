import os
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import truncnorm
from matplotlib import pyplot as plt


STYLESHEET_PATH = r"C:\LibsAndApps\Python config files\proplot_style.mplstyle"

if os.path.exists(STYLESHEET_PATH):
    plt.style.use(STYLESHEET_PATH)


def A_drift(x, s):
    '''Return the drift coefficient A(x) = s x(1-x)'''
    return s * x * (1.0 - x)


def D_diff(x, Ne):
    '''Return the diffusion coefficient D(x) = x(1-x)/(2Ne)'''
    return x * (1.0 - x) / (2.0 * Ne)


def D_derivative(x, Ne):
    '''Return the derivative of the diffusion coefficient D'(x) = (1-2x)/(2Ne)'''
    return (1.0 - 2.0 * x) / (2.0 * Ne)


def numerical_derivative(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return a centered finite-difference derivative for vector inputs."""
    h = eps * np.maximum(1.0, np.abs(x))
    return (func(x + h) - func(x - h)) / (2.0 * h)


def solve_fokker_planck_pde_chang_cooper(A: callable, D: callable, Phi0: callable, D_prime: callable = None,
        Nx: int = 500, Nt: int = 5000, tmax: float = 1000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    '''
    Solve the drift-diffusion (Fokker-Planck) PDE with absorbing boundaries at x = 0 and x = 1 
    via the implicit Chang-Cooper scheme:

    ∂φ/∂t = -∂(A(x) φ)/∂x + ∂²(D(x) φ)/∂x², 0 < x < 1, t > 0

    where A(x) is the drift and D(x) is the diffusion coefficient, with absorbing boundaries at x = 0 and x = 1.

    The initial condition φ(x, 0) is provided by the function `Phi0`.

    The function J(x, t) = A(x) φ(x, t) - ∂(D(x) φ(x, t))/∂x is the probability flux, such that
    the PDE can be written as ∂φ/∂t = -∂J/∂x. The absorption probabilities are given by
    P_loss(t) = ∫₀ᵗ J(0, τ) dτ and P_fix(t) = ∫₀ᵗ J(1, τ) dτ, while the interior probability is
    P_interior(t) = ∫₀¹ φ(x, t) dx.

    The function returns the solution φ(x, t) on a grid, as well as these integrals of probability fluxes.
    
    ### Arguments
    - `A` (callable): Drift coefficient function A(x).
    - `D` (callable): Diffusion coefficient function D(x).
    - `Phi0` (callable): Initial condition function φ(x, 0) to be evaluated on the spatial grid.
    - `D_prime` (callable, optional): Derivative of the diffusion coefficient D'(x). 
    If None, it will be computed numerically.
    - `Nx` (int, default = 500): Number of spatial grid points.
    - `Nt` (int, default = 5000): Number of time steps.
    - `tmax` (float, default = 1000.0): Maximum simulation time.
    
    ### Returns
    - `tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]`: Tuple containing the spatial grid `x`, 
    time grid `T`, solution matrix `Phi`, and a dictionary `results` for flux integrals with keys 
    `["P_loss", "P_fix", "P_interior", "P_total", "mass_error"]`.
    
    ### Raises
    - `ValueError`: If `Nx` is less than 3.
    '''    
    x = np.linspace(0.0, 1.0, Nx)
    dx = x[1] - x[0]
    dt = tmax / Nt

    if Nx < 3:
        raise ValueError("Nx must be at least 3 to have interior nodes.")

    Nint = Nx - 2

    # evaluate transport coefficients at cell interfaces for finite-volume fluxes.
    x_half = 0.5 * (x[:-1] + x[1:])
    D_half = D(x_half)
    if D_prime is None:
        B_half = A(x_half) - numerical_derivative(D, x_half)
    else:
        B_half = A(x_half) - D_prime(x_half)

    # Chang-Cooper upwind/exponential fitting weight, enforces correct steady states
    eps = 1e-15
    w = B_half * dx / (D_half + eps)
    delta = np.empty_like(w)
    small = np.abs(w) < 1e-8
    delta[small] = 0.5
    delta[~small] = 1.0 / w[~small] - 1.0 / np.expm1(w[~small])

    # interface flux is linear in neighboring cell values: J_{i+1/2} = alpha*phi_i + beta*phi_{i+1}.
    alpha = B_half * delta + D_half / dx
    beta = B_half * (1.0 - delta) - D_half / dx

    lower = np.zeros(Nint - 1)
    diag = np.zeros(Nint)
    upper = np.zeros(Nint - 1)

    # backward Euler on conservative flux form
    for k in range(Nint):
        i = k + 1
        diag[k] = 1.0 + (dt / dx) * (alpha[i] - beta[i - 1])

        if k > 0:
            lower[k - 1] = -(dt / dx) * alpha[i - 1]

        if k < Nint - 1:
            upper[k] = (dt / dx) * beta[i]

    M = diags(diagonals=[lower, diag, upper], offsets=[-1, 0, 1], format="csc")
    
    T = np.linspace(0.0, tmax, Nt + 1)

    phi = Phi0(x)  # current solution Phi(x, t_n)
    Phi = np.zeros((Nt + 1, Nx))  # full solution array Phi(x, t)
    Phi[0] = phi.copy()  # set initial condition

    P_loss = np.zeros(Nt + 1)
    P_fix = np.zeros(Nt + 1)
    P_interior = np.zeros(Nt + 1)
    P_total = np.zeros(Nt + 1)

    P_interior[0] = np.trapezoid(phi, x)
    P_total[0] = P_interior[0] + P_loss[0] + P_fix[0]

    for n in range(Nt):
        phi_in_old = phi[1:-1]
        phi_in_new = spsolve(M, phi_in_old)

        phi[:] = 0.0
        phi[1:-1] = phi_in_new

        J_left = beta[0] * phi[1]
        J_right = alpha[-1] * phi[-2]

        # boundary flux integrals are the absorbed loss/fixation probabilities.
        P_loss[n + 1] = P_loss[n] + dt * (-J_left)
        P_fix[n + 1] = P_fix[n] + dt * J_right

        P_interior[n + 1] = np.trapezoid(phi, x)
        P_total[n + 1] = P_interior[n + 1] + P_loss[n + 1] + P_fix[n + 1]

        Phi[n + 1] = phi.copy()

    mass_error = np.abs(P_total - 1.0)

    results = {
        "P_loss": P_loss,
        "P_fix": P_fix,
        "P_interior": P_interior,
        "P_total": P_total,
        "mass_error": mass_error,
    }

    return x, T, Phi, results


def solve_kimura_pde(Ne: float = 1000, s: float = 0.01, Nx: int = 500, Nt: int = 5000, tmax: float = 1000.0,
        x0: float = 0.3, sigma0: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Set up and solve Kimura's PDE using the Chang-Cooper scheme."""
    A = lambda x: A_drift(x, s)
    D = lambda x: D_diff(x, Ne)

    dD_dx = lambda x: D_derivative(x, Ne)

    a = (0.0 - x0) / sigma0
    b = (1.0 - x0) / sigma0
    Phi0 = lambda x: truncnorm.pdf(x, a, b, loc=x0, scale=sigma0)  # initial condition Phi(x, 0)

    return solve_fokker_planck_pde_chang_cooper(A, D, Phi0, D_prime=dD_dx, Nx=Nx, Nt=Nt, tmax=tmax)


def calc_mean_absorption_times(Ne: float = 1000, s: float = 0.01, 
        x0_arr: np.ndarray | None = None, Nx: int = 2000) -> np.ndarray:
    '''
    Compute mean absorption (loss-or-fixation) time from

    T''(x) + 2 N_e s T'(x) = -2 N_e / (x(1 - x)), 0 < x < 1

    where the boundary conditions are T(0) = T(1) = 0.

    A direct BVP solve on [0, 1] is numerically problematic because the RHS is
    singular at x = 0 and x = 1. We therefore solve the ODE on interior nodes
    with centered finite differences and then interpolate onto x0_arr.
    '''
    if x0_arr is None:
        x0_arr = np.linspace(0.0, 1.0, 101)
    else:
        x0_arr = np.asarray(x0_arr, dtype=float)

    if x0_arr.ndim != 1:
        raise ValueError("x0_arr must be a 1D array of initial frequencies.")
    if np.any((x0_arr < 0.0) | (x0_arr > 1.0)):
        raise ValueError("x0_arr values must lie in [0, 1].")
    if Ne <= 0.0:
        raise ValueError("Ne must be positive.")
    if Nx < 5:
        raise ValueError("Nx must be at least 5.")

    x = np.linspace(0.0, 1.0, Nx)
    dx = x[1] - x[0]
    x_in = x[1:-1]
    n_in = x_in.size

    drift = 2.0 * Ne * s
    rhs = -2.0 * Ne / (x_in * (1.0 - x_in))

    # tridiagonal matrix for finite difference scheme
    lower = np.full(n_in - 1, 1.0 / dx**2 - drift / (2.0 * dx))
    diag = np.full(n_in, -2.0 / dx**2)
    upper = np.full(n_in - 1, 1.0 / dx**2 + drift / (2.0 * dx))

    A = diags([lower, diag, upper], offsets=[-1, 0, 1], format="csc")
    T_in = spsolve(A, rhs)

    T_grid = np.zeros_like(x)
    T_grid[1:-1] = T_in

    T_means = np.interp(x0_arr, x, T_grid)
    T_means[x0_arr <= 0.0] = 0.0
    T_means[x0_arr >= 1.0] = 0.0

    return T_means


if __name__ == "__main__":

    Ne = 1000
    s = 0.01
    x, T, Phi, results = solve_kimura_pde(Ne=Ne, s=s)

    print(
        "Max conservation error "
        f"|P_interior + P_loss + P_fix - 1| = {np.max(results['mass_error']):.3e}"
    )
    print(
        "Final "
        f"P_interior={results['P_interior'][-1]:.6f}, "
        f"P_loss={results['P_loss'][-1]:.6f}, "
        f"P_fix={results['P_fix'][-1]:.6f}"
    )
