import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import truncnorm


def A_drift(x, s):
    '''Return the drift coefficient A(x) = s x(1-x)'''
    return s * x * (1.0 - x)


def D_diff(x, Ne):
    '''Return the diffusion coefficient D(x) = x(1-x)/(2Ne)'''
    return x * (1.0 - x) / (2.0 * Ne)


def D_prime(x, Ne):
    '''Return the derivative of the diffusion coefficient D'(x) = (1-2x)/(2Ne)'''
    return (1.0 - 2.0 * x) / (2.0 * Ne)


def B_eff(x, s, Ne):
    '''Return the effective drift coefficient B(x) = A(x) - D'(x)'''
    # Rewrite J = A phi - (D phi)_x as J = B phi - D phi_x for Chang-Cooper.
    return A_drift(x, s) - D_prime(x, Ne)


def solve_kimura_pde_chang_cooper(Ne: float = 1000, s: float = 0.01, Nx: int = 500, Nt: int = 5000, tmax: float = 1000.0, 
        x0: float = 0.3, sigma0: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    '''
    Solve Kimura's drift-diffusion PDE with absorbing boundaries via the implicit Chang-Cooper scheme:

    ∂φ/∂t = -∂(A(x) φ)/∂x + ∂²(D(x) φ)/∂x², 0 < x < 1, t > 0

    where A(x) = s x(1-x) and D(x) = x(1-x)/(2Ne), with absorbing boundaries at x = 0 and x = 1.

    The initial condition φ(x, 0) is a Gaussian centered at x0 with standard deviation sigma0 
    (truncated to [0, 1]).

    The function J(x, t) = A(x) φ(x, t) - ∂(D(x) φ(x, t))/∂x is the probability flux, such that
    the PDE can be written as ∂φ/∂t = -∂J/∂x. The absorption probabilities are given by
    P_loss(t) = ∫₀ᵗ J(0, τ) dτ and P_fix(t) = ∫₀ᵗ J(1, τ) dτ, while the interior probability is
    P_interior(t) = ∫₀¹ φ(x, t) dx.

    The function returns the solution φ(x, t) on a grid, as well as these integrals of probability fluxes.
    
    ### Arguments
    - `Ne` (float, default = 1000): Effective population size.
    - `s` (float, default = 0.01): Selection coefficient.
    - `Nx` (int, default = 500): Number of spatial grid points.
    - `Nt` (int, default = 5000): Number of time steps.
    - `tmax` (float, default = 1000.0): Maximum simulation time.
    - `x0` (float, default = 0.3): Initial allele frequency.
    - `sigma0` (float, default = 0.03): Standard deviation of the initial Gaussian distribution.
    
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
    a = (0.0 - x0) / sigma0
    b = (1.0 - x0) / sigma0
    phi = truncnorm.pdf(x, a, b, loc=x0, scale=sigma0)  # initial condition

    x_half = 0.5 * (x[:-1] + x[1:])
    # Evaluate transport coefficients at cell interfaces for finite-volume fluxes.
    D_half = D_diff(x_half, Ne)
    B_half = B_eff(x_half, s, Ne)

    eps = 1e-15
    w = B_half * dx / (D_half + eps)

    delta = np.empty_like(w)
    small = np.abs(w) < 1e-8
    # Chang-Cooper upwind/exponential fitting weight enforces correct steady states.
    delta[small] = 0.5
    delta[~small] = 1.0 / w[~small] - 1.0 / np.expm1(w[~small])

    # Interface flux is linear in neighboring cell values: J_{i+1/2} = alpha*phi_i + beta*phi_{i+1}.
    alpha = B_half * delta + D_half / dx
    beta = B_half * (1.0 - delta) - D_half / dx

    lower = np.zeros(Nint - 1)
    diag = np.zeros(Nint)
    upper = np.zeros(Nint - 1)

    for k in range(Nint):
        i = k + 1
        diag[k] = 1.0 + (dt / dx) * (alpha[i] - beta[i - 1])

        if k > 0:
            lower[k - 1] = -(dt / dx) * alpha[i - 1]

        if k < Nint - 1:
            upper[k] = (dt / dx) * beta[i]

    M = diags(diagonals=[lower, diag, upper], offsets=[-1, 0, 1], format="csc")
    # Backward Euler on conservative flux form yields an M-matrix-like implicit update.

    T = np.linspace(0.0, tmax, Nt + 1)

    Phi = np.zeros((Nt + 1, Nx))
    Phi[0] = phi.copy()

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

        # Boundary flux integrals are exactly the absorbed loss/fixation probabilities.
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


if __name__ == "__main__":

    x, T, Phi, results = solve_kimura_pde_chang_cooper()

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