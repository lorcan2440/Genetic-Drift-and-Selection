import numpy as np
from scipy.stats import truncnorm


def a_drift(x, s):
    # Deterministic selection term in the Wright-Fisher diffusion limit.
    return s * x * (1.0 - x)


def b_diffusion(x, Ne):
    # Diffusion amplitude satisfies b(x)^2 = 2D(x) = x(1-x)/Ne.
    return np.sqrt(np.maximum(x * (1.0 - x), 0.0) / Ne)


def bb_prime(x, Ne):
    # Product b*b' gives the scalar Milstein correction coefficient.
    return (1.0 - 2.0 * x) / (2.0 * Ne)


def simulate_kimura_sde_milstein(Ne: float = 1000, s: float = 0.01, Nt: int = 5000, tmax: float = 1000.0, 
        x0: float = 0.3, sigma0: float = 0.0, n_paths: int = 1500, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    '''
    Solve Kimura's drift-diffusion SDE with absorbing boundaries via the Milstein scheme:

    dx = A(x) dt + b(x) dW_t, 0 < x < 1, t > 0

    where A(x) = s x(1-x) and b(x) = sqrt(x(1-x)/Ne), with absorbing boundaries at x = 0 and x = 1,
    and W_t is the standard Wiener process (Brownian motion).

    The initial condition x(0) is drawn from a truncated Gaussian centered at x0 with standard deviation sigma0,
    truncated to the interval [0, 1].

    ### Arguments
    - `Ne` (float, default = 1000): Effective population size.
    - `s` (float, default = 0.01): Selection coefficient.
    - `Nt` (int, default = 5000): Number of time steps.
    - `tmax` (float, default = 1000.0): Maximum simulation time.
    - `x0` (float, default = 0.3): Initial allele frequency (mean).
    - `sigma0` (float, default = 0.0): Standard deviation of the initial Gaussian distribution.
    - `n_paths` (int, default = 1500): Number of SDE trajectories to simulate.
    - `seed` (int, default = 12345): Random seed for reproducibility.
    '''
    T = np.linspace(0.0, tmax, Nt + 1)
    dt = T[1] - T[0]

    rng = np.random.default_rng(seed)

    X = np.empty((Nt + 1, n_paths), dtype=float)
    if sigma0 == 0.0:
        X[0, :] = x0  # deterministic initial condition
    else:
        a = (0.0 - x0) / sigma0
        b = (1.0 - x0) / sigma0
        # initial condition drawn from a truncated Gaussian over allele frequency in [0, 1]
        X[0, :] = truncnorm.rvs(a, b, loc=x0, scale=sigma0, size=n_paths, random_state=rng)

    absorbed = np.zeros(n_paths, dtype=bool)

    for n in range(Nt):
        x_now = X[n, :]
        x_next = x_now.copy()

        active = ~absorbed
        if np.any(active):
            xa = x_now[active]
            dW = np.sqrt(dt) * rng.standard_normal(xa.size)

            xa_next = (
                xa
                + a_drift(xa, s) * dt
                + b_diffusion(xa, Ne) * dW
                # Milstein strong-order-1 correction for multiplicative Ito noise
                + 0.5 * bb_prime(xa, Ne) * (dW * dW - dt)
            )

            hit_left = xa_next <= 0.0
            hit_right = xa_next >= 1.0

            xa_next = np.where(hit_left, 0.0, xa_next)
            xa_next = np.where(hit_right, 1.0, xa_next)

            # project to absorbing states once trajectories cross the boundaries
            x_next[active] = xa_next
            absorbed[active] = hit_left | hit_right

        X[n + 1, :] = x_next

    return T, X


def calc_absorption_fractions(X):
    """Calculate the fractions of paths absorbed at 0, absorbed at 1, and still in the interior."""

    final_loss = np.mean(X[-1, :] == 0.0)
    final_fix = np.mean(X[-1, :] == 1.0)
    final_interior = np.mean((X[-1, :] > 0.0) & (X[-1, :] < 1.0))

    return final_loss, final_fix, final_interior


if __name__ == "__main__":
    T, X = simulate_kimura_sde_milstein()
    final_loss, final_fix, final_interior = calc_absorption_fractions(X)
    tmax = T[-1]
    print(
        f"Final fractions at t = {tmax:g}: "
        f"loss = {final_loss:.4f}, "
        f"fix = {final_fix:.4f}, "
        f"interior = {final_interior:.4f}"
    )
