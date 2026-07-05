import numpy as np
from scipy.stats import truncnorm


def A_drift(x, s):
    # Deterministic selection term in the Wright-Fisher diffusion limit.
    return s * x * (1.0 - x)


def b_diffusion(x, Ne):
    # Diffusion amplitude satisfies b(x)^2 = 2D(x) = x(1-x)/Ne.
    return np.sqrt(np.maximum(x * (1.0 - x), 0.0) / Ne)


def bb_prime(x, Ne):
    # Product b b' gives the scalar Milstein correction coefficient.
    return (1.0 - 2.0 * x) / (2.0 * Ne)


def numerical_derivative(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return a centered finite-difference derivative for vector inputs."""
    h = eps * np.maximum(1.0, np.abs(x))
    return (func(x + h) - func(x - h)) / (2.0 * h)


def solve_ito_sde_milstein(A: callable, b: callable, X0: callable, b_prime: callable = None,
        Nt: int = 5000, tmax: float = 1000.0, n_paths: int = 1500, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    '''
    Solve the 1D drift-diffusion SDE with absorbing boundaries via the Milstein scheme:

    dx = A(x) dt + b(x) dW_t, 0 < x < 1, t > 0

    where A(x) is the drift and b(x) is the diffusion amplitude, with absorbing boundaries at x = 0 and x = 1,
    and W_t is the standard Wiener process (Brownian motion).

    The initial condition x(0) for all trajectories is provided by the function `X0`.

    ### Arguments
    - `A` (callable): Drift coefficient function A(x).
    - `b` (callable): Diffusion amplitude function b(x).
    - `X0` (callable): Initial-condition sampler `X0(n_paths, rng) -> np.ndarray`.
    - `b_prime` (callable, optional): Derivative b'(x). If None, computed numerically.
    - `Nt` (int, default = 5000): Number of time steps.
    - `tmax` (float, default = 1000.0): Maximum simulation time.
    - `n_paths` (int, default = 1500): Number of SDE trajectories to simulate.
    - `seed` (int, default = 12345): Random seed for reproducibility.
    '''
    T = np.linspace(0.0, tmax, Nt + 1)
    dt = T[1] - T[0]

    rng = np.random.default_rng(seed)

    X = np.empty((Nt + 1, n_paths), dtype=float)
    x_init = np.asarray(X0(n_paths, rng), dtype=float)
    if x_init.shape != (n_paths,):
        raise ValueError("X0(n_paths, rng) must return a 1D array with length n_paths.")
    X[0, :] = x_init

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
                + A(xa) * dt
                + b(xa) * dW
                # Milstein strong-order-1 correction for multiplicative Ito noise
                + 0.5
                * b(xa)
                * (numerical_derivative(b, xa) if b_prime is None else b_prime(xa))
                * (dW * dW - dt)
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


def solve_kimura_sde(Ne: float = 1000, s: float = 0.01, Nt: int = 5000, tmax: float = 1000.0,
        x0: float = 0.3, sigma0: float = 0.0, n_paths: int = 1500, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """Set up and solve Kimura's SDE using the Milstein scheme."""
    A = lambda x: A_drift(x, s)
    b = lambda x: b_diffusion(x, Ne)
    bb_dx = lambda x: bb_prime(x, Ne)

    if sigma0 == 0.0:
        X0 = lambda n_paths, rng: np.full(n_paths, x0, dtype=float)
    else:
        lower = (0.0 - x0) / sigma0
        upper = (1.0 - x0) / sigma0
        X0 = lambda n_paths, rng: truncnorm.rvs(
            lower, upper, loc=x0, scale=sigma0, size=n_paths, random_state=rng
        )

    return solve_ito_sde_milstein(A=A, b=b, X0=X0, b_prime=bb_dx, Nt=Nt, tmax=tmax, n_paths=n_paths, seed=seed)


def calc_absorption_fractions(X):
    """Calculate the fractions of paths absorbed at 0, absorbed at 1, and still in the interior."""

    final_loss = np.mean(X[-1, :] == 0.0)
    final_fix = np.mean(X[-1, :] == 1.0)
    final_interior = np.mean((X[-1, :] > 0.0) & (X[-1, :] < 1.0))

    return final_loss, final_fix, final_interior


if __name__ == "__main__":
    Ne = 1000
    s = 0.01
    T, X = solve_kimura_sde(Ne=Ne, s=s)
    final_loss, final_fix, final_interior = calc_absorption_fractions(X)
    tmax = T[-1]
    print(
        f"Final fractions at t = {tmax:g}: "
        f"loss = {final_loss:.4f}, "
        f"fix = {final_fix:.4f}, "
        f"interior = {final_interior:.4f}"
    )
