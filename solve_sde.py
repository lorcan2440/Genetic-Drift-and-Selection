import numpy as np
from scipy.stats import truncnorm

"""Kimura SDE solver with Milstein discretization and absorbing boundaries.

This module exposes a callable function `simulate_kimura_sde_milstein` and does
not perform plotting at import/runtime.
"""


def a_drift(x, s):
    return s * x * (1.0 - x)


def b_diffusion(x, Ne):
    return np.sqrt(np.maximum(x * (1.0 - x), 0.0) / Ne)


def bb_prime(x, Ne):
    return (1.0 - 2.0 * x) / (2.0 * Ne)


def simulate_kimura_sde_milstein(Ne=1000, s=0.01, Nt=5000, tmax=1000.0, x0=0.3, sigma0=0.0, n_paths=1500, seed=12345):
    """Simulate Kimura SDE trajectories with Milstein's method.

    Returns a dictionary with the time grid, trajectories, and summary fractions.
    """
    T = np.linspace(0.0, tmax, Nt + 1)
    dt = T[1] - T[0]

    rng = np.random.default_rng(seed)

    X = np.empty((Nt + 1, n_paths), dtype=float)
    if sigma0 == 0.0:
        X[0, :] = x0
    else:
        a = (0.0 - x0) / sigma0
        b = (1.0 - x0) / sigma0
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
                + 0.5 * bb_prime(xa, Ne) * (dW * dW - dt)
            )

            hit_left = xa_next <= 0.0
            hit_right = xa_next >= 1.0

            xa_next = np.where(hit_left, 0.0, xa_next)
            xa_next = np.where(hit_right, 1.0, xa_next)

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
