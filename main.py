import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from solve_pde import solve_kimura_pde, calc_mean_absorption_times
from solve_sde import solve_kimura_sde, calc_absorption_fractions


STYLESHEET_PATH = r"C:\LibsAndApps\Python config files\proplot_style.mplstyle"

if os.path.exists(STYLESHEET_PATH):
    plt.style.use(STYLESHEET_PATH)


def create_plots(Ne: float = 100, s: float = 0.003, Nx: int = 1000, Nt: int = 10000, tmax: float = 500.0, 
        x0: float = 0.3, sigma0: float = 0.08, n_paths: int = 500, seed: int = 12345, save_folder: str = None) -> tuple[dict, np.ndarray]:
    '''
    Create plots for Kimura's PDE and SDE simulations.
    
    ### Arguments
    - `Ne` (int, default = 100): Effective population size
    - `s` (float, default = 0.003): Selection coefficient
    - `Nx` (int, default = 1000): Number of spatial grid points for PDE
    - `Nt` (int, default = 10000): Number of time steps for PDE
    - `tmax` (float, default = 500.0): Maximum simulation time
    - `x0` (float, default = 0.3): Initial allele frequency
    - `sigma0` (float, default = 0.08): Standard deviation of initial allele frequency
    - `n_paths` (int, default = 500): Number of SDE trajectories to simulate
    - `seed` (int, default = 12345): Random seed for reproducibility
    - `save_folder` (str, default = None): Folder to save the figures. If None, the figures are not saved.
    
    ### Returns
    - `pde_results` (dict): Results from the PDE simulation
    - `X_sde` (np.ndarray): Simulated SDE trajectories
    '''

    # solve the PDE to get the probability density function and absorption probabilities
    x_pde, T_pde, Phi, pde_results = solve_kimura_pde(Ne=Ne, s=s, Nx=Nx, Nt=Nt, tmax=tmax, x0=x0, sigma0=sigma0)

    # solve the SDE to get sampled trajectories
    T_sde, X_sde = solve_kimura_sde(Ne=Ne, s=s, Nt=Nt, tmax=tmax, x0=x0, sigma0=sigma0, n_paths=n_paths, seed=seed)

    # get mean absorption time according to theory
    x0_arr = np.linspace(0, 1, 100)
    T_means = calc_mean_absorption_times(Ne=Ne, s=s, x0_arr=x0_arr)
    T_x0 = np.interp(x0, x0_arr, T_means)

    # get mean absorption time according to empirical SDE trajectories
    # NOTE: this is biassed to be a slight underestimate since we only consider trajectories 
    # that have been absorbed already by the end of the simulation
    absorbed_mask = (X_sde <= 0.0) | (X_sde >= 1.0)
    absorbed_any = np.any(absorbed_mask, axis=0)
    first_abs_idx = np.argmax(absorbed_mask, axis=0)
    t_abs_empirical = np.mean(T_sde[first_abs_idx[absorbed_any]]) if np.any(absorbed_any) else np.nan

    # get eventual fixation probabilities according to theory
    gamma = 2.0 * Ne * s
    if np.isclose(gamma, 0.0):
        p_fix_theory = x0
    else:
        p_fix_theory = (1.0 - np.exp(-gamma * x0)) / (1.0 - np.exp(-gamma))
    p_loss_theory = 1.0 - p_fix_theory

    fig1 = plt.figure(figsize=(17, 9), constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig1, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.08, hspace=0.08)

    # ax1: PDF solution of Kimura's PDE
    ax1 = fig1.add_subplot(gs[:, 0], projection="3d")
    X_pde_grid, T_pde_grid = np.meshgrid(x_pde, T_pde)
    surf = ax1.plot_surface(X_pde_grid, T_pde_grid, Phi, cmap="RdPu", linewidth=0)
    ax1.set_box_aspect((1.3, 1.3, 0.55))  # 3D data box uses more of its subplot area
    ax1.set_xlabel("Allele frequency, $ x $")
    ax1.set_ylabel("Time in generations, $ t $")
    ax1.set_zlabel(r"Probability density, $\phi(x,t)$")
    ax1.view_init(elev=30, azim=-75, roll=0)
    ax1.set_title("Numerical Solution to Kimura's Drift-Diffusion PDE (Chang-Cooper scheme)")
    fig1.colorbar(surf, ax=ax1, shrink=0.78, aspect=20, pad=0.03)

    # ax2: sampled SDE trajectories
    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.plot(T_sde, X_sde, color="tab:blue", alpha=0.03, linewidth=0.7)
    ax2.axvline(t_abs_empirical, color="tab:green", ls="--", lw=2.0, alpha=0.95,
        label=rf"Empirical mean absorption time = {t_abs_empirical:.2f}")
    ax2.axvline(T_x0, color="tab:red", ls="--", lw=2.0, alpha=0.95,
        label=rf"Theoretical mean absorption time = {T_x0:.2f}")
    ax2.set_ylabel("Allele frequency, $ x $")
    ax2.set_title("Trajectories of Kimura's SDE (Milstein's method)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    # ax3: absorption probabilities from PDE and theoretical values
    ax3 = fig1.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(T_pde, pde_results["P_loss"], label=r"$P_{\mathrm{loss}}(t)$", lw=2)
    ax3.plot(T_pde, pde_results["P_fix"], label=r"$P_{\mathrm{fix}}(t)$", lw=2)
    ax3.plot(T_pde, pde_results["P_interior"], label=r"$P_{\mathrm{interior}}(t)$", lw=1.7, ls="--")
    ax3.plot(T_pde, pde_results["P_total"], label=r"$P_{\mathrm{total}}(t)$", lw=1.5, ls=":")
    ax3.axhline(p_loss_theory, color="tab:blue", ls=(0, (5, 3)), lw=1.8, alpha=0.9,
        label=rf"$P_{{\mathrm{{loss}}}}^{{\mathrm{{theory}}}}={p_loss_theory:.3f}$")
    ax3.axhline(p_fix_theory, color="tab:orange", ls=(0, (5, 3)), lw=1.8, alpha=0.9,
        label=rf"$P_{{\mathrm{{fix}}}}^{{\mathrm{{theory}}}}={p_fix_theory:.3f}$")
    ax3.set_xlabel("Time in generations, $ t $")
    ax3.set_ylabel("Probability")
    ax3.set_title("Theoretical Probabilities of Fixation and Loss")
    ax3.set_xlim(0, tmax)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper right", ncol=2)

    fig1.suptitle(f"Genetic drift with weak natural selection: $N_e={Ne}$, $s={s}$, $x_0={x0}$, $\\sigma_0={sigma0}$")

    if save_folder is not None:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        fig1.savefig(os.path.join(save_folder, "kimura_pde_sde_comparison.svg"), dpi=300, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x0_arr, T_means, color="blue")
    ax.plot(x0, T_x0, marker="o", color="red", markersize=6, label=f"Initial $x_0={x0}$, $T(x_0)={T_x0:.2f}$")
    ax.legend(loc="lower center")
    ax.set_xlabel("Initial allele frequency, $x_0$")
    ax.set_ylabel("Mean absorption time, $T(x_0)$")
    ax.set_title(f"Mean absorption (fixation or loss) time for Kimura's PDE: N_e = {Ne}, s = {s}")

    if save_folder is not None:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        fig2.savefig(os.path.join(save_folder, "kimura_mean_absorption_times.svg"), dpi=300, bbox_inches="tight")

    return pde_results, X_sde


if __name__ == "__main__":

    pde_results, X_sde = create_plots(Ne=20, s=0.05, x0=0.2, sigma0=0.03, n_paths=500, tmax=100, save_folder="output")

    print(
        "PDE mass error "
        f"|P_interior + P_loss + P_fix - 1| = {np.max(pde_results['mass_error']):.3e}"
    )

    final_loss, final_fix, final_interior = calc_absorption_fractions(X_sde)

    print(
        f"SDE final fractions: loss={final_loss:.4f}, "
        f"fix={final_fix:.4f}, "
        f"interior={final_interior:.4f}"
    )
    plt.show()
