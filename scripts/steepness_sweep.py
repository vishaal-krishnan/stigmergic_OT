"""
Steepness sweep for the SI figure: heterogeneity scale exploration.

Sweeps the sigmoid steepness of the refractive index transition, which controls
the environmental heterogeneity scale |grad log nu|^{-1}. For each steepness
value, runs the full APIC loop (Algorithm 1) and records the final
forward-pass trajectory, the converged pheromone field, and the trajectory's
interface-crossing position.

Produces three figures:
  - trajectories_vs_steepness.pdf     (2x5 grid of trajectory + pheromone panels)
  - x_crossing_vs_steepness.pdf       (crossing position vs. control strength)
  - traversal_time_vs_steepness.pdf   (normalized traversal time vs. control strength)

Used in the SI of "Stigmergic optimal transport".
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.apic import run_apic_loop, make_init_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(OUT_DIR, exist_ok=True)

POINT_A = jnp.array([0.0, 0.0])
POINT_B = jnp.array([1.0, 1.0])
BASE_NU = 1.0
JUMP_NU = 10.0
BOUNDARY_Y = 0.5

STEEP_VALS = np.linspace(1.0, 100.0, 10)
NUM_CYCLES = 5
BATCH_SIZE = 32
NUM_STEPS = 1520
DT = 0.001
SIGMA_NOISE = 1.0
PHER_SIGMA = 0.05
SEED = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_nu_fn(steep, base=BASE_NU, jump=JUMP_NU, boundary=BOUNDARY_Y):
    """Sigmoid refractive index field with adjustable steepness."""
    def nu_fn(x, y):
        s = 1.0 / (1.0 + jnp.exp(-steep * (y - boundary)))
        return base * (1 - s) + jump * s
    return nu_fn


def control_strength(steep, base=BASE_NU, jump=JUMP_NU):
    """Dimensionless control strength ell_0 * |grad log nu| at the interface."""
    nu_mid = 0.5 * (base + jump)
    log_nu_grad_y = steep * (jump - base) / (4 * nu_mid)
    ell_0 = float(jnp.linalg.norm(POINT_B - POINT_A))
    return log_nu_grad_y * ell_0


def compute_interface_crossing_x(traj, y_interface=BOUNDARY_Y):
    """Linear-interp x at which trajectory crosses y = y_interface."""
    y, x = traj[:, 1], traj[:, 0]
    for i in range(len(y) - 1):
        if (y[i] < y_interface) and (y[i + 1] >= y_interface):
            t = (y_interface - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return np.nan


def snell_optimal_x(point_a, point_b, nu1, nu2, y_interface=BOUNDARY_Y):
    """Snell-optimal interface crossing x-coordinate via Fermat's principle."""
    xa, ya = float(point_a[0]), float(point_a[1])
    xb, yb = float(point_b[0]), float(point_b[1])

    def optical_path(xc):
        L1 = np.sqrt((xc - xa) ** 2 + (y_interface - ya) ** 2)
        L2 = np.sqrt((xb - xc) ** 2 + (yb - y_interface) ** 2)
        return nu1 * L1 + nu2 * L2

    result = minimize_scalar(optical_path, bounds=(0.0, 1.0), method='bounded')
    return float(result.x), float(optical_path(result.x))


def trajectory_traversal_time(traj, nu_fn):
    """Sum of nu * ds along the trajectory."""
    x, y = traj[:, 0], traj[:, 1]
    nu_vals = np.array([float(nu_fn(xi, yi)) for xi, yi in zip(x, y)])
    dx = np.diff(x); dy = np.diff(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    nu_mid = 0.5 * (nu_vals[:-1] + nu_vals[1:])
    return float(np.sum(nu_mid * ds))


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

def run_sweep():
    init_fn = make_init_fn(POINT_A, POINT_B, BATCH_SIZE)
    forward_params = {"dt": DT, "num_steps": NUM_STEPS, "sigma_noise": SIGMA_NOISE}
    backward_params = forward_params

    representative_trajs = []
    representative_pheromones = []
    x_crossings = []
    traversal_times = []

    for k, steep in enumerate(STEEP_VALS):
        print(f"[{k+1}/{len(STEEP_VALS)}] steepness = {steep:.2f}")
        nu_fn = make_nu_fn(steep)

        all_forward, _, pher_pts, _ = run_apic_loop(
            num_cycles=NUM_CYCLES,
            key=random.PRNGKey(SEED),
            init_fn=init_fn,
            point_a=POINT_A,
            point_b=POINT_B,
            pher_sigma=PHER_SIGMA,
            forward_params=forward_params,
            backward_params=backward_params,
            nu_fn=nu_fn,
        )

        last_cycle = np.array(all_forward[-1])  # (T, B, 3)
        representative_trajs.append(last_cycle[:, 0])  # one agent for the panel

        # Store a subsample of pheromone points for visualization only
        pher_np = np.array(pher_pts)
        if pher_np.shape[0] > 20000:
            rng = np.random.default_rng(SEED + k)
            pher_np = pher_np[rng.choice(pher_np.shape[0], size=20000, replace=False)]
        representative_pheromones.append(pher_np)

        # Average convergence metrics over the batch (skip non-crossing trajectories)
        batch_crossings = [compute_interface_crossing_x(last_cycle[:, b])
                           for b in range(last_cycle.shape[1])]
        batch_times = [trajectory_traversal_time(last_cycle[:, b], nu_fn)
                       for b in range(last_cycle.shape[1])]
        x_crossings.append(np.nanmean(batch_crossings))
        traversal_times.append(np.nanmean(batch_times))

        # Free JAX arrays explicitly to avoid accumulation between sweeps
        del all_forward, pher_pts, last_cycle
        jax.clear_caches()

    return representative_trajs, representative_pheromones, x_crossings, traversal_times


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def pheromone_field_grid(pher_pts, X, Y, sigma=PHER_SIGMA,
                         chunk_size=2000, max_points=20000, rng_seed=0):
    """Gaussian KDE on a grid. Chunked to bound memory.

    If the trajectory cloud has > max_points, subsample uniformly so the
    figure stays representative without OOM. Each chunk allocates
    chunk_size * H * W floats ~ 2000*100*100*8 = 160MB transiently.
    """
    pts = np.asarray(pher_pts)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    field = np.zeros_like(X)
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start:start + chunk_size]
        dx = X[None, :, :] - chunk[:, 0][:, None, None]
        dy = Y[None, :, :] - chunk[:, 1][:, None, None]
        field += np.exp(-(dx ** 2 + dy ** 2) * inv_2s2).sum(axis=0)
    return field


def plot_trajectory_panels(trajs, pheros, control_strengths, out_path):
    grid = 100
    x = np.linspace(-0.1, 1.1, grid)
    y = np.linspace(-0.1, 1.1, grid)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    cmap = ListedColormap(["white"] + plt.get_cmap('YlOrRd')(np.linspace(0.2, 1.0, 256)).tolist())
    norm = LogNorm(vmin=1e-3, vmax=1.0)

    for i, (traj, pher, cs) in enumerate(zip(trajs, pheros, control_strengths)):
        ax = axes[i]
        field = pheromone_field_grid(pher, X, Y)
        ax.contourf(X, Y, field, levels=50, cmap=cmap, norm=norm)
        ax.plot(traj[:, 0], traj[:, 1], color='black', lw=1.5)
        ax.scatter([POINT_A[0]], [POINT_A[1]], color='red', s=30, zorder=3)
        ax.scatter([POINT_B[0]], [POINT_B[1]], color='blue', s=30, zorder=3)
        ax.axhline(BOUNDARY_Y, color='gray', linestyle='--', lw=1)
        ax.set_title(rf"$\ell_0 |\nabla \log \nu| = {cs:.2f}$", fontsize=10)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Trajectories and pheromone fields vs.\\ heterogeneity scale", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_crossing_vs_steepness(control_strengths, x_crossings, out_path):
    x_opt, _ = snell_optimal_x(POINT_A, POINT_B, BASE_NU, JUMP_NU)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(control_strengths, x_crossings, 'o-', color='black', label='APIC trajectory')
    ax.axhline(x_opt, color='C3', ls='--', label=f"Snell optimum $x={x_opt:.3f}$")
    ax.set_xlabel(r'$\ell_0 |\nabla \log \nu|$ (control strength)')
    ax.set_ylabel(r'$x$ at interface crossing ($y=0.5$)')
    ax.legend()
    ax.grid(True, ls='--', lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_traversal_time_vs_steepness(control_strengths, traversal_times, out_path):
    _, T_opt = snell_optimal_x(POINT_A, POINT_B, BASE_NU, JUMP_NU)
    ratios = np.array(traversal_times) / T_opt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(control_strengths, ratios, 'o-', color='black')
    ax.axhline(1.0, color='C3', ls='--', label=r'Snell optimum')
    ax.set_xlabel(r'$\ell_0 |\nabla \log \nu|$ (control strength)')
    ax.set_ylabel(r'$\langle T \rangle / T^*$')
    ax.legend()
    ax.grid(True, ls='--', lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running steepness sweep ...")
    trajs, pheros, x_crossings, traversal_times = run_sweep()
    control_strengths = [control_strength(s) for s in STEEP_VALS]

    print("\nGenerating figures ...")
    plot_trajectory_panels(
        trajs, pheros, control_strengths,
        os.path.join(OUT_DIR, 'trajectories_vs_steepness.pdf'),
    )
    plot_crossing_vs_steepness(
        control_strengths, x_crossings,
        os.path.join(OUT_DIR, 'x_crossing_vs_steepness.pdf'),
    )
    plot_traversal_time_vs_steepness(
        control_strengths, traversal_times,
        os.path.join(OUT_DIR, 'traversal_time_vs_steepness.pdf'),
    )

    # Save raw data alongside
    np.savez(
        os.path.join(OUT_DIR, 'steepness_sweep_data.npz'),
        steepness=STEEP_VALS,
        control_strengths=np.array(control_strengths),
        x_crossings=np.array(x_crossings),
        traversal_times=np.array(traversal_times),
    )
    print(f"  wrote {os.path.join(OUT_DIR, 'steepness_sweep_data.npz')}")


if __name__ == '__main__':
    main()
