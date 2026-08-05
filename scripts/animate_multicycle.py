"""Multi-cycle stigmergic-convergence animation.

Renders several full stigmergic cycles of Algorithm 1 as an mp4:

  Cycle 1 -> Cycle 2 -> ... -> Cycle N -> final side-by-side summary

Within each cycle the video shows, in sequence:

  Stage 1  forward pass         (trajectories A -> B under current phi)
  Stage 2  adjoint sweep        (Gamma(s) built backward, Gamma(1) = 0)
  Stage 3  controlled backward  (B -> A under omega_ctrl = -Gamma/gamma)
  Stage 4  pheromone update     (backward trajectories -> new phi)

The pheromone field is carried across cycles, so viewers see the bundle
tighten toward the Snell-optimal path as the loop converges.

Usage:
  python scripts/animate_multicycle.py [num_cycles] [out.mp4]
Defaults: 5 cycles, figures/demo/multicycle_convergence.mp4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.apic import (  # noqa: E402
    downsample_recent_weighted_trajectories,
    integrate_costate,
    make_init_fn,
    simulate_controlled_backward_pass,
    simulate_forward_batch,
)

NUM_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "figures/demo/multicycle_convergence.mp4"
)

W, H, DPI = 1600, 900, 100
FPS = 30
STAGE1_FRAMES = 60   # 2.0 s
STAGE2_FRAMES = 60
STAGE3_FRAMES = 60
STAGE4_FRAMES = 30   # 1.0 s
TITLE_FRAMES = 45    # 1.5 s per section title

POINT_A = jnp.array([0.0, 0.0])
POINT_B = jnp.array([1.0, 1.0])
BOUNDARY_Y = 0.5
BASE_NU = 1.0
JUMP_NU = 10.0

BATCH = 32
NUM_STEPS = 1520
DT = 0.001
PHER_SIGMA = 0.05


# ---------------------------------------------------------------------------
# Snell-optimal reference path
# ---------------------------------------------------------------------------

def snell_optimal_x():
    """Fermat-principle interface crossing x."""
    xa, ya = 0.0, 0.0
    xb, yb = 1.0, 1.0
    def cost(xc):
        return BASE_NU * np.hypot(xc - xa, BOUNDARY_Y - ya) \
             + JUMP_NU * np.hypot(xb - xc, yb - BOUNDARY_Y)
    r = minimize_scalar(cost, bounds=(0.0, 1.0), method='bounded')
    return float(r.x)


SNELL_X = snell_optimal_x()
SNELL_PATH = (np.array([0.0, SNELL_X, 1.0]),
              np.array([0.0, BOUNDARY_Y, 1.0]))


# ---------------------------------------------------------------------------
# Run all cycles up front and cache per-cycle arrays
# ---------------------------------------------------------------------------

def run_all_cycles(num_cycles: int, seed: int = 0):
    """Run the stigmergic loop and return the per-cycle stage arrays.

    Returns
    -------
    cycles : list of dicts with keys
        'fwd'        : (T, B, 3) forward-pass trajectory
        'lam'        : (T, B, 3) costate along the forward trajectory
        'bwd'        : (T, B, 3) controlled backward trajectory
        'pher_in'    : (Nin, 2) pheromone field entering this cycle
        'pher_out'   : (Nout, 2) pheromone field after Stage 4
    """
    key = random.PRNGKey(seed)
    init_fn = make_init_fn(POINT_A, POINT_B, BATCH)

    pher_pts = jnp.empty((0, 2))
    pher_wts = jnp.empty((0,))
    all_backward = []
    cycles = []

    for c in range(num_cycles):
        key, k_init, k_fwd, k_bwd, k_phen = random.split(key, 5)
        init_states = init_fn(k_init)
        pher_in = np.asarray(pher_pts)

        fwd = simulate_forward_batch(
            k_fwd, init_states, pher_pts, pher_wts,
            PHER_SIGMA, DT, NUM_STEPS, POINT_B,
        )
        lam = integrate_costate(fwd, pher_pts, pher_wts, PHER_SIGMA, DT)

        final = fwd[-1]
        theta_flip = jnp.mod(final[:, 2] + jnp.pi, 2 * jnp.pi)
        final_rot = jnp.stack([final[:, 0], final[:, 1], theta_flip], axis=1)

        bwd = simulate_controlled_backward_pass(
            k_bwd, final_rot, lam, pher_pts, pher_wts, PHER_SIGMA,
            NUM_STEPS, DT, POINT_A,
        )
        all_backward.append(bwd)

        pher_pts, pher_wts = downsample_recent_weighted_trajectories(
            all_backward_trajectories=all_backward,
            num_trajs_to_sample=200,
            weight=1.0, key=k_phen,
        )

        cycles.append({
            'fwd': np.asarray(fwd),
            'lam': np.asarray(lam),
            'bwd': np.asarray(bwd),
            'pher_in': pher_in,
            'pher_out': np.asarray(pher_pts),
        })
    return cycles


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fig_to_frame(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _kde_field(points: np.ndarray, X, Y, sigma=PHER_SIGMA,
               max_points=5000):
    if points.shape[0] == 0:
        return np.zeros_like(X)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)
    field = np.zeros_like(X)
    step = 500
    for s in range(0, points.shape[0], step):
        chunk = points[s:s + step]
        dx = X[None] - chunk[:, 0][:, None, None]
        dy = Y[None] - chunk[:, 1][:, None, None]
        field += np.exp(-(dx ** 2 + dy ** 2) * inv_2s2).sum(axis=0)
    return field


def _traj_axes(ax, title, subtitle=None):
    ax.add_patch(Rectangle((-0.1, BOUNDARY_Y), 1.3, 1.3 - BOUNDARY_Y,
                           facecolor='#e9e9ef', edgecolor='none', zorder=0))
    ax.axhline(BOUNDARY_Y, color='k', lw=0.8, zorder=4)
    ax.plot(*POINT_A, marker='o', mec='k', mfc='#1f77b4', ms=8, zorder=5)
    ax.plot(*POINT_B, marker='o', mec='k', mfc='#2ca02c', ms=8, zorder=5)
    ax.plot(SNELL_PATH[0], SNELL_PATH[1], color='#e07a1f', lw=1.6, ls='--',
            zorder=3, label='Snell optimum')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(title, fontsize=16, weight='bold')
    if subtitle:
        ax.text(0.5, 1.05, subtitle, transform=ax.transAxes,
                ha='center', fontsize=11, color='#555')


def _pher_layer(ax, pher_pts, X, Y):
    field = _kde_field(pher_pts, X, Y)
    if field.max() == 0:
        return
    vmax = float(np.percentile(field, 99)) or 1.0
    ax.imshow(field, origin='lower',
              extent=[X.min(), X.max(), Y.min(), Y.max()],
              cmap='Purples', vmin=0, vmax=vmax, alpha=0.7,
              aspect='auto', zorder=1)


def _to_uniform(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to exactly (H, W, 3).  imageio's ffmpeg backend
    requires every frame in a movie to have the same shape; matplotlib's
    canvas sometimes returns off-by-one pixel dimensions between figures
    with different layouts, so we normalize here."""
    from PIL import Image as _PILImage
    if frame.shape[:2] == (H, W):
        return frame
    im = _PILImage.fromarray(frame).resize((W, H), _PILImage.LANCZOS)
    return np.asarray(im)


def section_title(writer, title, subtitle, seconds=1.5,
                  title_size=40, subtitle_size=18, bg=(20, 25, 45)):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=None)
    ax = fig.add_axes([0.05, 0, 0.9, 1])
    ax.set_facecolor(tuple(v / 255 for v in bg))
    ax.set_axis_off()
    fig.patch.set_facecolor(tuple(v / 255 for v in bg))
    ax.text(0.5, 0.60, title, ha='center', va='center',
            fontsize=title_size, color='white', weight='bold', wrap=True)
    ax.text(0.5, 0.44, subtitle, ha='center', va='center',
            fontsize=subtitle_size, color='#b5bce0', wrap=True)
    frame = _to_uniform(_fig_to_frame(fig))
    plt.close(fig)
    for _ in range(int(FPS * seconds)):
        writer.append_data(frame)


def animate_forward(writer, fwd, pher_in, cycle_idx, num_cycles):
    T, B, _ = fwd.shape
    steps = np.linspace(2, T, STAGE1_FRAMES).astype(int)
    grid = 80
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    for t in steps:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout='constrained')
        ax = fig.add_subplot(111)
        _pher_layer(ax, pher_in, X, Y)
        for b in range(B):
            ax.plot(fwd[:t, b, 0], fwd[:t, b, 1],
                    color='#c1272d', alpha=0.35, lw=0.6, zorder=2)
        ax.plot(fwd[t - 1, :, 0], fwd[t - 1, :, 1], 'o',
                color='#c1272d', ms=4, alpha=0.9, zorder=3)
        s_norm = (t - 1) / (T - 1)
        _traj_axes(ax,
                   f"Cycle {cycle_idx + 1}/{num_cycles}  ·  "
                   "Stage 1: forward pass  (Eq. 4 + Eq. 2)",
                   subtitle=f"integrate A -> B under current pheromone "
                            f"field  ·  $s = {s_norm:.2f}$")
        ax.legend(loc='lower right', fontsize=10)
        writer.append_data(_to_uniform(_fig_to_frame(fig)))
        plt.close(fig)


def animate_adjoint(writer, lam, cycle_idx, num_cycles):
    T, B, _ = lam.shape
    s = np.linspace(0.0, 1.0, T)
    starts = np.linspace(T, 2, STAGE2_FRAMES).astype(int)
    ymin = float(lam[:, :, 2].min()) * 1.1
    ymax = float(lam[:, :, 2].max()) * 1.1
    for s_start in starts:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout='constrained')
        ax = fig.add_subplot(111)
        for b in range(B):
            ax.plot(s[s_start - 1:], lam[s_start - 1:, b, 2],
                    color='#7b3f99', alpha=0.35, lw=0.7)
        ax.scatter([1.0], [0.0], color='k', zorder=10, s=60,
                   label=r'terminal:  $\Gamma(1)=0$')
        ax.axhline(0.0, color='k', ls=':', lw=0.6, alpha=0.6)
        ax.axvline(s[s_start - 1], color='#7b3f99', lw=1.2, alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'arc length  $s \in [0,1]$')
        ax.set_ylabel(r'heading costate  $\Gamma(s)$')
        ax.set_title(
            f"Cycle {cycle_idx + 1}/{num_cycles}  ·  "
            "Stage 2: adjoint sweep  (Eq. 6, backward in $s$)",
            fontsize=15, weight='bold',
        )
        ax.text(0.5, 1.03,
                f"sweep frontier  $s = {s[s_start - 1]:.2f}$  "
                r"$\to$  $s = 0$",
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=11, color='#555')
        ax.legend(loc='upper left', fontsize=10)
        writer.append_data(_to_uniform(_fig_to_frame(fig)))
        plt.close(fig)


def animate_backward(writer, fwd, bwd, pher_in, cycle_idx, num_cycles):
    T, B, _ = bwd.shape
    steps = np.linspace(2, T, STAGE3_FRAMES).astype(int)
    grid = 80
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    for t in steps:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout='constrained')
        ax = fig.add_subplot(111)
        _pher_layer(ax, pher_in, X, Y)
        for b in range(B):
            ax.plot(fwd[:, b, 0], fwd[:, b, 1],
                    color='#c1272d', alpha=0.10, lw=0.4, zorder=2)
        for b in range(B):
            ax.plot(bwd[:t, b, 0], bwd[:t, b, 1],
                    color='#1f5aa8', alpha=0.4, lw=0.7, zorder=3)
        ax.plot(bwd[t - 1, :, 0], bwd[t - 1, :, 1], 'o',
                color='#1f5aa8', ms=4, alpha=0.9, zorder=4)
        _traj_axes(ax,
                   f"Cycle {cycle_idx + 1}/{num_cycles}  ·  "
                   "Stage 3: controlled backward  (Eq. 7)",
                   subtitle=r"$\tilde\omega^{\mathrm{ctrl}} = "
                            r"-\Gamma(s)/\gamma$,  "
                            r"$\gamma = \beta D_\theta$  ·  "
                            f"step {t}/{T}")
        ax.legend(loc='lower right', fontsize=10)
        writer.append_data(_to_uniform(_fig_to_frame(fig)))
        plt.close(fig)


def animate_pheromone_update(writer, pher_in, pher_out, cycle_idx,
                             num_cycles):
    grid = 80
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    field_in = _kde_field(pher_in, X, Y)
    field_out = _kde_field(pher_out, X, Y)
    for k in range(STAGE4_FRAMES):
        alpha = (k + 1) / STAGE4_FRAMES
        field = (1 - alpha) * field_in + alpha * field_out
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout='constrained')
        ax = fig.add_subplot(111)
        vmax = float(np.percentile(field, 99)) or 1.0
        ax.imshow(field, origin='lower',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  cmap='Purples', vmin=0, vmax=vmax, alpha=0.85,
                  aspect='auto', zorder=1)
        _traj_axes(ax,
                   f"Cycle {cycle_idx + 1}/{num_cycles}  ·  "
                   "Stage 4: pheromone update",
                   subtitle="deposit backward trajectories into $\\phi$")
        ax.legend(loc='lower right', fontsize=10)
        writer.append_data(_to_uniform(_fig_to_frame(fig)))
        plt.close(fig)


def animate_convergence_summary(writer, cycles, seconds=6.0):
    """Final panel: forward trajectories of all cycles side by side."""
    N = len(cycles)
    grid = 80
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, N, figsize=(W / DPI, H / DPI),
                             dpi=DPI, sharex=True, sharey=True,
                             layout='constrained')
    if N == 1:
        axes = [axes]
    for i, (ax, cyc) in enumerate(zip(axes, cycles)):
        _pher_layer(ax, cyc['pher_out'], X, Y)
        fwd = cyc['fwd']
        for b in range(fwd.shape[1]):
            ax.plot(fwd[:, b, 0], fwd[:, b, 1],
                    color='#c1272d', alpha=0.25, lw=0.5, zorder=2)
        ax.plot(SNELL_PATH[0], SNELL_PATH[1], color='#e07a1f', lw=1.4,
                ls='--', zorder=3)
        ax.add_patch(Rectangle((-0.1, BOUNDARY_Y), 1.3, 1.3 - BOUNDARY_Y,
                               facecolor='#e9e9ef', edgecolor='none',
                               zorder=0))
        ax.axhline(BOUNDARY_Y, color='k', lw=0.6, zorder=4)
        ax.plot(*POINT_A, marker='o', mec='k', mfc='#1f77b4', ms=6,
                zorder=5)
        ax.plot(*POINT_B, marker='o', mec='k', mfc='#2ca02c', ms=6,
                zorder=5)
        ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f"Cycle {i + 1}", fontsize=12, weight='bold')
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
    fig.suptitle(
        "Convergence: forward trajectories sharpen around the Snell-optimal path "
        "(orange dashed) as the stigmergic loop iterates",
        fontsize=14, y=1.02,
    )
    frame = _to_uniform(_fig_to_frame(fig))
    plt.close(fig)
    for _ in range(int(FPS * seconds)):
        writer.append_data(frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {NUM_CYCLES} stigmergic cycles ...")
    cycles = run_all_cycles(NUM_CYCLES)
    print(f"  done.  Each cycle: forward {cycles[0]['fwd'].shape}, "
          f"backward {cycles[0]['bwd'].shape}.")

    writer = imageio.get_writer(
        OUT, fps=FPS, codec='libx264', quality=8, macro_block_size=1,
    )
    try:
        section_title(
            writer,
            "Stigmergic optimal transport",
            f"Algorithm 1 (APIC): {NUM_CYCLES} cycles to convergence",
            seconds=2.0, title_size=34, subtitle_size=18,
        )

        for i, cyc in enumerate(cycles):
            section_title(
                writer,
                f"Cycle {i + 1} of {NUM_CYCLES}",
                "Stage 1 -> Stage 2 -> Stage 3 -> Stage 4",
                seconds=1.2, title_size=32,
            )
            animate_forward(writer, cyc['fwd'], cyc['pher_in'], i, NUM_CYCLES)
            animate_adjoint(writer, cyc['lam'], i, NUM_CYCLES)
            animate_backward(writer, cyc['fwd'], cyc['bwd'], cyc['pher_in'],
                             i, NUM_CYCLES)
            animate_pheromone_update(writer, cyc['pher_in'], cyc['pher_out'],
                                     i, NUM_CYCLES)

        section_title(
            writer,
            "Convergence",
            "trajectory ensemble tightens around the Snell-optimal path",
            seconds=1.5, title_size=34, subtitle_size=16,
        )
        animate_convergence_summary(writer, cycles, seconds=6.0)
    finally:
        writer.close()

    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
