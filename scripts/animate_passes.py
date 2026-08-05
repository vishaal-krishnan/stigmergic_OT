"""Animated mp4 visualizing the three passes of one APIC cycle.

Section 1 — Forward pass: trajectories drawn progressively, head moving A -> B.
Section 2 — Adjoint sweep: lambda_theta(s) drawn right-to-left, starting from
            the terminal condition lambda(T)=0 and accumulating backward.
Section 3 — Controlled backward pass: trajectories drawn B -> A, with the
            forward bundle shown faded underneath for reference.
Final frame: three-panel summary identical to figures/demo/forward_backward_panels.png.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.figure import Figure

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.apic import (
    integrate_costate,
    make_init_fn,
    simulate_controlled_backward_pass,
    simulate_forward_batch,
    smooth_piecewise_nu,
)

OUT = Path("figures/demo/forward_backward_animated.mp4")
W, H, DPI = 1600, 900, 100
FPS = 30


def run_cycle():
    batch_size = 32
    num_steps = 1520
    dt = 0.001
    sigma_noise = 1.0
    pher_sigma = 0.05
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([1.0, 1.0])

    key = random.PRNGKey(0)
    k_init, k_fwd, k_bwd = random.split(key, 3)
    init_fn = make_init_fn(point_a, point_b, batch_size)
    init_states = init_fn(k_init)
    pher_pts = jnp.empty((0, 2))
    pher_wts = jnp.empty((0,))

    fwd = simulate_forward_batch(
        k_fwd, init_states, pher_pts, pher_wts,
        pher_sigma, dt, num_steps, sigma_noise, point_b,
    )
    lam = integrate_costate(fwd, smooth_piecewise_nu, dt)
    final_states = fwd[-1]
    theta_flipped = jnp.mod(final_states[:, 2] + jnp.pi, 2 * jnp.pi)
    final_rot = jnp.stack([final_states[:, 0], final_states[:, 1], theta_flipped], axis=1)
    bwd = simulate_controlled_backward_pass(
        k_bwd, final_rot, lam, pher_pts, pher_wts, pher_sigma,
        num_steps, dt, sigma_noise, point_a,
    )
    return (
        np.asarray(fwd),
        np.asarray(lam),
        np.asarray(bwd),
        np.asarray(point_a),
        np.asarray(point_b),
        dt,
    )


def fig_to_frame(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def setup_trajectory_axes(ax, point_a, point_b, title, subtitle):
    ax.axhline(0.5, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.plot(*point_a, "o", color="C0", markersize=10,
            label=r"source  $s=0$")
    ax.plot(*point_b, "o", color="C2", markersize=10,
            label=r"target  $s=1$")
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\tilde X$")
    ax.set_ylabel(r"$\tilde Y$")
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=16, weight="bold")
    ax.legend(loc="lower right", fontsize=11)


def write_section_title(writer, title, subtitle, seconds=1.5,
                        title_size=36, subtitle_size=16):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#141a30")
    # leave 8% horizontal margin on each side so long titles never clip
    ax = fig.add_axes([0.08, 0.0, 0.84, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.60, title, ha="center", va="center",
            fontsize=title_size, color="white", weight="bold", wrap=True)
    ax.text(0.5, 0.42, subtitle, ha="center", va="center",
            fontsize=subtitle_size, color="#b5bce0", wrap=True)
    frame = fig_to_frame(fig)
    plt.close(fig)
    for _ in range(int(FPS * seconds)):
        writer.append_data(frame)


def animate_forward(writer, fwd, point_a, point_b, n_frames=90):
    T, B, _ = fwd.shape
    steps = np.linspace(2, T, n_frames).astype(int)
    for t in steps:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout="constrained")
        ax = fig.add_subplot(111)
        for b in range(B):
            ax.plot(fwd[:t, b, 0], fwd[:t, b, 1], color="C3", alpha=0.3, lw=0.6)
        ax.plot(fwd[t - 1, :, 0], fwd[t - 1, :, 1], "o", color="C3",
                markersize=4, alpha=0.9)
        s_norm = (t - 1) / (T - 1)
        setup_trajectory_axes(
            ax, point_a, point_b,
            "Step 1-2: Forward pass  (simulate_forward_batch)",
            r"reparametrized Langevin (Eq. 4) with trail-following "
            r"$\tilde\omega^{\mathrm{tf}}$ (Eq. 2)"
            f"\narc length  $s = {s_norm:.2f} \\in [0,1]$",
        )
        ax.text(0.02, 0.97, "src/apic.py:72-104", transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="#88a"))
        writer.append_data(fig_to_frame(fig))
        plt.close(fig)


def animate_adjoint(writer, lam, dt, n_frames=90):
    T, B, _ = lam.shape
    s_norm = np.linspace(0.0, 1.0, T)
    starts = np.linspace(T, 2, n_frames).astype(int)
    ymin = float(lam[:, :, 2].min()) * 1.1
    ymax = float(lam[:, :, 2].max()) * 1.1
    for s_start in starts:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout="constrained")
        ax = fig.add_subplot(111)
        for b in range(B):
            ax.plot(s_norm[s_start - 1:], lam[s_start - 1:, b, 2],
                    color="C4", alpha=0.35, lw=0.7)
        ax.scatter([1.0], [0.0], color="k", zorder=10, s=60,
                   label=r"terminal:  $\Gamma(1)=0$")
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.6)
        ax.axvline(s_norm[s_start - 1], color="C4", lw=1.2, alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"arc length  $s \in [0,1]$")
        ax.set_ylabel(r"heading costate  $\Gamma(s)$")
        ax.set_title(
            "Step 3: Backward adjoint sweep  (integrate_costate)\n"
            r"Eq. 6:  $\varepsilon_\theta\, \Gamma'(s) = "
            r"\nu\, \partial_{\theta\theta}E\,\Gamma + "
            r"\mu\cdot(-\sin\tilde\Theta, \cos\tilde\Theta)^\top$"
            "\n"
            f"sweep frontier  $s = {s_norm[s_start - 1]:.2f}$  "
            r"$\to$  $s = 0$",
            fontsize=13, weight="bold",
        )
        ax.text(0.02, 0.97, "src/apic.py:111-133", transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="#88a"))
        ax.legend(loc="upper left", fontsize=11)
        writer.append_data(fig_to_frame(fig))
        plt.close(fig)


def animate_backward(writer, fwd, bwd, point_a, point_b, n_frames=90):
    T, B, _ = bwd.shape
    steps = np.linspace(2, T, n_frames).astype(int)
    for t in steps:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI,
                          layout="constrained")
        ax = fig.add_subplot(111)
        # forward bundle, faded
        for b in range(B):
            ax.plot(fwd[:, b, 0], fwd[:, b, 1], color="C3", alpha=0.10, lw=0.4)
        # backward, growing
        for b in range(B):
            ax.plot(bwd[:t, b, 0], bwd[:t, b, 1], color="C0", alpha=0.4, lw=0.7)
        ax.plot(bwd[t - 1, :, 0], bwd[t - 1, :, 1], "o", color="C0",
                markersize=4, alpha=0.9)
        s_norm = (t - 1) / (T - 1)
        setup_trajectory_axes(
            ax, point_a, point_b,
            "Step 4: Controlled backward pass  (simulate_controlled_backward_pass)",
            r"Eq. 7:  $\tilde\omega^{\mathrm{ctrl}}(s) = -\Gamma(s)/\gamma$, "
            r"$\gamma = \beta D_\theta$"
            f"\nreturn from target ($s=1$) toward source ($s=0$)  ·   "
            f"$s = {s_norm:.2f}$",
        )
        ax.text(0.02, 0.97, "src/apic.py:140-175", transform=ax.transAxes,
                fontsize=10, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="#88a"))
        writer.append_data(fig_to_frame(fig))
        plt.close(fig)


def final_summary(writer, fwd, lam, bwd, point_a, point_b, dt, seconds=4.0):
    fig, axes = plt.subplots(1, 3, figsize=(W / DPI, H / DPI), dpi=DPI,
                              layout="constrained")

    ax = axes[0]
    for b in range(fwd.shape[1]):
        ax.plot(fwd[:, b, 0], fwd[:, b, 1], color="C3", alpha=0.3, lw=0.5)
    setup_trajectory_axes(ax, point_a, point_b,
                          "Forward pass",
                          "Eqs. 4, 2")

    ax = axes[1]
    s = np.linspace(0.0, 1.0, lam.shape[0])
    for b in range(lam.shape[1]):
        ax.plot(s, lam[:, b, 2], color="C4", alpha=0.3, lw=0.6)
    ax.axhline(0.0, color="k", ls=":", lw=0.6)
    ax.scatter([s[-1]], [0.0], color="k", zorder=5,
               label=r"$\Gamma(1)=0$")
    ax.set_xlabel(r"$s \in [0,1]$")
    ax.set_ylabel(r"$\Gamma(s)$")
    ax.set_title("Adjoint sweep\n(Eq. 6, backward in $s$)",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper left", fontsize=10)

    ax = axes[2]
    for b in range(fwd.shape[1]):
        ax.plot(fwd[:, b, 0], fwd[:, b, 1], color="C3", alpha=0.15, lw=0.5)
    for b in range(bwd.shape[1]):
        ax.plot(bwd[:, b, 0], bwd[:, b, 1], color="C0", alpha=0.4, lw=0.5)
    setup_trajectory_axes(ax, point_a, point_b,
                          "Controlled backward",
                          r"$\tilde\omega^{\mathrm{ctrl}} = -\Gamma/\gamma$ (Eq. 7)")

    frame = fig_to_frame(fig)
    plt.close(fig)
    for _ in range(int(FPS * seconds)):
        writer.append_data(frame)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Running one APIC cycle to collect trajectories...")
    fwd, lam, bwd, A, B, dt = run_cycle()
    print(f"  forward:  {fwd.shape}    A->B endpoint err = "
          f"{np.linalg.norm(fwd[-1, :, :2].mean(0) - B):.4f}")
    print(f"  adjoint:  {lam.shape}    max|lambda(T)| = "
          f"{np.max(np.abs(lam[-1])):.3e}")
    print(f"  backward: {bwd.shape}    B->A endpoint err = "
          f"{np.linalg.norm(bwd[-1, :, :2].mean(0) - A):.4f}")

    writer = imageio.get_writer(
        OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
    )
    try:
        write_section_title(
            writer,
            "Stigmergic optimal transport (PRL)",
            "Algorithm 1 (APIC) — one cycle, three sweeps",
            seconds=2.0, title_size=32, subtitle_size=18,
        )
        write_section_title(
            writer, "Step 1-2  ·  Forward pass",
            r"reparametrized Langevin (Eq. 4) + trail-following $\tilde\omega^{\mathrm{tf}}$ (Eq. 2)"
            "\n"
            r"integrate state $(\tilde X_i(s),\tilde Y_i(s),\tilde\Theta_i(s))$ from $s=0$ to $s=1$",
        )
        animate_forward(writer, fwd, A, B)

        write_section_title(
            writer, "Step 3  ·  Adjoint sweep",
            r"costates $(\mu(s),\Gamma(s))$ from Eq. 6,  "
            r"terminal conditions $\Gamma(1)=0$, $\mu(1)=-\nabla\Psi$"
            "\n"
            r"integrate BACKWARD in $s$ from $s=1$ to $s=0$",
        )
        animate_adjoint(writer, lam, dt)

        write_section_title(
            writer, "Step 4  ·  Controlled backward pass",
            r"feedback law (Eq. 7):  $\tilde\omega^{\mathrm{ctrl}}(s) = -\Gamma(s)/\gamma$,  "
            r"$\gamma=\beta D_\theta$"
            "\n"
            r"return from target ($s=1$) to source ($s=0$) under $\tilde\omega^{\mathrm{ctrl}}$",
        )
        animate_backward(writer, fwd, bwd, A, B)

        write_section_title(
            writer, "Three sweeps  ·  side-by-side",
            "forward (A->B),  adjoint backward in $s$,  controlled backward (B->A)",
            seconds=1.5,
        )
        final_summary(writer, fwd, lam, bwd, A, B, dt)
    finally:
        writer.close()

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
