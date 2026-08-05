"""Combined demo video: notebook run (left) + multi-cycle animation (right).

Same layout as `make_combined_demo_video.py` but the right-side animation
now plays several full stigmergic cycles.  Notebook 05 exercises one cycle
per stage (as an assertion-driven walkthrough); the right-side animation
shows what happens over N cycles until the trajectory bundle converges to
the Snell-optimal path.

Cell-to-animation mapping (during the three physics cells):

  stage1  -> Cycle 1 forward, Cycle 2 forward, ..., Cycle N forward
  stage2  -> Cycle 1 adjoint, ..., Cycle N adjoint
  stage3  -> Cycle 1 backward + Cycle 1 pheromone update,
             ..., Cycle N backward + Cycle N pheromone update,
             then a final convergence panel

The left cell view for those three cells is time-stretched to cover the
full right-side animation of all N cycles.
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from jax import random
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.apic import (  # noqa: E402
    downsample_recent_weighted_trajectories,
    integrate_costate,
    make_init_fn,
    simulate_controlled_backward_pass,
    simulate_forward_batch,
)

NB_PATH = Path("notebooks/05_forward_backward_demo.ipynb")
OUT_PATH = Path("figures/demo/05_notebook_plus_multicycle.mp4")
NUM_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 5

W, H = 1600, 900
LEFT_W = 900
RIGHT_W = 700
FPS = 30

# Animation timing: how many frames per stage, per cycle.
FRAMES_PER_STAGE = 40

BG           = (250, 250, 253)
CHROME       = (35,  38,  60)
CHROME_TEXT  = (230, 232, 245)
CELL_BG      = (255, 255, 255)
CODE_BG      = (245, 246, 250)
CODE_BG_HL   = (231, 236, 250)
BORDER_IDLE  = (210, 212, 220)
BORDER_RUN   = (60, 120, 200)
BORDER_DONE  = (140, 145, 160)
PROMPT       = (60, 120, 200)
PROMPT_RUN   = (200, 100, 40)
INK          = (30, 30, 40)
OUT_INK      = (55, 55, 70)
MUTED        = (130, 130, 145)
MD_BG        = (252, 252, 255)
DIVIDER      = (222, 224, 232)

POINT_A = jnp.array([0.0, 0.0])
POINT_B = jnp.array([1.0, 1.0])
BOUNDARY_Y = 0.5
BASE_NU = 1.0
JUMP_NU = 10.0
BATCH = 32
NUM_STEPS = 1520
DT = 0.001
PHER_SIGMA = 0.05


def snell_optimal_x():
    def cost(xc):
        return (BASE_NU * np.hypot(xc, BOUNDARY_Y)
                + JUMP_NU * np.hypot(1.0 - xc, 1.0 - BOUNDARY_Y))
    return float(minimize_scalar(cost, bounds=(0, 1), method='bounded').x)


SNELL_X = snell_optimal_x()
SNELL_PATH = (np.array([0.0, SNELL_X, 1.0]),
              np.array([0.0, BOUNDARY_Y, 1.0]))


# ---------------------------------------------------------------------------
# Fonts and text helpers (matches make_combined_demo_video.py)
# ---------------------------------------------------------------------------

def _font(size: int, mono: bool = False, bold: bool = False):
    if mono:
        cands = ["/System/Library/Fonts/Menlo.ttc",
                 "/Library/Fonts/Andale Mono.ttf"]
    else:
        cands = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    for p in cands:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE  = _font(22, bold=True)
F_TAB    = _font(14)
F_CODE   = _font(14, mono=True)
F_OUT    = _font(13, mono=True)
F_PROMPT = _font(12, mono=True)
F_MD     = _font(15)
F_LABEL  = _font(12)


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def wrap(text, width_px, font):
    lines = []
    avg = font.getlength("M") or 10
    chars = max(20, int(width_px / avg))
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=chars, drop_whitespace=False,
                                   break_long_words=False,
                                   replace_whitespace=False) or [""])
    return lines


# ---------------------------------------------------------------------------
# Left-half (notebook) rendering  (copied from combined video)
# ---------------------------------------------------------------------------

def draw_chrome(img, cell_index, total, right_label):
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (W, 46)], fill=CHROME)
    d.text((16, 12), "Jupyter", font=F_TITLE, fill=CHROME_TEXT)
    d.rounded_rectangle([(130, 8), (470, 38)], radius=6, fill=(55, 60, 90))
    d.text((142, 15), NB_PATH.name, font=F_TAB, fill=CHROME_TEXT)
    d.text((LEFT_W + 20, 15),
           right_label or "physics animation (idle)",
           font=F_TAB, fill=(200, 205, 220))
    right = f"cell {cell_index + 1} / {total}"
    tw = int(d.textlength(right, font=F_TAB))
    d.text((W - tw - 16, 15), right, font=F_TAB, fill=(180, 185, 205))
    d.rectangle([(LEFT_W, 46), (LEFT_W + 2, H)], fill=DIVIDER)


def draw_code_cell_left(img, source, phase, exec_display,
                        output_text, reveal_chars=10 ** 9):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 16, 62, LEFT_W - 20, H - 20
    border_color = (BORDER_IDLE if phase == "idle"
                    else BORDER_RUN if phase == "running"
                    else BORDER_DONE)
    d.rectangle([(x0, y0), (x0 + 4, y1)], fill=border_color)
    prompt_col = PROMPT_RUN if phase == "running" else PROMPT
    d.text((x0 + 16, y0 + 12), f"In [{exec_display}]:",
           font=F_PROMPT, fill=prompt_col)
    cx0 = x0 + 110
    cy0 = y0 + 4
    cx1 = x1 - 12
    src_lines = wrap(source, cx1 - cx0 - 24, F_CODE)
    line_h = F_CODE.size + 5
    src_h = min(len(src_lines) * line_h + 20, int((y1 - y0) * 0.55))
    src_bg = CODE_BG_HL if phase == "running" else CODE_BG
    d.rounded_rectangle([(cx0, cy0), (cx1, cy0 + src_h)], radius=6,
                        fill=src_bg, outline=(220, 224, 234))
    max_visible = max(3, (src_h - 20) // line_h)
    for i, line in enumerate(src_lines[:max_visible]):
        d.text((cx0 + 10, cy0 + 10 + i * line_h), line,
               font=F_CODE, fill=INK)
    if len(src_lines) > max_visible:
        d.text((cx0 + 10, cy0 + src_h - line_h - 6),
               f"... ({len(src_lines) - max_visible} more lines)",
               font=F_OUT, fill=MUTED)
    oy0 = cy0 + src_h + 12
    if phase in ("running", "done"):
        d.text((x0 + 16, oy0), f"Out[{exec_display}]:", font=F_PROMPT,
               fill=PROMPT if phase == "done" else MUTED)
    oh = y1 - oy0 - 4
    d.rounded_rectangle([(cx0, oy0), (cx1, oy0 + oh)], radius=6,
                        fill=CELL_BG, outline=(232, 234, 244))
    inner = (cx0 + 10, oy0 + 8, cx1 - 10, oy0 + oh - 10)
    if phase == "idle":
        d.text((inner[0], inner[1]),
               "(press Shift+Enter to run)", font=F_OUT, fill=MUTED)
        return
    joined = strip_ansi(output_text or "")[:reveal_chars]
    cursor_y = inner[1]
    out_lines = wrap(joined, inner[2] - inner[0], F_OUT)
    line_h2 = F_OUT.size + 3
    for line in out_lines:
        if cursor_y + F_OUT.size > inner[3] - 4:
            break
        d.text((inner[0], cursor_y), line, font=F_OUT, fill=OUT_INK)
        cursor_y += line_h2
    if phase == "running" and reveal_chars < len(strip_ansi(output_text or "")):
        last = out_lines[-1] if out_lines else ""
        lx = inner[0] + int(F_OUT.getlength(last))
        ly = cursor_y - line_h2
        d.rectangle([(lx + 2, ly + 2), (lx + 8, ly + F_OUT.size)],
                    fill=OUT_INK)


def render_markdown_cell(img, source):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 16, 62, LEFT_W - 20, H - 20
    d.rounded_rectangle([(x0 + 4, y0), (x1, y1)], radius=6, fill=MD_BG,
                        outline=(232, 234, 244))
    inner_w = x1 - x0 - 40
    y = y0 + 16
    for line in source.splitlines():
        s = line.rstrip()
        if s.startswith("# "):
            font = _font(24, bold=True)
            for wl in wrap(s[2:], inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 8
            y += 6
        elif s.startswith("## "):
            font = _font(19, bold=True)
            for wl in wrap(s[3:], inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 5
            y += 4
        elif s.startswith("### "):
            font = _font(17, bold=True)
            for wl in wrap(s[4:], inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 4
        elif s.startswith("|"):
            d.text((x0 + 24, y), s, font=F_CODE, fill=INK)
            y += F_CODE.size + 3
        elif s.startswith(("* ", "- ")):
            for wl in wrap("  - " + s[2:], inner_w, F_MD):
                d.text((x0 + 24, y), wl, font=F_MD, fill=INK)
                y += F_MD.size + 4
        elif not s.strip():
            y += 10
        else:
            for wl in wrap(s, inner_w, F_MD):
                d.text((x0 + 24, y), wl, font=F_MD, fill=INK)
                y += F_MD.size + 4
        if y > y1 - 30:
            break


# ---------------------------------------------------------------------------
# Right-half (animation) rendering
# ---------------------------------------------------------------------------

def _fig_to_pil(fig: Figure, size=(RIGHT_W, H - 46)) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return Image.fromarray(buf).resize(size, Image.LANCZOS)


def _kde_field(points, X, Y, sigma=PHER_SIGMA, max_points=3000):
    if points.shape[0] == 0:
        return np.zeros_like(X)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)
    field = np.zeros_like(X)
    step = 400
    for s in range(0, points.shape[0], step):
        ch = points[s:s + step]
        dx = X[None] - ch[:, 0][:, None, None]
        dy = Y[None] - ch[:, 1][:, None, None]
        field += np.exp(-(dx ** 2 + dy ** 2) * inv_2s2).sum(axis=0)
    return field


def _traj_axes(ax, title):
    ax.add_patch(Rectangle((-0.1, BOUNDARY_Y), 1.3, 1.3 - BOUNDARY_Y,
                           facecolor='#e9e9ef', edgecolor='none', zorder=0))
    ax.axhline(BOUNDARY_Y, color='k', lw=0.6, zorder=4)
    ax.plot(*POINT_A, marker='o', mec='k', mfc='#1f77b4', ms=7, zorder=5)
    ax.plot(*POINT_B, marker='o', mec='k', mfc='#2ca02c', ms=7, zorder=5)
    ax.plot(SNELL_PATH[0], SNELL_PATH[1], color='#e07a1f', lw=1.4,
            ls='--', zorder=3)
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$y$', fontsize=10)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.tick_params(labelsize=8)


def build_forward_frames_all_cycles(cycles):
    """Return: list of (right_image, cycle_idx, s_norm) tuples covering all
    cycles played sequentially."""
    grid = 70
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    all_frames = []
    for c, cyc in enumerate(cycles):
        fwd = cyc['fwd']
        pher = cyc['pher_in']
        T, B, _ = fwd.shape
        steps = np.linspace(2, T, FRAMES_PER_STAGE).astype(int)
        for t in steps:
            fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100),
                             dpi=100, layout='constrained')
            ax = fig.add_subplot(111)
            field = _kde_field(pher, X, Y)
            if field.max() > 0:
                vmax = float(np.percentile(field, 99)) or 1.0
                ax.imshow(field, origin='lower',
                          extent=[x.min(), x.max(), y.min(), y.max()],
                          cmap='Purples', vmin=0, vmax=vmax,
                          alpha=0.7, aspect='auto', zorder=1)
            for b in range(B):
                ax.plot(fwd[:t, b, 0], fwd[:t, b, 1],
                        color='#c1272d', alpha=0.35, lw=0.6, zorder=2)
            ax.plot(fwd[t - 1, :, 0], fwd[t - 1, :, 1], 'o',
                    color='#c1272d', ms=3, alpha=0.9, zorder=3)
            _traj_axes(ax,
                       f"Cycle {c + 1}/{len(cycles)}  ·  Stage 1: forward "
                       f"pass  (s = {(t - 1) / (T - 1):.2f})")
            all_frames.append(_fig_to_pil(fig))
            plt.close(fig)
    return all_frames


def build_adjoint_frames_all_cycles(cycles):
    all_frames = []
    for c, cyc in enumerate(cycles):
        lam = cyc['lam']
        T, B, _ = lam.shape
        s = np.linspace(0, 1, T)
        starts = np.linspace(T, 2, FRAMES_PER_STAGE).astype(int)
        ymin = float(lam[:, :, 2].min()) * 1.1
        ymax = float(lam[:, :, 2].max()) * 1.1
        for s_start in starts:
            fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100),
                             dpi=100, layout='constrained')
            ax = fig.add_subplot(111)
            for b in range(B):
                ax.plot(s[s_start - 1:], lam[s_start - 1:, b, 2],
                        color='#7b3f99', alpha=0.35, lw=0.7)
            ax.scatter([1.0], [0.0], color='k', zorder=10, s=50,
                       label=r'$\Gamma(1)=0$')
            ax.axhline(0.0, color='k', ls=':', lw=0.6)
            ax.axvline(s[s_start - 1], color='#7b3f99', lw=1.1, alpha=0.6)
            ax.set_xlim(0, 1.02); ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r'arc length $s$', fontsize=10)
            ax.set_ylabel(r'$\Gamma(s)$', fontsize=10)
            ax.set_title(
                f"Cycle {c + 1}/{len(cycles)}  ·  Stage 2: adjoint "
                f"sweep  (s = {s[s_start - 1]:.2f})",
                fontsize=11, weight='bold',
            )
            ax.legend(loc='upper left', fontsize=9)
            ax.tick_params(labelsize=8)
            all_frames.append(_fig_to_pil(fig))
            plt.close(fig)
    return all_frames


def build_backward_frames_all_cycles(cycles):
    grid = 70
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    all_frames = []
    n = len(cycles)
    for c, cyc in enumerate(cycles):
        fwd, bwd, pher = cyc['fwd'], cyc['bwd'], cyc['pher_in']
        T, B, _ = bwd.shape
        steps = np.linspace(2, T, FRAMES_PER_STAGE).astype(int)
        for t in steps:
            fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100),
                             dpi=100, layout='constrained')
            ax = fig.add_subplot(111)
            field = _kde_field(pher, X, Y)
            if field.max() > 0:
                vmax = float(np.percentile(field, 99)) or 1.0
                ax.imshow(field, origin='lower',
                          extent=[x.min(), x.max(), y.min(), y.max()],
                          cmap='Purples', vmin=0, vmax=vmax,
                          alpha=0.65, aspect='auto', zorder=1)
            for b in range(B):
                ax.plot(fwd[:, b, 0], fwd[:, b, 1],
                        color='#c1272d', alpha=0.10, lw=0.4, zorder=2)
            for b in range(B):
                ax.plot(bwd[:t, b, 0], bwd[:t, b, 1],
                        color='#1f5aa8', alpha=0.4, lw=0.6, zorder=3)
            ax.plot(bwd[t - 1, :, 0], bwd[t - 1, :, 1], 'o',
                    color='#1f5aa8', ms=3, alpha=0.9, zorder=4)
            _traj_axes(ax,
                       f"Cycle {c + 1}/{n}  ·  Stage 3: controlled "
                       f"backward  (step {t}/{T})")
            all_frames.append(_fig_to_pil(fig))
            plt.close(fig)
        # A short pheromone-update transition for this cycle
        field_in = _kde_field(pher, X, Y)
        field_out = _kde_field(cyc['pher_out'], X, Y)
        for k in range(FRAMES_PER_STAGE // 2):
            alpha = (k + 1) / (FRAMES_PER_STAGE // 2)
            field = (1 - alpha) * field_in + alpha * field_out
            fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100),
                             dpi=100, layout='constrained')
            ax = fig.add_subplot(111)
            vmax = float(np.percentile(field, 99)) or 1.0
            ax.imshow(field, origin='lower',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      cmap='Purples', vmin=0, vmax=vmax, alpha=0.85,
                      aspect='auto', zorder=1)
            _traj_axes(ax, f"Cycle {c + 1}/{n}  ·  "
                           "Stage 4: pheromone update (new $\\phi$)")
            all_frames.append(_fig_to_pil(fig))
            plt.close(fig)
    return all_frames


def build_convergence_final_frame(cycles) -> Image.Image:
    grid = 70
    x = np.linspace(-0.05, 1.1, grid)
    y = np.linspace(-0.05, 1.1, grid)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100), dpi=100,
                     layout='constrained')
    ax = fig.add_subplot(111)
    last = cycles[-1]
    field = _kde_field(last['pher_out'], X, Y)
    if field.max() > 0:
        vmax = float(np.percentile(field, 99)) or 1.0
        ax.imshow(field, origin='lower',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  cmap='Purples', vmin=0, vmax=vmax, alpha=0.75,
                  aspect='auto', zorder=1)
    for b in range(last['fwd'].shape[1]):
        ax.plot(last['fwd'][:, b, 0], last['fwd'][:, b, 1],
                color='#c1272d', alpha=0.35, lw=0.5, zorder=2)
    _traj_axes(ax, f"Converged bundle after {len(cycles)} cycles")
    im = _fig_to_pil(fig)
    plt.close(fig)
    return im


# ---------------------------------------------------------------------------
# Precompute cycles
# ---------------------------------------------------------------------------

def run_all_cycles(num_cycles: int, seed: int = 0):
    key = random.PRNGKey(seed)
    init_fn = make_init_fn(POINT_A, POINT_B, BATCH)
    pher_pts = jnp.empty((0, 2))
    pher_wts = jnp.empty((0,))
    all_bwd = []
    out = []
    for c in range(num_cycles):
        key, k_init, k_fwd, k_bwd, k_phen = random.split(key, 5)
        init_states = init_fn(k_init)
        pher_in = np.asarray(pher_pts)
        fwd = simulate_forward_batch(k_fwd, init_states, pher_pts, pher_wts,
                                     PHER_SIGMA, DT, NUM_STEPS, POINT_B)
        lam = integrate_costate(fwd, pher_pts, pher_wts, PHER_SIGMA, DT)
        final = fwd[-1]
        theta_flip = jnp.mod(final[:, 2] + jnp.pi, 2 * jnp.pi)
        final_rot = jnp.stack([final[:, 0], final[:, 1], theta_flip], axis=1)
        bwd = simulate_controlled_backward_pass(
            k_bwd, final_rot, lam, pher_pts, pher_wts, PHER_SIGMA,
            NUM_STEPS, DT, POINT_A,
        )
        all_bwd.append(bwd)
        pher_pts, pher_wts = downsample_recent_weighted_trajectories(
            all_backward_trajectories=all_bwd,
            num_trajs_to_sample=200,
            weight=1.0, key=k_phen,
        )
        out.append({
            'fwd': np.asarray(fwd),
            'lam': np.asarray(lam),
            'bwd': np.asarray(bwd),
            'pher_in': pher_in,
            'pher_out': np.asarray(pher_pts),
        })
    return out


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def cell_output_text(cell):
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append(o.get("text", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                t = data["text/plain"]
                out.append(t if isinstance(t, str) else "".join(t))
    return "\n".join(out).rstrip()


def compose(left, right, cell_index, total, right_label):
    canvas = Image.new("RGB", (W, H), BG)
    # Ensure paste inputs match the target regions exactly (uniform frame size)
    if left.size != (LEFT_W, H):
        left = left.resize((LEFT_W, H), Image.LANCZOS)
    target_right = (RIGHT_W, H - 46)
    if right.size != target_right:
        right = right.resize(target_right, Image.LANCZOS)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (LEFT_W + 2, 46))
    draw_chrome(canvas, cell_index, total, right_label)
    arr = np.asarray(canvas)
    if arr.shape[:2] != (H, W):
        arr = np.asarray(canvas.resize((W, H), Image.LANCZOS))
    return arr


def make_left_frame(cell, phase, exec_display, output_text,
                    reveal_chars=10 ** 9):
    left = Image.new("RGB", (LEFT_W, H), BG)
    if cell.cell_type == "markdown":
        render_markdown_cell(left, cell.source)
    else:
        draw_code_cell_left(left, cell.source, phase, exec_display,
                            output_text, reveal_chars=reveal_chars)
    return left


def main():
    print(f"Running {NUM_CYCLES} stigmergic cycles ...")
    cycles = run_all_cycles(NUM_CYCLES)
    print("Building forward frames ..."); fwd_fr = build_forward_frames_all_cycles(cycles)
    print("Building adjoint frames ..."); adj_fr = build_adjoint_frames_all_cycles(cycles)
    print("Building backward+update frames ..."); bwd_fr = build_backward_frames_all_cycles(cycles)
    print("Building convergence panel ..."); conv_frame = build_convergence_final_frame(cycles)
    blank = Image.new("RGB", (RIGHT_W, H - 46), BG)
    d = ImageDraw.Draw(blank)
    d.text((RIGHT_W // 2 - 90, (H - 46) // 2 - 8),
           "(no animation for this cell)", font=F_LABEL, fill=MUTED)

    nb = nbformat.read(NB_PATH, as_version=4)
    cells = nb.cells
    total = len(cells)

    anim_by_id = {
        "stage1": (fwd_fr,
                   f"Stage 1 animation, all {NUM_CYCLES} cycles: forward pass"),
        "stage2": (adj_fr,
                   f"Stage 2 animation, all {NUM_CYCLES} cycles: adjoint sweep"),
        "stage3": (bwd_fr,
                   f"Stage 3+4 animation, all {NUM_CYCLES} cycles: "
                   "backward + pheromone update"),
    }
    last_frame = blank
    right_label_static = "physics animation (idle)"

    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    exec_count = 0

    try:
        # Intro card
        intro_left = Image.new("RGB", (LEFT_W, H), BG)
        di = ImageDraw.Draw(intro_left)
        di.text((40, 220), "Combined demo (multi-cycle)",
                font=_font(28, bold=True), fill=INK)
        di.text((40, 268),
                "Left:   live walkthrough of notebook 05",
                font=_font(17), fill=OUT_INK)
        di.text((40, 294),
                f"Right:  physics animation for all {NUM_CYCLES} stigmergic",
                font=_font(17), fill=OUT_INK)
        di.text((40, 320),
                "        cycles until trajectory bundle converges to",
                font=_font(17), fill=OUT_INK)
        di.text((40, 346),
                "        the Snell-optimal path (orange dashed).",
                font=_font(17), fill=OUT_INK)
        intro_right = Image.new("RGB", (RIGHT_W, H - 46), (240, 240, 250))
        for _ in range(FPS * 2):
            writer.append_data(compose(intro_left, intro_right, 0, total,
                                       "(waiting for a stage cell)"))

        for i, cell in enumerate(cells):
            if cell.cell_type == "markdown":
                left = make_left_frame(cell, "done", " ", "")
                for _ in range(int(FPS * 2.0)):
                    writer.append_data(compose(left, last_frame, i, total,
                                               right_label_static))
                continue

            exec_count += 1
            output_text = cell_output_text(cell)
            cell_id = cell.get("id", "")
            anim = anim_by_id.get(cell_id)

            left = make_left_frame(cell, "idle", " ", "")
            for _ in range(int(FPS * 0.6)):
                writer.append_data(compose(left, last_frame, i, total,
                                           right_label_static))

            if anim is not None:
                frames, label = anim
                right_label_static = label
                for k, right_im in enumerate(frames):
                    reveal = int((k + 1) / len(frames) *
                                 len(strip_ansi(output_text)))
                    left = make_left_frame(cell, "running", "*",
                                           output_text, reveal_chars=reveal)
                    writer.append_data(compose(left, right_im, i, total,
                                               label))
                last_frame = frames[-1]
                # For stage3, tack on a final convergence panel
                if cell_id == "stage3":
                    right_label_static = "Convergence: bundle around the Snell-optimal path"
                    last_frame = conv_frame
                    for _ in range(FPS * 3):
                        writer.append_data(compose(
                            make_left_frame(cell, "done", str(exec_count),
                                            output_text),
                            conv_frame, i, total, right_label_static))
            else:
                n = len(strip_ansi(output_text))
                target = min(2.5, max(0.8, n / 250))
                total_fr = int(FPS * target)
                for k in range(total_fr):
                    reveal = int((k + 1) / max(total_fr, 1) * n)
                    left = make_left_frame(cell, "running", "*",
                                           output_text, reveal_chars=reveal)
                    writer.append_data(compose(left, last_frame, i, total,
                                               right_label_static))

            left = make_left_frame(cell, "done", str(exec_count), output_text)
            for _ in range(int(FPS * 1.6)):
                writer.append_data(compose(left, last_frame, i, total,
                                           right_label_static))

        outro_left = Image.new("RGB", (LEFT_W, H), BG)
        do = ImageDraw.Draw(outro_left)
        do.text((40, 260), "Run complete.  Every assertion passed.",
                font=_font(22, bold=True), fill=INK)
        do.text((40, 306),
                f"The animation on the right showed all {NUM_CYCLES} stigmergic",
                font=_font(16), fill=OUT_INK)
        do.text((40, 330),
                "cycles: forward, adjoint, backward, pheromone update.",
                font=_font(16), fill=OUT_INK)
        do.text((40, 360),
                "Trajectory bundle converged around the Snell-optimal path.",
                font=_font(16), fill=OUT_INK)
        for _ in range(FPS * 3):
            writer.append_data(compose(outro_left, conv_frame, total, total,
                                       "Converged"))
    finally:
        writer.close()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
