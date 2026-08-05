"""Combined demo video: notebook run on the left, physics animation on the right.

The left half plays through `notebooks/05_forward_backward_demo.ipynb`
cell-by-cell like `make_notebook_run_video.py`.  The right half stays
blank for setup/import cells; for the three physics cells it shows the
matching trajectory animation:

  * cell `stage1`  -> forward pass trajectories growing A -> B
  * cell `stage2`  -> adjoint costate Gamma(s) built backward from s=1
  * cell `stage3`  -> controlled backward trajectories B -> A over the
                       faded forward bundle

The left cell view for those three cells is time-stretched so the entire
animation plays while the output types out.
"""
from __future__ import annotations

import base64
import io
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.apic import (  # noqa: E402
    integrate_costate,
    make_init_fn,
    simulate_controlled_backward_pass,
    simulate_forward_batch,
    smooth_piecewise_nu,
)

NB_PATH = Path("notebooks/05_forward_backward_demo.ipynb")
OUT_PATH = Path("figures/demo/05_notebook_plus_animation.mp4")

W, H = 1600, 900
LEFT_W = 900        # notebook cell area
RIGHT_W = 700       # animation area
FPS = 30
ANIM_SECS = 4.0     # animation length for physics cells
ANIM_FRAMES = int(FPS * ANIM_SECS)

# Palette (matches make_notebook_run_video.py)
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


def wrap(text: str, width_px: int, font):
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


# =============================================================================
# Left half: notebook cell rendering
# =============================================================================

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
    # Divider between panels
    d.rectangle([(LEFT_W, 46), (LEFT_W + 2, H)], fill=DIVIDER)


def draw_code_cell_left(img, source, phase, exec_display,
                        output_text, reveal_chars=10**9):
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
        d.text((x0 + 16, oy0), f"Out[{exec_display}]:",
               font=F_PROMPT,
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


# =============================================================================
# Right half: physics animation frames (matplotlib -> PIL images)
# =============================================================================

def run_cycle():
    batch_size = 32
    num_steps = 1520
    dt = 0.001
    pher_sigma = 0.05
    A = jnp.array([0.0, 0.0])
    B = jnp.array([1.0, 1.0])
    key = random.PRNGKey(0)
    k_init, k_fwd, k_bwd = random.split(key, 3)
    init_fn = make_init_fn(A, B, batch_size)
    init_states = init_fn(k_init)
    pher_pts = jnp.empty((0, 2))
    pher_wts = jnp.empty((0,))

    fwd = simulate_forward_batch(k_fwd, init_states, pher_pts, pher_wts,
                                 pher_sigma, dt, num_steps, B)
    lam = integrate_costate(fwd, pher_pts, pher_wts, pher_sigma, dt)
    final = fwd[-1]
    theta_flip = jnp.mod(final[:, 2] + jnp.pi, 2 * jnp.pi)
    final_rot = jnp.stack([final[:, 0], final[:, 1], theta_flip], axis=1)
    bwd = simulate_controlled_backward_pass(
        k_bwd, final_rot, lam, pher_pts, pher_wts, pher_sigma,
        num_steps, dt, A,
    )
    return (
        np.asarray(fwd),
        np.asarray(lam),
        np.asarray(bwd),
        np.asarray(A),
        np.asarray(B),
        dt,
    )


def _fig_to_pil(fig: Figure, size=(RIGHT_W, H - 46)) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    im = Image.fromarray(buf)
    return im.resize(size, Image.LANCZOS)


def _traj_axes(ax, A, B, title):
    ax.axhspan(0.5, 1.15, color="#e9e9ef", zorder=0)
    ax.axhline(0.5, color="k", lw=0.6, zorder=4)
    ax.plot(*A, "o", mec="k", mfc="#1f77b4", ms=8, zorder=5)
    ax.plot(*B, "o", mec="k", mfc="#2ca02c", ms=8, zorder=5)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$y$", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.tick_params(labelsize=8)


def forward_frames(fwd, A, B):
    T, B_ct, _ = fwd.shape
    steps = np.linspace(2, T, ANIM_FRAMES).astype(int)
    frames = []
    for t in steps:
        fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100), dpi=100,
                          layout="constrained")
        ax = fig.add_subplot(111)
        for b in range(B_ct):
            ax.plot(fwd[:t, b, 0], fwd[:t, b, 1],
                    color="#c1272d", alpha=0.35, lw=0.6, zorder=2)
        ax.plot(fwd[t - 1, :, 0], fwd[t - 1, :, 1], "o",
                color="#c1272d", ms=4, alpha=0.9, zorder=3)
        s_norm = (t - 1) / (T - 1)
        _traj_axes(ax, A, B,
                   f"Stage 1: forward pass  (Eq. 4)\n"
                   f"arc length  $s = {s_norm:.2f}$")
        frames.append(_fig_to_pil(fig))
        plt.close(fig)
    return frames


def adjoint_frames(lam, dt):
    T, B_ct, _ = lam.shape
    s = np.linspace(0.0, 1.0, T)
    starts = np.linspace(T, 2, ANIM_FRAMES).astype(int)
    ymin = float(lam[:, :, 2].min()) * 1.1
    ymax = float(lam[:, :, 2].max()) * 1.1
    frames = []
    for s_start in starts:
        fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100), dpi=100,
                          layout="constrained")
        ax = fig.add_subplot(111)
        for b in range(B_ct):
            ax.plot(s[s_start - 1:], lam[s_start - 1:, b, 2],
                    color="#7b3f99", alpha=0.35, lw=0.7)
        ax.scatter([1.0], [0.0], color="k", zorder=10, s=60,
                   label=r"terminal:  $\Gamma(1)=0$")
        ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.6)
        ax.axvline(s[s_start - 1], color="#7b3f99", lw=1.2, alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"arc length  $s \in [0,1]$", fontsize=10)
        ax.set_ylabel(r"heading costate  $\Gamma(s)$", fontsize=10)
        ax.set_title(
            "Stage 2: adjoint sweep  (Eq. 6, backward in $s$)\n"
            f"sweep frontier  $s = {s[s_start - 1]:.2f}$",
            fontsize=12, weight="bold",
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.tick_params(labelsize=8)
        frames.append(_fig_to_pil(fig))
        plt.close(fig)
    return frames


def backward_frames(fwd, bwd, A, B):
    T, B_ct, _ = bwd.shape
    steps = np.linspace(2, T, ANIM_FRAMES).astype(int)
    frames = []
    for t in steps:
        fig = plt.figure(figsize=(RIGHT_W / 100, (H - 46) / 100), dpi=100,
                          layout="constrained")
        ax = fig.add_subplot(111)
        for b in range(B_ct):
            ax.plot(fwd[:, b, 0], fwd[:, b, 1],
                    color="#c1272d", alpha=0.10, lw=0.4, zorder=1)
        for b in range(B_ct):
            ax.plot(bwd[:t, b, 0], bwd[:t, b, 1],
                    color="#1f5aa8", alpha=0.4, lw=0.7, zorder=2)
        ax.plot(bwd[t - 1, :, 0], bwd[t - 1, :, 1], "o",
                color="#1f5aa8", ms=4, alpha=0.9, zorder=3)
        _traj_axes(ax, A, B,
                   f"Stage 3: controlled backward  (Eq. 7)\n"
                   r"$\tilde\omega^{\mathrm{ctrl}} = -\Gamma/\gamma$, "
                   f"step {t}/{T}")
        frames.append(_fig_to_pil(fig))
        plt.close(fig)
    return frames


def blank_right() -> Image.Image:
    img = Image.new("RGB", (RIGHT_W, H - 46), BG)
    d = ImageDraw.Draw(img)
    d.text((RIGHT_W // 2 - 60, (H - 46) // 2 - 8),
           "(no animation for this cell)",
           font=F_LABEL, fill=MUTED)
    return img


# =============================================================================
# Notebook helpers
# =============================================================================

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


# =============================================================================
# Compose and write
# =============================================================================

def compose(left: Image.Image, right: Image.Image,
            cell_index: int, total: int, right_label: str) -> np.ndarray:
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (LEFT_W + 2, 46))
    draw_chrome(canvas, cell_index, total, right_label)
    return np.asarray(canvas)


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
    print("Precomputing one APIC cycle for the animations...")
    fwd, lam, bwd, A, B, dt = run_cycle()
    print("  building forward frames..."); fwd_fr = forward_frames(fwd, A, B)
    print("  building adjoint frames..."); adj_fr = adjoint_frames(lam, dt)
    print("  building backward frames..."); bwd_fr = backward_frames(fwd, bwd, A, B)
    blank = blank_right()

    nb = nbformat.read(NB_PATH, as_version=4)
    cells = nb.cells
    total = len(cells)

    # Map cell id -> (animation frames, label)
    anim_by_id = {
        "stage1": (fwd_fr,
                   "Stage 1 animation: forward pass (A -> B)"),
        "stage2": (adj_fr,
                   "Stage 2 animation: adjoint sweep (backward in $s$)"),
        "stage3": (bwd_fr,
                   "Stage 3 animation: controlled backward (B -> A)"),
    }
    # Freeze frames: after an animation plays, later cells continue to
    # display the final frame of the most recently played animation.
    right_label_static = "physics animation (idle)"
    last_frame = blank

    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    exec_count = 0
    try:
        # Intro card (2s)
        intro_left = Image.new("RGB", (LEFT_W, H), BG)
        di = ImageDraw.Draw(intro_left)
        di.text((40, 200), "Combined demo",
                font=_font(28, bold=True), fill=INK)
        di.text((40, 250), "Left:   live notebook walkthrough",
                font=_font(17), fill=OUT_INK)
        di.text((40, 278), "Right:  physics animation for the",
                font=_font(17), fill=OUT_INK)
        di.text((40, 302), "        forward / adjoint / backward cells",
                font=_font(17), fill=OUT_INK)
        di.text((40, 360), NB_PATH.name,
                font=_font(15, mono=True), fill=(80, 100, 160))
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

            # Idle (0.6s)
            left = make_left_frame(cell, "idle", " ", "")
            for _ in range(int(FPS * 0.6)):
                writer.append_data(compose(left, last_frame, i, total,
                                           right_label_static))

            # Running phase - if animation, play it over ANIM_FRAMES; else 1s
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
            else:
                n_chars = len(strip_ansi(output_text))
                target = min(2.5, max(0.8, n_chars / 250))
                total_fr = int(FPS * target)
                for k in range(total_fr):
                    reveal = int((k + 1) / max(total_fr, 1) * n_chars)
                    left = make_left_frame(cell, "running", "*",
                                           output_text, reveal_chars=reveal)
                    writer.append_data(compose(left, last_frame, i, total,
                                               right_label_static))

            # Done hold (1.8s)
            left = make_left_frame(cell, "done", str(exec_count),
                                   output_text)
            for _ in range(int(FPS * 1.8)):
                writer.append_data(compose(left, last_frame, i, total,
                                           right_label_static))

        # Outro (3s)
        outro_left = Image.new("RGB", (LEFT_W, H), BG)
        do = ImageDraw.Draw(outro_left)
        do.text((40, 260), "Run complete.  Every assertion passed.",
                font=_font(24, bold=True), fill=INK)
        do.text((40, 310), "The animation on the right corresponds to",
                font=_font(17), fill=OUT_INK)
        do.text((40, 336), "the same stigmergic cycle exercised by the",
                font=_font(17), fill=OUT_INK)
        do.text((40, 362), "notebook.", font=_font(17), fill=OUT_INK)
        for _ in range(FPS * 3):
            writer.append_data(compose(outro_left, last_frame, total, total,
                                       right_label_static))
    finally:
        writer.close()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
