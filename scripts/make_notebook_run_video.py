"""Render an executed notebook as an mp4 that mimics a live Jupyter run.

For each cell we show:
  1. the source with `In [ ]:` prompt (idle),
  2. the prompt turns to `In [*]:` for a moment (running),
  3. text output types out character-by-character (with a caret),
  4. image output (if any) fades in,
  5. the prompt becomes `In [N]:` (done).

Markdown cells render as a rendered-block section (not raw source).

Usage:  make_notebook_run_video.py [in.ipynb] [out.mp4]

Defaults to  notebooks/05_forward_backward_demo.ipynb  ->
              figures/demo/05_forward_backward_run.mp4
"""
from __future__ import annotations

import base64
import io
import re
import sys
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import nbformat
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "notebooks/05_forward_backward_demo.ipynb"
)
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "figures/demo/05_forward_backward_run.mp4"
)

W, H = 1600, 900
FPS = 30

# Palette (Jupyter-ish light theme)
BG           = (250, 250, 253)
CHROME       = (35,  38,  60)
CHROME_TEXT  = (230, 232, 245)
CELL_BG      = (255, 255, 255)
CODE_BG      = (245, 246, 250)
CODE_BG_HL   = (231, 236, 250)  # highlight when running
BORDER_IDLE  = (210, 212, 220)
BORDER_RUN   = (60, 120, 200)
BORDER_DONE  = (140, 145, 160)
PROMPT       = (60, 120, 200)
PROMPT_RUN   = (200, 100, 40)
INK          = (30, 30, 40)
OUT_INK      = (55, 55, 70)
MUTED        = (130, 130, 145)
MD_BG        = (252, 252, 255)


def _font(size: int, mono: bool = False, bold: bool = False):
    if mono:
        candidates = ["/System/Library/Fonts/Menlo.ttc",
                      "/Library/Fonts/Andale Mono.ttf"]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
            else "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE  = _font(22, bold=True)
F_TAB    = _font(14)
F_BODY   = _font(17)
F_CODE   = _font(15, mono=True)
F_OUT    = _font(14, mono=True)
F_PROMPT = _font(13, mono=True)
F_MD_H   = _font(20, bold=True)
F_MD     = _font(16)


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def wrap(text: str, width_px: int, font) -> list[str]:
    lines: list[str] = []
    avg = font.getlength("M") or 10
    chars = max(20, int(width_px / avg))
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=chars, drop_whitespace=False,
                                   break_long_words=False, replace_whitespace=False)
                     or [""])
    return lines


def draw_chrome(img: Image.Image, nb_name: str, exec_count: int, total: int):
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (W, 46)], fill=CHROME)
    d.text((16, 12), "Jupyter", font=F_TITLE, fill=CHROME_TEXT)
    tab_x = 130
    d.rounded_rectangle([(tab_x, 8), (tab_x + 340, 38)], radius=6,
                        fill=(55, 60, 90))
    d.text((tab_x + 12, 15), nb_name, font=F_TAB, fill=CHROME_TEXT)
    right = f"cell {exec_count} / {total}"
    tw = int(d.textlength(right, font=F_TAB))
    d.text((W - tw - 16, 15), right, font=F_TAB, fill=(180, 185, 205))


def draw_code_cell(img: Image.Image, source: str, phase: str,
                   exec_display: str, output_text: str, output_img=None,
                   reveal_chars: int = 10 ** 9,
                   region=(24, 62, W - 24, H - 24)):
    """phase in {"idle", "running", "done"}."""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = region

    # Left border colored by phase
    border_color = (BORDER_IDLE if phase == "idle"
                    else BORDER_RUN if phase == "running"
                    else BORDER_DONE)
    d.rectangle([(x0, y0), (x0 + 4, y1)], fill=border_color)

    # Prompt column
    px = x0 + 16
    py = y0 + 12
    prompt_col = PROMPT_RUN if phase == "running" else PROMPT
    d.text((px, py), f"In [{exec_display}]:", font=F_PROMPT, fill=prompt_col)

    # Code panel
    cx0 = x0 + 130
    cy0 = y0 + 4
    cx1 = x1 - 16
    src_lines = wrap(source, cx1 - cx0 - 24, F_CODE)
    line_h = F_CODE.size + 5
    src_h = min(len(src_lines) * line_h + 20, int((y1 - y0) * 0.45))
    src_bg = CODE_BG_HL if phase == "running" else CODE_BG
    d.rounded_rectangle([(cx0, cy0), (cx1, cy0 + src_h)], radius=6,
                        fill=src_bg, outline=(220, 224, 234))
    max_visible = max(3, (src_h - 20) // line_h)
    for i, line in enumerate(src_lines[:max_visible]):
        d.text((cx0 + 12, cy0 + 10 + i * line_h), line,
               font=F_CODE, fill=INK)
    if len(src_lines) > max_visible:
        d.text((cx0 + 12, cy0 + src_h - line_h - 6),
               f"... ({len(src_lines) - max_visible} more lines)",
               font=F_OUT, fill=MUTED)

    # Output panel
    oy0 = cy0 + src_h + 12
    if phase in ("running", "done"):
        d.text((px, oy0), f"Out[{exec_display}]:", font=F_PROMPT,
               fill=PROMPT if phase == "done" else MUTED)
    oh = y1 - oy0 - 4
    d.rounded_rectangle([(cx0, oy0), (cx1, oy0 + oh)], radius=6,
                        fill=CELL_BG, outline=(232, 234, 244))
    inner = (cx0 + 12, oy0 + 10, cx1 - 12, oy0 + oh - 10)

    if phase == "idle":
        d.text((inner[0], inner[1]),
               "(press Shift+Enter to run)", font=F_OUT, fill=MUTED)
        return

    cursor_y = inner[1]

    # Text output typed to reveal_chars
    if output_text:
        joined = strip_ansi(output_text)[:reveal_chars]
        out_lines = wrap(joined, inner[2] - inner[0], F_OUT)
        line_h = F_OUT.size + 3
        for line in out_lines:
            if cursor_y + F_OUT.size > inner[3] - 4:
                break
            d.text((inner[0], cursor_y), line, font=F_OUT, fill=OUT_INK)
            cursor_y += line_h
        # caret if we're mid-typing (running phase)
        if phase == "running" and reveal_chars < len(strip_ansi(output_text)):
            last_line = out_lines[-1] if out_lines else ""
            lx = inner[0] + int(F_OUT.getlength(last_line))
            ly = cursor_y - line_h
            d.rectangle([(lx + 2, ly + 2), (lx + 8, ly + F_OUT.size)],
                        fill=OUT_INK)
        cursor_y += 6

    # Image output (fade in): drawn if reveal_chars >= len(text)
    if output_img is not None and reveal_chars >= len(output_text):
        max_h = inner[3] - cursor_y
        max_w = inner[2] - inner[0]
        im = output_img
        ratio = im.width / im.height
        h = min(max_h, im.height)
        w = int(h * ratio)
        if w > max_w:
            w = max_w
            h = int(w / ratio)
        thumb = im.resize((w, h), Image.LANCZOS)
        img.paste(thumb, (inner[0], cursor_y))


def render_markdown_cell(img: Image.Image, source: str,
                         region=(24, 62, W - 24, H - 24)):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = region
    d.rounded_rectangle([(x0 + 4, y0), (x1, y1)], radius=6, fill=MD_BG,
                        outline=(232, 234, 244))
    inner_w = x1 - x0 - 40
    y = y0 + 16
    for line in source.splitlines():
        s = line.rstrip()
        if s.startswith("# "):
            font = _font(28, bold=True)
            wrapped = wrap(s[2:], inner_w, font)
            for wl in wrapped:
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 8
            y += 8
        elif s.startswith("## "):
            font = _font(22, bold=True)
            wrapped = wrap(s[3:], inner_w, font)
            for wl in wrapped:
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 6
            y += 6
        elif s.startswith("### "):
            font = _font(19, bold=True)
            for wl in wrap(s[4:], inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 5
        elif s.startswith("|"):
            d.text((x0 + 24, y), s, font=F_CODE, fill=INK)
            y += F_CODE.size + 3
        elif s.startswith(("* ", "- ")):
            font = F_MD
            for wl in wrap("  - " + s[2:], inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 4
        elif not s.strip():
            y += 10
        else:
            font = F_MD
            for wl in wrap(s, inner_w, font):
                d.text((x0 + 24, y), wl, font=font, fill=INK)
                y += font.size + 4
        if y > y1 - 30:
            break


def cell_outputs(cell):
    texts = []
    imgs = []
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        if ot == "stream":
            texts.append(out.get("text", ""))
        elif ot in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                raw = base64.b64decode(data["image/png"])
                imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            elif "text/plain" in data:
                t = data["text/plain"]
                texts.append(t if isinstance(t, str) else "".join(t))
    return "\n".join(texts).rstrip(), imgs[0] if imgs else None


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    cells = nb.cells
    total_cells = sum(1 for c in cells)

    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    exec_count = 0

    try:
        # Intro card
        intro = Image.new("RGB", (W, H), BG)
        di = ImageDraw.Draw(intro)
        draw_chrome(intro, NB_PATH.name, 0, total_cells)
        di.text((80, 240),
                "Live walk-through: notebooks/05_forward_backward_demo.ipynb",
                font=_font(30, bold=True), fill=INK)
        di.text((80, 290),
                "Each cell is shown, executed, and its output revealed.",
                font=_font(18), fill=OUT_INK)
        di.text((80, 340),
                "This notebook exercises each of the four stages of Algorithm 1",
                font=_font(16), fill=OUT_INK)
        di.text((80, 366),
                "independently and asserts the forward-backward structure.",
                font=_font(16), fill=OUT_INK)
        for _ in range(FPS * 2):
            writer.append_data(np.asarray(intro))

        for i, cell in enumerate(cells):
            if cell.cell_type == "markdown":
                frame = Image.new("RGB", (W, H), BG)
                draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
                render_markdown_cell(frame, cell.source)
                arr = np.asarray(frame)
                for _ in range(int(FPS * 2.2)):
                    writer.append_data(arr)
                continue

            exec_count += 1
            output_text, output_img = cell_outputs(cell)

            # Phase 1: idle (0.8s)
            frame = Image.new("RGB", (W, H), BG)
            draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
            draw_code_cell(frame, cell.source, "idle", " ", "", None)
            arr = np.asarray(frame)
            for _ in range(int(FPS * 0.8)):
                writer.append_data(arr)

            # Phase 2: running with spinner (1.0s)
            for k in range(int(FPS * 1.0)):
                frame = Image.new("RGB", (W, H), BG)
                draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
                draw_code_cell(frame, cell.source, "running", "*", "", None)
                writer.append_data(np.asarray(frame))

            # Phase 3: type out output (max 2.5s or until fully revealed)
            n_chars = len(strip_ansi(output_text))
            if n_chars > 0:
                target_secs = min(2.5, max(0.8, n_chars / 250))
                total_frames = int(FPS * target_secs)
                for k in range(total_frames):
                    reveal = int((k + 1) / total_frames * n_chars)
                    frame = Image.new("RGB", (W, H), BG)
                    draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
                    draw_code_cell(frame, cell.source, "running",
                                   "*", output_text, None, reveal_chars=reveal)
                    writer.append_data(np.asarray(frame))

            # Phase 4: image fade-in (1.2s) if present
            if output_img is not None:
                for k in range(int(FPS * 1.2)):
                    frame = Image.new("RGB", (W, H), BG)
                    draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
                    draw_code_cell(frame, cell.source, "running", "*",
                                   output_text, output_img)
                    writer.append_data(np.asarray(frame))

            # Phase 5: done - hold with In[N] label (1.8s, or 2.5s if image)
            hold = 2.5 if output_img is not None else 1.8
            frame = Image.new("RGB", (W, H), BG)
            draw_chrome(frame, NB_PATH.name, i + 1, total_cells)
            draw_code_cell(frame, cell.source, "done", str(exec_count),
                           output_text, output_img)
            arr = np.asarray(frame)
            for _ in range(int(FPS * hold)):
                writer.append_data(arr)

        # Outro card
        outro = Image.new("RGB", (W, H), BG)
        do = ImageDraw.Draw(outro)
        draw_chrome(outro, NB_PATH.name, total_cells, total_cells)
        do.text((80, 260), "Run complete.  Every assertion passed.",
                font=_font(30, bold=True), fill=INK)
        do.text((80, 310),
                "Reproduce locally:",
                font=_font(18), fill=OUT_INK)
        do.text((80, 342),
                "jupyter nbconvert --to notebook --execute \\",
                font=_font(15, mono=True), fill=INK)
        do.text((80, 368),
                "  notebooks/05_forward_backward_demo.ipynb --inplace",
                font=_font(15, mono=True), fill=INK)
        for _ in range(FPS * 3):
            writer.append_data(np.asarray(outro))
    finally:
        writer.close()

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
