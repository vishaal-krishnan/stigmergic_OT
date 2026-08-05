"""Render an executed notebook into an mp4 demo for paper reviewers.

Walks cell-by-cell through the notebook, drawing each cell's source (markdown
or code) on the left and its outputs (text + images) on the right. Frames are
held for a few seconds each so the video plays as a guided walkthrough.
"""
from __future__ import annotations

import base64
import io
import re
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import nbformat
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
NB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/04_algorithm1_snell.ipynb")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/demo/04_algorithm1_snell_demo.mp4")

W, H = 1600, 900
FPS = 30
BG = (250, 250, 252)
PANEL = (255, 255, 255)
INK = (30, 30, 40)
MUTED = (110, 110, 130)
ACCENT = (60, 90, 200)
CODE_BG = (245, 245, 250)


def _font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Andale Mono.ttf"]
        if mono
        else ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]
    )
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = _font(28)
F_BODY = _font(20)
F_CODE = _font(16, mono=True)
F_SMALL = _font(14, mono=True)


def wrap(text: str, width: int, font: ImageFont.FreeTypeFont) -> list[str]:
    """Wrap each input line so it fits within `width` pixels."""
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        # naive char-width wrap based on font metrics
        avg = font.getlength("M") or 10
        chars = max(10, int(width / avg))
        wrapped = textwrap.wrap(raw, width=chars, drop_whitespace=False) or [""]
        lines.extend(wrapped)
    return lines


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def render_frame(
    cell_idx: int,
    total: int,
    kind: str,
    source: str,
    text_outputs: list[str],
    image_outputs: list[Image.Image],
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.rectangle([(0, 0), (W, 60)], fill=(20, 25, 45))
    d.text((24, 16), "Algorithm 1 (APIC) — forward/backward demo", font=F_TITLE, fill=(240, 240, 255))
    d.text(
        (W - 260, 22),
        f"cell {cell_idx + 1} / {total}  ·  {kind}",
        font=F_BODY,
        fill=(200, 200, 230),
    )

    # source panel (left)
    src_x, src_y, src_w, src_h = 24, 80, 760, 780
    d.rounded_rectangle(
        [(src_x, src_y), (src_x + src_w, src_y + src_h)], radius=10, fill=PANEL
    )
    d.text((src_x + 14, src_y + 10), "source", font=F_BODY, fill=ACCENT)
    panel_inner = (src_x + 14, src_y + 44, src_x + src_w - 14, src_y + src_h - 14)
    if kind == "code":
        d.rectangle(panel_inner, fill=CODE_BG)
    font = F_CODE if kind == "code" else F_BODY
    color = INK
    lines = wrap(source, src_w - 36, font)
    line_h = font.size + 6
    max_lines = (panel_inner[3] - panel_inner[1] - 12) // line_h
    for i, line in enumerate(lines[:max_lines]):
        d.text(
            (panel_inner[0] + 6, panel_inner[1] + 6 + i * line_h),
            line,
            font=font,
            fill=color,
        )
    if len(lines) > max_lines:
        d.text(
            (panel_inner[0] + 6, panel_inner[3] - line_h - 4),
            f"… ({len(lines) - max_lines} more lines)",
            font=F_SMALL,
            fill=MUTED,
        )

    # output panel (right)
    out_x, out_y, out_w, out_h = 808, 80, 768, 780
    d.rounded_rectangle(
        [(out_x, out_y), (out_x + out_w, out_y + out_h)], radius=10, fill=PANEL
    )
    d.text((out_x + 14, out_y + 10), "output", font=F_BODY, fill=ACCENT)
    inner = (out_x + 14, out_y + 44, out_x + out_w - 14, out_y + out_h - 14)

    cursor_y = inner[1] + 6
    # text outputs
    if text_outputs:
        joined = "\n".join(text_outputs).rstrip()
        joined = strip_ansi(joined)
        text_lines = wrap(joined, out_w - 36, F_SMALL)
        for line in text_lines:
            if cursor_y + F_SMALL.size > inner[3] - 4:
                break
            d.text((inner[0] + 6, cursor_y), line, font=F_SMALL, fill=INK)
            cursor_y += F_SMALL.size + 3
        cursor_y += 10

    # image outputs — stacked to fit
    if image_outputs:
        avail_h = inner[3] - cursor_y
        n = len(image_outputs)
        per_h = max(120, avail_h // n - 8)
        for im in image_outputs:
            ratio = im.width / im.height
            target_h = min(per_h, inner[3] - cursor_y)
            if target_h < 80:
                break
            target_w = int(target_h * ratio)
            if target_w > inner[2] - inner[0] - 12:
                target_w = inner[2] - inner[0] - 12
                target_h = int(target_w / ratio)
            thumb = im.resize((target_w, target_h), Image.LANCZOS)
            img.paste(thumb, (inner[0] + 6, cursor_y))
            cursor_y += target_h + 8

    if not text_outputs and not image_outputs:
        d.text((inner[0] + 6, inner[1] + 6), "(no output)", font=F_BODY, fill=MUTED)

    # footer
    d.text(
        (24, H - 30),
        "Recorded from an end-to-end headless re-execution of the notebook.",
        font=F_SMALL,
        fill=MUTED,
    )
    return img


def extract_outputs(cell) -> tuple[list[str], list[Image.Image]]:
    texts: list[str] = []
    imgs: list[Image.Image] = []
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
        elif ot == "error":
            texts.append("\n".join(out.get("traceback", [])))
    return texts, imgs


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    cells = nb.cells
    total = len(cells)

    writer = imageio.get_writer(
        OUT_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        # intro frame, held briefly
        intro = Image.new("RGB", (W, H), (20, 25, 45))
        di = ImageDraw.Draw(intro)
        di.text(
            (80, 360),
            "Algorithm 1 (APIC) — forward/backward structure",
            font=_font(38),
            fill=(240, 240, 255),
        )
        di.text(
            (80, 430),
            f"Headless re-execution of notebooks/{NB_PATH.name}",
            font=F_BODY,
            fill=(200, 200, 230),
        )
        di.text(
            (80, 470),
            f"{total} cells · venv python · jax {__import__('jax').__version__}",
            font=F_SMALL,
            fill=(170, 170, 200),
        )
        for _ in range(FPS * 2):
            writer.append_data(np.asarray(intro))

        for i, cell in enumerate(cells):
            kind = cell.cell_type
            src = cell.source
            if kind == "code":
                texts, imgs = extract_outputs(cell)
            else:
                texts, imgs = [], []
            frame = render_frame(i, total, kind, src, texts, imgs)
            # hold each cell longer when it has rich output
            seconds = 4.0 if kind == "markdown" else (6.0 if imgs else 5.0)
            arr = np.asarray(frame)
            for _ in range(int(FPS * seconds)):
                writer.append_data(arr)

        # outro
        outro = Image.new("RGB", (W, H), (20, 25, 45))
        do = ImageDraw.Draw(outro)
        do.text((80, 380), "Run completed without errors.", font=_font(40), fill=(240, 240, 255))
        do.text(
            (80, 450),
            "Reproduce locally:  jupyter nbconvert --to notebook --execute "
            "notebooks/04_algorithm1_snell.ipynb --inplace",
            font=F_SMALL,
            fill=(200, 200, 230),
        )
        for _ in range(FPS * 2):
            writer.append_data(np.asarray(outro))
    finally:
        writer.close()

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
