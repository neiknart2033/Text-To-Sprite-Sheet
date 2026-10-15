"""Đóng gói sprite sheet + manifest.json + preview.gif.

Chuyển thể từ `pipeline/spritesheet.py` của Pixel Art Studio (GPLv3). Sửa:
  - bản gốc resize frame lệch cỡ về ô vuông (méo hình); ở đây mọi frame BẮT
    BUỘC cùng kích thước, sai là báo lỗi;
  - manifest có toạ độ từng frame + thông tin tái lập (seed, prompt...);
  - GIF giữ nền trong suốt, phóng NEAREST để xem rõ pixel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def pack(frames: list[Image.Image], columns: int) -> Image.Image:
    if not frames:
        raise ValueError("Không có frame nào để đóng gói")
    sizes = {f.size for f in frames}
    if len(sizes) != 1:
        raise ValueError(f"Các frame không cùng kích thước: {sorted(sizes)}")
    fw, fh = frames[0].size
    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGBA", (columns * fw, rows * fh), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame.convert("RGBA"), ((i % columns) * fw, (i // columns) * fh))
    return sheet


def manifest(action: str, frames: list[Image.Image], columns: int, fps: int,
             anchor: dict, image_name: str = "spritesheet.png", loop: bool = True,
             extra: dict | None = None) -> dict:
    fw, fh = frames[0].size
    columns = max(1, min(columns, len(frames)))
    return {
        "version": 1,
        "action": action,
        "image": image_name,
        "frameWidth": fw,
        "frameHeight": fh,
        "columns": columns,
        "rows": math.ceil(len(frames) / columns),
        "frameCount": len(frames),
        "fps": fps,
        "loop": loop,
        "anchor": anchor,
        "frames": [{"index": i, "x": (i % columns) * fw, "y": (i // columns) * fh,
                    "w": fw, "h": fh, "durationMs": round(1000 / fps)}
                   for i in range(len(frames))],
        **({"meta": extra} if extra else {}),
    }


def save_gif(frames: list[Image.Image], path: str | Path, fps: int, scale: int = 1) -> None:
    """GIF trong suốt, palette dùng chung, không dither."""
    rgba = [np.asarray(f.convert("RGBA")) for f in frames]
    opaque_px = np.concatenate([a[a[:, :, 3] > 0][:, :3] for a in rgba] or [np.zeros((0, 3))])
    colors = np.unique(opaque_px.astype(np.uint8), axis=0) if len(opaque_px) else np.zeros((0, 3), np.uint8)
    if len(colors) > 255:
        # Hiếm (palette nhân vật thường <= 32 màu): lượng tử chung về 255 màu
        strip = Image.fromarray(opaque_px.reshape(1, -1, 3).astype(np.uint8), "RGB")
        q = strip.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        colors = np.array(q.getpalette()[:765], np.uint8).reshape(-1, 3)
    transparent = len(colors)
    palette = list(colors.flatten()) + [0, 0, 0] * (256 - len(colors))

    lut = {tuple(c): i for i, c in enumerate(colors.tolist())}
    images = []
    for a in rgba:
        idx = np.full(a.shape[:2], transparent, np.uint8)
        mask = a[:, :, 3] > 0
        px = a[mask][:, :3]
        if len(px):
            keys = [lut.get(tuple(p)) for p in px.tolist()]
            if any(k is None for k in keys):  # chỉ xảy ra khi đã lượng tử lại ở trên
                d = ((px[:, None, :].astype(np.int32) - colors[None].astype(np.int32)) ** 2).sum(-1)
                keys = d.argmin(1).tolist()
            idx[mask] = np.array(keys, np.uint8)
        im = Image.fromarray(idx, "P")
        im.putpalette(palette)
        if scale > 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        images.append(im)

    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps),
                   loop=0, transparency=transparent, disposal=2, optimize=False)


def write_outputs(out_dir: str | Path, action: str, cells: list[Image.Image], columns: int,
                  fps: int, anchor: dict, preview_scale: int = 3, loop: bool = True,
                  extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "spritesheet.png"
    manifest_path = out_dir / "manifest.json"
    gif_path = out_dir / "preview.gif"

    pack(cells, columns).save(sheet_path)
    data = manifest(action, cells, columns, fps, anchor, sheet_path.name, loop, extra)
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    save_gif(cells, gif_path, fps, preview_scale)
    return {"spritesheet": sheet_path, "manifest": manifest_path, "preview": gif_path}
