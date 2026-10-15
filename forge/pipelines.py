"""Ba bước của SpriteForge: nhân vật -> chuyển động -> sprite sheet.

1. `generate_character`  SD1.5 pixel art txt2img (ý tưởng "anchor" của Pixel
                         Art Studio, nhưng prompt rút gọn để vừa 77 token CLIP).
2. `animate_video`       WAN 2.2 TI2V 5B: nhân vật + prompt hành động -> clip,
                         chọn N frame cách đều (luồng của Sprite Sheep).
3. `finalize`            tách nền, về pixel gốc, ép palette, đặt vào ô đều nhau,
                         xuất spritesheet.png + manifest.json + preview.gif.
"""
from __future__ import annotations

import logging
import random
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import matting, pixel, sheet, workflows
from .comfy import ComfyClient, ComfyError
from .config import VIEWS, Action, Settings

log = logging.getLogger(__name__)

# CLIP của SD1.5 chỉ đọc ~77 token: prompt ngắn, ý quan trọng đặt trước.
CHAR_PROMPT = ("pixel art, game sprite, {description}, single character, full body, "
               "{view}, standing pose, centered, plain flat {bg} background")
CHAR_NEGATIVE = ("multiple characters, cropped, cut off, out of frame, blurry, 3d render, "
                 "realistic, photo, text, watermark, signature, border, frame, scenery, "
                 "detailed background, shadow, gradient background")

WAN_PROMPT = ("{action}. {character}2D pixel art game sprite animation, {view}, static camera, "
              "no camera movement, the character stays in the center of the frame, "
              "plain flat {bg} background.")
# Từ NEGATIVO của Sprite Sheep (lóe sáng, vệt chuyển động bị tách nền thành lỗ
# thủng trên nhân vật) + các thứ phá nền một màu.
WAN_NEGATIVE = ("blurry, low quality, jpeg artifacts, watermark, text, letters, signature, "
                "extra limbs, deformed anatomy, changing design, flickering, "
                "camera motion, zoom, pan, cut, scene change, multiple characters, "
                "white flash, glow, sparkles, speed lines, motion trails, "
                "turning around, back view, shadow, ground, floor, scenery, 3d render, realistic")


def resolve_seed(*candidates: int | None) -> int:
    """Seed đầu tiên khác None; giá trị âm (-1) = ngẫu nhiên mỗi lần gọi."""
    for c in candidates:
        if c is not None:
            return random.randint(0, 2**31 - 1) if c < 0 else int(c)
    return random.randint(0, 2**31 - 1)


# ------------------------------------------------------------------ bước 1
def generate_character(client: ComfyClient, cfg: Settings, description: str,
                       seed: int | None = None) -> tuple[Image.Image, dict]:
    bg = cfg.bg_color if cfg.bg_color in ("green", "magenta", "blue", "white") else "green"
    seed = resolve_seed(seed, cfg.seed)
    prompt = CHAR_PROMPT.format(description=description.strip().rstrip("."),
                                view=VIEWS[cfg.view], bg=bg)
    graph = workflows.sd15_txt2img(cfg.checkpoint, prompt, CHAR_NEGATIVE, seed,
                                   steps=cfg.steps, cfg=cfg.cfg_scale,
                                   sampler=cfg.sampler, scheduler=cfg.scheduler)
    files = client.run(graph, workflows.SD15_OUTPUT, timeout_s=900)
    raw = client.fetch(files[0]).convert("RGB")
    return raw, {"seed": seed, "prompt": prompt}


def cutout(image: Image.Image) -> Image.Image:
    """Ảnh có nền trong suốt thật thì giữ nguyên; không thì tách nền đo ở góc."""
    rgba = image.convert("RGBA")
    if (np.asarray(rgba)[:, :, 3] < 128).mean() > 0.05:
        return rgba
    return matting.remove_background(image.convert("RGB"), "auto")


def prepare_character(image: Image.Image, cfg: Settings) -> tuple[Image.Image, np.ndarray]:
    return pixel.prepare_character(cutout(image), cfg.char_height, cfg.palette)


def palette_of(native: Image.Image, max_colors: int) -> np.ndarray:
    a = np.asarray(native.convert("RGBA"))
    colors = np.unique(a[a[:, :, 3] > 0][:, :3], axis=0)
    return colors if 0 < len(colors) <= max_colors else pixel.extract_palette(native, max_colors)


# ------------------------------------------------------------------ bước 2
def animate_video(client: ComfyClient, cfg: Settings, native: Image.Image,
                  palette: np.ndarray, action: Action, description: str | None = None,
                  seed: int | None = None) -> tuple[list[Image.Image], pixel.VideoLayout, dict]:
    wan = cfg.wan
    key = cfg.bg_color if cfg.bg_color in ("green", "magenta", "blue") else pixel.key_color_for(palette)
    canvas, layout = pixel.compose_video_input(native, int(wan["width"]), int(wan["height"]),
                                               matting.KEY_FILL[key], float(wan["char_fill"]))
    image_name = client.upload_image(canvas, f"spriteforge_{uuid.uuid4().hex[:12]}.png")

    length = workflows.wan_length(action.duration)
    indices = workflows.frame_indices(length, action.frames, float(wan["useful_fraction"]))
    character = f"The character: {description.strip().rstrip('.')}. " if description else ""
    prompt = WAN_PROMPT.format(action=action.prompt.strip().rstrip("."), character=character,
                               view=VIEWS[cfg.view], bg=key)
    seed = resolve_seed(action.seed, seed, cfg.seed)
    graph = workflows.wan22_i2v(
        image_name, prompt, WAN_NEGATIVE, seed, length,
        models={"unet": wan["unet"], "text_encoder": wan["text_encoder"], "vae": wan["vae"]},
        width=int(wan["width"]), height=int(wan["height"]), steps=int(wan["steps"]),
        cfg=float(wan["cfg"]), shift=float(wan["shift"]))
    files = client.run(graph, workflows.WAN_OUTPUT, timeout_s=float(wan["timeout_s"]))
    if len(files) < length:
        log.warning("WAN trả %d frame, ít hơn %d dự kiến", len(files), length)
        indices = workflows.frame_indices(len(files), action.frames, float(wan["useful_fraction"]))

    frames, empty = [], 0
    for i in indices:
        raw = client.fetch(files[i]).convert("RGB")
        if float(np.asarray(raw).std()) < 2.0:
            raise ComfyError(f"Frame {i} gần như một màu (đen/trống). WAN có thể bị lỗi fp16 "
                             "trên T4; xem kaggle/README.md mục 'Xử lý sự cố'.")
        native_frame = pixel.video_frame_to_native(matting.remove_background(raw, key),
                                                   layout, palette)
        empty += pixel.bbox(native_frame) is None
        frames.append(native_frame)
    if empty == len(frames):
        raise ComfyError("Mọi frame đều trống sau khi tách nền: nhân vật biến mất hoặc nền "
                         f"không phải màu {key}. Thử đặt \"bg_color\" khác.")
    if empty:
        log.warning("%d/%d frame trống sau khi tách nền", empty, len(frames))
    return frames, layout, {"seed": seed, "prompt": prompt, "clip_length": length,
                            "frame_indices": indices, "key_color": key, "scale_k": layout.k}


# ------------------------------------------------------------------ bước 3
def finalize(frames: list[Image.Image], layout: pixel.VideoLayout, action: Action,
             cfg: Settings, out_dir: str | Path, meta: dict | None = None) -> dict:
    cells, placed = pixel.place_in_cells(frames, layout, cfg.frame_size,
                                         cfg.ground_margin, action.align)
    if placed["clipped_px"]:
        log.warning("%s: %d pixel bị cắt ngoài ô %dpx (tăng frame_size hoặc giảm char_height)",
                    action.name, placed["clipped_px"], cfg.frame_size)
    if action.align == "fixed" and placed["drift_px"] > 0.15 * cfg.frame_size:
        log.warning("%s: nhân vật trôi ngang ~%.0fpx qua các frame; cân nhắc \"align\": \"track\"",
                    action.name, placed["drift_px"])
    paths = sheet.write_outputs(
        out_dir, action.name, cells, action.columns, cfg.fps, placed["anchor"],
        cfg.preview_scale, action.loop,
        extra={**(meta or {}), "align": action.align, "drift_px": placed["drift_px"],
               "clipped_px": placed["clipped_px"]})
    return {"paths": paths, **placed}
