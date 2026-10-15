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
import math
import random
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import matting, pixel, poses, sheet, workflows
from .comfy import ComfyClient, ComfyError
from .config import VIEWS, Action, Settings

log = logging.getLogger(__name__)

# CLIP của SD1.5 chỉ đọc ~77 token: prompt ngắn, ý quan trọng đặt trước.
# Nền TRẮNG, không trọng số. Hai lần chạy thật đã sai theo hai kiểu:
#   - "green background" thường: "dark samurai, black armor" lấn át -> SD vẽ nền đen;
#   - "(green background:1.4)": màu xanh lan sang quần áo (kunoichi tím -> áo xanh lá).
# Nền trắng + tách nền loang từ mép (cutout_flat) không cần màu nền đặc biệt.
CHAR_BG = "white"
CHAR_SIZE = (512, 768)  # ảnh dọc: đủ chỗ cho cả người, đỡ bị cắt chân
CHAR_PROMPT = ("pixel art, game sprite, {description}, single character, full body from head "
               "to feet, ({view}:1.2), standing pose, centered, simple white background")
CHAR_NEGATIVE = ("multiple characters, cropped, cut off feet, cut off head, out of frame, "
                 "blurry, 3d render, realistic, photo, text, watermark, signature, border, frame, "
                 "scenery, detailed background, gradient background, shadow, {other_view}")
OTHER_VIEW = {"side": "front view, facing the viewer", "front": "side view, profile"}
CHAR_ATTEMPTS = 3  # số lần gen lại khi nền không đều hoặc nhân vật chạm mép ảnh

WAN_PROMPT = ("{action}. {character}2D pixel art game sprite animation, {view}, static camera, "
              "fixed framing, no zoom, the character keeps the same size and stays in the "
              "center of the frame, plain flat {bg} background.")
# Từ NEGATIVO của Sprite Sheep (lóe sáng, vệt chuyển động bị tách nền thành lỗ thủng)
# + các lỗi thấy trong clip thật: zoom vào nhân vật, biến hình, tan biến.
WAN_NEGATIVE = ("blurry, low quality, jpeg artifacts, watermark, text, letters, signature, "
                "extra limbs, deformed anatomy, changing design, flickering, morphing, "
                "transformation, melting, disappearing, camera motion, zoom in, zoom out, "
                "close-up, camera shake, scale change, pan, cut, scene change, multiple characters, "
                "white flash, glow, sparkles, speed lines, motion trails, turning around, "
                "back view, shadow, ground, floor, scenery, 3d render, realistic")
FRAME_SEARCH = 2  # mỗi frame cần lấy được chọn frame "lành" nhất trong ±2 frame của clip

# Engine pose: mô tả nhân vật + động tác; khung xương quyết định tư thế, IP-Adapter giữ ngoại hình.
POSE_PROMPT = ("pixel art, game sprite, {character}{action}, single character, full body, "
               "({view}:1.2), centered, simple white background")


def resolve_seed(*candidates: int | None) -> int:
    """Seed đầu tiên khác None; giá trị âm (-1) = ngẫu nhiên mỗi lần gọi."""
    for c in candidates:
        if c is not None:
            return random.randint(0, 2**31 - 1) if c < 0 else int(c)
    return random.randint(0, 2**31 - 1)


# ------------------------------------------------------------------ bước 1
def generate_character(client: ComfyClient, cfg: Settings, description: str,
                       seed: int | None = None) -> tuple[Image.Image, dict]:
    """SD1.5 txt2img nền trắng; nền không đều hoặc nhân vật chạm mép -> gen lại seed khác."""
    prompt = CHAR_PROMPT.format(description=description.strip().rstrip("."), view=VIEWS[cfg.view])
    negative = CHAR_NEGATIVE.format(other_view=OTHER_VIEW[cfg.view])
    first = resolve_seed(seed, cfg.seed)
    # Engine pose: vẽ nhân vật theo khung xương đứng của chính thư viện tư thế -> đúng
    # hướng nhìn ngang (SD hay tự vẽ nhìn thẳng) và cùng khung hình với các frame sau.
    pose_name = None
    if cfg.engine == "pose" and cfg.pose.get("controlnet"):
        size = int(cfg.pose["size"])
        pose_name = client.upload_image(poses.render(poses.NEUTRAL, size),
                                        f"spriteforge_pose_{uuid.uuid4().hex[:12]}.png")
    tries, best = [], None
    for attempt in range(CHAR_ATTEMPTS):
        s = first + attempt
        if pose_name:
            graph = workflows.sd15_pose(
                cfg.checkpoint_file or cfg.checkpoint, prompt, negative, s, pose_name,
                cfg.pose["controlnet"], float(cfg.pose["strength"]), size=size,
                steps=cfg.steps, cfg=cfg.cfg_scale, sampler=cfg.sampler, scheduler=cfg.scheduler)
        else:
            graph = workflows.sd15_txt2img(cfg.checkpoint_file or cfg.checkpoint, prompt, negative, s,
                                           width=CHAR_SIZE[0], height=CHAR_SIZE[1], steps=cfg.steps,
                                           cfg=cfg.cfg_scale, sampler=cfg.sampler,
                                           scheduler=cfg.scheduler)
        raw = client.fetch(client.run(graph, workflows.SD15_OUTPUT, timeout_s=900)[0]).convert("RGB")
        bg_ok, measured = matting.matches_key(raw, CHAR_BG)
        inside = not pixel.touches_edge(matting.cutout_flat(raw))
        tries.append({"seed": s, "background": measured, "background_ok": bg_ok,
                      "inside_frame": inside})
        if best is None or bg_ok + inside > best[0]:
            best = (bg_ok + inside, raw, s)
        if bg_ok and inside:
            break
        problems = ([] if bg_ok else [f"nền {measured} không phải nền trắng đều"]) + \
                   ([] if inside else ["nhân vật bị cắt ở mép ảnh"])
        log.warning("Nhân vật lần %d (seed %d): %s -> %s", attempt + 1, s, "; ".join(problems),
                    "thử seed khác" if attempt + 1 < CHAR_ATTEMPTS else "dùng ảnh tốt nhất")
    _, raw, s = best
    return raw, {"seed": s, "prompt": prompt, "negative": negative, "attempts": tries}


def cutout(image: Image.Image) -> Image.Image:
    """Ảnh có nền trong suốt thật thì giữ nguyên; không thì tách nền loang từ mép ảnh."""
    rgba = image.convert("RGBA")
    if (np.asarray(rgba)[:, :, 3] < 128).mean() > 0.05:
        return rgba
    return matting.cutout_flat(image.convert("RGB"))


def prepare_character(image: Image.Image, cfg: Settings,
                      user_image: bool = False) -> tuple[Image.Image, np.ndarray]:
    """Ảnh nhân vật -> sprite pixel gốc + palette.

    user_image (mode "image"): giữ đúng sprite mẫu của người dùng. Ảnh phóng to (VD x4)
    được thu về pixel gốc theo đúng lưới; không ép về char_height/palette mặc định trừ
    khi jobs.json khai báo rõ (hoặc nhân vật quá lớn so với ô frame_size).
    """
    cut = cutout(image)
    height, colors = cfg.char_height, cfg.palette
    if user_image:
        k = pixel.detect_block_size(cut)
        if k > 1:
            cut = pixel.resize_premultiplied(cut, (cut.width // k, cut.height // k))
            log.info("Ảnh mẫu đang phóng x%d -> thu về pixel gốc %dx%d", k, cut.width, cut.height)
        if not cfg.char_height_explicit:
            box = pixel.bbox(cut)
            native_h = (box[3] - box[1]) if box else height
            height = min(native_h, round(cfg.frame_size * 0.8))
        if not cfg.palette_explicit:
            colors = 255
    native, palette = pixel.prepare_character(cut, height, colors)
    holes = pixel.hole_ratio(native)
    if holes > 0.04:
        log.warning("Nhân vật bị thủng %.0f%% thân sau khi tách nền: animation sẽ lỗi. Xem "
                    "character_preview.png; đổi seed hoặc dùng mode \"image\" với ảnh nền trong suốt.",
                    holes * 100)
    return native, palette


def palette_of(native: Image.Image, max_colors: int) -> np.ndarray:
    a = np.asarray(native.convert("RGBA"))
    colors = np.unique(a[a[:, :, 3] > 0][:, :3], axis=0)
    return colors if 0 < len(colors) <= max_colors else pixel.extract_palette(native, max_colors)


# ------------------------------------------------------------------ bước 2
def animate_video(client: ComfyClient, cfg: Settings, native: Image.Image,
                  palette: np.ndarray, action: Action, description: str | None = None,
                  seed: int | None = None, debug_dir: str | Path | None = None,
                  ) -> tuple[list[Image.Image], pixel.VideoLayout, dict]:
    wan = cfg.wan
    key = cfg.bg_color if cfg.bg_color in ("green", "magenta", "blue") else pixel.key_color_for(palette)
    tolerance = matting.key_tolerance(matting.KEY_FILL[key], palette)
    canvas, layout = pixel.compose_video_input(native, int(wan["width"]), int(wan["height"]),
                                               matting.KEY_FILL[key], float(wan["char_fill"]))
    image_name = client.upload_image(canvas, f"spriteforge_{uuid.uuid4().hex[:12]}.png")

    length = workflows.wan_length(action.duration)
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
    targets = workflows.frame_indices(len(files), action.frames, float(wan["useful_fraction"]))

    raw_cache: dict[int, Image.Image] = {}
    native_cache: dict[int, Image.Image] = {}

    def raw(i: int) -> Image.Image:
        if i not in raw_cache:
            raw_cache[i] = client.fetch(files[i]).convert("RGB")
        return raw_cache[i]

    def native_frame(i: int) -> Image.Image:
        if i not in native_cache:
            if float(np.asarray(raw(i)).std()) < 2.0:
                raise ComfyError(f"Frame {i} gần như một màu (đen/trống). WAN có thể bị lỗi fp16 "
                                 "trên T4; xem kaggle/README.md mục 'Xử lý sự cố'.")
            cut = matting.remove_background(raw(i), key, tolerance=tolerance)
            native_cache[i] = pixel.video_frame_to_native(cut, layout, palette)
        return native_cache[i]

    if debug_dir:  # ảnh đưa vào WAN + toàn bộ clip WAN trả về, để soi lỗi
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(debug_dir / "input.png")
        _save_clip([raw(i) for i in range(len(files))], debug_dir / "clip.gif")

    chosen = _pick_frames(targets, len(files), native_frame)
    frames = [native_frame(i) for i in chosen]
    empty = sum(pixel.bbox(f) is None for f in frames)
    if empty == len(frames):
        raise ComfyError("Mọi frame đều trống sau khi tách nền: nhân vật biến mất hoặc nền "
                         f"không phải màu {key}. Thử đặt \"bg_color\" khác.")
    if empty:
        log.warning("%d/%d frame trống sau khi tách nền", empty, len(frames))
    return frames, layout, {"engine": "video", "seed": seed, "prompt": prompt, "clip_length": len(files),
                            "target_indices": targets, "frame_indices": chosen, "key_color": key,
                            "key_tolerance": round(tolerance, 1), "scale_k": layout.k}


def _pick_frames(targets: list[int], n_clip: int, native_frame) -> list[int]:
    """Với mỗi vị trí cần lấy, chọn frame "lành" nhất trong ±FRAME_SEARCH frame.

    Clip thật có frame hỏng rải rác: khung nền lệch màu, nhân vật mờ/vỡ mảnh, phóng to
    đột ngột. Frame lành = diện tích gần với nhân vật gốc (frame 0) và ít mảnh rời. Thứ
    tự frame luôn tăng dần; trong ±2 frame (0.08 giây) chuyển động thật đổi rất ít.
    """
    ref_area = pixel.frame_stats(native_frame(0))[0]

    def badness(i: int) -> float:
        area, parts = pixel.frame_stats(native_frame(i))
        if not area or not ref_area:
            return float("inf")
        return abs(math.log(area / ref_area)) + 0.15 * max(0, parts - 1)

    chosen: list[int] = []
    for t in targets:
        lo = max(t - FRAME_SEARCH, chosen[-1] + 1 if chosen else 0)
        hi = min(t + FRAME_SEARCH, n_clip - 1)
        if lo > hi:  # hết chỗ (clip ngắn hơn số frame cần): giữ vị trí gần nhất
            chosen.append(min(lo, n_clip - 1))
            continue
        chosen.append(min(range(lo, hi + 1), key=lambda i: (badness(i), abs(i - t))))
    return chosen


def reference_image(native: Image.Image, size: int) -> Image.Image:
    """Ảnh mẫu cho IP-Adapter: nhân vật phóng NEAREST, đứng đúng chỗ khung xương, nền trắng."""
    k = max(1, round(poses.standing_height_px(size) / max(1, native.height)))
    up = pixel.upscale(native, k)
    hip_x, ground_y = poses.ground_px(size)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(up, (round(hip_x - up.width / 2), round(ground_y - up.height)), up)
    return canvas


def animate_pose(client: ComfyClient, cfg: Settings, native: Image.Image,
                 palette: np.ndarray, action: Action, description: str | None = None,
                 seed: int | None = None, debug_dir: str | Path | None = None,
                 ) -> tuple[list[Image.Image], pixel.VideoLayout, dict]:
    """Engine pose: mỗi frame = SD1.5 + ControlNet OpenPose (+ IP-Adapter) theo thư viện tư thế.

    Mọi frame cùng seed, cùng khung xương tỉ lệ, cùng đường nền -> không zoom, không trôi,
    số frame đúng chính xác. Ảnh SD (nền trắng) được tách nền và thu về pixel gốc theo
    MỘT tỉ lệ chung cho mọi action (chiều cao đứng của nhân vật = chiều cao sprite gốc).
    """
    p = cfg.pose
    size = int(p["size"])
    keys = poses.sequence(action.pose or action.name, action.frames)
    seed = resolve_seed(action.seed, seed, cfg.seed)
    ref_name = None
    if p.get("ipadapter"):
        ref_name = client.upload_image(reference_image(native, size),
                                       f"spriteforge_ref_{uuid.uuid4().hex[:12]}.png")
    character = f"{description.strip().rstrip('.')}, " if description else ""
    prompt = POSE_PROMPT.format(character=character, action=action.prompt.strip().rstrip("."),
                                view=VIEWS[cfg.view])
    negative = CHAR_NEGATIVE.format(other_view=OTHER_VIEW[cfg.view])

    scale = native.height / poses.standing_height_px(size)
    side = max(8, round(size * scale))
    hip_x, ground_y = poses.ground_px(size)
    layout = pixel.VideoLayout(k=1, width=side, height=side,
                               center_x=hip_x * scale, ground_y=ground_y * scale)
    frames, skeletons, raws = [], [], []
    for key in keys:
        skeleton = poses.render(key, size)
        pose_name = client.upload_image(skeleton, f"spriteforge_pose_{uuid.uuid4().hex[:12]}.png")
        graph = workflows.sd15_pose(
            cfg.checkpoint_file or cfg.checkpoint, prompt, negative, seed, pose_name,
            p["controlnet"], float(p["strength"]), reference_image=ref_name,
            ipadapter_weight=float(p["ipadapter_weight"]), ipadapter_preset=str(p["ipadapter_preset"]),
            size=size, steps=cfg.steps, cfg=cfg.cfg_scale, sampler=cfg.sampler,
            scheduler=cfg.scheduler)
        raw = client.fetch(client.run(graph, workflows.POSE_OUTPUT, timeout_s=900)[0]).convert("RGB")
        frames.append(pixel.frame_to_native(matting.cutout_flat(raw), (side, side), palette))
        skeletons.append(skeleton)
        raws.append(raw)
    if debug_dir:  # khung xương + ảnh SD gốc từng frame, để soi lỗi
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        _strip(skeletons, debug_dir / "poses.png")
        _strip(raws, debug_dir / "raw.png")
    empty = sum(pixel.bbox(f) is None for f in frames)
    if empty == len(frames):
        raise ComfyError("Mọi frame đều trống sau khi tách nền: SD không vẽ nhân vật trên nền "
                         "trắng. Xem debug/raw.png.")
    if empty:
        log.warning("%d/%d frame trống sau khi tách nền", empty, len(frames))
    return frames, layout, {"engine": "pose", "pose": action.pose, "seed": seed, "prompt": prompt,
                            "ipadapter": bool(ref_name), "scale": round(scale, 4)}


def _strip(images: list[Image.Image], path: Path, height: int = 256) -> None:
    """Ghép các ảnh thành một hàng (thu nhỏ) để xem nhanh."""
    small = [im.resize((height * im.width // im.height, height)) for im in images]
    out = Image.new("RGB", (sum(im.width for im in small), height), (0, 0, 0))
    x = 0
    for im in small:
        out.paste(im, (x, 0))
        x += im.width
    out.save(path)


def _save_clip(clip: list[Image.Image], path: Path) -> None:
    small = [f.resize((f.width // 2, f.height // 2), Image.BILINEAR) for f in clip]
    small[0].save(path, save_all=True, append_images=small[1:],
                  duration=round(1000 / workflows.WAN_FPS), loop=0)


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
