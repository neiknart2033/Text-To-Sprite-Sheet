"""Ba bước của SpriteForge: nhân vật -> chuyển động -> sprite sheet.

1. `generate_character_edit`  FLUX.2 klein tạo nhân vật từ mô tả (engine "edit", mặc định).
   `generate_character`       SD1.5 pixel art txt2img (engine "pose"/"video"; ý tưởng
                              "anchor" của Pixel Art Studio).
2. `animate_edit`        FLUX.2 klein vẽ lại "con rối" (sprite mẫu uốn theo khung xương)
                         theo chính sprite mẫu (mặc định).
   `animate_pose`        SD1.5 + ControlNet OpenPose + IP-Adapter theo khung xương.
   `animate_video`       WAN 2.2 TI2V 5B: nhân vật + prompt hành động -> clip,
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
from scipy import ndimage

from . import cleanup, grid, matting, pixel, poses, puppet, sheet, workflows
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
# Chạy thật: checkpoint pixel art hay vẽ 3 nhân vật kiểu "character sheet", đĩa tròn sau
# lưng, vệt đất dưới chân -> cấm thẳng trong negative (ComfyUI tự chia prompt > 77 token).
CHAR_NEGATIVE = ("multiple characters, character sheet, multiple views, turnaround, reference sheet, "
                 "cropped, cut off feet, cut off head, out of frame, circle, sun, halo, round backdrop, "
                 "emblem, logo, banner, building, ground, floor, platform, "
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
# Bảng tư thế: cả action trong một ảnh -> prompt nói rõ "cùng một nhân vật lặp lại", và
# KHÔNG cấm "character sheet / multiple characters" như khi vẽ lẻ: ở đây đó là thứ cần vẽ.
GRID_PROMPT = ("pixel art, game sprite sheet, the same character repeated in every frame, "
               "{character}{action}, full body, ({view}:1.2), simple white background")
GRID_NEGATIVE = ("different characters, inconsistent design, different clothes, different colors, "
                 "cropped, cut off feet, cut off head, circle, sun, halo, round backdrop, emblem, "
                 "logo, banner, building, ground, floor, platform, grid lines, panel border, frame, "
                 "blurry, 3d render, realistic, photo, text, watermark, signature, scenery, "
                 "detailed background, gradient background, shadow, {other_view}")
GRID_TIMEOUT_S = 1800  # bảng 3x3 ô 512px trên T4 ~2 phút; chừa rộng cho GPU đang bận


# Engine edit (FLUX.2 klein): bộ mã hoá chữ là Qwen3 4B, đọc được câu dài tự nhiên.
EDIT_VIEWS = {"side": "strict side view (profile), facing right",
              "front": "front view, facing the viewer"}
EDIT_CHAR_SIZE = (768, 1024)  # ảnh dọc, chừa chỗ quanh nhân vật
EDIT_CHAR_PROMPT = ("Pixel art game sprite of {description}. Full body from head to feet, standing "
                    "idle pose, {view}, a single character centered with empty space around it, "
                    "plain pure white background, clean retro 16-bit pixel art with crisp pixels and "
                    "a dark outline, no shadow, no ground, no text, no border.")
# Ảnh 1 = con rối (đúng tư thế, khớp còn thô), ảnh 2 = sprite mẫu (đúng ngoại hình). Thử
# thật: klein giữ rất chặt bố cục + tư thế của ảnh 1 -> ảnh 1 phải là con rối.
EDIT_FRAME_PROMPT = (
    "Image 1 is a rough animation frame of a 2D pixel art game character, made by cutting the "
    "character from image 2 into body parts and rotating them into a new pose: {action}. Redraw "
    "image 1 as a clean, finished pixel art sprite. Keep exactly the same pose, body position, "
    "size and facing direction as image 1. Repair broken joints, gaps and stray pixels so the "
    "anatomy looks natural. The character must look exactly like image 2: same design, outfit, "
    "colors, proportions and weapon. {character}Single character, full body, {view}, plain pure "
    "white background, no shadow, no ground, no text.")


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
        cut = matting.cutout_flat(raw)
        clean_ok, cleaned = True, cut
        if pose_name:
            # Bỏ nhân vật thừa / đĩa tròn / vệt đất theo khung xương. Ảnh phải bỏ nhiều
            # (>15%) thì vẫn dùng được nhưng ưu tiên thử seed khác cho sạch hẳn.
            cleaned, stats = cleanup.clean_figure(cut, poses.NEUTRAL, measured)
            removed = stats["removed_figures_px"] + stats["peeled_px"]
            clean_ok = removed <= 0.15 * max(1, int((np.asarray(cut)[:, :, 3] > 0).sum()))
        inside = not pixel.touches_edge(cleaned)
        tries.append({"seed": s, "background": measured, "background_ok": bg_ok,
                      "inside_frame": inside, "clean": clean_ok})
        score = bg_ok + inside + clean_ok
        if best is None or score > best[0]:
            best = (score, raw, s, cleaned)
        if bg_ok and inside and clean_ok:
            break
        problems = ([] if bg_ok else [f"nền {measured} không phải nền trắng đều"]) + \
                   ([] if inside else ["nhân vật bị cắt ở mép ảnh"]) + \
                   ([] if clean_ok else ["SD vẽ thêm nhân vật/phông nền (đã tự xoá)"])
        log.warning("Nhân vật lần %d (seed %d): %s -> %s", attempt + 1, s, "; ".join(problems),
                    "thử seed khác" if attempt + 1 < CHAR_ATTEMPTS else "dùng ảnh tốt nhất")
    _, raw, s, cleaned = best
    # "cutout": nhân vật đã tách nền + làm sạch, dùng thay cho raw khi chuẩn bị sprite
    return raw, {"seed": s, "prompt": prompt, "negative": negative, "attempts": tries,
                 "cutout": cleaned}


def _klein_models(cfg: Settings) -> dict:
    return {k: cfg.edit[k] for k in ("unet", "text_encoder", "vae")}


def _klein_draw(client: ComfyClient, cfg: Settings, prompt: str, seed: int, width: int,
                height: int, **kw) -> Image.Image:
    e = cfg.edit
    graph = workflows.klein(_klein_models(cfg), prompt, seed, width, height, steps=int(e["steps"]),
                            cfg=float(e["cfg"]), **kw)
    files = client.run(graph, workflows.KLEIN_OUTPUT, timeout_s=float(e["timeout_s"]))
    raw = client.fetch(files[0]).convert("RGB")
    if float(np.asarray(raw).std()) < 2.0:
        raise ComfyError("FLUX.2 klein trả về ảnh gần như một màu (đen/trống): lỗi fp16 trên GPU "
                         "này? Xem kaggle/README.md mục 'Xử lý sự cố'.")
    return raw


def _main_figure(rgba: Image.Image, gap: float = 0.03) -> tuple[Image.Image, float]:
    """Giữ mảng lớn nhất + mảng nằm sát nó (kiếm tách rời vài pixel); trả về (ảnh, tỉ lệ
    pixel của phần giữ lại). Tỉ lệ thấp = ảnh có nhiều nhân vật / nhiều thứ rời nhau."""
    a = np.asarray(rgba.convert("RGBA")).copy()
    mask = a[:, :, 3] > 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n <= 1:
        return Image.fromarray(a, "RGBA"), 1.0
    sizes = ndimage.sum_labels(mask, labels, np.arange(1, n + 1))
    main = labels == int(np.argmax(sizes)) + 1
    near = ndimage.distance_transform_edt(~main) <= gap * a.shape[0]
    keep_ids = np.unique(labels[near & mask])
    keep = np.isin(labels, keep_ids[keep_ids > 0])
    a[~keep] = 0
    return Image.fromarray(a, "RGBA"), float(keep.sum() / mask.sum())


def generate_character_edit(client: ComfyClient, cfg: Settings, description: str,
                            seed: int | None = None) -> tuple[Image.Image, dict]:
    """FLUX.2 klein tạo nhân vật từ mô tả, nền trắng. Nền không đều, nhân vật chạm mép hoặc
    ảnh có nhiều nhân vật -> gen lại seed khác."""
    prompt = EDIT_CHAR_PROMPT.format(description=description.strip().rstrip("."),
                                     view=EDIT_VIEWS[cfg.view])
    first = resolve_seed(seed, cfg.seed)
    tries, best = [], None
    for attempt in range(CHAR_ATTEMPTS):
        s = first + attempt
        raw = _klein_draw(client, cfg, prompt, s, *EDIT_CHAR_SIZE)
        bg_ok, measured = matting.matches_key(raw, CHAR_BG)
        figure, share = _main_figure(matting.cutout_flat(raw))
        single, inside = share >= 0.85, not pixel.touches_edge(figure)
        tries.append({"seed": s, "background": measured, "background_ok": bg_ok,
                      "inside_frame": inside, "single": single})
        score = bg_ok + inside + single
        if best is None or score > best[0]:
            best = (score, raw, s, figure)
        if bg_ok and inside and single:
            break
        problems = ([] if bg_ok else [f"nền {measured} không phải nền trắng đều"]) + \
                   ([] if inside else ["nhân vật bị cắt ở mép ảnh"]) + \
                   ([] if single else [f"có nhiều nhân vật/mảng rời (nhân vật chính {share:.0%})"])
        log.warning("Nhân vật lần %d (seed %d): %s -> %s", attempt + 1, s, "; ".join(problems),
                    "thử seed khác" if attempt + 1 < CHAR_ATTEMPTS else "dùng ảnh tốt nhất")
    _, raw, s, figure = best
    return raw, {"engine": "edit", "seed": s, "prompt": prompt, "attempts": tries, "cutout": figure}


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


def animate_edit(client: ComfyClient, cfg: Settings, native: Image.Image,
                 palette: np.ndarray, action: Action, description: str | None = None,
                 seed: int | None = None, debug_dir: str | Path | None = None,
                 ) -> tuple[list[Image.Image], pixel.VideoLayout, dict]:
    """Engine edit: mỗi frame = FLUX.2 klein vẽ lại con rối (ảnh 1) theo sprite mẫu (ảnh 2).

    Con rối (forge/puppet.py) đã đúng tư thế và đúng màu, chỉ còn thô ở khớp -> model chỉ
    phải "sửa cho sạch", việc mà model sửa ảnh làm tốt nhất. Mỗi frame được kiểm tra: hình
    bóng phải khớp con rối (không phải tư thế đứng của ảnh mẫu); không khớp thì vẽ lại kiểu
    img2img từ chính con rối (`fallback_denoise`); vẫn không khớp thì dùng thẳng con rối.
    Mọi frame cùng seed, cùng chỗ đặt, cùng tỉ lệ xk -> thu về pixel gốc khớp đúng lưới.
    """
    e = cfg.edit
    size = int(e["size"])
    fit = 16 * math.ceil(max(native.height / poses.UNIT, native.width / 0.8) / 16)
    if fit > size:  # sprite to hơn ảnh model vẽ: phóng x1 vẫn không vừa
        log.warning("Sprite %dx%d quá lớn cho edit.size %d -> dùng %d", *native.size, size, fit)
        size = fit
    keys = poses.sequence(action.pose or action.name, action.frames)
    seed = resolve_seed(action.seed, seed, cfg.seed)
    rig = puppet.Rig(native, size)
    pl = rig.pl
    rest = np.asarray(rig.canvas)[:, :, 3] > 0
    tag = uuid.uuid4().hex[:12]
    ref_name = client.upload_image(puppet.on_white(rig.canvas), f"spriteforge_ref_{tag}.png")
    character = f"The character: {description.strip().rstrip('.')}. " if description else ""
    prompt = EDIT_FRAME_PROMPT.format(action=action.prompt.strip().rstrip("."),
                                      character=character, view=EDIT_VIEWS[cfg.view])
    attempts = (("edit", 1.0), ("img2img", float(e["fallback_denoise"])))
    layout = pixel.VideoLayout(k=pl.k, width=size, height=size,
                               center_x=pl.hip_x / pl.k, ground_y=pl.ground_y / pl.k)
    frames, puppets, raws, cleans, checks = [], [], [], [], []
    for i, key in enumerate(keys):
        pup = rig.pose(key)
        pup_mask = np.asarray(pup)[:, :, 3] > 0
        pup_name = client.upload_image(puppet.on_white(pup), f"spriteforge_puppet_{tag}_{i}.png")
        best = None
        for mode, denoise in attempts:
            raw = _klein_draw(client, cfg, prompt, seed, size, size,
                              references=[pup_name, ref_name], init_image=pup_name, denoise=denoise)
            cut = puppet.keep_near(matting.cutout_flat(raw), pup_mask, pl)
            ok, follow, gain = puppet.follows_pose(np.asarray(cut)[:, :, 3] > 0, pup_mask, rest,
                                                   pl.unit, float(e["follow_threshold"]))
            cand = {"ok": ok, "follow": follow, "gain": gain, "mode": mode, "raw": raw, "cut": cut}
            if best is None or (ok, follow) > (best["ok"], best["follow"]):
                best = cand
            if ok:
                break
        raws.append(best["raw"])  # debug luôn giữ ảnh model vẽ, kể cả khi không dùng
        if not best["ok"]:
            # Lớp lùi cuối: cả hai cách vẽ đều lệch tư thế -> dùng thẳng con rối. Thô ở khớp
            # nhưng chắc chắn đúng nhân vật (pixel của chính sprite mẫu) và đúng tư thế.
            log.warning("%s frame %d: ảnh vẽ lại không khớp tư thế (khớp %.0f%%) -> dùng con rối",
                        action.name, i + 1, best["follow"] * 100)
            best = {**best, "mode": "puppet", "cut": pup}
        checks.append({"mode": best["mode"], "follow": round(best["follow"], 3),
                       "moved_covered": round(best["gain"], 3), "ok": best["ok"]})
        frames.append(pixel.video_frame_to_native(best["cut"], layout, palette))
        puppets.append(puppet.on_white(pup))
        cleans.append(best["cut"])
    if debug_dir:  # con rối (ảnh 1 đưa vào model) + ảnh model vẽ + sau tách nền, từng frame
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        puppet.on_white(rig.canvas).save(debug_dir / "reference.png")
        _strip(puppets, debug_dir / "puppet.png")
        _strip(raws, debug_dir / "raw.png")
        _strip([_on_gray(c) for c in cleans], debug_dir / "clean.png")
    empty = sum(pixel.bbox(f) is None for f in frames)
    if empty == len(frames):
        raise ComfyError("Mọi frame đều trống sau khi tách nền: model không vẽ nhân vật trên nền "
                         "trắng. Xem debug/raw.png.")
    if empty:
        log.warning("%d/%d frame trống sau khi tách nền", empty, len(frames))
    for mode, what in (("img2img", "phải vẽ lại kiểu img2img từ con rối"),
                       ("puppet", "dùng thẳng con rối (model vẽ lệch tư thế)")):
        count = sum(c["mode"] == mode for c in checks)
        if count:
            log.info("%s: %d/%d frame %s", action.name, count, len(checks), what)
    return frames, layout, {"engine": "edit", "pose": action.pose, "seed": seed, "prompt": prompt,
                            "scale_k": pl.k, "frames": checks}


def animate_pose(client: ComfyClient, cfg: Settings, native: Image.Image,
                 palette: np.ndarray, action: Action, description: str | None = None,
                 seed: int | None = None, debug_dir: str | Path | None = None,
                 ) -> tuple[list[Image.Image], pixel.VideoLayout, dict]:
    """Engine pose: SD1.5 + ControlNet OpenPose (+ IP-Adapter) theo thư viện tư thế.

    layout "grid" (mặc định): cả action vẽ trong MỘT ảnh, nhân vật mẫu giữ nguyên ở ô đầu
    (forge/grid.py) -> các frame giống nhau hơn hẳn vẽ lẻ. Bảng lỗi (VD hết VRAM) thì tự
    chuyển sang vẽ từng frame. layout "single": mỗi frame một ảnh, cùng seed.
    Mọi frame cùng khung xương tỉ lệ, cùng đường nền -> không zoom, không trôi, số frame
    đúng chính xác. Ảnh SD (nền trắng) được tách nền, làm sạch theo khung xương, rồi thu
    về pixel gốc theo MỘT tỉ lệ chung cho mọi action (chiều cao đứng = sprite gốc).
    """
    p = cfg.pose
    size = int(p["size"])
    keys = poses.sequence(action.pose or action.name, action.frames)
    seed = resolve_seed(action.seed, seed, cfg.seed)
    anchor = reference_image(native, size)
    ref_name = None
    if p.get("ipadapter"):
        ref_name = client.upload_image(anchor, f"spriteforge_ref_{uuid.uuid4().hex[:12]}.png")
    text = {"character": f"{description.strip().rstrip('.')}, " if description else "",
            "action": action.prompt.strip().rstrip("."), "view": VIEWS[cfg.view]}
    common = {"checkpoint": cfg.checkpoint_file or cfg.checkpoint, "controlnet": p["controlnet"],
              "strength": float(p["strength"]), "reference_image": ref_name,
              "ipadapter_weight": float(p["ipadapter_weight"]),
              "ipadapter_preset": str(p["ipadapter_preset"]), "size": size, "steps": cfg.steps,
              "cfg": cfg.cfg_scale, "sampler": cfg.sampler, "scheduler": cfg.scheduler, "seed": seed}

    raws, boards, layout_used = None, [], "single"
    if p.get("layout") == "grid":
        prompt = GRID_PROMPT.format(**text)
        negative = GRID_NEGATIVE.format(other_view=OTHER_VIEW[cfg.view])
        try:
            raws, boards = _draw_boards(client, keys, anchor, int(p["grid_max_cells"]),
                                        prompt, negative, common)
            layout_used = "grid"
        except ComfyError as e:
            log.warning("Vẽ bảng tư thế lỗi (%s) -> vẽ từng frame. Nếu lỗi hết VRAM: đặt \"pose\": "
                        "{\"grid_max_cells\": 5} (bảng nhỏ hơn) hoặc {\"layout\": \"single\"}.", e)
    if raws is None:
        prompt = POSE_PROMPT.format(**text)
        negative = CHAR_NEGATIVE.format(other_view=OTHER_VIEW[cfg.view])
        raws = [_draw_single(client, key, prompt, negative, common) for key in keys]

    scale = native.height / poses.standing_height_px(size)
    side = max(8, round(size * scale))
    hip_x, ground_y = poses.ground_px(size)
    layout = pixel.VideoLayout(k=1, width=side, height=side,
                               center_x=hip_x * scale, ground_y=ground_y * scale)
    frames, cleans = [], []
    for key, raw in zip(keys, raws):
        bg = matting.border_color(np.asarray(raw).astype(np.int16))
        cleaned, _ = cleanup.clean_figure(matting.cutout_flat(raw), key, bg)
        frames.append(pixel.frame_to_native(cleaned, (side, side), palette))
        cleans.append(cleaned)
    if debug_dir:  # khung xương + ảnh SD gốc + sau làm sạch, từng frame, để soi lỗi
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        _strip([poses.render(k, size) for k in keys], debug_dir / "poses.png")
        _strip(raws, debug_dir / "raw.png")
        _strip([_on_gray(c) for c in cleans], debug_dir / "clean.png")
        for i, board in enumerate(boards):
            board.save(debug_dir / ("board.png" if i == 0 else f"board_{i + 1}.png"))
    empty = sum(pixel.bbox(f) is None for f in frames)
    if empty == len(frames):
        raise ComfyError("Mọi frame đều trống sau khi tách nền: SD không vẽ nhân vật trên nền "
                         "trắng. Xem debug/raw.png.")
    if empty:
        log.warning("%d/%d frame trống sau khi tách nền", empty, len(frames))
    return frames, layout, {"engine": "pose", "pose": action.pose, "layout": layout_used,
                            "boards": len(boards), "seed": seed, "prompt": prompt,
                            "ipadapter": bool(ref_name), "scale": round(scale, 4)}


def _draw_single(client: ComfyClient, key: dict, prompt: str, negative: str,
                 common: dict) -> Image.Image:
    """Một frame = một ảnh SD vẽ theo một khung xương."""
    pose_name = client.upload_image(poses.render(key, common["size"]),
                                    f"spriteforge_pose_{uuid.uuid4().hex[:12]}.png")
    graph = workflows.sd15_pose(prompt=prompt, negative=negative, pose_image=pose_name, **common)
    return client.fetch(client.run(graph, workflows.POSE_OUTPUT, timeout_s=900)[0]).convert("RGB")


def _draw_boards(client: ComfyClient, keys: list[dict], anchor: Image.Image, max_cells: int,
                 prompt: str, negative: str, common: dict,
                 ) -> tuple[list[Image.Image], list[Image.Image]]:
    """Vẽ frame theo bảng tư thế: ô 0 = nhân vật mẫu (giữ nguyên), ô 1.. = các frame.
    Trả về (ảnh từng frame cỡ size x size, ảnh từng bảng)."""
    size = common["size"]
    raws, boards = [], []
    for idx in grid.chunks(len(keys), max_cells):
        cols, rows = grid.shape(len(idx) + 1)
        cells = list(range(1, len(idx) + 1))
        skeletons = [poses.render(poses.NEUTRAL, size)] + [poses.render(keys[i], size) for i in idx]
        init = grid.compose([anchor], cols, rows, size, (255, 255, 255))
        tag = uuid.uuid4().hex[:12]
        pose_name, init_name, mask_name = (
            client.upload_image(im, f"spriteforge_{kind}_{tag}.png") for kind, im in (
                ("pose", grid.compose(skeletons, cols, rows, size, (0, 0, 0))), ("board", init),
                ("mask", grid.mask(cells, cols, rows, size))))
        graph = workflows.sd15_pose(prompt=prompt, negative=negative, pose_image=pose_name,
                                    init_image=init_name, init_mask=mask_name, **common)
        board = client.fetch(client.run(graph, workflows.POSE_OUTPUT,
                                        timeout_s=GRID_TIMEOUT_S)[0]).convert("RGB")
        if board.size != init.size:
            raise ComfyError(f"bảng tư thế trả về ảnh {board.size[0]}x{board.size[1]}, "
                             f"cần {init.size[0]}x{init.size[1]}")
        raws += grid.split(board, cells, cols, size)
        boards.append(board)
    return raws, boards


def _on_gray(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, (90, 90, 100, 255))
    bg.alpha_composite(rgba.convert("RGBA"))
    return bg.convert("RGB")


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
