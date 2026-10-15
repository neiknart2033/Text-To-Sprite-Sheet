"""Hậu xử lý pixel art: thu nhỏ về pixel gốc, ép palette, đặt frame vào ô.

Kế thừa ý tưởng từ Pixel Art Studio (GPLv3): lượng tử hoá palette
(`mixel_fix.quantize_palette`), lấy đa số trong khối pixel (`pixel_snap`), neo
chân nhân vật vào một đường nền (`frame_align`). Khác bản gốc:
  - palette lấy từ nhân vật gốc rồi ÁP cho mọi frame -> màu không nhảy giữa
    các frame;
  - alpha nhị phân (pixel art không có viền mờ);
  - căn frame theo MỘT độ lệch chung cho cả action (mode "fixed") thay vì neo
    đáy bbox từng frame, để động tác nhảy/lùi không bị "dí" xuống đất.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .matting import KEY_FILL

_EIGHT = np.ones((3, 3), bool)  # nối 8 hướng: nét chéo pixel art không bị coi là đốm rời


# --------------------------------------------------------------- tiện ích alpha
def binarize_alpha(rgba: Image.Image, threshold: int = 128) -> Image.Image:
    a = np.asarray(rgba.convert("RGBA")).copy()
    opaque = a[:, :, 3] >= threshold
    a[:, :, 3] = np.where(opaque, 255, 0)
    a[~opaque, :3] = 0
    return Image.fromarray(a, "RGBA")


def bbox(rgba: Image.Image) -> tuple[int, int, int, int] | None:
    return rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()


def remove_specks(rgba: Image.Image, min_area: int = 3) -> Image.Image:
    """Xoá cụm pixel rời nhỏ hơn `min_area` (nhiễu sau khi tách nền/thu nhỏ)."""
    if min_area <= 1:
        return rgba
    a = np.asarray(rgba).copy()
    mask = a[:, :, 3] > 0
    labels, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return rgba
    sizes = ndimage.sum_labels(mask, labels, np.arange(1, n + 1))
    small = np.isin(labels, np.flatnonzero(sizes < min_area) + 1)
    a[small] = 0
    return Image.fromarray(a, "RGBA")


def resize_premultiplied(rgba: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Thu nhỏ kiểu trung bình khối, có tính alpha (pixel trong suốt không làm bẩn màu)."""
    a = np.asarray(rgba.convert("RGBA")).astype(np.float32) / 255.0
    alpha = a[:, :, 3]
    channels = []
    for ch in (a[:, :, 0] * alpha, a[:, :, 1] * alpha, a[:, :, 2] * alpha, alpha):
        channels.append(np.asarray(Image.fromarray(ch, "F").resize(size, Image.BOX)))
    r, g, b, al = channels
    safe = np.maximum(al, 1e-6)
    rgb = np.dstack([r / safe, g / safe, b / safe])
    out = np.dstack([np.clip(rgb, 0, 1), np.clip(al, 0, 1)]) * 255
    return Image.fromarray(np.round(out).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------- palette
def extract_palette(rgba: Image.Image, max_colors: int = 32) -> np.ndarray:
    """Palette (k x 3, uint8) từ các pixel đục, median-cut, không dither."""
    a = np.asarray(rgba.convert("RGBA"))
    px = a[a[:, :, 3] >= 128][:, :3]
    if len(px) == 0:
        return np.zeros((1, 3), np.uint8)
    strip = Image.fromarray(px.reshape(1, -1, 3).astype(np.uint8), "RGB")
    q = strip.quantize(colors=max(1, min(256, max_colors)),
                       method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    used = np.unique(np.asarray(q))
    pal = np.array(q.getpalette()[: 3 * 256], np.uint8).reshape(-1, 3)
    return pal[used]


def apply_palette(rgba: Image.Image, palette: np.ndarray) -> Image.Image:
    """Đưa mọi pixel đục về màu gần nhất trong palette."""
    a = np.asarray(rgba.convert("RGBA")).copy()
    opaque = a[:, :, 3] > 0
    px = a[opaque][:, :3].astype(np.int32)
    if len(px):
        pal = palette.astype(np.int32)
        idx = np.empty(len(px), np.int64)
        for s in range(0, len(px), 65536):
            chunk = px[s:s + 65536]
            d = ((chunk[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
            idx[s:s + 65536] = d.argmin(1)
        a[opaque, :3] = palette[idx]
    return Image.fromarray(a, "RGBA")


def key_color_for(palette: np.ndarray) -> str:
    """Chọn màu nền (green/magenta/blue) khác xa nhất mọi màu của nhân vật."""
    best, best_d = "green", -1.0
    for name in ("green", "magenta", "blue"):
        c = np.array(KEY_FILL[name], np.float32)
        d = float(np.sqrt(((palette.astype(np.float32) - c) ** 2).sum(1)).min())
        if d > best_d:
            best, best_d = name, d
    return best


# ---------------------------------------------------------------- nhân vật
def prepare_character(rgba: Image.Image, char_height: int, palette_size: int,
                      speck_area: int = 3) -> tuple[Image.Image, np.ndarray]:
    """Ảnh nhân vật (đã tách nền) -> sprite gốc cao <= char_height px + palette."""
    box = bbox(rgba)
    if box is None:
        raise ValueError("Ảnh nhân vật trống sau khi tách nền")
    rgba = rgba.crop(box)
    w, h = rgba.size
    if h > char_height:
        s = char_height / h
        rgba = resize_premultiplied(rgba, (max(1, round(w * s)), char_height))
    rgba = remove_specks(binarize_alpha(rgba), speck_area)
    palette = extract_palette(rgba, palette_size)
    rgba = apply_palette(rgba, palette)
    box = bbox(rgba)
    return (rgba.crop(box) if box else rgba), palette


def upscale(rgba: Image.Image, factor: int) -> Image.Image:
    return rgba.resize((rgba.width * factor, rgba.height * factor), Image.NEAREST)


# ------------------------------------------------------------ ảnh vào WAN
@dataclass
class VideoLayout:
    """Nhân vật gốc được phóng xk (NEAREST) và đặt trên canvas WAN.

    Toạ độ `center_x`, `ground_y` tính theo pixel GỐC (canvas / k) - dùng để
    đặt các frame trả về vào ô sprite với cùng một độ lệch.
    """
    k: int
    width: int
    height: int
    center_x: float
    ground_y: float

    @property
    def native_size(self) -> tuple[int, int]:
        return self.width // self.k, self.height // self.k


def compose_video_input(native: Image.Image, width: int, height: int,
                        fill: tuple[int, int, int], char_fill: float = 0.6,
                        ground: float = 0.88) -> tuple[Image.Image, VideoLayout]:
    """Canvas RGB nền một màu, nhân vật phóng nguyên lần, chân ở `ground` * height."""
    w, h = native.size
    k = max(1, min(math.floor(char_fill * height / h), math.floor(0.9 * width / w)))
    up = upscale(native, k)
    x = (width - up.width) // 2
    y = max(0, round(ground * height) - up.height)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(up, (x, y), up)
    layout = VideoLayout(k=k, width=width, height=height,
                         center_x=(x + up.width / 2) / k, ground_y=(y + up.height) / k)
    return canvas, layout


def video_frame_to_native(frame_rgba: Image.Image, layout: VideoLayout,
                          palette: np.ndarray, speck_area: int = 3) -> Image.Image:
    """Frame WAN (đã tách nền) -> pixel gốc, alpha nhị phân, đúng palette nhân vật."""
    nw, nh = layout.native_size
    frame_rgba = frame_rgba.crop((0, 0, nw * layout.k, nh * layout.k))
    small = resize_premultiplied(frame_rgba, (nw, nh))
    small = remove_specks(binarize_alpha(small), speck_area)
    return apply_palette(small, palette)


# ------------------------------------------------------------ đặt vào ô
def _centroid_x(rgba: Image.Image) -> float | None:
    alpha = np.asarray(rgba)[:, :, 3] > 0
    xs = np.nonzero(alpha)[1]
    return float(xs.mean()) if len(xs) else None


def place_in_cells(frames: list[Image.Image], layout: VideoLayout, frame_size: int,
                   ground_margin: int, mode: str = "fixed") -> tuple[list[Image.Image], dict]:
    """Đặt các frame gốc vào ô vuông frame_size, mọi ô cùng kích thước.

    mode "fixed": một độ lệch chung cho cả action (giữ nguyên chuyển động thật,
                  nhảy vẫn bay lên, chém vẫn vươn ra).
    mode "track": bỏ phần trôi ngang tuyến tính (nhân vật đi/chạy bị dạt dần
                  trong video) nhưng giữ dao động trong chu kỳ.
    """
    anchor_x = frame_size // 2
    anchor_y = frame_size - ground_margin
    dy = round(anchor_y - layout.ground_y)

    centroids = [_centroid_x(f) for f in frames]
    known = [(i, c) for i, c in enumerate(centroids) if c is not None]
    slope, intercept = 0.0, layout.center_x
    if len(known) >= 2:
        xs = np.array([i for i, _ in known], float)
        ys = np.array([c for _, c in known], float)
        slope, intercept = np.polyfit(xs, ys, 1)
    drift = abs(slope) * max(0, len(frames) - 1)

    cells, clipped = [], 0
    for i, frame in enumerate(frames):
        if mode == "track":
            dx = round(anchor_x - (intercept + slope * i))
        else:
            dx = round(anchor_x - layout.center_x)
        cell, lost = _paste_clipped(frame, frame_size, dx, dy)
        cells.append(cell)
        clipped += lost
    return cells, {"clipped_px": int(clipped), "drift_px": round(float(drift), 1),
                   "anchor": {"x": anchor_x, "y": anchor_y}}


def _paste_clipped(frame: Image.Image, size: int, dx: int, dy: int) -> tuple[Image.Image, int]:
    src = np.asarray(frame.convert("RGBA"))
    out = np.zeros((size, size, 4), np.uint8)
    h, w = src.shape[:2]
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(size, dx + w), min(size, dy + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    lost = int((src[:, :, 3] > 0).sum() - (out[:, :, 3] > 0).sum())
    return Image.fromarray(out, "RGBA"), lost
