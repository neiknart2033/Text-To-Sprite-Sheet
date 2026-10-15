"""Tách nền chroma key (nền một màu: xanh lá, magenta, xanh dương...).

Chuyển thể từ `sidecar/scontorno.py` của Sprite Sheep
(MIT, Copyright (c) 2026 Silvio Manfrinato), chỉ giữ nhánh "tinta" (nền một
màu) và đổi tên sang tiếng Anh. Ý tưởng chính của tác giả gốc:

- Màu nền được ĐO trên ảnh (gần màu yêu cầu), không lấy hằng số: cùng một
  "green screen" mỗi model/chương trình xuất ra một sắc khác nhau.
- Hai ngưỡng: `strict` = chắc chắn là nền; `wide` = nền có thể lan tới đây.
  Một vùng nằm trong ngưỡng rộng là nền NẾU nó chứa ít nhất một pixel trong
  ngưỡng hẹp. Nhờ vậy khoảng xanh kẹt giữa tay và thân vẫn bị xoá, còn một
  pixel vàng lỡ giống màu xanh thì vẫn được giữ.
- Khử "spill": màu nền hắt lên viền nhân vật, chỉ sửa ở kênh trội của màu nền.

Alpha trả về còn mềm; với pixel art, `forge.pixel` sẽ nhị phân hoá sau.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

PRESETS = {
    "green": (0, 177, 64),
    "magenta": (255, 0, 255),
    "blue": (0, 71, 187),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# Màu tô nền khi CHÚNG TA dựng ảnh (canvas đưa vào WAN): màu thuần, dễ tách nhất
KEY_FILL = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
}


def corner_color(a: np.ndarray, q: int = 6) -> np.ndarray:
    """Màu nền đo ở 4 góc ảnh (trung vị)."""
    corners = np.concatenate([a[:q, :q].reshape(-1, 3), a[:q, -q:].reshape(-1, 3),
                              a[-q:, :q].reshape(-1, 3), a[-q:, -q:].reshape(-1, 3)])
    return np.median(corners, 0)


def corner_spread(image: Image.Image) -> float:
    """Độ lệch lớn nhất giữa các góc so với màu nền: > ~12 là nền không đồng nhất."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    tint = corner_color(a)
    corners = [a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]
    return max(float(np.abs(c - tint).max()) for c in corners)


def resolve_color(spec, a: np.ndarray) -> np.ndarray:
    """"auto" | tên preset | "#RRGGBB" | "r,g,b" | (r, g, b) -> mảng RGB."""
    if spec is None or spec in ("", "auto"):
        return corner_color(a)
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        return np.array([float(x) for x in spec])
    s = str(spec).strip().lower()
    if s in PRESETS:
        return np.array(PRESETS[s], float)
    if s.startswith("#") and len(s) == 7:
        return np.array([int(s[i:i + 2], 16) for i in (1, 3, 5)], float)
    if "," in s and len(s.split(",")) == 3:
        return np.array([float(p) for p in s.split(",")])
    raise ValueError(f"Màu nền không hợp lệ: {spec!r}")


def snap_tint(a: np.ndarray, tint: np.ndarray, radius: float = 110.0) -> np.ndarray:
    """Dời màu yêu cầu về màu nền thật có trong ảnh."""
    d = np.sqrt(((a.astype(np.float32) - tint) ** 2).sum(2))
    near = d < radius
    if near.sum() < 64:
        return tint
    return np.median(a[near], axis=0).astype(np.float32)


def _alpha(a: np.ndarray, tint: np.ndarray, wide: float) -> np.ndarray:
    strict = max(4.0, wide * 0.4)
    d = np.sqrt(((a.astype(np.float32) - tint) ** 2).sum(2))
    seed = d < strict
    if not seed.any():
        # Không có pixel nào đúng màu nền: thà giữ nguyên còn hơn đục lỗ bừa
        return np.ones(a.shape[:2], np.float32)
    labels, n = ndimage.label(d < wide)
    is_bg = np.zeros(n + 1, bool)
    is_bg[1:] = ndimage.sum_labels(seed, labels, np.arange(1, n + 1)) > 0
    background = is_bg[labels]
    ramp = np.clip((d - strict) / max(1e-6, wide - strict), 0.0, 1.0)
    return np.where(background, ramp, 1.0).astype(np.float32)


def _despill(a: np.ndarray, tint: np.ndarray) -> np.ndarray:
    channel = int(np.argmax(tint))
    others = [i for i in (0, 1, 2) if i != channel]
    c = a[:, :, channel]
    mean_others = (a[:, :, others[0]].astype(np.int16) + a[:, :, others[1]]) // 2
    dominant = c > np.maximum(a[:, :, others[0]], a[:, :, others[1]])
    out = a.copy()
    out[:, :, channel] = np.where(dominant, mean_others, c)
    return out


def remove_background(image: Image.Image, color="auto", tolerance: float = 66,
                      despill: bool = True) -> Image.Image:
    """Ảnh nền một màu -> RGBA (alpha mềm)."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    tint = snap_tint(a, resolve_color(color, a))
    alpha = _alpha(a, tint, float(tolerance))
    # Khử spill chỉ khi nền có kênh trội rõ ràng (không áp dụng cho nền trắng/đen)
    rgb = _despill(a, tint) if despill and np.ptp(tint) > 60 else a
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")
