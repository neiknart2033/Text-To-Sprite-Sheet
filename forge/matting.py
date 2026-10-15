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


def border_color(a: np.ndarray) -> np.ndarray:
    """Màu nền = trung vị của toàn bộ pixel viền ảnh."""
    return np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)


def cutout_flat(image: Image.Image, tolerance: float | None = None,
                despill: bool = True) -> Image.Image:
    """Tách nền phẳng BẤT KỲ màu (kể cả đen/trắng) bằng cách loang từ mép ảnh.

    Dùng cho ảnh nhân vật (SD sinh ra, hoặc ảnh người dùng). Khác
    `remove_background` (xoá MỌI vùng cùng màu nền - đúng với canvas key do ta
    tự dựng): ở đây chỉ xoá nền NỐI VỚI MÉP ảnh. Lỗi gặp thật trên Kaggle: SD vẽ
    nền đen cho "dark samurai", cách cũ xoá luôn tóc/giáp đen -> thủng 24% thân.
    """
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = border_color(a)
    if tolerance is None:  # theo độ nhiễu thật của nền, kẹp trong [8, 48]
        noise = np.percentile(np.abs(border - bg).max(1), 95)
        tolerance = float(np.clip(noise + 10, 8, 48))
    candidate = np.abs(a - bg).max(2) <= tolerance
    labels, n = ndimage.label(candidate)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge = edge[edge > 0]
    foreground = ~np.isin(labels, edge)

    # Khoảng nền bị kẹp (giữa tay và thân, giữa hai chân - bị vệt bóng dưới chân bịt
    # kín): gỡ nếu đủ lớn. Nền key (xanh lá...) gần như không trùng màu nhân vật nên
    # ngưỡng 1.5% thân; nền đen/trắng/xám dễ trùng mảng giáp/áo nên đòi 4%.
    # (Ảnh thật trên Kaggle: khoảng đen giữa hai chân = 9% thân, mảng tối khác <= 1.3%.)
    if n:
        sizes = ndimage.sum_labels(candidate, labels, np.arange(1, n + 1))
        share = 0.015 if np.ptp(bg) > 60 else 0.04
        min_hole = max(16.0, share * float(foreground.sum()))
        trapped = [i + 1 for i, s in enumerate(sizes) if s >= min_hole and (i + 1) not in edge]
        foreground &= ~np.isin(labels, trapped)

    rgb = _despill(a, bg) if despill and np.ptp(bg) > 60 else a
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8),
                     np.where(foreground, 255, 0).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def matches_key(image: Image.Image, key: str) -> tuple[bool, list[int]]:
    """Nền ảnh có đúng màu key yêu cầu và đồng nhất không (để quyết định gen lại)."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    bg = border_color(a)
    targets = [np.array(KEY_FILL.get(key, PRESETS.get(key, (0, 255, 0))), float)]
    if key in PRESETS:
        targets.append(np.array(PRESETS[key], float))
    close = min(float(np.sqrt(((bg - t) ** 2).sum())) for t in targets) < 120
    return close and corner_spread(image) <= 24, [int(c) for c in bg]
