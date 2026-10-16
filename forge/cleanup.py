"""Làm sạch ảnh SD theo khung xương: chỉ giữ đúng một nhân vật.

Lỗi thấy khi chạy thật với checkpoint pixel art (aziibpixelmix):
  - vẽ 3 nhân vật cạnh nhau kiểu "character sheet";
  - vẽ đĩa tròn / mảng hình học lớn phía sau nhân vật;
  - vẽ vệt đất tối kéo dài ngang dưới chân.
Tách nền trắng giữ hết những thứ đó, rồi IP-Adapter chép chúng sang MỌI frame. Engine
pose biết chính xác khung xương nằm đâu, nên dùng nó để lọc:
  1. chỉ giữ các mảng chạm vào lõi khung xương (bỏ nhân vật thừa, mảng rời);
  2. bóc các mảng MỘT MÀU lớn nằm sát nền và phần lớn ở ngoài thân (đĩa tròn, mái nhà...);
  3. dưới đường nền, chỉ giữ phần trong vùng thân (bỏ vệt đất ngoài bàn chân).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from . import poses

_EIGHT = np.ones((3, 3), bool)

CORE_RADIUS = 0.06    # lõi khung xương: mảng nào chạm vào đây mới là nhân vật
BODY_RADIUS = 0.20    # thân người + quần áo (đơn vị chiều cao)
HEAD_RADIUS = 0.17
WIDE_RADIUS = 0.30    # ngoài vùng này mà "dày" (không phải vũ khí mảnh) -> người thừa/phông nền
WIDE_HEAD = 0.24
THICK_PX = 7          # bán kính mở hình thái: nét mảnh hơn ~14px (kiếm, cán giáo) được giữ


def _keep_touching(mask: np.ndarray, core: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return mask
    hit = np.unique(labels[core & mask])
    hit = hit[hit > 0]
    if len(hit) == 0:  # không mảng nào chạm khung xương: giữ mảng lớn nhất
        sizes = ndimage.sum_labels(mask, labels, np.arange(1, n + 1))
        hit = np.array([int(np.argmax(sizes)) + 1])
    return np.isin(labels, hit)


def _peel(rgb: np.ndarray, mask: np.ndarray, body: np.ndarray, tol: int = 42,
          min_share: float = 0.05, outside_share: float = 0.6, rounds: int = 4) -> np.ndarray:
    """Bóc các mảng một màu lớn nằm sát nền, phần lớn ở ngoài thân (lặp vài vòng: đĩa
    tròn có viền nhiều lớp màu, bóc lớp ngoài xong mới lộ lớp trong).

    Màu SD không phẳng tuyệt đối (đĩa cam thật: (228,62,0) cạnh (243,81,32)), so màu
    thẳng thì mảng vỡ vụn thành trăm mảnh nhỏ -> làm mịn bằng median 5x5 trước khi so.
    """
    rgb = np.dstack([ndimage.median_filter(rgb[:, :, c], size=5) for c in range(3)]).astype(np.int16)
    for _ in range(rounds):
        # Lấy mẫu màu sâu 3px từ mép: mép ngoài thường là viền chuyển màu, không phải màu mảng
        edge = mask & ndimage.binary_dilation(~mask, iterations=3)
        if not edge.any():
            break
        total = int(mask.sum())
        bins = (rgb[edge] // 24).astype(np.int32)
        keys = bins[:, 0] * 10000 + bins[:, 1] * 100 + bins[:, 2]
        values, counts = np.unique(keys, return_counts=True)
        changed = False
        for key in values[np.argsort(counts)[::-1][:4]]:
            color = np.median(rgb[edge][keys == key], axis=0)
            similar = mask & (np.abs(rgb - color).max(2) <= tol)
            labels, _ = ndimage.label(similar, structure=_EIGHT)
            for lab in np.unique(labels[edge & similar]):
                if lab == 0:
                    continue
                comp = labels == lab
                area = int(comp.sum())
                outside = int((comp & ~body).sum())
                # Chỉ bóc khi >= 60% mảng nằm ngoài thân. Đã thử nới cho mảng rất lớn: bóc mất
                # luôn áo choàng tím của nhân vật thật (dính liền áo nhân vật thừa) -> thủng
                # thân. Bóc sót một ít phông nền còn hơn làm hỏng nhân vật.
                if area >= min_share * total and outside >= outside_share * area:
                    mask = mask & ~comp
                    changed = True
        if not changed:
            break
    return mask


def _drop_bulky_outside(mask: np.ndarray, wide: np.ndarray) -> np.ndarray:
    """Bỏ phần "dày" nằm ngoài vùng thân rộng: nhân vật thừa đứng dính sát, mảng phông nền.
    Nét mảnh (kiếm, giáo vươn xa) không sống sót qua phép mở hình thái nên được giữ."""
    yy, xx = np.mgrid[-THICK_PX:THICK_PX + 1, -THICK_PX:THICK_PX + 1]
    disk = xx * xx + yy * yy <= THICK_PX * THICK_PX
    bulky = ndimage.binary_opening(mask, structure=disk)
    return mask & ~(bulky & ~wide)


def _reflood(rgb: np.ndarray, mask: np.ndarray, bg_rgb, tol: int = 20) -> np.ndarray:
    """Nền bị bao trong đĩa tròn lộ ra sau khi bóc: xoá phần cùng màu nền chạm vùng trong suốt."""
    near_bg = mask & (np.abs(rgb - np.asarray(bg_rgb, np.int16)).max(2) <= tol)
    labels, _ = ndimage.label(near_bg, structure=_EIGHT)
    touching = np.unique(labels[near_bg & ndimage.binary_dilation(~mask)])
    touching = touching[touching > 0]
    return mask & ~np.isin(labels, touching) if len(touching) else mask


def clean_figure(rgba: Image.Image, key: dict, bg_rgb=None) -> tuple[Image.Image, dict]:
    """Ảnh SD đã tách nền (cỡ size x size, vẽ theo khung xương `key`) -> chỉ còn nhân vật.

    bg_rgb: màu nền gốc của ảnh SD (VD trắng) để xoá nốt nền lộ ra sau khi bóc đĩa tròn.
    """
    size = rgba.width
    a = np.asarray(rgba.convert("RGBA")).copy()
    rgb = a[:, :, :3].astype(np.int16)
    mask = a[:, :, 3] > 0
    before = int(mask.sum())
    core = poses.body_mask(key, size, CORE_RADIUS, CORE_RADIUS * 1.5)
    body = poses.body_mask(key, size, BODY_RADIUS, HEAD_RADIUS)

    mask = _keep_touching(mask, core)
    after_figures = int(mask.sum())
    if bg_rgb is not None:  # viền sáng mờ quanh mảng phía sau (chuyển màu sang nền trắng)
        mask = _reflood(rgb, mask, bg_rgb, tol=40)
    mask = _peel(rgb, mask, body)
    if bg_rgb is not None:
        mask = _reflood(rgb, mask, bg_rgb)
    mask = _drop_bulky_outside(mask, poses.body_mask(key, size, WIDE_RADIUS, WIDE_HEAD))
    after_peel = int(mask.sum())
    # Vệt đất: dưới đường nền chỉ giữ phần nằm trong vùng thân (bàn chân)
    ground_y = int(poses.ground_px(size)[1] - 0.02 * poses.standing_height_px(size))
    band = np.zeros_like(mask)
    band[ground_y:] = True
    mask &= ~(band & ~body)
    mask = _keep_touching(mask, core)

    a[~mask] = 0
    kept = int(mask.sum())
    outside = int((mask & ~body).sum())
    return Image.fromarray(a, "RGBA"), {
        "removed_figures_px": before - after_figures,
        "peeled_px": after_figures - after_peel,
        "outside_share": round(outside / kept, 3) if kept else 1.0,
    }
