"""Con rối 2D: uốn chính sprite mẫu theo khung xương tư thế (engine "edit").

Vì sao: FLUX.2 klein (model sửa ảnh theo ảnh mẫu) giữ nhân vật mẫu rất tốt, nhưng giữ
luôn TƯ THẾ và bố cục của ảnh thứ nhất. Thử thật trên Space chính thức (16/10/2026):
sprite đang đứng + ma-nơ-canh đang chạy + prompt "vẽ theo tư thế ma-nơ-canh" -> ra đúng
nhân vật nhưng vẫn đứng. Vì vậy ảnh thứ nhất phải ĐÃ ở đúng tư thế:
  1. đặt sprite lên khung xương đứng (NEUTRAL) của forge/poses.py;
  2. chia pixel thành đầu, thân, tay, chân theo đoạn xương gần nhất;
  3. xoay từng phần theo khung xương đích (giữ nguyên chiều dài xương).
Ảnh con rối còn thô (khớp gãy, hở, tay sau chỉ là bản sao tối màu của tay trước), klein
vẽ lại cho sạch mà vẫn giữ tư thế; ảnh thứ hai (sprite gốc) giữ đúng ngoại hình.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from . import poses

_EIGHT = np.ones((3, 3), bool)

ARM_RADIUS = 0.05      # pixel cách xương tay <= 0.05 chiều cao -> thuộc tay (tay sprite ~4-6px / 64px)
ARM_BELOW_HIP = 0.08   # bàn tay buông thõng thấp hơn hông một chút
TORSO_FILL = 0.11      # chỗ tay trước rời đi: lấp màu thân trong bán kính này quanh xương sống
FAR_ARM_SHADE = 0.70   # tay phía xa = bản sao tay gần, tối hơn (thường bị thân che)

# (tên, điểm đầu, điểm cuối) theo chỉ số OpenPose; "hip" = giữa hai hông.
# Thứ tự vẽ: sau ra trước (tay xa, chân xa, chân gần, thân, đầu, tay gần).
PARTS = [("fa_up", 5, 6), ("fa_lo", 6, 7), ("fl_up", 11, 12), ("fl_lo", 12, 13),
         ("nl_up", 8, 9), ("nl_lo", 9, 10), ("torso", "hip", 1), ("head", 1, 0),
         ("na_up", 2, 3), ("na_lo", 3, 4)]
# Tay xa không có pixel riêng (sprite nhìn ngang: thân che mất) -> lấy pixel tay gần, nên
# xoay từ chỗ của tay gần ở tư thế đứng.
SOURCE = {"fa_up": (2, 3), "fa_lo": (3, 4)}


@dataclass
class Placement:
    """Sprite gốc phóng NEAREST xk trên canvas vuông `size`; khung xương khớp theo nó."""
    size: int
    k: int
    unit: float      # chiều cao đứng của nhân vật trên canvas (px) = 1 đơn vị khung xương
    hip_x: float
    ground_y: float
    x0: int          # góc trên-trái của sprite đã phóng
    y0: int


def place(native: Image.Image, size: int) -> Placement:
    """Phóng nguyên lần sao cho nhân vật cao ~UNIT * size; chân trên đường nền GROUND.

    Mọi toạ độ là bội của k: ảnh model trả về thu lại xk là khớp đúng lưới pixel gốc.
    Hông đặt theo trung vị cột của phần chân (35% dưới), không theo khung bao: vũ khí chĩa
    ra trước không kéo lệch nhân vật.
    """
    w, h = native.size
    k = max(1, min(math.floor(poses.UNIT * size / h), math.floor(0.8 * size / w)))
    alpha = np.asarray(native.convert("RGBA"))[:, :, 3] > 0
    legs = alpha[int(h * 0.65):]
    xs = np.nonzero(legs if legs.any() else alpha)[1]
    cx = float(np.median(xs)) + 0.5 if len(xs) else w / 2
    ground = k * math.floor(poses.GROUND * size / k)
    x0 = k * round((poses.CENTER * size - k * cx) / k)
    return Placement(size=size, k=k, unit=float(k * h), hip_x=x0 + k * cx,
                     ground_y=float(ground), x0=x0, y0=ground - k * h)


def points(key: dict, pl: Placement) -> dict:
    """Điểm khung xương (px trên canvas) theo chỉ số OpenPose, thêm "hip" và "head"."""
    pts = {}
    for i, p in enumerate(poses.skeleton(key)):
        if p is not None:
            pts[i] = np.array([pl.hip_x + p[0] * pl.unit, pl.ground_y - p[1] * pl.unit])
    pts["hip"] = (pts[8] + pts[11]) / 2
    pts["head"] = pts[1] + (pts[0] - pts[1]) * 0.75 - np.array([0, 0.03 * pl.unit])
    return pts


def sprite_canvas(native: Image.Image, pl: Placement) -> Image.Image:
    """Sprite gốc phóng xk trên canvas trong suốt, đúng chỗ của khung xương đứng."""
    canvas = Image.new("RGBA", (pl.size, pl.size), (0, 0, 0, 0))
    up = native.convert("RGBA").resize((native.width * pl.k, native.height * pl.k), Image.NEAREST)
    canvas.alpha_composite(up, (pl.x0, pl.y0))
    return canvas


def on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba.convert("RGBA"))
    return bg.convert("RGB")


def _seg_dist(px: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((px - a) @ ab) / max(1e-9, float(ab @ ab)), 0, 1)
    return np.linalg.norm(px - (a + t[:, None] * ab), axis=1)


class Rig:
    """Sprite đã chia thành các phần theo khung xương đứng; `pose(key)` -> ảnh con rối."""

    def __init__(self, native: Image.Image, size: int):
        self.pl = place(native, size)
        self.canvas = sprite_canvas(native, self.pl)
        self.rest = points(poses.NEUTRAL, self.pl)
        a = np.asarray(self.canvas).copy()
        mask = a[:, :, 3] > 0
        ys, xs = np.nonzero(mask)
        px = np.stack([xs + 0.5, ys + 0.5], axis=1)
        dist = {name: _seg_dist(px, self.rest[s], self.rest[e]) for name, s, e in PARTS}
        u = self.pl.unit
        neck_y, hip_y = self.rest[1][1], self.rest["hip"][1]

        label = np.full(len(px), "torso", dtype=object)
        y = px[:, 1]
        arm_d = np.minimum(dist["na_up"], dist["na_lo"])
        arm = (y >= neck_y - ARM_RADIUS * u) & (y <= hip_y + ARM_BELOW_HIP * u) & (arm_d <= ARM_RADIUS * u)
        label[arm] = np.where(dist["na_up"][arm] <= dist["na_lo"][arm], "na_up", "na_lo")
        label[(y < neck_y) & ~arm] = "head"
        legs = (y > hip_y) & ~arm
        leg_names = ["nl_up", "nl_lo", "fl_up", "fl_lo"]
        nearest = np.argmin(np.stack([dist[n][legs] for n in leg_names]), axis=0)
        label[legs] = np.array(leg_names, dtype=object)[nearest]

        self.layers: dict[str, Image.Image] = {}
        for name, _, _ in PARTS:
            if name in ("fa_up", "fa_lo"):
                continue
            part = np.zeros_like(a)
            sel = label == name
            part[ys[sel], xs[sel]] = a[ys[sel], xs[sel]]
            self.layers[name] = part
        # Thân: lấp chỗ tay trước để lại (không thì tay vung đi để lại "tay ma" ở chỗ cũ)
        torso = self.layers["torso"]
        has = torso[:, :, 3] > 0
        hole = np.zeros_like(has)
        hole[ys[arm], xs[arm]] = True
        if has.any() and hole.any():
            gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            near_spine = _seg_dist(np.stack([gx[hole] + 0.5, gy[hole] + 0.5], axis=1),
                                   self.rest["hip"], self.rest[1]) <= TORSO_FILL * u
            _, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
            fy, fx = np.nonzero(hole)
            fy, fx = fy[near_spine], fx[near_spine]
            torso[fy, fx] = torso[iy[fy, fx], ix[fy, fx]]
        for near, far in (("na_up", "fa_up"), ("na_lo", "fa_lo")):
            self.layers[far] = _shade(self.layers[near], FAR_ARM_SHADE)
        self.layers = {name: Image.fromarray(v, "RGBA") for name, v in self.layers.items()}
        self.counts = {name: int((label == name).sum()) for name, _, _ in PARTS}

    def pose(self, key: dict) -> Image.Image:
        target = points(key, self.pl)
        out = Image.new("RGBA", (self.pl.size, self.pl.size), (0, 0, 0, 0))
        for name, s, e in PARTS:
            s0, e0 = SOURCE.get(name, (s, e))
            a0, b0, a1, b1 = self.rest[s0], self.rest[e0], target[s], target[e]
            phi = (math.atan2(b1[1] - a1[1], b1[0] - a1[0]) -
                   math.atan2(b0[1] - a0[1], b0[0] - a0[0]))
            c, sn = math.cos(phi), math.sin(phi)
            # ảnh đích q -> ảnh nguồn p = R(-phi) (q - a1) + a0
            data = (c, sn, a0[0] - c * a1[0] - sn * a1[1],
                    -sn, c, a0[1] + sn * a1[0] - c * a1[1])
            out.alpha_composite(self.layers[name].transform(out.size, Image.AFFINE, data,
                                                            resample=Image.NEAREST))
        return out


def _shade(layer: np.ndarray, factor: float) -> np.ndarray:
    out = layer.copy()
    out[:, :, :3] = (out[:, :, :3].astype(np.float32) * factor).astype(np.uint8)
    return out


def match(a: np.ndarray, b: np.ndarray, unit: float, radius: float = 0.04) -> float:
    """Hai hình bóng (mask bool) khớp nhau tới đâu, 0..1: min của "phần a nằm gần b" và
    "phần b nằm gần a" (sai lệch <= `radius` chiều cao được bỏ qua: model vẽ lại nét viền,
    áo choàng, vũ khí khác con rối một chút là bình thường)."""
    if not a.any() or not b.any():
        return 0.0
    r = radius * unit
    near_b = ndimage.distance_transform_edt(~b) <= r
    near_a = ndimage.distance_transform_edt(~a) <= r
    return float(min((a & near_b).sum() / a.sum(), (b & near_a).sum() / b.sum()))


def follows_pose(out: np.ndarray, puppet: np.ndarray, rest: np.ndarray, unit: float,
                 threshold: float = 0.7, radius: float = 0.04) -> tuple[bool, float, float]:
    """Ảnh model có theo tư thế con rối không? Lỗi cần bắt: model chép nguyên tư thế ĐỨNG
    của ảnh mẫu (đã thấy khi thử thật). Trả về (đạt, độ khớp hình bóng, độ phủ phần dời chỗ).

    Chỉ so hình bóng thì không đủ: giơ kiếm lên đầu chỉ đổi ~5% hình bóng. Nên đo riêng
    "phần dời chỗ" = pixel con rối nằm xa tư thế đứng (tay giơ lên, chân bước ra): ảnh đạt
    phải phủ được ít nhất nửa phần đó."""
    follow = match(out, puppet, unit, radius)
    r = radius * unit
    moved = puppet & (ndimage.distance_transform_edt(~rest) > r)
    if moved.sum() < 0.03 * max(1, puppet.sum()):  # gần như tư thế đứng (VD idle)
        return follow >= threshold, follow, 1.0
    gain = float((moved & (ndimage.distance_transform_edt(~out) <= r)).sum() / moved.sum())
    return follow >= threshold and gain >= 0.5, follow, gain


def keep_near(rgba: Image.Image, guide: np.ndarray, pl: Placement, near: float = 0.06,
              below: float = 0.04) -> Image.Image:
    """Ảnh model đã tách nền -> chỉ giữ phần gần con rối: bỏ mảnh lạc ở xa (người thừa,
    hoạ tiết nền) và vệt bóng/đất dưới đường nền ngoài bàn chân."""
    a = np.asarray(rgba.convert("RGBA")).copy()
    mask = a[:, :, 3] > 0
    if not mask.any() or not guide.any():
        return Image.fromarray(a, "RGBA")
    dist = ndimage.distance_transform_edt(~guide)
    labels, n = ndimage.label(mask, structure=_EIGHT)
    close = np.unique(labels[mask & (dist <= near * pl.unit)])
    keep = np.isin(labels, close[close > 0])
    ground = int(pl.ground_y + 0.02 * pl.unit)
    keep[ground:] &= dist[ground:] <= below * pl.unit
    a[~keep] = 0
    return Image.fromarray(a, "RGBA")
