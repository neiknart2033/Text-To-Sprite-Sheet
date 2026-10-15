"""Thư viện tư thế: khung xương OpenPose nhìn ngang (quay phải) cho từng action.

Ý tưởng "pose asset library" của Pixel Art Studio, nhưng sinh bằng code thay cho ảnh
có sẵn (thư viện gốc chỉ có 2-6 frame/action, thiếu attack/hurt): mỗi action là vài tư
thế mốc (keyframe), nội suy ra đúng số frame cần. Khung xương luôn cùng tỉ lệ và chân
luôn chạm cùng một đường nền -> mọi frame/mọi action khớp vị trí, không zoom, không trôi.

Quy ước góc (độ, theo hướng nhìn của nhân vật): 0 = chĩa thẳng xuống, 90 = chĩa ra phía
trước (sang phải), -90 = ra sau, 180 = thẳng lên. Khuỷu tay/đầu gối là góc gập cộng thêm
vào đoạn xương cha (khuỷu gập ra trước: dương; gối gập ra sau: âm).
Tay/chân "near" = phía gần người xem (bên phải của nhân vật khi quay phải), "far" = phía xa.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

# Chiều dài xương, đơn vị = chiều cao đứng thẳng (mắt cá -> đỉnh đầu ~ 1.0)
SHIN, THIGH, TORSO = 0.24, 0.24, 0.32
UPPER_ARM, FOREARM = 0.15, 0.14
NOSE, EYE, EAR = 0.11, 0.12, 0.08

# Định dạng OpenPose 18 điểm (COCO) và màu chuẩn của ControlNet (controlnet_aux.draw_bodypose)
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
         (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
          (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

# Bố cục trên ảnh sinh (vuông, cạnh `size`): tỉ lệ theo cạnh ảnh
UNIT = 0.58      # 1 đơn vị chiều cao = 58% cạnh ảnh
GROUND = 0.92    # đường nền (mắt cá chân) ở 92% chiều cao ảnh
CENTER = 0.46    # hông ở 46% chiều ngang (chừa chỗ phía trước cho vũ khí)

NEUTRAL = {"lean": 0, "head": 0, "na": (6, 14), "fa": (-6, 14), "nl": (8, -3), "fl": (-8, -3),
           "dx": 0.0, "air": 0.0}


def _k(**kw) -> dict:
    return {**NEUTRAL, **kw}


def _mirror(key: dict) -> dict:
    """Đổi bên tay/chân (nửa sau của chu kỳ đi/chạy)."""
    return {**key, "na": key["fa"], "fa": key["na"], "nl": key["fl"], "fl": key["nl"]}


_WALK = [_k(lean=3, nl=(28, -3), fl=(-22, -8), na=(-22, 10), fa=(22, 20)),
         _k(lean=3, nl=(18, -12), fl=(-14, -22), na=(-12, 12), fa=(12, 18)),
         _k(lean=3, nl=(2, -4), fl=(8, -55), na=(0, 12), fa=(0, 15)),
         _k(lean=3, nl=(-15, -2), fl=(25, -30), na=(12, 15), fa=(-12, 12))]
_RUN = [_k(lean=12, nl=(35, -10), fl=(-30, -40), na=(-35, 80), fa=(40, 90)),
        _k(lean=12, nl=(20, -35), fl=(-10, -80), na=(-20, 85), fa=(25, 90)),
        _k(lean=14, nl=(-10, -20), fl=(45, -70), na=(10, 85), fa=(-10, 85), air=0.06),
        _k(lean=13, nl=(-35, -30), fl=(50, -40), na=(30, 85), fa=(-35, 85), air=0.04)]
_READY = _k(lean=5, na=(30, 60), fa=(10, 40), nl=(20, -5), fl=(-15, -5))
_CROUCH = _k(lean=15, na=(-40, 20), fa=(-30, 20), nl=(55, -100), fl=(45, -95))

PRESETS: dict[str, dict] = {
    "idle": {"loop": True, "keys": [
        NEUTRAL, _k(lean=2, na=(8, 24), fa=(-3, 24), nl=(9, -8), fl=(-7, -8))]},
    "walk": {"loop": True, "keys": _WALK + [_mirror(k) for k in _WALK]},
    "run": {"loop": True, "keys": _RUN + [_mirror(k) for k in _RUN]},
    "attack": {"loop": False, "keys": [
        _READY,
        _k(lean=-8, na=(-120, 40), fa=(30, 30), nl=(20, -15), fl=(-20, -10)),
        _k(lean=15, na=(95, 0), fa=(-30, 20), nl=(35, -20), fl=(-30, -5), dx=0.05),
        _k(lean=15, na=(50, 10), fa=(-20, 20), nl=(35, -20), fl=(-30, -5), dx=0.05),
        _READY]},
    "heavy": {"loop": False, "keys": [
        _READY,
        _k(lean=-5, na=(175, 20), fa=(160, 30), nl=(15, -5), fl=(-15, -5)),
        _k(lean=25, na=(70, 0), fa=(65, 10), nl=(40, -40), fl=(-35, -10), dx=0.06),
        _k(lean=30, na=(40, 0), fa=(35, 10), nl=(45, -60), fl=(-40, -20), dx=0.06),
        _READY]},
    "hurt": {"loop": False, "keys": [
        NEUTRAL,
        _k(lean=-22, head=15, na=(40, 50), fa=(60, 40), nl=(15, -25), fl=(-25, -15), dx=-0.06),
        _k(lean=-12, head=5, na=(25, 40), fa=(40, 40), nl=(10, -15), fl=(-20, -10), dx=-0.08),
        NEUTRAL]},
    "jump": {"loop": False, "keys": [
        NEUTRAL, _CROUCH,
        _k(lean=5, na=(160, 10), fa=(150, 10), nl=(5, -5), fl=(-5, -10), air=0.12),
        _k(lean=0, na=(130, 30), fa=(120, 30), nl=(70, -110), fl=(60, -100), air=0.28),
        _k(lean=0, na=(100, 20), fa=(90, 20), nl=(25, -20), fl=(10, -30), air=0.10),
        _k(lean=10, na=(30, 30), fa=(20, 30), nl=(45, -80), fl=(35, -75)),
        NEUTRAL]},
    "death": {"loop": False, "keys": [
        NEUTRAL,
        _k(lean=-15, head=20, na=(30, 40), fa=(50, 40), nl=(10, -20), fl=(-20, -10), dx=-0.03),
        _k(lean=10, head=-10, na=(10, 20), fa=(20, 20), nl=(25, -115), fl=(10, -110)),
        _k(lean=50, head=-20, na=(60, 10), fa=(70, 10), nl=(40, -90), fl=(30, -90)),
        _k(lean=90, head=-10, na=(95, 0), fa=(100, 0), nl=(-88, 0), fl=(-92, 0), dx=0.05)]},
    "block": {"loop": False, "keys": [
        NEUTRAL,
        _k(lean=-5, na=(110, 100), fa=(80, 90), nl=(25, -15), fl=(-20, -10)),
        _k(lean=-6, na=(110, 100), fa=(80, 90), nl=(25, -18), fl=(-20, -12))]},
    "dash": {"loop": False, "keys": [
        _READY,
        _k(lean=30, na=(-60, 10), fa=(-50, 10), nl=(50, -30), fl=(-45, -10), dx=0.10),
        _k(lean=25, na=(-50, 20), fa=(-40, 20), nl=(45, -40), fl=(-40, -15), dx=0.12),
        _READY]},
    "crouch": {"loop": True, "keys": [
        _CROUCH, _k(lean=17, na=(-38, 24), fa=(-28, 24), nl=(58, -106), fl=(48, -100))]},
    "fall": {"loop": True, "keys": [
        _k(na=(150, 20), fa=(140, 20), nl=(20, -30), fl=(5, -20), air=0.20),
        _k(na=(140, 30), fa=(130, 30), nl=(25, -40), fl=(0, -25), air=0.20)]},
    "cast": {"loop": False, "keys": [
        NEUTRAL,
        _k(lean=-5, na=(-30, 90), fa=(-20, 90), nl=(15, -10), fl=(-15, -10)),
        _k(lean=10, na=(95, 0), fa=(85, 0), nl=(25, -10), fl=(-20, -5)),
        _k(lean=10, na=(95, 0), fa=(85, 0), nl=(25, -10), fl=(-20, -5)),
        NEUTRAL]},
    "kick": {"loop": False, "keys": [
        NEUTRAL,
        _k(lean=-10, na=(40, 60), fa=(-20, 40), nl=(90, -100), fl=(-5, 0)),
        _k(lean=-15, na=(30, 40), fa=(-30, 40), nl=(95, 0), fl=(-5, 0)),
        _k(lean=-10, na=(40, 60), fa=(-20, 40), nl=(90, -100), fl=(-5, 0)),
        NEUTRAL]},
    "punch": {"loop": False, "keys": [
        _READY,
        _k(lean=0, na=(70, 110), fa=(40, 90), nl=(20, -10), fl=(-15, -5)),
        _k(lean=12, na=(90, 0), fa=(30, 90), nl=(30, -15), fl=(-20, -5), dx=0.03),
        _READY]},
}


# ------------------------------------------------------------------ hình học
def _dir(angle: float) -> np.ndarray:
    """Vector đơn vị theo góc (y hướng LÊN, x hướng mặt nhân vật)."""
    r = math.radians(angle)
    return np.array([math.sin(r), -math.cos(r)])


def skeleton(key: dict) -> list[np.ndarray | None]:
    """18 điểm OpenPose (đơn vị chiều cao, y hướng lên, mắt cá thấp nhất chạm y=air)."""
    pts: list[np.ndarray | None] = [None] * 18
    hip = np.zeros(2)
    torso_a = 180 - key["lean"]
    neck = hip + TORSO * _dir(torso_a)
    shoulder = neck + 0.02 * _dir(torso_a + 180)
    nose_a = torso_a - 55 + key["head"]
    pts[1] = neck
    pts[0] = neck + NOSE * _dir(nose_a)
    pts[14] = neck + EYE * _dir(nose_a + 18)      # mắt phải (phía gần)
    pts[16] = neck + EAR * _dir(torso_a + 8)      # tai phải
    for side, (s_idx, e_idx, w_idx), dx in (("na", (2, 3, 4), 0.012), ("fa", (5, 6, 7), -0.012)):
        sh, el = key[side]
        s = shoulder + np.array([dx, 0])
        e = s + UPPER_ARM * _dir(sh)
        pts[s_idx], pts[e_idx], pts[w_idx] = s, e, e + FOREARM * _dir(sh + el)
    for side, (h_idx, k_idx, a_idx), dx in (("nl", (8, 9, 10), 0.012), ("fl", (11, 12, 13), -0.012)):
        hp, kn = key[side]
        h = hip + np.array([dx, 0])
        k = h + THIGH * _dir(hp)
        pts[h_idx], pts[k_idx], pts[a_idx] = h, k, k + SHIN * _dir(hp + kn)
    lowest = min(p[1] for p in pts if p is not None)
    shift = np.array([key["dx"], key["air"] - lowest])
    return [None if p is None else p + shift for p in pts]


def _lerp_key(a: dict, b: dict, t: float) -> dict:
    t = (1 - math.cos(math.pi * t)) / 2  # ease in-out
    out = {}
    for name, va in a.items():
        vb = b[name]
        if isinstance(va, tuple):
            out[name] = tuple(x + (y - x) * t for x, y in zip(va, vb))
        else:
            out[name] = va + (vb - va) * t
    return out


def sequence(name: str, frames: int) -> list[dict]:
    """N tư thế cho action: loop chia đều một vòng; không loop đi từ mốc đầu tới mốc cuối."""
    preset = PRESETS[name]
    keys, loop = preset["keys"], preset["loop"]
    if frames == 1:
        return [keys[0]]
    segments = len(keys) if loop else len(keys) - 1
    out = []
    for i in range(frames):
        pos = (i / frames if loop else i / (frames - 1)) * segments
        j = min(int(pos), segments - 1)
        out.append(_lerp_key(keys[j], keys[(j + 1) % len(keys)], pos - j))
    return out


def to_pixels(points: list[np.ndarray | None], size: int) -> list[tuple[float, float] | None]:
    unit = UNIT * size
    return [None if p is None else (CENTER * size + p[0] * unit, GROUND * size - p[1] * unit)
            for p in points]


def standing_height_px(size: int) -> float:
    """Chiều cao nhân vật đứng thẳng trên ảnh sinh: dùng để quy đổi về pixel gốc."""
    return UNIT * size


def ground_px(size: int) -> tuple[float, float]:
    """(x hông, y đường nền) trên ảnh sinh."""
    return CENTER * size, GROUND * size


def render(key: dict, size: int = 512) -> Image.Image:
    """Ảnh OpenPose (nền đen) đúng kiểu ControlNet được huấn luyện."""
    pts = to_pixels(skeleton(key), size)
    canvas = np.zeros((size, size, 3), np.float32)
    stick = max(2, round(4 * size / 512))
    for i, p in enumerate(pts):
        if p is not None:
            layer = Image.new("L", (size, size), 0)
            ImageDraw.Draw(layer).ellipse([p[0] - stick, p[1] - stick, p[0] + stick, p[1] + stick], fill=255)
            canvas[np.asarray(layer) > 0] = COLORS[i]
    for i, (a, b) in enumerate(LIMBS):
        if pts[a] is None or pts[b] is None:
            continue
        (x0, y0), (x1, y1) = pts[a], pts[b]
        length, ang = math.hypot(x1 - x0, y1 - y0) / 2, math.atan2(y1 - y0, x1 - x0)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        poly = [(mx + length * math.cos(t) * math.cos(ang) - stick * math.sin(t) * math.sin(ang),
                 my + length * math.cos(t) * math.sin(ang) + stick * math.sin(t) * math.cos(ang))
                for t in np.linspace(0, 2 * math.pi, 36, endpoint=False)]
        layer = Image.new("L", (size, size), 0)
        ImageDraw.Draw(layer).polygon(poly, fill=255)
        mask = np.asarray(layer) > 0
        canvas[mask] = canvas[mask] * 0.4 + np.array(COLORS[i], np.float32) * 0.6
    return Image.fromarray(canvas.astype(np.uint8), "RGB")
