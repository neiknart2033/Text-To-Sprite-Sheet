"""Đọc và kiểm tra jobs.json (schema: xem KAGGLE_DESIGN.md mục 6)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .poses import PRESETS as POSE_PRESETS

ENGINES = ("edit", "pose", "video")
# Engine "edit" (mặc định): FLUX.2 [klein] 4B, Apache-2.0. Tạo nhân vật từ mô tả và vẽ từng
# frame bằng cách sửa "con rối" (sprite mẫu uốn theo khung xương, forge/puppet.py) theo
# chính sprite mẫu. Model sửa ảnh giữ nhân vật mẫu tốt hơn hẳn IP-Adapter của SD1.5.
EDIT_DEFAULTS = {
    # 3 file của template chính thức ComfyUI (cell 2 notebook tự tải nếu Input chưa có)
    "unet": "flux-2-klein-4b-fp8.safetensors",   # 3.8GB, black-forest-labs/FLUX.2-klein-4b-fp8
    "text_encoder": "qwen_3_4b.safetensors",     # 7.5GB, Comfy-Org/z_image_turbo
    "vae": "flux2-vae.safetensors",              # 0.3GB, Comfy-Org/flux2-dev
    "size": 768,          # cạnh ảnh model vẽ mỗi frame (bội của 16); nhân vật cao ~58%
    "steps": 4,           # bản distilled: 4 bước
    "cfg": 1.0,
    # Ảnh vẽ lại không theo tư thế con rối (model chép tư thế đứng của ảnh mẫu) -> vẽ lại
    # kiểu img2img từ chính con rối: giữ 30% ảnh con rối -> tư thế chắc chắn đúng.
    "fallback_denoise": 0.7,
    "follow_threshold": 0.7,   # độ khớp hình bóng tối thiểu với con rối (forge/puppet.py)
    "timeout_s": 900,
}
POSE_DEFAULTS = {
    "controlnet": "control_v11p_sd15_openpose_fp16.safetensors",
    "strength": 1.0,           # độ bám khung xương của ControlNet
    "ipadapter": True,         # giữ ngoại hình theo ảnh nhân vật (cần ComfyUI_IPAdapter_plus)
    "ipadapter_preset": "PLUS (high strength)",
    "ipadapter_weight": 0.8,
    "size": 512,               # cạnh ảnh SD sinh mỗi frame (= cạnh một ô của bảng tư thế)
    # "grid": cả action trong MỘT ảnh, nhân vật mẫu nằm sẵn ở ô đầu -> frame giống nhau
    # hơn hẳn (forge/grid.py). "single": mỗi frame một ảnh (cách cũ, ít VRAM nhất).
    "layout": "grid",
    "grid_max_cells": 9,       # số ô tối đa một bảng (kể cả ô mẫu): 9 = 3x3 = 1536x1536px
}
POSE_LAYOUTS = ("grid", "single")

WAN_DEFAULTS = {
    "unet": "wan2.2_ti2v_5B_fp16.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
    # WAN 2.2 5B nén 32x32 pixel thành 1 token. Ở 448px nhân vật gầy chỉ rộng ~3 token
    # -> clip thật bị biến hình/zoom/tan biến. 640px: gấp đôi số token, chậm ~2 lần.
    "width": 640,
    "height": 640,
    "steps": 20,
    "cfg": 5.0,
    "shift": 8.0,
    "duration": 2.0,
    "useful_fraction": 0.85,
    "char_fill": 0.6,
    "timeout_s": 3600,
}

VIEWS = {
    "side": "side view, facing right",
    "front": "front view, facing the viewer",
}

ALIGN_MODES = ("fixed", "track")
_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


class ConfigError(ValueError):
    pass


@dataclass
class Action:
    name: str
    prompt: str
    frames: int
    columns: int
    duration: float
    align: str = "fixed"
    loop: bool = True
    seed: int | None = None
    engine: str = "edit"       # "edit" (FLUX.2 klein) | "pose" (khung xương + SD1.5) | "video" (WAN 2.2)
    pose: str | None = None    # preset trong forge/poses.py (engine edit/pose)


@dataclass
class Job:
    name: str
    mode: str
    prompt: str | None
    character_image: str | None
    actions: list[Action]
    seed: int | None = None


@dataclass
class Settings:
    comfy_url: str = "http://127.0.0.1:8188"
    checkpoint: str | None = None          # như trong jobs.json (có thể là "auto")
    checkpoint_file: str | None = None     # file thật khi checkpoint = "auto" (preflight điền)
    seed: int = -1
    steps: int = 30
    cfg_scale: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    frame_size: int = 128
    char_height: int = 64
    palette: int = 32
    fps: int = 10
    view: str = "side"
    bg_color: str = "auto"
    preview_scale: int = 3
    debug: bool = True
    char_height_explicit: bool = False  # jobs.json có khai báo char_height không
    palette_explicit: bool = False      # jobs.json có khai báo palette không
    engine: str = "edit"
    edit: dict = field(default_factory=lambda: dict(EDIT_DEFAULTS))
    pose: dict = field(default_factory=lambda: dict(POSE_DEFAULTS))
    wan: dict = field(default_factory=lambda: dict(WAN_DEFAULTS))
    jobs: list[Job] = field(default_factory=list)

    @property
    def ground_margin(self) -> int:
        return max(1, round(self.frame_size * 0.06))


def load(path: str | Path) -> Settings:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ConfigError(f"Không tìm thấy file {path}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} không phải JSON hợp lệ (dòng {e.lineno}, cột {e.colno}): {e.msg}") from None
    return parse(raw)


def parse(raw: dict) -> Settings:
    if not isinstance(raw, dict):
        raise ConfigError("jobs.json phải là một object { ... }")
    s = Settings()
    for key in ("comfy_url", "checkpoint", "sampler", "scheduler", "view", "bg_color"):
        if raw.get(key) is not None:
            setattr(s, key, str(raw[key]))
    for key in ("seed", "steps", "frame_size", "palette", "fps", "preview_scale"):
        if raw.get(key) is not None:
            setattr(s, key, _int(raw[key], key))
    if raw.get("cfg_scale") is not None:
        s.cfg_scale = float(raw["cfg_scale"])
    if raw.get("debug") is not None:
        s.debug = bool(raw["debug"])
    s.char_height = _int(raw.get("char_height", s.frame_size // 2), "char_height")
    s.char_height_explicit = raw.get("char_height") is not None
    s.palette_explicit = raw.get("palette") is not None
    s.wan = {**WAN_DEFAULTS, **(raw.get("wan") or {})}
    s.edit = {**EDIT_DEFAULTS, **(raw.get("edit") or {})}
    s.edit["size"] = _int(s.edit["size"], "edit.size")
    if s.edit["size"] % 16 or not 256 <= s.edit["size"] <= 1536:
        raise ConfigError("edit.size phải là bội của 16, trong khoảng 256..1536")
    if not 0.1 <= float(s.edit["fallback_denoise"]) <= 1.0:
        raise ConfigError("edit.fallback_denoise phải trong khoảng 0.1..1.0")
    s.pose = {**POSE_DEFAULTS, **(raw.get("pose") or {})}
    if s.pose["layout"] not in POSE_LAYOUTS:
        raise ConfigError(f"pose.layout phải là một trong {POSE_LAYOUTS}, đang là {s.pose['layout']!r}")
    s.pose["grid_max_cells"] = _int(s.pose["grid_max_cells"], "pose.grid_max_cells")
    if not 2 <= s.pose["grid_max_cells"] <= 16:
        raise ConfigError("pose.grid_max_cells phải trong khoảng 2..16")
    s.engine = str(raw.get("engine", s.engine))
    if s.engine not in ENGINES:
        raise ConfigError(f"engine phải là một trong {ENGINES}, đang là {s.engine!r}")

    if s.view not in VIEWS:
        raise ConfigError(f"view phải là một trong {list(VIEWS)}, đang là {s.view!r}")
    if not 16 <= s.frame_size <= 1024:
        raise ConfigError("frame_size phải trong khoảng 16..1024")
    if not 8 <= s.char_height <= s.frame_size:
        raise ConfigError(f"char_height phải trong khoảng 8..frame_size ({s.frame_size})")
    if not 2 <= s.palette <= 255:
        raise ConfigError("palette phải trong khoảng 2..255")
    if s.fps < 1:
        raise ConfigError("fps phải >= 1")

    jobs = raw.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ConfigError("jobs.json cần mảng \"jobs\" có ít nhất 1 job")
    seen = set()
    for j, job in enumerate(jobs):
        parsed = _job(job, j, s)
        if parsed.name in seen:
            raise ConfigError(f"Trùng tên job: {parsed.name}")
        seen.add(parsed.name)
        s.jobs.append(parsed)
    if needs_sd(s) and not s.checkpoint:
        raise ConfigError("Cần khai báo \"checkpoint\" (SD1.5 pixel art, hoặc \"auto\"): có job "
                          "mode \"prompt\" với engine \"pose\"/\"video\", hoặc action dùng engine "
                          "\"pose\" (SD vẽ từng frame). Engine \"edit\" mặc định không cần checkpoint.")
    return s


def char_engine(s: Settings) -> str:
    """Engine tạo nhân vật từ mô tả: "edit" = FLUX.2 klein, còn lại = SD1.5."""
    return "edit" if s.engine == "edit" else "sd"


def needs_sd(s: Settings) -> bool:
    return any((job.mode == "prompt" and char_engine(s) == "sd")
               or any(a.engine == "pose" for a in job.actions) for job in s.jobs)


def _job(raw: dict, j: int, s: Settings) -> Job:
    where = f"jobs[{j}]"
    name = str(raw.get("name", ""))
    if not _NAME.match(name):
        raise ConfigError(f"{where}.name chỉ được dùng chữ, số, '_' và '-' (đang là {name!r})")
    mode = raw.get("mode", "prompt")
    if mode == "prompt" and not raw.get("prompt"):
        raise ConfigError(f"{where} ({name}): mode \"prompt\" cần trường \"prompt\" mô tả nhân vật")
    if mode == "image" and not raw.get("character_image"):
        raise ConfigError(f"{where} ({name}): mode \"image\" cần \"character_image\" (đường dẫn ảnh)")
    if mode not in ("prompt", "image"):
        raise ConfigError(f"{where} ({name}): mode phải là \"prompt\" hoặc \"image\"")
    actions_raw = raw.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise ConfigError(f"{where} ({name}): cần ít nhất 1 action")
    actions, seen = [], set()
    for a, act in enumerate(actions_raw):
        aw = f"{where}.actions[{a}]"
        aname = str(act.get("name", ""))
        if not _NAME.match(aname):
            raise ConfigError(f"{aw}.name không hợp lệ: {aname!r}")
        if aname in seen:
            raise ConfigError(f"{where} ({name}): trùng tên action {aname}")
        seen.add(aname)
        if not act.get("prompt"):
            raise ConfigError(f"{aw} ({aname}): thiếu \"prompt\" mô tả hành động")
        frames = _int(act.get("frames", 8), f"{aw}.frames")
        if not 1 <= frames <= 64:
            raise ConfigError(f"{aw} ({aname}): frames phải trong khoảng 1..64")
        align = act.get("align", "fixed")
        if align not in ALIGN_MODES:
            raise ConfigError(f"{aw} ({aname}): align phải là một trong {ALIGN_MODES}")
        engine = str(act.get("engine", s.engine))
        if engine not in ENGINES:
            raise ConfigError(f"{aw} ({aname}): engine phải là một trong {ENGINES}")
        pose = None
        if engine in ("edit", "pose"):
            pose = str(act.get("pose", aname))
            if pose not in POSE_PRESETS:
                raise ConfigError(
                    f"{aw} ({aname}): chưa có tư thế mẫu \"{pose}\". Có sẵn: "
                    f"{', '.join(POSE_PRESETS)}. Action tên khác thì thêm \"pose\": \"<tên có sẵn>\" "
                    f"(VD \"slash2\" dùng \"pose\": \"attack\").")
        default_loop = POSE_PRESETS[pose]["loop"] if pose else True
        actions.append(Action(
            name=aname, prompt=str(act["prompt"]), frames=frames,
            columns=_int(act.get("columns", frames), f"{aw}.columns"),
            duration=float(act.get("duration", s.wan["duration"])),
            align=align, loop=bool(act.get("loop", default_loop)),
            seed=_int(act["seed"], f"{aw}.seed") if act.get("seed") is not None else None,
            engine=engine, pose=pose,
        ))
    return Job(name=name, mode=mode, prompt=raw.get("prompt"),
               character_image=raw.get("character_image"), actions=actions,
               seed=_int(raw["seed"], f"{where}.seed") if raw.get("seed") is not None else None)


def _int(value, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} phải là số nguyên, đang là {value!r}") from None
