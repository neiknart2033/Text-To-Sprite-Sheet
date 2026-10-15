"""Đọc và kiểm tra jobs.json (schema: xem KAGGLE_DESIGN.md mục 6)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

WAN_DEFAULTS = {
    "unet": "wan2.2_ti2v_5B_fp16.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
    "width": 448,
    "height": 448,
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
    checkpoint: str | None = None
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
    s.char_height = _int(raw.get("char_height", s.frame_size // 2), "char_height")
    s.wan = {**WAN_DEFAULTS, **(raw.get("wan") or {})}

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
    if any(job.mode == "prompt" for job in s.jobs) and not s.checkpoint:
        raise ConfigError("Có job mode \"prompt\" nên cần khai báo \"checkpoint\" (SD1.5 pixel art)")
    return s


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
        actions.append(Action(
            name=aname, prompt=str(act["prompt"]), frames=frames,
            columns=_int(act.get("columns", frames), f"{aw}.columns"),
            duration=float(act.get("duration", s.wan["duration"])),
            align=align, loop=bool(act.get("loop", True)),
            seed=_int(act["seed"], f"{aw}.seed") if act.get("seed") is not None else None,
        ))
    return Job(name=name, mode=mode, prompt=raw.get("prompt"),
               character_image=raw.get("character_image"), actions=actions,
               seed=_int(raw["seed"], f"{where}.seed") if raw.get("seed") is not None else None)


def _int(value, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} phải là số nguyên, đang là {value!r}") from None
