#!/usr/bin/env python
"""Chạy batch SpriteForge headless: jobs.json -> sprite sheet cho mọi action.

    python kaggle/run_batch.py --jobs jobs.json --out /kaggle/working/output

Hai lượt để không nạp đi nạp lại model:
  lượt 1  nhân vật của mọi job (SD1.5, trên ComfyUI đầu tiên), rồi gỡ SD khỏi VRAM;
  lượt 2  animation của mọi action (WAN 2.2), chia đều cho mọi ComfyUI trong
          --comfy-url (Kaggle "GPU T4 x2": 2 ComfyUI, mỗi cái một GPU -> nhanh ~2 lần).
Resume: action đã "done" trong <job>/progress.json thì bỏ qua. Một action lỗi
được ghi log rồi chạy tiếp action sau, không dừng cả batch.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from forge import config, pipelines, pixel, workflows  # noqa: E402
from forge.comfy import ComfyClient, ComfyError  # noqa: E402

log = logging.getLogger("spriteforge")


# GPU thứ 2 trở đi chờ trước khi nhận việc đầu tiên: tránh hai tiến trình ComfyUI
# cùng nạp ~17GB model vào RAM (~29GB trên Kaggle) một lúc.
STAGGER_S = 45.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SpriteForge batch: jobs.json -> sprite sheets")
    ap.add_argument("--jobs", required=True, help="đường dẫn jobs.json")
    ap.add_argument("--out", required=True, help="thư mục xuất kết quả")
    ap.add_argument("--comfy-url", help="ghi đè comfy_url trong jobs.json; nhiều GPU = nhiều URL "
                                        "cách nhau dấu phẩy, mỗi URL một ComfyUI")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out / "run_log.txt")

    try:
        cfg = config.load(args.jobs)
    except config.ConfigError as e:
        log.error("jobs.json lỗi: %s", e)
        return 2
    urls = [u.strip() for u in (args.comfy_url or cfg.comfy_url).split(",") if u.strip()]
    clients = []
    for url in urls:
        c = ComfyClient(url)
        try:
            c.ping()
            clients.append(c)
        except ComfyError as e:
            log.warning("Bỏ qua %s: %s", url, e)
    if not clients:
        log.error("Không kết nối được ComfyUI nào: %s", ", ".join(urls))
        return 2
    client = clients[0]
    try:
        problems = preflight(client, cfg, Path(args.jobs).resolve().parent, out)
    except ComfyError as e:
        log.error("%s", e)
        return 2
    if problems:
        for p in problems:
            log.error("Thiếu: %s", p)
        log.error("Dừng trước khi chạy: sửa các mục trên rồi chạy lại.")
        return 2

    started = time.time()
    total_actions = sum(len(j.actions) for j in cfg.jobs)
    log.info("Bắt đầu: %d job, %d action, %d GPU -> %s", len(cfg.jobs), total_actions,
             len(clients), out)

    # ---------------- lượt 1: nhân vật
    characters, char_errors, generated = {}, {}, False
    for job in cfg.jobs:
        job_dir = out / job.name
        try:
            native, palette, made = ensure_character(client, cfg, job, job_dir,
                                                     Path(args.jobs).resolve().parent)
            characters[job.name] = (native, palette)
            generated |= made
        except Exception as e:  # noqa: BLE001 - lỗi 1 job không được dừng cả batch
            char_errors[job.name] = f"{type(e).__name__}: {e}"
            log.error("[%s] Lỗi tạo nhân vật: %s\n%s", job.name, e, traceback.format_exc())
    if generated:
        client.free()

    # ---------------- lượt 2: animation, mỗi GPU (ComfyUI) một luồng lấy việc từ hàng đợi
    counts = {"done": 0, "skipped": 0, "error": 0}
    progress_by_job: dict[str, dict] = {}
    tasks: queue.Queue = queue.Queue()
    for job in cfg.jobs:
        job_dir = out / job.name
        progress = progress_by_job[job.name] = _load_progress(job_dir)
        if job.name not in characters:
            for act in job.actions:
                progress[act.name] = {"status": "error", "error": f"nhân vật lỗi: {char_errors[job.name]}"}
                counts["error"] += 1
            _save_progress(job_dir, progress)
            continue
        native = characters[job.name][0]
        for act in job.actions:
            entry = progress.get(act.name)
            fingerprint = action_fingerprint(cfg, job, act, native)
            if (_status(entry) == "done" and (job_dir / act.name / "spritesheet.png").exists()
                    and isinstance(entry, dict) and entry.get("fingerprint") == fingerprint):
                log.info("[%s/%s] đã xong trước đó, bỏ qua", job.name, act.name)
                counts["skipped"] += 1
                continue
            if _status(entry) == "done":
                log.info("[%s/%s] cấu hình, nhân vật hoặc thuật toán đã đổi -> làm lại",
                         job.name, act.name)
            tasks.put((job, act, fingerprint))

    lock = threading.Lock()

    def worker(index: int, worker_client: ComfyClient) -> None:
        wait_until = time.time() + STAGGER_S * index
        while time.time() < wait_until and not tasks.empty():
            time.sleep(0.5)
        gpu = f"gpu{index} " if len(clients) > 1 else ""
        while True:
            try:
                job, act, fingerprint = tasks.get_nowait()
            except queue.Empty:
                return
            status = run_action(worker_client, cfg, job, act, characters[job.name],
                                out / job.name, f"[{gpu}{job.name}/{act.name}]")
            status["fingerprint"] = fingerprint
            with lock:
                progress_by_job[job.name][act.name] = status
                counts[status["status"]] += 1
                _save_progress(out / job.name, progress_by_job[job.name])

    threads = [threading.Thread(target=worker, args=(i, c), daemon=True) for i, c in enumerate(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log.info("=" * 60)
    log.info("Tổng kết: %d xong, %d bỏ qua (đã có), %d lỗi / %d action. Thời gian: %s",
             counts["done"], counts["skipped"], counts["error"], total_actions,
             _fmt(time.time() - started))
    if counts["error"]:
        log.info("Chạy lại cùng lệnh để thử lại các action lỗi (action đã xong sẽ được bỏ qua).")
    if counts["skipped"] and not counts["done"] and not counts["error"]:
        log.info("Mọi action đã có sẵn từ lần chạy trước, không tạo gì mới. Muốn tạo lại: đổi seed/"
                 "prompt trong jobs.json, hoặc xoá thư mục %s", out)
    return 1 if counts["error"] else 0


# Tăng số này khi thuật toán đổi: kết quả cũ (tạo bằng code cũ) tự được làm lại.
# v2: tách nền nhân vật loang từ mép (sửa lỗi nền đen ăn thủng nhân vật).
# v3: nhân vật nền trắng 512x768 + loại ảnh bị cắt mép; WAN 640px, key màu phông thật,
#     chọn frame lành trong ±2 frame.
# v4: engine "pose" (khung xương + SD1.5 + ControlNet OpenPose + IP-Adapter) mặc định.
PIPELINE_VERSION = 4


def _digest(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def character_fingerprint(cfg: config.Settings, job: config.Job, jobs_dir: Path) -> str:
    """Mọi thứ quyết định nhân vật. Đổi một trong số này -> tạo lại nhân vật.

    Mode "image" tính cả NỘI DUNG file ảnh mẫu: upload bản sửa cùng tên vẫn được nhận ra.
    """
    source = None
    if job.mode == "image":
        src = _resolve_path(job.character_image, jobs_dir)
        source = hashlib.sha1(src.read_bytes()).hexdigest() if src else "missing"
    return _digest(PIPELINE_VERSION, job.mode, job.prompt, job.character_image, source, job.seed,
                   cfg.seed, cfg.checkpoint, cfg.steps, cfg.cfg_scale, cfg.sampler, cfg.scheduler,
                   cfg.char_height, cfg.char_height_explicit, cfg.palette, cfg.palette_explicit,
                   cfg.frame_size, cfg.view, cfg.bg_color, cfg.engine,
                   {k: cfg.pose[k] for k in ("controlnet", "strength", "size")})


def action_fingerprint(cfg: config.Settings, job: config.Job, act: config.Action, native) -> str:
    """Mọi thứ quyết định một action, gồm cả ảnh nhân vật thật (đổi nhân vật -> làm lại)."""
    engine_cfg = cfg.pose if act.engine == "pose" else cfg.wan
    return _digest(PIPELINE_VERSION, dataclasses.asdict(act), job.prompt, job.seed, cfg.seed,
                   cfg.steps, cfg.cfg_scale, cfg.sampler, cfg.scheduler, cfg.checkpoint,
                   cfg.frame_size, cfg.fps, cfg.preview_scale, cfg.view, cfg.bg_color, engine_cfg,
                   hashlib.sha1(native.tobytes()).hexdigest())


def _character_is_current(cfg: config.Settings, job: config.Job, job_dir: Path,
                          jobs_dir: Path) -> bool:
    try:
        info = json.loads((job_dir / "character.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return False
    return ((job_dir / "character.png").exists()
            and info.get("fingerprint") == character_fingerprint(cfg, job, jobs_dir))


def run_action(client: ComfyClient, cfg: config.Settings, job: config.Job, act: config.Action,
               character, job_dir: Path, tag: str) -> dict:
    """Một action: tạo frame (pose/video) -> hậu xử lý -> 3 file. Không ném lỗi ra ngoài."""
    native, palette = character
    t0 = time.time()
    if act.engine == "pose":
        log.info("%s bắt đầu: %d frame, tư thế \"%s\"", tag, act.frames, act.pose)
    else:
        log.info("%s bắt đầu: %d frame, clip WAN %.1fs", tag, act.frames, act.duration)
    animate = pipelines.animate_pose if act.engine == "pose" else pipelines.animate_video
    try:
        frames, layout, info = animate(
            client, cfg, native, palette, act,
            description=job.prompt, seed=job.seed,  # mode "image": prompt (nếu có) mô tả ảnh mẫu
            debug_dir=job_dir / act.name / "debug" if cfg.debug else None)
        result = pipelines.finalize(frames, layout, act, cfg, job_dir / act.name,
                                    meta={"job": job.name, **info})
        secs = time.time() - t0
        sizes = ", ".join(f"{p.name} {_kb(p)}" for p in result["paths"].values())
        log.info("%s xong trong %s (%s)", tag, _fmt(secs), sizes)
        return {"status": "done", "seconds": round(secs, 1), "seed": info["seed"]}
    except Exception as e:  # noqa: BLE001 - lỗi 1 action không được dừng cả batch
        secs = time.time() - t0
        log.error("%s LỖI sau %s: %s\n%s", tag, _fmt(secs), e, traceback.format_exc())
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "seconds": round(secs, 1)}


def preflight(client: ComfyClient, cfg: config.Settings, jobs_dir: Path, out: Path) -> list[str]:
    """Kiểm tra node, model, ảnh đầu vào TRƯỚC khi tốn giờ GPU."""
    problems = []
    actions = [a for j in cfg.jobs for a in j.actions]
    use_video = any(a.engine == "video" for a in actions)
    need_char = any(j.mode == "prompt" and not _character_is_current(cfg, j, out / j.name, jobs_dir)
                    for j in cfg.jobs)
    use_pose = any(a.engine == "pose" for a in actions) or (need_char and cfg.engine == "pose")
    need_sd = need_char or use_pose
    nodes = ((workflows.WAN_NODES if use_video else []) + (workflows.SD15_NODES if need_sd else [])
             + (workflows.POSE_NODES if use_pose else []))
    for node in client.missing_nodes(sorted(set(nodes))):
        problems.append(f"node ComfyUI '{node}' (cập nhật ComfyUI lên bản mới)")

    wanted = []
    if use_video:
        wanted += [("diffusion_models", cfg.wan["unet"]), ("text_encoders", cfg.wan["text_encoder"]),
                   ("vae", cfg.wan["vae"])]
    if use_pose:
        wanted.append(("controlnet", cfg.pose["controlnet"]))
        if cfg.pose.get("ipadapter"):
            reason = None
            if client.missing_nodes(workflows.IPADAPTER_NODES):
                reason = "chưa cài custom node ComfyUI_IPAdapter_plus"
            elif not any("ip-adapter-plus_sd15" in m for m in client.list_models("ipadapter")):
                reason = "thiếu model ipadapter/ip-adapter-plus_sd15.safetensors"
            elif not any("CLIP-ViT-H-14" in m for m in client.list_models("clip_vision")):
                reason = "thiếu model clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            if reason:
                cfg.pose["ipadapter"] = False
                log.warning("Tắt IP-Adapter (%s): nhân vật sẽ kém giống nhau giữa các frame.", reason)
    if need_sd and cfg.checkpoint == "auto":
        checkpoints = client.list_models("checkpoints")
        if checkpoints:
            cfg.checkpoint_file = checkpoints[0]
            log.info("checkpoint \"auto\" -> %s", cfg.checkpoint_file)
        else:
            problems.append("checkpoint SD1.5 nào đó trong checkpoints/ (Dataset chưa có file .safetensors)")
    elif need_sd:
        wanted.append(("checkpoints", cfg.checkpoint))
    for folder, name in wanted:
        available = client.list_models(folder)
        if available and name not in available:
            problems.append(f"model '{name}' trong {folder}/ (ComfyUI đang thấy: "
                            f"{', '.join(available[:8]) or 'không có gì'})")

    for job in cfg.jobs:
        if job.mode == "image" and _resolve_path(job.character_image, jobs_dir) is None:
            problems.append(f"ảnh nhân vật '{job.character_image}' của job {job.name} (đã tìm "
                            f"cạnh jobs.json và theo tên file trong {', '.join(map(str, SEARCH_ROOTS))})")
    return problems


def ensure_character(client: ComfyClient, cfg: config.Settings, job: config.Job,
                     job_dir: Path, jobs_dir: Path):
    """Trả về (sprite gốc, palette, có_gọi_GPU).

    Dùng lại character.png nếu được tạo với đúng cấu hình + thuật toán hiện tại; muốn
    nhân vật khác với cùng cấu hình thì xoá character.png hoặc đổi seed.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    char_path = job_dir / "character.png"
    user_image = job.mode == "image"
    if _character_is_current(cfg, job, job_dir, jobs_dir):
        native = Image.open(char_path).convert("RGBA")
        log.info("[%s] dùng lại nhân vật có sẵn %s (%dx%d)", job.name, char_path.name, *native.size)
        limit = 255 if user_image and not cfg.palette_explicit else cfg.palette
        return native, pipelines.palette_of(native, limit), False
    if char_path.exists():
        log.info("[%s] cấu hình/ảnh nhân vật hoặc thuật toán đã đổi -> tạo lại nhân vật", job.name)

    t0 = time.time()
    info: dict = {"mode": job.mode, "fingerprint": character_fingerprint(cfg, job, jobs_dir)}
    if job.mode == "prompt":
        log.info("[%s] tạo nhân vật từ prompt...", job.name)
        raw, gen = pipelines.generate_character(client, cfg, job.prompt, job.seed)
        raw.save(job_dir / "character_raw.png")
        info.update(gen)
    else:
        src = _resolve_path(job.character_image, jobs_dir)
        info["source"] = str(src)
        log.info("[%s] dùng ảnh mẫu %s", job.name, src)
        raw = Image.open(src)
        raw.load()
    native, palette = pipelines.prepare_character(raw, cfg, user_image=user_image)
    native.save(char_path)
    pixel.upscale(native, max(1, cfg.preview_scale * 2)).save(job_dir / "character_preview.png")
    info.update({"size": list(native.size), "colors": len(palette)})
    (job_dir / "character.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    log.info("[%s] nhân vật %dx%d, %d màu (%s)", job.name, *native.size, len(palette),
             _fmt(time.time() - t0))
    return native, palette, job.mode == "prompt"


# ------------------------------------------------------------------ tiện ích
# Nơi tìm ảnh mẫu theo TÊN FILE khi đường dẫn không trỏ thẳng tới file. Kaggle gắn
# dataset vào đường dẫn dài (VD /kaggle/input/datasets/<user>/<dataset>/...), người
# dùng chỉ cần ghi "hero.png".
SEARCH_ROOTS = [Path("/kaggle/input")]


def _resolve_path(path: str | None, jobs_dir: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    for candidate in ([p] if p.is_absolute() else [jobs_dir / p, Path.cwd() / p]):
        if candidate.is_file():
            return candidate
    for root in SEARCH_ROOTS:
        found = sorted(f for f in root.rglob(p.name) if f.is_file()) if root.is_dir() else []
        if found:
            if len(found) > 1:
                log.warning("Có %d file tên %s, dùng %s", len(found), p.name, found[0])
            return found[0]
    return None


def _status(entry) -> str | None:
    return entry if isinstance(entry, str) else (entry or {}).get("status")


def _load_progress(job_dir: Path) -> dict:
    try:
        return json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save_progress(job_dir: Path, progress: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_dir / "progress.json")


def _kb(path: Path) -> str:
    return f"{path.stat().st_size / 1024:.1f}KB"


def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s" if m else f"{secs:.1f}s"


def _setup_logging(log_file: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    if hasattr(sys.stdout, "reconfigure"):  # console Windows không phải UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


if __name__ == "__main__":
    sys.exit(main())
