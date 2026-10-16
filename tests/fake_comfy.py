"""ComfyUI giả để test toàn luồng không cần GPU.

Mô phỏng đúng các endpoint SpriteForge dùng: /system_stats, /object_info,
/models/<folder>, /upload/image, /prompt, /history/<id>, /view, /free.
- Graph SD1.5 -> ảnh nhân vật "pixel" khối 8px trên nền xanh hơi nhiễu.
- Graph WAN   -> `length` frame: ảnh đầu vào bị dịch lên/xuống như đang nhảy,
                 màu bị nhiễu nhẹ (để kiểm tra bước ép palette).
- Graph bảng tư thế (SetLatentNoiseMask) -> giữ nguyên vùng đen của mặt nạ, vẽ nhân vật
  vào từng ô trắng theo khung xương của ô đó.
- Graph FLUX.2 klein (Flux2Scheduler): không ảnh mẫu -> nhân vật nền trắng; có ảnh mẫu
  -> trả lại ảnh 1 (con rối) như thể đã vẽ sạch. Prompt có "COPYREF" -> lỗi thật đã thấy:
  model chép nguyên ảnh 2 (tư thế đứng), trừ khi chạy img2img (SetFirstSigma).
  "BADDRAW" -> chép ảnh 2 cả khi chạy img2img (để test lớp lùi cuối: dùng con rối).
- Prompt chứa "FAIL_ME" -> history trả lỗi execution_error; "GRIDOOM" -> chỉ graph bảng
  tư thế lỗi hết VRAM (để test đường lùi về vẽ từng frame).
"""
from __future__ import annotations

import io
import json
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

from forge import workflows

MODELS = {
    "checkpoints": ["pixel_test.safetensors"],
    "diffusion_models": ["wan2.2_ti2v_5B_fp16.safetensors", "flux-2-klein-4b-fp8.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "qwen_3_4b.safetensors"],
    "vae": ["wan2.2_vae.safetensors", "flux2-vae.safetensors"],
    "controlnet": ["control_v11p_sd15_openpose_fp16.safetensors"],
    "ipadapter": ["ip-adapter-plus_sd15.safetensors"],
    "clip_vision": ["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"],
}


def fake_character(seed: int, bg: str = "green", cut: bool = False) -> Image.Image:
    """bg="black" mô phỏng lỗi thật: SD vẽ nền đen tuyền cho nhân vật tối màu.
    cut=True: nhân vật tụt xuống, chân bị cắt ở mép dưới ảnh."""
    rng = np.random.default_rng(seed)
    a = np.zeros((512, 512, 3), np.int16)
    if bg != "black":
        a[:] = {"green": (40, 190, 80), "white": (245, 245, 245)}[bg]
        a += rng.integers(-6, 7, a.shape, dtype=np.int16)  # nền không phẳng tuyệt đối
    dy = 15 if cut else 0

    def block(x0, y0, x1, y1, color):
        a[(y0 + dy) * 8:(y1 + dy) * 8, x0 * 8:x1 * 8] = color

    block(28, 14, 36, 22, (230, 190, 150))   # đầu
    block(27, 13, 37, 14, (20, 20, 30))      # tóc
    block(26, 22, 38, 38, (40, 40, 110))     # thân
    block(22, 24, 26, 34, (40, 40, 110))     # tay trái
    block(38, 24, 42, 34, (40, 40, 110))     # tay phải
    block(42, 18, 43, 34, (200, 200, 210))   # kiếm
    block(27, 38, 31, 52, (25, 25, 25))      # chân trái
    block(33, 38, 37, 52, (25, 25, 25))      # chân phải
    block(30, 16, 31, 17, (200, 30, 30))     # mắt đỏ
    block(33, 16, 34, 17, (200, 30, 30))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def fake_video(start: Image.Image, length: int, seed: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    base = np.asarray(start.convert("RGB")).astype(np.int16)
    fill = base[0, 0].copy()
    frames = []
    for i in range(length):
        dy = -int(round(40 * math.sin(math.pi * i / max(1, length - 1))))  # nhảy lên rồi rơi
        f = np.empty_like(base)
        f[:] = fill
        if dy < 0:
            f[:dy] = base[-dy:]
        else:
            f[:] = base
        f += rng.integers(-5, 6, f.shape, dtype=np.int16)
        frames.append(Image.fromarray(np.clip(f, 0, 255).astype(np.uint8), "RGB"))
    return frames


def fake_pose_frame(pose: Image.Image, sprite: Image.Image, bg: str, cut: bool,
                    seed: int, disk: bool = False, extra: bool = False) -> Image.Image:
    """Mô phỏng SD + ControlNet: vẽ nhân vật đúng chỗ và đúng chiều cao của khung xương."""
    rng = np.random.default_rng(seed)
    size = pose.size
    color = {"black": (0, 0, 0), "white": (245, 245, 245), "green": (40, 190, 80)}[bg]
    a = np.zeros((size[1], size[0], 3), np.int16)
    a[:] = color
    if bg != "black":
        a += rng.integers(-4, 5, a.shape, dtype=np.int16)
    canvas = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    x0, y0, x1, y1 = pose.convert("L").point(lambda v: 255 if v > 20 else 0).getbbox()
    h = round((y1 - y0) * 1.08)  # khung xương không có đỉnh đầu
    w = max(1, round(sprite.width * h / sprite.height))
    body = sprite.resize((w, h), Image.NEAREST)
    top = y1 - h + (150 if cut else 0)
    cx = (x0 + x1) / 2
    if disk:  # lỗi thật: đĩa tròn to phía sau nhân vật (to cỡ ảnh thật so với dáng người)
        r = h * 0.65
        ImageDraw.Draw(canvas).ellipse([cx - r, top + h * 0.45 - r, cx + r, top + h * 0.45 + r],
                                        fill=(230, 60, 20))
    canvas.paste(body, (round(cx - w / 2), top), body)
    if extra:  # lỗi thật: thêm nhân vật thứ hai cạnh bên (kiểu character sheet)
        canvas.paste(body, (round(cx - w / 2 + w * 1.7), top), body)
    return canvas


def _cell_size(m: np.ndarray) -> int:
    """Cạnh ô của bảng tư thế = ước chung của cỡ ảnh và mọi chỗ mặt nạ đổi giá trị
    (ô 0 luôn giữ nguyên nên luôn có một chỗ đổi ở đúng cạnh ô đầu tiên)."""
    d = m.astype(np.int8)
    edges = [m.shape[0], m.shape[1]]
    edges += list(np.flatnonzero(np.diff(d, axis=1).any(0)) + 1)
    edges += list(np.flatnonzero(np.diff(d, axis=0).any(1)) + 1)
    return int(np.gcd.reduce(np.array(edges)))


def fake_board(pose: Image.Image, init: Image.Image, mask: Image.Image, sprite: Image.Image,
               bg: str, cut: bool, seed: int, disk: bool = False, extra: bool = False) -> Image.Image:
    m = np.asarray(mask.convert("RGB"))[:, :, 0] > 127
    cell = _cell_size(m)
    board = init.convert("RGB").copy()
    for y in range(0, board.height, cell):
        for x in range(0, board.width, cell):
            if m[y + cell // 2, x + cell // 2]:
                tile = fake_pose_frame(pose.crop((x, y, x + cell, y + cell)), sprite, bg, cut,
                                       seed + x + y, disk=disk, extra=extra)
                board.paste(tile, (x, y))
    return board


def fake_klein(graph: dict, uploads: dict, texts: str) -> Image.Image:
    seed = graph["8"]["inputs"]["noise_seed"]
    rng = np.random.default_rng(seed)
    refs = [graph[k]["inputs"]["image"] for k in sorted(
        (k for k, n in graph.items() if n["class_type"] == "LoadImage" and k.isdigit()
         and 20 <= int(k) < 100), key=int)]
    if not refs:  # tạo nhân vật
        img = fake_character(seed, bg="white", cut="CUTFEET" in texts)
        if "EXTRAFIG" in texts:  # lỗi thật kiểu "character sheet": 2 nhân vật cạnh nhau
            a = np.asarray(img).copy()
            a[:, :160] = a[:, 176:336]
            img = Image.fromarray(a)
        return img
    img2img = any(n["class_type"] == "SetFirstSigma" for n in graph.values())
    copy = len(refs) > 1 and ("BADDRAW" in texts or ("COPYREF" in texts and not img2img))
    src = uploads[refs[1] if copy else refs[0]]
    a = np.asarray(src.convert("RGB")).astype(np.int16)
    a += rng.integers(-3, 4, a.shape, dtype=np.int16)  # model vẽ lại: màu lệch nhẹ
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _sprite_from(image: Image.Image) -> Image.Image:
    """Nhân vật (RGBA, cắt sát) từ ảnh nền trơn: dùng cho ảnh mẫu IP-Adapter."""
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    mask = np.abs(a - a[0, 0]).max(2) > 30
    rgba = Image.fromarray(np.dstack([a.astype(np.uint8), (mask * 255).astype(np.uint8)]), "RGBA")
    return rgba.crop(rgba.getbbox())


class FakeComfy:
    def __init__(self, ipadapter: bool = True, grid: bool = True, klein: bool = True):
        self.ipadapter = ipadapter
        self.grid = grid
        self.klein = klein
        self.uploads: dict[str, Image.Image] = {}
        self.files: dict[str, Image.Image] = {}
        self.history: dict[str, dict] = {}
        self.prompts: list[dict] = []
        self.lock = threading.Lock()
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    # -------------------------------------------------------------- logic
    def run_graph(self, graph: dict) -> str:
        pid = uuid.uuid4().hex
        self.prompts.append(graph)
        classes = {n["class_type"] for n in graph.values()}
        texts = " ".join(n["inputs"].get("text", "") for n in graph.values()
                         if n["class_type"] == "CLIPTextEncode")
        if "FAIL_ME" in texts or ("GRIDOOM" in texts and "SetLatentNoiseMask" in classes):
            self.history[pid] = {"status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_id": "8", "node_type": "KSampler",
                                     "exception_type": "RuntimeError",
                                     "exception_message": "fake CUDA out of memory"}]]},
                "outputs": {}}
            return pid
        if "Flux2Scheduler" in classes:
            images, out_node = [fake_klein(graph, self.uploads, texts)], workflows.KLEIN_OUTPUT
        elif "UNETLoader" in classes:
            node = next(n for n in graph.values() if n["class_type"] == "Wan22ImageToVideoLatent")
            start = self.uploads[graph["4"]["inputs"]["image"]]
            seed = next(n for n in graph.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
            images, out_node = fake_video(start, node["inputs"]["length"], seed), workflows.WAN_OUTPUT
        else:
            seed = graph["3"]["inputs"]["seed"]
            bg = ("black" if "BLACKBG" in texts else
                  "white" if "white background" in texts else "green")
            if "ControlNetApplyAdvanced" in classes:
                pose = self.uploads[graph["21"]["inputs"]["image"]]
                ref = graph.get("31", {}).get("inputs", {}).get("image")
                flags = {"disk": "DISKBG" in texts, "extra": "EXTRAFIG" in texts}
                if "SetLatentNoiseMask" in classes:
                    init = self.uploads[graph["40"]["inputs"]["image"]]
                    sprite = _sprite_from(self.uploads[ref] if ref else init)
                    images = [fake_board(pose, init, self.uploads[graph["42"]["inputs"]["image"]],
                                         sprite, bg, "CUTFEET" in texts, seed, **flags)]
                else:
                    sprite = _sprite_from(self.uploads[ref] if ref else fake_character(seed, bg="white"))
                    images = [fake_pose_frame(pose, sprite, bg, "CUTFEET" in texts, seed, **flags)]
            else:
                images = [fake_character(seed, bg=bg, cut="CUTFEET" in texts)]
            out_node = workflows.SD15_OUTPUT
        infos = []
        for im in images:
            name = f"{uuid.uuid4().hex}.png"
            self.files[name] = im
            infos.append({"filename": name, "subfolder": "", "type": "temp"})
        self.history[pid] = {"status": {"status_str": "success", "completed": True, "messages": []},
                             "outputs": {out_node: {"images": infos}}}
        return pid

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, data, code=200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/system_stats":
                    return self._json({"system": {"os": "fake"}, "devices": []})
                if url.path == "/object_info":
                    nodes = set(workflows.WAN_NODES) | set(workflows.SD15_NODES) | set(workflows.POSE_NODES)
                    if fake.ipadapter:
                        nodes |= set(workflows.IPADAPTER_NODES)
                    if fake.grid:
                        nodes |= set(workflows.GRID_NODES)
                    if fake.klein:
                        nodes |= set(workflows.KLEIN_NODES)
                    return self._json({n: {} for n in nodes})
                if url.path.startswith("/models/"):
                    return self._json(MODELS.get(url.path.split("/")[-1], []))
                if url.path.startswith("/history/"):
                    pid = url.path.split("/")[-1]
                    entry = fake.history.get(pid)
                    return self._json({pid: entry} if entry else {})
                if url.path == "/view":
                    name = parse_qs(url.query)["filename"][0]
                    buf = io.BytesIO()
                    fake.files[name].save(buf, format="PNG")
                    body = buf.getvalue()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return None
                return self._json({"error": "not found"}, 404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/prompt":
                    graph = json.loads(body)["prompt"]
                    with fake.lock:
                        pid = fake.run_graph(graph)
                    return self._json({"prompt_id": pid, "number": 0, "node_errors": {}})
                if self.path == "/upload/image":
                    ctype = self.headers["Content-Type"]
                    boundary = ctype.split("boundary=")[1].encode()
                    for part in body.split(b"--" + boundary):
                        if b'name="image"' in part:
                            header, data = part.split(b"\r\n\r\n", 1)
                            name = header.split(b'filename="')[1].split(b'"')[0].decode()
                            fake.uploads[name] = Image.open(io.BytesIO(data.rstrip(b"\r\n"))).copy()
                            return self._json({"name": name, "subfolder": "", "type": "input"})
                    return self._json({"error": "no image"}, 400)
                if self.path == "/free":
                    return self._json({})
                return self._json({"error": "not found"}, 404)

        return Handler
