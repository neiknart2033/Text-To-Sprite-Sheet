"""ComfyUI giả để test toàn luồng không cần GPU.

Mô phỏng đúng các endpoint SpriteForge dùng: /system_stats, /object_info,
/models/<folder>, /upload/image, /prompt, /history/<id>, /view, /free.
- Graph SD1.5 -> ảnh nhân vật "pixel" khối 8px trên nền xanh hơi nhiễu.
- Graph WAN   -> `length` frame: ảnh đầu vào bị dịch lên/xuống như đang nhảy,
                 màu bị nhiễu nhẹ (để kiểm tra bước ép palette).
- Prompt chứa "FAIL_ME" -> history trả lỗi execution_error.
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
from PIL import Image

from forge import workflows

MODELS = {
    "checkpoints": ["pixel_test.safetensors"],
    "diffusion_models": ["wan2.2_ti2v_5B_fp16.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan2.2_vae.safetensors"],
}


def fake_character(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.zeros((512, 512, 3), np.int16)
    a[:] = (40, 190, 80)
    a += rng.integers(-6, 7, a.shape, dtype=np.int16)  # nền không phẳng tuyệt đối

    def block(x0, y0, x1, y1, color):
        a[y0 * 8:y1 * 8, x0 * 8:x1 * 8] = color

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


class FakeComfy:
    def __init__(self):
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
        if "FAIL_ME" in texts:
            self.history[pid] = {"status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_id": "8", "node_type": "KSampler",
                                     "exception_type": "RuntimeError",
                                     "exception_message": "fake CUDA out of memory"}]]},
                "outputs": {}}
            return pid
        if "UNETLoader" in classes:
            node = next(n for n in graph.values() if n["class_type"] == "Wan22ImageToVideoLatent")
            start = self.uploads[graph["4"]["inputs"]["image"]]
            seed = next(n for n in graph.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
            images, out_node = fake_video(start, node["inputs"]["length"], seed), workflows.WAN_OUTPUT
        else:
            seed = graph["3"]["inputs"]["seed"]
            images, out_node = [fake_character(seed)], workflows.SD15_OUTPUT
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
                    nodes = set(workflows.WAN_NODES) | set(workflows.SD15_NODES)
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
