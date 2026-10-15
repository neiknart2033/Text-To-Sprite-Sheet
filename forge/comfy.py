"""Client HTTP tối giản cho ComfyUI (chỉ dùng thư viện chuẩn).

Viết lại thay cho `api/comfyui_client.py` của Pixel Art Studio, vì bản gốc:
  - đổi seed -1 thành 0 (mọi lần gen ra cùng một ảnh),
  - bỏ qua tham số ControlNet/LoRA,
  - timeout cứng 150 giây (không đủ cho video diffusion).

Cách chờ kết quả học từ `comfyui_bridge._attendi` của Sprite Sheep: khi VAE
decode, ComfyUI chiếm GPU 100% và server HTTP có thể im hàng chục giây. Một lần
timeout không phải lỗi, chỉ là "đang bận"; chỉ bỏ cuộc khi im liên tục quá lâu.
"""
from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image

log = logging.getLogger(__name__)

POLL_S = 2.0  # chu kỳ hỏi /history (test đặt nhỏ lại)


class ComfyError(RuntimeError):
    """Lỗi từ ComfyUI (graph bị từ chối, node lỗi khi chạy, mất kết nối...)."""


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, data: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise ComfyError(f"{method} {path} -> HTTP {e.code}: {_explain(body)}") from e

    def _get_json(self, path: str, timeout: float | None = None):
        return json.loads(self._request("GET", path, timeout=timeout))

    def _post_json(self, path: str, payload: dict, timeout: float | None = None):
        raw = self._request("POST", path, json.dumps(payload).encode(),
                            {"Content-Type": "application/json"}, timeout)
        return json.loads(raw) if raw.strip() else {}

    # ------------------------------------------------------------- truy vấn
    def ping(self) -> dict:
        try:
            return self._get_json("/system_stats", timeout=5)
        except (urllib.error.URLError, OSError) as e:
            raise ComfyError(f"Không kết nối được ComfyUI tại {self.url}: {e}") from e

    def missing_nodes(self, class_types: list[str]) -> list[str]:
        info = self._get_json("/object_info", timeout=60)
        return [c for c in class_types if c not in info]

    def list_models(self, folder: str) -> list[str]:
        try:
            return list(self._get_json(f"/models/{folder}"))
        except ComfyError:
            return []

    def free(self) -> None:
        """Gỡ model khỏi VRAM/RAM (giữa lượt SD1.5 và lượt WAN)."""
        try:
            self._post_json("/free", {"unload_models": True, "free_memory": True})
        except (ComfyError, urllib.error.URLError, OSError) as e:
            log.warning("Không gọi được /free: %s", e)

    # ------------------------------------------------------------ ảnh vào/ra
    def upload_image(self, image: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in (("type", "input"), ("overwrite", "true")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; '
                         f'name="{key}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                     f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode())
        parts.append(buf.getvalue())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        raw = self._request("POST", "/upload/image", b"".join(parts),
                            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
                            timeout=120)
        return json.loads(raw).get("name", name)

    def fetch(self, file_info: dict) -> Image.Image:
        query = urllib.parse.urlencode({
            "filename": file_info["filename"],
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        })
        raw = self._request("GET", f"/view?{query}", timeout=120)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image

    # ------------------------------------------------------------ chạy graph
    def queue(self, graph: dict) -> str:
        r = self._post_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        prompt_id = r.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI từ chối graph: {_explain(json.dumps(r))}")
        return prompt_id

    def wait(self, prompt_id: str, timeout_s: float = 3600, poll_s: float | None = None,
             max_silent_s: float = 900) -> dict:
        """Chờ prompt chạy xong, trả về `outputs` của history."""
        deadline = time.time() + timeout_s
        silent_since = None
        while time.time() < deadline:
            time.sleep(POLL_S if poll_s is None else poll_s)
            try:
                hist = self._get_json(f"/history/{prompt_id}", timeout=30)
                silent_since = None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                silent_since = silent_since or time.time()
                if time.time() - silent_since > max_silent_s:
                    raise ComfyError(f"ComfyUI không phản hồi quá {max_silent_s:.0f}s: {e}") from e
                continue
            except ComfyError:
                continue
            entry = hist.get(prompt_id)
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(_execution_error(status))
            outputs = entry.get("outputs") or {}
            if outputs and status.get("completed", True):
                return outputs
            if status.get("completed"):
                # Xong mà không có ảnh (VD toàn bộ graph lấy từ cache): báo ngay, đừng chờ hết giờ
                raise ComfyError("ComfyUI chạy xong nhưng không trả về ảnh nào (graph bị cache?). "
                                 "Đổi seed rồi chạy lại.")
        raise ComfyError(f"Quá thời gian chờ {timeout_s:.0f}s cho prompt {prompt_id}")

    def run(self, graph: dict, output_node: str, timeout_s: float = 3600) -> list[dict]:
        """Chạy graph, trả về danh sách file ảnh (theo thứ tự batch) của node đầu ra."""
        outputs = self.wait(self.queue(graph), timeout_s=timeout_s)
        images = outputs.get(output_node, {}).get("images", [])
        if not images:
            raise ComfyError(f"Node {output_node} không trả về ảnh nào")
        return images


def _explain(body: str) -> str:
    """Rút gọn lỗi validate của ComfyUI (VD: thiếu file model) cho dễ đọc."""
    try:
        data = json.loads(body)
    except ValueError:
        return body[:500]
    msgs = []
    err = data.get("error")
    if isinstance(err, dict):
        msgs.append(f"{err.get('message', '')} {err.get('details', '')}".strip())
    for node_id, node_err in (data.get("node_errors") or {}).items():
        for e in node_err.get("errors", []):
            msgs.append(f"node {node_id} ({node_err.get('class_type')}): "
                        f"{e.get('message')} {e.get('details', '')}".strip())
    return " | ".join(m for m in msgs if m) or body[:500]


def _execution_error(status: dict) -> str:
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            return (f"Node {data.get('node_id')} ({data.get('node_type')}) lỗi: "
                    f"{data.get('exception_type')}: {data.get('exception_message', '').strip()}")
    return f"ComfyUI báo lỗi: {status}"
