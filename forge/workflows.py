"""Graph ComfyUI (định dạng API), chỉ dùng node có sẵn trong core của ComfyUI.

- `sd15_txt2img`: sinh nhân vật gốc bằng checkpoint SD1.5 pixel art.
- `wan22_i2v`: ảnh nhân vật + prompt hành động -> clip video (WAN 2.2 TI2V 5B).
  Phỏng theo `backend/comfyui_wan.py` của Sprite Sheep (MIT), nhưng trả thẳng
  batch frame qua `PreviewImage` thay cho `ImageGrid` + `SaveImage`: không phải
  ghép rồi cắt lại lưới, và chỉ cần tải về đúng những frame được chọn.
"""
from __future__ import annotations

SD15_NODES = ["CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
              "KSampler", "VAEDecode", "PreviewImage"]

WAN_NODES = ["UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "CLIPTextEncode",
             "ModelSamplingSD3", "Wan22ImageToVideoLatent", "KSampler", "VAEDecode",
             "PreviewImage"]

POSE_NODES = SD15_NODES + ["LoadImage", "ControlNetLoader", "ControlNetApplyAdvanced"]
# Custom node ComfyUI_IPAdapter_plus (cubiq). Thiếu thì engine pose vẫn chạy, chỉ yếu
# khả năng giữ ngoại hình nhân vật giữa các frame.
IPADAPTER_NODES = ["IPAdapterUnifiedLoader", "IPAdapter"]

# Node đầu ra của từng graph (để lấy ảnh từ history)
SD15_OUTPUT = "9"
POSE_OUTPUT = "9"
WAN_OUTPUT = "10"

# Ô chọn kiểu "enum" mà các phiên bản node hay đổi tên giá trị: giá trị không còn hợp lệ
# thì thay bằng lựa chọn đầu tiên. KHÔNG áp dụng cho tên file (model, ảnh): thay âm
# thầm sang file khác còn tệ hơn báo lỗi.
FLEXIBLE_CHOICES = {"weight_type", "preset", "combine_embeds", "embeds_scaling"}

# WAN 2.2 TI2V 5B: 24 fps, độ dài clip hợp lệ = 4n + 1
WAN_FPS = 24
WAN_MAX_LENGTH = 241


def sd15_txt2img(checkpoint: str, prompt: str, negative: str, seed: int,
                 width: int = 512, height: int = 512, steps: int = 30,
                 cfg: float = 7.0, sampler: str = "euler",
                 scheduler: str = "normal") -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
            "scheduler": scheduler, "denoise": 1.0, "model": ["4", 0],
            "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        SD15_OUTPUT: {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    }


def sd15_pose(checkpoint: str, prompt: str, negative: str, seed: int, pose_image: str,
              controlnet: str, strength: float = 1.0, reference_image: str | None = None,
              ipadapter_weight: float = 0.8, ipadapter_preset: str = "PLUS (high strength)",
              size: int = 512, steps: int = 30, cfg: float = 7.0, sampler: str = "euler",
              scheduler: str = "normal") -> dict:
    """SD1.5 + ControlNet OpenPose (+ IP-Adapter giữ ngoại hình nếu có ảnh mẫu)."""
    graph = sd15_txt2img(checkpoint, prompt, negative, seed, width=size, height=size,
                         steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler)
    graph["20"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}}
    graph["21"] = {"class_type": "LoadImage", "inputs": {"image": pose_image}}
    graph["22"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["6", 0], "negative": ["7", 0], "control_net": ["20", 0], "image": ["21", 0],
        "strength": strength, "start_percent": 0.0, "end_percent": 1.0}}
    graph["3"]["inputs"]["positive"] = ["22", 0]
    graph["3"]["inputs"]["negative"] = ["22", 1]
    if reference_image:
        graph["30"] = {"class_type": "IPAdapterUnifiedLoader",
                       "inputs": {"model": ["4", 0], "preset": ipadapter_preset}}
        graph["31"] = {"class_type": "LoadImage", "inputs": {"image": reference_image}}
        graph["32"] = {"class_type": "IPAdapter", "inputs": {
            "model": ["30", 0], "ipadapter": ["30", 1], "image": ["31", 0],
            "weight": ipadapter_weight, "start_at": 0.0, "end_at": 1.0, "weight_type": "standard"}}
        graph["3"]["inputs"]["model"] = ["32", 0]
    return graph


def fill_defaults(graph: dict, object_info: dict) -> list[str]:
    """Bổ sung input bắt buộc còn thiếu (lấy giá trị mặc định từ /object_info) và sửa giá
    trị enum đã đổi tên. Giúp graph chạy được qua nhiều phiên bản ComfyUI/custom node.
    Trả về danh sách thay đổi để ghi log."""
    changes = []
    for node_id, node in graph.items():
        spec = (object_info.get(node["class_type"]) or {}).get("input", {}).get("required", {})
        for name, desc in spec.items():
            if not isinstance(desc, (list, tuple)) or not desc:
                continue
            kind = desc[0]
            opts = desc[1] if len(desc) > 1 and isinstance(desc[1], dict) else {}
            choices = kind if isinstance(kind, list) else (opts.get("options") if kind == "COMBO" else None)
            value = node["inputs"].get(name)
            if value is None:
                if "default" in opts:
                    node["inputs"][name] = opts["default"]
                elif choices:
                    node["inputs"][name] = choices[0]
                else:
                    continue
                changes.append(f"{node['class_type']}.{name} = {node['inputs'][name]!r} (mặc định)")
            elif (choices and name in FLEXIBLE_CHOICES and not isinstance(value, list)
                  and value not in choices):
                node["inputs"][name] = choices[0]
                changes.append(f"{node['class_type']}.{name}: {value!r} không còn hợp lệ -> {choices[0]!r}")
    return changes


def wan22_i2v(image_name: str, prompt: str, negative: str, seed: int,
              length: int, models: dict, width: int = 448, height: int = 448,
              steps: int = 20, cfg: float = 5.0, shift: float = 8.0) -> dict:
    # shift=8 theo workflow chính thức của ComfyUI cho TI2V 5B; Sprite Sheep ghi
    # nhận thiếu shift thì nhân vật phình dần qua từng frame và lỗi màu ở góc.
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": models["text_encoder"], "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {
            "vae": ["3", 0], "start_image": ["4", 0], "batch_size": 1,
            "width": width, "height": height, "length": length}},
        "12": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["12", 0], "positive": ["5", 0], "negative": ["6", 0],
            "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        WAN_OUTPUT: {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }


def wan_length(duration_s: float) -> int:
    """Độ dài hợp lệ đầu tiên (4n+1) phủ được thời lượng yêu cầu."""
    wanted = max(1.0, float(duration_s)) * WAN_FPS
    n = max(1, -(-int(wanted - 1) // 4))
    return min(4 * n + 1, WAN_MAX_LENGTH)


def frame_indices(length: int, n_frames: int, useful_fraction: float = 0.85) -> list[int]:
    """N chỉ số cách đều trên phần đầu của clip.

    Từ `parametri.indici_frame` của Sprite Sheep: các frame cuối clip model hay
    dùng để "đứng lại", nên chỉ lấy mẫu phần đầu (mặc định 85%).
    """
    n_frames = max(1, int(n_frames))
    last = max(0, int(length * useful_fraction) - 1)
    if n_frames == 1:
        return [0]
    raw = [round(i * last / (n_frames - 1)) for i in range(n_frames)]
    out: list[int] = []
    for v in raw:
        if out and v <= out[-1]:
            v = out[-1] + 1
        out.append(min(v, length - 1))
    return out
