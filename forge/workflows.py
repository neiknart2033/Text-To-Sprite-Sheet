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

# Node đầu ra của từng graph (để lấy ảnh từ history)
SD15_OUTPUT = "9"
WAN_OUTPUT = "10"

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
