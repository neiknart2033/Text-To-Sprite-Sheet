# Mã nguồn bên thứ ba

SpriteForge ghép và chỉnh sửa code từ hai dự án dưới đây. Vì có code GPLv3,
toàn bộ SpriteForge phát hành theo **GNU GPL v3** (xem `LICENSE`). Sprite sheet
do tool tạo ra là của bạn; giấy phép code không áp lên ảnh đầu ra.

## Pixel Art Studio: GPL-3.0

- Nguồn: https://github.com/velteyn/pixel-art-studio (commit `979168c`, 28/07/2026)
- Tác giả: velteyn
- Phần dùng lại (đã chỉnh sửa):
  - `forge/sheet.py` ← `pipeline/spritesheet.py` (đóng gói sheet, manifest, GIF)
  - `forge/pixel.py` ← ý tưởng từ `pipeline/mixel_fix.py` (lượng tử palette),
    `pipeline/pixel_snap.py` (lấy đa số trong khối), `pipeline/frame_align.py` (neo chân)
  - `forge/pipelines.py` ← quy trình "anchor" của `pipeline/character_gen.py`
- Không dùng: GUI PyQt6, client A1111/ComfyUI gốc (`api/`). Client ComfyUI gốc bỏ
  qua ControlNet/LoRA và đổi seed -1 thành 0, nên được viết lại trong `forge/comfy.py`.

## Sprite Sheep: MIT

- Nguồn: https://github.com/MANFRI-DEV/sprite-sheep (commit `8bdc585`, 27/09/2026)
- Phần dùng lại (đã chỉnh sửa, dịch tên biến sang tiếng Anh):
  - `forge/matting.py` ← `sidecar/scontorno.py` (nhánh tách nền một màu)
  - `forge/workflows.py` ← `sidecar/backend/comfyui_wan.py` (graph WAN 2.2 TI2V 5B),
    `sidecar/parametri.py` (độ dài clip 4n+1, chọn frame cách đều)
  - `forge/comfy.py` ← cách chờ kết quả chịu được timeout của `comfyui_bridge._attendi`

```
MIT License

Copyright (c) 2026 Silvio Manfrinato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model (không kèm trong repo, tải riêng)

| Model | Giấy phép | Ghi chú |
|---|---|---|
| WAN 2.2 TI2V 5B (`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`) | Apache-2.0 | Không hạn chế |
| Checkpoint SD1.5 pixel art (Civitai) | Theo từng model | Đọc giấy phép trên trang model, nhất là khi dùng thương mại |
| ~~MiniMax H3~~ | MiniMax H3 Community License | **Không dùng**: cấm EU/UK/Hàn Quốc/Mỹ (cả ảnh đầu ra), quá nặng cho Kaggle |

ComfyUI (GPL-3.0) được cài riêng lúc chạy và giao tiếp qua HTTP.
