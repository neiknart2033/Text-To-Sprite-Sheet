# SpriteForge

Mô tả nhân vật (hoặc ảnh sprite có sẵn) + danh sách hành động → sprite sheet
pixel art cho game: `spritesheet.png` + `manifest.json` + `preview.gif` cho mỗi hành động.
Chạy headless trên **Kaggle GPU miễn phí** qua ComfyUI.

Ghép từ hai dự án mã nguồn mở:
- [Pixel Art Studio](https://github.com/velteyn/pixel-art-studio) (GPLv3): sinh nhân
  vật SD1.5, hậu xử lý pixel, đóng gói sheet.
- [Sprite Sheep](https://github.com/MANFRI-DEV/sprite-sheep) (MIT): chuyển động bằng
  video WAN 2.2 TI2V 5B (engine `video`), tách nền chroma key.

Engine mặc định `edit`: **FLUX.2 [klein] 4B** (model sửa ảnh theo ảnh mẫu, Apache-2.0).
Sprite mẫu được uốn thành "con rối" theo thư viện khung xương tư thế (`forge/poses.py`,
`forge/puppet.py`), rồi klein vẽ lại từng frame cho sạch mà vẫn giữ đúng nhân vật mẫu.
Engine cũ: `pose` (SD1.5 ControlNet OpenPose + IP-Adapter), `video` (WAN 2.2).

| Đường dẫn | Nội dung |
|---|---|
| `kaggle/README.md` | **Hướng dẫn sử dụng** (bắt đầu từ đây) |
| `kaggle/SpriteForge_Kaggle.ipynb` | Notebook chính import vào Kaggle (hỗ trợ GPU T4 x2) |
| `kaggle/SpriteForge_Models.ipynb` | Notebook phụ: tải model (FLUX.2 klein; engine cũ tuỳ chọn) một lần, lưu thành Output để gắn làm Input |
| `forge/poses.py` | Thư viện khung xương tư thế (15 action) |
| `forge/puppet.py` | Con rối: cắt sprite mẫu thành các phần, xoay theo khung xương; kiểm tra frame có đúng tư thế |
| `kaggle/run_batch.py` | Chạy batch: `python kaggle/run_batch.py --jobs jobs.json --out output` |
| `kaggle/jobs.example.json` | Mẫu: tạo nhân vật từ mô tả (Hắc Ronin), 8 hành động |
| `kaggle/jobs.image.example.json` | Mẫu: dùng sprite có sẵn (`hero.png`), 7 hành động |
| `forge/` | Thư viện: `comfy` (client), `workflows` (graph), `pipelines`, `pixel`, `matting`, `sheet`, `config` |
| `tests/` | Test không cần GPU (ComfyUI giả): `python -m unittest tests.test_forge -v` |
| `KAGGLE_DESIGN.md` | Thiết kế kỹ thuật |

Giấy phép: GPL-3.0 (xem `LICENSE`, `THIRD_PARTY_NOTICES.md`).
