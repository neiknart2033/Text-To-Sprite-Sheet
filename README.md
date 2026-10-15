# SpriteForge

Mô tả nhân vật (hoặc ảnh sprite có sẵn) + danh sách hành động → sprite sheet
pixel art cho game: `spritesheet.png` + `manifest.json` + `preview.gif` cho mỗi hành động.
Chạy headless trên **Kaggle GPU miễn phí** qua ComfyUI.

Ghép từ hai dự án mã nguồn mở:
- [Pixel Art Studio](https://github.com/velteyn/pixel-art-studio) (GPLv3): sinh nhân
  vật SD1.5, hậu xử lý pixel, đóng gói sheet.
- [Sprite Sheep](https://github.com/MANFRI-DEV/sprite-sheep) (MIT): chuyển động bằng
  video WAN 2.2 TI2V 5B, tách nền chroma key.

| Đường dẫn | Nội dung |
|---|---|
| `kaggle/README.md` | **Hướng dẫn sử dụng** (bắt đầu từ đây) |
| `kaggle/SpriteForge_Kaggle.ipynb` | Notebook chính import vào Kaggle (hỗ trợ GPU T4 x2) |
| `kaggle/SpriteForge_Models.ipynb` | Notebook phụ: tải model WAN 17GB một lần, lưu thành Output để gắn làm Input |
| `kaggle/run_batch.py` | Chạy batch: `python kaggle/run_batch.py --jobs jobs.json --out output` |
| `kaggle/jobs.example.json` | Mẫu: nhân vật Hắc Ronin, 8 hành động |
| `forge/` | Thư viện: `comfy` (client), `workflows` (graph), `pipelines`, `pixel`, `matting`, `sheet`, `config` |
| `tests/` | Test không cần GPU (ComfyUI giả): `python -m unittest tests.test_forge -v` |
| `KAGGLE_DESIGN.md` | Thiết kế kỹ thuật |

Giấy phép: GPL-3.0 (xem `LICENSE`, `THIRD_PARTY_NOTICES.md`).
