# SpriteForge trên Kaggle: hướng dẫn sử dụng

Gõ mô tả nhân vật và danh sách hành động. Kaggle GPU (miễn phí) sẽ tạo cho mỗi
hành động: `spritesheet.png` + `manifest.json` + `preview.gif`.

Cách hoạt động (chi tiết: `KAGGLE_DESIGN.md`):

1. **Nhân vật:** checkpoint SD1.5 pixel art vẽ nhân vật trên nền xanh. Sau đó
   tách nền, thu về kích thước pixel gốc (mặc định cao 64px) và rút palette màu.
2. **Chuyển động:** WAN 2.2 TI2V 5B nhận ảnh nhân vật + câu mô tả hành động,
   rồi tạo một clip video ngắn. Tool lấy N frame cách đều từ clip.
3. **Hậu xử lý:** tách nền, đưa về pixel gốc, ép đúng palette của nhân vật,
   đặt vào các ô vuông cùng kích thước, rồi xuất sheet, manifest và GIF.

---

## 1. Chuẩn bị (làm một lần)

### 1.1 Verify số điện thoại
Vào kaggle.com → avatar → **Settings** → *Phone verification*. Chưa verify thì
không bật được GPU và Internet.

### 1.2 Dataset model `spriteforge-models`
1. Tải về máy một checkpoint **SD1.5** pixel art dạng `.safetensors` (2–4GB),
   ví dụ `aziibpixelmix`, `PixelMonster` hoặc `Pixel Art Sprite Diffusion` trên Civitai.
2. kaggle.com → **Datasets** → **New Dataset** → upload file → đặt tên
   `spriteforge-models` → chọn **Private** → Create.
3. *(Tùy chọn, giúp khởi động nhanh hơn)* thêm 3 file WAN vào cùng dataset.
   Nếu không thêm, notebook sẽ tự tải từ HuggingFace mỗi session (~17GB, vài phút):
   - `wan2.2_ti2v_5B_fp16.safetensors` (9.3GB)
   - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (6.3GB)
   - `wan2.2_vae.safetensors` (1.3GB)

   Cả ba lấy từ `huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, thư mục `split_files/`.

Notebook tự tìm mọi file `.safetensors` trong các dataset đã gắn, nên tên thư
mục bên trong dataset không quan trọng. Chỉ để checkpoint trong dataset này,
**không** để LoRA, vì mọi file `.safetensors` khác 3 file WAN đều bị coi là checkpoint.

### 1.3 Đưa code SpriteForge lên Kaggle (chọn 1 cách)
- **Cách A (không cần GitHub):** nén cả thư mục dự án (có `forge/` và `kaggle/`)
  thành zip → New Dataset → tên `spriteforge-code`, Private.
- **Cách B:** đẩy dự án lên GitHub, rồi điền `SPRITEFORGE_REPO` ở cell 1 của notebook.
  Repo phải public, hoặc bạn tự thêm token.

### 1.4 Tạo notebook
1. kaggle.com → **Code** → **New Notebook** → *File → Import Notebook* → chọn
   `kaggle/SpriteForge_Kaggle.ipynb`.
2. Panel phải → *Session options*: **Accelerator: GPU T4** (T4 x2 cũng được, ComfyUI chỉ dùng
   1 card; tránh P100 vì chậm với fp16), **Internet: On**.
3. **Add Input** → chọn `spriteforge-models` (và `spriteforge-code` nếu dùng cách A).

---

## 2. Chạy

**Lần đầu**, đặt `RUN_BENCHMARK = True` ở cell 1 và chạy tương tác (Run All).
Cell 4 sẽ chạy thử 1 action và in thời gian, để bạn biết batch thật sẽ mất bao lâu.

**Chạy batch:**
1. Sửa nội dung `jobs.json` ở **cell 5** (xem mục 3).
2. Chọn một trong hai cách:
   - **Tương tác:** *Run All*, giữ tab mở. Khi xong, tải `output.zip` ở panel **Output** bên phải.
   - **Chạy nền (không cần canh):** *Save Version* → **Save & Run All (Commit)**.
     Kaggle chạy tối đa 12 giờ, kể cả khi bạn tắt máy. Khi xong, mở bản version
     đó → tab **Output** → tải `output.zip`.
3. Có action lỗi? Chạy lại cell 6 (trong cùng session). Action đã xong sẽ được bỏ qua.

> Thư mục `/kaggle/working` bị xoá khi session kết thúc. Chỉ những gì nằm trong
> Output của bản Commit mới được giữ lại. Nhớ tải `output.zip` trước khi tắt session tương tác.

---

## 3. `jobs.json`

```json
{
  "checkpoint": "auto",
  "seed": -1,
  "frame_size": 128,
  "char_height": 64,
  "palette": 32,
  "fps": 10,
  "view": "side",
  "jobs": [
    {
      "name": "hac_ronin",
      "mode": "prompt",
      "prompt": "dark samurai in black-indigo armor, oni mask with glowing red eyes",
      "actions": [
        {"name": "idle",   "prompt": "standing still in a ready stance, calm breathing", "frames": 8},
        {"name": "walk",   "prompt": "walking in place, steady walk cycle", "frames": 8, "align": "track"},
        {"name": "attack", "prompt": "slashes the katana horizontally", "frames": 8, "columns": 4},
        {"name": "hurt",   "prompt": "recoils backward from a hit", "frames": 6, "loop": false}
      ]
    },
    {
      "name": "chibi",
      "mode": "image",
      "character_image": "/kaggle/input/spriteforge-chars/chibi.png",
      "actions": [{"name": "jump", "prompt": "jumps straight up and lands", "frames": 6}]
    }
  ]
}
```

**Trường chung:**

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `checkpoint` | (bắt buộc nếu có job `prompt`) | Tên file checkpoint SD1.5, hoặc `"auto"` để lấy file đầu tiên tìm thấy |
| `seed` | `-1` | `-1` = ngẫu nhiên. Seed thực tế được ghi trong `manifest.json` để tái lập |
| `steps`, `cfg_scale` | `30`, `7` | Tham số SD1.5 khi tạo nhân vật |
| `frame_size` | `128` | Cạnh mỗi ô trong sheet (px). Mọi frame đúng bằng cỡ này |
| `char_height` | `frame_size/2` | Chiều cao nhân vật tính bằng pixel gốc |
| `palette` | `32` | Số màu tối đa. Mọi frame dùng chung palette của nhân vật |
| `fps` | `10` | FPS ghi vào manifest và GIF |
| `view` | `"side"` | `"side"` (nhìn ngang, quay phải) hoặc `"front"` |
| `bg_color` | `"auto"` | Màu nền dùng để tách. `"auto"` chọn màu khác xa màu nhân vật nhất. Có thể đặt `green`/`magenta`/`blue` |
| `preview_scale` | `3` | Hệ số phóng của GIF xem trước |
| `wan` | xem dưới | `steps` (20), `cfg` (5), `duration` (2.0 giây), `width`/`height` (448), `useful_fraction` (0.85), `char_fill` (0.6), `timeout_s` (3600) |

**Mỗi job:** `name` (chỉ dùng chữ, số, `_`, `-`); `mode` là `"prompt"` (cần
`prompt`) hoặc `"image"` (cần `character_image`); `seed` (tùy chọn); `actions`.
Với `mode: "image"`, ảnh nên có nền trong suốt hoặc nền một màu. Upload ảnh
thành dataset (ví dụ `spriteforge-chars`) rồi trỏ đường dẫn tuyệt đối vào `/kaggle/input/...`.

**Mỗi action:**

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `name`, `prompt` | bắt buộc | Tên thư mục; câu mô tả động tác **bằng tiếng Anh** |
| `frames` | `8` | Số frame trong sheet (1–64) |
| `columns` | `= frames` | Số cột (mặc định 1 hàng) |
| `duration` | `wan.duration` | Độ dài clip (giây). Động tác dài như `death` thì đặt 3 |
| `align` | `"fixed"` | `fixed` giữ nguyên vị trí thật (nhảy, chém). `track` bỏ phần trôi ngang (đi, chạy) |
| `loop` | `true` | Ghi vào manifest |
| `seed` | theo job | Seed riêng cho action |

---

## 4. Kết quả

```
output/
├── run_log.txt                  log đầy đủ của lần chạy
└── hac_ronin/
    ├── character.png            nhân vật ở pixel gốc (dùng lại cho mọi action)
    ├── character_preview.png    bản phóng to để xem
    ├── character_raw.png        ảnh gốc SD1.5 (mode prompt)
    ├── character.json           seed, prompt, kích thước, số màu
    ├── progress.json            trạng thái từng action (done / error + lý do)
    └── idle/
        ├── spritesheet.png      frame_size x frame_size mỗi ô, nền trong suốt
        ├── manifest.json        kích thước, số cột/hàng, fps, anchor, toạ độ từng frame, seed...
        └── preview.gif
```

`manifest.json` có `anchor` (điểm chân nhân vật trong ô) để engine game căn sprite.

---

## 5. Mẹo chất lượng

- **Không ưng nhân vật?** Đặt `"seed"` khác cho job, xoá thư mục `output/<job>/`
  rồi chạy lại. Hoặc chọn một `character.png` ưng ý từ lần trước và dùng `mode: "image"`.
- **Đi/chạy bị trôi khỏi khung:** viết "in place" trong prompt và đặt `"align": "track"`.
- **Nhân vật có màu trùng nền** (VD áo xanh lá): giữ `bg_color: "auto"` hoặc đặt `"magenta"`.
- **Nhân vật bị cắt** (log báo "pixel bị cắt"): tăng `frame_size` hoặc giảm `char_height`.
- **Động tác lặp lại / đứng im ở cuối:** giảm `wan.useful_fraction` (VD 0.7) hoặc giảm `duration`.

---

## 6. Xử lý sự cố

| Thông báo | Cách xử lý |
|---|---|
| `Thiếu: node ComfyUI 'Wan22ImageToVideoLatent'` | ComfyUI quá cũ: để `COMFY_REF = ""` để dùng bản mới nhất |
| `Thiếu: model '...' trong checkpoints/` | Tên trong `jobs.json` sai: xem danh sách "ComfyUI đang thấy", hoặc dùng `"checkpoint": "auto"` |
| `Frame ... gần như một màu (đen/trống)` | WAN chạy fp16 trên T4 có thể sinh NaN. Thử thêm `"--fp32-vae"` vào lệnh khởi động ComfyUI ở cell 3, hoặc đổi seed |
| `out of memory` | Giảm `wan.width`/`wan.height` (VD 384, phải là bội của 32) hoặc `duration` |
| Quá chậm | Giảm `wan.steps` (VD 12–15, chất lượng giảm nhẹ) hoặc `duration` |
| `Mọi frame đều trống sau khi tách nền` | Đặt `bg_color` thành màu khác |
| `nền ảnh SD không đồng nhất` | Xem `character_raw.png`: đổi seed, hoặc dùng `mode: "image"` |
| `ComfyUI không khởi động được` | Đọc log in ra ở cell 3 (thường do pip cài lỗi) |
| Hết 30 giờ GPU/tuần | Chờ reset hằng tuần, hoặc thuê Vast.ai/RunPod (notebook chạy được trên mọi máy Linux có GPU) |

---

## 7. Thời gian

| Bước | Ước tính |
|---|---|
| Cài đặt (clone + pip) | ~5–8 phút |
| Tải WAN từ HuggingFace (nếu không có trong dataset) | vài phút (~17GB) |
| Khởi động ComfyUI | ~1–2 phút |
| 1 nhân vật SD1.5 (512px, 30 steps) | ~10–30 giây |
| **1 action WAN (448², 49 frame, 20 steps)** | **chưa đo trên T4. Chạy benchmark (cell 4) trước** |

Hai dự án gốc chưa có số liệu WAN 2.2 trên T4. Lấy thời gian benchmark nhân với
số action để biết batch có vừa một session 12 giờ không.

---

## 8. Test ở máy local (không cần GPU)

```bash
python -m unittest tests.test_forge -v
```

Bộ test dựng một ComfyUI giả, kiểm tra cả luồng: tạo nhân vật, tạo animation,
sinh sheet/manifest/GIF, chạy tiếp từ chỗ dừng, và việc một action lỗi không làm dừng cả batch.
