# SpriteForge trên Kaggle: hướng dẫn sử dụng

Gõ mô tả nhân vật và danh sách hành động. Kaggle GPU (miễn phí) sẽ tạo cho mỗi
hành động: `spritesheet.png` + `manifest.json` + `preview.gif`.

Cách hoạt động (chi tiết: `KAGGLE_DESIGN.md`):

1. **Nhân vật:** checkpoint SD1.5 pixel art vẽ nhân vật đứng nhìn ngang trên nền trắng (theo
   khung xương đứng). Hoặc dùng sprite có sẵn của bạn. Sau đó tách nền, thu về pixel gốc,
   rút palette màu.
2. **Chuyển động (engine `pose`, mặc định):** mỗi action có sẵn một chuỗi **khung xương tư
   thế** (đi, chạy, chém, nhảy...). Mỗi frame được SD1.5 vẽ lại theo đúng khung xương đó
   (ControlNet OpenPose), dùng ảnh nhân vật làm mẫu để giữ ngoại hình (IP-Adapter), cùng
   một seed cho cả action. Vì khung xương cố định tỉ lệ và đường nền nên nhân vật không
   bị zoom, trôi hay biến mất, và số frame luôn đúng.
3. **Hậu xử lý:** tách nền, đưa về pixel gốc, ép đúng palette của nhân vật,
   đặt vào các ô vuông cùng kích thước, rồi xuất sheet, manifest và GIF.

Engine cũ `"engine": "video"` (WAN 2.2 tạo clip rồi cắt frame) vẫn dùng được, nhưng chạy
thật cho thấy WAN hay làm nhân vật pixel nhỏ bị zoom, biến hình hoặc tan biến.

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
### 1.2b Notebook `SpriteForge Models` (làm 1 lần, khỏi tải model mỗi session)
Mặc định notebook này lưu model của engine tư thế (~3.3GB): ControlNet OpenPose, IP-Adapter Plus,
CLIP-ViT-H. Đặt `INCLUDE_WAN = True` nếu cần cả WAN (~17GB) cho engine video. Bỏ qua bước
này thì cell 2 của notebook chính tự tải các model engine tư thế mỗi session (vài phút).
Không cần upload từ máy bạn: Kaggle tự tải từ HuggingFace về rồi lưu lại.
1. **Code** → **New Notebook** → *File → Import Notebook* → chọn `kaggle/SpriteForge_Models.ipynb`.
2. *Session options*: **Accelerator = None** (chỉ dùng CPU, không tốn quota GPU), **Internet = On**.
3. *Save Version* → **Save & Run All (Commit)**. Chờ khoảng 5–15 phút đến khi version báo chạy xong.
   Output của nó sẽ có 3 file: `wan2.2_ti2v_5B_fp16.safetensors` (9.3GB),
   `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (6.3GB), `wan2.2_vae.safetensors` (1.3GB).

Bỏ qua bước này thì notebook chính vẫn chạy được, nhưng mỗi session sẽ tự tải lại 17GB.

Notebook tự tìm mọi file `.safetensors` trong các dataset đã gắn, nên tên thư
mục bên trong dataset không quan trọng. Chỉ để checkpoint trong dataset này,
**không** để LoRA, vì mọi file `.safetensors` khác 3 file WAN đều bị coi là checkpoint.

### 1.3 Đưa code SpriteForge lên Kaggle (chọn 1 cách)
- **Cách A (không cần GitHub):** nén cả thư mục dự án (có `forge/` và `kaggle/`)
  thành zip → New Dataset → tên `spriteforge-code`, Private.
- **Cách B (GitHub):** đẩy dự án lên GitHub, rồi ở cell 1 của notebook điền
  `SPRITEFORGE_REPO` và `SPRITEFORGE_BRANCH` (nhánh chứa code).
  - Repo **public**: không cần làm gì thêm.
  - Repo **private**: tạo token trên GitHub → *Settings → Developer settings →
    Fine-grained tokens* → chỉ chọn repo này, quyền **Contents: Read-only**. Trong
    notebook Kaggle: *Add-ons → Secrets* → *Add Secret*, label `GITHUB_TOKEN`, dán
    token, rồi bật (tick) secret đó cho notebook. Token không bị in ra output.
  - Sửa code rồi push: chạy lại cell 2 là notebook tự `git pull` bản mới.

### 1.4 Tạo notebook
1. kaggle.com → **Code** → **New Notebook** → *File → Import Notebook* → chọn
   `kaggle/SpriteForge_Kaggle.ipynb`.
2. Panel phải → *Session options*: **Accelerator: GPU T4 x2**, **Internet: On**.
   Có 2 GPU thì notebook chạy 2 ComfyUI song song, mỗi GPU một cái, nên nhanh gần gấp đôi.
   Tránh P100 vì chậm với fp16.
3. **Add Input** → chọn `spriteforge-models`, notebook **SpriteForge Models** (tab *Notebooks* /
   *Your Work*), và `spriteforge-code` nếu dùng cách A.

---

## 2. Chạy

**Lần đầu**, đặt `RUN_BENCHMARK = True` ở cell 1 và chạy tương tác (Run All).
Cell 4 sẽ chạy thử 1 action và in thời gian, để bạn biết batch thật sẽ mất bao lâu.

**Chạy batch:**
1. Sửa nội dung `jobs.json` ở **cell 5** (xem mục 3).
2. Chọn một trong hai cách:
   - **Tương tác:** *Run All*, giữ tab mở. Khi xong, bấm link `output.zip` mà cell 7 in ra.
     Kaggle không cho tải cả thư mục, chỉ tải từng file, nên kết quả được gom vào 1 file zip.
     Cách khác: panel phải → **Output** → bấm ⋮ cạnh `output.zip` → *Download* (bấm biểu tượng làm mới nếu chưa thấy file).
   - **Chạy nền (không cần canh):** *Save Version* → **Save & Run All (Commit)**.
     Kaggle chạy tối đa 12 giờ, kể cả khi bạn tắt máy. Khi xong, mở bản version
     đó → tab **Output** → tải `output.zip`.
3. Có action lỗi, hoặc vừa sửa `jobs.json`? Chạy lại cell 6 trong cùng session. Tool chỉ làm lại
   những gì cần: action lỗi, action có prompt/tham số vừa sửa, và mọi thứ bị ảnh hưởng khi
   nhân vật đổi hoặc code SpriteForge được nâng cấp. Phần còn lại được bỏ qua.

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
      "character_image": "chibi.png",
      "prompt": "chibi samurai with a red headband",
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
| `bg_color` | `"auto"` | Màu nền của canvas đưa vào WAN (nhân vật SD luôn vẽ nền trắng). `"auto"` chọn màu khác xa màu nhân vật nhất. Có thể đặt `green`/`magenta`/`blue` |
| `preview_scale` | `3` | Hệ số phóng của GIF xem trước |
| `engine` | `"pose"` | `"pose"` (khung xương + SD1.5) hoặc `"video"` (WAN 2.2). Có thể đặt riêng cho từng action |
| `pose` | xem dưới | `strength` (1.0: độ bám khung xương), `ipadapter` (true), `ipadapter_weight` (0.8: độ giống nhân vật mẫu; cao quá thì tư thế kém đúng), `size` (512) |
| `debug` | `true` | Lưu ảnh soi lỗi cho mỗi action trong `debug/`: engine pose có `poses.png` (khung xương) + `raw.png` (ảnh SD gốc) + `clean.png` (sau khi làm sạch); engine video có `input.png` + `clip.gif` |
| `wan` | xem dưới | `steps` (20), `cfg` (5), `duration` (2.0 giây), `width`/`height` (640), `useful_fraction` (0.85), `char_fill` (0.6), `timeout_s` (3600) |

**Mỗi job:** `name` (chỉ dùng chữ, số, `_`, `-`); `mode` là `"prompt"` (cần
`prompt`) hoặc `"image"` (cần `character_image`); `seed` (tùy chọn); `actions`.

### Dùng sprite mẫu có sẵn (`mode: "image"`)
Không cần checkpoint SD: tool lấy thẳng nhân vật trong ảnh của bạn rồi tạo các action.
Mẫu đầy đủ: `kaggle/jobs.image.example.json`.
1. **Chuẩn bị ảnh**: PNG **nền trong suốt** (tốt nhất) hoặc nền **một màu phẳng**. Một nhân
   vật, đứng, nhìn ngang quay phải (khớp `"view": "side"`). Ảnh pixel gốc hoặc đã phóng
   to đều được: ảnh phóng x2, x4... tự được thu về pixel gốc.
2. **Upload lên Kaggle**: trong notebook → **Add Input** → **Upload** (hoặc *New Dataset*) →
   kéo thả file PNG (nhiều nhân vật thì kéo nhiều file) → đặt tên dataset, ví dụ `my-sprites` → Create.
3. **Ghi vào `jobs.json`** (cell 5): `"character_image": "hero.png"`. **Chỉ cần tên file**: tool tự
   tìm trong mọi Input đã gắn. Thêm `"prompt"` mô tả nhân vật (tiếng Anh) để WAN hiểu
   đúng nhân vật hơn (không bắt buộc).
4. Chạy cell 6 như bình thường.

Tool giữ nguyên sprite của bạn: không ép về `char_height`, giữ mọi màu gốc (tối đa 255).
Chỉ thu nhỏ khi nhân vật cao hơn 80% `frame_size`; khi đó hãy tăng `frame_size`
(VD sprite cao 100px → `"frame_size": 160`). Muốn ép cỡ hoặc số màu thì khai báo rõ
`char_height`, `palette`. Upload bản sửa của ảnh (cùng tên) rồi chạy lại → tool tự nhận
ra và làm lại.

**Mỗi action:**

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `name`, `prompt` | bắt buộc | Tên thư mục; câu mô tả động tác **bằng tiếng Anh** |
| `pose` | `= name` | Tư thế mẫu (engine pose). Có sẵn: `idle`, `walk`, `run`, `attack`, `heavy`, `hurt`, `jump`, `death`, `block`, `dash`, `crouch`, `fall`, `cast`, `kick`, `punch`. Action tên khác thì ghi rõ, VD `{"name": "slash2", "pose": "attack", ...}` |
| `engine` | theo `engine` chung | `"pose"` hoặc `"video"` cho riêng action này |
| `frames` | `8` | Số frame trong sheet (1–64) |
| `columns` | `= frames` | Số cột (mặc định 1 hàng) |
| `duration` | `wan.duration` | (engine video) Độ dài clip (giây) |
| `align` | `"fixed"` | `fixed` giữ nguyên vị trí thật (nhảy, chém). `track` bỏ phần trôi ngang (engine video) |
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
        ├── preview.gif
        └── debug/               (khi "debug": true) input.png + clip.gif gốc của WAN
```

`manifest.json` có `anchor` (điểm chân nhân vật trong ô) để engine game căn sprite.

---

## 5. Mẹo chất lượng

- **Không ưng nhân vật?** Đặt `"seed"` khác cho job rồi chạy lại cell 6. Nhân vật và mọi
  action của job đó tự làm lại. Hoặc chọn một `character.png` ưng ý từ lần trước và dùng `mode: "image"`.
- **Đi/chạy bị trôi khỏi khung:** viết "in place" trong prompt và đặt `"align": "track"`.
- **Nhân vật có màu trùng nền** (VD áo xanh lá): giữ `bg_color: "auto"` hoặc đặt `"magenta"`.
- **Nhân vật bị cắt** (log báo "pixel bị cắt"): tăng `frame_size` hoặc giảm `char_height`.
- **Động tác lặp lại / đứng im ở cuối:** giảm `wan.useful_fraction` (VD 0.7) hoặc giảm `duration`.

---

## 6. Xử lý sự cố

| Thông báo | Cách xử lý |
|---|---|
| `Tắt IP-Adapter (...)` | Thiếu custom node hoặc model IP-Adapter: engine pose vẫn chạy nhưng nhân vật kém giống nhau giữa các frame. Xem cell 2 có tải đủ `ip-adapter-plus_sd15` và `CLIP-ViT-H-14...` không |
| `chưa có tư thế mẫu "..."` | Tên action không trùng tư thế có sẵn: thêm `"pose": "<tên có sẵn>"` vào action đó |
| `SD vẽ thêm nhân vật/phông nền (đã tự xoá)` | Checkpoint hay vẽ nhiều nhân vật kiểu bảng thiết kế, đĩa tròn sau lưng, vệt đất. Tool tự xoá theo khung xương và thử seed khác để có nhân vật mẫu sạch. Xem `character_preview.png`; vẫn bẩn thì đổi `seed` của job |
| Frame còn sót mảng màu sau lưng | Mở `debug/clean.png` (frame sau khi làm sạch) so với `debug/raw.png`. Thường do nhân vật mẫu còn bẩn: đổi `seed` của job |
| Nhân vật sai tư thế / sai hướng (engine pose) | Mở `debug/poses.png` (khung xương) và `debug/raw.png` (ảnh SD gốc). Nhân vật không theo khung xương: tăng `pose.strength` (VD 1.2) hoặc giảm `pose.ipadapter_weight` (VD 0.6) |
| Nhân vật khác nhau giữa các frame (engine pose) | Tăng `pose.ipadapter_weight` (VD 0.9–1.0), viết `prompt` của job mô tả rõ màu sắc và trang phục |
| `Thiếu: node ComfyUI 'Wan22ImageToVideoLatent'` | ComfyUI quá cũ: để `COMFY_REF = ""` để dùng bản mới nhất |
| `Thiếu: model '...' trong checkpoints/` | Tên trong `jobs.json` sai: xem danh sách "ComfyUI đang thấy", hoặc dùng `"checkpoint": "auto"` |
| `Frame ... gần như một màu (đen/trống)` | WAN chạy fp16 trên T4 có thể sinh NaN. Thử thêm `"--fp32-vae"` vào lệnh khởi động ComfyUI ở cell 3, hoặc đổi seed |
| `out of memory` | Giảm `wan.width`/`wan.height` (VD 384, phải là bội của 32) hoặc `duration` |
| Quá chậm | Giảm `wan.steps` (VD 12–15, chất lượng giảm nhẹ) hoặc `duration` |
| `Mọi frame đều trống sau khi tách nền` | Đặt `bg_color` thành màu khác |
| `Nhân vật lần N: nền là [...], không phải green` | SD vẽ sai màu nền, tool tự gen lại (tối đa 3 lần). Hết 3 lần vẫn tách nền được bằng cách loang từ mép ảnh. Xem `character_preview.png` |
| `Nhân vật bị thủng ...% thân` | Tách nền ăn mất chi tiết: đổi seed của job, hoặc dùng `mode: "image"` với ảnh nền trong suốt |
| Animation méo, vỡ mảnh | Mở `debug/clip.gif`: nếu clip WAN đã méo thì sửa prompt action/đổi seed; nếu clip đẹp mà sheet vỡ thì là lỗi hậu xử lý, gửi log + clip cho người bảo trì |
| `ComfyUI không khởi động được` | Đọc log in ra ở cell 3 (thường do pip cài lỗi) |
| Session chết / "Kernel died" / lỗi kết nối giữa chừng khi dùng T4 x2 | Thiếu RAM khi 2 ComfyUI cùng giữ model: ở cell 1 đặt `NUM_GPUS = 1`, chạy lại từ cell 1. Action đã xong không bị chạy lại |
| Hết 30 giờ GPU/tuần | Chờ reset hằng tuần, hoặc thuê Vast.ai/RunPod (notebook chạy được trên mọi máy Linux có GPU) |

---

## 7. Thời gian

| Bước | Ước tính |
|---|---|
| Cài đặt (clone + pip) | ~5–8 phút |
| Tải WAN từ HuggingFace (nếu không có trong dataset) | vài phút (~17GB) |
| Khởi động ComfyUI | ~1–2 phút |
| 1 nhân vật SD1.5 (512px, 30 steps) | ~10–30 giây |
| **1 frame engine pose (512², 30 steps, ControlNet + IP-Adapter)** | **~6–10 giây** (ước tính, chưa đo) |
| **1 action 8 frame, engine pose** | **~1–1.5 phút** mỗi GPU (ước tính) |
| 1 action WAN ở 448² (clip 2s = 49 frame, 20 steps) | ~1 phút 24 giây mỗi GPU (đo trên Kaggle T4, 10/2026) |
| 1 action WAN ở 640² | ~3 phút mỗi GPU (ước tính) |

Vì sao 640 mà không 448: WAN 2.2 5B nén 32×32 pixel thành 1 token; ở 448 nhân vật chỉ rộng
vài token nên clip thật bị biến hình, zoom, tan biến. Muốn nhanh hơn (chấp nhận rủi ro đó): `"wan": {"width": 448, "height": 448}`.

Thời gian tạo sprite do GPU quyết định. Đo trên log thật: thời gian tăng đúng theo
số frame video WAN (~1,8 giây/frame), phần việc của code SpriteForge chỉ mất vài giây.

Thời gian gần như tỉ lệ với `duration` × `wan.steps`. Còn dư nhiều quota, nên có
thể tăng `wan.steps` lên 30 để thử chất lượng tốt hơn (~2 phút/action).

---

## 8. Test ở máy local (không cần GPU)

```bash
python -m unittest tests.test_forge -v
```

Bộ test dựng một ComfyUI giả, kiểm tra cả luồng: tạo nhân vật, tạo animation,
sinh sheet/manifest/GIF, chạy tiếp từ chỗ dừng, và việc một action lỗi không làm dừng cả batch.
