# SpriteForge trên Kaggle: hướng dẫn sử dụng

Gõ mô tả nhân vật (hoặc đưa sprite có sẵn) và danh sách hành động. Kaggle GPU (miễn phí)
sẽ tạo cho mỗi hành động: `spritesheet.png` + `manifest.json` + `preview.gif`.

Cách hoạt động (chi tiết: `KAGGLE_DESIGN.md`), engine mặc định `"edit"` dùng
**FLUX.2 [klein] 4B** (Black Forest Labs, 01/2026, Apache-2.0), một model **sửa ảnh theo
ảnh mẫu**:

1. **Nhân vật:** klein vẽ nhân vật đứng nhìn ngang trên nền trắng từ mô tả của bạn. Hoặc
   dùng sprite có sẵn. Sau đó tách nền, thu về pixel gốc, rút palette màu.
2. **Chuyển động:** mỗi action có sẵn một chuỗi **khung xương tư thế** (đi, chạy, chém,
   nhảy...). Với mỗi frame:
   - **Con rối:** chính sprite mẫu được cắt thành đầu, thân, tay, chân rồi xoay theo khung
     xương của frame (`forge/puppet.py`). Ảnh này đúng tư thế, đúng màu, nhưng khớp còn thô.
   - **Vẽ lại:** klein nhận ảnh 1 = con rối, ảnh 2 = sprite mẫu, và vẽ lại con rối cho sạch
     mà vẫn giữ nguyên tư thế, ngoại hình theo sprite mẫu.
   - **Kiểm tra:** hình bóng ảnh vẽ lại phải khớp con rối. Nếu model "lười", chép lại tư
     thế đứng của ảnh mẫu, tool tự vẽ lại kiểu img2img từ chính con rối, khi đó tư thế
     chắc chắn đúng.
3. **Hậu xử lý:** tách nền, thu về pixel gốc **đúng lưới** của sprite mẫu, ép đúng palette
   của nhân vật, đặt vào các ô cùng kích thước, rồi xuất sheet, manifest và GIF.

Vì sao đổi sang klein: engine cũ (SD1.5 + ControlNet + IP-Adapter) vẽ lại nhân vật từ đầu
mỗi frame, IP-Adapter chỉ giữ được "na ná", nên đưa sprite mẫu vào vẫn ra nhân vật khác.
Model sửa ảnh nhìn trực tiếp vào sprite mẫu khi vẽ. Các model mạnh hơn (Qwen-Image-2.1,
FLUX.3 klein, Qwen-Image-Edit 20B) cần GPU có bf16 (từ RTX 30xx/A100); T4 của Kaggle không
có nên phải chạy fp32, chậm gấp nhiều lần, và một số còn cấm dùng thương mại. FLUX.2 klein 4B
chạy fp16 được trên T4, chỉ cần 4 bước, và được dùng thương mại.

Engine cũ vẫn dùng được: `"engine": "pose"` (SD1.5 + ControlNet OpenPose + IP-Adapter, cần
checkpoint SD1.5) và `"engine": "video"` (WAN 2.2, xem mục 9).

---

## 1. Chuẩn bị (làm một lần)

### 1.1 Verify số điện thoại
Vào kaggle.com → avatar → **Settings** → *Phone verification*. Chưa verify thì
không bật được GPU và Internet.

### 1.2 (Tuỳ chọn) Notebook `SpriteForge Models`: khỏi tải model mỗi session
Cell 2 của notebook chính tự tải 3 file FLUX.2 klein (~11.6GB) từ HuggingFace mỗi session
(vài phút). Muốn lưu sẵn trên Kaggle:
1. **Code** → **New Notebook** → *File → Import Notebook* → chọn `kaggle/SpriteForge_Models.ipynb`.
2. *Session options*: **Accelerator = None** (chỉ dùng CPU, không tốn quota GPU), **Internet = On**.
3. *Save Version* → **Save & Run All (Commit)**. Chờ khoảng 5–15 phút đến khi version báo chạy xong.
   Output sẽ có `flux-2-klein-4b-fp8.safetensors` (3.8GB), `qwen_3_4b.safetensors` (7.5GB),
   `flux2-vae.safetensors` (0.3GB).
4. Đã từng chạy notebook này (bản cũ chỉ có ControlNet/IP-Adapter hoặc WAN)? Import bản mới
   rồi Commit lại: nó tự xoá file không dùng để không vượt giới hạn Output 20GB.

`INCLUDE_POSE_SD = True` thêm model engine cũ "pose" (~3.3GB); `INCLUDE_WAN = True` thêm WAN
(~17GB, không bật cùng klein vì vượt 20GB).

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
3. **Add Input** (tuỳ chọn): notebook **SpriteForge Models** (tab *Notebooks* / *Your Work*),
   dataset chứa sprite mẫu của bạn, và `spriteforge-code` nếu dùng cách A.

Notebook cũ (đã import trước bản FLUX.2 klein) vẫn chạy được: cell 2 lấy code mới qua
`git pull`, nhưng **danh sách model cần tải nằm trong chính cell 2**, nên hãy import lại
notebook (hoặc chép cell 1–3 từ `kaggle/SpriteForge_Kaggle.ipynb`). Nội dung `jobs.json` cũ ở
cell 5 giữ nguyên được: `checkpoint`, `steps`, `cfg_scale`, `wan` bị bỏ qua với engine "edit".

---

## 2. Chạy

**Lần đầu**, đặt `RUN_BENCHMARK = True` ở cell 1 và chạy tương tác (Run All).
Cell 4 tạo thử 1 nhân vật + 1 action chém 6 frame và in thời gian (`/kaggle/working/bench`),
để bạn biết batch thật sẽ mất bao lâu và xem chất lượng trước.

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
  "engine": "edit",
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
| `engine` | `"edit"` | `"edit"` (FLUX.2 klein), `"pose"` (SD1.5 + ControlNet, engine cũ), `"video"` (WAN 2.2). Có thể đặt riêng cho từng action |
| `seed` | `-1` | `-1` = ngẫu nhiên. Seed thực tế được ghi trong `manifest.json` để tái lập |
| `frame_size` | `128` | Cạnh mỗi ô trong sheet (px). Mọi frame đúng bằng cỡ này |
| `char_height` | `frame_size/2` | Chiều cao nhân vật tính bằng pixel gốc (mode `prompt`) |
| `palette` | `32` | Số màu tối đa. Mọi frame dùng chung palette của nhân vật |
| `fps` | `10` | FPS ghi vào manifest và GIF |
| `view` | `"side"` | `"side"` (nhìn ngang, quay phải). Thư viện tư thế chỉ có góc nhìn ngang |
| `preview_scale` | `3` | Hệ số phóng của GIF xem trước |
| `edit` | xem dưới | Tham số engine edit |
| `debug` | `true` | Lưu ảnh soi lỗi cho mỗi action trong `debug/` (mục 4) |
| `checkpoint`, `steps`, `cfg_scale`, `pose`, `wan`, `bg_color` | | Chỉ cho engine cũ, xem mục 9 |

**`edit`** (thường không cần đổi):

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `size` | `768` | Cạnh ảnh model vẽ mỗi frame (bội của 16). Nhỏ hơn (VD 640) thì nhanh hơn, chi tiết kém hơn |
| `steps`, `cfg` | `4`, `1.0` | Bản klein distilled được huấn luyện cho 4 bước, cfg 1 |
| `fallback_denoise` | `0.7` | Khi phải vẽ lại kiểu img2img: phần ảnh con rối bị vẽ lại (0.7 = giữ 30%). Thấp hơn: giữ tư thế chắc hơn nhưng khớp thô còn lộ; cao hơn: sạch hơn nhưng dễ lệch tư thế |
| `follow_threshold` | `0.7` | Độ khớp hình bóng tối thiểu với con rối để nhận ảnh |
| `unet`, `text_encoder`, `vae` | 3 file klein 4B | Tên file model |

**Mỗi job:** `name` (chỉ dùng chữ, số, `_`, `-`); `mode` là `"prompt"` (cần
`prompt`) hoặc `"image"` (cần `character_image`); `seed` (tùy chọn); `actions`.

### Dùng sprite mẫu có sẵn (`mode: "image"`)
Tool lấy thẳng nhân vật trong ảnh của bạn rồi tạo các action. Mẫu đầy đủ:
`kaggle/jobs.image.example.json`.
1. **Chuẩn bị ảnh**: PNG **nền trong suốt** (tốt nhất) hoặc nền **một màu phẳng**. Một nhân
   vật, **đứng thẳng, hai tay buông** (dễ cắt thành con rối nhất), nhìn ngang quay phải. Ảnh
   pixel gốc hoặc đã phóng to đều được: ảnh phóng x2, x4... tự được thu về pixel gốc.
2. **Upload lên Kaggle**: trong notebook → **Add Input** → **Upload** (hoặc *New Dataset*) →
   kéo thả file PNG (nhiều nhân vật thì kéo nhiều file) → đặt tên dataset, ví dụ `my-sprites` → Create.
3. **Ghi vào `jobs.json`** (cell 5): `"character_image": "hero.png"`. **Chỉ cần tên file**: tool tự
   tìm trong mọi Input đã gắn. Nên thêm `"prompt"` mô tả nhân vật (tiếng Anh, VD "a knight in
   silver armor with a red cape and a long sword") để model biết chi tiết nào là gì.
4. Chạy cell 6 như bình thường.

Tool giữ nguyên sprite của bạn: không ép về `char_height`, giữ mọi màu gốc (tối đa 255), mọi
frame chỉ dùng màu của sprite mẫu. Chỉ thu nhỏ khi nhân vật cao hơn 80% `frame_size`; khi đó
hãy tăng `frame_size` (VD sprite cao 100px → `"frame_size": 160`). Upload bản sửa của ảnh
(cùng tên) rồi chạy lại → tool tự nhận ra và làm lại.

**Mỗi action:**

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `name`, `prompt` | bắt buộc | Tên thư mục; câu mô tả động tác **bằng tiếng Anh** (đưa vào prompt của model) |
| `pose` | `= name` | Tư thế mẫu. Có sẵn: `idle`, `walk`, `run`, `attack`, `heavy`, `hurt`, `jump`, `death`, `block`, `dash`, `crouch`, `fall`, `cast`, `kick`, `punch`. Action tên khác thì ghi rõ, VD `{"name": "slash2", "pose": "attack", ...}` |
| `engine` | theo `engine` chung | Engine riêng cho action này |
| `frames` | `8` | Số frame trong sheet (1–64) |
| `columns` | `= frames` | Số cột (mặc định 1 hàng) |
| `align` | `"fixed"` | `fixed` giữ nguyên vị trí thật (nhảy, chém). `track` bỏ phần trôi ngang |
| `loop` | theo tư thế | Ghi vào manifest |
| `seed` | theo job | Seed riêng cho action |

---

## 4. Kết quả

```
output/
├── run_log.txt                  log đầy đủ của lần chạy
└── hac_ronin/
    ├── character.png            nhân vật ở pixel gốc (dùng lại cho mọi action)
    ├── character_preview.png    bản phóng to để xem
    ├── character_raw.png        ảnh gốc model vẽ (mode prompt)
    ├── character.json           seed, prompt, kích thước, số màu
    ├── progress.json            trạng thái từng action (done / error + lý do)
    └── idle/
        ├── spritesheet.png      frame_size x frame_size mỗi ô, nền trong suốt
        ├── manifest.json        kích thước, số cột/hàng, fps, anchor, toạ độ từng frame, seed...
        ├── preview.gif
        └── debug/               (khi "debug": true)
            ├── reference.png    ảnh 2: sprite mẫu đưa vào model
            ├── puppet.png       ảnh 1: con rối từng frame
            ├── raw.png          ảnh model vẽ lại, từng frame
            └── clean.png        sau khi tách nền
```

`manifest.json` có `anchor` (điểm chân nhân vật trong ô) để engine game căn sprite, và
`meta.frames`: với mỗi frame, ảnh được vẽ kiểu nào (`edit` hoặc `img2img`) và độ khớp
với con rối (`follow`).

---

## 5. Mẹo chất lượng

- **Không ưng nhân vật?** Đặt `"seed"` khác cho job rồi chạy lại cell 6. Nhân vật và mọi
  action của job đó tự làm lại. Hoặc chọn một `character.png` ưng ý từ lần trước và dùng `mode: "image"`.
- **Frame sai tư thế:** mở `debug/puppet.png`. Con rối sai (tay/chân lấy nhầm phần) thường do
  sprite mẫu không đứng thẳng hai tay buông: dùng sprite đứng thẳng. Con rối đúng mà
  `raw.png` sai: giảm `edit.fallback_denoise` (VD 0.55) hoặc tăng `edit.follow_threshold` (VD 0.8).
- **Frame còn thô ở khớp tay/chân:** tăng `edit.fallback_denoise` (VD 0.8).
- **Đi/chạy bị trôi khỏi khung:** viết "in place" trong prompt và đặt `"align": "track"`.
- **Nhân vật bị cắt** (log báo "pixel bị cắt"): tăng `frame_size` hoặc giảm `char_height`.

---

## 6. Xử lý sự cố

| Thông báo | Cách xử lý |
|---|---|
| `Thiếu: model 'flux-2-klein-4b-fp8.safetensors'` (hoặc `qwen_3_4b`, `flux2-vae`) | Cell 2 của notebook là bản cũ: import lại `kaggle/SpriteForge_Kaggle.ipynb` (mục 1.4), chạy lại cell 2 rồi cell 3 |
| `Thiếu: node ComfyUI 'Flux2Scheduler'` (hoặc `ReferenceLatent`, `EmptyFlux2LatentImage`) | ComfyUI quá cũ: để `COMFY_REF = ""` ở cell 1 để dùng bản mới nhất |
| `FLUX.2 klein trả về ảnh gần như một màu (đen/trống)` | fp16 tràn số trên T4: ở cell 1 đặt `COMFY_ARGS = "--bf16-unet"` (chậm hơn) rồi chạy lại cell 3 và cell 6 |
| `frame N: ảnh vẽ lại không khớp tư thế` | Cả hai cách vẽ đều lệch: xem mục 5 "Frame sai tư thế" |
| `N/M frame phải vẽ lại kiểu img2img` | Bình thường: model chép tư thế đứng ở các frame đó nên tool vẽ lại từ con rối (tốn thêm thời gian) |
| `Nhân vật lần N: có nhiều nhân vật/mảng rời` | Model vẽ nhiều nhân vật, tool tự gen lại (tối đa 3 lần) và chỉ giữ nhân vật lớn nhất |
| `chưa có tư thế mẫu "..."` | Tên action không trùng tư thế có sẵn: thêm `"pose": "<tên có sẵn>"` vào action đó |
| `out of memory` | Giảm `edit.size` (VD 640) |
| `Nhân vật bị thủng ...% thân` | Tách nền ăn mất chi tiết: đổi seed của job, hoặc dùng `mode: "image"` với ảnh nền trong suốt |
| `ComfyUI không khởi động được` | Đọc log in ra ở cell 3 (thường do pip cài lỗi) |
| Session chết / "Kernel died" / lỗi kết nối giữa chừng khi dùng T4 x2 | Thiếu RAM khi 2 ComfyUI cùng giữ model: ở cell 1 đặt `NUM_GPUS = 1`, chạy lại từ cell 1. Action đã xong không bị chạy lại |
| Hết 30 giờ GPU/tuần | Chờ reset hằng tuần, hoặc thuê Vast.ai/RunPod (notebook chạy được trên mọi máy Linux có GPU) |

---

## 7. Thời gian

| Bước | Ước tính |
|---|---|
| Cài đặt (clone + pip) | ~5–8 phút |
| Tải FLUX.2 klein từ HuggingFace (nếu không có trong Input) | vài phút (~11.6GB) |
| Khởi động ComfyUI | ~1–2 phút |
| 1 nhân vật (768x1024, 4 bước) | ~15–30 giây (ước tính, chưa đo trên T4) |
| 1 frame (768², 2 ảnh mẫu, 4 bước) | ~10–20 giây; gấp đôi nếu phải vẽ lại img2img (ước tính, chưa đo trên T4) |
| **1 action 8 frame** | **~1.5–4 phút** mỗi GPU (ước tính, chưa đo trên T4) |

Bật `RUN_BENCHMARK` (mục 2) để đo thật trên GPU của bạn. Muốn nhanh hơn: `"edit": {"size": 640}`.

---

## 8. Test ở máy local (không cần GPU)

```bash
python -m unittest tests.test_forge -v
```

Bộ test dựng một ComfyUI giả, kiểm tra cả luồng: tạo nhân vật, con rối, kiểm tra tư thế và
vẽ lại img2img, sinh sheet/manifest/GIF, chạy tiếp từ chỗ dừng, và việc một action lỗi
không làm dừng cả batch.

---

## 9. Engine cũ

**`"engine": "pose"`**: SD1.5 pixel art + ControlNet OpenPose + IP-Adapter vẽ theo khung
xương, mặc định cả action trong một "bảng tư thế". Cần:
- `"checkpoint"`: tên file checkpoint SD1.5 pixel art, hoặc `"auto"` (lấy file đầu tiên trong
  Input). Upload checkpoint thành dataset Kaggle; mọi file `.safetensors` lạ trong Input được
  coi là checkpoint.
- Cell 1: `DOWNLOAD_POSE_SD = True` (ControlNet + IP-Adapter, ~3.3GB).
- Khoá `pose`: `strength` (1.0), `ipadapter` (true), `ipadapter_weight` (0.8), `size` (512),
  `layout` (`"grid"` hoặc `"single"`), `grid_max_cells` (9). `steps`/`cfg_scale` (30/7) là tham số SD1.5.
- Debug: `board.png`, `poses.png`, `raw.png`, `clean.png`.

**`"engine": "video"`**: WAN 2.2 TI2V 5B tạo clip rồi cắt frame. Chạy thật cho thấy WAN hay
làm nhân vật pixel nhỏ bị zoom, biến hình hoặc tan biến. Cần `DOWNLOAD_WAN = True` (~17GB).
Khoá `wan`: `steps` (20), `cfg` (5), `duration` (2.0 giây), `width`/`height` (640),
`useful_fraction` (0.85), `char_fill` (0.6), `timeout_s` (3600). `bg_color` (`"auto"`) là màu
nền canvas đưa vào WAN. Debug: `input.png` + `clip.gif`. Một action WAN ở 448² mất ~1 phút 24
giây trên T4 (đo 10/2026).
