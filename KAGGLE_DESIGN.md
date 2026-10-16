# THIẾT KẾ: Chạy SpriteForge trên Kaggle GPU (miễn phí)

> Tài liệu kỹ thuật cho người và coding agent bảo trì dự án.
> Ngôn ngữ UI/docs: tiếng Việt. Tên biến/hàm/file: tiếng Anh.
> Hướng dẫn sử dụng cho người dùng cuối: `kaggle/README.md`.

**v2 (15/10/2026):** ghép hai dự án mã nguồn mở. Engine animation Phase 1 là
**WAN 2.2 TI2V 5B (video)**, thay cho "grid img2img" của v1.

**v7 (16/10/2026): engine mặc định đổi sang `edit`** (mục 4c): FLUX.2 [klein] 4B, model sửa
ảnh theo ảnh mẫu, vẽ lại "con rối" (sprite mẫu uốn theo khung xương). Lý do: chạy thật với
engine pose, đưa sprite mẫu vào vẫn ra nhân vật khác (IP-Adapter SD1.5 chỉ giữ "na ná").

**v6 (16/10/2026): bảng tư thế** (mục 4b): cả action vẽ trong một ảnh, nhân vật mẫu giữ
nguyên ở ô đầu -> frame nhất quán hơn vẽ lẻ.

**v4 (15/10/2026): engine mặc định đổi sang `pose`** (mục 4b). Ba lần chạy thật với WAN
(448px) cho clip bị zoom, biến hình, bóng mờ, tan biến (xem `debug/clip.gif`): WAN 5B nén
32x32 px thành 1 token, sprite nhỏ chỉ vài token nên không giữ được hình; hậu xử lý không cứu
được lỗi kéo dài nhiều frame. WAN vẫn còn qua `"engine": "video"`.

---

## 1. Mục tiêu

Chạy pipeline text-to-sprite-sheet trên **Kaggle Notebooks GPU miễn phí**
(Tesla T4 16GB, 30 giờ/tuần), vì máy local của user không đủ cấu hình chạy model AI.

- Input: 1 file `jobs.json` (mô tả nhân vật + danh sách action cần gen).
- Output: với mỗi action → `spritesheet.png` + `manifest.json` + `preview.gif`.
- Chạy **batch headless, không cần canh** (Save & Run All), xong tải 1 file zip về.

## 2. Ràng buộc Kaggle

1. **GPU miễn phí**: T4 16GB (hoặc T4 x2 / P100). Quota **30 giờ GPU/tuần**. T4 **không có bf16/fp8 phần cứng** → model chạy fp16.
2. **Session tối đa 12 giờ**. Commit ("Save & Run All") chạy nền, không cần mở trình duyệt.
3. **Máy ephemeral**: chỉ còn lại `/kaggle/input/` (Dataset, chỉ đọc) và Output của bản Commit (`/kaggle/working`, ≤ 20GB).
4. **Cần verify số điện thoại** mới bật được GPU + Internet.
5. **Không UI web**: chạy headless `run_batch.py`.
6. RAM ~30GB: đủ cho WAN 5B (~17GB file), **không** đủ cho MiniMax H3 (~39GB).
7. Không bf16 nên model chỉ hỗ trợ bf16/fp32 trong ComfyUI (`supported_inference_dtypes` của
   QwenImage, QwenImage21, Flux3, ZImage...) chạy fp32 trên T4: chậm gấp nhiều lần. Flux2
   (FLUX.2 klein) và WAN hỗ trợ fp16.

## 3. Nguồn gốc code: ghép hai dự án

Đã phân tích mã nguồn của cả hai repo (10/2026). Kết luận:

| | Pixel Art Studio (velteyn, **GPLv3**) | Sprite Sheep (MANFRI-DEV, **MIT**) |
|---|---|---|
| Ý tưởng | Text → nhân vật SD1.5 → pose ControlNet → sheet | Ảnh sprite + prompt → video diffusion → sheet |
| Lấy dùng | Prompt "anchor", lượng tử palette, khối pixel, neo chân, đóng gói sheet + manifest + GIF | Graph WAN 2.2 TI2V 5B (chỉ dùng node core), tách nền chroma 2 ngưỡng + khử spill, chọn frame cách đều, chờ ComfyUI chịu được timeout |
| **Không lấy** | GUI PyQt6. **Triple-ControlNet chỉ chạy với A1111**: `ComfyUIClient` bỏ qua `controlnet`/`loras`, đổi seed -1 → 0, timeout 150s. Thư viện pose chỉ 2–6 frame/action, không có attack/hurt. Prompt dài >1000 ký tự (CLIP chỉ đọc 77 token) | GUI Godot, MiniMax H3 (39GB, giấy phép cấm EU/UK/KR/US), sidecar HTTP server, watermark (đang tắt) |

Cách tích hợp: **chép có chọn lọc và sửa** vào `forge/` (ghi nguồn ở đầu mỗi file
và trong `THIRD_PARTY_NOTICES.md`), KHÔNG gọi CLI Sprite Sheep qua subprocess, vì
sidecar có tác dụng phụ khi import (tạo thư mục, đổi `HF_HOME`), import trần
trùng tên (`config`, `sheet`...) và đòi model nằm trong thư mục riêng của nó.
Dự án theo **GPL-3.0** (vì có code GPLv3). Ảnh đầu ra không bị ràng buộc.

## 4. Kiến trúc

```
jobs.json
  │
  ├─ LƯỢT 1  nhân vật (mọi job)                         forge.pipelines.generate_character / prepare_character
  │    mode "prompt": SD1.5 txt2img 512x768 "simple white background" (không trọng số: nền xanh
  │                   có trọng số làm màu xanh lan sang quần áo); nền viền không đều hoặc nhân vật
  │                   chạm mép ảnh -> gen lại seed+1 (tối đa 3 lần), lấy ảnh tốt nhất
  │    mode "image" : ảnh có sẵn (trong suốt hoặc nền một màu)
  │    ──► tách nền LOANG TỪ MÉP (matting.cutout_flat): chỉ xoá nền nối với mép ảnh + khoảng
  │        nền bị kẹp đủ lớn; KHÔNG xoá mọi pixel cùng màu nền (lỗi thật: nền đen ăn 24% thân
  │        nhân vật tối màu). remove_background (xoá mọi vùng cùng màu) chỉ dùng cho canvas WAN.
  │    ──► crop ──► thu về cao char_height (trung bình khối có alpha) ──► alpha nhị phân
  │    ──► rút palette (median-cut, ≤ palette màu) ──► character.png (pixel gốc)
  │    ──► POST /free (gỡ SD khỏi VRAM)
  │
  └─ LƯỢT 2  animation (mọi action, resume theo progress.json)   forge.pipelines.animate_video + finalize
       character.png phóng ×k (NEAREST, nhân vật ≈ 60% chiều cao) lên canvas 640² nền key (màu phông thật: xanh (0,177,64)/xanh dương/magenta dịu)
       (key = green/magenta/blue khác xa palette nhất) ──► upload
       ──► WAN 2.2 TI2V 5B (uni_pc, 20 steps, cfg 5, shift 8) clip 4n+1 frame @24fps
       ──► lấy N frame cách đều trên 85% đầu clip (chỉ tải về N frame đó)
       ──► (debug) lưu input.png + clip.gif đầy đủ
       ──► mỗi frame: tách nền key ──► thu ÷k về pixel gốc ──► alpha nhị phân ──► ép palette nhân vật
           ──► xoá mảnh vụn < max(3px, 0.6% thân)
       ──► đặt vào ô frame_size² (cùng một độ lệch cho cả action; canvas gốc nhỏ hơn ô thì
           đặt trọn trong ô, không cắt pixel nào; "track" bỏ trôi ngang)
       ──► spritesheet.png + manifest.json + preview.gif
```

Vì sao hai lượt: SD1.5 (~2GB) và WAN (~17GB) không cần nằm cùng lúc trong
VRAM/RAM; chạy xen kẽ theo job sẽ nạp lại WAN nhiều lần.

Vì sao nhân vật dựng ở pixel gốc rồi phóng ×k: biết chính xác k nên thu frame
WAN về đúng lưới pixel, và biết vị trí chân → mọi frame đặt bằng một độ lệch
chung (giữ chuyển động thật: nhảy lên vẫn bay lên, không bị "dí" xuống đất như
`align_frames` gốc).

## 5. Bố cục mã nguồn

```
forge/
  comfy.py       ComfyClient (urllib): ping, missing_nodes, list_models, upload_image,
                 queue/wait/run (chịu timeout khi GPU bận), fetch, free; lỗi -> ComfyError dễ đọc
  workflows.py   sd15_txt2img(), wan22_i2v() (node core + PreviewImage), wan_length(), frame_indices()
  matting.py     remove_background() chroma 2 ngưỡng + despill (từ Sprite Sheep)
  pixel.py       prepare_character, compose_video_input, video_frame_to_native, place_in_cells,
                 extract/apply_palette, key_color_for, resize_premultiplied, remove_specks
  sheet.py       pack (bắt buộc cùng cỡ), manifest, save_gif (trong suốt), write_outputs
  config.py      load/parse jobs.json -> Settings/Job/Action, thông báo lỗi tiếng Việt
  pipelines.py   generate_character, prepare_character, animate_video, finalize, resolve_seed
kaggle/
  run_batch.py               CLI batch (mục 7)
  SpriteForge_Kaggle.ipynb   notebook (mục 5b)
  jobs.example.json          Hắc Ronin, 8 action
  jobs.bench.json            1 action để đo tốc độ
  README.md                  hướng dẫn người dùng
tests/
  fake_comfy.py   ComfyUI giả (HTTP) sinh ảnh/clip tổng hợp, có tiêm lỗi "FAIL_ME"
  test_forge.py   unit + toàn luồng (resume, cô lập lỗi, preflight)
```

## 5b. Notebook Kaggle

| Cell | Nội dung |
|---|---|
| 0 (md) | 4 bước chuẩn bị + cách chạy |
| 1 | Cấu hình: `SPRITEFORGE_REPO`, `COMFY_REF`, `RUN_BENCHMARK`, `WORK=/tmp/spriteforge`, `OUTPUT` |
| 2 | Code SpriteForge (dataset `spriteforge-code` hoặc git clone) → clone ComfyUI + `pip install` → **gom model bằng symlink** vào `WORK/models/{checkpoints,diffusion_models,text_encoders,vae}`: checkpoint = mọi `.safetensors/.ckpt` trong `/kaggle/input` trừ 3 file WAN; file WAN lấy từ dataset nếu có, không thì `hf_hub_download` từ `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` → ghi `extra_model_paths.yaml` |
| 3 | Mỗi GPU (`nvidia-smi -L`, giới hạn bởi `NUM_GPUS`) một ComfyUI nền: `CUDA_VISIBLE_DEVICES=i`, port `8188+i`, thư mục input/output/temp/user riêng (ComfyUI cũ không có cờ → chạy lại không cờ). Chờ `/system_stats`, in log nếu lỗi. Xuất `COMFY_URLS` |
| 4 | (tùy chọn) benchmark `jobs.bench.json` |
| 5 | `%%writefile /kaggle/working/jobs.json` (cell duy nhất user sửa) |
| 6 | `run_batch.py --jobs ... --out /kaggle/working/output` |
| 7 | zip → `output.zip` |
| 8 | (tùy chọn) hiển thị các `preview.gif` |

ComfyUI và model đặt ở `/tmp` để không lẫn vào Output (giới hạn 20GB).
`run_batch.py --comfy-url url1,url2`: lượt 1 (SD) chạy trên url1; lượt 2 mỗi URL một luồng
lấy action từ hàng đợi chung (GPU thứ i chờ 45s×i trước việc đầu tiên để không nạp WAN vào RAM
cùng lúc); `progress.json` ghi dưới khoá. URL không ping được thì bỏ qua.

**Notebook phụ `kaggle/SpriteForge_Models.ipynb`** (CPU, Internet): tải 3 file WAN từ HF vào
`/kaggle/working` (16.9GB < 20GB), Commit một lần; notebook chính Add Input output này. Cell 2
tìm file theo tên trong `/kaggle/input/**`, nên không cần upload 17GB từ máy người dùng.
Notebook sinh từ script, nội dung cell 5 lấy nguyên `kaggle/jobs.example.json`.

## 6. `jobs.json`: schema

```json
{
  "comfy_url": "http://127.0.0.1:8188",
  "checkpoint": "auto",
  "seed": -1,
  "steps": 30,
  "cfg_scale": 7,
  "sampler": "euler",
  "scheduler": "normal",
  "frame_size": 128,
  "char_height": 64,
  "palette": 32,
  "fps": 10,
  "view": "side",
  "bg_color": "auto",
  "preview_scale": 3,
  "wan": {
    "unet": "wan2.2_ti2v_5B_fp16.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
    "width": 640, "height": 640,
    "steps": 20, "cfg": 5.0, "shift": 8.0,
    "duration": 2.0, "useful_fraction": 0.85,
    "char_fill": 0.6, "timeout_s": 3600
  },
  "jobs": [
    {
      "name": "hac_ronin",
      "mode": "prompt",
      "prompt": "dark samurai in black-indigo armor, oni mask with glowing red eyes, tattered black cloak, two katanas",
      "seed": null,
      "actions": [
        {"name": "idle",   "prompt": "standing still in a ready stance, calm breathing", "frames": 8},
        {"name": "walk",   "prompt": "walking in place, steady walk cycle", "frames": 8, "align": "track"},
        {"name": "attack", "prompt": "draws a katana and slashes horizontally", "frames": 8, "columns": 4},
        {"name": "hurt",   "prompt": "recoils backward from a hit, then recovers", "frames": 6, "loop": false},
        {"name": "death",  "prompt": "collapses to the ground", "frames": 8, "duration": 3.0, "loop": false}
      ]
    },
    {
      "name": "chibi_samurai",
      "mode": "image",
      "character_image": "/kaggle/input/spriteforge-chars/chibi.png",
      "actions": [{"name": "jump", "prompt": "jumps straight up and lands", "frames": 6}]
    }
  ]
}
```

Quy tắc (kiểm tra trong `forge/config.py`):
- `mode: "prompt"` cần `prompt`. Có ít nhất 1 job prompt thì cần `checkpoint` (tên file hoặc `"auto"`).
- `mode: "image"` cần `character_image`: đường dẫn tuyệt đối, hoặc tương đối so với thư mục chứa `jobs.json`.
- `name` của job/action: `[A-Za-z0-9_-]+`, không trùng nhau.
- Mỗi action có `frames` riêng (1–64), `columns` mặc định bằng `frames` (1 hàng).
- `align`: `fixed` (mặc định) | `track`. `duration` theo action, mặc định `wan.duration`.
- `char_height` mặc định `frame_size/2`, phải ≤ `frame_size`.
- Seed: action.seed → job.seed → seed chung. `-1` = ngẫu nhiên mỗi lần gọi; seed thực tế ghi vào manifest.
- `denoise` của v1 đã bỏ (không còn img2img).

## 7. `kaggle/run_batch.py`

```
parse args: --jobs, --out, [--comfy-url]
log ra stdout + out/run_log.txt
cfg = config.load(jobs)                 # lỗi schema -> exit 2, thông báo rõ trường nào
client.ping()                           # không nối được -> exit 2
preflight: node core còn thiếu (Wan22ImageToVideoLatent...), file model không có trong
           /models/<folder> (in danh sách ComfyUI đang thấy), ảnh character_image không tồn tại,
           checkpoint "auto" -> chọn file đầu tiên.   Có vấn đề -> exit 2 TRƯỚC khi tốn giờ GPU
LƯỢT 1: for job: character.png có rồi -> dùng lại; chưa -> gen/đọc ảnh -> lưu
        character.png, character_preview.png, character_raw.png, character.json
        lỗi -> ghi nhận, mọi action của job đó = error, chạy job khác
        có gen -> client.free()
LƯỢT 2: for job, action:
        progress.json[action].status == "done" và spritesheet.png tồn tại -> "bỏ qua"
        try animate_video + finalize -> progress = {status: done, seconds, seed}
        except -> log traceback, progress = {status: error, error, seconds}; TIẾP TỤC
        ghi progress.json ngay sau mỗi action (ghi file tạm rồi rename)
tổng kết: số xong / bỏ qua / lỗi, tổng thời gian. exit 0 nếu không lỗi, 1 nếu có lỗi
```

Log mỗi action: tên, thời gian, dung lượng 3 file; cảnh báo pixel bị cắt và trôi ngang lớn (gợi ý `track`).

## 8. Tài nguyên (T4 16GB)

| Hạng mục | Ước tính | Ghi chú |
|---|---|---|
| Cell 2 setup (clone + pip) | ~5–8 phút | |
| Tải WAN từ HF (khi không có trong dataset) | vài phút | ~17GB |
| Cell 3 start ComfyUI | ~1–2 phút | model nạp ở prompt đầu |
| Nhân vật SD1.5 512x768, 30 steps | ~20–60s/job | tối đa 3 lần gen nếu ảnh hỏng |
| **1 action WAN 448², 49 frame, 20 steps** | **~1m24s** | đo thật trên Kaggle T4 (10/2026), ổn định giữa các action |
| **21 action** | **~30 phút** | ≪ 3 giờ của DoD; 30h/tuần đủ chạy hàng chục batch |
| 21 action trên T4 x2 (448²) | ~15 phút | 2 GPU chạy được trên Kaggle (10/2026) |
| 1 action WAN 640² (mặc định v3) | ~3 phút (ước tính) | 2× token; cần đo lại |

Thời gian bị giới hạn bởi GPU, không phải code: log thật cho thấy thời gian tỉ lệ đúng
với số frame video (49 frame 84s, 73 frame 128s, ≈1.8s/frame, phần cố định ≈0). Lý thuyết:
5B tham số × 2 (CFG) × ~2.5k token ≈ 50 TFLOP/step; T4 thực tế ~15–20 TFLOPS → ~3s/step.
Còn có thể giảm (đánh đổi chất lượng): bớt `wan.steps`, clip ngắn hơn, `wan.width/height` 384.

VRAM: WAN 5B fp16 ~10GB + latent nhỏ (448²×49 frame ≈ 2.5k token) → dự kiến vừa
16GB. VAE decode là đỉnh VRAM. Rủi ro: fp16 trên T4 có thể sinh NaN (frame đen),
pipeline phát hiện và báo lỗi rõ ràng. Cách thử: `--fp32-vae`.

## 9. Bảo mật & chi phí

- Token HuggingFace (nếu sau này push output): Kaggle Secrets → `os.environ["HF_TOKEN"]`, không hardcode. Hiện tải WAN **không cần token** (repo public).
- Dataset `spriteforge-models`, `spriteforge-code`: để Private.
- Không phát sinh chi phí (quota miễn phí). Hết 30h/tuần: chờ reset hoặc thuê Vast.ai/RunPod (notebook chạy trên mọi máy Linux + GPU).

## 10. Tiêu chí nghiệm thu (Definition of Done)

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 1 | Notebook chạy Cell 1 → 7 không can thiệp tay (ngoài sửa jobs.json), ra đủ sheet cho mọi action mẫu | ⏳ cần chạy thật trên Kaggle |
| 2 | Mỗi action đủ `spritesheet.png`, `manifest.json`, `preview.gif`; mọi frame cùng kích thước tuyệt đối | ✅ `sheet.pack` từ chối frame lệch cỡ; test toàn luồng |
| 3 | Chạy lại cùng `jobs.json` → bỏ qua action `done` (resume qua `progress.json`), không gen lại nhân vật | ✅ test |
| 4 | 1 action lỗi không làm dừng cả batch | ✅ test (tiêm lỗi `FAIL_ME`) |
| 5 | Lỗi cấu hình/model/node phát hiện TRƯỚC khi chạy GPU | ✅ test preflight |
| 6 | Tổng thời gian 21 action < 3 giờ trên T4 (không tính setup) | ✅ ~1m24s/action → ~30 phút (đo 10/2026) |
| 7 | `kaggle/README.md`: verify phone → GPU T4 → dataset → Internet → jobs.json → Run All/Commit → tải `output.zip` | ✅ |
| 8 | `kaggle/jobs.example.json` đúng schema mục 6 (Hắc Ronin) | ✅ test parse |
| 9 | Ghi nguồn và giấy phép hai dự án gốc | ✅ `THIRD_PARTY_NOTICES.md`, `LICENSE` |

Chạy test local: `python -m unittest tests.test_forge -v` (không cần GPU).

## 4b. Engine `pose` (mặc định từ v4)

Ý tưởng triple-ControlNet của Pixel Art Studio, làm lại cho ComfyUI:

- **Thư viện tư thế** `forge/poses.py`: 15 preset nhìn ngang quay phải (`idle walk run attack
  heavy hurt jump death block dash crouch fall cast kick punch`), mỗi preset là vài keyframe
  (góc khớp: thân, đầu, vai/khuỷu, hông/gối cho phía gần/xa, dx, độ cao nhảy). Động học thuận
  -> 18 điểm OpenPose; nội suy cosine ra đúng N frame (loop: chia đều một vòng, không lặp
  frame cuối). Điểm thấp nhất luôn chạm đường nền (+ `air` khi nhảy) -> mọi frame/action cùng
  tỉ lệ, cùng đường nền. Vẽ đúng màu/độ dày của `controlnet_aux.draw_bodypose`.
- **Mỗi frame**: `CheckpointLoaderSimple` -> (`IPAdapterUnifiedLoader` "PLUS" + `IPAdapter` với
  ảnh nhân vật phóng NEAREST trên nền trắng) -> `ControlNetApplyAdvanced` (openpose fp16) ->
  `KSampler` (cùng seed cho cả action) -> nền trắng. Hậu xử lý: `cutout_flat` -> thu về pixel
  gốc theo tỉ lệ `chiều cao sprite / chiều cao đứng của khung xương` (chung cho mọi action)
  -> ép palette nhân vật -> `place_in_cells` (layout cố định, không trôi).
- **Làm sạch theo khung xương** (`forge/cleanup.py`, v5): chạy thật thấy checkpoint vẽ 3 nhân
  vật kiểu character sheet, đĩa tròn sau lưng, vệt đất; IP-Adapter chép chúng sang mọi frame.
  Lọc: giữ mảng chạm lõi khung xương; bóc mảng một màu lớn sát nền nằm >= 60% ngoài thân
  (đã thử nới cho mảng rất lớn: bóc mất áo choàng thật -> bỏ); xoá viền/nền trắng lộ ra; bỏ
  phần "dày" ngoài vùng thân rộng (giữ kiếm mảnh); dưới đường nền chỉ giữ bàn chân. Nhân vật
  mẫu bẩn (> 15% bị xoá) -> thử seed khác.
- **Bảng tư thế** (`forge/grid.py`, v6, `pose.layout = "grid"` mặc định): vẽ lẻ từng frame thì
  mỗi lần SD "nghĩ lại" chi tiết (hoa văn, màu dây lưng, độ dài kiếm) -> frame nhấp nháy dù cùng
  seed + IP-Adapter. Giờ cả action là MỘT ảnh lưới (`grid.shape`: 9 ô -> 3x3, 7 -> 4x2; ô 512px):
  ô 0 = nhân vật mẫu (`reference_image`, khung xương `NEUTRAL`), ô 1.. = các frame, ô thừa trắng.
  Graph: `LoadImage`(bảng) -> `VAEEncode` -> `SetLatentNoiseMask`(`LoadImageMask` kênh red: trắng
  = ô frame) -> `KSampler` denoise 1.0. Vùng ngoài mặt nạ được giữ đúng ảnh gốc ở mỗi bước khử
  nhiễu, nên self-attention của các ô frame luôn "nhìn" thấy nhân vật mẫu và chép theo (giống
  reference-only, không cần model thêm). Xu hướng tự vẽ character sheet của checkpoint thành lợi
  thế: `GRID_PROMPT` nói "the same character repeated", `GRID_NEGATIVE` bỏ các từ cấm nhiều nhân
  vật. Action dài hơn `grid_max_cells - 1` frame -> chia đều nhiều bảng (`grid.chunks`), mỗi bảng
  có ô mẫu riêng. Bảng lỗi (VD hết VRAM) -> tự vẽ từng frame (`layout` trong manifest = "single").
  Chưa đo trên T4: 1536² SD1.5 + ControlNet + IP-Adapter ước tính 1.5–2.5 phút/bảng.
- **Nhân vật** (mode prompt): vẽ bằng ControlNet theo khung xương đứng `NEUTRAL` -> đúng
  hướng nhìn ngang và cùng khung hình với các frame.
- **IP-Adapter** (cubiq/ComfyUI_IPAdapter_plus, repo đã lưu trữ): thiếu node/model -> preflight
  tắt kèm cảnh báo, engine vẫn chạy. `workflows.fill_defaults` đọc `/object_info` để điền input
  bắt buộc còn thiếu và sửa enum đổi tên (`weight_type`, `preset`...) giữa các phiên bản.
- Model: `control_v11p_sd15_openpose_fp16` (0.7GB), `ip-adapter-plus_sd15` (0.1GB),
  `CLIP-ViT-H-14-laion2B-s32B-b79K` (2.5GB) - cell 2 tìm trong Input theo tên, thiếu thì tải HF.

## 4c. Engine `edit` (mặc định từ v7)

**Chọn model (16/10/2026)**, đứng đầu bảng sửa ảnh open-weights (Artificial Analysis):

| Model | Elo | Giấy phép | Trên T4 |
|---|---|---|---|
| Qwen-Image-2.1 (7B, 09/2026) | 1083 | Qwen Research: **cấm thương mại** | chỉ bf16/fp32 -> fp32, chậm |
| HunyuanImage 3.0 Instruct | 1064 | | 80B MoE, không vừa |
| FLUX.2 [klein] 9B | 1015 | FLUX non-commercial | fp16 được; 2 ComfyUI x (9B + Qwen3 8B) vượt RAM 30GB |
| Qwen-Image-Edit 2511 (20B) | 991 | Apache-2.0 | chỉ bf16/fp32; PR fp16 (#12522) chưa merge |
| **FLUX.2 [klein] 4B** (01/2026) | 937 | **Apache-2.0** | **fp16, 4 bước, 3.8GB fp8 + Qwen3 4B 7.5GB** |
| FLUX.3 Image Klein (5.6B, 15/10/2026) | | Kommunity (< 5 triệu USD) | chỉ bf16/fp32, 16 bước |

**Thử thật trên Space chính thức `black-forest-labs/FLUX.2-klein-4B`** (sprite ronin 36x64 phóng
NEAREST lên 768, ảnh 1 = sprite đứng, ảnh 2 = ma-nơ-canh đang chạy/chém, prompt bảo vẽ theo
ma-nơ-canh): nhân vật giữ **đúng y** (giáp, khăn đỏ, kiếm), nhưng **tư thế bị chép từ ảnh 1**
(đứng). Kết luận: klein giữ rất chặt bố cục + tư thế của ảnh 1 -> ảnh 1 phải đã ở đúng tư thế.

**Con rối** (`forge/puppet.py`):
- `place`: phóng sprite NEAREST xk (k nguyên, nhân vật cao ~`poses.UNIT` cạnh ảnh), toạ độ là
  bội của k (frame trả về thu lại xk là khớp đúng lưới pixel gốc, `VideoLayout(k=k)`); hông theo
  trung vị cột của 35% dưới (vũ khí không kéo lệch); khung xương đặt theo chính sprite
  (1 đơn vị = chiều cao sprite).
- `Rig`: chia pixel theo khung xương `NEUTRAL`: trên cổ = đầu; cách xương tay gần <= 0.05 = tay
  (cánh/cẳng theo xương gần hơn); dưới hông = chân (4 đoạn, theo xương gần nhất -> chân dính
  nhau bị chia đôi); còn lại = thân. Chỗ tay rời đi được lấp màu thân gần nhất (không thì "tay
  ma"). Tay xa = bản sao tay gần tối 70% (sprite nhìn ngang không có pixel tay xa).
- `pose(key)`: mỗi phần xoay cứng quanh khớp đầu theo đoạn xương đích (PIL affine NEAREST), vẽ từ
  sau ra trước: tay xa, chân xa, chân gần, thân, đầu, tay gần.
- `follows_pose`: khớp hình bóng (min hai chiều "phần A nằm trong 0.04 đơn vị của B") >= 0.7 VÀ
  "phần dời chỗ" (pixel con rối cách tư thế đứng > 0.04) được phủ >= 50%. Chỉ so hình bóng
  không bắt được lỗi: giơ kiếm qua đầu chỉ đổi ~5% hình bóng. Kiểm trên ảnh thật của Space:
  ảnh chép tư thế đứng bị loại.
- `keep_near`: bỏ mảng xa con rối và vệt bóng dưới đường nền ngoài bàn chân.

**Mỗi frame** (`pipelines.animate_edit`): upload con rối + sprite mẫu (cùng chỗ đặt) ->
`workflows.klein` (UNETLoader + CLIPLoader `flux2` + VAELoader; `CLIPTextEncode` ->
`ReferenceLatent` x2 (con rối, mẫu; negative = `ConditioningZeroOut` cũng gắn 2 ảnh);
`CFGGuider` cfg 1; `Flux2Scheduler` 4 bước; `EmptyFlux2LatentImage`; `SamplerCustomAdvanced`
euler) -> kiểm `follows_pose`; không đạt -> img2img: latent = VAEEncode(con rối),
`SplitSigmasDenoise(fallback_denoise=0.7)`; chọn ảnh khớp nhất -> `cutout_flat` -> `keep_near`
-> `video_frame_to_native` (thu xk, ép palette). Cùng seed mọi frame. ComfyUI cache
`CLIPTextEncode`/`VAEEncode` ảnh mẫu giữa các frame cùng action.

**Nhân vật** (mode prompt, `generate_character_edit`): klein text-to-image 768x1024, nền trắng;
kiểm nền, chạm mép, "một nhân vật" (mảng lớn nhất + mảng sát nó >= 85%); tối đa 3 seed; chỉ giữ
nhân vật chính.

**Chưa đo trên T4.** Ước tính: 4B x 2 x ~6.9k token (3 ảnh 768²) ≈ 55 TFLOP + attention
≈ 15 TFLOP/bước; T4 ~20–25 TFLOPS thực -> ~3s/bước -> ~12–20s/frame. RAM: mỗi ComfyUI giữ
~11.6GB (klein fp8 3.8 + Qwen3 4B bf16 7.5 + VAE); 2 GPU ~23GB < 30GB (WAN 2 GPU ~31GB đã chạy
được với `STAGGER_S`). Rủi ro: Flux2 fp16 trên T4 sinh NaN -> `_klein_draw` báo lỗi ảnh một màu,
README hướng dẫn `COMFY_ARGS = "--bf16-unet"`.

Hướng nâng cấp nếu chưa đạt: LoRA RefControl (thedeoxen, Apache-2.0) cho klein 4B **Base**
(ảnh độ sâu + ảnh mẫu, ~50 bước + CFG -> chậm ~10 lần); LoRA pixel art cho klein 4B.

## 11. Phase 2 (chưa làm)

- Thêm preset tư thế theo 21 trạng thái game.
- Engine edit: con rối hiểu tỉ lệ chibi (đầu to), vũ khí cầm tay đi theo cổ tay; LoRA RefControl.
- Nhiều hướng nhìn (Pixel Art Studio có `direction_gen`: south/east/west/north).
- Train LoRA riêng cho từng nhân vật để khoá identity.
- Auto-push output lên HuggingFace Dataset sau mỗi batch.
- Resume xuyên session: gắn Output của lần Commit trước làm Input.
- Nút "mở bằng Colab" dự phòng khi hết quota Kaggle.
