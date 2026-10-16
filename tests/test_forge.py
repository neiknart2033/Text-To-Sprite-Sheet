"""Test SpriteForge không cần GPU: đơn vị + toàn luồng với ComfyUI giả.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))

from forge import cleanup, comfy, config, matting, pipelines, pixel, poses, sheet, workflows  # noqa: E402
from tests.fake_comfy import FakeComfy, fake_character  # noqa: E402

import run_batch  # noqa: E402

comfy.POLL_S = 0.01


def _close_log_handlers():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


class UnitTests(unittest.TestCase):
    def test_wan_length_is_4n_plus_1(self):
        for d in (1.0, 2.0, 2.1, 3.3, 10.0, 99.0):
            n = workflows.wan_length(d)
            self.assertEqual((n - 1) % 4, 0)
            self.assertGreaterEqual(n, min(d * 24, workflows.WAN_MAX_LENGTH))
        self.assertEqual(workflows.wan_length(2.0), 49)

    def test_frame_indices(self):
        idx = workflows.frame_indices(49, 8, 0.85)
        self.assertEqual(len(idx), 8)
        self.assertEqual(idx, sorted(set(idx)))
        self.assertLessEqual(idx[-1], int(49 * 0.85))
        self.assertEqual(len(workflows.frame_indices(5, 8)), 8)  # clip ngắn hơn số frame

    def test_matting_green(self):
        a = np.zeros((64, 64, 3), np.uint8)
        a[:] = (0, 255, 0)
        a[20:40, 20:40] = (200, 30, 30)
        out = np.asarray(matting.remove_background(Image.fromarray(a), "green"))
        self.assertEqual(out[0, 0, 3], 0)
        self.assertEqual(out[30, 30, 3], 255)

    def test_prepare_character_height_and_palette(self):
        native, palette = pipelines.prepare_character(fake_character(1), self._cfg())
        self.assertEqual(native.height, 64)
        colors = np.unique(np.asarray(native)[np.asarray(native)[:, :, 3] > 0][:, :3], axis=0)
        self.assertLessEqual(len(colors), 16)
        self.assertTrue(set(np.asarray(native)[:, :, 3].ravel()) <= {0, 255})

    def test_dark_character_on_black_background_keeps_body(self):
        # Lỗi thật trên Kaggle: nền đen -> cách cũ xoá luôn tóc/chân đen của nhân vật
        raw = fake_character(4, bg="black")
        leg = (29 * 8 + 4, 45 * 8)  # một pixel ở chân màu (25,25,25)
        self.assertLess(matting.remove_background(raw, "auto").getpixel(leg)[3], 128)  # cách cũ: mất chân
        self.assertEqual(matting.cutout_flat(raw).getpixel(leg)[3], 255)
        new, _ = pipelines.prepare_character(raw, self._cfg())
        self.assertEqual(new.height, 64)
        self.assertLess(pixel.hole_ratio(new), 0.01)
        self.assertFalse(matting.matches_key(raw, "green")[0])
        self.assertTrue(matting.matches_key(fake_character(4), "green")[0])

    def test_vignette_corners_do_not_shrink_character(self):
        # Lỗi thật: 4 góc ảnh SD tối hơn nền -> khung bao = cả ảnh -> nhân vật 52px thay vì 64px
        a = np.asarray(fake_character(5, bg="white")).copy()
        for ys, xs in ((slice(0, 4), slice(0, 4)), (slice(0, 4), slice(-4, None)),
                       (slice(-4, None), slice(0, 4)), (slice(-4, None), slice(-4, None))):
            a[ys, xs] = (200, 200, 200)
        raw = Image.fromarray(a)
        self.assertTrue(matting.matches_key(raw, "white")[0])  # góc tối không làm hỏng kiểm tra nền
        native, _ = pipelines.prepare_character(raw, self._cfg())
        self.assertEqual(native.height, 64)

    def test_detect_block_size(self):
        native, _ = pipelines.prepare_character(fake_character(9), self._cfg())
        self.assertEqual(pixel.detect_block_size(native), 1)
        for k in (2, 4, 7):
            self.assertEqual(pixel.detect_block_size(pixel.upscale(native, k)), k)

    def test_touches_edge(self):
        self.assertFalse(pixel.touches_edge(matting.cutout_flat(fake_character(6, bg="white"))))
        self.assertTrue(pixel.touches_edge(matting.cutout_flat(fake_character(6, bg="white", cut=True))))

    def test_pick_frames_skips_glitches(self):
        native, _ = pipelines.prepare_character(fake_character(8), self._cfg())
        good = Image.new("RGBA", (112, 112), (0, 0, 0, 0))
        good.alpha_composite(native, (30, 30))
        blank = Image.new("RGBA", (112, 112), (0, 0, 0, 0))
        big = pixel.upscale(native, 2).crop((0, 0, 112, 112))  # "zoom" đột ngột
        frames = [good] * 49
        glitch = {6: blank, 17: big, 23: blank, 24: blank}
        frames = [glitch.get(i, f) for i, f in enumerate(frames)]
        targets = workflows.frame_indices(49, 8, 0.85)
        chosen = pipelines._pick_frames(targets, 49, lambda i: frames[i])
        self.assertFalse(set(chosen) & set(glitch))
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertTrue(all(abs(c - t) <= pipelines.FRAME_SEARCH for c, t in zip(chosen, targets)))

    def test_place_in_cells_uniform_and_clipping(self):
        native, _ = pipelines.prepare_character(fake_character(2), self._cfg())
        canvas, layout = pixel.compose_video_input(native, 448, 448, (0, 255, 0))
        frame = pixel.video_frame_to_native(matting.remove_background(canvas, "green"), layout,
                                            pipelines.palette_of(native, 16))
        cells, info = pixel.place_in_cells([frame] * 3, layout, 128, 8)
        self.assertEqual({c.size for c in cells}, {(128, 128)})
        self.assertEqual(info["clipped_px"], 0)
        cells, info = pixel.place_in_cells([frame], layout, 32, 2)
        self.assertGreater(info["clipped_px"], 0)

    def test_canvas_smaller_than_cell_never_clips(self):
        # Lỗi thật trên Kaggle: canvas gốc 112px < ô 128px nhưng vẫn cắt mất phần dưới chân
        native, _ = pipelines.prepare_character(fake_character(3), self._cfg())
        _, layout = pixel.compose_video_input(native, 448, 448, (0, 255, 0))
        self.assertEqual(layout.native_size, (112, 112))
        frame = Image.new("RGBA", layout.native_size, (0, 0, 0, 0))
        frame.paste((200, 30, 30, 255), (0, 100, 112, 112))  # vươn sát đáy và hai mép canvas
        cells, info = pixel.place_in_cells([frame], layout, 128, 8)
        self.assertEqual(info["clipped_px"], 0)
        a = info["anchor"]
        self.assertTrue(0 < a["y"] <= 128 - 8 and abs(a["x"] - 64) <= 1)

    def test_gif_keeps_transparency(self):
        f = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        f.paste((255, 0, 0, 255), (4, 4, 12, 12))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "a.gif"
            sheet.save_gif([f, f.transpose(Image.FLIP_LEFT_RIGHT)], path, fps=10, scale=2)
            with Image.open(path) as g:
                self.assertEqual(g.size, (32, 32))
                self.assertIn("transparency", g.info)

    def test_pack_rejects_mixed_sizes(self):
        with self.assertRaises(ValueError):
            sheet.pack([Image.new("RGBA", (8, 8)), Image.new("RGBA", (9, 8))], 2)

    def test_config_errors(self):
        good = {"checkpoint": "x.safetensors",
                "jobs": [{"name": "a", "mode": "prompt", "prompt": "knight",
                          "actions": [{"name": "idle", "prompt": "idle", "frames": 4}]}]}
        config.parse(json.loads(json.dumps(good)))
        bad = json.loads(json.dumps(good))
        del bad["checkpoint"]
        with self.assertRaisesRegex(config.ConfigError, "checkpoint"):
            config.parse(bad)
        bad = json.loads(json.dumps(good))
        bad["jobs"][0]["name"] = "có dấu"
        with self.assertRaisesRegex(config.ConfigError, "name"):
            config.parse(bad)

    def test_explain_validation_error(self):
        body = json.dumps({"error": {"message": "Prompt outputs failed validation"},
                           "node_errors": {"4": {"class_type": "CheckpointLoaderSimple", "errors": [
                               {"message": "Value not in list", "details": "ckpt_name: 'x'"}]}}})
        msg = comfy._explain(body)
        self.assertIn("CheckpointLoaderSimple", msg)
        self.assertIn("ckpt_name", msg)

    @staticmethod
    def _cfg():
        return config.parse({"engine": "video", "checkpoint": "x", "frame_size": 128, "char_height": 64, "palette": 16,
                             "jobs": [{"name": "a", "prompt": "p",
                                       "actions": [{"name": "i", "prompt": "i"}]}]})


class EndToEndTests(unittest.TestCase):
    def test_batch_resume_and_error_isolation(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            chibi, _ = pipelines.prepare_character(fake_character(7), UnitTests._cfg())
            chibi.save(d / "chibi.png")
            jobs = {
                "comfy_url": fake.url, "engine": "video",
                "checkpoint": "pixel_test.safetensors",
                "seed": -1, "frame_size": 128, "char_height": 64, "palette": 32, "fps": 10,
                "jobs": [
                    {"name": "hac_ronin", "mode": "prompt", "prompt": "dark samurai, red eyes",
                     "actions": [
                         {"name": "idle", "prompt": "idle breathing", "frames": 8},
                         {"name": "attack", "prompt": "slashing katana", "frames": 6, "columns": 3},
                         {"name": "broken", "prompt": "FAIL_ME", "frames": 4},
                     ]},
                    {"name": "chibi", "mode": "image", "character_image": "chibi.png",
                     "actions": [{"name": "jump", "prompt": "jumping up", "frames": 6,
                                  "align": "track"}]},
                ],
            }
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            out = d / "out"

            code = run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(out)])
            self.assertEqual(code, 1)  # 1 action cố tình lỗi

            for job, act, n, cols in [("hac_ronin", "idle", 8, 8), ("hac_ronin", "attack", 6, 3),
                                      ("chibi", "jump", 6, 6)]:
                adir = out / job / act
                m = json.loads((adir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((m["frameWidth"], m["frameHeight"], m["frameCount"]), (128, 128, n))
                with Image.open(adir / "spritesheet.png") as s:
                    self.assertEqual(s.size, (cols * 128, -(-n // cols) * 128))
                    self.assertEqual(s.mode, "RGBA")
                with Image.open(adir / "preview.gif") as g:
                    self.assertEqual(g.n_frames, n)
                self.assertIn("seed", m["meta"])

            progress = json.loads((out / "hac_ronin" / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["idle"]["status"], "done")
            self.assertEqual(progress["broken"]["status"], "error")
            self.assertIn("out of memory", progress["broken"]["error"])
            self.assertTrue((out / "hac_ronin" / "character.png").exists())
            self.assertTrue((out / "run_log.txt").exists())

            # Chạy lại: chỉ action lỗi được thử lại, không gen lại nhân vật
            before = len(fake.prompts)
            code = run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(out)])
            self.assertEqual(code, 1)
            new = fake.prompts[before:]
            self.assertEqual(len(new), 1)
            self.assertIn("UNETLoader", {n["class_type"] for n in new[0].values()})
            self.assertIn("bỏ qua", (out / "run_log.txt").read_text(encoding="utf-8"))

            # Bỏ action lỗi -> batch sạch
            jobs["jobs"][0]["actions"].pop()
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            self.assertEqual(run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(out)]), 0)
            _close_log_handlers()  # Windows không xoá được thư mục tạm khi file log còn mở

    def test_stale_output_is_redone_and_changes_are_detected(self):
        # Lỗi thật: chạy lại trong cùng session -> output cũ (code cũ) bị "bỏ qua" hết
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d, out = Path(d), Path(d) / "o"
            acts = [{"name": "idle", "prompt": "idle", "frames": 4},
                    {"name": "walk", "prompt": "walk", "frames": 4}]
            jobs = {"comfy_url": fake.url, "checkpoint": "auto", "debug": False, "engine": "video",
                    "jobs": [{"name": "r", "prompt": "samurai", "actions": acts}]}
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            run = lambda: run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(out)])
            # Giả lập output của bản cũ: progress "done" và character.json không có fingerprint
            (out / "r" / "idle").mkdir(parents=True)
            Image.new("RGBA", (40, 64), (9, 9, 9, 255)).save(out / "r" / "character.png")
            (out / "r" / "character.json").write_text("{}", encoding="utf-8")
            Image.new("RGBA", (8, 8)).save(out / "r" / "idle" / "spritesheet.png")
            (out / "r" / "progress.json").write_text(json.dumps({"idle": "done"}), encoding="utf-8")

            self.assertEqual(run(), 0)
            classes = [{n["class_type"] for n in g.values()} for g in fake.prompts]
            self.assertEqual(sum("CheckpointLoaderSimple" in c for c in classes), 1)  # nhân vật tạo lại
            self.assertEqual(sum("UNETLoader" in c for c in classes), 2)  # cả 2 action làm lại

            n = len(fake.prompts)
            self.assertEqual(run(), 0)  # không đổi gì -> không gọi GPU
            self.assertEqual(len(fake.prompts), n)

            jobs["jobs"][0]["actions"][1]["prompt"] = "walking slowly"  # chỉ sửa walk
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            self.assertEqual(run(), 0)
            redone = fake.prompts[n:]
            self.assertEqual(len(redone), 1)
            self.assertIn("walking slowly", json.dumps(redone[0]))
            _close_log_handlers()

    def test_two_gpus_split_actions(self):
        with FakeComfy() as gpu0, FakeComfy() as gpu1, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            run_batch.STAGGER_S = 0.3
            acts = [{"name": f"a{i}", "prompt": "idle", "frames": 4} for i in range(6)]
            acts[2]["prompt"] = "FAIL_ME"
            jobs = {"checkpoint": "pixel_test.safetensors", "debug": False, "engine": "video",
                    "jobs": [{"name": "r", "prompt": "samurai", "actions": acts}]}
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            urls = f"{gpu0.url},http://127.0.0.1:9,{gpu1.url}"  # URL giữa không chạy -> bị bỏ qua
            code = run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o"),
                                   "--comfy-url", urls])
            self.assertEqual(code, 1)
            wan = lambda f: sum("UNETLoader" in {n["class_type"] for n in g.values()} for g in f.prompts)
            self.assertEqual(wan(gpu0) + wan(gpu1), 6)
            self.assertGreater(wan(gpu1), 0)  # GPU thứ 2 thật sự nhận việc
            progress = json.loads((d / "o/r/progress.json").read_text(encoding="utf-8"))
            self.assertEqual([progress[f"a{i}"]["status"] for i in range(6)],
                             ["done", "done", "error", "done", "done", "done"])
            self.assertIn("gpu1", (d / "o/run_log.txt").read_text(encoding="utf-8"))
            _close_log_handlers()

    def test_wrong_background_retries_then_cuts_cleanly(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            jobs = {"comfy_url": fake.url, "checkpoint": "pixel_test.safetensors", "seed": 10, "engine": "video",
                    "jobs": [{"name": "dark", "prompt": "BLACKBG dark samurai",
                              "actions": [{"name": "idle", "prompt": "idle", "frames": 4}]}]}
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            self.assertEqual(run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o")]), 0)
            sd = [g for g in fake.prompts if "CheckpointLoaderSimple" in {n["class_type"] for n in g.values()}]
            self.assertEqual(len(sd), pipelines.CHAR_ATTEMPTS)  # nền sai -> gen lại đủ số lần
            info = json.loads((d / "o/dark/character.json").read_text(encoding="utf-8"))
            self.assertEqual([t["seed"] for t in info["attempts"]], [10, 11, 12])
            native = Image.open(d / "o/dark/character.png")
            self.assertEqual(native.height, 64)
            self.assertLess(pixel.hole_ratio(native), 0.01)
            self.assertTrue((d / "o/dark/idle/debug/input.png").exists())
            with Image.open(d / "o/dark/idle/debug/clip.gif") as g:
                self.assertEqual(g.n_frames, 49)
            _close_log_handlers()

    def test_user_sprite_by_filename_keeps_native_pixels(self):
        # Trường hợp có sẵn sprite mẫu: upload ảnh (phóng x4), jobs.json chỉ ghi tên file
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            cfg48 = config.parse({"engine": "video", "checkpoint": "x", "char_height": 48, "palette": 40, "jobs": [
                {"name": "a", "prompt": "p", "actions": [{"name": "i", "prompt": "i"}]}]})
            native, _ = pipelines.prepare_character(fake_character(11), cfg48)
            (d / "kaggle_input" / "datasets" / "me" / "sprites").mkdir(parents=True)
            sprite_path = d / "kaggle_input" / "datasets" / "me" / "sprites" / "hero.png"
            pixel.upscale(native, 4).save(sprite_path)
            old_roots, run_batch.SEARCH_ROOTS = run_batch.SEARCH_ROOTS, [d / "kaggle_input"]
            try:
                jobs = {"comfy_url": fake.url, "frame_size": 160, "debug": False, "engine": "video",
                        "jobs": [{"name": "hero", "mode": "image", "character_image": "hero.png",
                                  "prompt": "a knight with a red cape",
                                  "actions": [{"name": "idle", "prompt": "breathing", "frames": 4}]}]}
                (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
                run = lambda: run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o")])
                self.assertEqual(run(), 0)
                char = Image.open(d / "o/hero/character.png")
                self.assertEqual(char.size, native.size)  # x4 -> đúng pixel gốc, không ép về 80px
                colors = lambda im: {tuple(p) for p in np.asarray(im)[np.asarray(im)[:, :, 3] > 0][:, :3]}
                self.assertEqual(colors(char), colors(native))  # giữ nguyên màu của người dùng
                self.assertIn("a knight with a red cape", json.dumps(fake.prompts[-1]))
                self.assertFalse(any("CheckpointLoaderSimple" in json.dumps(g) for g in fake.prompts))

                n = len(fake.prompts)
                self.assertEqual(run(), 0)  # không đổi gì -> không chạy lại
                self.assertEqual(len(fake.prompts), n)
                flipped = pixel.upscale(native.transpose(Image.FLIP_LEFT_RIGHT), 4)
                flipped.save(sprite_path)  # upload bản sửa, cùng tên file
                self.assertEqual(run(), 0)
                self.assertEqual(len(fake.prompts), n + 1)  # nhận ra ảnh mới -> làm lại action
            finally:
                run_batch.SEARCH_ROOTS = old_roots
                _close_log_handlers()

    def test_character_cut_at_edge_is_regenerated(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            jobs = {"comfy_url": fake.url, "checkpoint": "pixel_test.safetensors", "seed": 20, "engine": "video",
                    "debug": False, "jobs": [{"name": "c", "prompt": "CUTFEET ninja",
                                              "actions": [{"name": "i", "prompt": "i", "frames": 2}]}]}
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            self.assertEqual(run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o")]), 0)
            info = json.loads((d / "o/c/character.json").read_text(encoding="utf-8"))
            self.assertEqual(len(info["attempts"]), pipelines.CHAR_ATTEMPTS)
            self.assertFalse(any(t["inside_frame"] for t in info["attempts"]))
            self.assertTrue(all(t["background_ok"] for t in info["attempts"]))
            self.assertIn("bị cắt", (d / "o/run_log.txt").read_text(encoding="utf-8"))
            _close_log_handlers()

    def test_example_jobs_parse_and_auto_checkpoint(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            for name in ("jobs.example.json", "jobs.bench.json"):
                cfg = config.load(ROOT / "kaggle" / name)
                self.assertEqual(cfg.checkpoint, "auto")
                problems = run_batch.preflight(comfy.ComfyClient(fake.url), cfg, Path(d), Path(d))
                self.assertEqual(problems, [])
                self.assertEqual(cfg.checkpoint_file, "pixel_test.safetensors")

    def test_preflight_reports_missing_model(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            cfg = config.parse({"checkpoint": "khong_co.safetensors", "engine": "video", "jobs": [
                {"name": "a", "prompt": "p", "actions": [{"name": "i", "prompt": "i"}]}]})
            problems = run_batch.preflight(comfy.ComfyClient(fake.url), cfg, Path(d), Path(d))
            self.assertTrue(any("khong_co.safetensors" in p for p in problems))


class PoseEngineTests(unittest.TestCase):
    def test_presets_are_grounded_and_inside_canvas(self):
        for name, preset in poses.PRESETS.items():
            for n in (1, 6, 8):
                keys = poses.sequence(name, n)
                self.assertEqual(len(keys), n)
                for key in keys:
                    pts = [p for p in poses.skeleton(key) if p is not None]
                    self.assertAlmostEqual(min(p[1] for p in pts), key["air"], places=6, msg=name)
                    for x, y in (q for q in poses.to_pixels(poses.skeleton(key), 512) if q):
                        self.assertTrue(8 <= x <= 504 and 8 <= y <= 504, f"{name}: ({x:.0f},{y:.0f})")

    def test_render_uses_controlnet_openpose_colors(self):
        img = np.asarray(poses.render(poses.NEUTRAL, 512)).reshape(-1, 3)
        colors = {tuple(c) for c in img.tolist()}
        self.assertIn((153, 0, 0), colors)        # cổ -> vai phải: (255,0,0) x 0.6
        self.assertIn((0, 153, 51), colors)       # cổ -> hông phải: (0,255,85) x 0.6
        self.assertGreater((img.sum(1) > 0).mean(), 0.003)

    def test_loop_has_no_duplicate_end_frame(self):
        walk = poses.sequence("walk", 8)
        self.assertNotEqual(walk[0], walk[-1])
        jump = poses.sequence("jump", 6)
        self.assertEqual(jump[0], poses.PRESETS["jump"]["keys"][0])
        self.assertEqual(jump[-1], poses.PRESETS["jump"]["keys"][-1])

    def test_config_pose_validation(self):
        base = {"checkpoint": "auto", "jobs": [{"name": "a", "prompt": "p", "actions": []}]}

        def parse(actions, **extra):
            raw = json.loads(json.dumps(base)); raw.update(extra)
            raw["jobs"][0]["actions"] = actions
            return config.parse(raw)

        with self.assertRaisesRegex(config.ConfigError, "attack"):
            parse([{"name": "slash2", "prompt": "x"}])  # tên lạ -> gợi ý danh sách tư thế
        cfg = parse([{"name": "slash2", "prompt": "x", "pose": "attack"},
                     {"name": "jump", "prompt": "x"}, {"name": "w", "prompt": "x", "engine": "video"}])
        acts = cfg.jobs[0].actions
        self.assertEqual((acts[0].engine, acts[0].pose, acts[0].loop), ("pose", "attack", False))
        self.assertEqual((acts[2].engine, acts[2].pose), ("video", None))
        with self.assertRaisesRegex(config.ConfigError, "engine"):
            parse([{"name": "idle", "prompt": "x"}], engine="gif")
        with self.assertRaisesRegex(config.ConfigError, "checkpoint"):  # ảnh mẫu + pose vẫn cần SD
            config.parse({"jobs": [{"name": "a", "mode": "image", "character_image": "a.png",
                                    "actions": [{"name": "idle", "prompt": "x"}]}]})

    def test_fill_defaults(self):
        graph = {"1": {"class_type": "IPAdapter", "inputs": {"weight_type": "standard", "weight": 0.8}},
                 "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "mine.safetensors"}}}
        info = {"IPAdapter": {"input": {"required": {
                    "weight": ["FLOAT", {"default": 1.0}],
                    "weight_type": [["linear", "ease in"]],
                    "start_at": ["FLOAT", {"default": 0.0}],
                    "embeds_scaling": ["COMBO", {"options": ["V only", "K+V"]}]}}},
                "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["other.safetensors"]]}}}}
        workflows.fill_defaults(graph, info)
        self.assertEqual(graph["1"]["inputs"]["weight"], 0.8)            # giá trị có sẵn giữ nguyên
        self.assertEqual(graph["1"]["inputs"]["weight_type"], "linear")  # enum đổi tên -> sửa
        self.assertEqual(graph["1"]["inputs"]["start_at"], 0.0)          # thiếu -> mặc định
        self.assertEqual(graph["1"]["inputs"]["embeds_scaling"], "V only")
        self.assertEqual(graph["2"]["inputs"]["ckpt_name"], "mine.safetensors")  # tên file: không đụng

    def _run(self, fake, d, actions, **extra):
        jobs = {"comfy_url": fake.url, "checkpoint": "pixel_test.safetensors", "seed": 5,
                "jobs": [{"name": "ronin", "prompt": "dark samurai", "actions": actions}], **extra}
        (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        return run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o")])

    def test_pose_engine_end_to_end(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            actions = [{"name": "idle", "prompt": "breathing", "frames": 8},
                       {"name": "attack", "prompt": "slashing", "frames": 6, "columns": 3},
                       {"name": "slash2", "prompt": "second slash", "frames": 4, "pose": "attack"},
                       {"name": "jump", "prompt": "jumping", "frames": 6}]
            self.assertEqual(self._run(fake, d, actions), 0)
            kinds = [{n["class_type"] for n in g.values()} for g in fake.prompts]
            self.assertTrue(all("ControlNetApplyAdvanced" in k for k in kinds))
            self.assertNotIn("IPAdapter", kinds[0])                      # nhân vật: chỉ khung xương
            self.assertTrue(all("IPAdapter" in k for k in kinds[1:]))     # frame: + giữ ngoại hình
            # Bảng tư thế: mỗi action MỘT ảnh SD (không phải mỗi frame một ảnh)
            self.assertEqual(len(fake.prompts), 1 + 4)
            self.assertTrue(all("SetLatentNoiseMask" in k for k in kinds[1:]))
            idle_graph = fake.prompts[1]
            init = fake.uploads[idle_graph["40"]["inputs"]["image"]]
            mask = np.asarray(fake.uploads[idle_graph["42"]["inputs"]["image"]].convert("L"))
            self.assertEqual(init.size, (1536, 1536))                    # 8 frame + ô mẫu = 3x3
            self.assertEqual(mask[256, 256], 0)                          # ô mẫu: giữ nguyên
            self.assertTrue((mask[512:, :] == 255).all() and (mask[:512, 512:] == 255).all())
            self.assertLess(np.asarray(init.crop((0, 0, 512, 512)).convert("L")).min(), 200)  # có nhân vật mẫu
            self.assertEqual(np.asarray(init.crop((512, 0, 1536, 1536)).convert("L")).min(), 255)
            anchors = set()
            for act, n, cols in (("idle", 8, 8), ("attack", 6, 3), ("slash2", 4, 4), ("jump", 6, 6)):
                m = json.loads((d / "o/ronin" / act / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((m["frameCount"], m["columns"], m["frameWidth"]), (n, cols, 128))
                self.assertEqual(m["meta"]["engine"], "pose")
                self.assertEqual((m["meta"]["layout"], m["meta"]["boards"]), ("grid", 1))
                anchors.add(json.dumps(m["anchor"]))
                self.assertTrue((d / "o/ronin" / act / "debug" / "poses.png").exists())
                self.assertTrue((d / "o/ronin" / act / "debug" / "board.png").exists())
            self.assertEqual(len(anchors), 1)  # mọi action cùng điểm neo chân
            self.assertFalse(json.loads((d / "o/ronin/jump/manifest.json").read_text())["loop"])
            # nhảy: frame giữa phải cao hơn frame đầu (chuyển động thật được giữ)
            sheet_img = Image.open(d / "o/ronin/jump/spritesheet.png")
            tops = [sheet_img.crop((i * 128, 0, i * 128 + 128, 128)).getbbox()[1] for i in range(6)]
            self.assertLess(min(tops[2:4]), tops[0] - 10)
            _close_log_handlers()

    def test_preflight_catches_empty_model_folder(self):
        # Lỗi thật: ComfyUI cũ chưa đọc thư mục controlnet -> "not in []" lúc đã chạy
        from tests import fake_comfy
        saved = fake_comfy.MODELS["controlnet"]
        fake_comfy.MODELS["controlnet"] = []
        try:
            with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                d = Path(d)
                self.assertEqual(self._run(fake, d, [{"name": "idle", "prompt": "x", "frames": 2}]), 2)
                self.assertEqual(fake.prompts, [])  # dừng trước khi tốn GPU
                log_text = (d / "o/run_log.txt").read_text(encoding="utf-8")
                self.assertIn("control_v11p_sd15_openpose_fp16", log_text)
                self.assertIn("cell 3", log_text)
                _close_log_handlers()
        finally:
            fake_comfy.MODELS["controlnet"] = saved

    def test_clean_figure_keeps_hero_and_sword_only(self):
        size, key = 512, poses.NEUTRAL
        unit = poses.standing_height_px(size)
        hip_x, ground_y = poses.ground_px(size)
        img = Image.new("RGB", (size, size), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        cx, cy, r = hip_x, ground_y - 0.6 * unit, 0.55 * unit
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 70, 20))       # đĩa tròn
        a = np.asarray(img).copy()
        body = poses.body_mask(key, size, 0.08)
        a[np.roll(body, int(0.6 * unit), axis=1)] = (40, 110, 40)              # nhân vật thừa
        a[body] = (40, 40, 110)                                                  # nhân vật chính
        img = Image.fromarray(a)
        draw = ImageDraw.Draw(img)
        wx, wy = poses.to_pixels(poses.skeleton(key), size)[4]
        draw.line([(wx, wy), (wx + 0.45 * unit, wy - 0.2 * unit)], fill=(200, 200, 215), width=5)  # kiếm
        draw.line([(40, ground_y + 6), (470, ground_y + 6)], fill=(30, 20, 20), width=5)        # vệt đất
        clean, stats = cleanup.clean_figure(matting.cutout_flat(img), key, (250, 250, 250))
        alpha = np.asarray(clean)[:, :, 3]
        neck = poses.to_pixels(poses.skeleton(key), size)[1]
        self.assertEqual(alpha[int(neck[1]), int(neck[0])], 255)                  # giữ nhân vật
        self.assertEqual(alpha[int(cy), int(cx - 0.9 * r)], 0)                    # bỏ đĩa tròn
        self.assertEqual(alpha[int(neck[1]), int(neck[0] + 0.6 * unit)], 0)      # bỏ người thừa
        tip = (int(wx + 0.42 * unit), int(wy - 0.187 * unit))
        self.assertEqual(alpha[tip[1] - 2:tip[1] + 3, tip[0] - 2:tip[0] + 3].max(), 255)  # giữ kiếm
        self.assertEqual(alpha[int(ground_y + 6), 50], 0)                         # bỏ vệt đất
        self.assertGreater(stats["peeled_px"], 0)

    def test_dirty_sd_output_gives_clean_single_character(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            jobs = {"comfy_url": fake.url, "checkpoint": "pixel_test.safetensors", "seed": 3,
                    "jobs": [{"name": "r", "prompt": "DISKBG EXTRAFIG samurai",
                              "actions": [{"name": "attack", "prompt": "slash", "frames": 4}]}]}
            (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            self.assertEqual(run_batch.main(["--jobs", str(d / "jobs.json"), "--out", str(d / "o")]), 0)
            info = json.loads((d / "o/r/character.json").read_text(encoding="utf-8"))
            self.assertEqual(len(info["attempts"]), pipelines.CHAR_ATTEMPTS)  # ảnh bẩn -> thử lại
            self.assertFalse(info["attempts"][0]["clean"])

            def disk_share(im):
                a = np.asarray(im.convert("RGBA")).astype(int)
                m = a[:, :, 3] > 0
                red = m & (a[:, :, 0] > 180) & (a[:, :, 1] < 110) & (a[:, :, 2] < 80)
                return red.sum() / max(1, m.sum())

            char = Image.open(d / "o/r/character.png")
            self.assertLess(disk_share(char), 0.08)  # còn sót mảnh kẹp giữa tay và thân (mắt đỏ ~0.5%)
            self.assertLess(char.width, char.height * 0.9)  # một nhân vật, không phải hai
            # Frame: lọc theo khung xương là lớp an toàn (ảnh giả còn vẽ đĩa ở MỌI frame dù
            # ảnh mẫu đã sạch). Nhân vật thừa phải biến mất ở mọi frame; đĩa ở đa số frame.
            sheet_img = Image.open(d / "o/r/attack/spritesheet.png")
            cells = [sheet_img.crop((i * 128, 0, i * 128 + 128, 128)) for i in range(4)]
            self.assertGreaterEqual(sum(disk_share(c) < 0.10 for c in cells), 3)
            for cell in cells:
                box = cell.getbbox()
                self.assertLess(box[3] - box[1], 128)
            _close_log_handlers()

    def test_grid_shape_and_chunks(self):
        from forge import grid
        self.assertEqual([grid.shape(n) for n in (2, 3, 4, 5, 7, 9)],
                         [(2, 1), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3)])
        self.assertEqual(grid.chunks(8, 9), [list(range(8))])
        self.assertEqual([len(c) for c in grid.chunks(10, 9)], [5, 5])  # chia đều, không 8 + 2
        self.assertEqual([len(c) for c in grid.chunks(16, 9)], [8, 8])
        self.assertEqual(sum(grid.chunks(13, 5), []), list(range(13)))

    def test_grid_failure_falls_back_to_single_frames(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            self.assertEqual(self._run(fake, d, [{"name": "walk", "prompt": "GRIDOOM walking",
                                                  "frames": 4}]), 0)
            # nhân vật + bảng lỗi + 4 frame vẽ lẻ
            self.assertEqual(len(fake.prompts), 1 + 1 + 4)
            m = json.loads((d / "o/ronin/walk/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((m["meta"]["layout"], m["frameCount"]), ("single", 4))
            self.assertIn("vẽ từng frame", (d / "o/run_log.txt").read_text(encoding="utf-8"))
            _close_log_handlers()

    def test_long_action_uses_several_boards(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            self.assertEqual(self._run(fake, d, [{"name": "run", "prompt": "running", "frames": 12}]), 0)
            self.assertEqual(len(fake.prompts), 1 + 2)  # 12 frame -> 2 bảng x 6 frame (4x2 ô)
            sizes = {fake.uploads[g["40"]["inputs"]["image"]].size for g in fake.prompts[1:]}
            self.assertEqual(sizes, {(2048, 1024)})
            self.assertTrue((d / "o/ronin/run/debug/board_2.png").exists())
            sheet_img = Image.open(d / "o/ronin/run/spritesheet.png")
            self.assertTrue(all(sheet_img.crop((i * 128, 0, i * 128 + 128, 128)).getbbox()
                                for i in range(12)))
            _close_log_handlers()

    def test_missing_grid_nodes_and_layout_config(self):
        with FakeComfy(grid=False) as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            self.assertEqual(self._run(fake, d, [{"name": "idle", "prompt": "x", "frames": 2}]), 0)
            self.assertFalse(any("SetLatentNoiseMask" in json.dumps(g) for g in fake.prompts))
            self.assertIn("thiếu node", (d / "o/run_log.txt").read_text(encoding="utf-8"))
            _close_log_handlers()
        base = {"checkpoint": "auto", "jobs": [{"name": "a", "prompt": "p",
                                                "actions": [{"name": "idle", "prompt": "x"}]}]}
        self.assertEqual(config.parse(base).pose["layout"], "grid")
        with self.assertRaisesRegex(config.ConfigError, "layout"):
            config.parse({**base, "pose": {"layout": "board"}})
        with self.assertRaisesRegex(config.ConfigError, "grid_max_cells"):
            config.parse({**base, "pose": {"grid_max_cells": 1}})

    def test_pose_engine_without_ipadapter_still_runs(self):
        with FakeComfy(ipadapter=False) as fake, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            self.assertEqual(self._run(fake, d, [{"name": "idle", "prompt": "x", "frames": 2}]), 0)
            self.assertFalse(any("IPAdapter" in json.dumps(g) for g in fake.prompts))
            self.assertIn("Tắt IP-Adapter", (d / "o/run_log.txt").read_text(encoding="utf-8"))
            _close_log_handlers()


if __name__ == "__main__":
    unittest.main()
