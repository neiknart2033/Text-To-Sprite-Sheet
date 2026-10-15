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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))

from forge import comfy, config, matting, pipelines, pixel, sheet, workflows  # noqa: E402
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
        raw = fake_character(4, black_bg=True)
        leg = (29 * 8 + 4, 45 * 8)  # một pixel ở chân màu (25,25,25)
        self.assertLess(matting.remove_background(raw, "auto").getpixel(leg)[3], 128)  # cách cũ: mất chân
        self.assertEqual(matting.cutout_flat(raw).getpixel(leg)[3], 255)
        new, _ = pipelines.prepare_character(raw, self._cfg())
        self.assertEqual(new.height, 64)
        self.assertLess(pixel.hole_ratio(new), 0.01)
        self.assertFalse(matting.matches_key(raw, "green")[0])
        self.assertTrue(matting.matches_key(fake_character(4), "green")[0])

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
        with tempfile.TemporaryDirectory() as d:
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
        return config.parse({"checkpoint": "x", "frame_size": 128, "char_height": 64, "palette": 16,
                             "jobs": [{"name": "a", "prompt": "p",
                                       "actions": [{"name": "i", "prompt": "i"}]}]})


class EndToEndTests(unittest.TestCase):
    def test_batch_resume_and_error_isolation(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory() as d:
            d = Path(d)
            chibi, _ = pipelines.prepare_character(fake_character(7), UnitTests._cfg())
            chibi.save(d / "chibi.png")
            jobs = {
                "comfy_url": fake.url,
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

    def test_two_gpus_split_actions(self):
        with FakeComfy() as gpu0, FakeComfy() as gpu1, tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run_batch.STAGGER_S = 0.3
            acts = [{"name": f"a{i}", "prompt": "idle", "frames": 4} for i in range(6)]
            acts[2]["prompt"] = "FAIL_ME"
            jobs = {"checkpoint": "pixel_test.safetensors", "debug": False,
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
        with FakeComfy() as fake, tempfile.TemporaryDirectory() as d:
            d = Path(d)
            jobs = {"comfy_url": fake.url, "checkpoint": "pixel_test.safetensors", "seed": 10,
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

    def test_example_jobs_parse_and_auto_checkpoint(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory() as d:
            for name in ("jobs.example.json", "jobs.bench.json"):
                cfg = config.load(ROOT / "kaggle" / name)
                self.assertEqual(cfg.checkpoint, "auto")
                problems = run_batch.preflight(comfy.ComfyClient(fake.url), cfg, Path(d), Path(d))
                self.assertEqual(problems, [])
                self.assertEqual(cfg.checkpoint, "pixel_test.safetensors")

    def test_preflight_reports_missing_model(self):
        with FakeComfy() as fake, tempfile.TemporaryDirectory() as d:
            cfg = config.parse({"checkpoint": "khong_co.safetensors", "jobs": [
                {"name": "a", "prompt": "p", "actions": [{"name": "i", "prompt": "i"}]}]})
            problems = run_batch.preflight(comfy.ComfyClient(fake.url), cfg, Path(d), Path(d))
            self.assertTrue(any("khong_co.safetensors" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
