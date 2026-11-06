from __future__ import annotations

import json
import hashlib
import copy
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
import importlib.util
from pathlib import Path
from unittest import mock

from test_runtime_contracts import CONTROL, DESKTOP, PROJECT_DIR, load_module


class BatchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_module("batch_regression_engine", "def request_main_window_diarization_action")

    def test_same_name_and_legacy_outputs_when_resuming(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, out = root / "inputs", root / "outputs"
            src.mkdir()
            out.mkdir()
            names = ["会议.mp3", "会议.wav", "单段.ogg", "缺JSON.flac", "内容.mp4", "内容.mp4.wav"]
            items = [src / n for n in names]
            for path in items:
                path.touch()
            for name in ["会议.txt", "会议.json", "单段.txt", "单段.json", "缺JSON.txt", "内容.mp4.txt", "内容.mp4.json"]:
                (out / name).write_text("legacy", encoding="utf-8")
            pending = self.engine.pending_media_files(items[1:], src, out, write_json=True, allow_legacy=True)
            self.assertEqual(pending, [items[1], items[3], items[4], items[5]])
            self.assertEqual(self.engine.get_output_paths(Path(names[4]), out, src)[0].name, "内容.mp4.转写.txt")
            self.assertEqual((out / "内容.mp4.txt").read_text(), "legacy")
            self.assertNotEqual(self.engine.get_output_paths(Path(names[0]), out),
                                self.engine.get_output_paths(Path(names[1]), out))
            (out / "会议.wav.txt").touch()
            self.assertIn(items[1], self.engine.pending_media_files(items, src, out, write_json=True, allow_legacy=True))
            (out / "会议.wav.json").touch()
            self.assertNotIn(items[1], self.engine.pending_media_files(items, src, out, write_json=True, allow_legacy=True))
            self.assertEqual((out / "会议.txt").read_text(), "legacy")

    def test_finished_batch_never_imports_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.wav").touch()
            (root / "sample.wav.txt").write_text("finished")
            cmd = [sys.executable, "-B", "-c",
                   "import sys,runpy;sys.modules['torch']=None;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')",
                   str(Path(self.engine.__file__)), "--input-dir", str(root), "--output-dir", str(root),
                   "--skip-existing", "--disable-diarization", "--worker-result-file", str(root / "result.json")]
            res = subprocess.run(cmd, capture_output=True, timeout=20, cwd=PROJECT_DIR)
            self.assertEqual(res.returncode, 0, res.stderr)
            meta = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["total_files"], meta["status"]), (1, "ok"))
            self.assertEqual((root / "sample.wav.txt").read_text(), "finished")


class InterfaceRegressionTests(unittest.TestCase):
    def test_progress_survives_tail_window_partial_line_and_file_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.log"
            reader = CONTROL.TaskProgress()
            with path.open("w", encoding="utf-8") as log:
                log.write("待处理文件数: 1000\n")
                for i in range(999):
                    log.write("[步骤] 处理中\n" * 9 + f"[worker] [完成] f{i}\n")
                log.write("[worker] [跳")
            state = reader.read(path)
            self.assertEqual((state["total_files"], state["finished_files"]), (1000, 999))
            with path.open("a", encoding="utf-8") as log:
                log.write("过] last\n全部成功。\n")
            self.assertEqual(reader.read(path)["finished_files"], 1000)
            self.assertEqual(reader.read(path)["finished_files"], 1000)
            path.write_text("待处理文件数: 2\n", encoding="utf-8")
            self.assertEqual(reader.read(path)["finished_files"], 0)

    def test_desktop_can_load_with_no_web_framework(self):
        with mock.patch.dict(sys.modules, {"streamlit": None}):
            control = DESKTOP.load_console_module()
            self.assertTrue(callable(control.start_task))

    @unittest.skipUnless(os.name == "nt", "Windows process handles")
    def test_unverified_pid_cannot_be_controlled(self):
        with subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(60)"],
                              creationflags=subprocess.CREATE_NO_WINDOW) as child:
            try:
                stamp = CONTROL.get_process_creation_date(child.pid)
                self.assertTrue(stamp.startswith("ft:"))
                meta = dict(pid=child.pid, process_creation_date=stamp)
                self.assertEqual(CONTROL.tracked_task_state(meta), "running")
                for bad in ("", "ft:1"):
                    other = dict(meta, process_creation_date=bad)
                    self.assertFalse(CONTROL.set_tracked_task_paused(other, pause=True)[0])
                    self.assertFalse(CONTROL.kill_tracked_task(other)[0])
                with mock.patch.object(CONTROL, "_creation_ticks", return_value=0):
                    self.assertEqual(CONTROL.tracked_task_state(meta), "unknown")
                    self.assertFalse(CONTROL.kill_tracked_task(meta)[0])
                self.assertIsNone(child.poll())
            finally:
                child.terminate()
                child.wait(timeout=10)


class PortablePackageTests(unittest.TestCase):
    def test_only_the_bundled_package_contains_downloadable_models(self):
        spec = importlib.util.spec_from_file_location("package_builder", PROJECT_DIR / "build_portable.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "portable_runtime"
            def put(relative, data=b"fixture"):
                path = runtime / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            put("python/pythonw.exe")
            put("python/Lib/site-packages/torch/lib/runtime.dll")
            put("python/Lib/site-packages/torch/lib/dnnl.lib")
            put("python/Lib/site-packages/streamlit/__init__.py")
            put("python/Doc/index.html")
            put("python/MicrosoftWebDriver.exe")
            put("python/chromedriver.exe")
            put("python/LICENSE.txt")
            put("ffmpeg/bin/ffmpeg.exe")
            put("hf_home/token")
            put(".runtime/hf_token.dpapi")
            for repo, files in builder.MODEL_FILES.items():
                revision = next(v["revision"] for v in builder.MODELS.values() if "models--" + v["repo"].replace("/", "--") == repo)
                put(f"hf_cache/{repo}/refs/main", revision.encode())
                for name in files:
                    put(f"hf_cache/{repo}/snapshots/{revision}/{name}")
            for name in builder.SILERO_FILES:
                put(f"torch_cache/hub/snakers4_silero-vad_master/{name}")
            dest = root / "zip"
            fixture_models = copy.deepcopy(builder.MODELS)
            for spec in fixture_models.values():
                for expected in spec["files"].values():
                    expected.update(size=7, sha256=hashlib.sha256(b"fixture").hexdigest())
            with mock.patch.dict(builder.MODELS, fixture_models, clear=True):
                builder.build_packages(PROJECT_DIR, runtime, dest)
                next(runtime.glob("hf_cache/*/snapshots/*/model.bin")).write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    list(builder.model_files(runtime))
            with zipfile.ZipFile(dest / "Windows-按需下载模型.zip") as online, zipfile.ZipFile(dest / "Windows-含模型.zip") as full:
                a, b = set(online.namelist()), set(full.namelist())
                self.assertLess(a, b)
                self.assertIn("portable_runtime/python/LICENSE.txt", a)
                self.assertIn("portable_runtime/python/Lib/site-packages/torch/lib/runtime.dll", a)
                self.assertFalse(any(n.endswith((".lib", ".dpapi", "/token", "WebDriver.exe", "chromedriver.exe")) or "/Doc/" in n or "/streamlit/" in n for n in b))
                self.assertTrue(all(n.startswith(("portable_runtime/hf_cache/models--", "portable_runtime/torch_cache/")) for n in b - a))
                self.assertEqual(online.testzip(), None)
                self.assertEqual(full.testzip(), None)
