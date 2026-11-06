from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from contextlib import ExitStack

import model_resources as resources
from test_runtime_contracts import CONTROL, DESKTOP


class ModelPreparationTests(unittest.TestCase):
    def test_partial_corrupt_and_complete_downloads_and_anonymous_public_access(self):
        data = b"real fixture model"
        spec = {"repo": "fixture/model", "revision": "1" * 40,
                "files": {"model.bin": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
        with TemporaryDirectory() as temp, mock.patch.dict(resources.MODELS, {"tiny": spec}, clear=True):
            root = Path(temp)
            path = root / "models--fixture--model" / "snapshots" / spec["revision"] / "model.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(data[:2])
            self.assertIsNone(resources.local_model("tiny", root))
            calls = []
            def download(*args, **kwargs):
                calls.append(kwargs)
                path.write_bytes(data)
                return str(path)
            with mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
                self.assertEqual(resources.prepare_model("tiny", "hf_must_not_be_sent", root), path.parent)
                self.assertIs(calls[0]["token"], False)
                self.assertEqual(calls[0]["revision"], spec["revision"])
                path.write_bytes(b"x" * len(data))
                self.assertIsNone(resources.local_model("tiny", root))
                resources.prepare_model("tiny", "", root)
                self.assertTrue(calls[-1]["force_download"])
                with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("offline cache must not connect")):
                    resources.prepare_model("tiny", "", root)

    def test_complete_local_diarization_needs_no_token_in_frontend(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "audio.wav").touch()
            cmd = ["python", "backend.py", "--input-dir", str(root), "--output-dir", str(root),
                   "--device", "cpu", "--parallel-workers", "1"]
            with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), \
                 mock.patch.object(resources, "local_model", return_value=root):
                self.assertEqual(CONTROL.get_task_start_issues(cmd, ""), [])
            with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), \
                 mock.patch.object(resources, "local_model", return_value=None):
                self.assertTrue(any("Token" in x for x in CONTROL.get_task_start_issues(cmd, "")))

    def test_setup_window_and_cancellation_preserve_downloaded_data(self):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            for name, value in (("load_user_settings", {}), ("load_hf_token", ""),
                                ("load_task_meta", None), ("load_last_diarization_abort", None)):
                stack.enter_context(mock.patch.object(CONTROL, name, return_value=value))
            stack.enter_context(mock.patch.object(CONTROL, "RUNTIME_DIR", root))
            stack.enter_context(mock.patch.dict(os.environ, {"HF_TOKEN": ""}))
            stack.enter_context(mock.patch("model_setup.local_model", return_value=None))
            app = DESKTOP.DesktopConsoleApp(CONTROL)
            try:
                app.show_model_setup()
                window = app.model_window
                app.update()
                self.assertIn("需要下载", window.status.get())
                self.assertTrue(window.download.winfo_ismapped())
                path = root / "partial"
                path.write_bytes(b"partial")
                process = mock.Mock()
                process.poll.return_value = None
                with mock.patch("model_setup.subprocess.Popen", return_value=process):
                    window.start()
                self.assertIs(window.process, process)
                window.cancel_download()
                process.terminate.assert_called_once()
                process.wait.assert_called_once()
                self.assertEqual(path.read_bytes(), b"partial")
                self.assertIsNone(window.process)
                self.assertEqual(str(window.download.cget("state")), "normal")
            finally:
                app.destroy()

