from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, marker: str):
    path = next(
        path
        for path in PROJECT_DIR.glob("*.py")
        if marker in path.read_text(encoding="utf-8-sig")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = load_module("control_under_test", "def get_task_start_issues")
DESKTOP = load_module("desktop_under_test", "class DesktopConsoleApp")


class StartPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        (self.input_dir / "sample.wav").write_bytes(b"test")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def command(self, *extra: str) -> list[str]:
        return [
            "python",
            "backend.py",
            "--input-dir",
            str(self.input_dir),
            "--output-dir",
            str(self.output_dir),
            "--device",
            "cpu",
            "--parallel-workers",
            "1",
            "--disable-diarization",
            *extra,
        ]

    def test_valid_pure_transcription_is_allowed(self) -> None:
        self.assertEqual(CONTROL.get_task_start_issues(self.command(), ""), [])

    def test_missing_input_is_blocked(self) -> None:
        cmd = self.command()
        cmd[cmd.index("--input-dir") + 1] = str(self.root / "missing")
        self.assertTrue(CONTROL.get_task_start_issues(cmd, ""))
        self.assertTrue(CONTROL.get_task_start_blocker(cmd, ""))

    def test_gpu_multiworker_is_blocked_before_start(self) -> None:
        cmd = self.command()
        cmd[cmd.index("--device") + 1] = "cuda"
        cmd[cmd.index("--parallel-workers") + 1] = "4"
        with mock.patch.object(CONTROL.torch.cuda, "is_available", return_value=True):
            issues = CONTROL.get_task_start_issues(cmd, "")
        self.assertTrue(any("worker" in issue for issue in issues))

    def test_auto_tune_may_resolve_gpu_multiworker(self) -> None:
        cmd = self.command("--auto-env-tune")
        cmd[cmd.index("--device") + 1] = "cuda"
        cmd[cmd.index("--parallel-workers") + 1] = "4"
        with mock.patch.object(CONTROL.torch.cuda, "is_available", return_value=True):
            self.assertEqual(CONTROL.get_task_start_issues(cmd, ""), [])

    def test_diarization_without_token_is_blocked(self) -> None:
        cmd = [item for item in self.command() if item != "--disable-diarization"]
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "empty-cache")}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            issues = CONTROL.get_task_start_issues(cmd, "")
        self.assertTrue(any("Token" in issue for issue in issues))

    def test_invalid_worker_value_has_one_clear_issue(self) -> None:
        cmd = self.command()
        cmd[cmd.index("--parallel-workers") + 1] = "invalid"
        issues = [issue for issue in CONTROL.get_task_start_issues(cmd, "") if "worker" in issue]
        self.assertEqual(len(issues), 1)


class DesktopLayoutTests(unittest.TestCase):
    def test_tabs_resize_scroll_and_keep_task_controls(self) -> None:
        def check_bounds(widget):
            for child in widget.winfo_children():
                if child.winfo_ismapped():
                    if child.winfo_manager() in ("grid", "pack"):
                        self.assertGreaterEqual(child.winfo_x(), -2)
                        self.assertLessEqual(child.winfo_x() + child.winfo_width(), widget.winfo_width() + 2)
                    check_bounds(child)

        with ExitStack() as stack:
            for name, value in (("load_user_settings", {}), ("load_hf_token", ""),
                                ("load_task_meta", None), ("load_last_diarization_abort", None),
                                ("save_user_settings", None), ("save_task_meta", None), ("save_hf_token", None)):
                stack.enter_context(mock.patch.object(CONTROL, name, return_value=value))
            stack.enter_context(mock.patch.dict(os.environ, {"HF_TOKEN": ""}))
            app = DESKTOP.DesktopConsoleApp(CONTROL)
            try:
                for size in ("1240x900", "980x680"):
                    app.geometry(size)
                    for index in range(3):
                        app.notebook.select(index)
                        app.update()
                        check_bounds(app)
                        self.assertTrue(app.start_button.winfo_ismapped())
                app.notebook.select(0)
                app.preview_notes_var.set("需要处理的配置提醒。\n" * 30)
                app.update()
                outer = app.nametowidget(app.notebook.select())
                canvas = next(w for w in outer.winfo_children() if isinstance(w, DESKTOP.tk.Canvas))
                inner = canvas.winfo_children()[0]
                self.assertGreaterEqual(inner.winfo_height(), inner.winfo_reqheight())
                app._scroll_tab(SimpleNamespace(widget=inner, delta=-120))
                self.assertGreater(canvas.yview()[0], 0, (canvas.cget('scrollregion'), canvas.winfo_height(), inner.winfo_height(), canvas.yview()))
                for running, verified in ((True, True), (True, False), (False, False)):
                    app.current_task_running = running
                    app.task_identity_verified = verified
                    app._update_buttons()
                    app._update_lock_state()
                    self.assertEqual(str(app.start_button['state']), "disabled" if running else "normal")
                    self.assertEqual(str(app.pause_button['state']), "normal" if verified else "disabled")
                    self.assertEqual(str(app.stop_button['state']), "normal" if verified else "disabled")
                    for widget, enabled, disabled in app.lockable_widgets:
                        self.assertEqual(str(widget.cget('state')), disabled if running else enabled)
                with mock.patch.object(CONTROL.os, "startfile") as open_folder:
                    app.output_dir_var.set(str(PROJECT_DIR))
                    app.open_output_dir()
                    open_folder.assert_called_once_with(str(PROJECT_DIR.resolve()))
            finally:
                app.destroy()


class ControllerLaunchTests(unittest.TestCase):
    def test_token_is_environment_only_and_result_channel_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            runtime_dir = root / "runtime"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "sample.wav").write_bytes(b"test")
            cmd = [
                "python",
                "backend.py",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--parallel-workers",
                "1",
                "--disable-diarization",
            ]
            fake_process = SimpleNamespace(pid=43210)
            with mock.patch.object(CONTROL, "RUNTIME_DIR", runtime_dir), \
                 mock.patch.object(CONTROL, "TASK_META_PATH", runtime_dir / "task_meta.json"), \
                 mock.patch.object(CONTROL, "get_process_creation_date", return_value="created"), \
                 mock.patch.object(CONTROL.subprocess, "Popen", return_value=fake_process) as popen:
                meta = CONTROL.start_task(cmd, "hf_test_secret")

            launched_cmd = [str(item) for item in popen.call_args.args[0]]
            launched_env = popen.call_args.kwargs["env"]
            self.assertNotIn("--hf-token", launched_cmd)
            self.assertNotIn("hf_test_secret", launched_cmd)
            self.assertEqual(launched_env.get("HF_TOKEN"), "hf_test_secret")
            self.assertIn("--worker-result-file", launched_cmd)
            self.assertIn("--interaction-dir", launched_cmd)
            self.assertNotIn("hf_test_secret", json.dumps(meta, ensure_ascii=False))
            self.assertTrue(Path(str(meta["interaction_dir"])).is_dir())


class InteractionChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = load_module(
            "backend_under_test",
            "def request_main_window_diarization_action",
        )

    def test_video_scan_resume_decode_and_output(self) -> None:
        backend = self.backend
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            nested = input_dir / "子目录"
            nested.mkdir(parents=True)
            output_dir = root / "output"
            extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".wav"}
            files = [nested / ("视频测试" + ext.upper()) for ext in extensions]
            for path in files:
                path.touch()
            (nested / "ignore.txt").touch()
            self.assertEqual(set(backend.iter_audio_files(input_dir)), set(files))
            self.assertEqual(CONTROL.count_audio_files(str(input_dir)), len(files))
            video = nested / "视频测试.MP4"
            cmd = ["python", "backend.py", "--input-dir", str(input_dir),
                   "--output-dir", str(output_dir), "--device", "cpu",
                   "--disable-diarization", "--parallel-workers", "1",
                   "--resume-from-file", str(video)]
            self.assertEqual(CONTROL.get_task_start_issues(cmd, ""), [])
            resumed, relative = backend.resolve_resume_from_file(sorted(files), input_dir, str(video))
            self.assertEqual(resumed[0], video)
            self.assertEqual(relative, video.relative_to(input_dir))
            ffmpeg = shutil.which("ffmpeg")
            self.assertIsNotNone(ffmpeg)
            subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=32x32:r=1:d=1", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=1", "-c:v", "mpeg4",
                            "-c:a", "aac", "-shortest", str(video)],
                           check=True, capture_output=True, timeout=30)
            audio = backend.load_audio(str(video))
            self.assertTrue(15000 <= len(audio) <= 18000)
            self.assertGreater(float(abs(audio).max()), 0.01)
            args = SimpleNamespace(skip_existing=False, write_json=True, language="en", heartbeat_sec=0)
            result = {"segments": [{"start": 0, "end": 1, "text": "video test"}]}
            legacy_txt = output_dir / "子目录" / "视频测试.txt"
            legacy_txt.parent.mkdir(parents=True)
            legacy_txt.write_text("existing audio result", encoding="utf-8")
            for process in (backend.process_one_file, backend.process_one_file_cooperative):
                extra = {"stage_executor": None} if process is backend.process_one_file_cooperative else {}
                with mock.patch.object(backend, "transcribe_with_fallback", return_value=result) as transcribe:
                    process(audio_file=video, input_root=input_dir, output_root=output_dir,
                            worker_name="test", asr_model=None, diarization_model=None,
                            args=args, batch_size=1, chunk_size=30, heartbeat_fp=None, **extra)
                    self.assertGreater(len(transcribe.call_args.kwargs["audio"]), 15000)
                txt, structured = backend.get_output_paths(relative, output_dir)
                self.assertEqual(txt.name, "视频测试.MP4.txt")
                self.assertIn("video test", txt.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(structured.read_text(encoding="utf-8")), result)
            self.assertEqual(legacy_txt.read_text(encoding="utf-8"), "existing audio result")
            self.assertEqual(backend.get_output_paths(Path("sample.wav"), output_dir)[0].name, "sample.wav.txt")
            silent = root / "silent.mp4"
            subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video),
                            "-c:v", "copy", "-an", str(silent)],
                           check=True, capture_output=True, timeout=30)
            with self.assertRaises(RuntimeError):
                backend.load_audio(str(silent))

    def test_backend_request_is_answered_by_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task_meta = {"interaction_dir": temp_dir}
            outcome: dict[str, object] = {}

            def backend_waiter() -> None:
                try:
                    outcome["value"] = self.backend.request_main_window_diarization_action(
                        worker_name="worker-test",
                        file_label="sample.wav",
                        reason="test-reason",
                        stage="test-stage",
                        args=SimpleNamespace(interaction_dir=temp_dir),
                    )
                except BaseException as exc:  # pragma: no cover - diagnostic capture
                    outcome["error"] = exc

            thread = threading.Thread(target=backend_waiter, daemon=True)
            thread.start()
            deadline = time.time() + 5
            requests: list[dict[str, object]] = []
            while time.time() < deadline and not requests:
                requests = CONTROL.list_pending_task_interactions(task_meta)
                time.sleep(0.05)

            self.assertEqual(len(requests), 1)
            ok, _ = CONTROL.respond_to_task_interaction(
                task_meta,
                str(requests[0]["request_id"]),
                "transcribe-only",
            )
            self.assertTrue(ok)
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcome.get("value"), "transcribe-only")
            self.assertNotIn("error", outcome)

    def test_desktop_routes_prompt_response_without_starting_another_process(self) -> None:
        responses: list[tuple[str, str]] = []
        fake = SimpleNamespace(
            current_task_meta={"interaction_dir": "unused"},
            deferred_interaction_until={},
            control=SimpleNamespace(
                list_pending_task_interactions=lambda _meta: [
                    {"request_id": "request-1", "summary": "test summary"}
                ],
                respond_to_task_interaction=lambda _meta, request_id, response: (
                    responses.append((request_id, response)) is None,
                    "ok",
                ),
            ),
        )
        with mock.patch.object(DESKTOP.messagebox, "askyesnocancel", return_value=True):
            with mock.patch.object(DESKTOP.messagebox, "showerror"):
                DESKTOP.DesktopConsoleApp.handle_pending_task_interaction(fake)
        self.assertEqual(responses, [("request-1", "transcribe-only")])


class BackendResultTests(unittest.TestCase):
    def test_system_exit_reason_is_saved_to_result_and_log(self) -> None:
        backend_path = next(
            path
            for path in PROJECT_DIR.glob("*.py")
            if "def request_main_window_diarization_action" in path.read_text(encoding="utf-8-sig")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "result.json"
            log_path = root / "run.log"
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(backend_path),
                    "--input-dir",
                    str(root / "missing-input"),
                    "--output-dir",
                    str(root / "output"),
                    "--device",
                    "cpu",
                    "--disable-diarization",
                    "--log-file",
                    str(log_path),
                    "--worker-result-file",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(result.get("status"), "exited")
            self.assertTrue(result.get("note"))
            self.assertIn(str(result["note"]), log_path.read_text(encoding="utf-8-sig"))

    def test_backend_help_does_not_offer_token_command_line_argument(self) -> None:
        backend_path = next(
            path
            for path in PROJECT_DIR.glob("*.py")
            if "def request_main_window_diarization_action" in path.read_text(encoding="utf-8-sig")
        )
        process = subprocess.run(
            [sys.executable, "-B", str(backend_path), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        self.assertEqual(process.returncode, 0)
        self.assertNotIn("--hf-token", process.stdout)


class ParallelAbortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = load_module(
            "backend_parallel_under_test",
            "def request_main_window_diarization_action",
        )

    def test_abort_from_one_worker_stops_the_whole_batch(self) -> None:
        class FakeProcess:
            def __init__(self, return_code: int | None) -> None:
                self.return_code = return_code
                self.terminated = False

            def poll(self):
                return self.return_code

            def terminate(self) -> None:
                self.terminated = True
                self.return_code = -15

            def wait(self, timeout=None):
                if self.return_code is None:
                    self.return_code = -15
                return self.return_code

            def kill(self) -> None:
                self.return_code = -9

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            runtime_dir = root / "runtime"
            input_dir.mkdir()
            output_dir.mkdir()
            files = [input_dir / "one.wav", input_dir / "two.wav"]
            for path in files:
                path.write_bytes(b"test")

            args = SimpleNamespace(
                speed_profile="balanced",
                compute_type="auto",
                threads=4,
                model=None,
                chunk_size=None,
                batch_size=None,
                beam_size=None,
                best_of=None,
                temperatures=None,
                heartbeat_sec=2,
                diarization_unavailable_policy="ask",
                language=None,
                write_json=False,
                skip_existing=False,
                auto_env_tune=False,
                disable_auto_env_tune=False,
                disable_diarization=False,
                log_file=None,
                heartbeat_log_file=None,
                interaction_dir=str(root / "interactions"),
                process_label="test",
                diarization_model="test-model",
                num_speakers=None,
                min_speakers=None,
                max_speakers=None,
            )
            processes: list[FakeProcess] = []

            def fake_popen(cmd, **_kwargs):
                result_path = Path(cmd[cmd.index("--worker-result-file") + 1])
                if not processes:
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps({"status": "aborted", "failed": [], "note": "stop-all"}),
                        encoding="utf-8",
                    )
                    process = FakeProcess(1)
                else:
                    process = FakeProcess(None)
                processes.append(process)
                return process

            with mock.patch.object(self.backend, "RUNTIME_DIR", runtime_dir), \
                 mock.patch.object(self.backend.subprocess, "Popen", side_effect=fake_popen):
                with self.assertRaises(self.backend.DiarizationAbortRequested):
                    self.backend.run_parallel_file_workers(
                        args=args,
                        audio_files=files,
                        input_root=input_dir,
                        output_root=output_dir,
                        resolved_device="cpu",
                        worker_count=2,
                        env={"cpu_count": 4},
                    )

            self.assertEqual(len(processes), 2)
            self.assertTrue(processes[1].terminated)


if __name__ == "__main__":
    unittest.main()
