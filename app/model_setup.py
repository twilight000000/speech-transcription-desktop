from __future__ import annotations

import os
import subprocess
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from model_resources import MODELS, local_model, default_asr_model


class ModelSetup(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.process = None
        self.log = None
        self.timer = None
        self.title("首次使用与模型准备")
        self.geometry("740x560")
        self.minsize(640, 500)
        self.transient(app)
        self.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="三步开始转写", font=("Microsoft YaHei UI", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text="① 选择音视频所在文件夹和结果保存位置。\n"
                  "② 准备当前设置需要的模型；已就绪时无需下载。\n"
                  "③ 返回主界面点击“开始转写”，完成后打开输出目录。",
                  justify="left").pack(anchor="w", pady=10)
        ttk.Label(frame, text="模型下载来自 Hugging Face 官方仓库；音视频在本机处理。\n"
                  "首次下载需要联网和磁盘空间，完成后可离线使用。", justify="left").pack(anchor="w")
        self.status = tk.StringVar()
        ttk.Label(frame, textvariable=self.status, justify="left", wraplength=680).pack(anchor="w", pady=12)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x")
        self.text = tk.Text(frame, height=8, wrap="word", state="disabled", font=("Microsoft YaHei UI", 9))
        self.text.pack(fill="both", expand=True, pady=10)
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        self.download = ttk.Button(actions, text="下载缺少的模型 / 重试", command=self.start)
        self.download.pack(side="left")
        self.cancel = ttk.Button(actions, text="取消下载", command=self.cancel_download, state="disabled")
        self.cancel.pack(side="left", padx=8)
        ttk.Button(actions, text="返回主界面", command=self.close).pack(side="right")
        links = ttk.Frame(frame)
        links.pack(fill="x", pady=(10, 0))
        ttk.Button(links, text="说话人模型授权页面", command=lambda: webbrowser.open(
            "https://huggingface.co/pyannote/speaker-diarization-community-1")).pack(side="left")
        ttk.Button(links, text="创建读取 Token", command=lambda: webbrowser.open(
            "https://huggingface.co/settings/tokens")).pack(side="left", padx=8)
        self.refresh_status()

    def required_models(self):
        device = "cuda" if self.app.device_mode_var.get() != "仅CPU" and self.app.control.torch.cuda.is_available() else "cpu"
        names = [default_asr_model(self.app.control.resolve_speed_profile(self.app.precision_var.get()), device)]
        if self.app.enable_diarization_var.get():
            names.append("diarization")
        return names

    def refresh_status(self):
        self.needed = [name for name in self.required_models() if local_model(name) is None]
        rows = []
        for name in self.required_models():
            mb = sum(f["size"] for f in MODELS[name]["files"].values()) / 1024**2
            rows.append(f"{'说话人区分' if name == 'diarization' else name}："
                        f"{'需要下载' if name in self.needed else '已就绪'}（约 {mb:.0f} MB）")
        rows.append("普通转写无需账号。仅在线下载说话人模型需要接受条款并填写 Token。")
        self.status.set("\n".join(rows))
        self.download.configure(state="normal" if self.needed else "disabled")

    def start(self):
        if self.process is not None:
            return
        self.refresh_status()
        if not self.needed:
            return
        token = self.app.hf_token_var.get().strip()
        if "diarization" in self.needed and not token:
            messagebox.showinfo("说话人模型", "请先打开模型授权页面，登录并接受条款；\n"
                                "在主界面填写读取 Token 后再下载。\n\n"
                                "也可以先关闭“区分不同说话人”，直接进行普通转写。", parent=self)
            return
        runtime = self.app.control.RUNTIME_DIR
        runtime.mkdir(parents=True, exist_ok=True)
        self.log_path = runtime / "model_download.log"
        env = dict(os.environ, HF_TOKEN=token if "diarization" in self.needed else "",
                   HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_ENDPOINT="https://huggingface.co",
                   HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONUTF8="1")
        env.pop("HF_HUB_OFFLINE", None)
        try:
            self.log = self.log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                [self.app.control.resolve_backend_python_executable(), "-u",
                 str(Path(__file__).with_name("model_resources.py")), *self.needed],
                stdout=self.log, stderr=subprocess.STDOUT, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            if self.log:
                self.log.close()
            messagebox.showerror("下载未启动", f"无法创建下载任务：{exc}", parent=self)
            return
        self.download.configure(state="disabled")
        self.cancel.configure(state="normal")
        self.progress.start(15)
        self.poll()

    def poll(self):
        self.timer = None
        try:
            data = self.log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", "\n".join(data.splitlines()[-14:]))
            self.text.configure(state="disabled")
            self.text.see("end")
        except OSError:
            pass
        if self.process.poll() is None:
            self.timer = self.after(500, self.poll)
            return
        success = self.process.returncode == 0
        self.finish()
        self.refresh_status()
        if success:
            self.status.set(self.status.get() + "\n模型已准备好，请返回主界面开始转写。")

    def finish(self):
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        self.process = None
        if self.log:
            self.log.close()
            self.log = None
        self.progress.stop()
        self.cancel.configure(state="disabled")

    def cancel_download(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                messagebox.showwarning("取消中", "下载进程暂未退出，请稍后重试。", parent=self)
                return
        self.finish()
        self.refresh_status()
        self.status.set(self.status.get() + "\n下载已取消；再次下载会复用已完成的文件。")

    def close(self):
        if self.process is not None:
            if not messagebox.askyesno("下载仍在进行", "取消下载并关闭引导？已下载的文件会保留。", parent=self):
                return False
            self.cancel_download()
            if self.process is not None:
                return False
        self.destroy()
        return True

