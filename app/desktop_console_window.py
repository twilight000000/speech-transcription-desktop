from __future__ import annotations

import importlib.util
import sys
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from model_resources import diarization_available, local_model
from model_setup import ModelSetup

PROJECT_DIR = Path(__file__).resolve().parent
CONSOLE_SCRIPT_PATH = PROJECT_DIR / "控制台.py"
WINDOW_TITLE = "语音转文字桌面工具 · v0.1.0-beta.1"
USER_GUIDE_TEXT = """快速开始

1. 在“输入音视频目录”选择需要处理的文件夹，可混放音频和视频。
2. 在“输出目录”选择结果保存位置。
3. 选择识别语言、精度和硬件模式；不确定时保持推荐值。
4. 只需要文字时，关闭“区分不同说话人”。
5. 点击“首次使用 / 模型”，下载当前设置缺少的模型。\n   需要区分不同说话人时，开启相应选项，再通过“首次使用 / 模型”检查模型。
6. 点击“开始转写”，随后可在“运行日志”页查看进度。

视频处理

• 支持 MP4、MKV、MOV、AVI、WebM 等常见视频格式，自动读取音轨，无需预先转换。
• 只转写视频中的语音，不识别画面或画面字幕；视频需要有可解码的音轨。
• 结果保留原扩展名，例如“会议.wav.txt”和“会议.mp4.txt”。同名的不同音视频各自保存。

说话人分离与 Token

• 仅在线下载说话人模型需要 Token；含模型完整版本地运行和普通转写均不需要账号。
• Token 会使用当前 Windows 用户密钥加密记忆，不会写入普通设置、任务快照或日志。
• 清空 Token 输入框后关闭窗口，可删除已经保存的 Token。
• 模型缺失时，点击“首次使用 / 模型”，按提示下载；可取消后重试。
• Hugging Face 下载需先接受模型条款，再创建有读取权限的 Token。

高级设置中的硬件模式与并行

• 单个音频，尤其是长音频：优先选择“仅GPU”。
• 多文件且需要说话人分离：可以考虑“CPU+GPU协同”。
• 没有可用 CUDA 时：选择“仅CPU”。
• 并行文件数 设为 0 表示自动。
• 多 worker 主要用于 CPU 批处理；单 GPU 或 CPU+GPU 协同模式不要设置多个 worker。

精度与占用

• 高精度：使用更大的模型或更充分的搜索，通常更慢；实际效果依音频而定。
• 均衡：通常推荐。
• 高速：处理更快，准确率可能略有下降。
• 低占用：适合后台运行。
• 性能拉满：提高线程、batch 和 chunk，占用更多资源。
• 高占用优化只有显式开启后才会生效。

输出与续跑（恢复文件在高级设置中）

• TXT：带时间戳、说话人标签和识别文字。
• JSON：结构化识别结果；可通过“同时导出 JSON”控制。
• “跳过已有结果”适合中断后继续批量任务。
• “从指定文件恢复”可从输入目录中的某个音频或视频继续处理。

任务控制与日志

• “暂停”会暂停当前后端进程，“继续”恢复执行。
• “停止”会结束当前任务，请确认输出是否已经写完。
• 关闭控制台窗口不会自动停止仍在运行的后端，关闭前程序会询问确认。
• 选择“运行时询问”后，说话人分离遇到问题会在主窗口弹出选择；若主窗口已关闭，重新打开即可继续处理该请求。
• 普通日志记录处理步骤和结果；心跳日志用于确认长步骤仍在运行。
• 日志过滤只影响窗口显示，不会修改日志文件。

常见情况

• 说话人模型下载提示未授权：先接受模型条款，再核对 Token；也可以关闭说话人区分。
• 提示 CUDA 不可用：改用仅CPU，或检查显卡驱动和 CUDA 环境。
• 单文件 GPU 任务提示 worker 不兼容：把并行文件数 改为 0 或 1。
• 后端已结束但日志没有完整收尾：查看普通日志最后内容，并检查当前配置提示。
"""


def show_error_dialog(message: str) -> None:
    root: tk.Tk | None = None
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(WINDOW_TITLE, message, parent=root)
    except Exception:
        sys.stderr.write(f"{message}\n")
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def load_console_module() -> Any:
    spec = importlib.util.spec_from_file_location("local_console_backend", CONSOLE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载控制台脚本: {CONSOLE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DesktopConsoleApp(tk.Tk):
    def __init__(self, control: Any) -> None:
        super().__init__()
        self.control = control
        self.title(WINDOW_TITLE)
        self.geometry(f"{min(1240, self.winfo_screenwidth() - 80)}x{min(900, self.winfo_screenheight() - 100)}")
        self.minsize(980, 680)
        self._configure_theme()

        self.lockable_widgets: list[tuple[tk.Widget, str, str]] = []
        self.current_task_meta: dict[str, Any] | None = None
        self.current_task_result: dict[str, Any] | None = None
        self.current_task_running = False
        self.task_identity_verified = False
        self.progress_reader = self.control.TaskProgress()
        self.recommendation_key = None
        self.current_task_paused = False
        self.current_abort_meta: dict[str, Any] | None = None
        self.synced_running_pid = 0
        self.refresh_after_id: str | None = None
        self.current_recommended_mode = ""
        self.current_recommendation_text = ""
        self.last_normal_log_text = ""
        self.last_heartbeat_log_text = ""
        self.suspend_preview_updates = False
        self.language_display_to_code: dict[str, str] = {}
        self.language_list_labels: list[str] = []
        self.model_window = None
        self.help_window: tk.Toplevel | None = None
        self.deferred_interaction_until: dict[str, float] = {}

        self._init_vars()
        self._build_ui()
        self._bind_events()
        self._refresh_hardware()
        self.refresh_runtime(force=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_vars(self) -> None:
        default_mode = "仅GPU" if self.control.torch.cuda.is_available() else "仅CPU"

        self.input_dir_var = tk.StringVar(value=self.control.DEFAULT_INPUT_DIR)
        self.output_dir_var = tk.StringVar(value=self.control.DEFAULT_OUTPUT_DIR)
        self.resume_from_file_var = tk.StringVar(value="")
        self.language_search_var = tk.StringVar(value="")
        self.language_code_var = tk.StringVar(value="")
        self.language_display_var = tk.StringVar(value="自动检测")
        self.precision_var = tk.StringVar(value="均衡")
        self.device_mode_var = tk.StringVar(value=default_mode)
        self.parallel_workers_var = tk.StringVar(value="0")
        self.utilization_var = tk.StringVar(value="均衡")
        self.heartbeat_var = tk.StringVar(value="15")
        self.diarization_policy_var = tk.StringVar(value="运行时询问")
        self.hf_token_var = tk.StringVar(value="")

        self.auto_env_tune_var = tk.BooleanVar(value=False)
        self.maximize_utilization_var = tk.BooleanVar(value=False)
        self.enable_diarization_var = tk.BooleanVar(value=False)
        self.write_json_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=True)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.refresh_interval_var = tk.StringVar(value="2")
        self.log_tail_lines_var = tk.StringVar(value="500")
        self.log_filter_var = tk.StringVar(value="")

        self.pid_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="空闲")
        self.progress_text_var = tk.StringVar(value="0/?")
        self.fail_count_var = tk.StringVar(value="0")
        self.stage_var = tk.StringVar(value="等待启动")
        self.step_var = tk.StringVar(value="-")
        self.summary_var = tk.StringVar(value="当前没有已登记任务。")
        self.status_note_var = tk.StringVar(value="任务参数空闲，可直接修改。")

        self.hardware_cpu_var = tk.StringVar(value="-")
        self.hardware_cuda_var = tk.StringVar(value="-")
        self.hardware_gpu_count_var = tk.StringVar(value="-")
        self.hardware_gpu_name_var = tk.StringVar(value="-")

        self.recommendation_var = tk.StringVar(value="模式建议加载中...")
        self.preview_notes_var = tk.StringVar(value="设置就绪，点击“开始转写”运行；如参数不兼容，启动前会提示。")
        self.log_path_var = tk.StringVar(value="-")
        self.heartbeat_log_path_var = tk.StringVar(value="-")
        self.snapshot_var = tk.StringVar(value="当前没有任务配置快照。")
        self.abort_time_var = tk.StringVar(value="-")
        self.abort_file_var = tk.StringVar(value="-")
        self.abort_stage_var = tk.StringVar(value="-")
        self.abort_reason_var = tk.StringVar(value="暂无已记录的中断恢复点。")
        self._load_persisted_settings()

    def _load_persisted_settings(self) -> None:
        try:
            settings = self.control.load_user_settings()
        except Exception:
            settings = {}
        if not isinstance(settings, dict):
            return

        string_map: dict[str, tk.StringVar] = {
            "task_input_dir": self.input_dir_var,
            "task_output_dir": self.output_dir_var,
            "task_resume_from_file": self.resume_from_file_var,
            "task_language_search": self.language_search_var,
            "task_language_code": self.language_code_var,
            "task_precision_mode": self.precision_var,
            "task_device_mode": self.device_mode_var,
            "task_parallel_workers": self.parallel_workers_var,
            "task_utilization": self.utilization_var,
            "task_heartbeat_sec": self.heartbeat_var,
            "task_diarization_unavailable_policy": self.diarization_policy_var,
            "ui_refresh_interval_sec": self.refresh_interval_var,
            "ui_log_tail_lines": self.log_tail_lines_var,
            "ui_log_filter_keyword": self.log_filter_var,
        }
        bool_map: dict[str, tk.BooleanVar] = {
            "task_auto_env_tune": self.auto_env_tune_var,
            "task_maximize_utilization": self.maximize_utilization_var,
            "task_enable_diarization": self.enable_diarization_var,
            "task_write_json": self.write_json_var,
            "task_skip_existing": self.skip_existing_var,
            "ui_auto_refresh": self.auto_refresh_var,
        }
        for key, var in string_map.items():
            if key in settings:
                var.set(str(settings[key]))
        for key, var in bool_map.items():
            if key in settings:
                var.set(bool(settings[key]))
        try:
            remembered_token = self.control.os.getenv("HF_TOKEN", "").strip() or self.control.load_hf_token()
        except Exception:
            remembered_token = self.control.os.getenv("HF_TOKEN", "").strip()
        if remembered_token:
            self.hf_token_var.set(remembered_token)

    def _configure_theme(self) -> None:
        self.option_add("*Font", ("Microsoft YaHei UI", 9))
        self.configure(background="#f3f6fb")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f3f6fb", foreground="#24334a", font=("Microsoft YaHei UI", 9))
        style.configure("TFrame", background="#f3f6fb")
        style.configure("TLabel", background="#f3f6fb")
        style.configure("TLabelframe", background="#f3f6fb", bordercolor="#dbe3ee", relief="solid")
        style.configure("TLabelframe.Label", foreground="#24334a", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=(12, 7), background="#ffffff", bordercolor="#ced8e6")
        style.map("TButton", background=[("active", "#e8effa")], foreground=[("disabled", "#8793a5")])
        style.configure("Accent.TButton", background="#2563eb", foreground="white", bordercolor="#2563eb", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#dbe3ee"), ("active", "#1d4ed8")], foreground=[("disabled", "#8793a5"), ("!disabled", "white")])
        style.configure("Danger.TButton", foreground="#b42332")
        style.configure("TEntry", padding=6, fieldbackground="#ffffff", bordercolor="#ced8e6")
        style.configure("TCombobox", padding=5, fieldbackground="#ffffff", arrowsize=14)
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff"), ("disabled", "#e8edf4")])
        style.configure("TCheckbutton", padding=(0, 4))
        style.configure("TNotebook", borderwidth=0, background="#f3f6fb")
        style.configure("TNotebook.Tab", padding=(20, 9), background="#e8edf5")
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#1d4ed8")])
        style.configure("Horizontal.TProgressbar", background="#2563eb", troughcolor="#e1e8f2", borderwidth=0, thickness=7)
        style.configure("Muted.TLabel", foreground="#5e6d82")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root = ttk.Frame(self, padding=14)
        root.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="音视频转写", font=("Microsoft YaHei UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="本地识别 · 批量处理 · 说话人区分", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.info_button = ttk.Button(header, text="使用帮助", command=self.show_help_window)
        self.info_button.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Button(header, text="首次使用 / 模型", command=self.show_model_setup).grid(row=0, column=1, rowspan=2, padx=8)

        dashboard = ttk.Frame(root, padding=(8, 4))
        dashboard.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for idx in range(4):
            dashboard.columnconfigure(idx, weight=1, uniform="metrics")
        self._metric(dashboard, "任务状态", self.status_var, 0)
        self._metric(dashboard, "已处理 / 总文件", self.progress_text_var, 1)
        self._metric(dashboard, "失败文件", self.fail_count_var, 2)
        self._metric(dashboard, "当前阶段", self.stage_var, 3)
        self.progress_bar = ttk.Progressbar(dashboard, mode="determinate", maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        self.summary_label = ttk.Label(dashboard, textvariable=self.summary_var, style="Muted.TLabel", wraplength=1000)
        self.summary_label.grid(row=2, column=0, columnspan=4, sticky="ew")
        dashboard.bind("<Configure>", lambda event: self.summary_label.configure(wraplength=max(100, event.width - 24)))

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        config_tab = self._create_scrollable_tab(self.notebook, "转写设置")
        advanced_tab = self._create_scrollable_tab(self.notebook, "高级设置")
        logs_tab = self._create_scrollable_tab(self.notebook, "运行日志")
        for tab in (config_tab, advanced_tab, logs_tab):
            tab.columnconfigure(0, weight=1)
        logs_tab.rowconfigure(1, weight=1)
        self._build_config_tab(config_tab, advanced_tab)
        self._build_logs_tab(logs_tab)

        actions = ttk.Frame(root, padding=(0, 14, 0, 0))
        actions.grid(row=3, column=0, sticky="ew")
        self.start_button = ttk.Button(actions, text="开始转写", style="Accent.TButton", command=self.on_start)
        self.pause_button = ttk.Button(actions, text="暂停", command=self.on_pause)
        self.resume_button = ttk.Button(actions, text="继续", command=self.on_resume)
        self.stop_button = ttk.Button(actions, text="停止", style="Danger.TButton", command=self.on_stop)
        for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
            button.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="打开输出目录", command=self.open_output_dir).pack(side="right")
        ttk.Button(actions, text="刷新状态", command=lambda: self.refresh_runtime(force=True)).pack(side="right", padx=8)
        self.bind("<MouseWheel>", self._scroll_tab)
        self.bind("<F1>", lambda _event: self.show_help_window())

    def open_output_dir(self) -> None:
        folder = Path(self.output_dir_var.get().strip()).expanduser()
        if not self.output_dir_var.get().strip() or not folder.is_dir():
            messagebox.showinfo(WINDOW_TITLE, "输出目录尚不存在，请先选择目录或完成一次转写。", parent=self)
            return
        try:
            self.control.os.startfile(str(folder.resolve()))
        except OSError as exc:
            messagebox.showerror(WINDOW_TITLE, f"无法打开输出目录：{exc}", parent=self)

    def show_model_setup(self) -> None:
        if self.current_task_running:
            messagebox.showinfo(WINDOW_TITLE, "请等待当前任务结束后再准备模型。", parent=self)
            return
        if self.model_window is not None and self.model_window.winfo_exists():
            self.model_window.lift()
            self.model_window.refresh_status()
            return
        self.model_window = ModelSetup(self)

    def show_help_window(self) -> None:
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.deiconify()
            self.help_window.lift()
            self.help_window.focus_force()
            return

        window = tk.Toplevel(self)
        self.help_window = window
        window.title("信息与教程")
        window.geometry("860x720")
        window.minsize(640, 480)
        window.transient(self)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        heading = ttk.Frame(window, padding=(18, 14, 18, 8))
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text="信息与教程",
            font=("Microsoft YaHei UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            heading,
            text="可在配置任务时保持此窗口打开",
            foreground="#666666",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.Frame(window, padding=(18, 0, 18, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        text_widget = tk.Text(
            body,
            wrap="word",
            padx=16,
            pady=14,
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
            spacing1=2,
            spacing3=4,
        )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_widget.insert("1.0", USER_GUIDE_TEXT)
        text_widget.configure(state="disabled")

        footer = ttk.Frame(window, padding=(18, 0, 18, 14))
        footer.grid(row=2, column=0, sticky="ew")
        ttk.Button(footer, text="关闭", command=self.close_help_window).pack(side="right")

        window.protocol("WM_DELETE_WINDOW", self.close_help_window)
        window.bind("<Escape>", lambda _event: self.close_help_window())
        window.focus_force()

    def close_help_window(self) -> None:
        if self.help_window is not None:
            try:
                self.help_window.destroy()
            except Exception:
                pass
        self.help_window = None

    def _build_config_tab(self, parent: ttk.Frame, advanced_parent: ttk.Frame) -> None:
        forms = ttk.Frame(parent)
        forms.grid(row=0, column=0, sticky="ew")
        forms.columnconfigure(0, weight=1, uniform="forms")
        forms.columnconfigure(1, weight=1, uniform="forms")
        basic = ttk.LabelFrame(forms, text="01  文件与语言", padding=16)
        options = ttk.LabelFrame(forms, text="02  转写选项", padding=16)
        basic.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        options.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        for card in (basic, options):
            card.columnconfigure(1, weight=1)
        self._path_row(basic, 0, "输入音视频目录", self.input_dir_var, self.browse_input_dir, "支持音频、MP4、MKV、MOV 等，可混合批量处理。")
        self._path_row(basic, 1, "输出目录", self.output_dir_var, self.browse_output_dir, "TXT 和 JSON 保存在此；视频结果如“会议.mp4.txt”。")
        self._entry_row(basic, 2, "搜索语言", self.language_search_var, "输入中文名或代码，例如：中文、zh、English。")
        self._language_list_row(basic, 3)
        ttk.Label(basic, textvariable=self.language_display_var, foreground="#1d4ed8").grid(row=8, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self._combo_row(options, 0, "识别精度", self.precision_var, ["均衡", "高精度", "高速"], "均衡适合日常；高精度更慢，高速更省时。")
        self._combo_row(options, 1, "硬件模式", self.device_mode_var, ["自动", "仅CPU", "仅GPU", "CPU+GPU协同"], "单文件优先仅GPU；多文件分工可在高级设置调整。")
        self._check_row(options, 2, "区分不同说话人", self.enable_diarization_var, "需要区分说话人时开启；本地模型已就绪时无需 Token。")
        self._entry_row(options, 3, "HF Token", self.hf_token_var, "仅在线下载说话人模型时需要；使用 Windows 用户密钥加密保存。", show="*")
        self._check_row(options, 4, "同时导出 JSON", self.write_json_var, "默认输出 TXT；JSON 适合后续整理和处理。")
        self._check_row(options, 5, "跳过已有结果", self.skip_existing_var, "已完成的文件无需重复处理。")
        notice = ttk.Label(parent, textvariable=self.preview_notes_var, foreground="#785418", wraplength=900, justify="left")
        notice.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        parent.bind("<Configure>", lambda event: notice.configure(wraplength=max(100, event.width - 24)), add="+")

        hardware = ttk.LabelFrame(advanced_parent, text="硬件信息", padding=12)
        hardware.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        hardware.columnconfigure(1, weight=1)
        for row, (title, variable) in enumerate((("CPU 线程", self.hardware_cpu_var), ("CUDA 可用", self.hardware_cuda_var), ("GPU 数量", self.hardware_gpu_count_var), ("GPU 信息", self.hardware_gpu_name_var))):
            ttk.Label(hardware, text=title, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 20), pady=3)
            ttk.Label(hardware, textvariable=variable).grid(row=row, column=1, sticky="w")
        adv = ttk.LabelFrame(advanced_parent, text="执行与恢复", padding=16)
        adv.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        adv.columnconfigure(1, weight=1)
        self._combo_row(adv, 0, "并行文件数", self.parallel_workers_var, ["0", "1", "2", "3", "4"], "0 为自动；多个 worker 主要用于 CPU 批处理。")
        self._combo_row(adv, 1, "资源占用", self.utilization_var, ["低占用", "均衡", "性能拉满"], "低占用适合后台运行；均衡适合日常使用。")
        self._check_row(adv, 2, "允许自动调整参数", self.auto_env_tune_var, "开启后允许修正不兼容参数；关闭时只校验，不自动修改。")
        self._check_row(adv, 3, "提高资源利用率", self.maximize_utilization_var, "增加线程和批处理规模，提高吞吐量。")
        self._combo_row(adv, 4, "分离失败时", self.diarization_policy_var, ["运行时询问", "自动仅转写", "立即中断任务"], "决定无法区分说话人时如何继续。")
        self._combo_row(adv, 5, "心跳间隔（秒）", self.heartbeat_var, ["5", "10", "15", "20", "30"], "长步骤定期记录运行状态。")
        self._entry_row(adv, 6, "从指定文件恢复", self.resume_from_file_var, "填写输入目录内的音视频路径；留空从头开始。")

        preview = ttk.LabelFrame(advanced_parent, text="执行预览与提醒", padding=12)
        preview.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        preview.columnconfigure(0, weight=1)
        ttk.Label(preview, textvariable=self.recommendation_var, wraplength=650, justify="left").grid(row=0, column=0, sticky="w")
        self.apply_recommended_button = ttk.Button(preview, text="应用当前推荐硬件模式", command=self.apply_recommended_mode)
        self.apply_recommended_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._register_lockable(self.apply_recommended_button)
        ttk.Label(preview, text="执行命令预览").grid(row=1, column=0, sticky="w", pady=(10, 4))
        self.command_text = tk.Text(preview, height=4, wrap="word", state="disabled", background="#ffffff", foreground="#24334a", relief="flat", padx=10, pady=8)
        self.command_text.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(preview, text="提醒").grid(row=3, column=0, sticky="w", pady=(10, 4))
        ttk.Label(preview, textvariable=self.preview_notes_var, wraplength=650, justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        abort = ttk.LabelFrame(advanced_parent, text="上次中断恢复点", padding=12)
        abort.grid(row=3, column=0, sticky="nsew")
        abort.columnconfigure(1, weight=1)
        ttk.Label(abort, text="时间").grid(row=0, column=0, sticky="w")
        ttk.Label(abort, textvariable=self.abort_time_var).grid(row=0, column=1, sticky="w")
        ttk.Label(abort, text="文件").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(abort, textvariable=self.abort_file_var, wraplength=680).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(abort, text="阶段").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(abort, textvariable=self.abort_stage_var).grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(abort, text="原因").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        ttk.Label(abort, textvariable=self.abort_reason_var, wraplength=680, justify="left").grid(row=3, column=1, sticky="w", pady=(6, 0))
        self.load_abort_button = ttk.Button(abort, text="加载恢复点到当前配置", command=self.load_abort_point)
        self.load_abort_button.grid(row=4, column=1, sticky="w", pady=(10, 0))

    def _build_logs_tab(self, parent: ttk.Frame) -> None:
        settings = ttk.Frame(parent, padding=(0, 0, 0, 12))
        settings.grid(row=0, column=0, sticky="ew")
        settings.columnconfigure(5, weight=1)
        ttk.Checkbutton(settings, text="自动刷新", variable=self.auto_refresh_var).grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="间隔（秒）").grid(row=0, column=1, padx=(12, 4))
        ttk.Combobox(settings, textvariable=self.refresh_interval_var, values=["1", "2", "5"], width=4, state="readonly").grid(row=0, column=2)
        ttk.Label(settings, text="显示行数").grid(row=0, column=3, padx=(12, 4))
        ttk.Combobox(settings, textvariable=self.log_tail_lines_var, values=["200", "500", "1000"], width=6, state="readonly").grid(row=0, column=4)
        ttk.Label(settings, text="筛选日志").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(settings, textvariable=self.log_filter_var).grid(row=1, column=1, columnspan=5, sticky="ew", pady=(10, 0))

        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")
        normal_tab = ttk.Frame(notebook)
        heartbeat_tab = ttk.Frame(notebook)
        info = ttk.Frame(notebook, padding=14)
        notebook.add(normal_tab, text="转写日志")
        notebook.add(heartbeat_tab, text="心跳日志")
        notebook.add(info, text="任务详情")
        self.normal_log_text = self._log_text(normal_tab)
        self.heartbeat_log_text = self._log_text(heartbeat_tab)
        info.columnconfigure(1, weight=1)
        for row, (title, variable) in enumerate((("任务 PID", self.pid_var), ("当前步骤", self.step_var), ("普通日志", self.log_path_var), ("心跳日志", self.heartbeat_log_path_var), ("配置快照", self.snapshot_var))):
            ttk.Label(info, text=title, style="Muted.TLabel").grid(row=row, column=0, sticky="nw", padx=(0, 14), pady=(0, 10))
            label = ttk.Label(info, textvariable=variable, wraplength=650, justify="left")
            label.grid(row=row, column=1, sticky="ew", pady=(0, 10))
            label.bind("<Configure>", lambda event, label=label: label.configure(wraplength=max(100, event.width)))

    def _create_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        notebook.add(outer, text=title)
        canvas = tk.Canvas(outer, highlightthickness=0, background="#f3f6fb")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(10, 14))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        return inner

    def _scroll_tab(self, event: Any) -> str | None:
        if isinstance(event.widget, (tk.Text, tk.Listbox, ttk.Combobox)):
            return None
        widget = event.widget
        while widget is not None:
            if isinstance(widget, tk.Canvas):
                if widget.yview() != (0.0, 1.0):
                    widget.yview_scroll(-int(event.delta / 120), "units")
                return "break"
            widget = getattr(widget, "master", None)
        return None

    def _metric(self, parent: ttk.Frame, title: str, value_var: tk.StringVar, column: int) -> None:
        box = ttk.Frame(parent, padding=(6, 2))
        box.grid(row=0, column=column, sticky="ew")
        ttk.Label(box, text=title, foreground="#666666").pack(anchor="w")
        ttk.Label(box, textvariable=value_var, font=("Microsoft YaHei UI", 11, "bold"), wraplength=205).pack(anchor="w", pady=(2, 0))

    def _form_hint(self, parent: ttk.Frame, row: int, text: str) -> None:
        hint = ttk.Label(parent, text=text, style="Muted.TLabel", wraplength=330, justify="left")
        hint.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 8))
        hint.bind("<Configure>", lambda event: hint.configure(wraplength=max(100, event.width)))

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, command: Any, help_text: str = "") -> None:
        row *= 2
        field = ttk.Frame(parent)
        field.grid(row=row, column=0, columnspan=3, sticky="ew")
        field.columnconfigure(0, weight=1)
        ttk.Label(field, text=label).grid(row=0, column=0, sticky="w", pady=(0, 5))
        entry = ttk.Entry(field, textvariable=variable, width=20)
        entry.grid(row=1, column=0, sticky="ew")
        button = ttk.Button(field, text="浏览", command=command, width=5)
        button.grid(row=1, column=1, padx=(8, 0))
        self._register_lockable(entry)
        self._register_lockable(button)
        self._form_hint(parent, row + 1, help_text)

    def _entry_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, help_text: str, *, show: str | None = None) -> None:
        row *= 2
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12))
        entry = ttk.Entry(parent, textvariable=variable, show=show or "", width=16)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew")
        self._register_lockable(entry)
        self._form_hint(parent, row + 1, help_text)

    def _language_list_row(self, parent: ttk.LabelFrame, row: int) -> None:
        row *= 2
        ttk.Label(parent, text="识别语言").grid(row=row, column=0, sticky="nw", padx=(0, 12))
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew")
        frame.columnconfigure(0, weight=1)
        self.language_listbox = tk.Listbox(frame, height=4, width=18, exportselection=False, relief="flat", highlightthickness=1, highlightbackground="#ced8e6", selectbackground="#2563eb", selectforeground="#ffffff", activestyle="dotbox")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.language_listbox.yview)
        self.language_listbox.configure(yscrollcommand=scrollbar.set)
        self.language_listbox.grid(row=0, column=0, sticky="ew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.language_listbox.bind("<<ListboxSelect>>", self._on_language_list_selected)
        self._register_lockable(self.language_listbox)
        self._form_hint(parent, row + 1, "选择一种语言，或保留自动检测。")

    def _combo_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, values: list[str], help_text: str = "", help_column: int = 3) -> ttk.Combobox:
        row *= 2
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=16)
        combo.grid(row=row, column=1, columnspan=2, sticky="ew")
        self._register_lockable(combo, enabled_state="readonly")
        self._form_hint(parent, row + 1, help_text)
        return combo

    def _check_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.BooleanVar, help_text: str = "") -> None:
        row *= 2
        widget = ttk.Checkbutton(parent, text=label, variable=variable)
        widget.grid(row=row, column=0, columnspan=3, sticky="w")
        self._register_lockable(widget)
        self._form_hint(parent, row + 1, help_text)

    def _log_text(self, parent: ttk.Frame) -> tk.Text:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        text = tk.Text(parent, wrap="none", state="disabled", height=14, background="#ffffff", foreground="#24334a", relief="flat", padx=12, pady=10, font=("Consolas", 10))
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return text

    def _register_lockable(self, widget: tk.Widget, enabled_state: str = "normal", disabled_state: str = "disabled") -> None:
        self.lockable_widgets.append((widget, enabled_state, disabled_state))

    def _bind_events(self) -> None:
        preview_vars = [
            self.input_dir_var, self.output_dir_var, self.resume_from_file_var, self.language_search_var, self.language_code_var,
            self.precision_var, self.device_mode_var, self.parallel_workers_var, self.utilization_var,
            self.heartbeat_var, self.diarization_policy_var, self.hf_token_var, self.auto_env_tune_var,
            self.maximize_utilization_var, self.enable_diarization_var, self.write_json_var, self.skip_existing_var,
        ]
        for var in preview_vars:
            var.trace_add("write", self._on_preview_var_changed)
        self.log_tail_lines_var.trace_add("write", self._on_log_setting_changed)
        self.log_filter_var.trace_add("write", self._on_log_setting_changed)
        self.auto_refresh_var.trace_add("write", self._on_refresh_setting_changed)
        self.refresh_interval_var.trace_add("write", self._on_refresh_setting_changed)
        self._refresh_language_options()

    def save_current_settings(self) -> None:
        settings: dict[str, Any] = {
            "task_input_dir": self.input_dir_var.get(),
            "task_output_dir": self.output_dir_var.get(),
            "task_resume_from_file": self.resume_from_file_var.get(),
            "task_language_search": self.language_search_var.get(),
            "task_language_code": self.language_code_var.get(),
            "task_precision_mode": self.precision_var.get(),
            "task_device_mode": self.device_mode_var.get(),
            "task_parallel_workers": self.parallel_workers_var.get(),
            "task_utilization": self.utilization_var.get(),
            "task_heartbeat_sec": self.heartbeat_var.get(),
            "task_diarization_unavailable_policy": self.diarization_policy_var.get(),
            "task_auto_env_tune": bool(self.auto_env_tune_var.get()),
            "task_maximize_utilization": bool(self.maximize_utilization_var.get()),
            "task_enable_diarization": bool(self.enable_diarization_var.get()),
            "task_write_json": bool(self.write_json_var.get()),
            "task_skip_existing": bool(self.skip_existing_var.get()),
            "ui_auto_refresh": bool(self.auto_refresh_var.get()),
            "ui_refresh_interval_sec": self.refresh_interval_var.get(),
            "ui_log_tail_lines": self.log_tail_lines_var.get(),
            "ui_log_filter_keyword": self.log_filter_var.get(),
        }
        try:
            self.control.save_user_settings(settings)
        except Exception:
            pass

    def _on_log_setting_changed(self, *_args: Any) -> None:
        self.save_current_settings()
        self.refresh_logs(force=True)

    def _on_refresh_setting_changed(self, *_args: Any) -> None:
        self.save_current_settings()
        self.schedule_refresh()

    def _refresh_hardware(self) -> None:
        torch_mod = self.control.torch
        self.hardware_cpu_var.set(str(self.control.os.cpu_count() or 0))
        self.hardware_cuda_var.set("是" if torch_mod.cuda.is_available() else "否")
        self.hardware_gpu_count_var.set(str(torch_mod.cuda.device_count() if torch_mod.cuda.is_available() else 0))
        if torch_mod.cuda.is_available():
            try:
                gpu_name = str(torch_mod.cuda.get_device_name(0))
                gpu_mem_gb = torch_mod.cuda.get_device_properties(0).total_memory / (1024**3)
                self.hardware_gpu_name_var.set(f"{gpu_name} / {gpu_mem_gb:.2f}GB")
            except Exception:
                self.hardware_gpu_name_var.set("已检测到 GPU")
        else:
            self.hardware_gpu_name_var.set("未检测到可用 CUDA")

    def refresh_abort_panel(self) -> None:
        self.current_abort_meta = self.control.load_last_diarization_abort()
        if not self.current_abort_meta:
            self.abort_time_var.set("-")
            self.abort_file_var.set("-")
            self.abort_stage_var.set("-")
            self.abort_reason_var.set("暂无已记录的中断恢复点。")
            self.load_abort_button.configure(state="disabled")
            return
        self.abort_time_var.set(str(self.current_abort_meta.get("timestamp", "-")))
        self.abort_file_var.set(str(self.current_abort_meta.get("resume_from_file", "-")))
        self.abort_stage_var.set(str(self.current_abort_meta.get("stage", "-")))
        self.abort_reason_var.set(str(self.current_abort_meta.get("reason", "-")))
        self.load_abort_button.configure(state="disabled" if self.current_task_running else "normal")

    def _on_preview_var_changed(self, *_args: Any) -> None:
        if self.suspend_preview_updates:
            return
        self.save_current_settings()
        self.refresh_preview()

    def _refresh_language_options(self) -> None:
        current_code = self.language_code_var.get().strip().lower()
        options = self.control.build_language_code_options(
            search_keyword=self.language_search_var.get().strip(),
            current_code=current_code,
        )
        if current_code and current_code not in options:
            options = [""] + [current_code] + [code for code in options if code and code != current_code]
        self.language_display_to_code = {"自动检测": ""}
        self.language_list_labels = ["自动检测"]
        for code in options:
            if not code:
                continue
            label = self.control.language_label(code)
            self.language_display_to_code[label] = code
            self.language_list_labels.append(label)

        display_value = self._language_display_from_code(current_code)
        if self.language_display_var.get() != display_value:
            self.language_display_var.set(display_value)
        self._sync_language_listbox_selection(display_value)

    def _language_display_from_code(self, code: str) -> str:
        code = code.strip().lower()
        if not code:
            return "自动检测"
        for label, mapped_code in self.language_display_to_code.items():
            if mapped_code == code:
                return label
        return self.control.language_label(code)

    def _sync_language_listbox_selection(self, display_value: str) -> None:
        if not hasattr(self, "language_listbox"):
            return
        self.language_listbox.delete(0, "end")
        for label in self.language_list_labels:
            self.language_listbox.insert("end", label)

        try:
            index = self.language_list_labels.index(display_value)
        except ValueError:
            index = 0
        self.language_listbox.selection_clear(0, "end")
        self.language_listbox.selection_set(index)
        self.language_listbox.activate(index)
        self.language_listbox.see(index)

    def _on_language_list_selected(self, _event: Any) -> None:
        if not hasattr(self, "language_listbox"):
            return
        selected_indexes = self.language_listbox.curselection()
        if not selected_indexes:
            return
        selected = str(self.language_listbox.get(selected_indexes[0])).strip()
        if self.language_display_var.get() != selected:
            self.language_display_var.set(selected)
        self.language_code_var.set(self.language_display_to_code.get(selected, ""))

    def refresh_preview(self) -> None:
        self._refresh_language_options()
        try:
            cmd = self.build_current_command()
        except Exception as exc:
            self._set_text(self.command_text, f"命令生成失败: {exc}")
            return

        preview = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
        self._set_text(self.command_text, preview)

        key = (self.input_dir_var.get().strip(), bool(self.enable_diarization_var.get()))
        if self.recommendation_key != key:
            self.current_recommended_mode, self.current_recommendation_text = self.control.recommend_device_mode(*key)
            self.recommendation_key = key
        self.recommendation_var.set(
            f"当前推荐：{self.current_recommended_mode}。{self.current_recommendation_text}"
        )

        notes: list[str] = []
        if self.resume_from_file_var.get().strip():
            notes.append(f"本次任务将从指定文件开始处理：{self.resume_from_file_var.get().strip()}")
        if self.maximize_utilization_var.get() and self.utilization_var.get().strip() == "低占用":
            notes.append("当前选择低占用，已优先采用低占用参数，不会追加高占用优化。")
        if (
            self.maximize_utilization_var.get()
            and self.utilization_var.get().strip() != "低占用"
            and self.device_mode_var.get().strip() == "CPU+GPU协同"
        ):
            notes.append("已开启高占用优化：当前将使用协同高占用（GPU 转写 + CPU 分离并行增强）。")
        if self.enable_diarization_var.get() and not diarization_available(self.hf_token_var.get()):
            notes.append("说话人模型尚未就绪；请在“首次使用 / 模型”中准备，或关闭说话人区分后仅转写。")
        elif self.enable_diarization_var.get() and self.diarization_policy_var.get().strip() == "运行时询问":
            notes.append("若某个文件无法做说话人分离，主窗口会弹出“仅转写继续”或“中断任务”的选择。")
        if self._parse_int(self.parallel_workers_var.get(), 0, minimum=0) > 1 and self.device_mode_var.get().strip() in {"仅GPU", "CPU+GPU协同"}:
            notes.append("当前硬件模式下不支持多个 worker；程序自动调节参数关闭时，启动会报错。")
        self.preview_notes_var.set("\n".join(f"- {item}" for item in notes) if notes else "设置就绪，点击“开始转写”运行；如参数不兼容，启动前会提示。")
        self.apply_recommended_button.configure(
            state="disabled" if self.current_task_running or self.device_mode_var.get().strip() == self.current_recommended_mode else "normal"
        )

    def browse_input_dir(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="选择输入音视频目录", initialdir=self._existing_dir(self.input_dir_var.get()))
        if chosen:
            self.input_dir_var.set(chosen)

    def browse_output_dir(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="选择输出目录", initialdir=self._existing_dir(self.output_dir_var.get()))
        if chosen:
            self.output_dir_var.set(chosen)

    def _existing_dir(self, raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        return str(path) if path.exists() and path.is_dir() else str(PROJECT_DIR)

    def _parse_int(self, raw: str, default: int, minimum: int | None = None) -> int:
        try:
            value = int(str(raw).strip())
        except Exception:
            value = default
        if minimum is not None:
            value = max(minimum, value)
        return value

    def build_current_command(self) -> list[str]:
        return self.control.build_command(
            input_dir=self.input_dir_var.get().strip(),
            output_dir=self.output_dir_var.get().strip(),
            language=self.language_code_var.get().strip(),
            precision_mode=self.precision_var.get().strip() or "均衡",
            utilization=self.utilization_var.get().strip() or "均衡",
            device_mode=self.device_mode_var.get().strip() or "自动",
            parallel_workers=self._parse_int(self.parallel_workers_var.get(), 0, minimum=0),
            resume_from_file=self.resume_from_file_var.get().strip(),
            enable_diarization=bool(self.enable_diarization_var.get()),
            diarization_unavailable_policy=self.diarization_policy_var.get().strip() or "运行时询问",
            hf_token=self.hf_token_var.get().strip(),
            write_json=bool(self.write_json_var.get()),
            skip_existing=bool(self.skip_existing_var.get()),
            auto_env_tune=bool(self.auto_env_tune_var.get()),
            maximize_utilization=bool(self.maximize_utilization_var.get()),
            heartbeat_sec=self._parse_int(self.heartbeat_var.get(), 15, minimum=2),
        )

    def load_abort_point(self) -> None:
        if not self.current_abort_meta:
            messagebox.showinfo(WINDOW_TITLE, "暂无已记录的中断恢复点。", parent=self)
            return
        if self.current_task_running:
            messagebox.showwarning(WINDOW_TITLE, "任务运行中，不能改写当前参数。", parent=self)
            return
        self.suspend_preview_updates = True
        try:
            self.input_dir_var.set(str(self.current_abort_meta.get("input_dir", self.input_dir_var.get())))
            self.output_dir_var.set(str(self.current_abort_meta.get("output_dir", self.output_dir_var.get())))
            self.resume_from_file_var.set(str(self.current_abort_meta.get("resume_from_file", "")))
        finally:
            self.suspend_preview_updates = False
        self.save_current_settings()
        self.refresh_preview()
        messagebox.showinfo(WINDOW_TITLE, "已把恢复点加载到当前配置。", parent=self)

    def apply_recommended_mode(self) -> None:
        if not self.current_task_running and self.current_recommended_mode:
            self.device_mode_var.set(self.current_recommended_mode)

    def refresh_runtime(self, force: bool = False) -> None:
        task_meta = self.control.load_task_meta()
        task_running = False
        task_paused = False
        self.task_identity_verified = False
        if task_meta and "pid" in task_meta:
            state = self.control.tracked_task_state(task_meta)
            task_running = state != "stopped"
            self.task_identity_verified = state == "running"
            if task_running:
                task_paused = bool(task_meta.get("paused", False))
                if self.synced_running_pid != int(task_meta["pid"]):
                    self._sync_from_meta(task_meta)
                    self.synced_running_pid = int(task_meta["pid"])
            elif task_meta.get("paused"):
                task_meta["paused"] = False
                self.control.save_task_meta(task_meta)
        else:
            self.synced_running_pid = 0

        self.current_task_meta = task_meta
        self.current_task_running = task_running
        self.current_task_paused = task_paused
        self.current_task_result = self.control.load_task_result(task_meta)

        if self.task_identity_verified:
            self.handle_pending_task_interaction()
        if task_running and not force and not self.auto_refresh_var.get():
            self._update_buttons()
            self.schedule_refresh()
            return

        progress = (self.progress_reader.read(Path(task_meta["log_path"]))
                    if task_meta and task_meta.get("log_path") else self.control.parse_progress([]))
        self._update_dashboard(progress)
        self._update_buttons()
        self._update_lock_state()
        self.refresh_abort_panel()
        self.refresh_logs(force=force)
        if force:
            if not task_running:
                self.recommendation_key = None
            self.refresh_preview()
        if task_running and not self.task_identity_verified:
            self.status_var.set("进程身份待核实")
            self.status_note_var.set("暂时无法核实任务进程身份，暂停、继续和停止操作已锁定。")
        if not task_running:
            self.notify_task_result_if_needed()
        self.schedule_refresh()

    def handle_pending_task_interaction(self) -> None:
        requests = self.control.list_pending_task_interactions(self.current_task_meta)
        if not requests:
            return
        request = requests[0]
        request_id = str(request.get("request_id", ""))
        if time.monotonic() < self.deferred_interaction_until.get(request_id, 0.0):
            return
        summary = str(request.get("summary", "")).strip() or "当前文件无法执行说话人分离。"
        decision = messagebox.askyesnocancel(
            "说话人分离需要选择",
            summary
            + "\n\n选择“是”：当前文件仅转写并继续。"
            + "\n选择“否”：立即中断任务。"
            + "\n选择“取消”：暂不处理，任务继续等待。",
            parent=self,
        )
        if decision is None:
            self.deferred_interaction_until[request_id] = time.monotonic() + 30.0
            return
        self.deferred_interaction_until.pop(request_id, None)
        response = "transcribe-only" if decision else "abort"
        ok, message = self.control.respond_to_task_interaction(
            self.current_task_meta,
            request_id,
            response,
        )
        if not ok:
            messagebox.showerror(WINDOW_TITLE, message, parent=self)

    def notify_task_result_if_needed(self) -> None:
        if not self.current_task_meta or not self.current_task_result:
            return
        if bool(self.current_task_meta.get("result_notified", False)):
            return

        status = str(self.current_task_result.get("status", "")).strip()
        note = str(self.current_task_result.get("note", "")).strip()
        failed = self.current_task_result.get("failed", [])
        failed_count = len(failed) if isinstance(failed, list) else 0
        if status in {"exited", "crashed"}:
            messagebox.showerror(
                "任务异常退出",
                note or "后端任务异常退出，未提供具体原因。",
                parent=self,
            )
        elif status == "aborted":
            messagebox.showwarning(
                "任务已中断",
                note or "任务已中断。",
                parent=self,
            )
        elif status == "partial-failed":
            messagebox.showwarning(
                "任务部分失败",
                f"任务已结束，失败文件数：{failed_count}。\n\n{note}".strip(),
                parent=self,
            )

        self.current_task_meta["result_notified"] = True
        try:
            self.control.save_task_meta(self.current_task_meta)
        except Exception:
            pass

    def on_close(self) -> None:
        if self.model_window is not None and self.model_window.winfo_exists():
            if not self.model_window.close():
                return
        if self.current_task_running:
            ok = messagebox.askyesno(WINDOW_TITLE, "后台任务仍在运行。\n关闭界面不会停止识别任务，是否继续关闭？", parent=self)
            if not ok:
                return
        self.save_current_settings()
        try:
            self.control.save_hf_token(self.hf_token_var.get())
        except Exception as exc:
            close_anyway = messagebox.askyesno(
                "Token 保存失败",
                f"Hugging Face Token 未能加密保存：\n{exc}\n\n"
                "如果现在关闭，下次启动可能需要重新输入。是否仍然关闭？",
                parent=self,
            )
            if not close_anyway:
                return
        if self.refresh_after_id:
            try:
                self.after_cancel(self.refresh_after_id)
            except Exception:
                pass
        self.destroy()

    def on_start(self) -> None:
        if self.current_task_running:
            messagebox.showwarning(WINDOW_TITLE, "已有任务在运行，请先停止当前任务。", parent=self)
            return
        if not self.input_dir_var.get().strip() or not self.output_dir_var.get().strip():
            messagebox.showwarning(WINDOW_TITLE, "请输入输入目录和输出目录。", parent=self)
            return
        try:
            self.save_current_settings()
            hf_token = self.hf_token_var.get().strip()
            cmd = self.build_current_command()
            if self.model_window is not None and self.model_window.winfo_exists() and self.model_window.process is not None:
                self.model_window.lift()
                return
            device = "cuda" if self.device_mode_var.get() != "仅CPU" and self.control.torch.cuda.is_available() else "cpu"
            model = self.control.default_asr_model(self.control.resolve_speed_profile(self.precision_var.get()), device)
            if local_model(model) is None or (self.enable_diarization_var.get() and local_model("diarization") is None):
                self.show_model_setup()
                return
            blocker = self.control.get_task_start_blocker(cmd, hf_token)
            if blocker:
                messagebox.showwarning(
                    "无法启动任务",
                    blocker,
                    parent=self,
                )
                return
            try:
                self.control.save_hf_token(hf_token)
            except Exception as exc:
                messagebox.showwarning(
                    WINDOW_TITLE,
                    f"Token 可用于本次任务，但本机加密保存失败：{exc}",
                    parent=self,
                )
            self.current_task_meta = self.control.start_task(cmd, hf_token)
        except Exception as exc:
            messagebox.showerror(WINDOW_TITLE, f"启动任务失败：{exc}", parent=self)
            return
        messagebox.showinfo(WINDOW_TITLE, f"任务已启动，PID={self.current_task_meta['pid']}", parent=self)
        self.refresh_runtime(force=True)

    def on_pause(self) -> None:
        if not self.current_task_meta:
            return
        ok, message = self.control.set_tracked_task_paused(self.current_task_meta, pause=True)
        if ok:
            self.current_task_meta["paused"] = True
            self.control.save_task_meta(self.current_task_meta)
            messagebox.showinfo(WINDOW_TITLE, message, parent=self)
        else:
            messagebox.showerror(WINDOW_TITLE, f"暂停失败：{message}", parent=self)
        self.refresh_runtime(force=True)

    def on_resume(self) -> None:
        if not self.current_task_meta:
            return
        ok, message = self.control.set_tracked_task_paused(self.current_task_meta, pause=False)
        if ok:
            self.current_task_meta["paused"] = False
            self.control.save_task_meta(self.current_task_meta)
            messagebox.showinfo(WINDOW_TITLE, message, parent=self)
        else:
            messagebox.showerror(WINDOW_TITLE, f"继续失败：{message}", parent=self)
        self.refresh_runtime(force=True)

    def on_stop(self) -> None:
        if not self.current_task_meta:
            return
        if not messagebox.askyesno(WINDOW_TITLE, "确认停止当前后台任务？", parent=self):
            return
        ok, message = self.control.kill_tracked_task(self.current_task_meta)
        if ok:
            self.current_task_meta["stop_requested"] = True
            self.current_task_meta["result_notified"] = True
            self.control.save_task_meta(self.current_task_meta)
            messagebox.showinfo(WINDOW_TITLE, f"已停止任务。\n{message}", parent=self)
        else:
            messagebox.showerror(WINDOW_TITLE, f"停止失败：{message}", parent=self)
        self.refresh_runtime(force=True)

    def _sync_from_meta(self, task_meta: dict[str, Any]) -> None:
        values, flags = self.control.parse_cli_tokens([str(item) for item in task_meta.get("cmd", [])])
        self.suspend_preview_updates = True
        try:
            self.input_dir_var.set(values.get("--input-dir", self.input_dir_var.get()))
            self.output_dir_var.set(values.get("--output-dir", self.output_dir_var.get()))
            self.resume_from_file_var.set(values.get("--resume-from-file", self.resume_from_file_var.get()))
            self.language_code_var.set(values.get("--language", self.language_code_var.get()))
            self.parallel_workers_var.set(values.get("--parallel-workers", self.parallel_workers_var.get()))
            self.precision_var.set({"accurate": "高精度", "fast": "高速"}.get(values.get("--speed-profile", "balanced"), "均衡"))
            self.device_mode_var.set({"cpu": "仅CPU", "cuda": "仅GPU", "auto": "CPU+GPU协同" if "--hybrid" in flags else "自动"}.get(values.get("--device", "auto"), "自动"))
            self.enable_diarization_var.set("--disable-diarization" not in flags)
            self.diarization_policy_var.set({"ask": "运行时询问", "transcribe-only": "自动仅转写", "abort": "立即中断任务"}.get(values.get("--diarization-unavailable-policy", "ask"), "运行时询问"))
            self.write_json_var.set("--write-json" in flags)
            self.skip_existing_var.set("--skip-existing" in flags)
            self.auto_env_tune_var.set("--auto-env-tune" in flags and "--disable-auto-env-tune" not in flags)
            self.maximize_utilization_var.set("--maximize-utilization" in flags)
            self.heartbeat_var.set(values.get("--heartbeat-sec", self.heartbeat_var.get()))
        finally:
            self.suspend_preview_updates = False

    def _update_dashboard(self, progress: dict[str, Any]) -> None:
        pid = int(self.current_task_meta.get("pid", 0)) if self.current_task_meta else 0
        total = int(progress.get("total_files", 0))
        finished = int(progress.get("finished_files", 0))
        failed = int(progress.get("fail_count", 0))
        self.pid_var.set(str(pid) if pid else "-")
        self.status_var.set("已暂停" if self.current_task_running and self.current_task_paused else "运行中" if self.current_task_running else "已结束" if self.current_task_meta else "就绪")
        self.progress_text_var.set(f"{finished}/{total}" if total > 0 else f"{finished}/?")
        self.fail_count_var.set(str(failed))
        self.stage_var.set(str(progress.get("stage", "等待启动")))
        self.step_var.set(str(progress.get("step", "")).strip() or "-")
        self.progress_bar.configure(value=min(100, (finished / max(total, 1)) * 100) if total > 0 else 0)

        if self.current_task_running and self.current_task_paused:
            self.summary_var.set(f"任务已暂停。阶段：{self.stage_var.get()}；步骤：{self.step_var.get()}。")
            self.status_note_var.set("当前任务参数已锁定；仅日志设置会持续刷新。")
        elif self.current_task_running:
            self.summary_var.set(f"任务运行中。阶段：{self.stage_var.get()}；步骤：{self.step_var.get()}。")
            self.status_note_var.set("当前任务参数已锁定；仅日志设置会持续刷新。")
        elif self.current_task_meta:
            result_status = str((self.current_task_result or {}).get("status", "")).strip()
            result_note = str((self.current_task_result or {}).get("note", "")).strip()
            result_failed = (self.current_task_result or {}).get("failed", [])
            if bool(self.current_task_meta.get("stop_requested", False)) and not self.current_task_result:
                self.status_var.set("已停止")
                self.summary_var.set("任务已由用户停止。")
            elif result_status == "ok":
                self.status_var.set("已完成")
                self.summary_var.set("任务已完成，后端结果状态正常。")
            elif result_status == "partial-failed":
                result_failed_count = len(result_failed) if isinstance(result_failed, list) else failed
                self.status_var.set("部分失败")
                self.summary_var.set(f"任务已结束，失败文件数：{result_failed_count}。")
            elif result_status in {"exited", "crashed", "aborted"}:
                self.status_var.set("已中断" if result_status == "aborted" else "异常退出")
                self.summary_var.set(result_note or "任务异常结束，未提供具体原因。")
            elif progress.get("done_flag") and progress.get("success_flag"):
                self.summary_var.set("任务已完成，日志显示全部成功。")
            elif progress.get("done_flag"):
                self.summary_var.set(f"任务已结束，但存在失败文件，失败数：{failed}。")
            else:
                self.summary_var.set("任务进程已结束，但日志未写入完整收尾信息，可能是手动停止。")
            self.status_note_var.set("任务已结束，可调整参数后重新启动。")
        else:
            self.summary_var.set("当前没有已登记任务。")
            self.status_note_var.set("任务参数空闲，可直接修改。")

    def _update_buttons(self) -> None:
        self.start_button.configure(state="disabled" if self.current_task_running else "normal")
        controllable = self.current_task_running and self.task_identity_verified
        self.pause_button.configure(state="normal" if controllable and not self.current_task_paused else "disabled")
        self.resume_button.configure(state="normal" if controllable and self.current_task_paused else "disabled")
        self.stop_button.configure(state="normal" if controllable else "disabled")

    def _update_lock_state(self) -> None:
        locked = self.current_task_running
        for widget, enabled_state, disabled_state in self.lockable_widgets:
            try:
                widget.configure(state=disabled_state if locked else enabled_state)
            except tk.TclError:
                pass
        self.load_abort_button.configure(state="disabled" if locked or not self.current_abort_meta else "normal")

    def refresh_logs(self, force: bool = False) -> None:
        if not self.current_task_meta:
            self.log_path_var.set("-")
            self.heartbeat_log_path_var.set("-")
            self.snapshot_var.set("当前没有任务配置快照。")
            if force:
                self._set_text(self.normal_log_text, "(暂无普通日志)")
                self._set_text(self.heartbeat_log_text, "(暂无心跳日志)")
            return

        log_path = Path(str(self.current_task_meta.get("log_path", "")))
        heartbeat_raw = str(self.current_task_meta.get("heartbeat_log_path", "")).strip()
        heartbeat_path = Path(heartbeat_raw) if heartbeat_raw else log_path.with_name(f"{log_path.stem}.heartbeat{log_path.suffix}")
        self.log_path_var.set(str(log_path))
        self.heartbeat_log_path_var.set(str(heartbeat_path))
        if "cmd" in self.current_task_meta:
            snapshot = self.control.extract_effective_config_from_cmd(self.current_task_meta["cmd"])
            self.snapshot_var.set(" | ".join(f"{k}={v}" for k, v in snapshot.items() if v))

        max_lines = self._parse_int(self.log_tail_lines_var.get(), 500, minimum=50)
        keyword = self.log_filter_var.get().strip().lower()
        normal_lines = self.control.read_log_tail(log_path, max_lines=max(2000, max_lines * 4))
        heartbeat_lines = self.control.read_log_tail(heartbeat_path, max_lines=max(2000, max_lines * 4))
        if keyword:
            normal_lines = [line for line in normal_lines if keyword in line.lower()]
            heartbeat_lines = [line for line in heartbeat_lines if keyword in line.lower()]
        normal_text = "\n".join(normal_lines[-max_lines:] or ["(暂无普通日志)"])
        heartbeat_text = "\n".join(heartbeat_lines[-max_lines:] or ["(暂无心跳日志)"])
        if force or normal_text != self.last_normal_log_text:
            self._set_text(self.normal_log_text, normal_text)
            self.last_normal_log_text = normal_text
        if force or heartbeat_text != self.last_heartbeat_log_text:
            self._set_text(self.heartbeat_log_text, heartbeat_text)
            self.last_heartbeat_log_text = heartbeat_text

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def schedule_refresh(self) -> None:
        if self.refresh_after_id:
            try:
                self.after_cancel(self.refresh_after_id)
            except Exception:
                pass
            self.refresh_after_id = None
        if not self.current_task_running:
            return
        delay_ms = max(1, self._parse_int(self.refresh_interval_var.get(), 2, minimum=1)) * 1000
        if not self.auto_refresh_var.get():
            delay_ms = 2000
        self.refresh_after_id = self.after(delay_ms, lambda: self.refresh_runtime(force=False))


def main() -> None:
    if not CONSOLE_SCRIPT_PATH.exists():
        show_error_dialog(f"未找到控制台入口文件：{CONSOLE_SCRIPT_PATH}")
        raise SystemExit(1)

    try:
        control = load_console_module()
    except ModuleNotFoundError as exc:
        show_error_dialog(
            "启动失败：缺少运行依赖。\n\n"
            f"缺少模块: {exc.name}\n\n"
            "便携包用户：请完整解压后从根目录“启动.cmd”运行。\n"
            "源码用户：请按 README 中的开发环境说明安装依赖。"
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        show_error_dialog(
            "加载控制台逻辑失败。\n\n"
            f"{exc}\n\n"
            f"{traceback.format_exc(limit=3)}"
        )
        raise SystemExit(1) from exc

    try:
        first_start = not control.USER_SETTINGS_PATH.exists()
        app = DesktopConsoleApp(control)
        if first_start:
            app.after(300, app.show_model_setup)
        app.mainloop()
    except Exception as exc:
        show_error_dialog(
            "本地控制台启动失败。\n\n"
            f"{exc}\n\n"
            f"{traceback.format_exc(limit=5)}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
