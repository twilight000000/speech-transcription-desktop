import argparse
from model_resources import diarization_available, local_model, default_asr_model, load_local_vad
from collections import Counter
from functools import lru_cache
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def configure_portable_runtime() -> None:
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    project_dir = Path(__file__).resolve().parent
    portable_runtime = project_dir.parent / "portable_runtime"
    if not portable_runtime.exists():
        return

    python_dir = portable_runtime / "python"
    hf_cache = portable_runtime / "hf_cache"
    hf_home = portable_runtime / "hf_home"
    torch_home = portable_runtime / "torch_cache"

    current_path = os.environ.get("PATH", "")
    existing_parts = {part.casefold() for part in current_path.split(os.pathsep) if part}
    prepend_parts: list[str] = []
    for path in (python_dir, python_dir / "Scripts"):
        if path.exists() and str(path).casefold() not in existing_parts:
            prepend_parts.append(str(path))
    if prepend_parts:
        os.environ["PATH"] = os.pathsep.join(prepend_parts + [current_path])

    if hf_cache.exists():
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ.setdefault("HF_HOME", str(hf_home))
    # 随包使用缓存，避免误用用户级旧缓存。
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


configure_portable_runtime()

import whisperx

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

WHISPERX_AUDIO_SAMPLE_RATE = 16000
MIN_DIARIZATION_AUDIO_SECONDS = 1.0
PYANNOTE_STD_WARNING_PATTERN = r"std\(\): degrees of freedom is <= 0\..*"
DIARIZATION_UNAVAILABLE_POLICIES = ("ask", "transcribe-only", "abort")

# pyannote 在 Windows + torchcodec 环境下会给出冗长告警，不影响本脚本流程。
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)
warnings.filterwarnings(
    "ignore",
    message=PYANNOTE_STD_WARNING_PATTERN,
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "input"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
MAX_TRANSCRIBE_CHUNK_SIZE = 30
MIN_TRANSCRIBE_CHUNK_SIZE = 5
MAX_PARALLEL_FILE_WORKERS = 8
RUNTIME_DIR = Path(__file__).resolve().parent / ".runtime"
LAST_DIARIZATION_ABORT_PATH = RUNTIME_DIR / "last_diarization_abort.json"


def set_console_title(title: str | None) -> None:
    if os.name != "nt" or not title:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW(str(title))
    except Exception:
        pass


def configure_windows_runtime_guards() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfault_errorbox = 0x0002
        sem_noopenfile_errorbox = 0x8000
        flags = sem_failcriticalerrors | sem_nogpfault_errorbox | sem_noopenfile_errorbox
        kernel32 = ctypes.windll.kernel32
        if hasattr(kernel32, "SetThreadErrorMode"):
            previous = ctypes.c_uint(0)
            kernel32.SetThreadErrorMode(flags, ctypes.byref(previous))
        kernel32.SetErrorMode(flags)
        return True
    except Exception:
        return False


def has_torchcodec_distribution() -> bool:
    try:
        importlib.metadata.version("torchcodec")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


class TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    @property
    def encoding(self) -> str:
        for stream in self.streams:
            enc = getattr(stream, "encoding", None)
            if enc:
                return str(enc)
        return "utf-8"


class DiarizationAbortRequested(SystemExit):
    """Abort the current task when diarization cannot proceed and the user declines fallback."""

    def __init__(
        self,
        message: str,
        *,
        file_label: str = "",
        stage: str = "",
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.file_label = file_label
        self.stage = stage
        self.reason = reason


def configure_log_mirroring(log_file: str | None) -> tuple[Any | None, Any, Any]:
    if not log_file:
        return None, sys.stdout, sys.stderr

    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = path.open("a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_fp)
    sys.stderr = TeeStream(original_stderr, log_fp)
    return log_fp, original_stdout, original_stderr


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def clear_last_diarization_abort_checkpoint() -> None:
    try:
        LAST_DIARIZATION_ABORT_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_last_diarization_abort_checkpoint(
    *,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    exc: DiarizationAbortRequested,
) -> None:
    file_label = str(exc.file_label or "").strip()
    if not file_label:
        return
    ensure_runtime_dir()
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "resume_from_file": file_label,
        "stage": str(exc.stage or ""),
        "reason": str(exc.reason or str(exc)),
        "policy": str(getattr(args, "diarization_unavailable_policy", "") or ""),
        "process_label": str(getattr(args, "process_label", "") or ""),
    }
    LAST_DIARIZATION_ABORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_parallel_worker_result(
    result_file: str | None,
    *,
    status: str,
    failed: list[str],
    total_files: int,
    worker_label: str,
    note: str = "",
) -> None:
    if not result_file:
        return
    payload = {
        "status": status,
        "failed": failed,
        "total_files": int(total_files),
        "worker_label": worker_label,
        "note": note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        write_json_file(Path(result_file).expanduser().resolve(), payload)
    except Exception:
        pass


def configure_heartbeat_log(heartbeat_log_file: str | None) -> Any | None:
    if not heartbeat_log_file:
        return None
    path = Path(heartbeat_log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8", buffering=1)


def heartbeat_write(heartbeat_fp: Any | None, worker_name: str, message: str) -> None:
    if heartbeat_fp is None:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        heartbeat_fp.write(f"{ts} [{worker_name}] [心跳] {message}\n")
        heartbeat_fp.flush()
    except Exception:
        pass


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes = int(seconds // 60)
    remain = int(seconds % 60)
    return f"{minutes}分{remain}秒"


def detect_runtime_environment() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_names: list[str] = []
    gpu_memory_gb: list[float] = []
    gpu_capabilities: list[str] = []
    if cuda_available:
        try:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            gpu_names = []
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
                gpu_memory_gb.append(round(float(props.total_memory) / (1024**3), 2))
                gpu_capabilities.append(f"{int(props.major)}.{int(props.minor)}")
            except Exception:
                gpu_memory_gb.append(0.0)
                gpu_capabilities.append("unknown")

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", "") or "",
        "cuda_available": cuda_available,
        "cuda_count": int(torch.cuda.device_count() if cuda_available else 0),
        "gpu_names": gpu_names,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_capabilities": gpu_capabilities,
        "torchcodec_present": has_torchcodec_distribution(),
    }


def print_runtime_environment(env: dict[str, Any]) -> None:
    print("========== 环境检测 ==========")
    print(f"系统: {env.get('os', 'unknown')}")
    print(f"Python: {env.get('python', 'unknown')}")
    print(f"Torch: {env.get('torch', 'unknown')}")
    print(f"CPU线程: {env.get('cpu_count', 0)}")
    print(
        f"CUDA: {'可用' if env.get('cuda_available') else '不可用'}"
        f" (GPU数={env.get('cuda_count', 0)})"
    )
    gpu_names = env.get("gpu_names", []) or []
    if gpu_names:
        print(f"GPU: {', '.join(str(x) for x in gpu_names)}")
    gpu_memory_gb = env.get("gpu_memory_gb", []) or []
    if gpu_memory_gb:
        print(f"GPU显存: {', '.join(f'{float(x):.2f}GB' for x in gpu_memory_gb)}")
    if env.get("torch_cuda"):
        print(f"Torch CUDA: {env.get('torch_cuda')}")
    print("音频解码: PyAV / 内置 FFmpeg 库")
    print(f"TorchCodec模块: {'存在' if env.get('torchcodec_present') else '未安装/未发现'}")
    print("==============================")


def resolve_single_device_for_tuning(args: argparse.Namespace, env: dict[str, Any]) -> str:
    if args.device == "cpu":
        return "cpu"
    if args.device == "cuda":
        return "cuda" if bool(env.get("cuda_available")) else "cpu"
    return "cuda" if bool(env.get("cuda_available")) else "cpu"


def suggest_high_util_batch(
    *,
    device: str,
    speed_profile: str,
    cpu_count: int,
    gpu_mem_gb: float | None,
) -> int:
    if device == "cuda":
        mem = gpu_mem_gb or 0.0
        if mem and mem < 5.5:
            if speed_profile == "accurate":
                return 4
            if speed_profile == "fast":
                return 12
            return 8
        if speed_profile == "accurate":
            return 12 if mem >= 7.5 else 8
        if speed_profile == "fast":
            return 32 if mem >= 7.5 else 20
        return 20 if mem >= 7.5 else 12

    if speed_profile == "accurate":
        return 4 if cpu_count >= 12 else 3
    if speed_profile == "fast":
        return 10 if cpu_count >= 12 else 6
    return 8 if cpu_count >= 12 else 5


def suggest_hardware_chunk(*, device: str, speed_profile: str, gpu_mem_gb: float | None) -> int:
    if device == "cuda":
        mem = gpu_mem_gb or 0.0
        if mem and mem < 5.5:
            if speed_profile == "accurate":
                return 16
            if speed_profile == "fast":
                return 22
            return 18
        if speed_profile == "accurate":
            return 20 if mem >= 7.5 else 18
        if speed_profile == "fast":
            return 28 if mem >= 7.5 else 24
        return 24 if mem >= 7.5 else 20
    if speed_profile == "accurate":
        return 18
    if speed_profile == "fast":
        return 22
    return 20


def primary_gpu_memory_from_env(env: dict[str, Any]) -> float | None:
    values = env.get("gpu_memory_gb", []) or []
    if not values:
        return None
    try:
        value = float(values[0])
    except Exception:
        return None
    return value if value > 0 else None


def get_primary_gpu_memory_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    except Exception:
        return None


def safe_cuda_batch_cap(speed_profile: str, gpu_mem_gb: float | None) -> int | None:
    mem = gpu_mem_gb or 0.0
    if mem <= 0:
        return None
    if mem < 4.0:
        if speed_profile == "accurate":
            return 2
        if speed_profile == "fast":
            return 8
        return 4
    if mem < 5.5:
        if speed_profile == "accurate":
            return 4
        if speed_profile == "fast":
            return 12
        return 8
    return None


def safe_cuda_chunk_cap(speed_profile: str, gpu_mem_gb: float | None) -> int | None:
    mem = gpu_mem_gb or 0.0
    if mem <= 0:
        return None
    if mem < 4.0:
        if speed_profile == "accurate":
            return 14
        if speed_profile == "fast":
            return 18
        return 16
    if mem < 5.5:
        if speed_profile == "accurate":
            return 16
        if speed_profile == "fast":
            return 22
        return 18
    return None


def run_with_heartbeat(
    worker_name: str,
    label: str,
    heartbeat_sec: int,
    heartbeat_fp: Any | None,
    fn: Any,
    *fn_args: Any,
    **fn_kwargs: Any,
) -> Any:
    interval = max(2, int(heartbeat_sec))
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as heartbeat_executor:
        future = heartbeat_executor.submit(fn, *fn_args, **fn_kwargs)
        while True:
            try:
                return future.result(timeout=interval)
            except concurrent.futures.TimeoutError:
                elapsed = time.perf_counter() - start
                heartbeat_write(
                    heartbeat_fp,
                    worker_name,
                    f"{label} 进行中，已耗时 {format_seconds(elapsed)}",
                )


def apply_accuracy_preserving_max_utilization_tuning(
    args: argparse.Namespace,
    env: dict[str, Any],
) -> tuple[list[str], list[str]]:
    import torch

    adjustments: list[str] = []
    notices: list[str] = []
    cpu_count = int(env.get("cpu_count", os.cpu_count() or 4))
    device_for_tune = resolve_single_device_for_tuning(args, env)

    if args.threads <= 0:
        args.threads = cpu_count
        adjustments.append(f"threads 自动提升为 {cpu_count}（使用全部逻辑核心）。")
    elif args.threads < cpu_count:
        prev = args.threads
        args.threads = cpu_count
        adjustments.append(f"threads {prev} -> {cpu_count}（提高 CPU 利用率）。")

    gpu_mem_gb = None
    if device_for_tune == "cuda":
        try:
            gpu_mem_gb = primary_gpu_memory_from_env(env) or torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            gpu_mem_gb = None
    suggested_batch = suggest_high_util_batch(
        device=device_for_tune,
        speed_profile=args.speed_profile,
        cpu_count=cpu_count,
        gpu_mem_gb=gpu_mem_gb,
    )
    if args.batch_size is None:
        args.batch_size = suggested_batch
        adjustments.append(f"batch_size 自动设为 {suggested_batch}（保精度吞吐优化）。")
    elif args.batch_size < suggested_batch:
        prev_batch = args.batch_size
        args.batch_size = suggested_batch
        adjustments.append(f"batch_size {prev_batch} -> {suggested_batch}（按硬件自动上调）。")

    suggested_chunk = suggest_hardware_chunk(
        device=device_for_tune,
        speed_profile=args.speed_profile,
        gpu_mem_gb=gpu_mem_gb,
    )
    if args.chunk_size is None:
        args.chunk_size = suggested_chunk
        adjustments.append(f"chunk_size 自动设为 {suggested_chunk}（按硬件建议）。")
    elif args.chunk_size < suggested_chunk:
        prev_chunk = args.chunk_size
        args.chunk_size = suggested_chunk
        adjustments.append(f"chunk_size {prev_chunk} -> {suggested_chunk}（按硬件自动上调）。")

    if args.chunk_size is not None and args.chunk_size > MAX_TRANSCRIBE_CHUNK_SIZE:
        args.chunk_size = MAX_TRANSCRIBE_CHUNK_SIZE
        adjustments.append(f"chunk_size 过大，已限制到 {MAX_TRANSCRIBE_CHUNK_SIZE}。")

    notices.append("未修改模型/解码参数（beam、best_of、temperatures），识别精度策略保持不变。")
    if args.hybrid:
        notices.append("当前为协同模式：已应用协同高占用（GPU转写参数 + CPU线程占用提升）。")
    if device_for_tune == "cpu" and args.speed_profile == "accurate":
        notices.append("当前为 CPU + 高精度，利用率可提高但整体耗时仍可能较长。")

    return adjustments, notices


def apply_auto_env_tuning(
    args: argparse.Namespace,
    env: dict[str, Any],
) -> tuple[list[str], list[str]]:
    adjustments: list[str] = []
    notices: list[str] = []
    cuda_available = bool(env.get("cuda_available"))

    if args.device == "cuda" and not cuda_available:
        args.device = "cpu"
        adjustments.append("未检测到 CUDA，device: cuda -> cpu")

    if args.hybrid and args.device == "cpu":
        args.hybrid = False
        adjustments.append("device=cpu 与 hybrid 冲突，hybrid 已自动关闭")
    elif args.hybrid and not cuda_available:
        args.hybrid = False
        adjustments.append("未检测到 CUDA，hybrid 已自动关闭")

    effective_device = "cuda" if args.device == "cuda" or (args.device == "auto" and cuda_available) else "cpu"
    if args.compute_type == "float16" and effective_device == "cpu":
        args.compute_type = "int8"
        adjustments.append("CPU 不适合 float16，compute_type: float16 -> int8")

    if effective_device == "cuda":
        gpu_mem_gb = primary_gpu_memory_from_env(env)
        batch_cap = safe_cuda_batch_cap(args.speed_profile, gpu_mem_gb)
        if batch_cap is not None and args.batch_size is not None and args.batch_size > batch_cap:
            prev_batch = args.batch_size
            args.batch_size = batch_cap
            adjustments.append(
                f"检测到低显存 GPU（约 {gpu_mem_gb:.2f}GB），batch_size: {prev_batch} -> {batch_cap}"
            )
        chunk_cap = safe_cuda_chunk_cap(args.speed_profile, gpu_mem_gb)
        if chunk_cap is not None and args.chunk_size is not None and args.chunk_size > chunk_cap:
            prev_chunk = args.chunk_size
            args.chunk_size = chunk_cap
            adjustments.append(
                f"检测到低显存 GPU（约 {gpu_mem_gb:.2f}GB），chunk_size: {prev_chunk} -> {chunk_cap}"
            )

    if args.chunk_size is not None:
        if args.chunk_size <= 0:
            args.chunk_size = None
            adjustments.append("chunk_size<=0 无效，已回退为自动")
        elif args.chunk_size > MAX_TRANSCRIBE_CHUNK_SIZE:
            args.chunk_size = MAX_TRANSCRIBE_CHUNK_SIZE
            adjustments.append(f"chunk_size 过大，已限制到 {MAX_TRANSCRIBE_CHUNK_SIZE}")
        elif args.chunk_size < MIN_TRANSCRIBE_CHUNK_SIZE:
            args.chunk_size = MIN_TRANSCRIBE_CHUNK_SIZE
            adjustments.append(f"chunk_size 过小，已提升到 {MIN_TRANSCRIBE_CHUNK_SIZE}")

    if args.batch_size is not None and args.batch_size <= 0:
        args.batch_size = None
        adjustments.append("batch_size<=0 无效，已回退为自动")

    if args.threads < 0:
        args.threads = 0
        adjustments.append("threads<0 无效，已回退为自动")

    if args.heartbeat_sec < 2:
        args.heartbeat_sec = 2
        adjustments.append("heartbeat_sec 过小，已提升到 2 秒")

    if not args.disable_diarization:
        token = os.getenv("HF_TOKEN")
        if not diarization_available(token or "", args.diarization_model):
            args.disable_diarization = True
            adjustments.append("未检测到 HF_TOKEN，已自动关闭说话人分离")
            if args.hybrid:
                args.hybrid = False
                adjustments.append("说话人分离已关闭，hybrid 同步关闭")

    if effective_device == "cpu" and args.speed_profile == "accurate":
        notices.append("当前使用 CPU + 高精度档，速度会明显变慢（这是预期行为）。")
    return adjustments, notices


def is_auto_env_tune_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "auto_env_tune", False)) and not bool(
        getattr(args, "disable_auto_env_tune", False)
    )


def validate_runtime_config_without_auto_tune(
    args: argparse.Namespace,
    env: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    notices: list[str] = []
    cuda_available = bool(env.get("cuda_available"))

    if args.device == "cuda" and not cuda_available:
        errors.append("你选择了 device=cuda，但当前未检测到可用 CUDA。请改为 CPU/自动，或开启程序自动调节参数。")

    if args.hybrid and args.device == "cpu":
        errors.append("你启用了 CPU+GPU 协同，但同时指定了 device=cpu。请改为自动/GPU，或关闭协同。")
    elif args.hybrid and not cuda_available:
        errors.append("你启用了 CPU+GPU 协同，但当前未检测到可用 CUDA。请关闭协同或安装可用 CUDA 环境。")

    effective_device = "cuda" if args.device == "cuda" or (args.device == "auto" and cuda_available) else "cpu"
    if args.compute_type == "float16" and effective_device == "cpu":
        errors.append("你选择了 compute_type=float16，但当前实际设备为 CPU。请改为 int8/float32，或开启程序自动调节参数。")

    if args.chunk_size is not None:
        if args.chunk_size <= 0:
            errors.append("chunk_size 必须大于 0；留空表示使用档位默认值。")
        elif args.chunk_size > MAX_TRANSCRIBE_CHUNK_SIZE:
            errors.append(f"chunk_size={args.chunk_size} 超过上限 {MAX_TRANSCRIBE_CHUNK_SIZE}。")
        elif args.chunk_size < MIN_TRANSCRIBE_CHUNK_SIZE:
            errors.append(f"chunk_size={args.chunk_size} 低于下限 {MIN_TRANSCRIBE_CHUNK_SIZE}。")

    if args.batch_size is not None and args.batch_size <= 0:
        errors.append("batch_size 必须大于 0；留空表示使用档位默认值。")

    if args.threads < 0:
        errors.append("threads 不能小于 0；0 表示使用程序默认线程策略。")

    if args.parallel_workers < 0:
        errors.append("parallel_workers 不能小于 0；0 表示由程序按当前模式自动选择。")

    if args.heartbeat_sec < 2:
        errors.append("heartbeat_sec 不能小于 2 秒。")

    if not args.disable_diarization:
        token = os.getenv("HF_TOKEN")
        if not diarization_available(token or "", args.diarization_model):
            policy = str(getattr(args, "diarization_unavailable_policy", "ask") or "ask").strip().lower()
            has_main_window_channel = bool(str(getattr(args, "interaction_dir", "") or "").strip())
            if policy == "transcribe-only":
                notices.append("未提供 HF_TOKEN；将按已选策略改为仅转写。")
            elif policy == "ask" and has_main_window_channel:
                notices.append("未提供 HF_TOKEN；将等待主窗口选择，不会直接关闭后端。")
            else:
                errors.append("你启用了说话人分离，但未提供 HF_TOKEN。请填写 Token、设置环境变量，或关闭说话人分离。")

    if effective_device == "cpu" and args.speed_profile == "accurate":
        notices.append("当前使用 CPU + 高精度档，速度会明显变慢（这是预期行为）。")

    if errors:
        message = "参数校验失败，程序未自动修改你的设置：\n" + "\n".join(f"- {item}" for item in errors)
        raise SystemExit(message)

    return notices


def validate_parallel_request_without_auto_tune(
    *,
    requested_workers: int,
    file_count: int,
    resolved_device: str,
    use_cooperative_hybrid: bool,
) -> None:
    if requested_workers <= 1:
        return
    errors: list[str] = []
    if requested_workers > MAX_PARALLEL_FILE_WORKERS:
        errors.append(f"parallel_workers={requested_workers} 超过上限 {MAX_PARALLEL_FILE_WORKERS}。")
    if requested_workers > file_count:
        errors.append(f"parallel_workers={requested_workers} 大于待处理文件数 {file_count}。")
    if file_count <= 1:
        errors.append("待处理文件不足 2 个，不能使用文件级多 worker。")
    if use_cooperative_hybrid:
        errors.append("CPU+GPU 协同模式暂不支持文件级多进程并行。")
    if resolved_device != "cpu":
        errors.append("GPU 单设备模式不支持文件级多进程并行，以避免多进程抢占同一张 GPU。")

    if errors:
        message = "文件级并行参数校验失败，程序未自动回退 worker 数：\n" + "\n".join(f"- {item}" for item in errors)
        raise SystemExit(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量音视频转写 + 说话人分离（WhisperX + pyannote）"
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="输入音视频目录（视频自动读取音轨）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument(
        "--speed-profile",
        default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="速度档位：fast 更快，accurate 更准",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper 模型，如 tiny/base/small/medium/large-v3；不填则按 speed-profile 自动选择",
    )
    parser.add_argument("--language", default=None, help="语言代码，如 zh / en，不填则自动检测")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="启用 CPU+GPU 协同分工（同一文件内 GPU 转写 + CPU 说话人分离，需要可用 CUDA）",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "float32", "int8"],
        help="推理精度，CPU 推荐 int8，GPU 推荐 float16",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="转写 batch size（不填按 speed-profile）")
    parser.add_argument("--chunk-size", type=int, default=None, help="转写 chunk_size（秒，不填按 speed-profile）")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数；0 表示自动")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="文件级并行 worker 数；0 表示自动（仅 CPU 批处理默认启用）",
    )
    parser.add_argument("--heartbeat-sec", type=int, default=15, help="长步骤心跳日志间隔（秒）")
    parser.add_argument("--beam-size", type=int, default=None, help="解码 beam size（不填按 speed-profile）")
    parser.add_argument("--best-of", type=int, default=None, help="解码 best_of（不填按 speed-profile）")
    parser.add_argument(
        "--temperatures",
        default=None,
        help="温度列表，逗号分隔，如 0.0 或 0.0,0.2,0.4（不填按 speed-profile）",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过")
    parser.add_argument(
        "--resume-from-file",
        default=None,
        help="从指定音视频文件继续处理；可填相对输入目录的路径，或输入目录内的绝对路径",
    )
    parser.add_argument("--write-json", action="store_true", help="额外输出 JSON")
    parser.add_argument("--disable-diarization", action="store_true", help="关闭说话人分离")
    parser.add_argument(
        "--diarization-unavailable-policy",
        default="ask",
        choices=list(DIARIZATION_UNAVAILABLE_POLICIES),
        help="当单个文件无法执行说话人分离时：ask=运行时询问，transcribe-only=改为仅转写，abort=立即中断任务",
    )
    parser.add_argument(
        "--maximize-utilization",
        action="store_true",
        help="保精度高占用优化（自动匹配当前硬件模式；协同模式下启用协同高占用）",
    )
    parser.add_argument(
        "--auto-env-tune",
        action="store_true",
        help="允许程序自动修正不兼容参数（默认关闭，默认只报错不改参数）",
    )
    parser.add_argument(
        "--disable-auto-env-tune",
        action="store_true",
        help="兼容旧参数：关闭程序自动调节参数",
    )
    parser.add_argument("--log-file", default=None, help="日志文件路径（将日志同时输出到控制台和该文件）")
    parser.add_argument("--heartbeat-log-file", default=None, help="心跳日志文件路径（仅记录心跳日志）")
    parser.add_argument("--process-label", default="", help="后端进程标记名称（用于窗口标题和日志标识）")
    parser.add_argument("--worker-file-list", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--interaction-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parallel-child-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help="说话人分离模型名称",
    )
    parser.add_argument("--num-speakers", type=int, default=None, help="已知说话人数")
    parser.add_argument("--min-speakers", type=int, default=None, help="最少说话人数")
    parser.add_argument("--max-speakers", type=int, default=None, help="最多说话人数")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_execution_mode_recommendation(
    *,
    file_count: int,
    diarization_enabled: bool,
    cuda_available: bool,
) -> tuple[str, str]:
    if not cuda_available:
        return "仅CPU", "未检测到 CUDA，只能使用单设备 CPU。"
    if not diarization_enabled:
        return "仅GPU", "当前未启用说话人分离，单 GPU 更直接。"
    if file_count <= 1:
        return "仅GPU", "单个音频尤其长音频时，优先单 GPU，避免协同模式把说话人分离压到 CPU。"
    return "CPU+GPU协同", "多文件批处理且启用说话人分离时，协同模式更适合提高总吞吐。"


def should_use_cooperative_hybrid(args: argparse.Namespace) -> bool:
    import torch

    if not args.hybrid:
        return False

    if args.device == "cpu":
        print("已启用 hybrid，但你指定了 --device cpu，仅使用 CPU。")
        return False

    if not torch.cuda.is_available():
        print("已启用 hybrid，但 CUDA 不可用，回退为 CPU。")
        return False

    return True


def resolve_parallel_worker_count(
    *,
    requested_workers: int,
    file_count: int,
    resolved_device: str,
    use_cooperative_hybrid: bool,
    cpu_count: int,
) -> tuple[int, list[str]]:
    notices: list[str] = []
    if file_count <= 1:
        if requested_workers > 1:
            notices.append("待处理文件不足 2 个，文件级并行已自动回退为 1。")
        return 1, notices

    if use_cooperative_hybrid:
        if requested_workers > 1:
            notices.append("协同模式暂不启用文件级多进程并行，已自动回退为 1。")
        return 1, notices

    if resolved_device != "cpu":
        if requested_workers > 1:
            notices.append("当前使用 GPU 单设备，文件级多进程并行已自动回退为 1，以避免重复占用同一张 GPU。")
        elif requested_workers == 0:
            notices.append("当前使用 GPU 单设备，文件级多并行默认关闭。")
        return 1, notices

    if requested_workers <= 0:
        if cpu_count >= 24:
            worker_count = 4
        elif cpu_count >= 16:
            worker_count = 3
        elif cpu_count >= 8:
            worker_count = 2
        else:
            worker_count = 1
        worker_count = min(file_count, worker_count)
        if worker_count > 1:
            notices.append(f"CPU 模式已自动启用 {worker_count} 个文件级 worker。")
        return worker_count, notices

    worker_count = max(1, min(int(requested_workers), int(file_count), MAX_PARALLEL_FILE_WORKERS))
    if requested_workers > MAX_PARALLEL_FILE_WORKERS:
        notices.append(f"parallel_workers 过大，已限制到 {MAX_PARALLEL_FILE_WORKERS}。")
    if worker_count > 1:
        notices.append(f"CPU 模式将使用 {worker_count} 个文件级 worker。")
    return worker_count, notices


def resolve_parallel_child_threads(total_threads: int, worker_count: int, cpu_count: int) -> int:
    if worker_count <= 1:
        return max(1, total_threads) if total_threads > 0 else max(1, cpu_count)
    baseline = total_threads if total_threads > 0 else cpu_count
    return max(1, baseline // worker_count)


def write_parallel_worker_file_list(
    *,
    input_root: Path,
    files: list[Path],
    worker_index: int,
) -> Path:
    ensure_runtime_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = RUNTIME_DIR / f"parallel_files_{stamp}_w{worker_index}.json"
    payload = {
        "input_root": str(input_root),
        "files": [str(file_path.relative_to(input_root).as_posix()) for file_path in files],
    }
    write_json_file(path, payload)
    return path


def build_parallel_child_command(
    *,
    args: argparse.Namespace,
    script_path: Path,
    input_root: Path,
    output_root: Path,
    resolved_device: str,
    model: str,
    compute_type: str,
    batch_size: int,
    chunk_size: int,
    beam_size: int,
    best_of: int,
    temperatures: list[float],
    child_threads: int,
    worker_file_list: Path,
    worker_result_file: Path,
    worker_index: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--input-dir",
        str(input_root),
        "--output-dir",
        str(output_root),
        "--speed-profile",
        str(args.speed_profile),
        "--model",
        model,
        "--device",
        resolved_device,
        "--compute-type",
        compute_type,
        "--threads",
        str(child_threads),
        "--parallel-workers",
        "1",
        "--batch-size",
        str(batch_size),
        "--chunk-size",
        str(chunk_size),
        "--beam-size",
        str(beam_size),
        "--best-of",
        str(best_of),
        "--temperatures",
        ",".join(str(item) for item in temperatures),
        "--heartbeat-sec",
        str(args.heartbeat_sec),
        "--diarization-unavailable-policy",
        str(args.diarization_unavailable_policy),
        "--worker-file-list",
        str(worker_file_list),
        "--worker-result-file",
        str(worker_result_file),
        "--parallel-child-index",
        str(worker_index),
    ]

    if args.language:
        cmd += ["--language", str(args.language)]
    if args.write_json:
        cmd.append("--write-json")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if is_auto_env_tune_enabled(args):
        cmd.append("--auto-env-tune")
    if args.disable_diarization:
        cmd.append("--disable-diarization")
    if args.log_file:
        cmd += ["--log-file", str(args.log_file)]
    if args.heartbeat_log_file:
        cmd += ["--heartbeat-log-file", str(args.heartbeat_log_file)]
    if args.interaction_dir:
        cmd += ["--interaction-dir", str(args.interaction_dir)]
    if args.process_label:
        cmd += ["--process-label", f"{args.process_label}/worker-{worker_index}"]
    if args.diarization_model:
        cmd += ["--diarization-model", str(args.diarization_model)]
    if args.num_speakers is not None:
        cmd += ["--num-speakers", str(args.num_speakers)]
    if args.min_speakers is not None:
        cmd += ["--min-speakers", str(args.min_speakers)]
    if args.max_speakers is not None:
        cmd += ["--max-speakers", str(args.max_speakers)]
    return cmd


def run_parallel_file_workers(
    *,
    args: argparse.Namespace,
    audio_files: list[Path],
    input_root: Path,
    output_root: Path,
    resolved_device: str,
    worker_count: int,
    env: dict[str, Any],
) -> list[str]:
    if worker_count <= 1:
        return []

    compute_type = resolve_compute_type(resolved_device, args.compute_type)
    model, chunk_size, batch_size, beam_size, best_of, temperatures = resolve_asr_runtime_config(
        args=args,
        device=resolved_device,
    )
    cpu_count = int(env.get("cpu_count", os.cpu_count() or 4))
    child_threads = resolve_parallel_child_threads(args.threads, worker_count, cpu_count)
    file_groups = partition_audio_files_balanced(audio_files, worker_count)
    script_path = Path(__file__).resolve()

    launch_items: list[tuple[int, subprocess.Popen[Any], Path, Path]] = []
    started_file_lists: list[Path] = []
    started_results: list[Path] = []

    print("[阶段] 启动文件级多并行处理 ...")
    print(
        f"[并行调度] 文件级 worker={len(file_groups)}, "
        f"单 worker threads={child_threads}, model={model}, batch={batch_size}, chunk={chunk_size}"
    )

    try:
        for worker_index, file_group in enumerate(file_groups, start=1):
            worker_file_list = write_parallel_worker_file_list(
                input_root=input_root,
                files=file_group,
                worker_index=worker_index,
            )
            worker_result_file = RUNTIME_DIR / f"parallel_result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_w{worker_index}.json"
            started_file_lists.append(worker_file_list)
            started_results.append(worker_result_file)

            child_cmd = build_parallel_child_command(
                args=args,
                script_path=script_path,
                input_root=input_root,
                output_root=output_root,
                resolved_device=resolved_device,
                model=model,
                compute_type=compute_type,
                batch_size=batch_size,
                chunk_size=chunk_size,
                beam_size=beam_size,
                best_of=best_of,
                temperatures=temperatures,
                child_threads=child_threads,
                worker_file_list=worker_file_list,
                worker_result_file=worker_result_file,
                worker_index=worker_index,
            )
            print(
                f"[并行调度] worker-{worker_index} 已分配 {len(file_group)} 个文件，"
                f"首个文件: {file_group[0].relative_to(input_root).as_posix()}"
            )
            child_env = os.environ.copy()
            process = subprocess.Popen(
                child_cmd,
                cwd=str(script_path.parent),
                env=child_env,
            )
            launch_items.append((worker_index, process, worker_file_list, worker_result_file))
    except Exception:
        for _, process, _, _ in launch_items:
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
        raise

    failed: list[str] = []
    completed_workers: set[int] = set()
    try:
        while len(completed_workers) < len(launch_items):
            found_completed = False
            for worker_index, process, _, worker_result_file in launch_items:
                if worker_index in completed_workers:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue

                found_completed = True
                completed_workers.add(worker_index)
                payload: dict[str, Any] = {}
                if worker_result_file.exists():
                    try:
                        loaded = json.loads(worker_result_file.read_text(encoding="utf-8-sig"))
                        if isinstance(loaded, dict):
                            payload = loaded
                    except Exception:  # noqa: BLE001
                        payload = {}

                worker_failed = payload.get("failed", [])
                if isinstance(worker_failed, list):
                    failed.extend(str(item) for item in worker_failed)
                status = str(payload.get("status", "")).strip()
                note = str(payload.get("note", "")).strip()

                if status == "aborted":
                    raise DiarizationAbortRequested(
                        note or f"worker-{worker_index} 已请求中断任务。",
                        file_label=f"worker-{worker_index}",
                        stage="文件级并行处理",
                        reason=note or "用户在主窗口选择中断任务",
                    )
                if return_code != 0:
                    if worker_result_file.exists():
                        failed.append(
                            f"worker-{worker_index} 异常退出(code={return_code}): "
                            f"{note or status or '未知错误'}"
                        )
                    else:
                        failed.append(
                            f"worker-{worker_index} 异常退出(code={return_code})，未生成结果摘要。"
                        )

            if not found_completed:
                time.sleep(0.2)
    except BaseException:
        # “立即中断任务”必须作用于整个批次，而不是只结束触发提示的一个 worker。
        for _, process, _, _ in launch_items:
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
        for _, process, _, _ in launch_items:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        raise
    finally:
        for path in started_file_lists + started_results:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    return failed


def resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def resolve_threads(threads: int) -> int:
    if threads > 0:
        return threads
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 1)


def configure_torch_threading(threads: int) -> None:
    import torch

    safe_threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(safe_threads)
    os.environ["MKL_NUM_THREADS"] = str(safe_threads)
    try:
        torch.set_num_threads(safe_threads)
    except Exception:
        pass
    try:
        interop = max(1, min(8, safe_threads // 2 or 1))
        torch.set_num_interop_threads(interop)
    except Exception:
        pass


def parse_temperatures(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return None
    return [float(v) for v in values]


def resolve_asr_runtime_config(
    args: argparse.Namespace,
    device: str,
) -> tuple[str, int, int, int, int, list[float]]:
    gpu_mem_gb = get_primary_gpu_memory_gb() if device == "cuda" and is_auto_env_tune_enabled(args) else None
    if args.speed_profile == "fast":
        chunk_cap = safe_cuda_chunk_cap(args.speed_profile, gpu_mem_gb)
        batch_cap = safe_cuda_batch_cap(args.speed_profile, gpu_mem_gb)
        chunk_default = min(28, chunk_cap) if device == "cuda" and chunk_cap is not None else 28
        batch_default = min(24, batch_cap) if device == "cuda" and batch_cap is not None else (24 if device == "cuda" else 8)
        beam_default = 1
        best_default = 1
        temperatures_default = [0.0]
    elif args.speed_profile == "accurate":
        chunk_cap = safe_cuda_chunk_cap(args.speed_profile, gpu_mem_gb)
        batch_cap = safe_cuda_batch_cap(args.speed_profile, gpu_mem_gb)
        chunk_default = min(20, chunk_cap) if device == "cuda" and chunk_cap is not None else 20
        batch_default = min(8, batch_cap) if device == "cuda" and batch_cap is not None else (8 if device == "cuda" else 2)
        beam_default = 5
        best_default = 5
        temperatures_default = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        chunk_cap = safe_cuda_chunk_cap(args.speed_profile, gpu_mem_gb)
        batch_cap = safe_cuda_batch_cap(args.speed_profile, gpu_mem_gb)
        chunk_default = min(30, chunk_cap) if device == "cuda" and chunk_cap is not None else 30
        batch_default = min(16, batch_cap) if device == "cuda" and batch_cap is not None else (16 if device == "cuda" else 4)
        beam_default = 3
        best_default = 3
        temperatures_default = [0.0, 0.2, 0.4]

    model = args.model or default_asr_model(args.speed_profile, device)
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else chunk_default
    chunk_size = max(MIN_TRANSCRIBE_CHUNK_SIZE, min(MAX_TRANSCRIBE_CHUNK_SIZE, chunk_size))
    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else batch_default
    beam_size = args.beam_size if args.beam_size and args.beam_size > 0 else beam_default
    best_of = args.best_of if args.best_of and args.best_of > 0 else best_default
    temperatures = parse_temperatures(args.temperatures) or temperatures_default

    return model, chunk_size, batch_size, beam_size, best_of, temperatures


def is_tensor_stack_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "stack expects each tensor to be equal size" in message


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def transcribe_with_fallback(
    *,
    worker_name: str,
    asr_model: Any,
    audio: Any,
    language: str | None,
    batch_size: int,
    chunk_size: int,
    heartbeat_sec: int,
    heartbeat_fp: Any | None,
) -> dict[str, Any]:
    try:
        return run_with_heartbeat(
            worker_name,
            "语音转写",
            heartbeat_sec,
            heartbeat_fp,
            asr_model.transcribe,
            audio,
            batch_size=batch_size,
            language=language,
            chunk_size=chunk_size,
        )
    except Exception as first_exc:  # noqa: BLE001
        if not (is_tensor_stack_mismatch_error(first_exc) or is_oom_error(first_exc)):
            raise

        if is_oom_error(first_exc):
            retry_batch_size = max(1, batch_size // 2)
            retry_chunk_size = max(MIN_TRANSCRIBE_CHUNK_SIZE, min(MAX_TRANSCRIBE_CHUNK_SIZE, chunk_size // 2 or 1))
            reason = "显存/内存不足"
        else:
            retry_batch_size = 1
            retry_chunk_size = max(MIN_TRANSCRIBE_CHUNK_SIZE, min(MAX_TRANSCRIBE_CHUNK_SIZE, chunk_size // 2 or 1))
            reason = "张量尺寸冲突"
        if retry_chunk_size == chunk_size and retry_batch_size == batch_size:
            raise

        print(
            f"[{worker_name}] [告警] 批量转写遇到{reason}，自动降级重试："
            f" batch {batch_size} -> {retry_batch_size}, chunk {chunk_size} -> {retry_chunk_size}"
        )
        try:
            return run_with_heartbeat(
                worker_name,
                "语音转写(降级重试)",
                heartbeat_sec,
                heartbeat_fp,
                asr_model.transcribe,
                audio,
                batch_size=retry_batch_size,
                language=language,
                chunk_size=retry_chunk_size,
            )
        except Exception as retry_exc:  # noqa: BLE001
            raise RuntimeError(
                "批量转写失败且降级重试失败。"
                f" 首次错误: {first_exc}; 重试错误: {retry_exc}"
            ) from retry_exc


def get_audio_duration_seconds(audio: Any) -> float | None:
    try:
        if hasattr(audio, "shape") and getattr(audio, "shape"):
            samples = int(audio.shape[-1])
        else:
            samples = int(len(audio))
    except Exception:
        return None
    if samples <= 0:
        return 0.0
    return float(samples) / float(WHISPERX_AUDIO_SAMPLE_RATE)


def get_diarization_skip_reason(audio: Any) -> str | None:
    duration = get_audio_duration_seconds(audio)
    if duration is None:
        return None
    if duration <= 0:
        return "音频为空"
    if duration < MIN_DIARIZATION_AUDIO_SECONDS:
        return f"音频过短（{duration:.2f}s < {MIN_DIARIZATION_AUDIO_SECONDS:.2f}s）"
    return None


def request_main_window_diarization_action(
    *,
    worker_name: str,
    file_label: str,
    reason: str,
    stage: str,
    args: argparse.Namespace,
) -> str:
    summary = f"{file_label} 无法执行说话人分离（{stage}: {reason}）"
    raw_interaction_dir = str(getattr(args, "interaction_dir", "") or "").strip()
    if not raw_interaction_dir:
        raise DiarizationAbortRequested(
            f"{summary}。当前任务没有主窗口交互通道，已安全中断。",
            file_label=file_label,
            stage=stage,
            reason=reason,
        )

    interaction_dir = Path(raw_interaction_dir).expanduser().resolve()
    interaction_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"diarization_{os.getpid()}_{uuid.uuid4().hex}"
    request_path = interaction_dir / f"{request_id}.json"
    request_payload = {
        "request_id": request_id,
        "type": "diarization-unavailable",
        "status": "pending",
        "worker_name": worker_name,
        "file_label": file_label,
        "stage": stage,
        "reason": reason,
        "summary": summary,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_file(request_path, request_payload)
    print(f"[{worker_name}] [主窗口交互] {summary}")
    print(f"[{worker_name}] [主窗口交互] 正在等待主窗口选择“仅转写继续”或“中断任务”。")

    while True:
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            write_json_file(request_path, request_payload)
            time.sleep(0.25)
            continue
        except Exception:
            time.sleep(0.25)
            continue

        if isinstance(payload, dict) and str(payload.get("status", "")) == "answered":
            response = str(payload.get("response", "")).strip()
            try:
                request_path.unlink(missing_ok=True)
                interaction_dir.rmdir()
            except Exception:
                pass
            if response == "transcribe-only":
                print(f"[{worker_name}] [主窗口交互] 已选择：当前文件改为仅转写并继续。")
                return "transcribe-only"
            if response == "abort":
                raise DiarizationAbortRequested(
                    f"{summary}。已在主窗口选择中断任务，请先完善音频文件后再继续。",
                    file_label=file_label,
                    stage=stage,
                    reason=reason,
                )
        time.sleep(0.25)


def resolve_diarization_unavailable_action(
    *,
    worker_name: str,
    file_label: str,
    reason: str,
    stage: str,
    args: argparse.Namespace,
) -> str:
    summary = f"{file_label} 无法执行说话人分离（{stage}: {reason}）"
    policy = str(getattr(args, "diarization_unavailable_policy", "ask") or "ask").strip().lower()
    if policy not in DIARIZATION_UNAVAILABLE_POLICIES:
        policy = "ask"

    if policy == "transcribe-only":
        print(f"[{worker_name}] [告警] {summary}，已按策略改为仅转写并继续。")
        return "transcribe-only"

    if policy == "abort":
        raise DiarizationAbortRequested(
            f"{summary}。已按策略中断任务，请先完善音频文件后再继续。",
            file_label=file_label,
            stage=stage,
            reason=reason,
        )

    return request_main_window_diarization_action(
        worker_name=worker_name,
        file_label=file_label,
        reason=reason,
        stage=stage,
        args=args,
    )


def run_diarization_pipeline(
    diarization_model: Any,
    audio: Any,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=PYANNOTE_STD_WARNING_PATTERN,
            category=UserWarning,
        )
        return diarization_model(
            audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )


def load_audio(path: str):
    from faster_whisper.audio import decode_audio
    try:
        audio = decode_audio(path, sampling_rate=16000)
    except Exception as exc:
        raise RuntimeError(f"无法读取音轨（{type(exc).__name__}），请检查文件是否损坏或没有音轨。") from exc
    if audio.size == 0:
        raise RuntimeError("文件没有可读取的音频样本。")
    return audio


def iter_audio_files(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in MEDIA_EXTENSIONS:
            yield file_path


def resolve_resume_from_file(
    files: list[Path],
    input_root: Path,
    resume_from_file: str | None,
) -> tuple[list[Path], Path | None]:
    raw = str(resume_from_file or "").strip()
    if not raw:
        return files, None

    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(input_root.resolve())
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"--resume-from-file 不在输入目录内: {candidate}") from exc
    else:
        relative = candidate

    target_key = relative.as_posix().lower().lstrip("./")
    for idx, file_path in enumerate(files):
        current_relative = file_path.relative_to(input_root)
        current_key = current_relative.as_posix().lower().lstrip("./")
        if current_key == target_key:
            return files[idx:], current_relative

    raise SystemExit(f"未在输入目录中找到恢复文件: {raw}")


def load_audio_files_from_worker_file_list(input_root: Path, worker_file_list: str) -> list[Path]:
    list_path = Path(worker_file_list).expanduser().resolve()
    if not list_path.exists():
        raise SystemExit(f"并行 worker 文件清单不存在: {list_path}")

    try:
        payload = json.loads(list_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"无法读取并行 worker 文件清单: {list_path}") from exc

    raw_files = payload.get("files", []) if isinstance(payload, dict) else []
    if not isinstance(raw_files, list):
        raise SystemExit(f"并行 worker 文件清单格式无效: {list_path}")

    selected: list[Path] = []
    input_root_resolved = input_root.resolve()
    for item in raw_files:
        relative = Path(str(item))
        candidate = (input_root / relative).resolve()
        try:
            candidate.relative_to(input_root_resolved)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"并行 worker 文件超出输入目录范围: {relative}") from exc
        if not candidate.exists():
            raise SystemExit(f"并行 worker 指定文件不存在: {relative}")
        selected.append(candidate)
    return selected


def partition_audio_files_balanced(files: list[Path], worker_count: int) -> list[list[Path]]:
    if worker_count <= 1 or len(files) <= 1:
        return [files]

    indexed_sizes: list[tuple[int, Path]] = []
    for file_path in files:
        try:
            size = int(file_path.stat().st_size)
        except Exception:
            size = 0
        indexed_sizes.append((size, file_path))
    indexed_sizes.sort(key=lambda item: item[0], reverse=True)

    buckets: list[list[Path]] = [[] for _ in range(worker_count)]
    bucket_sizes = [0] * worker_count
    for size, file_path in indexed_sizes:
        target = min(range(worker_count), key=lambda idx: bucket_sizes[idx])
        buckets[target].append(file_path)
        bucket_sizes[target] += max(1, size)

    return [bucket for bucket in buckets if bucket]


def to_jsonable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"JSON serialization error: {type(obj)!r}")


def segment_to_line(segment: dict[str, Any]) -> str | None:
    text = str(segment.get("text", "")).strip()
    if not text:
        return None
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    speaker = str(segment.get("speaker", "unknown"))
    return f"[{start:08.2f}-{end:08.2f}] {speaker}: {text}"


@lru_cache(maxsize=256)
def _old_audio_stems(directory: Path) -> Counter:
    return Counter(p.stem.casefold() for p in directory.iterdir()
                   if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file())


def get_output_paths(relative: Path, output_root: Path, input_root: Path | None = None) -> tuple[Path, Path]:
    name = relative.name
    if input_root is not None:
        old_names = _old_audio_stems((input_root / relative).parent)
        # 例：旧版 会议.mp4.wav 的结果也叫 会议.mp4.txt。歧义旧稿须原样保留。
        while name.casefold() in old_names:
            name += ".转写"
    target = output_root / relative.parent / name
    return target.with_name(name + ".txt"), target.with_name(name + ".json")


def output_exists_for_skip(txt_out: Path, json_out: Path, *, write_json: bool) -> bool:
    return txt_out.is_file() and (not write_json or json_out.is_file())


def pending_media_files(
    files: list[Path], input_root: Path, output_root: Path,
    *, write_json: bool, allow_legacy: bool,
) -> list[Path]:
    # 旧版音频输出省略扩展名。恢复起点之前的文件也要参与消歧。
    pending = []
    for source in files:
        relative = source.relative_to(input_root)
        txt, structured = get_output_paths(relative, output_root, input_root)
        complete = output_exists_for_skip(txt, structured, write_json=write_json)
        stem = source.with_suffix("")
        if (not complete and not txt.exists() and not structured.exists()
                and allow_legacy and source.suffix.lower() in AUDIO_EXTENSIONS
                and _old_audio_stems(source.parent)[source.stem.casefold()] == 1
                and stem.suffix.lower() not in MEDIA_EXTENSIONS):
            old_target = output_root / relative
            complete = output_exists_for_skip(old_target.with_suffix(".txt"), old_target.with_suffix(".json"), write_json=write_json)
        if complete:
            print(f"[调度] [跳过] {relative}（已存在）")
        else:
            pending.append(source)
    return pending



def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def write_result_files(
    *,
    result: dict[str, Any],
    txt_out: Path,
    json_out: Path,
    write_json: bool,
) -> None:
    lines = []
    for seg in result.get("segments", []):
        line = segment_to_line(seg)
        if line:
            lines.append(line)

    write_text_atomic(txt_out, "\n".join(lines))
    if write_json:
        write_text_atomic(
            json_out,
            json.dumps(result, ensure_ascii=False, indent=2, default=to_jsonable),
        )


def process_one_file(
    audio_file: Path,
    input_root: Path,
    output_root: Path,
    worker_name: str,
    asr_model: Any,
    diarization_model: Any,
    args: argparse.Namespace,
    batch_size: int,
    chunk_size: int,
    heartbeat_fp: Any | None,
) -> None:
    file_start = time.perf_counter()
    relative = audio_file.relative_to(input_root)
    relative_label = relative.as_posix()
    txt_out, json_out = get_output_paths(relative, output_root, input_root)
    txt_out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_exists_for_skip(txt_out, json_out, write_json=args.write_json):
        print(f"[{worker_name}] [跳过] {relative}（已存在）")
        return
    if args.skip_existing and txt_out.exists() and args.write_json and not json_out.exists():
        print(f"[{worker_name}] [续跑] {relative} 已有 TXT 但缺少 JSON，将重新处理补齐。")

    print(f"[{worker_name}] [开始] {relative}")
    print(f"[{worker_name}] [步骤] 1/4 读取音频 ...")
    audio = load_audio(str(audio_file))

    diarization_precheck_reason: str | None = None
    diarization_enabled_for_file = diarization_model is not None
    if diarization_enabled_for_file:
        diarization_precheck_reason = get_diarization_skip_reason(audio)
        if diarization_precheck_reason:
            resolve_diarization_unavailable_action(
                worker_name=worker_name,
                file_label=relative_label,
                reason=diarization_precheck_reason,
                stage="预检查",
                args=args,
            )
            diarization_enabled_for_file = False

    print(f"[{worker_name}] [步骤] 2/4 语音转写 ...")
    result = transcribe_with_fallback(
        worker_name=worker_name,
        asr_model=asr_model,
        audio=audio,
        language=args.language,
        batch_size=batch_size,
        chunk_size=chunk_size,
        heartbeat_sec=args.heartbeat_sec,
        heartbeat_fp=heartbeat_fp,
    )

    diarize_df = None
    if diarization_model is not None and diarization_enabled_for_file:
        print(f"[{worker_name}] [步骤] 3/4 说话人分离 ...")
        try:
            diarize_df = run_with_heartbeat(
                worker_name,
                "说话人分离",
                args.heartbeat_sec,
                heartbeat_fp,
                run_diarization_pipeline,
                diarization_model,
                audio,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
        except Exception as exc:  # noqa: BLE001
            brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
            resolve_diarization_unavailable_action(
                worker_name=worker_name,
                file_label=relative_label,
                reason=brief,
                stage="运行失败",
                args=args,
            )
    elif diarization_model is not None:
        print(
            f"[{worker_name}] [步骤] 3/4 跳过说话人分离"
            f"（已按选择改为仅转写；原因: {diarization_precheck_reason or '不可用'}）"
        )
    else:
        print(f"[{worker_name}] [步骤] 3/4 跳过说话人分离（未启用）")

    if diarize_df is not None:
        result = whisperx.assign_word_speakers(diarize_df, result, fill_nearest=True)

    print(f"[{worker_name}] [步骤] 4/4 写入输出文件 ...")
    write_result_files(
        result=result,
        txt_out=txt_out,
        json_out=json_out,
        write_json=args.write_json,
    )

    elapsed = time.perf_counter() - file_start
    print(f"[{worker_name}] [完成] {relative} -> {txt_out}")
    print(f"[{worker_name}] [耗时] {relative.name}: {format_seconds(elapsed)}")


def process_one_file_cooperative(
    audio_file: Path,
    input_root: Path,
    output_root: Path,
    worker_name: str,
    asr_model: Any,
    diarization_model: Any,
    args: argparse.Namespace,
    batch_size: int,
    chunk_size: int,
    stage_executor: concurrent.futures.ThreadPoolExecutor,
    heartbeat_fp: Any | None,
) -> None:
    file_start = time.perf_counter()
    relative = audio_file.relative_to(input_root)
    relative_label = relative.as_posix()
    txt_out, json_out = get_output_paths(relative, output_root, input_root)
    txt_out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_exists_for_skip(txt_out, json_out, write_json=args.write_json):
        print(f"[{worker_name}] [跳过] {relative}（已存在）")
        return
    if args.skip_existing and txt_out.exists() and args.write_json and not json_out.exists():
        print(f"[{worker_name}] [续跑] {relative} 已有 TXT 但缺少 JSON，将重新处理补齐。")

    print(f"[{worker_name}] [开始] {relative}")
    print(f"[{worker_name}] [步骤] 1/4 读取音频 ...")
    audio = load_audio(str(audio_file))

    diar_future: concurrent.futures.Future[Any] | None = None
    diarization_precheck_reason: str | None = None
    diarization_enabled_for_file = diarization_model is not None
    if diarization_enabled_for_file:
        diarization_precheck_reason = get_diarization_skip_reason(audio)
        if diarization_precheck_reason:
            resolve_diarization_unavailable_action(
                worker_name=worker_name,
                file_label=relative_label,
                reason=diarization_precheck_reason,
                stage="预检查",
                args=args,
            )
            diarization_enabled_for_file = False
            print(
                f"[{worker_name}] [步骤] 2/4 GPU 语音转写"
                f"（当前文件按选择仅转写；原因: {diarization_precheck_reason}） ..."
            )
        else:
            print(f"[{worker_name}] [步骤] 2/4 并行执行：GPU 语音转写 + CPU 说话人分离 ...")
            diar_future = stage_executor.submit(
                run_diarization_pipeline,
                diarization_model,
                audio,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
    else:
        print(f"[{worker_name}] [步骤] 2/4 GPU 语音转写 ...")

    result = transcribe_with_fallback(
        worker_name=worker_name,
        asr_model=asr_model,
        audio=audio,
        language=args.language,
        batch_size=batch_size,
        chunk_size=chunk_size,
        heartbeat_sec=args.heartbeat_sec,
        heartbeat_fp=heartbeat_fp,
    )

    diarize_df = None
    if diar_future is not None:
        print(f"[{worker_name}] [步骤] 3/4 等待 CPU 说话人分离结果 ...")
        wait_start = time.perf_counter()
        try:
            while True:
                try:
                    diarize_df = diar_future.result(timeout=max(2, int(args.heartbeat_sec)))
                    break
                except concurrent.futures.TimeoutError:
                    elapsed = time.perf_counter() - wait_start
                    heartbeat_write(
                        heartbeat_fp,
                        worker_name,
                        f"CPU 说话人分离进行中，已等待 {format_seconds(elapsed)}",
                    )
        except Exception as exc:  # noqa: BLE001
            brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
            resolve_diarization_unavailable_action(
                worker_name=worker_name,
                file_label=relative_label,
                reason=brief,
                stage="运行失败",
                args=args,
            )
    elif diarization_model is not None and not diarization_enabled_for_file:
        print(
            f"[{worker_name}] [步骤] 3/4 跳过说话人分离"
            f"（已按选择改为仅转写；原因: {diarization_precheck_reason or '不可用'}）"
        )
    else:
        print(f"[{worker_name}] [步骤] 3/4 跳过说话人分离（未启用）")

    if diarize_df is not None:
        result = whisperx.assign_word_speakers(diarize_df, result, fill_nearest=True)

    print(f"[{worker_name}] [步骤] 4/4 写入输出文件 ...")
    write_result_files(
        result=result,
        txt_out=txt_out,
        json_out=json_out,
        write_json=args.write_json,
    )

    elapsed = time.perf_counter() - file_start
    print(f"[{worker_name}] [完成] {relative} -> {txt_out}")
    print(f"[{worker_name}] [耗时] {relative.name}: {format_seconds(elapsed)}")


def build_worker(
    *,
    worker_name: str,
    device: str,
    args: argparse.Namespace,
    diarization_device: str | None = None,
) -> dict[str, Any]:
    compute_type = resolve_compute_type(device, args.compute_type)
    threads = resolve_threads(args.threads)
    configure_torch_threading(threads)
    model, chunk_size, batch_size, beam_size, best_of, temperatures = resolve_asr_runtime_config(
        args=args,
        device=device,
    )
    if args.chunk_size and args.chunk_size > MAX_TRANSCRIBE_CHUNK_SIZE:
        print(
            f"[{worker_name}] [提示] 你设置的 chunk_size={args.chunk_size} 过大，"
            f"已自动限制为 {chunk_size}。"
        )

    print(
        f"[{worker_name}] device={device}, compute_type={compute_type}, threads={threads}, "
        f"model={model}, batch={batch_size}, chunk={chunk_size}, beam={beam_size}, best_of={best_of}"
    )

    asr_options = {
        "beam_size": beam_size,
        "best_of": best_of,
        "temperatures": temperatures,
        "condition_on_previous_text": False,
    }
    asr_model = whisperx.load_model(
        str(local_model(model) or model),
        device=device,
        compute_type=compute_type,
        language=args.language,
        vad_method="silero",
        vad_model=load_local_vad(chunk_size),
        threads=threads,
        asr_options=asr_options,
    )
    print(f"[{worker_name}] ASR 模型加载完成。")

    diarization_runtime_device = diarization_device or device
    diarization_model = None
    if not args.disable_diarization:
        token = os.getenv("HF_TOKEN")
        if not diarization_available(token or "", args.diarization_model):
            if is_auto_env_tune_enabled(args):
                print(f"[{worker_name}] 未提供 HF_TOKEN，自动关闭说话人分离（仅转写文本）。")
            else:
                resolve_diarization_unavailable_action(
                    worker_name=worker_name,
                    file_label="任务初始化",
                    reason="未提供 Hugging Face Token",
                    stage="模型准备",
                    args=args,
                )
                print(f"[{worker_name}] 已按所选策略关闭说话人分离（仅转写文本）。")
        else:
            try:
                from whisperx.diarize import DiarizationPipeline

                diarization_model = DiarizationPipeline(
                    model_name=str(local_model(args.diarization_model) or args.diarization_model),
                    token=token,
                    device=diarization_runtime_device,
                )
                print(
                    f"[{worker_name}] 说话人分离模型加载完成。"
                    f" (device={diarization_runtime_device})"
                )
            except Exception as exc:  # noqa: BLE001
                brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
                if is_auto_env_tune_enabled(args):
                    print(
                        f"[{worker_name}] [告警] 说话人分离模型加载失败，自动降级为仅转写。"
                        f" 原因: {brief}"
                    )
                else:
                    resolve_diarization_unavailable_action(
                        worker_name=worker_name,
                        file_label="任务初始化",
                        reason=brief,
                        stage="说话人分离模型加载失败",
                        args=args,
                    )
                    print(f"[{worker_name}] 已按所选策略关闭说话人分离（仅转写文本）。")
                print(
                    f"[{worker_name}] [提示] 如需说话人分离，请先在 HuggingFace 接受模型条款："
                    f" https://huggingface.co/{args.diarization_model}"
                )
                diarization_model = None

    return {
        "name": worker_name,
        "device": device,
        "model": model,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "asr_model": asr_model,
        "diarization_model": diarization_model,
        "diarization_device": diarization_runtime_device,
    }


def run_worker_files(
    *,
    worker: dict[str, Any],
    files: list[Path],
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    heartbeat_fp: Any | None,
) -> list[str]:
    failed: list[str] = []
    total = len(files)
    print(f"[{worker['name']}] [批次] 开始处理，共 {total} 个文件。")
    batch_start = time.perf_counter()
    for idx, audio_file in enumerate(files, start=1):
        percent = (idx / total) * 100 if total > 0 else 100
        print(f"\n[{worker['name']}] [{idx}/{total}] 当前进度 {percent:.1f}%")
        try:
            process_one_file(
                audio_file=audio_file,
                input_root=input_root,
                output_root=output_root,
                worker_name=worker["name"],
                asr_model=worker["asr_model"],
                diarization_model=worker["diarization_model"],
                args=args,
                batch_size=worker["batch_size"],
                chunk_size=worker["chunk_size"],
                heartbeat_fp=heartbeat_fp,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"{audio_file}: {exc}"
            print(f"[{worker['name']}] [失败] {msg}")
            failed.append(msg)
    elapsed = time.perf_counter() - batch_start
    print(f"[{worker['name']}] [批次] 结束，耗时 {format_seconds(elapsed)}，失败 {len(failed)} 个。")
    return failed


def run_worker_files_cooperative(
    *,
    worker: dict[str, Any],
    files: list[Path],
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    heartbeat_fp: Any | None,
) -> list[str]:
    failed: list[str] = []
    total = len(files)
    print(f"[{worker['name']}] [批次] 开始处理，共 {total} 个文件。")
    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as stage_executor:
        for idx, audio_file in enumerate(files, start=1):
            percent = (idx / total) * 100 if total > 0 else 100
            print(f"\n[{worker['name']}] [{idx}/{total}] 当前进度 {percent:.1f}%")
            try:
                process_one_file_cooperative(
                    audio_file=audio_file,
                    input_root=input_root,
                    output_root=output_root,
                    worker_name=worker["name"],
                    asr_model=worker["asr_model"],
                    diarization_model=worker["diarization_model"],
                    args=args,
                    batch_size=worker["batch_size"],
                    chunk_size=worker["chunk_size"],
                    stage_executor=stage_executor,
                    heartbeat_fp=heartbeat_fp,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{audio_file}: {exc}"
                print(f"[{worker['name']}] [失败] {msg}")
                failed.append(msg)

    elapsed = time.perf_counter() - batch_start
    print(f"[{worker['name']}] [批次] 结束，耗时 {format_seconds(elapsed)}，失败 {len(failed)} 个。")
    return failed


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    set_console_title(args.process_label)
    log_fp, original_stdout, original_stderr = configure_log_mirroring(args.log_file)
    heartbeat_fp = configure_heartbeat_log(args.heartbeat_log_file)
    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    worker_result_status = "unknown"
    worker_result_failed: list[str] = []
    worker_result_total_files = 0
    worker_result_note = ""
    worker_label = (
        f"worker-{args.parallel_child_index}" if args.parallel_child_index > 0 else (args.process_label or "main")
    )
    try:
        if not input_root.is_dir():
            raise SystemExit(f"输入目录不存在或不是目录: {input_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        _old_audio_stems.cache_clear()
        resume_relative = None
        if args.worker_file_list:
            all_files = load_audio_files_from_worker_file_list(input_root, args.worker_file_list)
            audio_files = all_files
        else:
            all_files = sorted(iter_audio_files(input_root))
            audio_files, resume_relative = resolve_resume_from_file(all_files, input_root, args.resume_from_file) if all_files else ([], None)
        worker_result_total_files = len(audio_files)
        if not args.worker_file_list:
            print(f"待处理文件数: {worker_result_total_files}")
        if args.skip_existing:
            audio_files = pending_media_files(
                audio_files, input_root, output_root, write_json=args.write_json,
                allow_legacy=not args.worker_file_list,
            )
        if not audio_files:
            if worker_result_total_files and not args.worker_file_list:
                clear_last_diarization_abort_checkpoint()
            print("全部已完成，无需加载模型。" if worker_result_total_files else "未找到可处理的音频或视频文件。")
            print("处理结束。")
            print("全部成功。")
            worker_result_status = "ok"
            return

        if args.process_label:
            print(f"[进程标记] {args.process_label} | PID={os.getpid()}")
        if heartbeat_fp is not None:
            heartbeat_write(heartbeat_fp, "system", "心跳日志通道已就绪。")
        if configure_windows_runtime_guards():
            print("[环境防护] 已启用 Windows DLL 弹窗抑制。")
        else:
            print("[环境防护] Windows DLL 弹窗抑制未启用（非Windows或系统不支持）。")

        if has_torchcodec_distribution():
            print("[环境防护] 检测到 torchcodec 已安装。若再次出现 DLL 兼容弹窗，请卸载 torchcodec。")
        else:
            print("[环境防护] 未检测到 torchcodec，已规避该类 DLL 兼容弹窗。")

        env = detect_runtime_environment()
        print_runtime_environment(env)

        auto_env_tune = is_auto_env_tune_enabled(args)
        if auto_env_tune:
            adjustments, notices = apply_auto_env_tuning(args, env)
            print("[程序自动调节参数] 已启用。")
            if adjustments:
                for idx, item in enumerate(adjustments, start=1):
                    print(f"[程序自动调节参数] 调整{idx}: {item}")
            else:
                print("[程序自动调节参数] 本次无需调整参数。")
            for notice in notices:
                print(f"[程序自动调节参数] 提示: {notice}")
        else:
            print("[程序自动调节参数] 已关闭，将只校验参数，不自动修改你的设置。")
            notices = validate_runtime_config_without_auto_tune(args, env)
            for notice in notices:
                print(f"[参数校验] 提示: {notice}")

        if args.maximize_utilization:
            util_adjustments, util_notices = apply_accuracy_preserving_max_utilization_tuning(args, env)
            print("[高占用优化] 已启用（保精度 + 自动匹配硬件模式）。")
            if util_adjustments:
                for idx, item in enumerate(util_adjustments, start=1):
                    print(f"[高占用优化] 调整{idx}: {item}")
            else:
                print("[高占用优化] 本次无需额外调整。")
            for notice in util_notices:
                print(f"[高占用优化] 提示: {notice}")
        else:
            print("[高占用优化] 未启用。")

        resolved_device = resolve_device(args.device)
        use_cooperative_hybrid = should_use_cooperative_hybrid(args)
        runtime_devices = ["cuda(asr)", "cpu(diarization)"] if use_cooperative_hybrid else [resolved_device]

        print("========== 任务配置 ==========")
        print(f"输入目录: {input_root}")
        print(f"输出目录: {output_root}")
        print(f"语言: {args.language or 'auto'}, 速度档位: {args.speed_profile}")
        print(f"心跳日志间隔: {args.heartbeat_sec}秒")
        print(f"恢复起点: {args.resume_from_file or '无'}")
        print(
            "说话人分离不可用时: "
            + {
                "ask": "运行时询问",
                "transcribe-only": "改为仅转写",
                "abort": "立即中断任务",
            }.get(str(args.diarization_unavailable_policy), str(args.diarization_unavailable_policy))
        )
        print(f"程序自动调节参数: {'开启' if auto_env_tune else '关闭'}")
        print(f"高占用优化: {'开启' if args.maximize_utilization else '关闭'}")
        print(
            f"device设置: {args.device}, hybrid={args.hybrid}, "
            f"执行模式: {'CPU+GPU协同（同一文件分工并行）' if use_cooperative_hybrid else '单设备'}"
        )
        print(f"实际运行设备: {runtime_devices}")
        if use_cooperative_hybrid and args.disable_diarization:
            print("[提示] 你启用了 hybrid 但关闭了说话人分离，CPU 协同负载会较低。")

        if use_cooperative_hybrid and worker_result_total_files <= 1:
            print(
                "[模式建议] 检测到单文件任务，CPU+GPU协同会把说话人分离压到 CPU；"
                "本次不会自动切换，如需调整请在控制台手动改为仅GPU/单设备。"
            )

        parallel_worker_count = 1
        parallel_notices: list[str] = []
        if not args.worker_file_list:
            recommended_mode, recommendation_reason = describe_execution_mode_recommendation(
                file_count=len(audio_files),
                diarization_enabled=not args.disable_diarization,
                cuda_available=bool(env.get("cuda_available")),
            )
            if not auto_env_tune:
                validate_parallel_request_without_auto_tune(
                    requested_workers=args.parallel_workers,
                    file_count=worker_result_total_files,
                    resolved_device=resolved_device,
                    use_cooperative_hybrid=use_cooperative_hybrid,
                )
            parallel_worker_count, parallel_notices = resolve_parallel_worker_count(
                requested_workers=args.parallel_workers,
                file_count=len(audio_files),
                resolved_device=resolved_device,
                use_cooperative_hybrid=use_cooperative_hybrid,
                cpu_count=int(env.get("cpu_count", os.cpu_count() or 4)),
            )
            if resume_relative is not None:
                print(f"[恢复] 已定位恢复文件: {resume_relative.as_posix()}")
            print("[模式声明] 单个音频优先单 GPU；多文件批处理且启用说话人分离时，再考虑 CPU+GPU 协同。")
            print(f"[模式声明] 当前任务推荐：{recommended_mode}。{recommendation_reason}")
            print(f"[并行调度] 文件级并行 worker: {parallel_worker_count}")
            for notice in parallel_notices:
                print(f"[并行调度] 提示: {notice}")
            print("==============================")
        else:
            print("==============================")

        failed: list[str] = []
        if args.worker_file_list:
            if use_cooperative_hybrid:
                print("[阶段] 子 worker 正在加载协同 worker 模型 ...")
                worker_name = f"worker-{args.parallel_child_index or 1}-coop"
                try:
                    worker = build_worker(
                        worker_name=worker_name,
                        device="cuda",
                        args=args,
                        diarization_device="cpu",
                    )
                except Exception as exc:  # noqa: BLE001
                    brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
                    raise SystemExit(f"{worker_name} 初始化失败，任务终止。原因: {brief}") from exc

                print("[阶段] 子 worker 启动协同处理（同一文件内 GPU 转写 + CPU 分离）...")
                failed.extend(
                    run_worker_files_cooperative(
                        worker=worker,
                        files=audio_files,
                        input_root=input_root,
                        output_root=output_root,
                        args=args,
                        heartbeat_fp=heartbeat_fp,
                    )
                )
            else:
                print("[阶段] 子 worker 正在加载 worker 模型 ...")
                worker_name = f"worker-{args.parallel_child_index or 1}-{resolved_device}"
                try:
                    worker = build_worker(
                        worker_name=worker_name,
                        device=resolved_device,
                        args=args,
                    )
                except Exception as exc:  # noqa: BLE001
                    brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
                    raise SystemExit(f"{worker_name} 初始化失败，任务终止。原因: {brief}") from exc

                print("[阶段] 子 worker 启动单设备处理 ...")
                failed.extend(
                    run_worker_files(
                        worker=worker,
                        files=audio_files,
                        input_root=input_root,
                        output_root=output_root,
                        args=args,
                        heartbeat_fp=heartbeat_fp,
                    )
                )
        elif parallel_worker_count > 1:
            failed.extend(
                run_parallel_file_workers(
                    args=args,
                    audio_files=audio_files,
                    input_root=input_root,
                    output_root=output_root,
                    resolved_device=resolved_device,
                    worker_count=parallel_worker_count,
                    env=env,
                )
            )
        elif use_cooperative_hybrid:
            print("[阶段] 正在加载协同 worker 模型 ...")
            worker_name = "worker-coop"
            try:
                worker = build_worker(
                    worker_name=worker_name,
                    device="cuda",
                    args=args,
                    diarization_device="cpu",
                )
            except Exception as exc:  # noqa: BLE001
                brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
                raise SystemExit(f"协同 worker 初始化失败，任务终止。原因: {brief}") from exc

            print("[阶段] 启动协同处理（同一文件内 GPU 转写 + CPU 分离）...")
            failed.extend(
                run_worker_files_cooperative(
                    worker=worker,
                    files=audio_files,
                    input_root=input_root,
                    output_root=output_root,
                    args=args,
                    heartbeat_fp=heartbeat_fp,
                )
            )
        else:
            print("[阶段] 正在加载 worker 模型 ...")
            worker_name = f"worker-1-{resolved_device}"
            try:
                worker = build_worker(
                    worker_name=worker_name,
                    device=resolved_device,
                    args=args,
                )
            except Exception as exc:  # noqa: BLE001
                brief = str(exc).splitlines()[0] if str(exc) else repr(exc)
                raise SystemExit(f"{worker_name} 初始化失败，任务终止。原因: {brief}") from exc

            print("[阶段] 启动单设备处理 ...")
            failed.extend(
                run_worker_files(
                    worker=worker,
                    files=audio_files,
                    input_root=input_root,
                    output_root=output_root,
                    args=args,
                    heartbeat_fp=heartbeat_fp,
                )
            )

        total_elapsed = time.perf_counter() - overall_start
        print("\n处理结束。")
        if failed:
            print(f"失败数量: {len(failed)}")
            for item in failed:
                print(f"- {item}")
            worker_result_status = "partial-failed"
        else:
            clear_last_diarization_abort_checkpoint()
            print("全部成功。")
            worker_result_status = "ok"
        worker_result_failed = list(failed)
        print(f"总耗时: {format_seconds(total_elapsed)}")
    except DiarizationAbortRequested as exc:
        worker_result_status = "aborted"
        worker_result_note = str(exc)
        print(f"[任务中断] {worker_result_note}", file=sys.stderr)
        save_last_diarization_abort_checkpoint(
            input_root=input_root,
            output_root=output_root,
            args=args,
            exc=exc,
        )
        print(f"[恢复点] 已记录到 {LAST_DIARIZATION_ABORT_PATH}")
        raise
    except SystemExit as exc:
        if worker_result_status == "unknown":
            worker_result_status = "exited"
            worker_result_note = str(exc) or "SystemExit"
        if worker_result_note:
            print(f"[任务退出] {worker_result_note}", file=sys.stderr)
        raise
    except Exception as exc:  # noqa: BLE001
        if worker_result_status == "unknown":
            worker_result_status = "crashed"
            worker_result_note = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[任务崩溃] {worker_result_note}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        save_parallel_worker_result(
            args.worker_result_file,
            status=worker_result_status,
            failed=worker_result_failed,
            total_files=worker_result_total_files,
            worker_label=worker_label,
            note=worker_result_note,
        )
        if heartbeat_fp is not None:
            heartbeat_write(heartbeat_fp, "system", "心跳日志通道已关闭。")
            try:
                heartbeat_fp.close()
            except Exception:
                pass
        if log_fp is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                log_fp.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

