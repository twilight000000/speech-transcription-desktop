from __future__ import annotations

import base64
from model_resources import diarization_available, local_model, default_asr_model, load_local_vad
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


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

import torch

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_DIR / "语音识别项目.py"
REDACTED_SECRET = "<已隐藏>"

RUNTIME_DIR = PROJECT_DIR / ".runtime"
TASK_META_PATH = RUNTIME_DIR / "task_meta.json"
USER_SETTINGS_PATH = RUNTIME_DIR / "user_settings.json"
HF_TOKEN_SECRET_PATH = RUNTIME_DIR / "hf_token.dpapi"
LAST_DIARIZATION_ABORT_PATH = RUNTIME_DIR / "last_diarization_abort.json"

DEFAULT_INPUT_DIR = str(PROJECT_DIR / "input")
DEFAULT_OUTPUT_DIR = str(PROJECT_DIR / "output")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

TOTAL_PATTERN = re.compile(r"待处理文件数:\s*(\d+)")
STAGE_PATTERN = re.compile(r"\[阶段\]\s*(.+)")
STEP_PATTERN = re.compile(r"\[步骤\]\s*(.+)")
FAIL_COUNT_PATTERN = re.compile(r"失败数量:\s*(\d+)")
PERSISTED_SETTING_KEYS = {
    "task_input_dir",
    "task_output_dir",
    "task_language_search",
    "task_language_code",
    "task_precision_mode",
    "task_device_mode",
    "task_utilization",
    "task_heartbeat_sec",
    "task_enable_diarization",
    "task_parallel_workers",
    "task_diarization_unavailable_policy",
    "task_resume_from_file",
    "task_write_json",
    "task_skip_existing",
    "task_auto_env_tune",
    "task_maximize_utilization",
    "ui_auto_refresh",
    "ui_refresh_interval_sec",
    "ui_log_tail_lines",
    "ui_log_filter_keyword",
}

LANGUAGE_NAME_ZH: dict[str, str] = {
    "af": "南非荷兰语",
    "am": "阿姆哈拉语",
    "ar": "阿拉伯语",
    "as": "阿萨姆语",
    "az": "阿塞拜疆语",
    "ba": "巴什基尔语",
    "be": "白俄罗斯语",
    "bg": "保加利亚语",
    "bn": "孟加拉语",
    "bo": "藏语",
    "br": "布列塔尼语",
    "bs": "波斯尼亚语",
    "ca": "加泰罗尼亚语",
    "cs": "捷克语",
    "cy": "威尔士语",
    "da": "丹麦语",
    "de": "德语",
    "el": "希腊语",
    "en": "英语",
    "es": "西班牙语",
    "et": "爱沙尼亚语",
    "eu": "巴斯克语",
    "fa": "波斯语",
    "fi": "芬兰语",
    "fo": "法罗语",
    "fr": "法语",
    "gl": "加利西亚语",
    "gu": "古吉拉特语",
    "ha": "豪萨语",
    "haw": "夏威夷语",
    "he": "希伯来语",
    "hi": "印地语",
    "hr": "克罗地亚语",
    "ht": "海地克里奥尔语",
    "hu": "匈牙利语",
    "hy": "亚美尼亚语",
    "id": "印度尼西亚语",
    "is": "冰岛语",
    "it": "意大利语",
    "ja": "日语",
    "jw": "爪哇语",
    "ka": "格鲁吉亚语",
    "kk": "哈萨克语",
    "km": "高棉语",
    "kn": "卡纳达语",
    "ko": "韩语",
    "la": "拉丁语",
    "lb": "卢森堡语",
    "ln": "林加拉语",
    "lo": "老挝语",
    "lt": "立陶宛语",
    "lv": "拉脱维亚语",
    "mg": "马达加斯加语",
    "mi": "毛利语",
    "mk": "马其顿语",
    "ml": "马拉雅拉姆语",
    "mn": "蒙古语",
    "mr": "马拉地语",
    "ms": "马来语",
    "mt": "马耳他语",
    "my": "缅甸语",
    "ne": "尼泊尔语",
    "nl": "荷兰语",
    "nn": "新挪威语",
    "no": "挪威语",
    "oc": "奥克语",
    "pa": "旁遮普语",
    "pl": "波兰语",
    "ps": "普什图语",
    "pt": "葡萄牙语",
    "ro": "罗马尼亚语",
    "ru": "俄语",
    "sa": "梵语",
    "sd": "信德语",
    "si": "僧伽罗语",
    "sk": "斯洛伐克语",
    "sl": "斯洛文尼亚语",
    "sn": "绍纳语",
    "so": "索马里语",
    "sq": "阿尔巴尼亚语",
    "sr": "塞尔维亚语",
    "su": "巽他语",
    "sv": "瑞典语",
    "sw": "斯瓦希里语",
    "ta": "泰米尔语",
    "te": "泰卢固语",
    "tg": "塔吉克语",
    "th": "泰语",
    "tk": "土库曼语",
    "tl": "他加禄语",
    "tr": "土耳其语",
    "tt": "鞑靼语",
    "uk": "乌克兰语",
    "ur": "乌尔都语",
    "uz": "乌兹别克语",
    "vi": "越南语",
    "yi": "意第绪语",
    "yo": "约鲁巴语",
    "zh": "中文",
    "yue": "粤语",
}

COMMON_LANGUAGE_CODES = [
    "zh",
    "yue",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "ru",
    "ar",
    "pt",
    "it",
    "hi",
    "th",
    "vi",
]

FALLBACK_LANGUAGE_CODES = tuple(LANGUAGE_NAME_ZH.keys())


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob_from_bytes(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    return blob, buffer


def _dpapi_protect(data: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Token 本机加密仅支持 Windows。")

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    input_blob, input_buffer = _blob_from_bytes(data)
    output_blob = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "WhisperX Hugging Face Token",
        None,
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        if output_blob.pbData:
            kernel32.LocalFree(output_blob.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Token 本机解密仅支持 Windows。")

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    input_blob, input_buffer = _blob_from_bytes(data)
    output_blob = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        if output_blob.pbData:
            kernel32.LocalFree(output_blob.pbData)


def load_hf_token() -> str:
    """Load the locally remembered token for the current Windows user."""
    if not HF_TOKEN_SECRET_PATH.exists():
        return ""
    try:
        payload = json.loads(HF_TOKEN_SECRET_PATH.read_text(encoding="utf-8"))
        if payload.get("scheme") != "windows-dpapi-current-user":
            return ""
        encrypted = base64.b64decode(str(payload["ciphertext"]), validate=True)
        return _dpapi_unprotect(encrypted).decode("utf-8").strip()
    except Exception:
        return ""


def save_hf_token(token: str) -> None:
    """Remember a token encrypted by Windows DPAPI, or forget it when blank."""
    token = str(token).strip()
    if not token:
        HF_TOKEN_SECRET_PATH.unlink(missing_ok=True)
        return

    ensure_runtime_dir()
    encrypted = _dpapi_protect(token.encode("utf-8"))
    payload = {
        "version": 1,
        "scheme": "windows-dpapi-current-user",
        "ciphertext": base64.b64encode(encrypted).decode("ascii"),
    }
    temp_path = HF_TOKEN_SECRET_PATH.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp_path, HF_TOKEN_SECRET_PATH)
    finally:
        temp_path.unlink(missing_ok=True)


def load_task_meta() -> dict[str, Any] | None:
    if not TASK_META_PATH.exists():
        return None
    try:
        data = json.loads(TASK_META_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            cmd = data.get("cmd")
            if isinstance(cmd, list):
                redacted_cmd = redact_command_secrets([str(item) for item in cmd])
                if redacted_cmd != cmd:
                    data["cmd"] = redacted_cmd
                    try:
                        save_task_meta(data)
                    except Exception:
                        pass
            return data
    except Exception:
        return None
    return None


def load_last_diarization_abort() -> dict[str, Any] | None:
    if not LAST_DIARIZATION_ABORT_PATH.exists():
        return None
    try:
        data = json.loads(LAST_DIARIZATION_ABORT_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def save_task_meta(meta: dict[str, Any]) -> None:
    ensure_runtime_dir()
    write_json_atomic(TASK_META_PATH, meta)


def load_user_settings() -> dict[str, Any]:
    if not USER_SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in PERSISTED_SETTING_KEYS if key in data}


def save_user_settings(settings: dict[str, Any]) -> None:
    ensure_runtime_dir()
    payload = {key: settings[key] for key in PERSISTED_SETTING_KEYS if key in settings}
    write_json_atomic(USER_SETTINGS_PATH, payload)


def resolve_backend_python_executable() -> str:
    executable = Path(sys.executable)
    if executable.name.lower() in {"pythonw.exe", "pyw.exe"}:
        python_exe = executable.with_name("python.exe")
        if python_exe.exists():
            return str(python_exe)
    return sys.executable


def redact_command_secrets(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for token in [str(item) for item in cmd]:
        if skip_next:
            redacted.append(REDACTED_SECRET)
            skip_next = False
            continue
        redacted.append(token)
        if token == "--hf-token":
            skip_next = True
    return redacted


def format_command_for_log(cmd: list[str]) -> str:
    try:
        return subprocess.list2cmdline([str(item) for item in redact_command_secrets(cmd)])
    except Exception:
        return " ".join(redact_command_secrets(cmd))


def build_hidden_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}

    kwargs: dict[str, Any] = {}
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    if creationflags:
        kwargs["creationflags"] = creationflags

    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0) or 0)
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0) or 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs


@contextmanager
def _live_process(pid: int):
    """Keep the process object alive while checking or controlling its PID."""
    if os.name != "nt" or pid <= 0:
        yield None
        return
    import _winapi
    try:
        handle = _winapi.OpenProcess(0x1000 | 0x100000, False, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    except OSError as exc:
        if exc.winerror not in (87, 1168):  # invalid PID / process gone
            raise
        yield None
        return
    try:
        yield handle if _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT else None
    finally:
        _winapi.CloseHandle(handle)


def _creation_ticks(handle: int) -> int:
    times = [wintypes.FILETIME() for _ in range(4)]
    if not ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(handle), *(ctypes.byref(item) for item in times)):
        return 0
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def _matches_creation(expected: str, ticks: int) -> bool:
    if not expected or not ticks:
        return False
    if expected.startswith("ft:"):
        return expected == f"ft:{ticks}"
    # Older task snapshots contain a CIM timestamp with microsecond precision.
    try:
        recorded = datetime.fromisoformat(expected).astimezone(timezone.utc)
        born = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)
        return recorded == born
    except (ValueError, OverflowError):
        return False


def get_process_creation_date(pid: int) -> str:
    try:
        with _live_process(pid) as handle:
            ticks = _creation_ticks(handle) if handle else 0
        return f"ft:{ticks}" if ticks else ""
    except (OSError, ValueError, OverflowError):
        return ""


def tracked_task_state(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "stopped"
    try:
        with _live_process(int(meta.get("pid", 0))) as handle:
            if handle is None:
                return "stopped"
            expected = str(meta.get("process_creation_date") or "")
            ticks = _creation_ticks(handle)
            if not expected or not ticks:
                return "unknown"
            return "running" if _matches_creation(expected, ticks) else "stopped"
    except OSError:
        return "unknown"
    except (ValueError, TypeError, OverflowError):
        return "stopped"


def _control_verified_task(meta: dict[str, Any], action, *args) -> tuple[bool, str]:
    try:
        pid = int(meta.get("pid", 0))
        with _live_process(pid) as handle:
            if not handle or not _matches_creation(str(meta.get("process_creation_date") or ""), _creation_ticks(handle)):
                return False, "任务已结束或无法核对进程身份，已取消操作。"
            return action(pid, *args)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return False, f"无法核对或操作任务进程：{exc}"


def kill_pid(pid: int) -> tuple[bool, str]:
    if pid <= 0:
        return False, "无效 PID。"
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **build_hidden_subprocess_kwargs(),
    )
    ok = result.returncode == 0
    message = (result.stdout or result.stderr or "").strip()
    return ok, (message or "已发送停止指令。")


def kill_tracked_task(task_meta: dict[str, Any]) -> tuple[bool, str]:
    return _control_verified_task(task_meta, kill_pid)


def set_pid_tree_paused(pid: int, pause: bool) -> tuple[bool, str]:
    if pid <= 0:
        return False, "无效 PID。"

    script = f"""
$ErrorActionPreference = 'Stop'
$root = {pid}
$pids = New-Object System.Collections.Generic.List[int]
$queue = New-Object System.Collections.Generic.Queue[int]
$pids.Add($root)
$queue.Enqueue($root)
while ($queue.Count -gt 0) {{
  $cur = $queue.Dequeue()
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$cur"
  foreach ($child in $children) {{
    $cid = [int]$child.ProcessId
    if (-not $pids.Contains($cid)) {{
      $pids.Add($cid)
      $queue.Enqueue($cid)
    }}
  }}
}}
$pids | Sort-Object -Unique -Descending | ForEach-Object {{ Write-Output $_ }}
"""
    tree_result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **build_hidden_subprocess_kwargs(),
    )
    if tree_result.returncode != 0:
        return False, (tree_result.stdout or tree_result.stderr or "").strip() or "读取进程树失败。"

    pid_list: list[int] = []
    for line in (tree_result.stdout or "").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            value = int(text)
            if value > 0:
                pid_list.append(value)
        except ValueError:
            continue
    if not pid_list:
        pid_list = [pid]

    try:
        import ctypes
        from ctypes import wintypes

        process_suspend_resume = 0x0800
        process_query_limited_info = 0x1000
        access_mask = process_suspend_resume | process_query_limited_info

        kernel32 = ctypes.windll.kernel32
        ntdll = ctypes.windll.ntdll

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        nt_suspend = ntdll.NtSuspendProcess
        nt_suspend.argtypes = [wintypes.HANDLE]
        nt_suspend.restype = ctypes.c_long

        nt_resume = ntdll.NtResumeProcess
        nt_resume.argtypes = [wintypes.HANDLE]
        nt_resume.restype = ctypes.c_long
    except Exception as exc:
        return False, f"加载系统挂起接口失败: {exc}"

    affected = 0
    skipped: list[str] = []
    action_name = "暂停" if pause else "继续"
    for target_pid in sorted(set(pid_list), reverse=True):
        handle = kernel32.OpenProcess(access_mask, False, int(target_pid))
        if not handle:
            skipped.append(f"{target_pid}: OpenProcess失败")
            continue
        try:
            status = nt_suspend(handle) if pause else nt_resume(handle)
            if int(status) == 0:
                affected += 1
            else:
                skipped.append(f"{target_pid}: NTSTATUS=0x{int(status) & 0xFFFFFFFF:08X}")
        finally:
            kernel32.CloseHandle(handle)

    if affected > 0:
        extra = f"，跳过 {len(skipped)} 个" if skipped else ""
        return True, f"已{action_name}任务进程树（影响 {affected} 个进程{extra}）。"
    if skipped:
        return False, "；".join(skipped[:3])
    return False, f"{action_name}任务失败。"


def set_tracked_task_paused(task_meta: dict[str, Any], pause: bool) -> tuple[bool, str]:
    return _control_verified_task(task_meta, set_pid_tree_paused, pause)


def parse_cli_tokens(tokens: list[str]) -> tuple[dict[str, str], set[str]]:
    values: dict[str, str] = {}
    flags: set[str] = set()
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx])
        if token.startswith("--"):
            if idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
                values[token] = str(tokens[idx + 1])
                idx += 2
            else:
                flags.add(token)
                idx += 1
        else:
            idx += 1
    return values, flags


def count_audio_files(input_dir: str, limit: int | None = None) -> int | None:
    try:
        root = Path(str(input_dir)).expanduser()
        if not root.exists() or not root.is_dir():
            return None
        count = 0
        for p in root.rglob("*"):
            if p.suffix.lower() in MEDIA_EXTENSIONS and p.is_file():
                count += 1
                if limit is not None and count >= limit:
                    break
        return count
    except Exception:
        return None


def recommend_device_mode(input_dir: str, enable_diarization: bool) -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "仅CPU", "未检测到 CUDA，只能使用单设备 CPU。"

    if not enable_diarization:
        return "仅GPU", "当前未启用说话人分离，单 GPU 更直接。"
    file_count = count_audio_files(input_dir, limit=2)
    if file_count == 1:
        return "仅GPU", "单个音频尤其长音频时，优先单 GPU，避免协同模式把说话人分离压到 CPU。"
    if isinstance(file_count, int) and file_count > 1:
        return "CPU+GPU协同", "多文件批处理且启用说话人分离时，协同模式更适合提高总吞吐。"
    return "仅GPU", "默认优先单 GPU；只有多文件批处理且启用说话人分离时，再考虑 CPU+GPU 协同。"


def get_supported_language_codes() -> list[str]:
    try:
        from faster_whisper import tokenizer as fw_tokenizer

        raw_codes = getattr(fw_tokenizer, "_LANGUAGE_CODES", None)
        if raw_codes:
            normalized = [str(code).lower() for code in raw_codes if str(code).strip()]
            return list(dict.fromkeys(normalized))
    except Exception:
        pass
    return list(FALLBACK_LANGUAGE_CODES)


def language_label(code: str) -> str:
    if not code:
        return "自动检测"
    zh_name = LANGUAGE_NAME_ZH.get(code, "未知语言")
    return f"{zh_name}（{code}）"


def build_language_code_options(search_keyword: str, current_code: str) -> list[str]:
    supported = get_supported_language_codes()
    common = [code for code in COMMON_LANGUAGE_CODES if code in supported]
    remaining = [code for code in supported if code not in common]
    ordered = common + remaining

    if current_code and current_code not in ordered:
        ordered.insert(0, current_code)

    keyword = search_keyword.strip().lower()
    if keyword:
        filtered = [
            code
            for code in ordered
            if keyword in code.lower() or keyword in language_label(code).lower()
        ]
    else:
        filtered = ordered
    return [""] + filtered


def resolve_speed_profile(precision_mode: str) -> str:
    if precision_mode == "高精度":
        return "accurate"
    if precision_mode == "高速":
        return "fast"
    return "balanced"


def resolve_utilization_preset(level: str, cpu_threads: int, speed_profile: str) -> dict[str, Any]:
    if speed_profile == "accurate":
        if level == "低占用":
            return {"threads": max(1, int(cpu_threads * 0.35)), "batch_size": 2, "chunk_size": 18}
        if level == "性能拉满":
            return {"threads": max(1, int(cpu_threads * 0.95)), "batch_size": 8, "chunk_size": 20}
        return {"threads": max(1, int(cpu_threads * 0.7)), "batch_size": 4, "chunk_size": 20}

    if speed_profile == "fast":
        if level == "低占用":
            return {"threads": max(1, int(cpu_threads * 0.35)), "batch_size": 2, "chunk_size": 20}
        if level == "性能拉满":
            return {"threads": max(1, int(cpu_threads * 0.95)), "batch_size": 16, "chunk_size": 28}
        return {"threads": max(1, int(cpu_threads * 0.7)), "batch_size": 8, "chunk_size": 24}

    if level == "低占用":
        return {"threads": max(1, int(cpu_threads * 0.35)), "batch_size": 2, "chunk_size": 20}
    if level == "性能拉满":
        return {"threads": max(1, int(cpu_threads * 0.95)), "batch_size": 12, "chunk_size": 24}
    return {"threads": max(1, int(cpu_threads * 0.7)), "batch_size": 6, "chunk_size": 22}


def resolve_device_mode(mode: str) -> tuple[str, bool]:
    if mode == "仅CPU":
        return "cpu", False
    if mode == "仅GPU":
        return "cuda", False
    if mode in {"CPU+GPU协同", "CPU+GPU混合"}:
        return "auto", True
    return "auto", False


def build_command(
    *,
    input_dir: str,
    output_dir: str,
    language: str,
    precision_mode: str,
    utilization: str,
    device_mode: str,
    parallel_workers: int,
    resume_from_file: str,
    enable_diarization: bool,
    diarization_unavailable_policy: str,
    hf_token: str,
    write_json: bool,
    skip_existing: bool,
    auto_env_tune: bool,
    maximize_utilization: bool,
    heartbeat_sec: int,
) -> list[str]:
    cpu_threads = os.cpu_count() or 4
    speed_profile = resolve_speed_profile(precision_mode)
    preset = resolve_utilization_preset(utilization, cpu_threads, speed_profile)
    device, hybrid = resolve_device_mode(device_mode)
    effective_maximize_utilization = bool(maximize_utilization) and str(utilization).strip() != "低占用"

    cmd = [
        resolve_backend_python_executable(),
        "-u",
        str(SCRIPT_PATH),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--speed-profile",
        speed_profile,
        "--device",
        device,
        "--threads",
        str(preset["threads"]),
        "--parallel-workers",
        str(max(0, int(parallel_workers))),
        "--batch-size",
        str(preset["batch_size"]),
        "--chunk-size",
        str(preset["chunk_size"]),
        "--heartbeat-sec",
        str(max(2, int(heartbeat_sec))),
        "--diarization-unavailable-policy",
        {
            "自动仅转写": "transcribe-only",
            "立即中断任务": "abort",
        }.get(str(diarization_unavailable_policy), "ask"),
    ]

    if language.strip():
        cmd += ["--language", language.strip()]
    if resume_from_file.strip():
        cmd += ["--resume-from-file", resume_from_file.strip()]

    if hybrid:
        cmd.append("--hybrid")
    if write_json:
        cmd.append("--write-json")
    if skip_existing:
        cmd.append("--skip-existing")
    if auto_env_tune:
        cmd.append("--auto-env-tune")
    if effective_maximize_utilization:
        cmd.append("--maximize-utilization")
    if not enable_diarization:
        cmd.append("--disable-diarization")

    return cmd


def get_task_start_issues(cmd: list[str], hf_token: str) -> list[str]:
    """Return predictable configuration problems before a backend is created."""
    values, flags = parse_cli_tokens([str(item) for item in cmd])
    issues: list[str] = []
    diarization_enabled = "--disable-diarization" not in flags
    auto_env_tune_enabled = (
        "--auto-env-tune" in flags
        and "--disable-auto-env-tune" not in flags
    )

    if (diarization_enabled and not diarization_available(hf_token, values.get("--diarization-model", "pyannote/speaker-diarization-community-1"))
            and not auto_env_tune_enabled):
        issues.append(
            "说话人模型尚未准备，且未填写 Hugging Face Token；请在“首次使用 / 模型”中完成下载，"
            "或者关闭说话人分离后仅进行语音转文字。"
        )

    input_raw = str(values.get("--input-dir", "")).strip()
    input_root: Path | None = None
    file_count: int | None = None
    if not input_raw:
        issues.append("没有填写输入音视频目录。")
    else:
        try:
            input_root = Path(input_raw).expanduser().resolve()
            if not input_root.exists():
                issues.append(f"输入目录不存在：{input_root}")
            elif not input_root.is_dir():
                issues.append(f"输入路径不是文件夹：{input_root}")
            else:
                file_count = count_audio_files(str(input_root))
                if file_count == 0:
                    issues.append("输入目录中没有可处理的音频或视频文件。")
        except Exception as exc:
            issues.append(f"无法检查输入目录：{exc}")

    output_raw = str(values.get("--output-dir", "")).strip()
    if not output_raw:
        issues.append("没有填写输出目录。")
    else:
        try:
            output_root = Path(output_raw).expanduser()
            if output_root.exists() and not output_root.is_dir():
                issues.append(f"输出路径已经存在，但它不是文件夹：{output_root}")
        except Exception as exc:
            issues.append(f"无法检查输出目录：{exc}")

    resume_raw = str(values.get("--resume-from-file", "")).strip()
    if resume_raw and input_root is not None and input_root.exists() and input_root.is_dir():
        try:
            candidate = Path(resume_raw).expanduser()
            if not candidate.is_absolute():
                candidate = input_root / candidate
            resolved_candidate = candidate.resolve()
            try:
                resolved_candidate.relative_to(input_root)
            except ValueError:
                issues.append("恢复文件必须位于输入目录内。")
            else:
                if not resolved_candidate.exists() or not resolved_candidate.is_file():
                    issues.append(f"没有找到指定的恢复文件：{resolved_candidate}")
                elif resolved_candidate.suffix.lower() not in MEDIA_EXTENSIONS:
                    issues.append(f"恢复文件不是支持的音频或视频格式：{resolved_candidate.name}")
        except Exception as exc:
            issues.append(f"无法检查恢复文件：{exc}")

    if not auto_env_tune_enabled:
        cuda_available = bool(torch.cuda.is_available())
        device = str(values.get("--device", "auto")).strip().lower()
        hybrid = "--hybrid" in flags
        if device == "cuda" and not cuda_available:
            issues.append("当前没有可用 CUDA，不能选择“仅GPU”。")
        if hybrid and not cuda_available:
            issues.append("当前没有可用 CUDA，不能使用“CPU+GPU协同”。")

        try:
            requested_workers: int | None = int(values.get("--parallel-workers", "0"))
        except Exception:
            requested_workers = None
            issues.append("文件级并行 worker 必须是整数。")

        if requested_workers is not None and requested_workers < 0:
            issues.append("文件级并行 worker 不能小于 0。")
        elif requested_workers is not None and requested_workers > 1:
            resolved_gpu = device == "cuda" or (device == "auto" and cuda_available)
            if hybrid:
                issues.append("CPU+GPU 协同模式不支持多个文件级 worker；请改为 0 或 1。")
            elif resolved_gpu:
                issues.append("单 GPU 模式不支持多个文件级 worker；请改为 0 或 1。")
            if isinstance(file_count, int) and requested_workers > file_count:
                issues.append(
                    f"文件级并行 worker={requested_workers} 大于待处理文件数 {file_count}。"
                )

    return issues


def get_task_start_blocker(cmd: list[str], hf_token: str) -> str:
    """Return a user-facing reason why the controller must not start a task."""
    issues = get_task_start_issues(cmd, hf_token)
    if issues:
        return (
            "任务尚未启动，后端不会被打开。\n\n发现以下问题：\n"
            + "\n".join(f"• {item}" for item in issues)
            + "\n\n请修改后重新点击“开始任务”。"
        )
    return ""


def load_task_result(task_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not task_meta:
        return None
    raw_path = str(task_meta.get("result_path", "")).strip()
    if not raw_path:
        return None
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def list_pending_task_interactions(task_meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not task_meta:
        return []
    raw_dir = str(task_meta.get("interaction_dir", "")).strip()
    if not raw_dir:
        return []
    interaction_dir = Path(raw_dir)
    if not interaction_dir.exists() or not interaction_dir.is_dir():
        return []

    pending: list[dict[str, Any]] = []
    for request_path in interaction_dir.glob("*.json"):
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, dict) or str(payload.get("status", "")) != "pending":
            continue
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id or request_path.name != f"{request_id}.json":
            continue
        pending.append(payload)
    return sorted(pending, key=lambda item: str(item.get("created_at", "")))


def respond_to_task_interaction(
    task_meta: dict[str, Any] | None,
    request_id: str,
    response: str,
) -> tuple[bool, str]:
    if response not in {"transcribe-only", "abort"}:
        return False, "不支持的交互选择。"
    if not task_meta:
        return False, "没有当前任务信息。"
    raw_dir = str(task_meta.get("interaction_dir", "")).strip()
    if not raw_dir:
        return False, "当前任务没有主窗口交互通道。"
    request_id = str(request_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
        return False, "交互请求编号无效。"

    interaction_dir = Path(raw_dir).resolve()
    request_path = (interaction_dir / f"{request_id}.json").resolve()
    if request_path.parent != interaction_dir or not request_path.exists():
        return False, "交互请求已经失效或不存在。"
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or str(payload.get("status", "")) != "pending":
            return False, "交互请求已经处理。"
        payload["status"] = "answered"
        payload["response"] = response
        payload["answered_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(request_path, payload)
    except Exception as exc:
        return False, f"无法提交选择：{exc}"
    return True, "选择已提交给后端。"


def start_task(cmd: list[str], hf_token: str) -> dict[str, Any]:
    blocker = get_task_start_blocker(cmd, hf_token)
    if blocker:
        raise ValueError(blocker)

    ensure_runtime_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_path = RUNTIME_DIR / f"run_{ts}.log"
    heartbeat_log_path = RUNTIME_DIR / f"run_{ts}.heartbeat.log"
    result_path = RUNTIME_DIR / f"run_{ts}.result.json"
    interaction_dir = RUNTIME_DIR / f"interaction_{ts}"
    interaction_dir.mkdir(parents=True, exist_ok=False)
    task_label = f"语音识别后端任务[{ts}]"
    task_cmd = [
        *cmd,
        "--log-file",
        str(log_path),
        "--heartbeat-log-file",
        str(heartbeat_log_path),
        "--process-label",
        task_label,
        "--worker-result-file",
        str(result_path),
        "--interaction-dir",
        str(interaction_dir),
    ]
    log_path.write_text(
        "========== 任务启动 ==========\n"
        f"时间: {datetime.now().isoformat(timespec='seconds')}\n"
        f"任务标记: {task_label}\n"
        f"命令: {format_command_for_log(task_cmd)}\n"
        "==============================\n\n",
        encoding="utf-8",
    )
    heartbeat_log_path.write_text(
        "========== 心跳日志启动 ==========\n"
        f"时间: {datetime.now().isoformat(timespec='seconds')}\n"
        f"任务标记: {task_label}\n"
        "=================================\n\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if hf_token.strip():
        env["HF_TOKEN"] = hf_token.strip()

    creationflags = (
        getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    )
    startupinfo = None
    if os.name == "nt" and hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.lpTitle = task_label
    try:
        process = subprocess.Popen(
            task_cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
    except OSError as exc:
        fallback_flags = (
            getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        process = subprocess.Popen(
            task_cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            creationflags=fallback_flags,
            startupinfo=startupinfo,
        )
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(
                "[控制台告警] CREATE_BREAKAWAY_FROM_JOB 启动失败，"
                "已自动回退为 CREATE_NEW_CONSOLE + CREATE_NEW_PROCESS_GROUP。\n"
                f"[控制台告警] 失败原因: {exc}\n\n"
            )

    meta = {
        "pid": process.pid,
        "task_label": task_label,
        "paused": False,
        "log_path": str(log_path),
        "heartbeat_log_path": str(heartbeat_log_path),
        "result_path": str(result_path),
        "interaction_dir": str(interaction_dir),
        "result_notified": False,
        "stop_requested": False,
        "cmd": redact_command_secrets(task_cmd),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "process_creation_date": get_process_creation_date(process.pid),
    }
    save_task_meta(meta)
    return meta


def read_log_tail(log_path: Path, max_lines: int = 500) -> list[str]:
    if not log_path.exists():
        return []
    if max_lines <= 0:
        return []

    try:
        with log_path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            file_size = fp.tell()
            block_size = 8192
            remaining = file_size
            chunks: list[bytes] = []
            newline_count = 0

            while remaining > 0 and newline_count <= max_lines:
                read_size = min(block_size, remaining)
                remaining -= read_size
                fp.seek(remaining)
                block = fp.read(read_size)
                chunks.append(block)
                newline_count += block.count(b"\n")

        merged = b"".join(reversed(chunks))
        text = merged.decode("utf-8", errors="replace")
        return text.splitlines()[-max_lines:]
    except Exception:
        with log_path.open("r", encoding="utf-8", errors="replace") as fp:
            lines = deque(fp, maxlen=max_lines)
        return [line.rstrip("\n") for line in lines]


class TaskProgress:
    def __init__(self):
        self.log_id = None
        self.offset = 0
        self.seen = set()
        self.summary = dict(total_files=0, finished_files=0, fail_count=0,
                            stage="等待启动", step="", done_flag=False, success_flag=False)

    def consume(self, line: str) -> None:
        for pattern, key, converter in ((TOTAL_PATTERN, "total_files", int), (STAGE_PATTERN, "stage", str),
                                        (STEP_PATTERN, "step", str), (FAIL_COUNT_PATTERN, "fail_count", int)):
            match = pattern.search(line)
            if match:
                self.summary[key] = converter(match.group(1).strip())
        if re.match(r"^\[[^]]+\]\s+\[(?:完成|跳过|失败)\]\s", line) and line not in self.seen:
            self.seen.add(line)
            self.summary["finished_files"] += 1
        if line == "处理结束。":
            self.summary["done_flag"] = True
        if line == "全部成功。":
            self.summary["success_flag"] = True

    def read(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("rb") as log:
                info = os.fstat(log.fileno())
                identity = (path, info.st_dev, info.st_ino)
                if identity != self.log_id or info.st_size < self.offset:
                    self.__init__()
                    self.log_id = identity
                log.seek(self.offset)
                for raw in log:
                    if not raw.endswith(b"\n"):
                        break  # the writer may still be appending this line
                    self.consume(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                    self.offset += len(raw)
        except OSError:
            if not self.log_id or self.log_id[0] != path:
                self.__init__()
        return self.summary.copy()


def parse_progress(lines: list[str]) -> dict[str, Any]:
    progress = TaskProgress()
    for line in lines:
        progress.consume(line.rstrip("\r\n"))
    return progress.summary


def extract_effective_config_from_cmd(cmd: list[str]) -> dict[str, str]:
    values, flags = parse_cli_tokens([str(item) for item in cmd])
    snapshot: dict[str, str] = {
        "输入目录": values.get("--input-dir", ""),
        "输出目录": values.get("--output-dir", ""),
        "语言": values.get("--language", "自动检测"),
        "速度档位": values.get("--speed-profile", ""),
        "设备": values.get("--device", ""),
        "恢复起点": values.get("--resume-from-file", ""),
        "文件级并行": "自动" if values.get("--parallel-workers", "0") == "0" else values.get("--parallel-workers", "1"),
        "线程": values.get("--threads", ""),
        "batch": values.get("--batch-size", ""),
        "chunk": values.get("--chunk-size", ""),
        "心跳秒数": values.get("--heartbeat-sec", "15"),
        "协同模式": "开启" if "--hybrid" in flags else "关闭",
        "高占用优化": "开启" if "--maximize-utilization" in flags else "关闭",
        "说话人分离": "关闭" if "--disable-diarization" in flags else "开启",
        "分离不可用时": {
            "ask": "运行时询问",
            "transcribe-only": "自动仅转写",
            "abort": "立即中断任务",
        }.get(values.get("--diarization-unavailable-policy", "ask"), "运行时询问"),
        "程序自动调节参数": "开启" if "--auto-env-tune" in flags and "--disable-auto-env-tune" not in flags else "关闭",
        "输出JSON": "开启" if "--write-json" in flags else "关闭",
        "跳过已处理": "开启" if "--skip-existing" in flags else "关闭",
    }
    return snapshot


def main() -> None:
    # Keep "streamlit run 控制台.py" as the optional web entry point.
    try:
        from web_console import main as run_web
    except ModuleNotFoundError as exc:
        if exc.name != "streamlit":
            raise
        raise SystemExit("网页控制台需要可选依赖，请运行 python -m pip install -r requirements-web.txt") from exc
    run_web()


if __name__ == "__main__":
    main()
