from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
MODELS = json.loads((APP_DIR / "models.json").read_text(encoding="utf-8"))


def cache_dir() -> Path:
    portable = APP_DIR.parent / "portable_runtime"
    if portable.is_dir():
        return portable / "hf_cache"
    return Path(os.getenv("HF_HUB_CACHE") or
                str(Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"))


def default_asr_model(profile: str, device: str) -> str:
    if profile == "fast":
        return "small" if device == "cuda" else "base"
    if profile == "accurate":
        return "medium" if device == "cuda" else "small"
    return "small"


def model_key(name: str) -> str | None:
    return next((key for key, spec in MODELS.items() if name in (key, spec["repo"])), None)


def local_model(name: str, cache: Path | None = None) -> Path | None:
    key = model_key(name)
    if key is None:
        path = Path(name).expanduser()
        return path.resolve() if path.is_dir() else None
    spec = MODELS[key]
    folder = (cache or cache_dir()) / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"]
    try:
        for name, expected in spec["files"].items():
            path = folder / name
            stat = path.stat()
            if stat.st_size != expected["size"]:
                return None
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    return None
        return folder
    except OSError:
        pass
    return None


def diarization_available(token: str = "", model: str = DEFAULT_DIARIZATION_MODEL) -> bool:
    return bool(token.strip() or os.getenv("HF_TOKEN", "").strip() or local_model(model))


def prepare_model(name: str, token: str = "", cache: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    key = model_key(name)
    if key is None:
        raise ValueError("此下载引导仅支持发布包列出的模型。")
    spec = MODELS[key]
    root = cache or cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected in spec["files"].items():
        print(f"准备 {key} / {filename}", flush=True)
        path = root / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"] / filename
        if not path.is_file() or path.stat().st_size != expected["size"]:
            path = Path(hf_hub_download(
                spec["repo"], filename, revision=spec["revision"], cache_dir=str(root),
                token=(token or False) if key == "diarization" else False,
            ))
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != expected["size"] or digest != expected["sha256"]:
            # 完整大小仍可能是损坏的缓存；强制重下当前文件，不删除用户的模型目录。
            print(f"校验失败，重新下载 {filename}", flush=True)
            path = Path(hf_hub_download(spec["repo"], filename, revision=spec["revision"],
                        cache_dir=str(root), token=(token or False) if key == "diarization" else False,
                        force_download=True))
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("模型校验失败，请重试或重新下载完整包。")
    result = local_model(name, root)
    if result is None:
        raise ValueError("模型文件不完整，请重试。")
    print(f"{key} 已就绪（SHA-256 校验通过）", flush=True)
    return result


def load_local_vad(chunk_size: int):
    import torch
    from whisperx.vads.silero import Silero
    from whisperx.vads.vad import Vad

    folder = Path(os.environ.get("TORCH_HOME", str(APP_DIR.parent / "portable_runtime/torch_cache"))) / "hub/snakers4_silero-vad_master"
    if not (folder / "src/silero_vad/data/silero_vad.jit").is_file():
        raise FileNotFoundError("缺少内置人声检测组件，请重新完整解压程序包。")
    class LocalSilero(Silero):
        def __init__(self):
            Vad.__init__(self, 0.5)
            self.vad_onset, self.chunk_size = 0.5, chunk_size
            self.vad_pipeline, utils = torch.hub.load(str(folder), "silero_vad", source="local", onnx=False)
            self.get_speech_timestamps, _, self.read_audio, _, _ = utils
    return LocalSilero()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Hugging Face 官方仓库准备固定版本模型。")
    parser.add_argument("models", nargs="+", choices=list(MODELS))
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        for name in args.models:
            prepare_model(name, os.getenv("HF_TOKEN", ""), args.cache)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            print("下载未获授权：请在模型页面接受条款，并填写有读取权限的 Hugging Face Token。", flush=True)
        else:
            print(f"准备失败（{type(exc).__name__}）。请检查网络、磁盘空间与文件写入权限，然后重试。", flush=True)
        raise SystemExit(1)

