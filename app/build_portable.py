"""Build portable desktop packages and their source archives from a prepared runtime."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from model_resources import MODELS

APP_FILES = ("启动控制台.bat", "desktop_console_window.py", "控制台.py", "语音识别项目.py",
             "model_resources.py", "model_setup.py", "models.json", "requirements.txt", "requirements-runtime.txt")
WEB_PACKAGES = {"streamlit", "pyarrow", "pydeck", "altair"}
MODEL_FILES = {("models--" + spec["repo"].replace("/", "--")): tuple(spec["files"])
               for spec in MODELS.values()}
SILERO_FILES = ("hubconf.py", "LICENSE", "CITATION.cff", "README.md",
                "src/silero_vad/__init__.py", "src/silero_vad/model.py",
                "src/silero_vad/utils_vad.py", "src/silero_vad/data/__init__.py",
                "src/silero_vad/data/silero_vad.jit")
DOC_FILES = ("README.md", "README.en.md", "LICENSE", "NOTICE.md", "MODEL_LICENSES.md",
             "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "CHANGELOG.md", "runtime-packages.json")


def excluded(rel: Path) -> bool:
    parts = tuple(part.lower() for part in rel.parts)
    if {".runtime", ".git", ".cache", "__pycache__"} & set(parts):
        return True
    if rel.suffix.lower() in {".lib", ".dpapi", ".log", ".pyc", ".pyo"} or parts[-1] in {
        ".env", "token", "stored_tokens", "direct_url.json"}:
        return True
    if parts[:2] == ("python", "doc") or parts in {
        ("python", "chromedriver.exe"), ("python", "microsoftwebdriver.exe")
    }:
        return True
    if parts[:2] == ("python", "scripts") and parts[-1].split(".")[0] == "streamlit":
        return True
    if parts[:3] == ("python", "lib", "site-packages") and len(parts) >= 4:
        pkg = parts[3]
        return (pkg in WEB_PACKAGES or pkg.removesuffix(".libs") in WEB_PACKAGES or
                (pkg.endswith(".dist-info") and pkg.split("-", 1)[0] in WEB_PACKAGES))
    return False


def tree_files(root: Path, folder: Path):
    for current, directories, names in os.walk(folder):
        current = Path(current)
        directories[:] = sorted(d for d in directories if not excluded((current / d).relative_to(root)))
        for name in sorted(names):
            p = current / name
            if excluded(p.relative_to(root)):
                continue
            if not p.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"文件链接指向目录以外：{p}")
            yield p


def model_files(rt: Path):
    for spec in MODELS.values():
        directory = rt / "hf_cache" / ("models--" + spec["repo"].replace("/", "--"))
        for name, expected in spec["files"].items():
            p = directory / "snapshots" / spec["revision"] / name
            if not p.is_file() or not p.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"模型文件缺失或越界：{p}")
            with p.open("rb") as handle:
                if p.stat().st_size != expected["size"] or hashlib.file_digest(handle, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError(f"模型校验失败：{p}")
            yield p


def silero_files(rt: Path):
    base = rt / "torch_cache/hub/snakers4_silero-vad_master"
    for name in SILERO_FILES:
        path = base / name
        if not path.is_file() or not path.resolve().is_relative_to(rt):
            raise ValueError(f"人声检测组件缺失或越界：{path}")
        yield path


def build_packages(app: Path, rt: Path, destination: Path) -> None:
    app, rt, destination = app.resolve(), rt.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(rt) or destination.is_relative_to(app):
        raise ValueError("请选择项目和运行环境以外的全新输出目录。")
    if not (rt / "python/pythonw.exe").is_file():
        raise FileNotFoundError(rt / "python/pythonw.exe")
    docs = app if (app / "README.md").is_file() else app.parent
    sources = docs / "third-party-sources"
    if not sources.is_dir():
        raise FileNotFoundError("缺少 third-party-sources 第三方来源材料，不能生成发布包。")
    models, vad = list(model_files(rt)), list(silero_files(rt))
    documentation = [(docs / n, n) for n in DOC_FILES]
    for folder in ("docs", "licenses"):
        documentation += [(p, p.relative_to(docs).as_posix()) for p in tree_files(docs, docs / folder)]
    for p, _ in documentation:
        if not p.is_file():
            raise FileNotFoundError(p)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-build-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "packages"
        stage.mkdir()
        online = stage / "Windows-按需下载模型.zip"
        bundled = stage / "Windows-含模型.zip"
        options = dict(compression=zipfile.ZIP_DEFLATED, compresslevel=6)
        with zipfile.ZipFile(online, "w", **options) as archive:
            for name in APP_FILES:
                archive.write(app / name, f"app/{name}")
            for p, name in documentation:
                archive.write(p, name)
            archive.writestr("启动.cmd", '@echo off\r\nchcp 65001 >nul\r\ncall "%~dp0app\\启动控制台.bat"\r\n'.encode("utf-8"))
            archive.mkdir("app/input/")
            archive.mkdir("portable_runtime/hf_cache/")
            for p in list(tree_files(rt, rt / "python")) + vad:
                archive.write(p, "portable_runtime/" + p.relative_to(rt).as_posix())
        shutil.copyfile(online, bundled)
        with zipfile.ZipFile(bundled, "a", **options) as archive:
            for p in models:
                archive.write(p, "portable_runtime/" + p.relative_to(rt).as_posix())
        with zipfile.ZipFile(stage / "源码.zip", "w", **options) as archive:
            for name in (*APP_FILES, "build_portable.py", "web_console.py", "requirements-web.txt"):
                archive.write(app / name, f"app/{name}")
            for p in tree_files(app, app / "tests"):
                archive.write(p, "app/" + p.relative_to(app).as_posix())
            for p, name in documentation:
                archive.write(p, name)
            archive.write(docs / ".gitignore", ".gitignore")
            for p in tree_files(docs, docs / ".github"):
                archive.write(p, p.relative_to(docs).as_posix())
            for p in vad:
                archive.write(p, "portable_runtime/" + p.relative_to(rt).as_posix())
        with zipfile.ZipFile(stage / "第三方源码资料.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            for p in tree_files(sources.resolve(), sources):
                archive.write(p, "third-party-sources/" + p.relative_to(sources).as_posix())
        hashes = []
        for path in sorted(stage.glob("*.zip")):
            with path.open("rb") as handle:
                hashes.append(f"{hashlib.file_digest(handle, 'sha256').hexdigest()}  {path.name}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
        stage.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成两个 Windows 便携包、源码及第三方来源材料。")
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent.parent / "portable_runtime")
    parser.add_argument("--output", type=Path, required=True, help="所有 ZIP 和校验值的全新目录")
    args = parser.parse_args()
    build_packages(Path(__file__).resolve().parent, args.runtime, args.output)

