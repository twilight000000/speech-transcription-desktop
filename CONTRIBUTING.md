# 开发与反馈

普通用户请使用 Releases 里的便携包。源码包不含运行环境和大模型。

## 本地运行

建议使用 Windows 64 位、Python 3.12。当前便携环境版本列在 runtime-packages.json。下面是对应版本的开发环境准备方式，仍需在目标电脑上验证。不要用一次“pip install 成功”代替实际转写验证。

在源码包根目录运行（CPU 示例）：

```text
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-deps -r app/requirements-runtime.txt
.venv\Scripts\python app/desktop_console_window.py
```

GPU 环境可将上面的 CPU 索引换成 https://download.pytorch.org/whl/cu128，前提是显卡和驱动兼容。requirements-runtime.txt 是随包环境的版本快照；使用 --no-deps 是为保留下面说明的 torchcodec 例外。requirements.txt 仅列核心依赖，不应直接用于重建这个便携环境。

桌面入口为 `desktop_console_window.py`，命令行入口为 `语音识别项目.py --help`。可选网页界面需要另装 requirements-web.txt，再运行 `streamlit run web_console.py`。

人声检测组件位于源码包随附的 `portable_runtime/torch_cache/hub/snakers4_silero-vad_master`，请保持源码 `app` 与 `portable_runtime` 并排。模型下载可从桌面引导完成；命令行可执行 `python model_resources.py small`。下载说话人模型时使用 HF_TOKEN 环境变量，不将 Token 写入源码。

已知环境差异：当前 Windows 便携环境有意不包含 torchcodec，程序通过 PyAV 解码后传入波形，避免其 DLL 兼容问题。whisperx/pyannote 的包元数据仍可能使 pip check 报告缺少 torchcodec；这不是完整依赖检查通过的环境。其他平台和使用 torchcodec 的第三方功能需要另行验证。

## 检查改动

在 app 目录执行 `python -m unittest discover -s tests`。界面测试会创建 Tk 窗口；音视频测试使用 FFmpeg 命令行生成无隐私样例，该测试工具不属于程序运行要求。发布前还应验证空缓存下载、完整本地模型离线转写和实际打包目录启动。

## 生成发布包

在与 app 并排的 portable_runtime 中准备 Python、依赖和已核对的模型。执行：

```text
python build_portable.py --runtime ../portable_runtime --output ../release-new
```

输出目录必须是新的，不能位于源码或运行环境内。脚本按白名单打包应用，剔除个人状态和无用开发文件，同时生成两个便携包、源码包、第三方源码资料包和 SHA-256。源资料见 third-party-sources/ 和 THIRD_PARTY_NOTICES.md，不能在发布时漏掉对应的来源材料。

不要提交大型转写模型权重、录音、输出、Token、.runtime、便携 Python 或个人路径。小型内置 Silero 组件随源码保留，其许可不变。提交代码表示你有权按本项目 GPL-3.0-only 提供该贡献；第三方代码须保留原许可和来源。

