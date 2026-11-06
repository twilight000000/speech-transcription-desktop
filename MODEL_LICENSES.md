# 模型来源与许可

模型与本项目代码分别授权。下表为本发布包使用的固定版本；文件大小与 SHA-256 记录在 app/models.json。程序下载时固定到该版本，不自动追随仓库最新分支。

| 用途 | 上游与署名 | 许可 | 本包改动 |
| --- | --- | --- | --- |
| 语音转写 tiny/base/small/medium | OpenAI Whisper；SYSTRAN faster-whisper 转换版本 | MIT | 不修改权重；按推理所需文件分发，保留模型卡与许可 |
| 说话人区分 community-1 | pyannote 团队及其贡献者 | CC BY 4.0 | 不修改权重与配置；随包保留署名、许可与版本信息 |
| 人声片段检测 Silero VAD | Silero Team / snakers4 | MIT | 不修改模型；发布包仅保留 Torch 推理需要的源码和 JIT 模型，未包含其他推理格式 |

来源：
- [Whisper](https://github.com/openai/whisper)，[SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper)。
- [tiny](https://huggingface.co/Systran/faster-whisper-tiny)、[base](https://huggingface.co/Systran/faster-whisper-base)、[small](https://huggingface.co/Systran/faster-whisper-small)、[medium](https://huggingface.co/Systran/faster-whisper-medium)。
- [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)，[CC BY 4.0 条款](https://creativecommons.org/licenses/by/4.0/)。
- [Silero VAD](https://github.com/snakers4/silero-vad)。

MIT 文本、ASR 模型卡和 CC BY 4.0 完整条款位于 licenses/。community-1 模型页另有论文引用信息，请科研使用者查阅上游模型卡。其官方仓库的在线文件下载要求先登录并接受访问条件；这与 CC BY 4.0 的署名等再分发条件分别适用。不要将个人 Token 随程序分发。

固定版本：
- tiny: d90ca5fe260221311c53c58e660288d3deb8d356
- base: ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
- small: 536b0662742c02347bc0e980a01041f333bce120
- medium: 08e178d48790749d25932bbc082711ddcfdfbc4f
- community-1: 3533c8cf8e369892e6b79ff1bf80f7b0286a54ee

本工具不对模型效果作保证，也不将这些模型宣称为作者原创。再分发模型时应继续保留相应许可、署名、来源、改动说明及其他适用通知。

