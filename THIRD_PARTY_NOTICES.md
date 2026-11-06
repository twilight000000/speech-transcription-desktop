# 第三方组件声明

本项目使用以下开源工具和模型。它们的名称、著作权与许可归各自作者及权利人；本项目的 GPL-3.0-only 不会替换这些原许可。

| 组件 | 本次便携环境版本 | 主要许可与来源 |
| --- | --- | --- |
| Python | 3.12.5 | Python Software Foundation License；包内 python/LICENSE.txt |
| WhisperX | 3.8.1 | BSD-2-Clause；https://github.com/m-bain/whisperX |
| faster-whisper | 1.2.1 | MIT；https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | 4.7.1 | MIT 及其第三方通知；https://github.com/OpenNMT/CTranslate2 |
| pyannote.audio | 4.0.4 | MIT；https://github.com/pyannote/pyannote-audio |
| PyTorch / torchaudio | 2.8.0+cu128 | 各自 BSD 类许可及第三方通知；https://github.com/pytorch/pytorch / https://github.com/pytorch/audio |
| PyAV | 16.1.0 | BSD-3-Clause；https://github.com/PyAV-Org/PyAV/tree/v16.1.0 |
| PyAV 所附 FFmpeg 与编解码库 | pyav-ffmpeg 8.0.1-3 | FFmpeg、x264、x265 及其他库分别适用 LGPL/GPL/BSD 等许可，见对应源码与许可文件 |
| Hugging Face Hub | 0.36.2 | Apache-2.0；https://github.com/huggingface/huggingface_hub |
| Silero VAD | 随包源码与 JIT 文件 | MIT；https://github.com/snakers4/silero-vad |

模型许可单列在 MODEL_LICENSES.md。完整 Python 分发版本清单位于 runtime-packages.json。分发元数据中的许可证字段仅供索引，权利与义务以每个组件完整许可文本为准。

## 随包许可位置

- Python：portable_runtime/python/LICENSE.txt 及其附带文件。
- Python 扩展包：portable_runtime/python/Lib/site-packages/ 中的各组件 LICENSE、NOTICE、COPYING 和 *.dist-info/licenses。
- 主要模型和 PyAV/FFmpeg 组件的许可副本：licenses/。
- Silero：portable_runtime/torch_cache/hub/snakers4_silero-vad_master/LICENSE。
- NVIDIA CUDA、cuDNN 等随 PyTorch 分发的库仍适用其专有再分发条款及附带通知，不属于本项目重新授权的自有代码。

程序没有修改随包 Python 扩展库的二进制内容。发布整理移除了不参与运行的导入库、开发文档、浏览器驱动和可选网页框架，以及重复的独立 FFmpeg 程序；模型权重不变。PyAV 所附动态库及其原有重定位内容保留。

## 对应源码资料

发布附件 **第三方源码资料.zip** 收集了 PyAV 16.1.0、pyav-ffmpeg 8.0.1-3 的构建脚本与补丁、FFmpeg 8.0.1 及其该构建声明的 Windows 编解码依赖来源。SOURCE_MANIFEST.json 提供下载来源与 SHA-256；UPSTREAM_RECIPES.json 保留上游版本及预期校验值。另附 libiconv 1.18 和 zlib 1.3.1 的源码。

这份资料包应与对应便携包一起提供下载，不应只留下一个不对应版本的源码主页。PyAV 及供应库构建方法见：
- https://github.com/PyAV-Org/PyAV/tree/v16.1.0
- https://github.com/PyAV-Org/pyav-ffmpeg/tree/8.0.1-3

随包没有对上述源码实施新的代码修改。其构建脚本及补丁是上游构建的一部分，均保留在归档中。Python 和其余扩展组件的上游地址可在 runtime-packages.json 与包内元数据中查询。

## 再分发和改动

再分发程序、模型或第三方运行库时，请一并保留适用的著作权、许可、署名、改动说明和对应源码获取方式。不得将模型或第三方库宣称为本项目原创。第三方名称不表示其作者为本项目背书。

若发现来源、版本或署名记录有遗漏，请通过项目 Issues 提醒维护者核对。

