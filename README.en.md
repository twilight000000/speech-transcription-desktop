# Local Speech Transcription Desktop

A Chinese-language Windows desktop interface for batch transcription of local audio and video, with CPU/NVIDIA GPU options and optional speaker diarization. Built with WhisperX, faster-whisper and pyannote; this project does not train its own recognition models.

**v0.1.0-beta.1 — first public beta.** Recognition and speaker labels require human review.

## Downloads

Use the project's **[Releases](https://github.com/twilight000000/speech-transcription-desktop/releases/tag/v0.1.0-beta.1)** page. GitHub's **Code → Download ZIP** contains source code, not a ready-to-run portable environment.

- **Download-on-demand (online)**: Python and dependencies included. Unzip everything, open `启动.cmd`, then use `首次使用 / 模型` to download the required model. Standard transcription needs no account. Downloading the optional pyannote speaker model requires accepting its Hugging Face conditions and providing a read token.
- **Bundled models (offline)**: the same application with tiny/base/small/medium transcription models and community-1 diarization preloaded. Local use of included models needs no token.

GitHub assets use English filenames. Download both online parts (.001 and .002), or all three offline parts (.001, .002 and .003), into one folder. Open .001 with 7-Zip and extract everything. Links and SHA-256 checksums are on the release page.

Both packages include the Silero voice activity detector. The download-on-demand package still contains large GPU runtime libraries.

Select an input directory and an output directory, prepare models, then choose `开始转写` (Start). Open results using `打开输出目录`. Results are TXT and optional JSON, with separate names for different media extensions. This version has no subtitle editor or SRT/VTT export.

Windows 10/11 x64 is the target. 16 GB RAM and 25 GB free disk space are recommended, not guaranteed minimums. GPU use requires a compatible NVIDIA GPU and driver. CPU mode remains available.

## Privacy and licensing

Audio/video processing is local. Model downloads access Hugging Face and its distribution infrastructure. The app disables pyannote usage metrics and Hugging Face telemetry. Saved tokens use Windows user-bound encryption.

The project's own code is **GPL-3.0-only**. Models and bundled dependencies retain their own licenses; see [model attribution](MODEL_LICENSES.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE). No warranty is provided to the extent allowed by the applicable licenses.

Use is free. Contributions, issue reports and Stars are welcome. Donations would be optional and would not unlock features or purchase support. No donation endpoint is configured yet.

[Chinese guide](README.md) · [Contributing](CONTRIBUTING.md)

