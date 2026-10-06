# Nemotron ASR — Local Speech-to-Text

Transcribe speech from your microphone **locally on your GPU** using [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

No cloud API. No API keys. Audio stays on your machine.

## Features

- Local inference on NVIDIA GPU
- Real-time streaming by default: text appears while you speak
- Optional batch mode (`--batch`): record a fixed duration, then transcribe
- Mandarin transcription (`zh-CN`) out of the box; other languages by editing one constant
- Model loads once per session; repeat without reloading

## Requirements

| Item | Notes |
|------|-------|
| **OS** | Windows (microphone via ffmpeg DirectShow) |
| **Python** | 3.11 – 3.13 (the pinned PyTorch build has no wheels for newer versions) |
| **GPU** | NVIDIA GPU with CUDA, 4 GB+ VRAM recommended |
| **ffmpeg** | Required for microphone capture |

## Setup

### 1. Install ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Restart your terminal after installation.

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

This installs the CUDA 12.4 build of PyTorch (`torch==2.6.0+cu124`) and `transformers>=5.13.0`, the first release line the model card lists for this model.

> **RTX 50-series:** these cards need a newer CUDA build than cu124. Before installing, change the first two lines of `requirements.txt` to a newer PyTorch CUDA index (for example `https://download.pytorch.org/whl/cu128`) and a matching `torch` version from that index.

### 3. First run downloads the model

The model (~2.5 GB) is cached under `~/.cache/huggingface/` after the first run.

## Usage

### Streaming (default)

```powershell
python test_nemotron_asr.py
```

Press **Enter** to start listening, speak, and press **Enter** again to stop. Text is printed live while you speak. Type **q** at the prompt to quit.

### Batch

```powershell
# Record 8 seconds (default), then transcribe
python test_nemotron_asr.py --batch

# Custom duration (seconds)
python test_nemotron_asr.py --batch -d 15
```

Press **Enter** to record, or type **q** to quit.

`-d` / `--duration` only applies with `--batch`. Without it, the script prints a warning and ignores the value.

### Example session (streaming)

```
========================================
  Speech-to-Text (local GPU · zh-CN · streaming)
========================================
Loading model (first run may take 30-60 seconds)...
Model ready (streaming, ~<N> ms latency).


Press Enter to start streaming, or q to quit:
Microphone: Microphone Array (Realtek(R) Audio)
Listening... press Enter to stop.

[Live] 你好，这是本地语音识别测试。

[Final] 你好，这是本地语音识别测试。
```

`<N>` is the latency reported by the model's processor. In batch mode the banner reads `(local GPU · zh-CN · batch)` followed by a `Duration: 8.0s` line, and the result is printed as `[Result] ...`.

## How it works

```
Streaming:  Microphone → ffmpeg PCM stream → Nemotron ASR (GPU) → live text
Batch:      Microphone → ffmpeg records WAV → Nemotron ASR (GPU) → text
```

1. **Capture** — ffmpeg reads from a microphone picked by name: a Realtek or "Microphone Array" device first, otherwise the first non-virtual audio device. This is not necessarily the Windows default device.
2. **Transcribe** — the Nemotron model runs via Hugging Face Transformers
3. **Repeat** — model stays loaded; press Enter again for another round

## Changing the language

Edit `LANGUAGE` in `test_nemotron_asr.py` (default `zh-CN`):

```python
LANGUAGE = "zh-CN"   # Mandarin (default)
LANGUAGE = "en-US"   # English
LANGUAGE = "auto"    # Auto-detect
```

The banner shows the value you set. The model supports [40 language-locales](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#supported-languages); use a code from that list.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ffmpeg not found` | Install ffmpeg and restart the terminal |
| Microphone error / `No audio received from microphone` | Settings → Privacy → Microphone → allow desktop apps |
| `Audio too quiet` (batch mode) | Speak closer to the mic, reduce background noise |
| `Transcription failed: ...` / out of GPU memory | Close other GPU apps; in batch mode use shorter recordings |
| `(Stream ended. Press Enter to continue.)` | The microphone stream stopped on its own; press Enter and start again |
| Slow first run | Normal — model download + GPU load takes 30-60s |

## Hardware notes

- **Model size:** ~0.6B parameters, ~2.5 GB on disk
- **VRAM:** ~2.4 GB for weights + inference overhead
- **GTX 1650 Ti (4 GB):** Works for short recordings; tight on memory
- **RTX 50-series:** needs a newer CUDA build of PyTorch (see Setup)

## Verification status

As of 2026-10-06:

- **Confirmed on a GTX 1070 (Python 3.12.10):** `pip install -r requirements.txt` succeeds, torch reports `2.6.0+cu124` with `torch.cuda.is_available() == True`, and transformers resolves to 5.18.0.
- **Stub-based tests only:** the streaming and batch code paths were checked with stub modules in place of the model, microphone and ffmpeg.
- **Not yet run:** a real microphone session has not been run since the streaming error-handling refactor.

## License

This project is a thin wrapper around NVIDIA's model. See the [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for the OpenMDW-1.1 license.
