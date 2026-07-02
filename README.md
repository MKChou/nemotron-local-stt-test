# Nemotron ASR — Local Speech-to-Text

Record from your microphone and transcribe speech **locally on your GPU** using [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

No cloud API. No API keys. Audio stays on your machine.

## Features

- Local inference on NVIDIA GPU
- Press **Enter** to record, get text when done
- Configurable recording duration
- English transcription (`en-US`) out of the box
- Model loads once per session; repeat recordings without reloading

## Requirements

| Item | Notes |
|------|-------|
| **OS** | Windows (microphone via ffmpeg DirectShow) |
| **Python** | 3.11+ |
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

> **Note:** PyTorch 2.6+ and Transformers from source are required for this model.

### 3. First run downloads the model

The model (~2.5 GB) is cached under `~/.cache/huggingface/` after the first run.

## Usage

```powershell
python test_nemotron_asr.py
```

Press **Enter** to start recording (default 8 seconds). Type **q** to quit.

```powershell
# Custom duration (seconds)
python test_nemotron_asr.py -d 15
```

### Example session

```
========================================
  Speech-to-Text (local GPU · English)
  Duration: 8.0s
========================================
Loading model (first run may take 30-60 seconds)...
Model ready.

Press Enter to record, or q to quit:

Recording 8s (Microphone Array (Realtek(R) Audio))...
Recording done.
Transcribing...

[Result] Hello, this is a test of local speech recognition.
```

## How it works

```
Microphone → ffmpeg records WAV → Nemotron ASR (GPU) → English text
```

1. **Record** — ffmpeg captures audio from the default microphone
2. **Transcribe** — the Nemotron model runs on your GPU via Hugging Face Transformers
3. **Repeat** — model stays loaded; press Enter again for another recording

## Changing the language

Edit `LANGUAGE` in `test_nemotron_asr.py`:

```python
LANGUAGE = "en-US"   # English
LANGUAGE = "zh-CN"   # Mandarin (Simplified)
LANGUAGE = "zh-TW"   # Mandarin (Traditional)
LANGUAGE = "auto"    # Auto-detect
```

The model supports [40 languages](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#supported-languages).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ffmpeg not found` | Install ffmpeg and restart the terminal |
| Microphone error | Settings → Privacy → Microphone → allow desktop apps |
| `Audio too quiet` | Speak closer to the mic, reduce background noise |
| Out of GPU memory | Close other GPU apps; use shorter recordings |
| Slow first run | Normal — model download + GPU load takes 30-60s |

## Hardware notes

- **Model size:** ~0.6B parameters, ~2.5 GB on disk
- **VRAM:** ~2.4 GB for weights + inference overhead
- **GTX 1650 Ti (4 GB):** Works for short recordings; tight on memory
- **RTX 5090 / server GPU:** Much faster inference, room for streaming later

## License

This project is a thin wrapper around NVIDIA's model. See the [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for the OpenMDW-1.1 license.
