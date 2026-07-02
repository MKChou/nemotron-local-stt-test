"""
Local microphone speech-to-text using NVIDIA Nemotron 3.5 ASR.
Usage: python test_nemotron_asr.py [--duration 8]
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
LANGUAGE = "en-US"
SAMPLE_RATE = 16000

_processor = None
_model = None


def load_model():
    global _processor, _model
    if _model is None:
        print("Loading model (first run may take 30-60 seconds)...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _processor.set_num_lookahead_tokens(13)
        _model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="auto")
        print("Model ready.\n")
    return _processor, _model


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError("ffmpeg not found. Install with: winget install Gyan.FFmpeg")


def pick_microphone() -> str:
    ffmpeg = find_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    devices = re.findall(r'"([^"]+)"\s*\(audio\)', result.stderr)

    for name in devices:
        lower = name.lower()
        if "realtek" in lower or "microphone array" in lower:
            return name
    for name in devices:
        if "virtual" not in name.lower() and "steam" not in name.lower():
            return name
    if not devices:
        raise RuntimeError("No microphone found. Check connection and permissions.")
    return devices[0]


def record(duration: float) -> str:
    mic = pick_microphone()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    print(f"Recording {duration}s ({mic})...")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-loglevel", "error",
            "-f", "dshow", "-i", f"audio={mic}",
            "-t", str(duration), "-ac", "1", "-ar", str(SAMPLE_RATE),
            wav_path,
        ],
        check=True,
    )
    print("Recording done.")
    return wav_path


def prepare_audio(wav_path: str) -> np.ndarray:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    peak = np.max(np.abs(audio))
    if peak < 0.01:
        raise RuntimeError("Audio too quiet. Move closer to the microphone.")
    return audio / peak * 0.95


def transcribe(wav_path: str) -> str:
    processor, model = load_model()
    audio = prepare_audio(wav_path)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, language=LANGUAGE)
    inputs = inputs.to(model.device, dtype=model.dtype)
    output = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(output.sequences, skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Local microphone speech-to-text")
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=8.0,
        help="Recording length in seconds (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 40)
    print("  Speech-to-Text (local GPU · English)")
    print(f"  Duration: {args.duration}s")
    print("=" * 40)

    load_model()

    while True:
        key = input("\nPress Enter to record, or q to quit: ").strip().lower()
        if key == "q":
            print("Bye.")
            break

        try:
            wav = record(args.duration)
            print("Transcribing...")
            result = transcribe(wav)
            print(f"\n[Result] {result or '(no speech detected)'}")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
