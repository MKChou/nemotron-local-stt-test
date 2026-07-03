"""
Local microphone speech-to-text using NVIDIA Nemotron 3.5 ASR.
Usage:
  python test_nemotron_asr.py              # real-time streaming (default)
  python test_nemotron_asr.py --batch -d 8 # record then transcribe
"""

import argparse
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
LANGUAGE = "zh-CN"
SAMPLE_RATE = 16000
MIC_READ_SAMPLES = 1600  # 100 ms @ 16 kHz

_processor = None
_model = None


def load_model(*, streaming: bool = True):
    global _processor, _model
    if _model is None:
        print("Loading model (first run may take 30-60 seconds)...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        lookahead = 6 if streaming else 13
        _processor.set_num_lookahead_tokens(lookahead)
        _model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="auto")
        mode = "streaming" if streaming else "batch"
        latency = getattr(_processor, "streaming_latency_ms", None)
        extra = f", ~{latency} ms latency" if latency else ""
        print(f"Model ready ({mode}{extra}).\n")
    elif streaming:
        _processor.set_num_lookahead_tokens(6)
    else:
        _processor.set_num_lookahead_tokens(13)
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


class MicrophoneStream:
    """Capture mono float32 PCM from the microphone via ffmpeg."""

    def __init__(self, mic_name: str):
        self.mic_name = mic_name
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._pending = np.array([], dtype=np.float32)
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.ended = False

    def start(self) -> None:
        ffmpeg = find_ffmpeg()
        self._proc = subprocess.Popen(
            [
                ffmpeg, "-loglevel", "error",
                "-f", "dshow", "-i", f"audio={self.mic_name}",
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "f32le", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            bytes_per_read = MIC_READ_SAMPLES * 4
            assert self._proc and self._proc.stdout
            while not self._stop.is_set():
                data = self._proc.stdout.read(bytes_per_read)
                if not data:
                    break
                self._queue.put(np.frombuffer(data, dtype=np.float32).copy())
        finally:
            self.ended = True
            self._queue.put(None)

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self.ended = True
                return
            self._pending = np.concatenate([self._pending, item])

    def read_samples(self, n: int) -> np.ndarray | None:
        """Return up to n samples; fewer if the stream ended."""
        deadline = time.time() + 300
        while len(self._pending) < n and not self.ended:
            if time.time() > deadline:
                break
            self._drain_queue()
            if len(self._pending) < n and not self.ended:
                try:
                    item = self._queue.get(timeout=0.05)
                    if item is None:
                        self.ended = True
                    else:
                        self._pending = np.concatenate([self._pending, item])
                except queue.Empty:
                    pass
        if len(self._pending) == 0:
            return None
        out = self._pending[:n]
        self._pending = self._pending[n:]
        return out


def _pad_audio(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def stream_transcribe(language: str = LANGUAGE) -> str:
    processor, model = load_model(streaming=True)
    mic = pick_microphone()
    print(f"Microphone: {mic}")
    print("Listening... press Enter to stop.\n")

    stop_requested = threading.Event()
    mic_stream = MicrophoneStream(mic)
    mic_stream.start()

    def wait_for_stop() -> None:
        input()
        stop_requested.set()
        mic_stream.stop()

    threading.Thread(target=wait_for_stop, daemon=True).start()

    first_audio = mic_stream.read_samples(processor.num_samples_first_audio_chunk)
    if first_audio is None or len(first_audio) == 0:
        mic_stream.stop()
        raise RuntimeError("No audio received from microphone.")

    first_audio = _pad_audio(first_audio, processor.num_samples_first_audio_chunk)
    audio_buffer = first_audio.copy()

    first_chunk_inputs = processor(
        first_audio,
        sampling_rate=SAMPLE_RATE,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    )
    first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)

    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft
    mel_frame_idx = processor.num_mel_frames_first_audio_chunk

    def input_features_generator():
        nonlocal mel_frame_idx, audio_buffer
        yield first_chunk_inputs.input_features[
            :, : processor.num_mel_frames_first_audio_chunk, :
        ]

        while True:
            start_idx = mel_frame_idx * hop_length - n_fft // 2
            end_idx = start_idx + processor.num_samples_per_audio_chunk

            while len(audio_buffer) < end_idx and not mic_stream.ended:
                chunk = mic_stream.read_samples(end_idx - len(audio_buffer))
                if chunk is None or len(chunk) == 0:
                    break
                audio_buffer = np.concatenate([audio_buffer, chunk])

            if len(audio_buffer) <= start_idx:
                break

            if len(audio_buffer) < end_idx:
                segment = _pad_audio(
                    audio_buffer[start_idx:],
                    processor.num_samples_per_audio_chunk,
                )
                finished = True
            else:
                segment = audio_buffer[start_idx:end_idx]
                finished = False

            inputs = processor(
                segment,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=language,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features

            if finished or stop_requested.is_set():
                break

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
    }

    print("[Live] ", end="", flush=True)
    parts: list[str] = []
    generate_thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    generate_thread.start()

    try:
        for text_chunk in streamer:
            print(text_chunk, end="", flush=True)
            parts.append(text_chunk)
    finally:
        mic_stream.stop()
        generate_thread.join(timeout=30)

    print()
    return "".join(parts).strip()


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


def transcribe_batch(wav_path: str) -> str:
    processor, model = load_model(streaming=False)
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
        "--batch",
        action="store_true",
        help="Record a fixed duration, then transcribe (non-streaming)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=8.0,
        help="Recording length in seconds for --batch mode (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 40)
    if args.batch:
        print("  Speech-to-Text (local GPU · 简体中文 · batch)")
        print(f"  Duration: {args.duration}s")
    else:
        print("  Speech-to-Text (local GPU · 简体中文 · streaming)")
    print("=" * 40)

    load_model(streaming=not args.batch)

    if args.batch:
        while True:
            key = input("\nPress Enter to record, or q to quit: ").strip().lower()
            if key == "q":
                print("Bye.")
                break
            try:
                wav = record(args.duration)
                print("Transcribing...")
                result = transcribe_batch(wav)
                print(f"\n[Result] {result or '(no speech detected)'}")
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
        return

    while True:
        key = input("\nPress Enter to start streaming, or q to quit: ").strip().lower()
        if key == "q":
            print("Bye.")
            break
        try:
            result = stream_transcribe()
            print(f"\n[Final] {result or '(no speech detected)'}")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
