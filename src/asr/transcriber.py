"""Speech-to-text transcriber using faster-whisper."""

import wave
from pathlib import Path
from threading import Lock
from typing import Union


class InvalidRecording(ValueError):
    """The input is not a readable microphone WAV."""


class TranscriptionError(RuntimeError):
    """The speech recognizer failed to transcribe a recording."""


class SpeechTranscriber:
    """Transcribes spoken user audio into text using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.lock = Lock()

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Return a transcript, or an empty string when no speech was detected."""
        audio_path = Path(audio_path)
        try:
            with wave.open(str(audio_path), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                if not frames or (wav.getsampwidth() > 1 and not any(frames)):
                    return ""
        except (EOFError, wave.Error) as exc:
            raise InvalidRecording("Could not read the recording. Record again or type a message.") from exc

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError("Speech recognition is unavailable. Install faster-whisper or type a message.") from exc

        with self.lock:
            try:
                if self.model is None:
                    self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                return self._read_segments(audio_path)
            except Exception as exc:
                self.model = None
                if self.device != "cuda":
                    raise TranscriptionError("Speech recognition is unavailable. Try again or type a message.") from exc
                try:
                    self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                    result = self._read_segments(audio_path)
                    self.device = "cpu"
                    self.compute_type = "int8"
                    return result
                except Exception as fallback_error:
                    self.model = None
                    raise TranscriptionError(
                        "Speech recognition is unavailable. Try again or type a message."
                    ) from fallback_error

    def _read_segments(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
        return " ".join(text for segment in segments if (text := segment.text.strip()))
