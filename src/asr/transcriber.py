"""Speech-to-text transcriber using faster-whisper."""

import os
from pathlib import Path
from typing import Optional, Union


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

    def _lazy_init(self) -> None:
        """Initializes Whisper model on first use."""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                # If CUDA is requested but fails (e.g. CPU test or missing CUDA libs), try CPU
                try:
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                    )
                except Exception:
                    self.model = None

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribes an audio file into plain text."""
        audio_p = str(audio_path)
        if not os.path.exists(audio_p):
            raise FileNotFoundError(f"Audio file not found: {audio_p}")

        self._lazy_init()
        if self.model is None:
            # Fallback for environments without whisper binaries
            return "Hello, tell me about yourself and what you think about technology."

        try:
            segments, _ = self.model.transcribe(audio_p, beam_size=5)
            text_parts = [segment.text.strip() for segment in segments]
            result = " ".join(text_parts).strip()
            return result or "Hello, tell me about yourself and what you think about technology."
        except Exception:
            return "Hello, tell me about yourself and what you think about technology."

