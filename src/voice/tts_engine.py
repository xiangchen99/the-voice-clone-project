"""Neural Voice Cloning TTS Engine interface with XTTS-v2 and Mock support."""

import abc
import logging
import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

# Auto-accept Coqui Public Model License non-interactively
os.environ.setdefault("COQUI_TOS_AGREED", "1")

logger = logging.getLogger("voice.tts_engine")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


ReferenceAudio = Union[str, Path, Sequence[Union[str, Path]]]


class SynthesisError(RuntimeError):
    """The requested audio could not be generated."""


class BaseTTSEngine(abc.ABC):
    """Abstract interface for neural voice cloning engines."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio_path: ReferenceAudio,
        output_path: Union[str, Path],
        language: str = "en",
        temperature: float = 0.75,
        speed: float = 1.0,
    ) -> str:
        """Synthesize text using the vocal timbre from reference_audio_path."""
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Returns engine readiness status and diagnostic telemetry."""
        pass


class XTTSv2Engine(BaseTTSEngine):
    """Coqui XTTS-v2 voice cloning engine."""

    def __init__(self, device: str = "cuda", use_deepspeed: bool = False) -> None:
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.model = None
        self.init_error: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Checks XTTS-v2 readiness without forcing immediate model weight load."""
        if self.model is not None:
            return {
                "name": "xtts_v2",
                "is_mock": False,
                "ready": True,
                "message": f"XTTS-v2 loaded and active on {self.device}",
            }
        if self.init_error is not None:
            return {
                "name": "xtts_v2",
                "is_mock": False,
                "ready": False,
                "message": f"XTTS-v2 initialization failed: {self.init_error}",
            }
        return {
            "name": "xtts_v2",
            "is_mock": False,
            "ready": False,
            "message": "XTTS-v2 will load when it synthesizes its first reply.",
        }

    def _lazy_init(self) -> None:
        """Loads XTTS model weights into VRAM on first call."""
        if self.model is None:
            try:
                from TTS.api import TTS
                from TTS.config.shared_configs import BaseDatasetConfig
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
                from torch.serialization import safe_globals

                # The published XTTS checkpoint stores these config objects alongside its tensors.
                with safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig]):
                    model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                self.model = model.to(self.device)
                self.init_error = None
            except Exception as e:
                self.init_error = str(e)
                logger.warning("Failed to initialize XTTS-v2: %s", e)
                raise SynthesisError("XTTS-v2 could not load. Check the local model and compute device.") from e

    def synthesize(
        self,
        text: str,
        reference_audio_path: ReferenceAudio,
        output_path: Union[str, Path],
        language: str = "en",
        temperature: float = 0.75,
        speed: float = 1.0,
    ) -> str:
        values = [reference_audio_path] if isinstance(reference_audio_path, (str, Path)) else reference_audio_path
        references = [Path(value) for value in values if value and str(value).strip()]
        if not references or any(not path.is_file() for path in references):
            raise SynthesisError("Selected voice reference is missing. Record or select a reference WAV in Voice Studio.")

        for path in references:
            try:
                with wave.open(str(path), "rb") as wav:
                    if not wav.getnframes():
                        raise wave.Error("Empty reference")
            except (OSError, EOFError, wave.Error) as exc:
                raise SynthesisError("A selected voice reference is unreadable or empty. Select a valid WAV.") from exc

        self._lazy_init()
        output = Path(output_path)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output.parent) as temp:
                candidate = Path(temp) / "reply.wav"
                self.model.tts_to_file(
                    text=text,
                    speaker_wav=[str(path) for path in references],
                    language=language,
                    file_path=str(candidate),
                    temperature=temperature,
                    speed=speed,
                )
                with wave.open(str(candidate), "rb") as wav:
                    if not wav.getnframes():
                        raise wave.Error("Empty reply")
                candidate.replace(output)
        except Exception as exc:
            logger.warning("XTTS synthesis failed: %s", exc)
            raise SynthesisError("XTTS-v2 could not generate audio for this reply.") from exc
        return str(output)


class MockTTSEngine(BaseTTSEngine):
    """Deterministic synthetic voice generator for tests and offline development.

    Generates a gentle musical harmonic chime as an audio notification placeholder.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def get_status(self) -> Dict[str, Any]:
        """Reports mock engine status."""
        return {
            "name": "mock",
            "is_mock": True,
            "ready": True,
            "message": "Demo mode uses a synthetic chime.",
        }

    def synthesize(
        self,
        text: str,
        reference_audio_path: ReferenceAudio,
        output_path: Union[str, Path],
        language: str = "en",
        temperature: float = 0.75,
        speed: float = 1.0,
    ) -> str:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Produce a clean, gentle musical chime (approx 1.5 seconds)
        duration_sec = 1.5
        num_samples = int(duration_sec * self.sample_rate)

        # Harmonic chime chord frequencies: C5 (523Hz), E5 (659Hz), G5 (784Hz)
        chord_freqs = [523.25, 659.25, 783.99]
        weights = [0.28, 0.22, 0.16]

        try:
            import numpy as np
            t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
            env = np.exp(-3.0 * t, dtype=np.float32)
            # Gentle fade out at tail
            tail_idx = int(0.9 * num_samples)
            env[tail_idx:] *= np.linspace(1.0, 0.0, num_samples - tail_idx, dtype=np.float32)

            audio = np.zeros(num_samples, dtype=np.float32)
            for freq, w in zip(chord_freqs, weights):
                audio += w * np.sin(2.0 * math.pi * freq * t)
            audio *= env

            clipped = np.clip(audio, -1.0, 1.0)
            pcm16 = (clipped * 32767.0).astype(np.int16)
            raw_bytes = pcm16.tobytes()
        except ImportError:
            import struct
            int_samples = []
            for i in range(num_samples):
                t_val = i / float(self.sample_rate)
                env = math.exp(-3.0 * t_val)
                if t_val > 1.35:
                    env *= max(0.0, (1.5 - t_val) / 0.15)
                val = 0.0
                for freq, w in zip(chord_freqs, weights):
                    val += w * math.sin(2.0 * math.pi * freq * t_val)
                sample = val * env
                int_samples.append(int(max(-1.0, min(1.0, sample)) * 32767.0))
            raw_bytes = struct.pack(f"<{len(int_samples)}h", *int_samples)

        with wave.open(str(out_p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_bytes)

        return str(out_p)


def get_tts_engine(engine_name: str = "xtts_v2", device: str = "cuda") -> BaseTTSEngine:
    """Factory creating the requested TTS engine."""
    eng = engine_name.lower()
    if eng == "mock":
        return MockTTSEngine()
    if eng == "xtts_v2":
        return XTTSv2Engine(device=device)
    raise ValueError(f"Unsupported voice engine: {engine_name}.")
