"""Neural Voice Cloning TTS Engine interface with XTTS-v2 and Mock support."""

import abc
import logging
import math
import os
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Auto-accept Coqui Public Model License non-interactively
os.environ.setdefault("COQUI_TOS_AGREED", "1")

logger = logging.getLogger("voice.tts_engine")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class BaseTTSEngine(abc.ABC):
    """Abstract interface for neural voice cloning engines."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio_path: Union[str, Path],
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
                "is_mock": True,
                "ready": False,
                "message": f"XTTS-v2 initialization failed: {self.init_error}",
            }
        try:
            import torch
            from TTS.api import TTS  # type: ignore
            cuda_avail = torch.cuda.is_available()
            if self.device == "cuda" and not cuda_avail:
                dev_note = "CUDA requested but not detected (will fall back to CPU)"
            else:
                dev_note = f"ready for lazy load on {self.device}"
            return {
                "name": "xtts_v2",
                "is_mock": False,
                "ready": True,
                "message": f"XTTS-v2 installed; {dev_note}",
            }
        except Exception as exc:
            return {
                "name": "xtts_v2",
                "is_mock": True,
                "ready": False,
                "message": f"Neural TTS dependencies not installed in this environment ({exc}). Run scripts/setup_environment.sh to activate.",
            }

    def _lazy_init(self) -> None:
        """Loads XTTS model weights into VRAM on first call."""
        if self.model is None:
            try:
                from TTS.api import TTS
                self.model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(self.device)
                self.init_error = None
            except Exception as e:
                self.init_error = str(e)
                logger.warning(
                    f"Failed to initialize XTTS-v2 on {self.device}: {e}. "
                    "Falling back to MockTTSEngine."
                )
                raise RuntimeError(
                    f"Failed to initialize XTTS-v2 on {self.device}. "
                    f"Ensure 'TTS' package and PyTorch CUDA are installed. Details: {e}"
                )

    def synthesize(
        self,
        text: str,
        reference_audio_path: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "en",
        temperature: float = 0.75,
        speed: float = 1.0,
    ) -> str:
        try:
            self._lazy_init()
        except RuntimeError:
            return MockTTSEngine().synthesize(
                text=text,
                reference_audio_path=reference_audio_path,
                output_path=output_path,
                language=language,
                temperature=temperature,
                speed=speed,
            )

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        ref_p = Path(reference_audio_path)
        if not ref_p.exists():
            alternatives = list(Path("data/audio/reference").glob("*.wav")) + list(Path("data/audio").glob("*.wav"))
            if alternatives:
                ref_p = alternatives[0]
            else:
                logger.warning(f"No reference audio found at {reference_audio_path}. Falling back to MockTTSEngine.")
                return MockTTSEngine().synthesize(
                    text=text,
                    reference_audio_path=reference_audio_path,
                    output_path=output_path,
                    language=language,
                    temperature=temperature,
                    speed=speed,
                )

        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=str(ref_p),
                language=language,
                file_path=str(out_p),
                temperature=temperature,
                speed=speed,
            )
            return str(out_p)
        except Exception as exc:
            logger.warning(f"XTTS synthesis error: {exc}. Falling back to MockTTSEngine.")
            return MockTTSEngine().synthesize(
                text=text,
                reference_audio_path=reference_audio_path,
                output_path=output_path,
                language=language,
                temperature=temperature,
                speed=speed,
            )


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
            "message": "Synthetic tone fallback engine (neural XTTS-v2 offline)",
        }

    def synthesize(
        self,
        text: str,
        reference_audio_path: Union[str, Path],
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
    elif eng == "xtts_v2":
        try:
            return XTTSv2Engine(device=device)
        except Exception:
            # Fallback to mock if weights/libraries not present yet
            return MockTTSEngine()
    else:
        return MockTTSEngine()
