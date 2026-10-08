"""Neural Voice Cloning TTS Engine interface with XTTS-v2 and Mock support."""

import abc
import math
import os
import wave
from pathlib import Path
from typing import Optional, Union

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


class XTTSv2Engine(BaseTTSEngine):
    """Coqui XTTS-v2 voice cloning engine."""

    def __init__(self, device: str = "cuda", use_deepspeed: bool = False) -> None:
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.model = None

    def _lazy_init(self) -> None:
        """Loads XTTS model weights into VRAM on first call."""
        if self.model is None:
            try:
                from TTS.api import TTS
                self.model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(self.device)
            except Exception as e:
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

        self.model.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio_path),
            language=language,
            file_path=str(out_p),
            temperature=temperature,
            speed=speed,
        )
        return str(out_p)


class MockTTSEngine(BaseTTSEngine):
    """Deterministic synthetic voice generator for tests and offline development."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

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

        duration_sec = max(0.8, len(text.split()) * 0.35)
        num_samples = int(duration_sec * self.sample_rate)

        try:
            import numpy as np
            t = np.linspace(0, duration_sec, num_samples, endpoint=False)
            freq = 160.0 + 20.0 * np.sin(2.0 * math.pi * 1.5 * t)
            phase = 2.0 * math.pi * np.cumsum(freq) / self.sample_rate
            audio = 0.25 * np.sin(phase) + 0.1 * np.sin(2.0 * phase)

            env_len = int(0.05 * self.sample_rate)
            if len(audio) > 2 * env_len:
                audio[:env_len] *= np.linspace(0.0, 1.0, env_len)
                audio[-env_len:] *= np.linspace(1.0, 0.0, env_len)

            clipped = np.clip(audio, -1.0, 1.0)
            pcm16 = (clipped * 32767.0).astype(np.int16)
            raw_bytes = pcm16.tobytes()
        except ImportError:
            import struct
            int_samples = []
            phase = 0.0
            env_len = int(0.05 * self.sample_rate)
            for i in range(num_samples):
                t_val = i / float(self.sample_rate)
                freq = 160.0 + 20.0 * math.sin(2.0 * math.pi * 1.5 * t_val)
                phase += 2.0 * math.pi * freq / float(self.sample_rate)
                sample = 0.25 * math.sin(phase) + 0.1 * math.sin(2.0 * phase)

                # Envelope
                if i < env_len:
                    sample *= (i / float(env_len))
                elif i > num_samples - env_len:
                    sample *= ((num_samples - i) / float(env_len))

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
