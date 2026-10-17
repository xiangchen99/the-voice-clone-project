"""Audio preprocessing module for voice clone dataset normalization and slicing."""

import array
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class AudioPreprocessor:
    """Preprocesses reference voice audio clips for neural voice cloning.

    Supports both NumPy (high performance) and pure Python standard library.
    """

    def __init__(
        self,
        target_sr: int = 24000,
        target_lufs_db: float = -20.0,
        silence_threshold_db: float = -40.0,
        min_silence_duration_ms: int = 300,
    ) -> None:
        self.target_sr = target_sr
        self.target_lufs_db = target_lufs_db
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration_ms = min_silence_duration_ms

    def load_wav(self, file_path: Union[str, Path]) -> Tuple[Union[List[float], any], int]:
        """Loads a WAV file into normalized float samples [-1.0, 1.0]."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size = p.stat().st_size
        if file_size < 12:
            raise ValueError(f"Audio file is too small ({file_size} bytes) to be a valid WAV file.")

        with open(p, "rb") as f:
            header = f.read(12)

        is_riff_wav = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        temp_converted: Optional[Path] = None

        if not is_riff_wav:
            # Check if ffmpeg is available to convert non-WAV formats (e.g. WebM, MP3, OGG)
            if shutil.which("ffmpeg"):
                temp_converted = p.with_name(f"{p.stem}_tmp_{os.getpid()}.wav")
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(p),
                            "-ar", str(self.target_sr),
                            "-ac", "1",
                            "-c:a", "pcm_s16le",
                            str(temp_converted),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    effective_path = str(temp_converted)
                except Exception:
                    if temp_converted.exists():
                        temp_converted.unlink()
                    temp_converted = None
                    effective_path = str(p)
            else:
                effective_path = str(p)
        else:
            effective_path = str(p)

        try:
            with wave.open(effective_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
        except wave.Error as e:
            if header[:4] == b"\x1a\x45\xdf\xa3":
                detected = "WebM/Opus"
            elif header[:4] == b"OggS":
                detected = "Ogg"
            elif header[4:8] == b"ftyp":
                detected = "MP4/AAC"
            else:
                detected = None
            hint = (
                f" Received compressed {detected} audio instead of PCM WAV; "
                f"reload the page (Ctrl+Shift+R) so the browser uses the raw PCM recorder."
                if detected
                else " Please ensure file is uncompressed PCM WAV format."
            )
            raise ValueError(
                f"Invalid or corrupted WAV file '{file_path}' (header: {header[:4]!r}): {e}.{hint}"
            ) from e
        finally:
            if temp_converted and temp_converted.exists():
                try:
                    temp_converted.unlink()
                except OSError:
                    pass

        if HAS_NUMPY:
            if sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sample_width == 1:
                data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 3:
                # 24-bit PCM conversion
                samples_list = []
                for i in range(0, len(raw_data), 3):
                    b0, b1, b2 = raw_data[i : i + 3]
                    val = b0 | (b1 << 8) | (b2 << 16)
                    if val & 0x800000:
                        val -= 0x1000000
                    samples_list.append(val / 8388608.0)
                data = np.array(samples_list, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")

            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = np.mean(data, axis=1)
            return data, framerate
        else:
            # Pure Python fallback
            samples: List[float] = []
            if sample_width == 2:
                count = n_frames * n_channels
                shorts = struct.unpack(f"<{count}h", raw_data)
                if n_channels == 1:
                    samples = [s / 32768.0 for s in shorts]
                else:
                    for i in range(0, count, n_channels):
                        avg_sample = sum(shorts[i : i + n_channels]) / (n_channels * 32768.0)
                        samples.append(avg_sample)
            elif sample_width == 1:
                bytes_data = struct.unpack(f"<{n_frames * n_channels}B", raw_data)
                samples = [(b - 128.0) / 128.0 for b in bytes_data]
                if n_channels > 1:
                    mono_samples = []
                    for i in range(0, len(samples), n_channels):
                        mono_samples.append(sum(samples[i : i + n_channels]) / n_channels)
                    samples = mono_samples
            elif sample_width == 3:
                # 24-bit signed PCM
                raw_samples = []
                for i in range(0, len(raw_data), 3):
                    b0, b1, b2 = raw_data[i : i + 3]
                    val = b0 | (b1 << 8) | (b2 << 16)
                    if val & 0x800000:
                        val -= 0x1000000
                    raw_samples.append(val / 8388608.0)
                if n_channels == 1:
                    samples = raw_samples
                else:
                    samples = []
                    for i in range(0, len(raw_samples), n_channels):
                        samples.append(sum(raw_samples[i : i + n_channels]) / n_channels)
            elif sample_width == 4:
                count = n_frames * n_channels
                ints = struct.unpack(f"<{count}i", raw_data)
                if n_channels == 1:
                    samples = [s / 2147483648.0 for s in ints]
                else:
                    samples = []
                    for i in range(0, count, n_channels):
                        samples.append(sum(ints[i : i + n_channels]) / (n_channels * 2147483648.0))
            else:
                raise ValueError(f"Unsupported sample width in standard lib fallback: {sample_width}")
            return samples, framerate

    def save_wav(self, file_path: Union[str, Path], audio: Union[List[float], any], sample_rate: int) -> None:
        """Saves float samples to a 16-bit PCM mono WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        if HAS_NUMPY and isinstance(audio, np.ndarray):
            clipped = np.clip(audio, -1.0, 1.0)
            pcm16 = (clipped * 32767.0).astype(np.int16)
            raw_bytes = pcm16.tobytes()
        else:
            clipped_samples = [max(-1.0, min(1.0, float(s))) for s in audio]
            int_samples = [int(s * 32767.0) for s in clipped_samples]
            raw_bytes = struct.pack(f"<{len(int_samples)}h", *int_samples)

        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)

    def resample_linear(
        self, audio: Union[List[float], Any], orig_sr: int, target_sr: int
    ) -> Union[List[float], Any]:
        """Linear resampling between sample rates."""
        if orig_sr == target_sr:
            return audio
        num_target_samples = int(round(len(audio) * float(target_sr) / float(orig_sr)))
        if num_target_samples == 0:
            return [] if not HAS_NUMPY else np.array([], dtype=np.float32)

        if HAS_NUMPY and isinstance(audio, np.ndarray):
            orig_indices = np.linspace(0, len(audio) - 1, len(audio))
            target_indices = np.linspace(0, len(audio) - 1, num_target_samples)
            return np.interp(target_indices, orig_indices, audio).astype(np.float32)
        else:
            resampled = []
            ratio = (len(audio) - 1) / max(1, num_target_samples - 1)
            for i in range(num_target_samples):
                src_pos = i * ratio
                idx = int(src_pos)
                frac = src_pos - idx
                if idx + 1 < len(audio):
                    val = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac
                else:
                    val = audio[idx]
                resampled.append(val)
            return resampled

    def calculate_rms_db(self, audio: Union[List[float], Any]) -> float:
        """Calculates root-mean-square loudness in dBFS."""
        if len(audio) == 0:
            return -100.0

        if HAS_NUMPY and isinstance(audio, np.ndarray):
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-7:
                return -100.0
            return float(20.0 * np.log10(rms))
        else:
            mean_sq = sum(s * s for s in audio) / len(audio)
            rms = math.sqrt(mean_sq)
            if rms < 1e-7:
                return -100.0
            return float(20.0 * math.log10(rms))

    def normalize_loudness(
        self, audio: Union[List[float], Any], target_db: Optional[float] = None
    ) -> Union[List[float], Any]:
        """Normalizes audio to target dBFS."""
        target = target_db if target_db is not None else self.target_lufs_db
        current_db = self.calculate_rms_db(audio)
        if current_db <= -90.0:
            return audio

        gain = 10.0 ** ((target - current_db) / 20.0)

        if HAS_NUMPY and isinstance(audio, np.ndarray):
            normalized = audio * gain
            peak = np.max(np.abs(normalized))
            if peak > 0.98:
                normalized = normalized * (0.98 / peak)
            return normalized.astype(np.float32)
        else:
            normalized_list = [s * gain for s in audio]
            peak = max(abs(s) for s in normalized_list) if normalized_list else 1.0
            if peak > 0.98:
                factor = 0.98 / peak
                normalized_list = [s * factor for s in normalized_list]
            return normalized_list

    def trim_silence(
        self, audio: Union[List[float], Any], sample_rate: int
    ) -> Union[List[float], Any]:
        """Trims leading and trailing silence."""
        frame_len = int(sample_rate * 0.02)
        if frame_len == 0 or len(audio) < frame_len:
            return audio

        num_frames = len(audio) // frame_len
        energies: List[float] = []
        for i in range(num_frames):
            chunk = audio[i * frame_len : (i + 1) * frame_len]
            energies.append(self.calculate_rms_db(chunk))

        start_frame = 0
        for i, energy in enumerate(energies):
            if energy > self.silence_threshold_db:
                start_frame = max(0, i - 1)
                break

        end_frame = num_frames
        for i in range(num_frames - 1, -1, -1):
            if energies[i] > self.silence_threshold_db:
                end_frame = min(num_frames, i + 2)
                break

        start_sample = start_frame * frame_len
        end_sample = min(len(audio), end_frame * frame_len)
        if start_sample >= end_sample:
            return audio
        return audio[start_sample:end_sample]

    def slice_into_chunks(
        self,
        audio: Union[List[float], Any],
        sample_rate: int,
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 3.0,
    ) -> List[Union[List[float], Any]]:
        """Slices continuous audio into natural reference chunks."""
        max_samples = int(max_duration_sec * sample_rate)
        min_samples = int(min_duration_sec * sample_rate)

        if len(audio) <= max_samples:
            return [audio] if len(audio) > 0 else []

        # Calculate optimal balanced chunk count
        num_chunks = int(math.ceil(len(audio) / float(max_samples)))
        chunk_size = int(math.ceil(len(audio) / float(num_chunks)))

        chunks: List[Union[List[float], Any]] = []
        curr = 0
        while curr < len(audio):
            end = min(curr + chunk_size, len(audio))
            chunk = audio[curr:end]
            if len(chunk) >= min_samples:
                chunks.append(chunk)
            elif chunks:
                # Merge small tail into previous chunk if needed
                if HAS_NUMPY and isinstance(chunks[-1], np.ndarray) and isinstance(chunk, np.ndarray):
                    chunks[-1] = np.concatenate([chunks[-1], chunk])
                else:
                    chunks[-1] = list(chunks[-1]) + list(chunk)
            elif len(chunk) > 0:
                chunks.append(chunk)
            curr += chunk_size

        return chunks

    def process_file(
        self, input_path: Union[str, Path], output_dir: Union[str, Path], base_name: Optional[str] = None
    ) -> Dict[str, Union[str, int, float, List[str]]]:
        """Processes an audio file end-to-end: loads, resamples, trims, normalizes, and saves reference clips."""
        in_p = Path(input_path)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)
        stem = base_name or in_p.stem

        raw_audio, sr = self.load_wav(in_p)
        resampled = self.resample_linear(raw_audio, sr, self.target_sr)
        trimmed = self.trim_silence(resampled, self.target_sr)
        normalized = self.normalize_loudness(trimmed, self.target_lufs_db)

        primary_path = out_d / f"{stem}_normalized.wav"
        self.save_wav(primary_path, normalized, self.target_sr)

        chunks = self.slice_into_chunks(normalized, self.target_sr)
        chunk_paths: List[str] = []
        for idx, chunk in enumerate(chunks):
            chunk_p = out_d / f"{stem}_chunk_{idx:02d}.wav"
            self.save_wav(chunk_p, chunk, self.target_sr)
            chunk_paths.append(str(chunk_p))

        total_duration = len(normalized) / float(self.target_sr)

        return {
            "source_file": str(in_p),
            "primary_reference": str(primary_path),
            "sample_rate": self.target_sr,
            "duration_seconds": round(total_duration, 2),
            "rms_db": round(self.calculate_rms_db(normalized), 2),
            "chunk_count": len(chunks),
            "chunk_paths": chunk_paths,
        }
