"""Interactive and programmatic audio recorder for voice capture."""

import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional

from src.audio.preprocessor import AudioPreprocessor
from src.audio.prompts import RECORDING_PROMPTS, get_prompt_by_id


class VoiceRecorder:
    """Manages recording sessions, audio capture, and immediate preprocessing."""

    def __init__(
        self,
        raw_dir: str = "data/audio/raw",
        reference_dir: str = "data/audio/reference",
        target_sr: int = 24000,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.reference_dir = Path(reference_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = AudioPreprocessor(target_sr=target_sr)

    def get_session_status(self) -> Dict[str, Dict[str, bool]]:
        """Returns the recording status of each prompt in the catalog."""
        status = {}
        for p in RECORDING_PROMPTS:
            pid = p["id"]
            raw_file = self.raw_dir / f"{pid}.wav"
            ref_file = self.reference_dir / f"{pid}_normalized.wav"
            status[pid] = {
                "recorded": raw_file.exists(),
                "processed": ref_file.exists(),
                "category": p["category"],
            }
        return status

    def ingest_audio_file(self, input_wav_path: str, prompt_id: str) -> Dict[str, object]:
        """Ingests an externally recorded audio file for a given prompt ID and preprocesses it."""
        in_p = Path(input_wav_path)
        if not in_p.exists():
            raise FileNotFoundError(f"Source audio file not found: {input_wav_path}")

        # Copy/save to raw_dir
        dest_raw = self.raw_dir / f"{prompt_id}.wav"
        raw_audio, sr = self.preprocessor.load_wav(in_p)
        self.preprocessor.save_wav(dest_raw, raw_audio, sr)

        # Process into reference_dir
        result = self.preprocessor.process_file(dest_raw, self.reference_dir, base_name=prompt_id)
        return result

    def create_master_reference(self, output_name: str = "master_ref.wav") -> Optional[str]:
        """Combines all processed reference chunks into a single concatenated master reference audio."""
        ref_files = sorted(list(self.reference_dir.glob("*_normalized.wav")))
        if not ref_files:
            return None

        sr = self.preprocessor.target_sr
        silence_samples = int(sr * 0.25)

        try:
            import numpy as np
            has_np = True
        except ImportError:
            has_np = False
            np = None  # type: ignore

        if has_np:
            audio_arrays = []
            for rf in ref_files:
                audio, _ = self.preprocessor.load_wav(rf)
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                audio_arrays.append(audio)
                audio_arrays.append(np.zeros(silence_samples, dtype=np.float32))

            if not audio_arrays:
                return None
            full_audio = np.concatenate(audio_arrays)
        else:
            combined_audio: List[float] = []
            silence_padding = [0.0] * silence_samples
            for rf in ref_files:
                audio, _ = self.preprocessor.load_wav(rf)
                combined_audio.extend(list(audio))
                combined_audio.extend(silence_padding)

            if not combined_audio:
                return None
            full_audio = combined_audio

        # Normalize master reference
        full_audio = self.preprocessor.normalize_loudness(full_audio, -20.0)

        master_path = self.reference_dir / output_name
        self.preprocessor.save_wav(master_path, full_audio, sr)
        return str(master_path)


def run_cli_session() -> None:
    """CLI runner for guided voice recording session."""
    recorder = VoiceRecorder()
    print("==================================================")
    print(" Guided Voice Capture Studio")
    print("==================================================")
    status = recorder.get_session_status()
    total = len(RECORDING_PROMPTS)
    recorded_count = sum(1 for v in status.values() if v["recorded"])

    print(f"Progress: {recorded_count}/{total} prompts recorded.\n")

    for idx, prompt in enumerate(RECORDING_PROMPTS, 1):
        pid = prompt["id"]
        is_done = status[pid]["recorded"]
        tag = "[COMPLETED]" if is_done else "[PENDING]"
        print(f"[{idx}/{total}] {tag} ({prompt['category']}) - {pid}")
        print(f"  Prompt: \"{prompt['prompt']}\"")
        print(f"  Intent: {prompt['intent']}\n")

    print("\nTo ingest an audio file for a prompt:")
    print("  python -m src.audio.recorder ingest <path_to_audio.wav> <prompt_id>")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "ingest":
        rec = VoiceRecorder()
        res = rec.ingest_audio_file(sys.argv[2], sys.argv[3])
        print("Ingestion complete:", res)
    else:
        run_cli_session()

