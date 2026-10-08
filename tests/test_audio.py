"""Unit tests for audio preprocessing and loudness normalization."""

import math
import tempfile
import unittest

from src.audio.preprocessor import AudioPreprocessor


class TestAudioPreprocessor(unittest.TestCase):

    def test_audio_preprocessor_loudness_and_normalization(self):
        preprocessor = AudioPreprocessor(target_sr=24000, target_lufs_db=-20.0)

        # Create synthetic sine wave test audio at 1000Hz, 1.0 second
        sr = 24000
        # Low amplitude wave
        test_audio = [0.05 * math.sin(2 * math.pi * 1000 * (i / sr)) for i in range(sr)]

        initial_rms_db = preprocessor.calculate_rms_db(test_audio)
        self.assertLess(initial_rms_db, -25.0)

        # Normalize to -20 dB
        normalized = preprocessor.normalize_loudness(test_audio, -20.0)
        norm_rms_db = preprocessor.calculate_rms_db(normalized)
        self.assertAlmostEqual(norm_rms_db, -20.0, delta=0.5)

    def test_audio_preprocessor_resampling_and_trimming(self):
        preprocessor = AudioPreprocessor(target_sr=24000)

        # 1 second of audio at 16000 Hz with 0.3s silence padding at start and end
        orig_sr = 16000
        silence_pad = [0.0] * int(0.3 * orig_sr)
        tone = [0.5 * math.sin(2 * math.pi * 440 * (i / orig_sr)) for i in range(int(0.4 * orig_sr))]
        full_audio = silence_pad + tone + silence_pad

        # Resample to 24000
        resampled = preprocessor.resample_linear(full_audio, orig_sr, 24000)
        expected_len = int(len(full_audio) * 24000 / orig_sr)
        self.assertEqual(len(resampled), expected_len)

        # Trim silence
        trimmed = preprocessor.trim_silence(resampled, 24000)
        self.assertLess(len(trimmed), len(resampled))

    def test_audio_preprocessor_slicing_and_file_io(self):
        preprocessor = AudioPreprocessor(target_sr=24000)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_wav = f"{tmpdir}/input.wav"
            out_dir = f"{tmpdir}/output"

            # Generate 12 seconds of test tone
            duration = 12.0
            sr = 24000
            total_samples = int(duration * sr)
            audio = [0.3 * math.sin(2 * math.pi * 440 * (i / sr)) for i in range(total_samples)]

            preprocessor.save_wav(test_wav, audio, sr)
            loaded, loaded_sr = preprocessor.load_wav(test_wav)
            self.assertEqual(loaded_sr, 24000)
            self.assertEqual(len(loaded), len(audio))

            result = preprocessor.process_file(test_wav, out_dir, base_name="test_clip")
            self.assertGreater(result["duration_seconds"], 10.0)
            self.assertGreaterEqual(result["chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()

