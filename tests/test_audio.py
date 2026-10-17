"""Unit tests for audio preprocessing and loudness normalization."""

import math
import tempfile
import unittest
from pathlib import Path

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

    def test_audio_preprocessor_short_clip_slicing(self):
        preprocessor = AudioPreprocessor(target_sr=24000)
        # 1.5 second clip (shorter than min_duration_sec 3.0s)
        sr = 24000
        short_audio = [0.2 * math.sin(2 * math.pi * 440 * (i / sr)) for i in range(int(1.5 * sr))]
        chunks = preprocessor.slice_into_chunks(short_audio, sr, min_duration_sec=3.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), len(short_audio))

    def test_audio_preprocessor_invalid_magic_bytes_and_file_checks(self):
        preprocessor = AudioPreprocessor(target_sr=24000)
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Non-existent file
            with self.assertRaises(FileNotFoundError):
                preprocessor.load_wav(f"{tmpdir}/non_existent.wav")

            # 2. File too small
            tiny_file = f"{tmpdir}/tiny.wav"
            with open(tiny_file, "wb") as f:
                f.write(b"RIFF")
            with self.assertRaises(ValueError):
                preprocessor.load_wav(tiny_file)

            # 3. WebM magic bytes pretending to be WAV
            fake_webm = f"{tmpdir}/fake_webm.wav"
            with open(fake_webm, "wb") as f:
                f.write(b"\x1a\x45\xdf\xa3" + b"\x00" * 100)
            with self.assertRaises(ValueError) as ctx:
                preprocessor.load_wav(fake_webm)
            self.assertIn("Invalid or corrupted WAV file", str(ctx.exception))

    def test_audio_preprocessor_32bit_pcm(self):
        import struct, wave
        preprocessor = AudioPreprocessor(target_sr=24000)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_32bit = f"{tmpdir}/audio32.wav"
            sr = 24000
            num_samples = sr // 2  # 0.5s
            # 32-bit signed ints
            int32_samples = [int(0.5 * 2147483647.0 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(num_samples)]
            with wave.open(wav_32bit, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)
                wf.setframerate(sr)
                wf.writeframes(struct.pack(f"<{num_samples}i", *int32_samples))

            loaded, loaded_sr = preprocessor.load_wav(wav_32bit)
            self.assertEqual(loaded_sr, 24000)
            self.assertEqual(len(loaded), num_samples)
            self.assertAlmostEqual(max(loaded), 0.5, delta=0.05)


class TestVoiceRecorder(unittest.TestCase):

    def test_create_master_reference_pure_python(self):
        from src.audio.recorder import VoiceRecorder
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = f"{tmpdir}/raw"
            ref_dir = f"{tmpdir}/ref"
            recorder = VoiceRecorder(raw_dir=raw_dir, reference_dir=ref_dir, target_sr=24000)

            # Initially no master reference
            self.assertIsNone(recorder.create_master_reference())

            # Create two sample normalized reference WAVs
            sr = 24000
            for name in ["clip_01", "clip_02"]:
                audio = [0.2 * math.sin(2 * math.pi * 300 * i / sr) for i in range(sr)]
                recorder.preprocessor.save_wav(f"{ref_dir}/{name}_normalized.wav", audio, sr)

            master_path = recorder.create_master_reference()
            self.assertIsNotNone(master_path)
            self.assertTrue(Path(master_path).exists())

            # Load master and verify duration has both clips plus silence padding
            loaded, loaded_sr = recorder.preprocessor.load_wav(master_path)
            self.assertEqual(loaded_sr, 24000)
            # 2 clips of 1s + 2 silences of 0.25s = 2.5s -> ~60,000 samples
            self.assertGreaterEqual(len(loaded), int(2.4 * sr))


class TestTTSEngine(unittest.TestCase):

    def test_mock_tts_engine_status_and_synthesis(self):
        from src.voice.tts_engine import MockTTSEngine, get_tts_engine
        engine = get_tts_engine("mock")
        self.assertIsInstance(engine, MockTTSEngine)

        status = engine.get_status()
        self.assertEqual(status["name"], "mock")
        self.assertTrue(status["is_mock"])
        self.assertTrue(status["ready"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_wav = f"{tmpdir}/chime.wav"
            res = engine.synthesize("Hello world test chime", "data/ref.wav", out_wav)
            self.assertEqual(res, out_wav)
            self.assertTrue(Path(out_wav).exists())

            import wave
            with wave.open(out_wav, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 24000)
                # Verify non-empty frames
                self.assertGreater(wf.getnframes(), 24000)

    def test_xtts_engine_status(self):
        from src.voice.tts_engine import XTTSv2Engine
        engine = XTTSv2Engine(device="cuda")
        status = engine.get_status()
        self.assertEqual(status["name"], "xtts_v2")
        self.assertIn("ready", status)
        self.assertIn("message", status)


if __name__ == "__main__":
    unittest.main()


