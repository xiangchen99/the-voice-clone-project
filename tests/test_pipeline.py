"""End-to-end integration test for the digital self orchestrator."""

import math
import os
import tempfile
import unittest

from src.audio.preprocessor import AudioPreprocessor
from src.pipeline.orchestrator import DigitalSelfOrchestrator


class TestDigitalSelfOrchestrator(unittest.TestCase):

    def test_orchestrator_text_turn_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = f"{tmpdir}/ref_audio.wav"
            out_audio = f"{tmpdir}/response.wav"

            pre = AudioPreprocessor(target_sr=24000)
            sr = 24000
            # Generate 1s tone with pure python math
            audio = [0.2 * math.sin(2 * math.pi * 300 * (i / sr)) for i in range(sr)]
            pre.save_wav(ref_path, audio, sr)

            orchestrator = DigitalSelfOrchestrator(settings_path="configs/settings.yaml")
            # Ingest a memory
            orchestrator.memory_store.add_document(
                doc_id="tech_creed",
                content="I love optimizing high-performance systems and writing clean code.",
                category="Work & Craft",
            )

            turn = orchestrator.process_text_turn(
                user_text="What kind of systems do you love building?",
                reference_audio_path=ref_path,
                output_audio_path=out_audio,
            )

            self.assertIn("response_text", turn)
            self.assertGreater(len(turn["response_text"]), 0)
            self.assertTrue(os.path.exists(turn["audio_path"]))
            self.assertIn("tts_is_mock", turn)
            self.assertIn("tts_engine", turn)
            self.assertGreater(turn["metrics"]["total_sec"], 0)
            self.assertEqual(len(orchestrator.chat_history), 2)

    def test_orchestrator_audio_turn_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_audio = f"{tmpdir}/mic_input.wav"
            ref_path = f"{tmpdir}/ref_audio.wav"

            pre = AudioPreprocessor(target_sr=24000)
            sr = 24000
            audio = [0.2 * math.sin(2 * math.pi * 350 * (i / sr)) for i in range(sr)]
            pre.save_wav(input_audio, audio, sr)
            pre.save_wav(ref_path, audio, sr)

            orchestrator = DigitalSelfOrchestrator(settings_path="configs/settings.yaml")
            turn = orchestrator.process_audio_turn(input_audio, reference_audio_path=ref_path)

            self.assertIn("user_text", turn)
            self.assertIn("response_text", turn)
            self.assertIn("audio_path", turn)
            self.assertTrue(os.path.exists(turn["audio_path"]))
            self.assertIn("asr_transcription_sec", turn["metrics"])

    def test_tts_engine_resilience_missing_ref(self):
        from src.voice.tts_engine import XTTSv2Engine
        with tempfile.TemporaryDirectory() as tmpdir:
            out_audio = f"{tmpdir}/output_tts.wav"
            engine = XTTSv2Engine(device="cpu")

            # Reference audio that definitely does not exist
            missing_ref = f"{tmpdir}/non_existent_ref.wav"
            result_path = engine.synthesize(
                text="Testing speech synthesis fallback.",
                reference_audio_path=missing_ref,
                output_path=out_audio,
            )

            self.assertTrue(os.path.exists(result_path))
            # Verify generated audio is a valid WAV
            pre = AudioPreprocessor(target_sr=24000)
            loaded, loaded_sr = pre.load_wav(result_path)
            self.assertGreater(len(loaded), 0)


if __name__ == "__main__":
    unittest.main()

