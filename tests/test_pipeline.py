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
            self.assertGreater(turn["metrics"]["total_sec"], 0)
            self.assertEqual(len(orchestrator.chat_history), 2)


if __name__ == "__main__":
    unittest.main()

