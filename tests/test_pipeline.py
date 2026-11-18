"""Integration tests for conversation outcomes and owner-approved evidence."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts.guided_interview import save_memory_card
from src.asr.transcriber import InvalidRecording
from src.brain.llm_client import GenerationError
from src.pipeline.orchestrator import DigitalSelfOrchestrator
from src.voice.tts_engine import SynthesisError, XTTSv2Engine


class TestDigitalSelfOrchestrator(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        settings = self.scratch / "settings.yaml"
        settings.write_text(f"memory:\n  db_dir: {self.scratch / 'memory'}\nllm:\n  backend: mock\ntts:\n  engine: mock\n")
        self.orchestrator = DigitalSelfOrchestrator(settings_path=str(settings))

    def test_explicit_demo_turn_writes_a_playable_wav(self):
        turn = self.orchestrator.process_text_turn("Hello", output_audio_path=str(self.scratch / "reply.wav"))

        self.assertEqual(turn["llm_backend"], "mock")
        self.assertTrue(turn["tts_is_mock"])
        self.assertEqual(len(self.orchestrator.chat_history), 2)
        with wave.open(turn["audio_path"], "rb") as audio:
            self.assertGreater(audio.getnframes(), 0)

    def test_silence_does_not_enter_conversation(self):
        recording = self.scratch / "silence.wav"
        with wave.open(str(recording), "wb") as audio:
            audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(48000))

        with self.assertRaisesRegex(InvalidRecording, "No speech detected"):
            self.orchestrator.process_audio_turn(recording)
        self.assertEqual(self.orchestrator.chat_history, [])

    def test_unavailable_language_model_leaves_history_empty(self):
        with patch.object(self.orchestrator.llm_client, "generate_response", side_effect=GenerationError("Unavailable")):
            with self.assertRaises(GenerationError):
                self.orchestrator.process_text_turn("Hello")
        self.assertEqual(self.orchestrator.chat_history, [])

    def test_citation_without_reply_does_not_commit_turn(self):
        with patch.object(self.orchestrator.llm_client, "generate_response", return_value="[M9]"):
            with self.assertRaisesRegex(GenerationError, "empty reply"):
                self.orchestrator.process_text_turn("Hello")
        self.assertEqual(self.orchestrator.chat_history, [])

    def test_synthesis_failure_keeps_reply_and_only_approved_citations(self):
        cards = self.scratch / "biography"
        save_memory_card("Origins", "lighthouse", "What did you build?", "A cardboard lighthouse.", cards, True)
        save_memory_card("Origins", "unapproved", "What else?", "Another lighthouse.", cards)
        self.orchestrator.memory_store.load_biography_directory(str(cards))
        self.orchestrator.prompt_builder.persona["communication_style"] = {
            "approved_replies": ["I usually start with a concrete example."]
        }
        self.orchestrator.tts_engine = XTTSv2Engine(device="cpu")
        self.orchestrator.llm_client.backend = "ollama"

        with patch.object(self.orchestrator.llm_client, "generate_response", return_value="My lighthouse was cardboard [M1]. [M2]") as llm:
            turn = self.orchestrator.process_text_turn(
                "Tell me about your lighthouse.",
                reference_audio_path=self.scratch / "missing.wav",
            )
        prompt = llm.call_args.args[0]
        self.assertIn("I usually start with a concrete example.", prompt)
        self.assertIn("A cardboard lighthouse.", prompt)
        self.assertNotIn("Another lighthouse.", prompt)
        self.assertEqual(turn["response_text"], "My lighthouse was cardboard [M1].")
        self.assertEqual([item["id"] for item in turn["cited_memories"]], ["bio_lighthouse"])
        self.assertIsNone(turn["audio_path"])
        self.assertIsNone(turn["tts_engine"])
        self.assertIn("reference is missing", turn["tts_message"])
        self.assertEqual(len(self.orchestrator.chat_history), 2)

    def test_missing_selected_reference_writes_no_audio(self):
        output = self.scratch / "output.wav"
        engine = XTTSv2Engine(device="cpu")
        with self.assertRaisesRegex(SynthesisError, "reference is missing"):
            engine.synthesize("Hello.", self.scratch / "missing.wav", output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
