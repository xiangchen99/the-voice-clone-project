"""Unit tests for prompt builder, sanitization, and LLM client."""

import unittest
from src.brain.llm_client import LocalLLMClient
from src.brain.prompt_builder import PromptBuilder


class TestBrainAndPromptBuilder(unittest.TestCase):

    def test_prompt_builder_assembly_and_rules(self):
        builder = PromptBuilder(config_path="configs/persona.yaml")
        system_prompt = builder.build_system_prompt(autobiographical_context="Memory: Loved robotics.")

        self.assertIn("CORE IDENTITY & ATTITUDE", system_prompt)
        self.assertIn("COMMUNICATION & SPEECH RULES", system_prompt)
        self.assertIn("Loved robotics", system_prompt)

    def test_tts_text_sanitization(self):
        raw_llm_output = "Sure! *smiles warmly* Here is the plan: # Steps\n1. Do this. 😊 Let's go!"
        sanitized = PromptBuilder.sanitize_for_tts(raw_llm_output)

        self.assertNotIn("*", sanitized)
        self.assertNotIn("😊", sanitized)
        self.assertNotIn("#", sanitized)
        self.assertTrue("Sure!" in sanitized and "Let's go!" in sanitized)

    def test_llm_sentence_chunking(self):
        client = LocalLLMClient(backend="mock")
        text = "First sentence here. Second sentence starts now! And what about the third?"
        chunks = list(client.stream_sentence_chunks(text))

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "First sentence here.")
        self.assertEqual(chunks[1], "Second sentence starts now!")
        self.assertEqual(chunks[2], "And what about the third?")


if __name__ == "__main__":
    unittest.main()

