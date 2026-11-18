"""Unit tests for prompt builder, sanitization, and LLM client."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.brain.llm_client import GenerationError, LocalLLMClient
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


class TestLLMStatus(unittest.TestCase):

    @patch("src.brain.llm_client.urllib.request.urlopen")
    def test_ollama_probes_the_configured_model(self, urlopen):
        client = LocalLLMClient(endpoint_url="http://localhost:9000/", model_name="local/test")
        urlopen.side_effect = URLError("Connection refused")
        self.assertEqual(client.get_status(), "unavailable")
        urlopen.assert_called_once_with("http://localhost:9000/api/tags", timeout=1)

        urlopen.side_effect = None
        for response, expected in (
            (b"{}", "unavailable"),
            (b"invalid JSON", "unavailable"),
            (b'{"models":[{"name":null}]}', "unavailable"),
            (b'{"models":[{"name":"local/test:small"}]}', "missing_model"),
            (b'{"models":[{"name":"local/test:latest"}]}', "installed"),
            (b'{"models":[{"model":"local/test"}]}', "installed"),
        ):
            with self.subTest(response=response):
                urlopen.return_value = io.BytesIO(response)
                self.assertEqual(client.get_status(), expected)

    @patch("src.brain.llm_client.urllib.request.urlopen")
    def test_ollama_status_tracks_successful_and_failed_replies(self, urlopen):
        client = LocalLLMClient(model_name="test:latest")

        def ollama_response(request, timeout):
            if isinstance(request, str):
                return io.BytesIO(b'{"models":[{"model":"test:latest"}]}')
            return io.BytesIO(json.dumps({"message": {"content": reply}}).encode())

        urlopen.side_effect = ollama_response
        self.assertEqual(client.get_status(), "installed")
        reply = "Hello."
        self.assertEqual(client.generate_response("System", "User"), reply)
        self.assertEqual(client.get_status(), "working")

        reply = " "
        with self.assertRaises(GenerationError):
            client.generate_response("System", "User")
        self.assertEqual(client.get_status(), "reply_failed")

        reply = "Hello again."
        self.assertEqual(client.generate_response("System", "User"), reply)
        self.assertEqual(client.get_status(), "working")

    @patch("src.brain.llm_client.urllib.request.urlopen")
    def test_demo_and_openai_compatible_status(self, urlopen):
        demo = LocalLLMClient(backend="mock")
        client = LocalLLMClient(backend="openai_compatible")
        self.assertEqual(demo.get_status(), "demo")
        demo.generate_response("System", "User")
        self.assertEqual(demo.get_status(), "demo")
        self.assertEqual(client.get_status(), "unchecked")
        urlopen.assert_not_called()

        urlopen.return_value = io.BytesIO(b'{"choices":[{"message":{"content":"Hello."}}]}')
        self.assertEqual(client.generate_response("System", "User"), "Hello.")
        self.assertEqual(client.get_status(), "working")
        urlopen.side_effect = URLError("Connection refused")
        with self.assertRaises(GenerationError):
            client.generate_response("System", "User")
        self.assertEqual(client.get_status(), "reply_failed")


if __name__ == "__main__":
    unittest.main()

