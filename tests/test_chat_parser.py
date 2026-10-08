"""Unit tests for chat log parsing and linguistic profiling."""

import unittest
from src.memory.chat_parser import ChatLogParser


class TestChatLogParser(unittest.TestCase):

    def test_whatsapp_chat_parsing(self):
        sample_whatsapp = """[10/12/23, 2:15:30 PM] Xiang: Hey! Let's check out the new model release tonight.
[10/12/23, 2:16:01 PM] Alex: Sounds awesome, what time?
[10/12/23, 2:17:15 PM] Xiang: Probably around eight PM.
I want to test the latency metrics first.
"""
        parser = ChatLogParser(target_author="Xiang")
        messages = parser.parse_whatsapp(sample_whatsapp)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["author"], "Xiang")
        self.assertIn("Hey! Let's check out", messages[0]["text"])
        self.assertIn("eight PM", messages[1]["text"])
        self.assertIn("latency metrics", messages[1]["text"])

    def test_linguistic_profile_extraction(self):
        messages = [
            {"author": "Xiang", "text": "To be honest, clean code and architecture are critical."},
            {"author": "Xiang", "text": "Let's break this down into three simple components!"},
            {"author": "Xiang", "text": "Why would anyone choose complexity over simplicity?"},
        ]
        parser = ChatLogParser()
        profile = parser.extract_linguistic_profile(messages)

        self.assertEqual(profile["message_count"], 3)
        self.assertGreater(profile["total_words"], 10)
        self.assertGreaterEqual(profile["punctuation_stats"]["!"], 1)
        self.assertGreaterEqual(profile["punctuation_stats"]["?"], 1)

    def test_chat_memory_chunking(self):
        parser = ChatLogParser()
        messages = [
            {"author": "Xiang", "text": f"This is message number {i} discussing software design and robotics."}
            for i in range(20)
        ]
        chunks = parser.chunk_messages_for_memory(messages, chunk_size_words=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("Xiang:" in c["content"] for c in chunks))


if __name__ == "__main__":
    unittest.main()

