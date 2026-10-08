"""Unit tests for autobiographical memory store and retriever."""

import tempfile
import unittest
from pathlib import Path
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore


class TestMemoryStoreAndRetriever(unittest.TestCase):

    def test_memory_store_indexing_and_bm25_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_dir=tmpdir, collection_name="test_memories")

            store.add_document(
                doc_id="childhood_01",
                content="When I was eight years old, I spent weekends building Lego cities with complex train layouts.",
                category="Origins & Family",
            )
            store.add_document(
                doc_id="tech_01",
                content="My philosophy on engineering is that clean architecture and deterministic tests beat quick hacks every time.",
                category="Work & Craft",
            )
            store.add_document(
                doc_id="food_01",
                content="My favorite comfort food is authentic hand-pulled noodles with chili oil.",
                category="Preferences",
            )

            self.assertEqual(len(store.documents), 3)

            # Query for Lego childhood
            results = store.query("Tell me about your Lego trains and childhood", top_k=2)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "childhood_01")

            # Query for code philosophy
            tech_results = store.query("What do you think about clean architecture and coding?", top_k=2)
            self.assertGreater(len(tech_results), 0)
            self.assertEqual(tech_results[0]["id"], "tech_01")

    def test_biography_markdown_directory_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bio_dir = Path(tmpdir) / "biography"
            bio_dir.mkdir()

            card = bio_dir / "origins.md"
            card.write_text(
                """---
category: "Early Years"
slug: "origins"
---
# Origins
I grew up surrounded by books and computers. My first computer was an old desktop that I took apart.
""",
                encoding="utf-8",
            )

            store = MemoryStore(db_dir=tmpdir, collection_name="bio_test")
            count = store.load_biography_directory(str(bio_dir))
            self.assertEqual(count, 1)
            self.assertEqual(store.documents[0]["id"], "bio_origins")
            self.assertIn("took apart", store.documents[0]["content"])

    def test_memory_retriever_context_formatting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(db_dir=tmpdir, collection_name="retriever_test")
            store.add_document(
                doc_id="failure_story",
                content="During a high school robotics competition our robot motor burned out right before the finals.",
                category="Milestones",
            )

            retriever = MemoryRetriever(memory_store=store, top_k=2)
            context = retriever.retrieve_context("Have you ever had a robot motor burn out or robot fail?")
            self.assertIn("RELEVANT AUTOBIOGRAPHICAL MEMORIES", context)
            self.assertIn("robotics competition", context)


if __name__ == "__main__":
    unittest.main()

