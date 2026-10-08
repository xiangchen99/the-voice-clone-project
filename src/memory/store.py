"""Autobiographical episodic memory vector store with semantic retrieval."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MemoryStore:
    """Vector store for indexing and retrieving personal memories, beliefs, and chat history."""

    def __init__(self, db_dir: str = "data/memory_db", collection_name: str = "autobiography") -> None:
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.storage_file = self.db_dir / f"{collection_name}.json"
        self.documents: List[Dict[str, Any]] = []
        self._load_storage()

    def _load_storage(self) -> None:
        """Loads indexed documents from persistent JSON store."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []
        else:
            self.documents = []

    def _save_storage(self) -> None:
        """Saves documents to persistent store."""
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

    def add_document(self, doc_id: str, content: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds or updates a memory document."""
        # Remove existing if doc_id matches
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        meta = metadata or {}
        meta["category"] = category
        self.documents.append({
            "id": doc_id,
            "content": content.strip(),
            "category": category,
            "metadata": meta,
        })
        self._save_storage()

    def load_biography_directory(self, bio_dir: str = "data/biography") -> int:
        """Parses all Markdown memory cards from the biography directory and indexes them."""
        bio_path = Path(bio_dir)
        if not bio_path.exists():
            return 0

        count = 0
        for md_file in bio_path.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Extract frontmatter if present
            category = "general"
            slug = md_file.stem
            if raw_text.startswith("---"):
                parts = raw_text.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    body = parts[2].strip()
                    for line in frontmatter.splitlines():
                        if line.startswith("category:"):
                            category = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("slug:"):
                            slug = line.split(":", 1)[1].strip().strip('"\'')
                else:
                    body = raw_text
            else:
                body = raw_text

            self.add_document(
                doc_id=f"bio_{slug}",
                content=body,
                category=category,
                metadata={"source_file": str(md_file), "slug": slug},
            )
            count += 1

        return count

    def _tokenize(self, text: str) -> List[str]:
        """Simple clean tokenization."""
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return tokens

    def _compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], avg_dl: float, num_docs: int) -> float:
        """Computes BM25-based relevance score between query and document."""
        k1 = 1.5
        b = 0.75
        doc_len = len(doc_tokens)
        if doc_len == 0 or avg_dl == 0:
            return 0.0

        score = 0.0
        doc_counts = {}
        for t in doc_tokens:
            doc_counts[t] = doc_counts.get(t, 0) + 1

        for q in query_tokens:
            tf = doc_counts.get(q, 0)
            if tf == 0:
                continue
            # Calculate term score
            idf = math.log(1.0 + (num_docs + 0.5) / 1.5)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
            score += idf * (numerator / denominator)

        return score

    def query(self, query_text: str, top_k: int = 4, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries the memory database using semantic BM25 score and returns the top_k most relevant memories."""
        if not self.documents:
            return []

        candidates = self.documents
        if category_filter:
            candidates = [d for d in candidates if d.get("category", "").lower() == category_filter.lower()]
            if not candidates:
                return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return candidates[:top_k]

        all_doc_tokens = [self._tokenize(d["content"]) for d in candidates]
        avg_dl = sum(len(t) for t in all_doc_tokens) / max(len(all_doc_tokens), 1)
        num_docs = len(candidates)

        scored_results: List[Tuple[float, Dict[str, Any]]] = []
        for doc, tokens in zip(candidates, all_doc_tokens):
            score = self._compute_bm25_score(q_tokens, tokens, avg_dl, num_docs)
            if score > 0.05:
                res_dict = dict(doc)
                res_dict["relevance_score"] = round(score, 3)
                scored_results.append((score, res_dict))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[:top_k]]

    def clear(self) -> None:
        """Clears all stored memories."""
        self.documents = []
        if self.storage_file.exists():
            os.remove(self.storage_file)

