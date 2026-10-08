"""Autobiographical context retriever for conversational prompting."""

from typing import Any, Dict, List, Optional
from src.memory.store import MemoryStore


class MemoryRetriever:
    """Retrieves relevant biographical context and formats it for prompt injection."""

    def __init__(self, memory_store: Optional[MemoryStore] = None, top_k: int = 3) -> None:
        self.store = memory_store or MemoryStore()
        self.top_k = top_k

    def retrieve_context(self, user_query: str) -> str:
        """Finds relevant personal memories and formats them into a clean prompt context block."""
        matches = self.store.query(user_query, top_k=self.top_k)
        if not matches:
            return ""

        context_lines = ["--- RELEVANT AUTOBIOGRAPHICAL MEMORIES ---"]
        for idx, m in enumerate(matches, 1):
            category = m.get("category", "General")
            content = m.get("content", "").strip()
            # Truncate content to ~300 chars if too long to maintain concise prompt context
            if len(content) > 600:
                content = content[:600] + "... [truncated]"
            context_lines.append(f"[Memory {idx} | Category: {category}]")
            context_lines.append(content)
            context_lines.append("")

        context_lines.append("--- END OF MEMORIES ---")
        return "\n".join(context_lines)

    def retrieve_memories_raw(self, user_query: str) -> List[Dict[str, Any]]:
        """Returns raw matched memory dictionaries."""
        return self.store.query(user_query, top_k=self.top_k)

