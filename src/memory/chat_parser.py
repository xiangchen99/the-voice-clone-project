"""Chat log parser and linguistic style extractor for personal preservation."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ChatLogParser:
    """Parses exported chat logs (WhatsApp, Discord, Telegram, or generic text)

    and extracts message corpora and linguistic style profiles.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "i", "you", "he",
        "she", "it", "we", "they", "my", "your", "his", "her", "its", "our",
        "their", "this", "that", "these", "those", "am", "so", "as", "if"
    }

    def __init__(self, target_author: Optional[str] = None) -> None:
        self.target_author = target_author.strip() if target_author else None

    def parse_whatsapp(self, text_content: str) -> List[Dict[str, str]]:
        """Parses WhatsApp export text lines into structured messages."""
        # Pattern 1: [12/31/23, 10:45:12 PM] Name: Message
        # Pattern 2: 12/31/23, 10:45 - Name: Message
        pattern_bracket = re.compile(
            r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s*([^:]+):\s*(.*)$"
        )
        pattern_dash = re.compile(
            r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)$"
        )

        messages: List[Dict[str, str]] = []
        current_msg: Optional[Dict[str, str]] = None

        for line in text_content.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            m1 = pattern_bracket.match(line_str)
            m2 = pattern_dash.match(line_str)
            match = m1 or m2

            if match:
                if current_msg:
                    messages.append(current_msg)
                timestamp = match.group(1).strip()
                author = match.group(2).strip()
                body = match.group(3).strip()
                current_msg = {"timestamp": timestamp, "author": author, "text": body}
            else:
                # Continuation of multi-line message
                if current_msg:
                    current_msg["text"] += "\n" + line_str

        if current_msg:
            messages.append(current_msg)

        return self._filter_author(messages)

    def parse_json_export(self, json_content: str) -> List[Dict[str, str]]:
        """Parses DiscordChatExporter or Telegram JSON exports."""
        data = json.loads(json_content)
        messages: List[Dict[str, str]] = []

        if isinstance(data, dict) and "messages" in data:
            raw_msgs = data["messages"]
        elif isinstance(data, list):
            raw_msgs = data
        else:
            raw_msgs = []

        for item in raw_msgs:
            if not isinstance(item, dict):
                continue
            author = item.get("author", {}).get("name") or item.get("from") or item.get("sender") or "Unknown"
            text = item.get("content") or item.get("text") or ""
            ts = item.get("timestamp") or item.get("date") or ""

            if isinstance(text, list):
                # Telegram sometimes breaks text into entities
                text_parts = [t.get("text", "") if isinstance(t, dict) else str(t) for t in text]
                text = "".join(text_parts)

            if str(text).strip():
                messages.append({"timestamp": str(ts), "author": str(author).strip(), "text": str(text).strip()})

        return self._filter_author(messages)

    def _filter_author(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filters messages if target_author is set."""
        if not self.target_author:
            return messages
        target_lower = self.target_author.lower()
        return [m for m in messages if target_lower in m["author"].lower()]

    def extract_linguistic_profile(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyzes text corpus to extract linguistic cadence, vocabulary, and statistics."""
        if not messages:
            return {
                "message_count": 0,
                "total_words": 0,
                "avg_words_per_message": 0.0,
                "top_words": [],
                "top_phrases": [],
                "punctuation_stats": {},
            }

        texts = [m["text"] for m in messages]
        total_msgs = len(texts)
        words: List[str] = []
        sentence_lengths: List[int] = []
        punct_counter: Counter = Counter()

        for t in texts:
            # Punctuation counts
            punct_counter["!"] += t.count("!")
            punct_counter["?"] += t.count("?")
            punct_counter["..."] += t.count("...")
            punct_counter[","] += t.count(",")

            cleaned_words = [re.sub(r"[^\w\s]", "", w).lower() for w in t.split()]
            cleaned_words = [w for w in cleaned_words if w]
            words.extend(cleaned_words)
            if cleaned_words:
                sentence_lengths.append(len(cleaned_words))

        # Filter stopwords for top vocabulary
        content_words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        word_freq = Counter(content_words).most_common(25)

        # Bigrams (two-word phrases)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        common_bigrams = [
            bg for bg, count in Counter(bigrams).most_common(20)
            if not all(w in self.STOPWORDS for w in bg.split())
        ]

        avg_words = float(sum(sentence_lengths) / max(len(sentence_lengths), 1))

        return {
            "message_count": total_msgs,
            "total_words": len(words),
            "avg_words_per_message": round(avg_words, 2),
            "top_words": word_freq,
            "top_phrases": common_bigrams[:10],
            "punctuation_stats": dict(punct_counter),
        }

    def chunk_messages_for_memory(
        self, messages: List[Dict[str, str]], chunk_size_words: int = 150
    ) -> List[Dict[str, Any]]:
        """Groups messages into conversational chunks suitable for RAG memory ingestion."""
        chunks: List[Dict[str, Any]] = []
        curr_chunk_words: List[str] = []
        curr_meta: List[str] = []

        for m in messages:
            msg_text = m["text"]
            curr_chunk_words.append(f"{m['author']}: {msg_text}")
            if sum(len(x.split()) for x in curr_chunk_words) >= chunk_size_words:
                chunks.append({
                    "content": "\n".join(curr_chunk_words),
                    "timestamp": m.get("timestamp", ""),
                    "source": "chat_history",
                })
                curr_chunk_words = []

        if curr_chunk_words:
            chunks.append({
                "content": "\n".join(curr_chunk_words),
                "timestamp": messages[-1].get("timestamp", "") if messages else "",
                "source": "chat_history",
            })

        return chunks

