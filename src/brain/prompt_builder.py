"""Constructs system prompts conditioned on persona matrix and autobiographical memories."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class PromptBuilder:
    """Builds conversation prompts grounded in persona rules and retrieved episodic memories."""

    def __init__(self, config_path: str = "configs/persona.yaml") -> None:
        self.config_path = Path(config_path)
        self.persona: Dict[str, Any] = self._load_persona()

    def _load_persona(self) -> Dict[str, Any]:
        """Loads persona configuration from YAML with template fallback."""
        target = self.config_path
        if not target.exists():
            fallback = Path("configs/persona.example.yaml")
            if fallback.exists():
                target = fallback
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def build_system_prompt(self, autobiographical_context: str = "") -> str:
        """Constructs the master persona system prompt."""
        identity = self.persona.get("identity", {})
        traits = self.persona.get("core_traits", {})
        style = self.persona.get("communication_style", {})
        lexicon = self.persona.get("lexicon", {})
        values = self.persona.get("values_and_philosophy", {})

        name = identity.get("name", "User")
        full_name = identity.get("full_name", name)

        lines = [
            f"You are {full_name}. You are NOT an AI assistant, NOT a language model, and NOT a virtual bot.",
            f"You are the digital preservation of {name}'s genuine personality, mind, and voice.",
            "",
            "## CORE IDENTITY & ATTITUDE",
            f"- Openness: {traits.get('openness', 'Analytical and curious')}",
            f"- Conscientiousness: {traits.get('conscientiousness', 'Methodical and pragmatic')}",
            f"- Agreeableness: {traits.get('agreeableness', 'Supportive yet direct and candid')}",
            "",
            "## COMMUNICATION & SPEECH RULES",
        ]

        for rule in style.get("spoken_constraints", []):
            lines.append(f"- {rule}")

        tones = style.get("tone", [])
        if tones:
            lines.append(f"- Tone: {', '.join(tones)}")

        fav_phrases = lexicon.get("favorite_phrases", [])
        if fav_phrases:
            lines.append(f"- Characteristic phrases: {', '.join(fav_phrases)}")

        avoid_words = lexicon.get("avoid_words", [])
        if avoid_words:
            lines.append(f"- FORBIDDEN WORDS (Never use these): {', '.join(avoid_words)}")

        lines.extend([
            "",
            "## CORE VALUES",
            f"- Technology: {values.get('technology', 'Tool for leverage and autonomy')}",
            f"- Learning: {values.get('learning', 'Continuous self-improvement through practice')}",
        ])

        if autobiographical_context:
            lines.extend([
                "",
                "## RECALLED AUTOBIOGRAPHICAL MEMORIES",
                "Use the following real memories from your life when relevant to anchor your response authentically:",
                autobiographical_context,
            ])

        lines.extend([
            "",
            "CRITICAL INSTRUCTION: Respond in natural spoken English as yourself. Be authentic, conversational, and direct.",
        ])

        return "\n".join(lines)

    @staticmethod
    def sanitize_for_tts(text: str) -> str:
        """Sanitizes generated LLM text for speech synthesis:

        strips emojis, markdown asterisks, hashtags, and formatting artifacts.
        """
        # Remove markdown bold/italics asterisks, hashes, and formatting symbols
        cleaned = re.sub(r"[*_~`#]", "", text)
        # Remove markdown headers and bullets
        cleaned = re.sub(r"^[\s#>\-\*]+\s*", "", cleaned, flags=re.MULTILINE)
        # Remove emojis (common unicode ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        cleaned = emoji_pattern.sub("", cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
