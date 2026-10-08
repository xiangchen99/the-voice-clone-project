"""Phonetically balanced and conversational reading prompts for voice capture."""

from typing import List, Dict

RECORDING_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "phonetic_01",
        "category": "Phonetic Calibration",
        "prompt": "The quick brown fox jumps over the lazy dog near the riverbank on a warm autumn morning.",
        "intent": "Warm-up pangram capturing full consonant and vowel range."
    },
    {
        "id": "phonetic_02",
        "category": "Phonetic Calibration",
        "prompt": "Oak is strong and also gives shade. Cats and dogs each hate the other. The pipe began to rust while new.",
        "intent": "Standard Harvard sentence measuring plosives and pauses."
    },
    {
        "id": "phonetic_03",
        "category": "Phonetic Calibration",
        "prompt": "Open the crate but don't drop the glass jar. A joy that's shared is a joy made double.",
        "intent": "Testing sibilance and dental sounds."
    },
    {
        "id": "casual_01",
        "category": "Conversational & Casual",
        "prompt": "Honestly, whenever I run into an unexpected bug late at night, I just pause, step back, and re-read the stack trace from the ground up.",
        "intent": "Natural, informal cadence with slight pauses."
    },
    {
        "id": "casual_02",
        "category": "Conversational & Casual",
        "prompt": "You know how some days everything just clicks? You sit down with a cup of coffee, fire up the terminal, and before you know it, three hours have flown by.",
        "intent": "Warm, reflective conversational tone."
    },
    {
        "id": "reflective_01",
        "category": "Reflective & Philosophical",
        "prompt": "I've always believed that building things from first principles gives you a clarity that shortcuts can never provide. You truly understand the foundation.",
        "intent": "Thoughtful, deliberate, grounded tempo."
    },
    {
        "id": "story_01",
        "category": "Narrative & Storytelling",
        "prompt": "Looking back at where I started compared to today, the most valuable skill wasn't knowing every answer right away. It was being persistent enough to keep digging until things made sense.",
        "intent": "Personal narrative prosody and pacing."
    },
    {
        "id": "dynamic_01",
        "category": "Enthusiastic & Animated",
        "prompt": "That's exactly what I'm talking about! When the entire system lines up, every microservice responds in milliseconds, and the architecture feels effortless.",
        "intent": "Higher vocal energy, exclamation, dynamic pitch variation."
    },
    {
        "id": "technical_01",
        "category": "Technical & Analytical",
        "prompt": "Let's break this down into three core components: the audio ingestion pipeline, the local neural model weights, and the dynamic memory vector index.",
        "intent": "Clear, precise enunciation with instructional rhythm."
    },
    {
        "id": "dialogue_01",
        "category": "Conversational Response",
        "prompt": "Yeah, I hear what you're saying, but let's look at the trade-offs first before making a final call.",
        "intent": "Casual conversational rebuttal tone."
    }
]


def get_all_prompts() -> List[Dict[str, str]]:
    """Return all recording prompts."""
    return RECORDING_PROMPTS


def get_prompt_by_id(prompt_id: str) -> Dict[str, str]:
    """Retrieve a specific prompt by identifier."""
    for p in RECORDING_PROMPTS:
        if p["id"] == prompt_id:
            return p
    raise KeyError(f"Prompt with id '{prompt_id}' not found.")

