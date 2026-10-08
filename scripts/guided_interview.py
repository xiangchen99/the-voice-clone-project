#!/usr/bin/env python3
"""Interactive Autobiographical Interview Questionnaire.

Conducts an interview across core life dimensions and saves structured
biographical memory cards in data/biography/.
"""

import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List

QUESTIONS: List[Dict[str, str]] = [
    {
        "category": "Origins & Early Foundations",
        "slug": "origins_early_life",
        "question": "Where were you born and raised? What are one or two vivid memories from your childhood that shaped who you are?",
    },
    {
        "category": "Origins & Early Foundations",
        "slug": "family_and_roots",
        "question": "How would you describe your relationship with your family, and what core lessons did your parents or mentors instill in you?",
    },
    {
        "category": "Pivotal Milestones",
        "slug": "turning_points",
        "question": "What is the single biggest turning point or fork in the road in your life so far, and why did you make that choice?",
    },
    {
        "category": "Pivotal Milestones",
        "slug": "major_setback",
        "question": "Tell the story of a major failure or setback you experienced. How did you react in the moment, and how did you overcome it?",
    },
    {
        "category": "Work, Craft & Ambition",
        "slug": "career_craft",
        "question": "What kind of work or projects make you lose track of time? What drives you to build and create?",
    },
    {
        "category": "Work, Craft & Ambition",
        "slug": "work_ethic",
        "question": "How do you handle intense pressure, difficult deadlines, or technical friction? What is your problem-solving philosophy?",
    },
    {
        "category": "Beliefs & Worldview",
        "slug": "core_principles",
        "question": "What are the 3 non-negotiable principles or values that guide your decisions, even when it's inconvenient?",
    },
    {
        "category": "Beliefs & Worldview",
        "slug": "technology_future",
        "question": "What is your honest outlook on the future of technology, artificial intelligence, and human autonomy?",
    },
    {
        "category": "Personality, Humor & Quirks",
        "slug": "humor_and_taste",
        "question": "What kind of humor makes you laugh out loud? What books, films, music, or creators have had the deepest impact on your taste?",
    },
    {
        "category": "Personality, Humor & Quirks",
        "slug": "pet_peeves_quirks",
        "question": "What are your biggest pet peeves? What is an idiosyncratic habit or quirk that people close to you know you have?",
    },
    {
        "category": "Human Connections",
        "slug": "friendship_loyalty",
        "question": "What does genuine friendship and loyalty mean to you? How do you support the people you care about?",
    },
    {
        "category": "Future Legacy",
        "slug": "personal_legacy",
        "question": "If you could leave behind one message or enduring piece of wisdom for future generations or people who love you, what would it be?",
    },
]


def save_memory_card(category: str, slug: str, question: str, answer: str, output_dir: Path) -> Path:
    """Saves an autobiographical memory entry as structured Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{slug}.md"
    timestamp = datetime.datetime.now().isoformat()

    content = f"""---
category: "{category}"
slug: "{slug}"
date_recorded: "{timestamp}"
---

# {category}: {slug.replace('_', ' ').title()}

## Elicitation Question
> {question}

## My Personal Memory & Reflections
{answer}
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def run_interview(auto_fill_samples: bool = False) -> None:
    """Runs the interactive interview in the terminal."""
    out_dir = Path("data/biography")
    print("================================================================")
    print(" Digital Self Preservation: Autobiographical Interview")
    print("================================================================")
    print("Answer each question in your own authentic voice and words.")
    print("Type your answer and press Enter. (Type 'skip' to skip any question)\n")

    for idx, item in enumerate(QUESTIONS, 1):
        slug = item["slug"]
        existing = out_dir / f"{slug}.md"
        status = "[DONE]" if existing.exists() else "[NEW]"

        print(f"[{idx}/{len(QUESTIONS)}] {status} Category: {item['category']}")
        print(f"Question: {item['question']}")

        if existing.exists():
            print(f"-> Existing record found at {existing}. Type 'overwrite' to change, or press Enter to keep.")
            if not auto_fill_samples:
                resp = input("Action: ").strip().lower()
                if resp != "overwrite":
                    print("Kept existing record.\n")
                    continue

        if auto_fill_samples:
            print("Auto-fill mode: Skipping interactive input.\n")
            continue

        print("\nEnter your reflection (Enter an empty line or 'DONE' on a new line when finished):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "DONE" or (not lines and line.strip().lower() == "skip"):
                    if line.strip().lower() == "skip":
                        lines = []
                    break
                lines.append(line)
                # single line entry shortcut if double enter
                if len(lines) >= 2 and lines[-1] == "" and lines[-2] == "":
                    lines = lines[:-2]
                    break
            except EOFError:
                break

        answer = "\n".join(lines).strip()
        if not answer:
            print("Skipped.\n")
            continue

        saved_path = save_memory_card(item["category"], slug, item["question"], answer, out_dir)
        print(f"-> Saved memory to {saved_path}\n")

    print("\nInterview session completed! Memories saved in data/biography/.")


if __name__ == "__main__":
    auto_mode = "--sample" in sys.argv
    run_interview(auto_fill_samples=auto_mode)

