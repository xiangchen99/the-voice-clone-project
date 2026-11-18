"""Generate blind listening pairs for a single voice clip and a set of clips."""

import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from src.voice.tts_engine import SynthesisError, XTTSv2Engine


LINES = (
    "Give me a minute to think it through. I want to make sure I got that right.",
    "That worked better than I expected! What would you try next?",
    "I have a few ideas, but I'm still figuring out which one makes the most sense.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", nargs="+", type=Path, help="First clip is used alone, then all clips are used together")
    parser.add_argument("--device", help="Override the device configured in configs/settings.yaml")
    args = parser.parse_args()
    references = list(dict.fromkeys(path.resolve() for path in args.references))
    if len(references) < 2:
        parser.error("Select at least two distinct reference WAV files.")

    settings = yaml.safe_load(Path("configs/settings.yaml").read_text(encoding="utf-8"))
    tts_options = settings.get("tts", {})
    device = args.device or settings.get("hardware", {}).get("device", "cuda")
    engine = XTTSv2Engine(device=device)
    conditions = [[references[0]], sorted(references, key=lambda path: (path.name, str(path)))]
    random.SystemRandom().shuffle(conditions)

    root = Path("data/audio/voice_comparisons")
    root.mkdir(parents=True, exist_ok=True)
    output = root / uuid4().hex[:12]
    try:
        with TemporaryDirectory(dir=root) as directory:
            staged = Path(directory) / "comparison"
            staged.mkdir()
            for number, line in enumerate(LINES, 1):
                for label, condition in zip(("A", "B"), conditions):
                    engine.synthesize(
                        text=line,
                        reference_audio_path=condition,
                        output_path=staged / f"{number}-{label}.wav",
                        language=tts_options.get("language", "en"),
                        temperature=tts_options.get("temperature", 0.75),
                        speed=tts_options.get("speed", 1.0),
                    )
            (staged / "lines.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
            (staged / "reveal.txt").write_text("\n".join(
                f"{label}: " + ", ".join(str(path) for path in condition)
                for label, condition in zip(("A", "B"), conditions)
            ) + "\n", encoding="utf-8")
            staged.rename(output)
    except SynthesisError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Listen to all three A/B pairs in {output}. Rate likeness and naturalness before opening reveal.txt.")


if __name__ == "__main__":
    main()
