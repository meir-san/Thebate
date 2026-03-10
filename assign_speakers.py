"""Standalone script to reassign speaker labels in an existing turns.json."""
import argparse
import json
import sys

from models import DebateResult
from speaker_assign import assign_speakers_interactive, apply_mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively assign speaker names in a turns.json file."
    )
    parser.add_argument("--input", required=True, help="Path to turns.json")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.input

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input} not found")
        sys.exit(1)

    result = DebateResult.from_dict(data)

    mapping = assign_speakers_interactive(result.turns)
    if not mapping:
        print("\nNo changes made.")
        return

    apply_mapping(result.turns, mapping)

    # Rebuild speakers and debaters lists
    speakers = []
    for t in result.turns:
        if t.speaker not in speakers:
            speakers.append(t.speaker)
    result.speakers = speakers

    # Update debaters: remap any that were renamed
    result.debaters = [mapping.get(d, d) for d in result.debaters]
    # Remove any debaters no longer in speakers
    result.debaters = [d for d in result.debaters if d in speakers]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"  Speakers: {', '.join(speakers)}")
    print(f"  Debaters: {', '.join(result.debaters)}")


if __name__ == "__main__":
    main()
