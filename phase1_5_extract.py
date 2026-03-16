#!/usr/bin/env python3
"""Phase 1.5 — Standalone CLI for LLM-based structure extraction.

Runs the preprocessor first (classify dialogue acts, merge fragments, clean text),
then sends only scorable turns to the LLM for structure extraction.

Usage:
    python3 phase1_5_extract.py --input flatearth/turns.json --output flatearth/structured_turns.json
"""

import argparse
import json
import os
import sys

from models import Turn
from pipeline.preprocessor import preprocess
from pipeline.structure_extractor import extract_structure


def run(args):
    """Core extraction logic. Called by main() or unified CLI."""
    # Load input
    with open(args.input) as f:
        raw = json.load(f)

    # Handle both formats: flat list of turns or full DebateResult with .turns
    if isinstance(raw, list):
        turn_dicts = raw
        topic = args.topic or "general debate"
        debaters = args.debaters
        metadata = None
    elif isinstance(raw, dict):
        turn_dicts = raw.get("turns", [])
        topic = args.topic or raw.get("topic", "general debate")
        debaters = args.debaters or raw.get("debaters")
        metadata = raw
    else:
        print("Error: input must be a JSON array or object with 'turns' key", file=sys.stderr)
        sys.exit(1)

    turns = [Turn.from_dict(t) for t in turn_dicts]
    print(f"Loaded {len(turns)} turns, topic: {topic}")
    if debaters:
        print(f"Debaters: {debaters}")

    # Step 1: Run preprocessor (no embeddings needed for basic preprocessing)
    print("\n--- Preprocessing ---")
    preprocess(turns, turn_embeddings=None, debaters=debaters)

    skipped = sum(1 for t in turns if not t.score_this)
    scorable = sum(1 for t in turns if t.score_this)
    debater_scorable = sum(
        1 for t in turns
        if t.score_this and (not debaters or t.speaker in debaters)
    )
    print(f"\nWill skip {skipped} turns (score_this=False)")
    print(f"Will send {debater_scorable} debater turns to LLM")

    # Step 2: Run structure extraction
    print("\n--- Structure Extraction ---")
    cache_path = None
    if not getattr(args, 'no_cache', False):
        cache_dir = os.path.dirname(args.output) or "."
        cache_path = os.path.join(cache_dir, ".structure_cache.json")

    extract_structure(turns, topic, debaters=debaters, cache_path=cache_path)

    # Write output
    if metadata:
        metadata["turns"] = [t.to_dict() for t in turns]
        output = metadata
    else:
        output = [t.to_dict() for t in turns]

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5: LLM structure extraction")
    parser.add_argument("--input", required=True, help="Path to turns.json (or scored.json with turns array)")
    parser.add_argument("--output", required=True, help="Path to write structured_turns.json")
    parser.add_argument("--topic", default=None, help="Debate topic (auto-detected from JSON if present)")
    parser.add_argument("--debaters", nargs="*", default=None, help="Debater names to process (all if omitted)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
