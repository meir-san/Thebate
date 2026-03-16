#!/usr/bin/env python3
"""Unified CLI — runs the full pipeline from YouTube URL to final report.

Intelligent caching: automatically skips phases whose output already exists.
Output goes to ./debates/{video_id}/ by default.

Usage:
    python3 debatestats.py --url "https://youtube.com/watch?v=..." \
        --topic "whether the earth is flat" \
        --speakers "Dave Farina,David Weiss"

    # Re-run from a specific phase:
    python3 debatestats.py --url "..." --topic "..." --force-from extract

    # Force re-run everything:
    python3 debatestats.py --url "..." --topic "..." --speakers "..." --force
"""

import argparse
import hashlib
import json
import os
import re
import sys
from types import SimpleNamespace


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', url)
    return m.group(1) if m else None


def file_hash(path: str) -> str:
    """SHA256 hash of a file's contents (first 16 hex chars)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full debate analysis pipeline from YouTube URL to report."
    )
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--topic", required=True, help="Debate topic in plain English")
    parser.add_argument(
        "--speakers", default=None,
        help="Comma-separated real names in order of first appearance "
             "(not needed if resuming from existing turns.json)"
    )
    parser.add_argument(
        "--debaters", default=None,
        help="Comma-separated names of speakers to score (default: all speakers)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ./debates/{video_id}/)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all phases, ignoring existing data"
    )
    parser.add_argument(
        "--force-from", choices=["ingest", "extract", "score", "render"],
        default=None,
        help="Re-run from a specific phase forward (keeps earlier data)"
    )
    parser.add_argument(
        "--skip-to", choices=["extract", "score", "render"], default=None,
        help="(legacy) Start from a specific phase"
    )
    parser.add_argument(
        "--adapter", default="assemblyai",
        choices=["assemblyai", "whisperx", "pyannote-api", "remote-whisperx"],
        help="Transcription adapter (default: assemblyai)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Extract video ID for directory naming
    video_id = extract_video_id(args.url)
    if not video_id:
        print(f"Could not extract video ID from: {args.url}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output or os.path.join("debates", video_id)
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    turns_path = os.path.join(output_dir, "turns.json")
    structured_path = os.path.join(output_dir, "structured_turns.json")
    scored_path = os.path.join(output_dir, "scored.json")
    report_path = os.path.join(output_dir, "report.html")
    overlay_path = os.path.join(output_dir, "overlay.html")
    hash_path = os.path.join(output_dir, ".turns_hash")

    # ── Determine which phases to run ────────────────────────────
    # Phase indices: ingest=0, extract=1, score=2, render=3
    phase_names = ["ingest", "extract", "score", "render"]
    phase_outputs = [turns_path, structured_path, scored_path, report_path]

    if args.force:
        start_from = 0
    elif args.force_from:
        start_from = phase_names.index(args.force_from)
    elif args.skip_to:
        start_from = {"extract": 1, "score": 2, "render": 3}[args.skip_to]
    else:
        # Auto-detect: find earliest missing output
        start_from = 4  # all done
        for i, path in enumerate(phase_outputs):
            if not os.path.exists(path):
                start_from = i
                break

        # Extra: if structured_turns.json exists but turns.json changed, re-extract
        if start_from > 1 and os.path.exists(turns_path) and os.path.exists(hash_path):
            current = file_hash(turns_path)
            with open(hash_path) as f:
                stored = f.read().strip()
            if current != stored:
                print("turns.json changed since last extraction — re-running from extract")
                start_from = 1

    # Print skip messages
    for i in range(min(start_from, 4)):
        if os.path.exists(phase_outputs[i]):
            print(f"Found existing {os.path.basename(phase_outputs[i])}, skipping {phase_names[i]}")

    if start_from >= 4:
        print("\nAll phases already complete. Use --force to re-run.")
        _print_scores(scored_path, output_dir)
        return

    # ── Phase 1: Ingest ──────────────────────────────────────────
    if start_from <= 0:
        if not args.speakers:
            print("Error: --speakers required for initial ingestion")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("PHASE 1: INGEST")
        print("=" * 60)
        from phase1_ingest import run as run_ingest
        ingest_args = SimpleNamespace(
            url=args.url,
            topic=args.topic,
            speakers=args.speakers,
            debaters=args.debaters,
            output=turns_path,
            adapter=args.adapter,
            voiceprints=None,
            speakers_dir="./speakers/",
            auto_enroll=False,
            enroll_model="base",
            keep_audio=True,  # keep audio for voiceprint matching
        )
        run_ingest(ingest_args)

    # ── Phase 1.5: Structure Extraction ──────────────────────────
    if start_from <= 1:
        # Check structure cache coverage before running
        cache_path = os.path.join(output_dir, ".structure_cache.json")
        skip_extract = False

        if os.path.exists(structured_path) and os.path.exists(cache_path) \
                and not args.force and args.force_from != "extract":
            try:
                with open(turns_path) as f:
                    data = json.load(f)
                with open(cache_path) as f:
                    cache = json.load(f)
                total = len(data.get("turns", []))
                cached = len(cache)
                if cached >= total:
                    print(f"\nStructure cache has {cached}/{total} turns, skipping extraction")
                    skip_extract = True
            except (json.JSONDecodeError, KeyError):
                pass

        if not skip_extract:
            print("\n" + "=" * 60)
            print("PHASE 1.5: STRUCTURE EXTRACTION")
            print("=" * 60)
            from phase1_5_extract import run as run_extract
            # Get debaters from args or from turns.json
            if args.debaters:
                debater_list = [n.strip() for n in args.debaters.split(",")]
            elif os.path.exists(turns_path):
                with open(turns_path) as f:
                    data = json.load(f)
                debater_list = data.get("debaters")
            else:
                debater_list = None
            extract_args = SimpleNamespace(
                input=turns_path,
                output=structured_path,
                topic=args.topic,
                debaters=debater_list,
                no_cache=False,
            )
            run_extract(extract_args)

        # Save turns.json hash for change detection
        with open(hash_path, "w") as f:
            f.write(file_hash(turns_path))

    # ── Phase 2: Score ───────────────────────────────────────────
    if start_from <= 2:
        print("\n" + "=" * 60)
        print("PHASE 2: SCORE")
        print("=" * 60)
        from phase2_score import run as run_score
        score_args = SimpleNamespace(
            input=structured_path,
            output=scored_path,
            threshold_engagement=None,
            threshold_dodge=None,
            threshold_drift=None,
        )
        run_score(score_args)

    # ── Phase 3: Render ──────────────────────────────────────────
    if start_from <= 3:
        print("\n" + "=" * 60)
        print("PHASE 3: RENDER")
        print("=" * 60)
        from phase3_render import run as run_render
        render_args = SimpleNamespace(
            input=scored_path,
            report=report_path,
            overlay=overlay_path,
        )
        run_render(render_args)

    _print_scores(scored_path, output_dir)


def _print_scores(scored_path, output_dir):
    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    try:
        with open(scored_path) as f:
            data = json.load(f)
        for speaker, stats in data.get("stats", {}).items():
            score = stats["overall_score"]
            rto = stats.get("responds_to_opponent_rate", 0)
            ss = stats.get("substance_share", 0)
            print(f"  {speaker:<24} {score:5.1f}/100  (rto: {rto:.3f}, substance: {ss:.3f})")
    except FileNotFoundError:
        print(f"  Could not read {scored_path}")

    report_path = os.path.join(output_dir, "report.html")
    overlay_path = os.path.join(output_dir, "overlay.html")
    print(f"\nOutput directory: {output_dir}")
    print(f"  Report:  {report_path}")
    print(f"  Overlay: {overlay_path}")
    print(f"  Data:    {scored_path}")


if __name__ == "__main__":
    main()
