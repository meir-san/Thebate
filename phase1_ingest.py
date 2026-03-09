import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

from dotenv import load_dotenv

from models import DebateResult
from pipeline.turn_builder import build_turns

YOUTUBE_RE = re.compile(r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+')


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Ingest YouTube debate video into diarized turns.")
    parser.add_argument("--url", required=True, help="Full YouTube URL")
    parser.add_argument("--topic", required=True, help="Debate topic in plain English")
    parser.add_argument("--speakers", default=None, help="Comma-separated real names in order of first appearance")
    parser.add_argument("--debaters", default=None, help="Comma-separated names of speakers to score (must be subset of --speakers). If omitted, all speakers are debaters.")
    parser.add_argument("--output", default="turns.json", help="Output JSON path (default: turns.json)")
    parser.add_argument("--adapter", default="assemblyai", choices=["assemblyai", "whisperx"], help="Transcription adapter")
    return parser.parse_args()


def get_video_metadata(url: str) -> dict | None:
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", url],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def format_duration(ms: int) -> str:
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def run(args):
    """Core ingestion logic. Called by main() or main.py wrapper."""
    load_dotenv()

    # Validate YouTube URL
    if not YOUTUBE_RE.match(args.url):
        print(f"Invalid YouTube URL: {args.url}")
        sys.exit(1)

    # Check API key
    if args.adapter == "assemblyai" and not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("Error: ASSEMBLYAI_API_KEY not set.")
        print("Get a free key at https://www.assemblyai.com")
        sys.exit(1)
    if args.adapter == "whisperx" and not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN not set.")
        print("Add HF_TOKEN=hf_xxx to your .env file.")
        sys.exit(1)

    # Get video metadata
    metadata = get_video_metadata(args.url)
    title = args.url
    if metadata:
        title = metadata.get("title", args.url)
        duration_secs = metadata.get("duration", 0)
        if duration_secs > 10800:
            print(f"Warning: Video is {duration_secs // 3600}h {(duration_secs % 3600) // 60}m long (>{3}h).")
            response = input("Continue? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                sys.exit(0)

    # Download audio to temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "audio.mp3")
    print(f"Downloading audio from: {args.url}")
    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", tmp_path, args.url],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"Failed to download: {args.url}")
        print("Video may be private or region-locked.")
        sys.exit(1)

    # Transcribe
    if args.adapter == "whisperx":
        print("Transcribing with WhisperX (this may take a while)...")
        from adapters.whisperx_adapter import WhisperXAdapter
        adapter = WhisperXAdapter()
    else:
        print("Transcribing with AssemblyAI (this may take a few minutes)...")
        from adapters.assemblyai_adapter import AssemblyAIAdapter
        adapter = AssemblyAIAdapter()
    segments = adapter.transcribe(tmp_path)

    # Build turns
    turns = build_turns(segments)

    # Apply speaker name mapping
    if args.speakers:
        names = [n.strip() for n in args.speakers.split(",")]
        speaker_labels = []
        for t in turns:
            if t.speaker not in speaker_labels:
                speaker_labels.append(t.speaker)

        if len(names) != len(speaker_labels):
            print(f"Warning: --speakers has {len(names)} names but {len(speaker_labels)} speakers detected.")
            print("Mapping what we can, leaving the rest as SPEAKER_XX.")

        mapping = {}
        for i, label in enumerate(speaker_labels):
            if i < len(names):
                mapping[label] = names[i]

        if mapping:
            parts = ", ".join(f"{k} → {v}" for k, v in mapping.items())
            print(f"Speaker mapping: {parts}")

        # Show first turn preview and confirm
        while True:
            # Apply current mapping
            for t in turns:
                if t.speaker in mapping:
                    t.speaker = mapping[t.speaker]

            # Show preview of each speaker's first turn
            print("\nSpeaker previews (first 20 words of first turn):")
            seen_speakers = []
            for t in turns:
                if t.speaker not in seen_speakers:
                    seen_speakers.append(t.speaker)
                    words = t.text.split()[:20]
                    preview = " ".join(words)
                    if len(t.text.split()) > 20:
                        preview += "..."
                    print(f"  {t.speaker}: \"{preview}\"")

            confirm = input("\nIs this mapping correct? [Y/n] ").strip().lower()
            if confirm in ("", "y", "yes"):
                break

            # Let user re-enter names — first revert to original labels
            for t in turns:
                for label, name in mapping.items():
                    if t.speaker == name:
                        t.speaker = label
                        break

            new_names_str = input(
                f"Re-enter speaker names in order of first appearance "
                f"({len(speaker_labels)} speakers, comma-separated): "
            ).strip()
            names = [n.strip() for n in new_names_str.split(",")]
            if len(names) != len(speaker_labels):
                print(f"Expected {len(speaker_labels)} names, got {len(names)}. Try again.")
                continue
            mapping = {label: name for label, name in zip(speaker_labels, names)}
            parts = ", ".join(f"{k} → {v}" for k, v in mapping.items())
            print(f"New mapping: {parts}")
    # Build speaker list in order of first appearance
    speakers = []
    for t in turns:
        if t.speaker not in speakers:
            speakers.append(t.speaker)

    # Determine debaters
    if args.debaters:
        debaters = [n.strip() for n in args.debaters.split(",")]
        invalid = [d for d in debaters if d not in speakers]
        if invalid:
            print(f"Warning: --debaters contains names not in speakers: {invalid}")
            print(f"Known speakers: {speakers}")
            debaters = [d for d in debaters if d in speakers]
    else:
        debaters = list(speakers)

    # Build DebateResult
    duration_ms = turns[-1].end_ms if turns else 0
    result = DebateResult(
        title=title,
        youtube_url=args.url,
        topic=args.topic,
        duration_ms=duration_ms,
        speakers=speakers,
        debaters=debaters,
        turns=turns,
        stats={},
        generated_at=datetime.utcnow().isoformat() + "Z",
    )

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # Print summary
    speaker_counts = {}
    for t in turns:
        speaker_counts[t.speaker] = speaker_counts.get(t.speaker, 0) + 1
    speaker_summary = ", ".join(f"{s} ({c} turns)" for s, c in speaker_counts.items())

    print(f"\n✓ Ingestion complete")
    print(f"  Turns: {len(turns)}")
    print(f"  Speakers: {speaker_summary}")
    print(f"  Debaters: {', '.join(result.debaters)}")
    print(f"  Duration: {format_duration(duration_ms)}")
    print(f"  Saved to: {args.output}")
    print(f"\n→ Next: python phase2_score.py --input {args.output}")

    # Cleanup temp audio
    try:
        os.remove(tmp_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
