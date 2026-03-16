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
from speaker_assign import assign_speakers_interactive, apply_mapping

YOUTUBE_RE = re.compile(r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+')


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Ingest YouTube debate video into diarized turns.")
    parser.add_argument("--url", required=True, help="Full YouTube URL")
    parser.add_argument("--topic", required=True, help="Debate topic in plain English")
    parser.add_argument("--speakers", default=None, help="Comma-separated real names in order of first appearance (skips interactive assignment)")
    parser.add_argument("--debaters", default=None, help="Comma-separated names of speakers to score (must be subset of --speakers). If omitted, all speakers are debaters.")
    parser.add_argument("--output", default="turns.json", help="Output JSON path (default: turns.json)")
    parser.add_argument("--adapter", default="assemblyai", choices=["assemblyai", "whisperx", "pyannote-api", "remote-whisperx"], help="Transcription adapter")
    parser.add_argument("--voiceprints", default=None, help="JSON file mapping speaker names to pyannoteAI voiceprint strings (for --adapter pyannote-api). Enroll voiceprints via pyannoteAI dashboard or API first")
    parser.add_argument("--speakers-dir", default="./speakers/", help="Directory with enrolled speaker .npy embeddings (default: ./speakers/)")
    parser.add_argument("--auto-enroll", action="store_true", help="Run interactive auto-enrollment if speakers dir is empty (whisperx only)")
    parser.add_argument("--enroll-model", default="base", help="WhisperX model for enrollment transcription (default: base). Use base or small for fast enrollment — accuracy doesn't matter for speaker ID")
    parser.add_argument("--keep-audio", action="store_true", help="Copy downloaded audio to output directory (e.g., ./sneako/audio.mp3) for later voiceprint matching")
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
    if args.adapter == "pyannote-api" and not os.environ.get("PYANNOTE_API_KEY"):
        print("Error: PYANNOTE_API_KEY not set.")
        print("Get an API key at https://dashboard.pyannote.ai")
        sys.exit(1)
    if args.adapter == "remote-whisperx" and not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN not set.")
        print("Add HF_TOKEN=hf_xxx to your .env file (needed for speaker diarization on GPU box).")
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

    # Auto-enroll if requested and speakers dir is empty
    if args.auto_enroll and args.adapter == "whisperx":
        import glob
        existing = glob.glob(os.path.join(args.speakers_dir, "*.npy"))
        if not existing:
            print("No enrolled speakers found — running auto-enrollment...")
            from auto_enroll import run_auto_enroll
            run_auto_enroll(tmp_path, args.speakers_dir, enroll_model=args.enroll_model)
        else:
            print(f"Found {len(existing)} enrolled speaker(s), skipping auto-enrollment.")

    # Transcribe
    if args.adapter == "whisperx":
        print("Transcribing with WhisperX (this may take a while)...")
        from adapters.whisperx_adapter import WhisperXAdapter
        adapter = WhisperXAdapter(speakers_dir=args.speakers_dir)
    elif args.adapter == "remote-whisperx":
        print("Transcribing with WhisperX on remote GPU server...")
        from adapters.remote_whisperx_adapter import RemoteWhisperXAdapter
        adapter = RemoteWhisperXAdapter()
    elif args.adapter == "pyannote-api":
        print("Transcribing with pyannoteAI cloud API (includes automated voiceprint enrollment)...")
        from adapters.pyannote_api_adapter import PyannoteAPIAdapter
        adapter = PyannoteAPIAdapter(voiceprints_path=args.voiceprints)
    else:
        print("Transcribing with AssemblyAI (this may take a few minutes)...")
        from adapters.assemblyai_adapter import AssemblyAIAdapter
        adapter = AssemblyAIAdapter()
    segments = adapter.transcribe(tmp_path)

    # Build turns
    turns = build_turns(segments)

    # Speaker assignment
    # Check if the adapter already named speakers (e.g. pyannote-api with voiceprints)
    speaker_labels = []
    for t in turns:
        if t.speaker not in speaker_labels:
            speaker_labels.append(t.speaker)
    has_named_speakers = any(not s.startswith("SPEAKER_") for s in speaker_labels)

    if args.speakers:
        # Fast path: use --speakers flag directly
        names = [n.strip() for n in args.speakers.split(",")]

        if len(names) != len(speaker_labels):
            print(f"Warning: --speakers has {len(names)} names but {len(speaker_labels)} speakers detected.")
            print("Mapping what we can, leaving the rest as SPEAKER_XX.")

        mapping = {}
        for i, label in enumerate(speaker_labels):
            if i < len(names):
                mapping[label] = names[i]
        apply_mapping(turns, mapping)
    elif has_named_speakers:
        # Adapter already named speakers (pyannote-api voiceprint flow)
        named = [s for s in speaker_labels if not s.startswith("SPEAKER_")]
        unnamed = [s for s in speaker_labels if s.startswith("SPEAKER_")]
        print(f"\nSpeakers already identified: {', '.join(named)}")
        if unnamed:
            print(f"Unnamed speakers: {', '.join(unnamed)} (use --speakers to name them)")
    else:
        # Interactive assignment
        mapping = assign_speakers_interactive(turns)
        apply_mapping(turns, mapping)

    # Build speaker list in order of first appearance
    speakers = []
    for t in turns:
        if t.speaker not in speakers:
            speakers.append(t.speaker)

    # Determine debaters (case-insensitive matching against speaker names)
    if args.debaters:
        debater_inputs = [n.strip() for n in args.debaters.split(",")]
        speakers_lower = {s.lower(): s for s in speakers}
        debaters = []
        invalid = []
        for d in debater_inputs:
            matched = speakers_lower.get(d.lower())
            if matched:
                debaters.append(matched)
            else:
                invalid.append(d)
        if invalid:
            print(f"Warning: --debaters contains names not in speakers: {invalid}")
            print(f"Known speakers: {speakers}")
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

    # Optionally keep audio for later voiceprint matching
    if args.keep_audio:
        import shutil
        output_dir = os.path.dirname(os.path.abspath(args.output))
        audio_dest = os.path.join(output_dir, "audio.mp3")
        try:
            shutil.copy2(tmp_path, audio_dest)
            print(f"  Audio saved to: {audio_dest}")
        except OSError as e:
            print(f"  Warning: could not save audio: {e}")

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
