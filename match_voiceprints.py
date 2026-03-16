"""Match speakers in a turns.json against saved voiceprint embeddings.

Takes a turns.json file and a voiceprints directory, extracts audio segments
per speaker, computes embeddings with resemblyzer, matches against saved
voiceprints by cosine similarity, and reassigns speaker labels.

Requires the source audio file to extract speaker segments from.
"""
import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match speakers in turns.json against saved voiceprints.",
        epilog="Example: python match_voiceprints.py --turns turns.json --audio audio.mp3 --voiceprints-dir ./voiceprints/",
    )
    parser.add_argument("--turns", required=True, help="Path to turns.json")
    parser.add_argument("--audio", default=None, help="Path to source audio file (auto-detected from turns.json directory if omitted)")
    parser.add_argument("--voiceprints-dir", default="./voiceprints/", help="Directory with .npy voiceprint files (default: ./voiceprints/)")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--min-similarity", type=float, default=0.85, help="Minimum cosine similarity to match (default: 0.85)")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without modifying the file")
    return parser.parse_args()


def load_voiceprints(voiceprints_dir: str) -> dict[str, np.ndarray]:
    """Load all .npy voiceprint files from directory."""
    voiceprints = {}
    if not os.path.isdir(voiceprints_dir):
        print(f"Error: voiceprints directory not found: {voiceprints_dir}")
        sys.exit(1)

    for fname in os.listdir(voiceprints_dir):
        if not fname.endswith(".npy"):
            continue
        name = fname[:-4]  # strip .npy
        path = os.path.join(voiceprints_dir, fname)
        emb = np.load(path)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        voiceprints[name] = emb
        print(f"  Loaded voiceprint: {name} (shape={emb.shape})")

    return voiceprints


def compute_speaker_embedding(
    audio_path: str,
    segments: list[dict],
    max_duration: float = 30.0,
) -> np.ndarray | None:
    """Compute a speaker embedding from their audio segments.

    Uses up to max_duration seconds of their longest segments.
    """
    import librosa
    from resemblyzer import VoiceEncoder, preprocess_wav

    # Sort by duration, pick longest segments up to max_duration total
    sorted_segs = sorted(segments, key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)

    audio_chunks = []
    total_duration = 0.0

    for seg in sorted_segs:
        start_s = seg["start_ms"] / 1000.0
        duration_s = (seg["end_ms"] - seg["start_ms"]) / 1000.0
        if duration_s < 1.0:
            continue

        try:
            y, sr = librosa.load(audio_path, sr=16000, offset=start_s, duration=min(duration_s, 10.0))
            if len(y) > 0:
                audio_chunks.append(y)
                total_duration += len(y) / 16000
        except Exception:
            continue

        if total_duration >= max_duration:
            break

    if not audio_chunks:
        return None

    # Concatenate and compute embedding
    combined = np.concatenate(audio_chunks)
    wav = preprocess_wav(combined, source_sr=16000)

    if len(wav) < 1600:  # less than 0.1s of usable audio
        return None

    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def main():
    load_dotenv()
    args = parse_args()

    if not os.path.exists(args.turns):
        print(f"Error: turns file not found: {args.turns}")
        sys.exit(1)

    # Auto-detect audio file if not specified
    if args.audio is None:
        turns_dir = os.path.dirname(os.path.abspath(args.turns))
        for audio_name in ["audio.mp3", "audio.wav", "audio.m4a"]:
            candidate = os.path.join(turns_dir, audio_name)
            if os.path.exists(candidate):
                args.audio = candidate
                print(f"Auto-detected audio: {candidate}")
                break
        if args.audio is None:
            print(f"Error: no --audio specified and no audio file found in {turns_dir}")
            print("Use --keep-audio with phase1_ingest.py to save audio for later matching.")
            sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    # Load turns
    with open(args.turns, "r", encoding="utf-8") as f:
        data = json.load(f)

    turns = data["turns"]
    print(f"Loaded {len(turns)} turns from {args.turns}")

    # Load voiceprints
    print(f"\nLoading voiceprints from {args.voiceprints_dir}...")
    voiceprints = load_voiceprints(args.voiceprints_dir)
    if not voiceprints:
        print("No voiceprints found. Enroll speakers first with enroll_speakers.py")
        sys.exit(1)
    print(f"  {len(voiceprints)} voiceprint(s) loaded\n")

    # Group turns by speaker
    speaker_turns: dict[str, list[dict]] = {}
    for t in turns:
        speaker_turns.setdefault(t["speaker"], []).append(t)

    # Compute embedding for each speaker and match against voiceprints
    print("Computing speaker embeddings and matching...")
    mapping: dict[str, str] = {}  # old_label -> new_name

    for speaker, segs in speaker_turns.items():
        total_ms = sum(s["end_ms"] - s["start_ms"] for s in segs)
        print(f"\n  {speaker} ({len(segs)} turns, {total_ms / 1000:.0f}s total)")

        embedding = compute_speaker_embedding(args.audio, segs)
        if embedding is None:
            print(f"    Could not compute embedding (no usable audio)")
            continue

        # Compare against all voiceprints
        best_name = None
        best_sim = -1.0
        for name, vp_emb in voiceprints.items():
            sim = float(np.dot(embedding, vp_emb))
            print(f"    vs {name}: {sim:.3f}")
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= args.min_similarity:
            mapping[speaker] = best_name
            print(f"    -> MATCH: {best_name} (similarity={best_sim:.3f})")
        else:
            print(f"    -> No match (best={best_name} at {best_sim:.3f}, threshold={args.min_similarity})")

    if not mapping:
        print("\nNo speakers matched. Try lowering --min-similarity or enrolling more speakers.")
        return

    # Apply mapping
    print(f"\n{'='*50}")
    print("Speaker mapping:")
    for old, new in mapping.items():
        print(f"  {old} -> {new}")

    if args.dry_run:
        print("\n(dry run — no changes written)")
        return

    # Update turns
    changed = 0
    for t in turns:
        if t["speaker"] in mapping:
            t["speaker"] = mapping[t["speaker"]]
            changed += 1

    # Update speakers list
    new_speakers = []
    for s in data.get("speakers", []):
        mapped = mapping.get(s, s)
        if mapped not in new_speakers:
            new_speakers.append(mapped)
    data["speakers"] = new_speakers

    # Update debaters list
    if "debaters" in data:
        new_debaters = []
        for d in data["debaters"]:
            mapped = mapping.get(d, d)
            if mapped not in new_debaters:
                new_debaters.append(mapped)
        data["debaters"] = new_debaters

    output_path = args.output or args.turns
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Updated {changed} turns")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
