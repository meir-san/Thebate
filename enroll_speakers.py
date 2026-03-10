"""Enroll a speaker's voice by extracting an embedding from a known audio segment."""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll a speaker voice from a YouTube URL or local audio file.",
        epilog="Example: python enroll_speakers.py --source https://youtu.be/XXX --name \"John\" --start 30 --end 45",
    )
    parser.add_argument("--source", required=True, help="YouTube URL or path to local audio file")
    parser.add_argument("--name", required=True, help="Speaker name (used as filename)")
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds where speaker is talking alone")
    parser.add_argument("--end", type=float, required=True, help="End time in seconds")
    parser.add_argument("--speakers-dir", default="./speakers/", help="Directory to save embeddings (default: ./speakers/)")
    return parser.parse_args()


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def main():
    load_dotenv()
    args = parse_args()

    if args.end <= args.start:
        print("Error: --end must be greater than --start")
        sys.exit(1)

    duration = args.end - args.start
    if duration < 3:
        print(f"Warning: segment is only {duration:.1f}s — recommend at least 5s for reliable embedding")
    if duration > 60:
        print(f"Warning: segment is {duration:.1f}s — only the first ~30s are needed")

    # Get audio file
    if _is_url(args.source):
        print(f"Downloading audio from: {args.source}")
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        try:
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, args.source],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("Failed to download audio.")
            sys.exit(1)
        cleanup = True
    else:
        if not os.path.exists(args.source):
            print(f"Error: {args.source} not found")
            sys.exit(1)
        audio_path = args.source
        cleanup = False

    # Compute embedding
    print(f"Computing voice embedding for {args.start:.1f}s – {args.end:.1f}s...")
    from pyannote.audio import Model, Inference
    from pyannote.core import Segment

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")

    excerpt = Segment(args.start, args.end)
    embedding = inference.crop(audio_path, excerpt).flatten()
    # embedding is now (256,) numpy array

    # Normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    # Speech energy check
    import librosa
    y, sr = librosa.load(audio_path, sr=None, offset=args.start, duration=args.end - args.start)
    rms = librosa.feature.rms(y=y)
    mean_rms = float(np.mean(rms))
    if mean_rms < 0.01:
        print(f"Warning: audio segment has very low energy (RMS={mean_rms:.4f}) — may not contain speech. Pick a louder segment.")

    # Save
    os.makedirs(args.speakers_dir, exist_ok=True)
    out_path = os.path.join(args.speakers_dir, f"{args.name}.npy")
    np.save(out_path, embedding)

    print(f"\n✓ Enrolled '{args.name}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Saved to: {out_path}")

    # Cleanup
    if cleanup:
        try:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
