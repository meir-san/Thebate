"""Voiceprint database manager — list, add, delete, compare, and match voiceprints.

Usage:
  voiceprint_db.py list                                          — show all saved voiceprints
  voiceprint_db.py add --name "X" --source audio.mp3 --start 30 --end 45  — enroll a speaker
  voiceprint_db.py delete --name "X"                             — remove a voiceprint
  voiceprint_db.py compare --name1 "X" --name2 "Y"              — show similarity between two voiceprints
  voiceprint_db.py match --audio file.mp3                        — match audio against all voiceprints

Default voiceprints directory: ./voiceprints/. Configurable via --db-dir.
"""
import argparse
import os
import sys

import numpy as np
from dotenv import load_dotenv

DEFAULT_DB_DIR = "./voiceprints/"


def cmd_list(args):
    """List all saved voiceprints."""
    db_dir = args.db_dir
    if not os.path.isdir(db_dir):
        print(f"No voiceprints directory at {db_dir}")
        return

    files = sorted(f for f in os.listdir(db_dir) if f.endswith(".npy"))
    if not files:
        print(f"No voiceprints in {db_dir}")
        return

    print(f"Voiceprints in {db_dir}:")
    for f in files:
        name = f[:-4]
        path = os.path.join(db_dir, f)
        emb = np.load(path)
        print(f"  {name:<25} shape={emb.shape}  file={path}")
    print(f"\n{len(files)} voiceprint(s) total")


def cmd_add(args):
    """Enroll a new voiceprint."""
    from enroll_speakers import compute_embedding, _is_url
    import subprocess
    import tempfile

    if args.end <= args.start:
        print("Error: --end must be greater than --start")
        sys.exit(1)

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

    print(f"Computing voice embedding for {args.start:.1f}s – {args.end:.1f}s...")
    embedding = compute_embedding(audio_path, args.start, args.end)

    os.makedirs(args.db_dir, exist_ok=True)
    out_path = os.path.join(args.db_dir, f"{args.name}.npy")

    if os.path.exists(out_path) and not args.force:
        print(f"Voiceprint '{args.name}' already exists. Use --force to overwrite.")
        sys.exit(1)

    np.save(out_path, embedding)
    print(f"\n✓ Enrolled '{args.name}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Saved to: {out_path}")

    if cleanup:
        try:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def cmd_delete(args):
    """Delete a voiceprint."""
    path = os.path.join(args.db_dir, f"{args.name}.npy")
    if not os.path.exists(path):
        print(f"Voiceprint '{args.name}' not found at {path}")
        sys.exit(1)

    os.remove(path)
    print(f"✓ Deleted voiceprint '{args.name}' ({path})")


def cmd_compare(args):
    """Compare two voiceprints."""
    path1 = os.path.join(args.db_dir, f"{args.name1}.npy")
    path2 = os.path.join(args.db_dir, f"{args.name2}.npy")

    for p, n in [(path1, args.name1), (path2, args.name2)]:
        if not os.path.exists(p):
            print(f"Voiceprint '{n}' not found at {p}")
            sys.exit(1)

    emb1 = np.load(path1)
    emb2 = np.load(path2)
    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1) if np.linalg.norm(emb1) > 0 else emb1
    emb2 = emb2 / np.linalg.norm(emb2) if np.linalg.norm(emb2) > 0 else emb2

    sim = float(np.dot(emb1, emb2))
    print(f"Cosine similarity between '{args.name1}' and '{args.name2}': {sim:.4f}")

    if sim > 0.90:
        print("  -> Very high — likely the same person")
    elif sim > 0.80:
        print("  -> High — possibly the same person or very similar voices")
    elif sim > 0.70:
        print("  -> Moderate — different people with some vocal similarity")
    else:
        print("  -> Low — clearly different people")


def cmd_match(args):
    """Match an audio file against all voiceprints."""
    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    from enroll_speakers import compute_embedding

    db_dir = args.db_dir
    if not os.path.isdir(db_dir):
        print(f"No voiceprints directory at {db_dir}")
        sys.exit(1)

    # Load all voiceprints
    voiceprints = {}
    for f in sorted(os.listdir(db_dir)):
        if not f.endswith(".npy"):
            continue
        name = f[:-4]
        emb = np.load(os.path.join(db_dir, f))
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        voiceprints[name] = emb

    if not voiceprints:
        print(f"No voiceprints in {db_dir}")
        sys.exit(1)

    start = args.start or 0
    end = args.end
    if end is None:
        # Use first 30 seconds by default
        import librosa
        duration = librosa.get_duration(path=args.audio)
        end = min(duration, 30.0)

    print(f"Computing embedding for {args.audio} ({start:.1f}s – {end:.1f}s)...")
    embedding = compute_embedding(args.audio, start, end)

    print(f"\nMatching against {len(voiceprints)} voiceprint(s):")
    results = []
    for name, vp_emb in voiceprints.items():
        sim = float(np.dot(embedding, vp_emb))
        results.append((name, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    for name, sim in results:
        marker = " <-- BEST MATCH" if name == results[0][0] else ""
        print(f"  {name:<25} {sim:.4f}{marker}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Voiceprint database manager")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help=f"Voiceprints directory (default: {DEFAULT_DB_DIR})")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    subparsers.add_parser("list", help="List all saved voiceprints")

    # add
    add_parser = subparsers.add_parser("add", help="Enroll a new voiceprint")
    add_parser.add_argument("--name", required=True, help="Speaker name")
    add_parser.add_argument("--source", required=True, help="Audio file path or YouTube URL")
    add_parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    add_parser.add_argument("--end", type=float, required=True, help="End time in seconds")
    add_parser.add_argument("--force", action="store_true", help="Overwrite existing voiceprint")

    # delete
    del_parser = subparsers.add_parser("delete", help="Delete a voiceprint")
    del_parser.add_argument("--name", required=True, help="Speaker name to delete")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two voiceprints")
    cmp_parser.add_argument("--name1", required=True, help="First speaker name")
    cmp_parser.add_argument("--name2", required=True, help="Second speaker name")

    # match
    match_parser = subparsers.add_parser("match", help="Match audio against all voiceprints")
    match_parser.add_argument("--audio", required=True, help="Audio file to match")
    match_parser.add_argument("--start", type=float, default=None, help="Start time (default: 0)")
    match_parser.add_argument("--end", type=float, default=None, help="End time (default: first 30s)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "list": cmd_list,
        "add": cmd_add,
        "delete": cmd_delete,
        "compare": cmd_compare,
        "match": cmd_match,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
