"""Interactive speaker enrollment with automatic segment selection.

Runs diarization to find speakers, picks the cleanest segments,
transcribes them, and prompts the user to name each speaker.
"""
import argparse
import gc
import os
import sys

import numpy as np
import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-enroll speakers from an audio file using diarization.",
        epilog="Example: python auto_enroll.py --audio audio.mp3 --num-speakers 2",
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--speakers-dir", default="./speakers/", help="Directory to save embeddings (default: ./speakers/)")
    parser.add_argument("--num-speakers", type=int, default=2, help="Number of speakers to enroll (default: 2)")
    parser.add_argument("--enroll-model", default="base", help="WhisperX model for enrollment transcription (default: base). Use base or small for fast enrollment — accuracy doesn't matter for speaker ID")
    return parser.parse_args()


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _find_clean_segments(all_segments: list[dict], speaker: str, margin: float = 0.5, min_duration: float = 3.0, n: int = 3) -> list[dict]:
    """Find the n longest segments for a speaker with no overlap from others within ±margin seconds."""
    speaker_segs = [s for s in all_segments if s["speaker"] == speaker]
    other_segs = [s for s in all_segments if s["speaker"] != speaker]

    clean = []
    for seg in speaker_segs:
        start_s = seg["start_ms"] / 1000.0
        end_s = seg["end_ms"] / 1000.0
        if end_s - start_s < min_duration:
            continue
        # Check for overlap with other speakers within margin
        overlaps = False
        for other in other_segs:
            o_start = other["start_ms"] / 1000.0 - margin
            o_end = other["end_ms"] / 1000.0 + margin
            if start_s < o_end and end_s > o_start:
                overlaps = True
                break
        if not overlaps:
            clean.append(seg)

    # Sort by duration descending, take top n
    clean.sort(key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)
    return clean[:n]


def run_auto_enroll(audio_path: str, speakers_dir: str = "./speakers/", num_speakers: int = 2, enroll_model: str = "base"):
    """Run auto enrollment. Can be called from phase1_ingest or standalone."""
    load_dotenv()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set.")
        print("Add HF_TOKEN=hf_xxx to your .env file.")
        sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Step 1: Transcribe with WhisperX
    print(f"Loading WhisperX model '{enroll_model}' on {device}...")
    model = whisperx.load_model(enroll_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16, verbose=True)
    language = result.get("language", "en")
    print(f"Detected language: {language}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 2: Align
    print("Aligning word timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 3: Diarize
    print("Running speaker diarization...")
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio_path)

    # Step 4: Assign speakers to transcribed segments
    print("Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    if not result.get("segments"):
        print("Error: no segments returned from diarization.")
        sys.exit(1)

    # Convert to standard format
    all_segments = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker.startswith("SPEAKER_"):
            pass
        elif speaker == "UNKNOWN":
            speaker = "SPEAKER_99"
        else:
            speaker = f"SPEAKER_{speaker}"

        text = seg.get("text", "").strip()
        if not text:
            continue

        all_segments.append({
            "speaker": speaker,
            "text": text,
            "start_ms": int(seg["start"] * 1000),
            "end_ms": int(seg["end"] * 1000),
        })

    # Find unique speakers sorted by total speaking time (descending)
    speaker_time: dict[str, int] = {}
    for seg in all_segments:
        dur = seg["end_ms"] - seg["start_ms"]
        speaker_time[seg["speaker"]] = speaker_time.get(seg["speaker"], 0) + dur
    speakers_by_time = sorted(speaker_time.keys(), key=lambda s: speaker_time[s], reverse=True)

    print(f"\nFound {len(speakers_by_time)} speakers:")
    for s in speakers_by_time:
        secs = speaker_time[s] / 1000.0
        print(f"  {s}: {_format_timestamp(secs)} total speaking time")

    if len(speakers_by_time) < num_speakers:
        print(f"\nWarning: only {len(speakers_by_time)} speakers detected, requested {num_speakers}.")
        num_speakers = len(speakers_by_time)

    # Take the top N speakers by speaking time
    target_speakers = speakers_by_time[:num_speakers]

    # Step 5: Pick clean candidate segments and present interactive prompt
    print(f"\n{'='*60}")
    print(f"Speaker Enrollment — {num_speakers} speakers")
    print(f"{'='*60}")

    from pyannote.audio import Model, Inference
    from pyannote.core import Segment

    emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(emb_model, window="whole")

    enrolled = {}  # name -> embedding

    for label in target_speakers:
        candidates = _find_clean_segments(all_segments, label)

        # Fallback: if no clean segments, take longest segments regardless of overlap
        if not candidates:
            speaker_segs = [s for s in all_segments if s["speaker"] == label]
            speaker_segs.sort(key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)
            candidates = [s for s in speaker_segs if (s["end_ms"] - s["start_ms"]) >= 3000][:3]

        if not candidates:
            print(f"\n  {label}: no segments >= 3s found, skipping.")
            continue

        # Display samples
        total_time = _format_timestamp(speaker_time[label] / 1000.0)
        seg_count = sum(1 for s in all_segments if s["speaker"] == label)
        print(f"\n  {label} ({seg_count} segments, {total_time} total)")
        print(f"  {'─'*54}")

        for seg in candidates:
            start = _format_timestamp(seg["start_ms"] / 1000.0)
            end = _format_timestamp(seg["end_ms"] / 1000.0)
            # Show first 25 words
            words = seg["text"].split()[:25]
            preview = " ".join(words)
            if len(seg["text"].split()) > 25:
                preview += "..."
            print(f"    [{start} - {end}] \"{preview}\"")

        name = input(f"\n  Who is this? Enter name (or 'skip'): ").strip()
        if not name or name.lower() == "skip":
            print(f"  → Skipping {label}")
            continue

        # Step 6: Compute embeddings from candidate segments
        print(f"  Computing voice embedding for {name}...")
        embeddings = []
        for seg in candidates:
            start_s = seg["start_ms"] / 1000.0
            end_s = seg["end_ms"] / 1000.0
            try:
                excerpt = Segment(start_s, end_s)
                emb = inference.crop(audio_path, excerpt).flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
            except Exception as e:
                print(f"    Warning: failed to extract embedding from [{_format_timestamp(start_s)} - {_format_timestamp(end_s)}]: {e}")
                continue

        if not embeddings:
            print(f"  Error: could not compute any embeddings for {name}, skipping.")
            continue

        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm

        # Save
        os.makedirs(speakers_dir, exist_ok=True)
        out_path = os.path.join(speakers_dir, f"{name}.npy")
        np.save(out_path, avg_emb)
        enrolled[name] = avg_emb

        print(f"  ✓ Enrolled '{name}' ({len(embeddings)} segments averaged)")
        print(f"    Saved to: {out_path}")

    # Step 7: Print similarity matrix
    if len(enrolled) >= 2:
        names = list(enrolled.keys())
        print(f"\n{'='*60}")
        print("Speaker Similarity Matrix (cosine)")
        print(f"{'='*60}")

        # Header
        col_width = max(len(n) for n in names) + 2
        header = " " * col_width
        for n in names:
            header += f"{n:>{col_width}}"
        print(header)

        # Rows
        for i, name_i in enumerate(names):
            row = f"{name_i:<{col_width}}"
            for j, name_j in enumerate(names):
                sim = float(np.dot(enrolled[name_i], enrolled[name_j]))
                if i == j:
                    row += f"{'1.000':>{col_width}}"
                else:
                    row += f"{sim:>{col_width}.3f}"
            print(row)

        # Warn if any pair is too similar
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = float(np.dot(enrolled[names[i]], enrolled[names[j]]))
                if sim > 0.75:
                    print(f"\n  ⚠ {names[i]} and {names[j]} are very similar ({sim:.3f}) — enrollment may be from the same speaker")

    elif len(enrolled) == 1:
        name = list(enrolled.keys())[0]
        print(f"\n✓ Enrolled 1 speaker: {name}")
    else:
        print("\nNo speakers enrolled.")
        return

    print(f"\n✓ Enrollment complete: {len(enrolled)} speaker(s) in {speakers_dir}")


def main():
    args = parse_args()
    run_auto_enroll(args.audio, args.speakers_dir, args.num_speakers, args.enroll_model)


if __name__ == "__main__":
    main()
