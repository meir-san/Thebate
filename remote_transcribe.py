#!/usr/bin/env python3
"""Standalone WhisperX transcription script for GPU server.

Deploy to the remote GPU box and run:
    python3 remote_transcribe.py --audio /tmp/audio.mp3 --output /tmp/segments.json

Outputs diarized segments as JSON. Prints STAGE: lines for progress tracking.
Caches intermediate results (transcription, alignment, diarization) so
re-runs skip completed GPU steps.

No dependency on the DebateStats codebase.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time


def audio_hash(path: str) -> str:
    """SHA256 hash of audio file (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="WhisperX GPU transcription")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to write segments JSON")
    parser.add_argument("--cache-dir", default=None, help="Directory for intermediate caches (auto-created from audio hash if omitted)")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for diarization (reads HF_TOKEN env if not set)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN required for speaker diarization. Set HF_TOKEN env var or use --hf-token.", file=sys.stderr)
        sys.exit(1)

    # Set up cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        ahash = audio_hash(args.audio)
        cache_dir = f"/tmp/debatestats_cache_{ahash}"
    os.makedirs(cache_dir, exist_ok=True)

    transcription_cache = os.path.join(cache_dir, "transcription.pkl")
    aligned_cache = os.path.join(cache_dir, "aligned.pkl")
    diarized_cache = os.path.join(cache_dir, "diarized.pkl")

    print(f"Cache dir: {cache_dir}", flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})", flush=True)

    import whisperx

    # Load audio once (needed by multiple stages)
    audio = whisperx.load_audio(args.audio)

    # Stage 1: Transcribe
    if os.path.exists(transcription_cache):
        print("STAGE:transcribing", flush=True)
        with open(transcription_cache, "rb") as f:
            result = pickle.load(f)
        print(f"  Loaded transcription from cache ({len(result['segments'])} segments)", flush=True)
    else:
        print("STAGE:downloading_model", flush=True)
        t0 = time.time()
        model = whisperx.load_model(args.model, device, compute_type=compute_type, language=args.language)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

        print("STAGE:transcribing", flush=True)
        t0 = time.time()
        result = model.transcribe(audio, batch_size=16, language=args.language)
        print(f"  Transcribed in {time.time() - t0:.1f}s ({len(result['segments'])} segments)", flush=True)

        with open(transcription_cache, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved transcription cache", flush=True)

        del model
        torch.cuda.empty_cache()

    # Stage 2: Align
    if os.path.exists(aligned_cache):
        print("STAGE:aligning", flush=True)
        with open(aligned_cache, "rb") as f:
            result = pickle.load(f)
        print(f"  Loaded alignment from cache", flush=True)
    else:
        print("STAGE:aligning", flush=True)
        t0 = time.time()
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.language, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
        print(f"  Aligned in {time.time() - t0:.1f}s", flush=True)

        with open(aligned_cache, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved alignment cache", flush=True)

        del align_model
        torch.cuda.empty_cache()

    # Stage 3: Diarize
    if os.path.exists(diarized_cache):
        print("STAGE:diarizing", flush=True)
        with open(diarized_cache, "rb") as f:
            result = pickle.load(f)
        print(f"  Loaded diarization from cache", flush=True)
    else:
        print("STAGE:diarizing", flush=True)
        t0 = time.time()
        from pyannote.audio import Pipeline as PyannotePipeline
        diarize_pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        diarize_pipeline = diarize_pipeline.to(torch.device(device))
        import torchaudio
        waveform, sample_rate = torchaudio.load(args.audio)
        diarize_segments = diarize_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        import pandas as pd
        annotation = getattr(diarize_segments, "exclusive_speaker_diarization", diarize_segments)
        diarize_df = pd.DataFrame([
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ])
        result = whisperx.assign_word_speakers(diarize_df, result)
        print(f"  Diarized in {time.time() - t0:.1f}s", flush=True)

        with open(diarized_cache, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved diarization cache", flush=True)

    # Build output segments
    segments = []
    for seg in result["segments"]:
        segments.append({
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "text": seg.get("text", "").strip(),
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
        })

    # Write output
    with open(args.output, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"STAGE:complete", flush=True)
    print(f"  Wrote {len(segments)} segments to {args.output}", flush=True)


if __name__ == "__main__":
    main()
