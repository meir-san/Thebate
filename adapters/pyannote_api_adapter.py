"""pyannoteAI cloud API adapter — automated voiceprint pipeline.

One command, one user interaction (name the speakers), correct speaker labels.

Flow:
  1. Upload audio to pyannoteAI
  2. Initial diarization with transcription
  3. Interactive speaker identification (show sample segments, user names them)
  4. Check local voiceprint cache — skip API creation for cached speakers
  5. For uncached speakers: try API voiceprint, fall back to resemblyzer
  6. Save new voiceprints to cache for future reuse
  7. Re-run with voiceprint identification OR local resemblyzer matching
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from adapters.base import BaseAdapter

DEFAULT_VOICEPRINTS_DIR = "./voiceprints/"


def _format_timestamp(ms: int) -> str:
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _extract_audio_clip(input_path: str, start_ms: int, duration_ms: int, output_path: str) -> bool:
    """Extract an audio clip using ffmpeg. Returns True on success."""
    start_s = start_ms / 1000.0
    duration_s = duration_ms / 1000.0
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ss", f"{start_s:.3f}",
                "-t", f"{duration_s:.3f}",
                "-q:a", "0",
                output_path,
            ],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    Warning: ffmpeg extraction failed: {e}")
        return False


def _load_cached_voiceprints(voiceprints_dir: str) -> dict[str, np.ndarray]:
    """Load all .npy voiceprint files from the cache directory."""
    cached = {}
    if not os.path.isdir(voiceprints_dir):
        return cached
    for fname in os.listdir(voiceprints_dir):
        if not fname.endswith(".npy"):
            continue
        name = fname[:-4]
        emb = np.load(os.path.join(voiceprints_dir, fname))
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        cached[name] = emb
    return cached


def _create_resemblyzer_voiceprint(audio_path: str, segments: list[dict]) -> np.ndarray | None:
    """Create a voiceprint using resemblyzer from speaker's audio segments.

    Same logic as match_voiceprints.compute_speaker_embedding but works
    with pyannote API segment format (start/end in seconds).
    """
    import librosa
    from resemblyzer import VoiceEncoder, preprocess_wav

    # Sort by duration, pick longest segments
    sorted_segs = sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True)

    audio_chunks = []
    total_duration = 0.0

    for seg in sorted_segs[:5]:  # use up to 5 longest segments
        start_s = seg["start"]
        duration_s = seg["end"] - seg["start"]
        if duration_s < 1.0:
            continue

        try:
            y, sr = librosa.load(audio_path, sr=16000, offset=start_s, duration=min(duration_s, 10.0))
            if len(y) > 0:
                audio_chunks.append(y)
                total_duration += len(y) / 16000
        except Exception:
            continue

        if total_duration >= 30.0:
            break

    if not audio_chunks:
        return None

    combined = np.concatenate(audio_chunks)
    wav = preprocess_wav(combined, source_sr=16000)

    if len(wav) < 1600:
        return None

    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def _match_segments_with_local_voiceprints(
    audio_path: str,
    diarize_turns: list[dict],
    speaker_names: dict[str, str],
    voiceprints_dir: str,
) -> list[dict]:
    """Match diarization segments against local voiceprints using resemblyzer.

    Returns segments with speaker labels replaced by matched voiceprint names.
    """
    cached = _load_cached_voiceprints(voiceprints_dir)
    if not cached:
        print("  No local voiceprints available for matching.")
        return []

    print(f"\n  Matching against {len(cached)} local voiceprint(s): {', '.join(cached.keys())}")

    # Group segments by diarization speaker label
    speaker_segments: dict[str, list[dict]] = {}
    for seg in diarize_turns:
        speaker = seg.get("speaker", "UNKNOWN")
        speaker_segments.setdefault(speaker, []).append(seg)

    # Compute embedding for each diarization speaker and match against voiceprints
    label_to_name: dict[str, str] = {}

    # Track which voiceprints have been claimed (exclusive matching)
    claimed: set[str] = set()

    # Process speakers by total speaking time (most audio = most reliable match)
    speakers_by_time = sorted(
        speaker_segments.keys(),
        key=lambda s: sum(seg["end"] - seg["start"] for seg in speaker_segments[s]),
        reverse=True,
    )

    for label in speakers_by_time:
        segs = speaker_segments[label]
        total_time = sum(seg["end"] - seg["start"] for seg in segs)
        if total_time < 5.0:  # skip very short speakers
            continue

        embedding = _create_resemblyzer_voiceprint(audio_path, segs)
        if embedding is None:
            continue

        best_name = None
        best_sim = -1.0
        for name, vp_emb in cached.items():
            if name in claimed:
                continue
            sim = float(np.dot(embedding, vp_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name and best_sim > 0.80:
            label_to_name[label] = best_name
            claimed.add(best_name)
            print(f"    {label} -> {best_name} (similarity={best_sim:.3f})")
        else:
            # Use user-provided name if available
            if label in speaker_names:
                label_to_name[label] = speaker_names[label]
                print(f"    {label} -> {speaker_names[label]} (user-named, no voiceprint match)")
            else:
                print(f"    {label} -> no match (best={best_name} at {best_sim:.3f})")

    # Build segments with matched names
    segments = []
    for turn in diarize_turns:
        speaker = turn.get("speaker", "UNKNOWN")
        if speaker in label_to_name:
            speaker = label_to_name[speaker]
        elif speaker in speaker_names:
            speaker = speaker_names[speaker]

        text = turn.get("text", "").strip()
        if not text:
            continue

        segments.append({
            "speaker": speaker,
            "text": text,
            "start_ms": int(turn["start"] * 1000),
            "end_ms": int(turn["end"] * 1000),
        })

    unique_speakers = set(s["speaker"] for s in segments)
    print(f"  Local voiceprint matching: {len(segments)} segments, {len(unique_speakers)} speakers")
    return segments


class PyannoteAPIAdapter(BaseAdapter):
    def __init__(self, voiceprints_path: str = None, voiceprints_dir: str = None):
        self.api_key = os.environ.get("PYANNOTE_API_KEY")
        if not self.api_key:
            print("Error: PYANNOTE_API_KEY not set.")
            print("Get an API key at https://dashboard.pyannote.ai")
            print("Add PYANNOTE_API_KEY=your_key to your .env file.")
            sys.exit(1)
        self.voiceprints_dir = voiceprints_dir or DEFAULT_VOICEPRINTS_DIR
        # Legacy: load pre-made voiceprints from file (skips interactive flow)
        self.voiceprints = self._load_voiceprints(voiceprints_path)

    def _load_voiceprints(self, path: str | None) -> dict[str, str] | None:
        if not path:
            return None
        if not os.path.exists(path):
            print(f"Warning: voiceprints file not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not data:
                print(f"Warning: voiceprints file is empty or not a dict: {path}")
                return None
            print(f"Loaded {len(data)} voiceprint(s): {', '.join(data.keys())}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read voiceprints file: {e}")
            return None

    def transcribe(self, audio_path: str) -> list[dict]:
        from pyannoteai.sdk import Client

        client = Client(token=self.api_key)

        # Upload audio once — reuse the media_url for all API calls
        print("Uploading audio to pyannoteAI...")
        media_url = client.upload(audio_path)

        # If pre-made voiceprints are provided, skip the interactive flow
        if self.voiceprints:
            return self._run_with_voiceprints(client, media_url, self.voiceprints)

        # === Step 2: Initial diarization with transcription ===
        print("Running initial diarization + transcription...")
        job_id = client.diarize(media_url, transcription=True)
        print(f"  Job submitted (ID: {job_id}). Waiting for results", end="", flush=True)
        try:
            diarize_result = client.retrieve(job_id, every_seconds=5)
        except Exception as e:
            print()
            print(f"Error: diarization failed ({type(e).__name__}): {e}")
            sys.exit(1)
        print(" done.")

        diarize_output = diarize_result.get("output", {})
        diarize_turns = diarize_output.get("turnLevelTranscription", [])
        if not diarize_turns:
            print("Warning: pyannoteAI returned no transcription segments.")
            sys.exit(1)

        # Group segments by speaker
        speaker_segments: dict[str, list[dict]] = {}
        for seg in diarize_turns:
            speaker = seg.get("speaker", "UNKNOWN")
            speaker_segments.setdefault(speaker, []).append(seg)

        # Sort speakers by total speaking time (descending)
        speaker_time = {}
        for speaker, segs in speaker_segments.items():
            speaker_time[speaker] = sum(s["end"] - s["start"] for s in segs)
        speakers_by_time = sorted(speaker_time.keys(), key=lambda s: speaker_time[s], reverse=True)

        print(f"\nFound {len(speakers_by_time)} speakers:")
        for s in speakers_by_time:
            secs = speaker_time[s]
            seg_count = len(speaker_segments[s])
            print(f"  {s}: {_format_timestamp(int(secs * 1000))} total ({seg_count} segments)")

        # === Step 3: Interactive speaker identification ===
        print(f"\n{'='*60}")
        print(f"Speaker Identification — name each speaker or type 'skip'")
        print(f"{'='*60}")

        speaker_names: dict[str, str] = {}  # diarization_label -> user_name

        for label in speakers_by_time:
            segs = speaker_segments[label]
            # Pick 2-3 longest segments as samples
            sorted_segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
            samples = sorted_segs[:3]

            total_time = _format_timestamp(int(speaker_time[label] * 1000))
            seg_count = len(segs)
            print(f"\n  {label} ({seg_count} segments, {total_time} total)")
            print(f"  {'─'*54}")

            for seg in samples:
                start = _format_timestamp(int(seg["start"] * 1000))
                end = _format_timestamp(int(seg["end"] * 1000))
                text = seg.get("text", "").strip()
                words = text.split()[:25]
                preview = " ".join(words)
                if len(text.split()) > 25:
                    preview += "..."
                print(f'    [{start} - {end}] "{preview}"')

            name = input(f"\n  Who is this? Enter name (or 'skip'): ").strip()
            if name and name.lower() != "skip":
                speaker_names[label] = name
                print(f"  -> {name}")
            else:
                print(f"  -> Keeping as {label}")

        if not speaker_names:
            print("\nNo speakers named — using diarization labels.")
            return self._build_segments_from_diarize(diarize_turns, {})

        # === Step 4: Create voiceprints (with caching) ===
        print(f"\n{'='*60}")
        print("Creating voiceprints...")
        print(f"{'='*60}")

        # Load cached voiceprints
        cached_voiceprints = _load_cached_voiceprints(self.voiceprints_dir)
        if cached_voiceprints:
            print(f"  Found {len(cached_voiceprints)} cached voiceprint(s): {', '.join(cached_voiceprints.keys())}")

        api_voiceprints: dict[str, str] = {}  # for pyannoteAI identify API
        any_new_voiceprints = False
        tmp_dir = tempfile.mkdtemp(prefix="thebate_vp_")

        for label, name in speaker_names.items():
            # Check cache first
            if name in cached_voiceprints:
                print(f"\n  Found cached voiceprint for {name} — skipping creation")
                continue

            print(f"\n  Creating voiceprint for {name} ({label})...")

            segs = speaker_segments[label]
            sorted_segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
            best_segs = sorted_segs[:3]

            # Extract audio clips
            clip_paths = []
            for i, seg in enumerate(best_segs):
                start_ms = int(seg["start"] * 1000)
                duration_ms = int((seg["end"] - seg["start"]) * 1000)
                clip_path = os.path.join(tmp_dir, f"{name}_{i}.mp3")
                if _extract_audio_clip(audio_path, start_ms, duration_ms, clip_path):
                    clip_paths.append(clip_path)
                    dur_s = duration_ms / 1000.0
                    print(f"    Extracted clip {i+1}: {_format_timestamp(start_ms)} ({dur_s:.1f}s)")

            if not clip_paths:
                print(f"    Error: no clips extracted for {name}, skipping.")
                continue

            # Concatenate clips
            concat_path = os.path.join(tmp_dir, f"{name}_concat.mp3")
            if len(clip_paths) == 1:
                concat_path = clip_paths[0]
            else:
                list_path = os.path.join(tmp_dir, f"{name}_list.txt")
                with open(list_path, "w") as f:
                    for cp in clip_paths:
                        f.write(f"file '{cp}'\n")
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", list_path, "-q:a", "0", concat_path,
                        ],
                        capture_output=True, check=True,
                    )
                except subprocess.CalledProcessError:
                    print(f"    Warning: concat failed, using first clip only.")
                    concat_path = clip_paths[0]

            # Try API voiceprint creation first
            api_success = False
            try:
                print(f"    Uploading clip for API voiceprint...")
                clip_url = client.upload(concat_path)
                print(f"    Creating API voiceprint...")
                vp_job_id = client.voiceprint(clip_url)
                vp_result = client.retrieve(vp_job_id, every_seconds=3)
                vp_string = vp_result.get("output", {}).get("voiceprint")
                if vp_string:
                    api_voiceprints[name] = vp_string
                    api_success = True
                    print(f"    API voiceprint created for {name}")
                else:
                    print(f"    Warning: API returned no voiceprint string.")
            except Exception as e:
                error_str = str(e)
                if "402" in error_str or "payment" in error_str.lower():
                    print(f"    API voiceprint failed (402 payment required).")
                else:
                    print(f"    API voiceprint failed: {type(e).__name__}: {e}")

            # Fall back to resemblyzer for local voiceprint
            if not api_success:
                print(f"    Creating local voiceprint with resemblyzer...")
                local_emb = _create_resemblyzer_voiceprint(audio_path, segs)
                if local_emb is not None:
                    os.makedirs(self.voiceprints_dir, exist_ok=True)
                    vp_path = os.path.join(self.voiceprints_dir, f"{name}.npy")
                    np.save(vp_path, local_emb)
                    cached_voiceprints[name] = local_emb
                    any_new_voiceprints = True
                    print(f"    Local voiceprint saved: {vp_path}")
                else:
                    print(f"    Warning: could not create local voiceprint for {name}")
            else:
                # Also save a local resemblyzer voiceprint for future cache
                print(f"    Also caching local resemblyzer voiceprint...")
                local_emb = _create_resemblyzer_voiceprint(audio_path, segs)
                if local_emb is not None:
                    os.makedirs(self.voiceprints_dir, exist_ok=True)
                    vp_path = os.path.join(self.voiceprints_dir, f"{name}.npy")
                    np.save(vp_path, local_emb)
                    any_new_voiceprints = True
                    print(f"    Cached: {vp_path}")

        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        if any_new_voiceprints:
            print(f"\n  New voiceprints saved to {self.voiceprints_dir}")

        # === Step 5: Identification ===
        # Try API identification if we have API voiceprints
        if api_voiceprints:
            print(f"\n{'='*60}")
            print(f"Re-running with {len(api_voiceprints)} API voiceprint(s)...")
            print(f"{'='*60}")

            return self._run_with_voiceprints(
                client, media_url, api_voiceprints,
                fallback_diarize_turns=diarize_turns,
                fallback_names=speaker_names,
                audio_path=audio_path,
            )

        # No API voiceprints — try local resemblyzer matching
        if cached_voiceprints:
            print(f"\n{'='*60}")
            print("No API voiceprints — running local resemblyzer matching...")
            print(f"{'='*60}")

            segments = _match_segments_with_local_voiceprints(
                audio_path, diarize_turns, speaker_names, self.voiceprints_dir,
            )
            if segments:
                return segments

        # Last resort: diarization labels with name mapping
        print("\nNo voiceprints available — using diarization labels with name mapping.")
        return self._build_segments_from_diarize(diarize_turns, speaker_names)

    def _run_with_voiceprints(
        self,
        client,
        media_url: str,
        voiceprints: dict[str, str],
        fallback_diarize_turns: list[dict] | None = None,
        fallback_names: dict[str, str] | None = None,
        audio_path: str | None = None,
    ) -> list[dict]:
        """Run identification with voiceprints and build segments."""
        # Debug: validate voiceprint values before sending
        print(f"Submitting identification job with {len(voiceprints)} voiceprint(s)...")
        clean_voiceprints: dict[str, str] = {}
        for name, vp in voiceprints.items():
            # Handle case where vp is a full API response dict instead of a string
            if isinstance(vp, dict):
                print(f"  Warning: voiceprint for '{name}' is a dict (keys: {list(vp.keys())}), extracting string...")
                vp = vp.get("voiceprint", vp.get("output", {}).get("voiceprint", ""))
            if not isinstance(vp, str) or len(vp) < 10:
                print(f"  Warning: voiceprint for '{name}' is invalid (type={type(vp).__name__}, len={len(vp) if isinstance(vp, str) else 'N/A'}), skipping.")
                continue
            clean_voiceprints[name] = vp
            print(f"  {name}: type={type(vp).__name__}, len={len(vp)}")

        if not clean_voiceprints:
            print("Error: no valid voiceprints to submit.")
            # Try local matching before giving up
            if fallback_diarize_turns and audio_path:
                segments = _match_segments_with_local_voiceprints(
                    audio_path, fallback_diarize_turns, fallback_names or {},
                    self.voiceprints_dir,
                )
                if segments:
                    return segments
            if fallback_diarize_turns:
                return self._build_segments_from_diarize(
                    fallback_diarize_turns, fallback_names or {}
                )
            sys.exit(1)

        try:
            job_id = client.identify(
                media_url,
                voiceprints=clean_voiceprints,
                exclusive_matching=True,
                confidence=True,
            )
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response'):
                try:
                    error_msg += f"\n  Response body: {e.response.text}"
                except Exception:
                    pass
            elif hasattr(e, '__cause__') and e.__cause__:
                error_msg += f"\n  Cause: {e.__cause__}"
            print(f"Error submitting identify job: {error_msg}")

            # Try local matching as fallback
            if fallback_diarize_turns and audio_path:
                print("Falling back to local voiceprint matching...")
                segments = _match_segments_with_local_voiceprints(
                    audio_path, fallback_diarize_turns, fallback_names or {},
                    self.voiceprints_dir,
                )
                if segments:
                    return segments

            if fallback_diarize_turns:
                print("Falling back to initial diarization result.")
                return self._build_segments_from_diarize(
                    fallback_diarize_turns, fallback_names or {}
                )
            sys.exit(1)

        print(f"  Job submitted (ID: {job_id}). Waiting for results", end="", flush=True)
        try:
            result = client.retrieve(job_id, every_seconds=5)
        except Exception as e:
            print()
            print(f"Error: identification failed ({type(e).__name__}): {e}")
            if fallback_diarize_turns:
                print("Falling back to initial diarization result.")
                return self._build_segments_from_diarize(
                    fallback_diarize_turns, fallback_names or {}
                )
            sys.exit(1)
        print(" done.")

        output = result.get("output", {})

        # Build speaker ID mapping from identification results
        id_mapping: dict[str, str] = {}
        if "identification" in output:
            for entry in output["identification"]:
                diar_speaker = entry.get("diarizationSpeaker", "")
                match = entry.get("match")
                if diar_speaker and match:
                    id_mapping[diar_speaker] = match
            if id_mapping:
                print(f"Speaker identification: {', '.join(f'{k} -> {v}' for k, v in id_mapping.items())}")

        # Try turn-level transcription first
        turns = output.get("turnLevelTranscription", [])
        if turns:
            return self._build_segments_from_turns(turns, id_mapping)

        # Try word-level transcription — merge consecutive words by same speaker
        words = output.get("wordLevelTranscription", [])
        if words:
            print("No turn-level transcription — merging word-level transcription...")
            return self._build_segments_from_words(words, id_mapping)

        # Fallback: match identification labels against original diarization text
        if fallback_diarize_turns:
            print("No transcription in identify result — matching against initial diarization...")
            diarization = output.get("diarization", [])
            if diarization:
                return self._build_segments_by_timestamp_overlap(
                    diarization, fallback_diarize_turns, id_mapping,
                )

            # Last resort: use diarize result with name mapping
            names = fallback_names or {}
            names.update(id_mapping)
            return self._build_segments_from_diarize(fallback_diarize_turns, names)

        print("Warning: identification returned no usable transcription.")
        sys.exit(1)

    def _build_segments_from_turns(
        self, turns: list[dict], id_mapping: dict[str, str],
    ) -> list[dict]:
        """Build standard segments from turnLevelTranscription."""
        segments = []
        for turn in turns:
            speaker = turn.get("speaker", "UNKNOWN")
            if speaker in id_mapping:
                speaker = id_mapping[speaker]
            else:
                speaker = self._normalize_speaker(speaker)

            text = turn.get("text", "").strip()
            if not text:
                continue

            segments.append({
                "speaker": speaker,
                "text": text,
                "start_ms": int(turn["start"] * 1000),
                "end_ms": int(turn["end"] * 1000),
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Transcription complete: {len(segments)} segments, {len(unique_speakers)} speakers")
        return segments

    def _build_segments_from_words(
        self, words: list[dict], id_mapping: dict[str, str],
    ) -> list[dict]:
        """Merge consecutive words by the same speaker into turns."""
        if not words:
            return []

        segments = []
        current_speaker = None
        current_text = []
        current_start = 0
        current_end = 0

        for w in words:
            speaker = w.get("speaker", "UNKNOWN")
            if speaker in id_mapping:
                speaker = id_mapping[speaker]
            else:
                speaker = self._normalize_speaker(speaker)

            word_text = w.get("word", w.get("text", "")).strip()
            if not word_text:
                continue

            if speaker != current_speaker:
                # Flush previous
                if current_speaker and current_text:
                    segments.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text),
                        "start_ms": int(current_start * 1000),
                        "end_ms": int(current_end * 1000),
                    })
                current_speaker = speaker
                current_text = [word_text]
                current_start = w.get("start", 0)
                current_end = w.get("end", 0)
            else:
                current_text.append(word_text)
                current_end = w.get("end", current_end)

        # Flush last
        if current_speaker and current_text:
            segments.append({
                "speaker": current_speaker,
                "text": " ".join(current_text),
                "start_ms": int(current_start * 1000),
                "end_ms": int(current_end * 1000),
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Merged word-level transcription: {len(segments)} segments, {len(unique_speakers)} speakers")
        return segments

    def _build_segments_by_timestamp_overlap(
        self,
        diarization: list[dict],
        diarize_turns: list[dict],
        id_mapping: dict[str, str],
    ) -> list[dict]:
        """Match identification diarization labels to original transcription by timestamp overlap."""
        id_ranges: list[tuple[float, float, str]] = []
        for entry in diarization:
            speaker = entry.get("speaker", "UNKNOWN")
            if speaker in id_mapping:
                speaker = id_mapping[speaker]
            else:
                speaker = self._normalize_speaker(speaker)
            id_ranges.append((entry["start"], entry["end"], speaker))

        segments = []
        for turn in diarize_turns:
            text = turn.get("text", "").strip()
            if not text:
                continue

            t_start = turn["start"]
            t_end = turn["end"]

            best_speaker = self._normalize_speaker(turn.get("speaker", "UNKNOWN"))
            best_overlap = 0

            for r_start, r_end, r_speaker in id_ranges:
                overlap_start = max(t_start, r_start)
                overlap_end = min(t_end, r_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = r_speaker

            segments.append({
                "speaker": best_speaker,
                "text": text,
                "start_ms": int(t_start * 1000),
                "end_ms": int(t_end * 1000),
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Timestamp-matched transcription: {len(segments)} segments, {len(unique_speakers)} speakers")
        return segments

    def _build_segments_from_diarize(
        self, diarize_turns: list[dict], name_mapping: dict[str, str],
    ) -> list[dict]:
        """Build segments from initial diarization result with optional name mapping."""
        segments = []
        for turn in diarize_turns:
            speaker = turn.get("speaker", "UNKNOWN")
            if speaker in name_mapping:
                speaker = name_mapping[speaker]
            else:
                speaker = self._normalize_speaker(speaker)

            text = turn.get("text", "").strip()
            if not text:
                continue

            segments.append({
                "speaker": speaker,
                "text": text,
                "start_ms": int(turn["start"] * 1000),
                "end_ms": int(turn["end"] * 1000),
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Transcription complete: {len(segments)} segments, {len(unique_speakers)} speakers")
        return segments

    @staticmethod
    def _normalize_speaker(speaker: str) -> str:
        """Normalize raw speaker label to SPEAKER_XX format."""
        if speaker.startswith("SPEAKER_"):
            return speaker
        if speaker == "UNKNOWN":
            return "SPEAKER_99"
        return f"SPEAKER_{speaker}"
