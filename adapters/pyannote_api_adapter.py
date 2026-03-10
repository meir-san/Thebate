"""pyannoteAI cloud API adapter — transcription + diarization in one API call."""
import json
import os
import sys

from adapters.base import BaseAdapter


class PyannoteAPIAdapter(BaseAdapter):
    def __init__(self, voiceprints_path: str = None):
        self.api_key = os.environ.get("PYANNOTE_API_KEY")
        if not self.api_key:
            print("Error: PYANNOTE_API_KEY not set.")
            print("Get an API key at https://dashboard.pyannote.ai")
            print("Add PYANNOTE_API_KEY=your_key to your .env file.")
            sys.exit(1)
        self.voiceprints = self._load_voiceprints(voiceprints_path)

    def _load_voiceprints(self, path: str | None) -> dict[str, str] | None:
        """Load voiceprint ID mapping from a JSON file.

        Expected format: {"SpeakerName": "base64_voiceprint_string", ...}
        Voiceprints are created via pyannoteAI's dashboard or API.
        """
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

        # Upload local file to pyannoteAI
        print("Uploading audio to pyannoteAI...")
        media_url = client.upload(audio_path)

        # Submit job — either identify (with voiceprints) or diarize
        if self.voiceprints:
            print(f"Submitting identification job with {len(self.voiceprints)} voiceprint(s)...")
            job_id = client.identify(
                media_url,
                voiceprints=self.voiceprints,
                transcription=True,
            )
        else:
            print("Submitting diarization + transcription job...")
            job_id = client.diarize(
                media_url,
                transcription=True,
            )

        # Poll for completion
        print(f"Job submitted (ID: {job_id}). Waiting for results", end="", flush=True)
        try:
            result = client.retrieve(job_id, every_seconds=5)
        except Exception as e:
            print()
            error_type = type(e).__name__
            print(f"Error: pyannoteAI job failed ({error_type}): {e}")
            sys.exit(1)
        print(" done.")

        output = result.get("output", {})

        # Build speaker name mapping from identification results
        id_mapping = {}
        if self.voiceprints and "identification" in output:
            for entry in output["identification"]:
                diar_speaker = entry.get("diarizationSpeaker", "")
                match = entry.get("match")
                if diar_speaker and match:
                    id_mapping[diar_speaker] = match

        # Parse turnLevelTranscription into standard adapter format
        turns = output.get("turnLevelTranscription", [])
        if not turns:
            print("Warning: pyannoteAI returned no transcription segments.")
            sys.exit(1)

        segments = []
        for turn in turns:
            speaker = turn.get("speaker", "UNKNOWN")

            # Apply voiceprint identification mapping
            if speaker in id_mapping:
                speaker = id_mapping[speaker]
            else:
                # Normalize to SPEAKER_XX format
                if not speaker.startswith("SPEAKER_"):
                    if speaker == "UNKNOWN":
                        speaker = "SPEAKER_99"
                    else:
                        speaker = f"SPEAKER_{speaker}"

            text = turn.get("text", "").strip()
            if not text:
                continue

            start_ms = int(turn["start"] * 1000)
            end_ms = int(turn["end"] * 1000)

            segments.append({
                "speaker": speaker,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Transcription complete: {len(segments)} segments, {len(unique_speakers)} speakers")
        if id_mapping:
            print(f"Identified speakers: {', '.join(f'{k} → {v}' for k, v in id_mapping.items())}")

        return segments
