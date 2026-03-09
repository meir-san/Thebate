import os
import sys

import assemblyai as aai

from adapters.base import BaseAdapter


class AssemblyAIAdapter(BaseAdapter):
    def __init__(self):
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            print("Error: ASSEMBLYAI_API_KEY not set.")
            print("Get a free key at https://www.assemblyai.com")
            sys.exit(1)
        aai.settings.api_key = api_key

    def transcribe(self, audio_path: str) -> list[dict]:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speech_models=["universal-3-pro"],
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)

        if not transcript.utterances:
            print("Warning: AssemblyAI returned no utterances.")
            print("Diarization may have failed. This can happen with:")
            print("  - Mono audio where speakers can't be distinguished")
            print("  - Very low quality audio")
            print("  - Audio with only one speaker")
            sys.exit(1)

        # Check for single-speaker case
        unique_labels = set(u.speaker for u in transcript.utterances)
        if len(unique_labels) == 1:
            label = next(iter(unique_labels))
            print(f"Warning: Only 1 unique speaker label detected: '{label}'")
            print("Verify the video has 2 speakers and try again, or use --speakers to label manually")

        chr_to_index = lambda c: ord(c.upper()) - ord("A")

        segments = []
        for u in transcript.utterances:
            speaker = f"SPEAKER_{chr_to_index(u.speaker):02d}"
            segments.append({
                "speaker": speaker,
                "text": u.text,
                "start_ms": u.start,
                "end_ms": u.end,
            })

        return segments
