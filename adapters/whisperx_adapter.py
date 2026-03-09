import gc
import os
import sys

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from adapters.base import BaseAdapter


class WhisperXAdapter(BaseAdapter):
    def __init__(self, model_name: str = "large-v2"):
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            print("Error: HF_TOKEN not set.")
            print("Add HF_TOKEN=hf_xxx to your .env file.")
            print("You must also accept terms for pyannote/segmentation-3.0")
            print("and pyannote/speaker-diarization-3.1 on huggingface.co")
            sys.exit(1)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def transcribe(self, audio_path: str) -> list[dict]:
        # Step 1: Transcribe
        print(f"Loading WhisperX model '{self.model_name}' on {self.device}...")
        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
        )
        audio = whisperx.load_audio(audio_path)
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=16, verbose=True)
        language = result.get("language", "en")
        print(f"Detected language: {language}")

        del model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Step 2: Align
        print("Aligning word timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        del model_a
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Step 3: Diarize
        print("Running speaker diarization...")
        diarize_model = DiarizationPipeline(
            token=self.hf_token,
            device=self.device,
        )
        diarize_segments = diarize_model(audio_path)

        # Step 4: Assign speakers
        print("Assigning speakers to segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if not result.get("segments"):
            print("Warning: WhisperX returned no segments.")
            sys.exit(1)

        # Convert to standard adapter format
        segments = []
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            # Normalize speaker labels to SPEAKER_XX format
            if speaker.startswith("SPEAKER_"):
                pass  # already in correct format
            elif speaker == "UNKNOWN":
                speaker = "SPEAKER_99"
            else:
                speaker = f"SPEAKER_{speaker}"

            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            text = seg.get("text", "").strip()
            if not text:
                continue

            segments.append({
                "speaker": speaker,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

        unique_speakers = set(s["speaker"] for s in segments)
        print(f"Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers")

        return segments
