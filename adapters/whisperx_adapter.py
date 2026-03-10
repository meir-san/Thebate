import gc
import glob
import os
import sys

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from adapters.base import BaseAdapter

SPEAKER_MATCH_THRESHOLD = 0.55


class WhisperXAdapter(BaseAdapter):
    def __init__(self, model_name: str = "large-v2", speakers_dir: str = None):
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            print("Error: HF_TOKEN not set.")
            print("Add HF_TOKEN=hf_xxx to your .env file.")
            print("You must also accept terms for pyannote/segmentation-3.0")
            print("and pyannote/speaker-diarization-3.1 on huggingface.co")
            sys.exit(1)
        self.model_name = model_name
        self.speakers_dir = speakers_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def _load_enrolled_speakers(self) -> dict[str, np.ndarray]:
        """Load enrolled speaker embeddings from speakers_dir."""
        if not self.speakers_dir or not os.path.isdir(self.speakers_dir):
            return {}
        enrolled = {}
        for path in glob.glob(os.path.join(self.speakers_dir, "*.npy")):
            name = os.path.splitext(os.path.basename(path))[0]
            emb = np.load(path)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            enrolled[name] = emb.flatten()
        return enrolled

    def _match_speakers(self, segments: list[dict], audio_path: str,
                        enrolled: dict[str, np.ndarray]) -> dict[str, str]:
        """Match diarized SPEAKER_XX labels to enrolled speakers by voice similarity."""
        from pyannote.audio import Model, Inference
        from pyannote.core import Segment
        from scipy.optimize import linear_sum_assignment

        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        inference = Inference(model, window="whole")

        # Group segments by speaker and pick representative time ranges
        speaker_segs: dict[str, list[dict]] = {}
        for seg in segments:
            speaker_segs.setdefault(seg["speaker"], []).append(seg)

        # Compute average embedding per diarized speaker
        labels = list(speaker_segs.keys())
        label_embeddings: dict[str, np.ndarray] = {}

        for label in labels:
            segs = speaker_segs[label]
            segs_sorted = sorted(segs, key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)
            embeddings = []
            for seg in segs_sorted[:5]:
                start_s = seg["start_ms"] / 1000.0
                end_s = seg["end_ms"] / 1000.0
                if end_s - start_s < 1.0:
                    continue
                try:
                    excerpt = Segment(start_s, end_s)
                    emb = inference.crop(audio_path, excerpt).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)
                except Exception:
                    continue

            if not embeddings:
                print(f"  Warning: could not compute embedding for {label} — no usable segments")
                continue

            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            label_embeddings[label] = avg_emb

        if not label_embeddings:
            return {}

        # Build similarity matrix and solve with Hungarian algorithm
        computed_labels = list(label_embeddings.keys())
        enrolled_names = list(enrolled.keys())
        sim_matrix = np.zeros((len(computed_labels), len(enrolled_names)))

        for i, label in enumerate(computed_labels):
            for j, name in enumerate(enrolled_names):
                sim_matrix[i, j] = float(np.dot(label_embeddings[label], enrolled[name]))

        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping = {}
        for r, c in zip(row_ind, col_ind):
            label = computed_labels[r]
            name = enrolled_names[c]
            sim = sim_matrix[r, c]

            if sim >= SPEAKER_MATCH_THRESHOLD:
                mapping[label] = name
                if sim <= 0.65:
                    print(f"  \u26a0 Low confidence match: {label} \u2192 {name} ({sim:.3f}) \u2014 verify manually")
                else:
                    print(f"  {label} \u2192 {name} (similarity: {sim:.3f})")
            else:
                print(f"  {label} \u2192 no match (best: {name} at {sim:.3f}, below {SPEAKER_MATCH_THRESHOLD})")

        # Log speakers that had no embeddings computed
        for label in labels:
            if label not in label_embeddings and label not in mapping:
                pass  # already warned above

        return mapping

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

        # Step 6: Match against enrolled speaker embeddings
        enrolled = self._load_enrolled_speakers()
        if enrolled:
            print(f"Matching against {len(enrolled)} enrolled speaker(s): {', '.join(enrolled.keys())}...")
            mapping = self._match_speakers(segments, audio_path, enrolled)
            if mapping:
                for seg in segments:
                    if seg["speaker"] in mapping:
                        seg["speaker"] = mapping[seg["speaker"]]
                matched = set(mapping.values())
                unmatched = set(enrolled.keys()) - matched
                if unmatched:
                    print(f"  Unmatched enrolled speakers: {', '.join(unmatched)}")
            else:
                print("  No speakers matched above threshold.")

        return segments
