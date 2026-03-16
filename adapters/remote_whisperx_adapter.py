"""Remote WhisperX adapter — runs transcription on a GPU server via SSH.

Uploads audio via SCP, runs remote_transcribe.py on the GPU box,
streams progress via SSH stdout, downloads the result.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import config
from adapters.base import BaseAdapter

# Only one remote transcription can run at a time (shared GPU)
_gpu_lock = threading.Lock()

REMOTE_SCRIPT = "/home/nun/remote_transcribe.py"
REMOTE_TMP_AUDIO = "/tmp/debatestats_audio.mp3"
REMOTE_TMP_OUTPUT = "/tmp/debatestats_segments.json"
REMOTE_CACHE_DIR = "/tmp/debatestats_cache"


class RemoteWhisperXAdapter(BaseAdapter):
    def __init__(self, progress_callback=None):
        """
        Args:
            progress_callback: optional callable(stage: str, message: str)
                stage is one of: uploading, downloading_model, transcribing,
                aligning, diarizing, complete, downloading_result
        """
        self.host = config.REMOTE_WHISPERX_HOST
        self.user = config.REMOTE_WHISPERX_USER
        self.progress_callback = progress_callback or (lambda s, m: None)

    def _ssh_target(self):
        return f"{self.user}@{self.host}"

    def _check_connectivity(self):
        """Verify SSH connectivity before starting."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                 self._ssh_target(), "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise ConnectionError(
                    f"Cannot reach GPU server at {self.host}. "
                    "Make sure the Ubuntu box is running and reachable via Tailscale."
                )
        except subprocess.TimeoutExpired:
            raise ConnectionError(
                f"SSH connection to {self.host} timed out. "
                "Make sure the Ubuntu box is running and reachable via Tailscale."
            )

    def _scp_upload(self, local_path: str, remote_path: str):
        subprocess.run(
            ["scp", "-q", local_path, f"{self._ssh_target()}:{remote_path}"],
            check=True, timeout=300,
        )

    def _scp_download(self, remote_path: str, local_path: str):
        subprocess.run(
            ["scp", "-q", f"{self._ssh_target()}:{remote_path}", local_path],
            check=True, timeout=120,
        )

    def _stop_ollama(self):
        """Stop ollama on the GPU server to free VRAM for transcription."""
        print("Stopping ollama on GPU server to free VRAM...")
        self.progress_callback("stopping_ollama", "Stopping ollama to free GPU memory")
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self._ssh_target(),
             "sudo systemctl stop ollama"],
            capture_output=True, timeout=15,
        )
        time.sleep(3)
        print("  Ollama stopped, GPU memory freed.")

    def _start_ollama(self):
        """Restart ollama on the GPU server for LLM extraction."""
        print("Restarting ollama on GPU server...")
        self.progress_callback("starting_ollama", "Restarting ollama for LLM extraction")
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self._ssh_target(),
             "sudo systemctl start ollama"],
            capture_output=True, timeout=15,
        )
        time.sleep(5)
        print("  Ollama restarted and ready.")

    def transcribe(self, audio_path: str) -> list[dict]:
        if not _gpu_lock.acquire(blocking=False):
            raise RuntimeError(
                "Another remote transcription is already running on the GPU. "
                "Wait for it to finish before starting a new one."
            )
        try:
            return self._transcribe_locked(audio_path)
        finally:
            _gpu_lock.release()

    def _transcribe_locked(self, audio_path: str) -> list[dict]:
        # Check connectivity
        print("Checking SSH connectivity to GPU server...")
        self._check_connectivity()

        # Upload audio (before stopping ollama — no GPU needed for SCP)
        print(f"Uploading audio to {self.host}...")
        self.progress_callback("uploading", f"Uploading audio to {self.host}")
        self._scp_upload(audio_path, REMOTE_TMP_AUDIO)
        print("  Upload complete.")

        # Compute audio hash for cache directory
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        audio_hash = h.hexdigest()[:16]
        cache_dir = f"{REMOTE_CACHE_DIR}_{audio_hash}"

        # Build remote command
        hf_token = os.environ.get("HF_TOKEN", "")
        remote_cmd = (
            f"HF_TOKEN='{hf_token}' python3 {REMOTE_SCRIPT} "
            f"--audio {REMOTE_TMP_AUDIO} --output {REMOTE_TMP_OUTPUT} "
            f"--cache-dir {cache_dir}"
        )

        # Stop ollama to free GPU VRAM, run transcription, restart ollama
        self._stop_ollama()
        try:
            # Run transcription via SSH, streaming stdout for progress
            print(f"Starting WhisperX on {self.host} (GPU)...")
            process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", self._ssh_target(), remote_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Read output line by line for progress tracking
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(f"  [remote] {line}")

                if line.startswith("STAGE:"):
                    stage = line.split(":", 1)[1].strip()
                    stage_labels = {
                        "downloading_model": "Loading WhisperX model on GPU",
                        "transcribing": "Transcribing with WhisperX (GPU)",
                        "aligning": "Aligning transcript",
                        "diarizing": "Diarizing speakers",
                        "complete": "Transcription complete",
                    }
                    self.progress_callback(stage, stage_labels.get(stage, stage))

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Remote transcription failed (exit code {process.returncode}). "
                    "Check the GPU server logs."
                )
        finally:
            # Always restart ollama, even if transcription failed
            self._start_ollama()

        # Download result
        print("Downloading transcription result...")
        self.progress_callback("downloading_result", "Downloading transcription result")
        local_tmp = tempfile.mktemp(suffix=".json")
        self._scp_download(REMOTE_TMP_OUTPUT, local_tmp)

        # Parse segments
        with open(local_tmp) as f:
            raw_segments = json.load(f)
        os.remove(local_tmp)

        # Cleanup remote files
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self._ssh_target(),
             f"rm -f {REMOTE_TMP_AUDIO} {REMOTE_TMP_OUTPUT}"],
            capture_output=True, timeout=10,
        )

        # Convert to standard format (start/end seconds → start_ms/end_ms)
        segments = []
        for seg in raw_segments:
            segments.append({
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip(),
                "start_ms": int(seg.get("start", 0) * 1000),
                "end_ms": int(seg.get("end", 0) * 1000),
            })

        print(f"  Got {len(segments)} segments from remote transcription.")
        return segments
