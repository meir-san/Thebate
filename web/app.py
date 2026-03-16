#!/usr/bin/env python3
"""Local web UI for debate analysis.

Intelligent caching: checks for existing data and lets the user resume or view results.
Output goes to ./debates/{video_id}/ by default.

Launch with:
    python3 web/app.py
    python3 -m web.app

Opens on http://localhost:8080
"""

import hashlib
import json
import os
import re
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from types import SimpleNamespace
from urllib.parse import urlparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PORT = 8080
VOICEPRINTS_DIR = os.path.join(PROJECT_ROOT, "voiceprints")
DEBATES_DIR = os.path.join(PROJECT_ROOT, "debates")


def extract_video_id(url: str) -> str | None:
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', url)
    return m.group(1) if m else None


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ── Global state ─────────────────────────────────────────────────
state = {
    "screen": "new",        # new | ingest | speakers | processing | results
    "phase": None,
    "message": "",
    "progress": 0,
    "total": 0,
    "speakers": [],         # [{label, turn_count, samples, suggested_name}]
    "results": None,
    "output_dir": None,
    "url": None,
    "topic": None,
    "error": None,
    "existing": None,       # {turns, structured, scored, report} — set by check-url
    "video_id": None,
    "pipeline_steps": [],   # [{id, label, status: pending|running|complete|failed, detail}]
    "pipeline_running": False,
}
state_lock = threading.Lock()


def _make_pipeline_steps():
    """Create fresh pipeline step list."""
    return [
        {"id": "download",   "label": "Download audio",                   "status": "pending", "detail": ""},
        {"id": "upload",     "label": "Upload audio to GPU server",       "status": "pending", "detail": ""},
        {"id": "transcribe", "label": "Transcribe with WhisperX (GPU)",   "status": "pending", "detail": ""},
        {"id": "align",      "label": "Align transcript",                 "status": "pending", "detail": ""},
        {"id": "diarize",    "label": "Diarize speakers",                 "status": "pending", "detail": ""},
        {"id": "speakers",   "label": "Speaker identification",           "status": "pending", "detail": ""},
        {"id": "extract",    "label": "Structure extraction (LLM)",       "status": "pending", "detail": ""},
        {"id": "score",      "label": "Scoring",                          "status": "pending", "detail": ""},
        {"id": "render",     "label": "Generating report",                "status": "pending", "detail": ""},
    ]


def _make_simple_pipeline_steps():
    """Pipeline steps for non-remote adapters (no upload/GPU stages)."""
    return [
        {"id": "download",   "label": "Download audio",                   "status": "pending", "detail": ""},
        {"id": "transcribe", "label": "Transcribing audio",               "status": "pending", "detail": ""},
        {"id": "speakers",   "label": "Speaker identification",           "status": "pending", "detail": ""},
        {"id": "extract",    "label": "Structure extraction (LLM)",       "status": "pending", "detail": ""},
        {"id": "score",      "label": "Scoring",                          "status": "pending", "detail": ""},
        {"id": "render",     "label": "Generating report",                "status": "pending", "detail": ""},
    ]


def _update_step(step_id, status, detail=""):
    """Update a pipeline step's status."""
    with state_lock:
        for step in state["pipeline_steps"]:
            if step["id"] == step_id:
                step["status"] = status
                step["detail"] = detail
                break


# ── HTTP Server ──────────────────────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            self._serve_file(os.path.join(static_dir, "index.html"), "text/html")

        elif path == "/api/status":
            with state_lock:
                self._send_json(dict(state))

        elif path == "/api/results":
            with state_lock:
                output_dir = state.get("output_dir")
            if not output_dir:
                self._send_json({"error": "no results"}, 404)
                return
            scored_path = os.path.join(output_dir, "scored.json")
            try:
                with open(scored_path) as f:
                    self._send_json(json.load(f))
            except FileNotFoundError:
                self._send_json({"error": "not ready"}, 404)

        elif path.startswith("/api/frame/"):
            # /api/frame/{video_id}/{filename}
            parts = path.split("/api/frame/", 1)[1].split("/", 1)
            if len(parts) != 2:
                self.send_error(404)
                return
            video_id, filename = parts
            # Sanitize: only allow alphanumeric, dash, underscore, dot in filename
            if not re.match(r'^[\w.-]+$', filename) or '..' in filename:
                self.send_error(403)
                return
            frame_dir = os.path.join(DEBATES_DIR, video_id, "frames")
            filepath = os.path.realpath(os.path.join(frame_dir, filename))
            if not filepath.startswith(os.path.realpath(frame_dir)):
                self.send_error(403)
                return
            self._serve_file(filepath, "image/jpeg")

        elif path.startswith("/output/"):
            with state_lock:
                output_dir = state.get("output_dir")
            if not output_dir:
                self.send_error(404)
                return
            filename = path.split("/output/", 1)[1]
            filepath = os.path.realpath(os.path.join(output_dir, filename))
            if not filepath.startswith(os.path.realpath(output_dir)):
                self.send_error(403)
                return
            ct = "text/html" if filename.endswith(".html") else "application/json"
            self._serve_file(filepath, ct)

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/check-url":
            data = json.loads(self._read_body())
            result = _check_url(data.get("url", ""))
            self._send_json(result)

        elif path == "/api/start":
            with state_lock:
                if state["pipeline_running"]:
                    self._send_json({"ok": False, "error": "Pipeline already running, please wait"}, 409)
                    return
            data = json.loads(self._read_body())
            _start_ingest(data)
            self._send_json({"ok": True})

        elif path == "/api/resume":
            with state_lock:
                if state["pipeline_running"]:
                    self._send_json({"ok": False, "error": "Pipeline already running, please wait"}, 409)
                    return
            data = json.loads(self._read_body())
            _resume_from(data)
            self._send_json({"ok": True})

        elif path == "/api/speakers":
            data = json.loads(self._read_body())
            _apply_speakers(data)
            self._send_json({"ok": True})

        elif path == "/api/reset":
            with state_lock:
                state.update({
                    "screen": "new", "phase": None, "message": "",
                    "progress": 0, "total": 0, "speakers": [],
                    "results": None, "output_dir": None, "url": None,
                    "topic": None, "error": None, "existing": None,
                    "video_id": None, "pipeline_steps": [],
                    "pipeline_running": False,
                })
            self._send_json({"ok": True})

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── URL Check: Detect Existing Data ──────────────────────────────
def _check_url(url: str) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        return {"video_id": None, "exists": False}

    output_dir = os.path.join(DEBATES_DIR, video_id)
    existing = {
        "turns": os.path.exists(os.path.join(output_dir, "turns.json")),
        "structured": os.path.exists(os.path.join(output_dir, "structured_turns.json")),
        "scored": os.path.exists(os.path.join(output_dir, "scored.json")),
        "report": os.path.exists(os.path.join(output_dir, "report.html")),
    }

    result = {
        "video_id": video_id,
        "exists": any(existing.values()),
        "existing": existing,
        "output_dir": output_dir,
    }

    # Load topic and speakers from existing turns.json
    if existing["turns"]:
        try:
            with open(os.path.join(output_dir, "turns.json")) as f:
                data = json.load(f)
            result["topic"] = data.get("topic", "")
            result["speakers"] = data.get("speakers", [])
            result["debaters"] = data.get("debaters", [])
            result["title"] = data.get("title", "")
        except (json.JSONDecodeError, IOError):
            pass

    # Load scores if available
    if existing["scored"]:
        try:
            with open(os.path.join(output_dir, "scored.json")) as f:
                data = json.load(f)
            result["scores"] = {
                name: {"overall_score": s["overall_score"]}
                for name, s in data.get("stats", {}).items()
            }
        except (json.JSONDecodeError, IOError):
            pass

    return result


# ── Resume From Existing Data ────────────────────────────────────
def _resume_from(data: dict):
    """Resume pipeline from a specific phase using existing data."""
    url = data.get("url", "").strip()
    topic = data.get("topic", "").strip()
    mode = data.get("mode", "results")  # results | extract | full

    video_id = extract_video_id(url)
    if not video_id:
        with state_lock:
            state["error"] = "Invalid YouTube URL"
        return

    output_dir = os.path.join(DEBATES_DIR, video_id)

    if mode == "results":
        # Jump straight to results
        scored_path = os.path.join(output_dir, "scored.json")
        try:
            with open(scored_path) as f:
                scored_data = json.load(f)
            results = {
                "speakers": scored_data.get("stats", {}),
                "url": scored_data.get("youtube_url", url),
                "title": scored_data.get("title", ""),
            }
            with state_lock:
                state.update({
                    "screen": "results", "phase": "done",
                    "message": "Loaded existing results",
                    "results": results, "output_dir": output_dir,
                    "url": url, "topic": topic, "video_id": video_id,
                })
        except FileNotFoundError:
            with state_lock:
                state["error"] = "scored.json not found"

    elif mode == "extract":
        # Re-run from extraction
        resume_steps = [
            {"id": "extract", "label": "Structure extraction (LLM)", "status": "pending", "detail": ""},
            {"id": "score",   "label": "Scoring",                    "status": "pending", "detail": ""},
            {"id": "render",  "label": "Generating report",          "status": "pending", "detail": ""},
        ]
        with state_lock:
            state.update({
                "screen": "processing", "phase": "extract",
                "message": "Starting structure extraction...",
                "progress": 0, "total": 0,
                "output_dir": output_dir, "url": url, "topic": topic,
                "video_id": video_id, "error": None,
                "pipeline_steps": resume_steps,
                "pipeline_running": True,
            })
        # Read debaters from turns.json
        turns_path = os.path.join(output_dir, "turns.json")
        with open(turns_path) as f:
            raw = json.load(f)
        debaters = raw.get("debaters", raw.get("speakers", []))
        thread = threading.Thread(
            target=_run_pipeline, args=(output_dir, topic, debaters, False),
            daemon=True,
        )
        thread.start()

    elif mode == "full":
        # Start from scratch (ingest)
        _start_ingest({"url": url, "topic": topic, "adapter": data.get("adapter", "assemblyai")})


# ── Phase 1: Ingest ──────────────────────────────────────────────
def _start_ingest(data):
    url = data.get("url", "").strip()
    topic = data.get("topic", "").strip()
    adapter = data.get("adapter", "assemblyai")

    if not url or not topic:
        with state_lock:
            state["error"] = "URL and topic are required"
        return

    video_id = extract_video_id(url)
    if not video_id:
        with state_lock:
            state["error"] = "Could not extract video ID from URL"
        return

    output_dir = os.path.join(DEBATES_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # If turns.json already exists, skip Phase 1 entirely
    turns_path = os.path.join(output_dir, "turns.json")
    if os.path.exists(turns_path):
        try:
            with open(turns_path) as f:
                raw = json.load(f)
            speaker_labels = raw.get("speakers", [])
            has_named = any(not s.startswith("SPEAKER_") for s in speaker_labels)

            if has_named:
                # Speakers already named — skip to extraction
                debaters = raw.get("debaters", speaker_labels)
                resume_steps = [
                    {"id": "extract", "label": "Structure extraction (LLM)", "status": "pending", "detail": ""},
                    {"id": "score",   "label": "Scoring",                    "status": "pending", "detail": ""},
                    {"id": "render",  "label": "Generating report",          "status": "pending", "detail": ""},
                ]
                with state_lock:
                    state.update({
                        "screen": "processing", "phase": "extract",
                        "message": "Found existing turns.json, skipping to extraction...",
                        "progress": 0, "total": 0,
                        "url": url, "topic": topic, "output_dir": output_dir,
                        "video_id": video_id,
                        "error": None, "results": None, "speakers": [],
                        "existing": None,
                        "pipeline_steps": resume_steps,
                        "pipeline_running": True,
                    })
                thread = threading.Thread(
                    target=_run_pipeline, args=(output_dir, topic, debaters, False),
                    daemon=True,
                )
                thread.start()
                return
            else:
                # Speakers not named yet — show speaker ID screen
                from models import Turn
                turns = [Turn.from_dict(t) for t in raw.get("turns", [])]
                audio_dest = os.path.join(output_dir, "audio.mp3")
                video_path = os.path.join(output_dir, "video.mp4")

                voiceprint_matches = _try_voiceprint_match(speaker_labels, turns, audio_dest)

                speaker_frames = {}
                if os.path.exists(video_path):
                    speaker_frames = _extract_speaker_frames(video_path, turns, speaker_labels, output_dir)

                speakers_info = []
                for label in speaker_labels:
                    st = [t for t in turns if t.speaker == label]
                    sample_turns = st[:3]
                    frames = speaker_frames.get(label, [])
                    samples = []
                    for idx, t in enumerate(sample_turns):
                        frame_file = frames[idx] if idx < len(frames) else None
                        frame_url = f"/api/frame/{video_id}/{frame_file}" if frame_file else None
                        samples.append({
                            "text": t.text[:250],
                            "start_ms": t.start_ms,
                            "end_ms": t.end_ms,
                            "frame": frame_url,
                        })
                    speakers_info.append({
                        "label": label,
                        "turn_count": len(st),
                        "samples": samples,
                        "suggested_name": voiceprint_matches.get(label, ""),
                    })

                with state_lock:
                    state.update({
                        "screen": "speakers", "phase": "speakers",
                        "message": "Identify the speakers",
                        "url": url, "topic": topic, "output_dir": output_dir,
                        "video_id": video_id,
                        "error": None, "results": None,
                        "speakers": speakers_info,
                        "pipeline_steps": [],
                        "pipeline_running": False,
                    })
                return
        except (json.JSONDecodeError, IOError, KeyError):
            pass  # Fall through to full ingest

    is_remote = (adapter == "remote-whisperx")
    steps = _make_pipeline_steps() if is_remote else _make_simple_pipeline_steps()

    with state_lock:
        state.update({
            "screen": "ingest", "phase": "ingest",
            "message": "Starting download...",
            "progress": 0, "total": 0,
            "url": url, "topic": topic, "output_dir": output_dir,
            "video_id": video_id,
            "error": None, "results": None, "speakers": [],
            "existing": None,
            "pipeline_steps": steps,
            "pipeline_running": True,
        })

    thread = threading.Thread(
        target=_run_ingest, args=(url, topic, adapter, output_dir),
        daemon=True,
    )
    thread.start()


def _extract_speaker_frames(video_path, turns, speaker_labels, output_dir):
    """Extract a video frame for each speaker's sample turns. Returns {label: [filenames]}."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(video_path):
        return {}

    # Check ffmpeg availability
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ffmpeg not available, skipping frame extraction")
        return {}

    result = {}
    for label in speaker_labels:
        speaker_turns = [t for t in turns if t.speaker == label][:3]
        filenames = []
        for idx, turn in enumerate(speaker_turns):
            # 2 seconds into the turn, clamped to turn duration
            ts_ms = turn.start_ms + min(2000, (turn.end_ms - turn.start_ms) // 2)
            ts_secs = ts_ms / 1000.0
            fname = f"frame_{label}_{idx}.jpg"
            frame_path = os.path.join(frames_dir, fname)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{ts_secs:.3f}", "-i", video_path,
                     "-frames:v", "1", "-q:v", "5", frame_path],
                    capture_output=True, timeout=15,
                )
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    filenames.append(fname)
                else:
                    filenames.append(None)
            except (subprocess.TimeoutExpired, OSError):
                filenames.append(None)
        result[label] = filenames
    return result


def _run_ingest(url, topic, adapter_name, output_dir):
    try:
        from dotenv import load_dotenv
        load_dotenv()

        import shutil
        import subprocess
        import tempfile
        from datetime import datetime
        from models import DebateResult
        from phase1_ingest import get_video_metadata
        from pipeline.turn_builder import build_turns

        # Step: Download
        _update_step("download", "running", "Fetching video info...")
        with state_lock:
            state["message"] = "Fetching video info..."

        metadata = get_video_metadata(url)
        title = metadata.get("title", url) if metadata else url

        _update_step("download", "running", f"Downloading: {title[:60]}")
        with state_lock:
            state["message"] = f"Downloading: {title[:60]}"

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "audio.mp3")
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", tmp_path, url],
            check=True, capture_output=True,
        )

        # Download low-res video for speaker frame extraction
        video_path = os.path.join(output_dir, "video.mp4")
        has_video = False
        try:
            _update_step("download", "running", "Downloading video (low-res)...")
            subprocess.run(
                ["yt-dlp", "-f", "worst[ext=mp4]", "-o", video_path, url],
                check=True, capture_output=True, timeout=300,
            )
            has_video = os.path.exists(video_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            print(f"Video download failed (frames unavailable): {e}")

        _update_step("download", "complete", "Downloaded")

        # Step: Transcribe (with remote-whisperx progress callbacks)
        if adapter_name == "remote-whisperx":
            def _remote_progress(stage, message):
                stage_to_step = {
                    "uploading": "upload",
                    "downloading_model": "transcribe",
                    "transcribing": "transcribe",
                    "aligning": "align",
                    "diarizing": "diarize",
                    "complete": "diarize",
                    "downloading_result": "diarize",
                }
                step_id = stage_to_step.get(stage)
                if not step_id:
                    return
                if stage == "uploading":
                    _update_step("upload", "running", message)
                elif stage == "downloading_model":
                    _update_step("upload", "complete")
                    _update_step("transcribe", "running", message)
                elif stage == "transcribing":
                    _update_step("transcribe", "running", message)
                elif stage == "aligning":
                    _update_step("transcribe", "complete")
                    _update_step("align", "running", message)
                elif stage == "diarizing":
                    _update_step("align", "complete")
                    _update_step("diarize", "running", message)
                elif stage == "complete":
                    _update_step("diarize", "complete", "Transcription complete")
                elif stage == "downloading_result":
                    _update_step("diarize", "running", message)
                with state_lock:
                    state["message"] = message

            _update_step("upload", "running", "Uploading audio to GPU server...")
            with state_lock:
                state["message"] = "Uploading audio to GPU server..."

            from adapters.remote_whisperx_adapter import RemoteWhisperXAdapter
            adapter = RemoteWhisperXAdapter(progress_callback=_remote_progress)
        elif adapter_name == "whisperx":
            _update_step("transcribe", "running", "Transcribing with WhisperX...")
            with state_lock:
                state["message"] = "Transcribing with WhisperX..."
            from adapters.whisperx_adapter import WhisperXAdapter
            adapter = WhisperXAdapter()
        elif adapter_name == "pyannote-api":
            _update_step("transcribe", "running", "Transcribing with pyannoteAI...")
            with state_lock:
                state["message"] = "Transcribing with pyannoteAI..."
            from adapters.pyannote_api_adapter import PyannoteAPIAdapter
            adapter = PyannoteAPIAdapter()
        else:
            _update_step("transcribe", "running", "Transcribing with AssemblyAI...")
            with state_lock:
                state["message"] = "Transcribing with AssemblyAI..."
            from adapters.assemblyai_adapter import AssemblyAIAdapter
            adapter = AssemblyAIAdapter()

        segments = adapter.transcribe(tmp_path)
        _update_step("transcribe", "complete")
        if adapter_name != "remote-whisperx":
            # For non-remote, mark transcribe complete
            pass

        turns = build_turns(segments)

        # Keep audio for voiceprint matching
        audio_dest = os.path.join(output_dir, "audio.mp3")
        try:
            shutil.copy2(tmp_path, audio_dest)
        except OSError:
            pass

        # Cleanup temp
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

        # Get speaker labels
        speaker_labels = []
        for t in turns:
            if t.speaker not in speaker_labels:
                speaker_labels.append(t.speaker)

        # Try voiceprint matching
        _update_step("speakers", "running", "Matching voiceprints...")
        voiceprint_matches = _try_voiceprint_match(speaker_labels, turns, audio_dest)

        # Extract frames from video for speaker identification
        speaker_frames = {}
        if has_video:
            _update_step("speakers", "running", "Extracting speaker frames...")
            speaker_frames = _extract_speaker_frames(video_path, turns, speaker_labels, output_dir)

        # Build speaker info for UI
        with state_lock:
            vid = state.get("video_id", "")

        speakers_info = []
        for label in speaker_labels:
            st = [t for t in turns if t.speaker == label]
            sample_turns = st[:3]
            frames = speaker_frames.get(label, [])
            samples = []
            for idx, t in enumerate(sample_turns):
                frame_file = frames[idx] if idx < len(frames) else None
                frame_url = f"/api/frame/{vid}/{frame_file}" if frame_file else None
                samples.append({
                    "text": t.text[:250],
                    "start_ms": t.start_ms,
                    "end_ms": t.end_ms,
                    "frame": frame_url,
                })
            speakers_info.append({
                "label": label,
                "turn_count": len(st),
                "samples": samples,
                "suggested_name": voiceprint_matches.get(label, ""),
            })

        # Save turns.json
        duration_ms = turns[-1].end_ms if turns else 0
        result = DebateResult(
            title=title,
            youtube_url=url,
            topic=topic,
            duration_ms=duration_ms,
            speakers=speaker_labels,
            debaters=list(speaker_labels),
            turns=turns,
            stats={},
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        turns_path = os.path.join(output_dir, "turns.json")
        with open(turns_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        _update_step("speakers", "running", "Waiting for speaker identification...")
        with state_lock:
            state.update({
                "screen": "speakers", "phase": "speakers",
                "message": "Identify the speakers",
                "speakers": speakers_info,
                "pipeline_running": False,
            })

    except ConnectionError as e:
        import traceback
        traceback.print_exc()
        with state_lock:
            state["screen"] = "new"
            state["error"] = f"SSH connection failed: {e}"
            state["message"] = ""
            state["pipeline_running"] = False
    except Exception as e:
        import traceback
        traceback.print_exc()
        with state_lock:
            state["screen"] = "new"
            state["error"] = str(e)
            state["message"] = ""
            state["pipeline_running"] = False


def _try_voiceprint_match(
    speaker_labels: list[str],
    turns,
    audio_path: str,
) -> dict[str, str]:
    """Try to match speakers against saved voiceprints. Returns {label: name}."""
    if not os.path.isdir(VOICEPRINTS_DIR) or not os.path.exists(audio_path):
        return {}

    npy_files = [f for f in os.listdir(VOICEPRINTS_DIR) if f.endswith(".npy")]
    if not npy_files:
        return {}

    try:
        import numpy as np
        from match_voiceprints import load_voiceprints, compute_speaker_embedding
    except ImportError:
        print("Voiceprint matching unavailable (missing resemblyzer/librosa)")
        return {}

    print("Attempting voiceprint matching...")
    voiceprints = load_voiceprints(VOICEPRINTS_DIR)
    if not voiceprints:
        return {}

    matches = {}
    used_names = set()

    for label in speaker_labels:
        segs = [
            {"start_ms": t.start_ms, "end_ms": t.end_ms}
            for t in turns if t.speaker == label
        ]
        embedding = compute_speaker_embedding(audio_path, segs)
        if embedding is None:
            continue

        best_name = None
        best_sim = -1.0
        for name, vp_emb in voiceprints.items():
            if name in used_names:
                continue
            sim = float(np.dot(embedding, vp_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name and best_sim >= 0.85:
            matches[label] = best_name
            used_names.add(best_name)
            print(f"  {label} -> {best_name} (similarity={best_sim:.3f})")
        else:
            print(f"  {label} -> no match (best: {best_name} at {best_sim:.3f})")

    return matches


# ── Speaker Assignment → Start Processing ────────────────────────
def _apply_speakers(data):
    mapping = data.get("mapping", {})
    debaters = data.get("debaters", [])

    with state_lock:
        output_dir = state["output_dir"]
        topic = state["topic"]

    turns_path = os.path.join(output_dir, "turns.json")
    with open(turns_path) as f:
        raw = json.load(f)

    for turn in raw["turns"]:
        old = turn["speaker"]
        if old in mapping:
            turn["speaker"] = mapping[old]

    # Deduplicate speakers list (handles merging: two labels → same name)
    seen = set()
    unique_speakers = []
    for s in raw["speakers"]:
        name = mapping.get(s, s)
        if name not in seen:
            seen.add(name)
            unique_speakers.append(name)
    raw["speakers"] = unique_speakers

    # Deduplicate debaters list
    if debaters:
        seen_d = set()
        unique_debaters = []
        for d in debaters:
            if d not in seen_d:
                seen_d.add(d)
                unique_debaters.append(d)
        raw["debaters"] = unique_debaters
    else:
        raw["debaters"] = list(raw["speakers"])

    with open(turns_path, "w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    # Clear stale downstream files since speakers changed
    for fname in ["structured_turns.json", ".structure_cache.json", "scored.json",
                   "report.html", "overlay.html"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    _update_step("speakers", "complete", "Speakers identified")

    with state_lock:
        state.update({
            "screen": "processing", "phase": "extract",
            "message": "Starting structure extraction...",
            "progress": 0, "total": 0,
            "pipeline_running": True,
        })

    thread = threading.Thread(
        target=_run_pipeline, args=(output_dir, topic, raw["debaters"], False),
        daemon=True,
    )
    thread.start()


# ── Pipeline: Extract → Score → Render ───────────────────────────
def _run_pipeline(output_dir, topic, debaters, skip_extract=False):
    turns_path = os.path.join(output_dir, "turns.json")
    structured_path = os.path.join(output_dir, "structured_turns.json")
    scored_path = os.path.join(output_dir, "scored.json")
    report_path = os.path.join(output_dir, "report.html")
    overlay_path = os.path.join(output_dir, "overlay.html")
    cache_path = os.path.join(output_dir, ".structure_cache.json")
    hash_path = os.path.join(output_dir, ".turns_hash")

    try:
        with open(turns_path) as f:
            raw = json.load(f)
        total_turns = len(raw["turns"])

        # Check if extraction can be skipped (cache fully covers turns)
        if not skip_extract and os.path.exists(structured_path) and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                if len(cache) >= total_turns:
                    # Also check turns.json hash
                    current_hash = file_hash(turns_path)
                    stored_ok = False
                    if os.path.exists(hash_path):
                        with open(hash_path) as f:
                            stored_ok = f.read().strip() == current_hash
                    if stored_ok:
                        with state_lock:
                            state["message"] = f"Structure cache has {len(cache)}/{total_turns} turns, skipping extraction"
                        print(f"Structure cache has {len(cache)}/{total_turns} turns, skipping extraction")
                        skip_extract = True
            except (json.JSONDecodeError, IOError):
                pass

        # Phase 1.5: Extract
        if not skip_extract:
            _update_step("extract", "running", f"0/{total_turns} turns")
            with state_lock:
                state.update({
                    "total": total_turns,
                    "message": f"Extracting structure (0/{total_turns})...",
                })

            stop_monitor = threading.Event()
            monitor = threading.Thread(
                target=_monitor_progress, args=(cache_path, total_turns, stop_monitor),
                daemon=True,
            )
            monitor.start()

            from phase1_5_extract import run as run_extract
            extract_args = SimpleNamespace(
                input=turns_path,
                output=structured_path,
                topic=topic,
                debaters=debaters,
                no_cache=False,
            )
            run_extract(extract_args)
            stop_monitor.set()
            _update_step("extract", "complete", f"{total_turns}/{total_turns} turns")

            # Save hash
            with open(hash_path, "w") as f:
                f.write(file_hash(turns_path))
        else:
            _update_step("extract", "complete", "Cached")

        # Phase 2: Score
        _update_step("score", "running")
        with state_lock:
            state.update({"phase": "score", "message": "Scoring debate...", "progress": 0})

        from phase2_score import run as run_score
        score_args = SimpleNamespace(
            input=structured_path,
            output=scored_path,
            threshold_engagement=None,
            threshold_dodge=None,
            threshold_drift=None,
        )
        run_score(score_args)
        _update_step("score", "complete")

        # Phase 3: Render
        _update_step("render", "running")
        with state_lock:
            state.update({"phase": "render", "message": "Generating report..."})

        from phase3_render import run as run_render
        render_args = SimpleNamespace(
            input=scored_path,
            report=report_path,
            overlay=overlay_path,
        )
        run_render(render_args)
        _update_step("render", "complete")

        # Build results
        with open(scored_path) as f:
            scored_data = json.load(f)

        results = {
            "speakers": scored_data.get("stats", {}),
            "url": scored_data.get("youtube_url", ""),
            "title": scored_data.get("title", ""),
        }

        with state_lock:
            state.update({
                "screen": "results", "phase": "done",
                "message": "Analysis complete!",
                "results": results,
                "pipeline_running": False,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        with state_lock:
            state["error"] = str(e)
            state["message"] = f"Error: {e}"
            state["pipeline_running"] = False


def _monitor_progress(cache_path, total_turns, stop_event):
    while not stop_event.is_set():
        try:
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cache = json.load(f)
                count = len(cache)
                with state_lock:
                    state["progress"] = count
                    state["message"] = f"Extracting structure ({count}/{total_turns})..."
                _update_step("extract", "running", f"{count}/{total_turns} turns")
        except (json.JSONDecodeError, IOError):
            pass
        stop_event.wait(2)


# ── Main ─────────────────────────────────────────────────────────
def main():
    os.makedirs(DEBATES_DIR, exist_ok=True)
    server = ThreadingHTTPServer(("", PORT), Handler)
    print(f"Debate Stats UI running at http://localhost:{PORT}")
    print(f"Debates stored in: {DEBATES_DIR}")
    print("Press Ctrl+C to stop\n")
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
