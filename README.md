# DebateStats

Score the process of argument, not the content.

## What it does

DebateStats takes a YouTube debate URL, transcribes and diarizes the audio, then scores each speaker on four argumentation mechanics: engagement with the opponent, question dodging, claim support, and topic focus. It produces a visual HTML report and a 1920x1080 broadcast overlay suitable for OBS. Every metric is computed mathematically from semantic similarity and string pattern matching. No LLM judges the content or opinions expressed. All math is open and verifiable.

## What it does NOT do

- Does not say who won on substance
- Does not fact-check claims
- Does not use an LLM to judge content
- Does not work in real-time (post-hoc analysis only)

## Prerequisites

- Python 3.10+
- ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux)
- AssemblyAI API key — free tier at https://www.assemblyai.com (covers ~5 hours/month)

## Install

```bash
git clone https://github.com/meir-san/Thebate
cd Thebate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ASSEMBLYAI_API_KEY
```

## Quick start

```bash
python main.py \
  --url "https://www.youtube.com/watch?v=6RQA9GZprqM" \
  --topic "Race, genetics, and American politics" \
  --speakers "JonTron,Destiny" \
  --output-dir ./output/
```

Then open `output/report.html` in any browser.

The JonTron vs Destiny debate (2017) is a well-known test case with clear stylistic differences between the two speakers.

## Phase-by-phase usage

For users who want to re-run scoring without re-fetching audio (saves API credits):

```bash
# Run ingestion once
python phase1_ingest.py --url "..." --topic "..." --speakers "A,B" --output turns.json

# Iterate on scoring without re-fetching
python phase2_score.py --input turns.json --output scored.json

# Re-render without re-scoring
python phase3_render.py --input scored.json
```

Or use the wrapper with skip flags:

```bash
# Re-score and re-render from existing turns
python main.py --url "..." --topic "..." --skip-ingest --output-dir ./output/

# Re-render only from existing scores
python main.py --url "..." --topic "..." --skip-ingest --skip-score --output-dir ./output/
```

## Metrics explained

**Engagement Score** — Did the speaker actually address what the opponent just said? Computed as semantic similarity between the response and the previous turn. Score near 0 means the response was semantically unrelated to what came before it.

**Question Dodge Rate** — When the opponent asked a direct question, did this speaker answer it? Computed by comparing the question's meaning to the response. Low similarity means the response talked about something else entirely.

**Claim Support Ratio** — What fraction of this speaker's assertions came with reasoning? A claim is a declarative sentence. A supported claim contains a reasoning connector ("because", "therefore", "evidence shows", "for example"). High ratio means the speaker tends to explain their reasoning. Low ratio means the speaker tends to assert without justifying.

**Topic Drift Index** — Did this speaker stay on the agreed debate topic, or steer the conversation elsewhere? Each turn is compared semantically to the topic string. High drift means the speaker consistently pulled the conversation away from the subject.

## Overall score

```
Score = (avg_engagement × 30)
      + ((1 − dodge_rate) × 25)
      + (claim_support_ratio × 25)
      + ((1 − avg_topic_drift) × 20)
```

Maximum: 100. This score measures argumentation process only. A speaker with strong opinions but poor mechanics scores low. A speaker with weak opinions but rigorous mechanics scores high.

## OBS overlay setup

1. Open OBS Studio
2. Add a new "Browser Source"
3. Check "Local file" and browse to `overlay.html`
4. Set Width: 1920, Height: 1080
5. Layer it above your debate video capture source
6. The overlay background is transparent — only the panels and bottom bar are visible

## Optional: Fully local processing (WhisperX)

For users who want no cloud API:

1. Install WhisperX: `pip install whisperx`
2. Get a HuggingFace token and accept the pyannote model licenses
3. Set `HF_TOKEN` in `.env`
4. Use `--adapter whisperx`

Note: significantly slower than AssemblyAI and requires a local GPU for reasonable speed.
