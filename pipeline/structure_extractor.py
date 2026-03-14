"""Phase 1.5 — LLM-based structure extraction using Qwen 2.5:14b via Ollama.

Expects turns to have been preprocessed (dialogue_act, clean_text, score_this set).
Skips turns where score_this=False, setting speech_act from dialogue_act directly.
"""

import hashlib
import json
import sys
import time

import requests

import config
from models import Turn

VALID_SPEECH_ACTS = {
    "claim", "rebuttal", "challenge", "correction", "dismissal",
    "insult", "agreement", "explanation", "concession",
}

PROMPT_TEMPLATE = """\
You are extracting argument structure from a debate transcript.

DEBATE TOPIC: {topic}
SPEAKER: {speaker}
{context_block}
CURRENT TURN TO ANALYZE: "{text}"

Classify this turn and extract its core proposition. Return JSON only.

SPEECH ACT DEFINITIONS (choose the one that BEST fits):
- "explanation": The speaker explains HOW or WHY something works, provides a mechanism, or teaches a concept. Example: "Gravity works by curving spacetime around massive objects."
- "correction": The speaker states that the opponent is WRONG and provides the correct information. Example: "Stars don't burn. They undergo nuclear fusion."
- "rebuttal": The speaker argues AGAINST the opponent's point without providing new explanatory information. Example: "That doesn't prove anything because it works on both models."
- "challenge": The speaker asks a question that tests or attacks the opponent's position. Example: "How does a distant sun orbit a tiny earth?"
- "claim": The speaker makes a standalone assertion not directly responding to the opponent. Example: "The earth is flat."
- "dismissal": The speaker rejects the opponent's point without any reasoning or counter-evidence. Example: "That doesn't make sense." or "Nobody believes that."
- "insult": The speaker attacks the opponent personally rather than their argument. Example: "You're an idiot." or "Stop being a prick."
- "agreement": The speaker agrees with something the opponent said. Example: "Yes, that's correct."
- "concession": The speaker partially yields a point. Example: "Okay, I'll grant you that, but..."

KEY DISTINCTION: If a speaker says WHY something is true or HOW something works (provides mechanism, evidence, or explanation), classify as "explanation" even if it also rebuts the opponent. Only use "rebuttal" when the speaker argues against the opponent WITHOUT explaining an alternative.

PROPOSITION RULES:
- Extract the ACTUAL CLAIM or POINT being made, stated as a simple declarative sentence
- DO NOT describe what the speaker is doing (wrong: "The speaker challenges the opponent's credibility")
- DO state what the speaker is claiming (right: "Ancient Greek scientists did not know about heliocentrism")
- If the turn is a question with no embedded claim, set proposition to null
- Strip all filler words, false starts, and disfluencies from the proposition

Return ONLY this JSON:
{{"speech_act": "...", "proposition": "..." or null, "responds_to_opponent": true/false}}"""


def _cache_key(turn: Turn) -> str:
    """Deterministic cache key from turn index + clean_text hash."""
    text = turn.clean_text or turn.text
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"{turn.index}_{text_hash}"


def _build_context(turns: list[Turn], current_idx: int, debaters: list[str] | None) -> str:
    """Build context block from up to 3 preceding scorable turns."""
    context_lines = []
    count = 0
    for j in range(current_idx - 1, -1, -1):
        t = turns[j]
        if debaters and t.speaker not in debaters:
            continue
        if not t.score_this:
            continue
        text = (t.clean_text or t.text)[:300]
        context_lines.insert(0, f'  {t.speaker}: "{text}"')
        count += 1
        if count >= 3:
            break
    if context_lines:
        return "PRECEDING TURNS:\n" + "\n".join(context_lines)
    return ""


def _call_ollama(prompt: str, timeout: float = 60.0) -> dict | None:
    """Call Ollama API and parse JSON response. Returns None on failure."""
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 256},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return json.loads(raw)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"    [ollama error] {e}", file=sys.stderr, flush=True)
        return None


def _validate_and_apply(turn: Turn, data: dict) -> bool:
    """Validate LLM output and apply to turn fields. Returns True if valid."""
    if not isinstance(data, dict):
        return False

    speech_act = data.get("speech_act", "").lower().strip()
    if speech_act not in VALID_SPEECH_ACTS:
        return False

    turn.speech_act = speech_act
    turn.proposition = data.get("proposition") if data.get("proposition") else None
    turn.responds_to_opponent = bool(data.get("responds_to_opponent", False))
    return True


# Map preprocessor dialogue_acts to speech_acts for skipped turns
_DIALOGUE_ACT_TO_SPEECH_ACT = {
    "backchannel": "backchannel",
    "fragment": "fragment",
}


def extract_structure(
    turns: list[Turn],
    topic: str,
    debaters: list[str] | None = None,
    cache_path: str | None = None,
) -> None:
    """Run LLM structure extraction on scorable turns, mutating them in place.

    Turns with score_this=False are skipped and get speech_act set from
    their dialogue_act. Only scorable turns are sent to the LLM.
    """
    # Load cache
    cache: dict[str, dict] = {}
    if cache_path:
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"  Loaded {len(cache)} cached extractions from {cache_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    total = len(turns)
    skipped = 0
    extracted = 0
    cached_hits = 0
    failures = 0
    start_time = time.time()

    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        # Skip non-scorable turns — set speech_act directly
        if not turn.score_this:
            turn.speech_act = _DIALOGUE_ACT_TO_SPEECH_ACT.get(
                turn.dialogue_act, "fragment"
            )
            turn.proposition = None
            turn.responds_to_opponent = False
            skipped += 1
            continue

        key = _cache_key(turn)

        # Check cache
        if key in cache:
            if _validate_and_apply(turn, cache[key]):
                cached_hits += 1
                extracted += 1
                continue

        # Build prompt using clean_text
        context = _build_context(turns, i, debaters)
        text = (turn.clean_text or turn.text)[:1500]
        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            speaker=turn.speaker,
            context_block=context,
            text=text,
        )

        # Call LLM
        data = _call_ollama(prompt)
        if data and _validate_and_apply(turn, data):
            cache[key] = data
            extracted += 1
        else:
            failures += 1
            # Set safe defaults on failure
            turn.speech_act = "claim"
            turn.responds_to_opponent = False

        # Progress logging + periodic cache save
        if (i + 1) % 20 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (extracted + failures) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{total}] extracted={extracted} cached={cached_hits} "
                  f"skipped={skipped} failures={failures} ({rate:.1f} turns/s)", flush=True)
            if cache_path and cache:
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)

    # Final cache save
    if cache_path and cache:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved {len(cache)} extractions to {cache_path}")

    elapsed = time.time() - start_time
    print(f"  Structure extraction complete: {extracted} extracted, "
          f"{cached_hits} from cache, {skipped} skipped, "
          f"{failures} failures in {elapsed:.1f}s")
