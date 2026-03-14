import re

import requests
from models import Turn
import config

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|the reason is|the reason being|"
    r"based on|for example|for instance|according to|as shown by|"
    r"as evidenced by|which shows|which demonstrates|"
    r"research shows|studies show|evidence shows|the data shows)\b",
    re.IGNORECASE,
)

# Comprehensive filler word/phrase list for spoken language cleanup
_FILLER_WORDS = re.compile(
    r"\b(?:uh|um|uh+m*|like|you know|I mean|sort of|kind of|right|okay|ok|"
    r"so|well|basically|actually|literally|really|just|very|pretty much|"
    r"honestly|frankly|obviously|clearly|of course|anyway|anyways|"
    r"look|hey|yeah|yes|no|ah|oh)\b[,.]?\s*",
    re.IGNORECASE,
)

# Leading conjunctions to strip from claim/reason after split
_LEADING_CONJUNCTIONS = re.compile(
    r"^(?:and|but|or|so|yet|then)[,]?\s+",
    re.IGNORECASE,
)

# False starts: repeated word sequences ("they can't they can't")
_FALSE_STARTS = re.compile(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", re.IGNORECASE)

# Trailing fragments without punctuation at end of text
_TRAILING_FRAGMENT = re.compile(r'[.!?]\s+[^.!?]*$')

_REPEATED_WORDS = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

# Stopwords for content word counting
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "that", "this",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "nor", "if", "then", "than", "too", "very", "just",
    "but", "and", "or", "so", "as", "at", "by", "for", "in", "of", "on",
    "to", "up", "out", "off", "with", "from", "into", "about", "all",
    "also", "more", "some", "any", "each", "only", "own", "same", "get",
    "got", "say", "said", "says", "go", "goes", "went", "come", "came",
    "make", "made", "take", "took", "see", "saw", "know", "knew",
    "think", "thought", "here", "there", "now", "then",
}

_PROMPT_TEMPLATE = """Evaluate the logical quality of this argument's reasoning.

Full sentence: "{full_sentence}"
Claim: "{claim}"
Stated reason: "{reason}"

Score 1-3 based on whether the reason provides a MECHANISM or EVIDENCE that connects to the claim:
1 = The reason is just a restatement, a bare assertion, an appeal to authority without evidence, or has no explanatory mechanism connecting it to the claim (e.g., "X is true because the Bible says so", "X because that's how it is", "X because I said so", "X because everyone knows")
2 = The reason is related to the claim and provides partial explanation, but the logical chain has gaps or missing steps
3 = The reason provides a clear mechanism, evidence, or chain of reasoning that directly explains WHY the claim would be true (e.g., "the earth is round because ships disappear bottom-first over the horizon, which can only happen on a curved surface")

The key question: does the reason explain HOW or WHY, or does it just assert WHAT?

Reply with ONLY the number 1, 2, or 3."""

_DIGIT_RE = re.compile(r"[123]")

MIN_SENTENCE_WORDS = 8  # total words after cleaning (lowered since we strip fillers)
MIN_CLAIM_CONTENT_WORDS = 2   # content words in claim (can be short: "the earth is round")
MIN_REASON_CONTENT_WORDS = 3  # content words in reason (needs substance)
SAMPLE_LOG_COUNT = 5


def _count_content_words(text: str) -> int:
    """Count substantive content words (non-stopword, >3 chars)."""
    return sum(
        1 for w in text.lower().split()
        if len(w) > 3 and w not in _STOPWORDS
    )


def _deep_clean_text(text: str) -> str:
    """Aggressively clean spoken language artifacts from text."""
    cleaned = text.strip()
    # Remove all filler words/phrases (multiple passes)
    for _ in range(5):
        prev = cleaned
        cleaned = _FILLER_WORDS.sub(" ", cleaned)
        if cleaned == prev:
            break
    # Remove false starts ("they can't they can't" -> "they can't")
    cleaned = _FALSE_STARTS.sub(r"\1", cleaned)
    # Collapse repeated words ("no no no no" -> "no")
    cleaned = _REPEATED_WORDS.sub(r"\1", cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _clean_part(text: str) -> str:
    """Clean a claim or reason part after splitting on premise indicator."""
    cleaned = _deep_clean_text(text)
    # Strip leading conjunctions ("and the earth...", "but we know...")
    cleaned = _LEADING_CONJUNCTIONS.sub("", cleaned).strip()
    return cleaned


def _extract_claim_reason_pairs(text: str) -> list[tuple[str, str, str]]:
    """Extract (claim, reason, full_sentence) triples from turn text.

    Aggressively cleans spoken language before extraction.
    Uses content word count instead of total word count for quality filter.
    """
    # Deep-clean the entire text first
    cleaned_text = _deep_clean_text(text)

    sentences = _SENTENCE_SPLIT.split(cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    pairs: list[tuple[str, str, str]] = []

    for sentence in sentences:
        # Skip short sentences (total words after cleaning)
        if len(sentence.split()) < MIN_SENTENCE_WORDS:
            continue

        # Find premise indicators in this sentence
        for m in PREMISE_INDICATORS.finditer(sentence):
            claim_raw = sentence[:m.start()].strip()
            reason_raw = sentence[m.end():].strip()

            # Clean both parts (strip leading conjunctions, etc.)
            claim = _clean_part(claim_raw)
            reason = _clean_part(reason_raw)

            # Both parts must have enough CONTENT words
            if _count_content_words(claim) < MIN_CLAIM_CONTENT_WORDS:
                continue
            if _count_content_words(reason) < MIN_REASON_CONTENT_WORDS:
                continue

            pairs.append((claim, reason, sentence))
            break  # one pair per sentence

    return pairs


def _query_ollama(claim: str, reason: str, full_sentence: str) -> int:
    """Send a claim-reason pair to ollama. Returns 1, 2, or 3."""
    prompt = _PROMPT_TEMPLATE.format(
        claim=claim, reason=reason, full_sentence=full_sentence,
    )
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        m = _DIGIT_RE.search(text)
        return int(m.group()) if m else 2
    except Exception:
        return 2  # default on failure


_EXPLANATION_PROMPT_TEMPLATE = """Evaluate the logical quality of this explanation.

Claim being made: "{claim}"
Full explanation: "{reason}"

Score 1-3 based on whether the explanation provides a MECHANISM or EVIDENCE that connects to the claim:
1 = The explanation is just a restatement, a bare assertion, an appeal to authority without evidence, or has no explanatory mechanism connecting it to the claim (e.g., "X is true because the Bible says so", "X because that's how it is", "X because I said so", "X because everyone knows")
2 = The explanation is related to the claim and provides partial explanation, but the logical chain has gaps or missing steps
3 = The explanation provides a clear mechanism, evidence, or chain of reasoning that directly explains WHY the claim would be true (e.g., "the earth is round because ships disappear bottom-first over the horizon, which can only happen on a curved surface")

The key question: does the explanation explain HOW or WHY, or does it just assert WHAT?

Reply with ONLY the number 1, 2, or 3."""


def _query_ollama_explanation(claim: str, reason: str) -> int:
    """Send a claim + explanation to ollama for scoring. Returns 1, 2, or 3."""
    prompt = _EXPLANATION_PROMPT_TEMPLATE.format(claim=claim, reason=reason)
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        m = _DIGIT_RE.search(text)
        return int(m.group()) if m else 2
    except Exception:
        return 2


def score_logical_coherence(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Score logical coherence of claim-reason pairs via remote ollama API.

    Two paths:
    1. If speech_act == "explanation" and proposition exists: send proposition + text to LLM
    2. Else if PREMISE_INDICATORS regex matches: use existing extraction logic
    """
    # Check connectivity first
    try:
        requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    except Exception as e:
        print(f"  Warning: ollama at {config.OLLAMA_URL} unreachable ({e}). Skipping logical coherence.")
        return

    total_turns = len(turns)
    scored_count = 0
    sample_log: dict[str, int] = {}  # speaker -> count of logged samples

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        # Path 1: structure-based — explanation turns with proposition
        if turn.speech_act == "explanation" and turn.proposition:
            claim = turn.proposition
            reason = (turn.clean_text or turn.text)[:1500]
            score = _query_ollama_explanation(claim, reason)

            logged = sample_log.get(turn.speaker, 0)
            if logged < SAMPLE_LOG_COUNT:
                print(f"  [{turn.speaker}] sample {logged + 1} (explanation):")
                print(f"    Claim:  {claim[:100]}")
                print(f"    Reason: {reason[:100]}")
                print(f"    Score:  {score}/3")
                sample_log[turn.speaker] = logged + 1

            turn.logical_coherence = round(score / 3, 4)
            scored_count += 1
            continue

        # Path 2: regex-based — extract claim-reason pairs from premise indicators
        pairs = _extract_claim_reason_pairs(turn.text)
        if not pairs:
            continue

        scores = []
        for claim, reason, full_sentence in pairs:
            score = _query_ollama(claim, reason, full_sentence)
            scores.append(score)

            logged = sample_log.get(turn.speaker, 0)
            if logged < SAMPLE_LOG_COUNT:
                print(f"  [{turn.speaker}] sample {logged + 1}:")
                print(f"    Claim:  {claim[:100]}")
                print(f"    Reason: {reason[:100]}")
                print(f"    Score:  {score}/3")
                sample_log[turn.speaker] = logged + 1

        turn.logical_coherence = round(sum(scores) / (len(scores) * 3), 4)
        scored_count += 1

        if scored_count % 20 == 0:
            print(f"  Logical coherence: {scored_count}/{total_turns} turns scored...")

    if scored_count > 0:
        print(f"  Logical coherence: {scored_count} turns scored")
