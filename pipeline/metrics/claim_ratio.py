import re

import numpy as np

from models import Turn
import config

FILLER_RE = re.compile(
    r"^(yeah|yep|no|yes|okay|ok|right|exactly|sure|absolutely|"
    r"i mean|you know|like i said|so|well|um|uh|hmm|wait|look|listen)\b",
    re.IGNORECASE
)

CLAIM_DEDUP_THRESHOLD = 0.85


def is_claim(sentence: str) -> bool:
    """A claim is a declarative sentence with enough content to evaluate."""
    s = sentence.strip()
    if len(s.split()) < 6:
        return False
    if s.endswith('?'):
        return False
    if FILLER_RE.match(s):
        return False
    return True


def is_supported(sentence: str) -> bool:
    """A supported claim contains at least one reasoning connector."""
    s_lower = sentence.lower()
    return any(connector in s_lower for connector in config.REASONING_CONNECTORS)


def _dedup_claims(claims: list[dict], embedder) -> list[dict]:
    """Deduplicate claims by cosine similarity on embeddings.

    Each claim is {"text": str, "supported": bool}.
    When merging duplicates, keep supported=True if any instance is supported.
    """
    if not claims or embedder is None:
        return claims

    texts = [c["text"] for c in claims]
    embs = embedder.embed_batch(texts)

    # Greedy clustering: assign each claim to the first cluster it matches
    clusters: list[dict] = []  # {"text": str, "supported": bool, "emb": ndarray}

    for i, claim in enumerate(claims):
        merged = False
        for cluster in clusters:
            sim = float(np.dot(embs[i], cluster["emb"]))
            if sim > CLAIM_DEDUP_THRESHOLD:
                # Merge: keep supported if either is supported
                if claim["supported"]:
                    cluster["supported"] = True
                merged = True
                break
        if not merged:
            clusters.append({
                "text": claim["text"],
                "supported": claim["supported"],
                "emb": embs[i],
            })

    return [{"text": c["text"], "supported": c["supported"]} for c in clusters]


def score_claims(turns: list[Turn], embedder=None) -> None:
    """Mutates turns in place. Sets turn.claim_count and turn.supported_claim_count.

    If embedder is provided, deduplicates claims per speaker before counting.
    """
    # First pass: extract raw claims per turn
    turn_claims: dict[int, list[dict]] = {}
    for turn in turns:
        sentences = [s.strip() for s in re.split(r'[.!?]+', turn.text) if s.strip()]
        claims = [{"text": s, "supported": is_supported(s)} for s in sentences if is_claim(s)]
        turn_claims[turn.index] = claims

    if embedder is None:
        # No dedup — original behavior
        for turn in turns:
            claims = turn_claims[turn.index]
            turn.claim_count = len(claims)
            turn.supported_claim_count = sum(1 for c in claims if c["supported"])
        return

    # Group claims by speaker for deduplication
    speaker_claims: dict[str, list[dict]] = {}
    for turn in turns:
        speaker_claims.setdefault(turn.speaker, []).extend(turn_claims[turn.index])

    # Dedup per speaker
    speaker_deduped: dict[str, list[dict]] = {}
    for speaker, claims in speaker_claims.items():
        speaker_deduped[speaker] = _dedup_claims(claims, embedder)

    # Distribute deduped counts back to turns proportionally
    # Each turn gets credit for its unique claims relative to the speaker total
    for speaker, deduped in speaker_deduped.items():
        total_unique = len(deduped)
        supported_unique = sum(1 for c in deduped if c["supported"])

        speaker_turns = [t for t in turns if t.speaker == speaker]
        raw_total = sum(len(turn_claims[t.index]) for t in speaker_turns)

        for turn in speaker_turns:
            raw = len(turn_claims[turn.index])
            if raw_total > 0 and raw > 0:
                ratio = raw / raw_total
                turn.claim_count = max(1, round(total_unique * ratio))
                turn.supported_claim_count = round(supported_unique * ratio)
            else:
                turn.claim_count = 0
                turn.supported_claim_count = 0
