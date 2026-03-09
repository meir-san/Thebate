import re

import numpy as np
from models import Turn, Flag
import config

_RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_RE_PERCENT = re.compile(r"\b\d+(?:\.\d+)?%")


def _extract_claims(text: str) -> set[str]:
    """Extract all numbers, years, and percentages from text as normalized strings."""
    claims = set()
    for m in _RE_YEAR.finditer(text):
        claims.add(m.group())
    for m in _RE_PERCENT.finditer(text):
        claims.add(m.group())
    for m in _RE_NUMBER.finditer(text):
        val = m.group().replace(",", "")
        # Skip single-digit numbers (too generic: "1", "2", etc.)
        if len(val.replace(".", "")) <= 1:
            continue
        claims.add(val)
    return claims


def _has_conflicting_claims(claims_a: set[str], claims_b: set[str]) -> bool:
    """Return True if both sets contain numeric/date claims but with differing values.
    We check: both have claims, they overlap in TYPE (both have years, both have
    numbers, etc.) but at least one value differs."""
    if not claims_a or not claims_b:
        return False
    # If they share any exact value, that's agreement not correction
    # We want: same category of claim but different values
    years_a = {c for c in claims_a if _RE_YEAR.fullmatch(c)}
    years_b = {c for c in claims_b if _RE_YEAR.fullmatch(c)}
    if years_a and years_b and years_a != years_b:
        return True

    pcts_a = {c for c in claims_a if _RE_PERCENT.fullmatch(c)}
    pcts_b = {c for c in claims_b if _RE_PERCENT.fullmatch(c)}
    if pcts_a and pcts_b and pcts_a != pcts_b:
        return True

    nums_a = claims_a - years_a - pcts_a
    nums_b = claims_b - years_b - pcts_b
    if nums_a and nums_b and nums_a != nums_b:
        return True

    return False


def score_corrections(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Detects factual correction events by finding
    consecutive opponent turns with conflicting numeric/date claims, then
    checks if the corrected speaker acknowledges the correction."""
    threshold = config.THRESHOLD_CORRECTION

    # Pre-extract claims for all turns
    turn_claims = {t.index: _extract_claims(t.text) for t in turns}

    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue
        claims = turn_claims[turn.index]
        if not claims:
            continue

        # Find the most recent opponent turn before this one
        prev_opponent = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker != turn.speaker:
                prev_opponent = turns[j]
                break
        if prev_opponent is None:
            continue
        if debaters and prev_opponent.speaker not in debaters:
            continue

        prev_claims = turn_claims[prev_opponent.index]
        if not _has_conflicting_claims(claims, prev_claims):
            continue

        # This turn is a correction of prev_opponent. Find prev_opponent's next response.
        responder_turn = None
        for j in range(i + 1, len(turns)):
            if turns[j].speaker != turn.speaker:
                responder_turn = turns[j]
                break
        if responder_turn is None:
            continue
        if debaters and responder_turn.speaker not in debaters:
            continue

        similarity = float(np.dot(
            turn_embeddings[turn.index],
            turn_embeddings[responder_turn.index],
        ))

        if similarity < threshold:
            responder_turn.flags.append(Flag(
                turn_index=responder_turn.index,
                flag_type="correction",
                score=similarity,
                threshold=threshold,
                explanation=(
                    f"Factual correction from {turn.speaker} not acknowledged — "
                    f"similarity: {similarity:.2f} (threshold: {threshold:.2f})"
                ),
            ))
