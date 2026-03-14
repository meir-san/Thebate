import re

import numpy as np
from models import Turn, Flag
import config

# --- Method A: Conflicting numeric claims ---

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
        if len(val.replace(".", "")) <= 1:
            continue
        claims.add(val)
    return claims


def _has_conflicting_claims(claims_a: set[str], claims_b: set[str]) -> bool:
    """Return True if both sets contain numeric/date claims but with differing values."""
    if not claims_a or not claims_b:
        return False
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


# --- Method B: Semantic rebuttal detection ---

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:"
    r"no[,.]"
    r"|that(?:'s| is) wrong"
    r"|that(?:'s| is) not true"
    r"|that(?:'s| is) not how"
    r"|that(?:'s| is) incorrect"
    r"|that(?:'s| is) false"
    r"|you(?:'re| are) wrong"
    r"|you(?:'re| are) mistaken"
    r"|that doesn'?t"
    r"|it doesn'?t work that way"
    r"|that(?:'s| is) a misunderstanding"
    r"|that(?:'s| is) not what"
    r"|actually,"
    r"|incorrect"
    r"|the problem with that"
    r"|that(?:'s| is) a straw\s?man"
    r"|that makes no sense"
    r"|that(?:'s| is) nonsense"
    r"|that(?:'s| is) absurd"
    r")",
    re.IGNORECASE,
)

MIN_WORDS_REBUTTAL = 15


def _is_semantic_rebuttal(text: str) -> bool:
    """Return True if the turn contains a rebuttal marker and enough substance."""
    if len(text.split()) < MIN_WORDS_REBUTTAL:
        return False
    return bool(_REBUTTAL_MARKERS.search(text))


def score_corrections(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Detects correction events.

    If speech_act data exists, uses it: opponent turns with speech_act == "correction"
    are corrections. Speaker responding with anything other than "dismissal"/"insult"
    counts as acknowledged.

    Falls back to two regex methods:
    A) Conflicting numeric/date claims between consecutive opponent turns
    B) Semantic rebuttal markers with substantive content
    """
    # Check if structure extraction data is available
    has_structure = any(t.speech_act is not None for t in turns)

    if has_structure:
        for i, turn in enumerate(turns):
            if debaters and turn.speaker not in debaters:
                continue
            if turn.speech_act != "correction":
                continue

            # This turn corrects the previous opponent — find who was corrected
            prev_opponent = None
            for j in range(i - 1, -1, -1):
                if turns[j].speaker != turn.speaker:
                    prev_opponent = turns[j]
                    break
            if prev_opponent is None:
                continue
            if debaters and prev_opponent.speaker not in debaters:
                continue

            corrected_speaker = prev_opponent.speaker

            # Find corrected speaker's next response
            responder_turn = None
            for j in range(i + 1, len(turns)):
                if turns[j].speaker == corrected_speaker:
                    responder_turn = turns[j]
                    break
            if responder_turn is None:
                continue

            # Check acknowledgment via speech_act
            acknowledged = responder_turn.speech_act not in ("dismissal", "insult", None)

            if not acknowledged:
                responder_turn.flags.append(Flag(
                    turn_index=responder_turn.index,
                    flag_type="correction",
                    score=0.0,
                    threshold=0.0,
                    explanation=(
                        f"Correction from {turn.speaker} not acknowledged — "
                        f"response speech_act: {responder_turn.speech_act}"
                    ),
                ))
        return

    threshold = config.THRESHOLD_CORRECTION

    # Pre-extract claims for all turns
    turn_claims = {t.index: _extract_claims(t.text) for t in turns}

    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
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

        # Method A: conflicting numeric claims
        is_correction = _has_conflicting_claims(
            turn_claims[turn.index], turn_claims[prev_opponent.index]
        )

        # Method B: semantic rebuttal markers
        if not is_correction:
            is_correction = _is_semantic_rebuttal(turn.text)

        if not is_correction:
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

        # Tighter acknowledgment: similarity alone is not enough.
        # Must also show engagement with the correction's content.
        acknowledged = False
        if similarity >= threshold:
            # Check for agreement markers
            _agreement = re.compile(
                r"\b(?:you(?:'re| are) right|fair point|I see|okay|"
                r"I agree|that(?:'s| is) (?:true|fair|valid)|granted|"
                r"I concede|I accept|good point|point taken)\b",
                re.IGNORECASE,
            )
            if _agreement.search(responder_turn.text):
                acknowledged = True

            # Check for shared content words (2+ non-trivial words)
            if not acknowledged:
                _content_re = re.compile(r"[a-zA-Z]{5,}")
                correction_words = set(_content_re.findall(turn.text.lower()))
                response_words = set(_content_re.findall(responder_turn.text.lower()))
                shared = correction_words & response_words
                if len(shared) >= 2:
                    acknowledged = True

            # Check for premise indicators responding to the correction
            if not acknowledged:
                _premise = re.compile(
                    r"\b(?:because|since|given that|due to|the reason is)\b",
                    re.IGNORECASE,
                )
                if _premise.search(responder_turn.text) and similarity > 0.3:
                    acknowledged = True

        if not acknowledged:
            responder_turn.flags.append(Flag(
                turn_index=responder_turn.index,
                flag_type="correction",
                score=similarity,
                threshold=threshold,
                explanation=(
                    f"Correction from {turn.speaker} not acknowledged — "
                    f"similarity: {similarity:.2f} (threshold: {threshold:.2f})"
                ),
            ))
