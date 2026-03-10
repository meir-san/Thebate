import re

from models import Turn

SCHEME_PATTERNS: dict[str, list[re.Pattern]] = {
    "analogy": [
        re.compile(r"\b(?:just like|similar to|the same (?:way|as)|analogous to|compared to)\b", re.IGNORECASE),
    ],
    "cause_effect": [
        re.compile(r"\b(?:because|caused by|leads to|results in|as a result|due to|consequently)\b", re.IGNORECASE),
    ],
    "authority": [
        re.compile(r"\b(?:according to|experts say|studies show|research shows|scientists|scholars)\b", re.IGNORECASE),
    ],
    "example": [
        re.compile(r"\b(?:for example|for instance|such as|consider the case|take the example)\b", re.IGNORECASE),
    ],
    "sign": [
        re.compile(r"\b(?:this (?:shows|indicates|suggests|proves|demonstrates)|evidence of|a sign of)\b", re.IGNORECASE),
    ],
    "practical_reasoning": [
        re.compile(r"\b(?:we should|we must|we need to|the (?:best|right) (?:way|approach)|in order to)\b", re.IGNORECASE),
    ],
    "values": [
        re.compile(r"\b(?:it(?:'s| is) (?:wrong|right|immoral|ethical|just|unjust|fair|unfair)|morally|values)\b", re.IGNORECASE),
    ],
}


def score_argumentation_schemes(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Identify argumentation schemes used in each turn and compute diversity per speaker."""
    speaker_schemes: dict[str, set[str]] = {}

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        matched = []
        for scheme_name, patterns in SCHEME_PATTERNS.items():
            if any(p.search(turn.text) for p in patterns):
                matched.append(scheme_name)

        turn.schemes = matched

        if turn.speaker not in speaker_schemes:
            speaker_schemes[turn.speaker] = set()
        speaker_schemes[turn.speaker].update(matched)

    # Compute scheme diversity per speaker across all their turns
    total_schemes = len(SCHEME_PATTERNS)
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue
        unique = len(speaker_schemes.get(turn.speaker, set()))
        turn.scheme_diversity = round(unique / total_schemes, 4) if total_schemes > 0 else 0.0
