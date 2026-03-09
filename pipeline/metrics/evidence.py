import re
from models import Turn, Flag
import config

_RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_RE_PERCENT = re.compile(r"\b\d+(\.\d+)?%")
_RE_STAT_NUMBER = re.compile(
    r"\b\d+[\.,]?\d*\s*(million|billion|thousand|people|deaths|casualties|years|months|weeks|days)\b",
    re.IGNORECASE,
)
_RE_NAMED_SOURCE = re.compile(
    r"(?:according to|per|from|citing|says|reported by|published by|based on)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)
_RE_DIRECT_QUOTE = re.compile(r'["\u201c]([^"\u201d]{12,})["\u201d]')


def _count_evidence_markers(text: str) -> int:
    count = 0
    count += len(_RE_YEAR.findall(text))
    count += len(_RE_PERCENT.findall(text))
    count += len(_RE_STAT_NUMBER.findall(text))
    count += len(_RE_NAMED_SOURCE.findall(text))
    count += len(_RE_DIRECT_QUOTE.findall(text))
    return count


def score_evidence(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Appends LOW_EVIDENCE flags for long turns
    with zero evidence markers."""
    min_words = config.MIN_WORDS_EVIDENCE

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        word_count = len(turn.text.split())
        markers = _count_evidence_markers(turn.text)
        turn.evidence_markers = markers
        turn.evidence_density = round(markers / word_count, 4) if word_count > 0 else 0.0

        if markers == 0 and word_count > min_words:
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="low_evidence",
                score=0.0,
                threshold=0.0,
                explanation=f"No evidence markers in {word_count}-word turn",
            ))
