import re

from models import Turn, Flag

FALSE_DICHOTOMY_PATTERNS = [
    r"\beither\b.{1,60}\bor\b",
    r"\byou(?:'re| are) either\b",
    r"\bthere(?:'s| is) only two\b",
    r"\bit(?:'s| is) either\b",
    r"\byou can either\b",
    r"\bthe only (?:choice|option|alternative)s?\b",
    r"\byou(?:'re| are) (?:with us|against us)\b",
    r"\bif you(?:'re| are) not .{1,30} then you(?:'re| are)\b",
    r"\bthere(?:'s| is) no (?:middle ground|third option|other way)\b",
    r"\byou have to (?:choose|pick|decide) between\b",
]

_DICHOTOMY_RES = [re.compile(p, re.IGNORECASE) for p in FALSE_DICHOTOMY_PATTERNS]


def score_false_dichotomy(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Detect false dichotomy framing via either/or patterns."""
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        for pattern in _DICHOTOMY_RES:
            if pattern.search(turn.text):
                turn.false_dichotomy_detected = True
                turn.flags.append(Flag(
                    turn_index=turn.index,
                    flag_type="false_dichotomy",
                    score=1.0,
                    threshold=0.0,
                    explanation="False dichotomy: either/or framing detected",
                ))
                break  # one flag per turn
