import re

import numpy as np
from models import Turn, Flag

DEFLECTION_PATTERNS = [
    r"\bwhat about\b",
    r"\bbut what about\b",
    r"\byeah but\b",
    r"\bwell what about\b",
    r"\bhow about\b",
    r"\bwhat about the fact\b",
    r"\bbut you (did|said|also)\b",
    r"\byou did the same\b",
    r"\byou also\b",
    r"\bhow come you\b",
    r"\bwhen you were\b",
]

_DEFLECTION_RES = [re.compile(p, re.IGNORECASE) for p in DEFLECTION_PATTERNS]


def score_whataboutism(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Detect whataboutism: deflection pattern + low similarity to opponent's previous turn."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        if not any(r.search(turn.text) for r in _DEFLECTION_RES):
            continue

        # Find most recent opponent turn
        opponent_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker != turn.speaker:
                if debaters and turns[j].speaker not in debaters:
                    continue
                opponent_turn = turns[j]
                break
        if opponent_turn is None:
            continue

        similarity = float(np.dot(
            turn_embeddings[turn.index],
            turn_embeddings[opponent_turn.index],
        ))

        if similarity < 0.25:
            turn.whataboutism_detected = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="whataboutism",
                score=similarity,
                threshold=0.25,
                explanation=(
                    f"Deflection pattern + low topical similarity: {similarity:.2f}"
                ),
            ))
