import re

import numpy as np
from models import Turn

_CONCESSION_MARKERS = re.compile(
    r"\b(?:I agree|you(?:'re| are) right|that(?:'s| is) (?:a )?(?:fair|good|valid) point|"
    r"I concede|I accept|granted|admittedly|fair enough)\b",
    re.IGNORECASE,
)

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:however|but|on the contrary|I disagree|that(?:'s| is) not|"
    r"the problem with|actually|no,)\b",
    re.IGNORECASE,
)

_SENTIMENT_NEGATIVE = re.compile(
    r"\b(?:terrible|horrible|awful|disgusting|pathetic|ridiculous|absurd|nonsense)\b",
    re.IGNORECASE,
)


def score_engagement_quality(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Classify each turn into DQI levels 0-3 based on engagement indicators.

    Level 0: No engagement (ignores opponent)
    Level 1: Negative engagement (attacks without substance)
    Level 2: Basic engagement (responds to opponent's topic)
    Level 3: Quality engagement (concession/rebuttal with substance)
    """
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
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
            turn.engagement_quality_level = 2  # first turn, default
            continue

        # Compute similarity to opponent
        if turn.index in turn_embeddings and opponent_turn.index in turn_embeddings:
            similarity = float(np.dot(
                turn_embeddings[turn.index],
                turn_embeddings[opponent_turn.index],
            ))
        else:
            similarity = 0.0

        has_concession = bool(_CONCESSION_MARKERS.search(turn.text))
        has_rebuttal = bool(_REBUTTAL_MARKERS.search(turn.text))
        has_negative = bool(_SENTIMENT_NEGATIVE.search(turn.text))

        # Classify
        if similarity < 0.10:
            level = 0  # no engagement
        elif has_negative and not has_rebuttal and not has_concession:
            level = 1  # negative only
        elif has_concession or (has_rebuttal and similarity > 0.30):
            level = 3  # quality engagement
        else:
            level = 2  # basic engagement

        turn.engagement_quality_level = level
