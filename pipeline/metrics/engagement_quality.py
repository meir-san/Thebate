import re

import numpy as np
from models import Turn
import config

_CONCESSION_MARKERS = re.compile(
    r"\b(?:I agree|you(?:'re| are) right|that(?:'s| is) (?:a )?(?:fair|good|valid) point|"
    r"I concede|I accept|granted|admittedly|fair enough)\b",
    re.IGNORECASE,
)

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:however|but|on the contrary|I disagree|that(?:'s| is) not|"
    r"the problem with|actually|no,|"
    r"regarding|you raise|you mention|you claim|you say|"
    r"the claim that|in contrast|while it is true|misleading|"
    r"factually incorrect|not correct)\b",
    re.IGNORECASE,
)

_SENTIMENT_NEGATIVE = re.compile(
    r"\b(?:terrible|horrible|awful|disgusting|pathetic|ridiculous|absurd|nonsense)\b",
    re.IGNORECASE,
)

_PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|owing to|as a result of|"
    r"on the grounds that|the reason is|based on|for example|"
    r"for instance|according to|as shown by|as evidenced by|"
    r"which shows|which demonstrates|research shows|studies show|"
    r"evidence shows|the data shows|a study by|found that|"
    r"published in)\b",
    re.IGNORECASE,
)

# Minimum content similarity to count as substantive engagement
# Name-dropping the opponent without addressing their argument doesn't count
CONTENT_SIMILARITY_THRESHOLD = 0.2

_REASONING_RE = None


def _has_reasoning_content(text: str) -> bool:
    """Check if text contains at least one reasoning connector or evidence indicator."""
    global _REASONING_RE
    if _REASONING_RE is None:
        connectors = config.REASONING_CONNECTORS
        escaped = [re.escape(c) for c in connectors]
        _REASONING_RE = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return bool(_REASONING_RE.search(text))


def score_engagement_quality(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Classify each turn into DQI levels 0-3 based on engagement indicators.

    Level 0: No engagement (ignores opponent, or only name-drops without addressing content)
    Level 1: Negative engagement (attacks without substance) or bare denial
    Level 2: Basic engagement (topically relevant + substantive + reasoning)
    Level 3: Quality engagement (concession/rebuttal + premise indicator + content similarity)

    Key: referencing the opponent by name alone doesn't count as engagement.
    The response must address the CONTENT of the opponent's argument (similarity > 0.2).
    """
    # Check if structure extraction data is available
    has_structure = any(t.responds_to_opponent is not None for t in turns)

    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        # Structure-based fast path: use responds_to_opponent + speech_act
        if has_structure and turn.responds_to_opponent is not None and turn.speech_act is not None:
            if not turn.responds_to_opponent:
                turn.engagement_quality_level = 0
                continue
            if turn.speech_act in ("explanation", "correction"):
                turn.engagement_quality_level = 3
                continue
            if turn.speech_act in ("rebuttal", "concession", "agreement"):
                turn.engagement_quality_level = 2
                continue
            if turn.speech_act in ("dismissal", "insult"):
                turn.engagement_quality_level = 1
                continue
            # challenge, claim — default to 1
            turn.engagement_quality_level = 1
            continue

        # Fallback: regex + embedding based classification
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

        # Compute similarity to opponent's actual content
        if turn.index in turn_embeddings and opponent_turn.index in turn_embeddings:
            similarity = float(np.dot(
                turn_embeddings[turn.index],
                turn_embeddings[opponent_turn.index],
            ))
        else:
            similarity = 0.0

        word_count = len(turn.text.split())
        has_concession = bool(_CONCESSION_MARKERS.search(turn.text))
        has_rebuttal = bool(_REBUTTAL_MARKERS.search(turn.text))
        has_negative = bool(_SENTIMENT_NEGATIVE.search(turn.text))
        has_premise = bool(_PREMISE_INDICATORS.search(turn.text))
        has_reasoning = _has_reasoning_content(turn.text)

        # Content similarity gate: must actually address opponent's argument
        # Name-dropping ("as Bill Nye said") without content relevance = no engagement
        addresses_content = similarity > CONTENT_SIMILARITY_THRESHOLD

        # Level 0: no content engagement
        if not addresses_content:
            turn.engagement_quality_level = 0
            continue

        # Level 3: quality engagement — needs concession or (rebuttal + premise)
        # AND must genuinely address opponent's content
        if has_concession and has_premise and similarity > 0.25:
            turn.engagement_quality_level = 3
            continue
        if has_rebuttal and has_premise and similarity > 0.25:
            turn.engagement_quality_level = 3
            continue

        # Level 2: basic engagement — needs similarity + substance + reasoning
        if similarity > 0.25 and word_count > 25 and (has_reasoning or has_premise):
            turn.engagement_quality_level = 2
            continue

        # Everything else is Level 1 — bare denial, negative attacks, or thin responses
        turn.engagement_quality_level = 1
