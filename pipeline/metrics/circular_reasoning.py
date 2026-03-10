import re

import numpy as np
from models import Turn, Flag

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

CONCLUSION_INDICATORS = re.compile(
    r"\b(?:therefore|thus|hence|so|consequently|this means|"
    r"which means|that's why|it follows|this proves|"
    r"this shows|this demonstrates|in conclusion)\b",
    re.IGNORECASE,
)


def score_circular_reasoning(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Detect circular reasoning: first and last assertion sentences are near-identical."""
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        sentences = _SENTENCE_SPLIT.split(turn.text.strip())
        assertions = [s.strip() for s in sentences if s.strip() and not s.strip().endswith("?")]

        if len(assertions) <= 2:
            continue

        first = assertions[0]
        last = assertions[-1]

        if not CONCLUSION_INDICATORS.search(last):
            continue

        first_emb = embedder.embed(first)
        last_emb = embedder.embed(last)
        similarity = float(np.dot(first_emb, last_emb))

        if similarity > 0.85:
            turn.circular_reasoning_detected = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="circular_reasoning",
                score=similarity,
                threshold=0.85,
                explanation=(
                    f"Circular reasoning: premise/conclusion similarity: {similarity:.2f}"
                ),
            ))
