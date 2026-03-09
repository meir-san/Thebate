import numpy as np
from models import Turn, Flag
import config


def score_topic_drift(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    topic_embedding: np.ndarray,
) -> None:
    """Mutates turns in place. Sets turn.topic_drift_score and appends to turn.flags."""
    threshold = config.THRESHOLD_TOPIC_DRIFT
    min_words = config.MIN_WORDS_TOPIC_DRIFT

    for turn in turns:
        # Skip short turns — "Yeah sure" is semantically far from any topic
        if len(turn.text.split()) < min_words:
            turn.topic_drift_score = None  # None = skipped
            continue

        similarity = float(np.dot(turn_embeddings[turn.index], topic_embedding))
        drift = round(1.0 - similarity, 4)
        turn.topic_drift_score = drift

        if drift > threshold:
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="topic_drift",
                score=drift,
                threshold=threshold,
                explanation=(
                    f"Semantic distance from debate topic: {drift:.2f} "
                    f"(threshold: {threshold:.2f})"
                )
            ))
