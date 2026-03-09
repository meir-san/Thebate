import numpy as np
from models import Turn, Flag
import config


def score_engagement(
    turns: list[Turn],
    embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Sets turn.engagement_score and appends to turn.flags.
    If debaters is provided, only looks at debater turns when finding the previous opponent turn
    (a moderator asking a question should not count as the 'opponent turn' to compare against).
    """
    threshold = config.THRESHOLD_ENGAGEMENT
    min_words = config.MIN_WORDS_ENGAGEMENT

    for i, turn in enumerate(turns):
        # Skip very short turns — "Yeah", "Right" etc have no semantic content to compare
        if len(turn.text.split()) < min_words:
            turn.engagement_score = None  # explicitly None = skipped
            continue

        # Find most recent turn by a different speaker
        # If debaters list provided, skip turns from non-debaters
        opponent_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker != turn.speaker:
                if debaters and turns[j].speaker not in debaters:
                    continue
                opponent_turn = turns[j]
                break

        if opponent_turn is None:
            # First speaker's first turn — no opponent yet
            turn.engagement_score = 1.0
            continue

        # If opponent's turn was also very short, skip (no reliable embedding)
        if len(opponent_turn.text.split()) < min_words:
            turn.engagement_score = None
            continue

        score = float(np.dot(embeddings[turn.index], embeddings[opponent_turn.index]))
        turn.engagement_score = round(score, 4)

        if score < threshold:
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="low_engagement",
                score=score,
                threshold=threshold,
                explanation=(
                    f"Response similarity to opponent's previous turn: {score:.2f} "
                    f"(threshold: {threshold:.2f})"
                )
            ))
