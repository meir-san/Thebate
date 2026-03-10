import re

import numpy as np
from models import Turn, Flag

TOPIC_CHANGE_MARKERS = re.compile(
    r"\b(?:speaking of|on another note|by the way|incidentally|"
    r"let me (?:change|shift|move)|changing the subject|"
    r"that reminds me|but (?:also|anyway)|moving on)\b",
    re.IGNORECASE,
)


def score_red_herring(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Detect red herrings: irrelevant responses to opponent questions or topic-change markers."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        word_count = len(turn.text.split())
        if word_count <= 20:
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

        has_question = "?" in opponent_turn.text
        has_topic_change = bool(TOPIC_CHANGE_MARKERS.search(turn.text))

        similarity = float(np.dot(
            turn_embeddings[turn.index],
            turn_embeddings[opponent_turn.index],
        ))

        flagged = False
        explanation = ""

        if has_question and similarity < 0.15:
            flagged = True
            explanation = f"Low relevance to opponent's question: {similarity:.2f}"
        elif has_topic_change:
            flagged = True
            explanation = f"Topic-change marker detected (similarity: {similarity:.2f})"

        if flagged:
            turn.red_herring_detected = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="red_herring",
                score=similarity,
                threshold=0.15,
                explanation=explanation,
            ))
