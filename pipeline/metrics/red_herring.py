import re

import numpy as np
from models import Turn, Flag

TOPIC_CHANGE_MARKERS = re.compile(
    r"\b(?:speaking of|on another note|by the way|incidentally|"
    r"let me (?:change|shift|move)|changing the subject|"
    r"that reminds me|but (?:also|anyway)|moving on)\b",
    re.IGNORECASE,
)

# Deflection patterns: explicitly redirecting to unrelated topics
_DEFLECTION_PATTERNS = re.compile(
    r"\b(?:should focus on|real problems|what about the economy|"
    r"what about immigration|this is (?:just )?about control|"
    r"taking away (?:our|your) freedoms|destroying? our economy|"
    r"focus on real)\b",
    re.IGNORECASE,
)

# Only flag red herrings in interactive segments — not during monologue presentations
MAX_TURN_DURATION_MS = 60_000  # 60 seconds


def score_red_herring(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Detect red herrings: irrelevant deflections from opponent's argument.

    Flags when:
    1. Response has very low similarity to opponent's turn (<0.15), OR
    2. Response uses explicit deflection patterns with low similarity (<0.25)
    """
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        word_count = len(turn.text.split())
        if word_count <= 10:
            continue

        # Only flag in interactive segments — presentations naturally change topics
        turn_duration = turn.end_ms - turn.start_ms
        if turn_duration > MAX_TURN_DURATION_MS:
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

        # Opponent turn must also be interactive
        opponent_duration = opponent_turn.end_ms - opponent_turn.start_ms
        if opponent_duration > MAX_TURN_DURATION_MS:
            continue

        has_question = "?" in opponent_turn.text
        has_topic_change = bool(TOPIC_CHANGE_MARKERS.search(turn.text))
        has_deflection = bool(_DEFLECTION_PATTERNS.search(turn.text))

        if turn.index not in turn_embeddings or opponent_turn.index not in turn_embeddings:
            continue

        similarity = float(np.dot(
            turn_embeddings[turn.index],
            turn_embeddings[opponent_turn.index],
        ))

        flagged = False
        explanation = ""

        # Very low relevance to opponent's argument (question or statement)
        if has_question and similarity < 0.15:
            flagged = True
            explanation = f"Low relevance to opponent's question: {similarity:.2f}"
        elif has_topic_change and similarity < 0.15:
            flagged = True
            explanation = f"Topic-change marker with low relevance ({similarity:.2f})"
        # Explicit deflection patterns with moderate-low similarity
        elif has_deflection and similarity < 0.25:
            flagged = True
            explanation = f"Deflection pattern with low relevance ({similarity:.2f})"
        # Very low similarity even without markers (non-sequitur)
        elif similarity < 0.10 and word_count > 20:
            flagged = True
            explanation = f"Non-sequitur: very low relevance ({similarity:.2f})"

        if flagged:
            turn.red_herring_detected = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="red_herring",
                score=similarity,
                threshold=0.15,
                explanation=explanation,
            ))
