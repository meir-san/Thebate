import numpy as np
from models import Turn, Flag
import config

# Strict negation signals — bare "actually" removed, requires clear correction language
CORRECTION_SIGNALS = [
    "that's not right", "that's incorrect", "that's false",
    "no that's wrong", "that's not accurate", "you're mistaken",
    "that's not true", "actually no", "actually that's",
    "no you said",
]

MIN_WORDS_CORRECTION = 10


def _is_correction_turn(text: str) -> bool:
    """Return True if the turn contains a genuine correction signal and is long enough."""
    if len(text.split()) < MIN_WORDS_CORRECTION:
        return False
    text_lower = text.lower()
    return any(sig in text_lower for sig in CORRECTION_SIGNALS)


def score_corrections(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Appends CORRECTION flags when a speaker
    ignores a correction directed at them.
    Only triggers when the correction turn follows the opponent's turn
    (i.e. it's directed at the opponent, not a self-correction)."""
    threshold = config.THRESHOLD_CORRECTION

    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        if not _is_correction_turn(turn.text):
            continue

        # The correction must follow an opponent's turn (directed at them)
        if i == 0:
            continue
        prev_turn = turns[i - 1]
        if prev_turn.speaker == turn.speaker:
            continue  # not directed at opponent

        # Find the next turn by the corrected speaker (the opponent)
        responder_turn = None
        for j in range(i + 1, len(turns)):
            if turns[j].speaker != turn.speaker:
                responder_turn = turns[j]
                break

        if responder_turn is None:
            continue

        if debaters and responder_turn.speaker not in debaters:
            continue

        similarity = float(np.dot(
            turn_embeddings[turn.index],
            turn_embeddings[responder_turn.index],
        ))

        if similarity < threshold:
            responder_turn.flags.append(Flag(
                turn_index=responder_turn.index,
                flag_type="correction",
                score=similarity,
                threshold=threshold,
                explanation=(
                    f"Correction from {turn.speaker} not acknowledged — "
                    f"similarity: {similarity:.2f} (threshold: {threshold:.2f})"
                ),
            ))
