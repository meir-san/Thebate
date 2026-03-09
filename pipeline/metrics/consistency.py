import numpy as np
from models import Turn, Flag
import config


def score_consistency(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Appends POSITION_SHIFT flags when a speaker's
    turn drifts from their opening position (average of first 3 turns)."""
    threshold = config.THRESHOLD_CONSISTENCY

    # Group turns by speaker
    speaker_turns: dict[str, list[Turn]] = {}
    for t in turns:
        if debaters and t.speaker not in debaters:
            continue
        speaker_turns.setdefault(t.speaker, []).append(t)

    for speaker, s_turns in speaker_turns.items():
        if len(s_turns) < 4:
            continue

        # Build opening position vector: average of first 3 turns
        anchor_turns = s_turns[:3]
        anchor_emb = np.mean(
            [turn_embeddings[t.index] for t in anchor_turns], axis=0
        )
        # Renormalize so dot product = cosine similarity
        norm = np.linalg.norm(anchor_emb)
        if norm > 0:
            anchor_emb = anchor_emb / norm

        # Compare every turn after the anchor to the opening position
        for turn in s_turns[3:]:
            turn_emb = turn_embeddings[turn.index]
            similarity = float(np.dot(turn_emb, anchor_emb))

            if similarity < threshold:
                turn.flags.append(Flag(
                    turn_index=turn.index,
                    flag_type="position_shift",
                    score=similarity,
                    threshold=threshold,
                    explanation=(
                        f"Drifted from opening position — "
                        f"similarity to opening: {similarity:.2f} (threshold: {threshold:.2f})"
                    ),
                ))
