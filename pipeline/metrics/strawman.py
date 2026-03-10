import re

import numpy as np
from models import Turn, Flag

RESTATEMENT_MARKERS = [
    r"so you(?:'re| are) saying",
    r"you(?:'re| are) basically saying",
    r"what you(?:'re| are) really saying",
    r"in other words,? you",
    r"you(?:'re| are) claiming",
    r"you(?:'re| are) arguing",
    r"your (?:argument|position|claim) is",
    r"you seem to (?:think|believe|suggest)",
]

_RESTATEMENT_RE = [re.compile(p, re.IGNORECASE) for p in RESTATEMENT_MARKERS]

EXAGGERATION_WORDS = re.compile(
    r"\b(?:all|every|never|always|nobody|everyone|nothing|everything|"
    r"completely|absolutely)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _extract_restated_text(text: str) -> str | None:
    """Extract text after a restatement marker up to the next sentence boundary."""
    for pattern in _RESTATEMENT_RE:
        m = pattern.search(text)
        if m:
            after = text[m.end():].strip()
            # Take up to next sentence boundary
            parts = _SENTENCE_SPLIT.split(after, maxsplit=1)
            return parts[0] if parts else after
    return None


def score_strawman(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Detect strawman arguments by comparing restated content to opponent's actual words."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        restated = _extract_restated_text(turn.text)
        if restated is None:
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

        # Embed the restated content and compare to opponent's turn
        restated_emb = embedder.embed(restated)
        opponent_emb = turn_embeddings[opponent_turn.index]
        similarity = float(np.dot(restated_emb, opponent_emb))

        has_exaggeration = bool(EXAGGERATION_WORDS.search(restated))

        # Strawman: low-to-medium similarity (distortion) or exaggeration present
        if (0.15 <= similarity <= 0.65) or has_exaggeration:
            turn.strawman_detected = True
            reason = f"Restatement similarity: {similarity:.2f}"
            if has_exaggeration:
                reason += " + exaggeration markers"
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="strawman",
                score=similarity,
                threshold=0.65,
                explanation=reason,
            ))
