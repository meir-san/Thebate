import re

import numpy as np
from models import Turn
import config

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_NAMED_ENTITY = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

EVIDENCE_MARKERS = re.compile(
    r"\b(?:according to|research shows|studies show|evidence shows|"
    r"data shows|the fact is|proven|demonstrated)\b",
    re.IGNORECASE,
)


def _sentence_strength(sentence: str) -> float:
    """Score a sentence's argumentative strength based on evidence markers."""
    score = 0.0
    score += len(_RE_NAMED_ENTITY.findall(sentence)) * 0.2
    score += len(_RE_NUMBER.findall(sentence)) * 0.15
    if EVIDENCE_MARKERS.search(sentence):
        score += 0.3
    connectors = config.REASONING_CONNECTORS
    text_lower = sentence.lower()
    if any(c in text_lower for c in connectors):
        score += 0.2
    return min(score, 1.0)


def score_strongest_point_targeting(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Measure whether a rebuttal targets the opponent's strongest point."""
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
            continue

        sentences = _SENTENCE_SPLIT.split(opponent_turn.text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and not s.strip().endswith("?")]

        if len(sentences) < 2:
            continue

        # Score each sentence's strength and rank them
        strengths = [(s, _sentence_strength(s)) for s in sentences]
        strengths.sort(key=lambda x: x[1], reverse=True)

        # Embed each opponent sentence and find which is most similar to the rebuttal
        turn_emb = turn_embeddings[turn.index]
        best_sim = -1.0
        best_rank = len(strengths)

        for rank, (sent, strength) in enumerate(strengths):
            sent_emb = embedder.embed(sent)
            sim = float(np.dot(turn_emb, sent_emb))
            if sim > best_sim:
                best_sim = sim
                best_rank = rank

        # Map rank to 0.0-1.0 score: targeting strongest = 1.0
        if len(strengths) > 1:
            targeting = 1.0 - (best_rank / (len(strengths) - 1))
        else:
            targeting = 1.0

        turn.targeting_score = round(targeting, 4)
