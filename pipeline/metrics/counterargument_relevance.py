import re

import numpy as np
from models import Turn

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:no[,.]|that(?:'s| is) (?:wrong|not true|incorrect|false)|"
    r"you(?:'re| are) (?:wrong|mistaken)|actually|the problem with that|"
    r"I disagree|on the contrary|however|but|"
    r"misleading|factually incorrect|not correct|"
    r"regarding|you raise|you mention|you claim|you say|"
    r"the claim that|in contrast|while it is true)\b",
    re.IGNORECASE,
)

_PREMISE_EVIDENCE = re.compile(
    r"\b(?:because|since|given that|due to|for example|for instance|"
    r"such as|according to|research shows|studies show|evidence shows|"
    r"the data shows|historically)\b",
    re.IGNORECASE,
)


def _extract_reasoning(text: str) -> list[str]:
    """Extract sentences containing premise/evidence indicators."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip() and _PREMISE_EVIDENCE.search(s)]


def _extract_claims(text: str) -> list[str]:
    """Extract claim sentences: first sentence or sentences without connectors."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    claims = []
    for i, s in enumerate(sentences):
        if i == 0 or not _PREMISE_EVIDENCE.search(s):
            claims.append(s)
    return claims[:3]  # limit to first 3 claim sentences


def score_counterargument_relevance(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Measure whether rebuttals are topically relevant while being oppositional."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        if not _REBUTTAL_MARKERS.search(turn.text):
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

        # Extract reasoning portions from both
        my_reasoning = _extract_reasoning(turn.text)
        opp_reasoning = _extract_reasoning(opponent_turn.text)

        # Extract claim portions from both
        my_claims = _extract_claims(turn.text)
        opp_claims = _extract_claims(opponent_turn.text)

        # Compute premise similarity
        if my_reasoning and opp_reasoning:
            my_r_embs = [embedder.embed(s) for s in my_reasoning]
            opp_r_embs = [embedder.embed(s) for s in opp_reasoning]
            avg_my_r = np.mean(my_r_embs, axis=0)
            avg_opp_r = np.mean(opp_r_embs, axis=0)
            norm_my = np.linalg.norm(avg_my_r)
            norm_opp = np.linalg.norm(avg_opp_r)
            if norm_my > 0 and norm_opp > 0:
                avg_my_r /= norm_my
                avg_opp_r /= norm_opp
            premise_sim = float(np.dot(avg_my_r, avg_opp_r))
        elif turn.index in turn_embeddings and opponent_turn.index in turn_embeddings:
            # Fallback: use full turn embeddings
            premise_sim = float(np.dot(
                turn_embeddings[turn.index],
                turn_embeddings[opponent_turn.index],
            ))
        else:
            continue

        # Compute conclusion similarity
        if my_claims and opp_claims:
            my_c_embs = [embedder.embed(s) for s in my_claims]
            opp_c_embs = [embedder.embed(s) for s in opp_claims]
            avg_my_c = np.mean(my_c_embs, axis=0)
            avg_opp_c = np.mean(opp_c_embs, axis=0)
            norm_my = np.linalg.norm(avg_my_c)
            norm_opp = np.linalg.norm(avg_opp_c)
            if norm_my > 0 and norm_opp > 0:
                avg_my_c /= norm_my
                avg_opp_c /= norm_opp
            conclusion_sim = float(np.dot(avg_my_c, avg_opp_c))
        else:
            conclusion_sim = 0.5  # neutral fallback

        # Relevance = how directly the response addresses the opponent's points
        # High similarity to opponent = high relevance (good counterargument)
        premise_sim = max(premise_sim, 0.0)
        turn.counterargument_relevance = round(min(premise_sim, 1.0), 4)
