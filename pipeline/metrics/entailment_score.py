import re

import numpy as np
from models import Turn

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_CONCLUSION_INDICATORS = re.compile(
    r"\b(?:therefore|thus|which means|this proves|hence|consequently|"
    r"it follows that|this shows|this demonstrates|"
    r"which is why|this is why|this means|meaning that|"
    r"indicating that|confirming that|which confirms|"
    r"which distinguishes|which explains|in other words)\b",
    re.IGNORECASE,
)

_PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|based on|"
    r"for example|for instance|according to|as shown by|"
    r"as evidenced by|research shows|studies show|evidence shows|"
    r"the data shows|a study by|a \d{4} study)\b",
    re.IGNORECASE,
)

MIN_WORDS = 20


def score_entailment(
    turns: list[Turn],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Score how well conclusions follow from premises within each turn.

    Looks for both conclusion indicators (therefore, which means) and
    premise indicators (because, based on, according to) to find
    premise-conclusion pairs.
    """
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        if len(turn.text.split()) < MIN_WORDS:
            continue

        sentences = _SENTENCE_SPLIT.split(turn.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            continue

        # Find conclusion and premise sentences
        conclusion_indices = [i for i, s in enumerate(sentences) if _CONCLUSION_INDICATORS.search(s)]
        premise_indices = [i for i, s in enumerate(sentences) if _PREMISE_INDICATORS.search(s)]

        if not conclusion_indices and not premise_indices:
            continue

        # Embed all sentences
        sent_embs = [embedder.embed(s) for s in sentences]

        best_score = 0.0

        # Path 1: For each conclusion, find best matching premise before it
        for ci in conclusion_indices:
            # Premises can be before OR adjacent to conclusion
            candidate_indices = [j for j in range(len(sentences)) if j != ci]
            if not candidate_indices:
                continue
            sims = [float(np.dot(sent_embs[ci], sent_embs[j])) for j in candidate_indices]
            best_sim = max(sims) if sims else 0.0
            best_score = max(best_score, best_sim)

        # Path 2: For each premise, find best matching claim/assertion sentence
        for pi in premise_indices:
            candidate_indices = [j for j in range(len(sentences)) if j != pi
                                and j not in premise_indices]
            if not candidate_indices:
                continue
            sims = [float(np.dot(sent_embs[pi], sent_embs[j])) for j in candidate_indices]
            best_sim = max(sims) if sims else 0.0
            best_score = max(best_score, best_sim)

        if best_score > 0:
            turn.entailment_score = round(max(best_score, 0.0), 4)
