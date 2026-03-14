import re

import numpy as np
from models import Turn, Flag
import config

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

CLAIM_THRESHOLD = 7           # minimum assertions to consider
CONNECTOR_RATIO_THRESHOLD = 0.3
CONNECTEDNESS_THRESHOLD = 0.25  # MEDIAN pairwise similarity above this = connected
DEVELOPMENT_RATIO_THRESHOLD = 0.4  # if >40% of claims are followed by development, not gish gallop
MAX_TURN_DURATION_MS = 60_000  # turns longer than 60s are presentation segments, not rapid-fire

# Evidence/development indicators that show a claim is being supported
_DEVELOPMENT_INDICATORS = re.compile(
    r"\b(?:because|since|therefore|for example|for instance|such as|"
    r"according to|evidence|data|study|studies|research|shows?|"
    r"demonstrates?|indicates?|suggests?|proves?|means? that|"
    r"the reason|this is why|in fact|specifically|in other words)\b",
    re.IGNORECASE,
)


def score_gish_gallop(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Detect Gish Gallop: rapid-fire disconnected, unsupported claims in interactive exchange.

    NOT gish gallop:
    - Structured presentations (long monologue segments > 60s)
    - Claims that are topically connected (median pairwise similarity > 0.25)
    - Claims that are followed by supporting reasoning or evidence
    """
    connectors = config.REASONING_CONNECTORS

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        # Skip presentation/monologue segments — gish gallop is a rapid-fire tactic
        turn_duration = turn.end_ms - turn.start_ms
        if turn_duration > MAX_TURN_DURATION_MS:
            continue

        sentences = _SENTENCE_SPLIT.split(turn.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Count assertion sentences (not questions)
        assertions = [s for s in sentences if not s.endswith("?")]
        claim_count = len(assertions)

        if claim_count < CLAIM_THRESHOLD:
            continue

        # Count reasoning connectors
        text_lower = turn.text.lower()
        connector_count = sum(1 for c in connectors if c in text_lower)
        connector_ratio = connector_count / claim_count if claim_count > 0 else 1.0

        if connector_ratio >= CONNECTOR_RATIO_THRESHOLD:
            continue

        # Check claim development: is each claim followed by supporting sentence?
        claim_indices = [i for i, s in enumerate(sentences) if not s.endswith("?")]
        developed_claims = 0
        for ci in claim_indices:
            # Check the next sentence (if it exists and is not itself a claim)
            next_idx = ci + 1
            if next_idx < len(sentences):
                next_sent = sentences[next_idx]
                if not next_sent.endswith("?") and _DEVELOPMENT_INDICATORS.search(next_sent):
                    developed_claims += 1

        development_ratio = developed_claims / len(claim_indices) if claim_indices else 0
        if development_ratio >= DEVELOPMENT_RATIO_THRESHOLD:
            continue  # claims are being developed/supported, not rapid-fire

        # Check if claims are topically connected (structured explanation)
        median_similarity = 0.0
        if len(assertions) >= 2:
            assertion_embs = np.array(embedder.embed_batch(assertions))
            sim_matrix = assertion_embs @ assertion_embs.T
            n = len(assertions)
            i_idx, j_idx = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[i_idx, j_idx]
            median_similarity = float(np.median(pairwise_sims))

            if median_similarity > CONNECTEDNESS_THRESHOLD:
                continue  # connected explanation, not gish gallop

        turn.gish_gallop_detected = True
        turn.flags.append(Flag(
            turn_index=turn.index,
            flag_type="gish_gallop",
            score=connector_ratio,
            threshold=CONNECTOR_RATIO_THRESHOLD,
            explanation=(
                f"Gish Gallop: {claim_count} assertions, "
                f"connector ratio: {connector_ratio:.2f}, "
                f"median claim similarity: {median_similarity:.2f}, "
                f"development ratio: {development_ratio:.2f}, "
                f"turn duration: {turn_duration / 1000:.0f}s"
            ),
        ))
