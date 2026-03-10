import re

from models import Turn, Flag
import config

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def score_gish_gallop(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Detect Gish Gallop: many assertions with few reasoning connectors."""
    connectors = config.REASONING_CONNECTORS

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        sentences = _SENTENCE_SPLIT.split(turn.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Count assertion sentences (not questions)
        assertions = [s for s in sentences if not s.endswith("?")]
        claim_count = len(assertions)

        if claim_count < 5:
            continue

        # Count reasoning connectors
        text_lower = turn.text.lower()
        connector_count = sum(1 for c in connectors if c in text_lower)
        connector_ratio = connector_count / claim_count if claim_count > 0 else 1.0

        if connector_ratio < 0.3:
            turn.gish_gallop_detected = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="gish_gallop",
                score=connector_ratio,
                threshold=0.3,
                explanation=(
                    f"Gish Gallop: {claim_count} assertions, "
                    f"connector ratio: {connector_ratio:.2f}"
                ),
            ))
