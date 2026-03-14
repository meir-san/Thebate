"""Substance ratio metrics: EXPLAIN_ATTACK_RATIO and SUBSTANCE_SHARE.

Only works when speech_act data exists from structure extraction.
Returns empty dict when no structure data is available.
"""

from models import Turn

_EXPLAIN_ACTS = {"explanation", "correction"}
_ATTACK_ACTS = {"challenge", "dismissal", "insult"}


def score_substance_ratio(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> dict[str, dict]:
    """Compute substance_share and explain_attack_ratio per speaker.

    Returns dict mapping speaker -> {
        "substance_share": float,
        "explain_attack_ratio": float,
        "explain_words": int,
        "attack_words": int,
        "substantive_engagement_words": int,
        "total_words": int,
    }

    Returns empty dict if no speech_act data exists.
    """
    # Check if structure data exists
    if not any(t.speech_act is not None for t in turns):
        return {}

    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, dict] = {}

    for speaker in speakers:
        speaker_turns = [
            t for t in turns
            if t.speaker == speaker and t.score_this
        ]

        total_words = sum(len(t.text.split()) for t in speaker_turns)
        if total_words == 0:
            results[speaker] = {
                "substance_share": 0.0,
                "explain_attack_ratio": 0.0,
                "explain_words": 0,
                "attack_words": 0,
                "substantive_engagement_words": 0,
                "total_words": 0,
            }
            continue

        explain_words = 0
        attack_words = 0
        substantive_engagement_words = 0

        for t in speaker_turns:
            if t.speech_act is None:
                continue
            words = len(t.text.split())

            if t.speech_act in _EXPLAIN_ACTS:
                explain_words += words
                if t.responds_to_opponent:
                    substantive_engagement_words += words
            elif t.speech_act in _ATTACK_ACTS:
                attack_words += words

        substance_share = substantive_engagement_words / total_words
        explain_attack_ratio = (
            explain_words / attack_words if attack_words > 0
            else float(explain_words) if explain_words > 0
            else 0.0
        )

        results[speaker] = {
            "substance_share": round(substance_share, 4),
            "explain_attack_ratio": round(explain_attack_ratio, 4),
            "explain_words": explain_words,
            "attack_words": attack_words,
            "substantive_engagement_words": substantive_engagement_words,
            "total_words": total_words,
        }

    return results
