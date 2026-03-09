from statistics import mean
from models import Turn, SpeakerStats
import config


def build_speaker_stats(turns: list[Turn], speaker: str) -> SpeakerStats:
    weights = config.SCORE_WEIGHTS
    speaker_turns = [t for t in turns if t.speaker == speaker]

    # --- Engagement ---
    # Exclude None (short turns that were skipped)
    engagement_scores = [t.engagement_score for t in speaker_turns
                         if t.engagement_score is not None]
    avg_engagement = mean(engagement_scores) if engagement_scores else 0.5

    # --- Dodge ---
    # Count questions asked TO this speaker (questions in other speakers' turns)
    # A dodge flag is on the RESPONDER'S turn, so count dodge flags on this speaker's turns
    dodge_flags = [f for t in speaker_turns for f in t.flags if f.flag_type == "dodge"]
    total_dodges = len(dodge_flags)

    # Count questions directed AT this speaker = questions in other speakers' turns
    from pipeline.metrics.dodge import extract_questions
    questions_faced = sum(
        len(extract_questions(t.text))
        for t in turns
        if t.speaker != speaker
    )
    dodge_rate = total_dodges / questions_faced if questions_faced > 0 else 0.0

    # --- Claims ---
    total_claims = sum(t.claim_count for t in speaker_turns)
    supported_claims = sum(t.supported_claim_count for t in speaker_turns)
    claim_support_ratio = supported_claims / total_claims if total_claims > 0 else 1.0

    # --- Topic drift ---
    # Exclude None (short turns that were skipped)
    drift_scores = [t.topic_drift_score for t in speaker_turns
                    if t.topic_drift_score is not None]
    avg_topic_drift = mean(drift_scores) if drift_scores else 0.0

    # --- Overall score ---
    overall_score = (
        avg_engagement * weights["engagement"] +
        (1 - dodge_rate) * weights["dodge"] +
        claim_support_ratio * weights["reasoning"] +
        (1 - avg_topic_drift) * weights["drift"]
    )

    return SpeakerStats(
        speaker=speaker,
        turn_count=len(speaker_turns),
        avg_engagement=round(avg_engagement, 3),
        total_dodges=total_dodges,
        questions_faced=questions_faced,
        dodge_rate=round(dodge_rate, 3),
        total_claims=total_claims,
        supported_claims=supported_claims,
        claim_support_ratio=round(claim_support_ratio, 3),
        avg_topic_drift=round(avg_topic_drift, 3),
        overall_score=round(overall_score, 1),
    )


def score_debate(result) -> None:
    """Mutates result.stats in place after all metrics have been run.
    Only builds stats for debaters, not moderators."""
    result.stats = {
        speaker: build_speaker_stats(result.turns, speaker)
        for speaker in result.debaters
    }
