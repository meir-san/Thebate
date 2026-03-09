from statistics import mean
from models import Turn, SpeakerStats
import config


def build_speaker_stats(
    turns: list[Turn],
    speaker: str,
    concession_counts: dict[str, int] | None = None,
) -> SpeakerStats:
    weights = config.SCORE_WEIGHTS
    speaker_turns = [t for t in turns if t.speaker == speaker]

    # --- Engagement ---
    engagement_scores = [t.engagement_score for t in speaker_turns
                         if t.engagement_score is not None]
    avg_engagement = mean(engagement_scores) if engagement_scores else 0.5

    # --- Dodge ---
    dodge_flags = [f for t in speaker_turns for f in t.flags if f.flag_type == "dodge"]
    total_dodges = len(dodge_flags)

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
    drift_scores = [t.topic_drift_score for t in speaker_turns
                    if t.topic_drift_score is not None]
    avg_topic_drift = mean(drift_scores) if drift_scores else 0.0

    # --- Correction absorption ---
    # Count correction flags on this speaker's turns (unacknowledged corrections)
    correction_flags = [f for t in speaker_turns for f in t.flags if f.flag_type == "correction"]
    # Count all factual correction events directed at this speaker
    from pipeline.metrics.correction import _extract_claims, _has_conflicting_claims
    turn_claims = {t.index: _extract_claims(t.text) for t in turns}
    corrections_received = 0
    for idx, t in enumerate(turns):
        if t.speaker == speaker:
            continue
        if debaters and t.speaker not in debaters:
            continue
        claims = turn_claims[t.index]
        if not claims:
            continue
        # Find opponent turn before this one (should be by 'speaker')
        prev_opponent = None
        for j in range(idx - 1, -1, -1):
            if turns[j].speaker != t.speaker:
                prev_opponent = turns[j]
                break
        if prev_opponent is None or prev_opponent.speaker != speaker:
            continue
        if not _has_conflicting_claims(claims, turn_claims[prev_opponent.index]):
            continue
        # This is a correction directed at 'speaker'. Check responder is 'speaker'.
        for j in range(idx + 1, len(turns)):
            if turns[j].speaker != t.speaker:
                if turns[j].speaker == speaker:
                    corrections_received += 1
                break
    corrections_unacknowledged = len(correction_flags)
    corrections_acknowledged = corrections_received - corrections_unacknowledged
    correction_absorption_rate = (
        corrections_acknowledged / corrections_received
        if corrections_received > 0 else 1.0
    )

    # --- Position consistency ---
    position_shift_flags = [f for t in speaker_turns for f in t.flags if f.flag_type == "position_shift"]
    position_shifts = len(position_shift_flags)
    consistency_score = 1 - (position_shifts / len(speaker_turns)) if speaker_turns else 1.0

    # --- Concessions ---
    concessions_made = (concession_counts or {}).get(speaker, 0)
    concession_rate = concessions_made / len(speaker_turns) if speaker_turns else 0.0

    # --- Evidence density ---
    total_evidence_markers = sum(t.evidence_markers for t in speaker_turns)
    densities = [t.evidence_density for t in speaker_turns if len(t.text.split()) > 0]
    avg_evidence_density = mean(densities) if densities else 0.0

    # --- Overall score ---
    # correction and consistency tracked but excluded from formula until reliable
    overall_score = (
        avg_engagement * weights["engagement"] +
        (1 - dodge_rate) * weights["dodge"] +
        claim_support_ratio * weights["reasoning"] +
        (1 - avg_topic_drift) * weights["drift"] +
        (concession_rate * 100) * weights["concession"] +
        (avg_evidence_density * 100) * weights["evidence"]
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
        corrections_received=corrections_received,
        corrections_acknowledged=corrections_acknowledged,
        correction_absorption_rate=round(correction_absorption_rate, 3),
        position_shifts=position_shifts,
        consistency_score=round(consistency_score, 3),
        concessions_made=concessions_made,
        concession_rate=round(concession_rate, 4),
        avg_evidence_density=round(avg_evidence_density, 4),
        total_evidence_markers=total_evidence_markers,
        overall_score=round(overall_score, 1),
    )


def score_debate(result, concession_counts: dict[str, int] | None = None) -> None:
    """Mutates result.stats in place after all metrics have been run.
    Only builds stats for debaters, not moderators."""
    result.stats = {
        speaker: build_speaker_stats(result.turns, speaker, concession_counts)
        for speaker in result.debaters
    }
