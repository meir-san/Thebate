from statistics import mean
from models import Turn, SpeakerStats
import config

FALLACY_FLAG_TYPES = {
    "ad_hominem", "strawman", "whataboutism", "red_herring",
    "gish_gallop", "circular_reasoning", "false_dichotomy",
}


def build_speaker_stats(
    turns: list[Turn],
    speaker: str,
    concession_counts: dict[str, int] | None = None,
    debaters: list[str] | None = None,
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
    correction_flags = [f for t in speaker_turns for f in t.flags if f.flag_type == "correction"]
    from pipeline.metrics.correction import _extract_claims, _has_conflicting_claims, _is_semantic_rebuttal
    turn_claims = {t.index: _extract_claims(t.text) for t in turns}
    corrections_received = 0
    for idx, t in enumerate(turns):
        if t.speaker == speaker:
            continue
        if debaters and t.speaker not in debaters:
            continue
        prev_opponent = None
        for j in range(idx - 1, -1, -1):
            if turns[j].speaker != t.speaker:
                prev_opponent = turns[j]
                break
        if prev_opponent is None or prev_opponent.speaker != speaker:
            continue
        is_correction = _has_conflicting_claims(
            turn_claims[t.index], turn_claims[prev_opponent.index]
        )
        if not is_correction:
            is_correction = _is_semantic_rebuttal(t.text)
        if not is_correction:
            continue
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
    speaker_conc = (concession_counts or {}).get(speaker, {"total": 0, "engaged": 0, "pivot": 0})
    concessions_made = speaker_conc["total"]
    concessions_engaged = speaker_conc["engaged"]
    concessions_pivot = speaker_conc["pivot"]
    concession_rate = concessions_engaged / len(speaker_turns) if speaker_turns else 0.0

    # --- Evidence density ---
    total_evidence_markers = sum(t.evidence_markers for t in speaker_turns)
    densities = [t.evidence_density for t in speaker_turns if len(t.text.split()) > 0]
    avg_evidence_density = mean(densities) if densities else 0.0

    # --- Fallacy rate ---
    fallacy_counts: dict[str, int] = {}
    total_fallacy_flags = 0
    for ft in FALLACY_FLAG_TYPES:
        count = sum(1 for t in speaker_turns for f in t.flags if f.flag_type == ft)
        fallacy_counts[ft] = count
        total_fallacy_flags += count
    fallacy_rate = total_fallacy_flags / len(speaker_turns) if speaker_turns else 0.0

    # --- Opponent term adoption ---
    adoption_vals = [t.opponent_term_adoption for t in speaker_turns
                     if t.opponent_term_adoption is not None]
    avg_opponent_term_adoption = mean(adoption_vals) if adoption_vals else 0.0

    # --- Targeting score ---
    targeting_vals = [t.targeting_score for t in speaker_turns
                      if t.targeting_score is not None]
    avg_targeting_score = mean(targeting_vals) if targeting_vals else 0.0

    # --- Scheme diversity ---
    all_schemes = set()
    for t in speaker_turns:
        all_schemes.update(t.schemes)
    scheme_diversity = len(all_schemes) / 7 if all_schemes else 0.0

    # --- Paraphrase fidelity ---
    fidelity_vals = [t.paraphrase_fidelity for t in speaker_turns
                     if t.paraphrase_fidelity is not None]
    avg_paraphrase_fidelity = mean(fidelity_vals) if fidelity_vals else 0.0

    # --- Engagement quality ---
    eq_vals = [t.engagement_quality_level for t in speaker_turns
               if t.engagement_quality_level is not None]
    avg_engagement_quality = (mean(eq_vals) / 3.0) if eq_vals else 0.0

    # --- Overall score ---
    overall_score = (
        claim_support_ratio * weights["premise_sufficiency"] +
        avg_engagement_quality * weights["engagement_quality"] +
        correction_absorption_rate * weights["correction"] +
        (1 - fallacy_rate) * weights["fallacy_free"] +
        avg_targeting_score * weights["argument_depth"] +
        avg_paraphrase_fidelity * weights["response_specificity"] +
        avg_opponent_term_adoption * weights["opponent_engagement"] +
        (avg_evidence_density * 100) * weights["evidence"] +
        (1 - dodge_rate) * weights["dodge"] +
        scheme_diversity * weights["scheme_diversity"] +
        (1 - avg_topic_drift) * weights["drift"] +
        (concession_rate * 100) * weights["concession"]
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
        concessions_engaged=concessions_engaged,
        concessions_pivot=concessions_pivot,
        concession_rate=round(concession_rate, 4),
        avg_evidence_density=round(avg_evidence_density, 4),
        total_evidence_markers=total_evidence_markers,
        fallacy_rate=round(fallacy_rate, 4),
        fallacy_counts=fallacy_counts,
        avg_opponent_term_adoption=round(avg_opponent_term_adoption, 4),
        avg_targeting_score=round(avg_targeting_score, 4),
        scheme_diversity=round(scheme_diversity, 4),
        avg_paraphrase_fidelity=round(avg_paraphrase_fidelity, 4),
        avg_engagement_quality=round(avg_engagement_quality, 4),
        overall_score=round(overall_score, 1),
    )


def score_debate(result, concession_counts: dict[str, int] | None = None) -> None:
    """Mutates result.stats in place after all metrics have been run.
    Only builds stats for debaters, not moderators."""
    result.stats = {
        speaker: build_speaker_stats(result.turns, speaker, concession_counts, result.debaters)
        for speaker in result.debaters
    }
