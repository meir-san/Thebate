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
    graph_scores: dict[str, float] | None = None,
    coverage_scores: dict[str, float] | None = None,
    flow_scores: dict[str, float] | None = None,
    defense_scores: dict[str, dict] | None = None,
    pivot_scores: dict[str, dict] | None = None,
    substance_scores: dict[str, dict] | None = None,
) -> SpeakerStats:
    weights = config.SCORE_WEIGHTS
    all_speaker_turns = [t for t in turns if t.speaker == speaker]
    speaker_turns = [t for t in all_speaker_turns if t.score_this]

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
    has_structure = any(t.speech_act is not None for t in turns)

    if has_structure:
        # Use speech_act for correction detection
        corrections_received = 0
        corrections_acknowledged = 0
        for idx, t in enumerate(turns):
            if t.speaker == speaker:
                continue
            if debaters and t.speaker not in debaters:
                continue
            if t.speech_act != "correction":
                continue
            # Check that the preceding turn was by this speaker
            prev_turn = None
            for j in range(idx - 1, -1, -1):
                if turns[j].speaker != t.speaker:
                    prev_turn = turns[j]
                    break
            if prev_turn is None or prev_turn.speaker != speaker:
                continue
            # Find this speaker's next turn after the correction
            response_turn = None
            for j in range(idx + 1, len(turns)):
                if turns[j].speaker == speaker:
                    response_turn = turns[j]
                    break
            if response_turn is None:
                corrections_received += 1
                continue
            corrections_received += 1
            if response_turn.speech_act not in ("dismissal", "insult", None):
                corrections_acknowledged += 1
    else:
        # Fallback: regex-based correction detection
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
    fallacy_rate = min(total_fallacy_flags / len(speaker_turns), 1.0) if speaker_turns else 0.0

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

    # --- Premise sufficiency ---
    ps_vals = [t.premise_sufficiency for t in speaker_turns
               if t.premise_sufficiency is not None]
    premise_sufficiency_score = mean(ps_vals) if ps_vals else 0.0
    # Cap at 1.0 (ratios above 1.0 are possible but represent saturation)
    premise_sufficiency_score = min(premise_sufficiency_score, 1.0)

    # --- Argument depth ---
    ad_vals = [t.argument_depth for t in speaker_turns
               if t.argument_depth is not None]
    argument_depth_score = mean(ad_vals) if ad_vals else 0.0

    # --- Response specificity ---
    rs_vals = [t.response_specificity for t in speaker_turns
               if t.response_specificity is not None]
    response_specificity_score = mean(rs_vals) if rs_vals else 0.0

    # --- Logical coherence ---
    lc_vals = [t.logical_coherence for t in speaker_turns
               if t.logical_coherence is not None]
    logical_coherence_score = mean(lc_vals) if lc_vals else 0.0

    # --- Argument graph (whole-debate, passed in) ---
    graph_coherence_score = (graph_scores or {}).get(speaker, 0.0)

    # --- Entailment score ---
    ent_vals = [t.entailment_score for t in speaker_turns
                if t.entailment_score is not None]
    avg_entailment_score = mean(ent_vals) if ent_vals else 0.0

    # --- Counterargument relevance ---
    car_vals = [t.counterargument_relevance for t in speaker_turns
                if t.counterargument_relevance is not None]
    avg_counterargument_relevance = mean(car_vals) if car_vals else 0.0

    # --- Argument coverage (whole-debate, passed in) ---
    argument_coverage_score = (coverage_scores or {}).get(speaker, 0.0)

    # --- Conversational flow (whole-debate, passed in) ---
    conversational_flow_score = (flow_scores or {}).get(speaker, 0.5)

    # --- Discourse quality ---
    dq_vals = [t.discourse_quality for t in speaker_turns
               if t.discourse_quality is not None]
    discourse_quality_score = mean(dq_vals) if dq_vals else 0.5

    # --- Claim defense (whole-debate, passed in) ---
    defense_data = (defense_scores or {}).get(speaker, {})
    claim_defense_rate = defense_data.get("defense_rate", 0.0)
    claims_challenged = defense_data.get("claims_challenged", 0)
    claims_defended = defense_data.get("claims_defended", 0)
    claims_abandoned = defense_data.get("claims_abandoned", 0)

    # --- Retreat pivot rate (whole-debate, passed in) ---
    pivot_data = (pivot_scores or {}).get(speaker, {})
    retreat_pivot_rate = pivot_data.get("retreat_pivot_rate", 0.0)

    # --- Structure extraction stats ---
    from collections import Counter
    if any(t.speech_act for t in speaker_turns):
        acts = Counter(t.speech_act for t in speaker_turns if t.speech_act)
        explanation_count = acts.get("explanation", 0)
        correction_count = acts.get("correction", 0)
        challenge_count = acts.get("challenge", 0)
        dismissal_count = acts.get("dismissal", 0)
        insult_count = acts.get("insult", 0)
        rebuttal_count = acts.get("rebuttal", 0)
        responds_to_opponent_rate = (
            sum(1 for t in speaker_turns if t.responds_to_opponent) / len(speaker_turns)
            if speaker_turns else 0.0
        )
    else:
        explanation_count = 0
        correction_count = 0
        challenge_count = 0
        dismissal_count = 0
        insult_count = 0
        rebuttal_count = 0
        responds_to_opponent_rate = 0.0

    # --- Substance ratio (whole-debate, passed in) ---
    substance_data = (substance_scores or {}).get(speaker, {})
    if substance_data:
        substance_share = substance_data.get("substance_share", 0.0)
        raw_explain_attack = substance_data.get("explain_attack_ratio", 0.0)
        explain_attack_ratio = min(raw_explain_attack / 4.0, 1.0)
    else:
        # No structure data — neutral score
        substance_share = 0.5
        explain_attack_ratio = 0.5

    # --- Overall score (two-metric system) ---
    has_structure = any(t.speech_act is not None for t in speaker_turns)
    if not has_structure:
        print(f"  [{speaker}] WARNING: No structure extraction data found. Run phase1_5_extract.py first for accurate scoring.")
        overall_score = 50.0
    else:
        overall_score = (
            responds_to_opponent_rate * weights["responds_to_opponent_rate"]
            + substance_share * weights["substance_share"]
        )

    # --- Diagnostic metrics (not scored, for detailed breakdown) ---
    diagnostics = {
        "premise_sufficiency": premise_sufficiency_score,
        "engagement_quality": avg_engagement_quality,
        "correction": correction_absorption_rate,
        "fallacy_free": 1 - fallacy_rate,
        "logical_coherence": logical_coherence_score,
        "argument_depth": argument_depth_score,
        "response_specificity": response_specificity_score,
        "opponent_engagement": avg_opponent_term_adoption,
        "evidence": avg_evidence_density,
        "dodge": (1 - dodge_rate) if questions_faced > 0 else 0.5,
        "argument_graph": graph_coherence_score,
        "counterargument_relevance": avg_counterargument_relevance,
        "argument_coverage": argument_coverage_score,
        "conversational_flow": conversational_flow_score,
        "discourse_quality": discourse_quality_score,
        "entailment": avg_entailment_score,
        "claim_defense": claim_defense_rate,
        "retreat_pivot": 1 - retreat_pivot_rate,
        "explain_attack_ratio": explain_attack_ratio,
    }

    rto_points = responds_to_opponent_rate * weights["responds_to_opponent_rate"]
    ss_points = substance_share * weights["substance_share"]

    print(f"\n  [{speaker}]")
    print(f"  === SCORING ===")
    print(f"    {'responds_to_opponent':<24} {rto_points:5.1f} / {weights['responds_to_opponent_rate']:>3} (raw: {responds_to_opponent_rate:.3f})")
    print(f"    {'substance_share':<24} {ss_points:5.1f} / {weights['substance_share']:>3} (raw: {substance_share:.3f})")
    print(f"    {'TOTAL':<24} {overall_score:5.1f} / 100")
    print(f"  === DIAGNOSTICS (not scored) ===")
    for name, raw in diagnostics.items():
        print(f"    {name:<28} {raw:.3f}")
    print()

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
        premise_sufficiency_score=round(premise_sufficiency_score, 4),
        argument_depth_score=round(argument_depth_score, 4),
        response_specificity_score=round(response_specificity_score, 4),
        logical_coherence_score=round(logical_coherence_score, 4),
        graph_coherence_score=round(graph_coherence_score, 4),
        avg_entailment_score=round(avg_entailment_score, 4),
        avg_counterargument_relevance=round(avg_counterargument_relevance, 4),
        argument_coverage_score=round(argument_coverage_score, 4),
        conversational_flow_score=round(conversational_flow_score, 4),
        discourse_quality_score=round(discourse_quality_score, 4),
        claim_defense_rate=round(claim_defense_rate, 4),
        claims_challenged=claims_challenged,
        claims_defended=claims_defended,
        claims_abandoned=claims_abandoned,
        retreat_pivot_rate=round(retreat_pivot_rate, 4),
        explanation_count=explanation_count,
        correction_count=correction_count,
        challenge_count=challenge_count,
        dismissal_count=dismissal_count,
        insult_count=insult_count,
        rebuttal_count=rebuttal_count,
        responds_to_opponent_rate=round(responds_to_opponent_rate, 4),
        substance_share=round(substance_data.get("substance_share", 0.0) if substance_data else 0.0, 4),
        explain_attack_ratio=round(substance_data.get("explain_attack_ratio", 0.0) if substance_data else 0.0, 4),
        overall_score=round(overall_score, 1),
    )


def score_debate(
    result,
    concession_counts: dict[str, int] | None = None,
    graph_scores: dict[str, float] | None = None,
    coverage_scores: dict[str, float] | None = None,
    flow_scores: dict[str, float] | None = None,
    defense_scores: dict[str, dict] | None = None,
    pivot_scores: dict[str, dict] | None = None,
    substance_scores: dict[str, dict] | None = None,
) -> None:
    """Mutates result.stats in place after all metrics have been run.
    Only builds stats for debaters, not moderators."""
    result.stats = {
        speaker: build_speaker_stats(
            result.turns, speaker, concession_counts, result.debaters,
            graph_scores=graph_scores,
            coverage_scores=coverage_scores,
            flow_scores=flow_scores,
            defense_scores=defense_scores,
            pivot_scores=pivot_scores,
            substance_scores=substance_scores,
        )
        for speaker in result.debaters
    }
