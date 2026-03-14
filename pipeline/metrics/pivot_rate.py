import re

import numpy as np
from models import Turn

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:no[,.]|that(?:'s| is) (?:wrong|not true|incorrect|false|not how)|"
    r"you(?:'re| are) (?:wrong|mistaken)|actually|the problem with that|"
    r"I disagree|on the contrary|however|but|"
    r"that doesn(?:'t| not)|incorrect|wrong|not true|"
    r"regarding|you raise|you mention|the claim that|"
    r"in contrast|while it is true|misleading|factually incorrect)\b",
    re.IGNORECASE,
)

# Broader challenge detection for implicit rebuttals
_CHALLENGE_WORDS = re.compile(
    r"\b(?:manipulated|debunked|hoax|hysteria|overblown|scam|fraud|fake|"
    r"propaganda|nonsense|rubbish|lie|lies|lying|absurd|ridiculous|"
    r"can not trust|cannot trust|pushing an agenda|nothing|"
    r"all wrong|always changed|not settled|just trying|scare|"
    r"what about)\b",
    re.IGNORECASE,
)

PIVOT_THRESHOLD = 0.30         # low similarity to own previous turn = pivot
CHALLENGE_SIM_THRESHOLD = 0.35 # opponent must address speaker's claim
RETREAT_SIM_THRESHOLD = 0.40   # response must be dissimilar to challenged claim to count as retreat
MAX_INTERVENING_TURNS = 2      # max other-speaker turns between consecutive own turns
SETUP_TURNS = 3
RECENT_CLAIMS_WINDOW = 5       # look at last N turns for challenged claims


def score_pivot_rate(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> dict[str, dict]:
    """Measure how often a speaker pivots to new topics, especially after being challenged.

    Returns dict mapping speaker -> {
        "total_pivots": int,
        "retreat_pivots": int,
        "retreat_pivot_rate": float,
        "claims_challenged": int,
    }
    """
    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, dict] = {}

    # Check if structure extraction data is available
    has_structure = any(t.speech_act is not None for t in turns)

    if has_structure:
        for speaker in speakers:
            total_pivots = 0
            retreat_pivots = 0
            claims_challenged = 0

            for i, turn in enumerate(turns):
                # Look for opponent challenges directed at this speaker
                if turn.speaker == speaker:
                    continue
                if debaters and turn.speaker not in debaters:
                    continue
                if turn.speech_act not in ("challenge", "correction"):
                    continue

                # Find this speaker's next turn after the challenge
                response_turn = None
                for j in range(i + 1, len(turns)):
                    if turns[j].speaker == speaker:
                        response_turn = turns[j]
                        break

                if response_turn is None:
                    continue

                claims_challenged += 1

                # Pivot = speaker doesn't respond to opponent or dismisses
                if response_turn.responds_to_opponent is False or response_turn.speech_act == "dismissal":
                    total_pivots += 1
                    retreat_pivots += 1

            retreat_pivot_rate = retreat_pivots / max(claims_challenged, 1)
            results[speaker] = {
                "total_pivots": total_pivots,
                "retreat_pivots": retreat_pivots,
                "retreat_pivot_rate": round(min(retreat_pivot_rate, 1.0), 4),
                "claims_challenged": claims_challenged,
            }
        return results

    for speaker in speakers:
        total_pivots = 0
        retreat_pivots = 0
        claims_challenged = 0
        speaker_turn_count = 0

        # Collect this speaker's recent turn indices for claim lookback
        speaker_turn_indices: list[int] = []

        for i, turn in enumerate(turns):
            if turn.speaker != speaker:
                continue

            speaker_turn_indices.append(i)
            speaker_turn_count += 1
            if speaker_turn_count <= SETUP_TURNS:
                continue

            if turn.index not in turn_embeddings:
                continue

            # Find this speaker's previous turn
            prev_own_idx = speaker_turn_indices[-2] if len(speaker_turn_indices) >= 2 else None
            if prev_own_idx is None:
                continue
            prev_own_turn = turns[prev_own_idx]

            if prev_own_turn.index not in turn_embeddings:
                continue

            # Check proximity: count intervening turns by other speakers
            intervening = 0
            for k in range(prev_own_idx + 1, i):
                if turns[k].speaker != speaker:
                    intervening += 1
            if intervening > MAX_INTERVENING_TURNS:
                continue  # too far apart, topic shift is natural

            # Compute similarity to own previous turn
            self_sim = float(np.dot(
                turn_embeddings[turn.index],
                turn_embeddings[prev_own_turn.index],
            ))

            if self_sim >= PIVOT_THRESHOLD:
                continue  # not a pivot

            total_pivots += 1

            # Check if the intervening exchange was a challenge to one of
            # this speaker's recent claims (last N turns)
            recent_own = [
                turns[idx] for idx in speaker_turn_indices[-RECENT_CLAIMS_WINDOW - 1:-1]
                if turns[idx].index in turn_embeddings
            ]

            was_challenged = False
            challenged_claim_emb = None
            for j in range(i - 1, prev_own_idx, -1):
                if turns[j].speaker == speaker:
                    continue
                if debaters and turns[j].speaker not in debaters:
                    continue

                opp_turn = turns[j]
                if not _REBUTTAL_MARKERS.search(opp_turn.text) and not _CHALLENGE_WORDS.search(opp_turn.text):
                    continue
                if opp_turn.index not in turn_embeddings:
                    continue

                opp_emb = turn_embeddings[opp_turn.index]

                # Check if opponent was addressing one of this speaker's recent claims
                for recent_t in recent_own:
                    sim_to_claim = float(np.dot(opp_emb, turn_embeddings[recent_t.index]))
                    if sim_to_claim > CHALLENGE_SIM_THRESHOLD:
                        was_challenged = True
                        challenged_claim_emb = turn_embeddings[recent_t.index]
                        claims_challenged += 1
                        break
                if was_challenged:
                    break

            if was_challenged and challenged_claim_emb is not None:
                # Check if response retreats from the challenged claim
                response_to_claim = float(np.dot(
                    turn_embeddings[turn.index], challenged_claim_emb
                ))
                if response_to_claim < RETREAT_SIM_THRESHOLD:
                    retreat_pivots += 1

        retreat_pivot_rate = retreat_pivots / max(claims_challenged, 1)

        results[speaker] = {
            "total_pivots": total_pivots,
            "retreat_pivots": retreat_pivots,
            "retreat_pivot_rate": round(min(retreat_pivot_rate, 1.0), 4),
            "claims_challenged": claims_challenged,
        }

    return results
