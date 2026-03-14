import re

import numpy as np
from models import Turn

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_REBUTTAL_MARKERS = re.compile(
    r"\b(?:no[,.]|that(?:'s| is) (?:wrong|not true|incorrect|false|not how)|"
    r"you(?:'re| are) (?:wrong|mistaken)|actually|the problem with that|"
    r"I disagree|on the contrary|however|but|"
    r"that doesn(?:'t| not)|incorrect|wrong|not true|"
    r"regarding|you raise|you mention|the claim that|"
    r"in contrast|while it is true|misleading|factually incorrect)\b",
    re.IGNORECASE,
)

# Broader challenge detection: implicit challenges that don't use standard rebuttal markers
_CHALLENGE_WORDS = re.compile(
    r"\b(?:manipulated|debunked|hoax|hysteria|overblown|scam|fraud|fake|"
    r"propaganda|nonsense|rubbish|lie|lies|lying|absurd|ridiculous|"
    r"can not trust|cannot trust|pushing an agenda|nothing|"
    r"all wrong|always changed|not settled|just trying|scare|"
    r"what about)\b",
    re.IGNORECASE,
)

_PREMISE_EVIDENCE = re.compile(
    r"\b(?:because|since|given that|due to|for example|for instance|"
    r"such as|according to|research shows|studies show|evidence shows|"
    r"the data shows|historically|therefore|thus|this proves|"
    r"which means|the reason is)\b",
    re.IGNORECASE,
)

_DEFENSE_MARKERS = re.compile(
    r"\b(?:that(?:'s| is) (?:exactly )?(?:what I said|my point)|"
    r"I just explained|as I said|my point was|let me repeat|"
    r"I already addressed|which is what I said|I stand by|"
    r"like I said|as I mentioned|I(?:'ve| have) already|"
    r"that(?:'s| is) what I(?:'m| am) saying)\b",
    re.IGNORECASE,
)

CHALLENGE_SIM_THRESHOLD = 0.25  # opponent must be addressing the claim
DEFENDED_SIM_THRESHOLD = 0.2    # speaker stays on their original claim
ABANDONED_SIM_THRESHOLD = 0.15  # speaker has moved on
MIN_CLAIM_WORDS = 10
MIN_NEW_CONTENT_WORDS = 3       # defense must add at least 3 new content words
MIN_CONTENT_WORD_LEN = 5        # content words must be >4 chars


def _has_new_content(response_text: str, original_claim: str) -> bool:
    """Check if the response adds new content words beyond the original claim.

    Simply restating the same claim is repetition, not defense.
    Defense requires introducing new evidence, reasoning, or detail.
    """
    claim_words = set(
        w.lower() for w in original_claim.split()
        if len(w) >= MIN_CONTENT_WORD_LEN
    )
    response_words = set(
        w.lower() for w in response_text.split()
        if len(w) >= MIN_CONTENT_WORD_LEN
    )
    new_words = response_words - claim_words
    return len(new_words) >= MIN_NEW_CONTENT_WORDS


def _extract_claims(text: str) -> list[str]:
    """Extract substantive assertion sentences from a turn."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.endswith("?"):
            continue
        if len(s.split()) < MIN_CLAIM_WORDS:
            continue
        # Skip pure rebuttals (starting with rebuttal marker only)
        if _REBUTTAL_MARKERS.match(s) and not _PREMISE_EVIDENCE.search(s):
            continue
        claims.append(s)
    return claims


def score_claim_defense(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> dict[str, dict]:
    """Track whether speakers defend their claims when challenged.

    Returns dict mapping speaker -> {
        "claims_challenged": int,
        "claims_defended": int,
        "claims_abandoned": int,
        "claims_deflected": int,
        "defense_rate": float,
    }
    """
    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, dict] = {}

    # Check if structure extraction data is available
    has_structure = any(t.speech_act is not None for t in turns)

    if has_structure:
        for speaker in speakers:
            claims_challenged = 0
            claims_defended = 0
            claims_abandoned = 0
            claims_deflected = 0

            for i, turn in enumerate(turns):
                # Look for opponent challenges directed at this speaker
                if turn.speaker == speaker:
                    continue
                if debaters and turn.speaker not in debaters:
                    continue
                if turn.speech_act not in ("challenge", "correction", "rebuttal"):
                    continue

                # Find this speaker's next turn after the challenge
                response_turn = None
                for j in range(i + 1, len(turns)):
                    if turns[j].speaker == speaker:
                        response_turn = turns[j]
                        break

                if response_turn is None:
                    claims_challenged += 1
                    claims_abandoned += 1
                    continue

                if response_turn.speech_act is None:
                    continue

                claims_challenged += 1

                if response_turn.speech_act in ("explanation", "correction", "rebuttal"):
                    claims_defended += 1
                elif response_turn.speech_act in ("dismissal", "insult"):
                    claims_abandoned += 1
                else:
                    claims_deflected += 1

            defense_rate = claims_defended / max(claims_challenged, 1)
            results[speaker] = {
                "claims_challenged": claims_challenged,
                "claims_defended": claims_defended,
                "claims_abandoned": claims_abandoned,
                "claims_deflected": claims_deflected,
                "defense_rate": round(defense_rate, 4),
            }
        return results

    for speaker in speakers:
        claims_challenged = 0
        claims_defended = 0
        claims_abandoned = 0
        claims_deflected = 0

        # Walk through the debate looking for this speaker's claims
        for i, turn in enumerate(turns):
            if turn.speaker != speaker:
                continue

            claims = _extract_claims(turn.text)
            if not claims:
                continue

            # Embed claims from this turn
            claim_embs = embedder.embed_batch(claims)

            # Find the next opponent turn
            opponent_turn = None
            opponent_idx = None
            for j in range(i + 1, len(turns)):
                if turns[j].speaker != speaker:
                    if debaters and turns[j].speaker not in debaters:
                        continue
                    opponent_turn = turns[j]
                    opponent_idx = j
                    break

            if opponent_turn is None:
                continue

            # Check if opponent challenges any of this speaker's claims
            if not _REBUTTAL_MARKERS.search(opponent_turn.text) and not _CHALLENGE_WORDS.search(opponent_turn.text):
                continue

            if opponent_turn.index not in turn_embeddings:
                continue

            opp_emb = turn_embeddings[opponent_turn.index]

            # Check each claim for challenge
            for claim_idx, claim_emb in enumerate(claim_embs):
                claim_to_opp_sim = float(np.dot(claim_emb, opp_emb))
                if claim_to_opp_sim < CHALLENGE_SIM_THRESHOLD:
                    continue  # opponent isn't addressing this specific claim

                claims_challenged += 1

                # Find speaker's next turn after the challenge
                response_turn = None
                for j in range(opponent_idx + 1, len(turns)):
                    if turns[j].speaker == speaker:
                        response_turn = turns[j]
                        break

                if response_turn is None:
                    claims_abandoned += 1
                    continue

                # Compare response to original claim
                if response_turn.index in turn_embeddings:
                    response_emb = turn_embeddings[response_turn.index]
                else:
                    continue

                response_to_claim_sim = float(np.dot(response_emb, claim_emb))
                has_reasoning = bool(_PREMISE_EVIDENCE.search(response_turn.text))
                has_defense_marker = bool(_DEFENSE_MARKERS.search(response_turn.text))
                adds_new_content = _has_new_content(response_turn.text, claims[claim_idx])

                # Defended: must add new content (not just repetition)
                # Defense marker alone isn't enough — "as I said" + same words = repetition
                if has_defense_marker and adds_new_content:
                    claims_defended += 1
                elif response_to_claim_sim > DEFENDED_SIM_THRESHOLD and adds_new_content:
                    claims_defended += 1
                elif response_to_claim_sim < ABANDONED_SIM_THRESHOLD:
                    claims_abandoned += 1
                else:
                    claims_deflected += 1

        defense_rate = claims_defended / max(claims_challenged, 1)

        results[speaker] = {
            "claims_challenged": claims_challenged,
            "claims_defended": claims_defended,
            "claims_abandoned": claims_abandoned,
            "claims_deflected": claims_deflected,
            "defense_rate": round(defense_rate, 4),
        }

    return results
