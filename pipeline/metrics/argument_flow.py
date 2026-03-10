import re

from models import Turn

STOPWORDS = {
    "about", "after", "also", "been", "before", "being", "between", "both",
    "could", "does", "doing", "down", "during", "each", "even", "every",
    "from", "have", "having", "here", "into", "just", "know", "like",
    "make", "many", "more", "most", "much", "must", "only", "other",
    "over", "same", "should", "some", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "thing", "things", "think",
    "this", "those", "through", "very", "want", "were", "what", "when",
    "where", "which", "while", "will", "with", "would", "your", "you're",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _extract_key_terms(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if len(w) > 4 and w not in STOPWORDS}


def score_argument_flow(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Measure how much a speaker adopts/engages with opponent's key terms."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        # Find most recent opponent turn
        opponent_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker != turn.speaker:
                if debaters and turns[j].speaker not in debaters:
                    continue
                opponent_turn = turns[j]
                break
        if opponent_turn is None:
            continue

        opp_terms = _extract_key_terms(opponent_turn.text)
        if not opp_terms:
            continue

        turn_terms = _extract_key_terms(turn.text)
        overlap = opp_terms & turn_terms
        ratio = len(overlap) / len(opp_terms)
        turn.opponent_term_adoption = round(ratio, 4)
