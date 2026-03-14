import re
from collections import Counter

from models import Turn

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Common stopwords to exclude
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "been", "were",
    "they", "them", "their", "there", "then", "than", "what", "when",
    "where", "which", "while", "will", "would", "could", "should", "about",
    "into", "just", "also", "very", "much", "more", "most", "some", "such",
    "only", "over", "your", "like", "know", "make", "take", "come", "want",
    "look", "give", "tell", "call", "keep", "going", "being", "does", "doing",
    "done", "didn", "doesn", "don", "isn", "aren", "wasn", "weren", "hasn",
    "hadn", "won", "wouldn", "couldn", "shouldn", "can", "not", "but",
    "for", "are", "was", "had", "has", "how", "its", "let", "may", "our",
    "out", "own", "say", "she", "too", "use", "who", "why", "you", "all",
    "any", "get", "got", "him", "his", "her", "here", "now", "one", "see",
    "way", "day", "did", "new", "old", "two", "back", "well", "even", "still",
    "those", "these", "other", "after", "before", "because", "since", "think",
    "thing", "things", "right", "really", "actually", "something", "anything",
    "everything", "nothing", "people", "never", "always", "every",
}

NUM_TALKING_POINTS = 20
SETUP_TURNS = 3
MIN_WORD_LEN = 5


def _extract_content_words(text: str) -> list[str]:
    """Extract content words: >4 chars, not stopwords."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if len(w) >= MIN_WORD_LEN and w not in _STOPWORDS]


def _get_talking_points(turns: list[Turn], speaker: str, max_turns: int) -> set[str]:
    """Extract top N content words from speaker's first few turns as talking points."""
    counter: Counter = Counter()
    count = 0
    for t in turns:
        if t.speaker == speaker:
            counter.update(_extract_content_words(t.text))
            count += 1
            if count >= max_turns:
                break
    return {w for w, _ in counter.most_common(NUM_TALKING_POINTS)}


def _compute_coverage(text: str, talking_points: set[str]) -> float:
    """Fraction of talking points mentioned in this text."""
    if not talking_points:
        return 0.0
    words = set(_extract_content_words(text))
    mentioned = words & talking_points
    return len(mentioned) / len(talking_points)


def _linear_slope(values: list[float]) -> float:
    """Simple linear regression slope."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def score_conversational_flow(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> dict[str, float]:
    """Track self-coverage vs opponent-coverage trends across debate.

    Returns dict mapping speaker -> conversational_flow_score (for SpeakerStats).
    """
    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, float] = {}

    for speaker in speakers:
        # Find all opponents
        opponents = [s for s in speakers if s != speaker]
        if not opponents:
            results[speaker] = 0.5
            continue

        # Get talking points from first few turns
        self_points = _get_talking_points(turns, speaker, SETUP_TURNS)
        opp_points: set[str] = set()
        for opp in opponents:
            opp_points |= _get_talking_points(turns, opp, SETUP_TURNS)

        if not self_points and not opp_points:
            results[speaker] = 0.5
            continue

        # Track coverage in subsequent turns
        self_coverages: list[float] = []
        opp_coverages: list[float] = []
        turn_count = 0
        for t in turns:
            if t.speaker != speaker:
                continue
            turn_count += 1
            if turn_count <= SETUP_TURNS:
                continue  # skip setup turns
            self_coverages.append(_compute_coverage(t.text, self_points))
            opp_coverages.append(_compute_coverage(t.text, opp_points))

        if len(self_coverages) < 2:
            results[speaker] = 0.5
            continue

        self_trend = _linear_slope(self_coverages)
        opp_trend = _linear_slope(opp_coverages)

        # Score: negative self_trend is good, positive opp_trend is good
        # Normalize to roughly 0-1
        self_component = max(min(0.5 - self_trend, 1.0), 0.0)
        opp_component = max(min(0.5 + opp_trend, 1.0), 0.0)
        flow_score = self_component * 0.5 + opp_component * 0.5

        results[speaker] = round(max(min(flow_score, 1.0), 0.0), 4)

    return results
