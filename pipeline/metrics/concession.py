import re
from models import Turn

CONCESSION_PHRASES = [
    "you're right", "you are right", "fair point", "good point",
    "i'll concede", "i concede", "granted", "that's fair",
    "i was wrong", "i stand corrected", "fair enough",
    "you make a good point", "i agree with that",
    "that's a valid point", "okay you got me",
]

ANTI_CONCESSION = [
    "but", "however", "although", "that said", "except",
    "still", "regardless", "even so", "yet",
]

MIN_WORDS_CONCESSION = 8


def _is_genuine_concession(text: str, phrase: str) -> bool:
    """Check that the concession phrase isn't immediately followed by a retraction."""
    idx = text.lower().find(phrase)
    if idx < 0:
        return False
    # Get the rest of the sentence after the phrase
    after = text[idx + len(phrase):]
    # Look at the next ~60 chars (same sentence window)
    window = after[:60].lower()
    # Check for sentence boundary — if we hit one, the concession stands
    sentence_end = re.search(r'[.!?]', window)
    if sentence_end:
        window = window[:sentence_end.start()]
    return not any(anti in window for anti in ANTI_CONCESSION)


def count_concessions(turns: list[Turn], debaters: list[str] | None = None) -> dict[str, int]:
    """Returns dict of speaker -> concession count.
    Filters out short turns and fake concessions (phrase + anti-concession retraction)."""
    counts: dict[str, int] = {}
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue
        if len(turn.text.split()) < MIN_WORDS_CONCESSION:
            continue
        text_lower = turn.text.lower()
        found = 0
        for phrase in CONCESSION_PHRASES:
            if phrase in text_lower:
                if _is_genuine_concession(turn.text, phrase):
                    found += 1
        counts[turn.speaker] = counts.get(turn.speaker, 0) + found
    return counts
