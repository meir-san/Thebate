import re

import numpy as np
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

NEGATIONS = ["haven't", "didn't", "don't", "not", "never", "no"]

QUOTE_REFS = ["i said", "he said", "you said", "she said", "they said"]

MIN_WORDS_CONCESSION = 8


def _is_genuine_concession(text: str, phrase: str) -> bool:
    """Multi-filter check for false-positive concessions."""
    text_lower = text.lower()
    # Word-boundary match to avoid substring hits (e.g. "conceded" matching "concede")
    match = re.search(r'\b' + re.escape(phrase) + r'\b', text_lower)
    if not match:
        return False
    idx = match.start()

    # --- Anti-concession retraction filter ---
    after = text[idx + len(phrase):]
    window = after[:60].lower()
    sentence_end = re.search(r'[.!?]', window)
    if sentence_end:
        window = window[:sentence_end.start()]
    if any(anti in window for anti in ANTI_CONCESSION):
        return False

    # --- Negation filter: negation within 5 words before OR after the phrase ---
    before_text = text_lower[:idx]
    before_words = before_text.split()
    last_5 = " ".join(before_words[-5:]) if before_words else ""
    if any(neg in last_5 for neg in NEGATIONS):
        return False
    after_words = text_lower[idx + len(phrase):].split()
    first_5_after = " ".join(after_words[:5]) if after_words else ""
    if any(neg in first_5_after for neg in NEGATIONS):
        return False

    # --- Quote/reference filter: "X said" within 10 words before ---
    last_10 = " ".join(before_words[-10:]) if before_words else ""
    if any(ref in last_10 for ref in QUOTE_REFS):
        return False

    # --- Attribution filter: "I was wrong/mistaken" when talking about opponent ---
    if phrase in ("i was wrong", "i was mistaken"):
        you_count = len(re.findall(r"\byou\b|\byour\b", text_lower))
        i_count = len(re.findall(r"\bi\b|\bmy\b", text_lower))
        if you_count > i_count:
            return False

    return True


def count_concessions(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray] | None = None,
    debaters: list[str] | None = None,
) -> dict[str, dict]:
    """Returns dict of speaker -> {total, engaged, pivot}.
    engaged = concession followed by a turn with similarity > 0.25 to the concession turn.
    pivot = concession followed by a topic change."""
    results: dict[str, dict] = {}

    for i, turn in enumerate(turns):
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

        if found == 0:
            continue

        # Check engagement of the next turn
        engaged = False
        if turn_embeddings is not None:
            next_turn = None
            for j in range(i + 1, len(turns)):
                if turns[j].speaker == turn.speaker:
                    next_turn = turns[j]
                    break
            if next_turn is not None and next_turn.index in turn_embeddings and turn.index in turn_embeddings:
                sim = float(np.dot(turn_embeddings[turn.index], turn_embeddings[next_turn.index]))
                engaged = sim > 0.25

        speaker = turn.speaker
        if speaker not in results:
            results[speaker] = {"total": 0, "engaged": 0, "pivot": 0}
        results[speaker]["total"] += found
        if engaged:
            results[speaker]["engaged"] += found
        else:
            results[speaker]["pivot"] += found

    return results
