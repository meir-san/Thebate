import re

from models import Turn, Flag

ATTACK_WORDS = [
    "stupid", "idiot", "fool", "ignorant", "incompetent", "liar", "dishonest",
    "clueless", "delusional", "hypocrite", "corrupt", "pathetic", "coward",
    "moron", "dumb", "naive", "arrogant", "selfish", "lazy", "ridiculous",
    "disgrace", "shameful", "unqualified", "biased", "bigot", "racist",
    "sexist", "extremist", "radical", "fanatic", "hack", "fraud", "phony",
    "crook", "criminal", "deplorable", "disgusting", "despicable",
]

_SECOND_PERSON = re.compile(r"\byou(?:'re|r|rself)?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z']+")


def score_ad_hominem(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Detect ad hominem attacks via 2nd-person pronoun + nearby attack word."""
    attack_set = set(ATTACK_WORDS)

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        words = _WORD_RE.findall(turn.text.lower())
        hit_count = 0

        for i, word in enumerate(words):
            if not _SECOND_PERSON.fullmatch(word):
                continue
            # Check window of 5 words around the pronoun
            window_start = max(0, i - 5)
            window_end = min(len(words), i + 6)
            window = words[window_start:window_end]
            for w in window:
                if w in attack_set:
                    hit_count += 1
                    turn.flags.append(Flag(
                        turn_index=turn.index,
                        flag_type="ad_hominem",
                        score=1.0,
                        threshold=0.0,
                        explanation=f"Ad hominem: \"{w}\" near 2nd-person pronoun",
                    ))
                    break  # one flag per pronoun occurrence

        turn.ad_hominem_count = hit_count
