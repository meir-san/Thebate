import re

from models import Turn, Flag

ATTACK_WORDS = [
    "stupid", "idiot", "fool", "ignorant", "incompetent", "liar", "dishonest",
    "clueless", "delusional", "hypocrite", "corrupt", "pathetic", "coward",
    "moron", "dumb", "naive", "arrogant", "selfish", "lazy",
    "disgrace", "shameful", "unqualified", "biased", "bigot", "racist",
    "sexist", "extremist", "radical", "fanatic", "hack", "fraud", "phony",
    "crook", "criminal", "deplorable", "disgusting", "despicable",
    "prick", "dick", "asshole", "ass", "jerk", "douche", "douchebag",
    "scumbag", "bastard", "bitch",
]

# Phrases that direct the attack at the ARGUMENT, not the person.
# If any of these appear within 10 words of the attack word, suppress the flag.
_ARGUMENT_DIRECTED = re.compile(
    r"\b(?:your\s+argument|your\s+claim|your\s+point|your\s+logic|"
    r"your\s+reasoning|your\s+evidence|your\s+theory|your\s+model|"
    r"your\s+explanation|what\s+you\s+said|what\s+you(?:'re|\s+are)\s+saying|"
    r"that(?:'s|\s+is)\s+ridiculous|that(?:'s|\s+is)\s+absurd|"
    r"that(?:'s|\s+is)\s+nonsense|that(?:'s|\s+is)\s+pathetic|"
    r"the\s+argument|the\s+claim|the\s+idea|this\s+idea|this\s+argument)\b",
    re.IGNORECASE,
)

# Motive-attacking phrases (ad hominem circumstantial)
# These attack the person's credibility/motives rather than their argument
_MOTIVE_ATTACK_PATTERNS = re.compile(
    r"\b(?:follow the money|pushing an agenda|keep (?:your|their) funding|"
    r"just trying to scare|paid to say|paid shill|in it for the money|"
    r"part of the system|part of the establishment|"
    r"making a big deal out of nothing|keep your funding|"
    r"you scientists|you people|you experts|"
    r"just about control|taking away (?:our|your) freedoms|"
    r"government agents?|bought and paid for)\b",
    re.IGNORECASE,
)

_SECOND_PERSON = re.compile(r"\byou(?:'re|r|rself)?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z']+")


def score_ad_hominem(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Detect ad hominem attacks.

    When structure data exists, speech_act == "insult" is the primary signal.
    Regex runs as supplement to catch insults the LLM missed.
    No double-counting: if LLM already flagged, regex won't add another flag.
    """
    attack_set = set(ATTACK_WORDS)

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        hit_count = 0
        llm_flagged = False

        # Primary: LLM classification (when structure data exists)
        if turn.speech_act == "insult":
            hit_count += 1
            llm_flagged = True
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="ad_hominem",
                score=1.0,
                threshold=0.0,
                explanation="Ad hominem: LLM classified as insult",
            ))

        # Supplement: regex detection (catches insults LLM missed)
        # Skip regex if LLM already flagged this turn
        if not llm_flagged:
            words = _WORD_RE.findall(turn.text.lower())

            for i, word in enumerate(words):
                if not _SECOND_PERSON.fullmatch(word):
                    continue
                window_start = max(0, i - 5)
                window_end = min(len(words), i + 6)
                window = words[window_start:window_end]

                attack_word = None
                for w in window:
                    if w in attack_set:
                        attack_word = w
                        break

                if attack_word is None:
                    continue

                ctx_start = max(0, i - 10)
                ctx_end = min(len(words), i + 11)
                ctx_text = " ".join(words[ctx_start:ctx_end])

                if _ARGUMENT_DIRECTED.search(ctx_text):
                    continue

                hit_count += 1
                turn.flags.append(Flag(
                    turn_index=turn.index,
                    flag_type="ad_hominem",
                    score=1.0,
                    threshold=0.0,
                    explanation=f"Ad hominem: \"{attack_word}\" near 2nd-person pronoun",
                ))

            # Secondary: motive-attacking phrases
            for m in _MOTIVE_ATTACK_PATTERNS.finditer(turn.text):
                hit_count += 1
                turn.flags.append(Flag(
                    turn_index=turn.index,
                    flag_type="ad_hominem",
                    score=1.0,
                    threshold=0.0,
                    explanation=f"Ad hominem (motive attack): \"{m.group()}\"",
                ))

        turn.ad_hominem_count = hit_count
